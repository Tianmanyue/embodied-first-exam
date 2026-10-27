"""
rcb_gsmini.py - task-local tooling for the GSMini tasks (dual-arm X-Trainer whose right gripper carries two GelSight Mini
tactile sensors; Isaac Sim 5.1 / Isaac Lab 2.3.2): frozen instance, action recording, fresh-process replay.
Same shape as the other benchmarks' rcb_*.py tools.

Concepts
--------
instance   : one task in its fixed nominal scene (`seed`), the physics device it is frozen on (`device`, the CPU: PhysX on the
             GPU does not replay bit for bit across processes) and the hash of the full scene state after reset
             (`state_sha`: robot joint positions / velocities and every object's pose and velocity). On load,
             `instance_exact` = hash equality and `instance_matches` = equality or every number within 2e-3.
trajectory : the (T, 18) rows handed to the robot, one per control step (60 Hz: physics 1/120 s, decimation 2):
               0-6   right end-effector target  x y z qw qx qy qz   (env frame, metres; the tactile gripper)
               7-13  left end-effector target   x y z qw qx qy qz
               14,15 right / left gripper: 0 = open, 1 = closed
               16,17 optional right / left finger gap override in metres (NaN = use the open / closed command)
             A NaN end-effector target holds that arm where it is. The targets are tracked by a differential IK and the
             joint PD drives. A replay holds the last row for SETTLE_STEPS more steps before the task is judged.
budget     : no step budget; MAX_STEPS only stops runaway trajectories.

CLI
---
  python rcb_gsmini.py make-instance --task insert_hdmi --seed 0 --out instance.json
  python rcb_gsmini.py replay        --task insert_hdmi --instance instance.json --traj traj.npz
  python rcb_gsmini.py run-oracle    --task insert_hdmi --instance instance.json --code oracle.py --out traj.npz

Python (what the agent gets)
----------------------------
  from rcb_gsmini import Session
  s = Session("insert_hdmi", instance="/app/instance.json")   # ~1 min: Isaac Sim launch + scene build
  s.reset()                       # back to the frozen initial state (unlimited)
  s.objects()                     # name -> {"position", "orientation_wxyz"}
  s.proprio()                     # joint_pos (16), right / left end-effector pose, gripper widths
  s.tactile()                     # per gel pad: height map (60, 80) mm, contact mask, press depth; RGB with tactile_rgb=True
  s.step(row)                     # one control step with an (18,) row (layout above)
  s.move_ee(arm, pose7, steps)    # straight-line interpolation of one arm's target over `steps` control steps
  s.set_gripper(arm, closed, steps)
  s.record_start(); ...; actions = s.record_stop()
  s.status()                      # {'success', 'sim_steps', 'resets', 'state_sha', 'metrics'}
  s.replay(actions)               # resets (not in a fresh process), replays open loop, returns status
  s.save_trajectory(actions, "/app/output/trajectory.npz")
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ACTION_DIM = 18
MAX_STEPS = 30000
SETTLE_STEPS = 60
DEVICE = os.environ.get("RCB_GSMINI_DEVICE", "cpu")
_APP = None


def _log(*a: Any) -> None:
    print("[rcb_gsmini]", *a, file=sys.stderr, flush=True)


def _launch_app(cameras: bool):
    global _APP
    if _APP is None:
        from gsmini_tasks.backends.isaaclab_scene import launch_app

        _APP = launch_app(headless=True, enable_cameras=bool(cameras))
    return _APP


def _flat(tree: Any) -> np.ndarray:
    parts: list[np.ndarray] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k in sorted(x):
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        else:
            try:
                parts.append(np.asarray(x, dtype=np.float64).ravel())
            except (TypeError, ValueError):
                pass

    walk(tree)
    return np.concatenate(parts) if parts else np.zeros(0)


class Session:
    def __init__(self, task: str, instance: str | Path | None = None, *, seed: int | None = None, cameras: bool = False,
                 tactile: bool = True, tactile_rgb: bool = False):
        self.task_name = str(task)
        self.instance = json.loads(Path(instance).read_text()) if instance else {}
        if self.instance and self.instance.get("task") != self.task_name:
            raise ValueError(f"instance is for task {self.instance.get('task')!r}, not {self.task_name!r}")
        self.seed = int(self.instance.get("seed", 0) if seed is None else seed)
        self.device = str(self.instance.get("device", DEVICE))
        _launch_app(cameras)
        import gsmini_tasks

        if self.task_name not in gsmini_tasks.TASKS:
            raise KeyError(f"unknown task {self.task_name!r}; known: {sorted(gsmini_tasks.TASKS)}")
        spec = gsmini_tasks.TASKS[self.task_name]
        self.cfg = spec.load_cfg()
        obs_mode = "full" if (cameras or tactile_rgb) else "state"
        self.task = spec.task_cls(self.cfg, obs_mode=obs_mode, device=self.device, cameras=bool(cameras), tactile=bool(tactile))
        self.task.build()
        self.cameras = bool(cameras)
        self.resets = -1
        self.sim_steps = 0
        self._rec: list[np.ndarray] | None = None
        self._last = None
        self.obs = None
        self.reset()

    # ------------------------------------------------------------------ state
    def _state(self) -> np.ndarray:
        st = dict(self.task.get_state())
        st.pop("_task", None)
        return _flat(st)

    def state_sha(self) -> str:
        return hashlib.sha1(self._state().tobytes()).hexdigest()

    def reset(self) -> None:
        self.obs = self.task.reset(seed=self.seed)
        self.resets += 1
        self.sim_steps = 0
        self._last = None
        s0 = self._state()
        self.reset_sha = hashlib.sha1(s0.tobytes()).hexdigest()
        if self.resets == 0:  # the instance is checked on the first reset of the process (later resets are not bit-exact)
            want = self.instance.get("state_sha")
            self.instance_exact = (want is None) or (want == self.reset_sha)
            ref = self.instance.get("state")
            self.instance_matches = self.instance_exact or (
                ref is not None and len(ref) == s0.size and float(np.max(np.abs(np.asarray(ref) - s0))) <= 2e-3)

    # ------------------------------------------------------------------ actions
    def _action(self, row):
        from gsmini_tasks.core.types import Action

        r = np.asarray(row, dtype=np.float64).reshape(-1)
        if r.size == 16:
            r = np.concatenate([r, [np.nan, np.nan]])
        if r.size != ACTION_DIM:
            raise ValueError(f"a row has {ACTION_DIM} numbers (or 16 without the gap override), got {r.size}")
        if not np.all(np.isfinite(r[14:16])):
            raise ValueError("gripper commands must be finite (0 = open, 1 = closed)")
        gap = None if np.all(np.isnan(r[16:18])) else r[16:18].copy()
        return r, Action(r[:14].copy(), np.rint(r[14:16]).astype(np.int64), gap)

    def step(self, row) -> None:
        if self.sim_steps >= MAX_STEPS:
            raise RuntimeError(f"MAX_STEPS ({MAX_STEPS}) reached")
        r, act = self._action(row)
        self.obs, _info = self.task.step(act)
        self.sim_steps += 1
        self._last = r
        if self._rec is not None:
            self._rec.append(r)

    def hold_action(self) -> np.ndarray:
        """A row that holds both arms and keeps the current gripper commands."""
        if self._last is not None:
            return self._last.copy()
        row = np.full(ACTION_DIM, np.nan)
        row[14:16] = 0.0
        return row

    def move_ee(self, arm: int, pose7, steps: int = 60) -> None:
        """Interpolate one arm's target (position linearly, orientation by slerp) from its current pose to `pose7`."""
        cur = np.asarray(self.task.scene.robot.ee_pose(int(arm)), dtype=np.float64)
        goal = np.asarray(pose7, dtype=np.float64).reshape(7)
        q0, q1 = cur[3:] / np.linalg.norm(cur[3:]), goal[3:] / np.linalg.norm(goal[3:])
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        for i in range(1, max(1, int(steps)) + 1):
            u = i / max(1, int(steps))
            row = self.hold_action()
            pos = (1.0 - u) * cur[:3] + u * goal[:3]
            d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
            if d > 0.9995:
                q = (1.0 - u) * q0 + u * q1
            else:
                th = math.acos(d)
                q = (math.sin((1.0 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
            row[7 * arm: 7 * arm + 7] = np.concatenate([pos, q / np.linalg.norm(q)])
            self.step(row)

    def set_gripper(self, arm: int, closed: bool, steps: int = 30) -> None:
        for _ in range(max(1, int(steps))):
            row = self.hold_action()
            row[14 + int(arm)] = 1.0 if closed else 0.0
            self.step(row)

    # ------------------------------------------------------------------ recording / replay
    def record_start(self) -> None:
        self._rec = []

    def record_stop(self) -> np.ndarray:
        rows, self._rec = self._rec or [], None
        return np.stack(rows) if rows else np.zeros((0, ACTION_DIM))

    def save_trajectory(self, actions: np.ndarray, path: str | Path) -> None:
        a = np.asarray(actions, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must be (T, {ACTION_DIM})")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, task=np.array(self.task_name), actions=a)

    def settle(self) -> None:
        row = self.hold_action()
        for _ in range(SETTLE_STEPS):
            self.step(row)

    def _frame(self) -> np.ndarray:
        obs = self.task.get_observation()["policy"]
        return np.concatenate([np.asarray(obs["wrist_right_rgb"]), np.asarray(obs["wrist_left_rgb"])], axis=1)

    def replay(self, actions: np.ndarray, settle: bool = True, video: str | None = None, fps: int = 15) -> dict[str, Any]:
        """Resets (unless nothing has run since the first reset of the process: then the replay is that first episode,
        exactly what the grader's fresh-process replay runs), replays the rows open loop, holds the last row."""
        if not (self.resets == 0 and self.sim_steps == 0):
            self.reset()
        frames: list[np.ndarray] = []
        every = max(1, int(round(60 / max(1, int(fps)))))  # 60 Hz control -> `fps` video frames per second
        rows = list(np.asarray(actions, dtype=np.float64))
        if settle:
            rows += [rows[-1]] * SETTLE_STEPS if rows else []
        for i, r in enumerate(rows):
            self.step(r)
            if video and self.cameras and i % every == 0 and len(frames) < 9000:
                frames.append(self._frame())
        st = self.status()
        if video and frames:  # after the verdict, never part of it
            try:
                import imageio.v2 as imageio

                imageio.mimsave(video, frames, fps=int(fps))
            except Exception as exc:  # noqa: BLE001
                _log(f"video not written: {exc!r}")
        return st

    # ------------------------------------------------------------------ queries
    def status(self) -> dict[str, Any]:
        m = {}
        try:
            m = {k: float(v) for k, v in self.task.metrics().items() if np.isscalar(v)}
        except Exception:
            pass
        return {"success": bool(self.task.is_success()), "sim_steps": int(self.sim_steps), "resets": int(self.resets),
                "state_sha": self.state_sha(), "metrics": m}

    def objects(self) -> dict[str, dict]:
        out = {}
        for o in self.cfg.get("scene", {}).get("objects", []):
            n = str(o["name"])
            try:
                p = np.asarray(self.task.object_pose(n), dtype=np.float64)
            except Exception:
                continue
            out[n] = {"position": p[:3].tolist(), "orientation_wxyz": p[3:7].tolist()}
        return out

    def proprio(self) -> dict[str, Any]:
        rob = self.task.scene.robot
        return {"joint_pos": np.asarray(rob.joint_pos(), dtype=np.float64).tolist(),
                "right_ee_pose": np.asarray(rob.ee_pose(0), dtype=np.float64).tolist(),
                "left_ee_pose": np.asarray(rob.ee_pose(1), dtype=np.float64).tolist(),
                "gripper_width": [float(rob.gripper_width(0)), float(rob.gripper_width(1))]}

    def tactile(self) -> dict[str, np.ndarray]:
        read = getattr(self.task, "sensors", {}).get("tactile")
        return {} if read is None else dict(read())

    def render_frame(self, camera: str = "wrist_right_rgb") -> np.ndarray:
        if not self.cameras:
            raise RuntimeError("cameras are off: Session(..., cameras=True)")
        obs = self.task.get_observation()
        return np.asarray(obs["policy"][camera])


# ---------------------------------------------------------------------- CLI
def _shutdown(code: int) -> None:
    import threading

    if _APP is not None:
        th = threading.Thread(target=_APP.close, daemon=True)
        th.start()
        th.join(timeout=20.0)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-instance")
    m.add_argument("--task", required=True); m.add_argument("--seed", type=int, default=0); m.add_argument("--out", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--task", required=True); r.add_argument("--instance", required=True); r.add_argument("--traj", required=True)
    r.add_argument("--video", default=None); r.add_argument("--fps", type=int, default=15)
    o = sub.add_parser("run-oracle")
    o.add_argument("--task", required=True); o.add_argument("--instance", required=True); o.add_argument("--code", required=True)
    o.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    code = 1
    try:
        if a.cmd == "make-instance":
            s = Session(a.task, seed=a.seed, tactile=False)
            st = s._state()
            inst = {"task": a.task, "seed": a.seed, "device": s.device, "state_sha": s.reset_sha, "state": st.tolist()}
            Path(a.out).write_text(json.dumps(inst, indent=1) + "\n")
            print(json.dumps({"task": a.task, "state_sha": s.reset_sha}), flush=True)
        elif a.cmd == "replay":
            d = np.load(a.traj, allow_pickle=False)
            acts = np.asarray(d["actions"], dtype=np.float64)
            s = Session(a.task, instance=a.instance, tactile=False, cameras=bool(a.video))
            t0 = time.time()
            st = s.replay(acts, video=a.video, fps=a.fps)
            print(json.dumps({"replays": [{"success": st["success"], "state_sha": st["state_sha"], "sim_steps": st["sim_steps"]}],
                              "instance_exact": bool(s.instance_exact), "instance_matches": bool(s.instance_matches),
                              "wall_s": round(time.time() - t0, 1)}), flush=True)
        elif a.cmd == "run-oracle":
            s = Session(a.task, instance=a.instance)
            ns = {"s": s, "Session": Session, "np": np}
            exec(compile(Path(a.code).read_text(), a.code, "exec"), ns)
            acts = ns.get("actions")
            if acts is None:
                raise RuntimeError("the oracle code must leave its rows in `actions`")
            s.settle()  # the live first episode, as a replay would end it
            st = s.status()
            s.save_trajectory(acts, a.out)
            print(json.dumps({"rows": int(len(acts)), **{k: st[k] for k in ("success", "state_sha", "sim_steps")}}), flush=True)
        code = 0
    except Exception:
        import traceback

        traceback.print_exc()
    _shutdown(code)
    return code


if __name__ == "__main__":
    main()
