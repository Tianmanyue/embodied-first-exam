# GSMini

Contact-rich manipulation with tactile sensing: a dual-arm X-Trainer whose right gripper carries two GelSight Mini
sensors, on Isaac Lab 2.3.2 / Isaac Sim 5.1. The tactile image of each gel pad is geometric: a 60 x 80 taxel raycast of
the pad against the touched meshes, rendered to 240 x 320 RGB with a Taxim look-up table (MIT; calibration from TacEx).

## Tasks
| Task | Mode | What is asked |
|---|---|---|
| `gsmini-insert-hdmi-i00` | privileged | pick up an HDMI plug and insert it into its port |

## Conventions (all tasks)
- Env frame, metres, z up; quaternions WXYZ. Table top z = 1.102, robot base at (-0.5, 0, 1.1) facing +x.
- Action row (18): right / left end-effector targets (7 each), right / left gripper command (0 open, 1 closed), optional
  right / left finger-gap override (NaN = unused). NaN in an arm's seven numbers holds that arm. 60 Hz control.
- Physics is PhysX on the CPU (bit-exact fresh-process replays); rendering and the tactile raycast on the GPU.
- Tooling: `rcb_gsmini.py` (`embodied_first_exam/benchmarks/gsmini/`), with the `gsmini_tasks` package as the agent may
  read it (scripted policies and expert parameters removed); both come from the files-only image `gsmini-files`.

## Budget
3600 s per task. Every Python process pays about 1 min of Isaac Sim launch and scene build; a replay of 1,000 rows takes
about 100 s with PhysX on the CPU.
