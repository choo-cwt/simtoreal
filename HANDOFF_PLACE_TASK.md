# Place Task Handoff

Date: 2026-06-11
Workspace: `/home/chichoo/squint-master6.6winproplace/squint-master`

This document records the current state of the XLeRobot Place task work. It is meant for continuing training/debugging without re-discovering the same details.

## Current Goal

Train `SO101PlaceCube-v1` for `xlerobot_right_head`, then deploy to the real robot.

Main real-robot symptoms observed so far:

- The robot can often approach the red cube.
- Grasp alignment is still sensitive to cube position and camera crop.
- It sometimes closes slightly before the gripper is fully around the cube.
- After grasping, it often does not reliably lift, move to the bin, and release.
- Real placement is not solved yet.

Current direction:

- Prefer pure visual+qpos training for the final route.
- Do not depend on privileged item/bin state for the final policy.
- Keep real-camera diagnostic recording and real-deploy crop/gripper controls available for testing.

## Important Files

Core training/task files:

```text
train_squint.py
envs/place.py
envs/base_random_env.py
envs/robot/xlerobot.py
scripts/train_place_6gb_with_logs.sh
```

Real robot deployment files:

```text
deploy.py
scripts/deploy_place_v11_routeA.sh
scripts/deploy_place_b1_diag_policy_input.sh
```

Experiment record:

```text
docs/place_experiment_versions.md
HANDOFF_PLACE_TASK.md
```

## Current Robot Setup

Use this robot id for training and deployment:

```text
--robot-uids=xlerobot_right_head
```

The agent is defined in:

```text
envs/robot/xlerobot.py
```

The active URDF is:

```text
envs/robot/xlerobot_right_arm_head_fixed.urdf
```

The root-level `urdf/` files are reference/source files, not the active agent URDF for this training path.

Controlled joints:

```text
Rotation_R
Pitch_R
Elbow_R
Wrist_Pitch_R
Wrist_Roll_R
Jaw_R
```

TCP is the average of:

```text
Fixed_Jaw_2
Moving_Jaw_2
```

## Current Controller

Use:

```text
--control-mode=pd_joint_target_delta_pos
```

Current action limits in `envs/robot/xlerobot.py`:

```python
delta_lower = [-0.07, -0.07, -0.07, -0.07, -0.07, -0.10]
delta_upper = [ 0.07,  0.07,  0.07,  0.07,  0.07,  0.10]
```

Important: wrist roll is currently unlocked. Earlier versions had wrist roll fixed at `0.0`; that is no longer true.

This is the v28/v29 slow-real action scale. Train a fresh checkpoint after reward or range changes. Do not resume from v21/v26 because the old checkpoint learned the faster action distribution.

## Current Place Environment State

File:

```text
envs/place.py
```

Current status:

- `xlerobot_right_head` is supported.
- Green bin material is retained.
- `privileged_state` switch is retained.
- Pure visual route uses `--no-privileged-state`.
- Current XLeRobot spawn area is narrowed from the default:

```python
spawn_box_pos = [0.265, 0.095]
spawn_box_half_size = [0.045, 0.035]
item_bin_min_center_dist = 0.12
item_bin_exclusion_margin = 0.01
```

With XLeRobot base `[0.05, 0, 0.068]`, this gives effective world/table XY ranges:

```text
x: [0.27, 0.36]
y: [0.06, 0.13]
```

Range visualization/check:

```text
docs/v33_effective_range.svg
deploy_utils/render_place_workspace.py
```

Current Place horizon:

```python
@register_env("SO101PlaceCube-v1", max_episode_steps=100)
@register_env("SO101PlaceCan-v1",  max_episode_steps=100)
```

This was restored because the 50-step run ended too quickly for stable grasp-lift-carry-release behavior.

## Current Top Camera Calibration

File:

```text
envs/base_random_env.py
deploy_utils/xlerobot_head_servos.json
```

Current top/head camera alignment:

```python
TOP_CAMERA_FOV = np.deg2rad(34.0)
TOP_CAMERA_LOCAL_Q = [0.180132, 0.14849, 0.750305, 0.618503]
```

Current head servo zero calibration:

```json
{
  "pan_center": 129,
  "tilt_center": 312
}
```

Use `deploy_utils/tune_top_camera.py` to compare raw real image, policy crop, sim image, and blended alignment before real deployment.

## Domain Randomization / RNG Fixes

Files:

```text
envs/place.py
envs/base_random_env.py
```

The cloud server hit multiple ManiSkill batched-RNG length errors with `NUM_EVAL_ENVS=16`, for example `index 15 is out of bounds for size 15`.

Current code has safe RNG helpers for:

- item/bin randomization in `place.py`
- lighting randomization in `base_random_env.py`
- camera FOV randomization in `base_random_env.py`
- robot color randomization in `base_random_env.py`
- gripper stiffness/damping randomization in `base_random_env.py`

Before uploading/running on cloud, verify syntax with:

```bash
cd /home/gpu/squint

python - <<'PY'
from pathlib import Path
for p in [Path("envs/place.py"), Path("envs/base_random_env.py")]:
    compile(p.read_text(), str(p), "exec")
    print("OK", p)
PY
```

Use this compile snippet instead of `python -m py_compile` if the uploaded folder has read-only `__pycache__` directories.

## Training Wrapper

Use:

```text
scripts/train_place_6gb_with_logs.sh
```

It saves:

- `runs/<EXP_NAME>/command.txt`
- `runs/<EXP_NAME>/train.log`
- `runs/<EXP_NAME>/eval_summary.txt`
- `runs/<EXP_NAME>/gpu_memory.csv`
- `runs/<EXP_NAME>/gpu_peak.txt`
- `runs/<EXP_NAME>/failure_notes.md`
- videos under `runs/<EXP_NAME>/videos`
- checkpoints under `runs/<EXP_NAME>/ckpt.pt` and `runs/<EXP_NAME>/best_ckpt.pt`

`train_squint.py` currently saves `best_ckpt.pt` whenever `success_at_end` improves.

## 4090 Cloud Fresh Training Command

Recommended v34 fresh stable-pregrasp visible range run:

```bash
cd /home/gpu/squint

env \
  EXP_NAME=place_xlerobot_v34_stable_pregrasp_range_x027_036_y006_013_64img_1024env_16eval_256upd_buf300k_5500k_4090 \
  NO_PRIVILEGED_STATE=true \
  IMAGE_SIZE=64 \
  RENDER_SIZE=128 \
  NUM_ENVS=1024 \
  NUM_EVAL_ENVS=16 \
  NUM_UPDATES=256 \
  BATCH_SIZE=512 \
  BUFFER_SIZE=300000 \
  TOTAL_TIMESTEPS=5500000 \
  EVAL_FREQ=100000 \
  GPU_LOG_INTERVAL=10 \
  scripts/train_place_6gb_with_logs.sh
```

This is pure visual+qpos because of:

```text
NO_PRIVILEGED_STATE=true
```

Do not use `CHECKPOINT=` for v34. The action scale, sampling range, spacing/exclusion rule, and reward differ from older checkpoints, so this should be a fresh training run.

## 3060 Local Training Command

Use this when limited to about 6 GB VRAM:

```bash
cd /home/chichoo/squint-master6.6winproplace/squint-master

env \
  EXP_NAME=place_xlerobot_v34_stable_pregrasp_range_x027_036_y006_013_64img_12env_8eval_8upd_buf40k_3500k_3060 \
  NO_PRIVILEGED_STATE=true \
  IMAGE_SIZE=64 \
  RENDER_SIZE=128 \
  NUM_ENVS=12 \
  NUM_EVAL_ENVS=8 \
  NUM_UPDATES=8 \
  BATCH_SIZE=48 \
  BUFFER_SIZE=40000 \
  TOTAL_TIMESTEPS=3500000 \
  EVAL_FREQ=100000 \
  GPU_LOG_INTERVAL=10 \
  scripts/train_place_6gb_with_logs.sh
```

If OOM occurs at replay buffer allocation, reduce in this order:

```text
BUFFER_SIZE=30000
BATCH_SIZE=32
NUM_ENVS=8
```

Keep `IMAGE_SIZE=64` for the current visual route unless doing an explicit low-memory comparison.

## Previous 4090 Result

Copied server folder:

```text
/home/chichoo/squint-master6.6winproplace/squint(2)/squint
```

Run:

```text
runs/place_xlerobot_v21_greenbin_wristroll_64img_1024env_16eval_256upd_buf300k_1500k_4090
```

Despite the name containing `1500k`, actual training used:

```text
TOTAL_TIMESTEPS=3500000
```

Result:

```text
Exit status: 0
Peak GPU memory: 14343 MiB
Best step: 3200000
Best success_at_end: 0.25
Best success_once: 0.25
Best return: 16.225
Final success_at_end: 0.06
Final success_once: 0.12
Final return: 9.62
```

Conclusion:

- It did not stably converge.
- Do not continue from the final `ckpt.pt`.
- If resuming that run, use `best_ckpt.pt`.
- Videos were about 2.55 seconds / 51 frames, matching the old 50-step horizon. This is why v27 restored 100 steps.

## Real Robot Deployment Notes

Real deployment has many helper flags in `deploy.py`. They are for testing and diagnosis, not part of pure training.

Useful real-test controls:

```text
--image-size
--max-episode-steps
--action-scale
--gripper-open-steps
--gripper-open-action
--gripper-close-action-limit
--real-crop-shift-x
--real-crop-shift-y
--record-dir
--record-resolution
--log-dir
```

Good real-test setting seen before:

```bash
RUN_NAME=real_place_v17_yminus20_strong_liftassist50_150 \
REAL_CROP_SHIFT_X=-20 \
REAL_CROP_SHIFT_Y=-20 \
GRIPPER_OPEN_STEPS=14 \
GRIPPER_CLOSE_LIMIT=0.15 \
POST_GRASP_ASSIST_START=50 \
POST_GRASP_ASSIST_END=150 \
POST_GRASP_LIFT_ACTION=0.12 \
POST_GRASP_WRIST_ACTION=0.08 \
POST_GRASP_CLOSE_ACTION=-0.08 \
scripts/deploy_place_v11_routeA.sh
```

But this includes assist logic and is not a clean policy-only measurement.

For clean policy evaluation, use the B1 diagnostic script or run `deploy.py` with only image size, gripper-open guard, record/log dirs, and no route-A state override.

## Known Failure Modes To Record

When reviewing eval or real videos, write notes in `runs/<EXP_NAME>/failure_notes.md`.

Use these categories:

```text
unreachable
cannot grasp
grasped but cannot place
camera unclear
bad gripper pose
cube pushed away
early close
no lift after grasp
no bin approach after grasp
```

For every promising run, keep:

- full command
- `train.log`
- `eval_summary.txt`
- eval videos, especially failures
- `best_ckpt.pt`
- `ckpt.pt`
- `gpu_peak.txt`
- real deployment videos/logs if tested

## Why The Robot Pushes The Cube

Likely causes:

- The learned sim action is faster than the real robot can accurately execute.
- Older v21/v26 runs used fast action deltas: arm `0.1`, gripper `0.2`.
- Current v29a uses v28b slow-real action deltas: arm `0.07`, gripper `0.10`.
- Current v34 effective shared sampling range is `x=[0.27,0.36]`, `y=[0.06,0.13]`, kept inside the current top-camera visible area but narrowed from v33 to reduce far/side grasps.
- Current v34 keeps item/bin random within the shared range but requires minimum center distance `0.12m`.
- Current v34 also rejects samples where the item starts inside the bin footprint plus `0.01m` margin.
- Current v34 uses stable pre-grasp shaping with stronger soft push, early-close, and near-fast-action penalties.
- Visual-only policy may approach the cube from a slightly wrong height/angle and collide before the gripper is centered.
- 64x64 helps compared with 32x32, but it does not guarantee millimeter-level alignment.
- If the real camera crop is shifted, the policy's perceived cube center is biased.

Most useful next fixes:

1. Train v34 fresh with slow-real action scale, narrowed visible range plus item/bin spacing/bin-footprint exclusion, stable pre-grasp reward, and 100-step horizon.
2. If cube pushing remains strong, inspect eval videos before increasing penalty weights; avoid making the policy conservative.
3. Keep 64x64 input for now.
4. Keep green bin if the real bin material is also green and visually distinct from black table and white robot.

## Do Not Accidentally Revert

Do not revert these unless explicitly testing a baseline:

- `xlerobot_right_head` robot support
- green bin
- real deploy crop shift and diagnostic video logic
- `best_ckpt.pt` saving on better `success_at_end`
- pure visual `--no-privileged-state` route
- safe RNG fixes for `NUM_EVAL_ENVS=16`
- 100-step Place horizon

## Version Log Location

Detailed change history is in:

```text
docs/place_experiment_versions.md
```

When changing code or training setup, add a new version entry there with:

- changed files
- command
- run directory
- best/final eval
- peak GPU memory
- observed failure modes
- decision
