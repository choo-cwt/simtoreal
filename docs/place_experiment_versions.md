# Place Experiment Versions

This document records deliberate changes for XLeRobot place-task experiments.
Append a new version whenever training parameters, environment sampling, reward,
robot setup, camera setup, or evaluation procedure changes.

## v1 - 6GB Baseline, Wide Place Sampling

Date: 2026-06-08

Purpose:
- Establish a first `SO101PlaceCube-v1` baseline for `xlerobot_right_head`.
- Keep the task code unchanged.
- Keep Squint acceleration enabled.

Training command wrapper:

```bash
./scripts/train_place_6gb_with_logs.sh
```

Expanded command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLCONFIGDIR=/tmp/mpl python train_squint.py \
  --env-id=SO101PlaceCube-v1 \
  --robot-uids=xlerobot_right_head \
  --control-mode=pd_joint_target_delta_pos \
  --num-envs=32 \
  --num-eval-envs=4 \
  --num-updates=8 \
  --batch-size=128 \
  --buffer-size=200000 \
  --render-size=128 \
  --image-size=16 \
  --total-timesteps=1500000 \
  --eval-freq=100000 \
  --exp-name=place_xlerobot_cube_topcam_6gb_32env_8upd_200kbuf
```

Important settings:
- `compile=True`
- `cudagraphs=True`
- `capture_video=True`
- `obs_mode=rgb+segmentation`
- `apply_jitter=True`
- `domain_randomization=True`

Place sampling in v1:
- `spawn_box_pos=[0.3, 0]`
- `spawn_box_half_size=0.1`
- XLeRobot base position: `[0.05, 0, 0.068]`
- Effective item/bin sampling center around XY `[0.35, 0]`
- Effective item/bin XY range before non-overlap sampling:
  - x: `[0.25, 0.45]`
  - y: `[-0.10, 0.10]`

Result summary:
- Run directory: `runs/place_xlerobot_cube_topcam_6gb_32env_8upd_200kbuf`
- Exit status: `0`
- Peak GPU memory: `4242 MiB`
- Final eval at 1.5M: `success_at_end=0.50`, `success_once=0.50`, `return=14.49`
- Eval uses only 4 envs, so success values are coarse: `0.00`, `0.25`, `0.50`, `0.75`, `1.00`.

Observation:
- The policy has learned partial behavior, but success is not stable.
- Some eval episodes appear to place item/bin positions where the arm cannot reliably reach or grip.
- Because v1 uses a wide XY sampling region for both item and bin, the learning problem may include too many difficult edge cases early.

## v2 - Narrow Reachable Sampling

Date: 2026-06-08
Status: applied, not yet trained.

Goal:
- Reduce early training difficulty by narrowing the item/bin sampling region.
- Preserve the same robot, camera, controller, reward, and algorithm settings.
- Keep this as a controlled environment-distribution change from v1.

Environment change:
- Changed default XLeRobot place sampling from:

```python
spawn_box_pos=[0.3, 0]
spawn_box_half_size=0.2 / 2
```

- To a narrower XLeRobot-specific distribution:

```python
spawn_box_pos=[0.275, 0.0]
spawn_box_half_size=[0.075, 0.08]
```

Expected effective XY range with XLeRobot base `[0.05, 0, 0.068]`:
- x: `[0.25, 0.40]`
- y: `[-0.08, 0.08]`

Reasoning:
- Keeps the task randomized.
- Removes the farthest `x=0.45` and `|y|=0.10` edge cases from early training.
- Should improve the fraction of episodes where the arm can actually reach and close around the cube.

Tradeoff:
- Easier simulation distribution.
- Policy may transfer less well to objects placed outside this narrower range.
- If v2 succeeds, later versions should gradually widen the range again.

Suggested v2 run name:

```text
place_xlerobot_cube_topcam_v2_range_x025_040_y008_32env_8eval_3500k
```

Suggested v2 training command:

```bash
EXP_NAME=place_xlerobot_cube_topcam_v2_range_x025_040_y008_32env_8eval_3500k \
./scripts/train_place_6gb_with_logs.sh
```

Files changed:
- `envs/place.py`
- `docs/place_experiment_versions.md`

## v3 - Best Checkpoint Saving

Date: 2026-06-08
Status: applied, not yet trained.

Goal:
- Prevent losing the best policy when later training degrades performance.
- Keep the normal `ckpt.pt` behavior unchanged.

Changed from previous version:
- `evaluate()` now returns the eval metrics dictionary.
- After each eval, `train_squint.py` still saves the latest checkpoint to `ckpt.pt`.
- If `eval/success_at_end` is higher than every previous eval in the same run, it also saves:
  - `best_ckpt.pt`
  - `best_metrics.txt`

Files changed:
- `train_squint.py`
- `docs/place_experiment_versions.md`

Notes:
- This does not change the training algorithm, reward, environment, camera, or robot.
- The previous 3M resume run reached `success_at_end=0.88` mid-run but ended at `0.25`; without best checkpoint saving, that better model was overwritten.

## v4 - Default Eval Envs 8

Date: 2026-06-08
Status: applied, not yet trained.

Goal:
- Match the requested eval visualization with 8 parallel evaluation environments.
- Make success metrics less coarse than 4 eval envs.

Changed from previous version:
- `scripts/train_place_6gb_with_logs.sh` default changed from:

```bash
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-4}"
```

- To:

```bash
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-8}"
```

Files changed:
- `scripts/train_place_6gb_with_logs.sh`
- `docs/place_experiment_versions.md`

Notes:
- Training envs remain `NUM_ENVS=32`.
- Update ratio remains paper-like for the 32-env training setup: `NUM_UPDATES=8`.
- Eval envs do not change policy learning directly; they improve evaluation coverage and video visualization.

## v5 - Default Total Timesteps 3.5M

Date: 2026-06-08
Status: trained.

Goal:
- Run longer by default for the narrowed v2 place distribution.
- Preserve best checkpoint saving so mid-run policy quality is not lost if late training degrades.

Changed from previous version:
- `scripts/train_place_6gb_with_logs.sh` default changed from:

```bash
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1500000}"
```

- To:

```bash
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3500000}"
```

Files changed:
- `scripts/train_place_6gb_with_logs.sh`
- `docs/place_experiment_versions.md`

Suggested v5 training command:

```bash
EXP_NAME=place_xlerobot_cube_topcam_v2_range_x025_040_y008_32env_8eval_3500k \
./scripts/train_place_6gb_with_logs.sh
```

Notes:
- This changes the wrapper script default only; `train_squint.py` still has its original default.
- With `NUM_ENVS=32`, `NUM_UPDATES=8`, and `TOTAL_TIMESTEPS=3500000`, the update-per-env-step ratio remains `0.25`.

Run result:
- Run directory: `runs/place_xlerobot_cube_topcam_v5_from_v2_ckpt_32env_8eval_3500k`
- Checkpoint initialized from: `runs/place_xlerobot_cube_topcam_v2_range_x025_040_y008_32env_8upd_1500k/ckpt.pt`
- Exit status: `0`
- Peak GPU memory: `4334 MiB`
- Best eval: `success_at_end=0.875`, `success_once=0.875`, `return=32.2749`
- Best step within this run: `0`
- Final eval: `success_at_end=0.50`, `success_once=0.50`, `return=16.40`
- Saved best checkpoint: `runs/place_xlerobot_cube_topcam_v5_from_v2_ckpt_32env_8eval_3500k/best_ckpt.pt`

Interpretation:
- The v2 checkpoint was already strong when v5 started.
- Additional 3.5M training steps did not improve the best success rate and reduced final policy quality.
- Use `best_ckpt.pt` from this run for evaluation/deployment-style testing, not the final `ckpt.pt`.

## v6 - Real Deployment Slow Gripper Close

Date: 2026-06-08
Status: applied, real deployment only.

Goal:
- Address real-robot failure where the policy reaches near the cube but closes the gripper too early/too fast.
- Keep the trained model and simulation task unchanged.

Observed real-robot runs:
- Run directory: `runs/real_place_v5_best_safe`
- Run directory: `runs/real_place_v5_best_try_grip35`
- In both runs the policy commands a strong negative gripper action as soon as forced-open steps end.
- With `gripper_open_steps=35`, first close command happens at step 35 and immediately reaches about `scaled_action_5=-0.15`.
- Video shows the gripper is near the cube but not reliably wrapped around it before closing.

Changed from previous version:
- Added `deploy.py` argument:

```python
gripper_close_action_limit: float = 0.0
```

- When set above `0`, negative gripper close commands are capped after the forced-open period:

```python
scaled_action[..., 5] = np.maximum(
    scaled_action[..., 5], -args.gripper_close_action_limit
)
```

Files changed:
- `deploy.py`
- `scripts/deploy_place_b1_diag_policy_input.sh`
- `docs/place_experiment_versions.md`

Notes:
- This does not change training, reward, checkpoint, camera, robot, or environment sampling.
- This is an execution-time real-robot guard to slow gripper closure.
- Recommended first test value: `--gripper-close-action-limit=0.04`.
- `deploy.py` prints deployment safety settings at startup so the active close limit is visible in the terminal.

## v7 - Real Deployment Timed Post-Grasp Assist

Date: 2026-06-08
Status: applied, real deployment only.

Goal:
- Diagnose and reduce the real-robot failure where the policy sometimes grasps the cube but then does not lift or move toward the bin.
- Keep the trained model and simulation task unchanged.

Observed real-robot behavior:
- Slower gripper close allows occasional grasps depending on cube pose.
- After a grasp-like state, the arm often stays near the cube instead of lifting and searching for the white bin.
- `Wrist_Roll_R` is intentionally fixed by the training controller (`delta_lower/upper=0.0`), so wrist roll cannot correct grasp orientation during policy execution.

Changed from previous version:
- Added `deploy.py` argument:

```python
post_grasp_assist_end: int = -1
```

- Post-grasp lift/wrist/close assist can now be limited to a time window:

```python
post_grasp_assist_start <= step_idx < post_grasp_assist_end
```

Files changed:
- `deploy.py`
- `docs/place_experiment_versions.md`

Notes:
- This is a deployment-time diagnostic/guard, not a training change.
- Use a short assist window to test whether lifting after grasp lets the policy transition toward the bin.
- Keeping assist active forever can prevent the final release/place phase.

## v8 - Raw Real-Camera Diagnostic Recording

Date: 2026-06-08
Status: applied, real deployment diagnostics only.

Goal:
- Make real-robot videos useful for diagnosing millimeter-level grasp failures.
- Keep the policy input, checkpoint, and robot actions unchanged.

Changed from previous version:
- `deploy.py` now stores the cropped raw camera frame before resizing it to the model input size.
- Episode recordings now prefer this raw pre-resize frame when `--record-dir` is enabled.
- The policy still receives the configured `--image-size` input, for example 16x16 for the current v5 checkpoint.

Files changed:
- `deploy.py`
- `docs/place_experiment_versions.md`

Notes:
- Previous real deployment videos were recorded from the already-resized model observation, so with `--image-size=16` they looked like enlarged 16x16 images.
- v8 only improves recording clarity. It does not improve policy perception by itself.
- Use this to distinguish early close, rear-side approach, unstable grasp, and grasped-but-no-lift failures.

## v9 - 32x32 Visual Training and Deploy Compatibility

Date: 2026-06-08
Status: trained, ready for real deployment test.

Goal:
- Test whether a higher policy input resolution improves real-robot grasp centering and place-stage robustness.
- Keep task range, reward, action space, and real deploy safety parameters otherwise unchanged.

Changed from previous version:
- Trained a new 32x32 input policy from scratch. The 16x16 v5 checkpoint cannot be directly resumed into 32x32 because the CNN encoder shape changes.
- Fixed `deploy.py` to construct `DeployAgent(..., target_image_size=args.image_size, device=device)`, so 32x32 checkpoints load with `--image-size=32`.

Files changed:
- `deploy.py`
- `docs/place_experiment_versions.md`

Training command:
```bash
EXP_NAME=place_xlerobot_cube_topcam_v9_32img_6gb_32env_8eval_3500k \
IMAGE_SIZE=32 \
BATCH_SIZE=96 \
BUFFER_SIZE=120000 \
NUM_ENVS=32 \
NUM_EVAL_ENVS=8 \
NUM_UPDATES=8 \
TOTAL_TIMESTEPS=3500000 \
EVAL_FREQ=100000 \
scripts/train_place_6gb_with_logs.sh
```

Run directory:
- `runs/place_xlerobot_cube_topcam_v9_32img_6gb_32env_8eval_3500k`

Result summary:
- Best eval: step 1300000, `success_at_end=0.875`, `success_once=1.0`, `return=32.1277`
- Best checkpoint: `runs/place_xlerobot_cube_topcam_v9_32img_6gb_32env_8eval_3500k/best_ckpt.pt`

Decision:
- Test v9 best on the real robot with `--image-size=32`.

## v10 - Real Deploy Auto Stall Assist

Date: 2026-06-08
Status: applied, real deployment only.

Goal:
- Address the v9 real-robot behavior where the policy can grasp or partially grasp the cube, then outputs near-zero actions and does not lift/place.
- Avoid unconditional timed lift assist before a grasp-like state.

Observed real-robot behavior:
- `real_place_v9_best_32img_open15_noassist/episode_7` shows strong close commands during steps 15-60, then near-zero arm and gripper actions from about step 60 onward.
- The robot appears to hold or contact the cube but does not lift. CSV confirms qpos stays nearly fixed after the stall.
- The ManiSkill control dt warning is small and not the primary cause of the no-lift behavior.

Changed from previous version:
- Added optional `deploy.py` arguments:

```python
auto_stall_assist: bool = False
stall_assist_gripper_qpos: float = 0.45
stall_assist_action_norm: float = 0.025
stall_assist_min_step: int = 50
stall_assist_duration: int = 100
```

- When enabled, deploy triggers post-grasp assist only if:
  - current step is at least `stall_assist_min_step`
  - gripper qpos is below `stall_assist_gripper_qpos`
  - scaled arm action norm is below `stall_assist_action_norm`
- CSV logs now include `assist_active` so triggered assist windows can be inspected later.

Files changed:
- `deploy.py`
- `docs/place_experiment_versions.md`

Notes:
- This is a diagnostic deployment assist, not a training change.
- Use it to test whether the v9 policy can continue toward the bin after being lifted out of the stalled grasp state.

## v11 - Route A Privileged State Override for Real Place Validation

Date: 2026-06-09
Status: applied, real deployment validation only.

Goal:
- Validate whether the current v9 policy can complete the place stage on real hardware when the privileged `item/bin` state is made consistent with the real setup.
- Keep this as a bridge toward a later Route B visual-only policy.

Finding that motivated this change:
- Simulation success is not purely visual. The training/deploy observation uses `state=True`, and `envs/place.py` includes privileged fields such as `item_pose`, `bin_pose`, `tcp_to_bin_pos`, and `item_to_bin_pos`.
- In real deployment, `Sim2RealEnv` reuses the simulation environment's `_get_obs_extra`; without an explicit real perception pipeline, these object/bin poses can be shadow-sim values rather than true real-world positions.
- This explains why simulation can place while real deployment can grasp/lift but not reliably move toward the white bin.

Changed from previous version:
- Added optional `deploy.py` route A arguments:

```python
route_a_state_override: bool = False
route_a_item_xy: tuple[float, float] = (0.325, 0.0)
route_a_override_item_before_grasp: bool = False
route_a_bin_xy: tuple[float, float] = (0.225, 0.0)
route_a_follow_tcp_after_grasp: bool = True
route_a_grasp_qpos: float = 0.45
route_a_tcp_item_z_offset: float = -0.015
```

- When enabled, deployment overwrites the shadow-sim item/bin poses before policy inference.
- After the gripper closes below `route_a_grasp_qpos`, the shadow item pose follows the simulated TCP so `item_to_bin_pos` and related privileged state move with the held object.
- Fixed route A observation refresh to return wrapper-formatted observations (`rgb` and `state`) instead of raw unflattened observations. The first route A script run failed with `KeyError: 'state'` before this fix.

Files changed:
- `deploy.py`
- `scripts/deploy_place_v11_routeA.sh`
- `docs/place_experiment_versions.md`

Notes:
- This is not the final sim2real solution. It deliberately restores privileged state consistency to verify that the place controller can work.
- Final Route B should remove object/bin privileged state from the policy input or replace it with a real perception pipeline.
- Use `scripts/deploy_place_v11_routeA.sh` for the default real deployment command. Override coordinates with `ITEM_X`, `ITEM_Y`, `BIN_X`, and `BIN_Y`.
- 2026-06-09 update: route A deploy script default changed to `--action-scale=0.15` and `--control-freq=30` for a faster paper-like real test.
- 2026-06-09 update: route A deploy script no longer fixes the red cube position by default. The cube can be placed freely and is only approximated by TCP after grasp; only the white bin center remains fixed through `BIN_X` and `BIN_Y`.
- 2026-06-09 update: initial gripper-open action increased from `0.12` to `0.20`; auto stall assist now triggers once per episode by default and the route A script shortens assist duration from `110` to `45` steps to avoid clamping the gripper closed through the release/place phase.
- 2026-06-09 update: latest v11 videos show grasp localization is still poor, with gripper closure starting at step 35 before the jaws are centered on the cube. Route A script defaults changed to `CONTROL_FREQ=20`, `ACTION_SCALE=0.08`, `GRIPPER_OPEN_STEPS=55`, and `GRIPPER_CLOSE_LIMIT=0.025`. The previous fast settings can still be restored with environment variables.

## v12 - Route A No Gripper Guard Test

Date: 2026-06-09
Status: applied, real deployment test only.

Goal:
- Test the raw policy gripper timing without deployment-time forced-open or close-limit guards.
- Keep route A state override, white-bin coordinate override, auto stall assist, checkpoint, image size, and controller settings unchanged.

Changed from previous version:
- `scripts/deploy_place_v11_routeA.sh` default run name changed from:

```bash
real_place_v11_routeA_state_override
```

- To:

```bash
real_place_v12_routeA_no_gripper_guard
```

- Disabled forced gripper opening by default:

```bash
GRIPPER_OPEN_STEPS=0
GRIPPER_OPEN_ACTION=0.0
```

- Disabled close-command limiting by default:

```bash
GRIPPER_CLOSE_LIMIT=0.0
```

- Set route A deploy speed back to the faster test setting:

```bash
CONTROL_FREQ=30
ACTION_SCALE=0.15
```

- Added deploy-time real camera crop shift controls for diagnosing systematic left/right visual bias:

```bash
--real-crop-shift-x
--real-crop-shift-y
```

- The route A script exposes them as:

```bash
REAL_CROP_SHIFT_X=0
REAL_CROP_SHIFT_Y=0
```

Files changed:
- `deploy.py`
- `scripts/deploy_place_v11_routeA.sh`
- `docs/place_experiment_versions.md`

Run command:

```bash
scripts/deploy_place_v11_routeA.sh
```

Notes:
- This does not remove the deploy.py options; it only changes this route A script's defaults.
- If the gripper closes too early again, compare `runs/real_place_v12_routeA_no_gripper_guard/logs/*.csv` with the prior guarded run to see whether the raw policy's `action_5` begins closing before the jaws are centered.
- If the robot consistently approaches to the same side of the cube, test a small real crop shift rather than retraining immediately. Positive `REAL_CROP_SHIFT_X` crops farther right from the 640x480 camera frame; negative values crop farther left.

## v13 - Route A 200 Step Crop-Tune Test

Date: 2026-06-09
Status: applied, real deployment test only.

Goal:
- Shorten real deployment episodes to 200 steps while testing the `REAL_CROP_SHIFT_X=-20` crop-shift alignment and moderate gripper timing.

Changed from previous version:
- Route A deploy script default run name changed to:

```bash
real_place_v13_routeA_200step_crop_tune
```

- Route A deploy script default max episode length changed to:

```bash
MAX_EPISODE_STEPS=200
```

- Route A deploy script default gripper settings use the moderate timing test:

```bash
GRIPPER_OPEN_STEPS=15
GRIPPER_OPEN_ACTION=0.20
GRIPPER_CLOSE_LIMIT=0.08
```

Files changed:
- `scripts/deploy_place_v11_routeA.sh`
- `docs/place_experiment_versions.md`

Run command:

```bash
REAL_CROP_SHIFT_X=-20 scripts/deploy_place_v11_routeA.sh
```

Notes:
- At `CONTROL_FREQ=30`, 200 total episode steps is about 6.7 seconds.
- The crop shift remains configurable and defaults to `0`; pass `REAL_CROP_SHIFT_X=-20` to reproduce the previous improved alignment test.

## v14 - Crop Left 20, Fast Close 15

Date: 2026-06-09
Status: applied, real deployment test.

Goal:
- Keep the improved left crop alignment and restore fast enough gripper closing after v13 showed `GRIPPER_CLOSE_LIMIT=0.08` was too slow.

Observed result:
- User reported this version has much higher grasp success.
- Remaining issue: grasp point is still slightly toward the cube tail/end rather than centered.

Changed from previous version:
- Route A deploy script default run name changed to:

```bash
real_place_v14_crop_left20_open15_close15
```

- Route A deploy script defaults changed to:

```bash
REAL_CROP_SHIFT_X=-20
GRIPPER_OPEN_STEPS=15
GRIPPER_OPEN_ACTION=0.20
GRIPPER_CLOSE_LIMIT=0.15
MAX_EPISODE_STEPS=200
```

Files changed:
- `scripts/deploy_place_v11_routeA.sh`
- `docs/place_experiment_versions.md`

Run command:

```bash
scripts/deploy_place_v11_routeA.sh
```

Notes:
- `GRIPPER_CLOSE_LIMIT=0.15` fixes the v13 failure where the gripper closed too slowly and reached full closure only after the hand had moved past the cube.
- Next small tests should keep `GRIPPER_CLOSE_LIMIT=0.15` fixed and only tune crop shift or `GRIPPER_OPEN_STEPS`.

## v15 - Open 14 Best Grasp, Post-Grasp Assist Configurable

Date: 2026-06-09
Status: applied, real deployment test.

Goal:
- Use the best observed grasp timing from real tests.
- Make post-grasp lift/hold assist configurable from the route A script for diagnosing grasped-but-no-lift and no-bin-search failures.

Observed result:
- `GRIPPER_OPEN_STEPS=14`, `REAL_CROP_SHIFT_X=-20`, and `GRIPPER_CLOSE_LIMIT=0.15` produced the best grasp success so far.
- Remaining issue: after a successful grasp, the robot often does not lift or move toward the white bin.
- Placing the white bin visibly in the camera view can change the approach trajectory and reduce grasp reliability, suggesting the visual policy is sensitive to the white object/background distribution.

Changed from previous version:
- Route A deploy script default gripper timing changed to:

```bash
GRIPPER_OPEN_STEPS=14
```

- Added route A script environment variables for post-grasp assist:

```bash
POST_GRASP_ASSIST_START=-1
POST_GRASP_ASSIST_END=-1
POST_GRASP_LIFT_ACTION=0.05
POST_GRASP_WRIST_ACTION=0.03
POST_GRASP_CLOSE_ACTION=-0.04
```

- Fixed `REAL_CROP_SHIFT_Y` so it applies a post-crop vertical image shift. This makes vertical/approach-direction visual bias tests possible even when the RealSense frame is 640x480 and the square crop only removes horizontal pixels.
- Route A deploy script default `MAX_EPISODE_STEPS` changed from `200` to `300` after the strong lift assist successfully lifted the cube and more post-grasp time was needed to evaluate place behavior.

Files changed:
- `deploy.py`
- `scripts/deploy_place_v11_routeA.sh`
- `docs/place_experiment_versions.md`

Recommended grasp-only command:

```bash
scripts/deploy_place_v11_routeA.sh
```

Recommended post-grasp diagnostic command:

```bash
RUN_NAME=real_place_v15_open14_liftassist55_100 \
POST_GRASP_ASSIST_START=55 \
POST_GRASP_ASSIST_END=100 \
POST_GRASP_LIFT_ACTION=0.06 \
POST_GRASP_WRIST_ACTION=0.04 \
POST_GRASP_CLOSE_ACTION=-0.04 \
scripts/deploy_place_v11_routeA.sh
```

Notes:
- Keep the white bin out of the camera view during grasp-only tests. If the bin must be present, place it at the fixed route A bin coordinate but as far left/out of the central crop as possible until grasp is stable.
- The current route A state override supplies the bin state numerically; the visual white bin can still perturb the policy because the model also uses RGB input.

## v16 - Scripted Route A Place Assist

Date: 2026-06-09
Status: applied, real deployment validation only.

Goal:
- Validate whether the real hardware can carry a grasped cube toward the configured white bin and release it.
- Keep this separate from policy quality; this is a scripted diagnostic assist, not the final learned solution.

Finding that motivated this change:
- Strong post-grasp lift assist can lift the cube.
- After lift assist ends, the policy often outputs actions that lower/return the arm instead of moving toward the bin.
- Increasing episode length to 300 steps did not fix this, so the bottleneck is policy transition into place behavior, not time.

Changed from previous version:
- Added deploy.py arguments:

```python
route_a_place_assist
route_a_place_assist_carry_start
route_a_place_assist_release_start
route_a_place_assist_end
route_a_place_assist_base_action
route_a_place_assist_elbow_action
route_a_place_assist_lift_action
route_a_place_assist_wrist_action
route_a_place_assist_close_action
route_a_place_assist_release_action
```

- Added matching route A script environment variables:

```bash
ROUTE_A_PLACE_ASSIST=false
ROUTE_A_PLACE_ASSIST_CARRY_START=150
ROUTE_A_PLACE_ASSIST_RELEASE_START=240
ROUTE_A_PLACE_ASSIST_END=285
ROUTE_A_PLACE_ASSIST_BASE_ACTION=0.08
ROUTE_A_PLACE_ASSIST_ELBOW_ACTION=-0.04
ROUTE_A_PLACE_ASSIST_LIFT_ACTION=0.04
ROUTE_A_PLACE_ASSIST_WRIST_ACTION=0.03
ROUTE_A_PLACE_ASSIST_CLOSE_ACTION=-0.06
ROUTE_A_PLACE_ASSIST_RELEASE_ACTION=0.18
```

Files changed:
- `deploy.py`
- `scripts/deploy_place_v11_routeA.sh`
- `docs/place_experiment_versions.md`

Recommended first test:

```bash
RUN_NAME=real_place_v19_bin_y08_placeassist_pos \
BIN_X=0.30 \
BIN_Y=0.08 \
REAL_CROP_SHIFT_X=-20 \
REAL_CROP_SHIFT_Y=-20 \
GRIPPER_OPEN_STEPS=14 \
GRIPPER_CLOSE_LIMIT=0.15 \
POST_GRASP_ASSIST_START=50 \
POST_GRASP_ASSIST_END=150 \
POST_GRASP_LIFT_ACTION=0.12 \
POST_GRASP_WRIST_ACTION=0.08 \
POST_GRASP_CLOSE_ACTION=-0.08 \
ROUTE_A_PLACE_ASSIST=true \
scripts/deploy_place_v11_routeA.sh
```

If the carry direction is wrong, flip:

```bash
ROUTE_A_PLACE_ASSIST_BASE_ACTION=-0.08
```

Notes:
- The CSV `assist_active` field is also set during scripted place assist.
- This assist does not use perception to servo to the bin; it only tests a fixed carry/release motion after the successful lift stage.

## v17 - B1 Visual+Robot-State Place Training

Date: 2026-06-09
Status: applied, ready to train.

Goal:
- Start route B1: train a place policy that cannot observe privileged object/bin state.
- Keep reward computation unchanged, but restrict policy observations to RGB plus robot/self state.

Changed from previous version:
- Added `privileged_state=True` environment argument to `envs/place.py`.
- When `privileged_state=False`, place observations remove:

```text
is_item_grasped
item_pose
bin_pose
tcp_to_item_grip_pos
tcp_to_bin_pos
item_to_bin_pos
item_dimensions
bin_dimensions
item_friction
item_density
```

- The policy still observes:

```text
noisy_qpos
controller state
qvel
tcp_pose
gripper_stiffness/gripper_damping when domain randomization is enabled
RGB/segmentation image
```

- Added train/deploy flag:

```bash
--no-privileged-state
```

- Added training script environment variable:

```bash
NO_PRIVILEGED_STATE=true
```

Files changed:
- `envs/place.py`
- `train_squint.py`
- `deploy.py`
- `scripts/train_place_6gb_with_logs.sh`
- `docs/place_experiment_versions.md`

Recommended first B1 training command:

```bash
EXP_NAME=place_xlerobot_b1_visual_qpos_32img_32env_8eval_3500k \
NO_PRIVILEGED_STATE=true \
IMAGE_SIZE=32 \
BATCH_SIZE=96 \
BUFFER_SIZE=120000 \
NUM_ENVS=32 \
NUM_EVAL_ENVS=8 \
NUM_UPDATES=8 \
TOTAL_TIMESTEPS=3500000 \
EVAL_FREQ=100000 \
scripts/train_place_6gb_with_logs.sh
```

Notes:
- Existing v9/v14 checkpoints are not compatible with B1 because the state dimension changes.
- First success may be lower/slower than privileged-state training. This run tests whether visual+robot-state alone can learn the full place behavior in simulation.
- Use the resulting B1 checkpoint with deploy flag `--no-privileged-state`; otherwise actor state dimensions will not match.

## v18 - Real Policy-Input Diagnostic Video

Date: 2026-06-09
Status: applied, real deployment diagnostics only.

Goal:
- Inspect the exact low-resolution RGB image used by the deployed policy.
- Avoid guessing from the high-resolution recording when the policy actually sees 32x32 input.

Changed from previous version:
- Added `deploy.py` arguments:

```bash
--policy-input-record-dir
--policy-input-record-resolution
```

- The real camera preprocessor now stores the post-crop, post-resize policy RGB frame before converting it to a tensor.
- Deployment can save a second per-episode video stream with those exact policy frames upscaled for inspection.
- This does not change observations, actions, rewards, checkpoints, crop settings, or any control behavior.

Files changed:
- `deploy.py`
- `docs/place_experiment_versions.md`

Recommended B1 diagnostic deploy command:

```bash
scripts/deploy_place_b1_diag_policy_input.sh
```

Equivalent direct deploy command:

```bash
python deploy.py --checkpoint=runs/place_xlerobot_b1_visual_qpos_32img_32env_8eval_3500k/best_ckpt.pt --env-id=SO101PlaceCube-v1 --robot-uids=xlerobot_right_head --control-mode=pd_joint_target_delta_pos --image-size=32 --no-privileged-state --max-episode-steps=300 --control-freq=30 --action-scale=0.15 --gripper-open-steps=14 --gripper-open-action=0.20 --gripper-close-action-limit=0.15 --post-grasp-assist-start=-1 --record-dir=runs/real_place_b1_diag_policy_input/videos --record-resolution=480 --policy-input-record-dir=runs/real_place_b1_diag_policy_input/policy_input_videos --policy-input-record-resolution=256 --log-dir=runs/real_place_b1_diag_policy_input/logs
```

Next analysis:
- Compare `videos/episode_*.mp4` against `policy_input_videos/episode_*.mp4`.
- Check whether the red cube and white bin are still separable in the actual 32x32 policy input during approach, grasp, and post-grasp.
- If the red cube or bin is not visible enough, prefer training-side camera/domain randomization or 64x64 input over further fixed crop offsets.

## v19 - Green Bin and Slower XLeRobot Sim Control

Date: 2026-06-10
Status: applied, requires new training run.

Goal:
- Make the bin visually distinct from the white robot and black table in real deployment.
- Reduce the sim-to-real mismatch where simulation learns fast one-shot grasps that the real robot cannot reproduce reliably.
- Give the policy more control steps to recover from imperfect approach and grasp attempts.

Changed from previous version:
- Changed the Place task bin visual material from white to green:

```python
base_color=[0.0, 0.8, 0.1, 1.0]
```

- Slowed the XLeRobot `pd_joint_delta_pos` and `pd_joint_target_delta_pos` action deltas:

```python
arm delta:     +/-0.10 -> +/-0.06 rad / control step
gripper delta: +/-0.20 -> +/-0.08 rad / control step
```

- Increased Place task horizon:

```python
SO101PlaceCube-v1: 50 -> 100 steps
SO101PlaceCan-v1:  50 -> 100 steps
```

Files changed:
- `envs/place.py`
- `envs/robot/xlerobot.py`
- `docs/place_experiment_versions.md`

Notes:
- This changes the training environment. Do not compare new v19 checkpoints directly against older white-bin/fast-control checkpoints without noting the environment change.
- Old checkpoints can still be loaded mechanically, but their learned timing and visual target color are from the previous environment.
- At the default sim control frequency of 10 Hz, 100 steps corresponds to about 10 seconds, closer to the real deployment horizon used in recent tests.

Recommended first v19 training command:

```bash
env \
  EXP_NAME=place_xlerobot_v19_greenbin_slowctrl_32img_32env_8eval_3500k \
  NO_PRIVILEGED_STATE=true \
  IMAGE_SIZE=32 \
  BATCH_SIZE=96 \
  BUFFER_SIZE=120000 \
  NUM_ENVS=32 \
  NUM_EVAL_ENVS=8 \
  NUM_UPDATES=8 \
  TOTAL_TIMESTEPS=3500000 \
  EVAL_FREQ=100000 \
  scripts/train_place_6gb_with_logs.sh
```

If GPU memory is tight, reduce `BATCH_SIZE=64` before reducing `NUM_ENVS`.

## v20 - Restore Paper-Style Place Timing and Unlock Wrist Roll

Date: 2026-06-10
Status: applied

Goal:
- Compare against the untouched `squint-master-0` baseline and undo recent training-environment changes that made the simulator less paper-like.
- Keep the changes needed for the current XLeRobot/real-robot workflow: green bin, top/head camera alignment, XLeRobot robot registration, real deploy mapping/logging, visual+qpos/no-privileged-state support, and existing `runs/` checkpoints.

Changed from v19:
- Restored XLeRobot `pd_joint_delta_pos` and `pd_joint_target_delta_pos` action deltas to the SO101 paper-style values:

```python
arm delta:     +/-0.06 -> +/-0.10 rad / control step
gripper delta: +/-0.08 -> +/-0.20 rad / control step
```

- Re-enabled the fifth arm action (`Wrist_Roll_R`) in the XLeRobot controller:

```python
Wrist_Roll_R delta: 0.0 -> +/-0.10
Wrist_Roll_R velocity action: 0.0 -> +/-1.0
```

- Restored Place task horizon to the original value:

```python
SO101PlaceCube-v1: 100 -> 50 steps
SO101PlaceCan-v1:  100 -> 50 steps
```

Files changed:
- `envs/robot/xlerobot.py`
- `envs/place.py`
- `docs/place_experiment_versions.md`

Notes:
- The green bin remains enabled.
- The top/head camera path remains enabled.
- The pure visual+qpos option remains enabled through `--no-privileged-state`.
- Existing checkpoints and videos under `runs/` were not changed.

## v21 - Explicit Place Randomization Sizes

Date: 2026-06-10
Status: applied

Problem:
- On the cloud server, the 4090 run failed during the initial eval reset with:

```text
IndexError: index 15 is out of bounds for axis 0 with size 15
```

- The run used `NUM_EVAL_ENVS=16`, but one domain-randomized item property array was created with length 15.

Changed from v20:
- In `envs/place.py`, all Place item/bin domain-randomization samples now explicitly request `size=(self.num_envs,)`.
- Randomized item colors now explicitly request `size=(self.num_envs, 3)`.

Files changed:
- `envs/place.py`
- `docs/place_experiment_versions.md`

Notes:
- This is a bug fix for vectorized env reset/reconfiguration. It does not change the intended randomization ranges.
- Existing checkpoints and videos under `runs/` were not changed.

## v22 - Per-Env Place Randomization Sampling

Date: 2026-06-10
Status: applied

Problem:
- The first v21 fix used `size=(self.num_envs,)` with ManiSkill's batched episode RNG.
- That produced a full vector for each env, so SAPIEN received arrays instead of scalar friction values:

```text
TypeError: PhysxMaterial(static_friction: float, dynamic_friction: float, restitution: float)
Invoked with static_friction=array(... shape=(1024,))
```

Changed from v21:
- Added `Place._uniform_per_env(...)`.
- Item dimensions, item friction/density, item colors, and bin dimensions now sample one value per env by indexing `self._batched_episode_rng[i]`.

Files changed:
- `envs/place.py`
- `docs/place_experiment_versions.md`

Notes:
- This keeps domain randomization enabled and preserves the intended ranges.
- Existing checkpoints and videos under `runs/` were not changed.

## v23 - Tolerate Short Batched RNG Lists

Date: 2026-06-10
Status: applied

Problem:
- On the cloud server, initial eval used `NUM_EVAL_ENVS=16`, but `self._batched_episode_rng.rngs` contained only 15 generators.
- The v22 per-env sampling helper indexed generator 15 and failed:

```text
IndexError: list index out of range
```

Changed from v22:
- `Place._uniform_per_env(...)` now checks the actual number of per-env RNGs.
- If ManiSkill provides fewer RNGs than `self.num_envs`, the helper cycles through the available RNGs until it returns exactly `self.num_envs` samples.
- Added a fallback path for RNG objects without a `.rngs` list.

Files changed:
- `envs/place.py`
- `docs/place_experiment_versions.md`

Notes:
- This is still only a domain-randomization sampling bug fix.
- Existing checkpoints and videos under `runs/` were not changed.

## v24 - Lighting Randomization RNG Length Fix

Date: 2026-06-10
Status: applied

Problem:
- A cloud-server log showed the same short batched-RNG issue in lighting randomization:

```text
File "/home/gpu/squint/envs/base_random_env.py", line 176, in _load_lighting
scene.render_system.ambient_light = ambient_colors[i]
IndexError: index 15 is out of bounds for axis 0 with size 15
```

- `NUM_EVAL_ENVS=16`, but `ambient_colors` had length 15.

Changed from v23:
- Added `BaseRandomEnv._uniform_per_sub_scene(...)`.
- `_load_lighting(...)` now samples one ambient color per actual sub-scene and tolerates a shorter ManiSkill batched RNG list.

Files changed:
- `envs/base_random_env.py`
- `docs/place_experiment_versions.md`

Notes:
- This is a domain-randomization bug fix only.
- Upload both `envs/place.py` and `envs/base_random_env.py` to the cloud server before rerunning v23/v24 commands.
- Existing checkpoints and videos under `runs/` were not changed.

## v25 - Camera FOV Randomization RNG Length Fix

Date: 2026-06-10
Status: applied

Problem:
- After fixing Place item/bin randomization and lighting randomization, cloud eval reset reached camera setup and failed with:

```text
camera.set_fovy(fovy[i], compute_x=True)
IndexError: index 15 is out of bounds for axis 0 with size 15
```

- The camera FOV noise path still used the ManiSkill batched RNG directly, producing 15 FOV values for 16 eval envs.

Changed from v24:
- Added `BaseRandomEnv._uniform_for_count(...)`.
- `BaseRandomEnv._uniform_per_sub_scene(...)` now uses that shared helper.
- Added `BaseRandomEnv._fov_noise_per_env(...)`.
- Third, wrist, and top camera FOV randomization now generate exactly `self.num_envs` values and tolerate a shorter RNG list.

Files changed:
- `envs/base_random_env.py`
- `docs/place_experiment_versions.md`

Notes:
- This is still a domain-randomization bug fix only.
- Existing checkpoints and videos under `runs/` were not changed.

## v26 - Centralize Safe Batched RNG Indexing

Date: 2026-06-10
Status: applied

Problem:
- After item/bin, lighting, and FOV randomization fixes, cloud eval reset failed in gripper speed randomization:

```text
File "/home/gpu/squint/envs/base_random_env.py", line 256, in _randomize_gripper_speed
stiffnesses = self._batched_episode_rng[env_idx].uniform(stiff_lo, stiff_hi)
IndexError: list index out of range
```

- The remaining direct batched-RNG index paths could still fail when ManiSkill provided fewer RNGs than env indices.

Changed from v25:
- Added `BaseRandomEnv._uniform_for_indices(...)`.
- `BaseRandomEnv._uniform_for_count(...)` now delegates to `_uniform_for_indices(...)`.
- Robot color randomization now uses safe indexed sampling.
- Gripper stiffness/damping randomization now uses safe indexed sampling and casts values to floats before applying drive properties.

Files changed:
- `envs/base_random_env.py`
- `docs/place_experiment_versions.md`

Notes:
- `rg "_batched_episode_rng" envs/base_random_env.py envs/place.py` should now show only helper implementations in `base_random_env.py` plus the local helper in `place.py`.
- Existing checkpoints and videos under `runs/` were not changed.

## v27 - Restore 100-Step Place Horizon for Pure-Visual Training

Date: 2026-06-10
Status: applied

Problem:
- The v21/v26 4090 run completed 3.5M steps but did not stably converge:

```text
best step=3200000
best success_at_end=0.25
final success_at_end=0.06
```

- Eval videos are about 2.55 seconds long, matching the 50-step horizon.
- Videos show the policy can sometimes grasp, but the episode often ends before a stable lift/carry/release-to-bin phase develops.

Changed from v26:
- Restored Place horizon to 100 steps:

```python
SO101PlaceCube-v1: 50 -> 100 steps
SO101PlaceCan-v1:  50 -> 100 steps
```

Files changed:
- `envs/place.py`
- `docs/place_experiment_versions.md`

Notes:
- This keeps green bin, wrist-roll action, 64x64 visual+qpos, and no-privileged-state training unchanged.
- Recommended next run: resume from the v21/v26 best checkpoint, not the final checkpoint.

## v28a - Slow-Real Action Scale Fresh Training

Date: 2026-06-11
Status: applied, ready to train.

Goal:
- Train a fresh pure visual+qpos place policy with an action scale closer to the real XLeRobot speed.
- Reduce sim-to-real failures where the fast v21/v26 policy pushes the cube, closes before alignment, or collides with the bin.
- Keep the change minimal before adding reward shaping, because earlier reward experiments did not reliably improve behavior.

Changed from v27:
- Reduced XLeRobot `pd_joint_delta_pos` and `pd_joint_target_delta_pos` action deltas:

```python
arm delta:     +/-0.10 -> +/-0.07 rad / control step
gripper delta: +/-0.20 -> +/-0.10 rad / control step
```

- Kept Place horizon at 100 steps:

```python
SO101PlaceCube-v1: 100 steps
SO101PlaceCan-v1:  100 steps
```

Files changed:
- `envs/robot/xlerobot.py`
- `envs/base_random_env.py`
- `deploy_utils/calibrate_head_servos.py`
- `deploy_utils/tune_top_camera.py`
- `deploy_utils/xlerobot_head_servos.json`
- `docs/place_experiment_versions.md`
- `HANDOFF_PLACE_TASK.md`
- `docs/codex_git_and_handoff_workflow.md`

Additional calibration changes included in the same workspace update:
- `deploy_utils/calibrate_head_servos.py` now connects with `handshake=False`.
- `deploy_utils/tune_top_camera.py` can load `xlerobot_head_servos.json` and displays raw real, policy crop, sim, and blended comparison views.
- Head servo centers changed to pan `129`, tilt `312`.
- Top camera sim alignment changed to FOV `34.0` degrees and quaternion `[0.180132, 0.14849, 0.750305, 0.618503]`.

Training command:

```bash
env \
  EXP_NAME=place_xlerobot_v28a_slowreal_64img_12env_8eval_8upd_buf40k_3500k_3060 \
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

Run directory:
- `runs/place_xlerobot_v28a_slowreal_64img_12env_8eval_8upd_buf40k_3500k_3060`

Result summary:
- Final eval:
- Best eval:
- Peak GPU memory:
- Runtime:

Observed failure modes to check:
- cube pushed away:
- early close:
- cannot grasp:
- bad gripper pose:
- no lift after grasp:
- no bin approach after grasp:

Decision:
- First inspect eval videos for reduced collision and better gripper centering before adding reward changes.

## Change Log Template

Copy this section for each new version.

```text
## vN - Short Name

Date:
Status: proposed / applied / trained / abandoned

Changed from previous version:
-

Files changed:
-

Training command:
```bash

```

Run directory:

Result summary:
- Final eval:
- Best eval:
- Peak GPU memory:
- Runtime:

Observed failure modes:
- unreachable:
- cannot grasp:
- grasped but cannot place:
- camera unclear:
- bad gripper pose:
- other:

Decision:
- continue training / widen range / narrow range / change reward / change robot setup / abandon
```
