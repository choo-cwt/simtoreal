#!/usr/bin/env bash
set -Eeuo pipefail

CHECKPOINT="${CHECKPOINT:-runs/place_xlerobot_b1_visual_qpos_32img_32env_8eval_3500k/best_ckpt.pt}"
RUN_NAME="${RUN_NAME:-real_place_b1_diag_policy_input}"

CONTROL_FREQ="${CONTROL_FREQ:-30}"
ACTION_SCALE="${ACTION_SCALE:-0.15}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
GRIPPER_OPEN_STEPS="${GRIPPER_OPEN_STEPS:-14}"
GRIPPER_OPEN_ACTION="${GRIPPER_OPEN_ACTION:-0.20}"
GRIPPER_CLOSE_LIMIT="${GRIPPER_CLOSE_LIMIT:-0.15}"
REAL_CROP_SHIFT_X="${REAL_CROP_SHIFT_X:-0}"
REAL_CROP_SHIFT_Y="${REAL_CROP_SHIFT_Y:-0}"

python deploy.py \
  --checkpoint="${CHECKPOINT}" \
  --env-id=SO101PlaceCube-v1 \
  --robot-uids=xlerobot_right_head \
  --control-mode=pd_joint_target_delta_pos \
  --image-size=32 \
  --no-privileged-state \
  --max-episode-steps="${MAX_EPISODE_STEPS}" \
  --control-freq="${CONTROL_FREQ}" \
  --action-scale="${ACTION_SCALE}" \
  --gripper-open-steps="${GRIPPER_OPEN_STEPS}" \
  --gripper-open-action="${GRIPPER_OPEN_ACTION}" \
  --gripper-close-action-limit="${GRIPPER_CLOSE_LIMIT}" \
  --post-grasp-assist-start=-1 \
  --real-crop-shift-x="${REAL_CROP_SHIFT_X}" \
  --real-crop-shift-y="${REAL_CROP_SHIFT_Y}" \
  --record-dir="runs/${RUN_NAME}/videos" \
  --record-resolution=480 \
  --policy-input-record-dir="runs/${RUN_NAME}/policy_input_videos" \
  --policy-input-record-resolution=256 \
  --log-dir="runs/${RUN_NAME}/logs"
