#!/usr/bin/env bash
set -Eeuo pipefail

CHECKPOINT="${CHECKPOINT:-runs/place_xlerobot_cube_topcam_v9_32img_6gb_32env_8eval_3500k/best_ckpt.pt}"
RUN_NAME="${RUN_NAME:-real_place_v14_crop_left20_open15_close15}"

BIN_X="${BIN_X:-0.225}"
BIN_Y="${BIN_Y:-0.0}"
CONTROL_FREQ="${CONTROL_FREQ:-30}"
ACTION_SCALE="${ACTION_SCALE:-0.15}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
GRIPPER_OPEN_STEPS="${GRIPPER_OPEN_STEPS:-14}"
GRIPPER_OPEN_ACTION="${GRIPPER_OPEN_ACTION:-0.20}"
GRIPPER_CLOSE_LIMIT="${GRIPPER_CLOSE_LIMIT:-0.15}"
REAL_CROP_SHIFT_X="${REAL_CROP_SHIFT_X:--20}"
REAL_CROP_SHIFT_Y="${REAL_CROP_SHIFT_Y:-0}"
POST_GRASP_ASSIST_START="${POST_GRASP_ASSIST_START:--1}"
POST_GRASP_ASSIST_END="${POST_GRASP_ASSIST_END:--1}"
POST_GRASP_LIFT_ACTION="${POST_GRASP_LIFT_ACTION:-0.05}"
POST_GRASP_WRIST_ACTION="${POST_GRASP_WRIST_ACTION:-0.03}"
POST_GRASP_CLOSE_ACTION="${POST_GRASP_CLOSE_ACTION:--0.04}"
ROUTE_A_PLACE_ASSIST="${ROUTE_A_PLACE_ASSIST:-false}"
ROUTE_A_PLACE_ASSIST_CARRY_START="${ROUTE_A_PLACE_ASSIST_CARRY_START:-150}"
ROUTE_A_PLACE_ASSIST_RELEASE_START="${ROUTE_A_PLACE_ASSIST_RELEASE_START:-240}"
ROUTE_A_PLACE_ASSIST_END="${ROUTE_A_PLACE_ASSIST_END:-285}"
ROUTE_A_PLACE_ASSIST_BASE_ACTION="${ROUTE_A_PLACE_ASSIST_BASE_ACTION:-0.08}"
ROUTE_A_PLACE_ASSIST_ELBOW_ACTION="${ROUTE_A_PLACE_ASSIST_ELBOW_ACTION:--0.04}"
ROUTE_A_PLACE_ASSIST_LIFT_ACTION="${ROUTE_A_PLACE_ASSIST_LIFT_ACTION:-0.04}"
ROUTE_A_PLACE_ASSIST_WRIST_ACTION="${ROUTE_A_PLACE_ASSIST_WRIST_ACTION:-0.03}"
ROUTE_A_PLACE_ASSIST_CLOSE_ACTION="${ROUTE_A_PLACE_ASSIST_CLOSE_ACTION:--0.06}"
ROUTE_A_PLACE_ASSIST_RELEASE_ACTION="${ROUTE_A_PLACE_ASSIST_RELEASE_ACTION:-0.18}"

ROUTE_A_PLACE_ASSIST_ARGS=()
if [ "${ROUTE_A_PLACE_ASSIST}" = "true" ]; then
  ROUTE_A_PLACE_ASSIST_ARGS+=(--route-a-place-assist)
fi

python deploy.py \
  --checkpoint="${CHECKPOINT}" \
  --env-id=SO101PlaceCube-v1 \
  --robot-uids=xlerobot_right_head \
  --control-mode=pd_joint_target_delta_pos \
  --image-size=32 \
  --max-episode-steps="${MAX_EPISODE_STEPS}" \
  --control-freq="${CONTROL_FREQ}" \
  --action-scale="${ACTION_SCALE}" \
  --gripper-open-steps="${GRIPPER_OPEN_STEPS}" \
  --gripper-open-action="${GRIPPER_OPEN_ACTION}" \
  --gripper-close-action-limit="${GRIPPER_CLOSE_LIMIT}" \
  --route-a-state-override \
  --route-a-bin-xy "${BIN_X}" "${BIN_Y}" \
  --route-a-follow-tcp-after-grasp \
  --route-a-grasp-qpos=0.45 \
  --route-a-tcp-item-z-offset=-0.015 \
  --auto-stall-assist \
  --stall-assist-min-step=55 \
  --stall-assist-gripper-qpos=0.45 \
  --stall-assist-action-norm=0.04 \
  --stall-assist-duration=45 \
  --post-grasp-assist-start="${POST_GRASP_ASSIST_START}" \
  --post-grasp-assist-end="${POST_GRASP_ASSIST_END}" \
  --post-grasp-lift-action="${POST_GRASP_LIFT_ACTION}" \
  --post-grasp-wrist-action="${POST_GRASP_WRIST_ACTION}" \
  --post-grasp-close-action="${POST_GRASP_CLOSE_ACTION}" \
  "${ROUTE_A_PLACE_ASSIST_ARGS[@]}" \
  --route-a-place-assist-carry-start="${ROUTE_A_PLACE_ASSIST_CARRY_START}" \
  --route-a-place-assist-release-start="${ROUTE_A_PLACE_ASSIST_RELEASE_START}" \
  --route-a-place-assist-end="${ROUTE_A_PLACE_ASSIST_END}" \
  --route-a-place-assist-base-action="${ROUTE_A_PLACE_ASSIST_BASE_ACTION}" \
  --route-a-place-assist-elbow-action="${ROUTE_A_PLACE_ASSIST_ELBOW_ACTION}" \
  --route-a-place-assist-lift-action="${ROUTE_A_PLACE_ASSIST_LIFT_ACTION}" \
  --route-a-place-assist-wrist-action="${ROUTE_A_PLACE_ASSIST_WRIST_ACTION}" \
  --route-a-place-assist-close-action="${ROUTE_A_PLACE_ASSIST_CLOSE_ACTION}" \
  --route-a-place-assist-release-action="${ROUTE_A_PLACE_ASSIST_RELEASE_ACTION}" \
  --record-dir="runs/${RUN_NAME}/videos" \
  --record-resolution=480 \
  --real-crop-shift-x="${REAL_CROP_SHIFT_X}" \
  --real-crop-shift-y="${REAL_CROP_SHIFT_Y}" \
  --log-dir="runs/${RUN_NAME}/logs"
