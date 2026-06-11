#!/usr/bin/env bash
set -Eeuo pipefail

EXP_NAME="${EXP_NAME:-place_xlerobot_cube_topcam_6gb_32env_8upd_200kbuf}"
ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
ROBOT_UIDS="${ROBOT_UIDS:-xlerobot_right_head}"
CONTROL_MODE="${CONTROL_MODE:-pd_joint_target_delta_pos}"
NUM_ENVS="${NUM_ENVS:-32}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-8}"
NUM_UPDATES="${NUM_UPDATES:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-200000}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3500000}"
EVAL_FREQ="${EVAL_FREQ:-100000}"
RENDER_SIZE="${RENDER_SIZE:-128}"
IMAGE_SIZE="${IMAGE_SIZE:-16}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-10}"
CHECKPOINT="${CHECKPOINT:-}"
NO_PRIVILEGED_STATE="${NO_PRIVILEGED_STATE:-false}"

RUN_DIR="runs/${EXP_NAME}"
mkdir -p "${RUN_DIR}"

CMD=(
  python train_squint.py
  --env-id="${ENV_ID}"
  --robot-uids="${ROBOT_UIDS}"
  --control-mode="${CONTROL_MODE}"
  --num-envs="${NUM_ENVS}"
  --num-eval-envs="${NUM_EVAL_ENVS}"
  --num-updates="${NUM_UPDATES}"
  --batch-size="${BATCH_SIZE}"
  --buffer-size="${BUFFER_SIZE}"
  --render-size="${RENDER_SIZE}"
  --image-size="${IMAGE_SIZE}"
  --total-timesteps="${TOTAL_TIMESTEPS}"
  --eval-freq="${EVAL_FREQ}"
  --exp-name="${EXP_NAME}"
)

if [ "${NO_PRIVILEGED_STATE}" = "true" ]; then
  CMD+=(--no-privileged-state)
fi

if [ -n "${CHECKPOINT}" ]; then
  CMD+=(--checkpoint="${CHECKPOINT}")
fi

{
  printf 'Started: %s\n' "$(date -Is)"
  printf 'Working directory: %s\n' "$(pwd)"
  printf 'Command:\n'
  printf 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLCONFIGDIR=/tmp/mpl'
  printf ' %q' "${CMD[@]}"
  printf '\n'
} | tee "${RUN_DIR}/command.txt"

cat > "${RUN_DIR}/failure_notes.md" <<'EOF'
# Failure Notes

Use this after watching eval videos.

- Run:
- Step/video:
- Failure type: unreachable / cannot grasp / grasped but cannot place / camera unclear / bad gripper pose / other
- Notes:
EOF

monitor_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi not found\n' > "${RUN_DIR}/gpu_memory.csv"
    return
  fi

  printf 'timestamp,index,name,memory.used MiB,memory.total MiB,utilization.gpu %%\n' > "${RUN_DIR}/gpu_memory.csv"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits >> "${RUN_DIR}/gpu_memory.csv" || true
    sleep "${GPU_LOG_INTERVAL}"
  done
}

monitor_gpu &
GPU_MONITOR_PID=$!
cleanup() {
  kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLCONFIGDIR=/tmp/mpl "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
STATUS=${PIPESTATUS[0]}
set -e

{
  printf 'Finished: %s\n' "$(date -Is)"
  printf 'Exit status: %s\n' "${STATUS}"
} | tee "${RUN_DIR}/run_status.txt"

if command -v awk >/dev/null 2>&1 && [ -s "${RUN_DIR}/gpu_memory.csv" ]; then
  awk -F',' '
    NR > 1 {
      gsub(/^ +| +$/, "", $4)
      if ($4 + 0 > max) max = $4 + 0
    }
    END {
      if (NR > 1) printf("Peak GPU memory used: %d MiB\n", max)
    }
  ' "${RUN_DIR}/gpu_memory.csv" | tee "${RUN_DIR}/gpu_peak.txt"
fi

grep -E 'success_at_end:|eval/success|eval/return|Step [0-9]+: model checkpoint saved' "${RUN_DIR}/train.log" > "${RUN_DIR}/eval_summary.txt" || true

exit "${STATUS}"
