#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Source run to warm-start from.
BASE_DIR="checkpoints/crl_longrun_v4_20260422_185938"
if [[ ! -d "$BASE_DIR" ]]; then
  echo "Base checkpoint dir not found: $BASE_DIR"
  exit 1
fi

RESUME_CKPT="$(ls "$BASE_DIR"/episode_*.pt | sort | tail -n 1)"
if [[ -z "${RESUME_CKPT}" ]]; then
  echo "No checkpoint found in $BASE_DIR"
  exit 1
fi

SESSION_NAME="crl_100k_bc_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/${SESSION_NAME}"
LOG_FILE="logs/${SESSION_NAME}.log"

mkdir -p checkpoints logs

CMD=(
  conda run -n scalingRL env PYTHONPATH=src
  python -m crl_widowx.train
  --device cuda
  --max-gpu-mode
  --resume-checkpoint "$RESUME_CKPT"
  --resume-load-optimizers
  --resume-load-alpha
  --robot-xml-path third_party/mujoco_menagerie/trossen_wx250s/wx250s.xml

  # Scripted-demo warmstart before RL updates.
  --scripted-demo-episodes 220
  --scripted-demo-max-steps 200
  --scripted-demo-success-threshold 1.0
  --scripted-demo-max-attempts-mult 20
  --scripted-demo-oracle-assist
  --bc-pretrain-steps 15000
  --bc-batch-size 512
  --post-bc-exploration-scale 0.10
  --post-bc-action-noise-scale 0.25

  # Long run target.
  --total-episodes 100000
  --episode-length 250

  # Exploration-heavy continuation.
  --warmup-episodes 300
  --exploration-total-steps 3000000
  --exploration-random-steps 0
  --exploration-eps-start 0.45
  --exploration-eps-end 0.04
  --action-noise-start 0.40
  --action-noise-end 0.03

  # Replay and training updates.
  --buffer-capacity-transitions 1000000
  --her-ratio 0.30
  --updates-per-episode 80
  --batch-size 512
  --critic-loss-type sym_infonce

  # Task settings.
  --zone-half-extent-xy 0.09
  --spawn-noise-xy 0.02

  # Reduce eval overhead.
  --video-interval 500
  --checkpoint-interval 25

  --wandb-mode online
  --wandb-project crl-widowx-pick-place
  --wandb-run-name "${SESSION_NAME}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
)

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

CMD_STR="$(printf '%q ' "${CMD[@]}")"
tmux new-session -d -s "$SESSION_NAME" "cd $ROOT_DIR && $CMD_STR"
tmux pipe-pane -o -t "$SESSION_NAME":0 "cat >> $ROOT_DIR/$LOG_FILE"

echo "RESUME_CKPT=$RESUME_CKPT"
echo "SESSION_NAME=$SESSION_NAME"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "Attach: tmux attach -t $SESSION_NAME"
