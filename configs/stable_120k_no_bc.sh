#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Start from the same baseline checkpoint but disable scripted demo / BC warmstart.
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

SESSION_NAME="crl_120k_nobc_$(date +%Y%m%d_%H%M%S)"
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

  # No scripted demos / no BC warmstart.
  --scripted-demo-episodes 0
  --bc-pretrain-steps 0

  # Heavy long but stable profile.
  --total-episodes 120000
  --episode-length 250
  --warmup-episodes 1200

  # Exploration budget for no-BC training.
  --exploration-total-steps 5000000
  --exploration-random-steps 1200000
  --exploration-eps-start 0.50
  --exploration-eps-end 0.08
  --action-noise-start 0.45
  --action-noise-end 0.05

  # Replay and optimization.
  --buffer-capacity-transitions 1000000
  --her-ratio 0.35
  --updates-per-episode 80
  --batch-size 512
  --critic-loss-type sym_infonce

  # Task settings.
  --zone-half-extent-xy 0.09
  --spawn-noise-xy 0.02

  # Reduce overhead for long stability.
  --video-interval 1000
  --checkpoint-interval 50

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
