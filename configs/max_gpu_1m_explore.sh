#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="crl_longrun_v4_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/${SESSION_NAME}"
LOG_FILE="logs/${SESSION_NAME}.log"

mkdir -p checkpoints logs

CMD=(
  conda run -n scalingRL env PYTHONPATH=src
  python -m crl_widowx.train
  --device cuda
  --max-gpu-mode
  --robot-xml-path third_party/mujoco_menagerie/trossen_wx250s/wx250s.xml
  --total-episodes 6000
  --episode-length 250
  --warmup-episodes 120
  --updates-per-episode 120
  --batch-size 512
  --buffer-capacity-transitions 1000000
  --her-ratio 0.30
  --critic-loss-type sym_infonce
  --zone-half-extent-xy 0.09
  --spawn-noise-xy 0.02
  --exploration-total-steps 1000000
  --exploration-random-steps 300000
  --exploration-eps-start 0.30
  --exploration-eps-end 0.04
  --action-noise-start 0.30
  --action-noise-end 0.04
  --video-interval 50
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

echo "SESSION_NAME=$SESSION_NAME"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "Attach: tmux attach -t $SESSION_NAME"
