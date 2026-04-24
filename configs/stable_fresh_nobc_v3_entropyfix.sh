#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="crl_fresh_nobc_v3_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/${SESSION_NAME}"
LOG_FILE="logs/${SESSION_NAME}.log"

mkdir -p checkpoints logs

CMD=(
  conda run -n scalingRL env PYTHONPATH=src
  python -m crl_widowx.train
  --device cuda
  --max-gpu-mode
  --robot-xml-path third_party/mujoco_menagerie/trossen_wx250s/wx250s.xml

  # Strictly fresh and no demo/BC warmstart.
  --scripted-demo-episodes 0
  --bc-pretrain-steps 0

  --total-episodes 60000
  --episode-length 250

  # Start learning earlier and keep stronger exploration pressure.
  --warmup-episodes 150
  --exploration-total-steps 3500000
  --exploration-random-steps 400000
  --exploration-eps-start 0.60
  --exploration-eps-end 0.15
  --action-noise-start 0.60
  --action-noise-end 0.20
  --action-noise-floor 0.15

  # Prevent alpha from collapsing to near-zero entropy.
  --target-entropy-scale -0.3
  --min-log-alpha -1.5
  --max-log-alpha 2.0
  --actor-reward-bonus-coeff 0.2

  # Less aggressive optimization to reduce early overfitting/collapse.
  --buffer-capacity-transitions 1000000
  --her-ratio 0.50
  --updates-per-episode 20
  --batch-size 256
  --critic-loss-type sym_infonce

  # Keep slightly easier geometry for early positive signal.
  --zone-half-extent-xy 0.10
  --spawn-noise-xy 0.015
  --goal-noise-xy 0.015

  --video-interval 500
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

echo "SESSION_NAME=$SESSION_NAME"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "Attach: tmux attach -t $SESSION_NAME"
