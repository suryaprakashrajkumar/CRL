#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="crl_fresh_nobc_v5_fast_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/${SESSION_NAME}"
LOG_FILE="logs/${SESSION_NAME}.log"

mkdir -p checkpoints logs

CMD=(
  conda run -n scalingRL env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -m crl_widowx.train
  --device cuda
  --max-gpu-mode
  --robot-xml-path third_party/mujoco_menagerie/trossen_wx250s/wx250s.xml

  # Fresh run, no demos/BC/checkpoint resume.
  --scripted-demo-episodes 0
  --bc-pretrain-steps 0

  --total-episodes 60000
  --episode-length 250

  # Exploration schedule tuned from current v4 branch.
  --warmup-episodes 120
  --exploration-total-steps 1200000
  --exploration-random-steps 150000
  --exploration-eps-start 0.65
  --exploration-eps-end 0.15
  --action-noise-start 0.65
  --action-noise-end 0.20
  --action-noise-floor 0.08

  # Keep entropy collapse in check while allowing lower alpha floor.
  --target-entropy-scale -0.25
  --min-log-alpha -4.0
  --max-log-alpha 2.0

  --reward-loss-coeff 0.3
  --actor-reward-bonus-coeff 0.2
  --actor-awbc-coeff 1.0
  --actor-awbc-temp 0.35

  --action-scale-arm 0.12
  --action-scale-gripper 0.010

  # Higher GPU work per update + reduced non-training overhead.
  --buffer-capacity-transitions 1000000
  --her-ratio 0.55
  --updates-per-episode 20
  --batch-size 512
  --critic-loss-type sym_infonce

  --zone-half-extent-xy 0.10
  --spawn-noise-xy 0.015
  --goal-noise-xy 0.015

  # Sparse eval/checkpoint cadence for faster wall-clock.
  --video-interval 2000
  --checkpoint-interval 200
  --disable-eval-stochastic-fallback
  --eval-fallback-noise-std 0.0

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
