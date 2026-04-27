#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="crl_scaling_iso_nobc_d32_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/${SESSION_NAME}"
LOG_FILE="logs/${SESSION_NAME}.log"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

mkdir -p checkpoints logs

CMD=(
  conda run -n scalingRL env PYTHONPATH=src MUJOCO_GL=egl
  python -m crl_widowx.train
  --device cuda
  --max-gpu-mode
  --robot-xml-path third_party/mujoco_menagerie/trossen_wx250s/wx250s.xml

  --observation-mode image
  --camera-name isometric
  --render-width 320
  --render-height 240
  --image-observation-width 48
  --image-observation-height 48
  --image-observation-grayscale

  --scripted-demo-episodes 200
  --bc-pretrain-steps 0
  --scripted-demo-max-steps 250
  --scripted-demo-success-threshold 1.0
  --disable-scripted-demo-oracle-assist

  --total-episodes 60000
  --episode-length 250

  --warmup-episodes 120
  --exploration-total-steps 1200000
  --exploration-random-steps 150000
  --exploration-eps-start 0.65
  --exploration-eps-end 0.15
  --action-noise-start 0.65
  --action-noise-end 0.20
  --action-noise-floor 0.08

  --target-entropy-scale -0.25
  --min-log-alpha -4.0
  --max-log-alpha 2.0

  --energy-fn dot
  --reward-loss-coeff 0.0
  --actor-reward-bonus-coeff 0.0
  --actor-awbc-coeff 0.05
  --actor-awbc-temp 0.35

  --hidden-dim 256
  --repr-dim 64
  --residual-depth 32
  --residual-block-size 4

  --action-scale-arm 0.12
  --action-scale-gripper 0.010
  --success-threshold-xy 0.04

  --buffer-capacity-episodes 480
  --buffer-capacity-transitions 120000
  --contrastive-goal-mode future
  --her-ratio 0.35
  --her-min-goal-dist 0.02
  --her-min-future-offset 4
  --updates-per-episode 20
  --batch-size 256
  --critic-loss-type sym_infonce

  --zone-half-extent-xy 0.10
  --spawn-noise-xy 0.015
  --goal-noise-xy 0.015

  --video-interval 500
  --checkpoint-interval 500
  --disable-eval-stochastic-fallback
  --eval-fallback-noise-std 0.0

  --wandb-mode online
  --wandb-project crl-widowx-pick-place
  --wandb-run-name "${SESSION_NAME}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume-checkpoint "$RESUME_CHECKPOINT" --resume-load-optimizers --resume-load-alpha)
fi

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
