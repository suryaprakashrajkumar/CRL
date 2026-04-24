# CRL WidowX250 Pick-and-Place (PyTorch + MuJoCo)

This project implements **Contrastive Reinforcement Learning (CRL)** for a WidowX250 (WX250S) robot arm in MuJoCo.

Task:
- Move a green cube from a red square zone to a blue square zone.
- Current setup uses pose/state observations.
- Camera rendering is included for periodic policy videos and future vision-based extension.

## Features

- PyTorch CRL agent inspired by JaxGCRL CRL design
- MuJoCo environment using Menagerie WX250S model
- HER-style future-goal replay sampling
- Weights & Biases logging (`wandb` default credentials)
- Periodic checkpoints
- Periodic rendered video upload to W&B

## Environment Requirement

Use your conda env:
- `scalingRL`

## Setup

1. Install dependencies in `scalingRL`:

```bash
conda run -n scalingRL pip install -r requirements.txt
```

2. Fetch MuJoCo Menagerie (includes `trossen_wx250s` model and assets):

```bash
bash scripts/fetch_wx250s.sh
```

## Train

```bash
conda run -n scalingRL python -m crl_widowx.train \
  --robot-xml-path third_party/mujoco_menagerie/trossen_wx250s/wx250s.xml \
  --wandb-project crl-widowx-pick-place \
  --checkpoint-dir checkpoints
```

## Useful Flags

- `--total-episodes 800`
- `--warmup-episodes 20`
- `--updates-per-episode 100`
- `--batch-size 256`
- `--video-interval 25`
- `--camera-name isometric` (or `topdown`)
- `--wandb-mode online|offline|disabled`
- `--scripted-demo-episodes 200` (collect scripted demonstrations before RL)
- `--scripted-demo-max-steps 180` (cap per scripted demo rollout)
- `--bc-pretrain-steps 3000` (actor behavior cloning steps on scripted demos)
- `--bc-batch-size 512`

### Scripted Demo Warmstart

You can warmstart training by first collecting scripted pick-and-place episodes and then running behavior cloning on those transitions:

```bash
conda run -n scalingRL env PYTHONPATH=src python -m crl_widowx.train \
  --device cuda \
  --max-gpu-mode \
  --robot-xml-path third_party/mujoco_menagerie/trossen_wx250s/wx250s.xml \
  --scripted-demo-episodes 200 \
  --scripted-demo-max-steps 180 \
  --bc-pretrain-steps 3000 \
  --bc-batch-size 512 \
  --checkpoint-dir checkpoints/crl_scripted_warmstart
```

## Outputs

- Checkpoints: `checkpoints/*.pt`
- Videos: `checkpoints/videos/*.mp4`
- W&B metrics and media: logged per run

## Notes

Design reasoning and implementation decisions are documented in `notes.md`.

## Stable Long-Run TMUX Workflow (April 23, 2026)

This project includes a stable long-run launcher:

```bash
bash configs/stable_8h_explore_resume.sh
```

What it does:
- Resumes from latest checkpoint in `checkpoints/crl_longrun_v4_20260422_185938`.
- Starts a detached tmux session named `crl_stable8h_<timestamp>`.
- Logs console output to `logs/crl_stable8h_<timestamp>.log`.
- Writes new checkpoints to `checkpoints/crl_stable8h_<timestamp>/`.

How to monitor:

```bash
tmux ls
tmux attach -t <session_name>
# detach without stopping: Ctrl-b then d
tail -f logs/<session_name>.log
```

How to stop:

```bash
tmux kill-session -t <session_name>
```

Quick health checks during run:
- `train/success` should eventually move above `0.00`.
- `train/distance_delta_xy` should trend positive over time.
- `eval/success` should stop being pinned at `0`.
