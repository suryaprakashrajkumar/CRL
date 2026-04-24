# Design Notes: WidowX250 Pick-and-Place with CRL

## Objective

Implement a MuJoCo + PyTorch training system where a WX250S arm moves a green cube from red zone to blue zone, with CRL as the primary algorithmic backbone.

## Why this architecture

1. Environment as explicit Goal-Conditioned MDP
- The task has a clear geometric goal (cube pose/position target).
- Observation is state-based now (pose/joint/object states), with clean extension path to camera observations later.

2. CRL mapping from JaxGCRL
- Kept core contrastive structure:
  - State-action encoder and goal encoder.
  - Batchwise logits matrix with diagonal positives and off-diagonal negatives.
  - InfoNCE variant losses and logsumexp regularization term.
- Actor uses tanh-squashed Gaussian with entropy temperature tuning, matching typical CRL/SAC-style continuous control behavior.

3. Replay uses future-goal relabeling
- Implemented episode-based replay and HER-style goal relabeling.
- Future goal sampling is biased by discount (`gamma^k`) to favor temporally-near goals, consistent with CRL future-state pairing ideas.

4. Menagerie model integration
- Uses Menagerie `trossen_wx250s/wx250s.xml` directly to avoid hand-maintaining robot dynamics definitions.
- Scene augmentation adds task objects/zones while leaving robot description canonical.

5. Logging, checkpoints, and videos
- W&B captures training diagnostics and rollout media.
- Checkpointing is periodic and final, enabling recovery and model selection.
- Video eval at intervals provides visual correctness signal (critical for manipulation).

## Environment choices

- Action space: 7D continuous
  - 6 arm joint delta-controls
  - 1 gripper delta-control
- Reward: sparse success on XY proximity of cube to blue zone center.
- Episode truncation: fixed horizon.
- Cameras: `isometric` and `topdown` for debug and future camera-policy transition.

Reasoning:
- Delta position control is stable and aligns with existing MuJoCo actuator setup in WX250S model.
- Sparse reward keeps alignment with goal-conditioned methods and avoids encoding brittle shaping assumptions too early.

## CRL choices

- Energy functions: selectable (`norm`, `dot`, `cosine`, `l2`).
- Contrastive objectives: selectable (`fwd_infonce`, `bwd_infonce`, `sym_infonce`, `binary_nce`).
- Logsumexp penalty included as explicit regularizer for critic stability.
- Representation normalization enabled by default for stable contrastive geometry.

Reasoning:
- This keeps the implementation faithful to referenced CRL patterns while enabling ablations without refactoring.

## Trade-offs accepted

1. No camera encoder yet
- Current policy is state/pose conditioned only.
- Camera path is intentionally deferred to avoid coupling visual representation learning to initial control/debug stage.

2. No target critic networks in this version
- Contrastive critic is directly optimized and actor queries live critic.
- This is simpler and easier to debug first; if instability appears, EMA targets can be added next.

3. Episode replay buffer (not giant ring)
- Easier to preserve future-goal sampling semantics and episode boundaries.
- Slightly less memory-efficient than low-level ring buffer, but cleaner and safer for manipulation training.

## Path to camera-based policy (isometric)

1. Keep same task/reward/replay interfaces.
2. Add image observations to env output.
3. Add visual encoder (CNN or ViT-lite) to produce latent state embedding.
4. Concatenate latent with proprioception for actor and state-action encoder.
5. Maintain same goal API (goal as pose now; later optional goal-image).

## Verification checklist used

- Module boundaries:
  - env
  - replay
  - agent
  - trainer
  - video/logging/checkpointing
- End-to-end train entrypoint implemented (`python -m crl_widowx.train`).
- W&B initialization and periodic media upload included.
- Checkpoint writes include model and optimizer state.
- Menagerie fetch script included.

## What to monitor first during training

- `train/success`
- `eval/success`
- `critic/loss` and `critic/logsumexp_penalty`
- `actor/q_pi` and `actor/log_prob`
- `alpha/value`

If success is flat for long runs, first changes to test:
1. Increase warmup and replay size.
2. Use `sym_infonce` and tune `logsumexp_penalty_coeff`.
3. Add mild shaping on cube-to-goal distance while keeping terminal sparse success metric.

## April 22 tuning update (after zero-success run)

Observed issues in the previous long run:
- `eval/success` stayed at 0.
- `alpha/value` collapsed to ~1e-6, causing weak exploration.
- Reward values were generated in replay but not consumed by agent optimization.

Implemented fixes:
1. Reward-aware CRL update
- Added a reward prediction head to the agent.
- Critic training now includes supervised reward loss.
- Actor objective now includes a reward bonus term from predicted reward.

2. Fixed entropy temperature update
- Replaced unstable alpha update with standard SAC-style log-alpha loss.
- Added log-alpha clamping bounds to avoid collapse/explosion.

3. Exploration upgrades
- Added epsilon-random exploration schedule.
- Added Gaussian action noise schedule.
- Increased warmup default.

4. Environment learnability improvements
- Added dense reward component (distance-based) plus sparse success bonus.
- Increased action delta scales for arm and gripper.
- Added optional grasp-assist latch to make cube transport attainable during early training.

5. Diagnostics additions
- Logged motion and task-progress metrics: action magnitude, start/end distance, distance delta.
- Logged reward-model metrics: `reward/loss`, `reward/pred_mean`, `reward/target_mean`, `actor/reward_pi`.

## April 22 v4 run objective

Requested training behavior:
- Larger red and blue regions.
- Success defined as cube entering blue region.
- Relaxed but long exploration (1,000,000 env steps).
- High-throughput GPU configuration preset.

Implemented for v4:
1. Zone-based task success
- Blue-zone membership now determines `is_success` in step/reset.

2. Larger region geometry
- Added `zone_half_extent_xy` and enlarged site boxes.
- Increased spawn/goal randomization inside larger regions.

3. Exploration in env-step units
- Added `exploration_total_steps` and `exploration_random_steps`.
- Schedules now decay by accumulated `env_steps` (not episode count).

4. Replay scale target
- Added `buffer_capacity_transitions` with default 1,000,000 transitions.

5. Max-GPU preset
- Added `--max-gpu-mode` runtime toggles (TF32 + cudnn benchmark + high matmul precision).
- Added launch script: `configs/max_gpu_1m_explore.sh`.
