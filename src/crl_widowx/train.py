from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from tqdm import trange

from crl_widowx.agents.crl_agent import CRLAgent, CRLConfig
from crl_widowx.envs.widowx_pick_place_env import WidowXEnvConfig, WidowXPickPlaceEnv
from crl_widowx.replay_buffer import EpisodeBatch, EpisodeReplayBuffer
from crl_widowx.utils.video import record_policy_rollout, write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRL on WidowX250 pick-and-place in MuJoCo")
    parser.add_argument(
        "--robot-xml-path",
        type=str,
        default="third_party/mujoco_menagerie/trossen_wx250s/wx250s.xml",
        help="Path to Menagerie wx250s.xml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--total-episodes", type=int, default=400)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--warmup-episodes", type=int, default=50)
    parser.add_argument("--updates-per-episode", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--scripted-demo-episodes", type=int, default=0)
    parser.add_argument("--scripted-demo-max-steps", type=int, default=200)
    parser.add_argument("--bc-pretrain-steps", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--scripted-demo-success-threshold", type=float, default=1.0)
    parser.add_argument("--scripted-demo-max-attempts-mult", type=int, default=20)
    parser.add_argument("--scripted-demo-oracle-assist", action="store_true", default=False)
    parser.add_argument("--disable-scripted-demo-oracle-assist", dest="scripted_demo_oracle_assist", action="store_false")
    parser.add_argument("--allow-oracle-future-goal-leak", action="store_true", default=False)
    parser.add_argument("--post-bc-exploration-scale", type=float, default=0.2)
    parser.add_argument("--post-bc-action-noise-scale", type=float, default=0.4)

    parser.add_argument("--exploration-eps-start", type=float, default=0.35)
    parser.add_argument("--exploration-eps-end", type=float, default=0.05)
    parser.add_argument("--exploration-eps-decay-episodes", type=int, default=600)
    parser.add_argument("--action-noise-start", type=float, default=0.35)
    parser.add_argument("--action-noise-end", type=float, default=0.05)
    parser.add_argument("--action-noise-decay-episodes", type=int, default=600)
    parser.add_argument("--exploration-total-steps", type=int, default=1_000_000)
    parser.add_argument("--exploration-random-steps", type=int, default=200_000)
    parser.add_argument("--action-noise-floor", type=float, default=0.0)

    parser.add_argument("--buffer-capacity-episodes", type=int, default=800)
    parser.add_argument("--buffer-capacity-transitions", type=int, default=1_000_000)
    parser.add_argument("--her-ratio", type=float, default=0.4)
    parser.add_argument(
        "--contrastive-goal-mode",
        type=str,
        default="future",
        choices=["future", "desired_or_her"],
        help="Use future achieved goals as contrastive positives, or the older desired-goal/HER mix.",
    )
    parser.add_argument("--her-min-goal-dist", type=float, default=0.03)
    parser.add_argument("--her-min-future-offset", type=int, default=4)
    parser.add_argument("--goal-sample-gamma", type=float, default=0.98)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--repr-dim", type=int, default=64)
    parser.add_argument(
        "--residual-depth",
        type=int,
        default=0,
        help="Total Dense layers inside residual blocks for actor and CRL critic encoders. Use 0 for shallow MLPs.",
    )
    parser.add_argument(
        "--residual-block-size",
        type=int,
        default=4,
        help="Dense->LayerNorm->SiLU units per residual block; the Scaling CRL paper uses 4.",
    )
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--energy-fn", type=str, default="norm", choices=["norm", "dot", "cosine", "l2"])
    parser.add_argument(
        "--critic-loss-type",
        type=str,
        default="fwd_infonce",
        choices=["fwd_infonce", "bwd_infonce", "sym_infonce", "binary_nce"],
    )
    parser.add_argument("--logsumexp-penalty-coeff", type=float, default=0.1)
    parser.add_argument("--reward-loss-coeff", type=float, default=1.0)
    parser.add_argument("--actor-reward-bonus-coeff", type=float, default=0.0)
    parser.add_argument("--actor-awbc-coeff", type=float, default=0.0)
    parser.add_argument("--actor-awbc-temp", type=float, default=0.5)
    parser.add_argument("--target-entropy-scale", type=float, default=-1.0)
    parser.add_argument("--min-log-alpha", type=float, default=-4.0)
    parser.add_argument("--max-log-alpha", type=float, default=2.0)

    parser.add_argument("--wandb-project", type=str, default="crl-widowx-pick-place")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--video-interval", type=int, default=25)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--eval-min-action-norm", type=float, default=0.08)
    parser.add_argument("--eval-fallback-noise-std", type=float, default=0.08)
    parser.add_argument("--eval-stochastic-fallback", action="store_true", default=True)
    parser.add_argument("--disable-eval-stochastic-fallback", dest="eval_stochastic_fallback", action="store_false")
    parser.add_argument("--continue-on-eval-error", action="store_true", default=True)
    parser.add_argument("--fail-on-eval-error", dest="continue_on_eval_error", action="store_false")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--resume-load-optimizers", action="store_true", default=False)
    parser.add_argument("--resume-load-alpha", action="store_true", default=False)

    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--camera-name", type=str, default="isometric", choices=["isometric", "topdown"])
    parser.add_argument("--observation-mode", type=str, default="state", choices=["state", "image"])
    parser.add_argument("--image-observation-width", type=int, default=64)
    parser.add_argument("--image-observation-height", type=int, default=64)
    parser.add_argument("--image-observation-grayscale", action="store_true", default=False)
    parser.add_argument("--max-gpu-mode", action="store_true", default=False)

    parser.add_argument("--action-scale-arm", type=float, default=0.08)
    parser.add_argument("--action-scale-gripper", type=float, default=0.006)
    parser.add_argument("--success-threshold-xy", type=float, default=0.04)
    parser.add_argument("--dense-reward-scale", type=float, default=0.12)
    parser.add_argument("--dense-reward-weight", type=float, default=0.3)
    parser.add_argument("--success-bonus", type=float, default=2.0)
    parser.add_argument("--zone-half-extent-xy", type=float, default=0.08)
    parser.add_argument("--spawn-noise-xy", type=float, default=0.04)
    parser.add_argument("--goal-noise-xy", type=float, default=0.02)
    parser.add_argument("--enable-grasp-assist", action="store_true", default=True)
    parser.add_argument("--disable-grasp-assist", dest="enable_grasp_assist", action="store_false")
    parser.add_argument(
        "--collect-with-stochastic-policy",
        action="store_true",
        default=False,
        help="If set, sample actions from stochastic policy during rollout; default uses deterministic mean + noise.",
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_episode(
    env: WidowXPickPlaceEnv,
    agent: CRLAgent | None,
    random_policy: bool,
    eps_random: float,
    action_noise_std: float,
    use_scripted_policy: bool = False,
    max_steps_override: int | None = None,
    scripted_oracle_assist: bool = False,
    collect_with_stochastic_policy: bool = False,
) -> tuple[EpisodeBatch, dict[str, float]]:
    obs, info = env.reset()

    states = [obs["observation"].astype(np.float32)]
    achieved_goals = [obs["achieved_goal"].astype(np.float32)]
    actions: list[np.ndarray] = []
    desired_goals: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[float] = []

    episode_return = 0.0
    success = float(info.get("is_success", 0.0))
    state_obs = obs.get("state", obs["observation"])
    reach_distances = [float(np.linalg.norm(state_obs[-4:-1]))]
    cube_goal_distances = [
        float(np.linalg.norm(obs["achieved_goal"][:2] - obs["desired_goal"][:2]))
    ]
    attached_steps = 0

    max_steps = env.max_steps if max_steps_override is None else max(1, int(max_steps_override))
    for step_idx in range(max_steps):
        state = obs["observation"]
        goal = obs["desired_goal"]

        if use_scripted_policy:
            action = env.scripted_action(obs).astype(np.float32)
        elif random_policy or agent is None or np.random.random() < eps_random:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(state, goal, deterministic=(not collect_with_stochastic_policy)).astype(np.float32)
            if action_noise_std > 0.0:
                action = np.clip(
                    action + np.random.normal(loc=0.0, scale=action_noise_std, size=action.shape).astype(np.float32),
                    -1.0,
                    1.0,
                )

        next_obs, reward, terminated, truncated, info = env.step(action)
        if (
            use_scripted_policy
            and scripted_oracle_assist
            and step_idx == (max_steps - 1)
            and float(info.get("is_success", 0.0)) < 1.0
        ):
            next_obs, reward, info = env.scripted_oracle_finalize()
        done = float(terminated or truncated)

        actions.append(action)
        desired_goals.append(goal.astype(np.float32))
        rewards.append(float(reward))
        dones.append(done)

        states.append(next_obs["observation"].astype(np.float32))
        achieved_goals.append(next_obs["achieved_goal"].astype(np.float32))

        episode_return += float(reward)
        success = max(success, float(info.get("is_success", 0.0)))
        next_state_obs = next_obs.get("state", next_obs["observation"])
        reach_distances.append(float(np.linalg.norm(next_state_obs[-4:-1])))
        cube_goal_distances.append(float(np.linalg.norm(next_obs["achieved_goal"][:2] - next_obs["desired_goal"][:2])))
        attached_steps += int(float(info.get("reward_attach_term", 0.0)) > 0.0)

        obs = next_obs
        if terminated or truncated:
            break

    episode = EpisodeBatch(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        desired_goals=np.asarray(desired_goals, dtype=np.float32),
        achieved_goals=np.asarray(achieved_goals, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
    )

    start_dist = float(np.linalg.norm(achieved_goals[0][:2] - desired_goals[0][:2])) if desired_goals else 0.0
    final_dist = float(np.linalg.norm(achieved_goals[-1][:2] - desired_goals[-1][:2])) if desired_goals else 0.0
    action_abs_mean = float(np.mean(np.abs(episode.actions))) if len(actions) > 0 else 0.0
    action_delta_abs_mean = (
        float(np.mean(np.abs(np.diff(episode.actions, axis=0)))) if len(actions) > 1 else 0.0
    )
    min_reach_dist = float(np.min(reach_distances)) if reach_distances else 0.0
    final_reach_dist = float(reach_distances[-1]) if reach_distances else 0.0
    min_cube_goal_dist = float(np.min(cube_goal_distances)) if cube_goal_distances else 0.0

    metrics = {
        "train/episode_return": float(episode_return),
        "train/episode_length": float(len(actions)),
        "train/success": float(success),
        "train/distance_start_xy": start_dist,
        "train/distance_end_xy": final_dist,
        "train/distance_delta_xy": start_dist - final_dist,
        "train/distance_min_xy": min_cube_goal_dist,
        "train/reach_distance_min": min_reach_dist,
        "train/reach_distance_final": final_reach_dist,
        "train/attached_fraction": float(attached_steps / max(1, len(actions))),
        "train/action_abs_mean": action_abs_mean,
        "train/action_delta_abs_mean": action_delta_abs_mean,
    }
    return episode, metrics


def linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 1:
        return float(end)
    frac = min(1.0, max(0.0, step / float(decay_steps - 1)))
    return float(start + frac * (end - start))


def sample_demo_batch(demo_transitions: dict[str, np.ndarray], batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    num_samples = int(demo_transitions["states"].shape[0])
    if num_samples < 1:
        raise RuntimeError("No scripted transitions available for BC warmstart.")
    idx = rng.integers(0, num_samples, size=(batch_size,))
    return {
        "states": demo_transitions["states"][idx],
        "goals": demo_transitions["goals"][idx],
        "actions": demo_transitions["actions"][idx],
    }


def save_checkpoint(
    path: Path,
    agent: CRLAgent,
    episode: int,
    gradient_steps: int,
    env_steps: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode": episode,
        "gradient_steps": gradient_steps,
        "env_steps": env_steps,
        "args": vars(args),
        "agent": agent.state_dict(),
    }
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def maybe_resume_from_checkpoint(
    agent: CRLAgent,
    resume_path: str | None,
    load_optimizers: bool,
    load_alpha: bool,
) -> tuple[int, int, int]:
    if not resume_path:
        return 1, 0, 0

    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    payload = torch.load(path, map_location=agent.device)
    state = payload.get("agent", payload)

    if "actor" in state:
        result = agent.actor.load_state_dict(state["actor"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"Warning: actor state mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}")
    if "sa_encoder" in state:
        result = agent.sa_encoder.load_state_dict(state["sa_encoder"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(
                f"Warning: sa_encoder state mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}"
            )
    if "g_encoder" in state:
        result = agent.g_encoder.load_state_dict(state["g_encoder"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"Warning: g_encoder state mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}")
    if "reward_head" in state:
        result = agent.reward_head.load_state_dict(state["reward_head"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(
                f"Warning: reward_head state mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}"
            )
    if load_alpha and "log_alpha" in state:
        agent.log_alpha.data.copy_(state["log_alpha"].to(agent.device))

    if load_optimizers:
        if "actor_opt" in state:
            try:
                agent.actor_opt.load_state_dict(state["actor_opt"])
            except Exception as exc:
                print(f"Warning: failed to load actor optimizer state: {exc}")
        if "critic_opt" in state:
            try:
                agent.critic_opt.load_state_dict(state["critic_opt"])
            except Exception as exc:
                print(f"Warning: failed to load critic optimizer state: {exc}")
        if "alpha_opt" in state:
            try:
                agent.alpha_opt.load_state_dict(state["alpha_opt"])
            except Exception as exc:
                print(f"Warning: failed to load alpha optimizer state: {exc}")

    start_episode = int(payload.get("episode", 0)) + 1
    gradient_steps = int(payload.get("gradient_steps", 0))
    env_steps = int(payload.get("env_steps", 0))
    return start_episode, gradient_steps, env_steps


def run_video_eval(
    eval_env: WidowXPickPlaceEnv,
    agent: CRLAgent,
    out_path: Path,
    episode_length: int,
    fps: int,
    eval_min_action_norm: float,
    eval_fallback_noise_std: float,
    eval_stochastic_fallback: bool,
) -> dict[str, float]:
    action_abs_values: list[float] = []
    fallback_steps = 0

    def policy(obs: dict[str, np.ndarray]) -> np.ndarray:
        nonlocal fallback_steps
        det_action = agent.act(obs["observation"], obs["desired_goal"], deterministic=True).astype(np.float32)

        use_fallback = eval_stochastic_fallback and float(np.linalg.norm(det_action)) < float(eval_min_action_norm)
        if use_fallback:
            fallback_steps += 1
            action = agent.act(obs["observation"], obs["desired_goal"], deterministic=False).astype(np.float32)
            if eval_fallback_noise_std > 0.0:
                action = np.clip(
                    action + np.random.normal(loc=0.0, scale=eval_fallback_noise_std, size=action.shape).astype(np.float32),
                    -1.0,
                    1.0,
                )
        else:
            action = det_action

        action_abs_values.append(float(np.mean(np.abs(action))))
        return action

    frames, metrics = record_policy_rollout(eval_env, policy, episode_length=episode_length, fps=fps)
    write_video(out_path, frames, fps=fps)
    metrics["action_abs_mean"] = float(np.mean(action_abs_values)) if action_abs_values else 0.0
    metrics["fallback_ratio"] = float(fallback_steps / max(1, len(action_abs_values)))
    return metrics


def main() -> None:
    args = parse_args()
    if (
        args.scripted_demo_episodes > 0
        and args.scripted_demo_oracle_assist
        and args.contrastive_goal_mode == "future"
        and not args.allow_oracle_future_goal_leak
    ):
        raise RuntimeError(
            "Oracle-assisted scripted demos leak forced future goals into future-goal CRL. "
            "Use real scripted successes, or pass --allow-oracle-future-goal-leak only for debugging."
        )
    if args.max_gpu_mode:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)

    env_cfg = WidowXEnvConfig(
        robot_xml_path=args.robot_xml_path,
        episode_length=args.episode_length,
        action_scale_arm=args.action_scale_arm,
        action_scale_gripper=args.action_scale_gripper,
        success_threshold_xy=args.success_threshold_xy,
        dense_reward_scale=args.dense_reward_scale,
        dense_reward_weight=args.dense_reward_weight,
        success_bonus=args.success_bonus,
        zone_half_extent_xy=args.zone_half_extent_xy,
        spawn_noise_xy=args.spawn_noise_xy,
        goal_noise_xy=args.goal_noise_xy,
        enable_grasp_assist=args.enable_grasp_assist,
        render_width=args.render_width,
        render_height=args.render_height,
        camera_name=args.camera_name,
        observation_mode=args.observation_mode,
        image_observation_width=args.image_observation_width,
        image_observation_height=args.image_observation_height,
        image_observation_grayscale=args.image_observation_grayscale,
        seed=args.seed,
    )

    train_render_mode = "rgb_array" if args.observation_mode == "image" else None
    env = WidowXPickPlaceEnv(env_cfg, render_mode=train_render_mode)
    eval_env = WidowXPickPlaceEnv(env_cfg, render_mode="rgb_array")

    first_obs, _ = env.reset(seed=args.seed)
    obs_dim = int(first_obs["observation"].shape[0])
    goal_dim = int(first_obs["desired_goal"].shape[0])
    action_dim = int(env.action_space.shape[0])
    obs_shape = None
    if args.observation_mode == "image":
        obs_channels = 1 if args.image_observation_grayscale else 3
        obs_shape = (obs_channels, args.image_observation_height, args.image_observation_width)

    agent_cfg = CRLConfig(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        repr_dim=args.repr_dim,
        residual_depth=args.residual_depth,
        residual_block_size=args.residual_block_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.goal_sample_gamma,
        energy_fn=args.energy_fn,
        critic_loss_type=args.critic_loss_type,
        logsumexp_penalty_coeff=args.logsumexp_penalty_coeff,
        reward_loss_coeff=args.reward_loss_coeff,
        actor_reward_bonus_coeff=args.actor_reward_bonus_coeff,
        actor_awbc_coeff=args.actor_awbc_coeff,
        actor_awbc_temp=args.actor_awbc_temp,
        target_entropy_scale=args.target_entropy_scale,
        min_log_alpha=args.min_log_alpha,
        max_log_alpha=args.max_log_alpha,
        obs_shape=obs_shape,
        device=args.device,
    )
    agent = CRLAgent(agent_cfg)

    start_episode, gradient_steps, env_steps = maybe_resume_from_checkpoint(
        agent,
        args.resume_checkpoint,
        load_optimizers=args.resume_load_optimizers,
        load_alpha=args.resume_load_alpha,
    )

    if env_steps <= 0 and start_episode > 1:
        env_steps = (start_episode - 1) * args.episode_length

    capacity_from_steps = int(np.ceil(args.buffer_capacity_transitions / max(1, args.episode_length)))
    buffer_capacity_episodes = max(args.buffer_capacity_episodes, capacity_from_steps)

    buffer = EpisodeReplayBuffer(
        capacity_episodes=buffer_capacity_episodes,
        her_ratio=args.her_ratio,
        reward_fn=lambda ag, dg: env.compute_reward(ag, dg),
        contrastive_goal_mode=args.contrastive_goal_mode,
        her_min_goal_dist=args.her_min_goal_dist,
        her_min_future_offset=args.her_min_future_offset,
        seed=args.seed,
    )

    demo_states: list[np.ndarray] = []
    demo_goals: list[np.ndarray] = []
    demo_actions: list[np.ndarray] = []

    scripted_metrics_acc = defaultdict(list)
    if args.scripted_demo_episodes > 0:
        accepted = 0
        attempts = 0
        max_attempts = max(args.scripted_demo_episodes, args.scripted_demo_episodes * args.scripted_demo_max_attempts_mult)
        while accepted < args.scripted_demo_episodes and attempts < max_attempts:
            attempts += 1
            demo_ep, demo_metrics = collect_episode(
                env,
                agent=None,
                random_policy=False,
                eps_random=0.0,
                action_noise_std=0.0,
                use_scripted_policy=True,
                max_steps_override=args.scripted_demo_max_steps,
                scripted_oracle_assist=args.scripted_demo_oracle_assist,
                collect_with_stochastic_policy=args.collect_with_stochastic_policy,
            )
            if float(demo_metrics.get("train/success", 0.0)) < float(args.scripted_demo_success_threshold):
                continue

            buffer.add_episode(demo_ep)
            demo_states.append(demo_ep.states[:-1])
            demo_goals.append(demo_ep.desired_goals)
            demo_actions.append(demo_ep.actions)
            for k, v in demo_metrics.items():
                scripted_metrics_acc[k.replace("train/", "scripted_demo/")].append(float(v))
            accepted += 1

        scripted_metrics_acc["scripted_demo/accepted_episodes"].append(float(accepted))
        scripted_metrics_acc["scripted_demo/attempted_episodes"].append(float(attempts))
        scripted_metrics_acc["scripted_demo/accept_rate"].append(float(accepted / max(attempts, 1)))

        if accepted < args.scripted_demo_episodes:
            raise RuntimeError(
                f"Collected {accepted}/{args.scripted_demo_episodes} successful scripted demos after {attempts} attempts. "
                "Increase --scripted-demo-max-attempts-mult or enable oracle assist."
            )

    demo_transitions: dict[str, np.ndarray] | None = None
    if demo_states:
        demo_transitions = {
            "states": np.concatenate(demo_states, axis=0).astype(np.float32),
            "goals": np.concatenate(demo_goals, axis=0).astype(np.float32),
            "actions": np.concatenate(demo_actions, axis=0).astype(np.float32),
        }

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "env_config": asdict(env_cfg),
            "agent_config": asdict(agent_cfg),
        },
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    video_dir = checkpoint_dir / "videos"

    if scripted_metrics_acc:
        wandb.log({k: float(np.mean(v)) for k, v in scripted_metrics_acc.items()}, step=max(0, start_episode - 1))

    if args.bc_pretrain_steps > 0:
        if demo_transitions is None:
            raise RuntimeError("BC pretrain requested but no scripted demos collected. Set --scripted-demo-episodes > 0.")
        bc_rng = np.random.default_rng(args.seed + 123)
        bc_acc = defaultdict(list)
        for _ in range(args.bc_pretrain_steps):
            bc_batch = sample_demo_batch(demo_transitions, args.bc_batch_size, bc_rng)
            bc_metrics = agent.behavior_clone_update(bc_batch)
            for k, v in bc_metrics.items():
                bc_acc[k].append(float(v))
        wandb.log({k: float(np.mean(v)) for k, v in bc_acc.items()}, step=max(0, start_episode - 1))

    has_bc_policy_warmstart = demo_transitions is not None and args.bc_pretrain_steps > 0

    progress = trange(start_episode, args.total_episodes + 1, desc="Training episodes")
    for episode_idx in progress:
        episode_rel_idx = episode_idx - start_episode + 1
        random_policy = (
            episode_rel_idx <= args.warmup_episodes
            and (not has_bc_policy_warmstart)
        ) or (env_steps < args.exploration_random_steps and (not has_bc_policy_warmstart))

        if args.exploration_total_steps > 0:
            schedule_step = min(env_steps, args.exploration_total_steps)
            schedule_horizon = args.exploration_total_steps
        else:
            schedule_step = max(0, episode_rel_idx - 1)
            schedule_horizon = max(1, args.exploration_eps_decay_episodes)

        eps_random = linear_schedule(
            args.exploration_eps_start,
            args.exploration_eps_end,
            schedule_step,
            schedule_horizon,
        )
        action_noise_std = linear_schedule(
            args.action_noise_start,
            args.action_noise_end,
            schedule_step,
            schedule_horizon,
        )
        action_noise_std = max(float(args.action_noise_floor), float(action_noise_std))
        if has_bc_policy_warmstart:
            eps_random *= max(0.0, args.post_bc_exploration_scale)
            action_noise_std *= max(0.0, args.post_bc_action_noise_scale)
        episode_batch, episode_metrics = collect_episode(
            env,
            agent,
            random_policy=random_policy,
            eps_random=eps_random,
            action_noise_std=action_noise_std,
            collect_with_stochastic_policy=args.collect_with_stochastic_policy,
        )
        buffer.add_episode(episode_batch)
        episode_steps = int(episode_metrics["train/episode_length"])
        env_steps += episode_steps

        update_acc = defaultdict(list)
        if buffer.num_transitions >= args.batch_size and episode_rel_idx > args.warmup_episodes:
            for _ in range(args.updates_per_episode):
                batch = buffer.sample(args.batch_size, gamma=args.goal_sample_gamma)
                update_metrics = agent.update(batch)
                for k, v in update_metrics.items():
                    update_acc[k].append(v)
                gradient_steps += 1

        mean_updates = {k: float(np.mean(v)) for k, v in update_acc.items()}

        log_data = {
            "episode": float(episode_idx),
            "train/buffer_episodes": float(buffer.num_episodes),
            "train/buffer_transitions": float(buffer.num_transitions),
            "train/gradient_steps": float(gradient_steps),
            "train/env_steps": float(env_steps),
            "train/eps_random": float(eps_random),
            "train/action_noise_std": float(action_noise_std),
            **episode_metrics,
            **mean_updates,
        }

        if run is not None:
            wandb.log(log_data, step=episode_idx)

        progress.set_postfix(
            success=f"{episode_metrics['train/success']:.2f}",
            ret=f"{episode_metrics['train/episode_return']:.2f}",
            buffer=buffer.num_transitions,
        )

        if episode_idx % args.checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"episode_{episode_idx:05d}.pt"
            save_checkpoint(ckpt_path, agent, episode_idx, gradient_steps, env_steps, args)

        if episode_idx % args.video_interval == 0:
            video_path = video_dir / f"episode_{episode_idx:05d}.mp4"
            try:
                eval_metrics = run_video_eval(
                    eval_env=eval_env,
                    agent=agent,
                    out_path=video_path,
                    episode_length=args.episode_length,
                    fps=args.video_fps,
                    eval_min_action_norm=args.eval_min_action_norm,
                    eval_fallback_noise_std=args.eval_fallback_noise_std,
                    eval_stochastic_fallback=args.eval_stochastic_fallback,
                )
                if run is not None:
                    wandb.log(
                        {
                            "eval/video": wandb.Video(video_path.as_posix(), fps=args.video_fps, format="mp4"),
                            "eval/return": eval_metrics["return"],
                            "eval/success": eval_metrics["success"],
                            "eval/frames": eval_metrics["length"],
                            "eval/action_abs_mean": eval_metrics.get("action_abs_mean", 0.0),
                            "eval/fallback_ratio": eval_metrics.get("fallback_ratio", 0.0),
                            "eval/error": 0.0,
                        },
                        step=episode_idx,
                    )
            except Exception as exc:
                msg = f"Warning: eval video failed at episode {episode_idx}: {exc}"
                print(msg)
                if run is not None:
                    wandb.log(
                        {
                            "eval/error": 1.0,
                        },
                        step=episode_idx,
                    )
                if not args.continue_on_eval_error:
                    raise

    final_ckpt = checkpoint_dir / "final.pt"
    save_checkpoint(final_ckpt, agent, args.total_episodes, gradient_steps, env_steps, args)

    env.close()
    eval_env.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
