from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class CRLConfig:
    obs_dim: int
    goal_dim: int
    action_dim: int
    hidden_dim: int = 256
    repr_dim: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.98
    energy_fn: str = "norm"
    critic_loss_type: str = "fwd_infonce"
    logsumexp_penalty_coeff: float = 0.1
    target_entropy_scale: float = -1.0
    reward_loss_coeff: float = 1.0
    actor_reward_bonus_coeff: float = 0.5
    actor_awbc_coeff: float = 0.0
    actor_awbc_temp: float = 0.5
    min_log_alpha: float = -4.0
    max_log_alpha: float = 2.0
    normalize_repr: bool = True
    obs_shape: tuple[int, int, int] | None = None
    device: str = "cpu"


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, obs_shape: tuple[int, int, int] | None = None) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        if obs_shape is None:
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.output_dim = hidden_dim
            return

        channels, height, width = obs_shape
        if obs_dim != channels * height * width:
            raise ValueError(f"obs_dim={obs_dim} does not match obs_shape={obs_shape}.")
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_dim = int(self.conv(dummy).shape[-1])
        self.proj = nn.Sequential(
            nn.Linear(conv_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_shape is None:
            return self.encoder(obs)
        batch = obs.shape[0]
        x = obs.reshape(batch, *self.obs_shape)
        return self.proj(self.conv(x))


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int,
        obs_shape: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.obs_encoder = ObservationEncoder(obs_dim, hidden_dim, obs_shape)
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_encoder.output_dim + goal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: torch.Tensor, goals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_features = self.obs_encoder(states)
        x = torch.cat([obs_features, goals], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)

        log_std = torch.tanh(self.log_std_head(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def sample(
        self, states: torch.Tensor, goals: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(states, goals)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        pre_tanh = mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob, mean


class StateActionEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        repr_dim: int,
        obs_shape: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.obs_encoder = ObservationEncoder(obs_dim, hidden_dim, obs_shape)
        self.net = nn.Sequential(
            nn.Linear(self.obs_encoder.output_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs_features = self.obs_encoder(states)
        return self.net(torch.cat([obs_features, actions], dim=-1))


class RewardModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_dim: int,
        hidden_dim: int,
        obs_shape: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.obs_encoder = ObservationEncoder(obs_dim, hidden_dim, obs_shape)
        self.net = nn.Sequential(
            nn.Linear(self.obs_encoder.output_dim + action_dim + goal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        obs_features = self.obs_encoder(states)
        return self.net(torch.cat([obs_features, actions, goals], dim=-1))


class CRLAgent:
    def __init__(self, config: CRLConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.actor = Actor(
            config.obs_dim,
            config.goal_dim,
            config.action_dim,
            config.hidden_dim,
            obs_shape=config.obs_shape,
        ).to(self.device)
        self.sa_encoder = StateActionEncoder(
            config.obs_dim,
            config.action_dim,
            config.hidden_dim,
            config.repr_dim,
            obs_shape=config.obs_shape,
        ).to(self.device)
        self.g_encoder = MLP(config.goal_dim, config.hidden_dim, config.repr_dim).to(self.device)
        self.reward_head = RewardModel(
            config.obs_dim,
            config.action_dim,
            config.goal_dim,
            config.hidden_dim,
            obs_shape=config.obs_shape,
        ).to(self.device)

        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.target_entropy = config.target_entropy_scale * float(config.action_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.sa_encoder.parameters())
            + list(self.g_encoder.parameters())
            + list(self.reward_head.parameters()),
            lr=config.critic_lr,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

    @staticmethod
    def _energy(x: torch.Tensor, y: torch.Tensor, fn_name: str) -> torch.Tensor:
        if fn_name == "norm":
            return -torch.sqrt(((x - y) ** 2).sum(dim=-1) + 1e-6)
        if fn_name == "dot":
            return (x * y).sum(dim=-1)
        if fn_name == "cosine":
            num = (x * y).sum(dim=-1)
            den = torch.linalg.norm(x, dim=-1) * torch.linalg.norm(y, dim=-1) + 1e-6
            return num / den
        if fn_name == "l2":
            return -((x - y) ** 2).sum(dim=-1)
        raise ValueError(f"Unknown energy function: {fn_name}")

    @staticmethod
    def _contrastive_loss(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
        diag = torch.diag(logits)

        if loss_type == "fwd_infonce":
            return -(diag - torch.logsumexp(logits, dim=1)).mean()
        if loss_type == "bwd_infonce":
            return -(diag - torch.logsumexp(logits, dim=0)).mean()
        if loss_type == "sym_infonce":
            row = torch.logsumexp(logits, dim=1)
            col = torch.logsumexp(logits, dim=0)
            return -(2.0 * diag - row - col).mean()
        if loss_type == "binary_nce":
            target = torch.eye(logits.shape[0], device=logits.device)
            return F.binary_cross_entropy_with_logits(logits, target)
        raise ValueError(f"Unknown contrastive loss: {loss_type}")

    def _encode_sa(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z = self.sa_encoder(states, actions)
        if self.config.normalize_repr:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def _encode_g(self, goals: torch.Tensor) -> torch.Tensor:
        z = self.g_encoder(goals)
        if self.config.normalize_repr:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def _q_values(self, states: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        sa_z = self._encode_sa(states, actions)
        g_z = self._encode_g(goals)
        return self._energy(sa_z, g_z, self.config.energy_fn)

    def _reward_values(self, states: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return self.reward_head(states, actions, goals).squeeze(-1)

    @torch.no_grad()
    def act(self, state: Any, goal: Any, deterministic: bool = False) -> Any:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        goal_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean = self.actor.sample(state_t, goal_t, deterministic=deterministic)
        mean_action = torch.tanh(mean)
        chosen = mean_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy()

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        goals = torch.as_tensor(batch["goals"], dtype=torch.float32, device=self.device)
        actor_goals = torch.as_tensor(batch.get("desired_goals", batch["goals"]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)

        sa_z = self._encode_sa(states, actions)
        g_z = self._encode_g(goals)
        logits = self._energy(sa_z.unsqueeze(1), g_z.unsqueeze(0), self.config.energy_fn)

        critic_loss = self._contrastive_loss(logits, self.config.critic_loss_type)
        # Stabilize logit scale without driving bounded energies (e.g. norm/cosine) to their minimum.
        lse = torch.logsumexp(logits, dim=1)
        lse_target = math.log(float(logits.shape[1]))
        logsumexp_penalty = self.config.logsumexp_penalty_coeff * torch.mean((lse - lse_target) ** 2)
        reward_pred = self._reward_values(states, actions, goals)
        reward_loss = F.mse_loss(reward_pred, rewards)
        critic_total = critic_loss + logsumexp_penalty + self.config.reward_loss_coeff * reward_loss
        if not torch.isfinite(critic_total):
            return {
                "critic/loss": float("nan"),
                "critic/logsumexp_penalty": float("nan"),
                "critic/diag_logits": float("nan"),
                "reward/loss": float("nan"),
                "reward/pred_mean": float("nan"),
                "reward/target_mean": float("nan"),
                "actor/loss": float("nan"),
                "actor/log_prob": float("nan"),
                "actor/q_pi": float("nan"),
                "actor/reward_pi": float("nan"),
                "alpha/loss": float("nan"),
                "alpha/value": float(self.log_alpha.exp().item()),
            }

        self.critic_opt.zero_grad(set_to_none=True)
        critic_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters()) + list(self.reward_head.parameters()),
            max_norm=10.0,
        )
        self.critic_opt.step()

        for p in self.sa_encoder.parameters():
            p.requires_grad_(False)
        for p in self.g_encoder.parameters():
            p.requires_grad_(False)
        for p in self.reward_head.parameters():
            p.requires_grad_(False)

        sampled_action, log_prob, mean = self.actor.sample(states, actor_goals, deterministic=False)
        q_pi = self._q_values(states, sampled_action, actor_goals)
        reward_pi = self._reward_values(states, sampled_action, actor_goals)

        alpha = self.log_alpha.exp()
        actor_sac_loss = (alpha.detach() * log_prob - q_pi - self.config.actor_reward_bonus_coeff * reward_pi).mean()

        mean_action = torch.tanh(mean)
        with torch.no_grad():
            if self.config.actor_awbc_temp > 0.0:
                norm_rewards = (rewards - rewards.mean()) / max(self.config.actor_awbc_temp, 1e-6)
                aw_weights = torch.softmax(norm_rewards, dim=0) * float(rewards.shape[0])
            else:
                aw_weights = torch.ones_like(rewards)
        awbc_per_sample = ((mean_action - actions) ** 2).mean(dim=-1)
        awbc_loss = (aw_weights * awbc_per_sample).mean()

        actor_loss = actor_sac_loss + self.config.actor_awbc_coeff * awbc_loss
        if not torch.isfinite(actor_loss):
            for p in self.sa_encoder.parameters():
                p.requires_grad_(True)
            for p in self.g_encoder.parameters():
                p.requires_grad_(True)
            for p in self.reward_head.parameters():
                p.requires_grad_(True)
            return {
                "critic/loss": float(critic_loss.item()),
                "critic/logsumexp_penalty": float(logsumexp_penalty.item()),
                "critic/diag_logits": float(torch.diag(logits).mean().item()),
                "reward/loss": float(reward_loss.item()),
                "reward/pred_mean": float(reward_pred.mean().item()),
                "reward/target_mean": float(rewards.mean().item()),
                "actor/loss": float("nan"),
                "actor/sac_loss": float("nan"),
                "actor/awbc_loss": float("nan"),
                "actor/log_prob": float("nan"),
                "actor/q_pi": float("nan"),
                "actor/reward_pi": float("nan"),
                "actor/det_action_abs_mean": float("nan"),
                "alpha/loss": float("nan"),
                "alpha/value": float(self.log_alpha.exp().item()),
            }

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        for p in self.sa_encoder.parameters():
            p.requires_grad_(True)
        for p in self.g_encoder.parameters():
            p.requires_grad_(True)
        for p in self.reward_head.parameters():
            p.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        if not torch.isfinite(alpha_loss):
            alpha_loss = torch.zeros_like(alpha_loss)
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.data.clamp_(self.config.min_log_alpha, self.config.max_log_alpha)

        with torch.no_grad():
            diag_logits = torch.diag(logits).mean().item()
            alpha_value = self.log_alpha.exp().item()
            mean_now, log_std_now = self.actor(states, goals)

        return {
            "critic/loss": float(critic_loss.item()),
            "critic/logsumexp_penalty": float(logsumexp_penalty.item()),
            "critic/diag_logits": float(diag_logits),
            "reward/loss": float(reward_loss.item()),
            "reward/pred_mean": float(reward_pred.mean().item()),
            "reward/target_mean": float(rewards.mean().item()),
            "actor/loss": float(actor_loss.item()),
            "actor/sac_loss": float(actor_sac_loss.item()),
            "actor/awbc_loss": float(awbc_loss.item()),
            "actor/log_prob": float(log_prob.mean().item()),
            "actor/q_pi": float(q_pi.mean().item()),
            "actor/reward_pi": float(reward_pi.mean().item()),
            "actor/det_action_abs_mean": float(mean_action.abs().mean().item()),
            "actor/mean_abs_pre_tanh": float(mean_now.abs().mean().item()),
            "actor/log_std_mean": float(log_std_now.mean().item()),
            "actor/log_std_min": float(log_std_now.min().item()),
            "actor/log_std_max": float(log_std_now.max().item()),
            "replay/contrastive_future_goal_ratio": float(np.mean(batch["her_used"])) if "her_used" in batch else 0.0,
            "replay/her_used_ratio": float(np.mean(batch["her_used"])) if "her_used" in batch else 0.0,
            "alpha/loss": float(alpha_loss.item()),
            "alpha/value": float(alpha_value),
        }

    def behavior_clone_update(self, batch: dict[str, Any]) -> dict[str, float]:
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        goals = torch.as_tensor(batch["goals"], dtype=torch.float32, device=self.device)
        expert_actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)

        mean, _ = self.actor(states, goals)
        pred_actions = torch.tanh(mean)

        bc_loss = F.mse_loss(pred_actions, expert_actions)

        self.actor_opt.zero_grad(set_to_none=True)
        bc_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            mae = torch.mean(torch.abs(pred_actions - expert_actions))

        return {
            "bc/loss": float(bc_loss.item()),
            "bc/action_mae": float(mae.item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "sa_encoder": self.sa_encoder.state_dict(),
            "g_encoder": self.g_encoder.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.sa_encoder.load_state_dict(state["sa_encoder"])
        self.g_encoder.load_state_dict(state["g_encoder"])
        if "reward_head" in state:
            self.reward_head.load_state_dict(state["reward_head"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])
