from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EpisodeBatch:
    states: np.ndarray
    actions: np.ndarray
    desired_goals: np.ndarray
    achieved_goals: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


class EpisodeReplayBuffer:
    def __init__(
        self,
        capacity_episodes: int,
        her_ratio: float,
        reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        contrastive_goal_mode: str = "future",
        her_min_goal_dist: float = 0.03,
        her_min_future_offset: int = 4,
        seed: int = 0,
    ) -> None:
        self.capacity_episodes = int(capacity_episodes)
        self.her_ratio = float(her_ratio)
        self.reward_fn = reward_fn
        if contrastive_goal_mode not in {"future", "desired_or_her"}:
            raise ValueError("contrastive_goal_mode must be 'future' or 'desired_or_her'.")
        self.contrastive_goal_mode = contrastive_goal_mode
        self.her_min_goal_dist = float(her_min_goal_dist)
        self.her_min_future_offset = int(max(1, her_min_future_offset))
        self._episodes: list[EpisodeBatch] = []
        self._rng = np.random.default_rng(seed)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        return int(sum(ep.actions.shape[0] for ep in self._episodes))

    def add_episode(self, episode: EpisodeBatch) -> None:
        t = episode.actions.shape[0]
        if episode.states.shape[0] != t + 1:
            raise ValueError("Episode states must have length T+1.")
        if episode.achieved_goals.shape[0] != t + 1:
            raise ValueError("Episode achieved_goals must have length T+1.")
        if episode.desired_goals.shape[0] != t:
            raise ValueError("Episode desired_goals must have length T.")

        self._episodes.append(episode)
        if len(self._episodes) > self.capacity_episodes:
            self._episodes.pop(0)

    def _sample_future_index(self, t: int, episode_len: int, gamma: float) -> int:
        start = min(episode_len, t + self.her_min_future_offset)
        future_idx = np.arange(start, episode_len + 1)
        if future_idx.size == 0:
            future_idx = np.arange(t + 1, episode_len + 1)
        if future_idx.size == 1:
            return int(future_idx[0])
        offsets = future_idx - future_idx[0]
        probs = np.power(gamma, offsets, dtype=np.float64)
        probs = probs / probs.sum()
        return int(self._rng.choice(future_idx, p=probs))

    def sample(self, batch_size: int, gamma: float = 0.98) -> dict[str, np.ndarray]:
        if self.num_transitions < batch_size:
            raise RuntimeError("Not enough transitions in replay buffer.")

        valid_eps = [ep for ep in self._episodes if ep.actions.shape[0] >= 1]
        if not valid_eps:
            raise RuntimeError("Replay buffer is empty.")

        state_dim = valid_eps[0].states.shape[1]
        action_dim = valid_eps[0].actions.shape[1]
        goal_dim = valid_eps[0].desired_goals.shape[1]

        states = np.zeros((batch_size, state_dim), dtype=np.float32)
        actions = np.zeros((batch_size, action_dim), dtype=np.float32)
        goals = np.zeros((batch_size, goal_dim), dtype=np.float32)
        desired_goals = np.zeros((batch_size, goal_dim), dtype=np.float32)
        her_used = np.zeros((batch_size,), dtype=np.float32)
        next_states = np.zeros((batch_size, state_dim), dtype=np.float32)
        rewards = np.zeros((batch_size,), dtype=np.float32)
        discounts = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            ep = valid_eps[self._rng.integers(0, len(valid_eps))]
            ep_len = ep.actions.shape[0]
            t = int(self._rng.integers(0, ep_len))

            sampled_from_her = False
            goal_idx = self._sample_future_index(t, ep_len, gamma)
            candidate_goal = ep.achieved_goals[goal_idx]
            if self.contrastive_goal_mode == "future":
                sampled_goal = candidate_goal
                sampled_from_her = True
            else:
                if self._rng.random() < self.her_ratio:
                    # Avoid trivial relabels where achieved_next already matches sampled goal.
                    if float(np.linalg.norm((ep.achieved_goals[t + 1] - candidate_goal)[:2])) >= self.her_min_goal_dist:
                        sampled_goal = candidate_goal
                        sampled_from_her = True
                    else:
                        sampled_goal = ep.desired_goals[t]
                else:
                    sampled_goal = ep.desired_goals[t]

            achieved_next = ep.achieved_goals[t + 1]
            reward = self.reward_fn(achieved_next, sampled_goal)
            reward = float(np.asarray(reward).reshape(-1)[0])

            done = float(ep.dones[t])

            states[i] = ep.states[t]
            actions[i] = ep.actions[t]
            goals[i] = sampled_goal
            desired_goals[i] = ep.desired_goals[t]
            her_used[i] = 1.0 if sampled_from_her else 0.0
            next_states[i] = ep.states[t + 1]
            rewards[i] = reward
            discounts[i] = 1.0 - done

        return {
            "states": states,
            "actions": actions,
            "goals": goals,
            "desired_goals": desired_goals,
            "her_used": her_used,
            "next_states": next_states,
            "rewards": rewards,
            "discounts": discounts,
        }
