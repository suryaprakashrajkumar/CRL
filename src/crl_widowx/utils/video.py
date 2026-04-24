from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def write_video(path: str | Path, frames: list[np.ndarray], fps: int = 25) -> None:
    if len(frames) == 0:
        raise RuntimeError("No frames were captured for video output.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path.as_posix(), frames, fps=fps)


def record_policy_rollout(
    env: Any,
    policy_fn: Any,
    episode_length: int,
    fps: int = 25,
) -> tuple[list[np.ndarray], dict[str, float]]:
    frames: list[np.ndarray] = []
    obs, info = env.reset()
    ep_return = 0.0
    success = float(info.get("is_success", 0.0))

    for _ in range(episode_length):
        try:
            frame = env.render()
        except Exception:
            frame = None
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))

        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        success = max(success, float(info.get("is_success", 0.0)))

        if terminated or truncated:
            break

    try:
        last_frame = env.render()
    except Exception:
        last_frame = None
    if last_frame is not None:
        frames.append(np.asarray(last_frame, dtype=np.uint8))

    metrics = {
        "return": ep_return,
        "success": success,
        "length": float(len(frames)),
    }
    return frames, metrics
