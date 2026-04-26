from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass
class WidowXEnvConfig:
    robot_xml_path: str
    episode_length: int = 200
    control_dt: float = 0.04
    physics_dt: float = 0.002
    action_scale_arm: float = 0.08
    action_scale_gripper: float = 0.006
    success_threshold_xy: float = 0.04
    dense_reward_scale: float = 0.12
    dense_reward_weight: float = 0.3
    reach_reward_scale: float = 0.08
    reach_reward_weight: float = 0.35
    lift_reward_weight: float = 0.35
    attach_bonus: float = 0.5
    success_bonus: float = 2.0
    enable_grasp_assist: bool = True
    grasp_distance_threshold: float = 0.03
    grasp_close_ctrl_threshold: float = 0.032
    grasp_release_ctrl_threshold: float = 0.022
    cube_size: float = 0.018
    red_zone_center_xy: tuple[float, float] = (0.24, -0.10)
    blue_zone_center_xy: tuple[float, float] = (0.24, 0.10)
    zone_half_extent_xy: float = 0.08
    spawn_noise_xy: float = 0.02
    goal_noise_xy: float = 0.02
    render_width: int = 640
    render_height: int = 480
    camera_name: str = "isometric"
    observation_mode: str = "state"
    image_observation_width: int = 64
    image_observation_height: int = 64
    image_observation_grayscale: bool = False
    seed: int = 0


class WidowXPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config: WidowXEnvConfig, render_mode: str | None = None):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self._rng = np.random.default_rng(config.seed)
        self._renderer: mujoco.Renderer | None = None
        self._render_failed = False
        self._render_error_logged = False

        self._generated_model_path = self._build_model_file()
        self.model = mujoco.MjModel.from_xml_path(self._generated_model_path.as_posix())
        self.model.opt.timestep = config.physics_dt
        self.data = mujoco.MjData(self.model)

        self.frame_skip = max(1, int(round(config.control_dt / config.physics_dt)))
        self.max_steps = config.episode_length

        self._arm_actuator_names = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
        ]
        self._actuator_ids = np.array(
            [self._name2id_or_raise(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self._arm_actuator_names],
            dtype=np.int32,
        )
        self._ctrl_range = self.model.actuator_ctrlrange[self._actuator_ids].copy()
        self._ctrl_target = np.zeros((len(self._actuator_ids),), dtype=np.float64)
        self._arm_dof_ids = np.array(
            [int(self.model.jnt_dofadr[int(self.model.actuator_trnid[aid, 0])]) for aid in self._actuator_ids[:6]],
            dtype=np.int32,
        )

        self._joint_names_for_obs = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "left_finger",
            "right_finger",
        ]
        self._joint_ids_for_obs = [self._name2id_or_raise(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self._joint_names_for_obs]
        self._qpos_ids = np.array([self.model.jnt_qposadr[jid] for jid in self._joint_ids_for_obs], dtype=np.int32)
        self._qvel_ids = np.array([self.model.jnt_dofadr[jid] for jid in self._joint_ids_for_obs], dtype=np.int32)

        self._cube_body_id = self._name2id_or_raise(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_joint_id = self._name2id_or_raise(mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        self._red_site_id = self._name2id_or_raise(mujoco.mjtObj.mjOBJ_SITE, "red_zone_site")
        self._blue_site_id = self._name2id_or_raise(mujoco.mjtObj.mjOBJ_SITE, "blue_zone_site")

        self._ee_body_id = self._resolve_ee_body_id()
        self._grasp_geom_ids = self._resolve_grasp_geom_ids()

        self._table_top_z = 0.04
        self._cube_rest_z = self._table_top_z + self.config.cube_size + 0.0005

        self._desired_goal = np.array([0.24, 0.10, self._cube_rest_z], dtype=np.float64)
        self._source_center = np.array([0.24, -0.10], dtype=np.float64)
        self._cube_spawn_xy = self._source_center.copy()
        self._cube_attached = False
        self._script_phase = "approach"

        zone_size = np.array([self.config.zone_half_extent_xy, self.config.zone_half_extent_xy, 0.001], dtype=np.float64)
        self.model.site_size[self._red_site_id] = zone_size
        self.model.site_size[self._blue_site_id] = zone_size

        if self.config.observation_mode not in {"state", "image"}:
            raise ValueError("observation_mode must be 'state' or 'image'.")

        obs, _ = self.reset(seed=config.seed)
        observation_dim = int(obs["observation"].shape[0])
        state_dim = int(obs["state"].shape[0])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )

    def _build_model_file(self) -> Path:
        template_path = Path(__file__).resolve().parent.parent / "assets" / "widowx_pick_place_template.xml"
        template = template_path.read_text(encoding="utf-8")
        robot_xml = Path(self.config.robot_xml_path).resolve()
        if not robot_xml.exists():
            raise FileNotFoundError(
                f"Robot XML not found: {robot_xml}. Fetch it with scripts/fetch_wx250s.sh first."
            )
        asset_dir = (robot_xml.parent / "assets").resolve()
        xml = template.replace("{{ROBOT_XML_PATH}}", robot_xml.name)
        xml = xml.replace("{{MESH_DIR}}", asset_dir.as_posix())
        xml = xml.replace("{{TEXTURE_DIR}}", asset_dir.as_posix())
        # Use a unique generated scene filename to avoid collisions across parallel runs.
        generated_path = robot_xml.parent / f"crl_widowx_pick_place_scene_{uuid4().hex}.xml"
        generated_path.write_text(xml, encoding="utf-8")
        return generated_path

    def _name2id_or_raise(self, obj_type: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise KeyError(f"Object not found in model: {name}")
        return int(idx)

    def _resolve_ee_body_id(self) -> int:
        candidates = ["wx250s/gripper_link", "gripper_link"]
        for name in candidates:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx >= 0:
                return int(idx)
        raise KeyError("Could not find end-effector body. Expected wx250s/gripper_link or gripper_link.")

    def _resolve_grasp_geom_ids(self) -> np.ndarray:
        candidates = ["left/left_g0", "left/left_g1", "right/right_g0", "right/right_g1"]
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in candidates]
        valid = [idx for idx in ids if idx >= 0]
        if valid:
            return np.asarray(valid, dtype=np.int32)
        raise KeyError("Could not find fingertip grasp geoms.")

    def _grasp_position(self) -> np.ndarray:
        return self.data.geom_xpos[self._grasp_geom_ids].mean(axis=0).copy()

    def _grasp_jacobian(self) -> np.ndarray:
        jacp_acc = np.zeros((3, self.model.nv), dtype=np.float64)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        for geom_id in self._grasp_geom_ids:
            jacp.fill(0.0)
            mujoco.mj_jacGeom(self.model, self.data, jacp, None, int(geom_id))
            jacp_acc += jacp
        return jacp_acc / float(len(self._grasp_geom_ids))

    def _apply_home_pose(self) -> None:
        if self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0]
            self.data.qvel[:] = 0.0
        else:
            self.data.qpos[:] = self.model.qpos0
            self.data.qvel[:] = 0.0

        for i, actuator_id in enumerate(self._actuator_ids):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qadr = int(self.model.jnt_qposadr[joint_id])
            self._ctrl_target[i] = self.data.qpos[qadr]
        self._ctrl_target = np.clip(self._ctrl_target, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
        self.data.ctrl[self._actuator_ids] = self._ctrl_target

    def _set_cube_pose(self, xy: np.ndarray) -> None:
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = np.array([xy[0], xy[1], self._cube_rest_z])
        self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.model.jnt_dofadr[self._cube_joint_id] : self.model.jnt_dofadr[self._cube_joint_id] + 6] = 0.0

    def _sample_task(self) -> None:
        # Keep source zone fixed, but randomize spawn and target within their respective zones.
        self._source_center = np.array(self.config.red_zone_center_xy, dtype=np.float64)
        target_center_xy = np.array(self.config.blue_zone_center_xy, dtype=np.float64)

        max_spawn = min(self.config.spawn_noise_xy, 0.4 * self.config.zone_half_extent_xy)
        spawn_noise = self._rng.uniform(-max_spawn, max_spawn, size=(2,))
        self._cube_spawn_xy = self._source_center + spawn_noise

        max_goal = min(self.config.goal_noise_xy, 0.5 * self.config.zone_half_extent_xy)
        goal_noise = self._rng.uniform(-max_goal, max_goal, size=(2,))
        target_xy = target_center_xy + goal_noise
        self._desired_goal = np.array([target_xy[0], target_xy[1], self._cube_rest_z], dtype=np.float64)

        self.model.site_pos[self._red_site_id] = np.array([self._source_center[0], self._source_center[1], 0.041], dtype=np.float64)
        self.model.site_pos[self._blue_site_id] = np.array([target_center_xy[0], target_center_xy[1], 0.041], dtype=np.float64)

    def _get_state_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self._qpos_ids].astype(np.float32)
        qvel = self.data.qvel[self._qvel_ids].astype(np.float32)

        ee_pos = self._grasp_position().astype(np.float32)
        cube_pos = self.data.xpos[self._cube_body_id].astype(np.float32)

        cube_to_ee = (cube_pos - ee_pos).astype(np.float32)
        grip_opening = np.array([qpos[6] - qpos[7]], dtype=np.float32)

        return np.concatenate(
            [
                qpos,
                qvel,
                ee_pos,
                cube_pos,
                cube_to_ee,
                grip_opening,
            ],
            axis=0,
        ).astype(np.float32)

    def _resize_frame_nearest(self, frame: np.ndarray) -> np.ndarray:
        target_h = int(self.config.image_observation_height)
        target_w = int(self.config.image_observation_width)
        src_h, src_w = frame.shape[:2]
        y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int32)
        x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int32)
        return frame[y_idx][:, x_idx]

    def _get_image_obs(self) -> np.ndarray:
        frame = self._render_rgb_frame(require=True)
        small = self._resize_frame_nearest(frame).astype(np.float32) / 255.0
        if self.config.image_observation_grayscale:
            small = (
                0.299 * small[..., 0]
                + 0.587 * small[..., 1]
                + 0.114 * small[..., 2]
            )[..., None]
        return np.transpose(small, (2, 0, 1)).reshape(-1).astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        state = self._get_state_obs()
        observation = state if self.config.observation_mode == "state" else self._get_image_obs()
        cube_pos = self.data.xpos[self._cube_body_id].astype(np.float32)

        return {
            "observation": observation.astype(np.float32),
            "state": state.astype(np.float32),
            "achieved_goal": cube_pos.astype(np.float32),
            "desired_goal": self._desired_goal.astype(np.float32),
        }

    def _distance_xy(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)

    def _inside_zone_xy(self, point_xyz: np.ndarray, zone_center_xy: np.ndarray) -> bool:
        dxy = np.abs(np.asarray(point_xyz)[..., :2] - np.asarray(zone_center_xy)[..., :2])
        return bool(np.all(dxy <= self.config.success_threshold_xy))

    def _is_success_blue_zone(self) -> bool:
        cube_pos = self.data.xpos[self._cube_body_id]
        return self._inside_zone_xy(cube_pos, self._desired_goal)

    def _update_grasp_assist(self) -> None:
        if not self.config.enable_grasp_assist:
            return

        ee_pos = self._grasp_position()
        cube_pos = self.data.xpos[self._cube_body_id]
        gripper_ctrl = float(self._ctrl_target[-1])
        ee_cube_xy = float(np.linalg.norm((ee_pos - cube_pos)[:2]))
        ee_cube_z = float(abs(ee_pos[2] - cube_pos[2]))

        if (
            (not self._cube_attached)
            and ee_cube_xy < self.config.grasp_distance_threshold
            and ee_cube_z < max(0.035, self.config.cube_size * 2.5)
            and gripper_ctrl >= self.config.grasp_close_ctrl_threshold
        ):
            self._cube_attached = True

        if self._cube_attached and gripper_ctrl <= self.config.grasp_release_ctrl_threshold:
            release_pos = ee_pos.copy()
            release_pos[2] = self._cube_rest_z
            self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = release_pos
            self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            dof_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0
            self._cube_attached = False

        if self._cube_attached:
            target_pos = ee_pos.copy()
            target_pos[2] = max(target_pos[2], self._cube_rest_z)
            self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = target_pos
            self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            dof_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        # Goal-only reward used by HER relabeling in replay.
        achieved = np.asarray(achieved_goal)
        desired = np.asarray(desired_goal)
        dist = self._distance_xy(achieved, desired)

        # Match env success semantics: being inside the target zone in XY.
        dxy = np.abs(achieved[..., :2] - desired[..., :2])
        success = np.all(dxy <= self.config.success_threshold_xy, axis=-1).astype(np.float32)

        dense = np.exp(-dist / max(self.config.dense_reward_scale, 1e-6)).astype(np.float32)
        reward = self.config.dense_reward_weight * dense + self.config.success_bonus * success
        return reward

    def _compute_step_reward(self) -> tuple[float, dict[str, float]]:
        ee_pos = self._grasp_position()
        cube_pos = self.data.xpos[self._cube_body_id]

        dist_goal = float(self._distance_xy(cube_pos, self._desired_goal))
        dist_reach = float(np.linalg.norm(ee_pos - cube_pos))
        lift = max(0.0, float(cube_pos[2] - self._cube_rest_z))

        goal_term = float(np.exp(-dist_goal / max(self.config.dense_reward_scale, 1e-6)))
        reach_term = float(np.exp(-dist_reach / max(self.config.reach_reward_scale, 1e-6)))
        lift_term = float(np.tanh(lift / 0.04))
        attached = 1.0 if self._cube_attached else 0.0
        success = 1.0 if self._is_success_blue_zone() else 0.0

        reward = (
            self.config.dense_reward_weight * goal_term
            + self.config.reach_reward_weight * reach_term
            + self.config.lift_reward_weight * lift_term
            + self.config.attach_bonus * attached
            + self.config.success_bonus * success
        )

        terms = {
            "reward_goal_term": goal_term,
            "reward_reach_term": reach_term,
            "reward_lift_term": lift_term,
            "reward_attach_term": attached,
            "reward_success_term": success,
        }
        return float(reward), terms

    def scripted_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Generate a scripted action using Jacobian-based Cartesian tracking plus gripper logic."""
        ee_pos = self._grasp_position()
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        goal_pos = np.asarray(obs["desired_goal"], dtype=np.float64)

        if self._is_success_blue_zone():
            self._script_phase = "release" if self._cube_attached else "retreat"

        dist_xy_to_cube = float(np.linalg.norm((ee_pos - cube_pos)[:2]))
        dist_z_to_cube = float(abs(ee_pos[2] - cube_pos[2]))
        cube_lifted = bool(cube_pos[2] > self._cube_rest_z + 0.02)

        if self._script_phase == "approach" and dist_xy_to_cube < 0.025:
            self._script_phase = "descend"
        if self._script_phase == "descend" and dist_xy_to_cube < 0.025 and dist_z_to_cube < 0.03:
            self._script_phase = "close"
        if self._script_phase == "close" and (self._cube_attached or cube_lifted):
            self._script_phase = "lift"
        if self._script_phase == "lift" and cube_pos[2] > self._cube_rest_z + 0.045:
            self._script_phase = "to_goal"
        if self._script_phase == "to_goal" and float(np.linalg.norm((cube_pos - goal_pos)[:2])) < 0.03:
            self._script_phase = "release"
        if self._script_phase == "release" and (not self._cube_attached) and self._is_success_blue_zone():
            self._script_phase = "retreat"

        hover_cube = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.09], dtype=np.float64)
        near_cube = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.005], dtype=np.float64)
        lift_pos = np.array([cube_pos[0], cube_pos[1], self._cube_rest_z + 0.10], dtype=np.float64)
        goal_hover = np.array([goal_pos[0], goal_pos[1], self._cube_rest_z + 0.10], dtype=np.float64)
        goal_drop = np.array([goal_pos[0], goal_pos[1], self._cube_rest_z + 0.02], dtype=np.float64)
        retreat_pos = np.array([ee_pos[0], ee_pos[1], self._cube_rest_z + 0.13], dtype=np.float64)

        grip_cmd = -1.0
        if self._script_phase == "approach":
            target_pos = hover_cube
        elif self._script_phase == "descend":
            target_pos = near_cube
        elif self._script_phase == "close":
            target_pos = near_cube
            grip_cmd = 1.0
        elif self._script_phase == "lift":
            target_pos = lift_pos
            grip_cmd = 1.0
        elif self._script_phase == "to_goal":
            target_pos = goal_hover
            grip_cmd = 1.0
        elif self._script_phase == "release":
            target_pos = goal_drop
            grip_cmd = -1.0
        else:
            target_pos = retreat_pos

        jacp = self._grasp_jacobian()
        j_ee = jacp[:, self._arm_dof_ids]

        pos_err = target_pos - ee_pos
        desired_ee_vel = np.clip(6.0 * pos_err, -0.5, 0.5)
        dq = np.linalg.pinv(j_ee, rcond=1e-3) @ desired_ee_vel

        arm_action = np.clip(dq / max(self.config.action_scale_arm, 1e-6), -1.0, 1.0)
        action = np.zeros((7,), dtype=np.float32)
        action[:6] = arm_action.astype(np.float32)
        action[6] = float(np.clip(grip_cmd, -1.0, 1.0))
        return action

    def scripted_oracle_finalize(self) -> tuple[dict[str, np.ndarray], float, dict[str, float]]:
        """Force-place the cube into the blue zone for high-quality successful demo trajectories."""
        goal_xy = self._desired_goal[:2].copy()
        self._cube_attached = False
        self._set_cube_pose(goal_xy)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        reward, reward_terms = self._compute_step_reward()
        dist = float(self._distance_xy(obs["achieved_goal"], obs["desired_goal"]))
        info = {
            "is_success": float(self._is_success_blue_zone()),
            "distance_to_goal_xy": dist,
            **reward_terms,
        }
        return obs, float(reward), info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._cube_attached = False
        self._script_phase = "approach"
        self._apply_home_pose()
        self._sample_task()
        self._set_cube_pose(self._cube_spawn_xy)

        mujoco.mj_forward(self.model, self.data)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        dist = float(self._distance_xy(obs["achieved_goal"], obs["desired_goal"]))
        info = {
            "is_success": float(self._is_success_blue_zone()),
            "distance_to_goal_xy": dist,
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        action = np.clip(action, -1.0, 1.0)
        scaled = np.concatenate(
            [
                action[:6] * self.config.action_scale_arm,
                action[6:] * self.config.action_scale_gripper,
            ]
        )
        self._ctrl_target = np.clip(self._ctrl_target + scaled, self._ctrl_range[:, 0], self._ctrl_range[:, 1])

        for _ in range(self.frame_skip):
            self.data.ctrl[self._actuator_ids] = self._ctrl_target
            mujoco.mj_step(self.model, self.data)
            self._update_grasp_assist()

        self._step_count += 1
        obs = self._get_obs()
        reward, reward_terms = self._compute_step_reward()

        dist = float(self._distance_xy(obs["achieved_goal"], obs["desired_goal"]))
        terminated = False
        truncated = self._step_count >= self.max_steps
        info = {
            "is_success": float(self._is_success_blue_zone()),
            "distance_to_goal_xy": dist,
            "ctrl_target": self._ctrl_target.copy(),
            **reward_terms,
        }
        return obs, reward, terminated, truncated, info

    def _render_rgb_frame(self, require: bool = False) -> np.ndarray | None:
        if self._render_failed:
            if require:
                raise RuntimeError("Rendering is unavailable after a previous renderer failure.")
            return None

        # Avoid known noisy GLFW failures on headless nodes without explicit MUJOCO_GL backend.
        if self._renderer is None:
            display = os.environ.get("DISPLAY", "")
            mujoco_gl = os.environ.get("MUJOCO_GL", "").lower()
            display_socket_missing = False
            if display.startswith(":"):
                disp_num = display[1:].split(".")[0]
                if disp_num.isdigit():
                    display_socket_missing = not Path(f"/tmp/.X11-unix/X{disp_num}").exists()

            if ((display == "") or display_socket_missing) and (mujoco_gl == ""):
                if not self._render_error_logged:
                    reason = "missing DISPLAY" if display == "" else f"invalid DISPLAY={display}"
                    print(f"Warning: render disabled ({reason}, MUJOCO_GL not set).")
                    self._render_error_logged = True
                self._render_failed = True
                if require:
                    raise RuntimeError("MUJOCO_GL or a valid DISPLAY is required for image observations.")
                return None

        try:
            if self._renderer is None:
                renderer = mujoco.Renderer(self.model, self.config.render_height, self.config.render_width)
                self._renderer = renderer

            self._renderer.update_scene(self.data, camera=self.config.camera_name)
            frame = self._renderer.render()
            return np.asarray(frame, dtype=np.uint8)
        except Exception as exc:
            if not self._render_error_logged:
                print(f"Warning: render disabled after failure: {exc}")
                self._render_error_logged = True
            self._render_failed = True
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:
                    pass
                self._renderer = None
            if require:
                raise RuntimeError(f"Renderer failed while building image observation: {exc}") from exc
            return None

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        return self._render_rgb_frame(require=False)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        if hasattr(self, "_generated_model_path") and self._generated_model_path.exists():
            self._generated_model_path.unlink(missing_ok=True)
