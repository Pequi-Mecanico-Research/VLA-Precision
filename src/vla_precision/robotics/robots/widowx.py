"""WidowX AI joint-space client — the first joint-space robot in this framework.

UR5e and Franka both control a Cartesian TCP pose (see `ur.py`/`franka.py`), with the reference-
frame and idle-hold math that implies. WidowX AI's native action space is the `trossen_arm` SDK's
own joint-space vector (6 arm joints in rad + 1 gripper in m), so none of that pose math applies
here — this class is considerably simpler as a result.

The gripper is the driver's 7th joint, not a separate device; see `grippers/widowx.py` for how
this `Robot` and its `Gripper` collaborator share one `TrossenArmDriver` connection.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from vla_precision.config.schema import RobotConfig
from vla_precision.robotics.grippers.base import Gripper
from vla_precision.robotics.grippers.widowx import get_shared_driver, release_shared_driver

LOGGER = logging.getLogger(__name__)

NUM_JOINTS = 7  # joint_0..joint_5 (rad) + gripper (m)


class WidowXRobot:
    """Joint-space client for the Trossen WidowX AI, talking directly to the `trossen_arm` SDK
    (no ROS, no `lerobot_robot_trossen`). Safety is two-layered, matching the same approach
    already used by the Caminho-A inference backend (`integrations/openpi/inference/backends/widowx.py`):
    absolute clipping to the driver's real joint limits, and a per-step relative-motion cap.
    """

    action_low = -np.inf
    action_high = np.inf

    def __init__(
        self,
        config: RobotConfig,
        *,
        gripper: Gripper,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.gripper = gripper
        self._sleep = sleep
        self._clock = clock

        self._ip = config.server_url
        self._model_name = str(config.options.get("model", "wxai_v0"))
        self._end_effector_name = str(config.options.get("end_effector", "wxai_v0_follower"))
        self.driver = get_shared_driver(self._ip, self._model_name, self._end_effector_name)

        num_joints = self.driver.get_num_joints()
        if num_joints != NUM_JOINTS:
            raise RuntimeError(f"Expected {NUM_JOINTS} joints (6 arm + gripper), driver reports {num_joints}.")
        self.joint_limits = self.driver.get_joint_limits()

        self.control_hz = float(config.control_hz)
        self.reset_positions = np.asarray(config.arms.left.reset_pose, dtype=np.float64)
        if self.reset_positions.shape != (NUM_JOINTS,):
            raise ValueError(
                f"robot.arms.left.reset_pose must contain {NUM_JOINTS} values (6 joints + gripper "
                f"in rad/m, this robot's native units — not a Cartesian pose), got "
                f"{self.reset_positions.shape}."
            )
        self.reset_pose_range = np.asarray(config.arms.left.reset_pose_range, dtype=np.float64)
        self.random_reset = bool(config.random_reset)
        self.reset_time_sec = float(config.options.get("reset_time_sec", 5.0))
        self.goal_time_per_step = float(config.options.get("goal_time_per_step", 1.5 / self.control_hz))

        max_relative_step = config.options.get("max_relative_step", {})
        arm_step = float(max_relative_step.get("arm", 0.15))
        gripper_step = float(max_relative_step.get("gripper", 0.02))
        self.max_relative_step = np.array([arm_step] * 6 + [gripper_step], dtype=float)

        self._state = np.asarray(self.driver.get_all_positions(), dtype=np.float64)

    @property
    def action_dimension(self) -> int:
        return 6 + self.gripper.action_dimension

    @property
    def currpos(self) -> np.ndarray:
        return self._state.copy()

    def refresh_state(self) -> None:
        self._state = np.asarray(self.driver.get_all_positions(), dtype=np.float64)

    def observations(self) -> dict[str, Any]:
        self.refresh_state()
        state: dict[str, Any] = {"joint_positions": self._state[:6].copy()}
        state.update(self.gripper.observations({"joint_positions": self._state}))
        return state

    def _clip_to_joint_limits(self, positions: np.ndarray) -> np.ndarray:
        lo = np.array([limit.position_min for limit in self.joint_limits], dtype=float)
        hi = np.array([limit.position_max for limit in self.joint_limits], dtype=float)
        return np.clip(positions, lo, hi)

    def _safe_target(self, target: np.ndarray) -> np.ndarray:
        """Absolute joint-limit clipping, then bound how far a single commanded step may move
        each joint (max_relative_target) — never trust a policy still in training beyond this."""
        target = self._clip_to_joint_limits(np.asarray(target, dtype=float))
        delta = np.clip(target - self._state, -self.max_relative_step, self.max_relative_step)
        return self._state + delta

    def execute_action_chunk(self, action: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None]
        if chunk.ndim != 2 or chunk.shape[-1] != self.action_dimension:
            raise ValueError(f"WidowX action must have shape (T, {self.action_dimension}), got {chunk.shape}")

        executed = []
        for row in chunk:
            started = self._clock()
            if not np.all(np.isfinite(row)):
                raise ValueError(f"WidowX action contains non-finite values: {row.tolist()}")
            safe_target = self._safe_target(row[:NUM_JOINTS])

            # One driver call moves the 6 arm joints; the gripper slot echoes the *current*
            # gripper reading so it is not disturbed by this call — the separate
            # self.gripper.command_chunk() below is what actually moves the gripper. Two SDK
            # calls instead of one atomic 7-joint move, mirroring how FrankaRobot separates arm
            # and gripper actuation; the driver serializes both onto the same physical arm.
            arm_target = list(safe_target[:6]) + [float(self._state[6])]
            self.driver.set_all_positions(arm_target, goal_time=self.goal_time_per_step, blocking=False)
            if self.gripper.action_dimension:
                self.gripper.command_chunk(safe_target[6:7])

            self._state = safe_target
            executed.append(row.copy())

            elapsed = self._clock() - started
            to_sleep = 1.0 / self.control_hz - elapsed
            if to_sleep > 0:
                self._sleep(to_sleep)
        return np.asarray(executed, dtype=np.float32)

    def _sample_reset_target(self) -> np.ndarray:
        target = self.reset_positions.copy()
        if not self.random_reset or self.reset_pose_range.size == 0:
            return target
        random_range = np.abs(self.reset_pose_range)
        target = target + np.random.uniform(-random_range, random_range)
        return target

    def reset(self, *, joint_reset: bool = False, options: dict[str, Any] | None = None) -> dict[str, Any]:
        del joint_reset, options
        target = self._clip_to_joint_limits(self._sample_reset_target())
        self.driver.set_all_positions(list(target), goal_time=self.reset_time_sec, blocking=True)
        self._state = np.asarray(self.driver.get_all_positions(), dtype=np.float64)
        return self.observations()

    def request(self, name: str, enabled: bool) -> Any:
        raise NotImplementedError(f"WidowX AI has no named request {name!r}={enabled!r}")

    def close(self) -> None:
        try:
            self.gripper.close()
        finally:
            release_shared_driver(self._ip, self._model_name, self._end_effector_name)
