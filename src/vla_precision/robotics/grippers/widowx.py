"""WidowX AI gripper — the driver's 7th joint, not a separate device.

Unlike the UR (serial DH gripper) or Franka (serial PGI gripper), the WidowX AI gripper is
controlled through the same `trossen_arm.TrossenArmDriver` connection as the arm itself. But
`robotics/environments/factory.py` builds the `Gripper` *before* the `Robot` and passes it in as a
collaborator (see `build_gripper`/`build_robot`) — there is no hook for them to share a connection
object directly. `get_shared_driver` below is the seam: both `WidowXGripper` and `WidowXRobot` ask
for a driver keyed by (ip, model, end_effector) and get back the *same* object, configured once.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from vla_precision.config.schema import RootConfig

_DRIVER_LOCK = threading.Lock()
_DRIVERS: dict[tuple[str, str, str], Any] = {}


def get_shared_driver(ip: str, model_name: str, end_effector_name: str):
    """Return the one `TrossenArmDriver` for this (ip, model, end_effector), configuring it
    on first use. Safe to call from both the gripper and the robot factory for the same arm."""
    import trossen_arm

    key = (ip, model_name, end_effector_name)
    with _DRIVER_LOCK:
        driver = _DRIVERS.get(key)
        if driver is None:
            driver = trossen_arm.TrossenArmDriver()
            driver.configure(
                getattr(trossen_arm.Model, model_name),
                getattr(trossen_arm.StandardEndEffector, end_effector_name),
                ip,
                clear_error=True,
            )
            driver.set_all_modes(trossen_arm.Mode.position)
            _DRIVERS[key] = driver
        return driver


def release_shared_driver(ip: str, model_name: str, end_effector_name: str) -> None:
    """Idle and clean up the cached driver. Called once, by whichever of Robot/Gripper closes
    last (currently `WidowXRobot.close()`, since `Robot.close()` owns the shared lifecycle)."""
    import trossen_arm

    key = (ip, model_name, end_effector_name)
    with _DRIVER_LOCK:
        driver = _DRIVERS.pop(key, None)
    if driver is not None:
        driver.set_all_modes(trossen_arm.Mode.idle)
        driver.cleanup()


class WidowXGripper:
    """Thin actuator over the arm's own driver — safety clipping happens in `WidowXRobot`,
    not here; this class only forwards an already-safe target position."""

    def __init__(self, config: RootConfig):
        robot = config.robot
        model_name = str(robot.options.get("model", "wxai_v0"))
        end_effector_name = str(robot.options.get("end_effector", "wxai_v0_follower"))
        self._driver = get_shared_driver(robot.server_url, model_name, end_effector_name)
        self._goal_time = float(config.gripper.options.get("goal_time", 0.2))

    @property
    def action_dimension(self) -> int:
        return 1

    def command_chunk(self, positions: np.ndarray) -> None:
        position = float(np.asarray(positions, dtype=float).reshape(-1)[-1])
        self._driver.set_gripper_position(position, goal_time=self._goal_time, blocking=False)

    def observations(self, robot_state: dict) -> dict[str, np.ndarray]:
        del robot_state
        return {"gripper_position": np.asarray([self._driver.get_gripper_position()], dtype=np.float32)}

    def close(self) -> None:
        # WidowXRobot.close() owns and releases the shared driver.
        pass
