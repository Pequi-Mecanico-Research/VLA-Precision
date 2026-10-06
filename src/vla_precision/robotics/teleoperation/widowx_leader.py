"""WidowX AI leader-arm teleoperation device for human intervention.

Unlike the keyboard experts (single_keyboard.py/dual_keyboard.py), this device does not produce
Cartesian pose deltas — it reads the ABSOLUTE 7-joint position (6 arm joints in rad + gripper in m)
of a second, physical, gravity-compensated leader arm, connected via its own `trossen_arm`
connection (a different IP than the follower's). See docs/widowx-intervencao-plano.md for the full
design: the leader arm is kept passively mirrored to the follower whenever the policy is in
control, so there is no jump in either direction when a human engages/releases control.
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NUM_JOINTS = 7  # joint_0..joint_5 (rad) + gripper (m) — same convention as WidowXRobot.

_DEFAULT_STAGED_POSITIONS = [0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0]


class WidowXLeaderArm:
    """Owns the leader arm's own `trossen_arm.TrossenArmDriver` connection.

    Two modes, switched explicitly on toggle edges (never polled): `Mode.position` while the
    policy drives the follower — this arm is actively mirrored to match it via `mirror()` — and
    `Mode.external_effort` (zero effort, gravity-compensated / backdrivable) while a human has
    engaged control.
    """

    def __init__(
        self,
        ip: str,
        *,
        model_name: str = "wxai_v0",
        end_effector_name: str = "wxai_v0_leader",
        staged_positions: list[float] | None = None,
        mirror_goal_time: float = 0.1,
    ):
        self._ip = ip
        self._model_name = model_name
        self._end_effector_name = end_effector_name
        self._staged_positions = list(staged_positions) if staged_positions is not None else list(
            _DEFAULT_STAGED_POSITIONS
        )
        if len(self._staged_positions) != NUM_JOINTS:
            raise ValueError(
                f"staged_positions must contain {NUM_JOINTS} values, got {len(self._staged_positions)}"
            )
        self._mirror_goal_time = float(mirror_goal_time)
        self._driver = None
        self._human_in_control = False

    def connect(self) -> None:
        import trossen_arm

        logger.info("[LEADER] Connecting to WidowX AI leader arm at %s", self._ip)
        self._driver = trossen_arm.TrossenArmDriver()
        self._driver.configure(
            getattr(trossen_arm.Model, self._model_name),
            getattr(trossen_arm.StandardEndEffector, self._end_effector_name),
            self._ip,
            clear_error=True,
        )
        self._driver.set_all_modes(trossen_arm.Mode.position)
        self._driver.set_all_positions(self._staged_positions, goal_time=2.0, blocking=True)
        logger.info("[LEADER] WidowX AI leader arm connected and staged.")

    def mirror(self, follower_positions: np.ndarray) -> None:
        """Keep the leader arm passively tracking the follower while the policy is in control.

        A no-op while a human is engaged, so engaging never fights this call.
        """
        if self._human_in_control or self._driver is None:
            return
        positions = np.asarray(follower_positions, dtype=float).reshape(-1)
        if positions.shape[0] != NUM_JOINTS:
            raise ValueError(f"mirror() expects {NUM_JOINTS} values, got {positions.shape}")
        self._driver.set_all_positions(list(positions), goal_time=self._mirror_goal_time, blocking=False)

    def engage_human_control(self) -> None:
        import trossen_arm

        if self._driver is None:
            raise RuntimeError("WidowXLeaderArm.connect() must be called first")
        self._human_in_control = True
        self._driver.set_all_modes(trossen_arm.Mode.external_effort)
        self._driver.set_all_external_efforts([0.0] * NUM_JOINTS, goal_time=0.0, blocking=True)
        logger.info("[LEADER] Human control engaged (gravity-compensated).")

    def disengage_human_control(self) -> None:
        import trossen_arm

        if self._driver is None:
            return
        self._human_in_control = False
        self._driver.set_all_modes(trossen_arm.Mode.position)
        logger.info("[LEADER] Human control released; resuming passive mirroring.")

    def get_action(self) -> np.ndarray:
        if not self._human_in_control:
            raise RuntimeError("get_action() called while the leader arm is not under human control")
        return np.asarray(self._driver.get_all_positions(), dtype=np.float32)

    def close(self) -> None:
        if self._driver is None:
            return
        self.disengage_human_control()
        self._driver.set_all_positions(self._staged_positions, goal_time=2.0, blocking=True)
        self._driver.set_all_positions([0.0] * NUM_JOINTS, goal_time=2.0, blocking=True)
        self._driver.cleanup()
        self._driver = None


class ToggleKeyListener:
    """Minimal single-key toggle, run in its own process.

    Mirrors SingleKeyboardExpert's `multiprocessing.Process` pattern (not `threading`) to keep
    `pynput`'s event loop isolated from the main actor process, which already runs JAX.
    """

    def __init__(self, key: str = "t"):
        self._key = key
        self._manager = multiprocessing.Manager()
        self._state = self._manager.Value("b", False)
        self._process = multiprocessing.Process(target=self._listen, daemon=True)
        self._process.start()

    def _listen(self) -> None:
        from pynput import keyboard

        target_key = self._key.lower()

        def on_press(key):
            char = getattr(key, "char", None)
            if char is not None and char.lower() == target_key:
                self._state.value = not self._state.value

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    @property
    def human_in_control(self) -> bool:
        return bool(self._state.value)

    def close(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


class WidowXLeaderExpert:
    """Combines the leader arm and the toggle listener into the shape LeaderArmIntervention
    expects. Built by `build_widowx_leader_expert`, registered as the "widowx_leader"
    teleoperation kind.
    """

    def __init__(self, arm: WidowXLeaderArm, toggle: ToggleKeyListener):
        self.arm = arm
        self.toggle = toggle
        self.arm.connect()

    @property
    def human_in_control(self) -> bool:
        return self.toggle.human_in_control

    def get_action(self) -> np.ndarray:
        return self.arm.get_action()

    def mirror(self, follower_positions: np.ndarray) -> None:
        self.arm.mirror(follower_positions)

    def engage(self) -> None:
        self.arm.engage_human_control()

    def disengage(self) -> None:
        self.arm.disengage_human_control()

    def close(self) -> None:
        self.toggle.close()
        self.arm.close()


def build_widowx_leader_expert(teleop_config: Any, dual_arm: bool, completion_event: Any) -> WidowXLeaderExpert:
    """Factory matching the `teleoperation_factories[kind](config, dual_arm, completion_event)`
    signature expected by `robotics/environments/factory.py:build_environment`.
    """
    del completion_event
    if dual_arm:
        raise NotImplementedError("WidowX leader-arm intervention does not support dual-arm setups yet")
    options = teleop_config.options
    arm = WidowXLeaderArm(
        ip=teleop_config.device,
        model_name=str(options.get("model", "wxai_v0")),
        end_effector_name=str(options.get("end_effector", "wxai_v0_leader")),
        staged_positions=options.get("staged_positions"),
        mirror_goal_time=float(options.get("mirror_goal_time", 0.1)),
    )
    toggle = ToggleKeyListener(key=str(options.get("trigger_key", "t")))
    return WidowXLeaderExpert(arm, toggle)
