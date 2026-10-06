#!/usr/bin/env python3
"""Standalone WidowX AI inference entrypoint for OpenPI + VLA-Precision checkpoints.

Robot control talks to the `trossen_arm` SDK directly (joint-space position control), not through
ROS or the full `lerobot_robot_trossen` package. Unlike UR5e/Franka, the WidowX AI action space is
already the driver's native joint-space vector (6 arm joints + 1 gripper, in rad/m) so there is no
Cartesian/quaternion math here, and the gripper is just the driver's 7th joint, not a separate
serial device.
"""

import argparse
import dataclasses
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(message)s")

import numpy as np
import trossen_arm
import yaml
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig

from vla_precision import image_tools
from vla_precision.integrations.openpi.checkpoints import (
    resolve_openpi_assets,
    resolve_openpi_checkpoint,
)

from .recording import Recorder
from .utils import FpsCounter

NUM_JOINTS = 7  # joint_0..joint_5 (rad) + gripper (m) — action space, and the position half of state.
# observation/state is 14D: the 7 joint positions above, THEN the 7 external efforts in the same
# order (Nm for arm joints, N for the gripper) — confirmed from the real dataset schema
# (docs/widowx-integration.md §6, docs/widowx-setup.md) and the training config that produced the
# checkpoints. WidowXInputs forwards this column verbatim with no slicing, so the live observation
# built here must match this shape and order exactly.
STATE_DIM = 2 * NUM_JOINTS


def _resolve_model_path(value) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _validate_widowx_norm_stats(norm_stats, *, path: Path | None = None) -> None:
    """WidowX AI action is 7D; state is 14D (7 positions + 7 external efforts) — see STATE_DIM."""

    def _dim(keys: tuple[str, ...]) -> int | None:
        if norm_stats is None:
            return None
        for key in keys:
            stats = norm_stats.get(key)
            if stats is None:
                continue
            mean = stats.get("mean") if isinstance(stats, dict) else getattr(stats, "mean", None)
            if mean is not None:
                return int(np.asarray(mean).reshape(-1).size)
        return None

    state_dim = _dim(("state", "observation/state"))
    action_dim = _dim(("actions", "action"))
    if state_dim != STATE_DIM or action_dim != NUM_JOINTS:
        raise ValueError(
            f"WidowX AI inference requires {STATE_DIM}-D state and {NUM_JOINTS}-D action norm "
            f"stats, got state={state_dim}, action={action_dim} in {path}."
        )


class Inference:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        model = cfg["model"]
        robot = cfg["robot"]
        self.remote_policy = cfg.get("policy", {}).get("location", "local") == "server"
        self.action_horizon = int(robot["action_horizon"])
        self.direct_action_horizon = bool(robot.get("direct_action_horizon", True))
        self.model_config = None
        if not self.remote_policy:
            from vla_precision.integrations.openpi import configs as _config

            self.model_config = _config.get_config(model["name"])
            if self.direct_action_horizon and int(self.model_config.model.action_horizon) != self.action_horizon:
                self.model_config = dataclasses.replace(
                    self.model_config,
                    model=dataclasses.replace(self.model_config.model, action_horizon=self.action_horizon),
                )

        self.checkpoint_dir: Path | None = None
        self.sample_steps = int(model.get("sample_steps", 10))
        self.norm_stats_path: Path | None = None
        self.norm_stats = None
        if not self.remote_policy:
            # Only resolved locally: in "server" mode the checkpoint lives on the GPU machine
            # running `openpi-inference serve-policy`, not on this robot-side client — resolving
            # it here would fail filesystem checks against a path that doesn't exist on this host.
            checkpoint = resolve_openpi_checkpoint(model["checkpoint_dir"], requested_step=model.get("checkpoint_step"))
            self.checkpoint_dir = Path(checkpoint.directory)
            self._configure_checkpoint_norm_stats(model)

        # Camera config
        cam = cfg["cameras"]
        self.cam_high_serial = cam["cam_high_serial"]
        self.cam_wrist_serial = cam["cam_wrist_serial"]
        self.cam_low_serial = cam["cam_low_serial"]
        self.cam_fps = cam.get("fps", 30)
        self.cam_width = cam.get("width", 640)
        self.cam_height = cam.get("height", 480)

        # Video config
        video = cfg["video"]
        self.video_fps = video.get("fps", 15)
        self.visualize = video["visualize"]
        record = cfg.get("record", {})
        self.num_episodes = record.get("num_episodes", 10)
        self.ep_timeout = record.get("episode_timeout_sec", 60)
        self.show_action_fps = record.get("show_action_fps", False)
        self.show_inference_fps = record.get("show_inference_fps", False)
        self.show_time = record.get("show_time", False)

        # Robot config
        self.robot_ip = robot["ip"]
        self.arm_model = getattr(trossen_arm.Model, robot.get("model", "wxai_v0"))
        self.end_effector = getattr(trossen_arm.StandardEndEffector, robot.get("end_effector", "wxai_v0_follower"))
        self.reset_positions = np.asarray(robot["reset_positions"], dtype=float)
        if self.reset_positions.shape != (NUM_JOINTS,):
            raise ValueError(f"robot.reset_positions must contain {NUM_JOINTS} values, got {self.reset_positions.shape}.")
        self.reset_time_sec = float(robot.get("reset_time_sec", 5.0))
        self.action_fps = float(robot["action_fps"])
        self.goal_time_per_step = float(robot.get("goal_time_per_step", 1.5 / self.action_fps))
        self.debug = robot.get("debug", False)

        max_relative_step = robot.get("max_relative_step", {})
        arm_step = float(max_relative_step.get("arm", 0.15))
        gripper_step = float(max_relative_step.get("gripper", 0.02))
        self.max_relative_step = np.array([arm_step] * 6 + [gripper_step], dtype=float)

        # Task config
        task = cfg["task"]
        self.task_description = task["description"]

        # timestamps / output paths (mirrors the other native backends)
        time_str = time.strftime("%Y%m%d-%H%M%S")
        time_path = time.strftime("%Y%m%d")
        base_dir = Path(cfg.get("output_root", "./results")).expanduser() / cfg["experiment"]["name"] / "openpi-native"
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path
        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        latest_path = log_dir / "latest.yaml"
        log_path = log_dir / "all_logs" / f"log_{time_str}.yaml"
        # The shared Recorder is shaped for at most 2 cameras plus an optional
        # third ("right_wrist") slot used by the dual-arm backends — WidowX has
        # 3 independent cameras from a single arm, so cam_low is recorded
        # through that third slot (see _transfer_obs_for_recorder below).
        wrist_video = video_dir / f"wrist_{time_str}.mp4"
        high_video = video_dir / f"high_{time_str}.mp4"
        low_video = video_dir / f"low_{time_str}.mp4"
        self.recorder = Recorder(
            log_path=log_path,
            video_path=[wrist_video, high_video, low_video],
            display_fps=self.video_fps,
            visualize=self.visualize,
        )
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(log_path)

        self.fps_action = FpsCounter(name="action")

        # Internal state
        self.driver: trossen_arm.TrossenArmDriver | None = None
        self.joint_limits: list[trossen_arm.JointLimit] | None = None
        self.cameras = None
        self._last_commanded_positions: np.ndarray | None = None
        self._ep_start = None
        self._ep_idx = 1
        self._ep_steps = 0
        self._ep_done = False
        self._success_episodes = 0
        self._failed_episodes = 0
        self._reset_requested = threading.Event()
        self._success_requested = threading.Event()
        self._keyboard_listener_stop = threading.Event()
        self._keyboard_listener_thread = None

    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to the WidowX AI arm and print its current state."""
        logging.info("\n===== [ROBOT] Connecting to WidowX AI =====")
        self.driver = trossen_arm.TrossenArmDriver()
        self.driver.configure(self.arm_model, self.end_effector, self.robot_ip, clear_error=True)
        num_joints = self.driver.get_num_joints()
        if num_joints != NUM_JOINTS:
            raise RuntimeError(f"Expected {NUM_JOINTS} joints (6 arm + gripper), driver reports {num_joints}.")
        self.joint_limits = self.driver.get_joint_limits()
        self.driver.set_all_modes(trossen_arm.Mode.position)
        positions = list(self.driver.get_all_positions())
        logging.info(f"[ROBOT] Current joint positions (rad, rad, ..., m): {[round(p, 4) for p in positions]}")
        logging.info("===== [ROBOT] WidowX AI initialized successfully =====\n")

    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect the 3 RealSense cameras."""
        try:
            logging.info("\n===== [CAM] Initializing cameras =====")

            def _cam_cfg(serial: str) -> RealSenseCameraConfig:
                return RealSenseCameraConfig(
                    serial_number_or_name=serial,
                    fps=self.cam_fps,
                    width=self.cam_width,
                    height=self.cam_height,
                    color_mode=ColorMode.RGB,
                    use_depth=False,
                    rotation=Cv2Rotation.NO_ROTATION,
                )

            camera_config = {
                "high_image": _cam_cfg(self.cam_high_serial),
                "wrist_image": _cam_cfg(self.cam_wrist_serial),
                "low_image": _cam_cfg(self.cam_low_serial),
            }
            self.cameras = make_cameras_from_configs(camera_config)
            for name, cam in self.cameras.items():
                cam.connect()
                logging.info(f"[CAM] {name} connected successfully.")
            logging.info("===== [CAM] Cameras initialized successfully =====\n")
        except Exception as e:
            logging.error("[ERROR] Failed to initialize cameras.")
            logging.error(f"Exception: {e}\n")
            self.cameras = None

    # --------------------------- SAFETY --------------------------- #
    def _clip_to_joint_limits(self, positions: np.ndarray) -> np.ndarray:
        if self.joint_limits is None:
            return positions
        lo = np.array([limit.position_min for limit in self.joint_limits], dtype=float)
        hi = np.array([limit.position_max for limit in self.joint_limits], dtype=float)
        return np.clip(positions, lo, hi)

    def _clip_relative_step(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Bound how far a single commanded step may move each joint (max_relative_target)."""
        delta = np.clip(target - current, -self.max_relative_step, self.max_relative_step)
        return current + delta

    def _safe_target(self, target: np.ndarray) -> np.ndarray:
        target = self._clip_to_joint_limits(np.asarray(target, dtype=float))
        if self._last_commanded_positions is None:
            self._last_commanded_positions = np.asarray(self.driver.get_all_positions(), dtype=float)
        safe = self._clip_relative_step(target, self._last_commanded_positions)
        self._last_commanded_positions = safe
        return safe

    # --------------------------- RESET --------------------------- #
    def reset_episode(self):
        logging.info("[RESET] Moving to reset joint positions.")
        self._last_commanded_positions = None
        target = self._safe_target(self.reset_positions)
        self.driver.set_all_positions(list(target), goal_time=self.reset_time_sec, blocking=True)
        self._last_commanded_positions = np.asarray(self.driver.get_all_positions(), dtype=float)

    # --------------------------- OBS --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        """Return the current observation in WidowXInputs format."""
        positions = np.asarray(self.driver.get_all_positions(), dtype=np.float32)
        efforts = np.asarray(self.driver.get_all_external_efforts(), dtype=np.float32)
        state = np.concatenate([positions, efforts])  # 14D: 7 positions then 7 efforts, see STATE_DIM
        obs = {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(self.cameras["high_image"].read(), 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(self.cameras["wrist_image"].read(), 224, 224)
            ),
            "observation/low_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(self.cameras["low_image"].read(), 224, 224)
            ),
            "prompt": self.task_description,
        }
        return obs

    @staticmethod
    def _transfer_obs_for_recorder(obs: Dict[str, Any]) -> Dict[str, Any]:
        """Reuse the shared Recorder's 3-image (dual-arm-shaped) visualization path."""
        return {
            "observation/wrist_image": obs["observation/wrist_image"],
            "observation/right_wrist_image": obs["observation/low_image"],
            "observation/exterior_image": obs["observation/image"],
        }

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray):
        """Execute a chunk of model actions as absolute joint-space targets."""
        if self.driver is None:
            logging.error("[ERROR] Robot driver not connected. Cannot execute actions.")
            return
        self.fps_action.reset()
        for action in np.asarray(actions)[: self.action_horizon]:
            start_time = time.perf_counter()
            action = np.asarray(action, dtype=float)
            if action.ndim != 1 or action.size < NUM_JOINTS:
                raise ValueError(f"Each model action must contain at least {NUM_JOINTS} values, got {action.shape}.")
            target = self._safe_target(action[:NUM_JOINTS])
            if not self.debug:
                self.driver.set_all_positions(list(target), goal_time=self.goal_time_per_step, blocking=False)

            elapsed = time.perf_counter() - start_time
            to_sleep = 1.0 / self.action_fps - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            self.fps_action.update(show=self.show_action_fps)

    # --------------------------- CHECKPOINT --------------------------- #
    def _configure_checkpoint_norm_stats(self, model: dict) -> None:
        assets = resolve_openpi_assets(
            model["checkpoint_dir"],
            default_asset_id=model.get("asset_id", ""),
            requested_step=model.get("checkpoint_step"),
        )
        if assets is None:
            raise FileNotFoundError(f"No norm_stats.json found under WidowX AI checkpoint: {self.checkpoint_dir}")
        self.norm_stats_path = Path(assets.norm_stats_path)
        data_assets = dataclasses.replace(
            self.model_config.data.assets,
            assets_dir=assets.directory,
            asset_id=assets.asset_id,
        )
        self.model_config = dataclasses.replace(
            self.model_config,
            data=dataclasses.replace(self.model_config.data, assets=data_assets),
        )
        data_config = self.model_config.data.create(self.model_config.assets_dirs, self.model_config.model)
        self.norm_stats = data_config.norm_stats
        _validate_widowx_norm_stats(self.norm_stats, path=self.norm_stats_path)
        logging.info(f"[MODEL] Using OpenPI norm stats from checkpoint: {self.norm_stats_path}")

    def _create_openpi_policy(self):
        from openpi.policies import policy_config as _policy_config

        if not (self.checkpoint_dir / "params").exists():
            raise FileNotFoundError(f"OpenPI checkpoint params not found: {self.checkpoint_dir / 'params'}")

        logging.info(f"[MODEL] OpenPI config: {self.model_config.name}")
        logging.info(f"[MODEL] checkpoint: {self.checkpoint_dir}")
        logging.info(f"[MODEL] norm stats: {self.norm_stats_path}")
        logging.info(f"[MODEL] diffusion sample steps: {self.sample_steps}")
        return _policy_config.create_trained_policy(
            self.model_config,
            self.checkpoint_dir,
            sample_kwargs={"num_steps": self.sample_steps},
        )

    # --------------------------- KEYBOARD EPISODE CONTROL --------------------------- #
    def _keyboard_episode_control_listener(self):
        """Request reset on double Enter, or success on double Space within 0.5 seconds."""
        import select
        import sys
        import termios
        import tty

        last_enter_time = None
        last_space_time = None
        old_termios = None
        if sys.stdin.isatty():
            old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        try:
            while not self._keyboard_listener_stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                char = sys.stdin.read(1)
                now = time.perf_counter()
                if char in ("\n", "\r"):
                    if last_enter_time is not None and now - last_enter_time <= 0.5:
                        self._reset_requested.set()
                        last_enter_time = None
                        logging.info("[RESET] Double Enter detected. Reset requested; current episode marked failed.")
                    else:
                        last_enter_time = now
                    continue
                if char != " ":
                    continue
                if last_space_time is not None and now - last_space_time <= 0.5:
                    self._success_requested.set()
                    last_space_time = None
                    logging.info("[EPISODE] Double Space detected. Success requested.")
                else:
                    last_space_time = now
        finally:
            if old_termios is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)

    def _start_keyboard_episode_control_listener(self):
        self._reset_requested.clear()
        self._success_requested.clear()
        self._keyboard_listener_stop.clear()
        self._keyboard_listener_thread = threading.Thread(target=self._keyboard_episode_control_listener, daemon=True)
        self._keyboard_listener_thread.start()
        logging.info("[EPISODE] Press Enter twice within 0.5s to fail/reset, or Space twice within 0.5s to mark success.")

    def _stop_keyboard_episode_control_listener(self):
        self._keyboard_listener_stop.set()
        if self._keyboard_listener_thread is not None:
            self._keyboard_listener_thread.join(timeout=0.2)
            self._keyboard_listener_thread = None

    def _wait_for_next_episode_start(self):
        input(f"Press Enter to start episode {self._ep_idx}...")

    # --------------------------- PIPELINE --------------------------- #
    def _prepare_inference(self):
        logging.info("========== Starting Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()
        self.reset_episode()

        policy = self._create_openpi_policy()
        obs = self.get_obs_state()
        logging.info("Warming up the WidowX AI OpenPI model")
        start = time.time()
        policy.infer(obs)
        logging.info(f"Model warmup completed, took {time.time() - start:.2f}s")
        input("Press Enter to continue inference...")
        return policy

    def _start_episode(self):
        self._ep_start = time.perf_counter()
        self._ep_steps = 0
        self._ep_done = False

    def _submit_step(self, result, obs, infer_idx: int):
        robot_actions = np.asarray(result["actions"])[..., :NUM_JOINTS]
        self.execute_actions(robot_actions)
        self._ep_steps += 1
        self.recorder.submit_actions(
            robot_actions[: self.action_horizon],
            infer_idx,
            obs["prompt"],
            state=obs["observation/state"],
            episode_index=self._ep_idx,
            episode_action_count=self._ep_steps,
            action_horizon=self.action_horizon,
        )
        self.recorder.submit_obs(self._transfer_obs_for_recorder(obs))

    def _submit_episode(self, status: str, completed: bool, **extra):
        duration = time.perf_counter() - self._ep_start
        self.recorder.submit_episode_result(
            episode_index=self._ep_idx,
            duration_sec=duration,
            status=status,
            action_batches=self._ep_steps,
            completed=completed,
            **extra,
        )
        if completed:
            self._success_episodes += 1
        else:
            self._failed_episodes += 1
        total_ended = self._success_episodes + self._failed_episodes
        logging.info(
            f"[EPISODE] Success/failure ratio after episode {self._ep_idx}: "
            f"success {self._success_episodes}/{total_ended}, failure {self._failed_episodes}/{total_ended}"
        )
        self._ep_done = True

    def _finish_episode_if_needed(self) -> bool:
        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self._success_requested.clear()
            logging.info("[RESET] Restarting current episode from reset position.")
            self._submit_episode("keyboard_reset", completed=False, reason="enter")
        elif self._success_requested.is_set():
            self._success_requested.clear()
            self._submit_episode("manual_success", completed=True, reason="double_space")
        elif time.perf_counter() - self._ep_start >= self.ep_timeout:
            logging.info(f"[RESET] Episode timeout ({self.ep_timeout}s). Restarting current episode.")
            self._submit_episode("timeout", completed=False, reason="episode_timeout")
        else:
            return False

        if self._ep_idx >= self.num_episodes:
            self._ep_idx += 1
            return True

        self._ep_idx += 1
        self._stop_keyboard_episode_control_listener()
        self.reset_episode()
        self._wait_for_next_episode_start()
        self._start_keyboard_episode_control_listener()
        self._start_episode()
        return True

    def _reset_before_exit(self):
        if self.driver is None:
            return
        try:
            logging.info("[RESET] Resetting robot before exit.")
            self.reset_episode()
        except Exception as e:
            logging.error(f"[ERROR] Failed to reset robot before exit: {e}")

    def _ask_save_video(self):
        try:
            ans = input("Save recorded videos before exiting? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving recorded videos before exiting...")
                self.recorder.save_video()
        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")

    def run(self):
        """Main pipeline: connect robot, cameras, and run inference."""
        try:
            policy = self._prepare_inference()
            infer_idx = 1
            self._start_keyboard_episode_control_listener()
            self._start_episode()
            logging.info("========== Starting Inference Loop ==========")
            while self._ep_idx <= self.num_episodes:
                loop_start = time.perf_counter()
                obs = self.get_obs_state()
                result = policy.infer(obs)
                self._submit_step(result, obs, infer_idx)
                self._finish_episode_if_needed()
                if self.show_inference_fps:
                    loop_elapsed = time.perf_counter() - loop_start
                    logging.info(f"[STATE] Inference loop rate: {1 / loop_elapsed:.1f} Hz")
                infer_idx += 1
            logging.info(f"[INFO] Finished {self.num_episodes} inference episodes.")
        except KeyboardInterrupt:
            logging.info("[INFO] KeyboardInterrupt detected. Saving recorded videos before exiting...")
        except Exception as e:
            logging.error(f"[ERROR] Inference loop encountered an error: {e}")
        finally:
            self._stop_keyboard_episode_control_listener()
            self._reset_before_exit()
            self.recorder.submit_episode_summary(
                action_horizon=self.action_horizon,
                description=self.task_description,
            )
            if self.driver is not None:
                self.driver.set_all_modes(trossen_arm.Mode.idle)
                self.driver.cleanup()

        self._ask_save_video()


# --------------------------- MAIN --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run standalone WidowX AI inference with an OpenPI checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "widowx.yaml",
        help="Path to the VLA-Precision WidowX inference YAML config.",
    )
    args = parser.parse_args()
    inference = Inference(args.config)
    inference.run()


if __name__ == "__main__":
    main()
