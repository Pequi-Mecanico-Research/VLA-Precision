"""Joint-space human-intervention wrapper for the WidowX AI, driven by a physical leader arm
instead of keyboard pose deltas.

A sibling of `KeyboardIntervention` (robotics/wrappers/keyboard_intervention.py), not a
replacement — UR/Franka keep using that wrapper unchanged. This one has no notion of Cartesian
pose deltas, "pose_pressed"/"gripper_pressed" buttons, or `base_action_to_tcp_action`: it relies on
`WidowXRobot.execute_action_chunk` already accepting absolute joint-space targets natively and
applying its own safety clipping. See docs/widowx-intervencao-plano.md for the full design.
"""

from __future__ import annotations

import threading
from typing import Any

import gymnasium as gym
import numpy as np


class LeaderArmIntervention(gym.ActionWrapper):
    """Toggle-key-driven intervention via a leader arm.

    While the toggle is off: actions pass straight through to the policy, and the leader arm is
    kept passively mirrored to the follower's current pose (so there is no jump whenever the human
    engages). While on: each control step executes the leader arm's live absolute joint position
    on the follower directly, and the executed sequence is packaged as `intervene_action` for the
    Correction Buffer — same contract `acob_stream/actor.py` already expects from
    `KeyboardIntervention`, minus the pose/gripper/stop-pressed metadata fields (deliberately
    omitted: `_effective_intervention` treats their absence as "no detailed metadata" and counts
    the whole transition as an effective intervention, which is exactly the semantics here).
    """

    def __init__(self, env, expert, completion_event: threading.Event | None = None):
        super().__init__(env)
        self.expert = expert
        self.completion_event = completion_event or threading.Event()
        self._was_human_in_control = False

    def _sync_toggle_edge(self) -> bool:
        """Engage/disengage the leader arm on a toggle transition; return the current state."""
        human_in_control = bool(self.expert.human_in_control)
        if human_in_control and not self._was_human_in_control:
            self.expert.engage()
        elif not human_in_control and self._was_human_in_control:
            self.expert.disengage()
        self._was_human_in_control = human_in_control
        return human_in_control

    def step(self, action):
        action = np.asarray(action)
        if action.ndim != 2:
            raise ValueError(f"LeaderArmIntervention expects action chunk shape (T, A), got {action.shape}")

        human_in_control = self._sync_toggle_edge()

        if not human_in_control:
            obs, rew, done, truncated, info = self.env.step(action)
            try:
                self.expert.mirror(self.unwrapped.robot.currpos)
            except Exception:
                # Mirroring is a convenience for a smooth future engage, never a reason to fail
                # an otherwise-normal policy step.
                pass
            info.setdefault("intervention_steps_executed", 0)
            info.setdefault("steps_executed", action.shape[0])
            return obs, rew, done, truncated, info

        human_actions = []
        last_action = None
        for step_idx in range(action.shape[0]):
            if bool(self.expert.human_in_control):
                leader_action = np.asarray(self.expert.get_action(), dtype=action.dtype)
                last_action = leader_action
            else:
                # Toggled off mid-chunk: hold the last human-commanded position for the rest of
                # this chunk rather than jumping back to a now-stale policy action. The next
                # step() call will resolve the toggle edge (disengage) at its own chunk boundary.
                leader_action = last_action if last_action is not None else action[step_idx]
            self.env.execute_action_chunk(leader_action[None, :])
            human_actions.append(leader_action)

        intervention_chunk = np.stack(human_actions, axis=0)

        obs, rew, done, truncated, info = self.env.finish_action_chunk_step()
        rew = np.asarray(rew, dtype=np.float32).reshape(-1)
        if rew.shape[0] != action.shape[0]:
            padded_rew = np.zeros((action.shape[0],), dtype=np.float32)
            padded_rew[: min(rew.shape[0], action.shape[0])] = rew[: action.shape[0]]
            rew = padded_rew
        info["intervene_action"] = intervention_chunk
        info["intervention_steps_executed"] = intervention_chunk.shape[0]
        info["steps_executed"] = action.shape[0]
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self._was_human_in_control = False
        return self.env.reset(**kwargs)

    def close(self):
        if hasattr(self.expert, "close"):
            self.expert.close()
        return self.env.close()
