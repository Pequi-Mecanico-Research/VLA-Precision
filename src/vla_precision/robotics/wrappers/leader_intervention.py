"""Intervenção humana pelo braço-líder do WidowX (tecla Espaço no serviço do braço).

O serviço do WidowX já faz a troca de modo e a teleoperação (o follower copia o leader). Este
wrapper só traduz o que aconteceu no chunk para as chaves que o actor do ACoB lê no ``info``
(acob_stream/actor.py):

- ``intervene_action``: o chunk que foi de fato executado (T, 7). Existe só se houve correção;
  é ele que vira a "ação boa" no buffer de correções.
- ``pose_intervention_pressed``: True quando o humano corrigiu neste chunk.
- ``intervention_steps_executed``: quantos passos do chunk vieram do leader.

O chunk corrigido tem sempre o mesmo tamanho do chunk da política, como o actor exige.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class LeaderArmIntervention(gym.Wrapper):
    def __init__(self, env: gym.Env, *, completion_event=None):
        super().__init__(env)
        # Sem evento, o detector de conclusão usa o teclado (ponto duas vezes = sucesso).
        self.completion_event = completion_event

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        report = getattr(self.unwrapped.robot, "last_chunk_report", None) or {}
        steps = int(report.get("intervened_steps", 0))
        info["pose_intervention_pressed"] = steps > 0
        info["gripper_intervention_pressed"] = False
        info["stop_intervention_pressed"] = False
        if steps > 0:
            executed = info.get("executed_action")
            if executed is None:
                executed = report["executed"]
            info["intervene_action"] = np.asarray(executed, dtype=np.float32).copy()
            info["intervention_steps_executed"] = steps
        return observation, reward, terminated, truncated, info
