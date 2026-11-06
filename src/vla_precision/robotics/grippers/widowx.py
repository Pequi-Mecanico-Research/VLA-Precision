"""Garra do WidowX AI.

No WidowX a garra é a 7ª junta do braço (``left_carriage_joint``): ela entra no vetor de ação 7D e
o serviço do braço a comanda junto com as outras juntas. Esta classe existe só para cumprir o
``Gripper`` Protocol (grippers/base.py): ela não manda nada sozinha.
"""

from __future__ import annotations

import numpy as np


class WidowXGripper:
    kind = "widowx"

    @property
    def action_dimension(self) -> int:
        return 1  # a última dimensão da ação 7D

    def command_chunk(self, positions: np.ndarray) -> None:
        del positions  # comandada pelo serviço do braço, junto com as juntas

    def observations(self, robot_state: dict) -> dict[str, np.ndarray]:
        del robot_state  # a posição da garra já está em left_carriage_joint
        return {}

    def close(self) -> None:
        return None
