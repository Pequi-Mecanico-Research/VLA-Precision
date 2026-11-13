"""Robô WidowX AI no VLA-Precision: cliente HTTP do serviço do braço.

O hardware fica no serviço (robotics/servers/widowx/service.py), que roda no ambiente Python 3.12
do ``lerobot_trossen``. Esta classe implementa o ``Robot`` Protocol (robots/base.py) chamando o
serviço, igual ao ``URRobot`` faz com o servidor do UR.

Convenção (diferente de UR/Franka): ação e estado em **posição absoluta de junta**, 7D
(``joint_0`` .. ``joint_5`` em rad + ``left_carriage_joint`` da garra em m), na ordem do dataset.
O estado sai como uma chave por junta, cada uma com 1 valor; o YAML da task lista as 7 em
``task.proprio_keys`` e o ``FlattenObservationWrapper`` junta tudo no vetor 7D.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vla_precision.config.schema import RobotConfig
    from vla_precision.robotics.grippers.base import Gripper

LOGGER = logging.getLogger(__name__)

JOINT_STATE_KEYS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "left_carriage_joint")
# Pose "staged" do lerobot_trossen (rad; garra em m).
DEFAULT_RESET_POSITIONS = (0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0)
HTTP_CONNECT_TIMEOUT_S = 2.0


class WidowXRobot:
    # Limites do espaço de ação do Gym. Posições de junta em rad: não podem ser cortadas em ±3
    # como as ações delta do UR.
    action_low = -2 * np.pi
    action_high = 2 * np.pi

    def __init__(
        self,
        config: RobotConfig,
        *,
        gripper: Gripper,
        post: Callable[..., Any] | None = None,
        wait_for_operator: Callable[[str], Any] = input,
    ):
        if post is None:
            import requests

            post = requests.post
        self._post_fn = post
        self._wait_for_operator = wait_for_operator
        self.url = config.server_url.rstrip("/") + "/"
        self.gripper = gripper
        self.control_hz = float(config.control_hz)
        options = dict(config.options)
        self.reset_positions = np.asarray(options.get("reset_positions", DEFAULT_RESET_POSITIONS), dtype=np.float64)
        if self.reset_positions.shape != (len(JOINT_STATE_KEYS),):
            raise ValueError(f"robot.options.reset_positions precisa de {len(JOINT_STATE_KEYS)} valores")
        self.reset_duration = float(options.get("reset_duration", 3.0))
        self.wait_at_reset = bool(options.get("wait_for_operator", True))
        self.last_chunk_report: dict[str, Any] = {"executed": [], "intervened_steps": 0, "intervention_active": False}

        self.server_config = self._post("config")
        names = tuple(self.server_config.get("joint_names", ()))
        if names != JOINT_STATE_KEYS:
            raise RuntimeError(f"Serviço do WidowX tem juntas {names}, esperado {JOINT_STATE_KEYS}")
        server_hz = float(self.server_config.get("control_hz", 0.0))
        if abs(server_hz - self.control_hz) > 1e-6:
            raise RuntimeError(
                f"robot.control_hz={self.control_hz} não bate com o serviço ({server_hz} Hz). "
                "O dataset foi gravado a 30 Hz; use o mesmo valor nos dois."
            )
        LOGGER.info("Conectado ao serviço do WidowX em %s (%s)", self.url, self.server_config.get("hardware"))
        self._state: dict[str, Any] = {}
        self.refresh_state()

    # ---------------------------------------------------------------- Robot Protocol
    @property
    def action_dimension(self) -> int:
        return len(JOINT_STATE_KEYS)

    @property
    def currpos(self) -> np.ndarray:
        return np.asarray(self._state["positions"], dtype=np.float64)

    def refresh_state(self) -> None:
        self._state = self._post("state")
        if self._state.get("fault"):
            LOGGER.error("Serviço do WidowX reportou erro no laço de controle: %s", self._state["fault"])

    def observations(self) -> dict[str, np.ndarray]:
        self.refresh_state()
        state = {
            key: np.asarray([value], dtype=np.float32)
            for key, value in zip(JOINT_STATE_KEYS, self._state["positions"], strict=True)
        }
        state.update(self.gripper.observations(self._state))
        return state

    def execute_action_chunk(self, action: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None]
        if chunk.ndim != 2 or chunk.shape[-1] != self.action_dimension:
            raise ValueError(f"Ação do WidowX precisa ter formato (T, {self.action_dimension}), veio {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("A política gerou uma ação com NaN ou infinito; o chunk não foi enviado ao braço")
        report = self._post(
            "execute_chunk",
            {"actions": chunk.tolist()},
            read_timeout=len(chunk) / self.control_hz + 15.0,
        )
        self.last_chunk_report = report
        # O que foi de fato mandado ao follower (com limite de passo, ou as posições do leader numa
        # intervenção). É isso que vai para o Replay.
        return np.asarray(report["executed"], dtype=np.float32)

    def reset(
        self,
        *,
        joint_reset: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        del joint_reset, options  # o WidowX sempre volta para a mesma pose de junta
        self._post(
            "reset",
            {"positions": self.reset_positions.tolist(), "duration": self.reset_duration},
            read_timeout=self.reset_duration + 20.0,
        )
        self.last_chunk_report = {"executed": [], "intervened_steps": 0, "intervention_active": False}
        if self.wait_at_reset:
            self._wait_for_operator("\n[WidowX] Arrume a cena e aperte Enter para começar o episódio... ")
        return self.observations()

    def request(self, name: str, enabled: bool) -> Any:
        if name == "force_pause":
            # O actor pede isto no fim de cada episódio (acob_stream/actor.py), enquanto empacota o
            # episódio para o learner. Vem do UR, que fica em modo de força e precisa ser congelado.
            # No WidowX não há nada a fazer: sem chunk em execução, o braço já fica parado na pose.
            return {"force_pause": bool(enabled)}
        return self._post(name, {name: bool(enabled)})

    def close(self) -> None:
        # Os braços pertencem ao serviço; ele os leva para o repouso quando é encerrado.
        self.gripper.close()

    # ---------------------------------------------------------------- HTTP
    def _post(self, endpoint: str, payload: dict | None = None, *, read_timeout: float = 5.0) -> dict:
        try:
            response = self._post_fn(
                self.url + endpoint,
                json=payload or {},
                timeout=(HTTP_CONNECT_TIMEOUT_S, read_timeout),
            )
        except Exception as error:
            raise RuntimeError(
                f"Sem resposta do serviço do WidowX em {self.url}{endpoint} ({type(error).__name__}: {error}). "
                "Ele está rodando? Ver README_WIDOWX.md."
            ) from None
        if response.status_code >= 400:
            raise RuntimeError(f"Serviço do WidowX recusou {endpoint} (HTTP {response.status_code}): {response.text}")
        return response.json()
