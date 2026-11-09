"""Serviço do braço WidowX AI para o Stage 2 do VLA-Precision.

Roda no PC ligado ao braço, no ambiente Python 3.12 do ``lerobot_trossen`` (third_party/), e **não**
importa nada do VLA-Precision (que exige Python 3.11). O VLA-Precision fala com ele por HTTP, pelo
``WidowXRobot`` (robotics/robots/widowx.py). É o mesmo padrão do UR (robotics/servers/ur/).

    uv run --project third_party/lerobot_trossen \\
        python src/vla_precision/robotics/servers/widowx/service.py

    # sem braços, simulando os dois (para testar o resto do sistema):
    python src/vla_precision/robotics/servers/widowx/service.py --fake --no-keyboard

O que ele faz:

- Abre o **follower** (executa as ações) e o **leader** (você move com a mão), usando as classes
  do ``lerobot_trossen`` que já funcionam na coleta de dados.
- Roda um laço de controle próprio, a ``--control-hz`` (30 Hz), com dois modos:

  | Modo        | Follower                          | Leader                                   |
  |-------------|-----------------------------------|------------------------------------------|
  | autônomo    | executa o chunk da política       | modo posição, **copia o follower**       |
  | intervenção | **copia o leader**                | gravidade compensada, você move com a mão |

- A tecla **Espaço** (lida via evdev, funciona sem foco na janela) liga e desliga a intervenção.
  Ao apertar, a política é interrompida na hora e o follower para; o leader, que já vinha copiando
  o follower, termina de alcançá-lo (``--sync-tolerance``, ~0,1-0,3 s) e só então é solto. Assim o
  follower não dá salto. O ``max_relative_target`` do follower limita cada passo (segunda proteção).
- Se a intervenção acabar no meio de um chunk, o resto do chunk **não** é executado: o follower
  fica parado na pose atual até a política mandar o próximo chunk, calculado a partir daí.

Endpoints (POST, corpo e resposta em JSON):

- ``/config``: nomes das juntas, frequência, modo.
- ``/state``: posições, velocidades e esforços do follower; se a intervenção está ligada.
- ``/execute_chunk`` ``{"actions": [[7 floats], ...]}``: executa o chunk (bloqueia até o fim) e
  devolve ``executed`` (o que foi de fato mandado ao follower, passo a passo) e ``intervened_steps``.
- ``/reset`` ``{"positions": [7 floats], "duration": s}``: desliga a intervenção e leva os dois
  braços para a pose dada.
- ``/intervention`` ``{"enabled": bool}``: liga/desliga a intervenção sem o teclado (para testes).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

LOGGER = logging.getLogger("widowx_service")

AUTONOMOUS, SYNCING, INTERVENTION = "autonomous", "syncing", "intervention"

JOINT_NAMES = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "left_carriage_joint")
NUM_JOINTS = len(JOINT_NAMES)
# Pose "staged" do lerobot_trossen: rad para as juntas, m para a garra.
STAGED_POSITIONS = (0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0)


# =============================================================================================
# Hardware
# =============================================================================================
class TrossenArms:
    """Os dois braços reais, pelas classes do lerobot_trossen. Só o laço de controle chama isto."""

    def __init__(
        self,
        *,
        follower_ip: str,
        leader_ip: str,
        control_hz: float,
        min_time_to_move_multiplier: float,
        max_relative_target: float | None,
    ):
        import trossen_arm
        from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
        from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower
        from lerobot_teleoperator_trossen.config_widowxai_leader import WidowXAILeaderTeleopConfig
        from lerobot_teleoperator_trossen.widowxai_leader import WidowXAILeaderTeleop

        self._mode = trossen_arm.Mode
        self.min_time_to_move = min_time_to_move_multiplier / control_hz
        self.follower = WidowXAIFollower(
            WidowXAIFollowerConfig(
                ip_address=follower_ip,
                cameras={},  # as câmeras ficam com o VLA-Precision
                loop_rate=int(control_hz),
                min_time_to_move_multiplier=min_time_to_move_multiplier,
                max_relative_target=max_relative_target,
                include_velocity=True,
                include_effort=True,
            )
        )
        self.leader = WidowXAILeaderTeleop(WidowXAILeaderTeleopConfig(ip_address=leader_ip))
        LOGGER.info("Conectando follower (%s) e leader (%s); os dois vão para a pose staged", follower_ip, leader_ip)
        self.follower.connect()
        self.leader.connect()  # termina com o leader em gravidade compensada
        self.leader.driver.set_all_modes(self._mode.position)
        self.leader.driver.set_all_positions(list(self.follower_positions()), goal_time=2.0, blocking=True)

    def follower_state(self) -> dict[str, list[float]]:
        joints = self.follower.driver.get_robot_output().joint.all
        return {
            "positions": list(joints.positions),
            "velocities": list(joints.velocities),
            "efforts": list(joints.efforts),
        }

    def follower_positions(self) -> np.ndarray:
        return np.asarray(self.follower.driver.get_all_positions(), dtype=np.float64)

    def send_follower(self, target: np.ndarray) -> np.ndarray:
        sent = self.follower.send_action({f"{name}.pos": float(v) for name, v in zip(JOINT_NAMES, target)})
        return np.asarray([sent[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float64)

    def leader_positions(self) -> np.ndarray:
        return np.asarray(self.leader.driver.get_all_positions(), dtype=np.float64)

    def leader_follow(self, target: np.ndarray) -> None:
        self.leader.driver.set_all_positions(list(map(float, target)), goal_time=self.min_time_to_move, blocking=False)

    def leader_release(self) -> None:
        """Gravidade compensada: igual ao final de ``go_to_staged_position`` do leader."""
        self.leader.driver.set_all_modes(self._mode.external_effort)
        self.leader.driver.set_all_external_efforts([0.0] * NUM_JOINTS, goal_time=0.0, blocking=True)

    def leader_hold(self) -> None:
        current = list(self.leader_positions())
        self.leader.driver.set_all_modes(self._mode.position)
        self.leader.driver.set_all_positions(current, goal_time=self.min_time_to_move, blocking=False)

    def move_both(self, goal: np.ndarray, duration: float) -> None:
        goal = list(map(float, goal))
        self.follower.driver.set_all_positions(goal, goal_time=duration, blocking=False)
        self.leader.driver.set_all_positions(goal, goal_time=duration, blocking=True)
        time.sleep(0.2)

    def close(self) -> None:
        for name, device in (("leader", self.leader), ("follower", self.follower)):
            try:
                device.disconnect()  # pose staged e depois repouso
            except Exception:
                LOGGER.exception("Falha ao desconectar o %s", name)


class FakeArms:
    """Simula os dois braços (``--fake``), para testar o resto do sistema sem hardware.

    O follower vai instantaneamente para o alvo, limitado por ``max_relative_target``. O leader
    copia o follower em modo posição; em intervenção, fica onde ``set_leader`` mandar (simulando a mão).
    """

    def __init__(self, *, max_relative_target: float | None = None):
        self.max_relative_target = max_relative_target
        self._follower = np.asarray(STAGED_POSITIONS, dtype=np.float64)
        self._leader = self._follower.copy()
        self.leader_free = False

    def set_leader(self, positions) -> None:
        self._leader = np.asarray(positions, dtype=np.float64)

    def follower_state(self) -> dict[str, list[float]]:
        return {"positions": list(self._follower), "velocities": [0.0] * NUM_JOINTS, "efforts": [0.0] * NUM_JOINTS}

    def follower_positions(self) -> np.ndarray:
        return self._follower.copy()

    def send_follower(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        if self.max_relative_target is not None:
            target = self._follower + np.clip(target - self._follower, -self.max_relative_target, self.max_relative_target)
        self._follower = target.copy()
        return target

    def leader_positions(self) -> np.ndarray:
        return self._leader.copy()

    def leader_follow(self, target: np.ndarray) -> None:
        if not self.leader_free:
            self._leader = np.asarray(target, dtype=np.float64).copy()

    def leader_release(self) -> None:
        self.leader_free = True

    def leader_hold(self) -> None:
        self.leader_free = False

    def move_both(self, goal: np.ndarray, duration: float) -> None:
        self._follower = np.asarray(goal, dtype=np.float64).copy()
        self._leader = self._follower.copy()

    def close(self) -> None:
        return None


# =============================================================================================
# Laço de controle
# =============================================================================================
class _ChunkJob:
    def __init__(self, actions: np.ndarray):
        self.actions = actions
        self.index = 0
        self.executed: list[list[float]] = []
        self.intervened_steps = 0
        self.hold_rest = False  # a intervenção acabou no meio: o resto do chunk fica parado
        self.done = threading.Event()

    def record(self, sent: np.ndarray, *, intervened: bool) -> None:
        self.executed.append([float(v) for v in sent])
        self.intervened_steps += int(intervened)
        self.index += 1
        if self.index >= len(self.actions):
            self.done.set()


class WidowXService:
    """Dono dos braços: todo acesso ao hardware acontece na thread do laço de controle."""

    def __init__(
        self,
        arms,
        *,
        control_hz: float = 30.0,
        sync_tolerance: float = 0.05,
        sync_timeout: float = 2.0,
    ):
        self.arms = arms
        self.control_hz = float(control_hz)
        self.period = 1.0 / self.control_hz
        self.sync_tolerance = float(sync_tolerance)
        self.sync_timeout = float(sync_timeout)
        # "autonomous" -> (Espaço) -> "syncing" -> "intervention" -> (Espaço) -> "autonomous"
        self.mode = AUTONOMOUS
        self._sync_target: np.ndarray | None = None
        self._sync_deadline = 0.0
        self.fault: str | None = None
        self._lock = threading.Lock()
        self._tick_done = threading.Condition(self._lock)  # avisa quando há uma leitura nova
        self._tick_count = 0
        self._chunk: _ChunkJob | None = None
        self._toggle_requested = False
        self._commands: list[tuple[Callable[[], object], dict]] = []
        self._latest = arms.follower_state()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="widowx-control", daemon=True)

    @property
    def intervention(self) -> bool:
        """True desde o Espaço (inclui a sincronização): a política já não comanda."""
        return self.mode != AUTONOMOUS

    # ---------------------------------------------------------------- ciclo de vida
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    # ---------------------------------------------------------------- chamado por outras threads
    def request_toggle(self) -> None:
        """Tecla Espaço: a troca de modo acontece no próximo passo do laço."""
        with self._lock:
            self._toggle_requested = True

    def state(self) -> dict:
        """Estado do follower lido no próximo ciclo do laço (depois do último comando enviado).

        Sem essa espera, o estado lido logo depois de um chunk seria o do começo do último ciclo,
        antes do último passo: a política decidiria o próximo chunk com um estado velho.
        """
        with self._tick_done:
            start = self._tick_count
            self._tick_done.wait_for(lambda: self._tick_count > start, timeout=2 * self.period + 0.5)
            return self._snapshot()

    def _snapshot(self) -> dict:
        return {**self._latest, "intervention": self.intervention, "mode": self.mode, "fault": self.fault}

    def config(self) -> dict:
        return {
            "joint_names": list(JOINT_NAMES),
            "action_dimension": NUM_JOINTS,
            "control_hz": self.control_hz,
            "mode": self.mode,
            "hardware": type(self.arms).__name__,
        }

    def execute_chunk(self, actions) -> dict:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != NUM_JOINTS or len(actions) == 0:
            raise ValueError(f"actions precisa ter formato (T, {NUM_JOINTS}), veio {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions tem NaN ou infinito")
        job = _ChunkJob(actions)
        with self._lock:
            if self._chunk is not None:
                raise RuntimeError("já existe um chunk em execução")
            self._chunk = job
        if not job.done.wait(timeout=len(actions) * self.period + 10.0):
            with self._lock:
                self._chunk = None
            raise TimeoutError("o chunk não terminou no tempo esperado (o laço de controle travou?)")
        with self._lock:
            self._chunk = None
            intervention_now = self.intervention
        return {
            "executed": job.executed,
            "intervened_steps": job.intervened_steps,
            "intervention_active": intervention_now,
        }

    def reset(self, positions, duration: float) -> dict:
        goal = np.asarray(positions if positions is not None else STAGED_POSITIONS, dtype=np.float64)
        if goal.shape != (NUM_JOINTS,) or not np.all(np.isfinite(goal)):
            raise ValueError(f"positions precisa ter {NUM_JOINTS} valores finitos")

        def run():
            if self._chunk is not None:
                raise RuntimeError("não dá para resetar com um chunk em execução")
            if self.intervention:
                self._end_intervention()
            LOGGER.info("Reset: levando os braços para %s em %.1f s", np.round(goal, 3).tolist(), duration)
            self.arms.move_both(goal, float(duration))
            self._latest = self.arms.follower_state()
            return self._snapshot()

        return self._run_in_loop(run, timeout=float(duration) + 15.0)

    def set_intervention(self, enabled: bool) -> dict:
        def run():
            if enabled and not self.intervention:
                self._begin_sync()
            elif not enabled and self.intervention:
                self._end_intervention()
            return {"intervention": self.intervention, "mode": self.mode}

        return self._run_in_loop(run, timeout=5.0)

    def _run_in_loop(self, fn: Callable[[], object], *, timeout: float):
        slot: dict = {"done": threading.Event()}
        with self._lock:
            self._commands.append((fn, slot))
        if not slot["done"].wait(timeout=timeout):
            raise TimeoutError("o laço de controle não respondeu")
        if "error" in slot:
            raise slot["error"]
        return slot["result"]

    # ---------------------------------------------------------------- thread do laço
    def _loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                commands, self._commands = self._commands, []
            for fn, slot in commands:
                try:
                    slot["result"] = fn()
                except Exception as error:
                    slot["error"] = error
                finally:
                    slot["done"].set()
            try:
                self._tick()
            except Exception as error:
                self.fault = f"{type(error).__name__}: {error}"
                LOGGER.exception("Erro no laço de controle")
            next_tick += self.period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # atrasou: não tenta compensar

    def _tick(self) -> None:
        state = self.arms.follower_state()
        with self._lock:
            self._latest = state
            self._tick_count += 1
            self._tick_done.notify_all()
            toggle, self._toggle_requested = self._toggle_requested, False
            chunk = self._chunk
        active_chunk = chunk if chunk is not None and not chunk.done.is_set() else None
        if toggle:
            if self.mode == AUTONOMOUS:
                self._begin_sync()
            else:
                self._end_intervention()

        if self.mode == SYNCING:
            # O follower fica parado; o leader (ainda em modo posição) termina de alcançá-lo.
            sent = self.arms.send_follower(self._sync_target)
            if active_chunk is not None:
                active_chunk.record(sent, intervened=True)  # a política já foi interrompida
            self.arms.leader_follow(self._sync_target)
            gap = float(np.max(np.abs(self.arms.leader_positions() - self._sync_target)))
            if gap <= self.sync_tolerance:
                self.arms.leader_release()
                self.mode = INTERVENTION
                LOGGER.warning(">>> INTERVENÇÃO LIGADA: mova o leader; o follower copia. Espaço desliga.")
            elif time.monotonic() > self._sync_deadline:
                LOGGER.error(
                    "Intervenção CANCELADA: o leader não alcançou o follower em %.1f s (distância %.3f rad). "
                    "O follower fica parado; aperte Espaço de novo para tentar.",
                    self.sync_timeout,
                    gap,
                )
                self._end_intervention()
            return

        if self.mode == INTERVENTION:
            sent = self.arms.send_follower(self.arms.leader_positions())
            if active_chunk is not None:
                active_chunk.record(sent, intervened=True)
            return

        positions = np.asarray(state["positions"], dtype=np.float64)
        if active_chunk is not None:
            if active_chunk.hold_rest:
                active_chunk.record(positions, intervened=False)  # parado na pose atual
            else:
                active_chunk.record(self.arms.send_follower(active_chunk.actions[active_chunk.index]), intervened=False)
        self.arms.leader_follow(positions)

    def _begin_sync(self) -> None:
        self._sync_target = self.arms.follower_positions()
        self._sync_deadline = time.monotonic() + self.sync_timeout
        self.mode = SYNCING
        LOGGER.warning("Espaço: política interrompida; sincronizando o leader com o follower...")

    def _end_intervention(self) -> None:
        if self.mode == INTERVENTION:
            self.arms.leader_hold()
            LOGGER.warning("<<< INTERVENÇÃO DESLIGADA: a política volta a comandar no próximo chunk.")
        self.mode = AUTONOMOUS
        with self._lock:
            if self._chunk is not None:
                self._chunk.hold_rest = True


# =============================================================================================
# Teclado (Espaço) e HTTP
# =============================================================================================
def start_space_listener(callback: Callable[[], None], *, debounce_s: float = 0.3) -> int:
    """Escuta a tecla Espaço em todos os teclados via evdev. Devolve quantos teclados achou."""
    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        LOGGER.error("evdev não instalado: use /intervention por HTTP ou rode no ambiente do lerobot_trossen")
        return 0

    devices = []
    for path in evdev.list_devices():
        try:
            device = evdev.InputDevice(path)
        except OSError:
            continue
        if ecodes.KEY_SPACE in device.capabilities().get(ecodes.EV_KEY, []):
            devices.append(device)
    if not devices:
        LOGGER.error(
            "Nenhum teclado acessível via evdev. Seu usuário precisa ler /dev/input/event* "
            "(sudo usermod -aG input $USER e entrar de novo)."
        )
        return 0

    last_press = [0.0]

    def listen(device) -> None:
        try:
            for event in device.read_loop():
                if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_SPACE and event.value == 1:
                    now = time.monotonic()
                    if now - last_press[0] >= debounce_s:
                        last_press[0] = now
                        callback()
        except OSError:
            LOGGER.warning("Teclado %s desconectado", device.path)

    for device in devices:
        threading.Thread(target=listen, args=(device,), name=f"space-{device.path}", daemon=True).start()
    LOGGER.info("Espaço liga/desliga a intervenção (teclados: %s)", ", ".join(d.name for d in devices))
    return len(devices)


def make_handler(service: WidowXService):
    routes: dict[str, Callable[[dict], dict]] = {
        "/config": lambda body: service.config(),
        "/state": lambda body: service.state(),
        "/execute_chunk": lambda body: service.execute_chunk(body["actions"]),
        "/reset": lambda body: service.reset(body.get("positions"), float(body.get("duration", 3.0))),
        "/intervention": lambda body: service.set_intervention(bool(body["enabled"])),
    }

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            route = routes.get(self.path.rstrip("/") or "/")
            if route is None:
                self._reply(404, {"error": f"rota desconhecida {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                self._reply(200, route(body))
            except (ValueError, KeyError, TypeError) as error:
                self._reply(400, {"error": f"{type(error).__name__}: {error}"})
            except RuntimeError as error:
                self._reply(409, {"error": str(error)})
            except Exception as error:
                LOGGER.exception("Erro em %s", self.path)
                self._reply(500, {"error": f"{type(error).__name__}: {error}"})

        do_GET = do_POST

        def log_message(self, format, *args) -> None:  # noqa: A002 - assinatura da stdlib
            LOGGER.debug("%s " + format, self.address_string(), *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--follower-ip", default="192.168.1.2")
    parser.add_argument("--leader-ip", default="192.168.1.23")
    parser.add_argument("--host", default="127.0.0.1", help="onde o HTTP escuta (o VLA-Precision roda na mesma máquina)")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--min-time-to-move-multiplier",
        type=float,
        default=4.0,
        help="suavização de cada comando: goal_time = valor / control-hz (4.0 = cliente do Stage 1; a coleta usou 3.0)",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=0.25,
        help="maior passo por junta a cada ciclo, em rad (proteção contra saltos); 0 desliga",
    )
    parser.add_argument(
        "--sync-tolerance",
        type=float,
        default=0.05,
        help="ao apertar Espaço, o leader só é solto quando estiver a menos disto do follower (rad)",
    )
    parser.add_argument("--sync-timeout", type=float, default=2.0, help="tempo máximo da sincronização (s)")
    parser.add_argument("--no-keyboard", action="store_true", help="não escuta o Espaço (use /intervention)")
    parser.add_argument("--fake", action="store_true", help="simula os dois braços, sem hardware")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    max_relative = args.max_relative_target if args.max_relative_target > 0 else None
    if args.fake:
        arms = FakeArms(max_relative_target=max_relative)
        LOGGER.warning("Modo --fake: nenhum braço real é usado")
    else:
        arms = TrossenArms(
            follower_ip=args.follower_ip,
            leader_ip=args.leader_ip,
            control_hz=args.control_hz,
            min_time_to_move_multiplier=args.min_time_to_move_multiplier,
            max_relative_target=max_relative,
        )
    service = WidowXService(
        arms,
        control_hz=args.control_hz,
        sync_tolerance=args.sync_tolerance,
        sync_timeout=args.sync_timeout,
    )
    service.start()
    if not args.no_keyboard:
        start_space_listener(service.request_toggle)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    LOGGER.info("Serviço do WidowX em http://%s:%d (Ctrl+C encerra e leva os braços para o repouso)", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        LOGGER.info("Encerrando...")
        server.server_close()
        service.stop()
        arms.close()


if __name__ == "__main__":
    main()
