"""Monta o ambiente do Stage 2 do WidowX de verdade, sem hardware.

Usa a mesma fábrica que o ``robot-agent-bridge`` usa (``build_environment``) com os YAMLs reais do
WidowX. Só troca o hardware: o serviço do WidowX roda com braços simulados (``FakeArms``) e as
RealSense viram câmeras falsas. Assim testa, de ponta a ponta, a pilha de wrappers, o estado 7D,
as imagens esticadas, o espaço de ação e as chaves de intervenção que chegam ao actor, inclusive
pela serialização do Pyro.

    uv run --no-sync python tests/widowx/check_environment_fake.py

Detalhes: docs/widowx.md, teste T2b.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "configs/stage2/tasks/wxai_organize_table.yaml"
DEPLOYMENT = ROOT / "configs/stage2/deployments/single_wxai.yaml"
# Cor da câmera falsa em BGR (como a RealSense entrega). No modelo deve chegar em RGB: (200, 20, 10).
FAKE_BGR = (10, 20, 200)

checks = 0


def check(condition, message: str) -> None:
    global checks
    if not condition:
        raise AssertionError(message)
    checks += 1
    print(f"  ok  {message}")


def load_service():
    path = ROOT / "src/vla_precision/robotics/servers/widowx/service.py"
    spec = importlib.util.spec_from_file_location("widowx_service", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeRealSense:
    """Câmera 640x480 de cor uniforme, no lugar da RealSense."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def capture(self) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = FAKE_BGR
        return frame

    def close(self) -> None:
        return None


def find_wrapper(env, name: str):
    while env is not None:
        if type(env).__name__ == name:
            return env
        env = getattr(env, "env", None)
    raise LookupError(name)


def wrapper_chain(env) -> list[str]:
    names = []
    while env is not None:
        names.append(type(env).__name__)
        env = getattr(env, "env", None)
    return names


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from vla_precision.config.loader import load_config
    from vla_precision.robotics.environments.factory import build_environment
    from vla_precision.robotics.environments.server import RemoteEnvironment
    from vla_precision.serialization import b64_to_numpy

    svc = load_service()
    arms = svc.FakeArms(max_relative_target=0.25)
    service = svc.WidowXService(arms, control_hz=30.0)
    service.start()
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), svc.make_handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"

    # Mesmo deployment, só com o endereço do serviço simulado e sem a janela das câmeras (o teste
    # pode rodar sem tela).
    deployment = (
        DEPLOYMENT.read_text()
        .replace("server_url: http://127.0.0.1:5002/", f"server_url: {url}")
        .replace("display_image: true", "display_image: false")
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(deployment)
    resolved = load_config(TASK, deployment=Path(handle.name))
    cameras = {"realsense": lambda name, device: FakeRealSense(name)}
    H = resolved.config.task.action_horizon  # passos por chunk, lido do YAML (D1)

    def build(**kwargs):
        env = build_environment(
            resolved,
            camera_factories=cameras,
            emergency_stop_factory=None,  # Esc via pynput precisa de tela; não faz parte deste teste
            completion_detection=False,  # idem para o ". ."
            **kwargs,
        )
        if hasattr(env.unwrapped, "robot"):
            env.unwrapped.robot._wait_for_operator = lambda message: None  # sem Enter no teste
        return env

    print("[1] pilha de wrappers")
    env = build()
    chain = wrapper_chain(env)
    print(f"      {' -> '.join(chain)}")
    check("LeaderArmIntervention" in chain, "intervenção pelo leader montada (teleoperation.kind: widowx_leader)")
    check("KeyboardIntervention" not in chain, "intervenção por teclado do UR não montada")
    check("QuaternionToEulerWrapper" not in chain, "conversão de quaternion (tcp_pose) desligada")
    relative = find_wrapper(env, "RelativeFrameWrapper")
    check(not relative.include_relative_pose and not relative.actions_in_tcp_frame, "frame relativo de TCP desligado")
    intervention = find_wrapper(env, "LeaderArmIntervention")
    check(intervention.completion_event is None, "sem completion_event: o sucesso continua no teclado (. .)")

    print("[2] espaços e observação")
    check(env.action_space.shape == (H, 7), f"ação ({H}, 7) = task.action_horizon x 7 juntas (veio {env.action_space.shape})")
    observation, _ = env.reset()
    check(set(observation) == {"state", "base_0_rgb", "left_wrist_0_rgb"}, f"observação: {sorted(observation)}")
    check(observation["state"].shape == (1, 7), f"estado 7D (veio {observation['state'].shape})")
    reset_pose = np.asarray(resolved.config.robot.options["reset_positions"])
    check(np.allclose(observation["state"][0], reset_pose, atol=1e-5), "estado = robot.options.reset_positions, na ordem do dataset")
    for key in ("base_0_rgb", "left_wrist_0_rgb"):
        image = observation[key][0]
        check(image.shape == (224, 224, 3) and image.dtype == np.uint8, f"{key}: 224x224x3 uint8")
        check(np.all(image == (200, 20, 10)), f"{key}: esticada (sem faixas pretas) e em RGB")

    print("[3] padding x esticar (a opção resize faz diferença)")
    base_env = env.unwrapped
    base_env.camera_options["base_0_rgb"] = {"resize": "pad"}
    padded = base_env._capture_images()["base_0_rgb"]
    check(np.all(padded[:28] == 0) and np.all(padded[-28:] == 0), "com resize: pad aparecem 28 linhas pretas em cima e embaixo")
    base_env.camera_options["base_0_rgb"] = {"resize": "stretch"}

    print("[4] chunk autônomo")
    start = observation["state"][0].astype(np.float64)
    chunk = start + np.linspace(0.0, 0.05, H)[:, None] * np.ones(7)
    observation, reward, terminated, truncated, info = env.step(chunk)
    check(reward.shape == (H,), "uma recompensa por passo do chunk")
    check(info["executed_action"].shape == (H, 7), f"executed_action ({H}, 7)")
    check(np.allclose(info["executed_action"], chunk, atol=1e-5), "executou exatamente o chunk")
    check(info["pose_intervention_pressed"] is False and "intervene_action" not in info, "sem intervenção marcada")
    check(np.allclose(observation["state"][0], chunk[-1], atol=1e-5), "estado novo = último passo do chunk")

    print("[5] chunk com intervenção")
    requests.post(url + "intervention", json={"enabled": True})
    time.sleep(0.2)
    check(service.mode == "intervention", "intervenção ligada")
    hand = observation["state"][0].astype(np.float64) + 0.1
    arms.set_leader(hand)
    observation, reward, terminated, truncated, info = env.step(chunk)
    check(info["pose_intervention_pressed"] is True, "pose_intervention_pressed = True")
    check(info["intervene_action"].shape == (H, 7), f"intervene_action ({H}, 7), mesmo formato do chunk da política")
    check(info["intervention_steps_executed"] == H, f"os {H} passos vieram do leader")
    check(np.allclose(info["intervene_action"][-1], hand, atol=1e-5), "o follower foi para a pose do leader")
    requests.post(url + "intervention", json={"enabled": False})

    print("[6] pelo Pyro (o que o actor recebe de verdade)")
    remote = RemoteEnvironment(env)
    observation_b64, _ = remote.env_reset(options={})
    values = remote.env_step(chunk.tolist())
    next_observation, reward, terminated, truncated, info = (b64_to_numpy(value) for value in values)
    check(np.asarray(next_observation["state"]).shape == (1, 7), "estado chega ao actor como (1, 7)")
    check(np.asarray(next_observation["base_0_rgb"]).shape == (1, 224, 224, 3), "imagem chega ao actor como (1, 224, 224, 3)")
    check(np.asarray(info["executed_action"]).shape == (H, 7), "executed_action chega ao actor")
    check("pose_intervention_pressed" in info, "chaves de intervenção chegam ao actor")
    check(np.asarray(b64_to_numpy(remote.env_action_space_sample())).shape == (H, 7), f"env_action_space_sample ({H}, 7)")
    env.close()

    print("[7] avaliação sem intervenção (evaluation.allow_intervention: false)")
    env = build(teleoperation_enabled=False)
    check("LeaderArmIntervention" not in wrapper_chain(env), "sem wrapper de intervenção")
    env.reset()
    _, _, _, _, info = env.step(np.repeat(reset_pose[None], H, axis=0))
    check("intervene_action" not in info, "nenhuma correção registrada")
    env.close()

    print("[8] modo fake_env do próprio VLA-Precision")
    env = build_environment(resolved, fake_env=True)
    observation, _ = env.reset()
    check(observation["state"].shape == (1, 7) and env.action_space.shape == (H, 7), f"fake_env monta com estado 7D e ação ({H}, 7)")
    env.close()

    server.shutdown()
    service.stop()
    Path(handle.name).unlink()
    print(f"\nTUDO CERTO: {checks} verificações")


if __name__ == "__main__":
    main()
