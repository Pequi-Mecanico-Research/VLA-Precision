"""Teste do lado do robô do Stage 2 sem hardware: serviço do WidowX com braços simulados.

Testa, de ponta a ponta e por HTTP de verdade: o serviço (robotics/servers/widowx/service.py), o
cliente ``WidowXRobot`` (robotics/robots/widowx.py) e o wrapper ``LeaderArmIntervention``
(robotics/wrappers/leader_intervention.py). Precisa só de numpy, requests e gymnasium:

    uv run --no-sync python tests/widowx/check_robot_side_fake.py

Detalhes: docs/widowx.md, teste T2.
"""

import importlib.util
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


svc = load("service", "src/vla_precision/robotics/servers/widowx/service.py")
robot_mod = load("widowx_robot", "src/vla_precision/robotics/robots/widowx.py")
gripper_mod = load("widowx_gripper", "src/vla_precision/robotics/grippers/widowx.py")
wrapper_mod = load("leader_intervention", "src/vla_precision/robotics/wrappers/leader_intervention.py")

ok_count = 0


def check(cond, msg):
    global ok_count
    if not cond:
        raise AssertionError(msg)
    ok_count += 1
    print(f"  ok  {msg}")


HZ = 30.0
arms = svc.FakeArms(max_relative_target=0.25)
service = svc.WidowXService(arms, control_hz=HZ, sync_tolerance=0.05, sync_timeout=2.0)
service.start()
server = ThreadingHTTPServer(("127.0.0.1", 0), svc.make_handler(service))
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/"

config = SimpleNamespace(server_url=url, control_hz=HZ, options={"reset_duration": 0.1, "wait_for_operator": True})
operator_prompts = []
robot = robot_mod.WidowXRobot(config, gripper=gripper_mod.WidowXGripper(), wait_for_operator=operator_prompts.append)

def pos():
    robot.refresh_state()  # o ambiente real faz isso via observations() depois de cada chunk
    return robot.currpos


print("[1] conexão e observação")
obs = robot.observations()
check(tuple(obs) == robot_mod.JOINT_STATE_KEYS, "observação tem as 7 chaves de junta, na ordem do dataset")
check(all(v.shape == (1,) for v in obs.values()), "cada chave tem 1 valor (o Flatten junta em 7D)")
check(np.allclose(pos(), svc.STAGED_POSITIONS), "começa na pose staged")

print("[2] chunk autônomo pequeno")
start = pos()
chunk = start + np.linspace(0, 0.1, 20)[:, None] * np.ones(7)
t = time.monotonic()
executed = robot.execute_action_chunk(chunk)
elapsed = time.monotonic() - t
check(executed.shape == (20, 7), "executed tem o formato do chunk (20, 7)")
check(np.allclose(executed, chunk, atol=1e-6), "passos pequenos são executados exatamente")
check(robot.last_chunk_report["intervened_steps"] == 0, "sem intervenção: intervened_steps = 0")
check(0.55 < elapsed < 1.2, f"20 passos a 30 Hz levam ~0,67 s (levou {elapsed:.2f} s)")
time.sleep(2 / HZ)  # o leader copia a posição medida do follower, um ciclo depois
check(np.allclose(arms.leader_positions(), arms.follower_positions()), "leader copiou o follower no modo autônomo")

print("[3] salto grande é limitado por passo")
here = pos()
executed = robot.execute_action_chunk(np.repeat((here + 1.0)[None], 3, axis=0))
steps = np.diff(np.vstack([here, executed]), axis=0)
check(np.all(np.abs(steps) <= 0.25 + 1e-5), "nenhum passo maior que max_relative_target (0,25 rad)")

print("[4] intervenção: follower copia o leader")
check(requests.post(url + "intervention", json={"enabled": True}).json()["mode"] == "syncing", "Espaço: primeiro sincroniza")
time.sleep(3 / HZ)
check(service.mode == "intervention", "depois de sincronizar, a intervenção liga")
check(arms.leader_free, "leader ficou solto (gravidade compensada)")
hand = pos() + 0.05
arms.set_leader(hand)  # "a mão" move o leader
policy = np.repeat((pos() - 0.5)[None], 10, axis=0)  # a política queria ir para outro lado
executed = robot.execute_action_chunk(policy)
check(robot.last_chunk_report["intervened_steps"] == 10, "os 10 passos vieram do leader")
check(np.allclose(executed[-1], hand), "o follower foi para a pose do leader, não para a da política")

print("[5] wrapper traduz para as chaves do actor do ACoB")
import gymnasium as gym


class FakeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,))
    action_space = gym.spaces.Box(-7, 7, (10, 7))

    def __init__(self, robot):
        self.robot = robot

    def step(self, action):
        executed = self.robot.execute_action_chunk(action)
        return np.zeros(1), np.zeros(len(action)), False, False, {"executed_action": executed}

    def reset(self, **kwargs):
        return np.zeros(1), {}


env = wrapper_mod.LeaderArmIntervention(FakeEnv(robot))
_, _, _, _, info = env.step(np.repeat(pos()[None], 10, axis=0))
check(info["pose_intervention_pressed"] is True, "pose_intervention_pressed = True")
check(info["intervene_action"].shape == (10, 7), "intervene_action tem o mesmo formato do chunk da política")
check(info["intervention_steps_executed"] == 10, "intervention_steps_executed = 10")
check(env.completion_event is None, "sem completion_event: sucesso continua no teclado")

print("[6] desligar no meio do chunk: o resto fica parado")
result = {}
policy = pos() + np.linspace(0, 0.2, 30)[:, None] * np.ones(7)
worker = threading.Thread(target=lambda: result.update(e=robot.execute_action_chunk(policy)))
worker.start()
time.sleep(10 / HZ)
service.request_toggle()  # = apertar Espaço
worker.join()
k = robot.last_chunk_report["intervened_steps"]
rest = result["e"][k:]
check(0 < k < 30, f"parte do chunk foi corrigida ({k} passos)")
check(np.allclose(rest, rest[0]), "depois de desligar, o follower ficou parado (não seguiu o resto da política)")
check(not service.intervention and not arms.leader_free, "intervenção desligada e leader de volta ao modo posição")

print("[7] Espaço no meio de um chunk autônomo: a política para na hora")
result = {}
start = pos()
policy = start + np.linspace(0, 0.3, 30)[:, None] * np.ones(7)
worker = threading.Thread(target=lambda: result.update(e=robot.execute_action_chunk(policy)))
worker.start()
time.sleep(8 / HZ)
service.request_toggle()  # = apertar Espaço
worker.join()
executed = result["e"]
k = 30 - robot.last_chunk_report["intervened_steps"]
check(5 <= k <= 12, f"a política executou só os {k} primeiros passos")
check(np.allclose(executed[k:], executed[k], atol=1e-5), "o resto do chunk ficou parado na pose da interrupção (leader não foi movido)")
check(service.mode == "intervention", "intervenção ativa depois da sincronização")
requests.post(url + "intervention", json={"enabled": False})

print("[8] proteção: sincronização que não converge é cancelada")
service.sync_timeout = 0.3
arms.leader_free = True  # simula um leader que não obedece (travado longe)
arms.set_leader(pos() + 0.5)
requests.post(url + "intervention", json={"enabled": True})
time.sleep(0.6)
check(service.mode == "autonomous", "cancelou depois do timeout e voltou ao autônomo")
arms.leader_free = False
service.sync_timeout = 2.0

print("[9] erros de entrada")
r = requests.post(url + "execute_chunk", json={"actions": [[0.0] * 6]})
check(r.status_code == 400, "chunk com 6 juntas é recusado (HTTP 400)")
import json
r = requests.post(url + "execute_chunk", data=json.dumps({"actions": [[float("nan")] * 7]}))
check(r.status_code == 400, "serviço recusa chunk com NaN (HTTP 400)")
try:
    robot.execute_action_chunk(np.full((3, 7), np.nan))
    check(False, "cliente deveria recusar NaN")
except ValueError as error:
    check("NaN" in str(error), "cliente recusa NaN antes de enviar, com mensagem clara")

print("[10] reset")
obs = robot.reset()
check(np.allclose(pos(), robot_mod.DEFAULT_RESET_POSITIONS), "reset leva para a pose de reset")
check(len(operator_prompts) == 1, "reset espera o operador (Enter)")

print("[11] frequência diferente é recusada")
try:
    robot_mod.WidowXRobot(SimpleNamespace(server_url=url, control_hz=10.0, options={}), gripper=gripper_mod.WidowXGripper())
    check(False, "deveria recusar")
except RuntimeError as error:
    check("control_hz" in str(error), "control_hz diferente do serviço dá erro claro")

server.shutdown()
service.stop()
print(f"\nTUDO CERTO: {ok_count} verificações")
