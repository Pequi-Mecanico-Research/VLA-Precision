"""T6b — ensaio completo do Stage 2 sem robô, numa máquina só, medindo a memória da GPU.

Sobe os processos de verdade do Stage 2 e troca só o hardware por simulação:

- o serviço do WidowX em ``--fake`` (braços simulados);
- o ``robot-agent-bridge`` com câmeras simuladas (``configs/stage2/deployments/single_wxai_dryrun.yaml``);
- o learner e o actor reais, na mesma GPU.

Enquanto rodam, o script liga e desliga a intervenção sozinho (pelo HTTP do serviço) para gerar as
correções que o learner exige antes de começar a treinar (~32 chunks corrigidos), e mede a memória
de GPU de cada processo com ``nvidia-smi``. No fim, mostra o pico de cada um e se o learner chegou
a treinar.

Pré-requisitos: o pré-processamento (T6) já rodou com o mesmo YAML de task; o checkpoint está no
caminho do YAML; a máquina tem tela (o robot-agent-bridge usa o teclado via pynput).

    uv run --no-sync python tests/widowx/dry_run_stage2.py --minutes 15

Os logs de cada processo ficam em ./results/dry_run/<data-hora>/. Detalhes: docs/widowx.md, T6b.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
TASK = "configs/stage2/tasks/wxai_organize_table.yaml"
DEPLOYMENT = "configs/stage2/deployments/single_wxai_dryrun.yaml"
SERVICE_URL = "http://127.0.0.1:5002/"


def gpu_memory_by_pid() -> dict[int, int]:
    """MiB de GPU usados por processo, segundo o nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    usage = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            usage[int(parts[0])] = int(parts[1])
    return usage


def gpu_total_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=15.0, help="quanto tempo o ensaio roda depois de tudo subir")
    parser.add_argument("--intervention-every", type=float, default=10.0, help="segundos entre intervenções simuladas")
    parser.add_argument("--intervention-seconds", type=float, default=2.0, help="duração de cada intervenção simulada")
    parser.add_argument(
        "--learner-fraction",
        default=None,
        help="XLA_PYTHON_CLIENT_MEM_FRACTION do learner (padrão: sem pré-reserva, para medir o uso real)",
    )
    parser.add_argument("--actor-fraction", default=None, help="idem para o actor")
    args = parser.parse_args()

    os.chdir(ROOT)
    run_dir = ROOT / "results" / "dry_run" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    common = ["--config", TASK, "--deployment", DEPLOYMENT]
    python = sys.executable

    def env_for(fraction: str | None) -> dict:
        env = dict(os.environ, WANDB_MODE="disabled")
        if fraction is None:
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # mede o uso real, não a reserva
        else:
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)
        return env

    processes: dict[str, subprocess.Popen] = {}

    def start(name: str, cmd: list[str], env: dict | None = None) -> None:
        log = (run_dir / f"{name}.log").open("w")
        processes[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env or os.environ)
        print(f"  {name:8s} pid {processes[name].pid}  log: {run_dir / (name + '.log')}")

    def alive(name: str) -> bool:
        return processes[name].poll() is None

    def stop_all() -> None:
        for name in ("actor", "learner", "bridge", "service"):
            proc = processes.get(name)
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        deadline = time.time() + 30
        for proc in processes.values():
            try:
                proc.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()

    print("Subindo os processos...")
    start("service", [python, "src/vla_precision/robotics/servers/widowx/service.py", "--fake", "--no-keyboard"])
    time.sleep(3)
    start("bridge", [python, "main.py", "--stage", "stage2", "--mode", "robot-agent-bridge", *common])
    start("learner", [python, "main.py", "--stage", "stage2", "--mode", "train", "--role", "learner", *common],
          env_for(args.learner_fraction))
    time.sleep(20)  # o learner publica os parâmetros iniciais antes de o actor conectar
    start("actor", [python, "main.py", "--stage", "stage2", "--mode", "train", "--role", "actor", *common],
          env_for(args.actor_fraction))

    peaks = {name: 0 for name in processes}
    corrections_sent = 0
    end = time.time() + args.minutes * 60
    next_intervention = time.time() + 60  # dá tempo de o actor carregar o modelo
    try:
        while time.time() < end:
            dead = [name for name in processes if not alive(name)]
            if dead:
                print(f"\nProcesso encerrou antes da hora: {', '.join(dead)}. Veja o log em {run_dir}.")
                break
            usage = gpu_memory_by_pid()
            for name, proc in processes.items():
                peaks[name] = max(peaks[name], usage.get(proc.pid, 0))
            if time.time() >= next_intervention:
                try:
                    requests.post(SERVICE_URL + "intervention", json={"enabled": True}, timeout=5)
                    time.sleep(args.intervention_seconds)
                    requests.post(SERVICE_URL + "intervention", json={"enabled": False}, timeout=5)
                    corrections_sent += 1
                except requests.RequestException as error:
                    print(f"  (intervenção simulada falhou: {error})")
                next_intervention = time.time() + args.intervention_every
            remaining = int(end - time.time())
            print(
                f"\r  restam {remaining:4d} s | GPU (MiB) actor {peaks.get('actor', 0):6d} learner {peaks.get('learner', 0):6d} "
                f"| intervenções simuladas {corrections_sent:3d}",
                end="",
                flush=True,
            )
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nInterrompido.")
    finally:
        print("\nEncerrando os processos...")
        stop_all()

    learner_log = (run_dir / "learner.log").read_text(errors="replace")
    trained = "Critic warmup" in learner_log or "Learner updates" in learner_log
    total = gpu_total_mib()
    print("\n===================== Resumo do ensaio =====================")
    print(f"  Pico de memória de GPU: actor {peaks.get('actor', 0)} MiB | learner {peaks.get('learner', 0)} MiB "
          f"| soma {peaks.get('actor', 0) + peaks.get('learner', 0)} MiB" + (f" | GPU {total} MiB" if total else ""))
    print(f"  Intervenções simuladas: {corrections_sent} (cada uma gera ~{max(1, int(args.intervention_seconds / 0.5))} chunks corrigidos)")
    print(f"  O learner chegou a treinar? {'sim' if trained else 'NÃO (veja Buffer status / Correction N/32 no learner.log)'}")
    print(f"  Logs: {run_dir}")
    print("  Me mande este resumo e, se algo falhou, o fim do log do processo que caiu.")


if __name__ == "__main__":
    main()
