"""Teste do serviço do WidowX com os braços reais (sem política, sem ACoB).

Antes, em outro terminal, no PC ligado aos braços:

    uv run --project third_party/lerobot_trossen python src/vla_precision/robotics/servers/widowx/service.py

Depois, neste terminal (qualquer ambiente com numpy e requests):

    uv run --no-sync python tests/widowx/check_service_hardware.py

Cada etapa espera o seu Enter. Os movimentos são pequenos (±0,05 rad) e lentos. Para parar tudo,
Ctrl+C no terminal do serviço: ele leva os braços para o repouso.

Detalhes: docs/widowx.md, teste T3.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import requests


def post(url: str, endpoint: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    response = requests.post(url + endpoint, json=payload or {}, timeout=(2.0, timeout))
    if response.status_code >= 400:
        raise SystemExit(f"Serviço recusou {endpoint}: HTTP {response.status_code} {response.text}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:5002/")
    parser.add_argument("--amplitude", type=float, default=0.05, help="amplitude do movimento de teste (rad)")
    parser.add_argument("--hold-seconds", type=float, default=15.0, help="duração da janela de intervenção")
    args = parser.parse_args()
    url = args.url.rstrip("/") + "/"

    config = post(url, "config")
    hz = float(config["control_hz"])
    print(f"Serviço: {config['hardware']}, {hz:.0f} Hz, juntas {config['joint_names']}")
    state = post(url, "state")
    start = np.asarray(state["positions"])
    print(f"Pose atual do follower: {np.round(start, 3).tolist()}")

    input(f"\n[1/3] Enter para mover a junta 0 em ±{args.amplitude} rad durante 3 s (o leader deve acompanhar)...")
    t = np.arange(int(3 * hz)) / hz
    chunk = np.repeat(start[None], len(t), axis=0)
    chunk[:, 0] += args.amplitude * np.sin(2 * np.pi * t / 3.0)
    report = post(url, "execute_chunk", {"actions": chunk.tolist()}, timeout=len(t) / hz + 10)
    executed = np.asarray(report["executed"])
    after = np.asarray(post(url, "state")["positions"])
    print(f"  passos executados: {len(executed)}; maior diferença comando×enviado: {np.max(np.abs(executed - chunk)):.4f} rad")
    print(f"  posição final: {np.round(after, 3).tolist()} (deveria estar perto da inicial)")
    print("  Confira a olho: o follower fez o vaivém suave? O leader acompanhou?")

    input(
        f"\n[2/3] Intervenção: o script vai mandar {args.hold_seconds:.0f} s de 'ficar parado'. Durante esse tempo:\n"
        "      1. aperte Espaço (o log do serviço mostra INTERVENÇÃO LIGADA);\n"
        "      2. mova o leader com a mão: o follower deve copiar;\n"
        "      3. aperte Espaço de novo: o follower deve parar onde está.\n"
        "      Enter para começar..."
    )
    here = np.asarray(post(url, "state")["positions"])
    hold = np.repeat(here[None], int(args.hold_seconds * hz), axis=0)
    report = post(url, "execute_chunk", {"actions": hold.tolist()}, timeout=args.hold_seconds + 10)
    executed = np.asarray(report["executed"])
    moved = float(np.max(np.abs(executed - here)))
    print(f"  passos vindos do leader: {report['intervened_steps']} de {len(executed)}")
    print(f"  maior deslocamento do follower: {moved:.3f} rad")
    if report["intervention_active"]:
        print("  ATENÇÃO: a intervenção ainda está ligada. Aperte Espaço para desligar.")
    if report["intervened_steps"] == 0:
        print("  ATENÇÃO: nenhuma intervenção registrada. O Espaço chegou ao serviço? (ver o log dele)")

    if input("\n[3/3] Levar os dois braços para a pose staged? [s/N] ").strip().lower() == "s":
        post(url, "reset", {"duration": 3.0}, timeout=20)
        print(f"  pose: {np.round(post(url, 'state')['positions'], 3).tolist()}")
    print("\nFim. Se tudo se comportou como descrito, o lado do robô está pronto para o Stage 2.")


if __name__ == "__main__":
    main()
