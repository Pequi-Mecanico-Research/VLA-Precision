#!/bin/bash
# Stage 2 - PC do braço (rodar primeiro): serviço dos dois braços + robot-agent-bridge.
# Teclas neste terminal: Enter = começar, . . = sucesso, Esc = fracasso. Espaço = intervenção.
cd "$(dirname "$0")/../.."

echo "Iniciando o serviço dos braços."
uv run --project third_party/lerobot_trossen python src/vla_precision/robotics/servers/widowx/service.py \
  --follower-ip 192.168.1.2 --leader-ip 192.168.1.23 &
trap "kill -TERM $!" EXIT   # ao sair, encerra o serviço (ele leva os braços ao repouso)
sleep 5

echo "Iniciando o robot-agent-bridge."
uv run --no-sync main.py --stage stage2 --mode robot-agent-bridge \
  --config configs/stage2/tasks/wxai_organize_table.yaml \
  --deployment configs/stage2/deployments/single_wxai.yaml
