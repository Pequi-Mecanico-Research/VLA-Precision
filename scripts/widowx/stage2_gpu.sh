#!/bin/bash
# Stage 2 - PC da GPU (rodar depois do stage2_robot.sh): learner + actor na mesma GPU.
# As frações dividem a memória da GPU entre os dois (sem isso, o JAX reserva 75% para cada um).
cd "$(dirname "$0")/../.."
CFG="--config configs/stage2/tasks/wxai_organize_table.yaml --deployment configs/stage2/deployments/single_wxai.yaml"

echo "Iniciando o learner."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 uv run --no-sync main.py --stage stage2 --mode train --role learner $CFG &
trap "kill -TERM $!" EXIT   # ao sair, encerra o learner
sleep 60                    # tempo para o learner carregar o modelo

echo "Iniciando o actor."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 uv run --no-sync main.py --stage stage2 --mode train --role actor $CFG
