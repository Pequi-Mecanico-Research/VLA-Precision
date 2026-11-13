#!/bin/bash
# Stage 2 - PC da GPU (rodar depois do stage2_robot.sh): learner + actor na mesma GPU.
# As frações dividem a memória da GPU entre os dois (sem isso, o JAX reserva 75% para cada um).
cd "$(dirname "$0")/../.."
CFG="--config configs/stage2/tasks/wxai_organize_table.yaml --deployment configs/stage2/deployments/single_wxai.yaml"

echo "Iniciando o learner."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.50 uv run --no-sync main.py --stage stage2 --mode train --role learner $CFG \
  > >(tee learner.log) 2>&1 &
trap "kill -TERM $!" EXIT   # ao sair, encerra o learner

# O actor só sobe quando o learner publica os pesos iniciais (antes disso, a GPU ainda está cheia).
until grep -q "Initial Actor parameters published" learner.log 2>/dev/null; do sleep 5; done

echo "Iniciando o actor."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run --no-sync main.py --stage stage2 --mode train --role actor $CFG
