#!/bin/bash
# Roda na máquina com GPU (a 4090): carrega o checkpoint e serve a política por WebSocket.
# Mesmo papel do openpi/server_inference.sh, mas pelo VLA-Precision.
#
# Pré-requisitos nesta máquina, antes de rodar:
#   uv sync --frozen --group stage1
#   checkpoint presente localmente — editar model.checkpoint_dir em
#   src/vla_precision/integrations/openpi/inference/configs/widowx.yaml
#   (hoje é um placeholder, "CHANGE_ME_DENSE_CHECKPOINT_PATH" — nenhuma fonte que já olhamos
#   tinha o caminho real; o próprio server_inference.sh do openpi também tem isso como
#   placeholder não resolvido, "path_to_police_localy")
set -e

cd "$(dirname "$0")"

echo "Iniciando servidor de política VLA-Precision (pi05)."

uv run --no-sync main.py \
  --stage stage1 --mode serve-openpi-policy \
  --config src/vla_precision/integrations/openpi/inference/configs/widowx.yaml
