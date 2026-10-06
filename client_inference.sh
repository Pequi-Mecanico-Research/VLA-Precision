#!/bin/bash
# Roda nesta máquina (PC cliente): conecta no braço WidowX AI + 3 câmeras via trossen_arm,
# e no servidor de política remoto (veja server_inference.sh, roda na máquina com GPU).
# Mesmo papel do openpi/client_inference.sh, mas pelo VLA-Precision.
set -e

cd "$(dirname "$0")"

echo "Iniciando cliente VLA-Precision (braço + câmeras) para inferência do pi05."

uv run --no-sync main.py \
  --stage stage1 --mode openpi-inference \
  --config src/vla_precision/integrations/openpi/inference/configs/widowx.yaml
