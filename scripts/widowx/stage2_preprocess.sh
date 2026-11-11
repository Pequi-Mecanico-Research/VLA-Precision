#!/bin/bash
# Stage 2 - PC da GPU, uma vez por dataset: prepara os buffers a partir das demonstrações.
cd "$(dirname "$0")/../.."

uv run --no-sync main.py --stage stage2 --mode preprocess \
  --config configs/stage2/tasks/wxai_organize_table.yaml \
  --deployment configs/stage2/deployments/single_wxai.yaml
