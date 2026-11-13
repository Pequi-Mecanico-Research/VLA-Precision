"""Tira do learner os pesos do PaliGemma que ele nunca usa (opção D, docs/widowx.md, T6/D9).

O learner do Stage 2 só roda o action expert em cima do prefixo já calculado (o cache de contexto do
pré-processamento e do actor): ``velocity_from_context`` chama o Gemma com ``[None, suffix_tokens]``,
e a parte ``None`` (o PaliGemma) não é executada. O ``embed_suffix`` do π0.5 também não toca no
PaliGemma. E o learner publica para o actor só os pesos treináveis (o LoRA).

Então, no learner, a visão (SigLIP), a tabela de tokens e as camadas de linguagem do Gemma 2B (~5,5 GB
em bf16) ficariam parados na GPU. Esta função troca esses pesos por arrays de tamanho zero **ainda na
CPU**, antes de o modelo ir para a GPU. Isso importa: se eles fossem para a GPU e só depois fossem
apagados, o JAX guardaria a memória liberada para si, e o actor (outro processo) não poderia usá-la.

- Nas camadas, que o ``nn.scan`` percorre, o formato vira ``(18, 0, ...)``: a 1ª dimensão (o número de
  camadas) precisa continuar igual, senão o scan quebra.
- No resto, ``(0, ...)``.

Se algum ponto do learner usar um desses pesos, o Flax acusa o formato errado na hora
(``ScopeParamShapeError``), em vez de calcular algo errado em silêncio.

O actor não usa isto: ele precisa do PaliGemma para ler as imagens.
"""

from __future__ import annotations

import re

import numpy as np

# No Gemma do OpenPI, os pesos do PaliGemma (o 1º "expert") não têm sufixo e os do action expert têm
# "_1" (openpi/models/gemma.py, ``_name``). É a mesma convenção do filtro de congelamento.
# "(.*/)?" tolera um nível a mais entre llm/ e layers/ (por exemplo, a coleção "params" do Flax).
_NOT_ACTION_EXPERT = r"(?!.*_1(/|$))"
_LLM_LAYERS = re.compile(rf".*PaliGemma/llm/(.*/)?layers/{_NOT_ACTION_EXPERT}.*")
_LLM_OTHERS = re.compile(rf".*PaliGemma/llm/(?!(.*/)?layers/){_NOT_ACTION_EXPERT}.*")
_VISION = re.compile(r".*PaliGemma/img/.*")


def empty_unused_paligemma(flat_params: dict) -> tuple[dict, int]:
    """Recebe os pesos achatados ({caminho: array}) e devolve os do PaliGemma vazios + bytes poupados."""
    result = {}
    saved = 0
    for key, value in flat_params.items():
        path = "/".join(str(part) for part in key)
        shape = getattr(value, "shape", None)
        if shape and _LLM_LAYERS.fullmatch(path) and len(shape) > 1:
            empty_shape = (shape[0], 0, *shape[2:])
        elif shape and (_LLM_OTHERS.fullmatch(path) or _VISION.fullmatch(path) or _LLM_LAYERS.fullmatch(path)):
            empty_shape = (0, *shape[1:])
        else:
            result[key] = value
            continue
        saved += int(np.prod(shape)) * np.dtype(value.dtype).itemsize
        result[key] = np.zeros(empty_shape, dtype=value.dtype)
    return result, saved
