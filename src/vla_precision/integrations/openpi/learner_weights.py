"""Tira do learner os pesos do PaliGemma que ele nunca usa (opção D, docs/widowx.md, T6/D9).

O learner do Stage 2 só roda o action expert em cima do prefixo já calculado (o cache de contexto do
pré-processamento e do actor): ``velocity_from_context`` chama o Gemma com ``[None, suffix_tokens]``,
e a parte ``None`` (o PaliGemma) não é executada. O ``embed_suffix`` do π0.5 também não toca no
PaliGemma. E o learner publica para o actor só os pesos treináveis (o LoRA).

Então, no learner, a visão (SigLIP), a tabela de tokens e as camadas de linguagem do Gemma 2B (~5,5 GB
em bf16) ficam parados na GPU. Esta função troca esses pesos por arrays de tamanho zero:

- nas camadas, que o ``nn.scan`` percorre, o formato vira ``(18, 0, ...)``: a 1ª dimensão (o número de
  camadas) precisa continuar igual, senão o scan quebra;
- no resto, ``(0, ...)``.

Se algum ponto do learner usar um desses pesos, o Flax acusa o formato errado na hora
(``ScopeParamShapeError``), em vez de calcular algo errado em silêncio.

O actor não usa esta função: ele precisa do PaliGemma para ler as imagens.
"""

from __future__ import annotations

import dataclasses
import logging

import jax
import jax.numpy as jnp
from openpi.shared import nnx_utils

LOGGER = logging.getLogger(__name__)

# No Gemma do OpenPI, os pesos do PaliGemma (o 1º "expert") não têm sufixo e os do action expert têm
# "_1" (openpi/models/gemma.py, ``_name``). É a mesma convenção do filtro de congelamento.
_NOT_ACTION_EXPERT = r"(?!.*_1(/|$))"
# "(.*/)?" tolera um nível a mais entre llm/ e layers/ (por exemplo, a coleção "params" do Flax).
_LLM_LAYERS = nnx_utils.PathRegex(rf".*PaliGemma/llm/(.*/)?layers/{_NOT_ACTION_EXPERT}.*")
_LLM_OTHERS = nnx_utils.PathRegex(rf".*PaliGemma/llm/(?!(.*/)?layers/){_NOT_ACTION_EXPERT}.*")
_VISION = nnx_utils.PathRegex(r".*PaliGemma/img/.*")


def _empty(keep_first_axis: bool):
    def fn(variable):
        value = variable.value
        if not hasattr(value, "shape") or len(value.shape) == 0:
            return variable
        shape = (value.shape[0], 0, *value.shape[2:]) if keep_first_axis and len(value.shape) > 1 else (0, *value.shape[1:])
        return variable.replace(jnp.zeros(shape, dtype=value.dtype))

    return fn


def _nbytes(state) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(state))


def _drop(params):
    params = nnx_utils.state_map(params, _LLM_LAYERS, _empty(keep_first_axis=True))
    params = nnx_utils.state_map(params, _LLM_OTHERS, _empty(keep_first_axis=False))
    return nnx_utils.state_map(params, _VISION, _empty(keep_first_axis=False))


def drop_unused_paligemma_weights(agent):
    """Devolve o agente com o PaliGemma vazio no ator e na referência (que compartilham esses pesos)."""
    pi_state = agent.state.pi_state
    before = _nbytes(pi_state.params)
    params = _drop(pi_state.params)
    reference = _drop(agent.state.ref_pi_params)
    after = _nbytes(params)
    if before - after < 1e9:
        LOGGER.warning("learner: quase nada do PaliGemma foi encontrado para remover; confira os nomes dos pesos")
    LOGGER.info(
        "learner: pesos do PaliGemma não usados removidos da GPU (%.2f GB -> %.2f GB)",
        before / 1e9,
        after / 1e9,
    )
    pi_state = dataclasses.replace(pi_state, params=params)
    return agent.replace(state=agent.state.replace(pi_state=pi_state, ref_pi_params=reference))
