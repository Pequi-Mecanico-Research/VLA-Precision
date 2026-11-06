"""WidowX AI (duas câmeras) para o adaptador ACoB <-> OpenPI.

As transformações de verdade estão no fork do OpenPI (third_party/openpi,
``openpi/policies/widowx_policy.py``) e são usadas sem alteração.

Este módulo só existe por um detalhe do VLA-Precision: para descobrir o formato da entrada, o
``OpenPIObservationAdapter`` (integrations/openpi/adapter.py) procura, no módulo da transformação de
entrada, uma função ``make_<nome>_example()``. Para ``WidowXBaseInputs`` ele procuraria
``make_widow_x_base_example``, que não existe no fork. A subclasse abaixo tem o nome ``WidowxInputs``,
então o adaptador procura ``make_widowx_example`` aqui.
"""

import dataclasses

import numpy as np
from openpi.policies import widowx_policy


@dataclasses.dataclass(frozen=True)
class WidowxInputs(widowx_policy.WidowXBaseInputs):
    """Exatamente a ``WidowXBaseInputs`` do fork: ``env`` -> imagem base, ``wrist`` -> pulso esquerdo,
    pulso direito vazio e mascarado. Nenhum comportamento é alterado."""


def make_widowx_example() -> dict:
    """Entrada de exemplo no formato que ``WidowxInputs`` espera (duas câmeras, estado 7D)."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "organize the table",
    }
