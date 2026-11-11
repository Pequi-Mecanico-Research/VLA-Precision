"""Câmera simulada, para ensaiar o Stage 2 sem hardware (``driver: fake`` no deployment).

Entrega um frame BGR do tamanho configurado, com um gradiente simples que muda devagar a cada
captura. Não serve para treinar nada de útil: serve para rodar o ambiente, o actor e o learner de
verdade e medir memória e tempo sem as RealSense. Ver docs/widowx.md, teste T6b.
"""

from __future__ import annotations

import numpy as np


class FakeDeviceCamera:
    def __init__(self, name: str, *, width: int = 640, height: int = 480):
        self._name = name
        self._shape = (int(height), int(width), 3)
        self._frame = 0

    @property
    def name(self) -> str:
        return self._name

    def capture(self) -> np.ndarray:
        self._frame += 1
        rows = np.linspace(0, 255, self._shape[0], dtype=np.float32)[:, None, None]
        image = np.broadcast_to((rows + self._frame) % 256, self._shape).astype(np.uint8)
        return np.ascontiguousarray(image)

    def close(self) -> None:
        return None
