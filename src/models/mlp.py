"""
M3 - MLP por minuto (deep learning SIN secuencia).

Procesa cada minuto de forma independiente: solo ve el estado actual (+ las features de
tendencia, si no se excluyen). Usa las mismas features que la GRU, así que la diferencia
M3 vs M4 aísla el efecto de ver la historia completa.
"""
import torch
from torch import nn


class MLPModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        blocks, width = [], n_features
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x).squeeze(-1)
