"""
M4 - GRU causal (modelo principal).

Causal = unidireccional: la predicción del minuto t solo depende de los minutos 0..t.
Nunca usar bidirectional=True: vería el futuro de la partida (fuga de información).

Arquitectura:
    X [B, T, F] -> Linear(F, hidden) + GELU + Dropout   (proyección de entrada)
                -> GRU(hidden, layers)                  (memoria de la partida)
                -> Linear(hidden, hidden/2) + GELU + Dropout -> Linear(., 1)
                -> logits [B, T]
"""
import torch
from torch import nn


class GRUModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                                   nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.gru(self.input(x))
        return self.head(h).squeeze(-1)
