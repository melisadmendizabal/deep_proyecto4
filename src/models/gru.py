"""
M4 - GRU causal (modelo principal).

Causal = unidireccional: la predicción del minuto t solo depende de los minutos 0..t.
Nunca usar bidirectional=True: vería el futuro de la partida (fuga de información).

Arquitectura:
    X [B, T, F] -> Linear(F, hidden) + GELU + Dropout   (proyección de entrada)
                + draft (opcional, champions=True: src/models/draft.py), igual en todos los minutos
                -> GRU(hidden, layers)                  (memoria de la partida)
                -> Linear(hidden, hidden/2) + GELU + Dropout -> Linear(., 1)
                -> logits [B, T]

Con champions=False (el default) el modelo y sus pesos son exactamente los de siempre.
"""
import torch
from torch import nn

from .draft import DraftEncoder


class GRUModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 1,
                 dropout: float = 0.1, champions: bool = False, champ_dim: int = 16,
                 champ_dropout: float = 0.1):
        super().__init__()
        self.uses_champions = bool(champions)
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                                   nn.Dropout(dropout))
        if self.uses_champions:
            self.draft = DraftEncoder(hidden, champ_dim, champ_dropout, dropout)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                champions: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input(x)
        if self.uses_champions:
            if champions is None:
                raise ValueError("Este modelo usa campeones: pasa champions=[B, 10]")
            h = h + self.draft(champions)[:, None, :]
        h, _ = self.gru(h)
        return self.head(h).squeeze(-1)
