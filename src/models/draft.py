"""
Codificador del draft (los 10 campeones), para agregarle campeones a cualquier modelo secuencial.

IDs de los 10 campeones (participantes 1-5 = azul, 6-10 = rojo; sequences.npz -> "champions").
Embedding por campeón, promedio por equipo (no importa el orden dentro del equipo) y
Linear([azul, rojo, azul − rojo]) -> hidden. En entrenamiento cada campeón se "olvida"
(id 0 = desconocido) con prob. champ_dropout, para que el modelo no dependa demasiado del draft
y funcione si aparece un campeón nuevo.

El vector resultante se SUMA a la entrada proyectada de cada minuto: es contexto fijo de la
partida, no cambia con el tiempo.
"""
import torch
from torch import nn

N_CHAMPION_IDS = 1024   # los championId de Riot llegan hasta ~950; 0 = desconocido


class DraftEncoder(nn.Module):
    def __init__(self, hidden: int, champ_dim: int = 16, champ_dropout: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.champ_dropout = champ_dropout
        self.champ = nn.Embedding(N_CHAMPION_IDS, champ_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(3 * champ_dim, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, champions: torch.Tensor) -> torch.Tensor:
        """champions [B, 10] (championId) -> [B, hidden]."""
        ids = champions.long().clamp(0, N_CHAMPION_IDS - 1)
        if self.training and self.champ_dropout > 0:
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < self.champ_dropout, 0)
        e = self.champ(ids)                                   # [B, 10, d]
        blue, red = e[:, :5].mean(1), e[:, 5:].mean(1)
        return self.proj(torch.cat([blue, red, blue - red], dim=-1))
