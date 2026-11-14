"""
M6 - Transformer causal (+ embeddings de campeones opcionales).

Causal: máscara triangular en la atención, así el minuto t solo atiende a los minutos 0..t
(igual que la GRU, sin ver el futuro). El relleno va al FINAL de cada partida, así que con la
máscara causal nunca afecta a los minutos válidos (no hace falta key_padding_mask).

Arquitectura:
    X [B, T, F] -> Linear(F, hidden) + GELU + Dropout        (proyección de entrada)
                + posición aprendida (minuto 0..max_len-1)
                + draft (opcional, ver abajo), el mismo vector en todos los minutos
                -> TransformerEncoder pre-LN (layers × [atención causal + FFN])
                -> LayerNorm -> Linear(hidden, hidden/2) + GELU + Dropout -> Linear(., 1)
                -> logits [B, T]

Draft (champions=True): IDs de los 10 campeones (participantes 1-5 = azul, 6-10 = rojo,
sequences.npz -> "champions"). Embedding por campeón, promedio por equipo (no importa el
orden dentro del equipo) y Linear([azul, rojo, azul − rojo]) -> hidden. En entrenamiento cada
campeón se "olvida" (id 0 = desconocido) con prob. champ_dropout, para que el modelo no dependa
demasiado del draft y funcione si falta un campeón nuevo.
"""
import torch
from torch import nn

N_CHAMPION_IDS = 1024   # los championId de Riot llegan hasta ~950; 0 = desconocido


class TransformerModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1, champions: bool = False, champ_dim: int = 16,
                 champ_dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.uses_champions = bool(champions)
        self.champ_dropout = champ_dropout
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Embedding(max_len, hidden)
        if self.uses_champions:
            self.champ = nn.Embedding(N_CHAMPION_IDS, champ_dim, padding_idx=0)
            self.draft = nn.Sequential(nn.Linear(3 * champ_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def draft_vector(self, champions: torch.Tensor) -> torch.Tensor:
        """champions [B, 10] (championId) -> [B, hidden]."""
        ids = champions.long().clamp(0, N_CHAMPION_IDS - 1)
        if self.training and self.champ_dropout > 0:
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < self.champ_dropout, 0)
        e = self.champ(ids)                                   # [B, 10, d]
        blue, red = e[:, :5].mean(1), e[:, 5:].mean(1)
        return self.draft(torch.cat([blue, red, blue - red], dim=-1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                champions: torch.Tensor | None = None) -> torch.Tensor:
        t = x.shape[1]
        h = self.input(x) + self.pos(torch.arange(t, device=x.device))[None]
        if self.uses_champions:
            if champions is None:
                raise ValueError("Este modelo usa campeones: pasa champions=[B, 10]")
            h = h + self.draft_vector(champions)[:, None, :]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.encoder(h, mask=causal)
        return self.head(self.norm(h)).squeeze(-1)
