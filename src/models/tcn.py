"""
M5 - TCN causal: convoluciones 1D dilatadas (B7).

Causal = cada convolución solo mira hacia atrás: se rellena con ceros a la IZQUIERDA
((k-1)·d posiciones) y nunca a la derecha, así que el minuto t solo depende de 0..t.
El padding del final de la secuencia (partidas más cortas que el batch) no afecta a los
minutos válidos, porque está a la derecha.

Arquitectura (estilo Bai et al., 2018):
    X [B, T, F] -> Linear(F, hidden) + GELU + Dropout
                -> `layers` bloques residuales; el bloque i usa dilatación 2^i:
                     [conv causal -> GELU -> Dropout] x 2  + conexión residual
                -> Linear(hidden, hidden/2) + GELU + Dropout -> Linear(., 1) -> logits [B, T]

Campo receptivo = 1 + 2·(k-1)·(2^layers - 1) minutos:
    k=3: layers=3 -> 29 min, layers=4 -> 61 min (toda la partida, T = 60).
A diferencia de la GRU, ve una ventana fija de historia y se entrena en paralelo en el tiempo.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__(channels, channels, kernel_size, dilation=dilation)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [B, C, T]
        return super().forward(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, kernel_size, dilation), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, kernel_size, dilation), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 4,
                 dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[TemporalBlock(hidden, kernel_size, 2 ** i, dropout)
                                      for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** layers - 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input(x).transpose(1, 2)                          # [B, hidden, T]
        h = self.blocks(h).transpose(1, 2)                         # [B, T, hidden]
        return self.head(h).squeeze(-1)
