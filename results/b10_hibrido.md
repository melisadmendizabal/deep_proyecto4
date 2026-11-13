# Iteración: detector híbrido GRU + R0

*`python src/hybrid_eval.py`. Mismo protocolo que B10 (`src/value_model.py`): θ* elegido en val (27 partidas), valor en test (28 partidas), parámetros base de `configs/value.yaml`, IC 95 % por bootstrap pareado sobre partidas. GRU: `models/_train/gru_final__s42`.*

**M4+R0** = momentos de la GRU (θ) + eventos de la regla R0 que no estén a menos de 2 min de uno ya elegido; máximo 5 por partida. Se propuso **después** de ver B10: es una iteración, no el plan original.

| Detector | θ* | Clips/partida | Precisión | Recall | F1 | Ahorro (min) | Valor USD/mes [IC 95 %] | Δ vs M0 | Δ vs M4 | Δ vs R0 | Valor en val |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | — | 3.9 | 0.48 | 0.71 | 0.57 | 22.2 | 148 [126, 169] | +11 [-8, +30] | +16 [-3, +36] | — | 139 |
| M0 | 0.17 | 2.2 | 0.51 | 0.43 | 0.47 | 20.6 | 137 [107, 168] | — | +5 [-7, +18] | -11 [-30, +8] | 133 |
| M4 | 0.23 | 1.9 | 0.49 | 0.35 | 0.41 | 19.7 | 132 [98, 163] | -5 [-18, +7] | — | -16 [-36, +3] | 132 |
| **M4+R0** | 0.34 | 3.8 | 0.49 | 0.72 | 0.58 | 22.5 | 150 [128, 170] | +13 [-5, +33] | +19 [-0, +38] | +2 [+0, +7] | 141 |
| Oráculo | — | 2.6 | 1.00 | 1.00 | 1.00 | 31.3 | 209 [185, 232] | +72 [+54, +90] | +77 [+60, +96] | +61 [+47, +76] | 198 |

**Grilla de θ:** 0.04–0.60 (B10 usa hasta 0.30). Con θ alto, la GRU aporta pocos clips y el híbrido tiende a R0: con θ* = 0.34, de los 200 clips del híbrido en las 55 partidas, **25 vienen de la GRU (12 %)** y 175 de R0.

| θ | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.50 | R0 solo |
|---|---|---|---|---|---|---|---|
| Valor en val | 127 | 139 | 140 | 139 | 139 | 139 | 139 |
| Valor en test | 145 | 151 | 153 | 148 | 148 | 148 | 148 |

**Sensibilidad** (39 escenarios: rango de cada parámetro, ±20 %, ±50 %, pesimista y optimista; θ* re-elegido en cada uno). Quién gana: M4+R0 35, M4 4.

