# Mejoras: ensamble de semillas (A) y M6 Transformer ± campeones (B)

*`python src/mejoras_eval.py`. Iteración posterior a B10: se propuso después de ver que la escalera M1–M5 quedó plana en AUC y que el valor de M4 empata con R0. Mismos datos, splits y protocolo de B5/B10; no sobrescribe sus resultados.*

**Modelos.** M4/M5 = GRU/TCN finales (semilla 42, los pesos de la app). *ens3* = promedio de las 3 semillas, recalibrado con una temperatura en val. M6 = Transformer causal (sin campeones); M6+draft = el mismo + embeddings de los 10 campeones. Configuración de M6 elegida en val (`results/hp/m6_hp.csv`). M4+draft = la GRU de M4 + los mismos embeddings de campeones (`src/models/draft.py`); lr y dropout elegidos por el promedio de val de 3 semillas (`results/hp/m4d_hp.csv`).

## Métricas técnicas (test, todas las partidas)

| Modelo | AUC | Log-loss | ECE | Salto medio/min | ΔAUC vs M4 [IC 95 %] | ΔLog-loss vs M4 [IC 95 %] |
|---|---|---|---|---|---|---|
| M0 | 0.7941 | 0.5432 | 0.0190 | 0.0511 | -0.0183 [-0.0222, -0.0141] | +0.0228 [+0.0178, +0.0279] |
| M2 | 0.8108 | 0.5220 | 0.0185 | 0.0654 | -0.0017 [-0.0029, -0.0005] | +0.0017 [-0.0001, +0.0034] |
| M4 | 0.8124 | 0.5204 | 0.0200 | 0.0604 | — | — |
| M4 ens3 | 0.8125 | 0.5205 | 0.0223 | 0.0613 | +0.0001 [-0.0003, +0.0004] | +0.0001 [-0.0004, +0.0007] |
| M4+draft | 0.8161 | 0.5161 | 0.0211 | 0.0604 | +0.0036 [-0.0002, +0.0080] | -0.0043 [-0.0094, +0.0006] |
| M4+draft ens3 | 0.8193 | 0.5129 | 0.0207 | 0.0614 | +0.0069 [+0.0040, +0.0102] | -0.0075 [-0.0116, -0.0038] |
| M5 | 0.8129 | 0.5198 | 0.0207 | 0.0591 | +0.0004 [-0.0001, +0.0009] | -0.0005 [-0.0012, +0.0002] |
| M5 ens3 | 0.8129 | 0.5201 | 0.0227 | 0.0602 | +0.0004 [-0.0001, +0.0010] | -0.0002 [-0.0010, +0.0005] |
| M6 | 0.8121 | 0.5211 | 0.0224 | 0.0615 | -0.0003 [-0.0010, +0.0004] | +0.0007 [-0.0002, +0.0016] |
| M6 ens3 | 0.8122 | 0.5217 | 0.0269 | 0.0616 | -0.0002 [-0.0009, +0.0004] | +0.0014 [+0.0003, +0.0024] |
| M6+draft | 0.8174 | 0.5148 | 0.0165 | 0.0613 | +0.0049 [+0.0013, +0.0088] | -0.0055 [-0.0104, -0.0009] |
| M6+draft ens3 | 0.8207 | 0.5112 | 0.0201 | 0.0609 | +0.0083 [+0.0047, +0.0121] | -0.0092 [-0.0137, -0.0050] |

AUC por semilla (test): M4 0.8123 ± 0.0002 · M4+draft 0.8163 ± 0.0015 · M5 0.8126 ± 0.0002 · M6 0.8119 ± 0.0003 · M6+draft 0.8184 ± 0.0010

## Valor (test: 28 partidas anotadas; θ* elegido en val: 27)

| Detector | θ* | Clips/partida | Precisión | Recall | F1 | Ahorro (min) | Valor USD/mes [IC 95 %] | Δ vs M4 | Δ vs R0 | Valor en val |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 0.30 | 3.9 | 0.48 | 0.71 | 0.57 | 22.2 | 148 [127, 168] | +16 [-2, +37] | — | 139 |
| M0 | 0.17 | 2.2 | 0.51 | 0.43 | 0.47 | 20.6 | 137 [107, 165] | +5 [-7, +17] | -11 [-31, +8] | 133 |
| M2 | 0.27 | 1.6 | 0.42 | 0.26 | 0.32 | 18.5 | 123 [93, 152] | -9 [-17, -0] | -25 [-44, -6] | 128 |
| M4 | 0.23 | 1.9 | 0.49 | 0.35 | 0.41 | 19.7 | 132 [99, 163] | — | -16 [-37, +2] | 132 |
| M4 ens3 | 0.25 | 1.4 | 0.53 | 0.29 | 0.37 | 19.5 | 130 [98, 160] | -1 [-9, +5] | -18 [-37, +2] | 130 |
| M4+draft | 0.24 | 1.4 | 0.57 | 0.32 | 0.41 | 20.1 | 134 [102, 166] | +2 [-5, +10] | -14 [-35, +7] | 129 |
| M4+draft ens3 | 0.26 | 1.1 | 0.58 | 0.25 | 0.35 | 19.5 | 130 [97, 161] | -2 [-9, +6] | -18 [-37, +1] | 130 |
| M5 | 0.16 | 3.3 | 0.39 | 0.50 | 0.44 | 19.2 | 128 [100, 154] | -4 [-14, +8] | -20 [-37, -3] | 129 |
| M5 ens3 | 0.24 | 1.6 | 0.51 | 0.31 | 0.39 | 19.6 | 131 [99, 161] | -1 [-9, +7] | -17 [-37, +2] | 130 |
| M6 | 0.21 | 2.4 | 0.43 | 0.40 | 0.41 | 19.3 | 129 [100, 158] | -3 [-10, +5] | -19 [-36, -2] | 128 |
| M6 ens3 | 0.16 | 3.5 | 0.39 | 0.53 | 0.45 | 19.4 | 129 [103, 154] | -3 [-14, +10] | -19 [-37, -1] | 130 |
| M6+draft | 0.21 | 2.1 | 0.47 | 0.38 | 0.42 | 19.7 | 131 [100, 161] | -0 [-9, +8] | -17 [-38, +4] | 130 |
| M6+draft ens3 | 0.15 | 3.6 | 0.39 | 0.55 | 0.46 | 19.4 | 129 [102, 156] | -2 [-13, +10] | -18 [-36, -1] | 128 |
| Oráculo | 0.30 | 2.6 | 1.00 | 1.00 | 1.00 | 31.3 | 209 [185, 232] | +77 [+60, +96] | +61 [+46, +76] | 198 |

*Valor: con pocas partidas anotadas los IC son anchos; una diferencia cuyo IC incluye 0 no es significativa.*

## M4+draft vs M6+draft (los dos con campeones, promedio de 3 semillas; pareado, test)

| Tramo | ΔAUC M6+draft − M4+draft [IC 95 %] |
|---|---|
| global | +0.0014 [−0.0007, +0.0034] (empate) |
| 5-9 | +0.0026 [−0.0006, +0.0056] |
| 10-14 | +0.0027 [+0.0002, +0.0049] |
| 15-19 | +0.0027 [+0.0006, +0.0046] |
| 20-24 | +0.0020 [+0.0003, +0.0036] |
| 25-29 | +0.0014 [−0.0004, +0.0031] |
| 30+ | −0.0039 [−0.0078, −0.0006] |

## Conclusiones

1. **Los campeones sí importan.** Con el draft, la GRU (M4+draft) y el Transformer (M6+draft) son los primeros modelos que mejoran de forma clara a M4: **+0.007 y +0.008 de AUC** con el promedio de 3 semillas (IC sin cruzar 0) y menor log-loss. Es entre 4 y 5 veces la diferencia entre la GRU y LightGBM (+0.0017). La mejora está en los minutos 5–29: la composición dice cómo va a escalar cada equipo antes de que se vea en el oro. En 30+ ya no aporta: el estado del juego ya lo dice todo. Gana en las 4 regiones y en los 6 tiers; pierde un poco solo en partidas de menos de 25 min.
2. **La arquitectura no importa.** Sin campeones, M6 = M4 = M5; con campeones, M6+draft ≈ M4+draft (+0.0014, no significativo). Lo que mueve el AUC es la **información nueva** (el draft), no el tipo de red. Coincide con la curva de aprendizaje (E2) y con las búsquedas planas de B6/B7.
3. **Con campeones, las semillas importan más.** Sin draft las 3 semillas aprenden lo mismo (sd del AUC 0.0002) y promediarlas no cambia nada. Con draft varían más (M6+draft sd 0.0010; M4+draft 0.0015: la semilla 44 de la GRU casi no aprovecha los campeones en ninguna de las 4 configuraciones probadas), y el promedio de 3 sí ayuda (+0.003). Por eso los dos modelos con campeones se usan como promedio de 3 entrenamientos. La GRU con campeones también necesitó un lr más alto (1e-3 vs 3e-4 de M4): con 3e-4 el early stopping la corta antes de que aprenda los embeddings.
4. **El valor no cambia.** Todos los modelos de ML quedan en 128–134 USD/mes, dentro del ruido entre sí, y ninguno supera a R0. El draft es igual en todos los minutos: **sube o baja la curva entera, pero no crea ni borra saltos**, y el detector de momentos vive de los saltos. Mejor AUC ≠ más valor, otra vez (mismo hallazgo de B10, sección 4.8).
5. **Para el producto:** una curva de probabilidad más precisa (útil para leer "¿cuándo se nos fue la partida?"), con los mismos momentos y el mismo valor. Los dos están en el selector de la app: `models/m4_draft` (θ* = 0.26) y `models/m6_draft` (θ* = 0.15), curva = promedio de 3 semillas, θ* elegido en val para ese promedio.

**Límites.** Iteración hecha después de B10 (no estaba en el plan principal; M6 era "Podría"). La configuración de M4+draft se eligió en val con campeones (la de M6, sin ellos). Las métricas técnicas usan las ~4,000 partidas de test; el valor, solo 28 partidas anotadas (IC anchos). El split es temporal, así que la mejora de los campeones se mantiene en partidas posteriores a train, pero el meta cambia con los parches: los modelos con draft habría que reentrenarlos con más frecuencia.
