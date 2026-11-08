# B7 · TCN, búsqueda del MLP (N1) y curva de aprendizaje (E2)

*7 nov 2026. Se reproduce con:*
```
python src/hp_search.py --config configs/mlp_base.yaml --grid configs/grid_mlp.yaml --prefix mlp_hp   # 3 min
python src/hp_search.py --config configs/tcn_base.yaml --grid configs/grid_tcn.yaml --prefix tcn_hp   # 5 min
python src/learning_curve.py --models m2                                                              # 1 min
python src/evaluate.py --run results/runs/m2_lgbm results/runs/mlp_hp__h64_l2_d0.3_lr0.001 results/runs/tcn_hp__h64_l3_d0.3_lr0.001 --name M2 M3_hp M5_hp
```

**Tiempos reales (RTX 4060):** una corrida de MLP tarda 5–20 s y una de TCN 10–40 s. Las búsquedas de 16 configuraciones duran **3–5 min**, no horas, así que no hace falta esperar a la noche para correrlas.

## M5 · TCN causal (`src/models/tcn.py`)

- Proyección lineal → `layers` bloques residuales con convoluciones causales dilatadas (dilatación 1, 2, 4, 8; kernel 3) → cabeza igual a la de la GRU.
- **Causal:** solo hay relleno a la izquierda. Los tests (`tests/test_b7.py`) verifican que cambiar el futuro no cambia el pasado, que el campo receptivo es exacto (`1 + 4·(2^layers − 1)`: 29 min con 3 bloques, 61 con 4) y que el relleno del final no afecta a los minutos válidos.
- Registrada en `src/models/__init__.py` como `tcn`, así `train.py` y `hp_search.py` la usan sin cambios. Config: `configs/tcn_base.yaml`; grid: `configs/grid_tcn.yaml`.

## Búsquedas de hiperparámetros (val, 1 semilla)

| Modelo | Mejor configuración | val_loss | val AUC | Rango de las 16 |
|---|---|---|---|---|
| M3 MLP | hidden 64, 2 capas, dropout 0.3, lr 1e-3 | 0.4930 | 0.8118 | 0.4930 – 0.4940 |
| M5 TCN | hidden 64, 3 bloques (29 min), dropout 0.3, lr 1e-3 | 0.4922 | 0.8121 | 0.4922 – 0.4934 |
| *M2 LightGBM (ref.)* | — | 0.4937 | 0.8113 | — |

- Los dos grids salen **planos**: la diferencia entre la mejor y la peor configuración es ~0.001 de log-loss. Los hiperparámetros casi no importan.
- En la TCN, 3 bloques (ve 29 min) le ganan levemente a 4 (61 min): ver la partida entera no aporta más que ver la última media hora.
- Tablas completas: `results/hp/mlp_hp.csv`, `results/hp/tcn_hp.csv`.

## Evaluación en test (preliminar, 1 semilla)

| Modelo | AUC | Log-loss | ECE | ΔAUC vs M2 (pareado) | ΔLog-loss vs M2 |
|---|---|---|---|---|---|
| M2 LightGBM | 0.811 | 0.522 | 0.019 | — | — |
| M3 MLP | 0.812 | 0.521 | 0.018 | +0.0012 [+0.0002, +0.0022] | −0.0014 [−0.0027, −0.0001] |
| M5 TCN | 0.813 | 0.520 | 0.021 | **+0.0021 [+0.0009, +0.0033]** | −0.0022 [−0.0040, −0.0005] |

La TCN es significativamente mejor que M2, pero por muy poco, y la ganancia está en los minutos 10–30 (`results/b5_comparacion_vs_M2.csv`). **Ver la historia aporta poco en AUC.** Eso hace más importante comparar los modelos por **valor** (B10/B12) y por suavidad de la curva (E6), no solo por AUC.

## E2 · Curva de aprendizaje (M2; M4 pendiente)

| Partidas de train | AUC test (media ± sd, 3 semillas) | Log-loss test | Entrenamiento |
|---|---|---|---|
| 1,000 | 0.7999 ± 0.0010 | 0.5367 | 1.3 s |
| 3,000 | 0.8062 ± 0.0010 | 0.5284 | 2.0 s |
| 6,000 | 0.8089 ± 0.0009 | 0.5247 | 3.2 s |
| 12,000 | 0.8100 ± 0.0004 | 0.5232 | 5.2 s |
| 18,294 (todo) | 0.8108 ± 0.0001 | 0.5221 | 6.6 s |

![Curva de aprendizaje](figures/b7_curva_aprendizaje.png)

- **Rendimientos decrecientes claros en M2:** de 1k a 6k gana +0.009 de AUC; de 12k a 18k, solo +0.0008. Con estas features, conseguir más partidas casi no mejoraría a LightGBM.
- **Pendiente:** la curva de M4 (GRU) necesita la mejor configuración de la búsqueda N1 de la GRU (B6, Melisa). Con ella:
  `python src/learning_curve.py --models m4 --set model.hidden=<h> model.layers=<l> model.dropout=<d> train.lr=<lr>`
  (15 corridas; reanudable). La pregunta clave para el informe: **¿la GRU sigue mejorando con más datos donde LightGBM ya se aplanó?**
