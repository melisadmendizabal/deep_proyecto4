# B6 · GRU (M4) y MLP (M3): búsqueda, modelos finales y evaluación

*7 nov 2026. Se reproduce con:*
```
python src/hp_search.py --config configs/gru_base.yaml --grid configs/grid_gru.yaml --prefix gru_hp        # 7 min (RTX 4050)
python src/hp_search.py --config configs/gru_base.yaml --prefix gru_final --seeds 42 43 44 \
    --set model.hidden=128 model.layers=2 model.dropout=0.1 train.lr=0.0003                              # ~1 min
python src/hp_search.py --config configs/mlp_base.yaml --prefix mlp_final --seeds 42 43 44 \
    --set model.hidden=64 model.layers=2 model.dropout=0.3 train.lr=0.001                                # ~30 s
python src/evaluate.py --run results/runs/m2_lgbm results/runs/mlp_final__s42 results/runs/gru_final__s42 --name M2 M3 M4
python src/seed_summary.py --prefix gru_final mlp_final --name M4 M3
```

## Búsqueda de la GRU (val, 16 configuraciones, 1 semilla)

- **Mejor:** hidden 128, 2 capas, dropout 0.1, lr 3e-4 → val_loss **0.4924**, val AUC 0.8123 (14 épocas, 12 s).
- La búsqueda sale **plana**, igual que las del MLP y la TCN: entre la mejor y la peor configuración hay 0.0008 de log-loss. Los hiperparámetros casi no importan con estas features.
- Tabla completa: `results/hp/gru_hp.csv`.

## Modelos finales (3 semillas) en test

| Modelo | AUC test (media ± sd) | Log-loss test | ECE | val_loss | Entrenar | Parámetros |
|---|---|---|---|---|---|---|
| M3 MLP (h64, 2 capas) | 0.8118 ± 0.0001 | 0.5211 ± 0.0006 | 0.020 | 0.4929 | 6 s | ~7k |
| **M4 GRU** (h128, 2 capas) | **0.8123 ± 0.0002** | **0.5207 ± 0.0006** | 0.020 | 0.4925 | 12 s | ~0.2M |

Por semilla: `results/b6_semillas.csv`. La variación entre semillas es diminuta: el resultado no depende de la suerte de la inicialización.

## Comparación pareada contra M2 (LightGBM), test, semilla 42

| Modelo | AUC (IC 95 %) | ΔAUC vs M2 | ΔLog-loss vs M2 |
|---|---|---|---|
| M2 LightGBM | 0.811 [0.802, 0.819] | — | — |
| M3 MLP | 0.812 [0.803, 0.820] | +0.0012 [+0.0003, +0.0023] | −0.0015 [−0.0028, −0.0002] |
| M4 GRU | 0.812 [0.804, 0.821] | **+0.0017 [+0.0005, +0.0029]** | −0.0017 [−0.0034, +0.0000] |
| M5 TCN (de B7) | 0.813 | +0.0021 [+0.0009, +0.0033] | −0.0022 [−0.0040, −0.0005] |

`results/b5_comparacion_vs_M2.csv` (tiene las cuatro filas: M3, M4 de B6 y M3_hp, M5_hp de B7).

## Qué significa

1. **La GRU supera a LightGBM de forma estadísticamente significativa, pero por muy poco** (+0.0017 de AUC). Los tres modelos de deep learning (MLP, GRU, TCN) y LightGBM quedan a ±0.002 de AUC entre sí.
2. **Ver la historia completa casi no aporta en AUC**: GRU (historia completa) ≈ TCN (29 min) ≈ MLP (solo el minuto actual + tendencias). El estado actual del juego ya resume casi todo lo que importa. Es un hallazgo honesto para el informe (sección 4.8).
3. **Por eso la comparación decisiva es por valor, no por AUC** (B10/B12): una curva más suave o mejor calibrada puede dar menos "momentos" falsos y ahorrar más tiempo aunque el AUC sea igual. E6 (pérdida que penaliza saltos) prueba justo eso.
4. Costo: entrenar la GRU cuesta 12 s de GPU y la inferencia es despreciable. La diferencia de costo entre modelos no es un factor.

## Detector de momentos sobre la GRU (B8, val)

| θ | Momentos por partida | % sin momentos |
|---|---|---|
| 0.10 | 4.4 | 0.0 |
| 0.12 | 4.1 | 0.1 |
| 0.15 | 3.5 | 1.6 |
| 0.20 | 2.4 | 13.8 |

Con θ ≤ 0.10 casi todas las partidas llegan al tope de 5. El θ útil probablemente está entre 0.12 y 0.20; se elige en B10 con las anotaciones de val.

## Para A (compañero): curva de aprendizaje de M4 (E2, B7)

```
python src/learning_curve.py --models m4 --set model.hidden=128 model.layers=2 model.dropout=0.1 train.lr=0.0003
```
