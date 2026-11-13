# Reporte de monitoreo

*`python app/monitor.py` · 300 análisis en `predicciones.jsonl` · 2026-11-13 06:26:51 → 2026-11-13 06:27:19 UTC.*

## Calidad por parche

| parche | partidas | log-loss | Brier | ECE | AUC min 15 | momentos/partida |
|---|---|---|---|---|---|---|
| 16.21 | 152 | 0.542 | 0.185 | 0.048 | 0.812 | 1.8 |
| 16.22 | 148 | 0.464 | 0.154 | 0.036 | 0.865 | 1.7 |

## Calidad por semana

| semana | partidas | log-loss | Brier | ECE | AUC min 15 | momentos/partida |
|---|---|---|---|---|---|---|
| 2026-W46 | 300 | 0.503 | 0.170 | 0.019 | 0.836 | 1.7 |

## Deriva de datos (PSI de las features del minuto 15 vs train)

| parche | partidas | PSI máx (feature) | features con PSI > 0.2 |
|---|---|---|---|
| 16.21 | 152 | 0.129 (gold_diff_bottom) | — |
| 16.22 | 148 | 0.140 (gold_diff_jungle) | — |

## Alertas

- Ninguna.

*Umbrales: {"ece": 0.05, "logloss_extra": 0.05, "psi": 0.2, "momentos_min": 0.8, "momentos_max": 4.0, "n_min": 30}. Sin alertas con menos de 30 partidas por grupo.*
