# B11 · Backend e inferencia

*12 nov 2026.*

## Cómo correrlo

```
python src/export_model.py            # empaqueta la GRU (3 semillas) + scaler + temperatura + θ* en models/m4_gru/
                                      # (--retrain si faltan los best.pt; los pesos no van a git)
uvicorn app.api:app --port 8000       # API en http://127.0.0.1:8000, documentación en /docs
python app/bench_latency.py           # latencia (--riot 10 para medir también la API de Riot; necesita .env)
```

## Qué hace

```
Riot ID / match ID ─► app/api.py ─► caché (memoria → data/app_cache → app/demo) ─► JSON crudo (caché, data/raw o API de Riot)
                                   └─► app/inference.py: parse_match → dedup_events → match_to_features → scaler.json
                                       → GRU × 3 semillas → temperatura → P(t) → detect_moments(θ* = 0.23)
                                       → explicación por momento (ablación por grupos) + eventos de la ventana
```

| Endpoint | Qué devuelve |
|---|---|
| `GET /health` | modelo, θ, semillas, parches de entrenamiento, si hay API key, partidas en caché y de demo |
| `GET /player?riot_id=Nombre%23TAG&platform=na1&count=5` | partidas ranked recientes: campeón, rol, lado, KDA, duración, si es analizable o ya está analizada |
| `GET /analyze?match_id=NA1_…&refresh=false` | curva `p_azul` por minuto con banda `p_min`–`p_max` (semillas) y `baja_certeza` (min 0–4); momentos con minutos del clip, equipo favorecido, Δ en puntos, confianza, explicación y eventos; avisos; tiempos |

- **Curva y momentos con la semilla 42**, el mismo modelo con el que B10 eligió θ* = 0.23 y calculó el valor, para que la app muestre lo mismo que se reporta. Las semillas 43 y 44 solo dan la **banda de incertidumbre**.
- **Explicación por momento (guía, sección 8):** se congela cada grupo de features (economía + rol, combate, objetivos, visión) en su valor del inicio del clip y se recalcula P al final. El grupo que más achica el salto es la explicación, y se devuelve el reparto completo. Ejemplo real: *"min 37–40: 35 pts de probabilidad a favor de rojo, sobre todo por objetivos (azul: 4 kills, torre; rojo: 6 kills, dragón, torre)"*.
  - **Limitación:** "economía" sale primero la mayoría de las veces, porque las kills y los objetivos se convierten en oro. Por eso cada momento lista los eventos concretos de la ventana y el reparto entre grupos.
- **Errores (guía, sección 8):**

  | Caso | Respuesta |
  |---|---|
  | ID inválido | 400 |
  | Partida inexistente | 404 |
  | Remake o partida de menos de 15 min | 422, "no se analiza" |
  | Key expirada o sin red | 503, pero si la partida está en caché se devuelve con un aviso |

  Además, se avisa si la partida no es ranked Solo/Duo o si su parche no estuvo en entrenamiento.
- **Monitoreo:** cada análisis nuevo se registra en `data/app_cache/predicciones.jsonl`: parche, P final, n.º de momentos, tiempos y avisos. Es la base de `docs/monitoreo.md` (B13).

## Verificación: la app calcula lo mismo que el entrenamiento

Se pasaron 20 partidas de test **desde el JSON crudo de Riot** por el pipeline de la app y se compararon con las predicciones del entrenamiento (`predictions.npz`). La diferencia máxima es 8.8e-5, que es el redondeo a 4 decimales.

**Bug encontrado y corregido.** La primera versión difería hasta en **0.09 de probabilidad**. El timeline de Riot repite algunos eventos tal cual (sobre todo `WARD_PLACED` "UNDEFINED" y alguna placa; hasta 200 en una partida), y `merge.py` los había borrado del dataset de entrenamiento (`drop_duplicates()` de filas idénticas). La app los contaba y el modelo veía más wards y placas de los que vio en el entrenamiento.
- **Arreglo:** `src/features.py: dedup_events`, que quita filas idénticas completas, igual que `merge.py`. Lo usan la app y `src/predict_matches.py` (las partidas propias vienen de `collect.py`, que no deduplica).
- No se deduplica solo por las columnas de features: eso borraría 809 eventos reales del entrenamiento (por ejemplo, dos larvas en el mismo milisegundo con posiciones distintas).
- **Consecuencia:** `results/b10_propias.*` se calculó antes del arreglo. Hay que volver a correr `python src/predict_matches.py` (en la PC que tenga `data/csv/na1_own/`).

## Latencia (`results/b11_latencia.md`)

100 partidas de test distintas, CPU (laptop de A, 4 hilos), modelo con 3 semillas:

| Etapa | p50 (ms) | p95 (ms) |
|---|---|---|
| Parse del JSON de Riot | 8.2 | 13.8 |
| Features | 11.7 | 21.3 |
| Modelo (3 semillas) | 15.6 | 22.4 |
| Explicación de momentos | 8.4 | 43.6 |
| **Total sin Riot** | **48.4** | **86.9** |

- **Con caché** (la misma partida otra vez), la respuesta tarda ~2 ms.
- **Falta:** medir la API de Riot (`--riot 10`) cuando haya `.env`. La guía estima 0.5–1.5 s, así que la meta de < 3 s por partida nueva se cumple con margen.

## Pesos usados

`models/m4_gru/` se generó reentrenando en la PC de A con el `config.yaml` exacto de `results/runs/gru_final__s4*` (`--retrain`). Los resultados son prácticamente iguales a los de B6:

| Semilla | val AUC (B6) | val AUC (reentrenado) |
|---|---|---|
| 42 | 0.8123 | 0.8123 |
| 43 | 0.8122 | 0.8122 |
| 44 | 0.8123 | 0.8124 |

Las temperaturas y las épocas también coinciden. Si se usan los `best.pt` originales de B, basta con correr `python src/export_model.py` sin `--retrain`.

## Pendiente

- [ ] `.env` con la API key en la PC donde se despliegue, para `/player` y las partidas nuevas, y para medir la latencia de Riot.
- [ ] Decidir qué detector usa el producto. Con los parámetros base, B10 da R0 > M0 > M4 en valor (la diferencia no es significativa). La app hoy usa la GRU (M4); la API está lista para exportar otro modelo o θ.
- [ ] Partidas de demo precalculadas en `app/demo/` (B13).
