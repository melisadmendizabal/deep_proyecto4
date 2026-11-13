# Plan de monitoreo (B13 · sección 4.6 del informe)

*13 nov 2026. Código: `app/api.py` (registro), `app/monitor.py` (reporte y alertas).*

## Por qué se puede monitorear tan directo

La app es **solo post-partida**: cuando alguien analiza una partida, el resultado ya se conoce. Así que **cada predicción se puede evaluar en cuanto se hace**, sin esperar etiquetas. No hace falta un proceso de "ground truth diferido", como en otros productos de ML.

## Qué se registra

Cada análisis nuevo agrega una línea a `data/app_cache/predicciones.jsonl` (fuera de git). Los errores de formato de Riot van a `errores.jsonl`.

| Campo | Para qué |
|---|---|
| `utc`, `match_id`, `plataforma`, `parche`, `inicio_utc` | agrupar por parche, región y semana |
| `curva_p_azul` (P por minuto) y `gana` | log-loss, Brier, ECE y AUC contra el resultado real |
| `n_momentos`, `n_candidatos`, `theta` | ver si el detector marca de más o de menos |
| `features_min15` (las 35 features crudas del minuto 15) | deriva de datos (PSI) contra train |
| `tiempos` (Riot, features, modelo, explicación) | latencia p50/p95 |
| `avisos` | cuántas partidas no son ranked o son de un parche nuevo |

**Privacidad:** no se guardan nombres, Riot IDs ni `puuid`. Los análisis cacheados tampoco los tienen. `/player` usa el `puuid` solo en memoria, para encontrar al jugador dentro de la partida.

## Métricas y alertas (`python app/monitor.py`)

| Métrica | Agrupada por | Valor de referencia (test) | Alerta si |
|---|---|---|---|
| Log-loss por minuto | parche, semana | 0.520 | > 0.570 |
| ECE (10 bins) | parche, semana | ~0.02 | > 0.05 → recalibrar la temperatura con datos recientes |
| AUC en el minuto 15 | parche, semana | ~0.84 | se reporta (sin umbral fijo) |
| Momentos por partida | parche, semana | 1.9 (B10) | fuera de [0.8, 4] → revisar θ |
| PSI de cada feature del minuto 15 vs train | parche | ~0 | > 0.2 → posible cambio de meta |
| Errores de formato de Riot | día | 0 | cualquiera → revisar `riot_data/parse.py` |
| Latencia | semana | p50 1.2 s, p95 < 3 s (B11) | p95 > 3 s |

- Las alertas solo se calculan con **≥ 30 partidas** por grupo, para no alarmarse por ruido.
- La referencia de PSI (deciles de cada feature en el minuto 15, 18,294 partidas de train) está en `app/monitor_ref.json`. Se regenera con `python app/monitor.py --build-ref`.

## Validación del monitoreo

Se pasaron 300 partidas de test por la app (`python app/monitor.py --simulate 300` → `results/monitoreo_simulacion.md`). El reporte reproduce la evaluación offline:

- **Calidad:** log-loss 0.503 y ECE 0.019 en total (test: 0.52 y 0.02).
- **Deriva:** PSI máximo 0.13 en 16.21 (parche de train) y 0.14 en 16.22 (parche no visto). Ninguna alerta, consistente con que en test el parche nuevo no se degradó (B11).
- **Bug encontrado al validar:** `pd.read_json` parsea floats de forma imprecisa (0.2 → 0.19999…), y features discretas como `level_diff`, que caen justo en los bordes de los bins, cambiaban de bin. Daba un PSI falso de 0.68. Ahora el registro se lee con `json`.

## Qué hacer ante cada alerta

| Alerta | Acción |
|---|---|
| ECE alto, AUC estable | Recalibrar: reajustar la temperatura con las partidas registradas del parche nuevo (no hace falta reentrenar) |
| PSI > 0.2 en varias features o log-loss alto | Cambió el meta. Descargar partidas del parche nuevo con `collect.py`, volver a correr `merge.py` → `build_features.py` → `train.py` y comparar contra el modelo actual en el mismo test |
| Momentos por partida fuera de rango | Volver a elegir θ con `src/value_model.py` (requiere nuevas anotaciones si cambió mucho) |
| Errores de formato de Riot | Riot cambió el esquema de match-v5: actualizar `riot_data/parse.py` y sus tests |
| Evento nuevo en el timeline (p. ej. un objetivo nuevo) | `src/features.py` lo ignora sin romperse; si es importante, agregar la feature y reentrenar |

**Rutina:** revisar el reporte una vez por semana y en cada parche nuevo (cada ~2 semanas). Por defecto, reentrenar cada 2 parches.

## Limitaciones

- El registro solo ve las partidas que la gente analiza: no es una muestra aleatoria (sesgo de selección; por ejemplo, más partidas perdidas que la gente quiere entender).
- La calidad de los **momentos** (precisión y recall) no se puede monitorear automáticamente: requiere anotación humana. Lo que sí se puede es preguntarle al coach "¿te sirvió este clip?" (sí/no) en la app. Queda como trabajo futuro.
