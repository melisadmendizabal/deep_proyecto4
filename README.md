# Fight Diff: ¿en qué minuto se decidió la partida?

Proyecto final de CC3092 Deep Learning (UVG, noviembre 2026). Melisa Mendizabal y Renato Rojas.

Dada una partida ranked de League of Legends, **Fight Diff** estima minuto a minuto la probabilidad de que gane el equipo azul con un modelo secuencial entrenado por nosotros (Transformer causal + campeones). Después marca los **momentos decisivos**, es decir, los saltos grandes de esa probabilidad, para que un coach revise solo esos clips y no la repetición completa. El valor se mide en minutos de revisión ahorrados y en USD al mes por equipo.

- **App:** `<URL de Render>` (plan gratis: la primera visita tarda ~1 min en despertar). También se puede correr en local: ver [Correr la app](#correr-la-app).
- **Presentación:** [`presentacion/Fight_Diff.pdf`](presentacion/Fight_Diff.pdf)
- **Guía de trabajo y bitácora de decisiones:** [`docs/GUIA_PROYECTO.md`](docs/GUIA_PROYECTO.md)

## Resultados principales

| | Resultado | Detalle |
|---|---|---|
| Datos | 27,000 partidas ranked propias (LAN, NA, EUW, KR × Silver–Master), split temporal 70/15/15 | [`results/b2_resumen.md`](results/b2_resumen.md), [`results/b4_eda.md`](results/b4_eda.md) |
| AUC en test | M0 (solo oro) 0.794 · M2 LightGBM 0.811 · GRU 0.812 · **M6 + campeones 0.821** | [`results/mejoras.md`](results/mejoras.md) |
| Valor | Revisión de ~40 → ~20 min por partida: **USD 123–148 al mes por equipo** según el detector (oráculo: 209) | [`results/b10_valor.md`](results/b10_valor.md) |
| Hallazgo | Más AUC no significa más valor: los campeones suben el AUC pero no cambian los saltos de la curva, y una regla simple (R0) compite en número de clips | [`results/b12_tabla_e1.md`](results/b12_tabla_e1.md) |
| Costo | Todo el entrenamiento: ~31 min de GPU de laptop. Hosting: USD 0. Una partida nueva tarda 1.2 s (95 % es la API de Riot) | [`results/b11_latencia.md`](results/b11_latencia.md) |
| Explicabilidad | SHAP de LightGBM, ablación por grupos y 3 casos de falla | [`results/b12_explicabilidad.md`](results/b12_explicabilidad.md), [`results/b12_fallas.md`](results/b12_fallas.md) |

## Qué hay en el repo y qué no

**Sí está en git:** el código, el split congelado (`splits/splits.csv`), las anotaciones de momentos (`annotations/`), las métricas y los resúmenes de cada corrida (`results/`, sin `predictions.npz` ni pesos), los modelos exportados que usa la app (`models/`, ~8 MB) y 5 partidas precalculadas para usar la app sin conexión (`app/demo/`).

**No está en git** (por tamaño o por privacidad): las partidas descargadas (`data/`, varios GB), las features (`features/sequences.npz`), los pesos de las corridas (`results/runs/*/best.pt`), las predicciones (`predictions.npz`) y el `.env` con la API key.

Con lo que está en git se puede **correr la app y las pruebas**. Para **reentrenar** hay que recolectar los datos con una API key de Riot (pasos 1–3) o pedirnos `features/sequences.npz` (~50 MB) y empezar en el paso 4.

## Instalación

Probado en Windows 11 con Python 3.12 y 3.13; la app en Render usa Python 3.12.10.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
# PyTorch aparte. Con GPU NVIDIA (lo que usamos, torch 2.6.0 + CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124
# Solo CPU (suficiente para la app y las pruebas):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Variables de entorno: copiar `.env.example` como `.env` y llenar:

| Variable | Para qué |
|---|---|
| `RIOT_API_KEY` | Key de https://developer.riotgames.com (la de desarrollo vence cada 24 h) |
| `PUUID_SALT` | Texto secreto para anonimizar a los jugadores (los PUUID se guardan como hash con sal) |

## Correr la app

```bash
uvicorn app.api:app --port 8000
```

Abrir http://127.0.0.1:8000. Se puede buscar por Riot ID (`Nombre#TAG` + servidor) o por match ID (`NA1_…`). La documentación de la API queda en `/docs`. Sin API key, la pestaña **Uso sin conexión** analiza las 5 partidas de ejemplo de `app/demo/`.

El modelo principal es M6 + campeones (`models/m6_draft`, θ* = 0.15); el selector ofrece M0–M5 y M4 + campeones. El despliegue en Render está explicado paso a paso en [`docs/despliegue.md`](docs/despliegue.md).

## Pruebas

```bash
python -m pytest -q
```

Cubren las features (sin fugas: borrar eventos futuros no cambia el pasado), los datasets, el entrenamiento, el detector de momentos, la función de valor y la API. No necesitan los datos reales: las de entrenamiento generan al vuelo un dataset falso con el mismo formato (`tests/make_fake_sequences.py`).

## Reproducir todo, en orden

Los tiempos son los de una laptop con RTX 4050/4060. Cada script explica sus opciones en el encabezado (`python src/<script>.py --help`).

### 1. Datos (B1–B2)

```bash
python collect.py                  # descarga partidas ranked por región × tier (riot_data/config.py)
python merge.py                    # 1,125 partidas por región × tier -> data/datos_merge/
python src/split.py                # split temporal 70/15/15 dentro de cada región × tier (congelado en splits/)
python src/quality_check.py        # partidas con AFK, duplicados y fugas de jugadores -> results/b2_*
```

`collect.py` tarda muchas horas: la key personal permite ~100 llamadas cada 2 min y cada partida necesita 2 (partida + timeline). Se puede cortar y retomar: salta las partidas que ya descargó.

### 2. EDA (B4)

```bash
python src/eda.py                  # ~3 min -> results/b4_eda.md y results/figures/b4_*
```

Los supuestos del modelo de valor, con sus fuentes, están en [`docs/supuestos_valor.md`](docs/supuestos_valor.md) y [`configs/value.yaml`](configs/value.yaml).

### 3. Features (B3)

```bash
python src/build_features.py       # ~5-10 min -> features/sequences.npz + features/scaler.json
```

35 features por equipo y minuto (azul − rojo), solo con eventos de timestamp ≤ t. El contrato está en [`docs/contrato_features.md`](docs/contrato_features.md). El scaler se ajusta solo con train.

### 4. Baselines (B5)

```bash
python src/baselines.py            # M0, M1, M2 + pruebas de cordura (~3-6 min, CPU)
python src/evaluate.py --run results/runs/m0_oro results/runs/m1_logit results/runs/m2_lgbm --name M0 M1 M2
```

### 5. Modelos de deep learning (B6–B7)

Cada búsqueda prueba 16 combinaciones (~3-7 min) y elige por log-loss de val. Las finales usan 3 semillas.

```bash
# búsquedas
python src/hp_search.py --config configs/mlp_base.yaml --grid configs/grid_mlp.yaml --prefix mlp_hp
python src/hp_search.py --config configs/gru_base.yaml --grid configs/grid_gru.yaml --prefix gru_hp
python src/hp_search.py --config configs/tcn_base.yaml --grid configs/grid_tcn.yaml --prefix tcn_hp
python src/hp_search.py --config configs/transformer_base.yaml --grid configs/grid_transformer.yaml --prefix m6_hp

# modelos finales (mejor configuración × 3 semillas)
python src/hp_search.py --config configs/mlp_base.yaml --prefix mlp_final --seeds 42 43 44 --set model.hidden=64 model.layers=2 model.dropout=0.3 train.lr=0.001
python src/hp_search.py --config configs/gru_base.yaml --prefix gru_final --seeds 42 43 44 --set model.hidden=128 model.layers=2 model.dropout=0.1 train.lr=0.0003
python src/hp_search.py --config configs/tcn_base.yaml --prefix tcn_final --seeds 42 43 44 --set model.hidden=64 model.layers=3 model.dropout=0.3 train.lr=0.001
python src/hp_search.py --config configs/transformer_base.yaml --prefix m6_final --seeds 42 43 44 --set model.dropout=0.3
python src/seed_summary.py --prefix gru_final mlp_final tcn_final --name M4 M3 M5
```

`evaluate.py` compara cualquier corrida contra las baselines con IC por bootstrap (`--run results/runs/gru_final__s42 --name M4`).

### 6. Experimentos (E2–E6)

```bash
# E2: curva de aprendizaje de M2 y M4
python src/learning_curve.py --gru-config configs/gru_base.yaml --set model.hidden=128 model.layers=2 model.dropout=0.1 train.lr=0.0003
# E3 (alto elo -> elo promedio) y E4 (región no vista)
python src/experiments_e3e4.py
# E6: GRU con penalización de saltos en la pérdida
python src/hp_search.py --config configs/gru_base.yaml --prefix e6_sm0.5 --seeds 42 43 44 --set model.hidden=128 model.layers=2 model.dropout=0.1 train.lr=0.0003 train.smooth_lambda=0.5
```

### 7. Momentos decisivos y anotación (B8–B9)

```bash
python src/select_annotation.py    # elige 55 partidas de val/test -> annotations/seleccion.csv + data/anotacion/
streamlit run src/annotate_app.py  # herramienta para anotar; protocolo en docs/protocolo_anotacion.md
```

Anotamos solo con el log de eventos, sin ver la curva del modelo. Las anotaciones están en `annotations/A.csv` y `annotations/B.csv`, y las revisiones de repeticiones (para medir f_rev) en `annotations/revisiones_*.csv`.

### 8. Valor (B10) y mejoras

```bash
python src/value_model.py          # valor de R0 y M0-M5: θ* elegido en val, valor en test, tornado, punto de equilibrio
python src/agreement.py            # acuerdo entre anotadores (F1 a ±2 min)
python src/predict_matches.py      # GRU vs las notas de las repeticiones de partidas propias
python src/hybrid_eval.py          # iteración GRU + regla R0

# M4 + campeones (búsqueda de lr y dropout con 3 semillas) y M6 + campeones
python src/hp_search.py --config configs/gru_base.yaml --grid configs/grid_gru_draft.yaml --prefix m4d_hp --seeds 42 43 44 --set model.hidden=128 model.layers=2 model.champions=true
python src/hp_search.py --config configs/transformer_base.yaml --prefix m6d_final --seeds 42 43 44 --set model.dropout=0.3 model.champions=true
python src/mejoras_eval.py         # ensambles de semillas, M6 y campeones, en AUC y en valor
```

### 9. Explicabilidad, casos de falla y tabla E1 (B12)

```bash
python src/b12_resultados.py       # ~3 min: SHAP, ablación por grupos, 3 casos de falla, tabla E1 con costos
```

### 10. Exportar los modelos para la app (B11)

```bash
python src/export_model.py --runs results/runs/m6d_final__s42 results/runs/m6d_final__s43 results/runs/m6d_final__s44 --name "M6+draft ens3" --manifest-name M6 --out models/m6_draft --promedio --theta 0.15
python app/make_demo.py            # rehace las partidas de ejemplo con el modelo nuevo
```

Los comandos para los demás modelos del selector están en el encabezado de [`src/export_model.py`](src/export_model.py). Monitoreo y latencia: `python app/monitor.py --simulate 300` y `python app/bench_latency.py` (plan en [`docs/monitoreo.md`](docs/monitoreo.md)).

## Estructura

```
collect.py, merge.py     descarga y selección de partidas (riot_data/: cliente de la API con límites de tasa)
src/                     split, features, baselines, modelos (src/models/), entrenamiento, evaluación,
                         momentos, modelo de valor, explicabilidad
app/                     API (FastAPI), motor de inferencia, frontend (app/static/), monitoreo
configs/                 configuraciones de los modelos, grids de búsqueda y parámetros de valor
annotations/             momentos anotados a mano y revisiones de repeticiones
splits/                  split congelado
models/                  modelos exportados que usa la app
results/                 métricas, reportes (.md), figuras y resumen de cada corrida
docs/                    guía, supuestos de valor, protocolo de anotación, despliegue, monitoreo
tests/                   pruebas (pytest)
```

## Privacidad y uso

- Los PUUID se guardan solo como hash con sal. La app no guarda nombres de jugadores en el registro de predicciones.
- El análisis es **solo post-partida** (match-v5 no da datos en vivo); usarlo durante una partida iría contra las políticas de Riot.
- El sistema mide equipos y momentos, no está pensado para juzgar a jugadores individuales.

*Fight Diff isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.*
