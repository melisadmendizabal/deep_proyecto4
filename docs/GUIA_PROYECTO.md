# Guía del proyecto final: probabilidad de victoria en LoL

*Actualizada: 5 de noviembre de 2026*

**Objetivo:** construir y desplegar un sistema que, a partir del timeline de una partida de LoL, estime minuto a minuto la probabilidad de victoria con un modelo secuencial de deep learning, detecte los momentos decisivos y cuantifique cuántas horas de revisión ahorra cada mejora del modelo.

**Fechas que mandan**

- **Domingo 15 nov:** demo, presentación y video listos, porque el turno puede caer el lunes 16.
- **Miércoles 18 nov:** entrega del trabajo escrito. Del 16 al 18 solo se pule el escrito.

**Punto de partida (ya hecho)**

- `collect.py` y `merge.py` funcionan.
- `data/datos_merge/` tiene 27,000 partidas balanceadas: 4 regiones × 6 tiers × 1,125.
- Quedan ~4,000 partidas de reserva en `data/csv/`.

---

## Avisos y estado (sáb 7 nov)

| Bloque | Estado |
|---|---|
| B1 Entorno + API key (A) | ✅ PC de A: `.venv` con Python 3.12.10, torch 2.6.0+cu124 (RTX 4060), LightGBM. La Personal API Key ya está registrada (Melisa) |
| B2 Split + calidad (B) | ✅ Completado |
| B3 Features (A) | ✅ Completado: `python src/build_features.py` (~5 min) → `features/` (pedirle a A el zip o regenerarlo). Detalle en `results/b5_resumen.md` |
| B4 EDA + supuestos (B) | ✅ Completado |
| B5 Baselines + evaluación (A) | ✅ Completado: M0 AUC 0.794, M1 0.810, M2 0.811 en test. **La GRU debe superar a M2.** Detalle en `results/b5_resumen.md` |
| B7 TCN + curva (A) | 🟡 TCN + búsquedas de MLP y TCN ✅; curva de M2 ✅. **Falta la curva de M4: espera la mejor config de la GRU (búsqueda N1 de B; tarda ~5 min)**. Detalle en `results/b7_resumen.md` |
| B6 Entrenamiento DL (B) | 🟡 Código listo y probado; smoke test con `sequences.npz` real ✅ (lo corrió A). **Ojo:** A corrigió `fit_temperature` en `src/train.py` (LBFGS sin búsqueda de línea daba T=0.0002 con modelos poco confiados; ahora usa `line_search_fn="strong_wolfe"` + test de regresión) |

**Para A (compañero), antes de empezar B3:**
1. Leer `docs/contrato_features.md`. El `.npz` debe cumplirlo exactamente; si no, `load_sequences` falla con un mensaje que dice qué falta.
2. Usar la columna `excluir_de_train` de `results/b2_calidad_partidas.csv` como el arreglo `afk`, y `splits/splits.csv` para `split` (no recalcularlo).
3. Ideas del EDA para las features: incluir **oro relativo** (diferencia / oro total), el **minuto**, objetivos (torres, dragones, barón) y no solo oro.
4. Para verificar el archivo al terminar: `python -c "from src.datasets import load_sequences; d = load_sequences('features/sequences.npz'); print(len(d), d.n_features)"`.

**Cuando exista `features/sequences.npz`, la noche N1 queda así** (los comandos ya están probados con datos sintéticos):
```
python src/train.py --config configs/gru_base.yaml --smoke                                    # 1-2 min, antes de dormir
python src/hp_search.py --config configs/gru_base.yaml --grid configs/grid_gru.yaml --prefix gru_hp   # GPU de B
python src/hp_search.py --config configs/mlp_base.yaml --grid configs/grid_mlp.yaml --prefix mlp_hp   # GPU de A
```
Resultados en `results/hp/gru_hp.csv` y `results/hp/mlp_hp.csv`, ordenados por val_loss.

**Calendario:** van ~1.5 días atrás. Las noches N1–N3 se corren desde la noche en que B3 esté listo; si B3 se termina el domingo 8, N1 es esa noche y no se pierde casi nada. Si se atrasa más, recortar según la tabla de riesgos (sección 11).

**Sin commits todavía.** Archivos nuevos para subir: `src/`, `tests/`, `configs/`, `splits/`, `results/` (excepto `results/runs/**/predictions.npz`, ya ignorado), `docs/`. `features/` está en `.gitignore`.

---

## Índice

0. [Reglas de trabajo](#0-reglas-de-trabajo)
1. [Calendario](#1-calendario)
2. [Modelos a comparar](#2-modelos-a-comparar)
3. [Datos: split, limpieza, fugas, features y normalización](#3-datos-split-limpieza-fugas-features-y-normalización)
4. [Experimentos](#4-experimentos)
5. [Modelo de valor](#5-modelo-de-valor)
6. [Bloques detallados B1–B16](#6-bloques-detallados-b1b16)
7. [Corridas nocturnas y paralelismo](#7-corridas-nocturnas-y-paralelismo)
8. [App, despliegue, explicabilidad y monitoreo](#8-app-despliegue-explicabilidad-y-monitoreo)
9. [Checklist contra la rúbrica](#9-checklist-contra-la-rúbrica)
10. [Estructura del repo](#10-estructura-del-repo)
11. [Riesgos, plan B y cola de reserva](#11-riesgos-plan-b-y-cola-de-reserva)
12. [Banco de preguntas para la defensa](#12-banco-de-preguntas-para-la-defensa)

---

## 0. Reglas de trabajo

- **Reparto:** bloques B1 a B16. Los impares son de **A (compañero)** y los pares de **B (Melisa)**. Cada par de bloques se hace en paralelo: B1 con B2, B3 con B4, etc.
- **Se balancean horas humanas, no tareas.** Cada bloque son 3–6 h frente a la compu. Los entrenamientos largos corren de noche y no cuentan como horas de nadie. Total estimado: ~41 h por persona.
- **Handoff de 20 min al cerrar cada bloque.** Quien lo hizo se lo explica al otro. En la ronda de preguntas *cualquiera* debe poder responder sobre *cualquier* parte.
- **Sync diaria de 15 min**, por ejemplo al empezar la tarde: qué terminé, qué sigue y qué dejo corriendo esta noche.
- **Git**
  - Una rama por bloque (`b05-baselines`), merge a `main` al cerrar.
  - Los datos nunca van a git.
  - Cada quien escribe sus resultados en su propio archivo `results/<bloque>_<experimento>.csv`, para no pisarse.
- **Si un bloque se atrasa**, el otro no se queda esperando: toma una tarea de la [cola de reserva](#11-riesgos-plan-b-y-cola-de-reserva).
- **Prioridades:** **Debe** (sin esto no cumple los requisitos mínimos), **Debería** (sube nota) y **Podría** (solo si sobra tiempo).

---

## 1. Calendario

```
              Jue5  Vie6  Sáb7  Dom8 | Lun9  Mar10 | Mié11 Jue12 | Vie13 Sáb14 | Dom15 || Lun16 – Mié18
A (compañero) B1    B3    B5    B7   | B9 ──────── | B11 ─────── | B13 ─────── | B15   || presentación
B (Melisa)    B2    B4    B6    B8   | B10 ─────── | B12 ─────── | B14 ─────── | B16   || + pulir escrito
Noche (GPUs)              N1    N2   | N3    (N4)  |             |             | ensayo||
                                                                                       entrega escrito: Mié 18
```

| Ronda | Fechas | A (compañero) | h | B (Melisa) | h | Noche |
|---|---|---|---|---|---|---|
| 1 | Jue 5 | B1 Entorno + API key | 3 | B2 Split + calidad de datos | 3 | — |
| 2 | Vie 6 | B3 Pipeline de features | 6 | B4 EDA + supuestos de valor | 5 | (build de features, ~30 min, en la tarde) |
| 3 | Sáb 7 | B5 Baselines + evaluación | 5 | B6 Entrenamiento DL (MLP, GRU) | 6 | **N1:** búsqueda de hiperparámetros |
| 4 | Dom 8 | B7 TCN + curva de aprendizaje | 5 | B8 Detector de momentos + anotación | 5 | **N2:** curva de aprendizaje + modelos finales |
| 5 | Lun 9 – Mar 10 | B9 Anotación + E3/E4 | 5 | B10 Anotación + función de valor | 6 | **N3:** E3/E4 + E6; **N4** opcional: Transformer |
| 6 | Mié 11 – Jue 12 | B11 Backend e inferencia | 6 | B12 Resultados de valor, sesgos, explicabilidad | 6 | — |
| 7 | Vie 13 – Sáb 14 | B13 Frontend + despliegue | 6 | B14 Usabilidad + redacción | 5 | — |
| 8 | Dom 15 | B15 Escrito + README + video | 5 | B16 Slides + pitch + defensa | 5 | Ensayo conjunto (2 h) |
| — | Lun 16 – Mié 18 | Presentación + revisión final | — | Presentación + revisión final | — | Entrega Mié 18 |

Las rondas 5 a 7 duran dos días porque son días con clases (~3 h por día por persona).

---

## 2. Modelos a comparar

Van a entrenar una escalera de modelos, del más barato al más complejo. Al final se presenta **el que genere más valor**, no necesariamente el de mejor AUC.

| ID | Modelo | Qué ve | Pregunta que responde | Costo | Prioridad |
|---|---|---|---|---|---|
| M0 | Logística **"solo oro"**: diferencia de oro + minuto | Oro actual | ¿Cuánto se logra con lo más obvio? Es el baseline chafa | CPU, segundos | Debe |
| R0 | **Regla sin ML** para momentos: barón, ace, inhibidor o swing de oro > 1,500 en 1 min | Eventos | ¿Hace falta un modelo para detectar momentos? | Nada | Debe |
| M1 | Regresión logística, todas las features | Minuto actual | ¿Ayudan las demás features? | CPU, < 1 min | Debe |
| M2 | **LightGBM** + features de tendencia (Δ de 1 y 3 min) | Minuto actual + algo de historia | El rival clásico más fuerte. Responde "¿se consideró algo más simple?" | CPU, minutos | Debe |
| M3 | MLP (PyTorch), mismas features que M2 | Minuto actual | ¿Ayuda el DL *sin* secuencia? | GPU, minutos | Debería |
| **M4** | **GRU causal** (modelo principal) | Toda la partida hasta t | ¿Aporta la historia completa? | GPU, ~5–15 min por corrida | **Debe** |
| M5 | TCN causal (convoluciones 1D dilatadas) | Ventana de historia | ¿Otra arquitectura secuencial rinde igual y más barata? | GPU, ~5–15 min | Debería |
| M6 | Transformer causal + embeddings de campeones | Historia + draft | ¿Importan los campeones? | GPU, 20–60 min | Podría |

**Cómo leer la escalera:**
- **M0 → M2:** cuánto rinde lo clásico.
- **M3 vs M4:** aísla el efecto de la secuencia, porque usan las mismas features.
- **M4 vs M5:** compara arquitecturas secuenciales.
- **M4 vs M6:** mide el aporte de los campeones.

### Receta de entrenamiento (M3–M6)

- **Salida:** P(gana azul | todo hasta el minuto t), una por minuto. La GRU predice en cada paso.
- **Pérdida:** BCE (log-loss) promediada sobre los minutos válidos de cada partida, con máscara de padding. Se usa log-loss porque es una *proper scoring rule*: premia probabilidades calibradas, que es lo que necesita el detector de momentos.
- **Optimizador**
  - AdamW, lr 1e-3, weight decay 1e-4.
  - Batch de 128 partidas.
  - Gradient clipping 1.0.
  - Early stopping por log-loss de validación (paciencia 5, máximo 50 épocas).
- **Búsqueda de hiperparámetros:** grid con hidden {64, 128} × capas {1, 2} × dropout {0.1, 0.3} × lr {1e-3, 3e-4} = 16 combinaciones. Se elige la mejor por log-loss de validación.
- **Calibración:** temperature scaling ajustado en validación, para todos los modelos (también M0–M2).
- **Incertidumbre:** los modelos finales se entrenan con 3 semillas y se reporta media ± desviación. Las métricas de test llevan IC 95 % por bootstrap sobre *partidas*, no sobre filas.
- **Modelos snapshot (M0–M3):** una fila por (partida, minuto), con el minuto como feature. Cada fila pesa 1/duración para que las partidas largas no dominen.
- **Truncado:** secuencias de máximo 60 minutos.

---

## 3. Datos: split, limpieza, fugas, features y normalización

### 3.1 Split (B2): se hace una vez y se congela · ✅ hecho

- **Por fecha, no al azar, dentro de cada subconjunto:** en cada plataforma × tier se ordena por `game_start_utc` y se asigna el 70 % más antiguo a **train**, el 15 % siguiente a **val** y el 15 % más reciente a **test**. Así se imita el uso real (entrenar con el pasado y predecir partidas nuevas) y cada subconjunto queda balanceado.
- **No se separa por parche** porque casi todo es 16.21 y el 16.22 tiene ~2 %. La fecha cumple el mismo propósito.
- **Por partida:** todas las filas de una partida van al mismo split.
- **Verificar:**
  - Que cada subconjunto (tier × región) tenga una proporción parecida en cada split.
  - El % de jugadores de test que también aparecen en train. Se reporta como posible fuga de jugador.
- **Guardar** en `data/splits.csv` (`match_id, split`) y copiarlo a `splits/splits.csv`, que **sí se sube a git** porque pesa ~1 MB y hace todo reproducible.
- **Regla de oro:** nadie modifica el split después. Todos los experimentos entrenan **solo** con train y se evalúan en el **mismo** test.
- **La reserva** (~4,000 partidas fuera del merge) queda intacta. Sirve como test extra "fresco" o para extender E2 por encima de 27k.

### 3.2 Limpieza (B2 detecta, B3 aplica)

| Problema | Cómo detectarlo | Qué hacer |
|---|---|---|
| Remakes / partidas < 15 min | Ya filtrados en `collect.py` | — |
| Jugador AFK o leaver | Sin ganar XP **y** sin moverse ≥ 2 min seguidos desde el minuto 4 (validado en B2: 3.3 % de las partidas) | Ya marcado en `results/b2_calidad_partidas.csv` (`excluir_de_train`): excluir de train y evaluar val/test con y sin ellas |
| Frames incompletos | Cada minuto debe tener 10 jugadores; `n_frames ≈ duración/60 + 1` | B2 encontró 0 casos; dejar el filtro por si llegan datos nuevos |
| `team_position` vacío | Valor `""` | Reemplazar por `UNKNOWN` |
| Partidas muy largas (> 60 min) | `duration_s` | Conservar, truncando la secuencia en 60 |
| Duplicados | Ya resueltos en `merge.py` | — |

### 3.3 Fugas de información: NUNCA usar como feature

- `duration_s`, `n_frames`, `ended_in_surrender`, `blue_win` y `win`.
- Las estadísticas **finales** de `participants.csv`: kills, deaths, gold_earned, champ_level, daño, vision_score.
- Los eventos `GAME_END` / `winning_team`.
- Eventos con `timestamp_ms` **posterior** al frame del minuto t.
- Cualquier normalización calculada con datos de la partida completa, como "oro / oro total final".
- Estadísticas del scaler calculadas con val o test. Se ajusta **solo con train**.

**Pruebas de cordura obligatorias**
- Con las etiquetas barajadas, el AUC debe dar ≈ 0.5.
- Un modelo que solo ve el minuto debe dar AUC ≈ 0.5.
- La predicción en el minuto t no debe cambiar si se borran los eventos posteriores a t (test unitario en B3).

### 3.4 Features por equipo y minuto (B3)

Todas son **azul − rojo** en el minuto t y causales, es decir, calculadas solo con datos ≤ t.

| Grupo | Features |
|---|---|
| Economía | Diferencia de oro total, de XP, de CS (minions + jungla), de nivel medio y de oro sin gastar. Diferencia de oro relativa = dif / (oro de ambos) |
| Por rol | Diferencia de oro en TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY (útil para explicar) |
| Combate | Kills acumuladas, kills en los últimos 2 min, first blood (+1/−1/0), aces |
| Objetivos | Torres por tipo (exterior/interior/base/nexo), placas, inhibidores caídos *ahora* (reaparecen a los 5 min), dragones por equipo, alma de dragón, buff de elder activo (150 s), buff de barón activo (180 s), heraldos, larvas (HORDE), Atakhan |
| Visión | Wards puestas y destruidas (acumuladas y últimos 3 min) |
| Tendencia (solo snapshot) | Δ de la diferencia de oro en 1 y 3 min |
| Contexto | Minuto. Tier y región en one-hot solo para los experimentos E3/E4 |
| Draft (solo M6) | IDs de los 10 campeones |

### 3.5 Normalización

- Diferencias continuas → `StandardScaler` ajustado **solo con train**, con media y desviación guardadas en `features/scaler.json`.
- Conteos sesgados (wards) → `log1p` y luego estandarizar.
- Flags (−1/0/1) → sin cambios.
- Después de estandarizar, recortar a [−5, 5] para que los outliers no desestabilicen la GRU.
- **Un solo `src/features.py`** lo usan el entrenamiento **y** la app. Si la app calcula las features distinto, el modelo falla en producción sin avisar.

### 3.6 Salidas de B3

- `features/team_minute.csv`: una fila por partida-minuto (~900k filas). Para *verlas* y para los modelos snapshot.
- `features/sequences.npz`: `X [N, 60, F]`, `mask [N, 60]`, `y [N]`, `match_id [N]`. Para los modelos secuenciales; carga en segundos y ocupa ~200 MB.
- Se construyen leyendo `datos_merge` por bloques de partidas, porque los archivos ya vienen agrupados por partida, para no llenar la RAM.

---

## 4. Experimentos

Todos entrenan con **train** (o un subconjunto de train) y se evalúan en el **mismo test**.

| ID | Experimento | Entrena con | Evalúa en | Pregunta | Prioridad | Cuándo |
|---|---|---|---|---|---|---|
| **E1** | Escalera de modelos M0–M5 (+R0) | Train completo | Test + partidas anotadas | ¿Cuánto **valor** agrega cada versión y con qué costo? | **Debe** | N1–N3, resultados en B12 |
| **E2** | Curva de aprendizaje | 1k, 3k, 6k, 12k y todo train; M2 y M4; 3 semillas | Test | ¿Faltan datos? ¿Dónde están los rendimientos decrecientes? | **Debe** | N2 |
| E3 | Alto elo vs mezclado | Diamante+Master de train vs la misma cantidad al azar de todos los tiers | Test Silver + Gold | ¿Sirve entrenar con alto elo para jugadores promedio? | Podría | N3 |
| E4 | Región no vista | Train sin LA1 vs train con LA1 submuestreado al mismo tamaño | Test LA1 | ¿Se puede lanzar en una región nueva sin reentrenar? | Podría | N3 |
| E5 | Sesgo por grupo | (Modelo final de E1) | Test por tier, región, minuto y duración | ¿Funciona peor para algún grupo? ¿Cuánto cuesta? | Debería | B12 |
| E6 | Pérdida alineada con el negocio | M4 con BCE + λ·(penalización de saltos de P en minutos sin eventos), λ ∈ {0, 0.1, 0.5}; con y sin calibración | Test + anotadas | ¿Menos falsos momentos = más valor, aunque el AUC casi no cambie? | Debería | N3 |
| E7 | Ablación de features | M2/M4 sin visión, sin objetivos, solo economía | Test | ¿Qué información importa? | Podría | Cola de reserva |

**E6 puede ser la historia más fuerte de la presentación.** La sección 4.8 pide mostrar que el cambio con más impacto en el valor no siempre es el que más mueve la métrica técnica. Una curva más suave y calibrada puede casi no cambiar el AUC y aun así reducir mucho los falsos "momentos decisivos", que son minutos de revisión perdidos.

**Reglas de comparación justa:**
- E3 y E4 comparan con el **mismo número** de partidas de entrenamiento.
- Todos los modelos usan el mismo `splits.csv`, el mismo `sequences.npz` y el mismo commit, que se anota en los resultados.

---

## 5. Modelo de valor

### 5.1 El escenario

- **Usuario:** coach de un equipo amateur o universitario que revisa partidas de su equipo.
- **Situación actual:** ve la grabación completa, unos 30–40 min por partida, sin saber dónde se decidió.
- **Con la herramienta:** abre el reporte y revisa solo los clips de los momentos detectados.

### 5.2 Momentos decisivos y su ground truth

- **Detector** (`src/moments.py`):
  - Marca el minuto t si |P(t) − P(t−w)| ≥ θ, con w ∈ {1, 2} min.
  - Une detecciones a menos de 2 min entre sí y se queda con las top-K por partida.
  - El umbral **θ es el punto de operación**.
- **Anotación humana (B8–B10)**
  - Muestra: 70 partidas de val+test, estratificadas por tier. Cada quien anota 35, y **15 las anotan ambos** para medir el acuerdo entre anotadores.
  - Partidas propias: 5–10 de ustedes, revisando la repetición real del cliente. Además sirven para **medir el tiempo de revisión completo**, que es la línea base del KPI.
  - Protocolo: hasta 5 momentos por partida, con minuto (±1) y tipo (pelea, objetivo, pick, error).
  - **La herramienta de anotación NO muestra la curva del modelo ni la gráfica de oro**, solo el log de eventos. Si no, la anotación queda sesgada hacia lo que el modelo ya ve.
- **Emparejamiento:** una detección es acierto (TP) si cae a ±2 min de un momento anotado. Las detecciones sobrantes son FP y los momentos anotados no detectados son FN.
- **Partición de las anotadas:** **θ se elige con las anotadas de val** y el valor se reporta con las de test.

### 5.3 Función de valor

Por partida:

```
T_base  = f_rev × D                                   (revisión completa)
T_tool  = t_setup + (TP + FP) × t_clip + FN × t_miss  (revisión con la herramienta)
Ahorro  = T_base − T_tool                             (minutos por partida)
```

Por mes:

```
Valor_mes = N_rev × (Ahorro_promedio / 60) × C_h − C_infra − N_rev × C_pred
```

| Parámetro | Significado | Valor inicial | Fuente / cómo se obtiene | Rango de sensibilidad |
|---|---|---|---|---|
| D | Duración media de partida | De los datos (~30 min) | `matches.csv` | Fijo (dato) |
| f_rev | Factor de revisión (pausas, retrocesos) | 1.3 | **Medido** en la prueba de usabilidad y con las partidas propias | ±20 %, ±50 % |
| t_setup | Abrir y leer el reporte | 2 min | Medido en usabilidad | ±50 % |
| t_clip | Revisar un momento (1 min antes y después + notas) | 2.5 min | Medido en usabilidad | ±50 % |
| t_miss | Costo de un momento no detectado (buscarlo a mano) | 8 min | Estimación: documentarla como supuesto | ±50 % |
| C_h | Costo por hora del coach | USD 10–25 | **Buscar y citar** tarifas reales de coaching de LoL | ±50 % |
| N_rev | Partidas revisadas al mes por equipo | 20 (≈ 5 scrims/semana) | Estimación; idealmente preguntar a 2–3 equipos o coaches | ±50 % |
| C_infra | Hosting | USD 0–7/mes | Plan del proveedor | — |
| C_pred | Costo por predicción | Medido (CPU ms × costo/h) | B11 | — |

### 5.4 Lo que el informe debe mostrar (B10, B12)

1. **Curva de valor vs θ**, una curva por modelo. El θ* óptimo **según valor**, que no es necesariamente el de mejor F1.
2. **Tabla E1 en dinero:** "M4 ahorra X min por partida = USD Y al mes por equipo frente a M0, con Z min extra de GPU de entrenamiento y W ms de inferencia".
3. **Valor vs tamaño de datos (E2):** el punto de rendimientos decrecientes.
4. **Tornado de sensibilidad:** cada parámetro a ±20 % y ±50 %. ¿Cuáles cambian la conclusión?
5. **Punto de equilibrio:** el mínimo de N_rev o C_h para que la herramienta valga la pena. Responde "¿lo pagaría alguien?".
6. **Costo de entrenamiento:** horas de GPU × costo equivalente en la nube (~USD 0.5/h), aunque ustedes usen su propia GPU.

---

## 6. Bloques detallados B1–B16

### B1 · A · Jue 5 · Entorno y API key (3 h) · Debe

- [x] **Hoy mismo**, solicitar la **Personal API Key** en developer.riotgames.com, registrando el producto como herramienta académica de análisis post-partida (registrada por Melisa).
- [~] Mismo Python en ambas PCs (3.12 o 3.13) dentro de un `.venv`. PC de A: Python 3.12.10, torch 2.6.0+cu124, numpy 2.5, pandas 3.1, scikit-learn 1.9, LightGBM 4.7. **Pendiente:** acordar versiones fijas con la PC de B en `requirements.txt`; `fastapi`, `uvicorn` y `captum` se instalan en B11/B12.
- [x] Verificar en la PC de A: `torch.cuda.is_available()` → `True` (RTX 4060 Laptop, 8 GB).
- [ ] Crear la estructura de carpetas de la [sección 10](#10-estructura-del-repo) y el esqueleto del README.
- [x] Descargar los datos que comparte B y comprobar los conteos (27,000 partidas).

**Terminado cuando:** las dos PCs entrenan en GPU y tienen los mismos datos.

### B2 · B · Jue 5 · Split y calidad de datos (3 h) · Debe · **COMPLETADO (sáb 7 nov)**

- [x] Comprimir `data/datos_merge/` y subirlo a un Drive compartido.
- [x] `src/split.py`: split por fecha 70/15/15 por partida → `data/splits.csv` + copia en `splits/splits.csv` (a git).
- [x] Tabla de tamaños por split × tier × región, y % de jugadores de test que están en train.
- [x] Detectores de calidad v0: frames incompletos, AFK y distribución de duración. Win rate de azul (esperado ~50–52 %).

**Terminado cuando:** `splits.csv` está congelado y subido. **Deja listo:** la lista de problemas de calidad para B3.

**Resultado** (detalle en `results/b2_resumen.md`; se reproduce con `python src/split.py` y `python src/quality_check.py`):
- Split temporal **estratificado por subconjunto**: train 18,912 · val 4,056 · test 4,032 (788/169/168 por subconjunto). Con un corte de fecha global, Master quedaba 51/23/26, porque sus jugadores juegan más seguido.
- 0 partidas con frames incompletos; gana azul 50.2 % (IC 49.6–50.8 %).
- AFK: 890 partidas (3.3 %) → columna `excluir_de_train` en `results/b2_calidad_partidas.csv`, **que debe usar B3**.
- 75 % de las partidas de test tiene ≥ 1 jugador que también está en train (~2 de 10). Es una limitación inevitable; se documenta.
- Las "kills de más" en eventos son ejecuciones (torre/súbditos), no errores.

### B3 · A · Vie 6 · Pipeline de features (6 h) · Debe · **COMPLETADO (sáb 7 nov)**

- [x] `src/features.py` con la función `match_to_features(frames, events, participants) → DataFrame por minuto`. La usan el entrenamiento y la app.
- [x] Todas las features de la sección 3.4, causales (eventos con `timestamp_ms ≤` timestamp del frame). Sin Atakhan (0 eventos en los datos).
- [x] Limpieza de la sección 3.2 (`afk` desde `excluir_de_train`; rol vacío → por orden de participante).
- [x] `scaler.json` ajustado solo con train (sin AFK).
- [x] `src/build_features.py`: procesa `datos_merge` por bloques → `team_minute.csv` + `sequences.npz` (~5 min).
- [x] Tests: causalidad (borrar eventos futuros no cambia el minuto t) y una partida revisada a mano (`tests/test_features.py`).

**Terminado cuando:** `sequences.npz` carga con las formas correctas y los tests pasan. El build tarda ~20–40 min, así que córranlo en la tarde.

### B4 · B · Vie 6 · EDA + supuestos de valor (5 h) · Debe · **COMPLETADO (sáb 7 nov)**

- [x] EDA como script reproducible `src/eda.py` (en vez de notebook; el resultado es el mismo cada vez):
  - Distribuciones por tier, región y parche; duración, rendiciones, nulos, outliers y AFK.
  - Relación de oro, kills, torres y dragones con el resultado en cada minuto (solo train, cálculo exploratorio sobre `frames.csv`).
- [x] Figuras en `results/figures/b4_*.png` + datos de cada figura en `results/b4_eda_data/` + reporte `results/b4_eda.md` (base de la sección 4.2).
- [x] `docs/supuestos_valor.md` + `configs/value.yaml`: escenario, función, tabla de parámetros con rangos y **fuentes abiertas y citadas**.

**Terminado cuando:** hay figuras + supuestos con fuentes. Si B3 todavía no termina, hagan el EDA con `frames.csv` crudo.

**Hallazgos clave** (detalle en `results/b4_eda.md`):
- **La diferencia de oro sola** da AUC 0.745 en el min 10, 0.816 en el 15 y 0.855 en el 20; su máximo es 0.887 en el min ~26 y luego baja (las partidas largas son parejas y deciden los objetivos). **Ese es el listón de M0.**
- La misma ventaja de oro pesa **más temprano**: +5.5k gana 96 % en el min 10 vs 86 % en el 20 → usar el **oro relativo** y el minuto como features (B3).
- 3.9 % de las partidas terminan a los 15–16 min (rendición temprana); solo 53 % llega al min 30 y 8 % al 40 → reportar métricas por tramo de minuto con su n.
- Remontadas desde −3k en el min 15: 12–15 % en todos los tiers, sin diferencia significativa.
- KR se rinde 35 % vs ~19–23 % en el resto; LAN tiene más AFK (5.8 %).
- 173 campeones, el menos jugado con 289 apariciones → embeddings pequeños son viables para M6.
- **Línea base del KPI:** revisar una partida ≈ 38 min → ~12.8 h ≈ USD 256/mes por equipo (valores base, por validar).
- **Pendiente para B10/B14:** medir f_rev, t_clip y t_setup; verificar a mano el precio de Mobalytics Plus (la página bloqueó el acceso automático).

### B5 · A · Sáb 7 · Baselines + evaluación (5 h) · Debe · **COMPLETADO (sáb 7 nov)**

- [x] `src/evaluate.py`:
  - AUC, log-loss, Brier, ECE (10 bins) y accuracy, globales y **por tramo de minuto** (5, 10, 15, 20, 25, 30+).
  - IC 95 % por bootstrap sobre partidas (1,000 réplicas) y curvas de calibración.
  - Escribe los resultados en `results/`. Además: **diferencias pareadas** vs el primer modelo (`results/b5_comparacion_vs_<ref>.csv`).
- [x] Entrenar M0, M1 y M2 (LightGBM con early stopping en val) y aplicarles calibración (`src/baselines.py`).
- [x] Pruebas de cordura de fuga (etiquetas barajadas → AUC 0.49; solo minuto → 0.50).
- [x] La regla R0 implementada como función que devuelve los minutos marcados (`rule_r0` en `src/baselines.py`).

**Resultado** (test; detalle en `results/b5_resumen.md`): M0 0.794, M1 0.810, M2 0.811 de AUC. M1/M2 superan a M0 de forma significativa (+0.016 pareado); **M2 ≈ M1** (sin diferencia global). Todos calibrados (ECE ≈ 0.02). Para evaluar la GRU: `python src/evaluate.py --run results/runs/m0_oro results/runs/m2_lgbm results/runs/<run> --name M0 M2 M4`.

**Noche N1:** tuning de LightGBM en CPU, si quieren.

### B6 · B · Sáb 7 · Entrenamiento DL: MLP y GRU (6 h) · Debe · **CÓDIGO LISTO, falta correr con datos reales (espera B3)**

- [x] `docs/contrato_features.md`: formato exacto de `features/sequences.npz` que **B3 debe generar**.
- [x] `src/datasets.py` (lee y **valida** `sequences.npz` contra el contrato, filtra por split/AFK/tier/región/tamaño, batches recortados).
- [x] `src/models/mlp.py` y `src/models/gru.py` (GRU **unidireccional**; B7 agrega TCN con `register('tcn', ...)` en `src/models/__init__.py`).
- [x] `src/train.py`:
  - Configs YAML en `configs/` (`gru_base.yaml`, `mlp_base.yaml`), semilla fija y la receta de la sección 2.
  - Guarda `best.pt`, `last.pt`, `metrics.csv`, `summary.json` y `predictions.npz` (val + test calibradas) en `results/runs/<run_id>/`.
  - Se puede **reanudar** (`--resume`) y tiene `--smoke`.
  - Temperature scaling al final y opción `train.smooth_lambda` lista para E6.
- [x] `src/hp_search.py` con el grid de 16 combinaciones (`configs/grid_gru.yaml`, `configs/grid_mlp.yaml`); al relanzarlo **salta las corridas terminadas**; `--seeds 42 43 44` para las finales.
- [x] Pruebas: `python -m pytest tests -q` → 11/11 (contrato, **causalidad de la GRU**, padding, aprendizaje, reanudar). Con datos sintéticos (`tests/make_fake_sequences.py`), una corrida de GRU tarda segundos en la RTX 4050.
- [ ] Smoke test con datos reales: `python src/train.py --config configs/gru_base.yaml --smoke` **(cuando exista `features/sequences.npz`)**.
- [ ] Corrida completa de GRU con datos reales. Debe superar a M0 (AUC del oro solo: 0.745 en el min 10, 0.816 en el 15); si no, hay un bug.
- [ ] **Subir a `main`** (sin commits todavía; ver avisos).

**Noche N1:** B corre la búsqueda de la GRU y A la del MLP (ver la [sección 7](#7-corridas-nocturnas-y-paralelismo)).

### B7 · A · Dom 8 · TCN + curva de aprendizaje (5 h) · Debería/Debe · **CASI COMPLETO (sáb 7 nov): falta la curva de M4**

- [~] Revisar con B los resultados de N1 y fijar la mejor configuración de GRU y de MLP. **MLP ✅** (búsqueda corrida por A: h64, 2 capas, dropout 0.3, lr 1e-3; `results/hp/mlp_hp.csv`). **GRU: falta la búsqueda de B.**
- [x] `src/models/tcn.py` (convoluciones causales dilatadas) + búsqueda de la TCN (adelantada de N2): h64, 3 bloques, dropout 0.3, lr 1e-3 (`results/hp/tcn_hp.csv`). M6 Transformer: no por ahora.
- [~] `src/learning_curve.py`: submuestras de train {1k, 3k, 6k, 12k, todo} con semilla fija; M2 y M4 con la mejor configuración y 3 semillas. **M2 ✅; M4 espera la mejor config de la GRU.**

**Resultado** (detalle en `results/b7_resumen.md`): las búsquedas tardan 3–5 min, no horas. Grids planos (±0.001 de log-loss). En test, TCN +0.002 de AUC vs M2 (significativo pero chico), MLP +0.001. M2 tiene rendimientos decrecientes claros: de 12k a 18k partidas gana solo +0.0008 de AUC.

**Noche N2:** A corre E2 + la búsqueda de TCN.

### B8 · B · Dom 8 · Detector de momentos + anotación (5 h) · Debe

- [ ] `src/moments.py` (ΔP, θ, w, top-K, unir detecciones cercanas).
- [ ] Herramienta de anotación (script simple o Streamlit): muestra **solo el log de eventos** por minuto y guarda en `annotations/<anotador>.csv` (match_id, minuto, tipo).
- [ ] `docs/protocolo_anotacion.md`: qué cuenta como momento decisivo, con 3 ejemplos.
- [ ] Elegir las 70 partidas (val+test, estratificadas) y las 15 compartidas. Anotar las primeras 10.

**Noche N2:** B corre GRU y MLP finales con 3 semillas + calibración.

### B9 · A · Lun 9 – Mar 10 · Anotación + E3/E4 (5 h) · Debe/Podría

- [ ] Anotar 35 partidas, incluidas las 15 compartidas.
- [ ] Configs de E3 y E4 (subconjuntos de train por tier y región, con tamaños iguales).
- [ ] Jugar o descargar 5–10 partidas propias **nuevas** con `collect.py --riot-id`, para la demo y como entradas fuera de entrenamiento.

**Noche N3:** A corre E3 + E4. **N4** (opcional): Transformer M6.

### B10 · B · Lun 9 – Mar 10 · Anotación + función de valor (6 h) · Debe

- [ ] Anotar 35 partidas (incluidas las 15 compartidas) y las partidas propias con la repetición, **cronometrando la revisión completa** (mide T_base y f_rev).
- [ ] `src/value_model.py` + `configs/value.yaml`:
  - Valor por partida y por mes.
  - Curva valor vs θ y elección de θ* con las anotadas de val.
  - Tornado ±20/±50 y punto de equilibrio.
- [ ] Acuerdo entre anotadores en las 15 compartidas: % de momentos que coinciden a ±2 min.

**Noche N3:** B corre TCN final con 3 semillas + E6.

### B11 · A · Mié 11 – Jue 12 · Backend e inferencia (6 h) · Debe

- [ ] Exportar el modelo ganador (state_dict en CPU o TorchScript) + `scaler.json` + temperatura + θ* a `models/` (fuera de git; subirlo a Drive o como Release).
- [ ] `app/api.py` (FastAPI):
  - `GET /player?riot_id=&platform=` → lista de partidas ranked recientes.
  - `GET /analyze?match_id=` → curva, momentos y explicaciones.
  - Reutiliza `riot_data/` y `src/features.py`.
- [ ] Explicación por momento (sección 8) y manejo de errores.
- [ ] Medir latencia (100 llamadas: p50/p95, separando API de Riot e inferencia) y agregar caché.

**Terminado cuando:** una partida propia nueva devuelve la curva y los momentos en < 3 s.

### B12 · B · Mié 11 – Jue 12 · Resultados de valor, sesgos y explicabilidad (6 h) · Debe

- [ ] Para cada modelo (R0, M0–M5): detector → P/R/F1 de momentos → valor al mes → **tabla maestra E1** (USD/mes, Δ vs M0, costo de entrenamiento e inferencia).
- [ ] Gráficas: valor vs modelo, valor vs θ y valor vs tamaño de datos (con E2).
- [ ] E5: AUC, ECE y F1 de momentos por tier, región, tramo de minuto y duración.
- [ ] Explicabilidad: SHAP global de LightGBM, ablación por grupos (o Integrated Gradients) para la GRU, y **3 casos de falla analizados**.

### B13 · A · Vie 13 – Sáb 14 · Frontend + despliegue (6 h) · Debe

- [ ] `app/static/index.html` + Chart.js:
  - Entrada por Riot ID o match ID.
  - Curva con banda de incertidumbre y marcadores de momentos.
  - Lista de momentos con su explicación.
- [ ] Desplegar (Render o Hugging Face Spaces, con la key como secreto).
- [ ] **Fallback offline:** 5 partidas precalculadas en caché por si la API falla durante la demo.
- [ ] Logging de predicciones + `docs/monitoreo.md`.

### B14 · B · Vie 13 – Sáb 14 · Usabilidad + redacción (5 h) · Debería

- [ ] Prueba con 4–6 personas que jueguen:
  - Cada una revisa una partida sin la herramienta y otra con ella.
  - Se mide el tiempo y si encontró los momentos clave.
  - Al final, una encuesta de 5 preguntas.
- [ ] Recalcular el valor con los tiempos medidos (f_rev, t_clip, t_setup).
- [ ] Borradores de las secciones 4.3 (valor), 4.5 (tabla de iteraciones) y 4.7 (explicabilidad y ética).

### B15 · A · Dom 15 · Escrito + README + video (5 h) · Debe

- [ ] Secciones 4.1, 4.4, 4.6 y 4.9.
- [ ] README de reproducción: comandos en orden, de `collect.py` a la app.
- [ ] Video .mp4 de 2–4 min (OBS): problema → demo con una partida nueva → resultado de valor.

### B16 · B · Dom 15 · Presentación + defensa (5 h) · Debe

- [ ] Slides para un pitch de 8–10 min: problema y usuario → por qué DL → modelo de valor → iteraciones → conclusión de negocio.
- [ ] Secciones 4.2 y 4.8 y el banco de preguntas (sección 12).
- [ ] **Ensayo conjunto, domingo en la noche (2 h):**
  - Cronometrar el pitch.
  - Demo con una partida jugada **ese mismo día**.
  - Cada quien hace preguntas sobre la parte del otro.

### Lun 16 – Mié 18 · Ambos

- [ ] Presentar en el turno asignado.
- [ ] Unir el escrito, revisarlo cruzado (cada quien lee lo que escribió el otro) y entregarlo el miércoles 18.

---

## 7. Corridas nocturnas y paralelismo

**Regla:** lo que tarde más de 30 min se lanza de noche (~22:00–23:00) y se revisa en la mañana. Durante el día, solo corridas cortas.

| Noche | A (GPU de A) | B (GPU de B) | Duración estimada | Qué revisar en la mañana |
|---|---|---|---|---|
| **N1** Sáb 7 → Dom 8 | Búsqueda del MLP (16 configs) + tuning de LightGBM (CPU) | Búsqueda de la GRU (16 configs) | 1–3 h | Mejor configuración por log-loss de val; curvas sin sobreajuste |
| **N2** Dom 8 → Lun 9 | E2 curva de aprendizaje (M2+M4 × 5 tamaños × 3 semillas) + búsqueda de TCN | GRU y MLP finales × 3 semillas + calibración | 2–4 h | ¿La curva se aplana? ¿Hay mucha variación entre semillas? |
| **N3** Lun 9 → Mar 10 | E3 + E4 | TCN final × 3 semillas + E6 (3 λ × 3 semillas) | 2–3 h | ¿E6 reduce los saltos espurios? |
| N4 Mar 10 → Mié 11 (opcional) | M6 Transformer + embeddings | — | 2–4 h | ¿Supera a M4? |

Las duraciones son estimaciones. Con la primera corrida completa de B6 midan cuánto tarda una época y ajusten.

**Antes de irse a dormir:**
1. **Smoke test** del mismo comando (1 época, 500 partidas, ~2 min). No dejen corriendo algo que falla a los 5 minutos.
2. **Desactivar la suspensión** y dejar la laptop conectada.
3. Cada corrida guarda `metrics.csv` por época y `best.pt`, y se puede reanudar.
4. Avisar en el chat qué quedó corriendo y con qué commit.

**Qué se puede hacer en paralelo:**
- **Las dos GPUs entrenan cosas distintas la misma noche** (tabla de arriba). Usan el mismo `splits.csv`, el mismo `sequences.npz` y el mismo commit.
- **LightGBM (CPU)** puede correr al mismo tiempo que la GPU en la misma máquina.
- **No construyan features y entrenen a la vez en la misma PC**, porque compiten por RAM.
- **Anotar es trabajo humano que no depende de la GPU:** se puede hacer mientras algo entrena de día.

---

## 8. App, despliegue, explicabilidad y monitoreo

### Arquitectura

```
Usuario (Riot ID o match ID)
        │
        ▼
 Frontend (HTML + Chart.js) ──► API FastAPI
                                   │
                                   ├─ riot_data.client  → match + timeline (API de Riot, rate limit, caché)
                                   ├─ riot_data.parse   → filas (mismas columnas que el entrenamiento)
                                   ├─ src.features      → features + scaler.json (MISMO código del entrenamiento)
                                   ├─ modelo (CPU)      → P(t) calibrada (temperatura)
                                   ├─ src.moments       → momentos con θ*
                                   └─ explicación       → qué grupo de features movió cada caída
                                   │
                                   ▼
                    JSON: curva + banda + momentos + explicación ──► gráfica y lista "ir al minuto"
```

- **Recomendación:** FastAPI + una página HTML, porque la API en sí ya cuenta como producto. **Plan B:** Streamlit si el tiempo aprieta.
- **Despliegue:** Render o Hugging Face Spaces, en el plan gratuito con CPU, y la API key como variable secreta.
- **Métricas operativas (sección 4.6)**
  - Latencia p50/p95, separando la API de Riot (domina, ~0.5–1.5 s) de la inferencia (esperado < 50 ms).
  - Throughput.
  - Costo por predicción = costo de hosting / predicciones al mes.

### Manejo de errores y casos límite

| Caso | Respuesta de la app |
|---|---|
| ID inválido o partida no encontrada | Mensaje claro y sugerencia de buscar por Riot ID |
| No es ranked Solo/Duo | Analiza, pero avisa: "el modelo se entrenó con ranked" |
| Remake o partida < 15 min | No analiza y explica por qué |
| Key expirada (401/403) o límite de tasa (429) | Mensaje + reintento; en la demo, usar el fallback de caché |
| Parche distinto al de entrenamiento | Aviso de posible degradación |

### Comunicar la incertidumbre

- Banda alrededor de la curva: dispersión entre las 3 semillas.
- Los minutos 0–5 se marcan como "baja certeza".
- Cada momento muestra el tamaño de su ΔP como nivel de confianza.

### Explicabilidad (sección 4.7)

- **Por momento (en la app):** ablación por grupos. Se recalcula P(t) reemplazando el cambio de un grupo de features (economía, combate, objetivos, visión) por su valor del minuto anterior, y el grupo que más reduce la caída es la explicación. Por ejemplo: "min 23: −18 pts, sobre todo por objetivos (barón del rival)". Es fácil de explicar en la defensa.
- **Global (en el informe):** SHAP de LightGBM e Integrated Gradients (Captum) de la GRU.
- **Casos de falla:** 3 partidas donde el modelo se equivocó mucho, con la explicación de por qué.

### Monitoreo (plan escrito en `docs/monitoreo.md`)

- Guardar cada predicción con su resultado real y calcular log-loss y ECE por semana.
- Alertar si el ECE supera un umbral o si las features cambian de distribución (PSI) con un parche nuevo.
- Validar el esquema de las respuestas de la API de Riot, por si cambian campos.
- Reentrenar cada 2 parches con `collect.py`.

---

## 9. Checklist contra la rúbrica

### Requisitos mínimos

| Requisito | Cómo lo cumplimos | Bloque |
|---|---|---|
| Implementación real de punta a punta | App: Riot ID → API → features → GRU → momentos → reporte web | B11, B13 |
| Deep learning en el núcleo | GRU entrenada por nosotros (+ MLP, TCN) | B6, B7 |
| Modelo de valor explícito y justificado | `value_model.py` + supuestos con fuentes + sensibilidad | B4, B10 |
| ≥ 2 versiones comparadas **en valor** | Tabla E1 en USD/mes con el costo de cómputo | B12 |
| Escenario real con restricciones | Coach amateur; límites de la API, latencia, privacidad (puuid con hash) | B4, B11 |
| Medible y reproducible | Split congelado en git, semillas, IC por bootstrap, README | B2, B5, B15 |

### Secciones del trabajo escrito (10 pts)

| Sección | Contenido principal | Material sale de | Redacta |
|---|---|---|---|
| 4.1 Contexto y problema | Usuario, costo actual, por qué DL, KPIs base | B4, B10 | A (B15) |
| 4.2 Datos y EDA | Fuente, volumen, EDA, limpieza, split, privacidad (política de Riot, hash) | B2, B4 | B (B16) |
| 4.3 Modelo de valor | Función, supuestos, curva, sensibilidad, θ* | B10, B12 | B (B14) |
| 4.4 Metodología | Arquitecturas, pérdida (y E6), optimizador, búsqueda, costo | B5–B7 | A (B15) |
| 4.5 Iteraciones | Tabla: cambio, métricas train/val/test, valor, costo; curvas | B5–B12 | B (B14) |
| 4.6 Despliegue y UX | Diagrama, latencia, usabilidad, errores, incertidumbre, monitoreo | B11, B13, B14 | A (B15) |
| 4.7 Explicabilidad y ética | SHAP/IG, sesgos (E5), usos indebidos, cuándo no usarlo | B12 | B (B14) |
| 4.8 Conclusiones | Qué movió el valor, rendimientos decrecientes, trade-offs | B12, E2, E6 | B (B16) |
| 4.9 Repositorio | Link + README de reproducción | B15 | A (B15) |

### Presentación (15 pts)

| Criterio | Qué preparar |
|---|---|
| Creatividad (3) | Datos propios de 4 regiones (no un dataset de Kaggle), momentos decisivos con explicación, partidas propias |
| Viabilidad (4) | Costos reales (hosting, inferencia), límites y política de la API, punto de equilibrio, competencia (op.gg, Mobalytics) y en qué nos diferenciamos |
| Vínculo métrica ↔ negocio (4) | Curva de valor vs θ, tabla E1 en USD, tornado, E6 |
| Dominio técnico (4) | Ensayo de preguntas, demo con partida nueva, explicar 3 fallas del modelo |

**Sobre uso indebido (4.7):** el sistema es **solo post-partida**. Usarlo en vivo, durante la partida, iría contra las políticas de Riot y no es posible con match-v5. No debe usarse para juzgar a jugadores individuales; mide equipos.

---

## 10. Estructura del repo

```
deep_proyecto4/
├── riot_data/            # (ya existe) cliente de la API, parseo, guardado
├── collect.py            # (ya existe) recolección
├── merge.py              # (ya existe) dataset balanceado
├── splits/splits.csv     # split congelado (SÍ va a git)
├── src/
│   ├── split.py
│   ├── features.py       # lo usan el entrenamiento Y la app
│   ├── build_features.py
│   ├── datasets.py
│   ├── models/           # baselines.py, mlp.py, gru.py, tcn.py, transformer.py
│   ├── train.py
│   ├── hp_search.py
│   ├── learning_curve.py
│   ├── evaluate.py
│   ├── calibrate.py
│   ├── moments.py
│   ├── value_model.py
│   └── explain.py
├── configs/              # YAML por experimento + value.yaml
├── tests/                # pytest + generador de datos sintéticos (python -m pytest tests -q)
├── app/                  # api.py + static/index.html
├── annotations/          # CSV de anotaciones (SÍ va a git)
├── results/              # métricas CSV y figuras (SÍ va a git; los modelos NO)
├── notebooks/            # solo EDA y análisis
├── docs/                 # esta guía, supuestos, protocolo, monitoreo, informe
└── data/, models/, features/   # NO van a git (Drive / Releases)
```

---

## 11. Riesgos, plan B y cola de reserva

| Riesgo | Señal | Plan B |
|---|---|---|
| La GRU no supera a LightGBM en AUC | Tabla E1 | Es un hallazgo válido. Comparar en **valor** (calibración, curvas más suaves → menos FP con E6). Si aun así pierde, presentar LightGBM + GRU y discutir el trade-off con honestidad |
| La anotación tarda demasiado | B9/B10 se alargan | Bajar a 40 partidas y agregar una métrica proxy automática ("último minuto en que el ganador cruzó 50 % y no volvió atrás") |
| La Personal API Key no llega a tiempo | Sin respuesta para el jueves 12 | Regenerar la dev key la mañana de la presentación + fallback de caché offline |
| Falla el despliegue | B13 | Demo local + video ya grabado |
| Un bloque se atrasa más de 1 día | Sync diaria | Recortar en este orden: M6 → E3/E4 → E7 → M5 → usabilidad completa (dejar una mini-prueba con 2 personas) |
| Falta RAM al construir features | Errores de memoria en B3 | Procesar por bloques más pequeños y no correr otra cosa a la vez |
| Fuga de información detectada tarde | AUC sospechosamente alto (> 0.9 en el minuto 10) | Pruebas de cordura de la sección 3.3 desde B5; revisar la lista de fugas |

**Cola de reserva** (para quien termine antes o espera algo):
- Figuras extra del EDA.
- Avanzar el README.
- Jugar o descargar partidas propias nuevas para la demo.
- Buscar fuentes de los parámetros de valor.
- Revisar la competencia y sus precios (sección de viabilidad).
- E7 (ablación de features).
- Preparar respuestas del banco de preguntas.

---

## 12. Banco de preguntas para la defensa

Ambos deben poder responder todas. Preparen la respuesta corta (30 s) de cada una.

1. ¿Por qué una GRU y no una LSTM o un Transformer? ¿Qué significa "causal"?
2. ¿Por qué log-loss como pérdida y no accuracy? ¿Qué es la calibración y por qué importa aquí?
3. ¿Cómo evitaron la fuga de información? ¿Por qué el split es por fecha?
4. ¿Por qué no basta con la diferencia de oro (M0)? ¿Cuánto mejora el modelo y dónde?
5. ¿Cómo eligieron el umbral θ? ¿Por qué no el de mejor F1?
6. ¿Qué pasa si sus supuestos de valor están 50 % equivocados?
7. ¿Cuándo deja de valer la pena mejorar el modelo (E2)?
8. ¿El modelo funciona peor en algún tier o región? ¿Por qué?
9. ¿Cuánto cuesta operar esto al mes? ¿Cuál es la latencia y qué la domina?
10. ¿Qué pasa con un parche nuevo? ¿Cómo lo detectarían?
11. Muestren un caso donde el modelo falla y expliquen por qué.
12. ¿Qué datos personales manejan y cómo los protegen? ¿Qué dice la política de Riot?
13. ¿En qué se diferencia de op.gg o Mobalytics? ¿Quién lo pagaría?
