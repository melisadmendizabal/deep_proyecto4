# Laboratorio 10 · Explainable AI sobre Fight Diff

CC3092 Deep Learning y Sistemas Inteligentes · Melisa Mendizabal (23778) y Renato Rojas (23813).

Explica el modelo final del proyecto, el mismo que usa la app: **M6 Transformer causal + campeones**, promedio de las semillas 42, 43 y 44 recalibrado con temperatura (`models/m6_draft/`). **No se reentrena nada.** El método es **Integrated Gradients (IG)**, implementado a mano en PyTorch (unas 15 líneas; no hace falta `captum`). Se aplica sobre el logit de la curva P(gana azul) que muestra la app. Por qué IG y no la oclusión o LIME: ver el informe del laboratorio (partes 1 y 2).

## Cómo ejecutarlo

Desde la raíz del repo, con el entorno del proyecto ya instalado (ver el [README general](../README.md)):

```bash
python lab10_xai/costo_metodos.py   # parte 2: costo de IG vs oclusión y error de completitud (~10 s)
python lab10_xai/explicar.py        # parte 3: explicaciones locales, globales, momentos y sensibilidad (4-11 min en CPU según la PC)
```

Prueba rápida (~2 min), que escribe en `lab10_xai/prueba/` para no pisar las figuras ni las tablas del informe:

```bash
python lab10_xai/explicar.py --n-global 100 --n-sens 30
```

**Qué datos necesita** (no van a git; se generan con los pasos del README general):

| Dato | Lo usa | Cómo se genera |
|---|---|---|
| `models/m6_draft/` | los dos scripts | sí está en git (paso 10 del README general para regenerarlo) |
| `features/sequences.npz` | `explicar.py` | paso 3 del README general (`src/build_features.py`) |
| `annotations/*.csv` | `explicar.py` (momentos E y F, tabla de momentos tempranos) | sí está en git |
| `data/anotacion/` (JSON crudos de las partidas anotadas) | `costo_metodos.py`; `explicar.py` solo para listar los eventos de los momentos E y F | `src/select_annotation.py` (paso de anotación). Sin esa carpeta, `explicar.py` sale igual, pero sin la lista de eventos |

## Qué genera

| Archivo | Contenido |
|---|---|
| `figuras/fig1_locales.png` | 6 instancias de test: acierto y error con alta confianza, caso cerca de 0.5, partida con AFK, momento verdadero y momento falso |
| `figuras/fig2_global.png` | % del aporte por grupo (minutos 10-25, vs TreeSHAP de M2), top 15 features y cuánto pesa la historia |
| `figuras/fig3_momentos.png` | IG vs la oclusión que usa hoy la app, sobre todos los momentos de la muestra |
| `resultados/resumen_parte3.md` | todas las tablas de la parte 3 en Markdown (se regenera; no editar a mano) |
| `resultados/locales*.csv` | las 6 instancias locales (resumen y atribución de cada feature) |
| `resultados/global_*.csv` | por grupo, por feature, por minuto de la historia y por subgrupo (parche, tier, AFK, aciertos/errores) |
| `resultados/momentos_*.csv` | IG vs oclusión en cada momento, grupos, feature principal y momentos tempranos anotados |
| `resultados/sensibilidad.csv` | robustez: una sola semilla, otra línea base, 16 y 64 pasos |
| `resultados/tiempos.csv` | costo de IG y de la oclusión por instancia |
| `resultados/costo_metodos*.csv` | parte 2: costo de cada método candidato y error de completitud |

## Parámetros de diseño

Están como constantes al inicio de `explicar.py`.

| Parámetro | Valor | Por qué |
|---|---|---|
| Salida explicada | logit de la curva de la app (promedio de 3 semillas, recalibrado) en el minuto *t* | Es lo que ve el coach. El logit no se satura cerca de 0 y 1; para mostrarlo se reparte en puntos de probabilidad de forma proporcional. |
| Línea base "partida pareja" (instancias A-D y global) | todas las diferencias en 0, **minuto real**, los 10 campeones = id 0 | Responde *"¿por qué P está donde está?"* contra un juego empatado en el mismo minuto. El id 0 es el "campeón desconocido" que el modelo vio en entrenamiento (*champion dropout*). |
| Línea base "partida congelada" (momentos E-F y global de momentos) | minutos *start+1…end* = estado del minuto *start* (menos el minuto) | Responde *"¿qué cambió en el salto?"*; es el mismo contrafactual que la oclusión de la app, así que las dos se comparan directamente. |
| Entradas integradas | 35 features × minutos 0…*t* y los embeddings de campeones de las 3 semillas, en un solo camino | La completitud cubre también el draft: Σ features + Σ campeones = logit(x) − logit(x′). |
| Pasos `STEPS` | 32 (regla del punto medio) | Error de completitud ≤ 4·10⁻⁴ logit; con 16 o 64 pasos los grupos no cambian (`sensibilidad.csv`). |
| Instancias locales | 6 de test, sorteadas con semilla 42 dentro de cada criterio | Acierto \|P − 0.5\| ≥ 0.40, error ≥ 0.30, umbral ≤ 0.02 (minuto 15, sin AFK); una con AFK; dos momentos de las 28 partidas anotadas (uno anotado por una persona, otro no). |
| Muestra global `N_GLOBAL` | 1,000 de las 4,032 partidas de test, minutos 10, 15, 20 y 25 | El minuto 15 lo tienen todas las partidas y el 25, el 81 %. |
| Conjunto de fondo | no aplica | IG usa una línea base, no un conjunto de fondo; la sensibilidad a la línea base se mide aparte. |

## Reproducibilidad

- **Semilla fija:** la muestra global y las instancias salen de `np.random.default_rng(42)`. La red está en modo `eval` (sin dropout ni *champion dropout*), así que IG es determinista.
- **Verificado:** al volver a correr `explicar.py` desde cero salen idénticos todos los CSV de resultados, salvo los tiempos (`tiempos.csv`, la columna `ms` de `locales.csv` y el tiempo total del resumen), que dependen de la carga de la PC.
- **Control interno:** antes de explicar, el script comprueba que su versión diferenciable de la curva coincide con la de la app (`app/inference.py`) con error < 10⁻⁴. Si alguien cambia el modelo y no coincide, se detiene.
- **Tiempos:** varían entre corridas, sobre todo en laptops con núcleos P y E. Para medir como en la tabla E1 del proyecto (un hilo y núcleos P), ver el encabezado de `costo_metodos.py`.
