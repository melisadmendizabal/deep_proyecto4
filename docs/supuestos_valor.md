# Supuestos del modelo de valor (B4)

*Versión 1 · 7 nov 2026. La implementan `src/value_model.py` y `configs/value.yaml` (B10). Los parámetros marcados **medir** se reemplazan por valores medidos en B10/B14.*

## 1. Escenario

**Usuario:** coach de un equipo amateur o universitario de LoL que revisa las partidas (scrims y ranked) de su equipo para encontrar errores.

**Hoy:** ve la grabación o la repetición completa de cada partida, pausando y retrocediendo, sin saber en qué minuto se decidió. Nada le dice dónde mirar.

**Con el sistema:** abre el reporte de la partida (curva de probabilidad + momentos decisivos), revisa solo los clips alrededor de cada momento detectado y salta el resto.

**KPI de negocio:** horas de revisión al mes por equipo y su costo en USD.

## 2. Función de valor

Por partida revisada:

```
T_base  = f_rev × D                                    # revisión completa (hoy)
T_tool  = t_setup + (TP + FP) × t_clip + FN × t_miss   # revisión con el sistema
Ahorro  = T_base − T_tool                              # minutos ahorrados
```

Por mes y por equipo:

```
Valor_mes = N_rev × (Ahorro / 60) × C_h  −  C_infra  −  N_rev × C_pred
```

- **TP:** momentos detectados que coinciden (±2 min) con uno anotado. Cada uno cuesta `t_clip` y *vale la pena*.
- **FP:** momentos detectados que no eran decisivos. Cuestan `t_clip` sin aportar nada.
- **FN:** momentos decisivos que el sistema no marcó. El coach los busca a mano o se los pierde, y eso cuesta `t_miss`.
- **El umbral θ del detector mueve TP, FP y FN.** Un θ bajo detecta todo (muchos FP), y un θ alto se pierde momentos (muchos FN). El θ óptimo es el que **maximiza Ahorro**, no el que maximiza F1.

## 3. Parámetros

| Parámetro | Qué es | Base | Rango (sensibilidad) | De dónde sale | Estado |
|---|---|---|---|---|---|
| D | Duración media de una partida | **29.5 min** | — | Nuestros datos: 27,000 partidas (`results/b4_eda.md`) | Dato |
| f_rev | Factor de revisión: minutos de revisión por minuto de partida (pausas, retrocesos, notas) | 1.3 | 1.0 – 2.0 | No hay estándar publicado; los anuncios de coaches hablan de ~1 h por partida revisada con el jugador | **Medir** (B10 con partidas propias, B14 con usuarios) |
| T_base | f_rev × D | 38 min | 30 – 59 min | Derivado | Derivado |
| t_setup | Abrir y leer el reporte | 2 min | 1 – 3 min | Estimación | **Medir** (B14) |
| t_clip | Revisar un momento (≈1 min antes y después + notas) | 2.5 min | 1.5 – 4 min | Estimación | **Medir** (B14) |
| t_miss | Costo de un momento decisivo no detectado (buscarlo a mano) | 8 min | 4 – 12 min | Estimación conservadora | Supuesto (sensibilidad ±50 %) |
| N_rev | Partidas revisadas al mes por equipo | 20 | 8 – 60 | Equipos universitarios: Ohio State practica 3 veces por semana, 3 h, partidas de ~40 min ([GBH/NPR, 2017](https://www.wgbh.org/news/2017-02-24/go-to-college-play-video-games-e-sports-make-a-play-for-the-big-ten)); el equipo A de McGill pide disponibilidad para 15 scrims por semana ([McGill Esports, 2024](https://mcgill-esports-association.vercel.app/teams/league-of-legends)). Eso son ~12–15 partidas por semana; suponemos que el coach revisa ~1 de cada 3 | Supuesto con fuente |
| C_h | Costo por hora del coach | USD 20 | 10 – 60 | Coaches amateur: USD 10–30/h; semi-pro: 30–60/h; pro: 60+/h ([WeCoach, 2023](https://wecoach.gg/blog/article/how-much-does-a-league-of-legends-coach-cost-in-2023)) | Fuente |
| C_infra | Hosting de la app | USD 0–7/mes | — | Plan gratuito o básico del proveedor (B13) | Confirmar en B13 |
| C_pred | Costo por predicción (CPU) | ≈ 0 | — | Medir latencia de inferencia en B11 | **Medir** (B11) |

## 4. Línea base (sin el sistema)

Con los valores base:

- Revisar una partida completa: 1.3 × 29.5 ≈ **38 min**.
- Al mes por equipo: 20 × 38 min ≈ **12.8 h**, que a USD 20/h son **≈ USD 256/mes** de tiempo del coach.

Esta es la línea base del KPI (sección 4.1 del informe).

## 5. Ejemplo ilustrativo (NO es un resultado)

Solo sirve para ver que la cuenta tiene sentido. Los números reales salen de B10/B12.

Supongamos 4 momentos decisivos por partida y un detector con recall 0.8 y precisión 0.7:
- TP = 3.2, detecciones = 3.2 / 0.7 ≈ 4.6, FP ≈ 1.4, FN = 0.8.
- T_tool = 2 + 4.6 × 2.5 + 0.8 × 8 ≈ **19.8 min** → ahorro ≈ **18 min por partida** (48 %).
- Al mes: 20 × 18 / 60 ≈ 6 h ≈ **USD 122/mes por equipo** a USD 20/h.

Lo que tiene que mostrar el informe es cómo cambia esto entre modelos (M0, M2, M4…) y con el umbral θ.

## 6. ¿Lo pagaría alguien? (viabilidad)

- **Comparación de precio:** la suscripción de Mobalytics Plus aparece en búsqueda a USD 4.99–7.99/mes. *No verificado: la página oficial (mobalytics.gg/lol/glp/plus) bloqueó el acceso automático. Confirmar a mano antes de citar.*
- **Argumento:** si el sistema ahorra ~6 h al mes de un coach de USD 20/h, el valor (~USD 120/mes) es más de 10 veces el precio de una suscripción de análisis típica. **Punto de equilibrio** con una suscripción de USD 8/mes: basta con ahorrar ~24 min al mes, es decir, menos de 2 partidas revisadas.
- **Riesgo de adopción:** el coach tiene que confiar en los momentos detectados. Por eso importan la explicación por momento y la calibración (secciones 4.6 y 4.7).

## 7. Análisis de sensibilidad que debe hacer B10

1. Cada parámetro a ±20 % y ±50 % (tornado), registrando cuánto cambia `Valor_mes` y si cambia **qué modelo gana**.
2. Punto de equilibrio en N_rev y en C_h.
3. Escenario pesimista (f_rev 1.0, t_clip 4, t_miss 12) y optimista (f_rev 2.0, t_clip 1.5, t_miss 4).
4. Separar lo que es **dato** (D), **fuente** (C_h, N_rev) y **estimación** (t_clip, t_miss) en la tabla del informe.

## Fuentes consultadas (abiertas el 7 nov 2026)

- [WeCoach – How Much Does a League of Legends Coach Cost in 2023?](https://wecoach.gg/blog/article/how-much-does-a-league-of-legends-coach-cost-in-2023): tarifas por hora por nivel de coach.
- [GBH/NPR (2017) – Go To College, Play Video Games: E-Sports Make A Play For The Big Ten](https://www.wgbh.org/news/2017-02-24/go-to-college-play-video-games-e-sports-make-a-play-for-the-big-ten): frecuencia de práctica de Ohio State y duración de partidas.
- [McGill Esports Association – League of Legends (archivo 2024)](https://mcgill-esports-association.vercel.app/teams/league-of-legends): scrims por semana del equipo A.
- Mobalytics Plus (mobalytics.gg/lol/glp/plus): **no se pudo abrir**; precio pendiente de verificar.
