# B10 · Modelo de valor: resultados

*Generado por `python src/value_model.py `. Se vuelve a correr cuando haya más anotaciones.*

## Datos

- Partidas anotadas con predicciones: **18 de val** (para elegir θ*) y **17 de test** (para reportar). Anotadores: B 35. Compartidas (A y B): 0.
- Momentos anotados por partida: 2.8; partidas sin momentos: 9 %. Confianza mínima usada: 1.
- Duración media de esas partidas: 30.1 min → revisión completa T_base = 1.3 × 30.1 = **39 min**.
- Tiempo medio de anotar una partida con la herramienta (solo el log): 9.6 min.
- Parámetros: f_rev 1.3, t_setup 2.0, t_clip 2.5, t_miss 8.0, N_rev 20, C_h USD 20 (`configs/value.yaml`).

> **Resultados preliminares:** con menos de ~20 partidas por split los IC son muy anchos y θ* puede cambiar. Volver a correr con las 35 + 35 anotaciones.

## Tabla principal (test, θ* elegido en val)

| Modelo | θ* | Clips/partida | Precisión | Recall | F1 | Ahorro (min/partida) | Ahorro % | Valor USD/mes [IC 95 %] | Δ vs M0 [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|
| R0 | — | 3.9 | 0.45 | 0.60 | 0.52 | 18.0 | 46 % | 120 [88, 153] | +2 [-17, +18] |
| M0 | 0.09 | 4.4 | 0.43 | 0.64 | 0.52 | 17.7 | 45 % | 118 [86, 152] | — |
| M1 | 0.05 | 4.9 | 0.33 | 0.54 | 0.41 | 14.1 | 36 % | 94 [57, 134] | -25 [-38, -10] |
| M2 | 0.04 | 4.9 | 0.32 | 0.54 | 0.40 | 13.9 | 36 % | 93 [56, 132] | -25 [-43, -8] |
| M3 | 0.18 | 3.3 | 0.38 | 0.42 | 0.40 | 15.2 | 39 % | 101 [63, 140] | -17 [-32, -2] |
| **M4** | 0.12 | 4.0 | 0.38 | 0.52 | 0.44 | 15.8 | 40 % | 105 [68, 144] | -13 [-28, +3] |
| M5 | 0.13 | 3.8 | 0.37 | 0.48 | 0.42 | 15.3 | 39 % | 102 [65, 140] | -16 [-32, -1] |
| M4 E6 λ0.1 | 0.12 | 4.0 | 0.38 | 0.52 | 0.44 | 15.8 | 40 % | 105 [68, 144] | -13 [-28, +3] |
| M4 E6 λ0.5 | 0.13 | 3.6 | 0.36 | 0.44 | 0.40 | 15.0 | 38 % | 100 [61, 139] | -19 [-35, -2] |
| Oráculo | — | 2.9 | 1.00 | 1.00 | 1.00 | 29.7 | 76 % | 198 [165, 235] | +80 [+61, +101] |

*Oráculo* = detector perfecto (marca exactamente lo anotado): es el techo del ahorro con estos parámetros. R0 = regla sin ML (barón, ace, inhibidor o swing de oro > 1,500).

![valor vs theta](figures/b10_valor_vs_theta.png)

## Sensibilidad (M4, θ* re-elegido en cada caso)

| Parámetro | Valor bajo → USD/mes | Valor alto → USD/mes | ¿Cambia el ganador? |
|---|---|---|---|
| f_rev (factor de revisión) | 1 → 45 | 2 → 246 | no |
| t_setup (min) | 1 → 112 | 3 → 99 | no |
| t_clip (min por clip) | 1.5 → 132 | 4 → 86 | sí: M0, M4 E6 λ0.5 |
| t_miss (min por momento perdido) | 4 → 166 | 12 → 68 | sí: M0, M4 E6 λ0.5 |
| N_rev (partidas al mes) | 8 → 42 | 60 → 316 | no |
| C_h (USD/h del coach) | 10 → 53 | 60 → 316 | no |

Ganador con los valores base: **R0**. Escenario pesimista (f_rev 1.0, t_clip 4, t_miss 12): M4 -34 USD/mes, gana R0. Optimista (f_rev 2.0, t_clip 1.5, t_miss 4): 310 USD/mes, gana R0. Los ±20 % y ±50 % de cada parámetro están en `results/b10_tornado.csv`.

![tornado](figures/b10_tornado.png)

## Punto de equilibrio (M4)

- El sistema ahorra tiempo mientras **f_rev > 0.77** (revisar la partida completa toma más que 0.77 min por minuto de juego) y mientras **t_clip < 6.5 min** por clip.
- Para pagar una suscripción de USD 8/mes basta con revisar **1.5 partidas al mes** (con C_h = 20), o un coach de **USD 1.52/h** (con N_rev = 20).
