# B12 · Tabla maestra E1: valor y costo de cada modelo

*`python src/b12_resultados.py --solo costo`. Valor: test (28 partidas anotadas), θ* elegido en val (`results/mejoras_valor.csv`; M1 y M3 de `results/b10_valor.md`). AUC: test, 4,032 partidas. Entrenamiento: suma de `train_seconds` de `results/runs/*/summary.json`. Inferencia: medida aquí en CPU con el motor de la app (`app/inference.py`), mediana de 28 partidas de test; *modelo* = solo la red (todas las semillas que usa la app), *análisis* = features + modelo + momentos + explicación.*

| Modelo | AUC test | Valor USD/mes [IC 95 %] | F1 momentos | Clips/partida | Parámetros | Entrenamiento final | Búsqueda de HP | Inferencia modelo | Análisis completo |
|---|---|---|---|---|---|---|---|---|---|
| R0 regla | — | 148 [127, 168] | 0.57 | 3.9 | — | 0 s | — | — | — |
| M0 logística (oro) | 0.794 | 137 [107, 165] | 0.47 | 2.2 | 3 | 0.1 s (CPU) | — | 0.1 ms | 13 ms |
| M1 logística | 0.810 | 123 [92, 154] | 0.31 | 1.5 | 34 | 0.8 s (CPU) | — | 0.1 ms | 12 ms |
| M2 LightGBM | 0.811 | 123 [93, 152] | 0.32 | 1.6 | 139 árboles | 4.2 s (CPU) | — | 0.2 ms | 19 ms |
| M3 MLP | 0.812 | 124 [94, 153] | 0.36 | 2.1 | 6,529 | 19 s (3 sem.) | 2.9 min (16 corridas) | 1.0 ms ×3 | 17 ms |
| M4 GRU | 0.812 | 132 [99, 163] | 0.41 | 1.9 | 210,817 | 37 s (3 sem.) | 6.7 min (16 corridas) | 13.3 ms ×3 | 50 ms |
| M5 TCN | 0.813 | 128 [100, 154] | 0.44 | 3.3 | 78,401 | 64 s (3 sem.) | 4.8 min (16 corridas) | 4.9 ms ×3 | 36 ms |
| M6 Transformer | 0.812 | 129 [100, 158] | 0.41 | 2.4 | 75,457 | 83 s (3 sem.) | 7.3 min (16 corridas) | — | — |
| M4 + campeones | 0.819 | 130 [97, 161] | 0.35 | 1.1 | 233,473 | 41 s (3 sem.) | 3.3 min (12 corridas) | 13.9 ms ×3 | 85 ms |
| M6 + campeones (app) | 0.821 | 129 [102, 156] | 0.46 | 3.6 | 94,977 | 93 s (3 sem.) | la de M6 | 4.7 ms ×3 | 57 ms |

- **Costo de entrenamiento:** todo el cómputo de entrenamiento de la tabla suma ~31 min de GPU de laptop (NVIDIA GeForce RTX 4050 Laptop GPU, NVIDIA GeForce RTX 4060 Laptop GPU). A precios de GPU en la nube es del orden de centavos de dólar: el costo real del proyecto es el tiempo de las personas (anotar, revisar), no el cómputo.
- **Costo de inferencia:** todos los modelos analizan una partida en milisegundos en CPU; en la app la llamada a la API de Riot (~1.2 s, `results/b11_latencia.md`) domina. Con el plan gratis de Render (512 MB) `C_infra` = USD 0 y `C_pred` ≈ 0, así que el costo no cambia qué modelo conviene.
- **Conclusión E1:** el valor queda en USD 123–148/mes para todos los detectores y ninguna diferencia entre modelos de ML es grande; como el costo es ~0 para todos, el modelo se elige por lo que da al usuario además de los clips (curva calibrada, incertidumbre, explicación).

