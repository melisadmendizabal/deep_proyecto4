# B11 · Latencia de la app

*`python app/bench_latency.py`. 100 partidas de test distintas (desde el JSON crudo), CPU (Intel64 Family 6 Model 191 Stepping 2, GenuineIntel, 4 hilos), modelo M4 con 3 semillas. 2026-11-12.*

| Etapa | p50 (ms) | p95 (ms) | máx (ms) |
|---|---|---|---|
| Parse del JSON de Riot | 8.2 | 13.8 | 45.6 |
| Features | 11.7 | 21.3 | 39.3 |
| Modelo (3 semillas) | 15.6 | 22.4 | 33.0 |
| Explicación de momentos | 8.4 | 43.6 | 60.8 |
| **Total sin Riot** | 48.4 | 86.9 | 126.9 |

- Partidas de 17–44 minutos; 1.6 momentos explicados por partida en promedio.
- La explicación cuesta 1 pasada extra del modelo por momento (5 variantes en un batch).
- **Falta medir la API de Riot** (`--riot 10`, necesita `.env`). La guía estima 0.5–1.5 s.
- Con caché (la misma partida otra vez) la respuesta es casi instantánea: no se llama a Riot ni al modelo.
- **Meta de B11:** < 3 s por partida nueva. Inferencia p95 = 86.9 ms.
