# B11 · Latencia de la app

*`python app/bench_latency.py`. 100 partidas de test distintas (desde el JSON crudo), CPU (Intel64 Family 6 Model 191 Stepping 2, GenuineIntel, 4 hilos), modelo M4 con 3 semillas. 2026-11-12.*

| Etapa | p50 (ms) | p95 (ms) | máx (ms) |
|---|---|---|---|
| Parse del JSON de Riot | 7.4 | 14.8 | 48.8 |
| Features | 11.5 | 20.3 | 45.3 |
| Modelo (3 semillas) | 16.7 | 27.4 | 36.9 |
| Explicación de momentos | 8.8 | 49.7 | 70.0 |
| **Total sin Riot** | 46.3 | 105.8 | 133.3 |
| API de Riot (match + timeline) | 914.3 | 3494.3 | 4302.4 |

- Partidas de 17–44 minutos; 1.6 momentos explicados por partida en promedio.
- La explicación cuesta 1 pasada extra del modelo por momento (5 variantes en un batch).
- La API de Riot domina: es el 95 % de la latencia total típica.
- Con caché (la misma partida otra vez) la respuesta es casi instantánea: no se llama a Riot ni al modelo.
- **Meta de B11:** < 3 s por partida nueva. Inferencia p95 = 105.8 ms.
