# B2 - Split y calidad de datos

*Generado por `src/split.py` y `src/quality_check.py`. No editar a mano.*

## Dataset

- Partidas: **27,000** (24 subconjuntos de plataforma × tier, 1,125 cada uno).
- Gana el lado azul: 50.2 % (IC 95 %: 49.6–50.8 %). Las clases están balanceadas, así que no hace falta reponderar.
- Terminan por rendición: 23.8 %.
- Duración (min): media 29.5, mediana 29.4, p5 17.4, p95 40.5, máx 61.3.
- Parches: 16.21 (19,516), 16.20 (5,075), 16.22 (1,991), 16.19 (418).

## Split (temporal, estratificado por subconjunto)

Dentro de cada plataforma × tier: 70 % más antiguo → train, 15 % → val, 15 % más reciente → test. Congelado en `splits/splits.csv`.

Totales: train 18,912 · val 4,056 · test 4,032. Por subconjunto: 788 / 169 / 168 (tabla completa en `results/b2_split_tamanos.csv`).

### Posible fuga por jugador

Partidas de val/test donde aparece al menos un jugador que también está en train. No se puede evitar del todo, porque los jugadores juegan muchas partidas; se reporta como limitación.

| split | partidas | jugadores_unicos | pct_jugadores_vistos_en_train | pct_partidas_con_>=1_jugador_de_train | promedio_jugadores_de_train_por_partida |
|---|---|---|---|---|---|
| val | 4056 | 35319 | 20.8 | 81.1 | 2.38 |
| test | 4032 | 34879 | 19.3 | 75.2 | 2.16 |

## Banderas de calidad

| bandera | partidas | % del total |
|---|---|---|
| flag_frames_incompletos | 0 | 0.00 |
| flag_afk | 890 | 3.30 |
| flag_kills_inconsistentes | 2 | 0.01 |
| flag_sin_game_end | 0 | 0.00 |
| excluir_de_train | 890 | 3.30 |

- **AFK:** un jugador ≥ 2 min seguidos sin ganar XP *y* sin moverse, desde el minuto 4. Rendición en partidas con AFK: 39.2 % vs 23.3 % sin AFK (validación del detector). Solo "sin XP ≥ 3 min" marcaba ~11 % de las partidas sin más rendiciones de lo normal: eran jugadores que morían mucho, no AFK.
- **Ejecuciones** (muertes por torre/súbditos/monstruos, sin jugador asesino): 5,046 en total, en 16.7 % de las partidas. Explican que haya más eventos CHAMPION_KILL que kills en participants; no es un error.
- **excluir_de_train** = frames incompletos o AFK. Recomendación para B3: excluirlas de train y reportar val/test con y sin ellas.
- Kills inconsistentes (kills de jugadores en eventos ≠ participants) y la falta de GAME_END son solo informativas.

Por split (% de partidas):

| split | flag_afk | excluir_de_train |
|---|---|---|
| train | 3.27 | 3.27 |
| val | 3.82 | 3.82 |
| test | 2.9 | 2.9 |

## Por tier

| tier | partidas | pct_afk | duracion_media_min | pct_rendicion | pct_gana_azul |
|---|---|---|---|---|---|
| DIAMOND | 4500 | 2.5 | 28.6 | 23.1 | 50.9 |
| EMERALD | 4500 | 2.5 | 29.6 | 22.3 | 49.6 |
| GOLD | 4500 | 4.3 | 30.4 | 22.7 | 51.4 |
| MASTER | 4500 | 1.7 | 27.4 | 28.8 | 49.2 |
| PLATINUM | 4500 | 3.4 | 30.1 | 22.0 | 49.3 |
| SILVER | 4500 | 5.4 | 30.7 | 24.0 | 50.8 |

## Por región

| platform | partidas | pct_afk | duracion_media_min | pct_rendicion | pct_gana_azul |
|---|---|---|---|---|---|
| euw1 | 6750 | 2.8 | 29.7 | 18.8 | 49.9 |
| kr | 6750 | 1.6 | 27.9 | 35.5 | 50.6 |
| la1 | 6750 | 5.8 | 30.6 | 18.5 | 50.8 |
| na1 | 6750 | 3.0 | 29.7 | 22.6 | 49.6 |
