# Contrato de datos: `features/sequences.npz`

Este archivo es el **puente entre B3 (features, compañero) y B6/B7 (modelos)**. B3 lo genera; `src/datasets.py` lo lee. Mientras ambos respeten este formato, pueden trabajar en paralelo sin pisarse.

Para probar sin datos reales: `python tests/make_fake_sequences.py` crea un archivo falso con exactamente este formato en `features/sequences_fake.npz`.

## Arreglos obligatorios

Se guarda con `np.savez_compressed(path, **arrays)`. Sin `object` arrays: los textos van como `np.str_`, y así se carga con `allow_pickle=False`.

| Clave | Forma | dtype | Contenido |
|---|---|---|---|
| `X` | `[N, T, F]` | `float32` | Features **ya normalizadas** (scaler ajustado solo con train). `T = 60` minutos. Los minutos que no existen (padding) van en `0.0` |
| `mask` | `[N, T]` | `bool` | `True` en los minutos que existen: del minuto 0 al último frame, truncado a 60 |
| `lengths` | `[N]` | `int16` | Minutos válidos por partida (= `mask.sum(1)`) |
| `y` | `[N]` | `int8` | 1 si gana el lado **azul**, 0 si gana rojo |
| `match_id` | `[N]` | `str` | ID de la partida |
| `split` | `[N]` | `str` | `train`, `val` o `test`, copiado de `splits/splits.csv` (nunca recalculado) |
| `platform` | `[N]` | `str` | `la1`, `na1`, `euw1`, `kr` |
| `tier` | `[N]` | `str` | `SILVER` … `MASTER` |
| `patch` | `[N]` | `str` | `16.21`, etc. |
| `afk` | `[N]` | `bool` | Columna `excluir_de_train` de `results/b2_calidad_partidas.csv` |
| `feature_names` | `[F]` | `str` | Nombre de cada feature, en el orden de `X` |
| `feature_groups` | `[F]` | `str` | Grupo de cada feature: `economia`, `rol`, `combate`, `objetivos`, `vision`, `tendencia`, `contexto`. Se usa para la ablación (E7) y para explicar momentos en la app |

## Opcionales

| Clave | Forma | dtype | Contenido |
|---|---|---|---|
| `champions` | `[N, 10]` | `int16` | `champion_id` de los participantes 1–10 (1–5 azul, 6–10 rojo). Solo para M6 |

## Reglas

1. **Fila `t` = estado en el frame del minuto `t`** (frame `0` = inicio). Solo puede usar eventos con `timestamp_ms ≤` el timestamp de ese frame (causalidad).
2. **Perspectiva azul:** toda diferencia es azul − rojo. Positivo = ventaja azul.
3. **Debe existir una feature llamada `minute`** (normalizada), porque los modelos snapshot (M0–M3) la necesitan.
4. **Las features de tendencia** (Δ de 1 y 3 min) van en el grupo `tendencia`. La GRU puede ignorarlas (`drop_groups: [tendencia]` en su config) porque ya ve la historia.
5. **No incluir nada de la lista de fugas** de la guía (sección 3.3).
6. Las partidas AFK **se incluyen** en el archivo con `afk = True`. El que entrena decide si las excluye (por defecto se excluyen de train y se conservan en val/test).
7. Junto al `.npz` va `features/scaler.json` con `{"feature_names": [...], "mean": [...], "std": [...]}` para que la app normalice igual.

## Cómo lo lee B6

```python
from src.datasets import load_sequences
data = load_sequences("features/sequences.npz")            # valida el contrato
train = data.subset(split="train", exclude_afk=True)        # filtros para experimentos
val = data.subset(split="val")
```

`load_sequences` revisa formas, dtypes y claves, y falla con un mensaje claro si algo no cumple el contrato.
