"""
Construye el dataset final: misma cantidad de partidas por subconjunto (plataforma x tier),
elegidas al azar y en orden aleatorio dentro de los archivos.

Uso:
  python merge.py                       # 1125 por subconjunto (MERGE_PER_SUBSET en config.py)
  python merge.py --per-subset 1200     # agrega partidas NUEVAS hasta llegar a 1200 por clase
  python merge.py --fresh               # ignora la selección anterior y elige todo de nuevo

Resultado en data/datos_merge/:
  matches.csv, participants.csv, frames.csv, events.csv   (mismas columnas que data/csv/)
  ids_usados_merge.csv   <- qué partidas están en el dataset, en qué orden y desde cuándo

Cómo se ordenan los datos:
  - Las PARTIDAS van en orden aleatorio (mezcladas entre regiones y tiers).
  - Las filas de UNA partida quedan juntas y en orden de tiempo (minuto, jugador), porque
    son una serie de tiempo: separarlas o desordenarlas rompería la secuencia.

ids_usados_merge.csv funciona como memoria: si lo vuelven a correr con un número mayor,
conserva todas las partidas ya usadas y solo agrega nuevas (columna `ronda`), así siempre
saben cuáles pudo haber visto un modelo entrenado antes.

Para no llenar la RAM, los archivos grandes se procesan por bloques en vez de cargarse
completos (los eventos de 27k partidas son ~13 millones de filas).
"""
import argparse
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from riot_data import config
from riot_data.parse import TABLE_FIELDS

IDS_FILE = "ids_usados_merge.csv"

# Cómo ordenar las filas dentro de cada partida, y qué define un duplicado.
SORT_KEYS = {
    "matches": [],
    "participants": ["participant_id"],
    "frames": ["minute", "participant_id"],
    "events": ["timestamp_ms"],
}
DEDUP_KEYS = {
    "matches": ["match_id"],
    "participants": ["match_id", "participant_id"],
    "frames": ["match_id", "minute", "participant_id"],
    "events": None,  # fila idéntica
}


def find_subsets() -> list[Path]:
    """Carpetas plataforma_tier con datos (sin carpetas internas '_...' ni partidas propias)."""
    if not config.CSV_DIR.exists():
        return []
    return [d for d in sorted(config.CSV_DIR.iterdir())
            if d.is_dir() and not d.name.startswith("_") and not d.name.endswith("_own")
            and (d / "matches.csv").exists()]


def load_available(subsets: list[Path]) -> pd.DataFrame:
    cols = ["match_id", "platform", "tier", "patch"]
    df = pd.concat([pd.read_csv(s / "matches.csv", usecols=cols, dtype=str) for s in subsets],
                   ignore_index=True)
    return df.drop_duplicates("match_id")


def select_ids(available: pd.DataFrame, previous: pd.DataFrame | None,
               per_subset: int, seed: int) -> pd.DataFrame:
    """Elige per_subset partidas por (plataforma, tier), conservando las ya usadas."""
    rng = np.random.default_rng(seed)
    prev_ids = set(previous["match_id"]) if previous is not None else set()
    prev_info = previous.set_index("match_id") if previous is not None else None
    if previous is not None:
        previous = previous.assign(ronda=previous["ronda"].astype(float).astype(int))
    next_round = int(previous["ronda"].max()) + 1 if previous is not None else 1
    today = date.today().isoformat()

    missing = prev_ids - set(available["match_id"])
    if missing:
        print(f"  aviso: {len(missing)} partidas de {IDS_FILE} ya no están en data/csv; "
              "se omiten")

    chosen, report = [], []
    for (platform, tier), group in available.groupby(["platform", "tier"], sort=True):
        kept = group[group["match_id"].isin(prev_ids)].copy()
        candidates = group[~group["match_id"].isin(prev_ids)].sort_values("match_id")
        need = max(0, per_subset - len(kept))
        take = candidates.iloc[rng.permutation(len(candidates))[:need]].copy()

        kept["ronda"] = kept["match_id"].map(prev_info["ronda"]) if len(kept) else []
        kept["fecha_agregado"] = (kept["match_id"].map(prev_info["fecha_agregado"])
                                  if len(kept) else [])
        take["ronda"] = next_round
        take["fecha_agregado"] = today
        chosen += [kept, take]

        total = len(kept) + len(take)
        note = ""
        if total < per_subset:
            note = f"  <- solo hay {total} disponibles"
        elif len(kept) > per_subset:
            note = f"  <- ya tenía {len(kept)} de antes (no se quitan)"
        report.append((f"{platform}_{tier.lower()}", len(kept), len(take),
                       len(candidates) - len(take), note))

    print(f"\n  {'subconjunto':<18}{'ya usadas':>10}{'nuevas':>8}{'reserva':>9}")
    for name, n_kept, n_new, n_reserve, note in report:
        print(f"  {name:<18}{n_kept:>10}{n_new:>8}{n_reserve:>9}{note}")

    selected = pd.concat([c for c in chosen if len(c)], ignore_index=True)
    selected["ronda"] = selected["ronda"].astype(int)
    # orden aleatorio global de las partidas
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected.insert(0, "orden", np.arange(1, len(selected) + 1))
    return selected[["orden", "match_id", "platform", "tier", "patch", "ronda",
                     "fecha_agregado"]]


def write_table(table: str, subsets: list[Path], order: dict[str, int], out: Path,
                block_matches: int):
    """
    Escribe una tabla con las partidas en el orden elegido, sin cargarla completa en RAM:
      1) lee cada subconjunto, se queda con las partidas elegidas y reparte sus filas en
         archivos temporales por bloque (bloque = rango de posiciones en el orden final);
      2) lee cada bloque (pequeño), lo ordena y lo agrega al archivo final.
    Todo se lee como texto: los valores se copian tal cual, sin convertir 7 en 7.0.
    """
    tmp = out / f"_tmp_{table}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    fields = TABLE_FIELDS[table]

    for sub in subsets:
        path = sub / f"{table}.csv"
        if not path.exists():
            continue
        for df in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=100_000):
            df = df[df["match_id"].isin(order.keys())]
            if df.empty:
                continue
            block = df["match_id"].map(order) // block_matches
            for b, part in df.groupby(block, sort=False):
                dest = tmp / f"{b:05d}.csv"
                part.to_csv(dest, mode="a", header=not dest.exists(), index=False,
                            columns=fields)

    dest = out / f"{table}.csv"
    dest.unlink(missing_ok=True)
    rows = 0
    for block_file in sorted(tmp.glob("*.csv")):
        df = pd.read_csv(block_file, dtype=str, keep_default_na=False)
        if DEDUP_KEYS[table]:
            df = df.drop_duplicates(DEDUP_KEYS[table])
        else:
            df = df.drop_duplicates()
        keys = pd.DataFrame({"_ord": df["match_id"].map(order)})
        for k in SORT_KEYS[table]:
            keys[k] = pd.to_numeric(df[k], errors="coerce")
        df = df.loc[keys.sort_values(list(keys.columns), kind="stable").index]
        df.to_csv(dest, mode="a", header=not dest.exists(), index=False, columns=fields)
        rows += len(df)
    shutil.rmtree(tmp)
    print(f"  {table + '.csv':<17} {rows:>12,} filas")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-subset", type=int, default=config.MERGE_PER_SUBSET,
                        help=f"partidas por subconjunto (default {config.MERGE_PER_SUBSET})")
    parser.add_argument("--seed", type=int, default=42, help="semilla del azar")
    parser.add_argument("--fresh", action="store_true",
                        help=f"ignorar {IDS_FILE} anterior y elegir todo de nuevo")
    parser.add_argument("--out", default=str(config.MERGE_DIR), help="carpeta de salida")
    parser.add_argument("--block-matches", type=int, default=500,
                        help="partidas por bloque al escribir (menos = menos RAM)")
    args = parser.parse_args()

    subsets = find_subsets()
    if not subsets:
        print("No hay datos en data/csv/. ¿Ya corrieron collect.py?")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / IDS_FILE

    previous = None
    if ids_path.exists() and not args.fresh:
        previous = pd.read_csv(ids_path, dtype=str)
        print(f"Usando selección anterior: {len(previous):,} partidas en {IDS_FILE} "
              "(usen --fresh para empezar de cero)")

    available = load_available(subsets)
    print(f"Disponibles: {len(available):,} partidas en {len(subsets)} subconjuntos. "
          f"Meta: {args.per_subset} por subconjunto.")
    selected = select_ids(available, previous, args.per_subset, args.seed)
    print(f"\n  TOTAL seleccionadas: {len(selected):,}   "
          f"reserva sin usar: {len(available) - len(selected):,}\n")

    order = dict(zip(selected["match_id"], selected["orden"]))
    for table in ("matches", "participants", "frames", "events"):
        write_table(table, subsets, order, out, args.block_matches)

    selected.to_csv(ids_path, index=False)
    print(f"  {IDS_FILE:<17} {len(selected):>12,} filas")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
