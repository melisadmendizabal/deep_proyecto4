"""
Mezcla subconjuntos (plataforma_tier) en un solo juego de CSV.

Ejemplos:
  python merge.py                                     # todo excepto partidas propias (OWN)
  python merge.py --platforms la1 na1                 # solo esas plataformas, todos los tiers
  python merge.py --tiers gold platinum --out data/csv/_merged/gold_plat
  python merge.py --include-own                       # incluye también las partidas propias
  python merge.py --per-subset 0                      # todas las partidas, sin muestrear

Por defecto toma una muestra aleatoria de MERGE_PER_SUBSET partidas (config.py) de cada
subconjunto plataforma_tier, siempre la misma gracias a --seed. Si un subconjunto tiene
menos, se usan todas las que haya y se avisa.

Además de concatenar, limpia: quita filas duplicadas y filas de partidas incompletas
(si el script de recolección se interrumpió a mitad de una partida).
"""
import argparse
from pathlib import Path

import pandas as pd

from riot_data import config
from riot_data.parse import TABLE_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platforms", nargs="*", help="filtrar plataformas (default: todas)")
    parser.add_argument("--tiers", nargs="*", help="filtrar tiers (default: todos menos OWN)")
    parser.add_argument("--include-own", action="store_true", help="incluir partidas propias")
    parser.add_argument("--per-subset", type=int, default=config.MERGE_PER_SUBSET,
                        help="partidas al azar por subconjunto (0 = todas). "
                             f"Default: {config.MERGE_PER_SUBSET}")
    parser.add_argument("--seed", type=int, default=42, help="semilla del muestreo")
    parser.add_argument("--out", help="carpeta de salida (default: data/csv/_merged/...)")
    args = parser.parse_args()
    if args.out is None:
        name = f"sample_{args.per_subset}" if args.per_subset else "all"
        args.out = str(config.CSV_DIR / "_merged" / name)

    platforms = {p.lower() for p in args.platforms} if args.platforms else None
    tiers = {t.lower() for t in args.tiers} if args.tiers else None

    subsets = []
    for sub in sorted(config.CSV_DIR.iterdir()) if config.CSV_DIR.exists() else []:
        if not sub.is_dir() or sub.name.startswith("_") or not (sub / "matches.csv").exists():
            continue
        platform, tier = sub.name.split("_", 1)
        if platforms and platform not in platforms:
            continue
        if tiers and tier not in tiers:
            continue
        if tier == "own" and not (args.include_own or (tiers and "own" in tiers)):
            continue
        subsets.append(sub)

    if not subsets:
        print("No hay subconjuntos que coincidan. ¿Ya corrieron collect.py?")
        return
    print("Subconjuntos:", ", ".join(s.name for s in subsets))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    matches = pd.concat([pd.read_csv(s / "matches.csv") for s in subsets], ignore_index=True)
    matches = matches.drop_duplicates("match_id")

    if args.per_subset:
        # barajar con semilla fija y quedarse con las primeras N de cada subconjunto
        matches = (matches.sample(frac=1, random_state=args.seed)
                   .groupby(["platform", "tier"]).head(args.per_subset)
                   .sort_values(["platform", "tier", "match_id"]))
        sizes = matches.groupby(["platform", "tier"]).size()
        print(f"\nMuestra de {args.per_subset} por subconjunto (seed={args.seed}):")
        for (platform, tier), n in sizes.items():
            flag = "" if n >= args.per_subset else f"   <- solo había {n}"
            print(f"  {platform}_{tier.lower():<14} {n:>6}{flag}")
        print()

    valid_ids = set(matches["match_id"])
    matches.to_csv(out / "matches.csv", index=False)
    print(f"  matches.csv       {len(matches):>10,} filas")

    keys = {
        "participants": ["match_id", "participant_id"],
        "frames": ["match_id", "minute", "participant_id"],
        "events": None,  # duplicado = fila idéntica
    }
    for table, key in keys.items():
        parts = [pd.read_csv(s / f"{table}.csv", low_memory=False)
                 for s in subsets if (s / f"{table}.csv").exists()]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df[df["match_id"].isin(valid_ids)]           # quita partidas incompletas
        df = df.drop_duplicates(subset=key)
        df = df[TABLE_FIELDS[table]]
        df.to_csv(out / f"{table}.csv", index=False)
        print(f"  {table + '.csv':<17} {len(df):>10,} filas")

    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
