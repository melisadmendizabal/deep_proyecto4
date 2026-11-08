"""
B8 - Elige las partidas a anotar y prepara sus datos para la herramienta de anotación.

Reglas:
  - Solo val y test (nunca train), sin partidas AFK.
  - Mitad val (para ELEGIR el umbral theta) y mitad test (para REPORTAR el valor).
  - Balanceadas por tier y mezcladas entre regiones.
  - n_shared partidas las anotan AMBOS (para medir el acuerdo entre anotadores) y
    n_each las anota solo una persona. Por defecto: 15 + 20 + 20 = 55 partidas distintas,
    35 por persona.

Uso:
  python src/select_annotation.py                 # crea annotations/seleccion.csv + data/anotacion/
  python src/select_annotation.py --add-own LA1_123 LA1_456   # agrega partidas propias (las anotan ambos)

Salidas:
  annotations/seleccion.csv        orden, match_id, split, tier, platform, duracion_min, anotador (A/B/ambos)
  data/anotacion/{matches,participants,events}.csv   solo las partidas elegidas (rápido de cargar)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from riot_data import config  # noqa: E402

SEL = ROOT / "annotations" / "seleccion.csv"
OUT = config.DATA_DIR / "anotacion"
TIERS = ["SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"]


def balanced_sample(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n partidas repartidas lo más parejo posible entre tiers (y al azar entre regiones)."""
    per_tier = {t: n // len(TIERS) for t in TIERS}
    for t in rng.permutation(TIERS)[: n % len(TIERS)]:
        per_tier[t] += 1
    parts = [pool[pool["tier"] == t].sample(k, random_state=int(rng.integers(1 << 31)))
             for t, k in per_tier.items()]
    return pd.concat(parts)


def select(n_shared: int, n_each: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = pd.read_csv(ROOT / "splits" / "splits.csv")
    q = pd.read_csv(ROOT / "results" / "b2_calidad_partidas.csv",
                    usecols=["match_id", "excluir_de_train", "duration_min"])
    pool = s.merge(q, on="match_id")
    pool = pool[pool["split"].isin(["val", "test"]) & ~pool["excluir_de_train"]]

    total = n_shared + 2 * n_each
    chosen = []
    for split, n in (("val", total // 2), ("test", total - total // 2)):
        chosen.append(balanced_sample(pool[pool["split"] == split], n, rng))
    chosen = pd.concat(chosen).sample(frac=1, random_state=seed).reset_index(drop=True)

    # reparto: ambos / A / B, cada grupo con mezcla de val y test
    chosen["anotador"] = ""
    order = chosen.sort_values(["split", "tier"]).index.to_numpy()
    labels = (["ambos"] * n_shared + ["A"] * n_each + ["B"] * n_each)
    labels = list(rng.permutation(labels))
    chosen.loc[order, "anotador"] = labels
    chosen.insert(0, "orden", np.arange(1, len(chosen) + 1))
    return chosen[["orden", "match_id", "split", "tier", "platform", "duration_min", "anotador"]]


def extract(match_ids: set[str]) -> None:
    """Copia a data/anotacion/ solo las filas de las partidas elegidas (todo como texto).

    Busca en data/datos_merge/ y en las carpetas de partidas propias data/csv/*_own/.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [config.MERGE_DIR] + sorted(config.CSV_DIR.glob("*_own"))
    for name in ("matches", "participants", "events"):
        parts = []
        for src in sources:
            path = src / f"{name}.csv"
            if not path.exists():
                continue
            for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=500_000):
                hit = chunk[chunk["match_id"].isin(match_ids)]
                if len(hit):
                    parts.append(hit)
        df = pd.concat(parts).drop_duplicates()
        df.to_csv(OUT / f"{name}.csv", index=False)
    missing = match_ids - set(pd.read_csv(OUT / "matches.csv", usecols=["match_id"])["match_id"])
    if missing:
        print(f"  aviso: no se encontraron datos de {sorted(missing)} "
              "(¿se descargaron con collect.py --riot-id?)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shared", type=int, default=15)
    ap.add_argument("--n-each", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add-own", nargs="*", help="match_id de partidas propias (ya descargadas)")
    ap.add_argument("--force", action="store_true", help="rehacer la selección aunque exista")
    args = ap.parse_args()

    if SEL.exists() and not args.force:
        sel = pd.read_csv(SEL)
        print(f"Ya existe {SEL.relative_to(ROOT)} ({len(sel)} partidas); se reutiliza "
              "(use --force para rehacerla).")
    else:
        sel = select(args.n_shared, args.n_each, args.seed)
    if args.add_own:
        own = pd.DataFrame({"match_id": [m for m in args.add_own if m not in set(sel["match_id"])]})
        own["split"], own["tier"], own["anotador"] = "propia", "OWN", "ambos"
        own["platform"] = own["match_id"].str.split("_").str[0].str.lower()
        own["duration_min"] = np.nan
        own["orden"] = np.arange(len(sel) + 1, len(sel) + 1 + len(own))
        sel = pd.concat([sel, own[sel.columns]], ignore_index=True)
    SEL.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(SEL, index=False)

    print("Extrayendo datos de las partidas elegidas...")
    extract(set(sel["match_id"]))
    print(f"\n{len(sel)} partidas -> {SEL.relative_to(ROOT)}")
    print(pd.crosstab(sel["anotador"], sel["split"], margins=True).to_string())
    print("\nPor tier:", sel["tier"].value_counts().to_dict())
    print("\nPor persona: A =", int(sel["anotador"].isin(["A", "ambos"]).sum()),
          "· B =", int(sel["anotador"].isin(["B", "ambos"]).sum()))


if __name__ == "__main__":
    main()
