"""
B2 - Split train / val / test (se corre UNA vez y se congela).

Estrategia: split temporal estratificado por subconjunto (plataforma x tier).
  - Dentro de cada subconjunto se ordenan las partidas por fecha de inicio y se asigna
    el 70 % más antiguo a train, el 15 % siguiente a val y el 15 % más reciente a test.
  - Así se imita el uso real (entrenar con el pasado, predecir partidas nuevas) y cada
    subconjunto queda con la misma proporción en cada split.
  - Por qué no un corte de fecha global: los jugadores de Master juegan más seguido y sus
    partidas son más recientes; con un corte global Master quedaba 51/23/26 y Gold 76/13/11.
  - Todas las filas de una partida van al mismo split (la unidad es la partida).

Uso:
  python src/split.py            # crea data/splits.csv y splits/splits.csv
  python src/split.py --force    # sobrescribe un split existente (¡solo si están seguros!)

Salidas:
  data/splits.csv, splits/splits.csv   match_id, split, platform, tier, patch, game_start_utc
  results/b2_split_tamanos.csv         tamaños por split x tier x región
  results/b2_fuga_jugadores.csv        cuántos jugadores de val/test aparecen en train
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from riot_data import config  # noqa: E402

FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS_DATA = config.DATA_DIR / "splits.csv"
SPLITS_REPO = ROOT / "splits" / "splits.csv"
RESULTS = ROOT / "results"


def temporal_split(matches: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in matches.groupby(["platform", "tier"], sort=True):
        g = g.sort_values(["game_start_utc", "match_id"]).reset_index(drop=True)
        n = len(g)
        n_train = round(n * FRACS["train"])
        n_val = round(n * FRACS["val"])
        g["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def size_tables(s: pd.DataFrame) -> pd.DataFrame:
    order = ["train", "val", "test"]
    table = (s.groupby(["platform", "tier", "split"]).size().unstack("split")[order]
             .reset_index())
    table["total"] = table[order].sum(axis=1)
    return table


def player_overlap(s: pd.DataFrame) -> pd.DataFrame:
    """% de partidas de val/test con al menos un jugador que también jugó en train."""
    parts = pd.read_csv(config.MERGE_DIR / "participants.csv", usecols=["match_id", "player"],
                        dtype=str)
    parts = parts.merge(s[["match_id", "split"]], on="match_id")
    train_players = set(parts.loc[parts.split == "train", "player"])
    rows = []
    for split in ("val", "test"):
        p = parts[parts.split == split].copy()
        p["in_train"] = p["player"].isin(train_players)
        per_match = p.groupby("match_id")["in_train"].sum()
        rows.append({
            "split": split,
            "partidas": len(per_match),
            "jugadores_unicos": p["player"].nunique(),
            "pct_jugadores_vistos_en_train": round(100 * p.drop_duplicates("player")["in_train"].mean(), 1),
            "pct_partidas_con_>=1_jugador_de_train": round(100 * (per_match >= 1).mean(), 1),
            "promedio_jugadores_de_train_por_partida": round(per_match.mean(), 2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="sobrescribir un split existente")
    args = parser.parse_args()

    if SPLITS_REPO.exists() and not args.force:
        print(f"Ya existe {SPLITS_REPO.relative_to(ROOT)}. El split está congelado; "
              "se usa el existente para regenerar las tablas (use --force para rehacerlo).")
        s = pd.read_csv(SPLITS_REPO, dtype={"patch": str})
    else:
        m = pd.read_csv(config.MERGE_DIR / "matches.csv", dtype={"patch": str},
                        usecols=["match_id", "platform", "tier", "patch", "game_start_utc"])
        s = temporal_split(m)
        s = s[["match_id", "split", "platform", "tier", "patch", "game_start_utc"]]
        for path in (SPLITS_DATA, SPLITS_REPO):
            path.parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(path, index=False)
        print(f"Split guardado en {SPLITS_DATA.relative_to(ROOT)} y {SPLITS_REPO.relative_to(ROOT)}")

    RESULTS.mkdir(exist_ok=True)
    sizes = size_tables(s)
    sizes.to_csv(RESULTS / "b2_split_tamanos.csv", index=False)
    overlap = player_overlap(s)
    overlap.to_csv(RESULTS / "b2_fuga_jugadores.csv", index=False)

    print("\nPartidas por split:")
    print(s["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("\nRango de fechas por split:")
    print(s.groupby("split")["game_start_utc"].agg(["min", "max"])
          .reindex(["train", "val", "test"]).to_string())
    print("\nParches por split:")
    print(pd.crosstab(s["patch"], s["split"])[["train", "val", "test"]].to_string())
    print("\nJugadores de val/test que también aparecen en train:")
    print(overlap.to_string(index=False))
    print(f"\nTabla completa de tamaños: results/b2_split_tamanos.csv")


if __name__ == "__main__":
    main()
