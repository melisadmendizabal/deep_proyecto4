"""
B10 - Acuerdo entre anotadores en las partidas que anotaron A y B.

Solo cuenta partidas de val/test (no las propias, que pueden anotarse en conjunto).
Por partida: los momentos de A y de B se emparejan 1 a 1 a ±tol minutos (moments.match_moments).
  acuerdo (F1) = 2 × coincidencias / (momentos de A + momentos de B)   (ambos "sin momentos" = 1)
  recall de B respecto a A = coincidencias / momentos de A (y al revés)
Entre las coincidencias: % en que coinciden el equipo beneficiado y el tipo.

El acuerdo humano es el TECHO realista del detector: si dos personas coinciden en el 60 % de
los momentos, un modelo no puede "acertar" mucho más que eso contra cualquiera de las dos.

Uso:
  python src/agreement.py              # tol = match_window_min de configs/value.yaml (2 min)
  python src/agreement.py --min-conf 2

Salida: results/b10_acuerdo.csv (una fila por partida compartida) + resumen en pantalla.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.moments import match_moments  # noqa: E402

ANN = ROOT / "annotations"


def load(who: str, min_conf: int) -> pd.DataFrame:
    df = pd.read_csv(ANN / f"{who}.csv")
    last = df.groupby("match_id")["guardado_utc"].transform("max")
    df = df[df["guardado_utc"] == last]
    none = df["sin_momentos"].astype(str).str.lower().eq("true")
    conf = pd.to_numeric(df["confianza"], errors="coerce").fillna(3)
    df["valido"] = ~none & (conf >= min_conf)
    return df


def agreement(a: pd.DataFrame, b: pd.DataFrame, tol: int, only: set | None = None) -> pd.DataFrame:
    rows = []
    shared = set(a["match_id"]) & set(b["match_id"])
    for mid in sorted(shared & only if only is not None else shared):
        ga = a[(a["match_id"] == mid) & a["valido"]]
        gb = b[(b["match_id"] == mid) & b["valido"]]
        ma, mb = ga["minuto"].astype(int).tolist(), gb["minuto"].astype(int).tolist()
        r = match_moments(mb, ma, tol=tol)  # "detectado" = B, "anotado" = A
        n = r["tp"]
        same_team = same_type = 0
        for mb_min, ma_min in r["pairs"]:
            ra = ga[ga["minuto"].astype(int) == ma_min].iloc[0]
            rb = gb[gb["minuto"].astype(int) == mb_min].iloc[0]
            same_team += ra["equipo"] == rb["equipo"]
            same_type += ra["tipo"] == rb["tipo"]
        rows.append({
            "match_id": mid, "n_A": len(ma), "n_B": len(mb), "coinciden": n,
            "acuerdo_f1": 1.0 if not ma and not mb else 2 * n / (len(ma) + len(mb)),
            "recall_B_vs_A": n / len(ma) if ma else float("nan"),
            "recall_A_vs_B": n / len(mb) if mb else float("nan"),
            "mismo_equipo": same_team, "mismo_tipo": same_type,
            "minutos_A": " ".join(map(str, ma)), "minutos_B": " ".join(map(str, mb)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-conf", type=int, default=1)
    ap.add_argument("--tol", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "value.yaml").read_text(encoding="utf-8"))
    tol = args.tol if args.tol is not None else int(cfg.get("match_window_min", 2))
    missing = [w for w in ("A", "B") if not (ANN / f"{w}.csv").exists()]
    if missing:
        raise SystemExit(f"Falta annotations/{missing[0]}.csv: el acuerdo necesita las anotaciones de A y B")
    # solo val/test: las partidas propias se pueden anotar juntos y no deben contar como acuerdo
    sel = pd.read_csv(ANN / "seleccion.csv")
    only = set(sel.loc[sel["split"].isin(["val", "test"]), "match_id"])
    df = agreement(load("A", args.min_conf), load("B", args.min_conf), tol, only)
    if df.empty:
        raise SystemExit("Todavía no hay partidas anotadas por A y por B a la vez")
    out = ROOT / "results" / "b10_acuerdo.csv"
    df.to_csv(out, index=False, float_format="%.3f")
    tot = df[["n_A", "n_B", "coinciden", "mismo_equipo", "mismo_tipo"]].sum()
    f1 = 2 * tot["coinciden"] / (tot["n_A"] + tot["n_B"])
    print(f"{len(df)} partidas compartidas, tolerancia ±{tol} min, confianza ≥ {args.min_conf}")
    print(f"  Momentos: A {tot['n_A']}, B {tot['n_B']}, coinciden {tot['coinciden']}")
    print(f"  Acuerdo (F1 global): {f1:.2f} · media por partida: {df['acuerdo_f1'].mean():.2f}")
    print(f"  De lo que marcó A, B encontró {tot['coinciden'] / max(tot['n_A'], 1):.0%}; "
          f"de lo que marcó B, A encontró {tot['coinciden'] / max(tot['n_B'], 1):.0%}")
    if tot["coinciden"]:
        print(f"  En las coincidencias: mismo equipo {tot['mismo_equipo'] / tot['coinciden']:.0%}, "
              f"mismo tipo {tot['mismo_tipo'] / tot['coinciden']:.0%}")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
