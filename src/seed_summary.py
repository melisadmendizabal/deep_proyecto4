"""
B6 - Variabilidad entre semillas: métricas de test de cada semilla y su media ± desviación.

Complementa a evaluate.py (que da IC por bootstrap sobre partidas para UNA corrida): aquí se
mide cuánto cambia el resultado si se reentrena con otra semilla. Usa las mismas métricas
(evaluate.Rows).

Uso:
  python src/seed_summary.py --prefix gru_final mlp_final --name M4 M3

Salida: results/b6_semillas.csv (una fila por modelo y semilla + filas "media" y "sd")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import Rows, load_predictions  # noqa: E402

METRICS = ["auc", "logloss", "logloss_partida", "brier", "ece", "accuracy"]


def summarize(prefix: str, name: str, split: str = "test") -> pd.DataFrame:
    runs = sorted((ROOT / "results" / "runs").glob(f"{prefix}__s*"))
    if not runs:
        raise SystemExit(f"No hay corridas results/runs/{prefix}__s*")
    rows = []
    for run in runs:
        pred = load_predictions(run)
        sel = pred["split"] == split
        m = Rows(pred["probs"][sel], pred["y"][sel]).metrics()
        s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        rows.append({"modelo": name, "semilla": run.name.split("__s")[-1], **m,
                     "val_loss": s["val_loss"], "train_seconds": s["train_seconds"],
                     "n_params": s["n_params"]})
    df = pd.DataFrame(rows)
    num = METRICS + ["val_loss", "train_seconds"]
    stats = pd.DataFrame([{"modelo": name, "semilla": "media", **df[num].mean().to_dict()},
                          {"modelo": name, "semilla": "sd", **df[num].std(ddof=1).to_dict()}])
    return pd.concat([df, stats], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--name", nargs="+", required=True)
    args = ap.parse_args()
    if len(args.prefix) != len(args.name):
        raise SystemExit("--prefix y --name deben tener la misma cantidad")
    out = pd.concat([summarize(p, n) for p, n in zip(args.prefix, args.name)], ignore_index=True)
    path = ROOT / "results" / "b6_semillas.csv"
    out.to_csv(path, index=False, float_format="%.5f")
    show = out[out["semilla"].isin(["media", "sd"])].copy()
    print(show[["modelo", "semilla", "auc", "logloss", "ece", "train_seconds"]].round(4).to_string(index=False))
    print(f"\n-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
