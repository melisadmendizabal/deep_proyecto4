"""
B7 / E2 - Curva de aprendizaje: ¿faltan datos? ¿dónde están los rendimientos decrecientes?

Para cada tamaño de train n en {1k, 3k, 6k, 12k, todo} y semilla s en {42, 43, 44}:
  M2  LightGBM      (src/baselines.py, CPU)
  M4  GRU           (src/train.py con la config que se pase; usar la MEJOR de N1)
Los dos modelos ven exactamente la misma submuestra de train (mismo subset() y semilla), y
siempre se evalúan en el mismo test completo.

Uso:
  python src/learning_curve.py --gru-config configs/gru_base.yaml --set model.hidden=128 model.layers=1
  python src/learning_curve.py --models m2 --sizes 1000 3000 --seeds 42      # prueba corta
  python src/learning_curve.py --gru-config configs/gru_base.yaml --smoke    # 1 época, para probar

Se puede cortar y relanzar: salta las corridas que ya tienen summary.json.

Salidas:
  results/runs/lc_<m2|m4>__n<n>_s<s>/     una carpeta por corrida (mismo formato que train.py)
  results/b7_curva_aprendizaje.csv         una fila por corrida: val_loss + métricas de test
  results/b7_curva_resumen.csv             media y desviación entre semillas por modelo y tamaño
  results/figures/b7_curva_aprendizaje.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.baselines import run_baseline  # noqa: E402
from src.datasets import load_sequences  # noqa: E402
from src.evaluate import Rows, load_predictions  # noqa: E402
from src.train import load_config, run as train_run  # noqa: E402

RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "b7_curva_aprendizaje.csv"
SUMMARY = ROOT / "results" / "b7_curva_resumen.csv"
FIG = ROOT / "results" / "figures" / "b7_curva_aprendizaje.png"
LABEL = {"m2": "M2 LightGBM", "m4": "M4 GRU"}


def test_metrics(run_dir: Path) -> dict:
    p = load_predictions(run_dir)
    t = p["split"] == "test"
    m = Rows(p["probs"][t], p["y"][t]).metrics()
    return {f"test_{k}": m[k] for k in ("auc", "logloss", "logloss_partida", "brier", "ece")}


def run_one(model: str, n: int | None, seed: int, full, gru_cfg: dict | None, smoke: bool) -> Path:
    tag = f"n{n if n else 'todo'}_s{seed}"
    run_dir = RUNS / f"lc_{model}{'_smoke' if smoke else ''}__{tag}"
    if (run_dir / "summary.json").exists():
        print(f"  {run_dir.name}: ya terminada, se salta")
        return run_dir
    t0 = time.time()
    if model == "m2":
        run_baseline("m2", full, run_name=run_dir.name, max_train_matches=n, subset_seed=seed,
                     verbose=False, seed=seed)
    else:
        cfg = json.loads(json.dumps(gru_cfg))       # copia profunda
        cfg["name"] = run_dir.name
        cfg["data"]["max_train_matches"] = n
        cfg["data"]["subset_seed"] = seed
        cfg["train"]["seed"] = seed
        if smoke:
            cfg["train"]["max_epochs"] = 1
        train_run(cfg, run_dir, resume=True, full=full, verbose=False)
    print(f"  {run_dir.name}: {time.time() - t0:.0f} s")
    return run_dir


def collect(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, float_format="%.5f")
    metrics = ["val_loss", "test_auc", "test_logloss", "test_brier", "test_ece", "train_seconds"]
    g = df.groupby(["modelo", "n_train"])[metrics]
    summ = g.mean().add_suffix("_media").join(g.std().add_suffix("_sd")).join(g.size().rename("semillas"))
    summ = summ.reset_index()
    summ.to_csv(SUMMARY, index=False, float_format="%.5f")
    return summ


def figure(summ: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    COLOR = {"m2": "#1baf7a", "m4": "#2a78d6"}
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], "test_auc", "AUC en test (más es mejor)"),
                              (axes[1], "test_logloss", "Log-loss en test (menos es mejor)")]:
        for model, d in summ.groupby("modelo"):
            d = d.sort_values("n_train")
            mean, sd = d[f"{metric}_media"], d[f"{metric}_sd"].fillna(0)
            ax.plot(d["n_train"], mean, marker="o", ms=4, color=COLOR.get(model), label=LABEL.get(model, model))
            ax.fill_between(d["n_train"], mean - sd, mean + sd, color=COLOR.get(model), alpha=0.18, lw=0)
        ax.set_xscale("log")
        ticks = sorted(summ["n_train"].unique())
        ax.set_xticks(ticks, [f"{t / 1000:.0f}k" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("partidas de entrenamiento (escala log)")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Curva de aprendizaje (media ± desv. entre 3 semillas)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["m2", "m4"], choices=["m2", "m4"])
    ap.add_argument("--sizes", nargs="*", type=int, default=[1000, 3000, 6000, 12000, 0],
                    help="partidas de train; 0 = todo train")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--gru-config", default=str(ROOT / "configs" / "gru_base.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="cambios a la config de la GRU (mejor de N1)")
    ap.add_argument("--smoke", action="store_true", help="GRU con 1 época (probar el comando)")
    args = ap.parse_args()

    gru_cfg = load_config(args.gru_config, args.set)
    data_path = Path(gru_cfg["data"]["path"])
    full = load_sequences(data_path if data_path.is_absolute() else ROOT / data_path)
    n_all = int(((full.split == "train") & ~full.afk).sum())
    print(f"Train completo (sin AFK): {n_all:,} partidas.  GRU: {gru_cfg['model']}  lr={gru_cfg['train']['lr']}")

    rows, t_start = [], time.time()
    total = len(args.models) * len(args.sizes) * len(args.seeds)
    i = 0
    for n in args.sizes:                      # primero los tamaños chicos: resultados útiles pronto
        for model in args.models:
            for seed in args.seeds:
                i += 1
                size = None if n == 0 or n >= n_all else n
                print(f"[{i}/{total}] {LABEL[model]}  n={size or n_all:,}  semilla {seed}")
                run_dir = run_one(model, size, seed, full, gru_cfg, args.smoke)
                s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                rows.append({"modelo": model, "n_train": s["n_train"], "semilla": seed,
                             "run_id": run_dir.name, "val_loss": s["val_loss"],
                             "train_seconds": s["train_seconds"], **test_metrics(run_dir)})
                summ = collect(rows)
    figure(summ)
    print(f"\nTerminado en {(time.time() - t_start) / 60:.1f} min")
    cols = ["modelo", "n_train", "test_auc_media", "test_auc_sd", "test_logloss_media", "train_seconds_media"]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Tablas: {OUT.name}, {SUMMARY.name}; figura: {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
