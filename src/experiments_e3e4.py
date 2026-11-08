"""
B9 - Experimentos E3 y E4 (guía, sección 4): ¿generaliza el modelo a otro grupo de jugadores?

  E3  Alto elo vs mezclado: train SOLO con DIAMOND+MASTER vs la MISMA cantidad de partidas
      al azar de todos los tiers. Se evalúa en test SILVER+GOLD.
      Pregunta: ¿sirve un modelo entrenado con alto elo para jugadores promedio?
  E4  Región no vista: train SIN LA1 vs train CON LA1 (submuestreado al mismo tamaño).
      Se evalúa en test LA1. Pregunta: ¿se puede lanzar en una región nueva sin reentrenar?

Comparación justa:
  - Mismo número de partidas de train en las dos condiciones (guía, sección 4).
  - El grupo evaluado (Silver+Gold en E3, LA1 en E4) se QUITA de val en ambas condiciones:
    val elige la época (early stopping) y la temperatura; si incluyera al grupo evaluado, el
    modelo "lo habría visto". El test no se toca.
  - Modelos: M2 (LightGBM) y M4 (GRU, mejor config de N1), 3 semillas cada uno.
  - Diferencia entre condiciones: bootstrap PAREADO sobre las partidas evaluadas, promediando
    la diferencia entre las 3 semillas en cada réplica.

Uso:
  python src/experiments_e3e4.py                         # E3 + E4, M2 + M4, 3 semillas (~5-10 min)
  python src/experiments_e3e4.py --exps E4 --models m2 --seeds 42      # prueba corta
Se puede cortar y relanzar: salta las corridas que ya tienen summary.json.

Salidas:
  results/runs/<e3|e4>_<m2|m4>__<condición>_s<semilla>/
  results/b9_e3e4_corridas.csv     una fila por corrida (métricas en el grupo evaluado y en el resto)
  results/b9_e3e4.csv              media ± sd por condición y diferencia pareada vs la referencia
  results/figures/b9_e3e4.png
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
from src.datasets import SequenceData, load_sequences  # noqa: E402
from src.evaluate import METRICS, Rows, load_predictions  # noqa: E402
from src.train import load_config, run as train_run  # noqa: E402

RUNS = ROOT / "results" / "runs"
OUT_RUNS = ROOT / "results" / "b9_e3e4_corridas.csv"
OUT = ROOT / "results" / "b9_e3e4.csv"
FIG = ROOT / "results" / "figures" / "b9_e3e4.png"
# mejor config de la GRU en la búsqueda N1 (results/b6_resumen.md)
BEST_GRU = ["model.hidden=128", "model.layers=2", "model.dropout=0.1", "train.lr=0.0003"]
LABEL = {"m2": "M2 LightGBM", "m4": "M4 GRU"}

EXPERIMENTS = {
    "E3": {
        "pregunta": "¿Sirve entrenar con alto elo para jugadores promedio?",
        "grupo": {"tiers": ["SILVER", "GOLD"]},
        "condiciones": {"alto_elo": {"tiers": ["DIAMOND", "MASTER"]},
                        "mezclado": {}},
        "referencia": "mezclado",
    },
    "E4": {
        "pregunta": "¿Se puede lanzar en una región nueva sin reentrenar?",
        "grupo": {"platforms": ["la1"]},
        "condiciones": {"sin_la1": {"platforms": ["na1", "euw1", "kr"]},
                        "con_la1": {}},
        "referencia": "con_la1",
    },
}


# ------------------------------------------------------------------ datos
def in_group(data: SequenceData, grupo: dict) -> np.ndarray:
    keep = np.ones(len(data), dtype=bool)
    if grupo.get("tiers"):
        keep &= np.isin(data.tier, grupo["tiers"])
    if grupo.get("platforms"):
        keep &= np.isin(data.platform, grupo["platforms"])
    return keep


def experiment_data(full: SequenceData, exp: dict) -> SequenceData:
    """Quita de val al grupo evaluado (en las dos condiciones por igual)."""
    drop = (full.split == "val") & in_group(full, exp["grupo"])
    return full._take(np.flatnonzero(~drop))


def train_size(data: SequenceData, cond: dict) -> int:
    return len(data.subset(split="train", exclude_afk=True, tiers=cond.get("tiers"),
                           platforms=cond.get("platforms")))


# ------------------------------------------------------------------ corridas
def run_one(exp_id: str, cond_name: str, cond: dict, model: str, seed: int, n: int,
            data: SequenceData, gru_cfg: dict) -> Path:
    run_dir = RUNS / f"{exp_id.lower()}_{model}__{cond_name}_s{seed}"
    if (run_dir / "summary.json").exists():
        print(f"  {run_dir.name}: ya terminada, se salta")
        return run_dir
    t0 = time.time()
    if model == "m2":
        run_baseline("m2", data, run_name=run_dir.name, max_train_matches=n, subset_seed=seed,
                     seed=seed, tiers=cond.get("tiers"), platforms=cond.get("platforms"),
                     verbose=False)
    else:
        cfg = json.loads(json.dumps(gru_cfg))
        cfg["name"] = run_dir.name
        cfg["data"].update({"tiers": cond.get("tiers"), "platforms": cond.get("platforms"),
                            "max_train_matches": n, "subset_seed": seed})
        cfg["train"]["seed"] = seed
        train_run(cfg, run_dir, resume=True, full=data, verbose=False)
    print(f"  {run_dir.name}: {time.time() - t0:.0f} s")
    return run_dir


def test_grid(run_dir: Path, match_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """probs [n, T] e y de las partidas pedidas, en ese orden."""
    p = load_predictions(run_dir)
    t = p["split"] == "test"
    pos = pd.Series(np.arange(t.sum()), index=p["match_id"][t]).reindex(match_ids).to_numpy()
    return p["probs"][t][pos], p["y"][t][pos]


# ------------------------------------------------------------------ análisis
def paired_delta(cond_rows: list[Rows], ref_rows: list[Rows], n: int, n_boot: int = 1000,
                 seed: int = 0) -> dict:
    """Δ = media sobre semillas de (métrica cond − métrica ref), con IC por bootstrap pareado."""
    def delta(counts=None):
        d = []
        for a, b in zip(cond_rows, ref_rows):
            ma, mb = a.metrics(counts), b.metrics(counts)
            d.append({k: ma[k] - mb[k] for k in METRICS})
        return {k: float(np.mean([x[k] for x in d])) for k in METRICS}
    rng = np.random.default_rng(seed)
    boots = pd.DataFrame([delta(np.bincount(rng.integers(0, n, n), minlength=n).astype(float))
                          for _ in range(n_boot)])
    point = delta()
    out = {}
    for k in METRICS:
        out[f"d_{k}"] = point[k]
        out[f"d_{k}_lo"], out[f"d_{k}_hi"] = boots[k].quantile(0.025), boots[k].quantile(0.975)
    return out


def figure(res: pd.DataFrame) -> None:
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
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })
    d = res[res["condicion"] != res["referencia"]].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, k, title in [(axes[0], "auc", "ΔAUC vs referencia (más es mejor)"),
                         (axes[1], "logloss", "ΔLog-loss vs referencia (menos es mejor)")]:
        labels = []
        for i, r in d.iterrows():
            y = len(d) - 1 - i
            ax.errorbar(r[f"d_{k}"], y, xerr=[[r[f"d_{k}"] - r[f"d_{k}_lo"]], [r[f"d_{k}_hi"] - r[f"d_{k}"]]],
                        fmt="o", color=COLOR[r["modelo"]], capsize=3, ms=5)
            labels.append((y, f"{r['experimento']} {r['condicion']} vs {r['referencia']}\n{LABEL[r['modelo']]}"))
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.set_yticks([y for y, _ in labels], [lab for _, lab in labels], fontsize=8.5)
        ax.set_title(title)
    axes[1].set_yticklabels([])
    fig.suptitle("E3 / E4: diferencia pareada en el grupo evaluado (IC 95 %, media de 3 semillas)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--models", nargs="*", default=["m2", "m4"], choices=["m2", "m4"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--gru-config", default=str(ROOT / "configs" / "gru_base.yaml"))
    ap.add_argument("--set", nargs="*", default=BEST_GRU, help="config de la GRU (default: mejor de N1)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    gru_cfg = load_config(args.gru_config, args.set)
    data_path = Path(gru_cfg["data"]["path"])
    full = load_sequences(data_path if data_path.is_absolute() else ROOT / data_path)
    t_start = time.time()
    run_rows, res_rows = [], []
    for exp_id in args.exps:
        exp = EXPERIMENTS[exp_id]
        data = experiment_data(full, exp)
        sizes = {c: train_size(data, cond) for c, cond in exp["condiciones"].items()}
        n = min(sizes.values())
        test = data.subset(split="test")
        target = in_group(test, exp["grupo"])
        ids_target, ids_rest = test.match_id[target], test.match_id[~target]
        print(f"\n{exp_id}: {exp['pregunta']}\n  train disponible {sizes} -> {n:,} partidas en cada "
              f"condición; test evaluado: {len(ids_target):,} partidas; val sin el grupo: "
              f"{(data.split == 'val').sum():,}")
        rows_by = {}
        for model in args.models:
            for cond_name, cond in exp["condiciones"].items():
                for seed in args.seeds:
                    run_dir = run_one(exp_id, cond_name, cond, model, seed, n, data, gru_cfg)
                    probs, y = test_grid(run_dir, ids_target)
                    r = Rows(probs, y)
                    rows_by[(model, cond_name, seed)] = r
                    rest = Rows(*test_grid(run_dir, ids_rest)).metrics()
                    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                    run_rows.append({"experimento": exp_id, "modelo": model, "condicion": cond_name,
                                     "semilla": seed, "n_train": s["n_train"], "run_id": run_dir.name,
                                     "val_loss": s["val_loss"], "temperatura": s["temperature"],
                                     **{f"grupo_{k}": v for k, v in r.metrics().items()},
                                     "resto_auc": rest["auc"], "resto_logloss": rest["logloss"]})
            ref = exp["referencia"]
            for cond_name in exp["condiciones"]:
                runs = [rows_by[(model, cond_name, s)] for s in args.seeds]
                ms = pd.DataFrame([r.metrics() for r in runs])
                row = {"experimento": exp_id, "modelo": model, "condicion": cond_name,
                       "referencia": ref, "n_train": n, "n_partidas_evaluadas": len(ids_target),
                       "semillas": len(args.seeds)}
                for k in METRICS:
                    row[k], row[f"{k}_sd"] = ms[k].mean(), ms[k].std()
                if cond_name != ref:
                    row.update(paired_delta(runs, [rows_by[(model, ref, s)] for s in args.seeds],
                                            len(ids_target), args.n_boot))
                res_rows.append(row)
        pd.DataFrame(run_rows).to_csv(OUT_RUNS, index=False, float_format="%.5f")
    res = pd.DataFrame(res_rows)
    res.to_csv(OUT, index=False, float_format="%.5f")
    figure(res)
    print(f"\nTerminado en {(time.time() - t_start) / 60:.1f} min")
    for _, r in res.iterrows():
        line = f"  {r['experimento']} {LABEL[r['modelo']]:<12} {r['condicion']:<9} AUC {r['auc']:.4f} ± {r['auc_sd']:.4f}  logloss {r['logloss']:.4f}"
        if r["condicion"] != r["referencia"]:
            line += (f"   ΔAUC {r['d_auc']:+.4f} [{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}]"
                     f"  Δlogloss {r['d_logloss']:+.4f} [{r['d_logloss_lo']:+.4f}, {r['d_logloss_hi']:+.4f}]")
        print(line)
    print(f"Tablas: {OUT_RUNS.name}, {OUT.name}; figura: {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
