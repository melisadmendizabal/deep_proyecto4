"""
B5 - Evaluación común para TODOS los modelos (M0-M6). Lee el predictions.npz que dejan
src/baselines.py y src/train.py (probs [N, T] calibradas, NaN = padding; match_id, split, y).

Uso:
  python src/evaluate.py --run results/runs/m0_oro results/runs/m2_lgbm --name M0 M2
  python src/evaluate.py --run results/runs/gru_base_2026... --name M4
  python src/evaluate.py --figures              # rehace las figuras con todo lo evaluado

Métricas (guía, sección B5), sobre filas (partida, minuto) válidas:
  auc, logloss (por fila), logloss_partida (promedio por partida = val_loss de train.py),
  brier, ece (10 bins), accuracy (umbral 0.5)
  - global y por tramo de minuto: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30+
  - IC 95 % por bootstrap sobre PARTIDAS (no filas), 1,000 réplicas
  - val y test, cada uno con todas las partidas y sin AFK (guía 3.2)

Salidas:
  results/b5_eval/<name>.csv              todas las métricas (split x subconjunto x tramo) con IC
  results/b5_eval/<name>_calibracion.csv  curva de calibración (test, 10 bins)
  results/b5_metricas.csv                 tabla maestra: una fila por modelo y split (global)
  results/b5_comparacion_vs_<ref>.csv     con 2+ runs: diferencias PAREADAS vs el primero, con IC
  results/figures/b5_*.png                calibración y AUC por tramo de todos los modelos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "b5_eval"
MASTER = ROOT / "results" / "b5_metricas.csv"
FIG = ROOT / "results" / "figures"
TRAMOS = [(0, 5, "0-4"), (5, 10, "5-9"), (10, 15, "10-14"), (15, 20, "15-19"),
          (20, 25, "20-24"), (25, 30, "25-29"), (30, 10_000, "30+")]
METRICS = ["auc", "logloss", "logloss_partida", "brier", "ece", "accuracy"]
N_BINS = 10
EPS = 1e-7


# ------------------------------------------------------------------ métricas ponderadas
class Rows:
    """Filas válidas de un conjunto de partidas, listas para métricas con pesos de bootstrap."""

    def __init__(self, probs: np.ndarray, y: np.ndarray, minutes_from: int = 0, minutes_to: int = 10_000):
        valid = ~np.isnan(probs)
        t = np.broadcast_to(np.arange(probs.shape[1]), probs.shape)
        valid &= (t >= minutes_from) & (t < minutes_to)
        self.match, self.minute = np.nonzero(valid)
        self.n_matches_total = len(y)
        self.p = np.clip(probs[valid].astype(np.float64), EPS, 1 - EPS)
        self.y = y[self.match].astype(np.float64)
        self.ll = -(self.y * np.log(self.p) + (1 - self.y) * np.log(1 - self.p))
        uniq, self.inv = np.unique(self.p, return_inverse=True)
        self.U = len(uniq)
        self.bin = np.minimum((self.p * N_BINS).astype(int), N_BINS - 1)
        cnt = np.bincount(self.match, minlength=len(y))
        self.has = cnt > 0
        self.ll_match = np.divide(np.bincount(self.match, self.ll, minlength=len(y)), cnt,
                                  out=np.zeros(len(y)), where=cnt > 0)

    def __len__(self):
        return len(self.p)

    def metrics(self, counts: np.ndarray | None = None) -> dict:
        """counts[m] = cuántas veces entra la partida m (None = 1 cada una)."""
        if counts is None:
            counts = np.ones(self.n_matches_total)
        w = counts[self.match]
        W = w.sum()
        if W == 0:
            return {k: np.nan for k in METRICS}
        out = {}
        pos = np.bincount(self.inv, w * self.y, minlength=self.U)
        neg = np.bincount(self.inv, w * (1 - self.y), minlength=self.U)
        P, N = pos.sum(), neg.sum()
        out["auc"] = float((pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (P * N)) if P > 0 and N > 0 else np.nan
        out["logloss"] = float((w * self.ll).sum() / W)
        cm = counts * self.has
        out["logloss_partida"] = float((cm * self.ll_match).sum() / cm.sum())
        out["brier"] = float((w * (self.p - self.y) ** 2).sum() / W)
        bw = np.bincount(self.bin, w, minlength=N_BINS)
        bp = np.bincount(self.bin, w * self.p, minlength=N_BINS)
        by = np.bincount(self.bin, w * self.y, minlength=N_BINS)
        out["ece"] = float(np.abs(bp - by).sum() / W)
        out["accuracy"] = float((w * ((self.p >= 0.5) == (self.y == 1))).sum() / W)
        return out

    def calibration(self) -> pd.DataFrame:
        b = np.bincount(self.bin, minlength=N_BINS)
        return pd.DataFrame({
            "bin": np.arange(N_BINS), "desde": np.arange(N_BINS) / N_BINS,
            "hasta": (np.arange(N_BINS) + 1) / N_BINS, "n": b,
            "p_media": np.bincount(self.bin, self.p, minlength=N_BINS) / np.maximum(b, 1),
            "frac_gana_azul": np.bincount(self.bin, self.y, minlength=N_BINS) / np.maximum(b, 1),
        })


def evaluate_set(probs: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Métricas global + por tramo con IC 95 % (bootstrap sobre partidas, mismas réplicas para todo)."""
    parts = [("global", Rows(probs, y))] + [(name, Rows(probs, y, a, b)) for a, b, name in TRAMOS]
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = {name: [] for name, _ in parts}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for name, r in parts:
            if len(r):
                boots[name].append(r.metrics(counts))
    rows = []
    for name, r in parts:
        if not len(r):
            continue
        row = {"tramo": name, "n_partidas": int(r.has.sum()), "n_filas": len(r)}
        point = r.metrics()
        b = pd.DataFrame(boots[name])
        for m in METRICS:
            row[m] = point[m]
            row[f"{m}_lo"], row[f"{m}_hi"] = (b[m].quantile(0.025), b[m].quantile(0.975)) if n_boot else (np.nan, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ predicciones de un run
def load_predictions(run: str | Path) -> dict:
    run = Path(run)
    path = run if run.suffix == ".npz" else run / "predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def afk_lookup() -> pd.Series:
    q = ROOT / "results" / "b2_calidad_partidas.csv"
    return pd.read_csv(q, usecols=["match_id", "excluir_de_train"]).set_index("match_id")["excluir_de_train"]


def evaluate_run(run: str | Path, name: str, n_boot: int = 1000) -> pd.DataFrame:
    pred = load_predictions(run)
    afk = afk_lookup().reindex(pred["match_id"]).fillna(False).to_numpy().astype(bool)
    frames = []
    for split in ("val", "test"):
        for subset, keep in (("todas", np.ones(len(afk), bool)), ("sin_afk", ~afk)):
            sel = (pred["split"] == split) & keep
            if not sel.any():
                continue
            df = evaluate_set(pred["probs"][sel], pred["y"][sel], n_boot=n_boot)
            df.insert(0, "subconjunto", subset)
            df.insert(0, "split", split)
            df.insert(0, "modelo", name)
            frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"{name}.csv", index=False, float_format="%.5f")
    test = pred["split"] == "test"
    Rows(pred["probs"][test], pred["y"][test]).calibration().assign(modelo=name) \
        .to_csv(OUT / f"{name}_calibracion.csv", index=False, float_format="%.5f")
    _update_master(res)
    return res


def _update_master(res: pd.DataFrame) -> None:
    g = res[res["tramo"] == "global"].drop(columns="tramo")
    if MASTER.exists():
        old = pd.read_csv(MASTER)
        old = old[~old["modelo"].isin(g["modelo"].unique())]
        g = pd.concat([old, g], ignore_index=True)
    g.sort_values(["split", "subconjunto", "modelo"]).to_csv(MASTER, index=False, float_format="%.5f")


def compare_runs(runs: list, names: list[str], n_boot: int = 1000, seed: int = 0,
                 split: str = "test") -> pd.DataFrame:
    """Diferencias PAREADAS contra el primer modelo (mismas partidas en cada réplica bootstrap).

    Un IC de la diferencia que no cruza 0 = diferencia significativa. Que los IC de cada
    modelo se traslapen NO dice eso.
    """
    preds = [load_predictions(r) for r in runs]
    ref_ids = preds[0]["match_id"][preds[0]["split"] == split]
    grids = []
    for p in preds:
        sel = p["split"] == split
        order = pd.Series(np.arange(sel.sum()), index=p["match_id"][sel]).reindex(ref_ids)
        if order.isna().any():
            raise ValueError("Los runs no tienen las mismas partidas de test")
        grids.append((p["probs"][sel][order.to_numpy()], p["y"][sel][order.to_numpy()]))
    tramos = [("global", 0, 10_000)] + [(n, a, b) for a, b, n in TRAMOS]
    rows_by = {(i, t): Rows(g[0], g[1], a, b) for i, g in enumerate(grids) for t, a, b in tramos}
    n = len(ref_ids)
    rng = np.random.default_rng(seed)
    boots = {(i, t): [] for i in range(1, len(grids)) for t, _, _ in tramos}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for t, _, _ in tramos:
            ref = rows_by[(0, t)].metrics(counts)
            for i in range(1, len(grids)):
                m = rows_by[(i, t)].metrics(counts)
                boots[(i, t)].append({k: m[k] - ref[k] for k in METRICS})
    out = []
    for i in range(1, len(grids)):
        for t, _, _ in tramos:
            ref, m = rows_by[(0, t)].metrics(), rows_by[(i, t)].metrics()
            b = pd.DataFrame(boots[(i, t)])
            row = {"modelo": names[i], "referencia": names[0], "split": split, "tramo": t}
            for k in METRICS:
                row[f"d_{k}"] = m[k] - ref[k]
                row[f"d_{k}_lo"], row[f"d_{k}_hi"] = b[k].quantile(0.025), b[k].quantile(0.975)
            out.append(row)
    return pd.DataFrame(out)


def fmt_ci(row, m, digits=3) -> str:
    return f"{row[m]:.{digits}f} [{row[m + '_lo']:.{digits}f}, {row[m + '_hi']:.{digits}f}]"


# ------------------------------------------------------------------ figuras
def make_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1", "#d1477a", "#52514e"]
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2,
    })
    FIG.mkdir(parents=True, exist_ok=True)
    evals = sorted(p for p in OUT.glob("*.csv") if not p.stem.endswith("_calibracion"))
    if not evals:
        print("No hay evaluaciones en results/b5_eval/")
        return
    names = [p.stem for p in evals]
    color = {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}

    # AUC por tramo (test, todas) con IC
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [t[2] for t in TRAMOS]
    for p in evals:
        df = pd.read_csv(p)
        d = df[(df.split == "test") & (df.subconjunto == "todas") & (df.tramo != "global")]
        d = d.set_index("tramo").reindex(labels)
        x = np.arange(len(labels))
        ax.plot(x, d["auc"], marker="o", ms=4, color=color[p.stem], label=p.stem)
        ax.fill_between(x, d["auc_lo"], d["auc_hi"], color=color[p.stem], alpha=0.15, lw=0)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlabel("minuto de la partida")
    ax.set_ylabel("AUC (test)")
    ax.set_title("AUC por tramo de minuto (IC 95 % bootstrap por partida)")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "b5_auc_por_tramo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # calibración (test)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfecta")
    for n in names:
        c = OUT / f"{n}_calibracion.csv"
        if c.exists():
            d = pd.read_csv(c)
            d = d[d.n > 0]
            ax.plot(d["p_media"], d["frac_gana_azul"], marker="o", ms=4, color=color[n], label=n)
    ax.set_xlabel("probabilidad predicha (azul gana)")
    ax.set_ylabel("fracción observada")
    ax.set_title("Calibración en test")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "b5_calibracion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figuras: {FIG / 'b5_auc_por_tramo.png'}, {FIG / 'b5_calibracion.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="*", default=[], help="carpetas de results/runs (o .npz)")
    ap.add_argument("--name", nargs="*", default=[], help="nombre de cada modelo (M0, M2, M4...)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--figures", action="store_true", help="solo rehacer las figuras")
    args = ap.parse_args()
    if args.run and len(args.run) != len(args.name):
        raise SystemExit("--run y --name deben tener la misma cantidad de elementos")
    for run, name in zip(args.run, args.name):
        res = evaluate_run(run, name, args.n_boot)
        r = res[(res.split == "test") & (res.subconjunto == "todas") & (res.tramo == "global")].iloc[0]
        print(f"{name:>6} test  AUC {fmt_ci(r, 'auc')}  logloss {fmt_ci(r, 'logloss')}  "
              f"Brier {fmt_ci(r, 'brier')}  ECE {fmt_ci(r, 'ece')}  acc {r['accuracy']:.3f}")
    if len(args.run) > 1:
        cmp = compare_runs(args.run, args.name, args.n_boot)
        path = ROOT / "results" / f"b5_comparacion_vs_{args.name[0]}.csv"
        cmp.to_csv(path, index=False, float_format="%.5f")
        print(f"\nDiferencias pareadas vs {args.name[0]} (test, global) -> {path.name}")
        for _, r in cmp[cmp.tramo == "global"].iterrows():
            print(f"  {r['modelo']:>6}  ΔAUC {r['d_auc']:+.4f} [{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}]"
                  f"  Δlogloss {r['d_logloss']:+.4f} [{r['d_logloss_lo']:+.4f}, {r['d_logloss_hi']:+.4f}]")
    make_figures()


if __name__ == "__main__":
    main()
