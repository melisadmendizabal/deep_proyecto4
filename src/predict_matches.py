"""
Pasa partidas crudas (CSV de collect.py) por un modelo entrenado y compara sus momentos con
los vistos por una persona en la repetición (annotations/revisiones_momentos.csv).

Pipeline (el mismo de build_features.py + train.py):
  frames/events/participants  ->  features.match_to_features  ->  scaler.json (de train)
  -> columnas que usó el modelo (best.pt)  ->  modelo  ->  logits / temperatura  ->  P(gana azul)
  -> moments.detect_moments con el θ* que eligió B10 en val para ese modelo.

Sirve para partidas que NO están en sequences.npz (las propias, la de la demo). Es la base de
la inferencia de la app (B11).

Uso:
  python src/predict_matches.py                       # partidas de revisiones_momentos.csv, GRU
  python src/predict_matches.py --match NA1_5676698765 --run results/runs/gru_final__s42
  python src/predict_matches.py --check               # verifica contra predictions.npz (val/test)

Salidas: results/b10_propias.md, results/b10_propias.csv (curvas), results/figures/b10_propias.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from riot_data import config  # noqa: E402
from src.features import (EVENT_COLS, EVENT_TYPES_USED, FEATURE_NAMES, FRAME_COLS,  # noqa: E402
                          Scaler, match_to_features)
from src.models import build_model  # noqa: E402
from src.moments import DEFAULTS, detect_moments, match_moments, moments_from_minutes  # noqa: E402

RES = ROOT / "results"
REVIEWS = ROOT / "annotations" / "revisiones_momentos.csv"


# ------------------------------------------------------------------ datos crudos
def _sources() -> list[Path]:
    return sorted(config.CSV_DIR.glob("*_own")) + [config.DATA_DIR / "anotacion"]


def load_raw(match_ids: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    """{match_id: {frames, events, participants, matches}} buscando en data/csv/*_own y data/anotacion."""
    ids, out = set(match_ids), {}
    for src in _sources():
        need = ids - set(out)
        if not need or not (src / "frames.csv").exists():
            continue
        tabs = {}
        for name in ("frames", "events", "participants", "matches"):
            df = pd.read_csv(src / f"{name}.csv", low_memory=False)
            tabs[name] = df[df["match_id"].isin(need)]
        for mid in need & set(tabs["frames"]["match_id"]):
            out[mid] = {k: v[v["match_id"] == mid] for k, v in tabs.items()}
    missing = ids - set(out)
    if missing:
        raise SystemExit(f"No hay datos crudos de {sorted(missing)} (¿collect.py --riot-id?)")
    return out


def raw_features(tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ev = tabs["events"]
    ev = ev[ev["type"].isin(EVENT_TYPES_USED)][["match_id"] + EVENT_COLS]
    return match_to_features(tabs["frames"][["match_id"] + FRAME_COLS], ev, tabs["participants"])


# ------------------------------------------------------------------ modelo
class Predictor:
    def __init__(self, run: Path, scaler_path: Path = ROOT / "features" / "scaler.json"):
        ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        self.cols = [FEATURE_NAMES.index(n) for n in ck["feature_names"]]
        self.model = build_model(ck["config"]["model"], len(self.cols))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.temperature = float(ck.get("temperature", 1.0))
        self.scaler = Scaler.load(scaler_path)
        if self.scaler.feature_names != list(FEATURE_NAMES):
            raise SystemExit("scaler.json no coincide con FEATURE_NAMES de src/features.py")

    @torch.no_grad()
    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        """P(gana azul) calibrada para cada minuto de la partida."""
        x = self.scaler.transform(raw.to_numpy(dtype=np.float32))[:, self.cols]
        X = torch.as_tensor(x[None], dtype=torch.float32)
        mask = torch.ones(X.shape[:2], dtype=torch.bool)
        logits = self.model(X, mask).float().numpy()[0]
        return 1 / (1 + np.exp(-logits / self.temperature))


def theta_star(name: str, default: float = DEFAULTS["theta"]) -> float:
    """θ* que eligió value_model.py en val para ese modelo (o el default si no hay)."""
    path = RES / "b10_valor_theta.csv"
    if not path.exists():
        return default
    a = pd.read_csv(path)
    a = a[(a["modelo"] == name) & (a["split"] == "val")]
    if a.empty:
        return default
    return float(a.sort_values(["ahorro", "theta"], ascending=False)["theta"].iloc[0])


# ------------------------------------------------------------------ verificación
def check(run: Path, n: int = 10) -> None:
    """Las probabilidades desde los CSV crudos deben ser iguales a las de predictions.npz."""
    with np.load(run / "predictions.npz", allow_pickle=False) as z:
        pred = {k: z[k] for k in ("match_id", "probs")}
    have = set(pd.read_csv(config.DATA_DIR / "anotacion" / "frames.csv", usecols=["match_id"])["match_id"])
    ids = [m for m in pred["match_id"] if m in have][:n]
    raw = load_raw(ids)
    p = Predictor(run)
    worst = 0.0
    for mid in ids:
        mine = p.predict(raw_features(raw[mid]))
        ref = pred["probs"][list(pred["match_id"]).index(mid)]
        ref = ref[~np.isnan(ref)]
        k = min(len(mine), len(ref))
        worst = max(worst, float(np.abs(mine[:k] - ref[:k]).max()))
        if len(mine) != len(ref):
            print(f"  aviso {mid}: {len(mine)} minutos aquí vs {len(ref)} en predictions.npz")
    print(f"Verificación en {len(ids)} partidas de val/test: diferencia máxima {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit("El pipeline NO reproduce las predicciones del entrenamiento")


# ------------------------------------------------------------------ comparación
def r0_minutes(tabs, raw: pd.DataFrame) -> list[int]:
    from src.baselines import rule_r0
    ev = tabs["events"]
    mins = rule_r0(raw, ev)
    g = raw["gold_diff"].to_numpy()
    swing = np.abs(np.diff(g, prepend=g[0]))
    return moments_from_minutes(mins, scores=[swing[m] for m in mins])


def compare(match_ids, run: Path, name: str, shift: int, tol: int):
    raw_all = load_raw(match_ids)
    p = Predictor(run)
    th = theta_star(name)
    reviews = pd.read_csv(REVIEWS) if REVIEWS.exists() else pd.DataFrame(columns=["match_id"])
    rows, curves, cards = [], [], []
    for mid in match_ids:
        tabs = raw_all[mid]
        raw = raw_features(tabs)
        probs = p.predict(raw)
        moms = detect_moments(probs, theta=th)
        m = tabs["matches"].iloc[0]
        rv = reviews[(reviews["match_id"] == mid) & reviews["es_momento"].astype(bool)
                     & reviews["minuto"].notna()]
        human = rv["minuto"].astype(int).tolist()
        r_model = match_moments([x.minute - shift for x in moms], human, tol=tol)
        r_rule = match_moments([x - shift for x in r0_minutes(tabs, raw)], human, tol=tol)
        same_team = sum(
            next(x.team for x in moms if x.minute - shift == d) ==
            rv.loc[rv["minuto"].astype(int) == a, "equipo"].fillna("").iloc[0]
            for d, a in r_model["pairs"])
        rows.append({"match_id": mid, "duracion_min": round(m["duration_s"] / 60, 1),
                     "gana": "azul" if m["blue_win"] else "rojo", "theta": th,
                     "momentos_humanos": len(human), "momentos_modelo": len(moms),
                     "tp": r_model["tp"], "fp": r_model["fp"], "fn": r_model["fn"],
                     "mismo_equipo": same_team,
                     "r0_tp": r_rule["tp"], "r0_fp": r_rule["fp"], "r0_fn": r_rule["fn"]})
        curves.append(pd.DataFrame({"match_id": mid, "minuto": np.arange(len(probs)),
                                    "p_azul": probs.round(4)}))
        cards.append((mid, probs, moms, rv, m))
    return pd.DataFrame(rows), pd.concat(curves), cards, th


SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED = "#2a78d6", "#e34948"


def plot(cards, name: str, shift: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, axes = plt.subplots(len(cards), 1, figsize=(9, 2.6 * len(cards)), squeeze=False)
    for ax, (mid, probs, moms, rv, m) in zip(axes[:, 0], cards):
        t = np.arange(len(probs))
        for mo in moms:  # clip que propone el modelo (en minutos de jugada)
            ax.axvspan(mo.start_minute - shift - 0.5, mo.minute - shift + 0.5, color=GRID, zorder=0)
        ax.plot(t, probs, color=INK, linewidth=2, zorder=3)
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        for r in rv.itertuples():
            c = BLUE if r.equipo == "azul" else RED if r.equipo == "rojo" else INK2
            ax.plot([r.minuto], [1.04], marker="v", color=c, markersize=8, clip_on=False, zorder=4)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(probs) - 0.5)
        ax.set_ylabel("P(gana azul)")
        ax.set_title(f"{mid} · gana {'azul' if m['blue_win'] else 'rojo'} · "
                     f"{m['duration_s'] / 60:.0f} min", pad=12)
    axes[-1, 0].set_xlabel("minuto")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([], [], color=INK, linewidth=2, label=f"curva de {name}"),
                        Patch(color=GRID, label=f"clip propuesto por {name}"),
                        Line2D([], [], marker="v", color=BLUE, linestyle="", label="momento en la repetición (beneficia a azul)"),
                        Line2D([], [], marker="v", color=RED, linestyle="", label="momento en la repetición (beneficia a rojo)")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = RES / "figures" / "b10_propias.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def report(df: pd.DataFrame, cards, name: str, th: float, shift: int, tol: int) -> Path:
    tot = df[["momentos_humanos", "tp", "fp", "fn", "mismo_equipo", "r0_tp", "r0_fp", "r0_fn"]].sum()

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        return p, r, (2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"))
    pm, rm, fm = prf(tot["tp"], tot["fp"], tot["fn"])
    pr, rr, fr = prf(tot["r0_tp"], tot["r0_fp"], tot["r0_fn"])
    L = ["# Partidas propias: modelo vs momentos vistos en la repetición", "",
         f"*`python src/predict_matches.py`. Modelo {name} con θ* = {th:.2f} (elegido en val por "
         f"B10), emparejado a ±{tol} min. Momentos humanos de `annotations/revisiones_momentos.csv` "
         "(solo `es_momento = True`). Estas partidas no están en el entrenamiento.*", "",
         "| Partida | Gana | Momentos en la repetición | Clips del modelo | Coinciden | FP | FN | R0: coinciden / clips |",
         "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        L.append(f"| {r.match_id} | {r.gana} | {r.momentos_humanos} | {r.momentos_modelo} | {r.tp} | "
                 f"{r.fp} | {r.fn} | {r.r0_tp} / {r.r0_tp + r.r0_fp} |")
    L += ["", f"**Total:** {name} precisión {pm:.2f}, recall {rm:.2f}, F1 {fm:.2f} "
          f"(en las coincidencias, mismo equipo {tot['mismo_equipo']}/{tot['tp']}). "
          f"R0: precisión {pr:.2f}, recall {rr:.2f}, F1 {fr:.2f}.", "",
          "> Son 3 partidas: sirve como ilustración y para la demo, no como métrica.", "",
          "![curvas](figures/b10_propias.png)", "", "## Detalle por partida", ""]
    for mid, probs, moms, rv, m in cards:
        L += [f"### {mid} (gana {'azul' if m['blue_win'] else 'rojo'})", "",
              f"Clips de {name} (minuto de la jugada):"]
        L += [f"- min {mo.start_minute - shift}–{mo.minute - shift}: P(azul) {mo.p_before:.2f} → "
              f"{mo.p_after:.2f} (favorece a {mo.team})" for mo in moms] or ["- (ninguno)"]
        L += ["", "Vistos en la repetición:"]
        L += [f"- min {int(r.minuto)}{'~' if r.minuto_aprox else ''} · {r.tipo} · "
              f"{r.equipo if isinstance(r.equipo, str) else '?'} · {r.nota}"
              for r in rv.itertuples()] or ["- (ninguno)"]
        L.append("")
    path = RES / "b10_propias.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", nargs="*", help="match_id (por defecto: las de revisiones_momentos.csv)")
    ap.add_argument("--run", default=str(RES / "runs" / "gru_final__s42"))
    ap.add_argument("--name", default="M4", help="nombre del modelo en b10_valor_theta.csv (para θ*)")
    ap.add_argument("--shift", type=int, default=1)
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    if args.check:
        check(run)
        return
    ids = args.match or list(dict.fromkeys(pd.read_csv(REVIEWS)["match_id"]))
    df, curves, cards, th = compare(ids, run, args.name, args.shift, args.tol)
    df.to_csv(RES / "b10_propias.csv", index=False)
    curves.to_csv(RES / "b10_propias_curvas.csv", index=False)
    plot(cards, args.name, args.shift)
    path = report(df, cards, args.name, th, args.shift, args.tol)
    print(df.to_string(index=False))
    print(f"\n-> {path.relative_to(ROOT)}, results/b10_propias.csv, results/b10_propias_curvas.csv, "
          "results/figures/b10_propias.png")


if __name__ == "__main__":
    main()
