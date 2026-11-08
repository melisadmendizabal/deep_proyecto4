"""
B8 - Detector de momentos decisivos a partir de la curva de probabilidad.

Idea: un momento decisivo es donde la probabilidad de victoria cambia de golpe.
  score(t) = max_w |P(t) - P(t-w)|,  w en `windows` (por defecto 1 y 2 minutos)
  - candidato si score(t) >= theta (el umbral = PUNTO DE OPERACIÓN del producto);
  - supresión de no-máximos: se toma el candidato más fuerte y se descartan los que
    estén a menos de `merge_gap` minutos de uno ya elegido (una pelea = un momento);
  - se quedan como máximo `top_k` momentos por partida.

El mismo post-proceso (unir + top-K) sirve para la regla sin ML R0 (`moments_from_minutes`),
así los modelos y R0 se comparan igual.

`match_moments` empareja detecciones con anotaciones humanas (±tol minutos) y devuelve
TP, FP, FN: es lo que usa el modelo de valor (B10).

Uso rápido (con las predicciones que deja train.py / baselines.py):
  python src/moments.py --run results/runs/<run> --match LA1_1234567890
  python src/moments.py --run results/runs/<run> --summary          # momentos por partida vs theta
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULTS = {"theta": 0.10, "windows": (1, 2), "merge_gap": 2, "top_k": 5}


@dataclass
class Moment:
    minute: int          # frame donde ya se ve el cambio (P(t))
    start_minute: int    # dónde empieza la jugada: t - w (el clip va de start_minute a minute)
    delta: float         # P(t) - P(t-w), con signo: + favorece a azul, - favorece a rojo
    score: float         # |delta|
    p_before: float
    p_after: float

    @property
    def team(self) -> str:
        return "azul" if self.delta > 0 else "rojo"


def _valid_curve(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    valid = ~np.isnan(p)
    if not valid.any():
        return p[:0]
    last = np.flatnonzero(valid)[-1]
    return p[: last + 1]


def score_curve(probs: np.ndarray, windows=DEFAULTS["windows"]):
    """Para cada minuto t: el mayor |P(t)-P(t-w)| y la ventana w que lo produce."""
    p = _valid_curve(probs)
    n = len(p)
    best = np.zeros(n)
    best_w = np.zeros(n, dtype=int)
    best_delta = np.zeros(n)
    for w in sorted(windows):
        if w >= n:
            continue
        d = np.zeros(n)
        d[w:] = p[w:] - p[:-w]
        better = np.abs(d) > best
        best[better], best_w[better], best_delta[better] = np.abs(d)[better], w, d[better]
    return p, best, best_w, best_delta


def _suppress(order: np.ndarray, minutes: np.ndarray, merge_gap: int, top_k: int | None) -> list[int]:
    chosen: list[int] = []
    for i in order:
        if all(abs(minutes[i] - minutes[j]) >= merge_gap for j in chosen):
            chosen.append(i)
            if top_k and len(chosen) >= top_k:
                break
    return chosen


def detect_moments(probs: np.ndarray, theta: float = DEFAULTS["theta"],
                   windows=DEFAULTS["windows"], merge_gap: int = DEFAULTS["merge_gap"],
                   top_k: int | None = DEFAULTS["top_k"]) -> list[Moment]:
    """Momentos de UNA partida. probs: P(gana azul) por minuto (NaN = la partida terminó)."""
    p, score, w, delta = score_curve(probs, windows)
    cand = np.flatnonzero(score >= theta)
    if len(cand) == 0:
        return []
    order = cand[np.argsort(-score[cand], kind="stable")]
    chosen = _suppress(order, np.arange(len(p)), merge_gap, top_k)
    out = []
    for t in chosen:
        # los candidatos vecinos del mismo signo son la misma jugada: el clip empieza en el primero
        same = [c for c in cand if abs(c - t) < merge_gap and np.sign(delta[c]) == np.sign(delta[t])]
        start = int(min(c - w[c] for c in same))
        out.append(Moment(minute=int(t), start_minute=start, delta=float(delta[t]),
                          score=float(score[t]), p_before=float(p[t - w[t]]), p_after=float(p[t])))
    return sorted(out, key=lambda m: m.minute)


def moments_from_minutes(minutes, scores=None, merge_gap: int = DEFAULTS["merge_gap"],
                         top_k: int | None = DEFAULTS["top_k"]) -> list[int]:
    """Mismo post-proceso para detectores que solo dan minutos (regla R0).

    scores: importancia de cada minuto (p. ej. |Δoro|); sin scores, gana el más temprano.
    """
    minutes = np.asarray(sorted(set(int(m) for m in minutes)))
    if len(minutes) == 0:
        return []
    s = np.ones(len(minutes)) if scores is None else np.asarray(scores, dtype=float)
    order = np.lexsort((minutes, -s))  # mayor score primero; empate: minuto más temprano
    chosen = _suppress(order, minutes, merge_gap, top_k)
    return sorted(int(minutes[i]) for i in chosen)


def match_moments(detected, annotated, tol: int = 2) -> dict:
    """Empareja detecciones con anotaciones (1 a 1, la más cercana primero, |Δ| <= tol).

    detected / annotated: listas de minutos (o Moments).
    Devuelve TP, FP, FN y los pares emparejados.
    """
    det = sorted(int(m.minute if isinstance(m, Moment) else m) for m in detected)
    ann = sorted(int(m) for m in annotated)
    pairs = sorted(((abs(d - a), i, j) for i, d in enumerate(det) for j, a in enumerate(ann)
                    if abs(d - a) <= tol))
    used_d, used_a, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in used_d and j not in used_a:
            used_d.add(i)
            used_a.add(j)
            matched.append((det[i], ann[j]))
    tp = len(matched)
    return {"tp": tp, "fp": len(det) - tp, "fn": len(ann) - tp, "pairs": sorted(matched)}


def detect_all(pred: dict, theta: float = DEFAULTS["theta"], split: str | None = None,
               **kwargs) -> pd.DataFrame:
    """Momentos de todas las partidas de un predictions.npz. Una fila por momento."""
    rows = []
    for mid, sp, probs in zip(pred["match_id"], pred["split"], pred["probs"]):
        if split and sp != split:
            continue
        for k, m in enumerate(detect_moments(probs, theta=theta, **kwargs)):
            rows.append({"match_id": mid, "split": sp, "rank_minute": k, **asdict(m)})
    return pd.DataFrame(rows)


def summary_by_theta(pred: dict, thetas=(0.05, 0.08, 0.10, 0.12, 0.15, 0.20), split="val",
                     **kwargs) -> pd.DataFrame:
    sel = pred["split"] == split
    out = []
    for th in thetas:
        counts = np.array([len(detect_moments(p, theta=th, **kwargs)) for p in pred["probs"][sel]])
        out.append({"theta": th, "momentos_por_partida": round(counts.mean(), 2),
                    "pct_partidas_sin_momentos": round(100 * (counts == 0).mean(), 1),
                    "p90_momentos": int(np.percentile(counts, 90))})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="carpeta de results/runs o un predictions.npz")
    ap.add_argument("--match", help="match_id para listar sus momentos")
    ap.add_argument("--summary", action="store_true", help="momentos por partida según theta (val)")
    ap.add_argument("--theta", type=float, default=DEFAULTS["theta"])
    ap.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    args = ap.parse_args()

    run = Path(args.run)
    with np.load(run if run.suffix == ".npz" else run / "predictions.npz", allow_pickle=False) as z:
        pred = {k: z[k] for k in z.files}
    if args.summary:
        print(summary_by_theta(pred, top_k=args.top_k).to_string(index=False))
    if args.match:
        idx = np.flatnonzero(pred["match_id"] == args.match)
        if not len(idx):
            raise SystemExit(f"{args.match} no está en las predicciones (solo hay val/test)")
        p = pred["probs"][idx[0]]
        print(f"{args.match} ({pred['split'][idx[0]]}), gana azul = {int(pred['y'][idx[0]])}")
        print("P(azul) por minuto:", " ".join(f"{v:.2f}" for v in _valid_curve(p)))
        for m in detect_moments(p, theta=args.theta, top_k=args.top_k):
            print(f"  min {m.start_minute:>2}-{m.minute:<2}  {m.p_before:.2f} -> {m.p_after:.2f}  "
                  f"({m.delta:+.2f}, favorece a {m.team})")


if __name__ == "__main__":
    main()
