"""
Iteración posterior a B10: ¿un detector HÍBRIDO (GRU + regla R0) genera más valor que cada uno?

  M4+R0: primero los momentos de la GRU con umbral θ (los más fuertes primero); después se agregan
         los minutos de la regla R0 (barón, ace, inhibidor, swing de oro > 1,500) que no caigan a
         menos de merge_gap minutos de uno ya elegido; tope top_k por partida.
         La GRU tiene prioridad porque es el núcleo del producto (curva + explicación).

Mismo protocolo que src/value_model.py (se importan sus funciones, sin modificarlas):
anotaciones de A y B, emparejamiento a ±2 min con shift 1, θ* elegido en VAL, valor reportado
en TEST, IC 95 % y Δ pareados por bootstrap sobre partidas, sensibilidad re-eligiendo θ*.
NO sobrescribe los resultados de B10: escribe results/b10_hibrido.*

Nota honesta: este detector se propuso DESPUÉS de ver los resultados de B10 (R0 ≈ M0 ≈ M4 en
valor). Se reporta como una iteración más (sección 4.5), no como el plan original.

Uso:
  python src/hybrid_eval.py
  python src/hybrid_eval.py --gru-run results/runs/gru_final__s42      # predicciones de la GRU
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import value_model as vm  # noqa: E402
from src.moments import DEFAULTS, detect_moments, match_moments  # noqa: E402

HYB = "M4+R0"
OUT_MD = vm.RES / "b10_hibrido.md"
OUT_CSV = vm.RES / "b10_hibrido.csv"


def hybrid_detections(probs: np.ndarray, r0_minutes: list[int], theta: float, top_k: int,
                      merge_gap: int) -> list[int]:
    """Minutos (frame) del detector híbrido para una partida."""
    moms = sorted(detect_moments(probs, theta=theta, merge_gap=merge_gap, top_k=top_k),
                  key=lambda d: -d.score)
    chosen = [d.minute for d in moms]
    for r in r0_minutes:
        if len(chosen) >= top_k:
            break
        if all(abs(r - c) >= merge_gap for c in chosen):
            chosen.append(int(r))
    return sorted(chosen)


def hybrid_unit_table(units: pd.DataFrame, gru_run: Path, tol: int, shift: int, top_k: int,
                      merge_gap: int) -> pd.DataFrame:
    mids = units["match_id"].unique().tolist()
    r0 = vm.r0_detections(mids, top_k, merge_gap)
    pred = vm.load_preds(gru_run)
    idx = {m: i for i, m in enumerate(pred["match_id"])}
    rows = []
    for th in vm.THETAS:
        det = {m: hybrid_detections(pred["probs"][idx[m]], r0[m], th, top_k, merge_gap)
               for m in mids if m in idx and m in r0}
        for u in units.itertuples():
            if u.match_id in det:
                r = match_moments([d - shift for d in det[u.match_id]], u.minutos, tol=tol)
                rows.append((HYB, th, u.anotador, u.match_id, r["tp"], r["fp"], r["fn"]))
    df = pd.DataFrame(rows, columns=["modelo", "theta", "anotador", "match_id", "tp", "fp", "fn"])
    meta = units.drop_duplicates("match_id")[["match_id", "split", "D"]]
    df = df.groupby(["modelo", "theta", "match_id"], as_index=False)[["tp", "fp", "fn"]].mean()
    return df.merge(meta, on="match_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gru-run", help="carpeta con predictions.npz de la GRU (default: gru_final__s42; "
                                      "si no existe, models/_train/gru_final__s42)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--theta-max", type=float, default=0.60,
                    help="grilla de θ: 0.04..theta_max en pasos de 0.01 (B10 usa hasta 0.30; con 0.30 el "
                         "θ* del híbrido queda en el borde, por eso se amplía)")
    args = ap.parse_args()
    vm.THETAS = np.round(np.arange(0.04, args.theta_max + 1e-9, 0.01), 2)

    gru = Path(args.gru_run) if args.gru_run else vm.RES / "runs" / "gru_final__s42"
    if not (gru / "predictions.npz").exists():
        gru = ROOT / "models" / "_train" / "gru_final__s42"
    print(f"GRU: {gru.relative_to(ROOT)}")
    P, ranges = vm.load_params()
    tol, shift = int(P.get("match_window_min", 2)), 1
    top_k, gap = DEFAULTS["top_k"], DEFAULTS["merge_gap"]

    units = vm.load_annotations()
    sel = pd.read_csv(vm.ANN_DIR / "seleccion.csv")
    units = units.merge(sel[["match_id", "split", "duration_min"]], on="match_id", how="left")
    units = units[units["split"].isin(["val", "test"])].rename(columns={"duration_min": "D"})
    print(f"{len(units)} anotaciones, {units['match_id'].nunique()} partidas "
          f"{units.drop_duplicates('match_id')['split'].value_counts().to_dict()}")

    models = [("R0", None), ("M0", vm.RES / "runs" / "m0_oro"), ("M4", gru)]
    ut = pd.concat([vm.unit_table(units, models, tol, shift, top_k, gap),
                    hybrid_unit_table(units, gru, tol, shift, top_k, gap)], ignore_index=True)
    agg = vm.aggregate(ut, P)
    theta_star = vm.choose_theta(agg, "val")
    cols = ["theta", "clips_por_partida", "precision", "recall", "f1", "ahorro", "ahorro_pct", "valor_mes"]
    order = ["R0", "M0", "M4", HYB, vm.ORACLE]
    val_tab = vm.results_at(agg, theta_star, "val").reindex(order)
    test_tab = vm.results_at(agg, theta_star, "test").reindex(order)
    deltas = {ref: vm.bootstrap(ut, theta_star, P, "test", ref, n_boot=args.n_boot) for ref in ("M0", "M4", "R0")}
    sens = vm.sensitivity(ut, P, ranges, HYB, "val", "test")
    wins = sens["ganador"].value_counts().to_dict()

    out = test_tab[cols].copy()
    out["valor_lo"], out["valor_hi"] = deltas["M0"]["valor_lo"], deltas["M0"]["valor_hi"]
    for ref, b in deltas.items():
        out[f"d_vs_{ref}"], out[f"d_vs_{ref}_lo"], out[f"d_vs_{ref}_hi"] = b["delta_ref"], b["delta_lo"], b["delta_hi"]
    out["valor_val"] = val_tab["valor_mes"]
    out.to_csv(OUT_CSV, float_format="%.4f")

    def ci(m, ref):
        b = deltas[ref]
        if m == ref or m not in b.index or pd.isna(b.loc[m, "delta_ref"]):
            return "—"
        return f"{b.loc[m, 'delta_ref']:+.0f} [{b.loc[m, 'delta_lo']:+.0f}, {b.loc[m, 'delta_hi']:+.0f}]"

    n_val = int((units.drop_duplicates("match_id")["split"] == "val").sum())
    n_test = int((units.drop_duplicates("match_id")["split"] == "test").sum())
    L = ["# Iteración: detector híbrido GRU + R0", "",
         f"*`python src/hybrid_eval.py`. Mismo protocolo que B10 (`src/value_model.py`): θ* elegido en val "
         f"({n_val} partidas), valor en test ({n_test} partidas), parámetros base de `configs/value.yaml`, "
         f"IC 95 % por bootstrap pareado sobre partidas. GRU: `{gru.relative_to(ROOT).as_posix()}`.*", "",
         "**M4+R0** = momentos de la GRU (θ) + eventos de la regla R0 que no estén a menos de 2 min de uno ya "
         "elegido; máximo 5 por partida. Se propuso **después** de ver B10: es una iteración, no el plan original.", "",
         "| Detector | θ* | Clips/partida | Precisión | Recall | F1 | Ahorro (min) | Valor USD/mes [IC 95 %] | Δ vs M0 | Δ vs M4 | Δ vs R0 | Valor en val |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        if m not in out.index or pd.isna(out.loc[m, "valor_mes"]):
            continue
        r = out.loc[m]
        th = "—" if m in ("R0", vm.ORACLE) else f"{r['theta']:.2f}"
        val = "" if pd.isna(r["valor_lo"]) else f" [{r['valor_lo']:.0f}, {r['valor_hi']:.0f}]"
        L.append(f"| {'**' + m + '**' if m == HYB else m} | {th} | {r['clips_por_partida']:.1f} | {r['precision']:.2f} | "
                 f"{r['recall']:.2f} | {r['f1']:.2f} | {r['ahorro']:.1f} | {r['valor_mes']:.0f}{val} | "
                 f"{ci(m, 'M0')} | {ci(m, 'M4')} | {ci(m, 'R0')} | {r['valor_val']:.0f} |")
    # de dónde salen los clips del híbrido con su θ*
    th_h = float(theta_star[HYB])
    mids = units["match_id"].unique().tolist()
    r0 = vm.r0_detections(mids, top_k, gap)
    pred = vm.load_preds(gru)
    idx = {m: i for i, m in enumerate(pred["match_id"])}
    n_gru = n_r0 = 0
    for m in mids:
        if m in idx and m in r0:
            gm = {d.minute for d in detect_moments(pred["probs"][idx[m]], theta=th_h, merge_gap=gap, top_k=top_k)}
            hy = hybrid_detections(pred["probs"][idx[m]], r0[m], th_h, top_k, gap)
            n_gru += sum(x in gm for x in hy)
            n_r0 += sum(x not in gm for x in hy)
    curve = agg[agg["modelo"] == HYB].pivot(index="theta", columns="split", values="valor_mes")
    r0_test = float(test_tab.loc["R0", "valor_mes"])
    L += ["", f"**Grilla de θ:** 0.04–{args.theta_max:.2f} (B10 usa hasta 0.30). Con θ alto, la GRU aporta "
          f"pocos clips y el híbrido tiende a R0: con θ* = {th_h:.2f}, de los {n_gru + n_r0} clips del híbrido en "
          f"las {len(mids)} partidas, **{n_gru} vienen de la GRU ({100 * n_gru / max(n_gru + n_r0, 1):.0f} %)** y "
          f"{n_r0} de R0.", "",
          "| θ | " + " | ".join(f"{t:.2f}" for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)) + " | R0 solo |",
          "|---|" + "---|" * 7,
          "| Valor en val | " + " | ".join(f"{curve.loc[t, 'val']:.0f}" for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50))
          + f" | {float(val_tab.loc['R0', 'valor_mes']):.0f} |",
          "| Valor en test | " + " | ".join(f"{curve.loc[t, 'test']:.0f}" for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50))
          + f" | {r0_test:.0f} |", "",
          f"**Sensibilidad** ({len(sens)} escenarios: rango de cada parámetro, ±20 %, ±50 %, pesimista y "
          f"optimista; θ* re-elegido en cada uno). Quién gana: "
          + ", ".join(f"{k} {v}" for k, v in sorted(wins.items(), key=lambda kv: -kv[1])) + ".", ""]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nval (selección de θ*):\n{val_tab[cols].round(3).to_string()}")
    print(f"\ntest:\n{out[cols + ['d_vs_M0', 'd_vs_M0_lo', 'd_vs_M0_hi', 'd_vs_M4', 'd_vs_M4_lo', 'd_vs_M4_hi', 'd_vs_R0', 'd_vs_R0_lo', 'd_vs_R0_hi']].round(2).to_string()}")
    print(f"\nganadores en sensibilidad: {wins}")
    print(f"-> {OUT_MD.relative_to(ROOT)}, {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
