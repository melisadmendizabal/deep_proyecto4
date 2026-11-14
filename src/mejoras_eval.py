"""
Iteración posterior a B10: dos mejoras baratas del modelo, medidas en AUC/log-loss Y en valor.

  A. Ensamble de semillas: promedio de las probabilidades de las 3 semillas (la app ya carga las 3,
     pero la curva sale solo de s42). Se recalibra con una temperatura ajustada en val.
  B. M6: Transformer causal, sin y con campeones (draft). Responde dos preguntas de la guía:
     "¿Transformer vs GRU?" (M6 vs M4) y "¿importan los campeones?" (M6+draft vs M6).

Protocolo (sin modificar B5/B10, se importan sus funciones):
  - Métricas técnicas: test, todas las partidas; Δ PAREADOS vs M4 s42 (bootstrap sobre partidas).
  - Suavidad: salto medio |p_t − p_{t−1}| por minuto en test (menos saltos = menos clips falsos).
  - Valor: anotaciones de A y B, ±2 min, shift 1, θ* elegido en VAL por modelo, valor en TEST,
    IC 95 % y Δ pareados por bootstrap (igual que src/value_model.py).
NO sobrescribe resultados anteriores: escribe results/mejoras.md y results/mejoras_*.csv

Uso:
  python src/mejoras_eval.py
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
from src.evaluate import Rows, compare_runs, evaluate_set, load_predictions  # noqa: E402
from src.moments import DEFAULTS  # noqa: E402

RUNS, TRAIN = ROOT / "results" / "runs", ROOT / "models" / "_train"
OUT_MD = ROOT / "results" / "mejoras.md"


def run_dir(name: str) -> Path:
    """La carpeta del run con predictions.npz (results/runs o models/_train, donde se reentrenó)."""
    for base in (RUNS, TRAIN):
        if (base / name / "predictions.npz").exists():
            return base / name
    raise FileNotFoundError(f"No hay predictions.npz de {name}")


# ------------------------------------------------------------------ ensamble
def _match_logloss(p: np.ndarray, y: np.ndarray) -> float:
    """Log-loss promediado por partida (como la pérdida de entrenamiento)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y[:, None] * np.log(p) + (1 - y[:, None]) * np.log(1 - p))
    return float(np.nanmean(np.nanmean(ll, axis=1)))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """T que minimiza el log-loss de val de sigmoid(z / T) (búsqueda dorada en [0.5, 2])."""
    f = lambda t: _match_logloss(1 / (1 + np.exp(-z / t)), y)  # noqa: E731
    a, b = 0.5, 2.0
    g = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - g * (b - a), a + g * (b - a)
        a, b = (a, d) if f(c) < f(d) else (c, b)
    return (a + b) / 2


def make_ensemble(members: list[str], out: Path) -> Path:
    """Promedio de probabilidades de varias semillas + recalibración en val -> out/predictions.npz."""
    preds = [load_predictions(run_dir(m)) for m in members]
    for p in preds[1:]:
        if not np.array_equal(p["match_id"], preds[0]["match_id"]):
            raise ValueError("Las semillas no tienen las mismas partidas en el mismo orden")
    mean = np.mean([p["probs"].astype(np.float64) for p in preds], axis=0)
    z = np.log(np.clip(mean, 1e-6, 1 - 1e-6) / np.clip(1 - mean, 1e-6, 1))
    val = preds[0]["split"] == "val"
    T = fit_temperature(z[val], preds[0]["y"][val].astype(np.float64))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", match_id=preds[0]["match_id"], split=preds[0]["split"],
                        y=preds[0]["y"], probs=(1 / (1 + np.exp(-z / T))).astype(np.float32),
                        temperature=np.float32(T))
    return out


# ------------------------------------------------------------------ métricas técnicas
def smoothness(probs: np.ndarray) -> float:
    """Salto medio por minuto |p_t − p_{t−1}| (promedio por partida)."""
    d = np.abs(np.diff(probs, axis=1))
    return float(np.nanmean(np.nanmean(d, axis=1)))


def technical(models: list[tuple[str, Path]], ref: str, n_boot: int) -> pd.DataFrame:
    rows = []
    for name, run in models:
        p = load_predictions(run)
        te = p["split"] == "test"
        g = evaluate_set(p["probs"][te], p["y"][te], n_boot=0).set_index("tramo").loc["global"]
        rows.append({"modelo": name, "auc": g["auc"], "logloss": g["logloss"], "brier": g["brier"],
                     "ece": g["ece"], "salto_medio": smoothness(p["probs"][te])})
    tab = pd.DataFrame(rows).set_index("modelo")
    names = [n for n, _ in models]
    i = names.index(ref)
    order = [models[i]] + [m for j, m in enumerate(models) if j != i]
    cmp = compare_runs([r for _, r in order], [n for n, _ in order], n_boot=n_boot)
    cmp = cmp[cmp["tramo"] == "global"].set_index("modelo")
    for m in ("auc", "logloss"):
        tab[f"d_{m}"], tab[f"d_{m}_lo"], tab[f"d_{m}_hi"] = cmp[f"d_{m}"], cmp[f"d_{m}_lo"], cmp[f"d_{m}_hi"]
    return tab


# ------------------------------------------------------------------ valor
def value(models: list[tuple[str, Path | None]], refs: list[str], n_boot: int):
    P, _ = vm.load_params()
    tol, shift = int(P.get("match_window_min", 2)), 1
    units = vm.load_annotations()
    sel = pd.read_csv(vm.ANN_DIR / "seleccion.csv")
    units = units.merge(sel[["match_id", "split", "duration_min"]], on="match_id", how="left")
    units = units[units["split"].isin(["val", "test"])].rename(columns={"duration_min": "D"})
    ut = vm.unit_table(units, models, tol, shift, DEFAULTS["top_k"], DEFAULTS["merge_gap"])
    agg = vm.aggregate(ut, P)
    theta_star = vm.choose_theta(agg, "val")
    tab = vm.results_at(agg, theta_star, "test")
    tab["valor_val"] = vm.results_at(agg, theta_star, "val")["valor_mes"]
    for ref in refs:
        b = vm.bootstrap(ut, theta_star, P, "test", ref, n_boot=n_boot)
        tab["valor_lo"], tab["valor_hi"] = b["valor_lo"], b["valor_hi"]
        tab[f"d_vs_{ref}"], tab[f"d_vs_{ref}_lo"], tab[f"d_vs_{ref}_hi"] = b["delta_ref"], b["delta_lo"], b["delta_hi"]
    n = units.drop_duplicates("match_id")["split"].value_counts().to_dict()
    return tab.reindex([m for m, _ in models] + [vm.ORACLE]), n


# ------------------------------------------------------------------ reporte
def _ci(v, lo, hi, d=4, signed=True):
    if pd.isna(v):
        return "—"
    f = f"{{:{'+' if signed else ''}.{d}f}}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m6", default="m6_final", help="prefijo de las corridas de M6 sin campeones")
    ap.add_argument("--m6d", default="m6d_final", help="prefijo de las corridas de M6 con campeones")
    ap.add_argument("--m4d", default="m4d_hp__lr0.001_d0.3_s{s}",
                    help="corridas de M4 + campeones ({s} = semilla); config elegida por el promedio de val "
                         "de 3 semillas en results/hp/m4d_hp.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    seeds = (42, 43, 44)

    ens = {
        "M4 ens3": make_ensemble([f"gru_final__s{s}" for s in seeds], RUNS / "gru_ens3"),
        "M5 ens3": make_ensemble([f"tcn_final__s{s}" for s in seeds], RUNS / "tcn_ens3"),
        "M6 ens3": make_ensemble([f"{args.m6}__s{s}" for s in seeds], RUNS / f"{args.m6}_ens3"),
        "M6+draft ens3": make_ensemble([f"{args.m6d}__s{s}" for s in seeds], RUNS / f"{args.m6d}_ens3"),
        "M4+draft ens3": make_ensemble([args.m4d.format(s=s) for s in seeds], RUNS / "m4d_ens3"),
    }
    models = [("M0", run_dir("m0_oro")), ("M2", run_dir("m2_lgbm")),
              ("M4", run_dir("gru_final__s42")), ("M4 ens3", ens["M4 ens3"]),
              ("M4+draft", run_dir(args.m4d.format(s=42))), ("M4+draft ens3", ens["M4+draft ens3"]),
              ("M5", run_dir("tcn_final__s42")), ("M5 ens3", ens["M5 ens3"]),
              ("M6", run_dir(f"{args.m6}__s42")), ("M6 ens3", ens["M6 ens3"]),
              ("M6+draft", run_dir(f"{args.m6d}__s42")), ("M6+draft ens3", ens["M6+draft ens3"])]

    print("métricas técnicas (test)...")
    tech = technical(models, "M4", args.n_boot)
    # semillas: media ± sd de AUC por familia (¿la diferencia es mayor que el ruido entre semillas?)
    seed_rows = []
    for fam, tmpl in (("M4", "gru_final__s{s}"), ("M4+draft", args.m4d), ("M5", "tcn_final__s{s}"),
                      ("M6", args.m6 + "__s{s}"), ("M6+draft", args.m6d + "__s{s}")):
        aucs = []
        for s in seeds:
            p = load_predictions(run_dir(tmpl.format(s=s)))
            te = p["split"] == "test"
            aucs.append(Rows(p["probs"][te], p["y"][te]).metrics()["auc"])
        seed_rows.append({"modelo": fam, "auc_media": np.mean(aucs), "auc_sd": np.std(aucs, ddof=1)})
    seeds_tab = pd.DataFrame(seed_rows).set_index("modelo")

    print("valor (anotaciones)...")
    vmodels = [("R0", None)] + models
    val_tab, n = value(vmodels, ["R0", "M4"], args.n_boot)

    tech.to_csv(ROOT / "results" / "mejoras_tecnicas.csv", float_format="%.5f")
    val_tab.to_csv(ROOT / "results" / "mejoras_valor.csv", float_format="%.4f")

    L = ["# Mejoras: ensamble de semillas (A) y M6 Transformer ± campeones (B)", "",
         "*`python src/mejoras_eval.py`. Iteración posterior a B10: se propuso después de ver que la escalera "
         "M1–M5 quedó plana en AUC y que el valor de M4 empata con R0. Mismos datos, splits y protocolo de "
         "B5/B10; no sobrescribe sus resultados.*", "",
         "**Modelos.** M4/M5 = GRU/TCN finales (semilla 42, los pesos de la app). *ens3* = promedio de las 3 "
         "semillas, recalibrado con una temperatura en val. M6 = Transformer causal (sin campeones); "
         "M6+draft = el mismo + embeddings de los 10 campeones. Configuración de M6 elegida en val "
         "(`results/hp/m6_hp.csv`). M4+draft = la GRU de M4 + los mismos embeddings de campeones "
         "(`src/models/draft.py`); lr y dropout elegidos por el promedio de val de 3 semillas "
         "(`results/hp/m4d_hp.csv`).", "",
         "## Métricas técnicas (test, todas las partidas)", "",
         "| Modelo | AUC | Log-loss | ECE | Salto medio/min | ΔAUC vs M4 [IC 95 %] | ΔLog-loss vs M4 [IC 95 %] |",
         "|---|---|---|---|---|---|---|"]
    for m, r in tech.iterrows():
        L.append(f"| {m} | {r['auc']:.4f} | {r['logloss']:.4f} | {r['ece']:.4f} | {r['salto_medio']:.4f} | "
                 f"{'—' if m == 'M4' else _ci(r['d_auc'], r['d_auc_lo'], r['d_auc_hi'])} | "
                 f"{'—' if m == 'M4' else _ci(r['d_logloss'], r['d_logloss_lo'], r['d_logloss_hi'])} |")
    L += ["", "AUC por semilla (test): " + " · ".join(
        f"{m} {r['auc_media']:.4f} ± {r['auc_sd']:.4f}" for m, r in seeds_tab.iterrows()), "",
        f"## Valor (test: {n.get('test', 0)} partidas anotadas; θ* elegido en val: {n.get('val', 0)})", "",
        "| Detector | θ* | Clips/partida | Precisión | Recall | F1 | Ahorro (min) | Valor USD/mes [IC 95 %] | Δ vs M4 | Δ vs R0 | Valor en val |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in val_tab.iterrows():
        if pd.isna(r.get("valor_mes")):
            continue
        d4 = "—" if m == "M4" else _ci(r["d_vs_M4"], r["d_vs_M4_lo"], r["d_vs_M4_hi"], 0)
        d0 = "—" if m == "R0" else _ci(r["d_vs_R0"], r["d_vs_R0_lo"], r["d_vs_R0_hi"], 0)
        L.append(f"| {m} | {r['theta']:.2f} | {r['clips_por_partida']:.1f} | {r['precision']:.2f} | "
                 f"{r['recall']:.2f} | {r['f1']:.2f} | {r['ahorro']:.1f} | "
                 f"{_ci(r['valor_mes'], r['valor_lo'], r['valor_hi'], 0, signed=False)} | {d4} | {d0} | "
                 f"{r['valor_val']:.0f} |")
    L += ["", "*Valor: con pocas partidas anotadas los IC son anchos; una diferencia cuyo IC incluye 0 no es "
          "significativa.*", ""]
    # la sección "## Conclusiones" se escribe a mano: se conserva si se vuelve a correr el script
    keep = ""
    if OUT_MD.exists():
        prev = OUT_MD.read_text(encoding="utf-8")
        if "## Conclusiones" in prev:
            keep = "\n" + prev[prev.index("## Conclusiones"):]
    OUT_MD.write_text("\n".join(L) + keep, encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {OUT_MD.relative_to(ROOT)}, results/mejoras_tecnicas.csv, results/mejoras_valor.csv")


if __name__ == "__main__":
    main()
