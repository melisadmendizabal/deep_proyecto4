"""
B10 - Modelo de valor: cuánto tiempo (y dinero) le ahorra a un coach cada versión del modelo.

Por partida revisada (docs/supuestos_valor.md):
  T_base = f_rev × D                                     revisión completa (hoy)
  T_tool = t_setup + (TP + FP) × t_clip + FN × t_miss    revisión con el sistema
  Ahorro = T_base − T_tool                               minutos
Por mes y por equipo:
  Valor_mes = N_rev × (Ahorro / 60) × C_h − C_infra − N_rev × C_pred

TP / FP / FN: momentos del detector (src/moments.py) contra los anotados por personas
(annotations/A.csv, annotations/B.csv), emparejados a ±match_window_min minutos.

Decisiones:
  - Minutos: el detector marca el frame t en el que la curva YA cambió (el frame t resume lo
    que pasó entre t−1 y t); la persona anota el minuto en que ocurre la jugada. La detección
    se compara como t − shift (shift = 1 por defecto). Lo mismo para la regla R0.
  - θ* (umbral del detector) se elige POR MODELO con las partidas de val (máximo ahorro medio;
    en empate, el θ más alto = menos clips) y el valor se reporta en TEST con ese θ*.
  - Una partida anotada por A y B cuenta una vez: TP/FP/FN se promedian entre los dos.
  - IC 95 %: bootstrap por partida. Δ vs referencia: pareado (mismas partidas remuestreadas).
  - Partidas propias (split "propia"): no tienen predicciones en los predictions.npz, se omiten.

Uso:
  python src/value_model.py
  python src/value_model.py --min-conf 2           # sensibilidad: sin momentos de confianza 1
  python src/value_model.py --run results/runs/gru_final__s42 --name M4

Salidas:
  results/b10_valor_theta.csv    modelo × θ × split: TP, FP, FN, P, R, F1, ahorro, valor_mes
  results/b10_tornado.csv        valor del modelo de referencia moviendo cada parámetro (+ quién gana)
  results/b10_valor.md           resumen: tabla principal, sensibilidad, equilibrio
  results/figures/b10_valor_vs_theta.png, results/figures/b10_tornado.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.moments import DEFAULTS, detect_moments, match_moments, moments_from_minutes  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
ANN_DIR = ROOT / "annotations"
ANOT_DATA = ROOT / "data" / "anotacion"

DEFAULT_MODELS = [  # (nombre, carpeta en results/runs; None = regla R0)
    ("R0", None), ("M0", "m0_oro"), ("M1", "m1_logit"), ("M2", "m2_lgbm"),
    ("M3", "mlp_final__s42"), ("M4", "gru_final__s42"), ("M5", "tcn_final__s42"),
    ("M4 E6 λ0.1", "e6_sm0.1__s42"), ("M4 E6 λ0.5", "e6_sm0.5__s42"),
]
ORACLE = "Oráculo"
THETAS = np.round(np.arange(0.04, 0.301, 0.01), 2)
TORNADO_PARAMS = ["f_rev", "t_setup", "t_clip", "t_miss", "N_rev", "C_h_usd"]
PARAM_LABEL = {"f_rev": "f_rev (factor de revisión)", "t_setup": "t_setup (min)",
               "t_clip": "t_clip (min por clip)", "t_miss": "t_miss (min por momento perdido)",
               "N_rev": "N_rev (partidas al mes)", "C_h_usd": "C_h (USD/h del coach)"}


# ------------------------------------------------------------------ parámetros
def load_params(path: Path = ROOT / "configs" / "value.yaml") -> tuple[dict, dict]:
    """(valores base, rangos {param: (low, high)}) desde configs/value.yaml."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base, ranges = {}, {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            base[k] = float(v["base"])
            if "low" in v and "high" in v:
                ranges[k] = (float(v["low"]), float(v["high"]))
        else:
            base[k] = v
    return base, ranges


def ahorro_min(D, tp, fp, fn, P: dict):
    """Minutos ahorrados por partida (acepta escalares o arreglos)."""
    t_base = P["f_rev"] * D
    t_tool = P["t_setup"] + (tp + fp) * P["t_clip"] + fn * P["t_miss"]
    return t_base - t_tool


def valor_mes(ahorro, P: dict):
    return P["N_rev"] * ahorro / 60 * P["C_h_usd"] - P["C_infra_usd_mes"] - P["N_rev"] * P["C_pred_usd"]


# ------------------------------------------------------------------ anotaciones
def load_annotations(paths=None, min_conf: int = 1) -> pd.DataFrame:
    """Una fila por (anotador, partida) con la lista de minutos anotados.

    sin_momentos = True -> lista vacía. Se descartan los momentos con confianza < min_conf.
    """
    paths = paths or [ANN_DIR / f"{w}.csv" for w in ("A", "B")]
    parts = [pd.read_csv(p) for p in paths if Path(p).exists()]
    if not parts:
        raise SystemExit("No hay anotaciones (annotations/A.csv ni annotations/B.csv)")
    ann = pd.concat(parts, ignore_index=True)
    last = ann.groupby(["anotador", "match_id"])["guardado_utc"].transform("max")
    ann = ann[ann["guardado_utc"] == last]  # la herramienta ya reemplaza; por si acaso
    rows = []
    for (who, mid), g in ann.groupby(["anotador", "match_id"], sort=False):
        none = g["sin_momentos"].astype(str).str.lower().eq("true")
        m = g[~none]
        conf = pd.to_numeric(m["confianza"], errors="coerce").fillna(3)
        mins = sorted(pd.to_numeric(m.loc[conf >= min_conf, "minuto"]).astype(int).tolist())
        rows.append({"anotador": who, "match_id": mid, "minutos": mins,
                     "segundos": float(g["segundos"].iloc[0])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ detectores
def load_preds(run: Path) -> dict:
    from src.evaluate import load_predictions
    return load_predictions(run)


def r0_detections(match_ids: list[str], top_k: int, merge_gap: int) -> dict[str, list[int]]:
    """Regla R0 (baselines.rule_r0) sobre las partidas anotadas, con el mismo post-proceso."""
    from src.baselines import rule_r0
    from src.features import match_to_features
    if not (ANOT_DATA / "frames.csv").exists():
        from src.select_annotation import extract
        print("  extrayendo frames de las partidas anotadas (una sola vez, ~1-2 min)...")
        extract(set(pd.read_csv(ROOT / "annotations" / "seleccion.csv")["match_id"]), names=("frames",))
    ids = set(match_ids)
    frames = pd.read_csv(ANOT_DATA / "frames.csv")
    events = pd.read_csv(ANOT_DATA / "events.csv", low_memory=False)
    parts = pd.read_csv(ANOT_DATA / "participants.csv")
    fr_g, ev_g, pa_g = (d[d["match_id"].isin(ids)].groupby("match_id") for d in (frames, events, parts))
    ev_ids = set(events["match_id"])
    out = {}
    for mid in match_ids:
        if mid not in fr_g.groups:
            continue
        ev = ev_g.get_group(mid) if mid in ev_ids else events.iloc[:0]
        feats = match_to_features(fr_g.get_group(mid), ev, pa_g.get_group(mid))
        mins = rule_r0(feats, ev)
        g = feats["gold_diff"].to_numpy()
        swing = np.abs(np.diff(g, prepend=g[0]))
        out[mid] = moments_from_minutes(mins, scores=[swing[m] for m in mins],
                                        merge_gap=merge_gap, top_k=top_k)
    return out


def unit_table(units: pd.DataFrame, models: list[tuple[str, Path | None]], tol: int, shift: int,
               top_k: int, merge_gap: int) -> pd.DataFrame:
    """TP/FP/FN por (modelo, θ, partida), promediando entre anotadores."""
    rows = []
    mids = units["match_id"].unique().tolist()
    for name, run in models:
        if run is None:
            det = r0_detections(mids, top_k, merge_gap)
            per_theta = {th: det for th in THETAS}
        else:
            pred = load_preds(run)
            idx = {m: i for i, m in enumerate(pred["match_id"])}
            per_theta = {}
            for th in THETAS:
                per_theta[th] = {m: [d.minute for d in detect_moments(
                    pred["probs"][idx[m]], theta=th, merge_gap=merge_gap, top_k=top_k)]
                    for m in mids if m in idx}
        for th, det in per_theta.items():
            for u in units.itertuples():
                if u.match_id not in det:
                    continue
                r = match_moments([d - shift for d in det[u.match_id]], u.minutos, tol=tol)
                rows.append((name, th, u.anotador, u.match_id, r["tp"], r["fp"], r["fn"]))
        # techo: un detector perfecto (marca justo lo anotado)
    for th in THETAS:
        for u in units.itertuples():
            n = len(u.minutos)
            rows.append((ORACLE, th, u.anotador, u.match_id, n, 0, 0))
    df = pd.DataFrame(rows, columns=["modelo", "theta", "anotador", "match_id", "tp", "fp", "fn"])
    meta = units.drop_duplicates("match_id")[["match_id", "split", "D"]]
    df = df.groupby(["modelo", "theta", "match_id"], as_index=False)[["tp", "fp", "fn"]].mean()
    return df.merge(meta, on="match_id")


# ------------------------------------------------------------------ agregación
def aggregate(ut: pd.DataFrame, P: dict) -> pd.DataFrame:
    d = ut.assign(ahorro=ahorro_min(ut["D"], ut["tp"], ut["fp"], ut["fn"], P),
                  t_base=P["f_rev"] * ut["D"])
    g = d.groupby(["modelo", "theta", "split"])
    out = g.agg(n=("match_id", "size"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                t_base=("t_base", "mean"), ahorro=("ahorro", "mean")).reset_index()
    out["precision"] = out["tp"] / (out["tp"] + out["fp"]).replace(0, np.nan)
    out["recall"] = out["tp"] / (out["tp"] + out["fn"]).replace(0, np.nan)
    out["f1"] = 2 * out["tp"] / (2 * out["tp"] + out["fp"] + out["fn"]).replace(0, np.nan)
    out["clips_por_partida"] = (out["tp"] + out["fp"]) / out["n"]
    out["t_tool"] = out["t_base"] - out["ahorro"]
    out["ahorro_pct"] = 100 * out["ahorro"] / out["t_base"]
    out["valor_mes"] = valor_mes(out["ahorro"], P)
    return out


def choose_theta(agg: pd.DataFrame, select_split: str = "val") -> pd.Series:
    """θ* por modelo: máximo ahorro medio en select_split; empate -> θ más alto."""
    a = agg[agg["split"] == select_split].sort_values(["modelo", "ahorro", "theta"],
                                                      ascending=[True, False, False])
    return a.groupby("modelo")["theta"].first()


def results_at(agg: pd.DataFrame, theta_star: pd.Series, split: str) -> pd.DataFrame:
    a = agg[agg["split"] == split]
    a = a[a["theta"] == a["modelo"].map(theta_star)]
    return a.set_index("modelo")


def bootstrap(ut: pd.DataFrame, theta_star: pd.Series, P: dict, split: str, ref: str,
              n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """IC 95 % de valor_mes por modelo y Δ pareado vs ref (remuestreando partidas)."""
    d = ut[(ut["split"] == split) & (ut["theta"] == ut["modelo"].map(theta_star))]
    d = d.assign(ahorro=ahorro_min(d["D"], d["tp"], d["fp"], d["fn"], P))
    wide = d.pivot_table(index="match_id", columns="modelo", values="ahorro")
    if wide.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    X = wide.to_numpy()
    boots = np.stack([np.nanmean(X[rng.integers(0, len(X), len(X))], axis=0) for _ in range(n_boot)])
    vb = valor_mes(boots, P)
    rows = []
    for j, m in enumerate(wide.columns):
        lo, hi = np.percentile(vb[:, j], [2.5, 97.5])
        row = {"modelo": m, "valor_lo": lo, "valor_hi": hi}
        if ref in wide.columns:
            dv = vb[:, j] - vb[:, list(wide.columns).index(ref)]
            row.update(delta_ref=valor_mes(np.nanmean(X[:, j]), P) - valor_mes(
                np.nanmean(X[:, list(wide.columns).index(ref)]), P),
                delta_lo=np.percentile(dv, 2.5), delta_hi=np.percentile(dv, 97.5))
        rows.append(row)
    return pd.DataFrame(rows).set_index("modelo")


# ------------------------------------------------------------------ sensibilidad
def evaluate_setting(ut: pd.DataFrame, P: dict, select_split: str, report_split: str) -> pd.DataFrame:
    agg = aggregate(ut, P)
    ts = choose_theta(agg, select_split)
    return results_at(agg, ts, report_split)


def sensitivity(ut: pd.DataFrame, P: dict, ranges: dict, ref: str, select_split: str,
                report_split: str) -> pd.DataFrame:
    """Cada parámetro en su rango (low/high) y a ±20 % / ±50 %; θ* se re-elige en cada caso."""
    rows = []

    def run(param, label, value):
        Q = dict(P, **{param: value}) if param else P
        r = evaluate_setting(ut, Q, select_split, report_split)
        ml = r.drop(index=[ORACLE], errors="ignore")
        rows.append({"parametro": param or "base", "escenario": label, "valor_param": value,
                     "valor_mes_ref": r.loc[ref, "valor_mes"] if ref in r.index else np.nan,
                     "theta_ref": r.loc[ref, "theta"] if ref in r.index else np.nan,
                     "ganador": ml["valor_mes"].idxmax() if len(ml) else ""})

    run(None, "base", np.nan)
    for p in TORNADO_PARAMS:
        if p in ranges:
            run(p, "rango_bajo", ranges[p][0])
            run(p, "rango_alto", ranges[p][1])
        for f in (-0.5, -0.2, 0.2, 0.5):
            run(p, f"{f:+.0%}", P[p] * (1 + f))
    for label, change in (("pesimista", {"f_rev": 1.0, "t_clip": 4.0, "t_miss": 12.0}),
                          ("optimista", {"f_rev": 2.0, "t_clip": 1.5, "t_miss": 4.0})):
        Q = dict(P, **change)
        r = evaluate_setting(ut, Q, select_split, report_split)
        ml = r.drop(index=[ORACLE], errors="ignore")
        rows.append({"parametro": "escenario", "escenario": label, "valor_param": np.nan,
                     "valor_mes_ref": r.loc[ref, "valor_mes"] if ref in r.index else np.nan,
                     "theta_ref": r.loc[ref, "theta"] if ref in r.index else np.nan,
                     "ganador": ml["valor_mes"].idxmax() if len(ml) else ""})
    return pd.DataFrame(rows)


def breakeven(row: pd.Series, D_mean: float, P: dict) -> dict:
    """Puntos de equilibrio del modelo de referencia (con sus TP/FP/FN medios en test)."""
    tp, fp, fn = row["tp"] / row["n"], row["fp"] / row["n"], row["fn"] / row["n"]
    t_tool = P["t_setup"] + (tp + fp) * P["t_clip"] + fn * P["t_miss"]
    ah = row["ahorro"]
    price = P.get("precio_ref_usd_mes", 8.0)
    per_match_usd = ah / 60 * P["C_h_usd"] - P["C_pred_usd"]
    return {
        "f_rev_equilibrio": t_tool / D_mean,
        "t_clip_equilibrio": (P["f_rev"] * D_mean - P["t_setup"] - fn * P["t_miss"]) / (tp + fp)
        if tp + fp > 0 else np.nan,
        "precio_ref": price,
        "N_rev_para_precio": price / per_match_usd if per_match_usd > 0 else np.inf,
        "C_h_para_precio": price / (P["N_rev"] * ah / 60) if ah > 0 else np.inf,
    }


# ------------------------------------------------------------------ figuras
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # paleta validada, orden fijo


def _usd(v: float) -> str:
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
    })
    return plt


def plot_value_vs_theta(agg: pd.DataFrame, theta_star: pd.Series, plot_models: list[str]) -> Path:
    plt = _style()
    splits = [s for s in ("val", "test") if s in set(agg["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4), sharey=True, squeeze=False)
    for ax, sp in zip(axes[0], splits):
        a = agg[agg["split"] == sp]
        for ref_name, ls in (("R0", "--"), (ORACLE, ":")):
            r = a[a["modelo"] == ref_name]
            if len(r):
                ax.axhline(r["valor_mes"].iloc[0], color=INK2, linewidth=1.2, linestyle=ls)
                ax.annotate(ref_name if ref_name == "R0" else "Oráculo (techo)",
                            (THETAS[0], r["valor_mes"].iloc[0]), xytext=(2, 3),
                            textcoords="offset points", color=INK2, fontsize=8)
        for color, m in zip(SERIES, [m for m in plot_models if m in set(a["modelo"])]):
            r = a[a["modelo"] == m].sort_values("theta")
            ax.plot(r["theta"], r["valor_mes"], color=color, label=m)
            if m in theta_star.index and sp == "val":
                s = r[r["theta"] == theta_star[m]]
                ax.plot(s["theta"], s["valor_mes"], "o", color=color, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=2)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{sp} (n = {a['n'].max()} partidas)")
        ax.set_xlabel("umbral θ del detector")
    axes[0][0].set_ylabel("Valor al mes por equipo (USD)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle("Valor según el umbral del detector (● = θ* elegido en val)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    path = FIG / "b10_valor_vs_theta.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tornado(sens: pd.DataFrame, ref: str) -> Path:
    plt = _style()
    base = sens.loc[sens["parametro"] == "base", "valor_mes_ref"].iloc[0]
    rows = []
    for p in TORNADO_PARAMS:
        s = sens[sens["parametro"] == p].set_index("escenario")
        if {"rango_bajo", "rango_alto"} <= set(s.index):
            rows.append((p, s.loc["rango_bajo"], s.loc["rango_alto"]))
    rows.sort(key=lambda r: abs(r[2]["valor_mes_ref"] - r[1]["valor_mes_ref"]))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(rows) + 1.4))
    for i, (p, lo, hi) in enumerate(rows):
        for r, color in ((lo, SERIES[0]), (hi, SERIES[1])):
            v = r["valor_mes_ref"]
            ax.barh(i, v - base, left=base, color=color, height=0.6)
            ax.annotate(f"{r['valor_param']:g} → {_usd(v)}", (v, i), xytext=(4 if v >= base else -4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2,
                        ha="left" if v >= base else "right")
    ax.set_yticks(range(len(rows)), [PARAM_LABEL[r[0]] for r in rows])
    ax.axvline(base, color=INK, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Valor al mes por equipo (USD)")
    ax.set_title(f"Sensibilidad del valor de {ref} (base {_usd(base)}/mes)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="parámetro en su valor bajo"),
                       Patch(color=SERIES[1], label="parámetro en su valor alto")],
              loc="lower right", fontsize=8)
    xmin, xmax = ax.get_xlim()
    pad = 0.18 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    fig.tight_layout()
    path = FIG / "b10_tornado.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ reporte
def _fmt(v, d=1):
    return "—" if pd.isna(v) else f"{v:,.{d}f}"


def write_report(main_tab: pd.DataFrame, sens: pd.DataFrame, be: dict, P: dict, info: dict,
                 ref: str, tornado_model: str, report_split: str) -> Path:
    L = ["# B10 · Modelo de valor: resultados", "",
         f"*Generado por `python src/value_model.py {info['argv']}`. Se vuelve a correr cuando haya "
         "más anotaciones.*", "",
         "## Datos", "",
         f"- Partidas anotadas con predicciones: **{info['n_val']} de val** (para elegir θ*) y "
         f"**{info['n_test']} de test** (para reportar). Anotadores: {info['por_anotador']}. "
         f"Compartidas (A y B): {info['n_shared']}.",
         f"- Momentos anotados por partida: {info['mom_por_partida']:.1f}; partidas sin momentos: "
         f"{info['pct_none']:.0f} %. Confianza mínima usada: {info['min_conf']}.",
         f"- Duración media de esas partidas: {info['D_mean']:.1f} min → revisión completa "
         f"T_base = {P['f_rev']} × {info['D_mean']:.1f} = **{P['f_rev'] * info['D_mean']:.0f} min**.",
         f"- Tiempo medio de anotar una partida con la herramienta (solo el log): "
         f"{info['seg_anotar'] / 60:.1f} min.",
         f"- Parámetros: f_rev {P['f_rev']}, t_setup {P['t_setup']}, t_clip {P['t_clip']}, "
         f"t_miss {P['t_miss']}, N_rev {P['N_rev']:g}, C_h USD {P['C_h_usd']:g} "
         "(`configs/value.yaml`).", ""]
    if info["n_test"] < 20 or info["n_val"] < 20:
        L += ["> **Resultados preliminares:** con menos de ~20 partidas por split los IC son muy "
              "anchos y θ* puede cambiar. Volver a correr con las 35 + 35 anotaciones.", ""]
    L += [f"## Tabla principal ({report_split}, θ* elegido en val)", "",
          f"| Modelo | θ* | Clips/partida | Precisión | Recall | F1 | Ahorro (min/partida) | "
          f"Ahorro % | Valor USD/mes [IC 95 %] | Δ vs {ref} [IC 95 %] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in main_tab.iterrows():
        th = "—" if m in ("R0", ORACLE) else f"{r['theta']:.2f}"
        ci = f" [{_fmt(r.get('valor_lo'), 0)}, {_fmt(r.get('valor_hi'), 0)}]" if "valor_lo" in r else ""
        dl = "—" if m == ref or pd.isna(r.get("delta_ref")) else \
            f"{r['delta_ref']:+,.0f} [{r['delta_lo']:+,.0f}, {r['delta_hi']:+,.0f}]"
        name = f"**{m}**" if m == tornado_model else m
        L.append(f"| {name} | {th} | {_fmt(r['clips_por_partida'])} | {_fmt(r['precision'], 2)} | "
                 f"{_fmt(r['recall'], 2)} | {_fmt(r['f1'], 2)} | {_fmt(r['ahorro'])} | "
                 f"{_fmt(r['ahorro_pct'], 0)} % | {_fmt(r['valor_mes'], 0)}{ci} | {dl} |")
    L += ["", "*Oráculo* = detector perfecto (marca exactamente lo anotado): es el techo del ahorro "
          "con estos parámetros. R0 = regla sin ML (barón, ace, inhibidor o swing de oro > 1,500).",
          "", "![valor vs theta](figures/b10_valor_vs_theta.png)", "",
          f"## Sensibilidad ({tornado_model}, θ* re-elegido en cada caso)", "",
          "| Parámetro | Valor bajo → USD/mes | Valor alto → USD/mes | ¿Cambia el ganador? |",
          "|---|---|---|---|"]
    base_win = sens.loc[sens["parametro"] == "base", "ganador"].iloc[0]
    for p in TORNADO_PARAMS:
        s = sens[(sens["parametro"] == p) & sens["escenario"].isin(["rango_bajo", "rango_alto"])]
        if len(s) != 2:
            continue
        lo, hi = s.iloc[0], s.iloc[1]
        win = {lo["ganador"], hi["ganador"]} - {base_win}
        L.append(f"| {PARAM_LABEL[p]} | {lo['valor_param']:g} → {_fmt(lo['valor_mes_ref'], 0)} | "
                 f"{hi['valor_param']:g} → {_fmt(hi['valor_mes_ref'], 0)} | "
                 f"{'sí: ' + ', '.join(sorted(win)) if win else 'no'} |")
    sc = sens[sens["parametro"] == "escenario"].set_index("escenario")
    L += ["", f"Ganador con los valores base: **{base_win}**. Escenario pesimista (f_rev 1.0, t_clip 4, "
          f"t_miss 12): {tornado_model} {_fmt(sc.loc['pesimista', 'valor_mes_ref'], 0)} USD/mes, gana "
          f"{sc.loc['pesimista', 'ganador']}. Optimista (f_rev 2.0, t_clip 1.5, t_miss 4): "
          f"{_fmt(sc.loc['optimista', 'valor_mes_ref'], 0)} USD/mes, gana {sc.loc['optimista', 'ganador']}. "
          "Los ±20 % y ±50 % de cada parámetro están en `results/b10_tornado.csv`.", "",
          "![tornado](figures/b10_tornado.png)", "",
          f"## Punto de equilibrio ({tornado_model})", "",
          f"- El sistema ahorra tiempo mientras **f_rev > {be['f_rev_equilibrio']:.2f}** (revisar la "
          f"partida completa toma más que {be['f_rev_equilibrio']:.2f} min por minuto de juego) y "
          f"mientras **t_clip < {_fmt(be['t_clip_equilibrio'], 1)} min** por clip.",
          f"- Para pagar una suscripción de USD {be['precio_ref']:g}/mes basta con revisar "
          f"**{_fmt(be['N_rev_para_precio'], 1)} partidas al mes** (con C_h = {P['C_h_usd']:g}), o un coach "
          f"de **USD {_fmt(be['C_h_para_precio'], 2)}/h** (con N_rev = {P['N_rev']:g}).", ""]
    path = RES / "b10_valor.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="*", help="carpetas de results/runs (por defecto: R0, M0–M5, E6)")
    ap.add_argument("--name", nargs="*")
    ap.add_argument("--min-conf", type=int, default=1, help="descarta momentos con confianza menor")
    ap.add_argument("--shift", type=int, default=1, help="minuto detectado − shift = minuto de la jugada")
    ap.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    ap.add_argument("--merge-gap", type=int, default=DEFAULTS["merge_gap"])
    ap.add_argument("--ref", default="M0", help="modelo contra el que se calcula Δ")
    ap.add_argument("--tornado-model", default="M4")
    ap.add_argument("--plot", nargs="*", default=["M0", "M2", "M4", "M4 E6 λ0.1", "M4 E6 λ0.5"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    P, ranges = load_params()
    tol = int(P.get("match_window_min", 2))
    if args.run:
        names = args.name or [Path(r).name for r in args.run]
        models = [(n, None if r.upper() == "R0" else Path(r)) for n, r in zip(names, args.run)]
    else:
        models = []
        for n, r in DEFAULT_MODELS:
            if r is not None and not (RES / "runs" / r / "predictions.npz").exists():
                print(f"  aviso: falta results/runs/{r}/predictions.npz, se omite {n}")
                continue
            models.append((n, None if r is None else RES / "runs" / r))

    units = load_annotations(min_conf=args.min_conf)
    sel = pd.read_csv(ANN_DIR / "seleccion.csv")
    units = units.merge(sel[["match_id", "split", "duration_min"]], on="match_id", how="left")
    units = units[units["split"].isin(["val", "test"])].rename(columns={"duration_min": "D"})
    if units.empty:
        raise SystemExit("Ninguna partida anotada es de val/test")
    print(f"{len(units)} anotaciones ({units['match_id'].nunique()} partidas): "
          f"{units.drop_duplicates('match_id')['split'].value_counts().to_dict()}")

    ut = unit_table(units, models, tol, args.shift, args.top_k, args.merge_gap)
    agg = aggregate(ut, P)
    have_val = (ut["split"] == "val").any()
    select_split = "val" if have_val else "test"
    if not have_val:
        print("  aviso: no hay partidas de val anotadas; θ* se elige en test (optimista)")
    theta_star = choose_theta(agg, select_split)
    report_split = "test" if (ut["split"] == "test").any() else "val"

    main_tab = results_at(agg, theta_star, report_split)
    boot = bootstrap(ut, theta_star, P, report_split, args.ref, n_boot=args.n_boot)
    main_tab = main_tab.join(boot)
    order = [n for n, _ in models] + [ORACLE]
    main_tab = main_tab.reindex([m for m in order if m in main_tab.index])

    tm = args.tornado_model if args.tornado_model in main_tab.index else main_tab.index[0]
    sens = sensitivity(ut, P, ranges, tm, select_split, report_split)
    meta = ut.drop_duplicates("match_id")
    D_mean = meta.loc[meta["split"] == report_split, "D"].mean()
    be = breakeven(main_tab.loc[tm], D_mean, P)

    agg.to_csv(RES / "b10_valor_theta.csv", index=False, float_format="%.4f")
    sens.to_csv(RES / "b10_tornado.csv", index=False, float_format="%.4f")
    FIG.mkdir(parents=True, exist_ok=True)
    plot_value_vs_theta(agg, theta_star, args.plot)
    plot_tornado(sens, tm)

    sel_u = units.drop_duplicates("match_id")
    info = {"argv": " ".join(sys.argv[1:]),
            "n_val": int((sel_u["split"] == "val").sum()), "n_test": int((sel_u["split"] == "test").sum()),
            "por_anotador": ", ".join(f"{k} {v}" for k, v in units["anotador"].value_counts().sort_index().items()),
            "n_shared": int((units.groupby("match_id")["anotador"].nunique() > 1).sum()),
            "mom_por_partida": units["minutos"].map(len).mean(),
            "pct_none": 100 * (units["minutos"].map(len) == 0).mean(),
            "min_conf": args.min_conf, "D_mean": D_mean, "seg_anotar": units["segundos"].mean()}
    path = write_report(main_tab, sens, be, P, info, args.ref, tm, report_split)

    cols = ["theta", "clips_por_partida", "precision", "recall", "f1", "ahorro", "ahorro_pct", "valor_mes"]
    print(f"\n{report_split}, θ* elegido en {select_split}:")
    print(main_tab[cols].round(3).to_string())
    print(f"\n-> {path.relative_to(ROOT)}, results/b10_valor_theta.csv, results/b10_tornado.csv, "
          "results/figures/b10_*.png")


if __name__ == "__main__":
    main()
