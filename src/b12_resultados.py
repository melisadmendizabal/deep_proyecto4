"""
B12 - Explicabilidad, casos de falla y tabla maestra E1 con costos.

  python src/b12_resultados.py                    # todo (~3-5 min; GPU si hay, si no CPU)
  python src/b12_resultados.py --solo shap ablacion
  python src/b12_resultados.py --solo fallas costo

Partes:
  shap      SHAP global de LightGBM (M2) en test, con el TreeSHAP que trae LightGBM
            (predict(pred_contrib=True), no hace falta el paquete `shap`). Por feature, por grupo
            y por tramo de minuto.
  ablacion  Ablación por grupos de features en los modelos secuenciales de la app (M4 GRU y
            M6+campeones): cada grupo se pone en 0 (= su media de train, porque las features
            están estandarizadas) y se mide cuánto empeoran AUC y log-loss en test. En M6+campeones
            también se "borran" los campeones (id 0 = desconocido).
  fallas    3 casos de falla del detector de la app (M6+campeones, θ* del manifest) en las
            partidas anotadas de test, comparados con las anotaciones y con la regla R0.
  costo     Tabla maestra E1: AUC, valor al mes, costo de entrenamiento (segundos de GPU de la
            búsqueda y de las 3 semillas finales), parámetros y latencia de inferencia en CPU
            medida aquí con el motor de la app.

Salidas:
  results/b12_explicabilidad.md   results/b12_shap.csv   results/b12_ablacion.csv
  results/b12_fallas.md           results/b12_fallas.csv
  results/b12_tabla_e1.md         results/b12_tabla_e1.csv
  results/figures/b12_shap_lgbm.png   b12_ablacion.png   b12_fallas.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.inference import Engine, load_member, window_events  # noqa: E402
from src.datasets import load_sequences  # noqa: E402
from src.evaluate import Rows  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402
from src.moments import DEFAULTS, detect_moments  # noqa: E402
from src.value_model import (ANN_DIR, INK, INK2, SERIES, SURFACE, _style,  # noqa: E402
                             r0_detections)

RES = ROOT / "results"
FIG = RES / "figures"
RUNS = RES / "runs"
MODELS = ROOT / "models"
SEQ = ROOT / "features" / "sequences.npz"
SHIFT = 1          # minuto detectado − 1 = minuto de la jugada (igual que value_model)
TOL = 2            # ±2 min para contar un acierto
BLUE, RED = "#3a8ee6", "#e8505b"   # colores de equipo de la app
GROUP_LABEL = {"contexto": "minuto", "economia": "economía", "rol": "oro por rol", "combate": "combate",
               "objetivos": "objetivos", "vision": "visión", "tendencia": "tendencia (Δ oro)",
               "campeones": "campeones (draft)"}
TRAMOS = [(0, 10, "0-9"), (10, 20, "10-19"), (20, 30, "20-29"), (30, 999, "30+")]


def _test():
    data = load_sequences(SEQ)
    if list(data.feature_names) != list(FEATURE_NAMES):
        raise SystemExit("features/sequences.npz no coincide con FEATURE_NAMES de src/features.py")
    return data.subset("test")


# ================================================================== SHAP de LightGBM
def shap_lgbm(test, max_rows: int = 60_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(RUNS / "m2_lgbm" / "model.txt"))
    names = booster.feature_name()
    cols = [FEATURE_NAMES.index(n) for n in names]
    groups = np.asarray(test.feature_groups)[cols]
    X = test.X[test.mask][:, cols]
    minute = np.broadcast_to(np.arange(test.mask.shape[1]), test.mask.shape)[test.mask]
    w = np.broadcast_to(1.0 / test.lengths[:, None], test.mask.shape)[test.mask]   # cada partida pesa 1
    rng = np.random.default_rng(0)
    if len(X) > max_rows:
        idx = np.sort(rng.choice(len(X), max_rows, replace=False))
        X, minute, w = X[idx], minute[idx], w[idx]
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]          # última columna = sesgo
    absc = np.abs(contrib)
    imp = (absc * w[:, None]).sum(0) / w.sum()
    # dirección: correlación entre el valor de la feature y su SHAP (+ = más alto ayuda a azul)
    direction = [np.corrcoef(X[:, j], contrib[:, j])[0, 1] if X[:, j].std() > 0 else np.nan
                 for j in range(len(names))]
    feat = pd.DataFrame({"feature": names, "grupo": groups, "shap_abs_medio": imp,
                         "pct": 100 * imp / imp.sum(), "direccion": direction})
    feat = feat.sort_values("shap_abs_medio", ascending=False).reset_index(drop=True)
    rows = []
    for a, b, lab in [(0, 999, "global")] + TRAMOS:
        sel = (minute >= a) & (minute < b)
        if not sel.any():
            continue
        g = pd.Series((absc[sel] * w[sel, None]).sum(0), index=groups).groupby(level=0).sum()
        g = 100 * g / g.sum()
        rows += [{"tramo": lab, "grupo": k, "pct": v} for k, v in g.items()]
    grp = pd.DataFrame(rows)
    return feat, grp


def plot_shap(feat: pd.DataFrame, grp: pd.DataFrame, top: int = 15) -> Path:
    plt = _style()
    order = list(GROUP_LABEL)[:7]
    color = {g: c for g, c in zip(["economia", "rol", "objetivos", "combate", "vision", "tendencia", "contexto"],
                                  SERIES + [INK2, "#9aa3b2"])}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    f = feat.head(top).iloc[::-1]
    ax1.barh(f["feature"], f["pct"], color=[color[g] for g in f["grupo"]], height=0.65)
    for y, (v, d) in enumerate(zip(f["pct"], f["direccion"])):
        ax1.annotate(f"{v:.1f} %", (v, y), xytext=(3, 0), textcoords="offset points", va="center",
                     fontsize=8, color=INK2)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("% del |SHAP| medio (test, cada partida pesa 1)")
    ax1.set_title(f"M2 LightGBM: top {top} features")
    tr = [t for t in ["0-9", "10-19", "20-29", "30+"] if t in set(grp["tramo"])]
    piv = grp[grp["tramo"].isin(tr)].pivot(index="tramo", columns="grupo", values="pct").reindex(tr)
    left = np.zeros(len(tr))
    for g in [g for g in order if g in piv.columns]:
        ax2.barh(range(len(tr)), piv[g], left=left, color=color[g], label=GROUP_LABEL[g], height=0.6)
        for i, v in enumerate(piv[g]):
            if v >= 6:
                ax2.annotate(f"{v:.0f}", (left[i] + v / 2, i), ha="center", va="center", fontsize=8,
                             color=SURFACE)
        left += piv[g].to_numpy()
    ax2.set_yticks(range(len(tr)), [f"min {t}" for t in tr])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("% del |SHAP| por grupo")
    ax2.set_title("Peso de cada grupo por minuto")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8)
    fig.tight_layout()
    path = FIG / "b12_shap_lgbm.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ================================================================== ablación por grupos
@torch.no_grad()
def _grid_probs(engine: Engine, X: np.ndarray, mask: np.ndarray, champs: np.ndarray | None,
                device: str, batch: int = 512) -> np.ndarray:
    """P(gana azul) [N, T] con la curva principal del motor (1 semilla o promedio recalibrado)."""
    out = []
    for m in engine.members:
        m.model.to(device)
    for i in range(0, len(X), batch):
        xb, mb = X[i:i + batch], torch.as_tensor(mask[i:i + batch], device=device)
        per_seed = []
        for m in engine.members:
            xt = torch.as_tensor(xb[..., m.cols], device=device)
            if getattr(m.model, "uses_champions", False):
                ch = torch.as_tensor(champs[i:i + batch].astype(np.int64), device=device)
                z = m.model(xt, mb, champions=ch)
            else:
                z = m.model(xt, mb)
            per_seed.append(1 / (1 + np.exp(-z.float().cpu().numpy().astype(np.float64) / m.temperature)))
        p = engine.combine(np.stack(per_seed)) if engine.average else per_seed[0]
        out.append(p)
    for m in engine.members:
        m.model.to("cpu")
    probs = np.concatenate(out)
    probs[~mask] = np.nan
    return probs


def ablation(test, device: str) -> pd.DataFrame:
    groups = np.asarray(test.feature_groups)
    rows = []
    for label, d in (("M4 GRU", "m4_gru"), ("M6 + campeones", "m6_draft")):
        if not (MODELS / d / "manifest.json").exists():
            print(f"  aviso: falta models/{d}, se omite {label}")
            continue
        eng = Engine(MODELS / d)
        used = sorted({groups[c] for m in eng.members for c in m.cols})
        has_draft = any(getattr(m.model, "uses_champions", False) for m in eng.members)
        variants = [("ninguno", None)] + [(g, g) for g in used] + ([("campeones", "campeones")] if has_draft else [])
        base = None
        for name, g in variants:
            X, ch = test.X, test.champions
            if g == "campeones":
                ch = np.zeros_like(test.champions)
            elif g is not None:
                X = test.X.copy()
                X[..., groups == g] = 0.0
            met = Rows(_grid_probs(eng, X, test.mask, ch, device), test.y).metrics()
            if base is None:
                base = met
            rows.append({"modelo": label, "grupo_quitado": name, "auc": met["auc"], "logloss": met["logloss"],
                         "d_auc": met["auc"] - base["auc"], "d_logloss": met["logloss"] - base["logloss"]})
            print(f"  {label:15s} sin {name:10s} AUC {met['auc']:.4f} ({met['auc'] - base['auc']:+.4f})")
    return pd.DataFrame(rows)


def plot_ablation(ab: pd.DataFrame) -> Path:
    plt = _style()
    models = list(dict.fromkeys(ab["modelo"]))
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 3.8), sharex=True, squeeze=False)
    for ax, (mod, color) in zip(axes[0], zip(models, SERIES)):
        a = ab[(ab["modelo"] == mod) & (ab["grupo_quitado"] != "ninguno")].sort_values("d_auc", ascending=False)
        ax.barh([GROUP_LABEL.get(g, g) for g in a["grupo_quitado"]], -a["d_auc"], color=color, height=0.6)
        for y, v in enumerate(-a["d_auc"]):
            ax.annotate(f"−{v:.3f}" if v >= 0 else f"+{-v:.3f}", (max(v, 0), y), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        base = ab[(ab["modelo"] == mod) & (ab["grupo_quitado"] == "ninguno")]["auc"].iloc[0]
        ax.set_title(f"{mod} (AUC test {base:.3f})")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUC que se pierde al quitar el grupo")
    fig.suptitle("Ablación por grupos: cada grupo se reemplaza por su media de train",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    path = FIG / "b12_ablacion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_explain(feat, grp, ab, figs) -> Path:
    gg = grp[grp["tramo"] == "global"].sort_values("pct", ascending=False)
    L = ["# B12 · Explicabilidad: SHAP de LightGBM y ablación por grupos", "",
         "*`python src/b12_resultados.py --solo shap ablacion`. Partidas de test (4,032). "
         "Cada partida pesa 1 (filas con peso 1/duración).*", "",
         "## SHAP global de LightGBM (M2)", "",
         "TreeSHAP exacto de LightGBM (`pred_contrib=True`) sobre una muestra de 60,000 filas "
         "(partida × minuto) de test. *Dirección* = correlación entre el valor de la feature y su SHAP "
         "(+1: más alto → más probable que gane azul).", "",
         "| # | Feature | Grupo | % del \\|SHAP\\| | Dirección |", "|---|---|---|---|---|"]
    for i, r in feat.head(15).iterrows():
        L.append(f"| {i + 1} | `{r.feature}` | {GROUP_LABEL.get(r.grupo, r.grupo)} | {r.pct:.1f} % | "
                 f"{r.direccion:+.2f} |")
    L += ["", "**Por grupo (global):** " + " · ".join(f"{GROUP_LABEL.get(r.grupo, r.grupo)} {r.pct:.0f} %"
                                                       for r in gg.itertuples()), "",
          "**Por tramo de minuto (% del |SHAP|):**", "",
          "| Grupo | " + " | ".join(f"min {t}" for _, _, t in TRAMOS) + " |",
          "|---|" + "---|" * len(TRAMOS)]
    piv = grp[grp["tramo"] != "global"].pivot(index="grupo", columns="tramo", values="pct")
    for g in gg["grupo"]:
        L.append(f"| {GROUP_LABEL.get(g, g)} | " + " | ".join(
            f"{piv.loc[g, t]:.0f} %" if t in piv.columns else "—" for _, _, t in TRAMOS) + " |")
    L += ["", f"![SHAP](figures/{figs['shap'].name})", ""]
    if ab is not None and len(ab):
        L += ["## Ablación por grupos en los modelos secuenciales (M4 GRU y M6 + campeones)", "",
              "Cada grupo se reemplaza por 0 en todos los minutos (= su media de train, porque las features "
              "están estandarizadas) y se mide el AUC y el log-loss en test. Es la misma idea que la explicación "
              "por momento de la app (`app/inference.py`, `explain`), pero sobre todo test. Los modelos secuenciales "
              "no usan el grupo *tendencia* (la historia ya la ven).", "",
              "| Modelo | Grupo quitado | AUC | ΔAUC | Log-loss | ΔLog-loss |", "|---|---|---|---|---|---|"]
        for r in ab.itertuples():
            L.append(f"| {r.modelo} | {GROUP_LABEL.get(r.grupo_quitado, r.grupo_quitado)} | {r.auc:.4f} | "
                     f"{r.d_auc:+.4f} | {r.logloss:.4f} | {r.d_logloss:+.4f} |")
        L += ["", f"![Ablación](figures/{figs['ablacion'].name})", ""]
    path = RES / "b12_explicabilidad.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ================================================================== casos de falla
def _annotations() -> pd.DataFrame:
    parts = [pd.read_csv(ANN_DIR / f"{w}.csv") for w in ("A", "B") if (ANN_DIR / f"{w}.csv").exists()]
    ann = pd.concat(parts, ignore_index=True)
    last = ann.groupby(["anotador", "match_id"])["guardado_utc"].transform("max")
    ann = ann[(ann["guardado_utc"] == last) & ~ann["sin_momentos"].astype(str).str.lower().eq("true")]
    ann = ann.dropna(subset=["minuto"]).copy()
    ann["minuto"] = ann["minuto"].astype(int)
    return ann


def _max_jump(p: np.ndarray, frame: int) -> tuple[float, int, str]:
    """Mayor |salto| (ventana 1 o 2 min) que termina a ±TOL del frame indicado, y a quién favorece."""
    best, at, team = 0.0, frame, "—"
    for t in range(max(1, frame - TOL), min(len(p), frame + TOL + 1)):
        for w in (1, 2):
            if t - w >= 0 and abs(p[t] - p[t - w]) > best:
                best, at = abs(p[t] - p[t - w]), t
                team = "azul" if p[t] > p[t - w] else "rojo"
    return best, at, team


def failure_cases(model_dir: Path = MODELS / "m6_draft") -> tuple[pd.DataFrame, list[dict]]:
    from src.predict_matches import load_raw
    sel = pd.read_csv(ANN_DIR / "seleccion.csv")
    ann = _annotations()
    test_ids = sorted(set(sel.loc[sel["split"] == "test", "match_id"]) & set(_all_annotated()))
    eng = Engine(model_dir)
    theta = eng.theta
    raw = load_raw(test_ids)
    r0 = r0_detections(test_ids, DEFAULTS["top_k"], DEFAULTS["merge_gap"])
    rows, detail = [], {}
    for mid in test_ids:
        res = eng.analyze(raw[mid])
        p = np.array([c["p_azul"] for c in res["curva"]])
        lo = np.array([c["p_min"] for c in res["curva"]])
        hi = np.array([c["p_max"] for c in res["curva"]])
        det = res["momentos"]
        a = ann[ann["match_id"] == mid]
        det_play = [m["minuto"] - SHIFT for m in det]
        r0_play = [m - SHIFT for m in r0.get(mid, [])]
        for r in a.itertuples():
            hit = [m for m in det if abs(m["minuto"] - SHIFT - r.minuto) <= TOL]
            jump, at, jteam = _max_jump(p, r.minuto + SHIFT)
            near_all = [m for m in detect_moments(p, theta=theta, windows=tuple(eng.det.get("windows", (1, 2))),
                                                  merge_gap=eng.det.get("merge_gap", 2), top_k=None)
                        if abs(m.minute - SHIFT - r.minuto) <= TOL]
            rows.append({"match_id": mid, "tipo_fila": "anotado", "anotador": r.anotador, "minuto_jugada": r.minuto,
                         "equipo_anotado": r.equipo, "tipo": r.tipo, "confianza": r.confianza,
                         "detectado": bool(hit), "equipo_ok": bool(hit) and hit[0]["favorece"] == r.equipo,
                         "r0_detecta": any(abs(m - r.minuto) <= TOL for m in r0_play),
                         "salto_max": round(jump, 3), "salto_favorece": jteam,
                         "cortado_top_k": bool(near_all) and not hit,
                         "p_antes": round(float(p[max(at - 1, 0)]), 3),
                         "theta": theta, "nota": r.nota})
        for m in det:
            near = a[(a["minuto"] - (m["minuto"] - SHIFT)).abs() <= TOL]
            rows.append({"match_id": mid, "tipo_fila": "detectado", "minuto_jugada": m["minuto"] - SHIFT,
                         "favorece": m["favorece"], "delta_pts": m["delta_pts"], "grupo": m["explicacion"]["grupo"],
                         "anotado": len(near) > 0, "texto": m["texto"],
                         "r0_cerca": any(abs(x - (m["minuto"] - SHIFT)) <= TOL for x in r0_play)})
        detail[mid] = {"res": res, "p": p, "lo": lo, "hi": hi, "ann": a, "det": det, "r0": r0_play,
                       "det_play": det_play, "events": raw[mid]["events"],
                       "ts": raw[mid]["frames"].groupby("minute")["timestamp_ms"].max().to_numpy()
                       if "minute" in raw[mid]["frames"] else None}
    raw_ann = pd.concat([pd.read_csv(ANN_DIR / f"{w}.csv") for w in ("A", "B") if (ANN_DIR / f"{w}.csv").exists()])
    none = raw_ann[raw_ann["sin_momentos"].astype(str).str.lower().eq("true")]
    for mid in detail:
        detail[mid]["sin_momentos"] = sorted(none.loc[none["match_id"] == mid, "anotador"].unique())
    tab = pd.DataFrame(rows)
    tab.attrs["n_partidas"] = len(test_ids)
    return tab, _pick_cases(tab, detail, theta)


def _all_annotated() -> set[str]:
    """Todas las partidas anotadas, también las marcadas 'sin momentos'."""
    return {m for w in ("A", "B") if (ANN_DIR / f"{w}.csv").exists()
            for m in pd.read_csv(ANN_DIR / f"{w}.csv")["match_id"].unique()}


def _split_tab(tab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filas de anotaciones y de detecciones, con las columnas booleanas como bool (el concat las deja object)."""
    an = tab[tab["tipo_fila"] == "anotado"].astype({"detectado": bool, "r0_detecta": bool, "equipo_ok": bool})
    de = tab[tab["tipo_fila"] == "detectado"].astype({"anotado": bool, "r0_cerca": bool})
    return an, de


def _pick_cases(tab: pd.DataFrame, detail: dict, theta: float) -> list[dict]:
    an, de = _split_tab(tab)
    used, cases = set(), []
    # 1) momento perdido que R0 sí ve: el anotado no detectado con el salto más chico (vs θ)
    c1 = an[~an["detectado"] & an["r0_detecta"]].copy()
    c1["abierta"] = (c1["p_antes"] - 0.5).abs() < 0.3          # la partida todavía no estaba decidida
    c1["bajo_theta"] = c1["salto_max"] < theta                 # la curva se movió, pero menos que θ*
    c1 = c1.sort_values(["abierta", "bajo_theta", "confianza", "salto_max"], ascending=[False, False, False, True])
    if len(c1):
        r = c1.iloc[0]
        used.add(r.match_id)
        title = ("Momento cortado por el límite de 5 clips (R0 sí lo marca)" if r.cortado_top_k
                 else "Momento perdido con la partida abierta (R0 sí lo marca)")
        cases.append({"caso": title, "match_id": r.match_id,
                      "minuto": int(r.minuto_jugada), "fila": r})
    # 2) falsa alarma: detección sin anotación cerca, con el salto más grande
    c2 = de[~de["anotado"].astype(bool) & ~de["match_id"].isin(used)].copy()
    c2["abs"] = c2["delta_pts"].abs()
    c2 = c2.sort_values(["r0_cerca", "abs"], ascending=[True, False])   # primero las que R0 tampoco ve
    if len(c2):
        r = c2.iloc[0]
        used.add(r.match_id)
        cases.append({"caso": "Falsa alarma: salto que nadie anotó" + ("" if r.r0_cerca else " y R0 no ve"),
                      "match_id": r.match_id,
                      "minuto": int(r.minuto_jugada), "fila": r})
    # 3) partida decidida antes: momento anotado sin detectar con la curva ya saturada (p < .2 o > .8);
    #    si no hay, el anotado sin detectar con mayor confianza de otra partida
    rest = an[~an["detectado"] & ~an["match_id"].isin(used)].copy()
    rest["sat"] = (rest["p_antes"] - 0.5).abs()
    c3 = rest[rest["sat"] >= 0.3].sort_values(["sat", "confianza"], ascending=False)
    title = "Momento en una partida ya decidida: la curva no tiene a dónde saltar"
    if not len(c3):
        c3, title = rest.sort_values(["confianza", "salto_max"], ascending=[False, True]), "Momento perdido"
    if len(c3):
        r = c3.iloc[0]
        cases.append({"caso": title, "match_id": r.match_id, "minuto": int(r.minuto_jugada), "fila": r})
    for c in cases:
        c.update(detail[c["match_id"]])
        c["theta"] = theta
    return cases


def plot_failures(cases: list[dict]) -> Path:
    plt = _style()
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 3.1 * len(cases)), squeeze=False)
    for ax, c in zip(axes[:, 0], cases):
        t = np.arange(len(c["p"])) - SHIFT       # eje en minuto de la jugada
        ax.fill_between(t, c["lo"], c["hi"], color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(t, c["p"], color=BLUE, label="P(gana azul), M6 + campeones")
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.axvspan(c["minuto"] - TOL, c["minuto"] + TOL, color=SERIES[3], alpha=0.18, linewidth=0)
        for r in c["ann"].itertuples():
            ax.axvline(r.minuto, color=BLUE if r.equipo == "azul" else RED, linewidth=1.2, linestyle="--")
        for m in c["det"]:
            x = m["minuto"] - SHIFT
            ax.plot(x, c["p"][m["minuto"]], "o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    color=BLUE if m["favorece"] == "azul" else RED)
        for x in c["r0"]:
            ax.plot(x, 0.03, "^", color=INK, markersize=7)
        ax.set_ylim(0, 1)
        ax.set_xlim(t[0], t[-1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("P(gana azul)")
        ax.set_title(f"{c['caso']} · {c['match_id']} · min {c['minuto']}")
    axes[-1, 0].set_xlabel("minuto de la jugada  ·  líneas punteadas: anotaciones (color = equipo)  ·  "
                           "●: momentos de la app  ·  ▲: regla R0  ·  franja: caso analizado")
    fig.tight_layout()
    path = FIG / "b12_fallas.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _events_text(c: dict) -> str:
    ev = c["events"]
    if c["ts"] is None or not len(ev):
        return "—"
    ts = c["ts"]
    f0, f1 = max(c["minuto"] + SHIFT - TOL - 1, 0), min(c["minuto"] + SHIFT + TOL, len(ts) - 1)
    d = window_events(ev, int(ts[f0]), int(ts[f1]))
    parts = []
    for team in ("azul", "rojo"):
        x = d.get(team, {})
        items = [f"{v} {k}" for k, v in x.items() if v and not isinstance(v, (list, dict))]
        items += [f"{k}: {', '.join(map(str, v))}" for k, v in x.items() if isinstance(v, list) and v]
        parts.append(f"{team}: " + (", ".join(items) if items else "nada"))
    return " · ".join(parts)


def _miss_breakdown(an: pd.DataFrame, theta: float) -> str:
    miss = an[~an["detectado"]]
    sat = (miss["p_antes"] - 0.5).abs() >= 0.3
    low = ~sat & (miss["salto_max"] < theta)
    loc = ~sat & ~low
    n = max(len(miss), 1)
    return (f"- **Por qué se pierden los {len(miss)} momentos no detectados:** partida ya decidida (curva < 20 % o "
            f"> 80 %) {int(sat.sum())} ({100 * sat.sum() / n:.0f} %); salto menor que θ* con la partida abierta "
            f"{int(low.sum())} ({100 * low.sum() / n:.0f} %); hubo salto ≥ θ* pero el clip quedó en otro minuto "
            f"(fuera del top 5 o fusionado con un vecino) {int(loc.sum())} ({100 * loc.sum() / n:.0f} %).")


def _diagnosis(c: dict) -> str:
    r, th = c["fila"], c["theta"]
    if r.tipo_fila == "detectado":
        dur = c["res"]["duracion_min"]
        if c["minuto"] >= dur - 3:
            return ("La app marca el **cierre de la partida**: en los últimos minutos el ganador tira torres e "
                    "inhibidor y la curva termina de irse a 0 o a 1, así que siempre hay un salto grande. Para el "
                    "coach ese clip casi no aporta (la partida ya se decidió antes), y los anotadores marcamos la "
                    "pelea que la decidió, no el cierre. Arreglo posible: no marcar momentos en los últimos 2–3 "
                    "minutos, o solo si la curva cruza el 50 %.")
        return ("La curva salta por un cambio de economía que los anotadores no consideramos decisivo. Es el error "
                "típico de un detector que solo mira la magnitud del salto, sin saber si cambió el rumbo de la partida.")
    if abs(r.p_antes - 0.5) >= 0.3:
        return (f"Antes del momento la curva ya estaba en {100 * r.p_antes:.0f} %: la partida ya estaba decidida y "
                f"la probabilidad casi no tiene a dónde moverse (salto máximo {100 * r.salto_max:.1f} pts). "
                "El momento es real (lo anotamos y R0 lo marca porque mira los eventos), pero un detector de "
                "saltos de probabilidad es ciego a él. Para el coach no es grave: confirma un resultado que se "
                "decidió antes, y ese momento anterior sí suele estar marcado.")
    if r.salto_max < th:
        return (f"La curva sí se movió ({100 * r.salto_max:.1f} pts), pero menos que θ* = {100 * th:.0f} pts. "
                "El modelo reparte el efecto de la jugada en varios minutos (el oro de las kills y las torres "
                "entra poco a poco), así que ningún salto de 1–2 min pasa el umbral. R0 lo ve porque cuenta los "
                "eventos directamente. Bajar θ* lo detectaría, pero también agrega falsas alarmas (por eso θ* "
                "se eligió maximizando el valor en val).")
    if r.cortado_top_k:
        why = (f"La curva sí saltó {100 * r.salto_max:.1f} pts (≥ θ*) cerca de ese minuto, pero la partida tuvo "
               f"más de {DEFAULTS['top_k']} saltos grandes y el detector solo muestra los {DEFAULTS['top_k']} mayores "
               "(`top_k`, para no darle al coach demasiados clips). En una partida muy movida el límite deja fuera "
               "momentos reales.")
    else:
        why = (f"Hubo un salto de {100 * r.salto_max:.1f} pts (≥ θ*), pero se fusionó con un momento vecino "
               "(`merge_gap`): una pelea larga queda como un solo clip, ubicado en el minuto del salto mayor.")
    if r.salto_favorece not in ("—", r.equipo_anotado):
        why += (f" Además, el salto va a favor de **{r.salto_favorece}** y el momento se anotó a favor de "
                f"**{r.equipo_anotado}**: fue una pelea pareja que la curva y el anotador leyeron distinto.")
    return why


def write_failures(tab: pd.DataFrame, cases: list[dict], fig: Path) -> Path:
    an, de = _split_tab(tab)
    theta = cases[0]["theta"] if cases else float("nan")
    L = ["# B12 · Casos de falla del detector de la app", "",
         f"*`python src/b12_resultados.py --solo fallas`. Detector de la app: M6 + campeones (promedio de 3 "
         f"semillas), θ* = {theta}. {tab.attrs.get('n_partidas', an['match_id'].nunique())} partidas anotadas de test; un acierto es una "
         f"detección a ±{TOL} min de una anotación (minuto detectado − {SHIFT} = minuto de la jugada).*", "",
         "## Resumen de errores en test", "",
         f"- Momentos anotados (contando cada anotador): **{len(an)}**, detectados {int(an['detectado'].sum())} "
         f"({100 * an['detectado'].mean():.0f} %); la regla R0 detecta {int(an['r0_detecta'].sum())} "
         f"({100 * an['r0_detecta'].mean():.0f} %). Aquí una detección puede cubrir dos anotaciones cercanas "
         f"(sin emparejar uno a uno), por eso sale más alto que el recall de `results/mejoras.md`.",
         f"- Detectados con el equipo correcto: {int(an['equipo_ok'].sum())} de {int(an['detectado'].sum())}.",
         f"- Perdidos por la app pero vistos por R0: {int((~an['detectado'] & an['r0_detecta']).sum())}; "
         f"al revés: {int((an['detectado'] & ~an['r0_detecta']).sum())}.",
         f"- Momentos perdidos: salto máximo cerca del minuto anotado, mediana "
         f"{100 * an.loc[~an['detectado'], 'salto_max'].median():.1f} pts (θ* = {100 * theta:.0f} pts); "
         f"{int(((an['p_antes'] - 0.5).abs() >= 0.3)[~an['detectado']].sum())} pasan con la curva ya en < 20 % o > 80 %.",
         _miss_breakdown(an, theta),
         f"- Detecciones de la app: {len(de)}, sin ninguna anotación cerca {int((~de['anotado'].astype(bool)).sum())} "
         f"({100 * (~de['anotado'].astype(bool)).mean():.0f} %).", "",
         f"![Casos de falla](figures/{fig.name})", ""]
    for i, c in enumerate(cases, 1):
        r = c["fila"]
        sel = c["ann"]
        L += [f"## Caso {i}. {c['caso']}", "",
              f"**{c['match_id']}**, minuto {c['minuto']} de la jugada. Gana {c['res']['gana']}, "
              f"{c['res']['duracion_min']} min.", "",
              "**Lo que anotamos en la partida:**", ""]
        for a in sel.itertuples():
            L.append(f"- {a.anotador}, min {a.minuto}, {a.tipo}, favorece a {a.equipo} "
                     f"(confianza {a.confianza:g}): {a.nota}")
        for who in c.get("sin_momentos", []):
            L.append(f"- {who}: la marcó **sin momentos decisivos**")
        L += ["", "**Lo que marcó la app:** " + ("; ".join(
            f"min {m['minuto'] - SHIFT} ({m['favorece']}, {m['delta_pts']:+.0f} pts, {m['explicacion']['grupo']})"
            for m in c["det"]) or "nada"), "",
              "**Regla R0:** " + (", ".join(f"min {x}" for x in c["r0"]) or "nada"), "",
              f"**Eventos alrededor del min {c['minuto']} (±{TOL}):** {_events_text(c)}", ""]
        if r.tipo_fila == "anotado":
            L.append(f"**Medido:** el mayor salto de la curva cerca de ese minuto fue de {100 * r.salto_max:.1f} pts "
                     f"(θ* = {100 * theta:.0f} pts); antes del salto la curva estaba en {100 * r.p_antes:.0f} %.")
        else:
            L.append(f"**Medido:** salto de {r.delta_pts:+.1f} pts a favor de {r.favorece}; la explicación de la app "
                     f"lo atribuye a *{r.grupo}*. Texto de la app: \"{r.texto}\"")
        L += ["", f"**Diagnóstico:** {_diagnosis(c)}", ""]
    path = RES / "b12_fallas.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ================================================================== tabla E1 con costos
E1 = [  # (nombre, carpeta de la app, corrida final, prefijo de la búsqueda, fuente del valor)
    ("R0 regla", None, None, None, "mejoras"),
    ("M0 logística (oro)", "m0_oro", "m0_oro", None, "mejoras"),
    ("M1 logística", "m1_logit", "m1_logit", None, "b10"),
    ("M2 LightGBM", "m2_lgbm", "m2_lgbm", None, "mejoras"),
    ("M3 MLP", "m3_mlp", "mlp_final", "mlp_hp", "b10"),
    ("M4 GRU", "m4_gru", "gru_final", "gru_hp", "mejoras"),
    ("M5 TCN", "m5_tcn", "tcn_final", "tcn_hp", "mejoras"),
    ("M6 Transformer", None, "m6_final", "m6_hp", "mejoras"),
    ("M4 + campeones", "m4_draft", "m4d_hp__lr0.001_d0.3_s4?", "m4d_hp", "mejoras"),   # la config elegida (mejoras_eval)
    ("M6 + campeones (app)", "m6_draft", "m6d_final", None, "mejoras"),
]
VALUE_KEY = {"R0 regla": "R0", "M0 logística (oro)": "M0", "M1 logística": "M1", "M2 LightGBM": "M2",
             "M3 MLP": "M3", "M4 GRU": "M4", "M5 TCN": "M5", "M6 Transformer": "M6",
             "M4 + campeones": "M4+draft ens3", "M6 + campeones (app)": "M6+draft ens3"}
AUC_KEY = {"M0 logística (oro)": "M0", "M2 LightGBM": "M2", "M4 GRU": "M4", "M5 TCN": "M5", "M6 Transformer": "M6",
           "M4 + campeones": "M4+draft ens3", "M6 + campeones (app)": "M6+draft ens3"}


def _summaries(prefix: str, final: bool) -> list[dict]:
    pat = prefix if "?" in prefix else (f"{prefix}__s*" if final else f"{prefix}__*")
    dirs = sorted(RUNS.glob(pat)) or ([RUNS / prefix] if (RUNS / prefix / "summary.json").exists() else [])
    return [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in dirs
            if (d / "summary.json").exists()]


def _b10_values() -> dict:
    """Valor y su IC de results/b10_valor.md (para M1 y M3, que no están en mejoras)."""
    out = {}
    text = (RES / "b10_valor.md").read_text(encoding="utf-8")
    for name in ("M1", "M3"):
        line = next((ln for ln in text.splitlines() if ln.startswith(f"| {name} |")), None)
        if line is None:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Modelo | θ* | Clips | Precisión | Recall | F1 | Ahorro | Ahorro % | Valor [IC] | Δ
        m = re.match(r"(-?\d+) \[(-?\d+), (-?\d+)\]", cells[8])
        out[name] = {"theta": float(cells[1]), "clips": float(cells[2]), "f1": float(cells[5]),
                     "valor": float(m.group(1)), "lo": float(m.group(2)), "hi": float(m.group(3))}
    return out


def _params(app_dir: str | None, fin: list[dict]) -> str:
    """Parámetros entrenables; para los baselines se leen del modelo exportado."""
    if fin and fin[0].get("n_params"):
        return f"{fin[0]['n_params']:,}"
    if app_dir and (MODELS / app_dir / "manifest.json").exists():
        man = json.loads((MODELS / app_dir / "manifest.json").read_text(encoding="utf-8"))
        f = MODELS / app_dir / man["members"][0]["file"]
        if man["members"][0].get("kind") == "logit":
            return f"{len(json.loads(f.read_text(encoding='utf-8'))['coef']) + 1}"
        if man["members"][0].get("kind") == "lgbm":
            import lightgbm as lgb
            return f"{lgb.Booster(model_file=str(f)).num_trees()} árboles"
    return "—"


def latency(model_dirs: list[str], n: int = 28) -> dict:
    from src.predict_matches import load_raw
    from src.features import EVENT_COLS, EVENT_TYPES_USED, FRAME_COLS, match_to_features
    from app.inference import champions_of
    # Un solo hilo, como en Render (OMP_NUM_THREADS=1). En una laptop con núcleos P y E conviene además fijar el
    # proceso a núcleos P con prioridad alta; si no, Windows lo manda a núcleos E y los tiempos varían ×3–4.
    torch.set_num_threads(1)
    sel = pd.read_csv(ANN_DIR / "seleccion.csv")
    ids = sel.loc[sel["split"] == "test", "match_id"].tolist()[:n]
    raw = load_raw(ids)
    feats = {}
    for mid, t in raw.items():
        ev = t["events"][t["events"]["type"].isin(EVENT_TYPES_USED)]
        feats[mid] = (match_to_features(t["frames"][FRAME_COLS], ev.reindex(columns=EVENT_COLS), t["participants"]),
                      champions_of(t["participants"]))
    out = {}
    for d in model_dirs:
        eng = Engine(MODELS / d)
        for f, ch in list(feats.values())[:3]:   # calentar
            eng.predict(f, ch)
        ms_pred, ms_full = [], []
        for mid, (f, ch) in feats.items():   # mínimo de 3 repeticiones: robusto a otra carga en la PC
            tp, tf = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                p = eng.predict(f, ch)
                if eng.average:
                    eng.combine(p)
                tp.append(1000 * (time.perf_counter() - t0))
                t0 = time.perf_counter()
                eng.analyze(raw[mid])
                tf.append(1000 * (time.perf_counter() - t0))
            ms_pred.append(min(tp))
            ms_full.append(min(tf))
        out[d] = {"modelo_ms": float(np.median(ms_pred)), "analisis_ms": float(np.median(ms_full)),
                  "semillas": len(eng.members)}
        print(f"  latencia {d:10s} modelo {out[d]['modelo_ms']:.1f} ms · análisis completo {out[d]['analisis_ms']:.1f} ms")
    return out


def cost_table() -> pd.DataFrame:
    val = pd.read_csv(RES / "mejoras_valor.csv").set_index("modelo")
    tec = pd.read_csv(RES / "mejoras_tecnicas.csv").set_index("modelo")
    b5 = pd.read_csv(RES / "b5_metricas.csv")
    b5 = b5[(b5["split"] == "test") & (b5["subconjunto"] == "todas")] if "todas" in set(b5["subconjunto"]) \
        else b5[b5["split"] == "test"].drop_duplicates("modelo")
    b5 = b5.set_index("modelo")
    b10 = _b10_values()
    lat = latency([d for _, d, *_ in E1 if d and (MODELS / d / "manifest.json").exists()])
    rows = []
    for name, app_dir, final, hp, src in E1:
        k = VALUE_KEY[name]
        if src == "mejoras" and k in val.index:
            v = val.loc[k]
            valor, lo, hi, theta = v["valor_mes"], v["valor_lo"], v["valor_hi"], v["theta"]
            f1, clips = v["f1"], v["clips_por_partida"]
        elif k in b10:
            valor, lo, hi, theta = b10[k]["valor"], b10[k]["lo"], b10[k]["hi"], b10[k]["theta"]
            f1, clips = b10[k]["f1"], b10[k]["clips"]
        else:
            valor = lo = hi = theta = f1 = clips = np.nan
        ak = AUC_KEY.get(name)
        auc = tec.loc[ak, "auc"] if ak in tec.index else (b5.loc[k, "auc"] if k in b5.index else np.nan)
        fin = _summaries(final, final=True) if final else []
        srch = _summaries(hp, final=False) if hp else []
        rows.append({
            "modelo": name, "auc_test": auc, "theta": theta, "clips_partida": clips, "f1_momentos": f1,
            "valor_usd_mes": valor, "valor_lo": lo, "valor_hi": hi,
            "params": _params(app_dir, fin),
            "seg_final": sum(s.get("train_seconds", 0) for s in fin) if fin else 0.0,
            "n_final": len(fin), "seg_busqueda": sum(s.get("train_seconds", 0) for s in srch),
            "n_busqueda": len(srch),
            "dispositivo": (fin[0].get("device", "") if fin else "—"),
            "inferencia_modelo_ms": lat.get(app_dir, {}).get("modelo_ms", np.nan),
            "inferencia_analisis_ms": lat.get(app_dir, {}).get("analisis_ms", np.nan),
            "semillas_app": lat.get(app_dir, {}).get("semillas", np.nan),
        })
    return pd.DataFrame(rows)


def write_cost(e1: pd.DataFrame) -> Path:
    def f(v, d=0, suf=""):
        return "—" if pd.isna(v) else f"{v:,.{d}f}{suf}"
    L = ["# B12 · Tabla maestra E1: valor y costo de cada modelo", "",
         "*`python src/b12_resultados.py --solo costo`. Valor: test (28 partidas anotadas), θ* elegido en val "
         "(`results/mejoras_valor.csv`; M1 y M3 de `results/b10_valor.md`). AUC: test, 4,032 partidas. "
         "Entrenamiento: suma de `train_seconds` de `results/runs/*/summary.json`. Inferencia: medida aquí en CPU "
         "con el motor de la app (`app/inference.py`), 1 hilo como en Render, proceso fijado a núcleos P "
         "(i7-13620H) con prioridad alta; mediana de 28 partidas de test (mínimo de 3 repeticiones por partida); entre corridas los tiempos varían hasta ~2× por el estado térmico "
         "de la laptop, sin cambiar el orden de magnitud. *modelo* = solo la red "
         "(todas las semillas que usa la app), *análisis* = features + modelo + momentos + explicación.*", "",
         "| Modelo | AUC test | Valor USD/mes [IC 95 %] | F1 momentos | Clips/partida | Parámetros | "
         "Entrenamiento final | Búsqueda de HP | Inferencia modelo | Análisis completo |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in e1.itertuples():
        fin = f"{r.seg_final:.0f} s ({r.n_final} sem.)" if r.n_final else ("0 s" if r.modelo.startswith("R0") else "—")
        if r.modelo.startswith("M0") or r.modelo.startswith("M1") or r.modelo.startswith("M2"):
            fin = f"{r.seg_final:.1f} s (CPU)" if r.seg_final else "< 1 s (CPU)"
        bus = f"{r.seg_busqueda / 60:.1f} min ({r.n_busqueda} corridas)" if r.n_busqueda else "—"
        if r.modelo.startswith("M6 + campeones"):
            bus = "la de M6"
        sem = f" ×{int(r.semillas_app)}" if not pd.isna(r.semillas_app) and r.semillas_app > 1 else ""
        L.append(f"| {r.modelo} | {f(r.auc_test, 3)} | {f(r.valor_usd_mes)} [{f(r.valor_lo)}, {f(r.valor_hi)}] | "
                 f"{f(r.f1_momentos, 2)} | {f(r.clips_partida, 1)} | {r.params} | {fin} | {bus} | "
                 f"{f(r.inferencia_modelo_ms, 1, ' ms')}{sem} | {f(r.inferencia_analisis_ms, 0, ' ms')} |")
    dev = sorted({d for d in e1["dispositivo"] if d and d != "—" and d != "cpu"})
    tot = (e1["seg_final"].sum() + e1["seg_busqueda"].sum()) / 60
    L += ["", f"- **Costo de entrenamiento:** todo el cómputo de entrenamiento de la tabla suma ~{tot:.0f} min "
          f"de GPU de laptop ({', '.join(dev)}). A precios de GPU en la nube es del orden de centavos de dólar: "
          "el costo real del proyecto es el tiempo de las personas (anotar, revisar), no el cómputo.",
          "- **Costo de inferencia:** todos los modelos analizan una partida en milisegundos en CPU; en la app "
          "la llamada a la API de Riot (~1.2 s, `results/b11_latencia.md`) domina. Con el plan gratis de Render "
          "(512 MB) `C_infra` = USD 0 y `C_pred` ≈ 0, así que el costo no cambia qué modelo conviene.",
          "- **Conclusión E1:** el valor queda en USD 123–148/mes para todos los detectores y ninguna diferencia "
          "entre modelos de ML es grande; como el costo es ~0 para todos, el modelo se elige por lo que da al "
          "usuario además de los clips (curva calibrada, incertidumbre, explicación).", ""]
    path = RES / "b12_tabla_e1.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", nargs="*", choices=["shap", "ablacion", "fallas", "costo"],
                    default=["shap", "ablacion", "fallas", "costo"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    if {"shap", "ablacion"} & set(args.solo):
        test = _test()
        figs, feat, grp, ab = {}, None, None, None
        if "shap" in args.solo:
            print("SHAP de LightGBM...")
            feat, grp = shap_lgbm(test)
            feat.to_csv(RES / "b12_shap.csv", index=False, float_format="%.5f")
            grp.to_csv(RES / "b12_shap_grupos.csv", index=False, float_format="%.3f")
            figs["shap"] = plot_shap(feat, grp)
        else:
            feat, grp = pd.read_csv(RES / "b12_shap.csv"), pd.read_csv(RES / "b12_shap_grupos.csv")
            figs["shap"] = FIG / "b12_shap_lgbm.png"
        if "ablacion" in args.solo:
            print(f"Ablación por grupos ({args.device})...")
            ab = ablation(test, args.device)
            ab.to_csv(RES / "b12_ablacion.csv", index=False, float_format="%.5f")
            figs["ablacion"] = plot_ablation(ab)
        elif (RES / "b12_ablacion.csv").exists():
            ab = pd.read_csv(RES / "b12_ablacion.csv")
            figs["ablacion"] = FIG / "b12_ablacion.png"
        print("  ->", write_explain(feat, grp, ab, figs))

    if "fallas" in args.solo:
        print("Casos de falla (M6 + campeones)...")
        tab, cases = failure_cases()
        tab.to_csv(RES / "b12_fallas.csv", index=False)
        fig = plot_failures(cases)
        print("  ->", write_failures(tab, cases, fig))

    if "costo" in args.solo:
        print("Tabla E1 con costos...")
        e1 = cost_table()
        e1.to_csv(RES / "b12_tabla_e1.csv", index=False, float_format="%.4f")
        print("  ->", write_cost(e1))


if __name__ == "__main__":
    main()
