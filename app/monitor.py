"""
B13 - Monitoreo del modelo en producción (plan en docs/monitoreo.md).

La app es post-partida: cuando analiza una partida, el resultado YA se conoce. Así que cada
predicción registrada (data/app_cache/predicciones.jsonl) se puede evaluar de inmediato.

  python app/monitor.py --build-ref          # referencia de train (features del min 15) -> app/monitor_ref.json
  python app/monitor.py                      # reporte del registro real -> results/monitoreo_reporte.md
  python app/monitor.py --simulate 300       # valida el monitoreo: pasa 300 partidas de test por la app
                                             # (desde data/raw) a un registro temporal y lo evalúa

Qué mide, por parche y por semana:
  - log-loss y Brier por minuto, ECE (10 bins), AUC en el minuto 15, n.º de momentos por partida
  - PSI de cada feature del minuto 15 contra train (deriva de datos: parches nuevos, cambios de meta)
Alertas (umbrales en ALERTS): ECE alto, log-loss peor que en test, PSI > 0.2, muy pocos o demasiados momentos.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from riot_data import config  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402

LOG = config.DATA_DIR / "app_cache" / "predicciones.jsonl"
REF = ROOT / "app" / "monitor_ref.json"
OUT = ROOT / "results" / "monitoreo_reporte.md"
MINUTE = 15
ALERTS = {
    "ece": 0.05,               # test: ~0.02
    "logloss_extra": 0.05,     # alerta si log-loss > log-loss de test + 0.05
    "psi": 0.20,               # > 0.2 = cambio de distribución importante (regla usual)
    "momentos_min": 0.8,       # media de momentos por partida fuera de [0.8, 4] = detector descalibrado
    "momentos_max": 4.0,
    "n_min": 30,               # con menos partidas no se calculan alertas
}


# ------------------------------------------------------------------ referencia
def build_ref() -> None:
    tm = pd.read_csv(ROOT / "features" / "team_minute.csv")
    row = tm[(tm["split"] == "train") & (~tm["afk"].astype(bool))]
    m = row.groupby("match_id").cumcount()            # minuto dentro de la partida
    row = row[m == MINUTE]
    ref = {"minute": MINUTE, "n_train": int(len(row)), "features": {}}
    for f in FEATURE_NAMES:
        if f == "minute":
            continue
        x = row[f].to_numpy(dtype=float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        if len(edges) < 3:                               # casi constante (p. ej. elder en el min 15)
            continue
        counts = np.histogram(x, bins=edges)[0]
        ref["features"][f] = {"edges": edges.tolist(), "p": (counts / counts.sum()).tolist()}
    s = pd.read_csv(ROOT / "results" / "b5_metricas.csv")
    m4 = s[(s["modelo"] == "M4") & (s["split"] == "test") & (s["subconjunto"] == "todas")]
    ref["test_logloss"] = float(m4["logloss"].iloc[0]) if len(m4) else 0.52
    REF.write_text(json.dumps(ref, indent=1), encoding="utf-8")
    print(f"Referencia: {len(ref['features'])} features del minuto {MINUTE} de {ref['n_train']:,} partidas de train "
          f"-> {REF.relative_to(ROOT)} (log-loss de test {ref['test_logloss']:.3f})")


def psi(values: np.ndarray, edges: list[float], p_ref: list[float]) -> float:
    e = np.array(edges, dtype=float)
    e[0], e[-1] = -np.inf, np.inf
    q = np.histogram(values, bins=e)[0] / max(len(values), 1)
    p = np.clip(np.array(p_ref), 1e-4, None)
    q = np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


# ------------------------------------------------------------------ métricas
def metrics(df: pd.DataFrame) -> dict:
    p = np.concatenate([np.clip(np.array(c, dtype=float), 1e-6, 1 - 1e-6) for c in df["curva_p_azul"]])
    y = np.concatenate([np.full(len(c), 1.0 if g == "azul" else 0.0) for c, g in zip(df["curva_p_azul"], df["gana"])])
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(p)
    out = {"n": len(df), "logloss": float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
           "brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
           "momentos_por_partida": float(df["n_momentos"].mean())}
    at15 = [(c[MINUTE], g == "azul") for c, g in zip(df["curva_p_azul"], df["gana"]) if len(c) > MINUTE]
    if len({b for _, b in at15}) == 2:
        from sklearn.metrics import roc_auc_score
        out["auc_min15"] = float(roc_auc_score([b for _, b in at15], [a for a, _ in at15]))
    return out


def report(log: Path, out: Path = OUT) -> str:
    if not log.exists():
        raise SystemExit(f"No hay registro en {log} (la app escribe una línea por análisis nuevo)")
    # json estándar y no pd.read_json: este último parsea floats de forma imprecisa (0.2 -> 0.19999…) y
    # features discretas como level_diff, que caen justo en los bordes de los bins, cambiaban de bin (PSI falso)
    with open(log, encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    ref = json.loads(REF.read_text(encoding="utf-8")) if REF.exists() else None
    df["semana"] = pd.to_datetime(df["utc"]).dt.strftime("%G-W%V")
    L = ["# Reporte de monitoreo", "", f"*`python app/monitor.py` · {len(df)} análisis en `{log.name}` · "
         f"{df['utc'].min()} → {df['utc'].max()} UTC.*", ""]
    alerts = []

    def table(by: str):
        rows = []
        for key, g in df.groupby(by):
            m = metrics(g)
            rows.append(f"| {key} | {m['n']} | {m['logloss']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | "
                        f"{m.get('auc_min15', float('nan')):.3f} | {m['momentos_por_partida']:.1f} |")
            if m["n"] >= ALERTS["n_min"]:
                if m["ece"] > ALERTS["ece"]:
                    alerts.append(f"{by} {key}: ECE {m['ece']:.3f} > {ALERTS['ece']} (descalibrado: recalibrar la temperatura)")
                if ref and m["logloss"] > ref["test_logloss"] + ALERTS["logloss_extra"]:
                    alerts.append(f"{by} {key}: log-loss {m['logloss']:.3f} > test {ref['test_logloss']:.3f} + {ALERTS['logloss_extra']}")
                if not ALERTS["momentos_min"] <= m["momentos_por_partida"] <= ALERTS["momentos_max"]:
                    alerts.append(f"{by} {key}: {m['momentos_por_partida']:.1f} momentos por partida (revisar θ)")
        return ["| " + by + " | partidas | log-loss | Brier | ECE | AUC min 15 | momentos/partida |",
                "|---|---|---|---|---|---|---|"] + rows

    L += ["## Calidad por parche", ""] + table("parche") + ["", "## Calidad por semana", ""] + table("semana") + [""]
    if ref is not None:
        L += [f"## Deriva de datos (PSI de las features del minuto {MINUTE} vs train)", "",
              "| parche | partidas | PSI máx (feature) | features con PSI > 0.2 |", "|---|---|---|---|"]
        feats = df.dropna(subset=["features_min15"])
        for key, g in feats.groupby("parche"):
            X = pd.DataFrame(list(g["features_min15"]))
            vals = {f: psi(X[f].to_numpy(dtype=float), r["edges"], r["p"]) for f, r in ref["features"].items() if f in X}
            worst = max(vals, key=vals.get)
            high = [f"{f} ({v:.2f})" for f, v in sorted(vals.items(), key=lambda kv: -kv[1]) if v > ALERTS["psi"]]
            L.append(f"| {key} | {len(g)} | {vals[worst]:.3f} ({worst}) | {', '.join(high) or '—'} |")
            if len(g) >= ALERTS["n_min"] and high:
                alerts.append(f"parche {key}: PSI > {ALERTS['psi']} en {', '.join(high)} (posible cambio de meta: revisar y reentrenar)")
        L.append("")
    L += ["## Alertas", ""] + ([f"- ⚠️ {a}" for a in alerts] or ["- Ninguna."]) + [
        "", f"*Umbrales: {json.dumps(ALERTS)}. Sin alertas con menos de {ALERTS['n_min']} partidas por grupo.*"]
    text = "\n".join(L) + "\n"
    out.write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ simulación
def simulate(n: int) -> None:
    """Pasa n partidas de test por la app (sin Riot: JSON de data/raw) a un registro temporal."""
    from fastapi.testclient import TestClient
    import app.api as api
    tmp = Path(tempfile.mkdtemp())
    api.CACHE_DIR = tmp
    s = pd.read_csv(ROOT / "splits" / "splits.csv", dtype={"patch": str})
    ids = [m for m in s.loc[s["split"] == "test", "match_id"]
           if (config.RAW_DIR / m.split("_")[0].lower() / f"{m}_timeline.json.gz").exists()][:n]
    with TestClient(api.app) as tc:
        for i, mid in enumerate(ids, 1):
            tc.get("/analyze", params={"match_id": mid})
            if i % 100 == 0:
                print(f"  {i}/{len(ids)}")
    print(report(tmp / "predicciones.jsonl", out=ROOT / "results" / "monitoreo_simulacion.md"))
    print("-> results/monitoreo_simulacion.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-ref", action="store_true")
    ap.add_argument("--simulate", type=int, default=0)
    ap.add_argument("--log", default=str(LOG))
    args = ap.parse_args()
    if args.build_ref:
        build_ref()
    elif args.simulate:
        simulate(args.simulate)
    else:
        print(report(Path(args.log)))


if __name__ == "__main__":
    main()
