"""
B11 - Latencia de la app (guía, sección 8: p50/p95 separando la API de Riot de la inferencia).

  python app/bench_latency.py                 # inferencia: 100 partidas de test desde data/raw (CPU)
  python app/bench_latency.py --riot 10       # + 10 descargas reales de la API de Riot (necesita .env)

Inferencia = JSON de Riot -> tablas (parse) -> features -> 3 semillas de la GRU -> momentos
-> explicación por momento. La API de Riot (match + timeline) se mide aparte porque domina y
no depende de nosotros. Salida: results/b11_latencia.csv y results/b11_latencia.md
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.inference import Engine, platform_of, tables_from_riot  # noqa: E402
from riot_data import config  # noqa: E402

OUT_CSV = ROOT / "results" / "b11_latencia.csv"
OUT_MD = ROOT / "results" / "b11_latencia.md"


def test_matches(n: int) -> list[tuple[str, Path, Path]]:
    s = pd.read_csv(ROOT / "splits" / "splits.csv")
    out = []
    for mid in s.loc[s["split"] == "test", "match_id"]:
        plat = platform_of(mid)
        m, t = config.RAW_DIR / plat / f"{mid}.json.gz", config.RAW_DIR / plat / f"{mid}_timeline.json.gz"
        if m.exists() and t.exists():
            out.append((mid, m, t))
            if len(out) >= n:
                break
    return out


def pct(x, q):
    return round(float(np.percentile(x, q)), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--riot", type=int, default=0, help="descargas reales de Riot a medir (2 llamadas c/u)")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "m4_gru"))
    args = ap.parse_args()

    eng = Engine(args.model_dir)
    matches = test_matches(args.n)
    print(f"{len(matches)} partidas de test con JSON crudo; modelo {eng.name} ({len(eng.members)} semillas), "
          f"CPU, {torch.get_num_threads()} hilos")
    rows = []
    eng.analyze(tables_from_riot(json.load(gzip.open(matches[0][1])), json.load(gzip.open(matches[0][2])),
                                 platform_of(matches[0][0])))         # calentamiento
    for mid, pm, pt in matches:
        match, timeline = json.load(gzip.open(pm)), json.load(gzip.open(pt))
        t0 = time.perf_counter()
        tables = tables_from_riot(match, timeline, platform_of(mid))
        parse_ms = 1000 * (time.perf_counter() - t0)
        res = eng.analyze(tables)
        rows.append({"match_id": mid, "minutos": len(res["curva"]), "momentos": len(res["momentos"]),
                     "parse_ms": parse_ms, **{k: v for k, v in res["tiempos"].items()},
                     "total_sin_riot_ms": parse_ms + res["tiempos"]["inferencia_total_ms"]})
    df = pd.DataFrame(rows)

    riot = []
    if args.riot:
        from riot_data.client import RiotClient
        client = RiotClient()
        for mid, _, _ in matches[: args.riot]:
            region = config.PLATFORM_TO_REGION[platform_of(mid)]
            t0 = time.perf_counter()
            client.match(region, mid)
            client.timeline(region, mid)
            riot.append(1000 * (time.perf_counter() - t0))
        df.loc[: len(riot) - 1, "riot_ms"] = riot
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.2f")

    parts = [("Parse del JSON de Riot", "parse_ms"), ("Features", "features_ms"),
             ("Modelo (3 semillas)", "modelo_ms"), ("Explicación de momentos", "explicacion_ms"),
             ("**Total sin Riot**", "total_sin_riot_ms")]
    L = ["# B11 · Latencia de la app", "",
         f"*`python app/bench_latency.py`. {len(df)} partidas de test distintas (desde el JSON crudo), "
         f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} hilos), "
         f"modelo {eng.name} con {len(eng.members)} semillas. {time.strftime('%Y-%m-%d')}.*", "",
         "| Etapa | p50 (ms) | p95 (ms) | máx (ms) |", "|---|---|---|---|"]
    for name, col in parts:
        L.append(f"| {name} | {pct(df[col], 50)} | {pct(df[col], 95)} | {round(df[col].max(), 1)} |")
    if riot:
        L.append(f"| API de Riot (match + timeline) | {pct(riot, 50)} | {pct(riot, 95)} | {round(max(riot), 1)} |")
    L += ["", f"- Partidas de {df['minutos'].min()}–{df['minutos'].max()} minutos; "
          f"{df['momentos'].mean():.1f} momentos explicados por partida en promedio.",
          "- La explicación cuesta 1 pasada extra del modelo por momento (5 variantes en un batch).",
          ("- La API de Riot domina: es el " +
           f"{100 * np.median(riot) / (np.median(riot) + df['total_sin_riot_ms'].median()):.0f} % de la latencia "
           "total típica." if riot else
           "- **Falta medir la API de Riot** (`--riot 10`, necesita `.env`). La guía estima 0.5–1.5 s."),
          "- Con caché (la misma partida otra vez) la respuesta es casi instantánea: no se llama a Riot ni al modelo.",
          f"- **Meta de B11:** < 3 s por partida nueva. Inferencia p95 = {pct(df['total_sin_riot_ms'], 95)} ms."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
