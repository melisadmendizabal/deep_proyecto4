"""
B3 - Construye las features de las 27,000 partidas de data/datos_merge/.

Uso:
  python src/build_features.py                       # todo (~5-10 min)
  python src/build_features.py --limit 300 --out-dir features_prueba   # prueba rápida

Salidas (en --out-dir, por defecto features/, que NO va a git):
  sequences.npz     X [N,60,F] normalizado, mask, lengths, y, match_id, split, ... (docs/contrato_features.md)
  scaler.json       media/desv. ajustadas SOLO con train (sin AFK); la app normaliza con esto
  team_minute.csv   una fila por partida-minuto con las features CRUDAS (para verlas y para M0-M2)

Lee frames.csv y events.csv por trozos, partida por partida (vienen agrupados y en el mismo
orden; si no, igual funciona, solo usa más RAM).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from riot_data import config  # noqa: E402
from src.features import (EVENT_COLS, EVENT_TYPES_USED, FEATURE_GROUPS, FEATURE_NAMES,  # noqa: E402
                          FRAME_COLS, T_MAX, Scaler, match_to_features)
from src.quality_check import iter_complete_matches  # noqa: E402

MERGE = config.MERGE_DIR


def _groups(path: Path, usecols: list[str], filter_types: set[str] | None = None):
    for chunk in iter_complete_matches(path, ["match_id"] + usecols):
        if filter_types is not None:
            chunk = chunk[chunk["type"].isin(filter_types)]
        for mid, g in chunk.groupby("match_id", sort=False):
            yield mid, g


def iter_matches(limit: int | None = None):
    """(match_id, frames, events) de cada partida, leyendo ambos CSV a la par."""
    ev_iter = _groups(MERGE / "events.csv", EVENT_COLS, EVENT_TYPES_USED)
    buffer: dict[str, pd.DataFrame] = {}
    exhausted = False
    empty = pd.DataFrame(columns=EVENT_COLS)
    for i, (mid, fr) in enumerate(_groups(MERGE / "frames.csv", FRAME_COLS)):
        if limit is not None and i >= limit:
            return
        while mid not in buffer and not exhausted:
            try:
                emid, eg = next(ev_iter)
                buffer[emid] = eg
            except StopIteration:
                exhausted = True
        yield mid, fr, buffer.pop(mid, empty)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "features"))
    ap.add_argument("--limit", type=int, default=None, help="solo las primeras N partidas (prueba)")
    ap.add_argument("--no-csv", action="store_true", help="no escribir team_minute.csv")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    matches = pd.read_csv(MERGE / "matches.csv",
                          usecols=["match_id", "platform", "tier", "patch", "blue_win"],
                          dtype={"patch": str}).set_index("match_id")
    splits = pd.read_csv(ROOT / "splits" / "splits.csv", usecols=["match_id", "split"]) \
        .set_index("match_id")["split"]
    afk = pd.read_csv(ROOT / "results" / "b2_calidad_partidas.csv",
                      usecols=["match_id", "excluir_de_train"]).set_index("match_id")["excluir_de_train"]
    parts = pd.read_csv(MERGE / "participants.csv",
                        usecols=["match_id", "participant_id", "team_id", "champion_id", "team_position"])
    parts_by = {mid: g for mid, g in parts.groupby("match_id", sort=False)}

    n_total = len(matches) if args.limit is None else min(args.limit, len(matches))
    F = len(FEATURE_NAMES)
    X_raw = np.zeros((n_total, T_MAX, F), dtype=np.float32)
    ts_all = np.zeros((n_total, T_MAX), dtype=np.int64)
    lengths = np.zeros(n_total, dtype=np.int16)
    ids: list[str] = []
    no_events = 0
    for i, (mid, fr, ev) in enumerate(iter_matches(args.limit)):
        if len(ev) == 0:
            no_events += 1
        raw = match_to_features(fr, ev, parts_by[mid])
        n = len(raw)
        X_raw[i, :n] = raw.to_numpy(dtype=np.float32)
        ts_all[i, :n] = raw.attrs["timestamp_ms"]
        lengths[i] = n
        ids.append(mid)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{n_total:,} partidas  ({time.time() - t0:.0f} s)", flush=True)
    N = len(ids)
    X_raw, ts_all, lengths = X_raw[:N], ts_all[:N], lengths[:N]
    match_id = np.array(ids)
    print(f"Features de {N:,} partidas en {time.time() - t0:.0f} s"
          + (f"  (aviso: {no_events} sin eventos)" if no_events else ""))

    missing = set(ids) - set(splits.index)
    if missing:
        raise SystemExit(f"{len(missing)} partidas sin split en splits/splits.csv, p. ej. {list(missing)[:3]}")
    meta = matches.loc[ids]
    split = splits.loc[ids].to_numpy().astype(str)
    afk_arr = afk.reindex(ids).fillna(False).to_numpy().astype(bool)
    y = meta["blue_win"].to_numpy().astype(np.int8)
    mask = np.arange(T_MAX)[None, :] < lengths[:, None]

    # ---- scaler: SOLO filas válidas de train sin AFK (guía 3.5)
    fit_rows = mask & ((split == "train") & ~afk_arr)[:, None]
    scaler = Scaler.fit(X_raw[fit_rows])
    scaler.save(out / "scaler.json")
    X = scaler.transform(X_raw)
    X[~mask] = 0.0

    champs = parts.pivot_table(index="match_id", columns="participant_id", values="champion_id",
                               aggfunc="first").reindex(index=ids, columns=range(1, 11)).fillna(0)

    arrays = {
        "X": X, "mask": mask, "lengths": lengths, "y": y,
        "match_id": match_id, "split": split,
        "platform": meta["platform"].to_numpy().astype(str),
        "tier": meta["tier"].to_numpy().astype(str),
        "patch": meta["patch"].to_numpy().astype(str),
        "afk": afk_arr,
        "feature_names": np.array(FEATURE_NAMES), "feature_groups": np.array(FEATURE_GROUPS),
        "champions": champs.to_numpy().astype(np.int16),
    }
    from src.datasets import validate  # el mismo validador que usa B6
    validate(arrays)
    np.savez_compressed(out / "sequences.npz", **arrays)
    print(f"Guardado {out / 'sequences.npz'}  X={X.shape}")

    if not args.no_csv:
        rows = np.flatnonzero(mask.ravel())
        mi, t = np.divmod(rows, T_MAX)
        tm = pd.DataFrame(X_raw.reshape(-1, F)[rows], columns=FEATURE_NAMES)
        tm.insert(0, "match_id", match_id[mi])
        tm.insert(1, "timestamp_ms", ts_all.reshape(-1)[rows])
        tm["split"], tm["afk"], tm["blue_win"] = split[mi], afk_arr[mi], y[mi]
        tm["platform"], tm["tier"] = arrays["platform"][mi], arrays["tier"][mi]
        tm.to_csv(out / "team_minute.csv", index=False, float_format="%.6g")
        print(f"Guardado {out / 'team_minute.csv'}  ({len(tm):,} filas)")

    # ---- cordura: AUC de la diferencia de oro en train (EDA B4: 0.745 / 0.816 / 0.855)
    g = FEATURE_NAMES.index("gold_diff")
    tr = (split == "train") & ~afk_arr
    for m in (10, 15, 20):
        ok = tr & (lengths > m)
        if ok.sum() > 50 and len(np.unique(y[ok])) == 2:
            print(f"  AUC dif. de oro, min {m}: {roc_auc_score(y[ok], X_raw[ok, m, g]):.3f}  (n={ok.sum():,})")
    const = [n for n, s in zip(FEATURE_NAMES, scaler.std) if s == 1.0 and X_raw[fit_rows][:, FEATURE_NAMES.index(n)].std() == 0]
    if const:
        print(f"  aviso: features constantes en train: {const}")
    print(f"Listo en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
