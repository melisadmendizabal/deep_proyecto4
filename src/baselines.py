"""
B5 - Baselines clásicos (guía, sección 2) + regla R0 + pruebas de cordura de fuga.

  M0  logística "solo oro": gold_diff + minute                         (el listón chafa)
  M1  logística, todas las features del minuto actual (sin tendencia)
  M2  LightGBM, todas las features + tendencia (Δ oro 1 y 3 min), early stopping en val
  R0  regla sin ML para momentos: barón, ace, inhibidor o swing de oro > 1,500 en 1 min

Uso:
  python src/baselines.py                     # M0, M1, M2 + cordura (~3-6 min, CPU)
  python src/baselines.py --models m2         # solo uno
  python src/baselines.py --no-sanity

Modelos snapshot: una fila por (partida, minuto), con el minuto como feature, y cada fila pesa
1/duración para que las partidas largas no dominen (guía, sección 2). Train sin AFK.
Calibración: temperature scaling en val (la misma función que usa train.py).

Salida en results/runs/<m0_oro|m1_logit|m2_lgbm>/, con el MISMO formato que train.py:
  predictions.npz (probs [N, T] calibradas, NaN = padding), summary.json, model.pkl / model.txt
y results/b5_cordura.csv con las pruebas de cordura. Luego: python src/evaluate.py ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datasets import SequenceData, load_sequences  # noqa: E402
from src.train import fit_temperature, git_commit, metrics_from_logits  # noqa: E402

RUNS = ROOT / "results" / "runs"
GOLD_SWING = 1500

MODELS = {
    "m0": {"run": "m0_oro", "kind": "logit", "names": ["gold_diff", "minute"]},
    "m1": {"run": "m1_logit", "kind": "logit", "drop_groups": ["tendencia"]},
    "m2": {"run": "m2_lgbm", "kind": "lgbm", "drop_groups": []},
}
LGBM_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbose": -1, "seed": 42, "num_threads": 0,
}


# ------------------------------------------------------------------ datos snapshot
def select(data: SequenceData, spec: dict) -> SequenceData:
    if "names" in spec:
        drop = [n for n in data.feature_names if n not in spec["names"]]
        return data.select_features(drop_names=drop)
    return data.select_features(drop_groups=spec.get("drop_groups", []))


def rows(data: SequenceData):
    """X [filas, F], y [filas], peso 1/duración normalizado a media 1."""
    X = data.X[data.mask]
    y = np.broadcast_to(data.y[:, None], data.mask.shape)[data.mask].astype(np.int8)
    w = np.broadcast_to(1.0 / data.lengths[:, None], data.mask.shape)[data.mask]
    return X, y, w * len(w) / w.sum()


def to_grid(values: np.ndarray, data: SequenceData) -> np.ndarray:
    """Filas -> [N, T] con NaN en el padding (formato de predictions.npz)."""
    out = np.full(data.mask.shape, np.nan, dtype=np.float32)
    out[data.mask] = values
    return out


# ------------------------------------------------------------------ entrenamiento
def fit_model(spec: dict, train: SequenceData, val: SequenceData):
    Xtr, ytr, wtr = rows(train)
    if spec["kind"] == "logit":
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(Xtr, ytr, sample_weight=wtr)
        return model, (lambda X: model.decision_function(X))
    import lightgbm as lgb
    Xva, yva, wva = rows(val)
    names = list(train.feature_names)
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr, feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, weight=wva, reference=dtr)
    model = lgb.train(LGBM_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
    return model, (lambda X: model.predict(X, raw_score=True, num_iteration=model.best_iteration))


def run_baseline(key: str, full: SequenceData) -> dict:
    spec = MODELS[key]
    data = select(full, spec)
    train = data.subset(split="train", exclude_afk=True)
    val, test = data.subset(split="val"), data.subset(split="test")
    run_dir = RUNS / spec["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{spec['run']}] train={len(train):,} partidas ({train.mask.sum():,} filas)  "
          f"features={train.n_features}")

    t0 = time.time()
    model, logit_fn = fit_model(spec, train, val)
    secs = time.time() - t0
    val_logits = to_grid(logit_fn(val.X[val.mask]), val)
    t1 = time.time()
    test_logits = to_grid(logit_fn(test.X[test.mask]), test)
    infer_ms_per_match = (time.time() - t1) * 1000 / len(test)
    temperature = fit_temperature(val_logits, val)
    raw = metrics_from_logits(val_logits, val)
    cal = metrics_from_logits(val_logits, val, temperature)

    logits_all = np.concatenate([val_logits, test_logits])
    np.savez_compressed(
        run_dir / "predictions.npz",
        match_id=np.concatenate([val.match_id, test.match_id]),
        split=np.concatenate([val.split, test.split]),
        y=np.concatenate([val.y, test.y]).astype(np.int8),
        probs=(1 / (1 + np.exp(-logits_all / temperature))).astype(np.float32),
        temperature=np.float32(temperature),
    )
    if spec["kind"] == "logit":
        with open(run_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        model.save_model(str(run_dir / "model.txt"), num_iteration=model.best_iteration)

    summary = {
        "run_id": spec["run"], "name": spec["run"], "model": spec["kind"],
        "val_loss": round(raw["loss"], 5), "val_auc": round(raw["auc"], 5),
        "val_brier": round(raw["brier"], 5), "val_loss_calibrated": round(cal["loss"], 5),
        "temperature": round(temperature, 4), "n_train": len(train), "n_val": len(val),
        "n_features": train.n_features, "feature_names": train.feature_names.tolist(),
        "train_seconds": round(secs, 1), "infer_ms_per_match": round(infer_ms_per_match, 4),
        "device": "cpu", "git_commit": git_commit(),
    }
    if spec["kind"] == "lgbm":
        summary["best_iteration"] = int(model.best_iteration)
        imp = model.feature_importance("gain")
        summary["importance_gain"] = dict(sorted(zip(train.feature_names.tolist(), imp.round(1).tolist()),
                                                 key=lambda kv: -kv[1]))
    else:
        summary["coef"] = dict(zip(train.feature_names.tolist(), model.coef_[0].round(4).tolist()))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  val_loss {summary['val_loss']} (calibrado {summary['val_loss_calibrated']}, "
          f"T={summary['temperature']})  val_auc {summary['val_auc']}  {secs:.1f}s")
    return summary


# ------------------------------------------------------------------ cordura (guía 3.3)
def sanity_checks(full: SequenceData) -> pd.DataFrame:
    """Etiquetas barajadas y 'solo minuto' deben dar AUC ~ 0.5 en val."""
    out = []
    data = select(full, MODELS["m1"])
    train = data.subset(split="train", exclude_afk=True)
    val = data.subset(split="val")
    rng = np.random.default_rng(0)

    # 5 permutaciones: con una sola, el AUC puede alejarse de 0.5 solo porque la dirección
    # aleatoria aprendida se correlaciona con el oro; la media debe quedar en ~0.5
    yv = np.broadcast_to(val.y[:, None], val.mask.shape)[val.mask]
    aucs = []
    for _ in range(5):
        shuffled = train._take(np.arange(len(train)))
        shuffled.y = rng.permutation(train.y)                   # baraja POR PARTIDA
        model, fn = fit_model({"kind": "logit"}, shuffled, val)
        aucs.append(roc_auc_score(yv, fn(val.X[val.mask])))
    out.append({"prueba": f"etiquetas barajadas (M1), media de 5 [min {min(aucs):.3f}, max {max(aucs):.3f}]",
                "auc_val": float(np.mean(aucs)), "esperado": "~0.5"})

    only_min = select(full, {"names": ["minute"]})
    tr_m, va_m = only_min.subset(split="train", exclude_afk=True), only_min.subset(split="val")
    model, fn = fit_model({"kind": "logit"}, tr_m, va_m)
    out.append({"prueba": "solo minuto", "auc_val": roc_auc_score(yv, fn(va_m.X[va_m.mask])),
                "esperado": "~0.5"})

    # alarma de fuga: AUC de M1 en el minuto 10 no debería pasar de ~0.9 (guía, riesgos)
    model, fn = fit_model({"kind": "logit"}, train, val)
    m10 = val.lengths > 10
    out.append({"prueba": "M1 en el minuto 10 (alarma si > 0.9)",
                "auc_val": roc_auc_score(val.y[m10], fn(val.X[m10, 10])), "esperado": "< 0.9"})
    df = pd.DataFrame(out)
    df["auc_val"] = df["auc_val"].round(4)
    df.to_csv(ROOT / "results" / "b5_cordura.csv", index=False)
    print(df.to_string(index=False))
    return df


# ------------------------------------------------------------------ regla R0
def rule_r0(features: pd.DataFrame, events: pd.DataFrame, gold_swing: float = GOLD_SWING) -> list[int]:
    """Minutos marcados por la regla sin ML (sin unir ni recortar a top-K; eso lo hace moments.py).

    features: salida de src.features.match_to_features (crudas, con attrs['timestamp_ms']).
    events:   eventos de la partida (columnas de events.csv).
    Un evento cuenta en el primer frame que ya lo "ve" (timestamp_ms <= timestamp del frame).
    """
    at = np.asarray(features.attrs["timestamp_ms"])
    minutes = set()
    if len(events):
        typ = events["type"].astype(str)
        hit = ((typ == "ELITE_MONSTER_KILL") & (events.get("monster_type") == "BARON_NASHOR")) \
            | ((typ == "CHAMPION_SPECIAL_KILL") & (events.get("kill_type") == "KILL_ACE")) \
            | ((typ == "BUILDING_KILL") & (events.get("building_type") == "INHIBITOR_BUILDING"))
        ts = pd.to_numeric(events.loc[hit, "timestamp_ms"]).to_numpy()
        idx = np.searchsorted(at, ts, side="left")
        minutes |= {int(i) for i in idx if i < len(at)}
    g = features["gold_diff"].to_numpy()
    minutes |= {int(t) for t in np.flatnonzero(np.abs(np.diff(g)) > gold_swing) + 1}
    return sorted(minutes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "features" / "sequences.npz"))
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()
    full = load_sequences(args.data)
    print(f"{len(full):,} partidas, {full.n_features} features")
    for key in args.models:
        run_baseline(key, full)
    if not args.no_sanity:
        sanity_checks(full)
    print("Siguiente: python src/evaluate.py --run "
          + " ".join(f"results/runs/{MODELS[k]['run']}" for k in args.models)
          + " --name " + " ".join(k.upper() for k in args.models))


if __name__ == "__main__":
    main()
