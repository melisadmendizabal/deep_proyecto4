"""
B11 - Exporta el modelo para la app: pesos en CPU de cada semilla + scaler + temperatura + θ*.

Uso:
  python src/export_model.py                       # M4 GRU, 3 semillas de results/runs/gru_final__s4*
  python src/export_model.py --retrain             # si faltan los best.pt: reentrena con el config.yaml
                                                   # de cada run (los pesos no van a git)
  python src/export_model.py --runs results/runs/tcn_final__s42 ... --name M5 --out models/m5_tcn
  python src/export_model.py --baseline m2 --name M2 --out models/m2_lgbm     # baselines de B5

Modelos que ofrece la app (selector; M4 es el principal):
  python src/export_model.py --runs results/runs/mlp_final__s4{2,3,4} --name M3 --out models/m3_mlp --retrain
  python src/export_model.py --runs results/runs/tcn_final__s4{2,3,4} --name M5 --out models/m5_tcn --retrain
  python src/export_model.py --baseline m0 --name M0 --out models/m0_oro
  python src/export_model.py --baseline m1 --name M1 --out models/m1_logit
  python src/export_model.py --baseline m2 --name M2 --out models/m2_lgbm
  python src/export_model.py --runs results/runs/m6d_final__s4{2,3,4} --name "M6+draft ens3" --out models/m6_draft       --promedio --theta 0.15      # M6 Transformer + campeones, curva = promedio de semillas (results/mejoras.md)

Salida (models/, fuera de git; para desplegar se sube a Drive o como Release):
  models/<out>/manifest.json   modelo, θ*, parámetros del detector, temperaturas, parches de train...
  models/<out>/seed_<s>.pt     {"model_cfg", "feature_names", "state_dict", "temperature"}
  models/<out>/scaler.json     copia de features/scaler.json (la app normaliza con esto)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datasets import load_sequences  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402
from src.moments import DEFAULTS  # noqa: E402
from src.predict_matches import theta_star  # noqa: E402
from src.train import git_commit, run as train_run  # noqa: E402

DEFAULT_RUNS = [ROOT / "results" / "runs" / f"gru_final__s{s}" for s in (42, 43, 44)]


def ensure_weights(run: Path, retrain: bool, full=None) -> Path:
    """Devuelve la carpeta que tiene best.pt (la del run, o una reentrenada en models/_train)."""
    if (run / "best.pt").exists():
        return run
    local = ROOT / "models" / "_train" / run.name
    if (local / "best.pt").exists() and (local / "summary.json").exists():
        return local
    if not retrain:
        raise SystemExit(f"No existe {run / 'best.pt'} (los pesos no van a git). Pídeselo a quien "
                         "lo entrenó o corre con --retrain")
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    print(f"  reentrenando {run.name} con su config.yaml -> {local.relative_to(ROOT)}")
    train_run(cfg, local, full=full, verbose=False)
    return local


def export_baseline(key: str, out: Path) -> tuple[list[dict], dict]:
    """M0/M1 (logística: coeficientes a JSON, la app no necesita sklearn) y M2 (LightGBM: model.txt)."""
    from src.baselines import MODELS, RUNS
    run = RUNS / MODELS[key]["run"]
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    T = float(s["temperature"])
    if MODELS[key]["kind"] == "logit":
        import pickle
        with open(run / "model.pkl", "rb") as f:
            model = pickle.load(f)
        (out / "logit.json").write_text(json.dumps({
            "feature_names": s["feature_names"], "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]), "temperature": T}, indent=1), encoding="utf-8")
        member = {"file": "logit.json", "kind": "logit", "seed": 0, "temperature": T}
    else:
        shutil.copy(run / "model.txt", out / "model.txt")
        member = {"file": "model.txt", "kind": "lgbm", "seed": 42, "temperature": T}
    member.update({"source": str(run.relative_to(ROOT)), "val_loss": s.get("val_loss"), "val_auc": s.get("val_auc")})
    print(f"  {out.relative_to(ROOT)}/{member['file']}  T={T:.3f}  val_auc={s.get('val_auc')}")
    return [member], s


def _average_info(runs: list[Path]) -> dict:
    """Temperatura del promedio de semillas, ajustada en val igual que en src/mejoras_eval.py."""
    from src.mejoras_eval import make_ensemble
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ens = make_ensemble([r.name for r in runs], Path(tmp))
        with np.load(ens / "predictions.npz", allow_pickle=False) as z:
            T = float(z["temperature"])
    print(f"  curva = promedio de {len(runs)} semillas, temperatura del promedio {T:.4f}")
    return {"curva": "promedio", "temperatura_promedio": T}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[str(r) for r in DEFAULT_RUNS])
    ap.add_argument("--name", default="M4", help="nombre en results/b10_valor_theta.csv (para θ*)")
    ap.add_argument("--out", default=str(ROOT / "models" / "m4_gru"))
    ap.add_argument("--theta", type=float, help="forzar θ (por defecto: θ* elegido en val por B10)")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--baseline", choices=["m0", "m1", "m2"], help="exportar un baseline de B5 en vez de --runs")
    ap.add_argument("--promedio", action="store_true",
                    help="curva = promedio de las semillas recalibrado en val (src/mejoras_eval.make_ensemble)")
    ap.add_argument("--manifest-name", help="nombre que muestra la app (default: --name)")
    args = ap.parse_args()

    runs = [] if args.baseline else [Path(r).resolve() for r in args.runs]
    full = None
    if args.retrain and any(not (r / "best.pt").exists() for r in runs):
        full = load_sequences(ROOT / "features" / "sequences.npz")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    members, summaries = [], []
    if args.baseline:
        members, s = export_baseline(args.baseline, out)
        summaries.append(s)
    for run in runs:
        src = ensure_weights(run, args.retrain, full)
        ck = torch.load(src / "best.pt", map_location="cpu", weights_only=False)
        seed = int(ck["config"]["train"]["seed"])
        path = out / f"seed_{seed}.pt"
        torch.save({"model_cfg": ck["config"]["model"], "feature_names": ck["feature_names"],
                    "state_dict": {k: v.cpu() for k, v in ck["model"].items()},
                    "temperature": float(ck.get("temperature", 1.0))}, path)
        s = json.loads((src / "summary.json").read_text(encoding="utf-8"))
        summaries.append(s)
        members.append({"file": path.name, "seed": seed, "temperature": float(ck.get("temperature", 1.0)),
                        "source": str(src.relative_to(ROOT)), "retrained_here": src != run,
                        "val_loss": s.get("val_loss"), "val_auc": s.get("val_auc")})
        print(f"  {path.relative_to(ROOT)}  T={members[-1]['temperature']:.3f}  val_auc={s.get('val_auc')}")

    shutil.copy(ROOT / "features" / "scaler.json", out / "scaler.json")
    theta = args.theta if args.theta is not None else theta_star(args.name)
    patches = []
    seq = ROOT / "features" / "sequences.npz"
    if seq.exists():
        with np.load(seq, allow_pickle=False) as z:
            patches = sorted(set(z["patch"][z["split"] == "train"].tolist()))
    manifest = {
        "name": args.manifest_name or args.name, "model_type": summaries[0].get("model"),
        "feature_names_all": list(FEATURE_NAMES), "members": members,
        "detector": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in DEFAULTS.items()},
                     "theta": theta},
        "train_patches": patches, "queue_id": 420, "min_duration_s": 15 * 60,
        **(_average_info(runs) if args.promedio else {}),
        "exported_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "git_commit": git_commit(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Listo: {out.relative_to(ROOT)}  ({len(members)} semillas, θ = {theta:.2f}, parches de train {patches})")


if __name__ == "__main__":
    main()
