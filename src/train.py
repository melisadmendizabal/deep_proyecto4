"""
B6 - Entrenamiento de modelos de deep learning (MLP, GRU y los que registre B7).

Uso:
  python src/train.py --config configs/gru_base.yaml
  python src/train.py --config configs/gru_base.yaml --set train.lr=3e-4 model.hidden=64
  python src/train.py --config configs/gru_base.yaml --smoke          # prueba rápida (~1 min)
  python src/train.py --config configs/gru_base.yaml --run-id mi_run --resume   # continuar

Receta (guía, sección 2):
  - Pérdida: BCE por minuto, promediada dentro de cada partida y luego entre partidas, con
    máscara para el padding. Así cada partida pesa lo mismo sin importar cuánto duró.
  - Opcional (E6): + smooth_lambda * promedio de (p_t - p_{t-1})^2, penaliza saltos bruscos.
  - AdamW, gradient clipping, ReduceLROnPlateau y early stopping por log-loss de validación.
  - Al final: temperature scaling ajustado en validación.

Salida en results/runs/<run_id>/:
  config.yaml      configuración completa usada
  metrics.csv      una fila por época (train_loss, val_loss, val_auc, val_brier, lr, segundos)
  best.pt          pesos del mejor modelo (+ config, features usadas, temperatura)
  last.pt          estado para reanudar (modelo, optimizador, época)
  summary.json     resumen (mejor época, métricas de val, tiempo, GPU, commit)
  predictions.npz  probabilidades calibradas por minuto para val y test: probs [N, T] (NaN = padding)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.datasets import SequenceData, iterate_batches, load_sequences  # noqa: E402
from src.models import build_model  # noqa: E402

DEFAULTS = {
    "name": "run",
    "data": {
        "path": "features/sequences.npz",
        "exclude_afk_train": True,
        "drop_groups": [],
        "drop_names": [],
        "tiers": None,           # E3: p. ej. [DIAMOND, MASTER]
        "platforms": None,       # E4: p. ej. [na1, euw1, kr]
        "max_train_matches": None,  # E2: curva de aprendizaje
        "subset_seed": 0,
    },
    "model": {"type": "gru", "hidden": 128, "layers": 1, "dropout": 0.1},
    "train": {
        "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 128, "max_epochs": 50,
        "patience": 5, "grad_clip": 1.0, "smooth_lambda": 0.0, "seed": 42,
        "lr_factor": 0.5, "lr_patience": 2, "device": "auto",
    },
    "output": {"dir": "results/runs"},
}


# ------------------------------------------------------------------ configuración
def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    *parents, last = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def load_config(path: str | Path | None, sets: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = deep_update(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in sets or []:
        key, _, raw = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


# ------------------------------------------------------------------ pérdida y métricas
def masked_bce(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
               smooth_lambda: float = 0.0) -> torch.Tensor:
    maskf = mask.float()
    target = y[:, None].expand_as(logits)
    per_step = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * maskf
    loss = (per_step.sum(1) / maskf.sum(1).clamp(min=1)).mean()
    if smooth_lambda > 0:
        p = torch.sigmoid(logits)
        pair = (mask[:, 1:] & mask[:, :-1]).float()
        jumps = ((p[:, 1:] - p[:, :-1]) ** 2 * pair).sum(1) / pair.sum(1).clamp(min=1)
        loss = loss + smooth_lambda * jumps.mean()
    return loss


@torch.no_grad()
def predict_logits(model, tensors, batch_size: int = 512) -> np.ndarray:
    """Logits [N, T_total] con NaN en el padding."""
    model.eval()
    n, t = tensors["mask"].shape
    out = np.full((n, t), np.nan, dtype=np.float32)
    for b in iterate_batches(tensors, batch_size, shuffle=False):
        logits = model(b["X"], b["mask"]).float()
        logits = logits.masked_fill(~b["mask"], float("nan"))
        out[b["idx"].cpu().numpy(), :logits.shape[1]] = logits.cpu().numpy()
    return out


def metrics_from_logits(logits: np.ndarray, data: SequenceData, temperature: float = 1.0) -> dict:
    mask = data.mask[:, :logits.shape[1]]
    z = logits / temperature
    y = np.broadcast_to(data.y[:, None], z.shape).astype(np.float64)
    zv, yv = z[mask].astype(np.float64), y[mask]
    # log-loss promediado por partida (igual que la pérdida de entrenamiento)
    per_step = np.logaddexp(0, zv) - yv * zv
    row = np.broadcast_to(np.arange(len(data))[:, None], z.shape)[mask]
    per_match = np.bincount(row, weights=per_step) / np.bincount(row)
    p = 1 / (1 + np.exp(-zv))
    return {
        "loss": float(per_match.mean()),
        "auc": float(roc_auc_score(yv, p)) if len(np.unique(yv)) > 1 else float("nan"),
        "brier": float(np.mean((p - yv) ** 2)),
    }


def fit_temperature(logits: np.ndarray, data: SequenceData) -> float:
    """Temperature scaling: un escalar T que minimiza el log-loss de validación."""
    mask = data.mask[:, :logits.shape[1]]
    z = torch.tensor(logits[mask], dtype=torch.float64)
    y = torch.tensor(np.broadcast_to(data.y[:, None], logits.shape)[mask], dtype=torch.float64)
    counts = mask.sum(1)
    w = torch.tensor(np.repeat(1.0 / counts, counts), dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    # sin búsqueda de línea, LBFGS con paso fijo se pasaba (T=0.0002 con modelos poco confiados)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(z / log_t.exp(), y, reduction="none") * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


# ------------------------------------------------------------------ utilidades
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def set_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def prepare_data(cfg: dict, full: SequenceData | None = None):
    d = cfg["data"]
    if full is None:
        full = load_sequences(ROOT / d["path"] if not Path(d["path"]).is_absolute() else d["path"])
    full = full.select_features(d.get("drop_groups") or [], d.get("drop_names") or [])
    train = full.subset(split="train", exclude_afk=d["exclude_afk_train"], tiers=d["tiers"],
                        platforms=d["platforms"], max_matches=d["max_train_matches"],
                        seed=d["subset_seed"])
    val = full.subset(split="val")
    test = full.subset(split="test")
    if len(train) == 0 or len(val) == 0:
        raise ValueError("train o val quedaron vacíos: revisa los filtros de data")
    return train, val, test


# ------------------------------------------------------------------ entrenamiento
def run(cfg: dict, run_dir: str | Path, resume: bool = False,
        full: SequenceData | None = None, verbose: bool = True) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg["train"]
    device = resolve_device(tc["device"])
    gen = set_seed(tc["seed"])
    log = print if verbose else (lambda *a, **k: None)

    train, val, test = prepare_data(cfg, full)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    log(f"[{run_dir.name}] train={len(train):,} val={len(val):,} test={len(test):,} "
        f"features={train.n_features} device={device}")

    model = build_model(cfg["model"], train.n_features).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tc["lr_factor"],
                                                       patience=tc["lr_patience"])
    tr_t, va_t = train.to_tensors(device), val.to_tensors(device)

    start_epoch, best_loss, best_epoch, bad_epochs, elapsed = 1, math.inf, 0, 0, 0.0
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu())
        start_epoch, best_loss = ck["epoch"] + 1, ck["best_loss"]
        best_epoch, bad_epochs, elapsed = ck["best_epoch"], ck["bad_epochs"], ck["elapsed"]
        log(f"  reanudando desde la época {start_epoch}")
    elif metrics_path.exists():
        metrics_path.unlink()

    meta = {"config": cfg, "feature_names": train.feature_names.tolist(),
            "feature_groups": train.feature_groups.tolist()}
    for epoch in range(start_epoch, tc["max_epochs"] + 1):
        if bad_epochs >= tc["patience"]:
            break
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for b in iterate_batches(tr_t, tc["batch_size"], shuffle=True, generator=gen):
            opt.zero_grad(set_to_none=True)
            loss = masked_bce(model(b["X"], b["mask"]), b["y"], b["mask"], tc["smooth_lambda"])
            loss.backward()
            if tc["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            total += loss.item() * len(b["y"])
            count += len(b["y"])
        vm = metrics_from_logits(predict_logits(model, va_t), val)
        sched.step(vm["loss"])
        secs = time.time() - t0
        elapsed += secs

        improved = vm["loss"] < best_loss - 1e-5
        if improved:
            best_loss, best_epoch, bad_epochs = vm["loss"], epoch, 0
            torch.save({**meta, "model": model.state_dict(), "epoch": epoch}, best_path)
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "train_loss": round(total / count, 5),
               "val_loss": round(vm["loss"], 5), "val_auc": round(vm["auc"], 5),
               "val_brier": round(vm["brier"], 5), "lr": opt.param_groups[0]["lr"],
               "seconds": round(secs, 2)}
        with metrics_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "gen": gen.get_state(), "epoch": epoch,
                    "best_loss": best_loss, "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs, "elapsed": elapsed}, last_path)
        log(f"  época {epoch:>3}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
            f"auc {row['val_auc']:.4f}  lr {row['lr']:.1e}  {secs:.1f}s{'  *' if improved else ''}")

    # ---- mejor modelo + calibración + predicciones
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    val_logits = predict_logits(model, va_t)
    temperature = fit_temperature(val_logits, val)
    raw, cal = metrics_from_logits(val_logits, val), metrics_from_logits(val_logits, val, temperature)
    best["temperature"] = temperature
    torch.save(best, best_path)

    te_t = test.to_tensors(device)
    test_logits = predict_logits(model, te_t)
    del te_t
    # probabilidades calibradas por minuto (NaN en el padding) para evaluate.py / moments.py
    logits_all = np.concatenate([val_logits, test_logits])
    np.savez_compressed(
        run_dir / "predictions.npz",
        match_id=np.concatenate([val.match_id, test.match_id]),
        split=np.concatenate([val.split, test.split]),
        y=np.concatenate([val.y, test.y]).astype(np.int8),
        probs=(1 / (1 + np.exp(-logits_all / temperature))).astype(np.float32),
        temperature=np.float32(temperature),
    )

    summary = {
        "run_id": run_dir.name, "name": cfg["name"], "model": cfg["model"]["type"],
        "best_epoch": best_epoch, "epochs_run": int(_count_rows(metrics_path)),
        "val_loss": round(raw["loss"], 5), "val_auc": round(raw["auc"], 5),
        "val_brier": round(raw["brier"], 5),
        "val_loss_calibrated": round(cal["loss"], 5), "temperature": round(temperature, 4),
        "n_train": len(train), "n_val": len(val), "n_features": train.n_features,
        "n_params": n_params, "train_seconds": round(elapsed, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "git_commit": git_commit(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"  mejor época {best_epoch}: val_loss {summary['val_loss']} (calibrado "
        f"{summary['val_loss_calibrated']}, T={summary['temperature']}), "
        f"val_auc {summary['val_auc']}, {summary['train_seconds']}s")
    return summary


def _count_rows(path: Path) -> int:
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="cambios puntuales: train.lr=3e-4")
    parser.add_argument("--run-id", help="nombre de la carpeta en results/runs")
    parser.add_argument("--resume", action="store_true", help="continuar desde last.pt")
    parser.add_argument("--smoke", action="store_true",
                        help="prueba rápida: 500 partidas de train, 1 época")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg["data"]["max_train_matches"] = 500
        cfg["train"]["max_epochs"] = 1
        cfg["name"] += "_smoke"
    run_id = args.run_id or f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run(cfg, ROOT / cfg["output"]["dir"] / run_id, resume=args.resume)


if __name__ == "__main__":
    main()
