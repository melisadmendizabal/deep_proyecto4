"""
B6 - Búsqueda de hiperparámetros (grid) y corridas con varias semillas.

Pensado para dejarlo corriendo de noche: si se corta (o se apaga la PC), al volver a
ejecutar el mismo comando SALTA las corridas ya terminadas y sigue con las demás.

Uso:
  python src/hp_search.py --config configs/gru_base.yaml --grid configs/grid_gru.yaml --prefix gru_hp
  python src/hp_search.py --config configs/gru_base.yaml --prefix gru_final --seeds 42 43 44 \\
      --set model.hidden=128 model.layers=1         # mejor config x 3 semillas
  python src/hp_search.py ... --smoke              # prueba rápida antes de irse a dormir

Salidas:
  results/runs/<prefix>__<combinación>/   una carpeta por corrida (ver src/train.py)
  results/hp/<prefix>.csv                 todas las corridas ordenadas por val_loss
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.datasets import load_sequences  # noqa: E402
from src.train import ROOT as _ROOT, load_config, run, set_dotted  # noqa: E402,F401

SHORT = {"model.hidden": "h", "model.layers": "l", "model.dropout": "d", "train.lr": "lr",
         "train.weight_decay": "wd", "train.batch_size": "bs", "train.smooth_lambda": "sm",
         "data.max_train_matches": "n", "train.seed": "s"}


def combo_name(combo: dict) -> str:
    parts = []
    for k, v in combo.items():
        key = SHORT.get(k, k.split(".")[-1])
        parts.append(f"{key}{v}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="config base (YAML)")
    parser.add_argument("--grid", help="YAML {clave.punteada: [valores]}")
    parser.add_argument("--prefix", required=True, help="nombre del grupo de corridas")
    parser.add_argument("--seeds", nargs="*", type=int, help="semillas (default: la del config)")
    parser.add_argument("--set", nargs="*", default=[], help="cambios fijos: model.hidden=128")
    parser.add_argument("--smoke", action="store_true",
                        help="cada corrida con 500 partidas y 1 época (para probar el comando)")
    args = parser.parse_args()

    base = load_config(args.config, args.set)
    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8")) if args.grid else {}
    if args.seeds:
        grid["train.seed"] = args.seeds
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*grid.values())] or [{}]
    prefix = args.prefix + ("_smoke" if args.smoke else "")

    out_dir = ROOT / base["output"]["dir"]
    print(f"{len(combos)} corridas en {out_dir.relative_to(ROOT)} (prefijo '{prefix}')")
    data_path = Path(base["data"]["path"])
    full = load_sequences(data_path if data_path.is_absolute() else ROOT / data_path)

    rows, t_start = [], time.time()
    for i, combo in enumerate(combos, 1):
        run_id = f"{prefix}__{combo_name(combo)}" if combo else prefix
        run_dir = out_dir / run_id
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            print(f"[{i}/{len(combos)}] {run_id}: ya terminada, se salta")
        else:
            cfg = load_config(args.config, args.set)
            for k, v in combo.items():
                set_dotted(cfg, k, v)
            cfg["name"] = run_id
            if args.smoke:
                cfg["data"]["max_train_matches"] = 500
                cfg["train"]["max_epochs"] = 1
            print(f"[{i}/{len(combos)}] {run_id}")
            try:
                run(cfg, run_dir, resume=True, full=full, verbose=True)
            except Exception:  # una corrida rota no detiene la búsqueda nocturna
                traceback.print_exc()
                (run_dir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
                continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({**combo, **summary})

        table = pd.DataFrame(rows).sort_values("val_loss")
        hp_dir = ROOT / "results" / "hp"
        hp_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(hp_dir / f"{prefix}.csv", index=False)

    if rows:
        cols = keys + ["val_loss", "val_loss_calibrated", "val_auc", "best_epoch",
                       "train_seconds"]
        print(f"\nTerminado en {(time.time() - t_start) / 60:.1f} min. Mejores corridas:")
        print(table[[c for c in cols if c in table]].head(8).to_string(index=False))
        print(f"\nTabla completa: results/hp/{prefix}.csv")


if __name__ == "__main__":
    main()
