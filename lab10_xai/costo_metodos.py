"""
Lab 10 · Parte 2: costo de los métodos de XAI candidatos sobre el modelo final (M6 + campeones, 3 semillas).

Mide, en CPU y con 1 hilo (como en Render), cuánto tarda cada método en explicar un minuto de una partida
de test, y verifica la propiedad de completitud de Integrated Gradients (suma de atribuciones = cambio del logit).

    python lab10_xai/costo_metodos.py            # escribe lab10_xai/resultados/costo_metodos.csv

En una laptop con núcleos P y E los tiempos varían mucho si Windows manda el proceso a núcleos E; para medir
igual que en la tabla E1 del proyecto (PowerShell):
    $env:OMP_NUM_THREADS='1'; $p = Start-Process python -ArgumentList 'lab10_xai/costo_metodos.py' -PassThru -NoNewWindow
    $p.ProcessorAffinity=[IntPtr]0x554; $p.PriorityClass='High'; $p.WaitForExit()

Qué se compara (minuto explicado t = mitad de la partida, ventana del "momento" = minutos t-2..t):
- IG con 16, 32 y 64 pasos (regla del punto medio). Línea base "partida congelada": los minutos t-1 y t se
  reemplazan por el estado del minuto t-2, el mismo contrafactual que usa la explicación actual de la app.
- Oclusión feature × minuto dentro de la ventana (3·F variantes, en un solo batch).
- Oclusión feature × minuto en toda la partida (F·T variantes, en batches de 256).
- Una pasada normal (forward) de las 3 semillas, como referencia.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.inference import Engine, champions_of  # noqa: E402
from src.features import EVENT_COLS, EVENT_TYPES_USED, FRAME_COLS, match_to_features  # noqa: E402
from src.predict_matches import load_raw  # noqa: E402
from src.value_model import ANN_DIR  # noqa: E402

SEED = 42
N_MATCHES = 5
OUT = Path(__file__).parent / "resultados"


def f_logit(eng: Engine, x: torch.Tensor, ch: torch.Tensor) -> torch.Tensor:
    """x [B, T, F] estandarizado -> logit de la curva de la app (promedio de semillas recalibrado) [B, T].
    Es la misma función que Engine.combine, pero diferenciable."""
    ps = []
    for m in eng.members:
        lg = m.model(x[..., m.cols], None, champions=ch.expand(x.shape[0], -1))
        ps.append(torch.sigmoid(lg / m.temperature))
    p = torch.stack(ps).mean(0).clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p)) / eng.avg_temperature


def integrated_gradients(eng, x, base, ch, t, steps):
    """IG del logit del minuto t respecto a toda la secuencia x [T, F]. Devuelve (atribuciones [T, F], Δ logit)."""
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    path = (base + alphas[:, None, None] * (x - base)).requires_grad_(True)
    g, = torch.autograd.grad(f_logit(eng, path, ch)[:, t].sum(), path)
    attr = (x - base) * g.mean(0)
    with torch.no_grad():
        delta = (f_logit(eng, x[None], ch)[0, t] - f_logit(eng, base[None], ch)[0, t]).item()
    return attr, delta


def main() -> None:
    torch.manual_seed(SEED)
    eng = Engine(ROOT / "models" / "m6_draft")
    torch.set_num_threads(1)
    sel = pd.read_csv(ANN_DIR / "seleccion.csv")
    ids = sel.loc[sel["split"] == "test", "match_id"].tolist()[:N_MATCHES]
    rows = []
    for mid, tb in load_raw(ids).items():
        ev = tb["events"][tb["events"]["type"].isin(EVENT_TYPES_USED)]
        feats = match_to_features(tb["frames"][FRAME_COLS], ev.reindex(columns=EVENT_COLS), tb["participants"])
        ch = torch.tensor(champions_of(tb["participants"]), dtype=torch.long).reshape(1, -1)
        X = torch.tensor(eng.scaler.transform(feats.to_numpy(dtype=np.float32)), dtype=torch.float32)
        T, F = X.shape
        t = T // 2
        base = X.clone()
        base[t - 1:t + 1] = X[t - 2]
        integrated_gradients(eng, X, base, ch, t, 8)   # calentar
        for steps in (16, 32, 64):
            t0 = time.perf_counter()
            attr, delta = integrated_gradients(eng, X, base, ch, t, steps)
            ms = 1000 * (time.perf_counter() - t0)
            rows.append(dict(match_id=mid, minutos=T, metodo=f"IG {steps} pasos", ms=ms, pasadas=steps,
                             delta_logit=abs(delta), error_completitud=abs(attr.sum().item() - delta)))
        with torch.no_grad():
            t0 = time.perf_counter()
            V = X[None].repeat(3 * F, 1, 1)
            for k, (mm, j) in enumerate((mm, j) for mm in range(t - 2, t + 1) for j in range(F)):
                V[k, mm, j] = X[t - 3, j]
            f_logit(eng, V, ch)
            rows.append(dict(match_id=mid, minutos=T, metodo="Oclusión feature×minuto (ventana de 3 min)",
                             ms=1000 * (time.perf_counter() - t0), pasadas=3 * F))
            t0 = time.perf_counter()
            V = X[None].repeat(F * T, 1, 1)
            for k, (mm, j) in enumerate((mm, j) for mm in range(T) for j in range(F)):
                V[k, mm, j] = 0.0   # 0 = media de train tras estandarizar
            for b in range(0, len(V), 256):
                f_logit(eng, V[b:b + 256], ch)
            rows.append(dict(match_id=mid, minutos=T, metodo="Oclusión feature×minuto (partida completa)",
                             ms=1000 * (time.perf_counter() - t0), pasadas=F * T))
            t0 = time.perf_counter()
            f_logit(eng, X[None], ch)
            rows.append(dict(match_id=mid, minutos=T, metodo="Pasada normal (referencia)",
                             ms=1000 * (time.perf_counter() - t0), pasadas=1))
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "costo_metodos_detalle.csv", index=False)
    summ = df.groupby("metodo", sort=False).agg(
        ms_mediana=("ms", "median"), pasadas_mediana=("pasadas", "median"), minutos_mediana=("minutos", "median"),
        delta_logit_mediana=("delta_logit", "median"), error_completitud_max=("error_completitud", "max"))
    summ.round(6).to_csv(OUT / "costo_metodos.csv")
    pd.set_option("display.width", 200)
    print(summ.round(5).to_string())


if __name__ == "__main__":
    main()
