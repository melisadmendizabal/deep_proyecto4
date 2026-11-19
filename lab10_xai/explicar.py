"""
Lab 10 · Parte 3: Integrated Gradients (IG) sobre el modelo final, sin reentrenarlo.

Modelo: M6 Transformer causal + campeones, promedio de las 3 semillas recalibrado (models/m6_draft/),
la misma curva que muestra la app. Datos: split de test de features/sequences.npz (ya estandarizado
con el scaler de train; reproduce las predicciones guardadas del ensamble con error < 1e-4).

    python lab10_xai/explicar.py                  # 4-11 min en CPU según la PC
    python lab10_xai/explicar.py --n-global 100 --n-sens 30   # prueba rápida (~2 min) -> lab10_xai/prueba/

Escribe lab10_xai/resultados/*.csv, lab10_xai/resultados/resumen_parte3.md y lab10_xai/figuras/*.png
(con --n-global/--n-sens distintos a los del informe escribe en lab10_xai/prueba/, que no va a git).
Todo es determinista: la muestra y las instancias salen de np.random.default_rng(42) y la red está en
modo eval (sin dropout), así que dos corridas dan los mismos números.

Qué se explica (salida = logit de la curva de la app en el minuto t; parámetros y su justificación en
lab10_xai/README.md):
  1. NIVEL: "¿por qué P(gana azul) está donde está en el minuto t?". Línea base = "partida pareja":
     todas las diferencias en 0 (cada feature cruda en 0 y luego estandarizada), el MINUTO REAL
     (para no atribuirle nada al paso del tiempo) y los 10 campeones = id 0 (desconocido, embedding
     nulo). IG integra a la vez las 35 features * minutos 0..t y los embeddings de campeones de las
     3 semillas, así que la completitud cubre también el draft:
         Σ atribuciones (features) + Σ atribuciones (campeones) = logit(x) - logit(línea base).
  2. MOMENTO: "¿qué cambió para que la curva saltara?". Línea base = "partida congelada": los minutos
     start+1..end se reemplazan por el estado del minuto start (el mismo contrafactual que usa la
     explicación por oclusión de la app, app/inference.py · explain), salvo el minuto. El draft no
     cambia dentro de un momento, así que recibe 0.

Atribuciones en logit; para mostrarlas en "puntos de probabilidad" se reparten proporcionalmente:
pts_j = 100 · (P(x) - P(x')) · a_j / Σ a (aproximación de presentación; la suma cuadra exacto).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.inference import EXPLAIN_GROUPS, Engine, window_events  # noqa: E402
from src.datasets import load_sequences  # noqa: E402
from src.features import EVENT_COLS, EVENT_TYPES_USED, FEATURE_GROUPS, FEATURE_NAMES, FRAME_COLS  # noqa: E402
from src.moments import detect_moments  # noqa: E402

# parámetros de diseño
SEED = 42                       # muestra global e instancias locales
STEPS = 32                      # pasos de IG (regla del punto medio); error de completitud
T_LOCAL = 15                    # minuto de las instancias locales de "nivel" (todas las partidas duran ≥ 15)
MINUTES_GLOBAL = (10, 15, 20, 25)
N_GLOBAL = 1000                 # partidas de test en la explicación global
N_SENS = 200                    # partidas para las pruebas de sensibilidad (semilla, línea base, pasos)
CHUNK = 32                      # partidas por batch de IG (32 pasos * 32 partidas = 1024 secuencias)
CONF_ALTA, CONF_ERROR, CONF_UMBRAL = 0.40, 0.30, 0.02   # |P - 0.5| para acierto / error / umbral

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "resultados", HERE / "figuras"
MODEL_DIR = ROOT / "models" / "m6_draft"

GROUP_LABEL = {"contexto": "minuto", "economia": "economía", "rol": "oro por rol", "combate": "combate",
               "objetivos": "objetivos", "vision": "visión", "tendencia": "tendencia"}
GROUPS = ["economía", "oro por rol", "objetivos", "combate", "visión", "minuto", "campeones"]
FEAT_GROUP = np.array([GROUP_LABEL[g] for g in FEATURE_GROUPS])
I_MIN = FEATURE_NAMES.index("minute")
NAMES = list(FEATURE_NAMES) + ["campeones_azul", "campeones_rojo"]          # 35 features + draft por equipo
NAME_GROUP = np.array(list(FEAT_GROUP) + ["campeones", "campeones"])
APP_GROUP = {"economía": "economía", "oro por rol": "economía", "combate": "combate",   # grupos de la app
             "objetivos": "objetivos", "visión": "visión"}
APP_GROUPS = list(EXPLAIN_GROUPS)

BLUE, RED, INK, MUTED, GRID = "#2a78d6", "#e34948", "#0b0b0b", "#52514e", "#e4e3de"
GRAYS = ["#d9d8d3", "#b5b3ab", "#8c8a82", "#5f5d57", "#2f2e2b"]
SEQ = ["#ffffff", "#ecebe7", "#d9d8d3", "#b5b3ab", "#8c8a82", "#5f5d57", "#2f2e2b"] 


#  modelo diferenciable
def _seed_logit(model, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """Forward de TransformerModel (src/models/transformer.py) recibiendo los embeddings de los campeones
    [B, 10, d] en vez de los ids, para poder integrar sobre ellos. Devuelve logits [B, T]."""
    t = x.shape[1]
    h = model.input(x) + model.pos(torch.arange(t))[None]
    blue, red = emb[:, :5].mean(1), emb[:, 5:].mean(1)
    h = h + model.draft(torch.cat([blue, red, blue - red], dim=-1))[:, None, :]
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    return model.head(model.norm(model.encoder(h, mask=causal))).squeeze(-1)


class Explainer:
    def __init__(self, eng: Engine):
        self.eng, self.members = eng, eng.members

    def embeddings(self, ch: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            return [m.model.champ(ch.long().clamp(0, m.model.champ.num_embeddings - 1)) for m in self.members]

    def logit(self, x: torch.Tensor, embs: list[torch.Tensor], seeds=None) -> torch.Tensor:
        """Logit de la curva de la app [B, T] (= Engine.combine, diferenciable). seeds=[k]: una sola semilla."""
        idx = range(len(self.members)) if seeds is None else seeds
        zs = [_seed_logit(self.members[k].model, x[..., self.members[k].cols], embs[k]) / self.members[k].temperature
              for k in idx]
        if len(zs) == 1:
            return zs[0]
        p = torch.stack([torch.sigmoid(z) for z in zs]).mean(0).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p)) / self.eng.avg_temperature

    def ig(self, x, xb, embs, embs_b, steps=STEPS, seeds=None):
        """IG del logit del ÚLTIMO minuto de x [B, L, F] (el modelo es causal: basta cortar en t+1).
        Camino recto conjunto (features y embeddings de campeones de todas las semillas, mismo α).
        Devuelve (atrib. features [B, L, F], atrib. campeones [B, 10], logit(x) [B], logit(x') [B])."""
        B, L, F = x.shape
        a = ((torch.arange(steps, dtype=torch.float32) + 0.5) / steps).repeat_interleave(B)[:, None, None]
        rep = lambda t: t.repeat(steps, 1, 1)  
        X = (rep(xb) + a * rep(x - xb)).requires_grad_(True)
        Es = [(rep(eb) + a * rep(e - eb)).requires_grad_(True) for e, eb in zip(embs, embs_b)]
        out = self.logit(X, Es, seeds)[:, -1].sum()
        grads = torch.autograd.grad(out, [X] + Es, allow_unused=True)
        attr_x = (x - xb) * grads[0].view(steps, B, L, F).mean(0)
        attr_ch = torch.zeros(B, 10)
        for e, eb, g in zip(embs, embs_b, grads[1:]):
            if g is not None:
                attr_ch += ((e - eb) * g.view(steps, B, 10, -1).mean(0)).sum(-1)
        with torch.no_grad():
            fx, fb = self.logit(x, embs, seeds)[:, -1], self.logit(xb, embs_b, seeds)[:, -1]
        return attr_x.detach().numpy(), attr_ch.numpy(), fx.numpy(), fb.numpy()

    @torch.no_grad()
    def curves(self, X: torch.Tensor, CH: torch.Tensor, lengths: np.ndarray) -> np.ndarray:
        """P(gana azul) de la app para cada partida [N, T] (NaN después del final)."""
        out = []
        for b in range(0, len(X), 256):
            z = self.logit(X[b:b + 256], self.embeddings(CH[b:b + 256]))
            out.append(torch.sigmoid(z).numpy().astype(np.float64))
        P = np.concatenate(out)
        P[np.arange(P.shape[1])[None] >= lengths[:, None]] = np.nan
        return P


def even_baseline(eng: Engine, x: torch.Tensor) -> torch.Tensor:
    """Partida pareja: todas las diferencias crudas = 0 (luego estandarizadas), con el minuto real."""
    row = torch.tensor(eng.scaler.transform(np.zeros((1, len(FEATURE_NAMES)), np.float32))[0])
    xb = row.expand_as(x).clone()
    xb[..., I_MIN] = x[..., I_MIN]
    return xb


def frozen_baseline(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Partida congelada: minutos start+1..end = estado del minuto start (menos el minuto)."""
    xb = x.clone()
    cols = [j for j in range(x.shape[-1]) if j != I_MIN]
    xb[:, start + 1:end + 1, cols] = x[:, start:start + 1, cols]
    return xb


def zeros_like_embs(embs):
    return [torch.zeros_like(e) for e in embs]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def per_name(attr_x: np.ndarray, attr_ch: np.ndarray) -> np.ndarray:
    """[B, L, F] + [B, 10] -> [B, 37]: suma sobre minutos por feature, y el draft por equipo."""
    return np.concatenate([attr_x.sum(1), attr_ch[:, :5].sum(1, keepdims=True),
                           attr_ch[:, 5:].sum(1, keepdims=True)], axis=1)


def group_sum(a: np.ndarray, absolute: bool) -> pd.DataFrame:
    """[B, 37] -> [B, grupos]: suma de |a| (importancia) o de a (con signo) por grupo."""
    v = np.abs(a) if absolute else a
    return pd.DataFrame({g: v[:, NAME_GROUP == g].sum(1) for g in GROUPS})


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else np.nan


def run_ig_batched(ex, X, CH, idx, t, baseline="even", seeds=None, steps=STEPS):
    """IG de nivel para las partidas idx en el minuto t, en batches. Devuelve dict de arrays."""
    res = {k: [] for k in ("a_x", "a_ch", "fx", "fb")}
    for b in range(0, len(idx), CHUNK):
        ii = idx[b:b + CHUNK]
        x = X[ii, :t + 1]
        embs = ex.embeddings(CH[ii])
        xb = even_baseline(ex.eng, x) if baseline == "even" else torch.zeros_like(x)
        for k, v in zip(res, ex.ig(x, xb, embs, zeros_like_embs(embs), steps=steps, seeds=seeds)):
            res[k].append(v)
    return {k: np.concatenate(v) for k, v in res.items()}


# anotaciones (momentos de test)
def load_notes() -> pd.DataFrame:
    ann = pd.concat([pd.read_csv(ROOT / "annotations" / f"{w}.csv") for w in ("A", "B")], ignore_index=True)
    last = ann.groupby(["anotador", "match_id"])["guardado_utc"].transform("max")
    ann = ann[(ann["guardado_utc"] == last)]
    return ann


def classify_moments(moms, mid: str, notes: pd.DataFrame) -> list[dict]:
    """Cada momento detectado -> ¿coincide con alguna anotación humana? (mismo criterio que B10:
    minuto detectado - 1 a ±2 min de lo anotado). Devuelve la nota humana más cercana si coincide."""
    g = notes[notes["match_id"] == mid]
    marked = g[~g["sin_momentos"].astype(str).str.lower().eq("true")]
    out = []
    for m in moms:
        d = (pd.to_numeric(marked["minuto"]) - (m.minute - 1)).abs()
        hit = marked[d <= 2]
        near = hit.loc[d[d <= 2].idxmin()] if len(hit) else None
        out.append({"tp": len(hit) > 0, "nota": None if near is None else str(near["nota"]),
                    "tipo": None if near is None else str(near["tipo"]),
                    "equipo": None if near is None else str(near["equipo"]),
                    "minuto_anotado": None if near is None else int(near["minuto"])})
    return out


def moment_events(mid: str, start: int, end: int) -> str:
    """Eventos reales del tramo (kills, objetivos, estructuras) con el resumen de la app. Requiere los CSV
    crudos de data/anotacion (los usa B10); si no están, devuelve ''."""
    try:
        from src.features import match_to_features
        from src.predict_matches import load_raw
        tb = load_raw([mid])[mid]
        ev = tb["events"][tb["events"]["type"].isin(EVENT_TYPES_USED)]
        raw = match_to_features(tb["frames"][FRAME_COLS], ev.reindex(columns=EVENT_COLS), tb["participants"])
        ts = np.asarray(raw.attrs["timestamp_ms"])
        e = window_events(tb["events"], ts[start], ts[end])
        txt = []
        for team in ("azul", "rojo"):
            d = e[team]
            items = ([f"{d['kills']} kills"] if d["kills"] else []) + d["objetivos"] + d["estructuras"]
            if items:
                txt.append(f"{team}: " + ", ".join(items))
        return "; ".join(txt) or "sin eventos"
    except (SystemExit, FileNotFoundError, KeyError) as exc:   # sin datos crudos: la explicación sigue igual
        return f"(eventos no disponibles: {exc})"


# main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-global", type=int, default=N_GLOBAL)
    ap.add_argument("--n-sens", type=int, default=N_SENS)
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta de salida (crea resultados/ y figuras/ dentro). Por defecto lab10_xai/, "
                         "o lab10_xai/prueba/ si se cambia --n-global o --n-sens, para no pisar lo del informe")
    args = ap.parse_args()
    global OUT, FIG
    base = args.salida or (HERE if (args.n_global, args.n_sens) == (N_GLOBAL, N_SENS) else HERE / "prueba")
    OUT, FIG = base / "resultados", base / "figuras"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    eng = Engine(MODEL_DIR)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ex = Explainer(eng)
    d = load_sequences(ROOT / "features" / "sequences.npz").subset("test")
    X = torch.tensor(d.X)
    CH = torch.tensor(d.champions.astype(np.int64))
    N = len(d)
    P = ex.curves(X, CH, d.lengths)
    # control: la curva reproducida = Engine (la app) en las primeras partidas
    for i in range(3):
        T = int(d.lengths[i])
        ref = eng.combine(np.stack([eng._probs(m, d.X[i:i + 1, :T], d.champions[i]) for m in eng.members]))[0]
        assert np.abs(ref - P[i, :T]).max() < 1e-4, "la curva diferenciable no coincide con la app"
    print(f"test: {N} partidas · threads: {torch.get_num_threads()}")

    #  1. explicaciones locales
    p15, y = P[:, T_LOCAL], d.y.astype(int)
    conf, correct = np.abs(p15 - 0.5), (p15 > 0.5) == (y == 1)
    clean = ~d.afk
    choices = {
        "A": ("Acierto, alta confianza", clean & correct & (conf >= CONF_ALTA)),
        "B": ("Error, alta confianza", clean & ~correct & (conf >= CONF_ERROR)),
        "C": ("Cerca del umbral", clean & (conf <= CONF_UMBRAL)),
        "D": ("Subgrupo: con AFK", d.afk.copy()),
    }
    local = []
    for key, (label, ok) in choices.items():
        i = int(rng.choice(np.flatnonzero(ok)))
        local.append(dict(key=key, caso=label, i=i, start=None, end=T_LOCAL, kind="nivel"))

    # momentos (E, F): partidas de test anotadas por personas (B10), detector de la app con θ* = 0.15
    notes = load_notes()
    sel = pd.read_csv(ROOT / "annotations" / "seleccion.csv")
    pos = {m: k for k, m in enumerate(d.match_id)}
    ann_ids = [m for m in sel.loc[sel["split"] == "test", "match_id"] if m in pos]
    cands = []
    for mid in ann_ids:
        i = pos[mid]
        p = P[i, :d.lengths[i]]
        moms = detect_moments(p, theta=eng.theta, windows=tuple(eng.det["windows"]),
                              merge_gap=eng.det["merge_gap"], top_k=eng.det["top_k"])
        for m, c in zip(moms, classify_moments(moms, mid, notes)):
            cands.append(dict(i=i, m=m, **c))
    tps = [c for c in cands if c["tp"] and c["nota"] not in (None, "nan")]
    fps = [c for c in cands if not c["tp"]]
    print(f"momentos en las {len(ann_ids)} partidas anotadas de test: {len(cands)} "
          f"({len(tps)} coinciden con una anotación con nota, {len(fps)} sin anotación)")
    for key, label, pool in (("E", "Momento verdadero", tps),     # coincide con una anotación humana
                             ("F", "Momento falso", fps)):         # nadie lo anotó
        c = pool[int(rng.integers(len(pool)))]
        local.append(dict(key=key, caso=label, i=c["i"], start=c["m"].start_minute, end=c["m"].minute,
                          kind="momento", nota=c["nota"], tipo=c["tipo"], equipo=c["equipo"],
                          minuto_anotado=c["minuto_anotado"]))

    loc_rows, loc_detail = [], []
    for inst in local:
        i, t = inst["i"], inst["end"]
        x = X[i:i + 1, :t + 1]
        embs = ex.embeddings(CH[i:i + 1])
        t0 = time.perf_counter()
        if inst["kind"] == "nivel":
            ax, ach, fx, fb = ex.ig(x, even_baseline(eng, x), embs, zeros_like_embs(embs))
        else:
            ax, ach, fx, fb = ex.ig(x, frozen_baseline(x, inst["start"], t), embs, embs)
        ms = 1000 * (time.perf_counter() - t0)
        a = per_name(ax, ach)[0]
        dlogit = float(fx[0] - fb[0])
        dp = float(sigmoid(fx[0]) - sigmoid(fb[0]))
        pts = 100 * dp * a / dlogit if abs(dlogit) > 1e-9 else np.zeros_like(a)
        inst.update(match_id=str(d.match_id[i]), tier=str(d.tier[i]), patch=str(d.patch[i]),
                    platform=str(d.platform[i]), y=int(y[i]), curve=P[i, :d.lengths[i]],
                    p=float(sigmoid(fx[0])), p_base=float(sigmoid(fb[0])), a=a, pts=pts,
                    a_minute=ax[0].sum(1), dlogit=dlogit, err=abs(a.sum() - dlogit), ms=ms)
        if inst["kind"] == "momento":
            inst["eventos"] = moment_events(inst["match_id"], inst["start"], t)
            inst["p_start"] = float(P[i, inst["start"]])
        order = np.argsort(-np.abs(a))
        top = ", ".join(f"{NAMES[j]} {pts[j]:+.1f}" for j in order[:4])
        loc_rows.append({
            "caso": inst["key"], "descripcion": inst["caso"], "match_id": inst["match_id"],
            "tier": inst["tier"], "parche": inst["patch"], "gana": "azul" if inst["y"] else "rojo",
            "minuto": t, "desde_min": inst["start"], "linea_base": "pareja" if inst["kind"] == "nivel" else "congelada",
            "p_azul": round(inst["p"], 3), "p_linea_base": round(inst["p_base"], 3),
            "delta_pts": round(100 * dp, 1), "delta_logit": round(dlogit, 4),
            "error_completitud": float(f"{inst['err']:.2e}"), "ms": round(ms, 1), "top4_pts": top,
            "nota_humana": inst.get("nota") or "", "eventos": inst.get("eventos") or "",
            "minuto_anotado": inst.get("minuto_anotado")})
        loc_detail += [{"caso": inst["key"], "feature": NAMES[j], "grupo": NAME_GROUP[j],
                        "atribucion_logit": round(float(a[j]), 5), "pts": round(float(pts[j]), 2)}
                       for j in range(len(NAMES))]
    loc_df = pd.DataFrame(loc_rows)
    loc_df.to_csv(OUT / "locales.csv", index=False)
    pd.DataFrame(loc_detail).to_csv(OUT / "locales_detalle.csv", index=False)
    print(loc_df[["caso", "match_id", "minuto", "p_azul", "gana", "delta_pts", "top4_pts"]].to_string(index=False))

    # 2. explicación global (nivel)
    sample = np.sort(rng.choice(N, size=min(args.n_global, N), replace=False))
    glob, lag_rows, times = [], [], []
    for t in MINUTES_GLOBAL:
        idx = sample[d.lengths[sample] > t]
        t0 = time.perf_counter()
        r = run_ig_batched(ex, X, CH, idx, t)
        times.append({"tarea": f"IG nivel, minuto {t} (batch de {CHUNK})", "n": len(idx),
                      "ms_por_instancia": 1000 * (time.perf_counter() - t0) / len(idx)})
        a = per_name(r["a_x"], r["a_ch"])
        err = np.abs(a.sum(1) - (r["fx"] - r["fb"]))
        # cuánto pesa cada minuto de la historia: Σ_features |a| en el minuto m, por "retraso" t − m
        per_min = np.abs(r["a_x"]).sum(2)                                  # [B, t+1]
        lag = t - np.arange(t + 1)
        bins = {"minuto t": lag == 0, "t−1": lag == 1, "t−2 a t−4": (lag >= 2) & (lag <= 4),
                "t−5 a t−9": (lag >= 5) & (lag <= 9), "≥ t−10": lag >= 10}
        tot = per_min.sum()
        lag_rows += [{"minuto": t, "retraso": k, "pct": 100 * per_min[:, m].sum() / tot} for k, m in bins.items()]
        for b, i in enumerate(idx):
            glob.append({"i": i, "t": t, "fx": r["fx"][b], "fb": r["fb"][b], "err": err[b],
                         "a": a[b], "v": d.X[i, t]})
        print(f"  global minuto {t}: {len(idx)} partidas, {times[-1]['ms_por_instancia']:.1f} ms/instancia, "
              f"error de completitud máx {err.max():.1e}")
    A = np.stack([g["a"] for g in glob])
    gi = np.array([g["i"] for g in glob])
    gt = np.array([g["t"] for g in glob])
    V = np.stack([g["v"] for g in glob])

    def shares(mask) -> pd.Series:
        s = group_sum(A[mask], absolute=True).sum()
        return 100 * s / s.sum()

    grp = pd.DataFrame({f"min {t}": shares(gt == t) for t in MINUTES_GLOBAL})
    grp["todos"] = shares(np.ones(len(gt), bool))
    grp = grp.reindex(GROUPS + ["tendencia"])          # tendencia: M6 no la usa (solo modelos snapshot)
    b12 = pd.read_csv(ROOT / "results" / "b12_shap_grupos.csv")
    b12 = b12[b12["tramo"] == "global"].assign(grupo=lambda f: f["grupo"].map(GROUP_LABEL)).set_index("grupo")["pct"]
    grp["M2 TreeSHAP (B12)"] = b12.reindex(grp.index)
    grp.round(2).to_csv(OUT / "global_grupos.csv", index_label="grupo")

    imp = np.abs(A).mean(0)
    feat = pd.DataFrame({"feature": NAMES, "grupo": NAME_GROUP, "media_abs_logit": imp,
                         "pct": 100 * imp / imp.sum()})
    # dirección: correlación entre el valor de la feature en el minuto t y su atribución (+ = más alto = azul)
    feat["direccion"] = [rank_corr(V[:, j], A[:, j]) if j < len(FEATURE_NAMES) else np.nan
                         for j in range(len(NAMES))]
    # campeones: ¿qué tan seguido el draft empuja hacia el equipo que terminó ganando?
    ywin = np.where(d.y[gi] == 1, 1, -1)
    draft_net = A[:, -2] + A[:, -1]
    feat = feat.sort_values("pct", ascending=False)
    feat.round(4).to_csv(OUT / "global_features.csv", index=False)
    lags = pd.DataFrame(lag_rows).pivot(index="minuto", columns="retraso", values="pct")[list(bins)]
    lags.round(2).to_csv(OUT / "global_retrasos.csv")

    # subgrupos (minuto 15): ¿el modelo usa las mismas señales en el parche no visto, por tier y con AFK?
    m15 = gt == 15
    sub = {"todas": m15, "parche 16.21": m15 & (d.patch[gi] == "16.21"),
           "parche 16.22 (no visto)": m15 & (d.patch[gi] == "16.22"),
           "con AFK": m15 & d.afk[gi]}
    for tier in ("SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"):
        sub[tier.lower()] = m15 & (d.tier[gi] == tier)
    pw = np.array([sigmoid(g["fx"]) for g in glob])
    hit = (pw > 0.5) == (d.y[gi] == 1)
    sub[f"aciertos |P−0.5| ≥ {CONF_ERROR}"] = m15 & hit & (np.abs(pw - 0.5) >= CONF_ERROR)
    sub[f"errores |P−0.5| ≥ {CONF_ERROR}"] = m15 & ~hit & (np.abs(pw - 0.5) >= CONF_ERROR)
    sg = pd.DataFrame({k: shares(v) for k, v in sub.items() if v.sum() >= 10}).T
    sg.insert(0, "n", [int(sub[k].sum()) for k in sg.index])
    sg["acierto_%"] = [100 * (((pw[sub[k]] > 0.5) == (d.y[gi][sub[k]] == 1)).mean()) for k in sg.index]
    sg.round(2).to_csv(OUT / "global_subgrupos.csv", index_label="subgrupo")

    # 3. momentos: IG vs oclusión de la app
    mrows = []
    t0 = time.perf_counter()
    n_m = 0
    for i in sample:
        p = P[i, :d.lengths[i]]
        for m in detect_moments(p, theta=eng.theta, windows=tuple(eng.det["windows"]),
                                merge_gap=eng.det["merge_gap"], top_k=eng.det["top_k"]):
            s, e = m.start_minute, m.minute
            x = X[i:i + 1, :e + 1]
            embs = ex.embeddings(CH[i:i + 1])
            tt = time.perf_counter()
            ax, _, fx, fb = ex.ig(x, frozen_baseline(x, s, e), embs, embs)
            ms_ig = 1000 * (time.perf_counter() - tt)
            a = per_name(ax, np.zeros((1, 10)))[0]
            sign = np.sign(p[e] - p[s])
            ig_g = {g: sign * a[[j for j in range(len(NAMES)) if APP_GROUP.get(NAME_GROUP[j]) == g]].sum()
                    for g in APP_GROUPS}
            # oclusión de la app (Engine.explain) en el espacio estandarizado: congelar un grupo es idéntico
            # porque el scaler es por columna
            tt = time.perf_counter()
            base = d.X[i, :e + 1]
            variants = [base]
            for groups in EXPLAIN_GROUPS.values():
                cols = [j for j, g in enumerate(FEATURE_GROUPS) if g in groups]
                v = base.copy()
                v[s + 1:e + 1, cols] = base[s, cols]
                variants.append(v)
            pv = eng._main(np.stack(variants), d.champions[i])[:, e]
            ms_occ = 1000 * (time.perf_counter() - tt)
            jump = abs(pv[0] - p[s])
            red = {g: jump - abs(q - p[s]) for g, q in zip(APP_GROUPS, pv[1:])}
            n_m += 1
            mrows.append({"match_id": d.match_id[i], "desde_min": s, "minuto": e, "salto_pts": 100 * (p[e] - p[s]),
                          "salto_cf_pts": 100 * (sigmoid(fx[0]) - sigmoid(fb[0])), "delta_logit": fx[0] - fb[0],
                          "error_completitud": abs(a.sum() - (fx[0] - fb[0])),
                          **{f"ig_{g}": ig_g[g] for g in APP_GROUPS}, **{f"oc_{g}": red[g] for g in APP_GROUPS},
                          "top_ig": max(ig_g, key=ig_g.get), "top_oc": max(red, key=red.get),
                          "top_feature": NAMES[int(np.argmax(sign * a))],
                          "pct_top_feature": 100 * float((sign * a).max() / (fx[0] - fb[0]) * sign)
                          if abs(fx[0] - fb[0]) > 1e-9 else np.nan,
                          # ¿algún grupo empuja en contra del salto con ≥ 10 % de su tamaño? (la app lo recorta a 0)
                          "ig_grupo_en_contra": min(ig_g.values()) <= -0.10 * abs(fx[0] - fb[0]),
                          "ig_feature_en_contra": (sign * a).min() <= -0.10 * abs(fx[0] - fb[0]),
                          "suma_oc_sobre_salto": sum(red.values()) / jump if jump > 0 else np.nan,
                          "ms_ig": ms_ig, "ms_oclusion": ms_occ})
    mom = pd.DataFrame(mrows)
    mom.round(5).to_csv(OUT / "momentos_ig_vs_oclusion.csv", index=False)
    times.append({"tarea": "IG momento (1 a la vez)", "n": n_m, "ms_por_instancia": mom["ms_ig"].median()})
    times.append({"tarea": "Oclusión de la app por momento", "n": n_m, "ms_por_instancia": mom["ms_oclusion"].median()})
    print(f"  momentos: {n_m} en {len(sample)} partidas ({time.perf_counter() - t0:.0f} s); "
          f"top-grupo IG = oclusión en {100 * (mom.top_ig == mom.top_oc).mean():.1f} %")

    def pos_share(df, pref):
        v = df[[f"{pref}_{g}" for g in APP_GROUPS]].clip(lower=0).to_numpy()
        tot = v.sum(1, keepdims=True)
        v = np.divide(v, tot, out=np.zeros_like(v), where=tot > 0)
        return pd.Series(100 * v.mean(0), index=APP_GROUPS)

    mom_sh = pd.DataFrame({"IG": pos_share(mom, "ig"), "Oclusión (app)": pos_share(mom, "oc"),
                           "IG top-1 %": 100 * mom["top_ig"].value_counts(normalize=True).reindex(APP_GROUPS).fillna(0),
                           "Oclusión top-1 %": 100 * mom["top_oc"].value_counts(normalize=True).reindex(APP_GROUPS).fillna(0)})
    mom_sh.round(2).to_csv(OUT / "momentos_grupos.csv", index_label="grupo")
    # ¿qué feature domina los momentos tempranos (zona de baja certeza, minuto < 5) vs el resto?
    early = mom["minuto"] < 5
    top_feat = pd.DataFrame({
        "minuto < 5": 100 * mom.loc[early, "top_feature"].value_counts(normalize=True),
        "minuto ≥ 5": 100 * mom.loc[~early, "top_feature"].value_counts(normalize=True)}).fillna(0)
    top_feat = top_feat.sort_values("minuto ≥ 5", ascending=False).head(8)
    top_feat.loc["n momentos"] = [early.sum(), (~early).sum()]
    top_feat.round(1).to_csv(OUT / "momentos_top_feature.csv", index_label="feature principal")
    # en las partidas anotadas: ¿los momentos tempranos son más a menudo falsos?
    ann_m = pd.DataFrame([{"temprano": c["m"].minute < 5, "verdadero": c["tp"]} for c in cands])
    ann_tab = ann_m.groupby("temprano")["verdadero"].agg(n="size", pct_verdaderos=lambda s: 100 * s.mean())
    ann_tab.index = ann_tab.index.map({True: "minuto < 5", False: "minuto ≥ 5"})
    ann_tab.round(1).to_csv(OUT / "momentos_anotados_tempranos.csv", index_label="momento")

    # 4. sensibilidad (semillas, línea base, pasos)
    sens_idx = sample[: args.n_sens]
    ref = run_ig_batched(ex, X, CH, sens_idx, T_LOCAL)
    ref_a = per_name(ref["a_x"], ref["a_ch"])
    ref_g = group_sum(ref_a, absolute=False).to_numpy()
    sens = []

    def compare(name, r):
        a = per_name(r["a_x"], r["a_ch"])
        g = group_sum(a, absolute=False).to_numpy()
        rc = [rank_corr(g[k], ref_g[k]) for k in range(len(g))]
        top = (np.abs(g).argmax(1) == np.abs(ref_g).argmax(1)).mean()
        sh = group_sum(a, absolute=True).sum()
        sh = 100 * sh / sh.sum()
        sens.append({"variante": name, "corr_rangos_grupos_mediana": np.nanmedian(rc), "mismo_grupo_top_%": 100 * top,
                     "corr_rangos_features_mediana": np.nanmedian([rank_corr(a[k], ref_a[k]) for k in range(len(a))]),
                     "error_completitud_max": np.abs(a.sum(1) - (r["fx"] - r["fb"])).max(),
                     **{f"pct_{k}": v for k, v in sh.items()}})

    compare("referencia: 3 semillas, partida pareja, 32 pasos", ref)
    for k, m in enumerate(eng.members):
        compare(f"solo semilla {m.seed}", run_ig_batched(ex, X, CH, sens_idx, T_LOCAL, seeds=[k]))
    compare("línea base 'todo en cero' (media de train, minuto incluido)",
            run_ig_batched(ex, X, CH, sens_idx, T_LOCAL, baseline="zeros"))
    for st in (16, 64):
        compare(f"{st} pasos", run_ig_batched(ex, X, CH, sens_idx, T_LOCAL, steps=st))
    sens = pd.DataFrame(sens)
    sens.round(4).to_csv(OUT / "sensibilidad.csv", index=False)
    pd.DataFrame(times).round(2).to_csv(OUT / "tiempos.csv", index=False)

    # figuras y resumen
    fig_locales(local)
    fig_global(grp, feat, lags)
    fig_momentos(mom_sh, mom)
    write_summary(loc_df, grp, feat, lags, sg, mom, mom_sh, sens, pd.DataFrame(times), glob, ywin, draft_net,
                  len(sample), time.perf_counter() - t_all, top_feat, ann_tab)
    print(f"listo en {time.perf_counter() - t_all:.0f} s -> {OUT}, {FIG}")


# figuras
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_locales(local: list[dict]) -> None:
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1.15, 1, 1.15], hspace=0.6, wspace=0.95)
    for k, inst in enumerate(local):
        r, c = divmod(k, 2)
        axc, axb = fig.add_subplot(gs[r, 2 * c]), fig.add_subplot(gs[r, 2 * c + 1])
        # curva
        cur = inst["curve"]
        axc.plot(np.arange(len(cur)), cur, color=INK, lw=1.6)
        axc.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        if inst["kind"] == "nivel":
            axc.axvline(inst["end"], color=MUTED, lw=0.8)
        else:
            axc.axvspan(inst["start"], inst["end"], color="#d9d8d3", lw=0)
        axc.plot([inst["end"]], [inst["p"]], "o", ms=6, color=INK, mec="white", mew=1.5)
        axc.set_ylim(0, 1)
        axc.set_xlabel(f"minuto\n{inst['match_id']} · {inst['tier'].lower()} · ganó {'azul' if inst['y'] else 'rojo'}",
                       fontsize=8, color=MUTED)
        axc.set_ylabel("P(gana azul)", fontsize=8, color=MUTED)
        _style(axc)
        axc.grid(axis="y", color=GRID, lw=0.6)
        axc.set_title(f"{inst['key']} · {inst['caso']}", fontsize=10.5, loc="left", color=INK, fontweight="bold")
        # barras: 8 mayores aportes en puntos de probabilidad
        pts = inst["pts"]
        order = np.argsort(-np.abs(pts))[:8][::-1]
        vals = pts[order]
        axb.barh(np.arange(len(order)), vals, color=[BLUE if v > 0 else RED for v in vals], height=0.7,
                 edgecolor="white", linewidth=1)
        axb.set_yticks(np.arange(len(order)))
        axb.set_yticklabels([f"{NAMES[j]}  ·  {NAME_GROUP[j]}" for j in order], fontsize=7.5, color=INK)
        axb.axvline(0, color=MUTED, lw=0.8)
        lim = max(1.0, np.abs(vals).max()) * 1.5
        axb.set_xlim(-lim, lim)
        for yy, v in enumerate(vals):
            axb.text(v + (0.03 * lim if v >= 0 else -0.03 * lim), yy, f"{v:+.1f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=7, color=INK)
        _style(axb)
        if inst["kind"] == "nivel":
            ttl = f"min {inst['end']}: P {inst['p_base']:.2f} (pareja) → {inst['p']:.2f}"
        else:
            ttl = (f"min {inst['start']}–{inst['end']}: {inst['p_start']:.2f} → {inst['p']:.2f} "
                   f"(congelada {inst['p_base']:.2f})")
        axb.set_title(ttl, fontsize=8.5, loc="left", color=MUTED)
        axb.set_xlabel("aporte (pts de prob.)   ← rojo · azul →", fontsize=8, color=MUTED)
    fig.suptitle("Integrated Gradients · explicaciones locales (M6 + campeones, test)", fontsize=13,
                 x=0.06, ha="left", color=INK, fontweight="bold")
    fig.text(0.06, 0.935, "Barras: 8 mayores aportes. Azul = empuja hacia azul, rojo = hacia rojo. "
             "A-D: línea base 'partida pareja' en el minuto 15. E-F: línea base 'partida congelada' (zona gris = "
             "momento); E coincide con una anotación humana, F no la anotó nadie.", fontsize=9, color=MUTED)
    fig.savefig(FIG / "fig1_locales.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_global(grp: pd.DataFrame, feat: pd.DataFrame, lags: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.15, 1.25, 1.1], "wspace": 0.55})
    # (a) % de |atribución| por grupo y minuto, vs TreeSHAP de LightGBM
    ax = axs[0]
    cols = [f"min {t}" for t in MINUTES_GLOBAL] + ["M2 TreeSHAP (B12)"]
    rows = GROUPS + ["tendencia"]
    M = grp.reindex(index=rows, columns=cols).to_numpy(dtype=float)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(np.nan_to_num(M), cmap=cmap, vmin=0, vmax=60, aspect="auto")
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, ("no usa" if rows[r] == "tendencia" else "—") if np.isnan(v) else f"{v:.0f}",
                ha="center", va="center", fontsize=8 if np.isnan(v) else 8.5,
                color="white" if (not np.isnan(v) and v > 30) else (MUTED if np.isnan(v) else INK))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["min 10", "min 15", "min 20", "min 25", "M2\nTreeSHAP"], fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5, color=INK)
    ax.axvline(len(MINUTES_GLOBAL) - 0.5, color="white", lw=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("a · % del |aporte| por grupo", fontsize=10.5, loc="left", color=INK, fontweight="bold")
    # (b) top-15 features
    ax = axs[1]
    top = feat.head(15).iloc[::-1]
    ax.barh(range(len(top)), top["pct"], color=GRAYS[3], height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{n}  ·  {g}" for n, g in zip(top["feature"], top["grupo"])], fontsize=7.5, color=INK)
    for yy, (v, dd) in enumerate(zip(top["pct"], top["direccion"])):
        ax.text(v + 0.4, yy, f"{v:.1f} %" + ("" if np.isnan(dd) else f"   dir {dd:+.2f}"),
                va="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, top["pct"].max() * 1.55)
    ax.set_xlabel("% del |aporte| (minutos 10-25)", fontsize=8, color=MUTED)
    _style(ax)
    ax.set_title("b · 15 features más usadas", fontsize=10.5, loc="left", color=INK, fontweight="bold")
    # (c) de qué minutos sale la atribución
    ax = axs[2]
    left = np.zeros(len(lags))
    for k, col in enumerate(lags.columns):
        v = lags[col].to_numpy()
        ax.barh(range(len(lags)), v, left=left, color=GRAYS[::-1][k], height=0.6, edgecolor="white", linewidth=2,
                label=col)
        for yy, (l0, vv) in enumerate(zip(left, v)):
            if vv >= 7:
                ax.text(l0 + vv / 2, yy, f"{vv:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if k < 3 else INK)
        left += v
    ax.set_yticks(range(len(lags)))
    ax.set_yticklabels([f"min {t}" for t in lags.index], fontsize=8.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% del |aporte| según el minuto de la historia", fontsize=8, color=MUTED)
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    _style(ax)
    ax.invert_yaxis()
    ax.set_title("c · ¿minuto actual o historia?", fontsize=10.5, loc="left", color=INK, fontweight="bold")
    fig.suptitle("Integrated Gradients · explicación global del nivel de P (línea base 'partida pareja')",
                 fontsize=13, x=0.06, ha="left", y=1.02, color=INK, fontweight="bold")
    fig.savefig(FIG / "fig2_global.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_momentos(mom_sh: pd.DataFrame, mom: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.35})
    ax = axs[0]
    yy = np.arange(len(APP_GROUPS))
    for k, (col, color) in enumerate((("IG", GRAYS[4]), ("Oclusión (app)", GRAYS[1]))):
        v = mom_sh[col].to_numpy()
        ax.barh(yy + (0.2 - 0.4 * k), v, height=0.38, color=color, label=col, edgecolor="white", linewidth=1)
        for y0, vv in zip(yy, v):
            ax.text(vv + 1, y0 + (0.2 - 0.4 * k), f"{vv:.0f} %", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels(APP_GROUPS, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("aporte medio al salto (% de los aportes a favor)", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    agree = 100 * (mom.top_ig == mom.top_oc).mean()
    ax.set_title(f"a · grupos que explican los {len(mom)} momentos\n(mismo grupo principal en {agree:.1f} %)",
                 fontsize=10, loc="left", color=INK, fontweight="bold")
    ax = axs[1]
    r = mom["suma_oc_sobre_salto"].clip(-0.5, 2.5)
    ax.hist(r, bins=30, color=GRAYS[2], edgecolor="white", linewidth=1)
    ax.axvline(1, color=INK, lw=1.2, ls="--")
    ax.text(1.03, ax.get_ylim()[1] * 0.92, "IG: siempre 1\n(completitud)", fontsize=8, color=INK)
    ax.set_xlabel("Σ reducciones de la oclusión ÷ salto real", fontsize=8, color=MUTED)
    ax.set_ylabel("momentos", fontsize=8, color=MUTED)
    _style(ax)
    q = mom["suma_oc_sobre_salto"]
    ax.set_title(f"b · ¿la oclusión suma el salto? mediana {q.median():.2f},\n"
                 f"80 % entre {q.quantile(.1):.2f} y {q.quantile(.9):.2f}", fontsize=10, loc="left", color=INK,
                 fontweight="bold")
    fig.savefig(FIG / "fig3_momentos.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# resumen
def _md(df: pd.DataFrame, fmt: str = "{:.1f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "—" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def write_summary(loc_df, grp, feat, lags, sg, mom, mom_sh, sens, times, glob, ywin, draft_net, n_sample, secs,
                  top_feat, ann_tab):
    errs = np.array([g["err"] for g in glob])
    dl = np.abs([g["fx"] - g["fb"] for g in glob])
    draft_ok = (np.sign(draft_net) == ywin)[np.abs(draft_net) > 1e-6].mean()
    lines = [
        "# Lab 10 · Parte 3 — resultados generados por `lab10_xai/explicar.py`",
        "",
        f"*Semilla {SEED} · IG {STEPS} pasos · {n_sample} partidas de test para lo global · {secs:.0f} s en total. "
        "No editar a mano: se regenera con el script.*",
        "",
        "## Explicaciones locales",
        "",
        _md(loc_df[["caso", "descripcion", "match_id", "minuto", "desde_min", "linea_base", "p_azul", "p_linea_base",
                    "delta_pts", "gana", "error_completitud", "top4_pts"]].astype({"desde_min": "object"}), "{:.3f}"),
        "",
        "Notas humanas / eventos de los momentos:",
        "",
    ]
    for r in loc_df.itertuples():
        if isinstance(r.nota_humana, str) and r.nota_humana:
            lines.append(f"- **{r.caso}** (anotado en el min {r.minuto_anotado}) nota: \"{r.nota_humana}\" · "
                         f"eventos: {r.eventos}")
        elif isinstance(r.eventos, str) and r.eventos:
            lines.append(f"- **{r.caso}** eventos: {r.eventos}")
    lines += [
        "",
        "![locales](../figuras/fig1_locales.png)",
        "",
        "## Global (nivel de P)",
        "",
        f"Completitud: error máx {errs.max():.1e} logit (mediana |Δ logit| explicado {np.median(dl):.2f}).",
        f"El draft empuja hacia el equipo que terminó ganando en {100 * draft_ok:.0f} % de las instancias.",
        "",
        _md(grp.reset_index(names="grupo")),
        "",
        _md(feat.head(15)[["feature", "grupo", "pct", "direccion"]], "{:.2f}"),
        "",
        "% del |aporte| según el minuto de la historia:",
        "",
        _md(lags.reset_index()),
        "",
        "Subgrupos (minuto 15):",
        "",
        _md(sg.reset_index(names="subgrupo")),
        "",
        "![global](../figuras/fig2_global.png)",
        "",
        "## Momentos: IG vs oclusión de la app",
        "",
        f"{len(mom)} momentos (θ* = 0.15). Mismo grupo principal: {100 * (mom.top_ig == mom.top_oc).mean():.1f} %. "
        f"Σ reducciones de la oclusión / salto: mediana {mom['suma_oc_sobre_salto'].median():.2f} "
        f"(p10 {mom['suma_oc_sobre_salto'].quantile(.1):.2f}, p90 {mom['suma_oc_sobre_salto'].quantile(.9):.2f}). "
        f"Salto contrafactual / salto de la curva: mediana "
        f"{(mom['salto_cf_pts'] / mom['salto_pts']).median():.2f}. "
        f"Error de completitud máx {mom['error_completitud'].max():.1e}. "
        f"Momentos con un grupo que empuja en contra del salto (≥ 10 % del Δ logit; la app lo recorta a 0): "
        f"{100 * mom['ig_grupo_en_contra'].mean():.1f} %; con alguna feature en contra: "
        f"{100 * mom['ig_feature_en_contra'].mean():.1f} %. Momentos antes del minuto 5: {100 * (mom['minuto'] < 5).mean():.1f} %.",
        "",
        _md(mom_sh.reset_index(names="grupo")),
        "",
        "Feature que más empuja el salto (% de momentos):",
        "",
        _md(top_feat.reset_index(names="feature principal")),
        "",
        "Momentos de las 28 partidas anotadas de test, según si caen en la zona de baja certeza:",
        "",
        _md(ann_tab.reset_index(names="momento")),
        "",
        "![momentos](../figuras/fig3_momentos.png)",
        "",
        "## Sensibilidad (minuto 15)",
        "",
        _md(sens, "{:.3g}"),
        "",
        "## Tiempos (CPU)",
        "",
        _md(times),
        "",
    ]
    (OUT / "resumen_parte3.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
