"""
B11 - Motor de inferencia de la app: partida cruda -> curva P(gana azul), momentos y explicaciones.

Mismo pipeline que el entrenamiento (no hay código de features aparte):
  JSON de Riot -> riot_data.parse.parse_match -> src.features.match_to_features
  -> scaler.json (de train) -> GRU (cada semilla) -> temperatura -> P(t)
  -> src.moments.detect_moments con el θ* que eligió B10 en val.

- Curva y momentos: la PRIMERA semilla del manifest (s42), la misma con la que B10 eligió θ* y
  calculó el valor, para que la app muestre lo mismo que se reporta.
- Banda de incertidumbre: mínimo y máximo entre las semillas (guía, sección 8).
- Explicación por momento (guía, sección 8): ablación por grupos. Se congela un grupo de
  features en su valor de ANTES de la jugada y se recalcula P(t); el grupo que más achica el
  salto es la explicación. "rol" se junta con "economía" (las dos son oro).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from riot_data.parse import parse_match  # noqa: E402
from src.features import (EVENT_COLS, EVENT_TYPES_USED, FEATURE_GROUPS, FEATURE_NAMES,  # noqa: E402
                          FRAME_COLS, Scaler, dedup_events, match_to_features)
from src.models import build_model  # noqa: E402
from src.moments import detect_moments  # noqa: E402

DEFAULT_MODEL_DIR = ROOT / "models" / "m4_gru"
LOW_CERTAINTY_MINUTES = 5
MAX_CANDIDATES = 3
MONITOR_MINUTE = 15   # minuto cuyas features crudas se registran para vigilar cambios de distribución (PSI)
EXPLAIN_GROUPS = {          # grupo de la explicación -> grupos del contrato de features
    "economía": ["economia", "rol"],
    "combate": ["combate"],
    "objetivos": ["objetivos"],
    "visión": ["vision"],
}


class AnalysisError(ValueError):
    """La partida no se puede analizar (remake, muy corta, sin frames...). status = código HTTP."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass
class Member:
    """Un modelo del ensamble. kind: "torch" (GRU/TCN/MLP), "logit" (M0/M1) o "lgbm" (M2).
    Los baselines son snapshot (una fila por minuto): ven el minuto actual, no la historia."""
    model: object
    cols: list[int]
    temperature: float
    seed: int
    kind: str = "torch"


# ------------------------------------------------------------------ datos
def tables_from_riot(match: dict, timeline: dict, platform: str, tier: str = "") -> dict[str, pd.DataFrame]:
    rows = parse_match(match, timeline, platform, tier)
    tables = {k: pd.DataFrame(v) for k, v in rows.items()}
    tables["events"] = dedup_events(tables["events"])   # igual que merge.py con el entrenamiento
    return tables


def platform_of(match_id: str) -> str:
    return match_id.split("_", 1)[0].lower()


# ------------------------------------------------------------------ motor
def load_member(model_dir: Path, m: dict) -> Member:
    """Un miembro del manifest. Archivos (los escribe src/export_model.py):
    torch: seed_<s>.pt · logit: <f>.json con coef/intercept · lgbm: model.txt (+ feature_names en el manifest)."""
    kind = m.get("kind", "torch")
    if kind == "logit":
        d = json.loads((model_dir / m["file"]).read_text(encoding="utf-8"))
        cols = [FEATURE_NAMES.index(n) for n in d["feature_names"]]
        params = {"coef": np.asarray(d["coef"], dtype=np.float32), "intercept": float(d["intercept"])}
        return Member(params, cols, float(d["temperature"]), int(m.get("seed", 0)), "logit")
    if kind == "lgbm":
        import lightgbm as lgb   # solo si se usa M2 (no lo necesitan los demás modelos)
        booster = lgb.Booster(model_file=str(model_dir / m["file"]))
        cols = [FEATURE_NAMES.index(n) for n in booster.feature_name()]
        return Member(booster, cols, float(m["temperature"]), int(m.get("seed", 0)), "lgbm")
    ck = torch.load(model_dir / m["file"], map_location="cpu", weights_only=False)
    cols = [FEATURE_NAMES.index(n) for n in ck["feature_names"]]
    model = build_model(ck["model_cfg"], len(cols))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return Member(model, cols, float(ck["temperature"]), int(m["seed"]), "torch")


class Engine:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        if not (model_dir / "manifest.json").exists():
            raise FileNotFoundError(
                f"No existe {model_dir / 'manifest.json'}. Exporten el modelo con "
                "`python src/export_model.py` (los pesos no van a git).")
        self.manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.scaler = Scaler.load(model_dir / "scaler.json")
        if self.scaler.feature_names != list(FEATURE_NAMES):
            raise ValueError("scaler.json no coincide con FEATURE_NAMES de src/features.py")
        self.members: list[Member] = []
        for m in self.manifest["members"]:
            self.members.append(load_member(model_dir, m))
        self.det = self.manifest["detector"]
        self.theta = float(self.det["theta"])
        # umbral de los "posibles momentos" (menos seguros): por defecto la mitad de θ*
        self.theta_cand = float(self.det.get("theta_candidato", round(self.theta / 2, 3)))
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    @property
    def name(self) -> str:
        return self.manifest.get("name", "modelo")

    # ---- predicción
    @torch.no_grad()
    def _probs(self, member: Member, X: np.ndarray) -> np.ndarray:
        """X normalizado [B, T, F_todas] -> P(gana azul) [B, T]."""
        x = X[..., member.cols].astype(np.float32)
        if member.kind == "logit":
            logits = x @ member.model["coef"] + member.model["intercept"]
        elif member.kind == "lgbm":
            b, t, f = x.shape
            logits = member.model.predict(x.reshape(-1, f), raw_score=True).reshape(b, t)
        else:
            xt = torch.as_tensor(x)
            mask = torch.ones(xt.shape[:2], dtype=torch.bool)
            logits = member.model(xt, mask).float().numpy()
        return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64) / member.temperature))

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        """Features crudas [T, F] -> P(gana azul) por semilla [S, T]."""
        X = self.scaler.transform(raw.to_numpy(dtype=np.float32))[None]
        return np.stack([self._probs(m, X)[0] for m in self.members])

    # ---- explicación
    def explain(self, raw: pd.DataFrame, start: int, end: int, p_before: float) -> dict:
        """Ablación por grupos sobre el modelo principal: congelar cada grupo en el minuto `start`."""
        base = raw.to_numpy(dtype=np.float32)
        variants = []
        for groups in EXPLAIN_GROUPS.values():
            cols = [j for j, g in enumerate(FEATURE_GROUPS) if g in groups]
            v = base.copy()
            v[start + 1:end + 1, cols] = base[start, cols]
            variants.append(v)
        X = self.scaler.transform(np.stack([base] + variants))
        p = self._probs(self.members[0], X)[:, end]
        jump = abs(p[0] - p_before)
        reduction = {g: float(jump - abs(pv - p_before)) for g, pv in zip(EXPLAIN_GROUPS, p[1:])}
        pos = {g: max(r, 0.0) for g, r in reduction.items()}
        total = sum(pos.values())
        share = {g: (r / total if total > 0 else 0.0) for g, r in pos.items()}
        top = max(reduction, key=reduction.get)
        return {"grupo": top, "aporte": {g: round(s, 3) for g, s in share.items()},
                "reduccion_salto": {g: round(r, 4) for g, r in reduction.items()}}

    # ---- análisis completo
    def analyze(self, tables: dict[str, pd.DataFrame], timings: dict | None = None) -> dict:
        t0 = time.perf_counter()
        timings = dict(timings or {})
        match = tables["matches"].iloc[0]
        frames, events, parts = tables["frames"], tables["events"], tables["participants"]
        if frames.empty:
            raise AnalysisError("La partida no tiene frames de timeline")
        dur = float(match["duration_s"])
        if dur < self.manifest.get("min_duration_s", 900):
            raise AnalysisError(f"La partida duró {dur / 60:.1f} min: los remakes y las partidas de "
                                "menos de 15 min no se analizan (el modelo no se entrenó con ellas)")
        warnings = []
        if int(match.get("queue_id", 420)) != self.manifest.get("queue_id", 420):
            warnings.append("No es ranked Solo/Duo: el modelo se entrenó con ranked y puede ser menos preciso.")
        patch = str(match.get("patch", ""))
        if patch and self.manifest.get("train_patches") and patch not in self.manifest["train_patches"]:
            warnings.append(f"Parche {patch} distinto a los de entrenamiento "
                            f"({', '.join(self.manifest['train_patches'])}): posible degradación "
                            "(en test, el parche no visto 16.22 no empeoró).")

        ev = events[events["type"].isin(EVENT_TYPES_USED)] if len(events) else events
        raw = match_to_features(frames[FRAME_COLS], ev.reindex(columns=EVENT_COLS), parts)
        ts = np.asarray(raw.attrs["timestamp_ms"])
        t1 = time.perf_counter()
        probs = self.predict(raw)                         # [S, T]
        main = probs[0]
        lo, hi = probs.min(0), probs.max(0)
        moms = detect_moments(main, theta=self.theta, windows=tuple(self.det.get("windows", (1, 2))),
                              merge_gap=self.det.get("merge_gap", 2), top_k=self.det.get("top_k", 5))
        t2 = time.perf_counter()

        # candidatos: saltos entre θ_cand y θ, lejos de los momentos principales (se muestran aparte)
        gap = self.det.get("merge_gap", 2)
        cands = [c for c in detect_moments(main, theta=self.theta_cand, windows=tuple(self.det.get("windows", (1, 2))),
                                           merge_gap=gap, top_k=None)
                 if c.score < self.theta and c.minute >= LOW_CERTAINTY_MINUTES   # no en la zona de baja certeza
                 and all(abs(c.minute - m.minute) >= gap for m in moms)]
        cands = sorted(sorted(cands, key=lambda c: -c.score)[:MAX_CANDIDATES], key=lambda c: c.minute)

        def describe(m, candidate: bool) -> dict:
            # el clip va de start_minute a minute: la explicación mide el salto en ese mismo tramo
            exp = self.explain(raw, m.start_minute, m.minute, float(main[m.start_minute]))
            evs = window_events(events, ts[m.start_minute], ts[m.minute])
            seed_deltas = probs[:, m.minute] - probs[:, m.start_minute]
            return {
                "minuto": int(m.minute), "desde_min": int(m.start_minute),
                "favorece": m.team, "delta_pts": round(100 * m.delta, 1),
                "p_antes": round(m.p_before, 3), "p_despues": round(m.p_after, 3),
                "confianza": "baja" if candidate else ("alta" if m.score >= 2 * self.theta else "media"),
                "semillas_de_acuerdo": int((np.abs(seed_deltas) >= (self.theta_cand if candidate else self.theta)).sum()),
                "candidato": candidate, "explicacion": exp, "eventos": evs,
                "texto": moment_text(m, exp["grupo"], evs),
            }

        moments = [describe(m, False) for m in sorted(moms, key=lambda x: x.minute)]
        candidates = [describe(c, True) for c in cands]
        t3 = time.perf_counter()
        timings.update({"features_ms": round(1000 * (t1 - t0), 1), "modelo_ms": round(1000 * (t2 - t1), 1),
                        "explicacion_ms": round(1000 * (t3 - t2), 1),
                        "inferencia_total_ms": round(1000 * (t3 - t0), 1)})
        return {
            "match_id": str(match["match_id"]), "plataforma": str(match["platform"]),
            "parche": patch, "duracion_min": round(dur / 60, 1),
            "gana": "azul" if int(match["blue_win"]) else "rojo",
            "inicio_utc": str(match.get("game_start_utc", "")),
            "modelo": {"nombre": self.name, "theta": self.theta, "theta_candidato": self.theta_cand,
                       "semillas": [m.seed for m in self.members]},
            "curva": [{"minuto": int(t), "p_azul": round(float(main[t]), 4),
                       "p_min": round(float(lo[t]), 4), "p_max": round(float(hi[t]), 4),
                       "baja_certeza": bool(t < LOW_CERTAINTY_MINUTES)} for t in range(len(main))],
            "momentos": moments,
            "candidatos": candidates,
            "avisos": warnings,
            "tiempos": timings,
            # solo para el registro de monitoreo (la API lo quita antes de responder)
            "_monitor": {"features_min15": {k: round(float(v), 4) for k, v in raw.iloc[MONITOR_MINUTE].items()}
                         if len(raw) > MONITOR_MINUTE else None},
        }


# ------------------------------------------------------------------ eventos y texto
MONSTER = {"BARON_NASHOR": "barón", "RIFTHERALD": "heraldo", "HORDE": "larvas", "DRAGON": "dragón"}


def window_events(events: pd.DataFrame, t_from: int, t_to: int) -> dict:
    """Resumen de lo que pasó en (t_from, t_to] por equipo: kills, objetivos y estructuras."""
    out = {"azul": {"kills": 0, "objetivos": [], "estructuras": []},
           "rojo": {"kills": 0, "objetivos": [], "estructuras": []}}
    if events is None or not len(events):
        return out
    events = events.reindex(columns=EVENT_COLS)
    ts = pd.to_numeric(events["timestamp_ms"])
    e = events[(ts > t_from) & (ts <= t_to)]
    team_of_pid = lambda pid: "azul" if 1 <= pid <= 5 else "rojo" if 6 <= pid <= 10 else None  # noqa: E731
    other = {"azul": "rojo", "rojo": "azul"}
    for r in e.itertuples():
        typ = r.type
        if typ == "CHAMPION_KILL":
            victim = team_of_pid(int(_num(r.victim_id)))
            if victim:
                out[other[victim]]["kills"] += 1
        elif typ == "CHAMPION_SPECIAL_KILL" and getattr(r, "kill_type", "") == "KILL_ACE":
            t = team_of_pid(int(_num(r.killer_id)))
            if t:
                out[t]["objetivos"].append("ace")
        elif typ == "ELITE_MONSTER_KILL":
            kt = int(_num(r.killer_team_id))
            t = "azul" if kt == 100 else "rojo" if kt == 200 else team_of_pid(int(_num(r.killer_id)))
            if t:
                name = "elder" if getattr(r, "monster_sub_type", "") == "ELDER_DRAGON" else \
                    MONSTER.get(getattr(r, "monster_type", ""), "monstruo")
                out[t]["objetivos"].append(name)
        elif typ == "BUILDING_KILL":
            owner = int(_num(r.team_id))
            t = "rojo" if owner == 100 else "azul" if owner == 200 else None
            if t:
                b = "inhibidor" if getattr(r, "building_type", "") == "INHIBITOR_BUILDING" else "torre"
                out[t]["estructuras"].append(b)
    return out


def _num(v) -> float:
    try:
        f = float(v)
        return 0.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _team_summary(d: dict) -> str:
    parts = []
    if d["kills"]:
        parts.append(f"{d['kills']} kill{'s' if d['kills'] != 1 else ''}")
    for k in ("objetivos", "estructuras"):
        counts = pd.Series(d[k]).value_counts() if d[k] else {}
        parts += [f"{n}× {name}" if n > 1 else name for name, n in dict(counts).items()]
    return ", ".join(parts)


def moment_text(m, group: str, evs: dict) -> str:
    detail = "; ".join(f"{t}: {s}" for t in ("azul", "rojo") if (s := _team_summary(evs[t])))
    txt = (f"min {m.start_minute}–{m.minute}: {abs(100 * m.delta):.0f} pts de probabilidad a favor "
           f"de {m.team}, sobre todo por {group}")
    return txt + (f" ({detail})" if detail else "")
