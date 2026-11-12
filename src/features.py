"""
B3 - Features por minuto de una partida (guía, sección 3.4). Lo usan el entrenamiento
(src/build_features.py) Y la app: si la app calcula distinto, el modelo falla sin avisar.

    from src.features import match_to_features, Scaler
    raw = match_to_features(frames, events, participants)   # DataFrame [minutos, F], sin normalizar
    X = Scaler.load("features/scaler.json").transform(raw.to_numpy())

Entradas: las filas de UNA partida, con las columnas de riot_data/parse.py
(frames.csv, events.csv, participants.csv). En la app basta con pd.DataFrame(parse_match(...)[tabla]).

Reglas:
  - Todo es azul - rojo (positivo = ventaja azul).
  - Causal: la fila del minuto t solo usa eventos con timestamp_ms <= timestamp del frame t.
  - Nada de la lista de fugas (guía 3.3): no se leen duration_s, win, GAME_END ni
    estadísticas finales de participants (solo campeón, equipo y rol).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

T_MAX = 60                        # minutos 0..59 (guía: truncado a 60)
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
BARON_BUFF_MS = 180_000
ELDER_BUFF_MS = 150_000
INHIB_RESPAWN_MS = 300_000
SOUL_DRAGONS = 4

# (nombre, grupo del contrato, transformación)
#   std  -> estandarizar con media/desv. de train
#   slog -> sign(x)*log1p(|x|) y luego estandarizar (conteos sesgados: wards)
#   flag -> -1/0/1, sin cambios
FEATURES: list[tuple[str, str, str]] = [
    ("minute", "contexto", "std"),
    # economía
    ("gold_diff", "economia", "std"),
    ("gold_diff_rel", "economia", "std"),        # dif / (oro de ambos), en el minuto t
    ("xp_diff", "economia", "std"),
    ("cs_diff", "economia", "std"),               # minions + jungla
    ("level_diff", "economia", "std"),            # nivel medio azul - nivel medio rojo
    ("current_gold_diff", "economia", "std"),     # oro sin gastar
    # por rol
    ("gold_diff_top", "rol", "std"),
    ("gold_diff_jungle", "rol", "std"),
    ("gold_diff_middle", "rol", "std"),
    ("gold_diff_bottom", "rol", "std"),
    ("gold_diff_utility", "rol", "std"),
    # combate
    ("kills_diff", "combate", "std"),
    ("kills_diff_2m", "combate", "std"),          # kills en (t-2 min, t]
    ("first_blood", "combate", "flag"),
    ("aces_diff", "combate", "std"),
    # objetivos
    ("towers_outer_diff", "objetivos", "std"),
    ("towers_inner_diff", "objetivos", "std"),
    ("towers_base_diff", "objetivos", "std"),
    ("towers_nexus_diff", "objetivos", "std"),
    ("plates_diff", "objetivos", "std"),
    ("inhibs_down_diff", "objetivos", "std"),     # inhibidores rojos caídos AHORA - azules
    ("dragons_diff", "objetivos", "std"),         # sin contar elder
    ("dragon_soul", "objetivos", "flag"),
    ("elder_active", "objetivos", "flag"),
    ("barons_diff", "objetivos", "std"),
    ("baron_active", "objetivos", "flag"),
    ("heralds_diff", "objetivos", "std"),
    ("grubs_diff", "objetivos", "std"),           # larvas del vacío (HORDE)
    # visión
    ("wards_placed_diff", "vision", "slog"),
    ("wards_killed_diff", "vision", "slog"),
    ("wards_placed_3m_diff", "vision", "slog"),
    ("wards_killed_3m_diff", "vision", "slog"),
    # tendencia (solo modelos snapshot; la GRU las quita con drop_groups)
    ("gold_diff_d1", "tendencia", "std"),
    ("gold_diff_d3", "tendencia", "std"),
]
FEATURE_NAMES = [f for f, _, _ in FEATURES]
FEATURE_GROUPS = [g for _, g, _ in FEATURES]
FEATURE_TRANSFORMS = [t for _, _, t in FEATURES]

FRAME_COLS = ["minute", "timestamp_ms", "participant_id", "total_gold", "current_gold", "xp",
              "level", "minions_killed", "jungle_minions_killed"]
EVENT_COLS = ["timestamp_ms", "type", "killer_id", "victim_id", "creator_id", "killer_team_id",
              "team_id", "monster_type", "monster_sub_type", "building_type", "tower_type",
              "kill_type"]
EVENT_TYPES_USED = {"CHAMPION_KILL", "CHAMPION_SPECIAL_KILL", "ELITE_MONSTER_KILL",
                    "BUILDING_KILL", "TURRET_PLATE_DESTROYED", "WARD_PLACED", "WARD_KILL"}


# ---------------------------------------------------------------- utilidades

def dedup_events(events: pd.DataFrame) -> pd.DataFrame:
    """Quita filas de eventos IDÉNTICAS en todas sus columnas, igual que merge.py (línea ~158).

    El timeline de Riot repite algunos eventos tal cual (sobre todo WARD_PLACED "UNDEFINED" y
    alguna placa). El dataset de entrenamiento (datos_merge) ya pasó por esta limpieza, así que
    los datos que NO vienen de datos_merge (la app, partidas propias de collect.py) deben pasar
    por aquí antes de match_to_features. No deduplicar solo por las columnas que usan las
    features: borraría eventos reales distintos (p. ej. dos larvas en el mismo milisegundo).
    """
    return events.drop_duplicates(ignore_index=True) if len(events) else events



def _side_of_pid(pid: np.ndarray, team_of: dict[int, int]) -> np.ndarray:
    """+1 si el participante es azul, -1 si es rojo, 0 si no hay participante (p. ej. torre)."""
    out = np.zeros(len(pid), dtype=np.int8)
    for i, p in enumerate(pid):
        if p > 0:
            team = team_of.get(int(p), 100 if p <= 5 else 200)
            out[i] = 1 if team == 100 else -1
    return out


def _side_of_team(team: np.ndarray) -> np.ndarray:
    return np.where(team == 100, 1, np.where(team == 200, -1, 0)).astype(np.int8)


def _cum(ts: np.ndarray, side: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Suma de `side` de los eventos con ts <= at (ts ordenado)."""
    if len(ts) == 0:
        return np.zeros(len(at))
    csum = np.concatenate([[0], np.cumsum(side, dtype=np.int64)])
    return csum[np.searchsorted(ts, at, side="right")].astype(float)


def _window(ts: np.ndarray, side: np.ndarray, at: np.ndarray, width_ms: int) -> np.ndarray:
    """Suma de `side` de los eventos con at - width < ts <= at."""
    return _cum(ts, side, at) - _cum(ts, side, at - width_ms)


def _active(ts: np.ndarray, side: np.ndarray, at: np.ndarray, dur_ms: int) -> np.ndarray:
    """Cuántos efectos activos (ts <= at < ts + dur) tiene azul menos rojo."""
    if len(ts) == 0:
        return np.zeros(len(at))
    live = (ts[None, :] <= at[:, None]) & (at[:, None] < ts[None, :] + dur_ms)
    return (live * side[None, :]).sum(1).astype(float)


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df:
        return np.zeros(len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()


def _str(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df:
        return np.full(len(df), "", dtype=object)
    return df[col].fillna("").astype(str).to_numpy()


def roles_of(participants: pd.DataFrame) -> dict[int, str]:
    """participant_id -> rol. Rol vacío -> por orden (1/6 TOP, 2/7 JUNGLE, ...), guía 3.2."""
    out = {}
    pos = dict(zip(participants["participant_id"].astype(int),
                   participants["team_position"].fillna("").astype(str)))
    for pid in range(1, 11):
        r = pos.get(pid, "")
        out[pid] = r if r in ROLES else ROLES[(pid - 1) % 5]
    return out


# ---------------------------------------------------------------- features de una partida

def match_to_features(frames: pd.DataFrame, events: pd.DataFrame,
                      participants: pd.DataFrame, t_max: int = T_MAX) -> pd.DataFrame:
    """Features crudas (sin normalizar) de una partida: una fila por minuto 0..min(n, t_max)-1."""
    team_of = dict(zip(participants["participant_id"].astype(int),
                       participants["team_id"].astype(int)))
    role_of = roles_of(participants)
    blue_pids = [p for p in range(1, 11) if team_of.get(p, 100 if p <= 5 else 200) == 100]

    # ---- frames -> matrices [minutos, 10]
    fr = frames[FRAME_COLS].copy()
    fr = fr[fr["minute"] < t_max]
    piv = fr.pivot_table(index="minute", columns="participant_id", aggfunc="first")
    piv = piv.sort_index().ffill().fillna(0)
    minutes = piv.index.to_numpy()
    if not np.array_equal(minutes, np.arange(len(minutes))):
        raise ValueError(f"Minutos no contiguos desde 0: {minutes[:5]}...")
    at = fr.groupby("minute")["timestamp_ms"].max().reindex(minutes).to_numpy(dtype=np.int64)

    def stat(col):
        m = piv[col].reindex(columns=range(1, 11), fill_value=0).to_numpy(dtype=float)
        sign = np.array([1.0 if p in blue_pids else -1.0 for p in range(1, 11)])
        return m, sign

    gold, sign = stat("total_gold")
    xp, _ = stat("xp")
    cs = stat("minions_killed")[0] + stat("jungle_minions_killed")[0]
    level, _ = stat("level")
    cur, _ = stat("current_gold")

    f = {}
    f["minute"] = minutes.astype(float)
    f["gold_diff"] = gold @ sign
    total = gold.sum(1)
    f["gold_diff_rel"] = np.divide(f["gold_diff"], total, out=np.zeros(len(total)), where=total > 0)
    f["xp_diff"] = xp @ sign
    f["cs_diff"] = cs @ sign
    f["level_diff"] = (level @ sign) / 5.0
    f["current_gold_diff"] = cur @ sign
    for role in ROLES:
        w = np.array([sign[p - 1] if role_of[p] == role else 0.0 for p in range(1, 11)])
        f[f"gold_diff_{role.lower()}"] = gold @ w

    # ---- eventos (ordenados por tiempo)
    ev = events[events["type"].isin(EVENT_TYPES_USED)] if len(events) else events
    ev = ev.sort_values("timestamp_ms", kind="stable")
    ts = _num(ev, "timestamp_ms").astype(np.int64)
    typ = _str(ev, "type")
    killer = _num(ev, "killer_id").astype(int)
    victim = _num(ev, "victim_id").astype(int)
    creator = _num(ev, "creator_id").astype(int)
    killer_team = _num(ev, "killer_team_id").astype(int)
    team = _num(ev, "team_id").astype(int)
    monster = _str(ev, "monster_type")
    sub = _str(ev, "monster_sub_type")
    building = _str(ev, "building_type")
    tower = _str(ev, "tower_type")
    kill_type = _str(ev, "kill_type")

    def sel(m, side):
        keep = m & (side != 0)
        return ts[keep], side[keep]

    # combate: la kill es del equipo contrario a la víctima (incluye ejecuciones)
    kills = sel(typ == "CHAMPION_KILL", -_side_of_pid(victim, team_of))
    f["kills_diff"] = _cum(*kills, at)
    f["kills_diff_2m"] = _window(*kills, at, 120_000)
    k_ts, k_side = kills
    f["first_blood"] = np.zeros(len(at))
    if len(k_ts):
        f["first_blood"] = np.where(at >= k_ts[0], float(k_side[0]), 0.0)
    f["aces_diff"] = _cum(*sel((typ == "CHAMPION_SPECIAL_KILL") & (kill_type == "KILL_ACE"),
                               _side_of_pid(killer, team_of)), at)

    # estructuras: team_id es el DUEÑO de la torre/placa/inhibidor (el que la pierde)
    owner_side = -_side_of_team(team)          # +1 = azul destruyó algo rojo
    is_tower = (typ == "BUILDING_KILL") & (building == "TOWER_BUILDING")
    for kind, name in [("OUTER_TURRET", "outer"), ("INNER_TURRET", "inner"),
                       ("BASE_TURRET", "base"), ("NEXUS_TURRET", "nexus")]:
        f[f"towers_{name}_diff"] = _cum(*sel(is_tower & (tower == kind), owner_side), at)
    f["plates_diff"] = _cum(*sel(typ == "TURRET_PLATE_DESTROYED", owner_side), at)
    inhib = sel((typ == "BUILDING_KILL") & (building == "INHIBITOR_BUILDING"), owner_side)
    f["inhibs_down_diff"] = _active(*inhib, at, INHIB_RESPAWN_MS)

    # monstruos épicos: equipo que lo mató
    mside = _side_of_team(killer_team)
    mside = np.where(mside == 0, _side_of_pid(killer, team_of), mside).astype(np.int8)
    elite = typ == "ELITE_MONSTER_KILL"
    is_elder = elite & (sub == "ELDER_DRAGON")
    drag_ts, drag_side = sel(elite & (monster == "DRAGON") & ~is_elder, mside)
    f["dragons_diff"] = _cum(drag_ts, drag_side, at)
    # alma: el primero en llegar a 4 dragones (DRAGON_SOUL_GIVEN trae team_id=0 en ~75 %)
    blue_d = _cum(drag_ts, (drag_side == 1).astype(np.int8), at)
    red_d = _cum(drag_ts, (drag_side == -1).astype(np.int8), at)
    f["dragon_soul"] = np.where(blue_d >= SOUL_DRAGONS, 1.0, np.where(red_d >= SOUL_DRAGONS, -1.0, 0.0))
    f["elder_active"] = np.clip(_active(*sel(is_elder, mside), at, ELDER_BUFF_MS), -1, 1)
    baron = sel(elite & (monster == "BARON_NASHOR"), mside)
    f["barons_diff"] = _cum(*baron, at)
    f["baron_active"] = np.clip(_active(*baron, at, BARON_BUFF_MS), -1, 1)
    f["heralds_diff"] = _cum(*sel(elite & (monster == "RIFTHERALD"), mside), at)
    f["grubs_diff"] = _cum(*sel(elite & (monster == "HORDE"), mside), at)

    # visión
    placed = sel(typ == "WARD_PLACED", _side_of_pid(creator, team_of))
    killed = sel(typ == "WARD_KILL", _side_of_pid(killer, team_of))
    f["wards_placed_diff"] = _cum(*placed, at)
    f["wards_killed_diff"] = _cum(*killed, at)
    f["wards_placed_3m_diff"] = _window(*placed, at, 180_000)
    f["wards_killed_3m_diff"] = _window(*killed, at, 180_000)

    # tendencia: Δ de la diferencia de oro (antes del minuto k se compara con el minuto 0)
    g = f["gold_diff"]
    for k in (1, 3):
        prev = np.concatenate([np.full(k, g[0]), g[:-k]])[: len(g)]
        f[f"gold_diff_d{k}"] = g - prev

    out = pd.DataFrame({name: f[name] for name in FEATURE_NAMES}, index=pd.Index(minutes, name="minute"))
    out.attrs["timestamp_ms"] = at
    return out


# ---------------------------------------------------------------- normalización

def apply_transforms(raw: np.ndarray, transforms=FEATURE_TRANSFORMS) -> np.ndarray:
    """Transformaciones previas al escalado (signed log1p en conteos sesgados)."""
    x = np.asarray(raw, dtype=np.float64).copy()
    for j, tr in enumerate(transforms):
        if tr == "slog":
            x[..., j] = np.sign(x[..., j]) * np.log1p(np.abs(x[..., j]))
    return x


class Scaler:
    """StandardScaler ajustado SOLO con train; flags sin cambios; recorte a [-clip, clip]."""

    def __init__(self, feature_names, transforms, mean, std, clip: float = 5.0):
        self.feature_names = list(feature_names)
        self.transforms = list(transforms)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.clip = clip

    @classmethod
    def fit(cls, raw_rows: np.ndarray, feature_names=FEATURE_NAMES,
            transforms=FEATURE_TRANSFORMS, clip: float = 5.0) -> "Scaler":
        x = apply_transforms(raw_rows, transforms)
        mean, std = x.mean(0), x.std(0)
        flags = np.array([t == "flag" for t in transforms])
        mean[flags], std[flags] = 0.0, 1.0
        std[std < 1e-8] = 1.0                       # columnas constantes
        return cls(feature_names, transforms, mean, std, clip)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        x = (apply_transforms(raw, self.transforms) - self.mean) / self.std
        return np.clip(x, -self.clip, self.clip).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "feature_names": self.feature_names, "transforms": self.transforms,
            "mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip,
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["feature_names"], d["transforms"], d["mean"], d["std"], d.get("clip", 5.0))
