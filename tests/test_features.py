"""
Pruebas de B3 (src/features.py):
  - una partida armada a mano con valores esperados conocidos;
  - causalidad: borrar los eventos posteriores a t no cambia los minutos <= t;
  - normalización: flags sin cambios, recorte a [-5, 5], ida y vuelta de scaler.json;
  - (si existen los datos reales) causalidad sobre una partida real.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import (FEATURE_NAMES, FEATURE_TRANSFORMS, Scaler,  # noqa: E402
                          match_to_features)

MIN = 60_000


def make_frames(n_min: int, gold_blue_per_min=400, gold_red_per_min=350) -> pd.DataFrame:
    rows = []
    for m in range(n_min):
        ts = m * MIN + (25 if m else 0)            # los frames reales no caen exacto en el minuto
        for pid in range(1, 11):
            blue = pid <= 5
            gold = 500 + m * (gold_blue_per_min if blue else gold_red_per_min)
            rows.append(dict(minute=m, timestamp_ms=ts, participant_id=pid,
                             total_gold=gold, current_gold=100 if blue else 50, xp=m * 300,
                             level=1 + m // 3, minions_killed=m * 6 if blue else m * 5,
                             jungle_minions_killed=0))
    return pd.DataFrame(rows)


def make_participants() -> pd.DataFrame:
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    return pd.DataFrame(dict(participant_id=range(1, 11),
                             team_id=[100] * 5 + [200] * 5,
                             team_position=roles + ["TOP", "JUNGLE", "", "BOTTOM", "UTILITY"]))


def ev(ts, type, **kw):
    return dict(timestamp_ms=ts, type=type, **kw)


def hand_events() -> pd.DataFrame:
    return pd.DataFrame([
        ev(3 * MIN + 10_000, "CHAMPION_KILL", killer_id=7, victim_id=2),       # rojo mata (first blood rojo)
        ev(4 * MIN + 5_000, "CHAMPION_KILL", killer_id=1, victim_id=6),        # azul mata
        ev(4 * MIN + 6_000, "CHAMPION_KILL", killer_id=0, victim_id=8),        # ejecución de rojo -> cuenta para azul
        ev(5 * MIN, "WARD_PLACED", creator_id=3),
        ev(5 * MIN + 1, "WARD_PLACED", creator_id=4),
        ev(5 * MIN + 2, "WARD_KILL", killer_id=9),
        ev(6 * MIN, "TURRET_PLATE_DESTROYED", team_id=200, killer_id=0),       # placa roja cae -> +1 azul
        ev(8 * MIN, "ELITE_MONSTER_KILL", killer_team_id=100, killer_id=2,
           monster_type="DRAGON", monster_sub_type="FIRE_DRAGON"),
        ev(10 * MIN, "BUILDING_KILL", team_id=200, building_type="TOWER_BUILDING",
           tower_type="OUTER_TURRET"),                                           # torre roja cae -> +1 azul
        ev(20 * MIN, "ELITE_MONSTER_KILL", killer_team_id=200, killer_id=7,
           monster_type="BARON_NASHOR"),                                         # barón rojo 20:00-23:00
        ev(21 * MIN, "BUILDING_KILL", team_id=100, building_type="INHIBITOR_BUILDING"),  # inhib azul 21-26
        ev(22 * MIN, "CHAMPION_SPECIAL_KILL", killer_id=8, kill_type="KILL_ACE"),
        ev(24 * MIN + 30_000, "GAME_END"),   # nunca se usa (fuga)
    ])


def at(df, minute, col):
    return df.loc[minute, col]


def test_hand_checked_match():
    f = match_to_features(make_frames(30), hand_events(), make_participants())
    assert list(f.columns) == FEATURE_NAMES
    assert len(f) == 30
    # economía: 5 jugadores x 50 de oro por minuto
    assert at(f, 10, "gold_diff") == 5 * 50 * 10
    assert at(f, 10, "gold_diff_rel") == pytest.approx(2500 / (10 * 500 + 10 * 10 * 375))
    assert at(f, 10, "cs_diff") == 5 * 10
    assert at(f, 10, "current_gold_diff") == 5 * 50
    assert at(f, 10, "gold_diff_middle") == 500        # el MIDDLE rojo vacío cae por orden en MIDDLE
    # combate
    assert at(f, 3, "kills_diff") == 0                  # el frame 3 es antes de la kill de 3:10
    assert at(f, 4, "kills_diff") == -1 and at(f, 4, "first_blood") == -1
    assert at(f, 5, "kills_diff") == 1
    assert at(f, 5, "kills_diff_2m") == 1               # (3:00.025, 5:00.025]: -1 +1 +1
    assert at(f, 6, "kills_diff_2m") == 2               # ya salió la de 3:10
    assert at(f, 7, "kills_diff_2m") == 0
    assert at(f, 22, "aces_diff") == -1
    # objetivos
    assert at(f, 6, "plates_diff") == 1
    assert at(f, 8, "dragons_diff") == 1
    assert at(f, 10, "towers_outer_diff") == 1
    assert at(f, 19, "baron_active") == 0
    assert at(f, 20, "baron_active") == -1 and at(f, 22, "baron_active") == -1
    assert at(f, 23, "baron_active") == 0               # 23:00.025 >= 20:00 + 180 s
    assert at(f, 23, "barons_diff") == -1
    assert at(f, 21, "inhibs_down_diff") == -1
    assert at(f, 26, "inhibs_down_diff") == 0            # reaparece a los 5 min
    # tendencia
    assert at(f, 10, "gold_diff_d1") == 250
    assert at(f, 10, "gold_diff_d3") == 750
    assert at(f, 1, "gold_diff_d3") == 250               # antes del minuto 3 se compara con el 0


def test_wards_timing():
    f = match_to_features(make_frames(10), hand_events(), make_participants())
    # wards a 5:00.000 y 5:00.001 (azul) y una destruida por rojo; frame 5 = 5:00.025
    assert at(f, 5, "wards_placed_diff") == 2
    assert at(f, 5, "wards_killed_diff") == -1
    assert at(f, 7, "wards_placed_3m_diff") == 2         # (4:00.025, 7:00.025]
    assert at(f, 8, "wards_placed_3m_diff") == 0         # 5:00.000 ya no está en (5:00.025, 8:00.025]


def test_dragon_soul_and_elder():
    drag = [ev((5 + 5 * i) * MIN, "ELITE_MONSTER_KILL", killer_team_id=100,
               monster_type="DRAGON", monster_sub_type="AIR_DRAGON") for i in range(4)]
    elder = ev(30 * MIN, "ELITE_MONSTER_KILL", killer_team_id=200,
               monster_type="DRAGON", monster_sub_type="ELDER_DRAGON")
    f = match_to_features(make_frames(35), pd.DataFrame(drag + [elder]), make_participants())
    assert at(f, 19, "dragon_soul") == 0
    assert at(f, 20, "dragon_soul") == 1
    assert at(f, 31, "dragons_diff") == 4                # el elder no cuenta como dragón
    assert at(f, 31, "elder_active") == -1
    assert at(f, 33, "elder_active") == 0                # 150 s


def random_events(rng, n_min, n=150) -> pd.DataFrame:
    types = ["CHAMPION_KILL", "WARD_PLACED", "WARD_KILL", "TURRET_PLATE_DESTROYED",
             "BUILDING_KILL", "ELITE_MONSTER_KILL", "CHAMPION_SPECIAL_KILL"]
    rows = []
    for _ in range(n):
        t = rng.choice(types)
        r = ev(int(rng.integers(0, n_min * MIN)), t,
               killer_id=int(rng.integers(0, 11)), victim_id=int(rng.integers(1, 11)),
               creator_id=int(rng.integers(1, 11)), team_id=int(rng.choice([100, 200])),
               killer_team_id=int(rng.choice([100, 200])))
        if t == "BUILDING_KILL":
            r["building_type"] = rng.choice(["TOWER_BUILDING", "INHIBITOR_BUILDING"])
            r["tower_type"] = rng.choice(["OUTER_TURRET", "INNER_TURRET", "BASE_TURRET", "NEXUS_TURRET"])
        if t == "ELITE_MONSTER_KILL":
            r["monster_type"] = rng.choice(["DRAGON", "BARON_NASHOR", "RIFTHERALD", "HORDE"])
            r["monster_sub_type"] = rng.choice(["FIRE_DRAGON", "ELDER_DRAGON"])
        if t == "CHAMPION_SPECIAL_KILL":
            r["kill_type"] = rng.choice(["KILL_ACE", "KILL_MULTI", "KILL_FIRST_BLOOD"])
        rows.append(r)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_causality_synthetic(seed):
    rng = np.random.default_rng(seed)
    n_min = 32
    frames, parts = make_frames(n_min), make_participants()
    events = random_events(rng, n_min)
    full = match_to_features(frames, events, parts)
    for t in (0, 5, 13, 20, 31):
        cut_ts = full.attrs["timestamp_ms"][t]
        cut = match_to_features(frames, events[events["timestamp_ms"] <= cut_ts], parts)
        pd.testing.assert_frame_equal(full.loc[:t], cut.loc[:t])


def test_truncates_to_60():
    f = match_to_features(make_frames(75), pd.DataFrame(columns=["timestamp_ms", "type"]),
                          make_participants())
    assert len(f) == 60 and f.index[-1] == 59


def test_scaler_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 100, size=(500, len(FEATURE_NAMES)))
    flags = [i for i, t in enumerate(FEATURE_TRANSFORMS) if t == "flag"]
    raw[:, flags] = rng.choice([-1, 0, 1], size=(500, len(flags)))
    raw[0, 1] = 1e9                                     # outlier
    sc = Scaler.fit(raw)
    x = sc.transform(raw)
    assert x.dtype == np.float32
    assert np.abs(x).max() <= 5.0
    np.testing.assert_array_equal(x[:, flags], raw[:, flags])
    sc.save(tmp_path / "scaler.json")
    np.testing.assert_allclose(Scaler.load(tmp_path / "scaler.json").transform(raw), x)


MERGE = ROOT / "data" / "datos_merge"


@pytest.mark.skipif(not (MERGE / "frames.csv").exists(), reason="sin datos reales")
def test_causality_real_match():
    from src.build_features import iter_matches
    parts = pd.read_csv(MERGE / "participants.csv", nrows=50)
    mid, frames, events = next(iter_matches(limit=1))
    p = parts[parts["match_id"] == mid]
    full = match_to_features(frames, events, p)
    for t in (5, 15, len(full) - 2):
        cut_ts = full.attrs["timestamp_ms"][t]
        cut = match_to_features(frames, events[events["timestamp_ms"] <= cut_ts], p)
        pd.testing.assert_frame_equal(full.loc[:t], cut.loc[:t])
