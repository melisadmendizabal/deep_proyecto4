"""
Pruebas de B11 (app/inference.py, app/api.py). Usan un modelo diminuto armado en el momento
(los pesos reales no van a git) y un cliente de Riot falso (no gastan la API).

    python -m pytest tests -q
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.inference import EXPLAIN_GROUPS, AnalysisError, Engine, tables_from_riot  # noqa: E402
from src.features import (FEATURE_GROUPS, FEATURE_NAMES, FEATURE_TRANSFORMS, Scaler,  # noqa: E402
                          dedup_events)
from src.models import build_model  # noqa: E402
from tests.test_features import MIN, hand_events, make_frames, make_participants  # noqa: E402


def make_engine_dir(tmp: Path, theta: float = 1e-6) -> Path:
    """Modelo aleatorio diminuto: su curva casi no se mueve, por eso θ ≈ 0 (siempre hay candidatos)."""
    d = tmp / "model"
    d.mkdir()
    Scaler(FEATURE_NAMES, FEATURE_TRANSFORMS, np.zeros(len(FEATURE_NAMES)),
           np.full(len(FEATURE_NAMES), 1000.0)).save(d / "scaler.json")
    names = [n for n, g in zip(FEATURE_NAMES, FEATURE_GROUPS) if g != "tendencia"]
    members = []
    for seed in (42, 43):
        torch.manual_seed(seed)
        cfg = {"type": "gru", "hidden": 8, "layers": 1, "dropout": 0.0}
        model = build_model(cfg, len(names))
        torch.save({"model_cfg": cfg, "feature_names": names, "state_dict": model.state_dict(),
                    "temperature": 1.0}, d / f"seed_{seed}.pt")
        members.append({"file": f"seed_{seed}.pt", "seed": seed, "temperature": 1.0})
    (d / "manifest.json").write_text(json.dumps({
        "name": "M4test", "members": members, "train_patches": ["16.21"], "queue_id": 420,
        "min_duration_s": 900,
        "detector": {"theta": theta, "windows": [1, 2], "merge_gap": 2, "top_k": 5}}))
    return d


def make_tables(minutes=30, duration_s=None, queue=420, patch="16.21"):
    parts = make_participants()
    parts["champion_name"] = "X"
    match = pd.DataFrame([{"match_id": "NA1_1234567", "platform": "na1", "duration_s": duration_s or minutes * 60,
                           "blue_win": 1, "queue_id": queue, "patch": patch, "game_start_utc": ""}])
    return {"frames": make_frames(minutes), "events": hand_events(), "participants": parts, "matches": match}


@pytest.fixture
def engine(tmp_path):
    return Engine(make_engine_dir(tmp_path))


def test_analyze_shape_and_band(engine):
    res = engine.analyze(make_tables())
    assert len(res["curva"]) == 30 and res["gana"] == "azul"
    for c in res["curva"]:
        assert c["p_min"] - 1e-6 <= c["p_azul"] <= c["p_max"] + 1e-6
        assert c["baja_certeza"] == (c["minuto"] < 5)
    assert 1 <= len(res["momentos"]) <= 5
    for m in res["momentos"]:
        assert m["desde_min"] < m["minuto"] and m["favorece"] in ("azul", "rojo")
        assert m["explicacion"]["grupo"] in EXPLAIN_GROUPS
        share = sum(m["explicacion"]["aporte"].values())
        assert share == pytest.approx(1.0, abs=1e-2) or share == 0
    assert {"features_ms", "modelo_ms", "inferencia_total_ms"} <= set(res["tiempos"])
    assert res["avisos"] == []


def test_warnings_and_errors(engine):
    res = engine.analyze(make_tables(queue=440, patch="16.23"))
    assert any("Solo/Duo" in w for w in res["avisos"]) and any("16.23" in w for w in res["avisos"])
    with pytest.raises(AnalysisError) as exc:
        engine.analyze(make_tables(minutes=12))
    assert exc.value.status == 422


def test_explanation_finds_the_frozen_group(tmp_path):
    """Si el único cambio entre dos minutos es de oro, la explicación debe ser 'economía'."""
    eng = Engine(make_engine_dir(tmp_path))
    tables = make_tables()
    tables["events"] = tables["events"].iloc[0:0]            # sin kills ni objetivos
    f = tables["frames"]
    f.loc[(f.minute >= 15) & (f.participant_id <= 5), "total_gold"] += 4000
    res = eng.analyze(tables)
    from src.features import match_to_features
    raw = match_to_features(f, tables["events"], tables["participants"])
    exp = eng.explain(raw, 14, 15, float(eng.predict(raw)[0][14]))
    assert exp["grupo"] == "economía"
    assert len(res["curva"]) == 30


def test_dedup_events_only_identical_rows():
    ev = pd.DataFrame([{"timestamp_ms": 1, "type": "WARD_PLACED", "creator_id": 3, "position_x": 1},
                       {"timestamp_ms": 1, "type": "WARD_PLACED", "creator_id": 3, "position_x": 1},
                       {"timestamp_ms": 1, "type": "WARD_PLACED", "creator_id": 3, "position_x": 2}])
    assert len(dedup_events(ev)) == 2


# ------------------------------------------------------------------ API con Riot falso
RAW = ROOT / "data" / "raw"


def _some_raw_match():
    for plat in ("na1", "kr", "euw1", "la1"):
        for p in sorted((RAW / plat).glob("*_timeline.json.gz"))[:50] if (RAW / plat).exists() else []:
            mid = p.name.replace("_timeline.json.gz", "")
            m = RAW / plat / f"{mid}.json.gz"
            if m.exists() and mid.lower().startswith(plat):
                match = json.load(gzip.open(m))
                if match["info"]["gameDuration"] >= 15 * 60:
                    return mid, match, json.load(gzip.open(p))
    return None


class FakeRiot:
    def __init__(self, mid, match, timeline):
        self.mid, self.m, self.t, self.calls = mid, match, timeline, 0

    def match(self, region, mid):
        self.calls += 1
        return self.m if mid == self.mid else None

    def timeline(self, region, mid):
        self.calls += 1
        return self.t if mid == self.mid else None

    def account_by_riot_id(self, region, name, tag):
        return {"puuid": self.m["info"]["participants"][0]["puuid"], "gameName": name, "tagLine": tag} \
            if name == "Yo" else None

    def match_ids(self, region, puuid, count, queue=None, start_time=None):
        return [self.mid]


@pytest.fixture
def client(tmp_path, monkeypatch):
    raw = _some_raw_match()
    if raw is None:
        pytest.skip("sin JSON crudos en data/raw")
    from fastapi.testclient import TestClient
    import app.api as api
    monkeypatch.setattr(api, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(api, "DEMO_DIR", tmp_path / "demo")
    fake = FakeRiot(*raw)
    api.init(engine=Engine(make_engine_dir(tmp_path)), riot=fake)
    return TestClient(api.app), raw[0], fake, api


def test_api_analyze_and_cache(client):
    tc, mid, fake, api = client
    r = tc.get("/analyze", params={"match_id": mid.lower()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["match_id"] == mid and body["fuente"] == "riot" and len(body["curva"]) > 10
    assert fake.calls == 2
    r2 = tc.get("/analyze", params={"match_id": mid})
    assert r2.json()["fuente"] == "memoria" and fake.calls == 2         # no vuelve a llamar a Riot
    assert (api.CACHE_DIR / "predicciones.jsonl").read_text(encoding="utf-8").count("\n") == 1
    api.S.memory.clear()
    assert tc.get("/analyze", params={"match_id": mid}).json()["fuente"] == "disco"


def test_api_errors(client):
    tc, mid, fake, api = client
    assert tc.get("/analyze", params={"match_id": "hola"}).status_code == 400
    assert tc.get("/analyze", params={"match_id": "NA1_9999999999"}).status_code == 404
    assert tc.get("/player", params={"riot_id": "sin-tag"}).status_code == 400
    assert tc.get("/player", params={"riot_id": "Otro#TAG"}).status_code == 404
    api.S.riot, api.S.riot_error = None, "sin key"
    assert tc.get("/analyze", params={"match_id": "NA1_9999999998"}).status_code == 503


def test_api_player_and_health(client):
    tc, mid, fake, api = client
    r = tc.get("/player", params={"riot_id": "Yo#LAN", "platform": mid.split("_")[0].lower()})
    assert r.status_code == 200, r.text
    p = r.json()["partidas"]
    assert p and p[0]["match_id"] == mid and p[0]["campeon"]
    h = tc.get("/health").json()
    assert h["ok"] and h["modelo"] == "M4test" and h["api_riot"]
