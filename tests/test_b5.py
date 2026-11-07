"""
Pruebas de B5: métricas de src/evaluate.py contra sklearn, bootstrap por partida y regla R0.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.baselines import rule_r0  # noqa: E402
from src.evaluate import Rows, evaluate_set  # noqa: E402
from src.features import match_to_features  # noqa: E402
from tests.test_features import MIN, ev, make_frames, make_participants  # noqa: E402


def fake_probs(n=300, t=40, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    lengths = rng.integers(15, t + 1, n)
    signal = (y * 2 - 1)[:, None] * np.linspace(0, 2, t)[None, :]
    z = signal + rng.normal(0, 1.5, (n, t))
    probs = 1 / (1 + np.exp(-z))
    probs[np.arange(t)[None, :] >= lengths[:, None]] = np.nan
    probs[:, 0] = 0.5                                     # empates (como el minuto 0 real)
    return probs.astype(np.float32), y


def test_metrics_match_sklearn():
    probs, y = fake_probs()
    r = Rows(probs, y)
    m = r.metrics()
    valid = ~np.isnan(probs)
    p = probs[valid].astype(np.float64)
    yy = np.broadcast_to(y[:, None], probs.shape)[valid]
    assert m["auc"] == pytest.approx(roc_auc_score(yy, p), abs=1e-9)
    assert m["logloss"] == pytest.approx(log_loss(yy, p), abs=1e-6)
    assert m["brier"] == pytest.approx(brier_score_loss(yy, p), abs=1e-9)
    assert m["accuracy"] == pytest.approx(((p >= 0.5) == yy).mean())
    assert 0 <= m["ece"] < 0.5


def test_bootstrap_counts_equal_duplication():
    probs, y = fake_probs(n=50)
    counts = np.zeros(50)
    counts[:20] = 1                                        # las partidas 20-49 no salen
    counts[5] = 3                                          # la partida 5 entra 3 veces
    dup_idx = np.repeat(np.arange(50), counts.astype(int))
    a = Rows(probs, y).metrics(counts)
    b = Rows(probs[dup_idx], y[dup_idx]).metrics()
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_evaluate_set_ci_contains_point():
    probs, y = fake_probs()
    df = evaluate_set(probs, y, n_boot=100)
    g = df[df.tramo == "global"].iloc[0]
    assert g["auc_lo"] <= g["auc"] <= g["auc_hi"]
    assert set(df.tramo) >= {"global", "0-4", "30+"}


def test_compare_runs_paired(tmp_path):
    from src.evaluate import compare_runs
    probs, y = fake_probs(n=200)
    ids = np.array([f"M_{i}" for i in range(200)])
    split = np.array(["test"] * 200)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, match_id=ids, split=split, y=y, probs=probs)
    perm = np.random.default_rng(0).permutation(200)          # mismo modelo, otro orden
    np.savez(b, match_id=ids[perm], split=split, y=y[perm], probs=probs[perm])
    cmp = compare_runs([a, b], ["A", "B"], n_boot=50)
    assert np.allclose(cmp["d_auc"], 0) and np.allclose(cmp["d_auc_lo"], 0) and np.allclose(cmp["d_auc_hi"], 0)


@pytest.mark.parametrize("scale", [0.15, 1.0, 4.0])
def test_fit_temperature_matches_grid(scale):
    """Regresión: con modelos poco confiados (logits pequeños) LBFGS daba T=0.0002."""
    from types import SimpleNamespace
    from src.train import fit_temperature
    rng = np.random.default_rng(1)
    n, t = 2000, 20
    true = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-true))).astype(np.int8)
    logits = np.repeat((true * scale)[:, None], t, axis=1).astype(np.float32)
    T = fit_temperature(logits, SimpleNamespace(mask=np.ones((n, t), bool), y=y))
    grid = np.exp(np.linspace(-4, 4, 1601))
    z = true * scale
    losses = [np.mean(np.logaddexp(0, z / g) - y * z / g) for g in grid]
    assert T == pytest.approx(grid[int(np.argmin(losses))], rel=0.02)


def test_rule_r0():
    events = pd.DataFrame([
        ev(10 * MIN + 30_000, "ELITE_MONSTER_KILL", monster_type="BARON_NASHOR", killer_team_id=100),
        ev(14 * MIN + 1, "CHAMPION_SPECIAL_KILL", kill_type="KILL_ACE", killer_id=2),
        ev(17 * MIN, "BUILDING_KILL", building_type="INHIBITOR_BUILDING", team_id=200),
        ev(5 * MIN, "ELITE_MONSTER_KILL", monster_type="DRAGON", killer_team_id=100),  # no cuenta
    ])
    frames = make_frames(25)
    # swing de oro: en el minuto 20 el azul gana 2,000 de golpe
    frames.loc[(frames.minute >= 20) & (frames.participant_id == 1), "total_gold"] += 2000
    f = match_to_features(frames, events, make_participants())
    # barón 10:30 -> frame 11; ace 14:00.001 -> frame 14 (14:00.025); inhib 17:00 -> frame 17
    assert rule_r0(f, events) == [11, 14, 17, 20]
