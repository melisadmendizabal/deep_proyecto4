"""
Pruebas de M6: Transformer causal con embeddings de campeones opcionales.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models import MODELS, build_model  # noqa: E402
from src.train import forward, load_config, run  # noqa: E402
from tests.make_fake_sequences import make  # noqa: E402

CFG = {"type": "transformer", "hidden": 16, "layers": 2, "heads": 2, "dropout": 0.0}


def champs(b=4, seed=0):
    return torch.randint(1, 950, (b, 10), generator=torch.Generator().manual_seed(seed))


def test_transformer_registered():
    assert "transformer" in MODELS


@pytest.mark.parametrize("use_champs", [False, True])
def test_transformer_is_causal(use_champs):
    torch.manual_seed(0)
    model = build_model({**CFG, "champions": use_champs}, 7).eval()
    c = champs() if use_champs else None
    x = torch.randn(4, 60, 7)
    x2 = x.clone()
    x2[:, 30:] = torch.randn(4, 30, 7) * 10           # cambia solo el "futuro"
    with torch.no_grad():
        a, b = model(x, champions=c), model(x2, champions=c)
    assert a.shape == (4, 60)
    assert torch.allclose(a[:, :30], b[:, :30], atol=1e-5), "el Transformer usa minutos futuros"
    assert not torch.allclose(a[:, 30:], b[:, 30:])


def test_transformer_padding_right_does_not_matter():
    """El relleno al final (partidas más cortas en el batch) no cambia los minutos válidos."""
    torch.manual_seed(0)
    model = build_model(CFG, 5).eval()
    x = torch.randn(1, 40, 5)
    with torch.no_grad():
        short = model(x[:, :25])
        long = model(torch.cat([x[:, :25], torch.zeros(1, 15, 5)], dim=1))
    assert torch.allclose(short, long[:, :25], atol=1e-5)


def test_draft_order_within_team_does_not_matter_but_teams_do():
    torch.manual_seed(0)
    model = build_model({**CFG, "champions": True}, 5).eval()
    x = torch.randn(2, 20, 5)
    c = champs(2)
    shuffled = torch.cat([c[:, [4, 2, 0, 3, 1]], c[:, [9, 5, 7, 6, 8]]], dim=1)   # otro orden dentro de cada equipo
    swapped = torch.cat([c[:, 5:], c[:, :5]], dim=1)                               # azul <-> rojo
    with torch.no_grad():
        base = model(x, champions=c)
        assert torch.allclose(base, model(x, champions=shuffled), atol=1e-5)
        assert not torch.allclose(base, model(x, champions=swapped))
    with pytest.raises(ValueError):
        model(x)                                   # un modelo con draft exige los campeones


def test_forward_passes_champions_only_to_draft_models():
    x, mask = torch.randn(2, 10, 5), torch.ones(2, 10, dtype=torch.bool)
    b = {"X": x, "mask": mask, "champions": champs(2)}
    for use in (False, True):
        model = build_model({**CFG, "champions": use}, 5).eval()
        assert forward(model, b).shape == (2, 10)
    gru = build_model({"type": "gru", "hidden": 8, "layers": 1, "dropout": 0.0}, 5).eval()
    assert forward(gru, b).shape == (2, 10)         # la GRU ignora los campeones


@pytest.mark.parametrize("use_champs", [False, True])
def test_transformer_training_learns(tmp_path, use_champs):
    fake = tmp_path / "fake.npz"
    np.savez_compressed(fake, **make(1500, seed=1))
    cfg = load_config(ROOT / "configs" / "transformer_base.yaml",
                      ["model.hidden=32", "model.layers=2", "train.max_epochs=6", "train.patience=3",
                       "train.batch_size=64", f"model.champions={str(use_champs).lower()}"])
    cfg["data"]["path"] = str(fake)
    summary = run(cfg, tmp_path / "m6", verbose=False)
    assert summary["val_auc"] > 0.75, summary
    assert (tmp_path / "m6" / "predictions.npz").exists()


# ------------------------------------------------------------------ M4 + draft (GRU con campeones)
GRU = {"type": "gru", "hidden": 16, "layers": 2, "dropout": 0.0}


def test_gru_with_draft_is_causal_and_team_aware():
    torch.manual_seed(0)
    model = build_model({**GRU, "champions": True}, 7).eval()
    c = champs()
    x = torch.randn(4, 40, 7)
    x2 = x.clone()
    x2[:, 20:] = torch.randn(4, 20, 7) * 10
    with torch.no_grad():
        a, b = model(x, champions=c), model(x2, champions=c)
        assert torch.allclose(a[:, :20], b[:, :20], atol=1e-6), "la GRU usa minutos futuros"
        shuffled = torch.cat([c[:, [3, 1, 4, 0, 2]], c[:, 5:]], dim=1)
        assert torch.allclose(a, model(x, champions=shuffled), atol=1e-5)
        assert not torch.allclose(a, model(x, champions=torch.cat([c[:, 5:], c[:, :5]], dim=1)))


def test_gru_without_draft_keeps_the_m4_weights():
    """champions=False: mismos parámetros que siempre -> los pesos exportados de M4 siguen cargando."""
    import json
    sd = torch.load(ROOT / "models" / "m4_gru" / "seed_42.pt", map_location="cpu", weights_only=False) \
        if (ROOT / "models" / "m4_gru" / "seed_42.pt").exists() else None
    if sd is None:
        pytest.skip("sin models/m4_gru")
    model = build_model(sd["model_cfg"], len(sd["feature_names"]))
    model.load_state_dict(sd["state_dict"])          # strict: falla si cambió algún parámetro
    assert not model.uses_champions
    assert json.loads((ROOT / "models" / "m4_gru" / "manifest.json").read_text(encoding="utf-8"))["name"] == "M4"


def test_gru_with_draft_training_learns(tmp_path):
    fake = tmp_path / "fake.npz"
    np.savez_compressed(fake, **make(1500, seed=1))
    cfg = load_config(ROOT / "configs" / "gru_base.yaml",
                      ["model.hidden=32", "train.max_epochs=6", "train.patience=3", "train.batch_size=64",
                       "model.champions=true"])
    cfg["data"]["path"] = str(fake)
    summary = run(cfg, tmp_path / "m4d", verbose=False)
    assert summary["val_auc"] > 0.75, summary
