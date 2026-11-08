"""
Pruebas de B7: TCN causal (M5).

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
from src.train import load_config, run  # noqa: E402
from tests.make_fake_sequences import make  # noqa: E402


def test_tcn_registered():
    assert "tcn" in MODELS


@pytest.mark.parametrize("layers", [2, 4])
def test_tcn_is_causal(layers):
    torch.manual_seed(0)
    model = build_model({"type": "tcn", "hidden": 16, "layers": layers, "dropout": 0.0}, 7).eval()
    x = torch.randn(4, 60, 7)
    x2 = x.clone()
    x2[:, 30:] = torch.randn(4, 30, 7) * 10           # cambia solo el "futuro"
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert a.shape == (4, 60)
    assert torch.allclose(a[:, :30], b[:, :30], atol=1e-6), "la TCN usa minutos futuros"
    assert not torch.allclose(a[:, 30:], b[:, 30:])


@pytest.mark.parametrize("layers", [2, 3])
def test_tcn_receptive_field(layers):
    """El minuto t depende de los minutos t-rf+1..t y de nada más antiguo."""
    torch.manual_seed(0)
    model = build_model({"type": "tcn", "hidden": 16, "layers": layers, "dropout": 0.0}, 5).eval()
    rf = model.receptive_field
    assert rf == 1 + 2 * 2 * (2 ** layers - 1)
    x = torch.randn(1, 60, 5)
    t = 55
    for past, should_change in [(t - rf, False), (t - rf + 1, True)]:
        x2 = x.clone()
        x2[:, past] += 5.0
        with torch.no_grad():
            changed = not torch.allclose(model(x)[0, t], model(x2)[0, t], atol=1e-6)
        assert changed == should_change, (past, rf)


def test_tcn_padding_right_does_not_matter():
    """El relleno al final (partidas más cortas en el batch) no cambia los minutos válidos."""
    torch.manual_seed(0)
    model = build_model({"type": "tcn", "hidden": 16, "layers": 3, "dropout": 0.0}, 5).eval()
    x = torch.randn(1, 40, 5)
    with torch.no_grad():
        short = model(x[:, :25])
        long = model(torch.cat([x[:, :25], torch.zeros(1, 15, 5)], dim=1))
    assert torch.allclose(short, long[:, :25], atol=1e-6)


def test_tcn_training_learns(tmp_path):
    fake = tmp_path / "fake.npz"
    np.savez_compressed(fake, **make(1500, seed=1))
    cfg = load_config(ROOT / "configs" / "tcn_base.yaml",
                      ["model.hidden=32", "model.layers=3", "train.max_epochs=6",
                       "train.patience=3", "train.batch_size=64"])
    cfg["data"]["path"] = str(fake)
    summary = run(cfg, tmp_path / "tcn", verbose=False)
    assert summary["val_auc"] > 0.75, summary
    assert (tmp_path / "tcn" / "predictions.npz").exists()
