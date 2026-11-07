"""
Pruebas de B6 (datasets, modelos, entrenamiento) con datos sintéticos.

    python -m pytest tests -q

Lo más importante que verifican:
  - el lector rechaza archivos que no cumplen el contrato;
  - la GRU es CAUSAL: cambiar el futuro no cambia la predicción del pasado (sin fuga);
  - el padding no afecta la pérdida;
  - el entrenamiento aprende (AUC de val alto en datos con señal), guarda todo y se puede reanudar.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from make_fake_sequences import make  # noqa: E402
from src.datasets import ContractError, load_sequences, validate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import load_config, masked_bce, run  # noqa: E402


@pytest.fixture(scope="module")
def fake_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "seq.npz"
    np.savez_compressed(path, **make(1500, seed=1))
    return path


# ------------------------------------------------------------------ contrato
def test_contract_ok(fake_path):
    data = load_sequences(fake_path)
    assert data.X.dtype == np.float32 and data.X.shape[1] == 60
    train = data.subset(split="train", exclude_afk=True)
    assert set(train.split) == {"train"} and not train.afk.any()
    small = data.subset(split="train", max_matches=100, seed=3)
    assert len(small) == 100
    assert np.array_equal(small.match_id, data.subset(split="train", max_matches=100, seed=3).match_id)
    no_trend = data.select_features(drop_groups=["tendencia"])
    assert "tendencia" not in set(no_trend.feature_groups)
    assert no_trend.X.shape[2] == data.X.shape[2] - 2


@pytest.mark.parametrize("breaker", [
    lambda a: a.pop("y"),
    lambda a: a.__setitem__("lengths", a["lengths"] + 1),
    lambda a: a.__setitem__("feature_groups", np.array(["otro"] * len(a["feature_groups"]))),
    lambda a: a.__setitem__("X", np.where(a["mask"][..., None], np.nan, a["X"]).astype(np.float32)),
])
def test_contract_rejects_bad_files(breaker):
    arrays = make(50)
    breaker(arrays)
    with pytest.raises(ContractError):
        validate(arrays)


# ------------------------------------------------------------------ modelos
@pytest.mark.parametrize("kind", ["gru", "mlp"])
def test_models_are_causal(kind):
    torch.manual_seed(0)
    model = build_model({"type": kind, "hidden": 32, "layers": 2, "dropout": 0.0}, 7).eval()
    x = torch.randn(4, 30, 7)
    x2 = x.clone()
    x2[:, 15:] = torch.randn(4, 15, 7) * 10          # cambia solo el "futuro"
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert a.shape == (4, 30)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-6), "la predicción usa minutos futuros"
    assert not torch.allclose(a[:, 15:], b[:, 15:])


def test_padding_does_not_change_loss():
    logits = torch.randn(3, 10)
    mask = torch.tensor([[True] * 10, [True] * 6 + [False] * 4, [True] * 3 + [False] * 7])
    y = torch.tensor([1.0, 0.0, 1.0])
    logits2 = logits.clone()
    logits2[~mask] = 99.0
    assert torch.allclose(masked_bce(logits, y, mask), masked_bce(logits2, y, mask))
    assert masked_bce(logits, y, mask, smooth_lambda=1.0) >= masked_bce(logits, y, mask)


# ------------------------------------------------------------------ entrenamiento
@pytest.mark.parametrize("kind,drop", [("gru", ["tendencia"]), ("mlp", [])])
def test_training_learns_and_saves(fake_path, tmp_path, kind, drop):
    cfg = load_config(None, [f"model.type={kind}", "model.hidden=32", "train.max_epochs=6",
                             "train.patience=3", "train.batch_size=64"])
    cfg["data"]["path"] = str(fake_path)
    cfg["data"]["drop_groups"] = drop
    summary = run(cfg, tmp_path / kind, verbose=False)
    assert summary["val_auc"] > 0.75, summary
    assert summary["temperature"] > 0
    for f in ("config.yaml", "metrics.csv", "best.pt", "last.pt", "summary.json",
              "predictions.npz"):
        assert (tmp_path / kind / f).exists(), f
    with np.load(tmp_path / kind / "predictions.npz") as p:
        probs, split = p["probs"], p["split"]
        assert set(split) == {"val", "test"}
        valid = probs[~np.isnan(probs)]
        assert valid.min() >= 0 and valid.max() <= 1


def test_resume_continues(fake_path, tmp_path):
    cfg = load_config(None, ["model.hidden=16", "train.max_epochs=2", "train.patience=10"])
    cfg["data"]["path"] = str(fake_path)
    run(cfg, tmp_path / "r", verbose=False)
    cfg["train"]["max_epochs"] = 4
    summary = run(cfg, tmp_path / "r", resume=True, verbose=False)
    assert summary["epochs_run"] == 4
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["epochs_run"] == 4
