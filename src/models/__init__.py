"""
Modelos de deep learning. Todos reciben X [B, T, F] y devuelven logits [B, T]:
una predicción de P(gana azul) por minuto.

    from src.models import build_model
    model = build_model({"type": "gru", "hidden": 128, "layers": 1, "dropout": 0.1}, n_features)
"""
from .gru import GRUModel
from .mlp import MLPModel

MODELS = {"gru": GRUModel, "mlp": MLPModel}


def build_model(cfg: dict, n_features: int):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in MODELS:
        raise ValueError(f"Modelo desconocido '{kind}'. Opciones: {list(MODELS)}")
    return MODELS[kind](n_features=n_features, **cfg)


def register(name: str, cls) -> None:
    """Para que B7 agregue TCN / Transformer sin tocar este archivo: register('tcn', TCNModel)."""
    MODELS[name] = cls


# B7: registrar aquí los modelos extra, para que train.py los encuentre sin importarlos aparte
from .tcn import TCNModel  # noqa: E402
register("tcn", TCNModel)
from .transformer import TransformerModel  # noqa: E402
register("transformer", TransformerModel)
