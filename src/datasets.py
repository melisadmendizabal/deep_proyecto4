"""
Carga y filtra features/sequences.npz según docs/contrato_features.md.

    from src.datasets import load_sequences
    data = load_sequences("features/sequences.npz")
    train = data.subset(split="train", exclude_afk=True)
    train = train.select_features(drop_groups=["tendencia"])

Los modelos secuenciales (GRU) y por minuto (MLP) usan el mismo formato [N, T, F]:
el MLP simplemente procesa cada minuto por separado.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

REQUIRED = {
    "X": (3, np.floating),
    "mask": (2, np.bool_),
    "lengths": (1, np.integer),
    "y": (1, np.integer),
    "match_id": (1, np.str_),
    "split": (1, np.str_),
    "platform": (1, np.str_),
    "tier": (1, np.str_),
    "patch": (1, np.str_),
    "afk": (1, np.bool_),
    "feature_names": (1, np.str_),
    "feature_groups": (1, np.str_),
}
VALID_GROUPS = {"economia", "rol", "combate", "objetivos", "vision", "tendencia", "contexto"}


class ContractError(ValueError):
    """El archivo no cumple docs/contrato_features.md."""


@dataclass
class SequenceData:
    X: np.ndarray            # [N, T, F] float32
    mask: np.ndarray         # [N, T] bool
    lengths: np.ndarray      # [N]
    y: np.ndarray            # [N]
    match_id: np.ndarray
    split: np.ndarray
    platform: np.ndarray
    tier: np.ndarray
    patch: np.ndarray
    afk: np.ndarray
    feature_names: np.ndarray  # [F]
    feature_groups: np.ndarray  # [F]
    champions: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    def _take(self, idx: np.ndarray) -> "SequenceData":
        per_match = {f.name: getattr(self, f.name)[idx]
                     for f in fields(self)
                     if f.name not in ("feature_names", "feature_groups")
                     and getattr(self, f.name) is not None}
        return replace(self, **per_match)

    def subset(self, split: str | list[str] | None = None, exclude_afk: bool = False,
               tiers: list[str] | None = None, platforms: list[str] | None = None,
               max_matches: int | None = None, seed: int = 0) -> "SequenceData":
        """Filtra partidas. max_matches toma una muestra aleatoria reproducible (curva de aprendizaje)."""
        keep = np.ones(len(self), dtype=bool)
        if split is not None:
            keep &= np.isin(self.split, [split] if isinstance(split, str) else split)
        if exclude_afk:
            keep &= ~self.afk
        if tiers:
            keep &= np.isin(self.tier, [t.upper() for t in tiers])
        if platforms:
            keep &= np.isin(self.platform, [p.lower() for p in platforms])
        idx = np.flatnonzero(keep)
        if max_matches is not None and max_matches < len(idx):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_matches, replace=False))
        return self._take(idx)

    def select_features(self, drop_groups=(), drop_names=()) -> "SequenceData":
        keep = ~np.isin(self.feature_groups, list(drop_groups)) & \
               ~np.isin(self.feature_names, list(drop_names))
        if not keep.any():
            raise ValueError("No queda ninguna feature después de filtrar")
        return replace(self, X=self.X[:, :, keep], feature_names=self.feature_names[keep],
                       feature_groups=self.feature_groups[keep])

    def to_tensors(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        return {
            "X": torch.as_tensor(self.X, dtype=torch.float32, device=device),
            "mask": torch.as_tensor(self.mask, dtype=torch.bool, device=device),
            "lengths": torch.as_tensor(self.lengths.astype(np.int64), device=device),
            "y": torch.as_tensor(self.y.astype(np.float32), device=device),
        }


def validate(arrays: dict[str, np.ndarray]) -> None:
    missing = [k for k in REQUIRED if k not in arrays]
    if missing:
        raise ContractError(f"Faltan claves en el .npz: {missing}")
    for key, (ndim, kind) in REQUIRED.items():
        a = arrays[key]
        if a.ndim != ndim:
            raise ContractError(f"'{key}' debe tener {ndim} dimensiones, tiene {a.ndim} {a.shape}")
        if not np.issubdtype(a.dtype, kind):
            raise ContractError(f"'{key}' debe ser {kind.__name__}, es {a.dtype}")
    n, t, f = arrays["X"].shape
    for key in REQUIRED:
        if key in ("X", "feature_names", "feature_groups"):
            continue
        if arrays[key].shape[0] != n:
            raise ContractError(f"'{key}' tiene {arrays[key].shape[0]} filas, X tiene {n}")
    if arrays["mask"].shape != (n, t):
        raise ContractError(f"'mask' debe ser {(n, t)}, es {arrays['mask'].shape}")
    if len(arrays["feature_names"]) != f or len(arrays["feature_groups"]) != f:
        raise ContractError("feature_names / feature_groups deben tener F elementos")
    if not np.array_equal(arrays["mask"].sum(1), arrays["lengths"]):
        raise ContractError("'lengths' no coincide con mask.sum(1)")
    if not (arrays["mask"][:, 0].all()):
        raise ContractError("Toda partida debe tener el minuto 0 válido")
    # la máscara debe ser un bloque contiguo desde el minuto 0
    expected = np.arange(t)[None, :] < arrays["lengths"][:, None]
    if not np.array_equal(arrays["mask"], expected):
        raise ContractError("'mask' debe ser True desde el minuto 0 hasta lengths-1, sin huecos")
    bad_groups = set(arrays["feature_groups"]) - VALID_GROUPS
    if bad_groups:
        raise ContractError(f"Grupos desconocidos: {bad_groups}. Válidos: {sorted(VALID_GROUPS)}")
    if "minute" not in set(arrays["feature_names"]):
        raise ContractError("Debe existir una feature llamada 'minute'")
    if not set(np.unique(arrays["y"])) <= {0, 1}:
        raise ContractError("'y' debe ser 0/1")
    if not set(np.unique(arrays["split"])) <= {"train", "val", "test"}:
        raise ContractError("'split' solo puede ser train/val/test")
    if not np.isfinite(arrays["X"]).all():
        raise ContractError("X tiene NaN o infinitos")
    if "champions" in arrays and arrays["champions"].shape != (n, 10):
        raise ContractError("'champions' debe ser [N, 10]")


def load_sequences(path: str | Path) -> SequenceData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Lo genera B3 (src/build_features.py); para probar sin datos "
            "reales: python tests/make_fake_sequences.py")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    validate(arrays)
    return SequenceData(
        X=arrays["X"].astype(np.float32, copy=False),
        mask=arrays["mask"], lengths=arrays["lengths"], y=arrays["y"],
        match_id=arrays["match_id"], split=arrays["split"], platform=arrays["platform"],
        tier=arrays["tier"], patch=arrays["patch"], afk=arrays["afk"],
        feature_names=arrays["feature_names"], feature_groups=arrays["feature_groups"],
        champions=arrays.get("champions"),
    )


def iterate_batches(tensors: dict[str, torch.Tensor], batch_size: int, shuffle: bool,
                    generator: torch.Generator | None = None):
    """Batches de partidas ya en el dispositivo, recortados al largo máximo del batch."""
    n = tensors["y"].shape[0]
    device = tensors["y"].device
    order = (torch.randperm(n, generator=generator).to(device) if shuffle
             else torch.arange(n, device=device))
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        t_max = int(tensors["lengths"][idx].max())
        yield {
            "X": tensors["X"][idx, :t_max],
            "mask": tensors["mask"][idx, :t_max],
            "y": tensors["y"][idx],
            "idx": idx,
        }
