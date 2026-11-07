"""
Genera un features/sequences_fake.npz con EXACTAMENTE el formato de docs/contrato_features.md,
para probar datasets/modelos/entrenamiento sin esperar a B3.

Los datos son sintéticos pero tienen señal secuencial: cada partida tiene una "ventaja
oculta" s que hace crecer la diferencia de oro con una pendiente; el resultado depende de s
y del oro final. Un modelo que ve la historia (GRU) puede estimar s mejor que uno que solo
ve el minuto actual.

Uso:  python tests/make_fake_sequences.py [--n 3000] [--out features/sequences_fake.npz]
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T = 60
FEATURES = [  # (nombre, grupo)
    ("minute", "contexto"),
    ("gold_diff", "economia"),
    ("xp_diff", "economia"),
    ("kills_diff", "combate"),
    ("towers_diff", "objetivos"),
    ("dragons_diff", "objetivos"),
    ("wards_placed_diff", "vision"),
    ("gold_diff_top", "rol"),
    ("gold_diff_d1", "tendencia"),
    ("gold_diff_d3", "tendencia"),
]


def make(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(16, 51, size=n)
    s = rng.normal(0, 1, size=n)                              # ventaja oculta
    steps = rng.normal(180 * s[:, None], 450, size=(n, T))    # oro por minuto
    steps[:, 0] = 0
    gold = np.cumsum(steps, axis=1)
    xp = gold * 0.8 + rng.normal(0, 300, size=(n, T))
    kills = np.cumsum(rng.poisson(0.35, (n, T)) - rng.poisson(0.35, (n, T))
                      + (rng.random((n, T)) < 0.05 * np.abs(s)[:, None]) * np.sign(s)[:, None],
                      axis=1)
    towers = np.floor(np.clip(gold / 2500, -11, 11))
    dragons = np.floor(np.clip(gold / 4000, -4, 4))
    wards = np.cumsum(rng.normal(0.2 * s[:, None], 1.5, (n, T)), axis=1)
    top = gold * 0.2 + rng.normal(0, 400, (n, T))
    minute = np.broadcast_to(np.arange(T), (n, T)).astype(float)
    d1 = np.diff(gold, axis=1, prepend=0)
    d3 = gold - np.concatenate([np.zeros((n, 3)), gold[:, :-3]], axis=1)

    raw = np.stack([minute, gold, xp, kills, towers, dragons, wards, top, d1, d3], axis=-1)
    mask = np.arange(T)[None, :] < lengths[:, None]
    final_gold = gold[np.arange(n), lengths - 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.3 * s + final_gold / 6000)))).astype(np.int8)

    mean = raw[mask].mean(0)
    std = raw[mask].std(0) + 1e-6
    X = np.where(mask[..., None], np.clip((raw - mean) / std, -5, 5), 0).astype(np.float32)

    split = rng.choice(["train", "val", "test"], size=n, p=[0.7, 0.15, 0.15])
    platform = rng.choice(["la1", "na1", "euw1", "kr"], size=n)
    tier = rng.choice(["SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"], size=n)
    return {
        "X": X, "mask": mask, "lengths": lengths.astype(np.int16), "y": y,
        "match_id": np.array([f"{p.upper()}_{i:07d}" for i, p in enumerate(platform)]),
        "split": split, "platform": platform, "tier": tier,
        "patch": rng.choice(["16.20", "16.21", "16.22"], size=n),
        "afk": rng.random(n) < 0.03,
        "feature_names": np.array([f for f, _ in FEATURES]),
        "feature_groups": np.array([g for _, g in FEATURES]),
        "champions": rng.integers(1, 900, size=(n, 10)).astype(np.int16),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "features" / "sequences_fake.npz"))
    args = parser.parse_args()
    arrays = make(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"Guardado {out} ({args.n} partidas, {arrays['X'].shape[2]} features)")


if __name__ == "__main__":
    main()
