"""Pruebas de B8: detector de momentos y emparejamiento con anotaciones."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.moments import detect_moments, match_moments, moments_from_minutes  # noqa: E402


def curve(*values, pad=5):
    return np.array(list(values) + [np.nan] * pad, dtype=float)


def test_detects_jump_and_direction():
    p = curve(0.5, 0.5, 0.52, 0.51, 0.75, 0.76, 0.77, 0.4, 0.38)
    m = detect_moments(p, theta=0.15, windows=(1,), top_k=None)
    assert [x.minute for x in m] == [4, 7]
    assert m[0].team == "azul" and m[1].team == "rojo"
    assert m[0].start_minute == 3 and abs(m[0].delta - 0.24) < 1e-9


def test_window_two_catches_slow_swings():
    p = curve(0.5, 0.5, 0.58, 0.66, 0.67)          # +0.08 por minuto: con w=1 no pasa 0.12
    assert detect_moments(p, theta=0.12, windows=(1,)) == []
    m = detect_moments(p, theta=0.12, windows=(1, 2))
    assert len(m) == 1 and m[0].minute == 3 and m[0].start_minute == 1


def test_merge_and_top_k():
    # dos saltos juntos (min 4 y 5) = una sola pelea; luego otros tres saltos separados
    p = curve(.5, .5, .5, .5, .7, .9, .9, .9, .6, .6, .6, .3, .3, .3, .6)
    m = detect_moments(p, theta=0.15, windows=(1,), merge_gap=2, top_k=None)
    assert len(m) == 4 and m[0].minute in (4, 5)       # los saltos de los min 4 y 5 son UNA pelea
    assert m[0].start_minute == 3                      # el clip cubre toda la pelea (min 3 a 5)
    assert [x.minute for x in m[1:]] == [8, 11, 14]
    top2 = detect_moments(p, theta=0.15, windows=(1,), merge_gap=2, top_k=2)
    assert len(top2) == 2
    assert all(x.score >= 0.29 for x in top2)


def test_padding_and_no_moments():
    assert detect_moments(curve(0.5, 0.51, 0.52), theta=0.1) == []
    assert detect_moments(np.full(10, np.nan)) == []


def test_r0_postprocess():
    assert moments_from_minutes([10, 11, 25, 26, 40], merge_gap=2, top_k=None) == [10, 25, 40]
    assert moments_from_minutes([10, 11], scores=[1, 5], merge_gap=2) == [11]
    assert moments_from_minutes([5, 20, 30], top_k=2) == [5, 20]


def test_matching_one_to_one():
    r = match_moments([10, 12, 30], [11, 25], tol=2)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 2, 1)
    assert r["pairs"] == [(10, 11)] or r["pairs"] == [(12, 11)]
    r = match_moments([], [5, 10])
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 2)
