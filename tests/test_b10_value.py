"""B10: función de valor, elección de θ*, lectura de anotaciones y acuerdo entre anotadores."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.agreement import agreement  # noqa: E402
from src.value_model import (aggregate, ahorro_min, choose_theta, load_annotations,  # noqa: E402
                             load_params, valor_mes)

P = {"f_rev": 1.3, "t_setup": 2.0, "t_clip": 2.5, "t_miss": 8.0, "N_rev": 20,
     "C_h_usd": 20, "C_infra_usd_mes": 0, "C_pred_usd": 0}


def test_formula_matches_doc_example():
    # docs/supuestos_valor.md §5: TP 3.2, FP 1.4, FN 0.8, D 29.5 -> T_tool ≈ 19.9, ahorro ≈ 18.5
    ah = ahorro_min(29.5, 3.2, 1.4, 0.8, P)
    assert ah == pytest.approx(1.3 * 29.5 - (2 + 4.6 * 2.5 + 0.8 * 8))
    assert valor_mes(ah, P) == pytest.approx(20 * ah / 60 * 20)


def test_params_load():
    base, ranges = load_params()
    assert base["f_rev"] > 0 and "t_clip" in ranges and base["match_window_min"] == 2


def test_choose_theta_max_ahorro_tie_goes_to_higher_theta():
    ut = pd.DataFrame({
        "modelo": ["M"] * 3, "theta": [0.1, 0.2, 0.3], "match_id": ["x"] * 3,
        "tp": [2, 2, 2], "fp": [3, 1, 1], "fn": [0, 0, 0], "split": ["val"] * 3, "D": [30.0] * 3})
    agg = aggregate(ut, P)
    assert choose_theta(agg)["M"] == 0.3  # 0.2 y 0.3 empatan (menos FP que 0.1): gana el más alto


def test_load_annotations_none_and_min_conf(tmp_path):
    f = tmp_path / "B.csv"
    pd.DataFrame([
        {"anotador": "B", "match_id": "m1", "minuto": 10, "tipo": "pelea", "equipo": "azul",
         "confianza": 3, "nota": "", "sin_momentos": False, "segundos": 100, "guardado_utc": "t1"},
        {"anotador": "B", "match_id": "m1", "minuto": 20, "tipo": "pelea", "equipo": "rojo",
         "confianza": 1, "nota": "", "sin_momentos": False, "segundos": 100, "guardado_utc": "t1"},
        {"anotador": "B", "match_id": "m2", "minuto": None, "tipo": None, "equipo": None,
         "confianza": None, "nota": "", "sin_momentos": True, "segundos": 50, "guardado_utc": "t1"},
    ]).to_csv(f, index=False)
    u = load_annotations([f]).set_index("match_id")
    assert u.loc["m1", "minutos"] == [10, 20] and u.loc["m2", "minutos"] == []
    u2 = load_annotations([f], min_conf=2).set_index("match_id")
    assert u2.loc["m1", "minutos"] == [10]


def test_agreement_counts():
    def ann(rows):
        df = pd.DataFrame(rows, columns=["match_id", "minuto", "equipo", "tipo"])
        return df.assign(valido=True)
    a = ann([("m", 10, "azul", "pelea"), ("m", 25, "rojo", "objetivo")])
    b = ann([("m", 11, "azul", "pelea"), ("m", 30, "rojo", "pelea")])
    r = agreement(a, b, tol=2).iloc[0]
    assert r["coinciden"] == 1 and r["acuerdo_f1"] == pytest.approx(0.5)
    assert r["mismo_equipo"] == 1 and r["mismo_tipo"] == 1
    assert np.isclose(r["recall_B_vs_A"], 0.5)
