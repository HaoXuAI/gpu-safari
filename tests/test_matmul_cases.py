import json
from pathlib import Path

import pytest

from benchmarks.cases import load_matmul_cases


ROOT = Path(__file__).parents[1]
CASES = ROOT / "experiments" / "cuda" / "matmul" / "cases.json"


def test_checked_in_matmul_cases_cover_learning_boundaries():
    cases = load_matmul_cases(CASES)
    assert all(case["precision"] == "float32" for case in cases)
    dimensions = {(case["m"], case["n"], case["k"]) for case in cases}
    assert (0, 17, 9) in dimensions
    assert (1, 1, 1) in dimensions
    assert (7, 5, 3) in dimensions
    assert (15, 15, 15) in dimensions
    assert (31, 17, 29) in dimensions
    assert (512, 512, 512) in dimensions
    assert (256, 1024, 128) in dimensions


@pytest.mark.parametrize("field,value", [("m", -1), ("warmup", -1), ("iterations", 0)])
def test_matmul_cases_reject_invalid_counts(tmp_path, field, value):
    case = {
        "name": "invalid",
        "m": 1,
        "n": 1,
        "k": 1,
        "precision": "float32",
        "warmup": 1,
        "iterations": 1,
    }
    case[field] = value
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case]))
    with pytest.raises(ValueError, match=field):
        load_matmul_cases(path)


def test_matmul_cases_reject_unknown_fields(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{
        "name": "invalid", "m": 1, "n": 1, "k": 1,
        "precision": "float32", "warmup": 1, "iterations": 1,
        "token": "not-a-token"
    }]))
    with pytest.raises(ValueError, match="fields"):
        load_matmul_cases(path)


@pytest.mark.parametrize("precision", ["float16", "", 32])
def test_matmul_cases_reject_non_float32_precision(tmp_path, precision):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{
        "name": "invalid", "m": 1, "n": 1, "k": 1,
        "precision": precision, "warmup": 1, "iterations": 1,
    }]))
    with pytest.raises(ValueError, match="precision"):
        load_matmul_cases(path)
