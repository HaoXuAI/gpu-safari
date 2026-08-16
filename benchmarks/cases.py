import json
from pathlib import Path


FIELDS = {"name", "m", "n", "k", "warmup", "iterations"}


def load_matmul_cases(path: Path) -> list[dict[str, int | str]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("matmul cases must be a non-empty list")
    cases = []
    for index, case in enumerate(raw):
        if not isinstance(case, dict) or set(case) != FIELDS:
            raise ValueError(f"case {index} fields must equal {sorted(FIELDS)}")
        if not isinstance(case["name"], str) or not case["name"]:
            raise ValueError(f"case {index} name must be non-empty")
        for field in ("m", "n", "k", "warmup", "iterations"):
            if type(case[field]) is not int:
                raise ValueError(f"case {index} {field} must be an integer")
        for field in ("m", "n", "k", "warmup"):
            if case[field] < 0:
                raise ValueError(f"case {index} {field} must be nonnegative")
        if case["iterations"] < 1:
            raise ValueError(f"case {index} iterations must be positive")
        cases.append(case)
    return cases
