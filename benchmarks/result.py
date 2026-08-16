import csv
import io
import json
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).parent / "schema" / "result-v1.schema.json"


def load_schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text())


def validate_result(document: dict[str, object]) -> None:
    Draft202012Validator(load_schema()).validate(document)


def parse_reduction_csv(output: str) -> list[dict[str, object]]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or lines[-1] != "ALL CORRECTNESS CHECKS PASSED":
        raise ValueError("reduction output did not pass correctness checks")
    rows = csv.DictReader(io.StringIO("\n".join(lines[:-1])))
    parsed = []
    for row in rows:
        parsed.append({
            "method": row["method"],
            "size": int(row["size"]),
            "result": float(row["result"]),
            "expected": float(row["expected"]),
            "avg_ms": float(row["avg_ms"]),
            "effective_gb_s": float(row["effective_gb_s"]),
            "passed": row["status"] == "PASS",
        })
    if not parsed or not all(row["passed"] for row in parsed):
        raise ValueError("reduction output did not pass correctness checks")
    return parsed


def build_reduction_result(
    rows: list[dict[str, object]],
    *,
    revision: str,
    gpu_model: str,
    cuda_version: str,
    compiler_version: str,
    provider: str,
) -> dict[str, object]:
    candidates = [row for row in rows if row["method"] == "cub_device_reduce"]
    if not candidates:
        raise ValueError("reduction output is missing cub_device_reduce")
    selected = max(candidates, key=lambda row: int(row["size"]))
    document = {
        "schema_version": "1.0.0",
        "experiment": {"id": "cuda/reduction", "implementation": "cub_device_reduce"},
        "source": {"revision": revision},
        "accelerator": {"vendor": "NVIDIA", "model": gpu_model, "count": 1},
        "platform": {"provider": provider},
        "software": {"cuda": cuda_version, "compiler": compiler_version},
        "workload": {"elements": selected["size"], "dtype": "float32"},
        "correctness": {"passed": all(bool(row["passed"]) for row in rows)},
        "measurements": [
            {"name": "latency", "value": selected["avg_ms"], "unit": "ms"},
            {"name": "effective_bandwidth", "value": selected["effective_gb_s"], "unit": "GB/s"},
        ],
    }
    validate_result(document)
    return document


def parse_matmul_csv(output: str) -> list[dict[str, object]]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or lines[-1] != "ALL MATMUL CORRECTNESS CHECKS PASSED":
        raise ValueError("matmul output did not pass correctness checks")

    fields = [
        "implementation",
        "m",
        "n",
        "k",
        "avg_ms",
        "tflops",
        "max_abs_error",
        "max_rel_error",
        "status",
    ]
    rows = csv.DictReader(io.StringIO("\n".join(lines[:-1])))
    if rows.fieldnames != fields:
        raise ValueError("matmul output did not pass correctness checks")

    parsed = []
    for row in rows:
        if set(row) != set(fields) or any(value is None for value in row.values()):
            raise ValueError("matmul output did not pass correctness checks")
        parsed.append({
            "implementation": row["implementation"],
            "m": int(row["m"]),
            "n": int(row["n"]),
            "k": int(row["k"]),
            "avg_ms": float(row["avg_ms"]),
            "tflops": float(row["tflops"]),
            "max_abs_error": float(row["max_abs_error"]),
            "max_rel_error": float(row["max_rel_error"]),
            "passed": row["status"] == "PASS",
        })
    if not parsed or not all(row["passed"] for row in parsed):
        raise ValueError("matmul output did not pass correctness checks")
    return parsed


def build_matmul_result(
    rows: list[dict[str, object]],
    *,
    revision: str,
    gpu_model: str,
    cuda_version: str,
    compiler_version: str,
    provider: str,
) -> dict[str, object]:
    candidates = [row for row in rows if row["implementation"] == "cublas"]
    if not candidates:
        raise ValueError("matmul output is missing cublas")
    selected = max(candidates, key=lambda row: int(row["m"]) * int(row["n"]) * int(row["k"]))
    document = {
        "schema_version": "1.0.0",
        "experiment": {"id": "cuda/matmul", "implementation": "cublas"},
        "source": {"revision": revision},
        "accelerator": {"vendor": "NVIDIA", "model": gpu_model, "count": 1},
        "platform": {"provider": provider},
        "software": {"cuda": cuda_version, "compiler": compiler_version},
        "workload": {
            "m": selected["m"], "n": selected["n"], "k": selected["k"],
            "dtype": "float32",
        },
        "correctness": {"passed": all(bool(row["passed"]) for row in rows)},
        "measurements": [
            {"name": "latency", "value": selected["avg_ms"], "unit": "ms"},
            {"name": "throughput", "value": selected["tflops"], "unit": "TFLOP/s"},
            {"name": "max_abs_error", "value": selected["max_abs_error"], "unit": "absolute"},
            {"name": "max_rel_error", "value": selected["max_rel_error"], "unit": "relative"},
        ],
    }
    validate_result(document)
    return document
