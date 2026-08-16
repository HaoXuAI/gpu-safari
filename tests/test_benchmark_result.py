import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from benchmarks.result import (
    build_reduction_result,
    load_schema,
    parse_reduction_csv,
    validate_result,
)


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "benchmarks" / "examples" / "cuda-reduction-l4.json"

VALID_OUTPUT = """method,size,result,expected,avg_ms,effective_gb_s,status
cub_device_reduce,1048576,11.5,11.5,0.005704,735.0,PASS
ALL CORRECTNESS CHECKS PASSED
"""


def test_schema_has_stable_identity():
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://gpu-safari.dev/schemas/result-v1.schema.json"


def test_checked_in_example_is_valid():
    validate_result(json.loads(EXAMPLE.read_text()))


def test_correctness_is_required():
    document = json.loads(EXAMPLE.read_text())
    del document["correctness"]
    with pytest.raises(ValidationError):
        validate_result(document)


def test_credentials_are_rejected_as_unknown_properties():
    document = json.loads(EXAMPLE.read_text())
    document["token"] = "not-a-real-token"
    with pytest.raises(ValidationError):
        validate_result(document)


def test_parse_reduction_csv_returns_typed_rows():
    assert parse_reduction_csv(VALID_OUTPUT) == [{
        "method": "cub_device_reduce",
        "size": 1048576,
        "result": 11.5,
        "expected": 11.5,
        "avg_ms": 0.005704,
        "effective_gb_s": 735.0,
        "passed": True,
    }]


def test_parse_reduction_csv_rejects_failed_run():
    failed = VALID_OUTPUT.replace("ALL CORRECTNESS CHECKS PASSED", "CORRECTNESS CHECK FAILED")
    with pytest.raises(ValueError, match="did not pass correctness checks"):
        parse_reduction_csv(failed)


def test_build_reduction_result_selects_largest_cub_workload():
    document = build_reduction_result(
        parse_reduction_csv(VALID_OUTPUT),
        revision="f932b741aee26b22c92111f987265e5132e5e1d5",
        gpu_model="NVIDIA L4",
        cuda_version="12.8.1",
        compiler_version="nvcc 12.8",
        provider="Modal",
    )
    validate_result(document)
    assert document["experiment"] == {
        "id": "cuda/reduction",
        "implementation": "cub_device_reduce",
    }
    assert document["measurements"] == [
        {"name": "latency", "value": 0.005704, "unit": "ms"},
        {"name": "effective_bandwidth", "value": 735.0, "unit": "GB/s"},
    ]
