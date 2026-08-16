import pytest

from benchmarks.result import build_matmul_result, parse_matmul_csv, validate_result


VALID = """implementation,m,n,k,avg_ms,tflops,max_abs_error,max_rel_error,status
naive,31,17,29,0.010000,0.003056,0.000001,0.000002,PASS
tiled,31,17,29,0.008000,0.003820,0.000001,0.000002,PASS
cublas,512,512,512,0.100000,2.684355,0.000001,0.000002,PASS
ALL MATMUL CORRECTNESS CHECKS PASSED
"""


def test_parse_matmul_csv_returns_typed_rows():
    rows = parse_matmul_csv(VALID)
    assert rows[0] == {
        "implementation": "naive", "m": 31, "n": 17, "k": 29,
        "avg_ms": 0.01, "tflops": 0.003056,
        "max_abs_error": 0.000001, "max_rel_error": 0.000002,
        "passed": True,
    }


@pytest.mark.parametrize("text", [
    VALID.replace("PASS", "FAIL", 1),
    VALID.replace("ALL MATMUL CORRECTNESS CHECKS PASSED", "MATMUL CORRECTNESS CHECK FAILED"),
])
def test_parse_matmul_csv_rejects_failed_correctness(text):
    with pytest.raises(ValueError, match="did not pass correctness checks"):
        parse_matmul_csv(text)


def test_build_matmul_result_selects_largest_cublas_case():
    document = build_matmul_result(
        parse_matmul_csv(VALID),
        revision="6496fe7e113f6ae12e564c3d46bd64647012be96",
        gpu_model="NVIDIA L4",
        cuda_version="12.8.1",
        compiler_version="nvcc 12.8",
        provider="Modal",
    )
    validate_result(document)
    assert document["experiment"] == {"id": "cuda/matmul", "implementation": "cublas"}
    assert document["workload"] == {"m": 512, "n": 512, "k": 512, "dtype": "float32"}
    assert document["measurements"] == [
        {"name": "latency", "value": 0.1, "unit": "ms"},
        {"name": "throughput", "value": 2.684355, "unit": "TFLOP/s"},
        {"name": "max_abs_error", "value": 0.000001, "unit": "absolute"},
        {"name": "max_rel_error", "value": 0.000002, "unit": "relative"},
    ]


def test_build_matmul_result_requires_cublas():
    rows = [row for row in parse_matmul_csv(VALID) if row["implementation"] != "cublas"]
    with pytest.raises(ValueError, match="missing cublas"):
        build_matmul_result(
            rows,
            revision="6496fe7e113f6ae12e564c3d46bd64647012be96",
            gpu_model="NVIDIA L4",
            cuda_version="12.8.1",
            compiler_version="nvcc 12.8",
            provider="Modal",
        )
