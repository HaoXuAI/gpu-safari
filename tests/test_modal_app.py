from pathlib import Path


APP = Path(__file__).parents[1] / "platforms" / "modal" / "reduction.py"
REDUCTION = (
    Path(__file__).parents[1]
    / "experiments"
    / "cuda"
    / "reduction"
    / "src"
    / "reduction.cu"
)


def test_modal_cuda_app_is_cost_bounded():
    source = APP.read_text()
    assert 'gpu="L4"' in source
    assert "timeout=600" in source
    assert "scaledown_window=2" in source
    assert "min_containers" not in source


def test_modal_cuda_app_checks_the_gpu_stack():
    source = APP.read_text()
    for command in ("nvidia-smi", "nvcc", "deviceQuery"):
        assert command in source


def test_modal_cuda_app_uses_result_contract():
    source = APP.read_text()
    assert "parse_reduction_csv" in source
    assert "build_reduction_result" in source
    assert 'modal.App("gpu-safari-cuda-reduction")' in source


def test_reduction_lab_contains_four_implementations():
    source = REDUCTION.read_text()
    for implementation in (
        "reduce_naive_atomic",
        "reduce_shared",
        "reduce_warp_shuffle",
        "cub::DeviceReduce::Sum",
    ):
        assert implementation in source


def test_reduction_lab_covers_edge_cases_and_cuda_timing():
    source = REDUCTION.read_text()
    for size in ("0", "1", "17", "255", "256", "1000", "1 << 20"):
        assert size in source
    assert "cudaEventElapsedTime" in source
    assert "std::mt19937 rng(42)" in source
