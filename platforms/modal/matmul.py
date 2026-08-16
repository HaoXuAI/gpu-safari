"""Cost-bounded CUDA matrix multiplication experiment for Modal.

Run with: modal run platforms/modal/matmul.py
"""

import json
import subprocess
from pathlib import Path

import modal


if modal.is_local():
    ROOT = Path(__file__).parents[2]
    MATMUL_SOURCE_DIR = ROOT / "experiments" / "cuda" / "matmul" / "src"
    CASES_SOURCE = ROOT / "experiments" / "cuda" / "matmul" / "cases.json"
    BENCHMARKS_SOURCE = ROOT / "benchmarks"
else:
    ROOT = Path("/opt/gpu-safari")
    MATMUL_SOURCE_DIR = ROOT
    CASES_SOURCE = ROOT / "cases.json"
    BENCHMARKS_SOURCE = ROOT / "benchmarks"
app = modal.App("gpu-safari-cuda-matmul")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "nlohmann-json3-dev")
    .pip_install("jsonschema>=4.23,<5")
    .add_local_file(
        MATMUL_SOURCE_DIR / "matmul.cu",
        "/opt/gpu-safari/matmul.cu",
        copy=True,
    )
    .add_local_file(
        MATMUL_SOURCE_DIR / "kernels.cuh",
        "/opt/gpu-safari/kernels.cuh",
        copy=True,
    )
    .add_local_file(
        MATMUL_SOURCE_DIR / "reference.cuh",
        "/opt/gpu-safari/reference.cuh",
        copy=True,
    )
    .add_local_file(CASES_SOURCE, "/opt/gpu-safari/cases.json", copy=True)
    .add_local_dir(BENCHMARKS_SOURCE, "/opt/gpu-safari/benchmarks", copy=True)
    .env({"PYTHONPATH": "/opt/gpu-safari"})
)


@app.function(
    image=cuda_image,
    gpu="L4",
    timeout=600,
    scaledown_window=2,
)
def run_matmul(revision: str) -> dict[str, object]:
    """Compile, benchmark, and return a validated matmul result document."""
    from benchmarks.result import build_matmul_result, parse_matmul_csv

    work = Path("/tmp/gpu-safari")
    work.mkdir(exist_ok=True)

    def run(*command: str) -> str:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()

    gpu_model = run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    cuda_version = run("nvcc", "--version")
    compiler_version = cuda_version.splitlines()[-1]
    binary = work / "matmul"
    run(
        "nvcc", "-O3", "--use_fast_math", "-std=c++17", "-arch=sm_89",
        "/opt/gpu-safari/matmul.cu", "-lcublas", "-o", str(binary),
    )
    output = run(str(binary), "/opt/gpu-safari/cases.json")
    print(output)
    return build_matmul_result(
        parse_matmul_csv(output),
        revision=revision,
        gpu_model=gpu_model,
        cuda_version=cuda_version,
        compiler_version=compiler_version,
        provider="Modal",
    )


@app.local_entrypoint()
def main():
    revision = subprocess.run(
        ("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    print(json.dumps(run_matmul.remote(revision), indent=2, sort_keys=True))
