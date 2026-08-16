"""Cost-bounded CUDA reduction experiment for Modal.

Run with: modal run platforms/modal/reduction.py
"""

import json
import subprocess
from pathlib import Path

import modal


if modal.is_local():
    ROOT = Path(__file__).parents[2]
    CUDA_SOURCE = ROOT / "experiments" / "cuda" / "reduction" / "src" / "reduction.cu"
    BENCHMARKS_SOURCE = ROOT / "benchmarks"
else:
    ROOT = Path("/opt/gpu-safari")
    CUDA_SOURCE = ROOT / "reduction.cu"
    BENCHMARKS_SOURCE = ROOT / "benchmarks"
app = modal.App("gpu-safari-cuda-reduction")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential")
    .pip_install("jsonschema>=4.23,<5")
    .add_local_file(
        CUDA_SOURCE,
        "/opt/gpu-safari/reduction.cu",
        copy=True,
    )
    .add_local_dir(BENCHMARKS_SOURCE, "/opt/gpu-safari/benchmarks", copy=True)
    .env({"PYTHONPATH": "/opt/gpu-safari"})
)


DEVICE_QUERY_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstdio>

int main() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count < 1) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(status));
    return 1;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 2;
  std::printf("deviceQuery: %s, compute capability %d.%d, %.1f GiB\n",
              prop.name, prop.major, prop.minor,
              prop.totalGlobalMem / 1073741824.0);
  return 0;
}
"""


@app.function(
    image=cuda_image,
    gpu="L4",
    timeout=600,
    scaledown_window=2,
)
def run_reduction(revision: str) -> dict[str, object]:
    """Compile, check, benchmark, and return a validated result document."""
    from benchmarks.result import build_reduction_result, parse_reduction_csv

    work = Path("/tmp/gpu-safari")
    work.mkdir(exist_ok=True)
    source = work / "deviceQuery.cu"
    binary = work / "deviceQuery"
    source.write_text(DEVICE_QUERY_SOURCE)

    def run(*command: str) -> str:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()

    gpu_model = run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    cuda_version = run("nvcc", "--version")
    compiler_version = cuda_version.splitlines()[-1]
    run("nvcc", "-O2", str(source), "-o", str(binary))
    print(run(str(binary)))  # deviceQuery
    reduction = work / "reduction"
    run(
        "nvcc", "-O3", "--use_fast_math", "-std=c++17", "-arch=sm_89",
        "/opt/gpu-safari/reduction.cu", "-o", str(reduction),
    )
    output = run(str(reduction))
    print(output)
    return build_reduction_result(
        parse_reduction_csv(output),
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
    print(json.dumps(run_reduction.remote(revision), indent=2, sort_keys=True))
