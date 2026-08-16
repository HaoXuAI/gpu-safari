"""Cost-bounded CUDA smoke test for Modal.

Run with: modal run modal_cuda.py
"""

import subprocess
from pathlib import Path

import modal


app = modal.App("gpu-kernel-lab")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential")
    .add_local_file(
        Path(__file__).parent / "cuda" / "reduction.cu",
        "/opt/gpu-kernel-lab/reduction.cu",
        copy=True,
    )
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
def cuda_smoke_test() -> dict[str, str]:
    """Compile and run a minimal CUDA device query on one economical L4."""
    work = Path("/tmp/gpu-kernel-lab")
    work.mkdir(exist_ok=True)
    source = work / "deviceQuery.cu"
    binary = work / "deviceQuery"
    source.write_text(DEVICE_QUERY_SOURCE)

    def run(*command: str) -> str:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()

    return {
        "nvidia-smi": run("nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"),
        "nvcc --version": run("nvcc", "--version"),
        "compile": run("nvcc", "-O2", str(source), "-o", str(binary)),
        "deviceQuery": run(str(binary)),
        "reduction benchmark": run(
            "nvcc",
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-arch=sm_89",
            "/opt/gpu-kernel-lab/reduction.cu",
            "-o",
            str(work / "reduction"),
        ) + run(str(work / "reduction")),
    }


@app.local_entrypoint()
def main():
    for name, output in cuda_smoke_test.remote().items():
        print(f"\n[{name}]\n{output}")
