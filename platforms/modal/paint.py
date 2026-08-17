"""Cost-bounded Triton paint-pixels lesson runner for Modal."""

import json

import modal


app = modal.App("gpu-safari-paint-pixels")
image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
).pip_install("torch==2.7.1", "triton==3.3.1")


@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    scaledown_window=2,
)
def run_paint(group_size: int = 10) -> dict[str, object]:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def paint_kernel(out, n_pixels: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_pixels
        tl.store(out + offsets, 0.5, mask=mask)

    pixels = 64
    launch_size = triton.next_power_of_2(group_size)
    output = torch.empty((pixels,), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(pixels, launch_size),)
    for _ in range(5):
        paint_kernel[grid](output, pixels, BLOCK_SIZE=launch_size)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        paint_kernel[grid](output, pixels, BLOCK_SIZE=launch_size)
    end.record()
    torch.cuda.synchronize()

    expected = torch.full_like(output, 0.5)
    max_abs_error = float((output - expected).abs().max().item())
    return {
        "schema_version": "1.0.0",
        "experiment": "paint-pixels",
        "provider": "modal-triton",
        "device": torch.cuda.get_device_name(0),
        "implementation": "triton",
        "workload": {"pixels": pixels, "dtype": "float32", "group_size": launch_size},
        "correctness": {"passed": max_abs_error <= 1e-6, "max_abs_error": max_abs_error},
        "measurements": [{"name": "latency", "value": start.elapsed_time(end) / 100, "unit": "ms"}],
        "output": {"checksum": float(output.sum().item())},
    }


@app.local_entrypoint()
def main(group_size: int = 10):
    print(json.dumps(run_paint.remote(group_size), sort_keys=True))
