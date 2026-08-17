"""Real Apple GPU execution through an MLX custom Metal kernel."""

import math
import time

from runtime.contracts import validate_execution_result


METAL_SOURCE = """
    uint pixel = thread_position_in_grid.x;
    uint n_pixels = N_PIXELS;
    if (pixel < n_pixels) {
        out[pixel] = color[0];
    }
"""


def build_result(
    *,
    device: str,
    pixels: int,
    group_size: int,
    latency_ms: float,
    checksum: float,
    max_abs_error: float,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "experiment": "paint-pixels",
        "provider": "apple-mlx",
        "device": device,
        "implementation": "metal",
        "workload": {"pixels": pixels, "dtype": "float32", "group_size": group_size},
        "correctness": {"passed": max_abs_error <= 1e-6, "max_abs_error": max_abs_error},
        "measurements": [{"name": "latency", "value": latency_ms, "unit": "ms"}],
        "output": {"checksum": checksum},
    }


def run_paint(*, pixels: int = 64, group_size: int = 10, iterations: int = 100) -> dict[str, object]:
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal backend is not available")
    if pixels <= 0 or group_size <= 0 or iterations <= 0:
        raise ValueError("pixels, group_size, and iterations must be positive")

    launched_threads = math.ceil(pixels / group_size) * group_size
    kernel = mx.fast.metal_kernel(
        name="gpu_safari_paint",
        input_names=["color"],
        output_names=["out"],
        source=METAL_SOURCE,
    )
    color = mx.array([0.5], dtype=mx.float32)

    def launch():
        output = kernel(
            inputs=[color],
            template=[("N_PIXELS", pixels)],
            grid=(launched_threads, 1, 1),
            threadgroup=(group_size, 1, 1),
            output_shapes=[(pixels,)],
            output_dtypes=[mx.float32],
        )[0]
        mx.eval(output)
        return output

    output = launch()
    for _ in range(5):
        output = launch()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        output = launch()
    latency_ms = (time.perf_counter_ns() - started) / iterations / 1_000_000

    expected = mx.full((pixels,), 0.5, dtype=mx.float32)
    max_abs_error = float(mx.max(mx.abs(output - expected)).item())
    checksum = float(mx.sum(output).item())
    info = mx.device_info()
    device = str(info.get("device_name") or info.get("architecture") or "Apple GPU")
    return validate_execution_result(build_result(
        device=device,
        pixels=pixels,
        group_size=group_size,
        latency_ms=latency_ms,
        checksum=checksum,
        max_abs_error=max_abs_error,
    ))
