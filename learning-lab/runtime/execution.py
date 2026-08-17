"""Capability detection and explicit provider dispatch."""

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable

from runtime.contracts import validate_execution_result


def detect_capabilities(
    *,
    system: str | None = None,
    machine: str | None = None,
    module_available: Callable[[str], bool] | None = None,
    command_available: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    system = system or platform.system()
    machine = machine or platform.machine()
    module_available = module_available or (lambda name: importlib.util.find_spec(name) is not None)
    command_available = command_available or (lambda name: shutil.which(name) is not None)

    apple_hardware = system == "Darwin" and machine == "arm64"
    mlx_available = apple_hardware and module_available("mlx")
    modal_available = command_available("modal")
    return {
        "schema_version": "1.0.0",
        "providers": [
            {
                "id": "apple-mlx",
                "label": "Apple GPU · MLX",
                "available": mlx_available,
                "kind": "local",
                "requires_confirmation": False,
                "reason": None if mlx_available else (
                    "Install MLX with `pip install mlx`. Apple silicon and macOS 14+ are required."
                    if apple_hardware else "Apple MLX requires an Apple silicon Mac."
                ),
            },
            {
                "id": "modal-triton",
                "label": "NVIDIA GPU · Modal",
                "available": modal_available,
                "kind": "cloud",
                "requires_confirmation": True,
                "reason": None if modal_available else "Install and authenticate the Modal CLI first.",
            },
        ],
    }


def run_provider(
    provider: str,
    *,
    confirmed: bool = False,
    group_size: int = 10,
    executor: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    if provider == "modal-triton" and not confirmed:
        raise PermissionError("billable Modal GPU execution requires explicit confirmation")
    if provider == "apple-mlx":
        from runtime.mlx_runner import run_paint

        return validate_execution_result(run_paint(group_size=group_size))
    if provider == "modal-triton":
        launch_size = 1 << (group_size - 1).bit_length()
        completed = executor(
            ["modal", "run", "--quiet", "platforms/modal/paint.py", "--group-size", str(launch_size)],
            check=False, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            print(completed.stderr.strip(), file=sys.stderr)
            raise RuntimeError("Modal execution failed; check the companion-server terminal.")
        for line in reversed(completed.stdout.splitlines()):
            try:
                return validate_execution_result(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
        raise RuntimeError("Modal returned no valid learning-lab result.")
    raise ValueError(f"unknown execution provider: {provider}")
