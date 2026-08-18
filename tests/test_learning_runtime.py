import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "learning-lab"))

from runtime.contracts import validate_execution_result
from runtime.execution import detect_capabilities, run_provider
from runtime.mlx_runner import METAL_SOURCE, build_result
from runtime.api import handle_api_request
from server import is_trusted_origin


def test_detect_capabilities_finds_apple_mlx_and_cost_gates_modal():
    capabilities = detect_capabilities(
        system="Darwin",
        machine="arm64",
        module_available=lambda name: name == "mlx",
        command_available=lambda name: name == "modal",
    )

    assert capabilities == {
        "schema_version": "1.0.0",
        "providers": [
            {
                "id": "apple-mlx",
                "label": "Apple GPU · MLX",
                "available": True,
                "kind": "local",
                "requires_confirmation": False,
                "reason": None,
            },
            {
                "id": "modal-triton",
                "label": "NVIDIA GPU · Modal",
                "available": True,
                "kind": "cloud",
                "requires_confirmation": True,
                "reason": None,
            },
        ],
    }


def test_detect_capabilities_explains_missing_mlx():
    capabilities = detect_capabilities(
        system="Darwin",
        machine="arm64",
        module_available=lambda _name: False,
        command_available=lambda _name: False,
    )

    apple, modal = capabilities["providers"]
    assert apple["available"] is False
    assert "pip install mlx" in apple["reason"]
    assert modal["available"] is False
    assert "Modal CLI" in modal["reason"]


def test_execution_result_contract_accepts_real_measured_gpu_output():
    result = {
        "schema_version": "1.0.0",
        "experiment": "paint-pixels",
        "provider": "apple-mlx",
        "device": "Apple M3 Max",
        "implementation": "metal",
        "workload": {"pixels": 64, "dtype": "float32", "group_size": 10},
        "correctness": {"passed": True, "max_abs_error": 0.0},
        "measurements": [{"name": "latency", "value": 0.02, "unit": "ms"}],
        "output": {"checksum": 32.0},
    }

    assert validate_execution_result(result) is result


def test_execution_result_rejects_simulated_or_invalid_measurements():
    with pytest.raises(ValueError, match="real GPU provider"):
        validate_execution_result({"provider": "simulation"})


def test_modal_execution_requires_explicit_cost_confirmation():
    called = False

    def executor(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(PermissionError, match="billable Modal GPU"):
        run_provider("modal-triton", confirmed=False, executor=executor)

    assert called is False


def test_confirmed_modal_execution_uses_one_quiet_argv_and_parses_final_json_line():
    expected = build_result(device="NVIDIA L4", pixels=64, group_size=16, latency_ms=0.02, checksum=32.0, max_abs_error=0.0)
    expected["provider"] = "modal-triton"
    expected["implementation"] = "triton"
    calls = []

    def executor(argv, **kwargs):
        calls.append((argv, kwargs))
        return type("Completed", (), {"returncode": 0, "stdout": "status\n" + __import__("json").dumps(expected) + "\n", "stderr": ""})()

    result = run_provider("modal-triton", confirmed=True, group_size=10, executor=executor)
    assert result["workload"]["group_size"] == 16
    assert calls[0][0] == ["modal", "run", "--quiet", "platforms/modal/paint.py", "--group-size", "16"]


def test_metal_kernel_maps_one_grid_thread_to_one_guarded_pixel():
    assert "thread_position_in_grid" in METAL_SOURCE
    assert "if (pixel < n_pixels)" in METAL_SOURCE
    assert "out[pixel] = color[0]" in METAL_SOURCE


def test_mlx_result_uses_the_provider_neutral_contract():
    result = build_result(
        device="Apple M3 Max",
        pixels=64,
        group_size=10,
        latency_ms=0.012,
        checksum=32.0,
        max_abs_error=0.0,
    )

    assert result["provider"] == "apple-mlx"
    assert result["implementation"] == "metal"
    assert result["correctness"]["passed"] is True
    validate_execution_result(result)


def test_companion_api_exposes_capabilities_and_dispatches_runs():
    capabilities = lambda: {"schema_version": "1.0.0", "providers": []}
    calls = []

    def runner(provider, *, confirmed, group_size):
        calls.append((provider, confirmed, group_size))
        return {"provider": provider}

    status, payload = handle_api_request("GET", "/api/capabilities", None, capabilities=capabilities, runner=runner)
    assert status == 200
    assert payload["providers"] == []

    status, payload = handle_api_request(
        "POST",
        "/api/run",
        {"provider": "apple-mlx", "confirmed": False, "group_size": 10},
        capabilities=capabilities,
        runner=runner,
    )
    assert status == 200
    assert payload == {"provider": "apple-mlx"}
    assert calls == [("apple-mlx", False, 10)]


def test_companion_api_returns_safe_errors_without_tracebacks():
    def failing_runner(*_args, **_kwargs):
        raise RuntimeError("Metal backend unavailable")

    status, payload = handle_api_request(
        "POST",
        "/api/run",
        {"provider": "apple-mlx", "confirmed": False, "group_size": 10},
        capabilities=lambda: {},
        runner=failing_runner,
    )

    assert status == 422
    assert payload == {"error": "GPU execution failed. Check the companion-server terminal for details."}


def test_modal_paint_launcher_is_cost_bounded_and_uses_masked_triton_store():
    source = (ROOT / "platforms" / "modal" / "paint.py").read_text()

    assert 'gpu="L4"' in source
    assert "timeout=600" in source
    assert "scaledown_window=2" in source
    assert "tl.program_id(0)" in source
    assert "tl.arange" in source
    assert "tl.store" in source
    assert "mask=mask" in source
    assert "next_power_of_2" in source


def test_loopback_server_rejects_cross_origin_and_rebound_hosts():
    assert is_trusted_origin("127.0.0.1:8000", "http://127.0.0.1:8000")
    assert is_trusted_origin("localhost:8000", "http://localhost:8000")
    assert not is_trusted_origin("127.0.0.1:8000", "https://evil.example")
    assert not is_trusted_origin("evil.example:8000", "http://evil.example:8000")
