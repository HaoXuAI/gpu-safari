import ast
import importlib.util
import shlex
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "platforms" / "modal" / "matmul.py"


class _FakeImage:
    @classmethod
    def from_registry(cls, *_args, **_kwargs):
        return cls()

    def apt_install(self, *_args, **_kwargs):
        return self

    def pip_install(self, *_args, **_kwargs):
        return self

    def add_local_file(self, *_args, **_kwargs):
        return self

    def add_local_dir(self, *_args, **_kwargs):
        return self

    def env(self, *_args, **_kwargs):
        return self


class _FakeApp:
    def __init__(self, _name):
        pass

    def function(self, **_kwargs):
        return lambda function: function

    def local_entrypoint(self):
        return lambda function: function


def _load_launcher(monkeypatch):
    fake_modal = types.ModuleType("modal")
    fake_modal.App = _FakeApp
    fake_modal.Image = _FakeImage
    fake_modal.is_local = lambda: True
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    spec = importlib.util.spec_from_file_location("matmul_launcher_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matmul_experiment_has_required_layout():
    required = [
        ROOT / "experiments" / "cuda" / "matmul" / "README.md",
        ROOT / "experiments" / "cuda" / "matmul" / "cases.json",
        ROOT / "experiments" / "cuda" / "matmul" / "src" / "kernels.cuh",
        ROOT / "experiments" / "cuda" / "matmul" / "src" / "reference.cuh",
        ROOT / "experiments" / "cuda" / "matmul" / "src" / "matmul.cu",
        LAUNCHER,
    ]
    assert [str(path.relative_to(ROOT)) for path in required if not path.is_file()] == []


def test_matmul_launcher_is_cost_bounded():
    tree = ast.parse(LAUNCHER.read_text())
    run_matmul = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_matmul"
    )
    decorator = next(
        item
        for item in run_matmul.decorator_list
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "function"
    )
    keywords = {item.arg: item.value for item in decorator.keywords}
    assert {
        key: ast.literal_eval(keywords[key])
        for key in ("gpu", "timeout", "scaledown_window")
    } == {"gpu": "L4", "timeout": 600, "scaledown_window": 2}
    assert "min_containers" not in keywords


def test_checked_subprocess_returns_trimmed_stdout(monkeypatch):
    launcher = _load_launcher(monkeypatch)
    assert launcher.run_checked(
        sys.executable,
        "-c",
        "print('  launcher output  ')",
    ) == "launcher output"


def test_checked_subprocess_failure_surfaces_diagnostics(monkeypatch):
    launcher = _load_launcher(monkeypatch)
    command = (
        sys.executable,
        "-c",
        (
            "import sys; print('launcher stdout'); "
            "print('launcher stderr', file=sys.stderr); raise SystemExit(7)"
        ),
    )
    with pytest.raises(RuntimeError) as error:
        launcher.run_checked(*command)

    message = str(error.value)
    assert "exit code 7" in message
    assert shlex.join(command) in message
    assert "stdout:\nlauncher stdout" in message
    assert "stderr:\nlauncher stderr" in message
