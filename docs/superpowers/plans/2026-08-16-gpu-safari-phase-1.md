# GPU Safari Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebrand the repository as GPU Safari, turn the existing CUDA reduction benchmark into the first self-contained experiment, and establish the versioned result and contribution contracts.

**Architecture:** Use a lab-first monorepo in which experiment code lives independently from thin platform launchers. The CUDA reduction executable continues to emit readable CSV, while a small Python result module converts captured output and environment metadata into a validated, versioned JSON document.

**Tech Stack:** Python 3.12+, pytest, jsonschema, CUDA C++17, NVIDIA CUB, Modal, JSON Schema Draft 2020-12, Git, GitHub CLI

**Spec:** `docs/superpowers/specs/2026-08-16-gpu-safari-design.md`

## Global Constraints

- The project name is **GPU Safari** and the repository slug is `gpu-safari`.
- The project remains vendor-neutral across CUDA, ROCm, MLX, and Triton.
- Experiment logic must remain independent from cloud launchers.
- An experiment needs one demonstrated backend, not every supported platform.
- Correctness must be checked before performance is reported.
- GPU execution must remain explicit because it may consume quotas or incur charges.
- Credentials must never be stored in configs, results, fixtures, or documentation.
- The maintained Phase 1 backend is Modal on one L4 with a 600-second timeout and a 2-second scale-down window.
- A shared orchestration framework and universal cloud API are outside Phase 1.

The following approved subsystems receive separate specifications and plans
after this foundation is working: CUDA and Triton expansion, MLX and ROCm
tracks, the GPU Safari Challenge and Kaggle adapter, and additional maintained
cloud launchers.

## Planned File Structure

```text
gpu-safari/
├── README.md
├── CONTRIBUTING.md
├── benchmarks/
│   ├── __init__.py
│   ├── result.py
│   ├── schema/
│   │   └── result-v1.schema.json
│   └── examples/
│       └── cuda-reduction-l4.json
├── contributor-guide/
│   └── experiment-template.md
├── experiments/
│   └── cuda/
│       └── reduction/
│           ├── README.md
│           └── src/
│               └── reduction.cu
├── platforms/
│   └── modal/
│       ├── README.md
│       └── reduction.py
├── tests/
│   ├── test_benchmark_result.py
│   ├── test_modal_reduction.py
│   ├── test_reduction_experiment.py
│   └── test_repository_contract.py
└── requirements-dev.txt
```

`benchmarks/result.py` owns result parsing, document construction, and schema validation. The CUDA source owns kernels, correctness checks, timing, and readable CSV. `platforms/modal/reduction.py` owns only Modal environment construction, GPU selection, command execution, metadata capture, and result persistence.

---

### Task 1: Establish the GPU Safari Identity and Repository Contract

**Files:**
- Modify: `README.md`
- Create: `tests/test_repository_contract.py`

**Interfaces:**
- Consumes: Existing project files and the approved design specification.
- Produces: A root README containing the stable project name, tagline, ecosystem scope, learning paths, safety note, and links used by every later task.

- [ ] **Step 1: Write the failing repository-contract tests**

Create `tests/test_repository_contract.py`:

```python
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_readme_defines_gpu_safari_scope():
    readme = (ROOT / "README.md").read_text()
    assert readme.startswith("# GPU Safari\n")
    for term in ("CUDA", "ROCm", "MLX", "Triton"):
        assert term in readme
    assert "A hands-on expedition through parallel computing" in readme


def test_readme_makes_gpu_spending_explicit():
    readme = (ROOT / "README.md").read_text().lower()
    assert "may incur charges" in readme
    assert "cpu-only" in readme
```

- [ ] **Step 2: Run the contract tests and confirm the old identity fails**

Run: `./.venv/bin/python -m pytest tests/test_repository_contract.py -v`

Expected: FAIL because the README starts with `# GPU Kernel Lab on Modal` and does not describe the four ecosystems.

- [ ] **Step 3: Replace the root README with the approved identity and navigation**

Write `README.md` with these exact sections:

```markdown
# GPU Safari

> A hands-on expedition through parallel computing—from CUDA, ROCm, MLX, and Triton kernels to AI workloads on local and cloud GPUs.

GPU Safari is a vendor-neutral curriculum and experiment gallery. Start with a guided learning path or choose a self-contained experiment.

## Start here

- Learn the foundations: execution models, memory, correctness, profiling, and benchmarking.
- Run the first experiment: [CUDA reduction](experiments/cuda/reduction/README.md).
- Choose an execution platform: local instructions or a maintained cloud launcher.
- Build toward complete workloads in the application challenges.

## Ecosystems

| Track | Phase 1 status |
| --- | --- |
| CUDA | First reduction experiment |
| Triton | Planned comparison track |
| ROCm | Planned starter experiment |
| MLX | Planned Apple Silicon experiment |

## Safe execution

CPU-only validation is available for repository contracts and result tooling. GPU commands are always explicit and may consume quotas or incur charges. Review the selected accelerator and provider limits before launching a benchmark.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [experiment template](contributor-guide/experiment-template.md). A new experiment needs one reproducible backend; support for every cloud is not required.
```

- [ ] **Step 4: Run the focused tests**

Run: `./.venv/bin/python -m pytest tests/test_repository_contract.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Commit the identity**

```bash
git add README.md tests/test_repository_contract.py
git commit -m "Rebrand project as GPU Safari"
```

---

### Task 2: Move CUDA Reduction into a Self-Contained Experiment

**Files:**
- Move: `cuda/reduction.cu` → `experiments/cuda/reduction/src/reduction.cu`
- Create: `experiments/cuda/reduction/README.md`
- Replace: `tests/test_modal_app.py` → `tests/test_reduction_experiment.py`

**Interfaces:**
- Consumes: The existing four-method CUDA C++17 benchmark.
- Produces: `experiments/cuda/reduction/src/reduction.cu`, which emits the existing CSV header and returns zero only when every correctness case passes.

- [ ] **Step 1: Write the failing experiment-location tests**

Create `tests/test_reduction_experiment.py`:

```python
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXPERIMENT = ROOT / "experiments" / "cuda" / "reduction"
SOURCE = EXPERIMENT / "src" / "reduction.cu"


def test_reduction_experiment_has_lesson_and_source():
    assert (EXPERIMENT / "README.md").is_file()
    assert SOURCE.is_file()


def test_reduction_contains_four_implementations():
    source = SOURCE.read_text()
    for implementation in (
        "reduce_naive_atomic",
        "reduce_shared",
        "reduce_warp_shuffle",
        "cub::DeviceReduce::Sum",
    ):
        assert implementation in source


def test_reduction_covers_edge_cases_and_cuda_timing():
    source = SOURCE.read_text()
    for size in ("0", "1", "17", "255", "256", "1000", "1 << 20"):
        assert size in source
    assert "cudaEventElapsedTime" in source
    assert "std::mt19937 rng(42)" in source
    assert "ALL CORRECTNESS CHECKS PASSED" in source
```

- [ ] **Step 2: Run the new experiment tests and confirm the new path fails**

Run: `./.venv/bin/python -m pytest tests/test_reduction_experiment.py -v`

Expected: FAIL because `experiments/cuda/reduction/` does not exist.

- [ ] **Step 3: Move the source without changing kernel behavior**

Create the destination directories and move the tracked source:

```bash
mkdir -p experiments/cuda/reduction/src
git mv cuda/reduction.cu experiments/cuda/reduction/src/reduction.cu
```

Delete `tests/test_modal_app.py` only after its reduction assertions are present in `tests/test_reduction_experiment.py`; Modal assertions move in Task 5.

- [ ] **Step 4: Add the reduction lesson**

Create `experiments/cuda/reduction/README.md` with:

```markdown
# CUDA Reduction

## What you will learn

Compare atomic accumulation, shared-memory tree reduction, warp shuffles, and CUB's production reduction while checking numerical correctness before interpreting timing.

## Implementations

| Method | Main idea |
| --- | --- |
| Naive atomic | Each block atomically contributes a partial sum |
| Shared memory | Threads reduce a block-local tree before one atomic update |
| Warp shuffle | Registers exchange values within warps before block aggregation |
| CUB DeviceReduce | NVIDIA's production library baseline |

## Correctness workload

The executable tests empty input, scalar input, non-power-of-two sizes, block boundaries, and one million deterministic random values. A benchmark result is valid only when all methods pass the tolerance check.

## Run

Use the maintained [Modal launcher](../../../platforms/modal/README.md), or compile `src/reduction.cu` with CUDA 12 and a target architecture supported by your GPU.
```

- [ ] **Step 5: Run the experiment tests and the full CPU suite**

Run: `./.venv/bin/python -m pytest tests/test_reduction_experiment.py -v`

Expected: `3 passed`.

Run: `./.venv/bin/python -m pytest -q`

Expected: all collected CPU tests pass.

- [ ] **Step 6: Commit the experiment migration**

```bash
git add experiments/cuda/reduction tests/test_reduction_experiment.py tests/test_modal_app.py
git commit -m "Organize CUDA reduction experiment"
```

---

### Task 3: Define and Validate Benchmark Result Version 1

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/result.py`
- Create: `benchmarks/schema/result-v1.schema.json`
- Create: `benchmarks/examples/cuda-reduction-l4.json`
- Create: `tests/test_benchmark_result.py`
- Modify: `requirements-dev.txt`

**Interfaces:**
- Produces: `load_schema() -> dict[str, object]` and `validate_result(document: dict[str, object]) -> None`.
- Produces schema ID: `https://gpu-safari.dev/schemas/result-v1.schema.json`.
- Produces result documents with top-level keys `schema_version`, `experiment`, `source`, `accelerator`, `software`, `workload`, `correctness`, `measurements`, and optional `platform`.

- [ ] **Step 1: Add the schema dependency**

Append this exact constraint to `requirements-dev.txt`:

```text
jsonschema>=4.23,<5
```

Run: `./.venv/bin/python -m pip install -r requirements-dev.txt`

Expected: installation succeeds with `jsonschema` 4.x.

- [ ] **Step 2: Write failing schema and validation tests**

Create `tests/test_benchmark_result.py`:

```python
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from benchmarks.result import load_schema, validate_result


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "benchmarks" / "examples" / "cuda-reduction-l4.json"


def test_schema_has_stable_identity():
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "https://gpu-safari.dev/schemas/result-v1.schema.json"


def test_checked_in_example_is_valid():
    document = json.loads(EXAMPLE.read_text())
    validate_result(document)


def test_correctness_is_required():
    document = json.loads(EXAMPLE.read_text())
    del document["correctness"]
    with pytest.raises(ValidationError):
        validate_result(document)


def test_credentials_are_rejected_as_unknown_properties():
    document = json.loads(EXAMPLE.read_text())
    document["token"] = "not-a-real-token"
    with pytest.raises(ValidationError):
        validate_result(document)
```

- [ ] **Step 3: Run the schema tests and confirm the module is missing**

Run: `./.venv/bin/python -m pytest tests/test_benchmark_result.py -v`

Expected: collection ERROR with `ModuleNotFoundError: No module named 'benchmarks'`.

- [ ] **Step 4: Implement the validator**

Create an empty `benchmarks/__init__.py`, then create `benchmarks/result.py`:

```python
import json
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).parent / "schema" / "result-v1.schema.json"


def load_schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text())


def validate_result(document: dict[str, object]) -> None:
    Draft202012Validator(load_schema()).validate(document)
```

- [ ] **Step 5: Add the closed version-one JSON Schema**

Create `benchmarks/schema/result-v1.schema.json`. Set `additionalProperties` to `false` at the root and for fixed-shape nested objects. Permit string-valued version keys in `software` and experiment-specific keys in `workload`. Require:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://gpu-safari.dev/schemas/result-v1.schema.json",
  "type": "object",
  "required": [
    "schema_version", "experiment", "source", "accelerator", "software",
    "workload", "correctness", "measurements"
  ],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "experiment": {
      "type": "object",
      "required": ["id", "implementation"],
      "properties": {
        "id": {"type": "string", "minLength": 1},
        "implementation": {"type": "string", "minLength": 1}
      },
      "additionalProperties": false
    },
    "source": {
      "type": "object",
      "required": ["revision"],
      "properties": {"revision": {"type": "string", "minLength": 7}},
      "additionalProperties": false
    },
    "accelerator": {
      "type": "object",
      "required": ["vendor", "model", "count"],
      "properties": {
        "vendor": {"type": "string", "minLength": 1},
        "model": {"type": "string", "minLength": 1},
        "architecture": {"type": "string"},
        "count": {"type": "integer", "minimum": 1}
      },
      "additionalProperties": false
    },
    "platform": {
      "type": "object",
      "required": ["provider"],
      "properties": {
        "provider": {"type": "string", "minLength": 1},
        "instance": {"type": "string"},
        "elapsed_seconds": {"type": "number", "minimum": 0},
        "estimated_cost_usd": {"type": "number", "minimum": 0}
      },
      "additionalProperties": false
    },
    "software": {"type": "object", "minProperties": 1, "additionalProperties": {"type": "string"}},
    "workload": {"type": "object", "minProperties": 1},
    "correctness": {
      "type": "object",
      "required": ["passed"],
      "properties": {"passed": {"type": "boolean"}, "tolerance": {"type": "number", "minimum": 0}},
      "additionalProperties": false
    },
    "measurements": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["name", "value", "unit"],
        "properties": {
          "name": {"type": "string", "minLength": 1},
          "value": {"type": "number"},
          "unit": {"type": "string", "minLength": 1}
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
```

- [ ] **Step 6: Add a valid checked-in L4 example**

Create `benchmarks/examples/cuda-reduction-l4.json` using revision `f932b741aee26b22c92111f987265e5132e5e1d5`, provider `Modal`, accelerator vendor `NVIDIA`, model `L4`, implementation `cub_device_reduce`, correctness `passed: true`, and the observed one-million-element measurements `0.005704 ms` and `735 GB/s`. Include `software` keys for `cuda` and `compiler`, and a `workload` object containing `elements: 1048576` and `dtype: "float32"`.

- [ ] **Step 7: Run the focused and complete CPU suites**

Run: `./.venv/bin/python -m pytest tests/test_benchmark_result.py -v`

Expected: `4 passed`.

Run: `./.venv/bin/python -m pytest -q`

Expected: all collected CPU tests pass.

- [ ] **Step 8: Commit the result contract**

```bash
git add benchmarks requirements-dev.txt tests/test_benchmark_result.py
git commit -m "Define benchmark result schema"
```

---

### Task 4: Parse Reduction Output into Standard Measurements

**Files:**
- Modify: `benchmarks/result.py`
- Modify: `tests/test_benchmark_result.py`

**Interfaces:**
- Produces: `parse_reduction_csv(output: str) -> list[dict[str, object]]`.
- Produces: `build_reduction_result(rows: list[dict[str, object]], *, revision: str, gpu_model: str, cuda_version: str, compiler_version: str, provider: str) -> dict[str, object]`.
- Each returned measurement row has `method: str`, `size: int`, `result: float`, `expected: float`, `avg_ms: float`, `effective_gb_s: float`, and `passed: bool`.
- Raises: `ValueError("reduction output did not pass correctness checks")` unless the final non-empty line is `ALL CORRECTNESS CHECKS PASSED`.

- [ ] **Step 1: Add failing parser tests**

Append to `tests/test_benchmark_result.py`:

```python
from benchmarks.result import build_reduction_result, parse_reduction_csv


VALID_OUTPUT = """method,size,result,expected,avg_ms,effective_gb_s,status
cub_device_reduce,1048576,11.5,11.5,0.005704,735.0,PASS
ALL CORRECTNESS CHECKS PASSED
"""


def test_parse_reduction_csv_returns_typed_rows():
    assert parse_reduction_csv(VALID_OUTPUT) == [{
        "method": "cub_device_reduce",
        "size": 1048576,
        "result": 11.5,
        "expected": 11.5,
        "avg_ms": 0.005704,
        "effective_gb_s": 735.0,
        "passed": True,
    }]


def test_parse_reduction_csv_rejects_failed_run():
    failed = VALID_OUTPUT.replace("ALL CORRECTNESS CHECKS PASSED", "CORRECTNESS CHECK FAILED")
    with pytest.raises(ValueError, match="did not pass correctness checks"):
        parse_reduction_csv(failed)


def test_build_reduction_result_selects_largest_cub_workload():
    rows = parse_reduction_csv(VALID_OUTPUT)
    document = build_reduction_result(
        rows,
        revision="f932b741aee26b22c92111f987265e5132e5e1d5",
        gpu_model="NVIDIA L4",
        cuda_version="12.8.1",
        compiler_version="nvcc 12.8",
        provider="Modal",
    )
    validate_result(document)
    assert document["experiment"] == {
        "id": "cuda/reduction",
        "implementation": "cub_device_reduce",
    }
    assert document["measurements"] == [
        {"name": "latency", "value": 0.005704, "unit": "ms"},
        {"name": "effective_bandwidth", "value": 735.0, "unit": "GB/s"},
    ]
```

- [ ] **Step 2: Run the parser tests and confirm the import fails**

Run: `./.venv/bin/python -m pytest tests/test_benchmark_result.py -v`

Expected: collection ERROR because `parse_reduction_csv` is undefined.

- [ ] **Step 3: Implement strict CSV parsing**

Add to `benchmarks/result.py`:

```python
import csv
import io


def parse_reduction_csv(output: str) -> list[dict[str, object]]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or lines[-1] != "ALL CORRECTNESS CHECKS PASSED":
        raise ValueError("reduction output did not pass correctness checks")
    rows = csv.DictReader(io.StringIO("\n".join(lines[:-1])))
    parsed = []
    for row in rows:
        parsed.append({
            "method": row["method"],
            "size": int(row["size"]),
            "result": float(row["result"]),
            "expected": float(row["expected"]),
            "avg_ms": float(row["avg_ms"]),
            "effective_gb_s": float(row["effective_gb_s"]),
            "passed": row["status"] == "PASS",
        })
    if not parsed or not all(row["passed"] for row in parsed):
        raise ValueError("reduction output did not pass correctness checks")
    return parsed


def build_reduction_result(
    rows: list[dict[str, object]],
    *,
    revision: str,
    gpu_model: str,
    cuda_version: str,
    compiler_version: str,
    provider: str,
) -> dict[str, object]:
    candidates = [row for row in rows if row["method"] == "cub_device_reduce"]
    if not candidates:
        raise ValueError("reduction output is missing cub_device_reduce")
    selected = max(candidates, key=lambda row: int(row["size"]))
    document = {
        "schema_version": "1.0.0",
        "experiment": {"id": "cuda/reduction", "implementation": "cub_device_reduce"},
        "source": {"revision": revision},
        "accelerator": {"vendor": "NVIDIA", "model": gpu_model, "count": 1},
        "platform": {"provider": provider},
        "software": {"cuda": cuda_version, "compiler": compiler_version},
        "workload": {"elements": selected["size"], "dtype": "float32"},
        "correctness": {"passed": all(bool(row["passed"]) for row in rows)},
        "measurements": [
            {"name": "latency", "value": selected["avg_ms"], "unit": "ms"},
            {
                "name": "effective_bandwidth",
                "value": selected["effective_gb_s"],
                "unit": "GB/s",
            },
        ],
    }
    validate_result(document)
    return document
```

- [ ] **Step 4: Run parser tests and the complete CPU suite**

Run: `./.venv/bin/python -m pytest tests/test_benchmark_result.py -v`

Expected: `7 passed`.

Run: `./.venv/bin/python -m pytest -q`

Expected: all collected CPU tests pass.

- [ ] **Step 5: Commit the parser**

```bash
git add benchmarks/result.py tests/test_benchmark_result.py
git commit -m "Parse CUDA reduction results"
```

---

### Task 5: Convert the Modal Script into a Thin Platform Launcher

**Files:**
- Move: `modal_cuda.py` → `platforms/modal/reduction.py`
- Create: `platforms/modal/README.md`
- Create: `tests/test_modal_reduction.py`

**Interfaces:**
- Consumes: `experiments/cuda/reduction/src/reduction.cu`, `parse_reduction_csv`, `build_reduction_result`, and `validate_result`.
- Produces: Modal app `gpu-safari-cuda-reduction` and remote function `run_reduction(revision: str) -> dict[str, object]`.
- Preserves: L4 accelerator, 600-second timeout, 2-second scale-down window, zero warm containers, CUDA 12.8.1 development image, and explicit local entry point.

- [ ] **Step 1: Write failing launcher-contract tests**

Create `tests/test_modal_reduction.py`:

```python
from pathlib import Path


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "platforms" / "modal" / "reduction.py"


def test_modal_launcher_is_cost_bounded():
    source = LAUNCHER.read_text()
    assert 'gpu="L4"' in source
    assert "timeout=600" in source
    assert "scaledown_window=2" in source
    assert "min_containers" not in source


def test_modal_launcher_uses_experiment_and_result_contract():
    source = LAUNCHER.read_text()
    assert "experiments/cuda/reduction/src/reduction.cu" in source
    assert "parse_reduction_csv" in source
    assert "build_reduction_result" in source
    assert 'modal.App("gpu-safari-cuda-reduction")' in source


def test_modal_launcher_checks_gpu_environment():
    source = LAUNCHER.read_text()
    for command in ("nvidia-smi", "nvcc", "deviceQuery"):
        assert command in source
```

- [ ] **Step 2: Run launcher tests and confirm the new path fails**

Run: `./.venv/bin/python -m pytest tests/test_modal_reduction.py -v`

Expected: FAIL because `platforms/modal/reduction.py` does not exist.

- [ ] **Step 3: Move and narrow the launcher**

Move `modal_cuda.py` to `platforms/modal/reduction.py`. Change the app name to `gpu-safari-cuda-reduction`. Resolve repository paths with `ROOT = Path(__file__).parents[2]`. Add the CUDA source as `/opt/gpu-safari/reduction.cu`, add the complete local `benchmarks/` directory as `/opt/gpu-safari/benchmarks`, and set `PYTHONPATH=/opt/gpu-safari` in the image.

Rename the remote function to `run_reduction(revision: str)`. Keep environment commands separate from benchmark output. After the executable succeeds, call `parse_reduction_csv`, pass the rows and captured versions to `build_reduction_result`, and return the validated document. Derive the source revision locally with `git rev-parse HEAD` and pass it to the remote call as a string; do not mount `.git` or credentials into the container. Use this data flow:

```python
from benchmarks.result import build_reduction_result, parse_reduction_csv


@app.function(image=cuda_image, gpu="L4", timeout=600, scaledown_window=2)
def run_reduction(revision: str) -> dict[str, object]:
    gpu_model = run(
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    )
    cuda_version = run("nvcc", "--version")
    compiler_version = cuda_version.splitlines()[-1]
    run("nvcc", "-O2", str(device_query_source), "-o", str(device_query_binary))
    print(run(str(device_query_binary)))
    run(
        "nvcc", "-O3", "--use_fast_math", "-std=c++17", "-arch=sm_89",
        "/opt/gpu-safari/reduction.cu", "-o", str(reduction_binary),
    )
    output = run(str(reduction_binary))
    print(output)
    rows = parse_reduction_csv(output)
    return build_reduction_result(
        rows,
        revision=revision,
        gpu_model=gpu_model,
        cuda_version=cuda_version,
        compiler_version=compiler_version,
        provider="Modal",
    )
```

- [ ] **Step 4: Add explicit Modal usage documentation**

Create `platforms/modal/README.md` with these commands:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m modal setup
modal run platforms/modal/reduction.py
```

State that the command launches billable L4 compute, normally finishes within a few minutes, and scales to zero. Explain that the returned JSON is validated against `benchmarks/schema/result-v1.schema.json`.

- [ ] **Step 5: Run CPU validation**

Run: `./.venv/bin/python -m pytest tests/test_modal_reduction.py -v`

Expected: `3 passed`.

Run: `./.venv/bin/python -m pytest -q`

Expected: all collected CPU tests pass.

- [ ] **Step 6: Run one explicit GPU smoke test**

Run: `./.venv/bin/modal run platforms/modal/reduction.py`

Expected: the CUDA source compiles for L4, every reduction row reports `PASS`, the terminal includes `ALL CORRECTNESS CHECKS PASSED`, and the entry point prints a JSON document accepted by `validate_result`.

- [ ] **Step 7: Commit the Modal migration**

```bash
git add platforms/modal modal_cuda.py tests/test_modal_reduction.py
git commit -m "Add Modal reduction launcher"
```

---

### Task 6: Publish the Contributor Contract and Rename GitHub Repository

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `contributor-guide/experiment-template.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: The experiment layout, three validation levels, and result schema established in Tasks 1–5.
- Produces: A repeatable experiment contribution checklist and the public repository URL `https://github.com/HaoXuAI/gpu-safari`.

- [ ] **Step 1: Extend repository-contract tests for contributor documentation**

Append to `tests/test_repository_contract.py`:

```python
def test_contributor_contract_is_discoverable():
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    template = (ROOT / "contributor-guide" / "experiment-template.md").read_text()
    for level in ("CPU validation", "GPU smoke test", "Benchmark run"):
        assert level in contributing
    for section in (
        "Learning objective",
        "Implementations",
        "Correctness",
        "Benchmark methodology",
        "Supported platforms",
        "Limitations",
    ):
        assert f"## {section}" in template
```

- [ ] **Step 2: Run the contract test and confirm missing documents fail**

Run: `./.venv/bin/python -m pytest tests/test_repository_contract.py -v`

Expected: FAIL with `FileNotFoundError` for `CONTRIBUTING.md`.

- [ ] **Step 3: Write the contribution guide**

Create `CONTRIBUTING.md` with prerequisites, experiment naming (`experiments/<ecosystem>/<experiment>/`), the requirement for one reproducible backend, credential safety, and these validation gates:

1. CPU validation: `python -m pytest -q`.
2. GPU smoke test: small and adversarial correctness cases on one documented accelerator.
3. Benchmark run: explicit command, full environment metadata, and result JSON validated against version one.

Require contributors to distinguish observations from universal performance claims and document unsupported platforms.

- [ ] **Step 4: Write the experiment template**

Create `contributor-guide/experiment-template.md` with the exact second-level headings asserted by the test. Under each heading, provide direct questions contributors must answer. Include commands for CPU tests, one backend smoke test, result validation, and a checklist that confirms no credentials or generated binaries are staged.

- [ ] **Step 5: Run all CPU tests and repository hygiene checks**

Run: `./.venv/bin/python -m pytest -q`

Expected: all collected tests pass.

Run: `git diff --check`

Expected: no output and exit status zero.

Run: `git status --short --ignored`

Expected: only intended source and documentation changes are unignored; `.venv/`, `.pytest_cache/`, `__pycache__/`, `*.cubin`, and `*.ptx` remain ignored.

- [ ] **Step 6: Commit the contributor contract**

```bash
git add CONTRIBUTING.md contributor-guide/experiment-template.md README.md tests/test_repository_contract.py
git commit -m "Document GPU Safari contributions"
```

- [ ] **Step 7: Verify the complete Phase 1 deliverable before publishing**

Run: `./.venv/bin/python -m pytest -q`

Expected: all collected tests pass with zero failures.

Run: `git status -sb`

Expected: clean `main` branch ahead of `origin/main` only by the intentional Phase 1 commits.

Run: `git log --show-signature --oneline origin/main..HEAD`

Expected: every Phase 1 commit reports a good signature.

- [ ] **Step 8: Rename the GitHub repository and update the remote**

Run:

```bash
gh repo rename gpu-safari --repo HaoXuAI/gpu-kernel-lab --yes
git remote set-url origin https://github.com/HaoXuAI/gpu-safari.git
git remote -v
```

Expected: both fetch and push URLs are `https://github.com/HaoXuAI/gpu-safari.git`.

- [ ] **Step 9: Push and verify the public repository**

Run:

```bash
git push origin main
gh repo view HaoXuAI/gpu-safari --json nameWithOwner,url,visibility,defaultBranchRef
git rev-parse HEAD
git rev-parse origin/main
```

Expected: GitHub reports a public repository named `HaoXuAI/gpu-safari` with default branch `main`, and local `HEAD` equals `origin/main`.
