# GPU Kernel Lab on Modal

This lab verifies the CUDA stack and compares four sum reductions on an L4:
naive atomic accumulation, shared-memory tree reduction, warp shuffles, and
CUB's production `DeviceReduce` implementation.

## One-time setup

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m modal setup
```

The Modal dashboard currently provides $30 monthly Starter credit. Keep the
workspace usage limit enabled; this account was observed with a $5 limit.

## Verify without using a GPU

```bash
python3 -m pytest -q
python3 -m modal --version
```

## Run the L4 smoke test

This launches billable GPU compute and normally completes in a few minutes:

```bash
modal run modal_cuda.py
```

The function has a 10-minute timeout, a 2-second scale-down window, and no warm
containers. It should print `nvidia-smi`, `nvcc --version`, and `deviceQuery`
results followed by correctness and CUDA-event benchmark CSV before scaling to
zero. The cases include empty input, non-power-of-two sizes, block boundaries,
and one million deterministic random values.

Use L4 for routine CUDA and Triton work. Change `gpu="L4"` to `gpu="A100-40GB"`
only for architecture-specific experiments, and return it to L4 afterward.
