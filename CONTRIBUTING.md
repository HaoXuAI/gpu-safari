# Contributing to GPU Safari

GPU Safari welcomes focused experiments that teach a transferable GPU concept. Place an experiment at `experiments/<ecosystem>/<experiment>/` and keep its implementation independent from cloud launchers.

## What a contribution needs

- A clear learning objective and concise lesson
- Correctness checks before performance measurements
- A documented benchmark method and versioned result JSON
- Reproducible commands for at least one backend
- Supported environment details and known limitations
- No credentials, generated binaries, or provider-specific experiment logic

Support for every cloud is not required. Platform code belongs under `platforms/<provider>/` and should only provision the environment, invoke an experiment, and retrieve results.

## Validation levels

### CPU validation

Run `python -m pytest -q` to validate result schemas, parsers, configurations, and reference behavior without using a GPU.

### GPU smoke test

Run small and adversarial correctness cases on one documented accelerator. Include empty inputs, awkward sizes, and numerical tolerances relevant to the experiment.

### Benchmark run

Run the explicit benchmark command, capture the hardware and software environment, and validate the result against `benchmarks/schema/result-v1.schema.json`. GPU runs may consume quotas or incur charges.

## Performance claims

Treat benchmark results as observations. State the accelerator, versions, workload, precision, warmup, measurement method, and limitations. Do not infer architectural superiority from one uncontrolled comparison.

## Before committing

```bash
python -m pytest -q
git diff --check
git status --short --ignored
```

Confirm that credentials are absent and `.venv/`, caches, `*.cubin`, and `*.ptx` remain ignored.
