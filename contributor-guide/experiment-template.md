# Experiment Template

Copy this structure into `experiments/<ecosystem>/<experiment>/README.md` and replace the prompts with concrete answers.

## Learning objective

What transferable GPU concept will a learner understand, and what should they be able to explain or change after completing the experiment?

## Implementations

Which implementations are compared? Explain the relevant execution, memory, synchronization, or library choices without assuming benchmark results.

## Correctness

What is the independent reference? List the boundary, empty, awkward-size, numerical-tolerance, and deterministic cases that must pass before timing begins.

## Benchmark methodology

Document the input, precision, warmup count, measured iterations, timer, synchronization points, metrics, and result-schema version.

## Supported platforms

Provide exact setup, CPU-validation, GPU-smoke-test, and benchmark commands for at least one backend. State the required accelerator, driver, runtime, compiler, framework, and libraries.

## Limitations

State unsupported hardware, portability constraints, numerical tradeoffs, and reasons the measurements should not be generalized beyond the recorded environment.

## Submission checklist

- [ ] `python -m pytest -q` passes.
- [ ] Correctness succeeds before benchmark data is accepted.
- [ ] Result JSON validates against `benchmarks/schema/result-v1.schema.json`.
- [ ] One backend has reproducible commands and environment details.
- [ ] No credentials, generated binaries, `.cubin`, or `.ptx` files are staged.
- [ ] Performance statements are scoped to the measured configuration.
