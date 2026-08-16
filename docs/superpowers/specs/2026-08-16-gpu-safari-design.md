# GPU Safari Design

## Purpose

GPU Safari is a hands-on, vendor-neutral learning project for parallel
computing. It helps learners progress from GPU fundamentals and isolated
kernels to profiling, portable implementations, real AI workloads, and
cost-aware execution on local and cloud hardware.

The project covers CUDA, ROCm, MLX, and Triton without making any one vendor,
framework, or cloud platform its identity. The existing CUDA reduction
benchmark becomes the first experiment rather than the scope of the project.

**Tagline:** A hands-on expedition through parallel computing—from CUDA,
ROCm, MLX, and Triton kernels to AI workloads on local and cloud GPUs.

## Audience and Learning Model

GPU Safari serves two complementary audiences:

1. Learners follow a guided path from architecture fundamentals to complete
   GPU-accelerated applications.
2. Contributors add self-contained experiments, platform recipes, and
   comparable implementations through a standard contract.

The repository is a hybrid curriculum and experiment gallery. The guided core
provides an intentional progression, while contributed experiments can explore
specialized hardware, techniques, and workloads independently.

## Design Principles

- Teach transferable GPU concepts before platform-specific mechanics.
- Keep experiment logic independent from cloud launchers.
- Require correctness before measuring performance.
- Record enough environment metadata to make results reproducible.
- Treat benchmark results as observations, not universal performance claims.
- Make the first contribution path small: one experiment needs one working
  backend, not support for every platform.
- Keep billable GPU runs explicit; CPU-only checks should be available where
  practical.
- Never store platform credentials in the repository.

## Repository Architecture

The project uses a lab-first monorepo:

```text
gpu-safari/
├── foundations/
├── experiments/
│   ├── cuda/
│   ├── triton/
│   ├── rocm/
│   └── mlx/
├── applications/
│   └── gpu-safari-challenge/
├── platforms/
│   ├── local/
│   ├── modal/
│   └── lightning/
├── benchmarks/
├── docs/
└── contributor-guide/
```

`foundations/` explains concepts shared across ecosystems, including execution
models, memory hierarchies, synchronization, numerical correctness, profiling,
and benchmarking. `experiments/` contains focused implementations.
`applications/` connects low-level lessons to end-to-end workloads.
`platforms/` contains launchers and setup recipes only. `benchmarks/` contains
schemas, result tooling, and published observations.

The current reduction implementation moves to
`experiments/cuda/reduction/`. Its Modal launcher moves under the Modal
platform integration and invokes the experiment without owning its logic.

## Experiment Contract

Every experiment contains:

- a concise lesson explaining the concept and expected learning outcome;
- one or more implementations;
- correctness tests, including boundary and awkward-size cases;
- a benchmark configuration with documented workload parameters;
- machine-readable results that follow the versioned result schema; and
- supported-platform notes with reproducible commands.

An experiment may be merged with one demonstrated execution backend. Other
backends can be added incrementally. Ecosystem-equivalent experiments, such as
CUDA and Triton reductions, remain separate implementations linked through
shared comparison documentation.

## Benchmark Result Model

Each recorded result includes:

- schema version and experiment identifier;
- source revision;
- accelerator vendor, model, architecture, and count;
- host and platform metadata when available;
- driver, runtime, compiler, framework, and library versions;
- workload parameters and numerical precision;
- correctness status and tolerances;
- warmup and measurement methodology;
- latency, throughput, and peak accelerator memory when measurable; and
- elapsed platform time and estimated cost when the provider exposes enough
  information to calculate it reliably.

The command-line output remains readable for learners and can additionally
write the same result as JSON. Comparisons must disclose hardware and software
differences rather than ranking unlike systems as if they were equivalent.

## Platform Architecture

Platform support uses thin launchers and recipes:

```text
experiment → platform launcher → GPU environment
```

Local, Modal, and Lightning AI are the maintained v1 examples. CoreWeave,
Nebius, AWS, and GCP begin as documented contribution targets and become
maintained integrations only after repeated use justifies that commitment.

Platform-specific code handles environment construction, accelerator
selection, file transfer, invocation, and result retrieval. It must not contain
kernel or workload logic. Missing accelerators, drivers, or toolchains produce
clear setup errors. Secrets are supplied through provider secret managers or
local credential stores and are never serialized into experiment configs or
results.

GPU execution is opt-in because it may consume quotas or incur charges.
Documentation distinguishes CPU validation, GPU smoke tests, and benchmark
runs.

## GPU Safari Challenge

The first application is a permanent, repository-owned Kaggle-style mini
competition. It uses a public, versioned dataset and includes:

- a simple CPU baseline;
- GPU training and inference baselines;
- a fixed predictive-quality metric;
- performance reporting for runtime, throughput, peak memory, and estimated
  cost;
- an offline-compatible submission format;
- a reproducible local scorer; and
- an optional Kaggle notebook and submission adapter.

The learning sequence begins with a correct submission. Learners then optimize
data movement, mixed precision, compilation, batching, and profiling before
introducing custom kernels where measurements justify them.

Active Kaggle competitions may be documented as optional case studies. They
cannot be required by the core curriculum because availability, hardware, data,
and rules change over time.

## Validation and Contribution Model

Validation has three levels:

1. **CPU validation** checks schemas, configuration, documentation examples,
   reference calculations, and code that does not require an accelerator.
2. **GPU smoke tests** verify correctness on small, empty, non-power-of-two,
   boundary, and other adversarial inputs relevant to the experiment.
3. **Benchmark runs** are explicit, potentially billable jobs that capture the
   complete result metadata.

Pull requests are not required to execute every cloud backend. A contributor
must provide reproducible commands, supported environment details, correctness
evidence, and limitations. Maintainers can validate selected hardware
separately. Benchmark contributions must not claim architectural superiority
from a single uncontrolled comparison.

## Version-One Scope

Version one contains:

- foundations for architecture, memory, correctness, profiling, and
  benchmarking;
- CUDA reduction, matrix multiplication, and softmax experiments;
- equivalent Triton experiments where the comparison teaches a clear lesson;
- one starter MLX experiment for Apple Silicon;
- ROCm setup guidance and one starter experiment;
- maintained local, Modal, and Lightning AI execution examples;
- adapter specifications for CoreWeave, Nebius, AWS, and GCP; and
- the first GPU Safari Challenge.

A shared orchestration framework is intentionally excluded from v1. A small
CLI may be introduced only after multiple experiments demonstrate stable,
repeated needs that cannot be met by simple commands.

## Delivery Sequence

1. Rename and restructure the existing CUDA reduction project as GPU Safari.
2. Define the experiment template and versioned benchmark-result schema.
3. Add CUDA matrix multiplication and softmax experiments.
4. Add equivalent Triton experiments and comparison lessons.
5. Add the first MLX and ROCm experiments.
6. Build the permanent GPU Safari Challenge and optional Kaggle adapter.
7. Expand maintained platform integrations based on demonstrated demand.

## Success Criteria

The first release succeeds when a new learner can:

- understand the repository's learning path without external orientation;
- run CPU validation without a GPU;
- execute at least one correct GPU experiment locally or on a documented cloud
  platform;
- interpret correctness and benchmark results with the environment context;
- compare two implementations of the same concept; and
- contribute a new single-backend experiment using the documented template.
