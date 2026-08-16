# GPU Safari

> A hands-on expedition through parallel computing—from CUDA, ROCm, MLX, and Triton kernels to AI workloads on local and cloud GPUs.

GPU Safari is a vendor-neutral curriculum and experiment gallery. Start with a guided learning path or choose a self-contained experiment.

## Start here

- Learn the foundations: execution models, memory, correctness, profiling, and benchmarking.
- Run the first experiment: [CUDA reduction](experiments/cuda/reduction/README.md).
- Continue with [CUDA matrix multiplication](experiments/cuda/matmul/README.md).
- Choose an execution platform: local instructions or a maintained cloud launcher.
- Build toward complete workloads in the application challenges.

## Ecosystems

| Track | Phase 1 status |
| --- | --- |
| CUDA | Reduction and matrix multiplication experiments |
| Triton | Planned comparison track |
| ROCm | Planned starter experiment |
| MLX | Planned Apple Silicon experiment |

## Safe execution

CPU-only validation is available for repository contracts and result tooling. GPU commands are always explicit and may consume quotas or incur charges. Review the selected accelerator and provider limits before launching a benchmark.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [experiment template](contributor-guide/experiment-template.md). A new experiment needs one reproducible backend; support for every cloud is not required.
