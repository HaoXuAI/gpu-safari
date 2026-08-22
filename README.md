# GPU Safari

> A hands-on expedition through parallel computing—from CUDA, ROCm, MLX, and Triton kernels to AI workloads on local and cloud GPUs.

GPU Safari is a vendor-neutral curriculum and experiment gallery. Start with a guided learning path or choose a self-contained experiment.

## Explore the website

The static website begins with a visual, 10-minute expedition that works without a GPU, installation, or account:

```bash
python -m http.server 8000
```

Open <http://localhost:8000/learning-lab/>. From there you can explore the trail map, complete **Paint Pixels in Parallel**, or connect the optional local companion for measured Apple GPU execution.

## Start here

- New to GPU programming? Begin with the interactive [Paint Pixels in Parallel](learning-lab/) learning lab. It runs in a browser without a GPU and connects familiar Python ideas to Triton and CUDA.
- Start the guided website: [GPU Safari learning experience](learning-lab/).
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
| MLX / Metal | First real Apple Silicon learning kernel |

## Safe execution

CPU-only validation is available for repository contracts and result tooling. GPU commands are always explicit and may consume quotas or incur charges. Review the selected accelerator and provider limits before launching a benchmark.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [experiment template](contributor-guide/experiment-template.md). A new experiment needs one reproducible backend; support for every cloud is not required.
