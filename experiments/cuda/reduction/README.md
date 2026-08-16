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
