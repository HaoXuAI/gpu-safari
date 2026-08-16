# CUDA Matrix Multiplication

This experiment follows one matrix-multiplication contract from a transparent
CUDA baseline through a shared-memory optimization to a vendor library:

```text
C[M, N] = A[M, K] x B[K, N]
```

All matrices are FP32 and stored in row-major order. In particular,
`A[row * K + inner]`, `B[inner * N + col]`, and `C[row * N + col]` name the
elements used to compute `C[row, col]`.

## What to compare

| Implementation | How it maps work | Memory strategy | Why it belongs in the lesson |
| --- | --- | --- | --- |
| Naive CUDA | One thread computes one `C[row, col]` | Each inner-loop iteration loads its A and B operands from global memory | Makes the indexing and thread-to-output mapping explicit |
| Tiled CUDA | One thread still computes one `C[row, col]` in a 16x16 block | The block cooperatively loads A and B tiles into shared memory, then reuses them | Shows how cooperation reduces repeated global-memory loads |
| cuBLAS SGEMM | cuBLAS chooses the GPU work decomposition | A tuned library implementation | Provides a production-library baseline for the same FP32 contract |

## Naive: make the mapping visible

The CUDA grid covers the output matrix. A thread's `threadIdx` and `blockIdx`
produce one `(row, col)` location, and that thread accumulates over `inner`:

```text
sum += A[row, inner] * B[inner, col]
```

The simple version is useful because it makes the row-major indexing clear.
Its drawback is global-memory traffic: neighboring output threads often need
overlapping A or B values, but each thread repeatedly loads its own operands
from global memory.

## Tiled: cooperate before computing

The tiled kernel uses 16x16 thread blocks. For each K-dimension tile, every
thread loads at most one A value and one B value into shared memory. A thread
then multiplies the row from the A tile by the column from the B tile and adds
the products to its private accumulator.

There are two required synchronization points in every tile iteration:

1. `__syncthreads()` after the cooperative loads ensures every thread sees a
   complete shared-memory tile before consuming it.
2. `__syncthreads()` after the multiply-accumulate loop ensures no thread
   overwrites shared memory for the next tile while another still reads it.

Shapes are not assumed to be multiples of 16. A load outside the logical
matrix writes `0.0f` into shared memory instead. This boundary zero-fill keeps
all threads participating in both barriers and makes partial tiles contribute
only their valid products. Only threads whose `(row, col)` falls inside C
write an output element.

## cuBLAS: keep the row-major contract

cuBLAS SGEMM is column-major by default. The experiment keeps the public
row-major `C = A x B` contract by viewing the same buffers as transposed
column-major matrices and reversing the operands: it calls SGEMM for
`B^T x A^T`, with dimensions `(N, M, K)`. The resulting column-major buffer is
the row-major C layout, so callers and correctness checks continue to see the
same matrix product.

## Validate and measure

Run the CPU-only repository validation first:

```bash
python -m pytest -q
```

Then run the maintained GPU experiment:

```bash
modal run platforms/modal/matmul.py
```

The GPU command uses billable L4 compute; review provider limits before
running it. It exercises zero-sized, scalar, rectangular, sub-tile, and
non-tile-aligned workloads before timing. Small cases compare with a CPU
reference that accumulates in double precision; larger cases use cuBLAS as the
reference. A result passes when its maximum absolute error is at most
`1e-3 * max(K, 1)` or its maximum relative error is at most `1e-3`.

Timing starts only after a correctness check passes. The benchmark performs
warmup launches, records CUDA events around the measured launches on the same
stream, synchronizes the stop event, and reports average milliseconds per
iteration. Effective throughput is:

```text
TFLOP/s = (2 * M * N * K) / (average_ms * 1e9)
```

The factor of two counts one multiply and one add for each inner-dimension
term. Compare only results from the same workload and accelerator; correctness
is a prerequisite for interpreting speed.

## Advanced next steps

These are intentionally separate future work, not changes to this FP32
baseline:

- **TF32:** evaluate its accuracy and throughput trade-off on supported GPUs.
- **Tensor Cores:** add a matrix-multiply path that explicitly targets them.
- **CUTLASS:** compare a composable library kernel with cuBLAS and the teaching
  kernels.
- **Triton:** implement the same row-major contract in a higher-level GPU
  kernel language.
