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
At a fixed `inner`, neighboring threads along the output-column direction read
neighboring `B[inner, col]` elements, so those B loads can be combined into
coalesced memory transactions. Those threads also request the same
`A[row, inner]` value, but the naive code still issues that request from every
thread and repeats it for each output tile. The access pattern is readable,
but it does not explicitly stage either operand for predictable block-wide
reuse.

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

The cooperative loads are arranged so adjacent `threadIdx.x` values fetch
adjacent elements from both A and B, giving coalesced global-memory accesses.
After the load, each A value is reused by the 16 threads producing different
columns in that tile, and each B value is reused by the 16 threads producing
different rows. A 16x16 tile therefore performs many multiply-adds for each
value fetched from global memory. This raises arithmetic intensity—the ratio
of arithmetic work to global-memory bytes—and gives the GPU more useful
computation with which to cover memory cost.

Tile size is a resource tradeoff, not a setting to increase without limit.
Larger tiles can increase reuse, but the shared-memory footprint grows with
both operand tiles, the thread count grows with the two-dimensional block, and
more ambitious per-thread accumulation uses more registers. Shared memory,
registers, and threads are finite per SM; consuming too much of any one can
reduce the number of resident blocks and warps, lowering occupancy. Smaller
tiles admit more blocks but offer less reuse. The 16x16 choice is a clear
teaching point, not a claim that one block shape is optimal for every GPU or
matrix shape.

## cuBLAS: keep the row-major contract

cuBLAS SGEMM is column-major by default. The experiment keeps the public
row-major `C = A x B` contract by viewing the same buffers as transposed
column-major matrices and reversing the operands: it calls SGEMM for
`B^T x A^T`, with dimensions `(N, M, K)`. The resulting column-major buffer is
the row-major C layout, so callers and correctness checks continue to see the
same matrix product.

cuBLAS is expected to lead these teaching kernels because it can select
architecture- and shape-specific kernels with tuned tile shapes, memory
staging, register use, instruction scheduling, and pipelining. Those choices
are specialized for the installed GPU and refined across many workloads,
whereas the custom kernels deliberately keep one readable strategy. The result
is a production baseline, not a universal performance promise: comparisons
still apply only to the recorded GPU, software stack, shapes, and precision.

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
Every workload records `"precision": "float32"`; both the Python case loader
and the CUDA executable reject missing fields or any other precision.

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
