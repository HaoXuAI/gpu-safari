#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expression)                                                   \
  do {                                                                           \
    const cudaError_t cuda_status_ = (expression);                               \
    if (cuda_status_ != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA call failed: %s at %s:%d: %s\n",              \
                   #expression, __FILE__, __LINE__,                              \
                   cudaGetErrorString(cuda_status_));                            \
      std::exit(2);                                                              \
    }                                                                            \
  } while (false)
#endif

constexpr int kMatmulTileSize = 16;

__global__ void matmul_naive_kernel(const float* a, const float* b, float* c,
                                    int m, int n, int k) {
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || col >= n) {
    return;
  }

  float sum = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    sum += a[static_cast<std::size_t>(row) * k + inner] *
           b[static_cast<std::size_t>(inner) * n + col];
  }
  c[static_cast<std::size_t>(row) * n + col] = sum;
}

__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c,
                                    int m, int n, int k) {
  __shared__ float a_tile[kMatmulTileSize][kMatmulTileSize];
  __shared__ float b_tile[kMatmulTileSize][kMatmulTileSize];

  const int row = static_cast<int>(blockIdx.y * kMatmulTileSize + threadIdx.y);
  const int col = static_cast<int>(blockIdx.x * kMatmulTileSize + threadIdx.x);
  float sum = 0.0f;

  const int tile_count = (k - 1) / kMatmulTileSize + 1;
  for (int tile = 0; tile < tile_count;
       ++tile) {
    const int a_col = tile * kMatmulTileSize + static_cast<int>(threadIdx.x);
    const int b_row = tile * kMatmulTileSize + static_cast<int>(threadIdx.y);
    a_tile[threadIdx.y][threadIdx.x] =
        row < m && a_col < k
            ? a[static_cast<std::size_t>(row) * k + a_col]
            : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        b_row < k && col < n
            ? b[static_cast<std::size_t>(b_row) * n + col]
            : 0.0f;
    __syncthreads();

    for (int inner = 0; inner < kMatmulTileSize; ++inner) {
      sum += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[static_cast<std::size_t>(row) * n + col] = sum;
  }
}

inline void launch_naive(const float* a, const float* b, float* c, int m, int n,
                         int k, cudaStream_t stream) {
  if (m == 0 || n == 0) {
    return;
  }
  if (k == 0) {
    CUDA_CHECK(cudaMemsetAsync(c, 0,
                               static_cast<std::size_t>(m) * n * sizeof(float),
                               stream));
    return;
  }

  const dim3 block(kMatmulTileSize, kMatmulTileSize);
  const dim3 grid((n - 1) / kMatmulTileSize + 1,
                  (m - 1) / kMatmulTileSize + 1);
  matmul_naive_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

inline void launch_tiled(const float* a, const float* b, float* c, int m, int n,
                         int k, cudaStream_t stream) {
  if (m == 0 || n == 0) {
    return;
  }
  if (k == 0) {
    CUDA_CHECK(cudaMemsetAsync(c, 0,
                               static_cast<std::size_t>(m) * n * sizeof(float),
                               stream));
    return;
  }

  const dim3 block(kMatmulTileSize, kMatmulTileSize);
  const dim3 grid((n - 1) / kMatmulTileSize + 1,
                  (m - 1) / kMatmulTileSize + 1);
  matmul_tiled_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}
