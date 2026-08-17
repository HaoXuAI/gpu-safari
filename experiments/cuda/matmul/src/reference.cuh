#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

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

inline const char* cublas_error_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown cuBLAS error";
}

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(expression)                                                 \
  do {                                                                           \
    const cublasStatus_t cublas_status_ = (expression);                          \
    if (cublas_status_ != CUBLAS_STATUS_SUCCESS) {                               \
      std::fprintf(stderr, "cuBLAS call failed: %s at %s:%d: %s\n",            \
                   #expression, __FILE__, __LINE__,                              \
                   cublas_error_string(cublas_status_));                         \
      std::exit(2);                                                              \
    }                                                                            \
  } while (false)
#endif

struct ErrorMetrics {
  double max_abs_error;
  double max_rel_error;
};

inline std::vector<float> cpu_reference(const std::vector<float>& a,
                                        const std::vector<float>& b, int m,
                                        int n, int k) {
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        sum +=
            static_cast<double>(a[static_cast<std::size_t>(row) * k + inner]) *
            static_cast<double>(b[static_cast<std::size_t>(inner) * n + col]);
      }
      c[static_cast<std::size_t>(row) * n + col] = static_cast<float>(sum);
    }
  }
  return c;
}

inline ErrorMetrics compute_errors(const std::vector<float>& actual,
                                   const std::vector<float>& expected) {
  if (actual.size() != expected.size()) {
    throw std::invalid_argument("actual and expected sizes differ");
  }

  ErrorMetrics metrics{0.0, 0.0};
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference =
        std::abs(static_cast<double>(actual[index]) - expected[index]);
    const double relative =
        difference /
        std::max(std::abs(static_cast<double>(expected[index])), 1e-7);
    if (!std::isfinite(difference) || !std::isfinite(relative)) {
      metrics.max_abs_error = std::numeric_limits<double>::infinity();
      metrics.max_rel_error = std::numeric_limits<double>::infinity();
      continue;
    }
    metrics.max_abs_error = std::max(metrics.max_abs_error, difference);
    metrics.max_rel_error = std::max(metrics.max_rel_error, relative);
  }
  return metrics;
}

inline void launch_cublas_row_major(cublasHandle_t handle, const float* a,
                                    const float* b, float* c, int m, int n,
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

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                           b, n, a, k, &beta, c, n));
}
