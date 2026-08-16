#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status_));                                 \
      std::exit(2);                                                              \
    }                                                                           \
  } while (0)

constexpr int kBlockSize = 256;

__global__ void reduce_naive_atomic(const float* input, size_t n, float* output) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  float local = 0.0f;
  for (; i < n; i += stride) local += input[i];
  atomicAdd(output, local);
}

__global__ void reduce_shared(const float* input, size_t n, float* output) {
  __shared__ float values[kBlockSize];
  size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x * 2;
  float sum = 0.0f;
  for (; i < n; i += stride) {
    sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
  }
  values[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) values[threadIdx.x] += values[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(output, values[0]);
}

__device__ float warp_sum(float value) {
  unsigned mask = __activemask();
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(mask, value, offset);
  return value;
}

__global__ void reduce_warp_shuffle(const float* input, size_t n, float* output) {
  __shared__ float warp_sums[kBlockSize / 32];
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  float sum = 0.0f;
  for (; i < n; i += stride) sum += input[i];
  sum = warp_sum(sum);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    if (lane == 0) atomicAdd(output, block_sum);
  }
}

enum class Method { Naive, Shared, Warp, Cub };

const char* method_name(Method method) {
  switch (method) {
    case Method::Naive: return "naive_atomic";
    case Method::Shared: return "shared_memory";
    case Method::Warp: return "warp_shuffle";
    case Method::Cub: return "cub_device_reduce";
  }
  return "unknown";
}

void launch(Method method, const float* input, size_t n, float* output,
            void* temp, size_t temp_bytes) {
  if (n == 0) {
    CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
    return;
  }
  int blocks = std::min<int>(1024, static_cast<int>((n + kBlockSize - 1) / kBlockSize));
  if (method != Method::Cub) CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
  switch (method) {
    case Method::Naive:
      reduce_naive_atomic<<<blocks, kBlockSize>>>(input, n, output);
      break;
    case Method::Shared:
      reduce_shared<<<blocks, kBlockSize>>>(input, n, output);
      break;
    case Method::Warp:
      reduce_warp_shuffle<<<blocks, kBlockSize>>>(input, n, output);
      break;
    case Method::Cub:
      CUDA_CHECK(cub::DeviceReduce::Sum(temp, temp_bytes, input, output, n));
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

double reference_sum(const std::vector<float>& values) {
  double result = 0.0;
  for (float value : values) result += value;
  return result;
}

float benchmark(Method method, const float* input, size_t n, float* output,
                void* temp, size_t temp_bytes, int iterations) {
  for (int i = 0; i < 5; ++i) launch(method, input, n, output, temp, temp_bytes);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i)
    launch(method, input, n, output, temp, temp_bytes);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / iterations;
}

int main() {
  const std::vector<size_t> sizes = {0, 1, 17, 255, 256, 1000, 1 << 20};
  const std::vector<Method> methods = {
      Method::Naive, Method::Shared, Method::Warp, Method::Cub};
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::puts("method,size,result,expected,avg_ms,effective_gb_s,status");
  bool all_correct = true;
  for (size_t n : sizes) {
    std::vector<float> host(n);
    for (float& value : host) value = distribution(rng);
    double expected = reference_sum(host);
    float *device_input = nullptr, *device_output = nullptr;
    if (n > 0) {
      CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(device_input, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
    size_t temp_bytes = 0;
    if (n > 0)
      CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, device_input, device_output, n));
    void* temp = nullptr;
    if (temp_bytes > 0) CUDA_CHECK(cudaMalloc(&temp, temp_bytes));

    for (Method method : methods) {
      launch(method, device_input, n, device_output, temp, temp_bytes);
      CUDA_CHECK(cudaDeviceSynchronize());
      float result = 0.0f;
      CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(float), cudaMemcpyDeviceToHost));
      double tolerance = 0.05 + std::abs(expected) * 1e-4;
      bool correct = std::abs(static_cast<double>(result) - expected) <= tolerance;
      all_correct &= correct;
      int iterations = n >= (1 << 20) ? 100 : 20;
      float ms = benchmark(method, device_input, n, device_output, temp, temp_bytes, iterations);
      double gb_s = ms > 0 ? (n * sizeof(float) / 1e6) / ms : 0.0;
      std::printf("%s,%zu,%.7g,%.7g,%.6f,%.3f,%s\n", method_name(method), n,
                  result, expected, ms, gb_s, correct ? "PASS" : "FAIL");
    }
    if (temp) CUDA_CHECK(cudaFree(temp));
    CUDA_CHECK(cudaFree(device_output));
    if (device_input) CUDA_CHECK(cudaFree(device_input));
  }
  std::puts(all_correct ? "ALL CORRECTNESS CHECKS PASSED" : "CORRECTNESS CHECK FAILED");
  return all_correct ? 0 : 1;
}
