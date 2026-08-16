#include "kernels.cuh"
#include "reference.cuh"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

struct MatmulCase {
  std::string name;
  int m;
  int n;
  int k;
  std::string precision;
  int warmup;
  int iterations;
};

enum class Implementation { Naive, Tiled, Cublas };

constexpr std::uint64_t kCpuReferenceOperationLimit = 1'000'000;

int read_nonnegative_int(const json& entry, const char* field,
                         std::size_t case_index) {
  if (!entry.at(field).is_number_integer() &&
      !entry.at(field).is_number_unsigned()) {
    throw std::invalid_argument("case " + std::to_string(case_index) + " " +
                                field + " must be an integer");
  }

  std::int64_t value = 0;
  if (entry.at(field).is_number_unsigned()) {
    const std::uint64_t unsigned_value = entry.at(field).get<std::uint64_t>();
    if (unsigned_value > static_cast<std::uint64_t>(
                             std::numeric_limits<int>::max())) {
      throw std::invalid_argument("case " + std::to_string(case_index) + " " +
                                  field + " is too large");
    }
    value = static_cast<std::int64_t>(unsigned_value);
  } else {
    value = entry.at(field).get<std::int64_t>();
  }

  if (value < 0) {
    throw std::invalid_argument("case " + std::to_string(case_index) + " " +
                                field + " must be nonnegative");
  }
  if (value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("case " + std::to_string(case_index) + " " +
                                field + " is too large");
  }
  return static_cast<int>(value);
}

std::vector<MatmulCase> load_cases(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::invalid_argument("could not open cases file: " + path);
  }

  json document;
  input >> document;
  if (!document.is_array() || document.empty()) {
    throw std::invalid_argument("matmul cases must be a non-empty array");
  }

  const std::set<std::string> required_fields{
      "name", "m", "n", "k", "precision", "warmup", "iterations"};
  std::vector<MatmulCase> cases;
  cases.reserve(document.size());
  for (std::size_t index = 0; index < document.size(); ++index) {
    const json& entry = document[index];
    if (!entry.is_object()) {
      throw std::invalid_argument("case " + std::to_string(index) +
                                  " must be an object");
    }

    std::set<std::string> actual_fields;
    for (auto field = entry.begin(); field != entry.end(); ++field) {
      actual_fields.insert(field.key());
    }
    if (actual_fields != required_fields) {
      throw std::invalid_argument("case " + std::to_string(index) +
                                  " has the wrong field set");
    }
    if (!entry.at("name").is_string() ||
        entry.at("name").get_ref<const std::string&>().empty()) {
      throw std::invalid_argument("case " + std::to_string(index) +
                                  " name must be non-empty");
    }
    if (!entry.at("precision").is_string() ||
        entry.at("precision").get_ref<const std::string&>() != "float32") {
      throw std::invalid_argument("case " + std::to_string(index) +
                                  " precision must be float32");
    }

    MatmulCase parsed{
        entry.at("name").get<std::string>(),
        read_nonnegative_int(entry, "m", index),
        read_nonnegative_int(entry, "n", index),
        read_nonnegative_int(entry, "k", index),
        entry.at("precision").get<std::string>(),
        read_nonnegative_int(entry, "warmup", index),
        read_nonnegative_int(entry, "iterations", index),
    };
    if (parsed.iterations < 1) {
      throw std::invalid_argument("case " + std::to_string(index) +
                                  " iterations must be positive");
    }
    cases.push_back(std::move(parsed));
  }
  return cases;
}

std::size_t matrix_elements(int rows, int columns) {
  const std::size_t row_count = static_cast<std::size_t>(rows);
  const std::size_t column_count = static_cast<std::size_t>(columns);
  if (column_count != 0 &&
      row_count > std::numeric_limits<std::size_t>::max() / column_count) {
    throw std::invalid_argument("matrix dimensions are too large");
  }
  const std::size_t elements = row_count * column_count;
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::invalid_argument("matrix allocation is too large");
  }
  return elements;
}

bool use_cpu_reference(const MatmulCase& test_case) {
  if (test_case.m == 0 || test_case.n == 0 || test_case.k == 0) {
    return true;
  }
  std::uint64_t operations = static_cast<std::uint64_t>(test_case.m);
  if (operations >
      kCpuReferenceOperationLimit / static_cast<std::uint64_t>(test_case.n)) {
    return false;
  }
  operations *= static_cast<std::uint64_t>(test_case.n);
  return operations <=
         kCpuReferenceOperationLimit / static_cast<std::uint64_t>(test_case.k);
}

const char* implementation_name(Implementation implementation) {
  switch (implementation) {
    case Implementation::Naive:
      return "naive";
    case Implementation::Tiled:
      return "tiled";
    case Implementation::Cublas:
      return "cublas";
  }
  return "unknown";
}

void launch(Implementation implementation, cublasHandle_t handle,
            const float* a, const float* b, float* c, int m, int n, int k,
            cudaStream_t stream) {
  switch (implementation) {
    case Implementation::Naive:
      launch_naive(a, b, c, m, n, k, stream);
      CUDA_CHECK(cudaGetLastError());
      return;
    case Implementation::Tiled:
      launch_tiled(a, b, c, m, n, k, stream);
      CUDA_CHECK(cudaGetLastError());
      return;
    case Implementation::Cublas:
      launch_cublas_row_major(handle, a, b, c, m, n, k, stream);
      return;
  }
}

float benchmark(Implementation implementation, cublasHandle_t handle,
                const float* a, const float* b, float* c,
                const MatmulCase& test_case, cudaStream_t stream) {
  for (int iteration = 0; iteration < test_case.warmup; ++iteration) {
    launch(implementation, handle, a, b, c, test_case.m, test_case.n,
           test_case.k, stream);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < test_case.iterations; ++iteration) {
    launch(implementation, handle, a, b, c, test_case.m, test_case.n,
           test_case.k, stream);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(test_case.iterations);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: matmul <cases.json>\n";
    return 1;
  }

  std::vector<MatmulCase> cases;
  try {
    cases = load_cases(argv[1]);
  } catch (const std::exception& error) {
    std::cerr << "invalid matmul cases: " << error.what() << '\n';
    return 1;
  }

  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(cublasCreate(&cublas));
  CUBLAS_CHECK(cublasSetStream(cublas, stream));
  CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));

  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  const std::vector<Implementation> implementations{
      Implementation::Naive, Implementation::Tiled, Implementation::Cublas};
  bool all_correct = true;

  std::cout << "implementation,m,n,k,avg_ms,tflops,max_abs_error,"
               "max_rel_error,status\n";
  std::cout << std::fixed << std::setprecision(6);

  try {
    for (const MatmulCase& test_case : cases) {
      const std::size_t a_elements = matrix_elements(test_case.m, test_case.k);
      const std::size_t b_elements = matrix_elements(test_case.k, test_case.n);
      const std::size_t c_elements = matrix_elements(test_case.m, test_case.n);

      std::vector<float> host_a(a_elements);
      std::vector<float> host_b(b_elements);
      std::vector<float> actual(c_elements);
      for (float& value : host_a) {
        value = distribution(generator);
      }
      for (float& value : host_b) {
        value = distribution(generator);
      }

      float* device_a = nullptr;
      float* device_b = nullptr;
      float* device_c = nullptr;
      if (a_elements != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a),
                              a_elements * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(device_a, host_a.data(),
                                   a_elements * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
      }
      if (b_elements != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b),
                              b_elements * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(device_b, host_b.data(),
                                   b_elements * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
      }
      if (c_elements != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c),
                              c_elements * sizeof(float)));
      }

      std::vector<float> expected;
      if (use_cpu_reference(test_case)) {
        expected = cpu_reference(host_a, host_b, test_case.m, test_case.n,
                                 test_case.k);
      } else {
        launch_cublas_row_major(cublas, device_a, device_b, device_c,
                                test_case.m, test_case.n, test_case.k, stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        expected.resize(c_elements);
        CUDA_CHECK(cudaMemcpy(expected.data(), device_c,
                              c_elements * sizeof(float),
                              cudaMemcpyDeviceToHost));
      }

      for (Implementation implementation : implementations) {
        if (c_elements != 0) {
          // All-one FP32 bit patterns are NaNs, so an unwritten element fails.
          CUDA_CHECK(cudaMemsetAsync(device_c, 0xFF,
                                     c_elements * sizeof(float), stream));
        }
        launch(implementation, cublas, device_a, device_b, device_c,
               test_case.m, test_case.n, test_case.k, stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (c_elements != 0) {
          CUDA_CHECK(cudaMemcpy(actual.data(), device_c,
                                c_elements * sizeof(float),
                                cudaMemcpyDeviceToHost));
        }

        const ErrorMetrics errors = compute_errors(actual, expected);
        const double absolute_tolerance =
            1e-3 * static_cast<double>(std::max(test_case.k, 1));
        const bool correct = errors.max_abs_error <= absolute_tolerance ||
                             errors.max_rel_error <= 1e-3;
        all_correct = all_correct && correct;

        float average_ms = 0.0f;
        if (correct) {
          average_ms = benchmark(implementation, cublas, device_a, device_b,
                                 device_c, test_case, stream);
        }
        const double tflops =
            average_ms > 0.0f
                ? (2.0 * static_cast<double>(test_case.m) * test_case.n *
                   test_case.k) /
                      (static_cast<double>(average_ms) * 1e9)
                : 0.0;
        std::cout << implementation_name(implementation) << ',' << test_case.m
                  << ',' << test_case.n << ',' << test_case.k << ','
                  << average_ms << ',' << tflops << ',' << errors.max_abs_error
                  << ',' << errors.max_rel_error << ','
                  << (correct ? "PASS" : "FAIL") << '\n';
      }

      if (device_c != nullptr) {
        CUDA_CHECK(cudaFree(device_c));
      }
      if (device_b != nullptr) {
        CUDA_CHECK(cudaFree(device_b));
      }
      if (device_a != nullptr) {
        CUDA_CHECK(cudaFree(device_a));
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "matmul experiment failed: " << error.what() << '\n';
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 1;
  }

  CUBLAS_CHECK(cublasDestroy(cublas));
  CUDA_CHECK(cudaStreamDestroy(stream));
  if (all_correct) {
    std::cout << "ALL MATMUL CORRECTNESS CHECKS PASSED\n";
    return 0;
  }
  std::cout << "MATMUL CORRECTNESS CHECK FAILED\n";
  return 1;
}
