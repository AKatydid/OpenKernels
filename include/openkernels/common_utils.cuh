#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA(cmd)                                                         \
  do {                                                                          \
    cudaError_t err = (cmd);                                                    \
    if (err != cudaSuccess) {                                                   \
      throw std::runtime_error(cudaGetErrorString(err));                        \
    }                                                                           \
  } while (0)

// Alias for consistency with common CUDA naming.
#define CUDA_CHECK(cmd) CHECK_CUDA(cmd)

class CudaTimer {
 public:
  CudaTimer() {
    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));
  }

  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t stream = 0) { CHECK_CUDA(cudaEventRecord(start_, stream)); }

  float stop(cudaStream_t stream = 0) {
    CHECK_CUDA(cudaEventRecord(stop_, stream));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

inline std::vector<float> make_random_f32(size_t n, float lo = -1.0f,
                                          float hi = 1.0f) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = dist(gen);
  }
  return v;
}

inline std::vector<int32_t> make_random_i32(size_t n, int lo = 1, int hi = 4) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int32_t> dist(lo, hi);
  std::vector<int32_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = dist(gen);
  }
  return v;
}

inline void print_head(const char *tag, const std::vector<float> &v, int k = 4) {
  std::cout << tag << ": ";
  for (int i = 0; i < k && i < static_cast<int>(v.size()); ++i) {
    std::cout << v[i] << (i + 1 == k ? "" : ", ");
  }
  std::cout << "\n";
}

inline void print_head(const char *tag, const std::vector<int32_t> &v, int k = 8) {
  std::cout << tag << ": ";
  for (int i = 0; i < k && i < static_cast<int>(v.size()); ++i) {
    std::cout << v[i] << (i + 1 == k ? "" : ", ");
  }
  std::cout << "\n";
}
