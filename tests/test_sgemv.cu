#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"
#include "openkernels/kernels.h"

int main() {
  try {
    const int M = 256;
    const int K = 128;
    std::vector<float> h_a = make_random_f32(static_cast<size_t>(M) * K);
    std::vector<float> h_x = make_random_f32(K);
    std::vector<float> h_y(M, 0.0f);

    float *d_a = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_y.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    sgemv_k128_f32x4(d_a, d_x, d_y, M, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    print_head("sgemv_k128_f32x4", h_y);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "sgemv_test failed: " << e.what() << "\n";
    return 1;
  }
}
