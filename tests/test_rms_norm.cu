#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"
#include "openkernels/kernels.h"

int main() {
  try {
    const int N = 16;
    const int K = 128;
    std::vector<float> h_x = make_random_f32(static_cast<size_t>(N) * K);
    std::vector<float> h_y(static_cast<size_t>(N) * K, 0.0f);

    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_y.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    rmsnorm_f32(d_x, d_y, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    print_head("rmsnorm_f32", h_y);

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "rms_norm_test failed: " << e.what() << "\n";
    return 1;
  }
}
