#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"
#include "openkernels/kernels.h"

int main() {
  try {
    const int N = 4096;
    std::vector<float> h_x = make_random_f32(N);
    std::vector<float> h_y(N, 0.0f);

    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    gelu_f32(d_x, d_y, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    print_head("gelu_f32", h_y);

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "gelu_test failed: " << e.what() << "\n";
    return 1;
  }
}
