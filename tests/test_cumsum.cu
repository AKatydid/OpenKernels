#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"
#include "openkernels/kernels.h"

int main() {
  try {
    const size_t N = 1 << 20;
    std::vector<int32_t> h_in = make_random_i32(N, 1, 3);
    std::vector<int32_t> h_out(N, 0);

    int32_t *d_in = nullptr;
    int32_t *d_out = nullptr;
    int32_t *d_buf = nullptr;

    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_buf, 2 * N * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));

    ScanThenFan(d_in, d_buf, d_out, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    print_head("cumsum", h_out);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_buf));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "cumsum_test failed: " << e.what() << "\n";
    return 1;
  }
}
