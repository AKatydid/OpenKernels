#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"
#include "openkernels/kernels.h"

int main(int argc, char **argv) {
  try {
    const auto positionals = collect_positionals(argc, argv);
    const size_t N = positional_arg<size_t>(positionals, 0, 1u << 20);
    const TestRunOptions opts = parse_run_options(argc, argv);

    std::vector<int32_t> h_in = make_random_i32(N, 1, 3);
    std::vector<int32_t> h_gpu(N, 0);
    std::vector<int32_t> h_cpu;

    int32_t *d_in = nullptr;
    int32_t *d_out = nullptr;
    int32_t *d_buf = nullptr;

    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_buf, 2 * N * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    auto run_gpu = [&]() {
      ScanThenFan(d_in, d_buf, d_out, N);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    };

    if (opts.verify) {
      run_gpu();
      CHECK_CUDA(cudaMemcpy(h_gpu.data(), d_out, N * sizeof(int32_t),
                            cudaMemcpyDeviceToHost));
      cumsum_ref(h_in, h_cpu);
      validate_close(h_gpu, h_cpu, "cumsum", 0.0, 0.0);
      std::cout << "VERIFY cumsum PASS\n";
    }

    if (opts.bench) {
      CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(int32_t),
                            cudaMemcpyHostToDevice));
      const double gpu_ms = benchmark_kernel(opts.iters, run_gpu);
      std::cout << "BENCH cumsum gpu_ms=" << gpu_ms << "\n";
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_buf));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "cumsum_test failed: " << e.what() << "\n";
    return 1;
  }
}
