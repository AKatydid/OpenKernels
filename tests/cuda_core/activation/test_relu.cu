#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"
#include "openkernels/kernels.h"

int main(int argc, char **argv) {
  try {
    const auto positionals = collect_positionals(argc, argv);
    const int N = positional_arg<int>(positionals, 0, 4096);
    const TestRunOptions opts = parse_run_options(argc, argv);

    std::vector<float> h_x = make_random_f32(N);
    std::vector<float> h_gpu(N, 0.0f);
    std::vector<float> h_cpu;

    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    auto run_gpu = [&]() {
      relu_f32(d_x, d_y, N);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    };

    if (opts.verify) {
      run_gpu();
      CHECK_CUDA(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(float),
                            cudaMemcpyDeviceToHost));
      relu_ref(h_x, h_cpu);
      validate_close(h_gpu, h_cpu, "relu_f32", 1e-5, 1e-5);
      std::cout << "VERIFY relu_f32 PASS\n";
    }

    if (opts.bench) {
      CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(float),
                            cudaMemcpyHostToDevice));
      const double gpu_ms = benchmark_kernel(opts.iters, run_gpu);
      std::cout << "BENCH relu_f32 gpu_ms=" << gpu_ms << "\n";
    }

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "relu_test failed: " << e.what() << "\n";
    return 1;
  }
}
