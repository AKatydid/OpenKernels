#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"
#include "openkernels/kernels.h"

int main(int argc, char **argv) {
  try {
    const auto positionals = collect_positionals(argc, argv);
    const int N = positional_arg<int>(positionals, 0, 16);
    const int K = positional_arg<int>(positionals, 1, 128);
    const TestRunOptions opts = parse_run_options(argc, argv);

    std::vector<float> h_x = make_random_f32(static_cast<size_t>(N) * K);
    std::vector<float> h_gpu(static_cast<size_t>(N) * K, 0.0f);
    std::vector<float> h_cpu;

    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    auto run_gpu = [&]() {
      rmsnorm_f32(d_x, d_y, N, K);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    };

    if (opts.verify) {
      run_gpu();
      CHECK_CUDA(cudaMemcpy(h_gpu.data(), d_y, h_gpu.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
      rmsnorm_ref(h_x, h_cpu, N, K);
      validate_close(h_gpu, h_cpu, "rmsnorm_f32", 1e-4, 1e-4);
      std::cout << "VERIFY rmsnorm_f32 PASS\n";
    }

    if (opts.bench) {
      CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      const double gpu_ms = benchmark_kernel(opts.iters, run_gpu);
      std::cout << "BENCH rmsnorm_f32 gpu_ms=" << gpu_ms << "\n";
    }

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "rms_norm_test failed: " << e.what() << "\n";
    return 1;
  }
}
