#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"
#include "openkernels/kernels.h"

int main(int argc, char **argv) {
  try {
    const auto positionals = collect_positionals(argc, argv);
    const int M = positional_arg<int>(positionals, 0, 256);
    const int K = positional_arg<int>(positionals, 1, 128);
    const TestRunOptions opts = parse_run_options(argc, argv);

    if (K % 128 != 0) {
      throw std::runtime_error("test_sgemv only supports K multiples of 128");
    }

    std::vector<float> h_a = make_random_f32(static_cast<size_t>(M) * K);
    std::vector<float> h_x = make_random_f32(K);
    std::vector<float> h_gpu(M, 0.0f);
    std::vector<float> h_cpu;

    float *d_a = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    auto run_gpu = [&]() {
      sgemv_k128_f32x4(d_a, d_x, d_y, M, K);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    };

    if (opts.verify) {
      run_gpu();
      CHECK_CUDA(cudaMemcpy(h_gpu.data(), d_y, h_gpu.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
      sgemv_ref(h_a, h_x, h_cpu, M, K);
      validate_close(h_gpu, h_cpu, "sgemv_k128_f32x4", 1e-4, 1e-4);
      std::cout << "VERIFY sgemv_k128_f32x4 PASS\n";
    }

    if (opts.bench) {
      CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      const double gpu_ms = benchmark_kernel(opts.iters, run_gpu);
      std::cout << "BENCH sgemv_k128_f32x4 gpu_ms=" << gpu_ms << "\n";
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "sgemv_test failed: " << e.what() << "\n";
    return 1;
  }
}
