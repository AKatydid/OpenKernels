#define OPENKERNELS_DISABLE_MAIN
#include "src/cuda_core/blas/sgemm_v1_tiling.cu"

#include <iostream>
#include <vector>

#include "openkernels/common_utils.cuh"

int main(int argc, char **argv) {
  try {
    const auto positionals = collect_positionals(argc, argv);
    const int M = positional_arg<int>(positionals, 0, 256);
    const int N = positional_arg<int>(positionals, 1, 256);
    const int K = positional_arg<int>(positionals, 2, 256);
    const TestRunOptions opts = parse_run_options(argc, argv);

    std::vector<float> h_a = make_random_f32(static_cast<size_t>(M) * K);
    std::vector<float> h_b = make_random_f32(static_cast<size_t>(K) * N);
    std::vector<float> h_c(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> h_gpu(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> h_cpu;

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, h_c.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    constexpr int BM = 16;
    constexpr int BN = 16;
    dim3 block(BM, BN);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto run_gpu = [&]() {
      sgemmKernelV1<<<grid, block>>>(d_a, d_b, d_c, M, N, K, alpha, beta);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    };

    if (opts.verify) {
      run_gpu();
      CHECK_CUDA(cudaMemcpy(h_gpu.data(), d_c, h_gpu.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
      sgemm_ref(h_a, h_b, h_cpu, M, N, K, alpha, beta);
      validate_close(h_gpu, h_cpu, "sgemm_v1", 1e-3, 1e-3);
      std::cout << "VERIFY sgemm_v1 PASS\n";
    }

    if (opts.bench) {
      CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      const double gpu_ms = benchmark_kernel(opts.iters, run_gpu);
      std::cout << "BENCH sgemm_v1 gpu_ms=" << gpu_ms << "\n";
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "sgemm_test failed: " << e.what() << "\n";
    return 1;
  }
}
