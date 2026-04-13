#ifndef _OPENKERNELS_CONTEXT_H_
#define _OPENKERNELS_CONTEXT_H_

#pragma once

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class OpenKernelsContext {
public:
    static OpenKernelsContext& instance() {
        static OpenKernelsContext ctx;
        return ctx;
    }

    void initialize(int device_idx = 0, bool deterministic = true) {
        if (initialized_) {
            return;
        }

        if (deterministic) {
            srand(0);
        }

        cudaSetDevice(device_idx);
        cudaGetDeviceProperties(&device_prop_, device_idx);

        cublasCreate(&cublas_handle_);
        cudaMalloc(&cublaslt_workspace_, cublaslt_workspace_size_);

        int enable_tf32 = device_prop_.major >= 8 ? 1 : 0;
        cublas_compute_type_ = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
        cublasSetMathMode(cublas_handle_, cublas_math_mode);

        initialized_ = true;
    }

    void shutdown() {
        if (!initialized_) {
            return;
        }

        if (cublaslt_workspace_ != NULL) {
            cudaFree(cublaslt_workspace_);
            cublaslt_workspace_ = NULL;
        }
        cublasDestroy(cublas_handle_);
        initialized_ = false;
    }

    bool is_initialized() const { return initialized_; }

    cublasHandle_t cublas_handle() const { return cublas_handle_; }
    void* cublaslt_workspace() const { return cublaslt_workspace_; }
    size_t cublaslt_workspace_size() const { return cublaslt_workspace_size_; }
    cublasComputeType_t cublas_compute_type() const { return cublas_compute_type_; }

    int cuda_arch_major() const { return device_prop_.major; }
    int cuda_arch_minor() const { return device_prop_.minor; }
    int cuda_num_sms() const { return device_prop_.multiProcessorCount; }
    int cuda_threads_per_sm() const { return device_prop_.maxThreadsPerMultiProcessor; }

private:
    OpenKernelsContext() = default;

    bool initialized_ = false;
    size_t cublaslt_workspace_size_ = 32 * 1024 * 1024;
    void* cublaslt_workspace_ = NULL;
    cublasComputeType_t cublas_compute_type_ = CUBLAS_COMPUTE_32F;
    cublasHandle_t cublas_handle_ = nullptr;
    cudaDeviceProp device_prop_{};
};

#endif // _OPENKERNELS_CONTEXT_H_