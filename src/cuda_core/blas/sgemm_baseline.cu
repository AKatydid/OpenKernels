#include "common_utils.cuh"

int sgemm_cpu(const float *A, const float *B, float *C, int M, int N, int K,
              float alpha, float beta) {
	if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || N <= 0 || K <= 0) {
		return -1;
	}

	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			float sum = 0.0f;
			for (int k = 0; k < K; ++k) {
				sum += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
			}
			C[OFFSET(m, n, N)] = alpha * sum + beta * C[OFFSET(m, n, N)];
		}
	}
	return 0;
}

// Row-major SGEMM via cuBLAS with an internally managed handle.
cublasStatus_t sgemm_cublas(const float *d_A, const float *d_B, float *d_C,
                            int M, int N, int K, float alpha, float beta) {
	cublasHandle_t handle = nullptr;
	cublasStatus_t create_st = cublasCreate(&handle);
	if (create_st != CUBLAS_STATUS_SUCCESS) {
		printf("cuBLAS handle creation failed, status=%d\n", (int)create_st);
		return create_st;
	}

	cublasStatus_t gemm_st = sgemm_cublas_with_handle(
        handle, d_A, d_B, d_C, M, N, K, alpha, beta
    );
	cublasStatus_t destroy_st = cublasDestroy(handle);
	if (destroy_st != CUBLAS_STATUS_SUCCESS) {
		printf("cuBLAS handle destroy failed, status=%d\n", (int)destroy_st);
		if (gemm_st == CUBLAS_STATUS_SUCCESS) {
			return destroy_st;
		}
	}

	return gemm_st;
}

cublasStatus_t sgemm_cublas_with_handle(cublasHandle_t handle, const float *d_A,
                                        const float *d_B, float *d_C, int M,
                                        int N, int K, float alpha, float beta) {
	if (handle == nullptr || d_A == nullptr || d_B == nullptr || d_C == nullptr ||
		M <= 0 || N <= 0 || K <= 0) {
		return CUBLAS_STATUS_INVALID_VALUE;
	}

	cublasStatus_t st = cublasSgemm_v2(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N
    );
	if (st != CUBLAS_STATUS_SUCCESS) {
		printf("cuBLAS SGEMM failed, status=%d\n", (int)st);
	}
	return st;
}

/**
 * C = αAB + βC, every thread calculates one element
 */
__global__ void naiveSGEMM(
    float* A, float* B, float* C, 
    const int M, const int N, const int K, 
    const float alpha, const float beta
){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        float sum = 0.0;

        for (int i = 0; i < K; ++i){
            sum += A[OFFSET(row, i, K)] * B[OFFSET(i, col, N)];
        }

        C[OFFSET(row, col, N)] = alpha * sum + beta * C[OFFSET(row, col, N)];
    }
}