#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) (((row) * ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)   \
{							    \
    cudaError_t e = (func);	    \
    if(e != cudaSuccess)	    \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));    \
}

/**
 * SGEMM Shared Mem
 */
__global__ void sgemmKernelV1(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K, const float alpha, const float beta
){
    const int TILE_SIZE = 16;

    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float sa[TILE_SIZE][TILE_SIZE];
    __shared__ float sb[TILE_SIZE][TILE_SIZE];

    float tileSum = 0.0f;
    int nIter = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < nIter; ++i){
        sa[threadIdx.x][threadIdx.y] = (m < M && (i * TILE_SIZE + threadIdx.y) < K) ? a[OFFSET(m, i * TILE_SIZE + threadIdx.y, K)] : 0.0f;
        sb[threadIdx.x][threadIdx.y] = (n < N && (i * TILE_SIZE + threadIdx.x) < K) ? b[OFFSET(i * TILE_SIZE + threadIdx.x, n, N)] : 0.0f;
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j){
            tileSum += sa[threadIdx.x][j] * sb[j][threadIdx.y];
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        c[m * N + n] = alpha * tileSum + beta * c[m * N + n];
    }
}

int main(int argc, char** argv){
    if (argc != 4){
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
        return -1;
    }

    size_t M = atoll(argv[1]);
    size_t N = atoll(argv[2]);
    size_t K = atoll(argv[3]);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float* h_A = (float *)malloc(bytes_A);
    float* h_B = (float *)malloc(bytes_B);
    float* h_C = (float *)malloc(bytes_C);
    float* h_C_ref = (float *)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    for (size_t i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (size_t i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;
    for (size_t i = 0; i < M * N; i++) h_C[i] = rand() / (float)RAND_MAX;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    float msec = 0;
    float flop = 2 * M * N * K;
    int warmup = 10;
    int nIter = 200;
    
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // warmup
    for (int i = 0; i < warmup; i++){
        float alpha = 1.0;
        float beta = 0;
        sgemmKernelV1<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    float alpha = 1.0;
    float beta = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventRecord(start));

    for (int i = 0; i < nIter; i++){
        sgemmKernelV1<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("Elapsed time (My SGEMM): %.2f ms / %.4f giga flops\n", msec, flop * 1e-9f / (msec * 1e-3f / nIter));

    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // cublas
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    alpha = 1.0;
    beta = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, M
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("Elapsed time (cuBLAS): %.2f ms / %.4f giga flops\n", msec, flop * 1e-9f / (msec * 1e-3f / nIter));

    checkCudaErrors(cudaMemcpy(h_C_ref, d_C, bytes_C, cudaMemcpyDeviceToHost));

    double eps = 1.e-6; 
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C_ref[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C_ref[col * M + row], eps);
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");

    cublasDestroy(blas_handle); 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
