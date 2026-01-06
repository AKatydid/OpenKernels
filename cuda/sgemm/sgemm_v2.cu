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
 * SGEMM Shared Mem V2: Every thread calculates (TM, TN)
 */
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmKernelV2(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K, const float alpha, const float beta
){
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int load_a_smem_m = tid / (BK / 4);           // tid / 2
    int load_a_smem_k = (tid % (BK / 4)) * 4;     // (tid % 2 == 0) ? 0 : 4
    int load_b_smem_k = tid >> 5;                 // tid / 32
    int load_b_smem_n = (tid & 31) << 2;          // (tid % 32) * 4

    int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
    int load_b_gmem_n = blockIdx.x * BN + load_b_smem_n;

    float tileSum[TM][TN] = {0.0};
    __shared__ float sa[BM][BK];
    __shared__ float sb[BK][BN];

    int nIter = (K + BK - 1) / BK;
    for (int bk = 0; bk < nIter; ++bk){
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        FLOAT4(sa[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        FLOAT4(sb[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k){
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm){
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    tileSum[tm][tn] += sa[threadIdx.y * TM + tm][k] * sb[k][threadIdx.x * TN + tn];
                }
            }
        }

        __syncthreads();
    }

    // write back to global mem
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(tileSum[i][j]);
        }
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
    
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block = dim3(BN / TN, BM / TM);
    dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);

    // warmup
    // for (int i = 0; i < warmup; i++){
    //     float alpha = 1.0;
    //     float beta = 0;
    //     sgemmKernelV2<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    // }

    float alpha = 1.0;
    float beta = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventRecord(start));

    for (int i = 0; i < nIter; i++){
        float alpha = 1.0;
        float beta = 0;
        sgemmKernelV2<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
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