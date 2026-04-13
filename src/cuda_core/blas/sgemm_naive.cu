#include "common_utils.cuh"

/**
 * Optimized Version 1: Using shared mem and every thread calculates (TN, TM) elements
 */
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void SGEMM_V1(
    float* A, float* B, float* C, 
    const int M, const int N, const int K, 
    const float alpha, const float beta
){
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float sa[BM][BK];
    __shared__ float sb[BK][BN];
    
    int load_a_smem_m = tid / (BK / 4);
    int load_a_smem_k = (tid % (BK / 4)) * 4;
    int load_b_smem_k = tid / (BN / 4);
    int load_b_smem_n = (tid % (BN / 4)) * 4;

    int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
    int load_b_gmem_n = blockIdx.x * BN + load_b_smem_n;

    float tileSum[TM][TN] = {0.0};
    int nIter = (K + BK - 1) / BK;
    for (int bk = 0; bk < nIter; ++bk){
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        FLOAT4(sa[load_a_smem_m][load_a_smem_k]) = FLOAT4(A[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        FLOAT4(sb[load_b_smem_k][load_b_smem_n]) = FLOAT4(B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]);

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

    // write back to gmem
    #pragma unroll
    for (int i = 0; i < TM; ++i){
        int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j+=4){
            int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(C[store_c_gmem_addr]) = FLOAT4(tileSum[i][j]);
        }
    }
}


int main(int argc, char** argv){
    if (argc != 4){
        printf("Usage: %s <M> <N> <K>", argv[0]);
        return -1;
    }

    size_t M = atoll(argv[1]);
    size_t N = atoll(argv[2]);
    size_t K = atoll(argv[3]);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C_ref = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < M * N; ++i) h_C[i] = rand() / (float)RAND_MAX;

    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    float alpha = 1.0;
    float beta = 0.0;

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    float msec = 0;
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventSynchronize(start));
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int i = 0; i < 10; ++i){
        // naiveSGEMM<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
        SGEMM_V1<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, end));
    printf("Elapsed time (My SGEMM): %.2f ms / %.4f giga flops\n", msec, 2 * M * N * K * 1e-9f / (msec * 1e-3f / 10));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // 验证
    cublasHandle_t blas_handle;
    cublasCreate_v2(&blas_handle);

    cublasSgemm_v2(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);

    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C, bytes_C, cudaMemcpyDeviceToHost));
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

    cublasDestroy_v2(blas_handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    return 0;
}