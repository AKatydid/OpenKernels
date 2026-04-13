#include "common_utils.cuh"

/**
 * SGEMM Shared Mem V4: Every thread calculates (TM, TN)
 */
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmKernelV4(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K, float alpha, float beta) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    __syncthreads();

    const int nTile = (K + BK - 1) / BK;
    for (int bk = 1; bk < nTile; bk++) {

        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

    int final_smem_sel = (nTile - 1) & 1;
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[final_smem_sel][tk][ty * TM / 2         ]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[final_smem_sel][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[final_smem_sel][tk][tx * TN / 2         ]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[final_smem_sel][tk][tx * TN / 2 + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}


int main(int argc, char** argv){
    if (argc < 4 || argc > 6){
        printf("Usage: %s <M> <N> <K> [warm_up] [test_iter]\n", argv[0]);
        return -1;
    }

    size_t M = atoll(argv[1]);
    size_t N = atoll(argv[2]);
    size_t K = atoll(argv[3]);
    int warmup = (argc >= 5) ? atoi(argv[4]) : 10;
    int nIter = (argc >= 6) ? atoi(argv[5]) : 200;

    if (warmup < 0 || nIter <= 0) {
        printf("warm_up must be >= 0 and test_iter must be > 0\n");
        return -1;
    }

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

    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    float msec = 0;
    float flop = 2 * M * N * K;
    
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block = dim3(BN / TN, BM / TM);
    dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);

    // warmup
    for (int i = 0; i < warmup; i++){
        float alpha = 1.0;
        float beta = 0;
        sgemmKernelV4<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    float alpha = 1.0;
    float beta = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventSynchronize(start));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < nIter; i++){
        float alpha = 1.0;
        float beta = 0;
        sgemmKernelV4<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));
    printf("Elapsed time (My SGEMM): %.2f ms / %.4f giga flops\n", msec, flop * 1e-9f / (msec * 1e-3f / nIter));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // cublas
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    alpha = 1.0;
    beta = 0;
    CHECK_CUDA(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventSynchronize(start));
    CHECK_CUDA(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, M
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));
    printf("Elapsed time (cuBLAS): %.2f ms / %.4f giga flops\n", msec, flop * 1e-9f / (msec * 1e-3f / nIter));

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
