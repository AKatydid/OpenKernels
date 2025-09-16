#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>


#define WARP_SIZE 32
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <unsigned int WarpSize>
__device__ __inline__ float warp_reduce_sum_f32(float sum){
    if (WarpSize >= 32) sum += __shfl_xor_sync(0xffffffff, sum, 16);
    if (WarpSize >= 16) sum += __shfl_xor_sync(0xffffffff, sum, 8);
    if (WarpSize >= 8) sum += __shfl_xor_sync(0xffffffff, sum, 4);
    if (WarpSize >= 4) sum += __shfl_xor_sync(0xffffffff, sum, 2);
    if (WarpSize >= 2) sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}

/**
 * block(32, 4)
 * grid((M+4-1) / 4)
 */
__global__ void sgemv_k32_f32_kernel(float* a, float* b, float* c, const int M, const int K){
  int tx = threadIdx.x; // 0~31
  int ty = threadIdx.y; // 0~4
  int bx = blockIdx.x;  // 0~M/4
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;

  if (m < M){
    float sum = 0;
    int niter = (K + WARP_SIZE -1) / WARP_SIZE;
    for (int i = 0; i < niter; ++i){
      int tid_in_block = i * WARP_SIZE + lane;
      sum += a[OFFSET(m, tid_in_block, K)] * b[tid_in_block];
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) c[m] = sum;
  }
}

/**
 * block(32, 4)
 * grid((M+4-1) / 4)
 */
__global__ void sgemv_k128_f32x4_kernel(float *a, float *b, float *c, int M, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = bx * blockDim.y + ty;

  if (m < M){
    float sum = 0;
    int niter = (K + WARP_SIZE * 4 - 1) / (WARP_SIZE * 4);
    for (int w = 0; w < niter; w++){
      int tid_in_block = 4 * (w * WARP_SIZE + lane);
      float4 cur_a = FLOAT4(a[OFFSET(m, tid_in_block, K)]);
      float4 cur_b = FLOAT4(b[tid_in_block]);
      sum += cur_a.x * cur_b.x;
      sum += cur_a.y * cur_b.y;
      sum += cur_a.z * cur_b.z;
      sum += cur_a.w * cur_b.w;
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if(lane == 0) c[m] = sum;
  }
}

/**
 * block(32, 4)
 * grid( (K + 4*2 -1) / (4*2) )
 */
template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16_f32_kernel(float *a, float *b, float *c, int M, int K) {
    constexpr int WARP_K = (WARP_SIZE + ROW_PER_WARP -1) / ROW_PER_WARP;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    int lane = tx % WARP_SIZE;
    int inner_col = lane % WARP_K;
    int inner_row = lane / WARP_K;
    int m = bx * blockDim.y * ROW_PER_WARP + ty * ROW_PER_WARP + inner_row;

    if (m < M){
      float sum = 0;
      sum = a[OFFSET(m, inner_col, K)] * b[inner_col];
      sum = warp_reduce_sum_f32<WARP_K>(sum);
      if (inner_col == 0) c[m] = sum;
    }
}

/**
 * Every Blcok calculate (BM, SPLIT_K)
 * const int BM = 4;
 * const int BK = WARP_SZIE;
 * const int SPLIT_K = 2048;
 * dim3 block(WARP_SZIE ,BM);
 * dim3 grid((K + SPLIT_K -1) / SPLIT_K, (M + BM -1) / BM);
 */
template <const int BM, const int BK, const int SPLIT_K>
__global__ void sgemv_splitk_f32_kernel(float *a, float *b, float *c, int M, int K){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int m = by * BM + ty;
  int lane = tx % WARP_SIZE;

  if (m < M){
    float sum = 0;
    int niter = SPLIT_K / WARP_SIZE / 4; 
    for (int w = 0; w < niter; ++w){ 
      int k = bx * SPLIT_K + (w * WARP_SIZE + lane) * 4;
      float4 reg_x = FLOAT4(a[OFFSET(m, k, K)]);
      float4 reg_y = FLOAT4(b[k]);
      sum += reg_x.x * reg_y.x;
      sum += reg_x.y * reg_y.y;
      sum += reg_x.z * reg_y.z;
      sum += reg_x.w * reg_y.w;
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) atomicAdd(&c[m], sum);
  }
}

/**
 * Every Blcok calculate (BM, SPLIT_K)
 * const int BM = 4;
 * const int BK = WARP_SZIE;
 * const int SPLIT_K = 2048;
 * dim3 block(WARP_SZIE ,BM);
 * dim3 grid((K + SPLIT_K -1) / SPLIT_K, (M + BM -1) / BM);
 * size_t shared_mem_bytes = SPLIT_K * sizeof(float);
 */

template <const int BM, const int BK, const int SPLIT_K>
__global__ void sgemv_splitk_smem_f32_kernel(const float * __restrict__ A,
                                             const float * __restrict__ B,
                                             float * __restrict__ C,
                                             int M, int K) {
    extern __shared__ float smem_b[];
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x; // 0..BK-1
    const int ty = threadIdx.y; // 0..BM-1

    const int block_threads = BK * BM;

    const int tid = ty * BK + tx;
    const int lane = tx;
    const int m = by * BM + ty;

    // compute tile range in K dimension
    const int k_start = bx * SPLIT_K;
    const int k_end   = min(k_start + SPLIT_K, K);
    const int tile_len = k_end - k_start;

    for (int idx = tid; idx < tile_len; idx += block_threads) {
        smem_b[idx] = B[k_start + idx];
    }
    __syncthreads();

    if (m < M){
      float sum = 0.0f;
      for (int k_local = lane; k_local < tile_len; k_local += BK) {
          float a_val = A[m * K + (k_start + k_local)];
          float b_val = smem_b[k_local];
          sum += a_val * b_val;
      }

      sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
      if (lane == 0) {
          atomicAdd(&C[m], sum);
      }
    }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define ASSERT_K_IS_MULTIBLE_OF(V)                                             \
  if (K % (V) != 0) {                                                          \
    throw std::runtime_error("K must be multiple of " #V);                     \
  }

#define ASSERT_K_IS_EQUAL_OF(V)                                                \
  if (K != (V)) {                                                              \
    throw std::runtime_error("K must be " #V);                                 \
  }

void sgemv_k32_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(32)

  dim3 block(32, 4);
  dim3 grid((M + 4 - 1) / 4);

  sgemv_k32_f32_kernel<<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                                        reinterpret_cast<float *>(x.data_ptr()),
                                        reinterpret_cast<float *>(y.data_ptr()),
                                        M, K);
}

void sgemv_k128_f32x4(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(128)

  dim3 block(32, 4);
  dim3 grid((M + 4 - 1) / 4);

  sgemv_k128_f32x4_kernel<<<grid, block>>>(
      reinterpret_cast<float *>(a.data_ptr()),
      reinterpret_cast<float *>(x.data_ptr()),
      reinterpret_cast<float *>(y.data_ptr()), M, K);
}

void sgemv_k16_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_EQUAL_OF(16)

  constexpr int NUM_THREADS = 128;
  constexpr int ROW_PER_WARP = 2;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE; // 4
  constexpr int NUM_ROWS = NUM_WARPS * ROW_PER_WARP; // 4 * 2 = 8

  dim3 block(32, NUM_WARPS);
  dim3 grid((M + NUM_ROWS - 1) / NUM_ROWS);

  sgemv_k16_f32_kernel<ROW_PER_WARP>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(x.data_ptr()),
                        reinterpret_cast<float *>(y.data_ptr()), M, K);
}

void sgemv_splitk_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)

  y.zero_();

  const int BM = 4;
  const int BK = 32;
  const int SPLIT_K = 1024;
  dim3 block(BK ,BM);
  dim3 grid((K + SPLIT_K -1) / SPLIT_K, (M + BM -1) / BM);

  sgemv_splitk_f32_kernel<BM, BK, SPLIT_K>
      <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(x.data_ptr()),
                        reinterpret_cast<float *>(y.data_ptr()), M, K);

}

void sgemv_splitk_smem_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)

  y.zero_();

  const int BM = 8;
  const int BK = 32;
  const int SPLIT_K = 512;
  dim3 block(BK ,BM);
  dim3 grid((K + SPLIT_K -1) / SPLIT_K, (M + BM -1) / BM);
  size_t shared_mem_bytes = SPLIT_K * sizeof(float);

  sgemv_splitk_smem_f32_kernel<BM, BK, SPLIT_K>
      <<<grid, block, shared_mem_bytes>>>(reinterpret_cast<float *>(a.data_ptr()),
                        reinterpret_cast<float *>(x.data_ptr()),
                        reinterpret_cast<float *>(y.data_ptr()), M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sgemv_k32_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemv_k128_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sgemv_k16_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemv_splitk_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemv_splitk_smem_f32)
}