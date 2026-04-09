#include <float.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FLOAT4(ptr) (reinterpret_cast<float*>(&(ptr))[0])

template<const int WarpSize>
__device__ __inline__ float warpReduceSum(float val){
#pragma unroll
    for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/**
 * const int BN = 2;
 * const int BK = 128;
 * dim3 grid((N + BN - 1)/BN);
 * dim3 block(BK, BN);
 */
template<const int BN, const int BK>
__global__ void rmsnorm_f32_kernel(float* x, float* out, const int N, const int K){
  const int WARP_NUM = BK / WARP_SIZE;
  __shared__ float smem[BN][WARP_NUM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;

  int lane = tx % WARP_SIZE;
  int warp_id = tx / WARP_SIZE;

  float* cur_line_addr = x + (bx * BN + ty) * K;
  float val = cur_line_addr[tx] * cur_line_addr[tx];
  for (int i = tx + BK; i < K; i += BK){
    val += cur_line_addr[i] * cur_line_addr[i];
  }

  val = warpReduceSum<WARP_SIZE>(val);

  if (lane == 0)
    smem[ty][warp_id] = val;
  __syncthreads();

  if (tx == 0){
    float norm = smem[ty][0] / K;
    for (int i = 1; i < WARP_NUM; ++i){
      norm += smem[ty][i] / K;
    }
    smem[ty][0] = norm;
  }
  __syncthreads();

  float norm_val = rsqrtf(smem[ty][0] + 1e-5);
  for (int i = tx; i < K; i += BK){
    out[(bx * BN + ty) * K + i] = cur_line_addr[i] * norm_val;
  }
}

void rmsnorm_f32(float *x, float *y, int N, int K){
  const int BN = 2;
  const int NUM_THREAD = 128;
  dim3 grid((N + BN - 1)/BN);
  dim3 block(NUM_THREAD, BN);

  rmsnorm_f32_kernel<BN, NUM_THREAD>
      <<<grid, block>>>(x, y, N, K);
}
