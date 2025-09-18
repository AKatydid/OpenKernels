#include <float.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

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

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }


void rmsnorm_f32(torch::Tensor x, torch::Tensor y){
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)

  const int N = x.size(0);
  const int K = x.size(1);

  const int BN = 2;
  const int NUM_THREAD = 128;
  dim3 grid((N + BN - 1)/BN);
  dim3 block(NUM_THREAD, BN);

  rmsnorm_f32_kernel<BN, NUM_THREAD>                                                    
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),              
                        reinterpret_cast<float *>(y.data_ptr()), N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    TORCH_BINDING_COMMON_EXTENSION(rmsnorm_f32)
}


