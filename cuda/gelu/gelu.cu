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
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define SQRT_2_PI (M_SQRT2 *M_2_SQRTPI * 0.5f)
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

__inline__ __device__ float gelu_tanh_approximate(float x){
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715 * x * x *x)));
}

// block(256)
// grid((N + 256 - 1) / 256)
__global__ void gelu_f32_kernel(float* x, float* y, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = gelu_tanh_approximate(v);
  }
}

// BN = 256
// block(256 / 4)
// grid((N + 256 - 1) / 256)
__global__ void gelu_f32x4_kernel(float* x, float* y, int N){
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;
  
  reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

  reg_y.x = gelu_tanh_approximate(reg_x.x);
  reg_y.y = gelu_tanh_approximate(reg_x.y);
  reg_y.z = gelu_tanh_approximate(reg_x.z);
  reg_y.w = gelu_tanh_approximate(reg_x.w);

  if(idx + 0 < N) 
    FLOAT4(y[idx]) = reg_y;
}


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_GELU(packed_type, th_type, element_type, n_elements)     \
  void gelu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      gelu_##packed_type##_kernel<<<grid, block>>>(                            \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        gelu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        gelu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_GELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_GELU(f32x4, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32x4)
}