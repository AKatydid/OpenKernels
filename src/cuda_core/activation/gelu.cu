#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define SQRT_2_PI (M_SQRT2 *M_2_SQRTPI * 0.5f)
#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])

/**
 * GELU activation function:
 *   raw: GELU(x) = x * 0.5 * (1.0 + erf(x * 1/sqrt(2)))
 *   approximate: GELU(x) = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */

__inline__ __device__ float gelu_tanh_approximate(float x){
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715 * x * x * x)));
}


__global__ void gelu_f32_kernel(float* x, float* y, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = gelu_tanh_approximate(v);
  }
}


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


void gelu_f32(float *x, float *y, int N) {
  dim3 block(256);
  dim3 grid((N + 256 - 1) / 256);
  gelu_f32_kernel<<<grid, block>>>(x, y, N);
}

void gelu_f32x4(float *x, float *y, int N) {
  dim3 block(256 / 4);
  dim3 grid((N + 256 - 1) / 256);
  gelu_f32x4_kernel<<<grid, block>>>(x, y, N);
}
