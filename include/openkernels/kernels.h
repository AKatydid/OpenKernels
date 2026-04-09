#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

void relu_f32(float *x, float *y, int N);
void relu_f32x4(float *x, float *y, int N);
void relu_f16(half *x, half *y, int N);
void relu_f16x2(half *x, half *y, int N);
void relu_f16x8(half *x, half *y, int N);
void relu_f16x8_pack(half *x, half *y, int N);

void gelu_f32(float *x, float *y, int N);
void gelu_f32x4(float *x, float *y, int N);

void rmsnorm_f32(float *x, float *y, int N, int K);

void sgemv_k32_f32(float *a, float *x, float *y, int M, int K);
void sgemv_k128_f32x4(float *a, float *x, float *y, int M, int K);
void sgemv_k16_f32(float *a, float *x, float *y, int M, int K);
void sgemv_splitk_f32(float *a, float *x, float *y, int M, int K);
void sgemv_splitk_smem_f32(float *a, float *x, float *y, int M, int K);

void softmax_f32_per_token(float *x, float *y, int S, int H);
void softmax_f32x4_per_token(float *x, float *y, int S, int H);
void safe_softmax_f32_per_token(float *x, float *y, int S, int H);
void safe_softmax_f32x4_per_token(float *x, float *y, int S, int H);
void safe_softmax_f16_f32_per_token(half *x, half *y, int S, int H);
void safe_softmax_f16x2_f32_per_token(half *x, half *y, int S, int H);
void safe_softmax_f16x8_pack_f32_per_token(half *x, half *y, int S, int H);
void online_safe_softmax_f32_per_token(float *x, float *y, int S, int H);
void online_safe_softmax_f32x4_pack_per_token(float *x, float *y, int S, int H);

void ScanThenFan(const int32_t *input, int32_t *buffer, int32_t *output, size_t n);
