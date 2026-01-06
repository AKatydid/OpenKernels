#ifndef _CUDA_C_H_
#define _CUDA_C_H_

#include<stdio.h>
#include<stdlib.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define checkCUDAErrors(func)   \
{                               \
    cudaError_t e = (func);     \
    if (e != cudaSuccess){      \
         printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));    \
    }                           \
}
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#endif
