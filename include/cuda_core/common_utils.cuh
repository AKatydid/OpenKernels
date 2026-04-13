#ifndef _OPENKERNELS_CUDA_CORE_BLAS_H_
#define _OPENKERNELS_CUDA_CORE_BLAS_H_

#include<stdio.h>
#include<stdlib.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include <math.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CHECK_CUDA(func)   \
{                               \
    cudaError_t e = (func);     \
    if (e != cudaSuccess){      \
         printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));    \
    }                           \
}

#endif // _OPENKERNELS_CUDA_CORE_BLAS_H_