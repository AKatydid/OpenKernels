#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ int32_t ScanWarp(int32_t val) {
  int32_t lane = threadIdx.x & 31;
  #pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int32_t n = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset) val += n;
  }
  return val;
}

__device__ __forceinline__ int32_t ScanBlock(int32_t val) {
    int32_t lane = threadIdx.x & 31;
    int32_t warp_id = threadIdx.x >> 5;
    extern __shared__ int32_t smem[];
    
    val = ScanWarp(val);  // scan each warp
    __syncthreads();
    
    if (lane == 31) smem[warp_id] = val;  // write sum of each warp to warp_sum
    __syncthreads();

    if (warp_id == 0) smem[lane] = ScanWarp(smem[lane]);  // scan warp_sum
    __syncthreads();

    if (warp_id > 0) val += smem[warp_id - 1];
    __syncthreads();
    return val;
}

__global__ void ScanAndWritePartSumKernel(
    const int32_t* input, int32_t* part, int32_t* output, size_t n, size_t part_num
){
  for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
    size_t index = part_i * blockDim.x + threadIdx.x;
    int32_t val = index < n ? input[index] : 0;
    val = ScanBlock(val);
    __syncthreads();
    if (index < n) output[index] = val;
    if (threadIdx.x == blockDim.x - 1) part[part_i] = val;
  }
}

__global__ void AddBaseSumKernel(int32_t* part, int32_t* output, size_t n, size_t part_num){
  for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
    if (part_i == 0) continue;
    int32_t index = part_i * blockDim.x + threadIdx.x;
    if (index < n) output[index] += part[part_i - 1];
  }
}

// 递归调用，buffer 总共开 2n * sizeof(int32_t) 即可缓存所有递归中间结果
void ScanThenFan(const int32_t* input, int32_t* buffer, int32_t* output, size_t n) {
  size_t part_size = 1024;  // tuned
  size_t part_num = (n + part_size - 1) / part_size;
  size_t block_num = std::min<size_t>(part_num, 128);
  int32_t* part = buffer;   // buffer[0:part_num] to save the metric of part
  size_t shm_size = 32 * sizeof(int32_t);
  ScanAndWritePartSumKernel<<<block_num, part_size, shm_size>>>(input, part, output, n, part_num);

  if (part_num >= 2) {
    ScanThenFan(part, buffer + part_num, part, part_num);
    AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num);
  }
}