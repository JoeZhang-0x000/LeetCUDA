#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cub/cub.cuh>

#pragma once
template<int _BLOCK_SIZE, int _ROWS_PER_WARP, int _VEC_SIZE>
struct Policy{
  static constexpr int BLOCK_SIZE = _BLOCK_SIZE;
  static constexpr int ROWS_PER_WARP = _ROWS_PER_WARP;
  static constexpr int VEC_SIZE = _VEC_SIZE;

  static constexpr int WARP_SIZE = 32;
  static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

  static_assert(BLOCK_SIZE % WARP_SIZE == 0, "BLOCK_SIZE must be divisible by WARP_SIZE");
};

template<typename T, const int VEC_SZIE>
__align__(sizeof(T) * VEC_SZIE) struct VEC{
  T data[VEC_SZIE];
};

template<typename T, const int VEC_SZIE>
__device__ __forceinline__  VEC<T, VEC_SZIE> load_vector(const T * addr){
  return  *reinterpret_cast<const VEC<T, VEC_SZIE>*>(addr);
}

template<typename Policy>
__global__ void cutlass_sgemv_kernel(float *a, float *x, float *y, int M, int K){
  using WarpReduce = cub::WarpReduce<float>;
  __shared__ typename WarpReduce::TempStorage temp_storage[Policy::WARPS_PER_BLOCK];

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int global_warp_id = (blockIdx.x * Policy::WARPS_PER_BLOCK) + warp_id;

  int row_start = global_warp_id * Policy::ROWS_PER_WARP;

  #pragma unroll
  for (int r = 0; r < Policy::ROWS_PER_WARP; r++){
    int current_row = row_start + r;
    if (current_row >= M) return;
    float row_sum = 0.0f;
    int stride = Policy::WARP_SIZE * Policy::VEC_SIZE;

    for (int k = lane_id * Policy::VEC_SIZE; k < K; k+= stride){
      auto vec_a = load_vector<float, Policy::VEC_SIZE>(&a[current_row * K + k]);
      auto vec_x = load_vector<float, Policy::VEC_SIZE>(&x[k]);

      #pragma unroll
      for (int i=0;i<Policy::VEC_SIZE;i++){
        row_sum += vec_a.data[i] * vec_x.data[i];
      }
    }
    float result = WarpReduce(temp_storage[warp_id]).Sum(row_sum);

    if (lane_id == 0){
      y[current_row] = result;
    }
  }
}



