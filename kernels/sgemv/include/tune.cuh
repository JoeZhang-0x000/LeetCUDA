#include "lib.cuh"
#include <iomanip>
#include <iostream>
#include <map>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <tuple>

#pragma once

using Policy_list =
    std::tuple<Policy<64, 1, 1>, Policy<64, 1, 4>, Policy<64, 4, 1>,
               Policy<128, 1, 1>, Policy<128, 1, 4>, Policy<128, 4, 1>,
               Policy<128, 4, 4>, Policy<256, 1, 1>, Policy<256, 1, 4>,
               Policy<256, 4, 1>, Policy<256, 4, 4>>;

std::map<std::tuple<int, int>, std::tuple<int, int, int>> best_config_map;

struct Tunner {

  template <const int I = 0>
  static void tune_step(float *a, float *x, float *y, int M, int K,
                        float &cur_min_time, std::string &cur_best_config,
                        std::tuple<int, int, int> &best_config) {
    using Cur_Policy = std::tuple_element_t<I, Policy_list>;

    // warm up
    for (int i = 0; i < 10; i++) {
      cutlass_sgemv_kernel<Cur_Policy>
          <<<(M + Cur_Policy::WARPS_PER_BLOCK - 1) /
                 Cur_Policy::WARPS_PER_BLOCK,
             Cur_Policy::BLOCK_SIZE>>>(a, x, y, M, K);
    }
    cudaDeviceSynchronize();

    // timing
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++) {
      cutlass_sgemv_kernel<Cur_Policy>
          <<<(M + Cur_Policy::WARPS_PER_BLOCK - 1) /
                 Cur_Policy::WARPS_PER_BLOCK,
             Cur_Policy::BLOCK_SIZE>>>(a, x, y, M, K);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100;

    // std::cout << "Policy<" << Cur_Policy::BLOCK_SIZE << ", "
    //           << Cur_Policy::ROWS_PER_WARP << ", " << Cur_Policy::VEC_SIZE
    //           << "> time: " << ms << " ms" << std::endl;

    if (ms < cur_min_time) {
      cur_min_time = ms;
      cur_best_config =
          std::string("Policy<") + std::to_string(Cur_Policy::BLOCK_SIZE) +
          std::string(", ") + std::to_string(Cur_Policy::ROWS_PER_WARP) +
          std::string(", ") + std::to_string(Cur_Policy::VEC_SIZE) +
          std::string(">");
      best_config = {Cur_Policy::BLOCK_SIZE, Cur_Policy::ROWS_PER_WARP,
                     Cur_Policy::VEC_SIZE};
    }

    // search next
    if constexpr (I + 1 < std::tuple_size<Policy_list>::value) {
      tune_step<I + 1>(a, x, y, M, K, cur_min_time, cur_best_config,
                       best_config);
    }
  }

  static void tune(float *a, float *x, float *y, int M, int K) {
    float min_time = 9999.0f;
    std::string cur_best_config;
    std::tuple<int, int, int> best_config = {1, 1, 1};
    tune_step(a, x, y, M, K, min_time, cur_best_config, best_config);
    // std::cout << "cur min time: " << min_time
    //           << " ms, best config: " << cur_best_config << std::endl;
    best_config_map.insert({{M, K}, best_config});
  }
};

template <int I = 0>
bool dispatch_kernel(const std::tuple<int, int, int> &config, float *a,
                     float *x, float *y, int M, int K) {
  using CurrentPolicy = std::tuple_element_t<I, Policy_list>;

  if (config == std::make_tuple(CurrentPolicy::BLOCK_SIZE,
                                CurrentPolicy::ROWS_PER_WARP,
                                CurrentPolicy::VEC_SIZE)) {
    cutlass_sgemv_kernel<CurrentPolicy>
        <<<(M + CurrentPolicy::WARPS_PER_BLOCK - 1) /
               CurrentPolicy::WARPS_PER_BLOCK,
           CurrentPolicy::BLOCK_SIZE>>>(a, x, y, M, K);
    return true;
  }
  if constexpr (I + 1 < std::tuple_size_v<Policy_list>) {
    return dispatch_kernel<I + 1>(config, a, x, y, M, K);
  }

  return false;
}

void dispatch(float *a, float *x, float *y, int M, int K) {
  if (best_config_map.find({M, K}) == best_config_map.end()) {
    std::cout << "first run, tuning...\n";
    Tunner::tune(a, x, y, M, K);
  }
  auto best_config = best_config_map.at({M, K});

  if (!dispatch_kernel(best_config, a, x, y, M, K)) {
    std::cerr << "Error: Best config found in map but no matching Policy was "
                 "found in Policy_list!"
              << std::endl;
  }
}