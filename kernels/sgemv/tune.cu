#include "include/tune.cuh"
int main() {
  // ==============================================================
  //  prepare data
  // ==============================================================
  int M = 128, N = 128, K = 1024;
  thrust::host_vector<float> h_a(M * K);
  thrust::host_vector<float> h_x(N);
  thrust::host_vector<float> h_y(M);

  thrust::device_vector<float> d_a = h_a;
  thrust::device_vector<float> d_x = h_x;
  thrust::device_vector<float> d_y = h_y;

  Tunner::tune(thrust::raw_pointer_cast(d_a.data()),
               thrust::raw_pointer_cast(d_x.data()),
               thrust::raw_pointer_cast(d_y.data()), M, K);

  return 0;
}