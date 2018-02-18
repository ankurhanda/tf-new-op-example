#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void AddOneKernel(const int* in, const int N, int* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] + 1;
  }
}

void AddOneKernelLauncher(const int* in, const int N, int* out) {
  AddOneKernel<<<32, 256>>>(in, N, out);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}