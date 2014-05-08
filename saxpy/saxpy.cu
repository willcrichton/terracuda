#include <stdlib.h>
#include <cuda.h>

__global__ void saxpy(int *X, int *Y, int a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  Y[i] += a * X[i];
}

int main() {
  int N = 10000000;
  int *X = (int*) malloc(sizeof(int) * N), 
      *Y = (int*) malloc(sizeof(int) * N),
       a = 2,
      *dX, *dY;
  
  cudaMalloc(&dX, sizeof(int) * N);
  cudaMalloc(&dY, sizeof(int) * N);

  for (int i = 0; i < N; i++) {
    X[i] = i;
    Y[i] = N - i;
  }

  cudaMemcpy(dX, X, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dY, Y, sizeof(int) * N, cudaMemcpyHostToDevice);

  dim3 blockDim(64, 1, 1);
  dim3 gridDim(N / blockDim.x, 1, 1);
  saxpy<<<gridDim, blockDim>>>(dX, dY, a);
  cudaDeviceSynchronize();

  cudaMemcpy(dY, Y, sizeof(int) * N, cudaMemcpyDeviceToHost);

  free(X); free(Y);
  cudaFree(dX); cudaFree(dY);
}