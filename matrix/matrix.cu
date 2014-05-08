#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>
#include <cuda.h>
#include <iterator>

using namespace std;

__global__ void multiply(int *A, int *B, int *C, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i = idx / N, j = idx % N;
  int sum = 0;
  for (int k = 0; k < N; k++) {
    sum += A[i * N + k] * B[k * N + j];
  }

  C[idx] = sum;
}

int main() {
  string line;
  int N = -1;
  int *A, *B, *C, *cur;
  int count = 0;
  while(getline(cin, line)) {
    if (N == -1) { 
      N = atoi(line.c_str());
      A = new int[N * N];
      B = new int[N * N];
      C = new int[N * N];
      cur = A;
    } else {
      vector<string> nums;
      istringstream iss(line);
      copy(istream_iterator<string>(iss),
           istream_iterator<string>(),
           back_inserter(nums));

      for (int i = 0; i < nums.size(); i++) {
        cur[count * N + i] = atoi(nums[i].c_str());
      }

      count++;
      if (count == N) {
        count = 0;
        cur = B;
      }
    }
  }

  int *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(int) * N * N);
  cudaMalloc(&dB, sizeof(int) * N * N);
  cudaMalloc(&dC, sizeof(int) * N * N);

  cudaMemcpy(dA, A, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(int) * N * N, cudaMemcpyHostToDevice);

  dim3 blockDim(64, 1, 1);
  dim3 gridDim(N * N / blockDim.x, 1, 1);
  multiply<<<gridDim, blockDim>>>(dA, dB, dC, N);

  cudaMemcpy(dA, A, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(dB, B, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(dC, C, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

  return 0;
}