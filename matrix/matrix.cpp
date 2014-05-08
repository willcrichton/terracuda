#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

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

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }

      C[i * N + j] = sum;
    }
  }

  return 0;
}