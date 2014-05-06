#include <stdlib.h>

int main() {
  int N = 10000000;
  int *A = malloc(sizeof(int) * N), 
      *B = malloc(sizeof(int) * N), 
      *C = malloc(sizeof(int) * N);
  
  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = N - i;
    C[i] = 0;
  }

  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }

  free(A); free(B); free(C);
}