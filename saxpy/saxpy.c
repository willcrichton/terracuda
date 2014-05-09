#include <stdlib.h>

int main() {
  int N = 10000000;
  int *X = malloc(sizeof(int) * N), 
      *Y = malloc(sizeof(int) * N), 
       a = 2;
  
  for (int i = 0; i < N; i++) {
    X[i] = i;
    Y[i] = N - i;
  }

  for (int i = 0; i < N; i++) {
    Y[i] += a * X[i];
  }

  free(X); free(Y);
}