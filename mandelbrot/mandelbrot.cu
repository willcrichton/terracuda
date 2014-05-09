#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <cuda.h>

__global__ void render(char *out, int width, int height) {
  int index = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
  int x_dim = (index / 3) % width, y_dim = (index / 3) / width;
  float x_origin = ((float) x_dim/width)*3.25 - 2;
  float y_origin = ((float) y_dim/width)*2.5 - 1.25;

  float x = 0.0;
  float y = 0.0;

  int iteration = 0;
  int scale = 8;
  int max_iteration = 256 * scale;
  while(x*x + y*y <= 4 && iteration < max_iteration) {
    float xtemp = x*x - y*y + x_origin;
    y = 2*x*y + y_origin;
    x = xtemp;
    iteration++;
  }

  if(iteration == max_iteration) {
    out[index] = 0;
    out[index + 1] = 0;
    out[index + 2] = 0;
  } else {
    out[index] = iteration / scale;
    out[index + 1] = iteration / scale;
    out[index + 2] = iteration / scale;
  }
}

void runCUDA(int width, int height)
{
  // Multiply by 3 here, since we need red, green and blue for each pixel
  size_t buffer_size = sizeof(char) * width * height * 3;
  
  char *image;
  cudaMalloc((void **) &image, buffer_size);

  char *host_image = (char *) malloc(buffer_size);

  dim3 blockDim(64, 1, 1);
  dim3 gridDim(width * height / blockDim.x, 1, 1);
  render<<< gridDim, blockDim, 0 >>>(image, width, height);

  cudaMemcpy(host_image, image, buffer_size, cudaMemcpyDeviceToHost);

  // Now write the file
  /*printf("P3\n%d %d\n255\n", width, height);
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      for (int i = 0; i < 3; i++) {
        unsigned char c = host_image[(row * width + col) * 3 + i];
        printf("%d ", c);
      }
    }
    printf("\n");
    }*/

  cudaFree(image);
  free(host_image);
}

int main(int argc, const char * argv[]) {
  int N = 1024;
  runCUDA(N, N);
  return 0;
}
