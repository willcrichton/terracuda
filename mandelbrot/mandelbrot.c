#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1024
#define HEIGHT 1024

static inline void render(char *out, int x_dim, int y_dim) {
  int index = 3*WIDTH*y_dim + x_dim*3;
  float x_origin = ((float) x_dim/WIDTH)*3.25 - 2;
  float y_origin = ((float) y_dim/WIDTH)*2.5 - 1.25;

  float x = 0.0;
  float y = 0.0;

  int iteration = 0;
  int scale = 32;
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

int main() {
  // Multiply by 3 here, since we need red, green and blue for each pixel
  size_t buffer_size = sizeof(char) * WIDTH * HEIGHT * 3;
  char *image = (char *) malloc(buffer_size);

  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      render(image, x, y);
    }
  }

  free(image);
  return 0;
}