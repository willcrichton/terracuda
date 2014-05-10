#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define rol(x, n) ((x << n) | (x >> (32-n)))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// error checker for cuda calls
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //if (abort) exit(code);
   }
}


__device__ unsigned int f(int t, unsigned int B, unsigned int C, unsigned int D) {
  if (t <= 19)      return ((B & C) | ((!B) & D));
  else if (t <= 39) return (B ^ C ^ D);
  else if (t <= 59) return ((B & C) | (B & D) | (C & D));
  else              return (B ^ C ^ D);
}

__device__ unsigned int k(int t) {
  if (t <= 19)      return 0x5A827999;
  else if (t <= 39) return 0x6ED9EBA1;
  else if (t <= 59) return 0x8F1BBCDC;
  else              return 0xCA62C1D6;
}

// sha kernel, compute the hash of the message assigned to this thread
__global__ void sha(unsigned int *msgs, unsigned int *results) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  printf("hello from thread %d\n", tid);

  unsigned int mask = 0x0000000F;
  unsigned int H0 =   0x67452301;
  unsigned int H1 =   0xEFCDAB89;
  unsigned int H2 =   0x98BADCFE;
  unsigned int H3 =   0x10325476;
  unsigned int H4 =   0xC3D2E1F0;

  unsigned int A, B, C, D, E;
  A = H0; B = H1; C = H2; D = H3; E = H4;

  unsigned int *W = &msgs[16*tid];

  for (int i = 0; i < 80; i++) {
    unsigned int s = i & mask;
    if (i >= 16) {
      W[s] = rol(
                 W[(s + 13) & mask] ^
                 W[(s + 8)  & mask] ^
                 W[(s + 2)  & mask] ^
                 W[s], 1);
    }
    unsigned int tmp = rol(A, 5) + f(i, B, C, D) + E + W[s] + k(i);
    E = D;
    D = C;
    C = rol(B, 30);
    B = A;
    A = tmp;
  }
  H0 = H0 + A;
  H1 = H1 + B;
  H2 = H2 + C;
  H3 = H3 + D;
  H4 = H4 + E;
  results[tid*5 + 0] = H0;
  results[tid*5 + 1] = H1;
  results[tid*5 + 2] = H2;
  results[tid*5 + 3] = H3;
  results[tid*5 + 4] = H4;
  printf("\n%d - > 0x%x 0x%x 0x%x 0x%x 0x%x\n", tid, H0, H1, H2, H3, H4);
}

// Copy over the messages and run the kernel, then copy the results back
void sha_cuda(unsigned int *msgs, unsigned int num_msgs) {
  // cudaMalloc space for the messages on the device
  unsigned int *device_msgs;
  size_t size_msgs = (size_t)(16*num_msgs);
  gpuErrchk(cudaMalloc((void **) &device_msgs, size_msgs));

  // cudaMalloc results space on the device
  unsigned int *device_results;
  size_t size_results = (size_t)(5*num_msgs);

  gpuErrchk(cudaMalloc((void **) &device_results, size_results));

  // malloc space for the results on the host
  unsigned int *host_results = (unsigned int *)malloc(sizeof(unsigned int)*5*num_msgs);

  printf("Host result array before work:\n");
  for (int i = 0; i < 5*num_msgs; i++) {
    if (i % 5 == 0 && i != 0) printf("\n");
    printf(" 0x%x ", host_results[i]);
  }
  printf("\n\n");

  // move messages over, yo
  // TROUBLESOME REGION RIGHT OVA HERE
  gpuErrchk(cudaMemcpy((void*)device_msgs,
                       (const void *)msgs,
                       sizeof(unsigned int)*16*num_msgs,
                       cudaMemcpyHostToDevice));
  sha<<<(64, 1, 1), (num_msgs/blockDim.x, 1 , 1), 1>>>(device_msgs, device_results);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy((void *)host_results,
                       (const void *)device_results,
                       sizeof(unsigned int)*5*num_msgs,
                       cudaMemcpyDeviceToHost));
  // END TROUBLESOME REGION
  printf("Host result array after work:\n");
  for (int i = 0; i < 5*num_msgs; i++) {
    if (i % 5 == 0 && i != 0) printf("\n");
    printf(" 0x%x ", host_results[i]);
  }
  printf("\n");
  gpuErrchk(cudaFree(device_msgs));
  gpuErrchk(cudaFree(device_results));
  return;
}

unsigned int bytes_to_word(char a, char b, char c, char d) {
  return (a*0x1000000 + b*0x10000 + c*0x100 + d);

}

int main(int argc, char *argv[]) {
  char *filename = argv[1];
  printf("Reading from file: %s\n", filename);
  FILE *f1 = fopen(filename, "r");

  if (f1 != NULL) {
    char line[64];
    int num_msgs = 0;
    int i = 0;
    // count the number of lines in the file so we can store it all in a big array
    // because I'm bad at C. What's a better way to do this?
    while (fgets(line, sizeof(line), f1) != NULL) {
      num_msgs++;
    }

    //printf("%d messages, so allocate 64*num_msgs = %d \n", num_msgs, 4*16*num_msgs);
    unsigned int *msgs = (unsigned int*)malloc(sizeof(unsigned int)*16*num_msgs);

    unsigned int msg_offset = 0;

    FILE *f2 = fopen(filename, "r");
    while (fgets(line, sizeof(line), f2) != NULL) {
      line[strlen(line)-1] = '\0'; // remove the \n character
      //printf("Input: %s  %d\n", line, strlen(line));
      unsigned int msg_len = strlen(line);
      unsigned int msg_idx = 16*msg_offset;
      unsigned char zeroed_msg[64];
      for (i = 0; i < 64; i++) {
        if (i < msg_len) {
          zeroed_msg[i] = line[i];
        }
        else {
          zeroed_msg[i] = 0;
        }
      }
      zeroed_msg[msg_len] = 0x80;
      zeroed_msg[63] = msg_len*8;
      for (i = 0; i < 16; i++) {
        unsigned int j = bytes_to_word(zeroed_msg[4*i],
                                       zeroed_msg[4*i+1],
                                       zeroed_msg[4*i+2],
                                       zeroed_msg[4*i+3]);
        msgs[msg_idx+i] = j;
      }
      msg_offset++;
    }
    /*
    printf("Padded strings as follows:\n");
    for (i = 0; i < 16*num_msgs; i++) {
      if (i % 16 == 0) printf("\n");
      printf(" 0x%x", msgs[i]);
    }
    printf("\n");
    */
    sha_cuda(msgs, num_msgs);
    free(msgs);
  }
}
