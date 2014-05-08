cuda:
	nvcc -O3 -arch compute_20 saxpy.cu
	time ./a.out

c:
	gcc -O3 saxpy.c
	time ./a.out

terra:
	time ../../terra-fork/terra saxpy.lua

tcuda:
	time ../../terra-fork/terra saxpy.t