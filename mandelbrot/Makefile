all:
	time ../../terra-fork/terra mandelbrot.t
#	sed -e '1,3d' < test.tmp > test.ppm
#	rm test.tmp
#	convert test.ppm test.jpg

cuda:
	nvcc -arch compute_20 main.cu
	time ./a.out
#	convert test.ppm test.jpg