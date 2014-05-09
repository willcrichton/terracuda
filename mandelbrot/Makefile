all:
	time ../../terra-fork/terra mandelbrot.t
#	sed -e '1,3d' < test.tmp > test.ppm
#	rm test.tmp
#	convert test.ppm test.jpg

cuda:
	nvcc -O3 -arch compute_20 mandelbrot.cu
	time ./a.out
#	convert test.ppm test.jpg
#	open test.jpg

c:
	gcc -O3 mandelbrot.c
	time ./a.out > test.ppm
	convert test.ppm test.jpg
	open test.jpg