all: mmultvector jacobi

mmultvector:
	nvcc -std=c++11 -o mmultvector mmultvector.cu

jacobi:
	nvcc -std=c++11 -o jacobi jacobi.cu

clean:
	rm -rf mmultvector
	rm -rf jacobi