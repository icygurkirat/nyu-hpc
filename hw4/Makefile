all: mmultvector mmult jacobi

mmultvector:
	nvcc -std=c++11 -o mmultvector mmultvector.cu

mmult:
	nvcc -std=c++11 -o mmult mmult.cu

jacobi:
	nvcc -std=c++11 -o jacobi jacobi.cu

clean:
	rm -rf mmultvector
	rm -rf mmult
	rm -rf jacobi