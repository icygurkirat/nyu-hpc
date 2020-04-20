all: mmultvector mmult jacobi

mmultvector:
	nvcc -std=c++11 -o mmultvector mmultvector.cu

mmult:
	nvcc -std=c++11 -o mmult mmult.cu

jacobi:
	nvcc -std=c++11 -o jacobi-gpu jacobi-gpu.cu
	g++ -std=c++11 -o jacobi-cpu jacobi-cpu.cpp -fopenmp

clean:
	rm -rf mmultvector
	rm -rf mmult
	rm -rf jacobi-cpu
	rm -rf jacobi-gpu