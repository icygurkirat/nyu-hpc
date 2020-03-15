all: test1 test2 ompbugs jacobi gauss mmult

test1:
	g++ -g -std=c++11 -o val_test01_solved val_test01_solved.cpp

test2:
	g++ -g -std=c++11 -o val_test02_solved val_test02_solved.cpp

mmult:
	g++ -fopenmp -O2 -march=native MMult1.cpp -o MMult1

ompbugs:
	g++ -std=c++11 -o omp_solved2 omp_solved2.c -fopenmp
	g++ -std=c++11 -o omp_solved3 omp_solved3.c -fopenmp
	g++ -std=c++11 -o omp_solved4 omp_solved4.c -fopenmp
	g++ -std=c++11 -o omp_solved5 omp_solved5.c -fopenmp
	g++ -std=c++11 -o omp_solved6 omp_solved6.c -fopenmp

jacobi:
	g++ -std=c++11 -o jacobi2D-omp jacobi2D-omp.cpp -fopenmp

gauss:
	g++ -std=c++11 -o gs2D-omp gs2D-omp.cpp -fopenmp

clean:
	rm -rf MMult1
	rm -rf jacobi2D-omp
	rm -rf gs2D-omp
	rm -rf val_test01_solved
	rm -rf val_test02_solved
	rm -rf omp_solved2
	rm -rf omp_solved3
	rm -rf omp_solved4
	rm -rf omp_solved5
	rm -rf omp_solved6