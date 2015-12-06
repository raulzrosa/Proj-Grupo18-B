# SmoothCUDA
Trabalho desenvolvido na disciplina de Programação Concorrente, oferecida no segundo semestre de 2015 pela Universidade de São Paulo. O trabalho tem como objetivo fazer uma comparação de desempenho de um algoritmo que passa um filtro smooth, e utilizando para isso 3 versões diferentes: uma execução sequencial, uma execução paralela usando OpenMPI e OpenMP e uma e versão paralela usando CUDA. A versão CUDA consiste em utilizar uma GPU Nividia para fazer a execução.

Para compila o MP-MPI:
mpic++ -o SmoothMPI-MP SmoothMPI-MP.cpp -L/usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -fopenmp -O0 -Wall -g
Para executar o MP-MPI :
mpirun --hostfile name_nodes.txt -np num_nodes SmoothMPI-MP img_in type_img img_out 



Para compilar o cuda:
nvcc smoothCuda.cu -o smoothCuda -L/usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -g 

Para executar o cuda:
./smoothCuda img_in type_img img_out
