#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <math.h>

using namespace cv;
using namespace std;

//Função que calcula a média de uma "matriz" 5x5 a partir de uma dada posição
__global__ void smooth(unsigned char *entrada, unsigned char *saida, int n_linhas, int n_colunas , int cor, int canais) {
    //Calcula a posição no vetor (id_bloco * total_blocos + id_thread)
    float media;
	//printf(" %d %d \n", n_linhas, n_colunas);
    int posicao = cor + canais*(blockIdx.x * blockDim.x + threadIdx.x);
    //Se a posição não é maior que o limite da imagem original...
    if(posicao < (n_linhas + 4)*(n_colunas + 4)*canais) {
        //soma o valor da região 5x5 em torno no pixel
        media  = (entrada[posicao]+
                        entrada[cor +canais*(posicao+(n_colunas+4))]+
                        entrada[cor +canais*(posicao+(2*(n_colunas+4)))]+
                        entrada[cor +canais*(posicao+((-1)*(n_colunas+4)))]+
                        entrada[cor +canais*(posicao+((-2)*(n_colunas+4)))]+
                        entrada[cor +canais*(posicao+1)]+
                        entrada[cor +canais*(posicao+(n_colunas+4)+1)]+
                        entrada[cor +canais*(posicao+(2*(n_colunas+4))+1)]+
                        entrada[cor +canais*(posicao+((-1)*(n_colunas+4))+1)]+
                        entrada[cor +canais*(posicao+((-2)*(n_colunas+4))+1)]+
                        entrada[cor +canais*(posicao+2)]+
                        entrada[cor +canais*(posicao+(n_colunas+4)+2)]+
                        entrada[cor +canais*(posicao+(2*(n_colunas+4))+2)]+
                        entrada[cor +canais*(posicao+((-1)*(n_colunas+4))+2)]+
                        entrada[cor +canais*(posicao+((-2)*(n_colunas+4))+2)]+
                        entrada[cor +canais*(posicao - 1)]+
                        entrada[cor +canais*(posicao+(n_colunas+4)-1)]+
                        entrada[cor +canais*(posicao+(2*(n_colunas+4))- 1)]+
                        entrada[cor +canais*(posicao+((-1)*(n_colunas+4)) - 1)]+
                        entrada[cor +canais*(posicao+((-2)*(n_colunas+4)) - 1)]+
                        entrada[cor +canais*(posicao - 2)]+
                        entrada[cor +canais*(posicao+(n_colunas+4) - 2)]+
                        entrada[cor +canais*(posicao+(2*(n_colunas+4)) - 2)]+
                        entrada[cor +canais*(posicao+((-1)*(n_colunas+4)) - 2)]+
                        entrada[cor +canais*(posicao+((-2)*(n_colunas+4)) - 2)])/25;
    //calcula a média
    saida[posicao] =  media; 
   // printf("%d %d\n", entrada[posicao], saida[posicao]);
    }
}

int main(int argc, char *argv[]) {
    //diz se a imagem é grayscale or color
    int tipo_img = atoi(argv[2]);
    //arquivo de entrada
    const char *fileIn, *fileOut;
    
    //numero maximo de threads da placa do andromeda
    int nthreads = 1024;

    int numBlocks;


    //matriz com a imagem de entrada
    Mat in;
    //matriz que receberá a imagem de saida
    Mat out;

    //le o nome da imagem
    fileIn = argv[1];
    fileOut = argv[3];
    //le e salva a imagem na matriz
    if(tipo_img == 0) {
        in = imread(fileIn, CV_LOAD_IMAGE_GRAYSCALE);
    } else if(tipo_img == 1) {
        in = imread(fileIn, CV_LOAD_IMAGE_COLOR);
    } else {
        cout << "Tipo de imagem nao suportado" << endl;
        return -1;
    }
    //caso nao consegui abrir a imagem
    if (in.empty()) {
        cout << "Nao foi possivel abrir a  imagem: " << endl;
        return -1;
    }
    int border = 2;
    //coloco as dimensoes da img em variaveis mais simples para facilitar
    int l_height = in.size().height, l_width = in.size().width;

    //numero de blocos é o total de pixels dividido pelo total de threads
	if(tipo_img == 0) {
   		 numBlocks = (l_height*l_width/nthreads) + 1;
	} else if (tipo_img == 1) {
		numBlocks = (l_height*l_width/nthreads)*3 + 1;
	}
    unsigned char *original,*saida;

    //poe uma borda na imagem
    copyMakeBorder(in, in, border, border, border, border, BORDER_REPLICATE);
    //alloca uma matriz que vai receber uma imagem com borda
	if(tipo_img == 0) {
   		 cudaMalloc(&original, (l_width + 4) * (l_height + 4));	
   		 cudaMalloc(&saida, l_width * l_height);
	} else if (tipo_img == 1) {
		cudaMalloc(&original, (l_width + 4) * (l_height + 4)*3);
  		cudaMalloc(&saida, l_width * l_height* 3);
	}
    //alloca a matriz de saida que nao tem borda
    out = Mat::zeros(in.size(), in.type());
    //inicializa o tipo Mat que vai receber a matriz de saida

    //pegar o tempo de inicio
    struct timeval inicio, fim;
    gettimeofday(&inicio,0);
    if(tipo_img == 0) {
   		 //copia a imagem original de entrada para a gpu
   		 cudaMemcpy(original, in.data,(l_width + 4) * (l_height + 4), cudaMemcpyHostToDevice);
	}
	else if(tipo_img == 1) {
	   	 //copia a imagem original de entrada para a gpu
   		 cudaMemcpy(original, in.data,(l_width + 4) * (l_height + 4)*3, cudaMemcpyHostToDevice);	
	}
    //chama a função que passa o filtro
	if(tipo_img == 0){ 
    	smooth<<<numBlocks,nthreads>>>(original, saida, l_height, l_width, 0, 1);
	    cudaMemcpy(out.data, saida, l_width*l_height,cudaMemcpyDeviceToHost);
    } else if(tipo_img == 1) {
		smooth<<<numBlocks,nthreads>>>(original, saida, l_height, l_width, 0, 3);
		smooth<<<numBlocks,nthreads>>>(original, saida, l_height, l_width, 1, 3);
		smooth<<<numBlocks,nthreads>>>(original, saida, l_height, l_width,2, 3);
   		cudaMemcpy(out.data, saida, l_width*l_height*3,cudaMemcpyDeviceToHost);
	}
    //copia a matriz que ja recebeu e que esta na gpu de volta pra cpu
    

    //pega o tempo de fim, faz a diferença e imprime na tela
    gettimeofday(&fim,0);
    float speedup = (fim.tv_sec + fim.tv_usec/1000000.0) - (inicio.tv_sec + inicio.tv_usec/1000000.0);
    cout << speedup << endl;
    
    //gera a imagem de saida
    imwrite(fileOut, out);
    
    //libera memória
    in.release();
    out.release();
    cudaFree(original);
    cudaFree(saida);

    return 0;
}
    
