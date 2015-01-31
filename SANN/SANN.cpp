// SANN.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include <iostream>
#include <stdlib.h>
#include <time.h> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

class SANN{
public:

	void train(cv::Mat Descriptors);
	void Match(cv::Mat &Descriptors1, cv::Mat &Descriptors2, std::vector<cv::DMatch> &Matches);
	float distance(int N, int M);
	

	SANN();
	~SANN();

private:
	cv::Mat Descriptores1;
	cv::Mat Descriptores2;
	cv::Mat Material;
	cv::Mat Sumas;
	cv::Mat Medias;
	cv::Mat Desviaciones;
	int muestrasEntrenamiento, muestrasClasificacion;
	int caracteristicas;

	void sortByCol(cv::Mat &src, cv::Mat &dst, int col);
	void randomDistribution(int N, int M);
	void proposeRandomPair();
	
};

SANN::SANN(){
	muestrasEntrenamiento = 0;
	muestrasClasificacion = 0;
	caracteristicas = 0;
}

SANN::~SANN(){

}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	Descriptors1 -> Matriz original con los descriptores de esntrenamiento
*	Descriptors2 -> Matriz original con los descriptores de clasificacion
*	Matches		 -> Indices Resultado de match
*********************************************************************************************/
void SANN::Match(cv::Mat &Descriptors1, cv::Mat &Descriptors2, std::vector<cv::DMatch> &Matches){
	//Primero, entrenar con el primer argumento
	if(Descriptors1.rows > 0 && Descriptors1.cols > 0)
		train(Descriptors1);
	else
		std::cout << "No se puede entrenar, la matriz 1 no contiene muestras o caracteristicas \n";

	if((Descriptors2.rows > 0 && Descriptors2.cols > 0))
		Descriptors2.copyTo(Descriptores2);
	else
		std::cout << "No se puede clasificar, la matriz 2 no contiene muestras o caracteristicas \n";

	if(Descriptors1.cols != Descriptors2.cols)
		std::cout << "No se puede clasificar, la matriz 1 y la matriz 2 no tienen el mismo numero de caracteristicas \n";

	muestrasClasificacion = Descriptors2.rows;
	//Crear el material, para optimizar el proceso es una matriz con N filas como Descriptores de entrenamiento
	//y 4 columnas:
	//1) Indice de fila de la muestra material de entrenamiento
	//2) Indice de fila de la muestra a clasificar (Inicialmente aleatorio por principio de diversidad), -1 defecto
	//3) Indice propuesto de cambio, -1 por defecto
	//4) Distancia entre las muestras
	Material = cv::Mat::zeros(muestrasEntrenamiento, 4, CV_32F);

	for(int i=0; i < muestrasEntrenamiento; i++){
		Material.at<float>(i,0) = i;
		Material.at<float>(i,1) = -1;
		Material.at<float>(i,2) = -1;
		Material.at<float>(i,3) = -150000;
	}
	
	//Distribuir aleatoriamente las muestras de clasificacion
	randomDistribution(muestrasEntrenamiento, muestrasClasificacion);

	//Calcular las distancias
	for(int i=0; i < muestrasEntrenamiento; i++){
		int indexN = Material.at<float>(i,0);
		int indexM = Material.at<float>(i,1);
		if(indexM != -1)
			Material.at<float>(i,3) = distance(indexN, indexM);
	}

	//Proponer vector de cambio
	proposeRandomPair();

	std::cout << "\n Material: \n";
	//Finalmente imprimir el material
	for(int i=0; i<muestrasEntrenamiento; i++){
		for(int j=0; j<4; j++)
			std::cout << Material.at<float>(i,j) << " ";
		std::cout << "\n";
	}
	std::cout << "Fin material \n";

}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	Descriptors -> Matriz original con los descriptores adentro
*
*********************************************************************************************/
void SANN::train(cv::Mat Descriptors){
	//Hallar el numero de muestras y el numero de caracteristicas
	muestrasEntrenamiento = Descriptors.rows;
	caracteristicas = Descriptors.cols;

	//Declarar las matrices de Sumas y desviaciones
	Sumas = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));
	Medias = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));
	Desviaciones = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));

	//Hallas las sumas
	for(int i=0; i<muestrasEntrenamiento; i++)
		for(int j=0; j<caracteristicas; j++)
			Sumas.at<float>(0,j) =  Sumas.at<float>(0,j) + Descriptors.at<float>(i,j);

	//Hallar las medias
	for(int j=0; j<caracteristicas; j++)
			Medias.at<float>(0,j) = Sumas.at<float>(0,j) / muestrasEntrenamiento;

	//Hallar las desviaciones estandar
	for(int i=0; i<muestrasEntrenamiento; i++)
		for(int j=0; j<caracteristicas; j++)
			Desviaciones.at<float>(0,j) =  Desviaciones.at<float>(0,j)  + cv::pow(Descriptors.at<float>(i,j) - Medias.at<float>(0,j), 2);
	
	for(int j=0; j<caracteristicas; j++)
		Desviaciones.at<float>(0,j) = cv::sqrt(Desviaciones.at<float>(0,j) / muestrasEntrenamiento);

	//Hallar la columna con mayor desviacion estandar
	int maxDesv = -9999999999, maxDesvIdx = 0;
	for(int i=0; i<caracteristicas; i++)
		if(Desviaciones.at<float>(0,i) > maxDesv){
			maxDesv = Desviaciones.at<float>(0,i);
			maxDesvIdx = i;
		}
	
	//Finalmente ordenar los descriptores por la columna con mayor desviacion
	sortByCol(Descriptors, Descriptores1, maxDesvIdx);
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	src -> Matriz original con las muestras desordenadas
*	dst -> Matriz resultado ordenada
*	col -> Indice de la columna que se quiere ordenar
*
*********************************************************************************************/
void SANN::sortByCol(cv::Mat &src, cv::Mat &dst, int col){
	//Primero hallar el orden de los indices de cada columna
	cv::Mat idx;
	cv::sortIdx(src, idx, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

	//Segundo se crea la matriz de destino igual a la matriz fuente
	cv::Mat sorted = cv::Mat::zeros(src.rows,src.cols,src.type());

	//Se itera y se copia fila por fila
	for(int i=0; i<sorted.rows; i++)
		src.row(idx.at<int>(i,col)).copyTo(sorted.row(i));

	//Se copia a la salida
	sorted.copyTo(dst);
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	N -> Numero de  muestras de entrenamiento
*	M -> Numero de muestras a clasificar
*
*	!IMPORTANTE:  N => M
*********************************************************************************************/
void SANN::randomDistribution(int N, int M){
	//Crear el vector base y llenarlo
	std::vector<int> listaBaseEntrenamiento;
	for(int i=0; i < M; i++)
		listaBaseEntrenamiento.push_back(i);

	//Crear la lista desordenada y llenarla
	std::vector<int> listaBaseClasificacion;
	srand (time(NULL));
	int El = M;
	for(int i=0; i < M; i++){
		int index = rand() % El;
		listaBaseClasificacion.push_back(listaBaseEntrenamiento.at(index));
		listaBaseEntrenamiento.erase(listaBaseEntrenamiento.begin() + index);
		if(!--El)
			break;
	}

	//Crear la lista de pareja iniciarla en -1
	std::vector<int> listaBasePareja;
	for(int i=0; i < N; i++)
		listaBasePareja.push_back(-1);

	//Ubicar los indice de la lista de baseClasificacion en la de pareja
	El = M;
	for(int i=0; i < M; i++){
		while(true){
			int index = rand() % N;
			if(listaBasePareja.at(index) == -1){
				listaBasePareja.at(index) = listaBaseClasificacion.at(i);
				break;
			}
		}
	}

	//Finalmente llenar los indices en el Material
	for(int i=0; i < N; i++)
		Material.at<float>(i,1) = listaBasePareja.at(i);
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	N -> Indice de la muestra de entrenamiento
*	M -> Indice de la muestra de clasificacion
*
*	IMPORTANTE: N,M deben ser valores existentes!
*	TODO: Optimizar con absdiff().sum()
*********************************************************************************************/
float SANN::distance(int N, int M){
	float d = 0;
	for(int i=0; i<caracteristicas; i++){
		float  resta = Descriptores1.at<float>(N,i) - Descriptores2.at<float>(M,i);
		d += std::abs(resta);
	}

	return d;
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN y metodo principal basado en SA
*
*	TODO: Optimizar con absdiff().sum()
*********************************************************************************************/
void SANN::proposeRandomPair(){

	//Recorrer el material en busca de particulas a clasificar
	srand (time(NULL));
	int El = muestrasEntrenamiento;
	for(int i=0; i < muestrasEntrenamiento; i++){
		int indexM = Material.at<float>(i,1);
		//Si encuentra una particula de clasificacion
		if(indexM != -1){
			//Buscar en el espacio de indices propuesto una posibilidad de cambio
			int indexA = rand() % El;
			//Se calcula la funcion de costo
			float d_proposed = distance(indexA,indexM);
			float d_actual = Material.at<float>(i,3);

			//Si se mejora la funcion de costo
			if(d_proposed < d_actual){
				//Si el espacio esta vacio, entonces se pasa la particula a ese espacio
				if(Material.at<float>(indexA,1) == -1){
					Material.at<float>(indexA,1) = indexM;
					Material.at<float>(indexA,3) = d_proposed;
					Material.at<float>(i,1) = -1;
					Material.at<float>(i,3) = -1;
				}
				//Sino esta vacio, se verifica que la funcion de costo mejore respecto al valor actual
				else if(d_proposed < Material.at<float>(indexA,3)){
					Material.at<float>(i,1) = Material.at<float>(indexA,1);
					Material.at<float>(i,3) = Material.at<float>(indexA,3);					
					Material.at<float>(indexA,1) = indexM;
					Material.at<float>(indexA,3) = d_proposed;
				}
			}
		}
	}
}

int _tmain(int argc, _TCHAR* argv[]){

	//Datos de prueba

	cv::Mat Imagen1 = cv::imread("cuadro_1_imagen.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	cv::Mat Imagen2 = cv::imread("cuadro_2_imagen.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	if(!Imagen1.data || !Imagen2.data){
		std::cout << "No se puede leer la imagen \n";
		return 1;
	}

	//Crear el objeto SURF
	int minHessian = 400;
	cv::SurfFeatureDetector detector( minHessian );

	//Calular los keypoints
	std::vector<cv::KeyPoint> keypoints_scene1, keypoints_scene2;
	detector.detect( Imagen1, keypoints_scene1 );
	detector.detect( Imagen2, keypoints_scene2 );
	
	//Calular los descriptores
	cv::SurfDescriptorExtractor extractor;
	cv::Mat descriptors_scene1, descriptors_scene2;

	extractor.compute( Imagen1, keypoints_scene1, descriptors_scene1);
	extractor.compute( Imagen2, keypoints_scene2, descriptors_scene2);

	//Metodo match PROPUESTO
	/*
	SANN bestMatcher;
	std::vector< cv::DMatch > matches1;
	bestMatcher.Match( descriptors_scene2, descriptors_scene1, matches1);

	
	//Metodo match SUGERIDO:
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches2;
	matcher.match( descriptors_scene1, descriptors_scene2, matches2 );

	cv::Mat img_matches;
	cv::drawMatches( Imagen1, keypoints_scene1, Imagen2, keypoints_scene2,
               matches2, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	cv::imshow( "Good Matches & Object detection", img_matches );
	cv::waitKey(0);
	*/
	/*
	SANN metodo(caracteristicas);
	metodo.Entrenar();
	*/


	//Test de sort
	cv::Mat source = cv::Mat::zeros(10,10,CV_32F);
	for(int i=0; i<10; i++)
		for(int j=0; j<10; j++)
			source.at<float>(i,j) = (i-5)*(i-5)*(j-5)*(j-5);
	
	std::cout << "Entrenamiento: \n \n";
	for(int i=0; i < source.rows; i++){
		for(int j=0; j < source.cols; j++)
			std::cout << source.at<float>(i,j) << " ";
		std::cout << "\n";
	}

	cv::Mat dst = cv::Mat::zeros(2,10,CV_32F);
	for(int i=0; i<2; i++)
		for(int j=0; j<10; j++)
			dst.at<float>(i,j) = (i+1)*(i+1)*(j-5)*(j-5);

	std::cout << "Clasificacion: \n \n";
	for(int i=0; i < dst.rows; i++){
		for(int j=0; j < dst.cols; j++)
			std::cout << dst.at<float>(i,j) << " ";
		std::cout << "\n";
	}

	std::vector<cv::DMatch> Matches;
	
	SANN tester;
	tester.Match(source, dst, Matches);

	return 0;
}

