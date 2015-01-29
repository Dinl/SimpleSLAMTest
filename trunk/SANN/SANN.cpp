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
	std::vector<int> randomDistribution(int N);
	

	SANN();
	~SANN();

private:
	cv::Mat Descriptores1;
	cv::Mat Descriptores2;
	cv::Mat Material;
	cv::Mat Sumas;
	cv::Mat Medias;
	cv::Mat Desviaciones;
	int muestras;
	int caracteristicas;

	void sortByCol(cv::Mat &src, cv::Mat &dst, int col);
	
};

SANN::SANN(){
	muestras = 0;
	caracteristicas = 0;
}

SANN::~SANN(){

}

void SANN::Match(cv::Mat &Descriptors1, cv::Mat &Descriptors2, std::vector<cv::DMatch> &Matches){
	//Primero, entrenar con el primer argumento
	if(Descriptors1.rows > 0 && Descriptors1.cols > 0)
		train(Descriptors1);
	else
		std::cout << "No se puede entrenar, la matriz 1 no contiene muestras o caracteristicas \n";

	//Crear el material, para optimizar el proceso es una matriz con N filas como Descriptores de entrenamiento
	//y 4 columnas:
	//1) Indice de fila de la muestra material de entrenamiento
	//2) Indice de fila de la muestra a clasificar (Inicialmente aleatorio por principio de diversidad), -1 defecto
	//3) Indice propuesto de cambio, -1 por defecto
	//4) Distancia entre las muestras
	Material = cv::Mat::zeros(muestras, 4, CV_8U);

	for(int i=0; i < muestras; i++){
		Material.at<uchar>(i,0) = i;
		Material.at<uchar>(i,1) = -1;
		Material.at<uchar>(i,2) = -1;
		Material.at<uchar>(i,3) = -15000;
	}




}

void SANN::train(cv::Mat Descriptors){
	//Hallar el numero de muestras y el numero de caracteristicas
	muestras = Descriptors.rows;
	caracteristicas = Descriptors.cols;

	//Declarar las matrices de Sumas y desviaciones
	Sumas = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));
	Medias = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));
	Desviaciones = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));

	//Hallas las sumas
	for(int i=0; i<muestras; i++)
		for(int j=0; j<caracteristicas; j++)
			Sumas.at<float>(0,j) =  Sumas.at<float>(0,j) + Descriptors.at<float>(i,j);

	//Hallar las medias
	for(int j=0; j<caracteristicas; j++)
			Medias.at<float>(0,j) = Sumas.at<float>(0,j) / muestras;

	//Hallar las desviaciones estandar
	for(int i=0; i<muestras; i++)
		for(int j=0; j<caracteristicas; j++)
			Desviaciones.at<float>(0,j) =  Desviaciones.at<float>(0,j)  + cv::pow(Descriptors.at<float>(i,j) - Medias.at<float>(0,j), 2);
	
	for(int j=0; j<caracteristicas; j++)
		Desviaciones.at<float>(0,j) = cv::sqrt(Desviaciones.at<float>(0,j) / muestras);

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

/**
	Funcion privada de la clase de clasificacion SANN
	N -> Numero de 
*/
std::vector<int> SANN::randomDistribution(int N){
	//Crear el vector base y llenarlo
	std::vector<int> listaBase;
	for(int i=0; i < N; i++)
		listaBase.push_back(i);
	
	//Crear la lista desordenada y llenarla
	std::vector<int> listaDesordenada;
	srand (time(NULL));
	int El = N;
	for(int i=0; i < N; i++){
		int index = rand() % El;
		listaDesordenada.push_back(listaBase.at(index));
		listaBase.erase(listaBase.begin() + index);
		El--;
	}

	return listaDesordenada;
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
	SANN bestMatcher;
	std::vector< cv::DMatch > matches1;
	bestMatcher.Match( descriptors_scene1, descriptors_scene2, matches1);


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

	/*
	SANN metodo(caracteristicas);
	metodo.Entrenar();
	*/


	//Test de sort
	cv::Mat source = cv::Mat::zeros(10,10,CV_32F), dst;

	for(int i=0; i<10; i++)
		for(int j=0; j<10; j++)
			source.at<float>(i,j) = (i-5)*(i-5)*(j-5)*(j-5);

	std::cout << "Original: \n \n";
	for(int i=0; i < source.rows; i++){
		for(int j=0; j < source.cols; j++)
			std::cout << source.at<float>(i,j) << " ";
		std::cout << "\n";
	}
	
	SANN tester;
	tester.train(source);
	//tester.sortByCol(source, dst, 1);
	tester.randomDistribution(10);

	

	std::cout << "Resultado: \n \n";
	for(int i=0; i < dst.rows; i++){
		for(int j=0; j < dst.cols; j++)
			std::cout << dst.at<int>(i,j) << " ";
		std::cout << "\n";
	}


	return 0;
}

