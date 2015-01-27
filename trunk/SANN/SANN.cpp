// SANN.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

class SANN{
public:

	void train(cv::Mat Descriptors);
	void Match(cv::Mat &Descriptors1, cv::Mat &Descriptors2, std::vector<cv::DMatch> &Matches);

	SANN();
	~SANN();

private:
	cv::Mat Descriptores1;
	cv::Mat Descriptores2;
	cv::Mat Sumas;
	cv::Mat Medias;
	cv::Mat Desviaciones;
	int muestras;
	int caracteristicas;
};

SANN::SANN(){
	muestras = 0;
	caracteristicas = 0;
}

SANN::~SANN(){

}

void SANN::Match(cv::Mat &Descriptors1, cv::Mat &Descriptors2, std::vector<cv::DMatch> &Matches){
	//Primero, entrenar con el primer argumento
	train(Descriptors1);
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
	
	for(int j=0; j<caracteristicas; j++)
		std::cout << Desviaciones.at<float>(0,j) << "\n";


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

	return 0;
}

