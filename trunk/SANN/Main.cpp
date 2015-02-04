#include "stdafx.h"
#include <iostream>
#include <stdlib.h>
#include <time.h> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "SANN.hpp";

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
	bestMatcher.Match(descriptors_scene2, descriptors_scene1, matches1);
	
	//bestMatcher.toString();
	
	//Metodo match SUGERIDO:
	/*
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches2;
	matcher.match( descriptors_scene1, descriptors_scene2, matches2 );

	std::cout << "Material: \n \n";
	for(int i=0; i < matches2.size(); i++){
		std::cout << matches2[i].queryIdx << " " << matches2[i].trainIdx << " " << matches2[i].distance << "\n";
	}
	*/
	cv::Mat img_matches;
	cv::drawMatches( Imagen2, keypoints_scene2, Imagen1, keypoints_scene1,
               matches1, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	cv::imshow( "Good Matches & Object detection", img_matches );
	cv::waitKey(0);
	
	/*
	SANN metodo(caracteristicas);
	metodo.Entrenar();
	*/


	//Test de sort
	/*
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

	cv::Mat dst = cv::Mat::zeros(5,10,CV_32F);
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
	*/
	return 0;
}