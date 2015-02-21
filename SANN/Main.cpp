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
#include "alineadorCERES.h";


//Variables globales
cv::Mat Imagen1, Imagen2;
std::vector<cv::KeyPoint> keypoints_scene1, keypoints_scene2;
std::vector< cv::DMatch > matches1, matches2;

void MetodoPropuesto(cv::Mat &descriptors_scene1, cv::Mat& descriptors_scene2){

	SANN bestMatcher;
	bestMatcher.Match(descriptors_scene2, descriptors_scene1, matches1);



	cv::Mat img_matches1;
	cv::drawMatches( Imagen2, keypoints_scene2, Imagen1, keypoints_scene1,
               matches1, img_matches1, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	cv::imshow( "Resultado propuesto", img_matches1 );
	cv::waitKey(0);
}

void MetodoSugerido(cv::Mat &descriptors_scene1, cv::Mat& descriptors_scene2){
	cv::FlannBasedMatcher matcher;
	
	matcher.match( descriptors_scene1, descriptors_scene2, matches2 );

	//Filtrar
	std::vector< cv::DMatch > matchesFilter;
	for(int i=0; i < matches2.size(); i++)
		if(matches2[i].distance < 0.25)
			matchesFilter.push_back(matches2[i]);

	//Realizar alineacion
	ceres::Problem problem;
	double matriz[9] = {0,0,0,0,0,0,0,0,1};

	for(int i=0; i < matchesFilter.size(); i++){
		double P1_x = keypoints_scene1[matchesFilter[i].queryIdx].pt.x;
		double P1_y = keypoints_scene1[matchesFilter[i].queryIdx].pt.y;

		double P2[3];
		P2[0] = keypoints_scene2[matchesFilter[i].trainIdx].pt.x;
		P2[1] = keypoints_scene2[matchesFilter[i].trainIdx].pt.y;
		P2[2] = 1000;

		double residuo[2];

		ceres::CostFunction* cost_function = alineadorM9::Create(P1_x, P1_y);
		problem.AddResidualBlock(cost_function, NULL, matriz, P2);//Pensar mejor

	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";


	cv::Mat img_matches1;
	cv::drawMatches( Imagen1, keypoints_scene1, Imagen2, keypoints_scene2,
               matchesFilter, img_matches1, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	cv::imshow( "Resultado sugerido", img_matches1 );
	//cv::waitKey(0);

	//Crear la imagen final
	//TODO: Mejorar con trasnformacion afin
	cv::Mat finalImage = cv::Mat(600,800,CV_8UC1, cv::Scalar(0));
	//Declarar la matriz de transformacion afin
	cv::Mat affineM = cv::Mat(2,3,CV_32FC1, cv::Scalar(0));
	affineM.at<uchar>(0,2) = (float)matriz[3];
	affineM.at<uchar>(1,2) = (float)matriz[4];

	std::cout << "Matriz : " << affineM << "\n";
	//Imagen1.copyTo(finalImage.rowRange(0,480).colRange(0,640));
	//Imagen2.copyTo(finalImage.rowRange(0-ty,480-ty).colRange(0-tx,640+tx));
	cv::transform(Imagen2,finalImage,affineM);
	cv::imshow( "Resultado traslapado", finalImage );
	cv::waitKey(0);

}

int _tmain(int argc, _TCHAR* argv[]){

	//Datos de prueba

	Imagen1 = cv::imread("cuadro_1_imagen.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	Imagen2 = cv::imread("cuadro_2_imagen.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	if(!Imagen1.data || !Imagen2.data){
		std::cout << "No se puede leer la imagen \n";
		return 1;
	}

	//Crear el objeto SURF
	int minHessian = 400;
	cv::SurfFeatureDetector detector( minHessian );

	//Calular los keypoints
	
	detector.detect( Imagen1, keypoints_scene1 );
	detector.detect( Imagen2, keypoints_scene2 );
	
	//Calular los descriptores
	cv::SurfDescriptorExtractor extractor;
	cv::Mat descriptors_scene1, descriptors_scene2;

	extractor.compute( Imagen1, keypoints_scene1, descriptors_scene1);
	extractor.compute( Imagen2, keypoints_scene2, descriptors_scene2);

	//Metodo match PROPUESTO
	//MetodoPropuesto(descriptors_scene1, descriptors_scene2);

	//Metodomatch sugerido
	MetodoSugerido(descriptors_scene1, descriptors_scene2);


	return 0;
}