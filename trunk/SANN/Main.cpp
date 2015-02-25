#include "stdafx.h"
#include <iostream>
#include <stdlib.h>
#include <time.h> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>


#include <pcl/io/pcd_io.h>
//TODO: Solucionar ambiguedades para que sea compatible con ceres
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>

#include <pcl/visualization/boost.h>
#include <pcl/console/print.h>
#include <pcl/filters/filter.h>

#include "SANN.hpp";
#include "alineadorCERES.h";

//Definiciones utiles
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

//Variables globales
cv::Mat Imagen1, Imagen2;
std::vector<cv::KeyPoint> keypoints_scene1, keypoints_scene2;
pcl::PointCloud<PointT>::ConstPtr cloud_scene1, cloud_scene2;
std::vector< cv::DMatch > matches;

void MetodoPropuesto(cv::Mat &descriptors_scene1, cv::Mat& descriptors_scene2){

	SANN bestMatcher;
	bestMatcher.Match(descriptors_scene2, descriptors_scene1, matches);



	cv::Mat img_matches1;
	cv::drawMatches( Imagen2, keypoints_scene2, Imagen1, keypoints_scene1,
               matches, img_matches1, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	cv::imshow( "Resultado propuesto", img_matches1 );
	cv::waitKey(0);
}

void MetodoSugerido(cv::Mat &descriptors_scene1, cv::Mat& descriptors_scene2){
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matchesFilter;

	matcher.match( descriptors_scene1, descriptors_scene2, matchesFilter );

	//Filtrar
	
	for(int i=0; i < matchesFilter.size(); i++)
		if(matchesFilter[i].distance < 0.05)
			matches.push_back(matchesFilter[i]);

	cv::Mat img_matches1;
	cv::drawMatches( Imagen1, keypoints_scene1, Imagen2, keypoints_scene2,
               matches, img_matches1, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	cv::imshow( "Resultado sugerido", img_matches1 );
	cv::waitKey(0);
}

void alinearCeres(){
	//Realizar alineacion
	ceres::Problem problem;

	//Crear el vector inicial de transformacion
	double *matriz;
	matriz = new double[6];
	matriz[0] = 0;
	matriz[1] = 0;
	matriz[2] = 0;
	matriz[3] = 0;
	matriz[4] = 0;
	matriz[5] = 0;

	for(int i=0; i < matches.size(); i++){
		//Obtener el punto en la imagen original
		double P1[2];
		P1[0] = keypoints_scene1[matches[i].queryIdx].pt.x;
		P1[1] = keypoints_scene1[matches[i].queryIdx].pt.y;

		double indiceP1 = ((Imagen1.cols * (P1[1]-1)) + P1[0]);
		//Obtener el correspondiente punto en la nube
		double C1[3];
		C1[0] = cloud_scene1->at(indiceP1).x;
		C1[1] = cloud_scene1->at(indiceP1).y;
		C1[2] = cloud_scene1->at(indiceP1).z;
		/**********************************************************************/
		//Obtener el punto en la imagen destino
		double P2[2];
		P2[0] = keypoints_scene2[matches[i].trainIdx].pt.x;
		P2[1] = keypoints_scene2[matches[i].trainIdx].pt.y;

		double indiceP2 = ((Imagen2.cols * (P2[1]-1)) + P2[0]);
		//Obtener el correspondiente punto en la nube
		double C2[3];
		C2[0] = cloud_scene2->at(indiceP2).x;
		C2[1] = cloud_scene2->at(indiceP2).y;
		C2[2] = cloud_scene2->at(indiceP2).z;

		double residuo[3];

		if(C1[0] == C1[0] && C2[0] == C2[0]){
			ceres::CostFunction* cost_function = alineadorM9::Create(C1[0], C1[1], C1[2]);
			problem.AddResidualBlock(cost_function, NULL, matriz, C2);//Pensar mejor
		}

	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;				//Por definir
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	//options.parameter_tolerance = 0.0000000000000000000000000000000001;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	std::cout << matriz[0] << " " << matriz[1] << " " << matriz[2] << " " << matriz[3] << " " << matriz[4] << " " << matriz[5] << "\n";

	//Crear la imagen final
	//TODO: Mejorar con trasnformacion afin
	cv::Mat finalImage = cv::Mat(600,800,CV_8UC1, cv::Scalar(0));
	//Declarar la matriz de transformacion afin
	cv::Mat affineM = cv::Mat(2,3,CV_32FC1, cv::Scalar(0));
	affineM.at<uchar>(0,2) = (uchar)((float)matriz[3]);
	affineM.at<uchar>(1,2) = (uchar)((float)matriz[4]);

	std::cout << "Matriz : " << affineM << "\n";
	//Imagen1.copyTo(finalImage.rowRange(0,480).colRange(0,640));
	//Imagen2.copyTo(finalImage.rowRange(0-ty,480-ty).colRange(0-tx,640+tx));
	cv::transform(Imagen2,finalImage,affineM);
	cv::imshow( "Resultado traslapado", finalImage );
	cv::waitKey(0);

}

int _tmain(int argc, _TCHAR* argv[]){

	//Cargar datos de prueba imagenes
	Imagen1 = cv::imread("cuadro_1_imagen_grises.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	Imagen2 = cv::imread("cuadro_13_imagen_grises.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	if(!Imagen1.data || !Imagen2.data){
		std::cout << "No se puede leer la imagen \n";
		return 1;
	}

	//Cargar datos de prueba nube
	pcl::PointCloud<PointT>::Ptr tmpscene1(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<PointT>::Ptr tmpscene2(new pcl::PointCloud<pcl::PointXYZRGBA>);

	if(pcl::io::loadPCDFile<PointT>("cuadro_1_nube.pcd",*tmpscene1) != 0
		|| pcl::io::loadPCDFile<PointT>("cuadro_13_nube.pcd",*tmpscene2) != 0){
		PCL_ERROR("Problem reading clouds \n");
		return 1;
	}
	cloud_scene1 = tmpscene1;
	cloud_scene2 = tmpscene2;

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

	//Realizar alineacion
	alinearCeres();

	return 0;
}