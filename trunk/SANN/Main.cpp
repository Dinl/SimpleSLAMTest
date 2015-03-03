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
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/visualization/boost.h>
#include <pcl/console/print.h>
#include <pcl/filters/filter.h>
#include <pcl/common/transforms.h>

#include "SANN.hpp";
#include "alineadorCERES.h";

//Definiciones utiles
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

//Variables globales
cv::Mat Imagen1, Imagen2;
std::vector<cv::KeyPoint> keypoints_scene1, keypoints_scene2;
pcl::PointCloud<PointT>::ConstPtr cloud_scene1, cloud_scene2;
pcl::PointCloud<PointT>::ConstPtr scene2transformed;


std::vector< cv::DMatch > matches;

float Rt[4][4];

void MetodoPropuesto(cv::Mat &descriptors_scene1, cv::Mat& descriptors_scene2){

	SANN bestMatcher;
	bestMatcher.Match(descriptors_scene1, descriptors_scene2, matches);



	cv::Mat img_matches1;
	cv::drawMatches( Imagen1, keypoints_scene1, Imagen2, keypoints_scene2,
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
		if(matchesFilter[i].distance < 0.150)
			matches.push_back(matchesFilter[i]);

	cv::Mat img_matches1;
	cv::drawMatches( Imagen1, keypoints_scene1, Imagen2, keypoints_scene2,
               matches, img_matches1, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	cv::imshow( "Resultado sugerido", img_matches1 );
	cv::waitKey(0);
}

void imprimirMatriz(double *M){
	float x = M[0];
	float y = M[1];
	float z = M[2];
	float yaw = M[3];
	float pitch = M[4];
	float roll = M[5];

	float sin_yaw = ceres::sin(yaw);
	float cos_yaw = ceres::cos(yaw);
	float sin_pitch = ceres::sin(pitch);
	float cos_pitch = ceres::cos(pitch);
	float sin_roll = ceres::sin(roll);
	float cos_roll = ceres::cos(roll);

	Rt[0][0] = cos_yaw * cos_pitch;
	Rt[0][1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll;
	Rt[0][2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll;
	Rt[0][3] = x;
	Rt[1][0] = sin_yaw * cos_pitch;
	Rt[1][1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll;
	Rt[1][2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll;
	Rt[1][3] = y;
	Rt[2][0] = -sin_pitch;
	Rt[2][1] = cos_pitch * sin_roll;
	Rt[2][2] = cos_pitch * cos_roll;
	Rt[2][3] = z;
	Rt[3][0] = 0.0;
	Rt[3][1] = 0.0;
	Rt[3][2] = 0.0;
	Rt[3][3] = 1.0;

	std::cout << "\n ************ \n";
	std::cout << " "<<Rt[0][0]<<"  "<<Rt[0][1]<<"  "<<Rt[0][2]<<"  "<<Rt[0][3]<<"\n";
	std::cout << " "<<Rt[1][0]<<"  "<<Rt[1][1]<<"  "<<Rt[1][2]<<"  "<<Rt[1][3]<<"\n"; 
	std::cout << " "<<Rt[2][0]<<"  "<<Rt[2][1]<<"  "<<Rt[2][2]<<"  "<<Rt[2][3]<<"\n"; 
	std::cout << " "<<Rt[3][0]<<"  "<<Rt[3][1]<<"  "<<Rt[3][2]<<"  "<<Rt[3][3]<<"\n"; 
	std::cout << "\n ************ \n";
}

void alinearCeres(){

	//Mostrar la nube transformada
	boost::shared_ptr<pcl::visualization::PCLVisualizer> v (new pcl::visualization::PCLVisualizer("OpenNI viewer"));

	//Crear las nubes de puntos originales
	pcl::PointCloud<PointT>::Ptr scene1(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr scene2(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr sceneSum(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr sceneT(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr sceneTSum(new pcl::PointCloud<PointT>);

	*scene1 = *cloud_scene1;
	*scene2 = *cloud_scene2;
	*sceneT = *scene2transformed;
	*sceneSum = *scene1 + *scene2;

	//Crear el primer viewport
	int v1(0);
	v->createViewPort(0.0,0.0,0.5,1.0,v1);
	v->setBackgroundColor(0,0,0,v1);
	v->addPointCloud(scene1, "sample cloud1", v1);

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
		C1[0] = cloud_scene1->at(P1[0],P1[1]).x;
		C1[1] = cloud_scene1->at(P1[0],P1[1]).y;
		C1[2] = cloud_scene1->at(P1[0],P1[1]).z;

		//Mostrar esferas en el lugar de los keypoints
		std::string sphereid = "sphere"+std::to_string(i);
		v->addSphere(cloud_scene1->at(P1[0],P1[1]),0.01,255,0,0,sphereid,v1);

		/**********************************************************************/
		//Obtener el punto en la imagen destino
		double P2[2];
		P2[0] = keypoints_scene2[matches[i].trainIdx].pt.x;
		P2[1] = keypoints_scene2[matches[i].trainIdx].pt.y;

		double indiceP2 = ((Imagen2.cols * (P2[1]-1)) + P2[0]);
		//Obtener el correspondiente punto en la nube
		double C2[3];
		C2[0] = cloud_scene2->at(P2[0],P2[1]).x;
		C2[1] = cloud_scene2->at(P2[0],P2[1]).y;
		C2[2] = cloud_scene2->at(P2[0],P2[1]).z;

		double residuo[3];

		/**Imprimir los 2 puntos espaciales**/
		/*
		std::cout<< "punto" << i << " : \n";
		std::cout << "Po: " << P1[0] << " " << P1[1] << " Pd: " << P2[0] << " " << P2[1] << "\n";
		std::cout << "Po: " << C1[0] << " " << C1[1] << " " << C1[2] << " Pd: " << C2[0] << " " << C2[1] << " " << C2[2] << "\n";
		*/
		if(C1[0] == C1[0] && C1[1] == C1[1] && C1[2] == C1[2] && 
			C2[0] == C2[0] && C2[1] == C2[1] && C2[2] == C2[2]){



			ceres::CostFunction* cost_function = alineadorM9::Create(C1[0], C1[1], C1[2]);
			problem.AddResidualBlock(cost_function, NULL, matriz, C2);//Pensar mejor
		}

	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;				//Por definir
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	std::cout << matriz[0] << " " << matriz[1] << " " << matriz[2] << " " << matriz[3] << " " << matriz[4] << " " << matriz[5] << "\n";
	imprimirMatriz(matriz);

	//Aplicar la transformacion de la nube
	
	Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

	transform_1 (0,0) = Rt[0][0];
	transform_1 (0,1) = Rt[0][1];
	transform_1 (0,2) = Rt[0][2];
	transform_1 (0,3) = Rt[0][3];

	transform_1 (1,0) = Rt[1][0];
	transform_1 (1,1) = Rt[1][1];
	transform_1 (1,2) = Rt[1][2];
	transform_1 (1,3) = Rt[1][3];

	transform_1 (2,0) = Rt[2][0];
	transform_1 (2,1) = Rt[2][1];
	transform_1 (2,2) = Rt[2][2];
	transform_1 (2,3) = Rt[2][3];

	transform_1 (3,0) = Rt[3][0];
	transform_1 (3,1) = Rt[3][1];
	transform_1 (3,2) = Rt[3][2];
	transform_1 (3,3) = Rt[3][3];
	
	pcl::transformPointCloud (*scene1, *sceneT, transform_1);
	*sceneTSum = *scene2 + *sceneT;

	int v2(0);
	v->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	v->setBackgroundColor(0,0,0,v2);
	v->addPointCloud(sceneTSum, "sample cloud2", v2);

	while(!v->wasStopped()){
		v->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}


	std::cout << "transformed \n";

}

int _tmain(int argc, _TCHAR* argv[]){

	//Cargar datos de prueba imagenes
	Imagen1 = cv::imread("cuadro_1_imagen_grises.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	Imagen2 = cv::imread("cuadro_5_imagen_grises.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	if(!Imagen1.data || !Imagen2.data){
		std::cout << "No se puede leer la imagen \n";
		return 1;
	}

	//Cargar datos de prueba nube
	pcl::PointCloud<PointT>::Ptr tmpscene1(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr tmpscene2(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr tmpscene3(new pcl::PointCloud<PointT>);

	if(pcl::io::loadPCDFile<PointT>("cuadro_1_nube.pcd",*tmpscene1) != 0
		|| pcl::io::loadPCDFile<PointT>("cuadro_5_nube.pcd",*tmpscene2) != 0){
		PCL_ERROR("Problem reading clouds \n");
		return 1;
	}
	cloud_scene1 = tmpscene1;
	cloud_scene2 = tmpscene2;
	scene2transformed = tmpscene3;

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