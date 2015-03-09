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

	
	//Halla min/max
	float min = 9999999, max = -9999999;
	for(int i=0; i < matchesFilter.size(); i++)
		if(matchesFilter[i].distance < min)
			min = matchesFilter[i].distance;
		else if(matchesFilter[i].distance > max)
			max = matchesFilter[i].distance;

	//Filtrar
	float limite = (max-min)*0.2 + min;
	for(int i=0; i < matchesFilter.size(); i++)
		if(matchesFilter[i].distance <= limite)
			matches.push_back(matchesFilter[i]);
	
	cv::Mat img_matches1;
	cv::drawMatches( Imagen1, keypoints_scene1, Imagen2, keypoints_scene2,
               matches, img_matches1, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	cv::imshow( "Resultado sugerido", img_matches1 );
	cv::waitKey(0);
}

void imprimirMatriz(double *M){
	
	float yaw = M[0];
	float pitch = M[1];
	float roll = M[2];
	float x = M[3];
	float y = M[4];
	float z = M[5];

	float sin_yaw = ceres::sin(yaw);
	float cos_yaw = ceres::cos(yaw);
	float sin_pitch = ceres::sin(pitch);
	float cos_pitch = ceres::cos(pitch);
	float sin_roll = ceres::sin(roll);
	float cos_roll = ceres::cos(roll);

	Rt[0][0] = cos_pitch * cos_roll;
	Rt[0][1] = sin_pitch*sin_yaw - cos_pitch*sin_roll*cos_yaw;
	Rt[0][2] = cos_pitch*sin_roll*sin_yaw + sin_pitch*cos_yaw;
	Rt[0][3] = x;
	Rt[1][0] = sin_roll;
	Rt[1][1] = cos_roll*cos_yaw;
	Rt[1][2] = -cos_roll*sin_yaw;
	Rt[1][3] = y;
	Rt[2][0] = -sin_pitch*cos_roll;
	Rt[2][1] = sin_pitch*sin_roll*cos_yaw + cos_pitch*sin_yaw;
	Rt[2][2] = -sin_pitch*sin_roll*sin_yaw + cos_pitch*cos_yaw;
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

	//Realizar alineacion
	ceres::Problem problem;

	//Crear el vector inicial de transformacion
	double *extrinseca;
	extrinseca = new double[6];
	extrinseca[0] = 0;
	extrinseca[1] = 0;
	extrinseca[2] = 0;
	extrinseca[3] = 0;
	extrinseca[4] = 0;
	extrinseca[5] = 0;

	double *intrinseca;
	intrinseca = new double[6];
	intrinseca[0] = 535.501896;	//Fx
	intrinseca[1] = 537.504906;	//Fy
	intrinseca[2] = 330.201700;	//CenterX
	intrinseca[3] = 248.2017;	//CenterY
	intrinseca[4] = 0.119773;	//K1
	intrinseca[5] = -0.369037;	//K2

	for(int i=0; i < matches.size(); i++){
		//Obtener el punto en la imagen original
		double *P1;
		P1 = new double[2];
		P1[0] = keypoints_scene1[matches[i].queryIdx].pt.x;
		P1[1] = keypoints_scene1[matches[i].queryIdx].pt.y;

		//Obtener el correspondiente punto en la nube
		double indiceP1 = ((Imagen1.cols * (P1[1]-1)) + P1[0]);		
		double *C1;
		C1 = new double[3];
		C1[0] = cloud_scene1->at(P1[0],P1[1]).x;
		C1[1] = cloud_scene1->at(P1[0],P1[1]).y;
		C1[2] = cloud_scene1->at(P1[0],P1[1]).z;

		//Mostrar esferas en el lugar de los keypoints
		//std::string sphereid = "sphere"+std::to_string(i);
		//v->addSphere(cloud_scene1->at(P1[0],P1[1]),0.01,255,0,0,sphereid,v1);

		/**********************************************************************/
		//Obtener el punto en la imagen destino
		double *P2;
		P2 = new double[2];
		P2[0] = keypoints_scene2[matches[i].trainIdx].pt.x;
		P2[1] = keypoints_scene2[matches[i].trainIdx].pt.y;

		//Obtener el correspondiente punto en la nube
		double indiceP2 = ((Imagen2.cols * (P2[1]-1)) + P2[0]);		
		double *C2;
		C2 = new double[3];
		C2[0] = cloud_scene2->at(P2[0],P2[1]).x;
		C2[1] = cloud_scene2->at(P2[0],P2[1]).y;
		C2[2] = cloud_scene2->at(P2[0],P2[1]).z;

		if(C1[0] == C1[0] && C1[1] == C1[1] && C1[2] == C1[2] && C2[0] == C2[0] && C2[1] == C2[1] && C2[2] == C2[2]){
			/**Imprimir los 2 puntos espaciales**/
			/*
			std::cout<< "punto" << i << " : \n";
			std::cout << "Po: " << P1[0] << " " << P1[1] << " Pd: " << P2[0] << " " << P2[1] << "\n";
			std::cout << "Po: " << C1[0] << " " << C1[1] << " " << C1[2] << " Pd: " << C2[0] << " " << C2[1] << " " << C2[2] << "\n";
			*/
		
			ceres::CostFunction* cost_function = alineadorM9::Create(C1[0], C1[1], C1[2]);
			problem.AddResidualBlock(cost_function, NULL, extrinseca, intrinseca, P2, C2);
		}

	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;				//Por definir
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;
	options.function_tolerance = 1e-16;
	options.gradient_tolerance = 1e-32;
	options.parameter_tolerance = 1e-16;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	std::cout << extrinseca[0] << " " << extrinseca[1] << " " << extrinseca[2] << " " << extrinseca[3] << " " << extrinseca[4] << " " << extrinseca[5] << "\n";
	std::cout << intrinseca[0] << " " << intrinseca[1] << " " << intrinseca[2] << " " << intrinseca[3] << " " << intrinseca[4] << " " << intrinseca[5] << "\n";
	//imprimirMatriz(matriz);

	double R[9];
	//matriz[2] = 0;
	ceres::AngleAxisToRotationMatrix(extrinseca,R);

	//alineadorM9::getRotationMatrix(matriz,R);

	//Aplicar la transformacion de la nube
	
	Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

	transform_1 (0,0) = R[0];
	transform_1 (0,1) = R[3];
	transform_1 (0,2) = R[6];
	transform_1 (0,3) = extrinseca[3];

	transform_1 (1,0) = R[1];
	transform_1 (1,1) = R[4];
	transform_1 (1,2) = R[7];
	transform_1 (1,3) = extrinseca[4];

	transform_1 (2,0) = R[2];
	transform_1 (2,1) = R[5];
	transform_1 (2,2) = R[8];
	transform_1 (2,3) = extrinseca[5];

	transform_1 (3,0) = 0;
	transform_1 (3,1) = 0;
	transform_1 (3,2) = 0;
	transform_1 (3,3) = 1;
	
	std::cout << transform_1 << "\n";

	pcl::transformPointCloud (*scene2, *sceneT, transform_1);
	*sceneTSum = *scene1 + *sceneT;

	int v2(0);
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
	//cv::SurfFeatureDetector detector( minHessian );
	cv::SiftFeatureDetector detector;
	//Calular los keypoints
	
	detector.detect( Imagen1, keypoints_scene1 );
	detector.detect( Imagen2, keypoints_scene2 );
	
	//Calular los descriptores
	//cv::SurfDescriptorExtractor extractor;
	cv::SiftDescriptorExtractor extractor;
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