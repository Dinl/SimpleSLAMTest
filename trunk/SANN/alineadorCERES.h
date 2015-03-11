// AlineadorCERES.h: define el metodo de calculo de la matriz de alineacion
//

//#ifndef __alineacionCERES_H
//#define __alineacionCERES_H


#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/rotation.h>

using namespace ceres;



/***********************************************************************************
	*	Clase T para realizar la alineacion de observaciones entre 2 imagenes que se
	*	proyectan al espacio.
************************************************************************************/
class alineador2D{
public:
	/***********************************************************************************
	*	Constructor:
	*	Se crea con el punto XY de la primera imagen 
	***********************************************************************************/
	alineador2D(double original_x, double original_y){
		P1_x = original_x;
		P1_y = original_y;
	};

	/***********************************************************************************
	*	Operador
	*	Es el operador quien recibe y calcula los residuos para cada punto
	*
	*	Textrinseca -> Array 6 double: [Rx Ry Rz X Y Z] rotaciones en el sentido de euler
	*	Tintrinseca -> Array 6 double: [Fx Fy Cx Cy K1 K2] parametros de calibacion
	*	punto2D		-> Punto XY en la imagen de destino
	*	punto3D		-> Punto XYZ correspondiente al punto2D
	*
	*	El proceso consiste en:
	*	
	*	punto2D -> (intrinseca) -> punto2d3d -> (extrinseca) -> transformedpoint3d
	*	-> (intrinseca) -> transformedPunto2d
	*	residuo = original - transformedPunto2d
	***********************************************************************************/
	template <typename T>
	bool operator()(const T* const Textrinseca, const T* const Tintrinseca, const T* const punto2D, const T* const punto3D, T* residuos) const {
		
		//PASO 1. OBTENER LOS PAREMTROS INTRINSECOS DE LA CAMARA
		const T& focalX = Tintrinseca[0];
		const T& focalY = Tintrinseca[1];
		const T& centerX = Tintrinseca[2];
		const T& centerY = Tintrinseca[3];
		const T& k1 = Tintrinseca[4];
		const T& k2 = Tintrinseca[5];

		//PASO 2. CONVERTIR EL PUNTO 2D DE LA SEGUNDA CAPTURA A 3D
		T punto2D3D[3];
		punto2D3D[2] = punto3D[2];
		punto2D3D[1] = (punto2D[1] - centerY)*(punto3D[2]/focalY);
		punto2D3D[0] = (punto2D[0] - centerX)*(punto3D[2]/focalX);

		//Obtener el punto transformado en rotacion y traslacion
		T transformedPoint3D[3];
		ceres::AngleAxisRotatePoint(Textrinseca, punto2D3D, transformedPoint3D);

		transformedPoint3D[0] += Textrinseca[3];
		transformedPoint3D[1] += Textrinseca[4];
		transformedPoint3D[2] += Textrinseca[5];

		//Transformar las coordenadas 3D a la imagen 2D
		// Xcorrected = focalX * (X/Z) + centerX
		// Ycorrected = focalY * (Y/Z) + centerY
		
		T xp = focalX*(transformedPoint3D[0] / transformedPoint3D[2]) + centerX;
		T yp = focalY*(transformedPoint3D[1] / transformedPoint3D[2]) + centerY;

		// Aplicar el primer y segundo parametro de distorcion radial
		
		T r2 = xp*xp + yp*yp;
		T distortion = T(1.0) + r2  * (k1 + k2  * r2);
		distortion = T(1.0);
		//Hallar la prediccion
		T predicted_x = distortion * xp;
		T predicted_y = distortion * yp;

		//Obtener la distancia residual
		residuos[0] = T(P1_x) - predicted_x;
		residuos[1] = T(P1_y) - predicted_y;

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<alineador2D, 2, 6, 6, 2, 3>(new alineador2D(observed_x, observed_y)));
	}

private:
	double P1_x, P1_y;
};

/***********************************************************************************
	*	Clase T para realizar la alineacion de observaciones entre 2 nubes
	*
************************************************************************************/
class alineador3D{
public:
	
	/***********************************************************************************
	*	Constructor:
	*	Se crea con el punto XYZ de la primera nube 
	***********************************************************************************/
	alineador3D(double original_x, double original_y, double original_z){
		P1_x = original_x;
		P1_y = original_y;
		P1_z = original_z;
	};

	/***********************************************************************************
	*	Operador
	*	Es el operador quien recibe y calcula los residuos para cada punto
	*
	*	Textrinseca -> Array 6 double: [Rx Ry Rz X Y Z] rotaciones en el sentido de euler
	*	punto3D		-> Punto XYZ correspondiente al punto2D
	*
	*	El proceso consiste en:
	*	
	*	punto2D -> (intrinseca) -> punto2d3d -> (extrinseca) -> transformedpoint3d
	*	-> (intrinseca) -> transformedPunto2d
	*	residuo = original - transformedPunto2d
	***********************************************************************************/
	template <typename T>
	bool operator()(const T* const Textrinseca, const T* const punto3D, T* residuos) const {
		
		//Obtener el punto transformado en rotacion y traslacion
		T transformedPoint3D[3];
		ceres::AngleAxisRotatePoint(Textrinseca, punto3D, transformedPoint3D);

		transformedPoint3D[0] += Textrinseca[3];
		transformedPoint3D[1] += Textrinseca[4];
		transformedPoint3D[2] += Textrinseca[5];

		//Obtener la distancia residual
		residuos[0] = T(P1_x) - transformedPoint3D[0];
		residuos[1] = T(P1_y) - transformedPoint3D[1];
		residuos[2] = T(P1_z) - transformedPoint3D[2];

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double observed_z) {
		return (new ceres::AutoDiffCostFunction<alineador3D, 3, 6, 3>(new alineador3D(observed_x, observed_y, observed_z)));
	}

private:
	double P1_x, P1_y, P1_z;
};

//#endif