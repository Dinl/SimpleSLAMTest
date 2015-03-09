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


class alineadorM9{
public:
	
	//Constructor, el punto de la primera imagen 
	alineadorM9(double original_x, double original_y, double original_z){
		P1_x = original_x;
		P1_y = original_y;
		P1_z = original_z;
	};

	/***********************************************************************************
	*	Operador de costo utilizado por CERES
	*	Tmatrix -> Ingresa como un array de 6 valores:
	*				Tmatrix[0] -> Angulo yaw
	*				Tmatrix[1] -> Angulo pitch
	*				Tmatrix[2] -> Angulo roll
	*				Tmatrix[3] -> Traslacion en X
	*				Tmatrix[4] -> Traslacion en Y
	*				Tmatrix[5] -> Traslacion en Z
	*				Tmatrix[6] -> distancia focal X
	*				Tmatrix[7] -> distancia focal Y
	*				Tmatrix[8] -> Centro de la imagen en X
	*				Tmatrix[9] -> Centro de la imagen en Y
	*				Tmatrix[10] -> K1
	*				Tmatrix[11] -> K2
	*
	*	punto -> Ingresa como un array de 3 valores con XYZ del destino
	*				punto[0] -> X
	*				punto[1] -> Y
	*				punto[2] -> Z
	*
	*	residuos -> Array de 3 valores con los residuos en XYZ usado por ceres
	*
	************************************************************************************/

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
		residuos[0] = P1_x - T(transformedPoint3D[0]);
		residuos[1] = P1_y - T(transformedPoint3D[1]);
		residuos[2] = P1_z - T(transformedPoint3D[2]);

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double observed_z) {
		return (new ceres::AutoDiffCostFunction<alineadorM9, 3, 6, 6, 2, 3>(new alineadorM9(observed_x, observed_y, observed_z)));
	}

private:
	double P1_x, P1_y, P1_z;
};

//#endif