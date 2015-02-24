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
	*				Tmatrix[0] -> Traslacion en X
	*				Tmatrix[1] -> Traslacion en Y
	*				Tmatrix[2] -> Traslacion en Z
	*				Tmatrix[3] -> Angulo yaw
	*				Tmatrix[4] -> Angulo pitch
	*				Tmatrix[5] -> Angulo roll
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
	bool operator()(const T* const Tmatrix, const T* const punto, T* residuos) const {
		
		//Establecer las constantes de la calibracion de camaa
		T fx = T(535.501896);
		T fy = T(537.504906);
		T inv_fx = T(1.) / fx;
		T inv_fy = T(1.) / fy;
		T cx = T(330.019632);
		T cy = T(248.201700);

		//Obtener las variables del vector de estado
		T x = Tmatrix[0];
		T y = Tmatrix[1];
		T z = Tmatrix[2];
		T yaw = Tmatrix[3];
		T pitch = Tmatrix[4];
		T roll = Tmatrix[5];

		//Calcular parametros utiles
		T sin_yaw = ceres::sin(yaw);
		T cos_yaw = ceres::cos(yaw);
		T sin_pitch = ceres::sin(pitch);
		T cos_pitch = ceres::cos(pitch);
		T sin_roll = ceres::sin(roll);
		T cos_roll = ceres::cos(roll);

		//Calcular la matriz de transformacion
		T Rt[4][4];
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
		Rt[3][0] = T(0.);
		Rt[3][1] = T(0.);
		Rt[3][2] = T(0.);
		Rt[3][3] = T(1.);

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y, , const double observed_Z) {
		return (new ceres::AutoDiffCostFunction<alineadorM9, 2, 9, 3>(new alineadorM9(observed_x, observed_y, observed_Z)));
	}

private:
	double P1_x, P1_y, P1_z;
};

//#endif