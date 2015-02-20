// AlineadorCERES.h: define el metodo de calculo de la matriz de alineacion
//

//#ifndef __alineacionCERES_H
//#define __alineacionCERES_H


#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace ceres;


class alineadorM9{
public:
	
	//Constructor, el punto de la primera imagen 
	alineadorM9(double observed_x, double observed_y){
		P1_x = observed_x;
		P1_y = observed_y;
	};

	//Operador de costo
	template <typename T>
	bool operator()(const T* const Tmatrix, const T* const punto, T* residuos) const {
		
		//Computar la rotacion
		T p[3];
		ceres::AngleAxisRotatePoint(Tmatrix, punto, p);

		//Computar la traslacion
		p[0] += Tmatrix[3]; 
		p[1] += Tmatrix[4]; 
		p[2] += Tmatrix[5];

		//Computar los centros de distorcion
		T xp = - p[0] / p[2];
		T yp = - p[1] / p[2];

		//Aplicar la distorsion radial
		const T& l1 = Tmatrix[7];
		const T& l2 = Tmatrix[8];
		T r2 = xp*xp + yp*yp;
		T distortion = T(1.0) + r2  * (l1 + l2  * r2);

		//Predecir la posicion final
		const T& focal = Tmatrix[6];
		T predicted_x = focal * distortion * xp;
		T predicted_y = focal * distortion * yp;

		// The error is the difference between the predicted and observed position.
		residuos[0] = predicted_x - T(P1_x);
		residuos[1] = predicted_y - T(P1_y);

		return true;
	}

	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<alineadorM9, 2, 9, 3>(new alineadorM9(observed_x, observed_y)));
	}

private:
	double P1_x, P1_y;
};

//#endif