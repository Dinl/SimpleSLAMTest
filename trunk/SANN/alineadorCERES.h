// AlineadorCERES.h: define el metodo de calculo de la matriz de alineacion
//

//#ifndef __alineacionCERES_H
//#define __alineacionCERES_H


#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace ceres;


class mifunciondecosto{
public:
	
	//Constructor
	mifunciondecosto(double k): k_(k){};

	//Operador de costo
	template <typename T>
	bool operator()(const T* const x, const T* const y, T* e) const {
		e[0] = T(k_) - x[0] * y[0] - x[1] * y[1];
		return true;
	}

private:
	double k_;
};

int main(int argc, char** argv) {

	double x[2];
	double y[2];

	x[0] = 0.0;
	x[1] = 1.0;
	y[0] = 0.0;
	y[1] = 2.0;

	Problem problem;
	CostFunction* funcioncosto = new AutoDiffCostFunction<mifunciondecosto,1,2,2>(new mifunciondecosto(1.0));
	problem.AddResidualBlock(funcioncosto, NULL, x, y);


	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	return 0;
}

//#endif