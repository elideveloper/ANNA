#include "activation_functions.h"


namespace ANNA {

    double logisticFunction(double x) {
        return (1 / (1 + exp(-x)));
    }

    double logisticFunctionDerivative(double x) {
        return (logisticFunction(x) * (1 - logisticFunction(x)));
    }

	double tanhFunction(double x) {
		return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
	}

	double tanhFunctionDerivative(double x) {
        return (1 - sqrt(tanhFunction(x)));
	}
}
