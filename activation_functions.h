#pragma once
#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <math.h>


namespace ANNA {

    typedef double(*ActivationFunc)(double inpSum);

    double logisticFunction(double x);
    double logisticFunctionDerivative(double x);
	double tanhFunction(double x);
	double tanhFunctionDerivative(double x);
}

#endif
