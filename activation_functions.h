#pragma once

#include <math.h>


namespace ANNA {

    typedef double(*ActivationFunc)(double inpSum);

    double logisticFunction(double inpSum);

    double logisticFunctionDerivative(double inpSum);
}
