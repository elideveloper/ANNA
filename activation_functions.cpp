#include "activation_functions.h"


namespace ANNA {

    double logisticFunction(double inpSum) {
        return (1 / (1 + exp(-inpSum)));
    }

    double logisticFunctionDerivative(double inpSum) {
        return logisticFunction(inpSum) * (1 - logisticFunction(inpSum));
    }
}
