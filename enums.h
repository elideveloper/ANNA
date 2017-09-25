#pragma once
#ifndef ENUMS_H
#define ENUMS_H


namespace ANNA {

    enum LearningMethod {
        BP, // Backpropagation
        GA  // Genetic algorithm
    };

    enum ActivationFunction {
        UNDEFINED_ACTIVATION_FUNCTION = 0,
        LOGISTIC_FUNCTION,   // (1 / (1 + exp(-x)))
		TANH_FUNCTION
    };
}


#endif // ENUMS_H
