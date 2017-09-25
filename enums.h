#pragma once
#ifndef ENUMS_H
#define ENUMS_H


namespace ANNA {

    enum LearningMethod {
        BP, // Backpropagation
        GA  // Genetic algorithm
    };

    enum ActivationFunction {
        LOGISTIC_FUNCTION,   // (1 / (1 + exp(-x)))
		TANH_FUNCTION		// ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
    };
}


#endif // ENUMS_H
