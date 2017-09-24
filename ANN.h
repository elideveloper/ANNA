#pragma once

#include "Layer.h"


namespace ANNA {

    class ANN {
        Layer hiddenLayer;      // numInput is equal to numNeurons in the hidden layer
        Layer outputLayer;      // numOutput is equal to numNeurons in the output layer
        ActivationFunc activFunc;
        double* output;
        double* hiddenOutput;

    public:
        ANN(int numInput, int numHiddenNeurons, int numOutput, ActivationFunc activationFunc);
        double* computeOutput(double* input);
        double* getOutput() const;
        void backPropagate(double* input, double* rightOutput, double d, ActivationFunc funcDerivative);	// d - шаг обучения
    };
}
