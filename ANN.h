#pragma once
#ifndef ANN_H
#define ANN_H

#include "Layer.h"
#include "enums.h"


namespace ANNA {

    class ANN {
        Layer hiddenLayer;      // numInput is equal to numNeurons in the hidden layer
        Layer outputLayer;      // numOutput is equal to numNeurons in the output layer
        ActivationFunc activFunc;
        ActivationFunc activFuncDerivative;
        double* output;
        double* hiddenOutput;

    public:
        ANN(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod, ANNA::ActivationFunction activFunc = ANNA::UNDEFINED_ACTIVATION_FUNCTION);
        double* computeOutput(double* input);
        double* getOutput() const;
        double backPropagate(double* input, double* rightOutput, double d);	// d - learning speed
        double train(int trainDatasetSize, double** trainInput, double** trainOutput, double d, double avgError, int maxIterations);
    };
}

#endif
