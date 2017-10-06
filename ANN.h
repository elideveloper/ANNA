#pragma once
#ifndef ANN_H
#define ANN_H

#include "Layer.h"
#include "enums.h"


namespace ANNA {

	struct TrainingResult {
		int numIterations;
		double avgError;

		TrainingResult(int numIter, double avgErr);
	};

    class ANN {
        Layer hiddenLayer;      // numInput is equal to numNeurons in the hidden layer
        Layer outputLayer;      // numOutput is equal to numNeurons in the output layer
        ActivationFunc activFunc;
        ActivationFunc activFuncDerivative;
		LearningMethod learnMethod;
        double* output;
        double* hiddenOutput;
    public:
        ANN(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod = ANNA::BP, ANNA::ActivationFunction activFunc = ANNA::TANH_FUNCTION);
        double* computeOutput(double* input);
        double* getOutput() const;
        double backPropagate(double* input, double* correctOutput, double d);																							// d - learning speed
		TrainingResult train(int trainDatasetSize, double** trainInput, double** trainOutput, double d, double avgError, int maxIterations);
        void exportNeuronsWeights() const;
        void importNeuronsWeights() const;
    };
}

#endif
