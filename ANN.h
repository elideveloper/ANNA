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
        double backPropagate(double* input, double* correctOutput, double d);                       // d - learning speed
        struct Individual {
            int numInput;
            int numHidden;
            int numOutput;
            Neuron* hiddenNeurons;
			Neuron* outputNeurons;
			Individual();
            Individual(int numInput, int numHidden, int numOutput);
			Individual(const Individual& ind);
            void init(int numInput, int numHidden, int numOutput);
			void refresh(int numInput, int numHidden, int numOutput);
            ~Individual();
        };
        struct Children {
            Individual* left;
            Individual* right;
            Children(int numInput, int numHidden, int numOutput);
        };
		void ANN::sortIndividuals(Individual** generation, int numIndividuals, int trainDatasetSize, double** input, double** correctOutput, int numBest);
        Children* cross(const Individual& mom, const Individual& dad);
		void ANN::goToNextGeneration(Individual** generation, int numIndividuals, int trainDatasetSize, double** input, double** correctOutput);      // get generation and returns next generation
        void importNeuronsWeights(const Individual& ind) const;
    public:
        ANN();
        ANN(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod = ANNA::BP, ANNA::ActivationFunction activFunc = ANNA::TANH_FUNCTION);
		ANN(const ANN& ann);
		~ANN();
        void init(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod = ANNA::BP, ANNA::ActivationFunction activFunc = ANNA::TANH_FUNCTION);
        double* computeOutput(double* input);
        double* getOutput() const;
        double getAvgError(double* correctOutput) const;
		TrainingResult train(int trainDatasetSize, double** trainInput, double** trainOutput, double d, double avgError, int maxIterations);
        void exportNeuronsWeights() const;
        void importNeuronsWeights() const;
    };
}

#endif
