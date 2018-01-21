#ifndef ANN_H
#define ANN_H

#include "Layer.h"
#include "enums.h"
#include "utility.h"


namespace ANNA {

	struct TrainingResult {
		int numIterations;
		double avgError;
		TrainingResult(int numIter, double avgErr);
	};

    struct MethodParams {
        virtual ~MethodParams() = 0;
		virtual MethodParams* clone() = 0;
    };

    struct GAParams : MethodParams {
        int populationSize;                 // number of Individuals in one generation
        int numElite;                   // num of best Individuals which go to the next generation
		int numNewcomers;
		int numParents;
		double mutationProbab;	
        int maxGenerations;
        GAParams(int populationSize, double elitePercentage, double parentsPercentage, double newcomersPercentage, double mutationProbab, int maxGenerations);
		MethodParams* clone() final;
	};

    struct BPParams : MethodParams {
        double learningSpeed;
        int repetitionFactor;
        BPParams(double learningSpeed, int repetitionFactor);
		MethodParams* clone() final;
    };

    class ANN {
        Layer hiddenLayer;                  // numInput is equal to numNeurons in the hidden layer
        Layer outputLayer;                  // numOutput is equal to numNeurons in the output layer
        ActivationFunc activFunc;
        ActivationFunc activFuncDerivative;
		LearningMethod learnMethod;
        double* output;
        double* hiddenOutput;
        MethodParams* params;
		double backPropagate(double* input, double* correctOutput);
        struct Individual {
            int numInput;
            int numHidden;
            int numOutput;
            Neuron* hiddenNeurons;
			Neuron* outputNeurons;
			Individual();
            Individual(int numInput, int numHidden, int numOutput);
			Individual(const Individual& ind);
			Individual& operator=(const Individual& ind);
            void init(int numInput, int numHidden, int numOutput);
			void getCrossed(const Individual& mom, const Individual& dad, double mutateProb);
            ~Individual();
        };
		Individual** createRandomGeneration();
		void destroyGeneration(Individual** generation);
        void sortIndividuals(Individual** generation, int datasetSize, double** input, double** correctOutput);
        void goToNextGeneration(Individual** generation, int trainDatasetSize, double** input, double** correctOutput);      // get generation and returns next generation
        void importNeuronsWeights(const Individual& ind) const;
		Individual* getSelfIndivid() const;
    public:
        ANN();
        ANN(int numInput, int numHiddenNeurons, int numOutput, ANNA::ActivationFunction activFunc = ANNA::TANH_FUNCTION, ANNA::LearningMethod learnMethod = ANNA::BP, ANNA::MethodParams* params = nullptr);
		ANN(const ANN& ann);
		ANN& operator=(const ANN& ann);
		~ANN();
        void init(int numInput, int numHiddenNeurons, int numOutput, ANNA::ActivationFunc activFunc, ANNA::ActivationFunc activFuncDeriv = nullptr, ANNA::LearningMethod learnMethod = ANNA::BP, ANNA::MethodParams* params = nullptr);
        double* computeOutput(double* input);
        double* getOutput() const;
        double getAvgError(double* correctOutput) const;
        TrainingResult train(int trainDatasetSize, double** trainInput, double** trainOutput, int pretestDatasetSize, double** pretestInput, double** pretestOutput, double acceptableError);
        void exportNeuronsWeights() const;
        void importNeuronsWeights() const;
    };
}

#endif
