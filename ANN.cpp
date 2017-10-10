#include "ANN.h"

#include <limits>
#include <fstream>
#include <queue>


namespace ANNA {

    ANN::ANN() : output(nullptr), hiddenOutput(nullptr)
    {
    }

    ANN::~ANN()
    {
        delete[] this->output;
        delete[] this->hiddenOutput;
    }


    ANN::ANN(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod, ANNA::ActivationFunction activFunc) : hiddenLayer(numHiddenNeurons, numInput), outputLayer(numOutput, numHiddenNeurons), output(nullptr), hiddenOutput(nullptr)
    {
        switch (learnMethod) {
            case BP: {
                this->learnMethod = BP;
            }
                break;
            case GA: {
                // GA details
                this->learnMethod = GA;
            }
                break;
            default: {
                // error
            }
        }
        switch (activFunc) {
            case LOGISTIC_FUNCTION: {
                this->activFunc = logisticFunction;
                this->activFuncDerivative = logisticDerivReceivingLogisticVal;
            }
                break;
            case TANH_FUNCTION: {
                this->activFunc = tanhFunction;
                this->activFuncDerivative = tanhDerivReceivingTanhVal;
            }
                break;
            default: {
                // error
            }
        }
    }

	ANN::ANN(const ANN& ann)
	{
		this->hiddenLayer = ann.hiddenLayer;
		this->outputLayer = ann.outputLayer;
		this->activFunc = ann.activFunc;
		this->activFuncDerivative = ann.activFuncDerivative;
		this->learnMethod = ann.learnMethod;
		for (int i = 0; i < this->outputLayer.getNumNeurons(); i++) {
			this->output[i] = ann.output[i];
		}
		for (int i = 0; i < this->hiddenLayer.getNumNeurons(); i++) {
			this->hiddenOutput[i] = ann.hiddenOutput[i];
		}
	}

    void ANN::init(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod, ANNA::ActivationFunction activFunc)
    {
        this->output = nullptr;
        this->hiddenOutput = nullptr;
        this->hiddenLayer.init(numInput, numHiddenNeurons);
        this->outputLayer.init(numHiddenNeurons, numOutput);
        switch (learnMethod) {
            case BP: {
                this->learnMethod = BP;
            }
                break;
            case GA: {
                // GA details
                this->learnMethod = GA;
            }
                break;
            default: {
                // error
            }
        }
        switch (activFunc) {
            case LOGISTIC_FUNCTION: {
                this->activFunc = logisticFunction;
                this->activFuncDerivative = logisticDerivReceivingLogisticVal;
            }
                break;
            case TANH_FUNCTION: {
                this->activFunc = tanhFunction;
                this->activFuncDerivative = tanhDerivReceivingTanhVal;
            }
                break;
            default: {
                // error
            }
        }
    }

    double* ANN::computeOutput(double* input)
    {
        delete[] this->output;
        delete[] this->hiddenOutput;
        this->hiddenOutput = this->hiddenLayer.computeOutput(input, this->activFunc);
        this->output = this->outputLayer.computeOutput(this->hiddenOutput, this->activFunc);
        return this->output;
    }

    double* ANN::getOutput() const
    {
        return this->output;
    }

    double ANN::getAvgError(double* correctOutput) const
    {
        double err = 0.0;
        int numOutput = this->outputLayer.getNumNeurons();
        for (int i = 0; i < numOutput; i++) {
            err += abs(correctOutput[i] - this->output[i]);
        }
        return (err / numOutput);
    }

    double ANN::backPropagate(double* input, double* correctOutput, double d)
    {
		this->computeOutput(input);																	// refreshs output

        int numOutput = this->outputLayer.getNumNeurons();
        double err = 0.0;

        // output layer errors
        double* outErrors = new double[numOutput];
        for (int i = 0; i < numOutput; i++) {
            outErrors[i] = correctOutput[i] - this->output[i];
            err += fabs(outErrors[i]);
        }

        // hidden layer errors
        double* hiddenErrors = this->hiddenLayer.computeLayerErrors(outErrors, this->outputLayer);

        // hidden layer weights correcting
        this->hiddenLayer.correctWeights(input, hiddenErrors, d, this->activFuncDerivative);
        delete[] hiddenErrors;

        // output layer weights correcting
        this->outputLayer.correctWeights(this->hiddenOutput, outErrors, d, this->activFuncDerivative);
        delete[] outErrors;

        return (err / numOutput);
    }

    TrainingResult ANN::train(int trainDatasetSize, double** trainInput, double** trainOutput, double d, double avgError, int maxIterations)
    {
        double avgErr = std::numeric_limits<double>::max();
        int m = 0;

        switch (this->learnMethod) {
            case BP: {
                // while avg error will not be acceptable or max iterations
                while (m < maxIterations && avgErr > avgError) {
                    avgErr = 0.0;
                    for (int i = 0; i < trainDatasetSize; i++, m++) {
                        avgErr += this->backPropagate(trainInput[i], trainOutput[i], d);
                    }
                    avgErr /= trainDatasetSize;
                }
            }
                break;
            case GA: {
                int numIndividuals = 10;
                int numInput = this->hiddenLayer.getNumInputs();
                int numHidden = this->hiddenLayer.getNumNeurons();
                int numOutput = this->outputLayer.getNumNeurons();
                Individual** generation = new Individual*[numIndividuals];									// create first generation
                for (int i = 0; i < numIndividuals; i++) {
                    generation[i] = new Individual(numInput, numHidden, numOutput);
                }
                while (m < maxIterations) {
                    this->goToNextGeneration(generation, numIndividuals, trainDatasetSize, trainInput, trainOutput);
                    m++;
                }
				avgErr = 0.0;
				this->importNeuronsWeights(*generation[0]);
				for (int i = 0; i < trainDatasetSize; i++) {
					this->computeOutput(trainInput[i]);														// refreshs output
					avgErr += this->getAvgError(trainOutput[i]);
				}
				avgErr /= trainDatasetSize;
				delete[] generation;
            }
                break;
            default: {

            }
        }

        return TrainingResult(m, avgErr);
    }

    void ANN::exportNeuronsWeights() const
    {
        std::ofstream outFile("hidden_layer.txt");
        this->hiddenLayer.exportWeights(outFile);
        outFile.close();
        outFile.open("output_layer.txt");
        this->outputLayer.exportWeights(outFile);
        outFile.close();
    }

    void ANN::importNeuronsWeights() const
    {
        std::ifstream inFile("hidden_layer.txt");
        this->hiddenLayer.importWeights(inFile);
        inFile.close();
        inFile.open("output_layer.txt");
        this->outputLayer.importWeights(inFile);
        inFile.close();
    }

    void ANN::importNeuronsWeights(const Individual& ind) const
    {
        this->hiddenLayer.importWeights(ind.hiddenNeurons);
        this->outputLayer.importWeights(ind.outputNeurons);
    }

    TrainingResult::TrainingResult(int numIter, double avgErr)
    {
        this->numIterations = numIter;
        this->avgError = avgErr;
    }

    ANN::Children::Children(int numInput, int numHidden, int numOutput) : left(new Individual(numInput, numHidden, numOutput)), right(new Individual(numInput, numHidden, numOutput))
    {
    }

	ANN::Individual::Individual() : numInput(0), numHidden(0), numOutput(0), hiddenNeurons(nullptr), outputNeurons(nullptr)
	{
	}

    ANN::Individual::Individual(int numInput, int numHidden, int numOutput) : numInput(numInput), numHidden(numHidden), numOutput(numOutput)
    {
		this->hiddenNeurons = new Neuron[numHidden];
		for (int i = 0; i < numHidden; i++) {
			this->hiddenNeurons[i] = Neuron(numInput);
		}
		this->outputNeurons = new Neuron[numOutput];
		for (int i = 0; i < numOutput; i++) {
			this->outputNeurons[i] = Neuron(numHidden);
		}
    }

    ANN::Individual::~Individual()
    {
        delete[] this->hiddenNeurons;
        delete[] this->outputNeurons;
    }

	ANN::Individual::Individual(const Individual& ind) {
		this->numInput = ind.numInput;
		this->numHidden = ind.numHidden;
		this->numOutput = ind.numOutput;
		this->hiddenNeurons = new Neuron[numHidden];
		for (int i = 0; i < numHidden; i++) {
			this->hiddenNeurons[i].importWeights(ind.hiddenNeurons[i]);
		}
		this->outputNeurons = new Neuron[numOutput];
		for (int i = 0; i < numOutput; i++) {
			this->outputNeurons[i].importWeights(ind.outputNeurons[i]);
		}
	}

    void ANN::Individual::init(int numInput, int numHidden, int numOutput)
    {
        this->hiddenNeurons = new Neuron[numHidden]();
        for (int i = 0; i < numHidden; i++) {
			this->hiddenNeurons[i].init(numInput);
        }
        this->outputNeurons = new Neuron[numOutput]();
        for (int i = 0; i < numOutput; i++) {
            this->outputNeurons[i].init(numHidden);
        }
        this->numInput = numInput;
        this->numHidden = numHidden;
        this->numOutput = numOutput;
    }

	void ANN::Individual::refresh(int numInput, int numHidden, int numOutput)
	{
		delete[] this->hiddenNeurons;
		delete[] this->outputNeurons;
		this->init(numInput, numHidden, numOutput);
	}

    void ANN::sortIndividuals(Individual** generation, int numIndividuals, int trainDatasetSize, double** input, double** correctOutput, int numBest)
    {
		int numInput = this->hiddenLayer.getNumInputs();
		int numHidden = this->hiddenLayer.getNumNeurons();
		int numOutput = this->outputLayer.getNumNeurons();

        std::priority_queue<double> errors;
        double* errorsArr = new double[numIndividuals];
        ANN* anns = new ANN[numIndividuals]();
        for (int i = 0; i < numIndividuals; i++) {
            anns[i].init(numInput, numHidden, numOutput);
            anns[i].importNeuronsWeights(*generation[i]);
            double avgErr = 0.0;
            for (int j = 0; j < trainDatasetSize; j++) {
                anns[i].computeOutput(input[j]);
                avgErr += anns[i].getAvgError(correctOutput[j]);
            }
            errorsArr[i] = avgErr / trainDatasetSize;
            errors.push(errorsArr[i]);
        }

		Individual* ind = nullptr;
        for (int i = numIndividuals - 1; i >= 0; i--) {
            for (int j = 0; j < numIndividuals; j++) {
                if (abs(errors.top() - errorsArr[j]) < std::numeric_limits<double>::min()) {
					ind = generation[j];
					generation[j] = generation[i];
					generation[i] = ind;
                    break;
                }
            }
            errors.pop();
        }

		delete[] anns;
		delete[] errorsArr;
    }

    ANN::Children* ANN::cross(const Individual& mom, const Individual& dad)
    {
        Children* children = new Children(mom.numInput, mom.numHidden, mom.numOutput);

		// crossover hidden layer
        int r = rand() % (mom.numHidden - 1);
        for (int i = 0; i <= r; i++) {
			children->left->hiddenNeurons[i].importWeights(mom.hiddenNeurons[i]);
			children->right->hiddenNeurons[i].importWeights(dad.hiddenNeurons[i]);
        }
        for (int i = r + 1; i < mom.numHidden; i++) {
			children->left->hiddenNeurons[i].importWeights(dad.hiddenNeurons[i]);
			children->right->hiddenNeurons[i].importWeights(mom.hiddenNeurons[i]);
        }

		// crossover output layer
        r = rand() % (mom.numOutput - 1);
        for (int i = 0; i <= r; i++) {
			children->left->outputNeurons[i].importWeights(mom.outputNeurons[i]);
			children->right->outputNeurons[i].importWeights(dad.outputNeurons[i]);
        }
        for (int i = r + 1; i < mom.numOutput; i++) {
			children->left->outputNeurons[i].importWeights(dad.outputNeurons[i]);
			children->right->outputNeurons[i].importWeights(mom.outputNeurons[i]);
        }

        return children;
    }

    void ANN::goToNextGeneration(Individual** generation, int numIndividuals, int trainDatasetSize, double** input, double** correctOutput)
    {
        sortIndividuals(generation, numIndividuals, trainDatasetSize, input, correctOutput, 5);
        Children* ch1 = this->cross(*generation[2], *generation[3]);
        Children* ch2 = this->cross(*generation[4], *generation[3]);
        Children* ch3 = this->cross(*generation[2], *generation[4]);
		delete generation[2];
		delete generation[3];
		delete generation[4];
		delete generation[5];
		delete generation[6];
		delete generation[7];
		generation[2] = ch1->left;
		generation[3] = ch1->right;
		generation[4] = ch2->left;
		generation[5] = ch2->right;
		generation[6] = ch3->left;
		generation[7] = ch3->right;
		// random individuals
		generation[8]->refresh(this->hiddenLayer.getNumInputs(), this->hiddenLayer.getNumNeurons(), this->outputLayer.getNumNeurons());
		generation[9]->refresh(this->hiddenLayer.getNumInputs(), this->hiddenLayer.getNumNeurons(), this->outputLayer.getNumNeurons());
    }
}
