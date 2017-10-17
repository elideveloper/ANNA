#include "ANN.h"

#include <limits>
#include <fstream>
#include <queue>


namespace ANNA {

    ANN::ANN() : output(nullptr), hiddenOutput(nullptr), params(nullptr)
    {
    }

    ANN::~ANN()
    {
        delete[] this->output;
        delete[] this->hiddenOutput;
        delete this->params;
    }

    ANN::ANN(int numInput, int numHiddenNeurons, int numOutput, ANNA::ActivationFunction activFunc, ANNA::LearningMethod learnMethod, ANNA::MethodParams* params) : hiddenLayer(numHiddenNeurons, numInput), outputLayer(numOutput, numHiddenNeurons), output(nullptr), hiddenOutput(nullptr), params(params)
    {
        switch (learnMethod) {
            case BP: {
                this->learnMethod = BP;
            }
                break;
            case GA: {
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
        this->params = ann.params;
		for (int i = 0; i < this->outputLayer.getNumNeurons(); i++) {
			this->output[i] = ann.output[i];
		}
		for (int i = 0; i < this->hiddenLayer.getNumNeurons(); i++) {
			this->hiddenOutput[i] = ann.hiddenOutput[i];
		}
	}

    void ANN::init(int numInput, int numHiddenNeurons, int numOutput, ANNA::ActivationFunc activFunc, ANNA::ActivationFunc activFuncDeriv, ANNA::LearningMethod learnMethod, ANNA::MethodParams* params)
    {
        this->output = nullptr;
        this->hiddenOutput = nullptr;
        this->params = params;
        this->hiddenLayer.init(numInput, numHiddenNeurons);
        this->outputLayer.init(numHiddenNeurons, numOutput);
        this->activFunc = activFunc;
        this->activFuncDerivative = activFuncDeriv;
        switch (learnMethod) {
            case BP: {
                this->learnMethod = BP;
            }
                break;
            case GA: {
                this->learnMethod = GA;
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

    double ANN::backPropagate(double* input, double* correctOutput)
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

        BPParams* bpParams = static_cast<BPParams*>(this->params);

        // hidden layer weights correcting
        this->hiddenLayer.correctWeights(input, hiddenErrors, bpParams->learningSpeed, this->activFuncDerivative);
        delete[] hiddenErrors;

        // output layer weights correcting
        this->outputLayer.correctWeights(this->hiddenOutput, outErrors, bpParams->learningSpeed, this->activFuncDerivative);
        delete[] outErrors;

        return (err / numOutput);
    }

    TrainingResult ANN::train(int trainDatasetSize, double** trainInput, double** trainOutput, int pretestDatasetSize, double** pretestInput, double** pretestOutput, double acceptableError)
    {
        double avgErr = std::numeric_limits<double>::max();
        int m = 0;

        switch (this->learnMethod) {
            case BP: {
                int maxIterations = static_cast<BPParams*>(this->params)->repetitionFactor * trainDatasetSize;
                while (m < maxIterations && avgErr > acceptableError) {
                    for (int i = 0; i < trainDatasetSize; i++, m++) {
                        this->backPropagate(trainInput[i], trainOutput[i]);
                    }
                    avgErr = 0.0;
                    for (int i = 0; i < pretestDatasetSize; i++) {
                        this->computeOutput(pretestInput[i]);
                        avgErr += this->getAvgError(pretestOutput[i]);
                    }
                    avgErr /= pretestDatasetSize;
                }
            }
                break;
            case GA: {
				Individual** generation = this->createRandomGeneration();									// create first generation
                while (m < static_cast<GAParams*>(this->params)->maxGenerations && avgErr > acceptableError) {
                    this->goToNextGeneration(generation, pretestDatasetSize, pretestInput, pretestOutput);
                    m++;
                    this->importNeuronsWeights(*generation[0]);
                    avgErr = 0.0;
                    for (int i = 0; i < pretestDatasetSize; i++) {
                        this->computeOutput(pretestInput[i]);
                        avgErr += this->getAvgError(pretestOutput[i]);
                    }
                    avgErr /= pretestDatasetSize;
                }
				this->destroyGeneration(generation);
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

	ANN::Individual* ANN::getSelfIndivid() const
	{
		Individual* ind = new Individual(this->hiddenLayer.getNumInputs(), this->hiddenLayer.getNumNeurons(), this->outputLayer.getNumNeurons());
		for (int i = 0; i < ind->numHidden; i++) {
			ind->hiddenNeurons[i].importWeights(this->hiddenLayer.getNeuron(i));
		}
		for (int i = 0; i < ind->numOutput; i++) {
			ind->outputNeurons[i].importWeights(this->outputLayer.getNeuron(i));
		}
		return ind;
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
            this->hiddenNeurons[i].init(numInput);
		}
		this->outputNeurons = new Neuron[numOutput];
		for (int i = 0; i < numOutput; i++) {
            this->outputNeurons[i].init(numHidden);
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
		this->hiddenNeurons = new Neuron[this->numHidden];
		for (int i = 0; i < this->numHidden; i++) {
			this->hiddenNeurons[i].importWeights(ind.hiddenNeurons[i]);
		}
		this->outputNeurons = new Neuron[this->numOutput];
		for (int i = 0; i < this->numOutput; i++) {
			this->outputNeurons[i].importWeights(ind.outputNeurons[i]);
		}
	}

	ANN::Individual& ANN::Individual::operator=(const Individual& ind)
	{
		this->numInput = ind.numInput;
		this->numHidden = ind.numHidden;
		this->numOutput = ind.numOutput;
		this->hiddenNeurons = new Neuron[this->numHidden];
		for (int i = 0; i < this->numHidden; i++) {
			this->hiddenNeurons[i].importWeights(ind.hiddenNeurons[i]);
		}
		this->outputNeurons = new Neuron[this->numOutput];
		for (int i = 0; i < this->numOutput; i++) {
			this->outputNeurons[i].importWeights(ind.outputNeurons[i]);
		}
		return *this;
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

	void ANN::Individual::tryToMutate(int mutationPercent)
	{
        if (mutationPercent <= 0) return;
        if (mutationPercent > 100) mutationPercent = 100;

		int rNo = rand() % (this->numHidden * (100 / mutationPercent));
		if (rNo < this->numHidden) {
			this->hiddenNeurons[rNo].importWeights(Neuron(this->numInput));
		}
		rNo = rand() % (this->numOutput * (100 / mutationPercent));
		if (rNo < this->numOutput) {
			this->outputNeurons[rNo].importWeights(Neuron(this->numHidden));
		}
	}

	ANN::Individual** ANN::createRandomGeneration()
	{
		int numIndividuals = static_cast<GAParams*>(this->params)->generationSize;
		Individual** generation = new Individual*[numIndividuals];									// create first generation
		generation[0] = this->getSelfIndivid();
		for (int i = 1; i < numIndividuals; i++) {
            generation[i] = new Individual(this->hiddenLayer.getNumInputs(), this->hiddenLayer.getNumNeurons(), this->outputLayer.getNumNeurons());
		}
		return generation;
	}

	void ANN::destroyGeneration(Individual ** generation)
	{
		int numIndividuals = static_cast<GAParams*>(this->params)->generationSize;
		for (int i = 0; i < numIndividuals; i++) {
			delete generation[i];
		}
		delete[] generation;
	}

    void ANN::sortIndividuals(Individual** generation, int datasetSize, double** input, double** correctOutput)
    {
        GAParams* gaParams = static_cast<GAParams*>(this->params);
        int numIndividuals = gaParams->generationSize;

        std::priority_queue<double> errors;
        double* errorsArr = new double[numIndividuals];
        ANN* anns = new ANN[numIndividuals]();
        for (int i = 0; i < numIndividuals; i++) {
            anns[i].init(this->hiddenLayer.getNumInputs(), this->hiddenLayer.getNumNeurons(), this->outputLayer.getNumNeurons(), this->activFunc);
            anns[i].importNeuronsWeights(*generation[i]);
            double avgErr = 0.0;
            for (int j = 0; j < datasetSize; j++) {
                anns[i].computeOutput(input[j]);
                avgErr += anns[i].getAvgError(correctOutput[j]);
            }
            errorsArr[i] = avgErr / datasetSize;
            errors.push(errorsArr[i]);
        }

		Individual* ind = nullptr;
        double err = 0.0;
        for (int i = numIndividuals - 1; i >= 0; i--) {
            for (int j = 0; j < numIndividuals; j++) {
                if (abs(errors.top() - errorsArr[j]) < std::numeric_limits<double>::min()) {
					ind = generation[j];
					generation[j] = generation[i];
					generation[i] = ind;
                    err = errorsArr[j];
                    errorsArr[j] = errorsArr[i];
                    errorsArr[i] = err;
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

    void ANN::goToNextGeneration(Individual** generation, int trainDatasetSize, double** input, double** correctOutput)
    {
        sortIndividuals(generation, trainDatasetSize, input, correctOutput);

        GAParams* gaParams = static_cast<GAParams*>(this->params);

		// now works only for even number of children
        int numChildren = gaParams->generationSize - gaParams->numRandomIndividuals - gaParams->numLeaveBest;
		int numParents = numChildren / 2;

		Individual** ch = new Individual*[numChildren]();												// two children for each parent
		int momIndex = 0, dadIndex = 0;
		for (int i = 0, j = 0; j < numParents; i+=2, j++) {
            momIndex = j + gaParams->numLeaveBest;
            dadIndex = (j < numParents - 1) ? momIndex + 1 : gaParams->numLeaveBest;
			Children* children = this->cross(*generation[momIndex], *generation[dadIndex]);
			ch[i] = children->left;
			ch[i + 1] = children->right;
		}

        for (int i = gaParams->numLeaveBest; i < gaParams->numLeaveBest + numChildren; i++) {
			delete generation[i];
            generation[i] = ch[i - gaParams->numLeaveBest];
            generation[i]->tryToMutate(gaParams->mutationPercent);
		}
		delete[] ch;

		// random individuals
        for (int i = gaParams->generationSize - gaParams->numRandomIndividuals; i < gaParams->generationSize; i++) {
			generation[i]->refresh(this->hiddenLayer.getNumInputs(), this->hiddenLayer.getNumNeurons(), this->outputLayer.getNumNeurons());
		}
    }

	TrainingResult::TrainingResult(int numIter, double avgErr)
	{
		this->numIterations = numIter;
		this->avgError = avgErr;
	}

    MethodParams::~MethodParams()
    {
    }

    GAParams::GAParams(int generationSize, int numLeaveBest, int numRandomIndividuals, int mutationPercent, int maxGenerations) : mutationPercent(mutationPercent), maxGenerations(maxGenerations)
	{
		// check for (generationSize - numLeaveBest - numRandomIndividuals) is even
		this->generationSize = generationSize;
		this->numLeaveBest = numLeaveBest;
		this->numRandomIndividuals = numRandomIndividuals;
	}

    BPParams::BPParams(double learningSpeed, int repetitionFactor) : learningSpeed(learningSpeed), repetitionFactor(repetitionFactor)
    {
    }
}
