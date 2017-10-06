#include "ANN.h"

#include <limits>
#include <fstream>
#include <queue>
#include <iostream>


namespace ANNA {

    ANN::ANN()
    {
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

    void ANN::init(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod, ANNA::ActivationFunction activFunc)
    {
        this->hiddenLayer = Layer(numHiddenNeurons, numInput);
        this->outputLayer = Layer(numOutput, numHiddenNeurons);
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
                        this->computeOutput(trainInput[i]);                                     // refreshs output
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
                Individual* generation = new Individual[numIndividuals]; // generate first generation
                for (int i = 0; i < numIndividuals; i++) {
                    generation[i].init(numInput, numHidden, numOutput);
                }
                while (m < maxIterations && avgErr > avgError) {
                    generation = this->makeGeneticTransformation(generation, numIndividuals, trainDatasetSize, trainInput, trainOutput);
                    m++;
                    this->importNeuronsWeights(&generation[0]);
                    avgErr = 0.0;
                    for (int i = 0; i < trainDatasetSize; i++) {
                        this->computeOutput(trainInput[i]);                                     // refreshs output
                        avgErr += this->getAvgError(trainOutput[i]);
                    }
                    avgErr /= trainDatasetSize;
                }
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

    void ANN::importNeuronsWeights(Individual* ind) const
    {
        this->hiddenLayer.importWeights(ind->hiddenNeurons);
        this->outputLayer.importWeights(ind->outputNeurons);
    }

    TrainingResult::TrainingResult(int numIter, double avgErr)
    {
        this->numIterations = numIter;
        this->avgError = avgErr;
    }

    ANN::Children::Children(int numInput, int numHidden, int numOutput)
    {
        this->left = new Individual();
        this->right = new Individual();
        this->left->numInput = numInput;
        this->left->numHidden = numHidden;
        this->left->numOutput = numOutput;
        this->right->numInput = numInput;
        this->right->numHidden = numHidden;
        this->right->numOutput = numOutput;
    }

    ANN::Individual::Individual()
    {
    }

    ANN::Individual::~Individual()
    {
        for (int i = 0; i < this->numHidden; i++) {
            delete[] this->hiddenNeurons[i];
        }
        delete[] this->hiddenNeurons;
        for (int i = 0; i < this->numOutput; i++) {
            delete[] this->outputNeurons[i];
        }
        delete[] this->outputNeurons;
    }

    void ANN::Individual::init(int numInput, int numHidden, int numOutput)
    {
        this->hiddenNeurons = new double*[numHidden];
        for (int i = 0; i<numHidden; i++) {
            this->hiddenNeurons[i] = new double[numInput];
            for (int j = 0; j<numInput; j++) {
                this->hiddenNeurons[i][j] = (rand() % 101 - 50) / 100.0;
            }
        }
        this->outputNeurons = new double*[numOutput];
        for (int i = 0; i<numOutput; i++) {
            this->outputNeurons[i] = new double[numHidden];
            for (int j = 0; j<numHidden; j++) {
                this->outputNeurons[i][j] = (rand() % 101 - 50) / 100.0;
            }
        }
        this->numInput = numInput;
        this->numHidden = numHidden;
        this->numOutput = numOutput;
    }

    ANN::Individual* ANN::getBestIndividuals(Individual* generation, int numIndividuals, int trainDatasetSize, double** input, double** correctOutput, int numBest)
    {
        Individual* bestInds = new Individual[numBest];
        std::priority_queue<double> errors;
        double* errorsArr = new double[numIndividuals];
        ANN* anns = new ANN[numIndividuals];
        int numInput = this->hiddenLayer.getNumInputs();
        int numHidden = this->hiddenLayer.getNumNeurons();
        int numOutput = this->outputLayer.getNumNeurons();
        for (int i = 0; i < numIndividuals; i++) {
            anns[i].init(numInput, numHidden, numOutput);
            anns[i].importNeuronsWeights(&generation[i]);
            double avgErr = 0.0;
            for (int j = 0; j < trainDatasetSize; j++) {
                anns[i].computeOutput(input[j]);
                avgErr += anns[i].getAvgError(correctOutput[j]);
            }
            errorsArr[i] = avgErr/trainDatasetSize;
            errors.push(avgErr/trainDatasetSize);
        }

        for (int i = 0; i < numIndividuals - numBest; i++) errors.pop();       // optimize
        for (int i = numBest - 1; i >= 0; i--) {
            for (int j = 0; j < numIndividuals; j++) {
                if (abs(errors.top() - errorsArr[j]) < std::numeric_limits<double>::min()) {
                    bestInds[i] = generation[j];
                    break;
                }
            }
            errors.pop();
        }
        delete[] anns;
        delete[] errorsArr;
        // delete[] left individuals from generation

        return bestInds;
    }

    ANN::Children* ANN::cross(const Individual& mom, const Individual& dad)
    {
        Children* children = new Children(mom.numInput, mom.numHidden, mom.numOutput);
        children->left->hiddenNeurons = new double*[mom.numHidden];
        children->right->hiddenNeurons = new double*[mom.numHidden];
        children->left->outputNeurons = new double*[mom.numOutput];
        children->right->outputNeurons = new double*[mom.numOutput];

        int r = rand() % (mom.numHidden - 1);
        for (int i = 0; i <= r; i++) {
            children->left->hiddenNeurons[i] = new double[mom.numInput];
            children->right->hiddenNeurons[i] = new double[mom.numInput];
            for (int k = 0; k < mom.numInput; k++) {
                children->left->hiddenNeurons[i][k] = mom.hiddenNeurons[i][k];
                children->right->hiddenNeurons[i][k] = dad.hiddenNeurons[i][k];
            }
        }
        for (int i = r + 1; i < mom.numHidden; i++) {
            children->left->hiddenNeurons[i] = new double[mom.numInput];
            children->right->hiddenNeurons[i] = new double[mom.numInput];
            for (int k = 0; k<mom.numInput; k++) {
                children->left->hiddenNeurons[i][k] = dad.hiddenNeurons[i][k];
                children->right->hiddenNeurons[i][k] = mom.hiddenNeurons[i][k];
            }
        }

        r = rand() % (mom.numOutput - 1);
        for (int i = 0; i <= r; i++) {
            children->left->outputNeurons[i] = new double[mom.numHidden];
            children->right->outputNeurons[i] = new double[mom.numHidden];
            for (int k = 0; k < mom.numHidden; k++) {
                children->left->outputNeurons[i][k] = mom.outputNeurons[i][k];
                children->right->outputNeurons[i][k] = dad.outputNeurons[i][k];
            }
        }
        for (int i = r + 1; i < mom.numOutput; i++) {
            children->left->outputNeurons[i] = new double[mom.numHidden];
            children->right->outputNeurons[i] = new double[mom.numHidden];
            for (int k = 0; k < mom.numHidden; k++) {
                children->left->outputNeurons[i][k] = dad.outputNeurons[i][k];
                children->right->outputNeurons[i][k] = mom.outputNeurons[i][k];
            }
        }

        return children;
    }

    ANN::Individual* ANN::makeGeneticTransformation(Individual* generation, int numIndividuals, int trainDatasetSize, double** input, double** correctOutput)
    {
        Individual* nextGen = new Individual[numIndividuals];
        Individual* nextbestInds = getBestIndividuals(generation, numIndividuals, trainDatasetSize, input, correctOutput, 5);
        nextGen[0] = nextbestInds[0];
        nextGen[1] = nextbestInds[1];
        Children* ch1 = this->cross(nextbestInds[2], nextbestInds[3]);
        Children* ch2 = this->cross(nextbestInds[4], nextbestInds[3]);
        Children* ch3 = this->cross(nextbestInds[2], nextbestInds[4]);
        nextGen[2] = *(ch1->left);
        nextGen[3] = *(ch1->right);
        nextGen[4] = *(ch2->left);
        nextGen[5] = *(ch2->right);
        nextGen[6] = *(ch3->left);
        nextGen[7] = *(ch3->right);
        int numInput = this->hiddenLayer.getNumInputs();
        int numHidden = this->hiddenLayer.getNumNeurons();
        int numOutput = this->outputLayer.getNumNeurons();
        nextGen[8].init(numInput, numHidden, numOutput);        // random individuals
        nextGen[9].init(numInput, numHidden, numOutput);
        return nextGen;
    }
}
