#include "ANN.h"

#include <limits>


namespace ANNA {

    ANN::ANN(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod, ANNA::ActivationFunction activFunc) : hiddenLayer(numHiddenNeurons, numInput), outputLayer(numOutput, numHiddenNeurons)
    {
        switch (learnMethod) {
            case BP: {
            switch (activFunc) {
            case LOGISTIC_FUNCTION: {
                this->activFunc = logisticFunction;
                this->activFuncDerivative = logisticFunctionDerivative;
            }
                break;
            default: {
                // error
            }
            }
        }
            break;
        case GA: {
            // GA details
        }
        default: {
            // error
        }
        }
    }

    double* ANN::computeOutput(double* input)
    {
        this->hiddenOutput = hiddenLayer.computeOutput(input, this->activFunc);
        this->output = outputLayer.computeOutput(this->hiddenOutput, this->activFunc);
        return this->output;
    }

    double* ANN::getOutput() const
    {
        return this->output;
    }

    double ANN::backPropagate(double* input, double* correctOutput, double d)
    {
        double err = 0.0;
        int numOutput = this->outputLayer.getNumNeurons();
        int numHidden = this->hiddenLayer.getNumNeurons();
        int numInput = this->hiddenLayer.getNumInputs();

        double* outErrors = new double[numOutput];
        for (int i = 0; i < numOutput; i++) {
            outErrors[i] = correctOutput[i] - this->output[i];
            err += outErrors[i];
        }
        err /= numOutput;                                           // avg output error

        double* hiddenErrors = new double[numHidden];
        for (int i = 0; i < numHidden; i++) {
            hiddenErrors[i] = 0.0;
            double* weights = this->outputLayer.getWeightsForNeuron(i);
            for (int j = 0; j < numOutput; j++) {
                hiddenErrors[i] += outErrors[j] * weights[j];
            }
            delete[] weights;
        }

        // hidden layer weights correcting
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInput; j++) {
                this->hiddenLayer.correctNeuronWeight(i, j, input, hiddenErrors[i], d, this->activFuncDerivative);
            }
        }
		delete[] hiddenErrors;

        // output layer weights correcting
        for (int i = 0; i < numOutput; i++) {
            for (int j = 0; j < numHidden; j++) {
                this->outputLayer.correctNeuronWeight(i, j, this->hiddenOutput, outErrors[i], d, this->activFuncDerivative);
            }
        }
		delete[] outErrors;

        return fabs(err);
    }

    double ANN::train(int trainDatasetSize, double** trainInput, double** trainOutput, double d, double avgError, int maxIterations)
    {
        double avgErr = std::numeric_limits<double>::max();
        int m = 0;

        // while avg error will not be acceptable or max iterations
        while (m < maxIterations && avgErr > avgError) {
            avgErr = 0.0;
            for (int i = 0; i < trainDatasetSize; i++, m++) {
                avgErr += this->backPropagate(trainInput[i], trainOutput[i], d);
                this->computeOutput(trainInput[i]);                                     // refreshs output
            }
            avgErr /= trainDatasetSize;
        }
        std::cout << "Number of iterations: " << m << std::endl;
        return avgErr;
    }
}
