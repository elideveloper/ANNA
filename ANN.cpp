#include "ANN.h"


namespace ANNA {

    ANN::ANN(int numInput, int numHiddenNeurons, int numOutput, ActivationFunc activationFunc) : hiddenLayer(numHiddenNeurons, numInput), outputLayer(numOutput, numHiddenNeurons)
    {
        this->activFunc = activationFunc;
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

    void ANN::backPropagate(double* input, double* rightOutput, double d, ActivationFunc funcDerivative)
{
	int numOutput = this->outputLayer.getNumNeurons();
	int numHidden = this->hiddenLayer.getNumNeurons();
	int numInput = this->hiddenLayer.getNumInputs();

    double* outErrors = new double[numOutput];
	for (int i = 0; i < numOutput; i++) {
		outErrors[i] = rightOutput[i] - this->output[i];
	}

	double* hiddenErrors = new double[numHidden];
	for (int i = 0; i < numHidden; i++) {
		hiddenErrors[i] = 0.0;
		double* weights = this->outputLayer.getWeightsForNeuron(i);
		for (int j = 0; j < numOutput; j++) {
			hiddenErrors[i] += outErrors[j] * weights[j];
		}
		delete[] weights;
	}

	// корректируем входные веса для скрытого слоя
	for (int i = 0; i < numHidden; i++) {
		for (int j = 0; j < numInput; j++) {
			this->hiddenLayer.correctNeuronWeight(i, j, input, hiddenErrors[i], d, funcDerivative);
		}
	}

	// корректируем веса выходного слоя
	for (int i = 0; i < numOutput; i++) {
		for (int j = 0; j < numHidden; j++) {
			this->outputLayer.correctNeuronWeight(i, j, this->hiddenOutput, outErrors[i], d, funcDerivative);
		}
	}
}
}
