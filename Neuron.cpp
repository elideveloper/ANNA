#include "Neuron.h"

#include <iostream>
#include <ctime>


namespace ANNA {

    Neuron::Neuron() : weights(nullptr)
    {
    }

    Neuron::Neuron(int numInput) : numInput(numInput)
    {
		std::srand(std::time(0));
		// how to protect from numInput = 0
        this->weights = new double[numInput];
        for (int i = 0; i < numInput; i++) {
			this->weights[i] = (rand() % 101 - 50) / 100.0;		// randow values
        }
    }

    double Neuron::computeOutput(double* input, ActivationFunc activFunc)
    {
		this->output = activFunc(this->computeInputSum(input));
        return this->output;
    }

    int Neuron::getNumInput() const
    {
        return this->numInput;
    }

    double Neuron::getOutput() const
	{
		return this->output;
	}

    double* Neuron::getWeights() const
    {
        return this->weights;
    }

    double Neuron::getWeight(int weightIndex) const
    {
        return this->weights[weightIndex];
    }

    void Neuron::setWeight(int weightNo, double newWeight)
    {
        this->weights[weightNo] = newWeight;
    }

    double Neuron::computeInputSum(double* input)
    {
        double sum = 0.0;
        for (int i = 0; i < this->numInput; i++) {
            sum += weights[i] * input[i];
        }
        return sum;
    }
}
