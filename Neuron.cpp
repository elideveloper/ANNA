#include "Neuron.h"

#include <iostream>
#include <ctime>


namespace ANNA {

    Neuron::Neuron()
    {
    }

    Neuron::Neuron(int numInput)
    {
		std::srand(std::time(0));
        this->numInput = numInput;
        this->weights = new double[numInput];
        for (int i = 0; i < numInput; i++) {
			this->weights[i] = (rand() % 11) / 100.0;		// randow values [0; 0.1]
        }
    }

    double Neuron::computeOutput(double* input, ActivationFunc activFunc)
    {
        return activFunc(this->computeInputSum(input));
    }

    int Neuron::getNumInput() const
    {
        return this->numInput;
    }

    double * Neuron::getWeights() const
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
