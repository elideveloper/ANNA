#include "Neuron.h"

#include <iostream>


namespace ANNA {

    Neuron::Neuron() : weights(nullptr)
    {
    }

    Neuron::Neuron(int numInput) : numInput(numInput)
    {
		// how to protect from numInput = 0
        this->weights = new double[numInput];
        for (int i = 0; i < numInput; i++) {
            this->weights[i] = (rand() % 1001 - 500) / 1000.0;		// randow values
        }
    }

	Neuron::Neuron(const Neuron & neuron)
	{
		this->numInput = neuron.getNumInput();
		this->weights = new double[this->numInput];
		this->importWeights(neuron);
	}

	Neuron& Neuron::operator=(const Neuron& neuron)
	{
		this->numInput = neuron.getNumInput();
		this->weights = new double[this->numInput];
		this->importWeights(neuron);
		return *this;
	}

	Neuron::~Neuron()
	{
		delete[] this->weights;
	}

	void Neuron::init(int numInput)
	{
		// how to protect from numInput = 0
		this->numInput = numInput;
		this->weights = new double[numInput];
		for (int i = 0; i < numInput; i++) {
            this->weights[i] = (rand() % 1001 - 500) / 1000.0;		// randow values
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

    double Neuron::getWeight(int weightNo) const
    {
        return this->weights[weightNo];
    }

    void Neuron::setWeight(int weightNo, double newWeight)
    {
        this->weights[weightNo] = newWeight;
    }

	void Neuron::importWeights(const Neuron& neuron)
	{
		for (int i = 0; i < this->numInput; i++) {
			this->weights[i] = neuron.getWeight(i);
		}
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
