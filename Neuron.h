#pragma once
#ifndef NEURON_H
#define NEURON_H

#include "activation_functions.h"


namespace ANNA {

    class Neuron {
        double* weights;
        int numInput;
		double output;		// neuron's output for the last input
    public:
        Neuron();
        Neuron(int numInput);
		Neuron(const Neuron& neuron);
		Neuron& operator=(const Neuron& neuron);
		~Neuron();
		void init(int numInput);
        double computeInputSum(double* input);
        double computeOutput(double* input, ActivationFunc activFunc);
        int getNumInput() const;
        double getOutput() const;
        double* getWeights() const;
        double getWeight(int weightIndex) const;
        void setWeight(int weightNo, double newWeight);
		void importWeights(const Neuron& neuron);
    };
}

#endif
