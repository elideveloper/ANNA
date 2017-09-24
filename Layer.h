#pragma once

#include <iostream>

#include "Neuron.h"


namespace ANNA {

    class Layer {
        int numNeurons;
        Neuron* neurons;
    public:
        Layer(int numInput, int numNeurons);
        double* computeOutput(double* input, ActivationFunc activFunc);
        double* getWeightsForNeuron(int neuronIndex) const;
        int getNumNeurons() const;
        int getNumInputs() const;
        void correctNeuronWeight(int neuronNo, int weightNo, double* input, double error, double d, ActivationFunc derivative);
    };
}
