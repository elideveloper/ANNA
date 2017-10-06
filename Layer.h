#pragma once
#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

#include <fstream>


namespace ANNA {

    class Layer {
        int numNeurons;
        Neuron* neurons;
    public:
        Layer();
        Layer(int numInput, int numNeurons);
        double* computeOutput(double* input, ActivationFunc activFunc);
        double* getWeightsForNeuron(int neuronIndex) const;
        int getNumNeurons() const;
        int getNumInputs() const;
        void correctNeuronWeight(int neuronNo, int weightNo, double* input, double error, double d, ActivationFunc derivative);
        void exportWeights(std::ofstream& file) const;
        void importWeights(std::ifstream& file) const;
        void importWeights(double** weights) const;
    };
}

#endif
