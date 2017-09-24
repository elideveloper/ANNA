#pragma once

#include "activation_functions.h"


namespace ANNA {

    class Neuron {
        double* weights;
        int numInput;
    public:

        Neuron();
        Neuron(int numInput);
        double computeInputSum(double* input);
        double computeOutput(double* input, ActivationFunc activFunc);
        int getNumInput() const;
        double* getWeights() const;
        double getWeight(int weightIndex) const;
        void setWeight(int weightNo, double newWeight);
    };
}
