#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

#include <fstream>


namespace ANNA {

    class Layer {
        int numNeurons;
        Neuron* neurons;
        void correctNeuronWeights(int neuronNo, int numWeights, double* input, double error, double d, ActivationFunc derivative);
    public:
        Layer();
        Layer(int numInput, int numNeurons);
		Layer(const Layer& layer);
		Layer& operator=(const Layer& layer);
		~Layer();
		void init(int numInput, int numNeurons);
        double* computeOutput(double* input, ActivationFunc activFunc);
        int getNumNeurons() const;
        int getNumInputs() const;
		void correctWeights(double* input, double* errors, double d, ActivationFunc derivative);
		double* computeLayerErrors(double* nextLayerErrors, const Layer& nextLayer);
        Neuron getNeuron(int neuronNo) const;
        void exportWeights(std::ofstream& file) const;
        void importWeights(std::ifstream& file) const;
        void importWeights(Neuron* neurons) const;
    };
}

#endif
