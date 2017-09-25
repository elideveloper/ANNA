#include "Layer.h"

#include <iostream>


namespace ANNA {

    Layer::Layer(int numNeurons, int numInput)
    {
        this->numNeurons = numNeurons;
        this->neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            this->neurons[i] = Neuron(numInput);
        }
    }

    double* Layer::computeOutput(double* input, ActivationFunc activFunc)
    {
        double* neuronsOutput = new double[this->numNeurons];
        for (int i = 0; i < this->numNeurons; i++) {
            neuronsOutput[i] = this->neurons[i].computeOutput(input, activFunc);
        }
        return neuronsOutput;
    }

    double* Layer::getWeightsForNeuron(int neuronIndex) const
    {
        double* weights = new double[this->numNeurons];
        for (int i = 0; i < this->numNeurons; i++) {
            weights[i] = this->neurons[i].getWeight(neuronIndex);
        }
        return weights;
    }

    int Layer::getNumNeurons() const
    {
        return this->numNeurons;
    }

    int Layer::getNumInputs() const
    {
        return neurons[this->numNeurons - 1].getNumInput();
    }

    void Layer::correctNeuronWeight(int neuronNo, int weightNo, double* input, double error, double d, ActivationFunc derivative)
    {
        double oldWeight = this->neurons[neuronNo].getWeight(weightNo);
        double corrWeight = oldWeight + d * error * derivative(this->neurons[neuronNo].computeInputSum(input)) * input[weightNo];
        this->neurons[neuronNo].setWeight(weightNo, corrWeight);
    }

    void Layer::exportWeights(std::ofstream& file) const
    {
        for (int i = 0; i < this->numNeurons; i++) {
            double* neuronWeights = this->neurons[i].getWeights();
            for (int j = 0; j < this->getNumInputs(); j++) {
                file << neuronWeights[j] << " ";
            }
        }
    }

    void Layer::importWeights(std::ifstream& file) const
    {
        double w = 0.0;
        for (int i = 0; i < this->numNeurons; i++) {
            for (int j = 0; j < this->getNumInputs(); j++) {
                file >> w;
                this->neurons[i].setWeight(j, w);
            }
        }
    }
}
