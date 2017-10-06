#include "Layer.h"


namespace ANNA {

    Layer::Layer()
    {
    }

    Layer::Layer(int numNeurons, int numInput) : numNeurons(numNeurons)
    {
		// how to protect from numNeurons = 0
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

    int Layer::getNumNeurons() const
    {
        return this->numNeurons;
    }

    int Layer::getNumInputs() const
    {
        return neurons[this->numNeurons - 1].getNumInput();
    }

	void Layer::correctWeights(double* input, double* errors, double d, ActivationFunc derivative)
	{
		int numWeights = this->getNumInputs();
		for (int i = 0; i < this->numNeurons; i++) {
			this->correctNeuronWeights(i, numWeights, input, errors[i], d, derivative);
		}
	}

    void Layer::correctNeuronWeights(int neuronNo, int numWeights, double* input, double error, double d, ActivationFunc derivative)
    {
        double* oldWeights = this->neurons[neuronNo].getWeights();
		for (int i = 0; i < numWeights; i++) {
			this->neurons[neuronNo].setWeight(i, oldWeights[i] + d * error * derivative(this->neurons[neuronNo].getOutput()) * input[i]);
		}
    }

	double* Layer::computeLayerErrors(double* nextLayerErrors, const Layer& nextLayer)
	{
		double* errors = new double[this->numNeurons]();
		int numNext = nextLayer.getNumNeurons();
		for (int i = 0; i < this->numNeurons; i++) {
			for (int j = 0; j < numNext; j++) {
				errors[i] += nextLayerErrors[j] * nextLayer.neurons[j].getWeight(i);
			}
		}
		return errors;
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

    void Layer::importWeights(double** weights) const
    {
        for (int i = 0; i < this->numNeurons; i++) {
            for (int j = 0; j < this->getNumInputs(); j++) {
                this->neurons[i].setWeight(j, weights[i][j]);
            }
        }
    }
}
