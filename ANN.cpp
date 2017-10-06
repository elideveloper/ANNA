#include "ANN.h"

#include <limits>
#include <fstream>


namespace ANNA {

    ANN::ANN(int numInput, int numHiddenNeurons, int numOutput, ANNA::LearningMethod learnMethod, ANNA::ActivationFunction activFunc) : hiddenLayer(numHiddenNeurons, numInput), outputLayer(numOutput, numHiddenNeurons)
    {
        switch (learnMethod) {
			case BP: {
				this->learnMethod = BP;
			}
				break;
			case GA: {
				// GA details
				this->learnMethod = GA;
			}
				break;
			default: {
				// error
			}
        }
		switch (activFunc) {
			case LOGISTIC_FUNCTION: {
				this->activFunc = logisticFunction;
				this->activFuncDerivative = logisticDerivReceivingLogisticVal;
			}
				break;
			case TANH_FUNCTION: {
				this->activFunc = tanhFunction;
				this->activFuncDerivative = tanhDerivReceivingTanhVal;
			}
				break;
			default: {
				// error
			}
		}
    }

    double* ANN::computeOutput(double* input)
    {
        this->hiddenOutput = hiddenLayer.computeOutput(input, this->activFunc);
        this->output = outputLayer.computeOutput(this->hiddenOutput, this->activFunc);
        return this->output;
    }

    double* ANN::getOutput() const
    {
        return this->output;
    }

    double ANN::backPropagate(double* input, double* correctOutput, double d)
    {
        double err = 0.0;
        int numOutput = this->outputLayer.getNumNeurons();

		// compute output layer errors
        double* outErrors = new double[numOutput];
        for (int i = 0; i < numOutput; i++) {
            outErrors[i] = correctOutput[i] - this->output[i];
            err += fabs(outErrors[i]);
        }
        err /= numOutput;																						// avg output error

		// compute hidden layer errors
		double* hiddenErrors = this->hiddenLayer.computeLayerErrors(outErrors, this->outputLayer);

        // hidden layer weights correcting
		this->hiddenLayer.correctWeights(input, hiddenErrors, d, this->activFuncDerivative);
		delete[] hiddenErrors;

        // output layer weights correcting
		this->outputLayer.correctWeights(this->hiddenOutput, outErrors, d, this->activFuncDerivative);
		delete[] outErrors;

        return err;
    }

	TrainingResult ANN::train(int trainDatasetSize, double** trainInput, double** trainOutput, double d, double avgError, int maxIterations)
    {
        double avgErr = std::numeric_limits<double>::max();
        int m = 0;

		switch (this->learnMethod) {
			case BP: {
				// while avg error will not be acceptable or max iterations
				while (m < maxIterations && avgErr > avgError) {
					avgErr = 0.0;
					for (int i = 0; i < trainDatasetSize; i++, m++) {
						avgErr += this->backPropagate(trainInput[i], trainOutput[i], d);
						this->computeOutput(trainInput[i]);                                     // refreshs output
					}
					avgErr /= trainDatasetSize;
				}
			}
				break;
			case GA: {

			}
				break;
			default: {

			}
		}

        return TrainingResult(m, avgErr);
    }

    void ANN::exportNeuronsWeights() const
    {
        std::ofstream outFile("hidden_layer.txt");
        this->hiddenLayer.exportWeights(outFile);
        outFile.close();
        outFile.open("output_layer.txt");
        this->outputLayer.exportWeights(outFile);
        outFile.close();
    }

    void ANN::importNeuronsWeights() const
    {
        std::ifstream inFile("hidden_layer.txt");
        this->hiddenLayer.importWeights(inFile);
        inFile.close();
        inFile.open("output_layer.txt");
        this->outputLayer.importWeights(inFile);
        inFile.close();
    }

	TrainingResult::TrainingResult(int numIter, double avgErr)
	{
		this->numIterations = numIter;
		this->avgError = avgErr;
	}
}
