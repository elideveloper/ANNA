#include <iostream>
#include <ctime>

#include "ANN.h"
#include "activation_functions.h"

void printArr(double* arr, int numElem) {
	for (int i = 0; i < numElem; i++) {
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}

double* randomlyDeviatedArray(double* arr, int numElem) {
	double* newArr = new double[numElem];
	for (int i = 0; i < numElem; i++) {
		newArr[i] = arr[i] + (rand() % 21 - 10) / 100.0;
	}
	return newArr;
}


/*
 * Add GA learning
 * #Automatic select of derivative for activation function
 * #High level setting of learning method
 * Dump and import neurons weights
 * M.b. return structure with error and number of iterations info.
 * */

int main() {
	std::srand(std::time(0));
    const double learningSpeed = 0.1;
	const int numInp = 9;
	const int numNeur = 3;
	const int numOutput = numInp;
    double inputData[numInp] = { 0.5, 0.5, 0.5,
							0.5, 0.5, 0.5, 
							0.5, 0.5, 0.5 };
	//double rightOut[numOutput] = inputData;

    ANNA::ANN myAnn(numInp, numNeur, numOutput, ANNA::BP, ANNA::LOGISTIC_FUNCTION);
	double* output = myAnn.computeOutput(inputData);
	std::cout << "Initial:\n";
	printArr(output, numOutput);

    // prepare training dataset
    int trainingSetSize = 100;
    double** trainInp = new double*[trainingSetSize];
    double** trainOut = new double*[trainingSetSize];
    for (int i = 0; i < trainingSetSize; i++) {
        trainInp[i] = randomlyDeviatedArray(inputData, numInp);
        trainOut[i] = new double[numInp];
        for (int j = 0; j < numInp; j++) {
            trainOut[i][j] = inputData[j];
        }
    }

    // train
    double err = myAnn.train(trainingSetSize, trainInp, trainOut, learningSpeed, 0.001, 10000);
    std::cout << "Error avg: " << err << std::endl;

    // test
	double* inpRand = randomlyDeviatedArray(inputData, numInp);
	std::cout << "Random input:\n";
	printArr(inpRand, numOutput);
	double* outputRand = myAnn.computeOutput(inpRand);
	std::cout << "Random output:\n";
	printArr(outputRand, numOutput);

	system("pause");

	return 0;
}
