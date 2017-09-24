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


int main() {
	std::srand(std::time(0));
	const int numInp = 9;
	const int numNeur = 3;
	const int numOutput = numInp;
	double inputData[9] = { 0.5, 0.5, 0.5, 
							0.5, 0.5, 0.5, 
							0.5, 0.5, 0.5 };
	//double rightOut[numOutput] = inputData;

    ANNA::ANN myAnn(numInp, numNeur, numOutput, ANNA::logisticFunction);
	double* output = myAnn.computeOutput(inputData);
	std::cout << "Initial:\n";
	printArr(output, numOutput);

	for (int i = 0; i < 10000; i++) {
        myAnn.backPropagate(randomlyDeviatedArray(inputData, numInp), inputData, 0.2, ANNA::logisticFunctionDerivative);
		output = myAnn.computeOutput(inputData);
	}

	double* inpRand = randomlyDeviatedArray(inputData, numInp);
	std::cout << "Random input:\n";
	printArr(inpRand, numOutput);
	double* outputRand = myAnn.computeOutput(inpRand);
	std::cout << "Random output:\n";
	printArr(outputRand, numOutput);

	system("pause");

	return 0;
}
