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
	std::srand(std::time(0));
	double* newArr = new double[numElem];
	for (int i = 0; i < numElem; i++) {
		newArr[i] = arr[i] + (rand() % 21 - 10) / 100.0;
	}
	return newArr;
}


/*
 * #Add GA learning
 * #Automatic select of derivative for activation function
 * #High level setting of learning method
 * #Dump and import neurons weights
 * #M.b. return structure with error and number of iterations info.
 * Different training methods are implemented in private functions of ANN, then just call them in train()
 * M.b. stop condition add in training loop
 * #make an object of parameters for GA and use it, instead of magic numbers =)
 * */


int main() {
	const int numInp = 16;
    const int numHiddenNeur = 5;
    const int numOutput = numInp;
    double inputData[numInp] = { 0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5 };
	//double rightOut[numOutput] = { 1 };

    ANNA::ANN myAnn(numInp, numHiddenNeur, numOutput, ANNA::GA, ANNA::TANH_FUNCTION, new ANNA::GAParams(10, 1, 3, 2));
	double* output = myAnn.computeOutput(inputData);
	std::cout << "Initial input:\n";
	printArr(inputData, numInp);
	std::cout << "Initial output:\n";
	printArr(output, numOutput);

    // prepare training dataset
    int trainingSetSize = 1000;
    double** trainInp = new double*[trainingSetSize];
    double** trainOut = new double*[trainingSetSize];
    for (int i = 0; i < trainingSetSize; i++) {
        trainInp[i] = randomlyDeviatedArray(inputData, numInp);
        trainOut[i] = new double[numOutput];
        for (int j = 0; j < numOutput; j++) {
            trainOut[i][j] = 1.0;
        }
    }

	std::cout << "Initial error avg: " << myAnn.getAvgError(trainOut[0]) << std::endl;

	unsigned int repetitionFactor = 1;
	const double learningSpeed = 0.01;
	const double acceptableError = 0.01;

	clock_t tStart = clock();

    // train
    ANNA::TrainingResult trainingOutput = myAnn.train(trainingSetSize, trainInp, trainOut, learningSpeed, acceptableError, trainingSetSize * repetitionFactor);
	std::cout << "\nNumber of iterations: " << trainingOutput.numIterations << std::endl;
	std::cout << "Error avg: " << trainingOutput.avgError << std::endl;

    // test
	double* inpRand = randomlyDeviatedArray(inputData, numInp);
	std::cout << "Test input:\n";
	printArr(inpRand, numInp);
	double* outputRand = myAnn.computeOutput(inpRand);
	std::cout << "Test output:\n";
	printArr(outputRand, numOutput);
    
	/*
	myAnn.exportNeuronsWeights();               // export neurons' weights of myAnn

    ANNA::ANN myAnn2(numInp, numHiddenNeur, numOutput, ANNA::BP, ANNA::TANH_FUNCTION);  // create a new ANN of the same structure
    myAnn2.importNeuronsWeights();              // import neurons' weights of myAnn2
    double* outputRand2 = myAnn.computeOutput(inpRand);
    std::cout << "Output:\n";
    printArr(outputRand2, numOutput);
	*/

	printf("Time taken: %.4fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	system("pause");
	return 0;
}
