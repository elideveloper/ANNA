#include <iostream>
#include <ctime>

#include "ANN.h"
#include "activation_functions.h"


void printArr(double* arr, int numElem) {
	for (int i = 0; i < numElem; i++) {
        //std::cout << arr[i] << " ";
        printf("%.2f ", arr[i]);
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
 * Different training methods are implemented in private functions of ANN, then just call them in train()
 * M.b. stop condition add in training loop
 * Add concurrency if possible
 * Exceptions, all checks of input parameters
 * Change size types to Unsigned int
 * */


int main() {
    std::srand(std::time(0));

	const int numInp = 16;
    const int numHiddenNeur = 5;
    const int numOutput = numInp;
    const double acceptableError = 0.01;
    const double learningSpeed = 0.2;
    unsigned int repetitionFactor = 100;
    unsigned int maxGenerations = 1000;
    unsigned int generationSize = 10;
    unsigned int numLeaveBest = 1;
    unsigned int numRandomInds = 3;
    unsigned int mutationProbability = 1;
    ANNA::BPParams* bpParams = new ANNA::BPParams(learningSpeed, repetitionFactor);
    ANNA::GAParams* gaParams = new ANNA::GAParams(generationSize, numLeaveBest, numRandomInds, mutationProbability, maxGenerations);

    double exampleInputData[numInp] = { 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5 };

    // prepare training dataset
    int trainingSetSize = 1000;
    double** trainInp = new double*[trainingSetSize];
    double** trainOut = new double*[trainingSetSize];
    for (int i = 0; i < trainingSetSize; i++) {
        trainInp[i] = randomlyDeviatedArray(exampleInputData, numInp);
        trainOut[i] = new double[numOutput];
        //for (int j = 0; j < numOutput; j++) {
            trainOut[i] = trainInp[i];
        //}
    }

    // pretesting dataset
    int pretestingSetSize = 10;
    double** pretestInp = new double*[pretestingSetSize];
    double** pretestOut = new double*[pretestingSetSize];
    for (int i = 0; i < pretestingSetSize; i++) {
        pretestInp[i] = randomlyDeviatedArray(exampleInputData, numInp);
        pretestOut[i] = new double[numOutput];
        //for (int j = 0; j < numOutput; j++) {
            pretestOut[i] = pretestInp[i];
        //}
    }

    // init ANN
    ANNA::ANN myAnn(numInp, numHiddenNeur, numOutput, ANNA::TANH_FUNCTION, ANNA::BP, bpParams);
    double* output = myAnn.computeOutput(trainInp[0]);
    std::cout << "Initial input:\n";
    printArr(trainInp[0], numInp);
    std::cout << "Initial output:\n";
    printArr(output, numOutput);
	std::cout << "Initial error avg: " << myAnn.getAvgError(trainOut[0]) << std::endl;
	clock_t tStart = clock();

    // train
    ANNA::TrainingResult trainingOutput = myAnn.train(trainingSetSize, trainInp, trainOut, pretestingSetSize, pretestInp, pretestOut, acceptableError);
	std::cout << "\nNumber of iterations: " << trainingOutput.numIterations << std::endl;
	std::cout << "Error avg: " << trainingOutput.avgError << std::endl;

    printf("Time taken: %.4fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    // test
    double* inpRand = randomlyDeviatedArray(exampleInputData, numInp);
	std::cout << "Test input:\n";
	printArr(inpRand, numInp);
	double* outputRand = myAnn.computeOutput(inpRand);
	std::cout << "Test output:\n";
	printArr(outputRand, numOutput);

    myAnn.exportNeuronsWeights();
    
    //system("pause");
	return 0;
}
