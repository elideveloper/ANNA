#include <ctime>

#include "ANN.h"
#include "activation_functions.h"


void printMatrix(double* arr, int numElem, int numInARow = 999999999) {
    for (int i = 0; i < numElem; i++) {
		if (i % numInARow == 0) printf("\n");
        printf("%.2f	", arr[i]);
    }
    printf("\n");
}

double* randomlyDeviatedArray(double* arr, int numElem) {
    double* newArr = new double[numElem];
    for (int i = 0; i < numElem; i++) {
        newArr[i] = arr[i] + (rand() % 21 - 10) / 100.0;
    }
    return newArr;
}

void testNoiseReduction() {
	const int numInp = 16;
	const int numHiddenNeur = 5;
	const int numOutput = numInp;
	const double acceptableError = 0.001;
	const double learningSpeed = 0.1;
	unsigned int repetitionFactor = 100;
	unsigned int maxGenerations = 10000;
	unsigned int generationSize = 10;
	unsigned int numLeaveBest = 2;
	unsigned int numRandomInds = 2;
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
		for (int j = 0; j < numOutput; j++) {
			trainOut[i][j] = 0.5;
		}
	}

	// pretesting dataset
	int pretestingSetSize = 10;
	double** pretestInp = new double*[pretestingSetSize];
	double** pretestOut = new double*[pretestingSetSize];
	for (int i = 0; i < pretestingSetSize; i++) {
		pretestInp[i] = randomlyDeviatedArray(exampleInputData, numInp);
		pretestOut[i] = new double[numOutput];
		for (int j = 0; j < numOutput; j++) {
			pretestOut[i][j] = 0.5;
		}
	}

	// init ANN
	ANNA::ANN myAnn(numInp, numHiddenNeur, numOutput, ANNA::TANH_FUNCTION, ANNA::BP, bpParams);

	// check initial error
	double avgErr = 0.0;
	for (int i = 0; i < pretestingSetSize; i++) {
		myAnn.computeOutput(pretestInp[i]);
		avgErr += myAnn.getAvgError(pretestOut[i]);
	}
	printf("Initial error avg: %.4f\n", avgErr / pretestingSetSize);
	
	// test before training
	double* inpRand = randomlyDeviatedArray(exampleInputData, numInp);
	printf("Test input:");
	printMatrix(inpRand, numInp, 4);
	double* outputRand = myAnn.computeOutput(inpRand);
	printf("Test output:");
	printMatrix(outputRand, numOutput, 4);

	clock_t tStart = clock();

	// train
	ANNA::TrainingResult trainingOutput = myAnn.train(trainingSetSize, trainInp, trainOut, pretestingSetSize, pretestInp, pretestOut, acceptableError);
	printf("\nNumber of iterations: %i\n", trainingOutput.numIterations);
	printf("Error avg: %.4f\n", trainingOutput.avgError);

	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	// test
	inpRand = randomlyDeviatedArray(exampleInputData, numInp);
	printf("Test input:");
	printMatrix(inpRand, numInp, 4);
	outputRand = myAnn.computeOutput(inpRand);
	printf("Test output:");
	printMatrix(outputRand, numOutput, 4);

	//myAnn.exportNeuronsWeights();
}

void testSquareVsX() {
	const int numInp = 25;
	const int numHiddenNeur = 5;
	const int numOutput = 2;
	const double acceptableError = 0.001;
	const double learningSpeed = 0.1;
	unsigned int repetitionFactor = 100;
	unsigned int maxGenerations = 1000;
	unsigned int generationSize = 40;
	unsigned int numLeaveBest = 4;
	unsigned int numRandomInds = 10;
	unsigned int mutationProbability = 1;
	ANNA::BPParams* bpParams = new ANNA::BPParams(learningSpeed, repetitionFactor);
	ANNA::GAParams* gaParams = new ANNA::GAParams(generationSize, numLeaveBest, numRandomInds, mutationProbability, maxGenerations);

	double inputDataX[numInp] =	{	1.0, 0.0, 0.0, 0.0, 1.0,
									0.0, 1.0, 0.0, 1.0, 0.0,
									0.0, 0.0, 1.0, 0.0, 0.0,
									0.0, 1.0, 0.0, 1.0, 0.0,
									1.0, 0.0, 0.0, 0.0, 1.0 };
	double inputDataSquare[numInp] = {	1.0, 1.0, 1.0, 1.0, 1.0,
										1.0, 0.0, 0.0, 0.0, 1.0,
										1.0, 0.0, 0.0, 0.0, 1.0,
										1.0, 0.0, 0.0, 0.0, 1.0,
										1.0, 1.0, 1.0, 1.0, 1.0 };

	// prepare training dataset
	int trainingSetSize = 1000;
	double** trainInp = new double*[trainingSetSize];
	double** trainOut = new double*[trainingSetSize];
	for (int i = 0; i < trainingSetSize; i++) {
		trainOut[i] = new double[numOutput];
		if (i < trainingSetSize / 2) {
			// squares noisy
			trainInp[i] = randomlyDeviatedArray(inputDataSquare, numInp);
			trainOut[i][0] = 1.0;
			trainOut[i][1] = 0.0;
		} 
		else {
			// Xs noisy
			trainInp[i] = randomlyDeviatedArray(inputDataX, numInp);
			trainOut[i][0] = 0.0;
			trainOut[i][1] = 1.0;
		}
	}

	// pretesting dataset
	int pretestingSetSize = 10;
	double** pretestInp = new double*[pretestingSetSize];
	double** pretestOut = new double*[pretestingSetSize];
	for (int i = 0; i < pretestingSetSize; i++) {
		pretestOut[i] = new double[numOutput];
		if (i < pretestingSetSize / 2) {
			// squares noisy
			pretestInp[i] = randomlyDeviatedArray(inputDataSquare, numInp);
			pretestOut[i][0] = 1.0;
			pretestOut[i][1] = 0.0;
		}
		else {
			// Xs noisy
			pretestInp[i] = randomlyDeviatedArray(inputDataX, numInp);
			pretestOut[i][0] = 0.0;
			pretestOut[i][1] = 1.0;
		}
	}

	// init ANN
	ANNA::ANN myAnn(numInp, numHiddenNeur, numOutput, ANNA::TANH_FUNCTION, ANNA::BP, bpParams);

	// check initial error
	double avgErr = 0.0;
	for (int i = 0; i < pretestingSetSize; i++) {
		myAnn.computeOutput(pretestInp[i]);
		avgErr += myAnn.getAvgError(pretestOut[i]);
	}
	printf("Initial error avg: %.4f\n", avgErr / pretestingSetSize);
	
	// test before training
	double* inpRand = randomlyDeviatedArray(inputDataSquare, numInp);
	printf("Test input for Square:");
	printMatrix(inpRand, numInp, 5);
	double* outputRand = myAnn.computeOutput(inpRand);
	printf("Test output for Square:");
	printMatrix(outputRand, numOutput, 5);
	inpRand = randomlyDeviatedArray(inputDataX, numInp);
	printf("Test input for X:");
	printMatrix(inpRand, numInp, 5);
	outputRand = myAnn.computeOutput(inpRand);
	printf("Test output for X:");
	printMatrix(outputRand, numOutput, 5);

	clock_t tStart = clock();

	// train
	ANNA::TrainingResult trainingOutput = myAnn.train(trainingSetSize, trainInp, trainOut, pretestingSetSize, pretestInp, pretestOut, acceptableError);
	printf("\nNumber of iterations: %i\n", trainingOutput.numIterations);
	printf("Error avg: %.4f\n", trainingOutput.avgError);

	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	// test
	inpRand = randomlyDeviatedArray(inputDataSquare, numInp);
	printf("Test input for Square:");
	printMatrix(inpRand, numInp, 5);
	outputRand = myAnn.computeOutput(inpRand);
	printf("Test output for Square:");
	printMatrix(outputRand, numOutput, 5);
	inpRand = randomlyDeviatedArray(inputDataX, numInp);
	printf("Test input for X:");
	printMatrix(inpRand, numInp, 5);
	outputRand = myAnn.computeOutput(inpRand);
	printf("Test output for X:");
	printMatrix(outputRand, numOutput, 5);

	//myAnn.exportNeuronsWeights();
}


/*
 * ~Different training methods are implemented in private functions of ANN, then just call them in train()
 * ~M.b. stop condition add in training loop
 * Add concurrency if possible
 * Exceptions, all checks of input parameters
 * ~Change size types to Unsigned int
 * Check and fix all copy constructors / operator= 
 * */


int main() {
    std::srand(std::time(0));

	//testNoiseReduction();

	//testSquareVsX();

    //system("pause");
    return 0;
}
