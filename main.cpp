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
 * Add mutation
 * Problem with 1 neuron in GA
 * #Automatic select of derivative for activation function
 * #High level setting of learning method
 * #Dump and import neurons weights
 * #M.b. return structure with error and number of iterations info.
 * Different traning methods are implemented in private functions of ANN, then just call them in train()
 * M.b. take random input from set while train() and check avg error after each iteration
 * M.b. stop condition add in training loop
 * */


int main() {
	const int numInp = 16;
    const int numHiddenNeur = 5;
<<<<<<< HEAD
    const int numOutput = 2;
=======
    const int numOutput = numInp;
>>>>>>> 3949e9ee07c2977e67e342d7d46bdcddc904902e
    double inputData[numInp] = { 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5,
		0.5, 0.5, 0.5, 0.5 };

<<<<<<< HEAD
    ANNA::ANN myAnn(numInp, numHiddenNeur, numOutput, ANNA::GA, ANNA::TANH_FUNCTION);
=======
    ANNA::ANN myAnn(numInp, numHiddenNeur, numOutput, ANNA::BP, ANNA::LOGISTIC_FUNCTION);
>>>>>>> 3949e9ee07c2977e67e342d7d46bdcddc904902e
	double* output = myAnn.computeOutput(inputData);
	std::cout << "Initial input:\n";
	printArr(inputData, numInp);
	std::cout << "Initial output:\n";
	printArr(output, numOutput);

    // prepare training dataset
    int trainingSetSize = 100;
    double** trainInp = new double*[trainingSetSize];
    double** trainOut = new double*[trainingSetSize];
    for (int i = 0; i < trainingSetSize; i++) {
        trainInp[i] = randomlyDeviatedArray(inputData, numInp);
        trainOut[i] = new double[numOutput];
        for (int j = 0; j < numOutput; j++) {
            trainOut[i][j] = 0.5;
        }
    }

	const double learningSpeed = 0.01;
    const double acceptableError = 0.01;

	clock_t tStart = clock();

    // train
    ANNA::TrainingResult trainingOutput = myAnn.train(trainingSetSize, trainInp, trainOut, learningSpeed, acceptableError, trainingSetSize * 100);
    std::cout << "Error avg: " << trainingOutput.avgError << std::endl;
	std::cout << "Number of iterations: " << trainingOutput.numIterations << std::endl;

    // test
	double* inpRand = randomlyDeviatedArray(inputData, numInp);
	std::cout << "Random input:\n";
	printArr(inpRand, numInp);
	double* outputRand = myAnn.computeOutput(inpRand);
	std::cout << "Output:\n";
	printArr(outputRand, numOutput);
    myAnn.exportNeuronsWeights();               // export neurons' weights of myAnn



    /*
    ANNA::ANN myAnn2(numInp, numHiddenNeur, numOutput, ANNA::BP, ANNA::TANH_FUNCTION);  // create a new ANN of the same structure
    myAnn2.importNeuronsWeights();              // import neurons' weights of myAnn2
    double* outputRand2 = myAnn.computeOutput(inpRand);
    std::cout << "Output:\n";
    printArr(outputRand2, numOutput);
	*/
<<<<<<< HEAD
=======

	printf("Time taken: %.4fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	system("pause");
>>>>>>> 3949e9ee07c2977e67e342d7d46bdcddc904902e
	return 0;
}
