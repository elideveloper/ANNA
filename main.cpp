#include <ctime>

#include "ANN.h"
#include "activation_functions.h"


void printArr(double* arr, int numElem) {
    for (int i = 0; i < numElem; i++) {
        printf("%.2f ", arr[i]);
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
    const double acceptableError = 0.001;
    const double learningSpeed = 0.2;
    unsigned int repetitionFactor = 10;
    unsigned int maxGenerations = 10000;
    unsigned int generationSize = 40;
    unsigned int numLeaveBest = 4;
    unsigned int numRandomInds = 10;
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
            trainOut[i][j] = 1.0;
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
            pretestOut[i][j] = 1.0;
        }
    }

    // init ANN
    ANNA::ANN myAnn(numInp, numHiddenNeur, numOutput, ANNA::TANH_FUNCTION, ANNA::GA, gaParams);

    // check initial error
    double avgErr = 0.0;
    for (int i = 0; i < pretestingSetSize; i++) {
        myAnn.computeOutput(pretestInp[i]);
        avgErr += myAnn.getAvgError(pretestOut[i]);
    }
    printf("Initial error avg: %.4f\n", avgErr / pretestingSetSize);

    clock_t tStart = clock();

    // train
    ANNA::TrainingResult trainingOutput = myAnn.train(trainingSetSize, trainInp, trainOut, pretestingSetSize, pretestInp, pretestOut, acceptableError);
    printf("\nNumber of iterations: %i\n", trainingOutput.numIterations);
    printf("Error avg: %.4f\n", trainingOutput.avgError);

    printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    // test
    double* inpRand = randomlyDeviatedArray(exampleInputData, numInp);
    printf("Test input:\n");
    printArr(inpRand, numInp);
    double* outputRand = myAnn.computeOutput(inpRand);
    printf("Test output:\n");
    printArr(outputRand, numOutput);

    myAnn.exportNeuronsWeights();

    //system("pause");
    return 0;
}
