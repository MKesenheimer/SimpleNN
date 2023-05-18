#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define NINPUTS 4
#define NOUTPUTS 3
#define NNEURONS 40
#define NDATASETS 4
#define NPARAMETERS (2*NINPUTS + NINPUTS * NNEURONS + NNEURONS + NNEURONS * NOUTPUTS + NOUTPUTS)

// choose parallelization
//#define PARALLEL1
//#define PARALLEL2
//#define PARALLEL3
#define NTHREADS 4

#if defined(PARALLEL1) || defined(PARALLEL2) || defined(PARALLEL3)
#include <omp.h>
#endif


// choose transfer function
#define SIGMOID
//#define RELU
//#define TANH

// whether to use linesearch for optimization or not
//#define LINESEARCH
#define NPOINTS 5
#define THRESHOLDL 1E-3

// use adaptive learningrate
//#define ADAPTIVLEARNING
#define THRESHOLDU 1E-1
#define THRESHOLDD 1E-3
#define RINCREASE 1E-2
#define NADAPTEND 5

// global variables for adaptive learning rates
double save;
int nadapt;

struct ILayer {
    double input;            // mat[NINPUTS]
    double weight;           // mat[NINPUTS]
    double output[NNEURONS]; // mat[NINPUTS][NNEURONS]
    double theta;            // mat[NINPUTS]
};

struct Neuron {
    double input[NINPUTS];   // mat[NNEURONS][NINPUTS]
    double weight[NINPUTS];  // mat[NNEURONS][NINPUTS]
    double output[NOUTPUTS]; // mat[NNEURONS][NOUTPUTS]
    double theta;            // mat[NNEURONS]
};

struct OLayer {
    double input[NNEURONS];  // mat[NOUTPUTS][NNEURONS]
    double weight[NNEURONS]; // mat[NOUTPUTS][NNEURONS]
    double output;           // mat[NOUTPUTS]
    double theta;            // mat[NOUTPUTS]
};

struct NN {
    struct ILayer ilayer[NINPUTS];
    struct Neuron neuron[NNEURONS];
    struct OLayer olayer[NOUTPUTS];
};

struct DataSet {
    double xx[NINPUTS];
    double yy[NOUTPUTS];
};

double getRandomNumber();
void initNN(struct NN *nn);
void calculateNN(const double xx[], struct NN *nn);
double lossFunction(struct NN *nn, const struct DataSet dataset[]);
void train1(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);
void train2(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);

void structToArray(const struct NN *nn, double par[]);
void arrayToStruct(const double par[], struct NN *nn);


void snapshot(struct NN *nn);
void load(struct NN *nn);

#endif