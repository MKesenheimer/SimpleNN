#pragma once

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#define NINPUTS 4
#define NOUTPUTS 2
#define NNEURONS 4
#define NDATASETS 2
#define NPARAMETERS (2*NINPUTS + NINPUTS * NNEURONS + NNEURONS + NNEURONS * NOUTPUTS + NOUTPUTS)

struct ILayer {
    double input;
    double weight;
    double output[NNEURONS];
    double theta;
};

struct Neuron {
    double input[NINPUTS];
    double weight[NINPUTS];
    double output[NOUTPUTS];
    double theta;
};

struct OLayer {
    double input[NNEURONS];
    double weight[NNEURONS];
    double output;
    double theta;
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
double costFunction(struct NN *nn, const struct DataSet dataset[]);
void train(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate);

#endif