#include "NN.h"

double getRandomNumber(){
    double a = 1.0;
    double x = (double)rand()/(double)(RAND_MAX/a);
    return x;
}

double transferFunction(double x, double theta) {
    double f;
    
    #define SIGMOID
    //#define RELU
    //#define TANH
    
    #ifdef SIGMOID
        f = 1/(1+exp(theta-x));
    #endif

    #ifdef RELU
        if(x+theta >= 0) {
            f = x+theta;
        } else {
            f = 0;
        }
    #endif

    #ifdef TANH
        f= tanh(theta+x);
    #endif

    return f;
}

void initNN(struct NN *nn) {
    // initialize input layer
    for(int i=0; i<NINPUTS; i++) {
        nn->ilayer[i].theta = getRandomNumber();
        // the first layer has by default only one input per node
        nn->ilayer[i].weight = getRandomNumber();
        nn->ilayer[i].input  = 0;
        for(int n=0; n<NNEURONS; n++) {
            nn->ilayer[i].output[n] = 0;
        }
    }

    // initialize neurons
    for(int i=0; i<NNEURONS; i++) {
        nn->neuron[i].theta = getRandomNumber();
        for(int n=0; n<NINPUTS; n++) {
            nn->neuron[i].weight[n] = getRandomNumber();
            nn->neuron[i].input[n]  = 0;
        }
        for(int n=0; n<NOUTPUTS; n++) {
            nn->neuron[i].output[n] = 0;
        }
    }
    
    // initialize output layer
    for(int i=0; i<NOUTPUTS; i++) {
        nn->olayer[i].theta = getRandomNumber();
        for(int n=0; n<NNEURONS; n++) {
            nn->olayer[i].weight[n] = getRandomNumber();
            nn->olayer[i].input[n]  = 0;
        }
        // the output layer has by default only one output per node
        nn->olayer[i].output = 0;
    }
}

void calculateNN(const double xx[], struct NN *nn) {
    for(int i=0; i<NINPUTS; i++) {
        double temp = nn->ilayer[i].weight*xx[i];
        
        for(int n=0; n<NNEURONS; n++) {
            //nn->ilayer[i].output[n] = 1/(1+exp(nn->ilayer[i].theta-temp));
            nn->ilayer[i].output[n] = transferFunction(temp, nn->ilayer[i].theta);
        }
    }
    
    for(int i=0; i<NNEURONS; i++) {
        double temp = 0;
        
        for(int n=0; n<NINPUTS; n++) {
            temp += nn->neuron[i].weight[n]*nn->ilayer[n].output[i];
        }
        
        for(int n=0; n<NOUTPUTS; n++) {
            //nn->neuron[i].output[n] = 1/(1+exp(nn->neuron[i].theta-temp));
            nn->neuron[i].output[n] = transferFunction(temp, nn->neuron[i].theta);
        }
    }
    
    for(int i=0; i<NOUTPUTS; i++) {
        double temp = 0;
        
        for(int n=0; n<NNEURONS; n++) {
            temp += nn->olayer[i].weight[n]*nn->neuron[n].output[i];
        }
        //nn->olayer[i].output = 1/(1+exp(nn->olayer[i].theta-temp));
        nn->olayer[i].output = transferFunction(temp, nn->olayer[i].theta);
    }
}

double costFunction(struct NN *nn, const struct DataSet dataset[]) {
    double delta = 0;
    
    for(int i=0; i<NDATASETS; i++) {
        calculateNN(dataset[i].xx, nn);
        
        double delta2 = 0;
        for(int j=0; j<NOUTPUTS; j++) {
            delta2 += pow(nn->olayer[j].output - dataset[i].yy[j],2);
        }
        delta += sqrt(delta2);
    }

    return delta/2;
}

void train(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    bool optimized = false;
    double h = 0.0001;
    
    while (!optimized) {
        if(costFunction(nn,dataset)<accuracy) {
            optimized = true;
        } else {
            // optimize the cost function
            struct NN nnh;
            double derivThetaIn[NINPUTS];
            double derivWeightIn[NINPUTS];
            double derivThetaNe[NNEURONS];
            double derivWeightNe[NNEURONS][NINPUTS];
            double derivThetaOut[NOUTPUTS];
            double derivWeightOut[NOUTPUTS][NNEURONS];
            
            // calculate the derivatives
            for(int i=0; i<NINPUTS; i++) {
                nnh = *nn;
                nnh.ilayer[i].theta = nn->ilayer[i].theta + h;
                derivThetaIn[i] = (costFunction(&nnh,dataset) - costFunction(nn,dataset))/h;
                nnh = *nn;
                nnh.ilayer[i].weight = nn->ilayer[i].weight + h;
                derivWeightIn[i] = (costFunction(&nnh,dataset) - costFunction(nn,dataset))/h;
            }
            for(int i=0; i<NNEURONS; i++) {
                nnh = *nn;
                nnh.neuron[i].theta = nn->neuron[i].theta + h;
                derivThetaNe[i] = (costFunction(&nnh,dataset) - costFunction(nn,dataset))/h;
                for(int n=0; n<NINPUTS; n++) {
                    nnh = *nn;
                    nnh.neuron[i].weight[n] = nn->neuron[i].weight[n] + h;
                    derivWeightNe[i][n] = (costFunction(&nnh,dataset) - costFunction(nn,dataset))/h;
                }
            }
            for(int i=0; i<NOUTPUTS; i++) {
                nnh = *nn;
                nnh.olayer[i].theta = nn->olayer[i].theta + h;
                derivThetaOut[i] = (costFunction(&nnh,dataset) - costFunction(nn,dataset))/h;
                for(int n=0; n<NNEURONS; n++) {
                    nnh = *nn;
                    nnh.olayer[i].weight[n] = nn->olayer[i].weight[n] + h;
                    derivWeightOut[i][n] = (costFunction(&nnh,dataset) - costFunction(nn,dataset))/h;
                }
            }
            
            // calculate the new parameters
            for(int i=0; i<NINPUTS; i++) {
                nn->ilayer[i].theta = nn->ilayer[i].theta - learningrate*derivThetaIn[i];
                nn->ilayer[i].weight = nn->ilayer[i].weight - learningrate*derivWeightIn[i];
            }
            for(int i=0; i<NNEURONS; i++) {
                nn->neuron[i].theta = nn->neuron[i].theta - learningrate*derivThetaNe[i];
                for(int n=0; n<NINPUTS; n++) {
                    nn->neuron[i].weight[n] = nn->neuron[i].weight[n] - learningrate*derivWeightNe[i][n];
                }
            }
            for(int i=0; i<NOUTPUTS; i++) {
                nn->olayer[i].theta = nn->olayer[i].theta - learningrate*derivThetaOut[i];
                for(int n=0; n<NNEURONS; n++) {
                    nn->olayer[i].weight[n] = nn->olayer[i].weight[n] - learningrate*derivWeightOut[i][n];
                }
            }
        }
    }
}