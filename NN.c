#include "NN.h"

double rnd(double a, double b){
    double x = (double)rand() / (double)(RAND_MAX) * (b - a) + a;
    return x;
}

void structToArray(const struct NN *nn, double par[]) {
    int counter = 0;
    for(int i = 0; i < NINPUTS; ++i) {
        par[counter++] = nn->ilayer[i].theta;
        par[counter++] = nn->ilayer[i].weight;
    }

    for(int i = 0; i < NNEURONS; ++i) {
        par[counter++] = nn->neuron[i].theta;
        for(int n = 0; n<NINPUTS; n++) {
            par[counter++] = nn->neuron[i].weight[n];
        }
    }

    for(int i = 0; i < NOUTPUTS; ++i) {
        par[counter++] = nn->olayer[i].theta;
        for(int n = 0; n < NNEURONS; n++) {
            par[counter++] = nn->olayer[i].weight[n];
        }
    }
}

void arrayToStruct(const double par[], struct NN *nn) {
    int counter = 0;
    for(int i = 0; i < NINPUTS; ++i) {
        nn->ilayer[i].theta = par[counter++];
        nn->ilayer[i].weight = par[counter++];
    }

    for(int i = 0; i < NNEURONS; ++i) {
        nn->neuron[i].theta = par[counter++];
        for(int n = 0; n<NINPUTS; n++) {
            nn->neuron[i].weight[n] = par[counter++];
        }
    }

    for(int i = 0; i < NOUTPUTS; ++i) {
        nn->olayer[i].theta = par[counter++];
        for(int n = 0; n < NNEURONS; n++) {
            nn->olayer[i].weight[n] = par[counter++];
        }
    }
}

double transferFunction(double x, double theta) {
    double f;

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
    save = 0.;
    nadapt = 0;

    // initialize input layer
    for(int i = 0; i < NINPUTS; ++i) {
        nn->ilayer[i].theta = 0;
        // the first layer has by default only one input per node
        nn->ilayer[i].weight = 0;
        nn->ilayer[i].input  = 0;
        for(int n = 0; n < NNEURONS; n++) {
            nn->ilayer[i].output[n] = 0;
        }
    }

    // initialize neurons
    for(int i = 0; i < NNEURONS; ++i) {
        nn->neuron[i].theta = 0;
        for(int n = 0; n<NINPUTS; n++) {
            nn->neuron[i].weight[n] = 0;
            nn->neuron[i].input[n]  = 0;
        }
        for(int n = 0; n < NOUTPUTS; n++) {
            nn->neuron[i].output[n] = 0;
        }
    }

    // initialize output layer
    for(int i = 0; i < NOUTPUTS; ++i) {
        nn->olayer[i].theta = 0;
        for(int n = 0; n < NNEURONS; n++) {
            nn->olayer[i].weight[n] = 0;
            nn->olayer[i].input[n]  = 0;
        }
        // the output layer has by default only one output per node
        nn->olayer[i].output = 0;
    }
}

void calculateNN(const double xx[], struct NN *nn) {
    #ifdef PARALLEL1
        #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(xx) shared(nn)
    #endif
    for(int i = 0; i < NINPUTS; ++i) {
        double temp = nn->ilayer[i].weight * xx[i];
        for(int n = 0; n < NNEURONS; n++) {
            nn->ilayer[i].output[n] = transferFunction(temp, nn->ilayer[i].theta);
        }
    }

    #ifdef PARALLEL1
        #pragma omp parallel for num_threads(NTHREADS) default(none) shared(nn)
    #endif
    for(int i = 0; i < NNEURONS; ++i) {
        double temp = 0;
        for(int n = 0; n < NINPUTS; n++) {
            temp += nn->neuron[i].weight[n] * nn->ilayer[n].output[i];
        }
        for(int n = 0; n < NOUTPUTS; n++) {
            nn->neuron[i].output[n] = transferFunction(temp, nn->neuron[i].theta);
        }
    }

    #ifdef PARALLEL1
        #pragma omp parallel for num_threads(NTHREADS) default(none) shared(nn)
    #endif
    for(int i = 0; i < NOUTPUTS; ++i) {
        double temp = 0;
        for(int n = 0; n < NNEURONS; n++) {
            temp += nn->olayer[i].weight[n] * nn->neuron[n].output[i];
        }
        nn->olayer[i].output = transferFunction(temp, nn->olayer[i].theta);
    }
}

double lossFunction(struct NN *nn, const struct DataSet dataset[]) {
    double delta = 0;
    #ifdef PARALLEL2
        #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(dataset, nn) reduction(+:delta)
    #endif
    for(int i = 0; i < NDATASETS; ++i) {
        calculateNN(dataset[i].xx, nn);
        double delta2 = 0;
        for(int j = 0; j < NOUTPUTS; ++j) {
            delta2 += pow(nn->olayer[j].output - dataset[i].yy[j], 2);
        }
        delta += sqrt(delta2);
    }
    return delta/2;
}

double linesearch(const double par[], const double deriv[], const struct DataSet dataset[], const int npoints, const double learningrate) {
    struct NN nn;
    double point[npoints];
    double parl[NPARAMETERS];
    double alpha;
    #ifdef PARALLEL3
        #pragma omp parallel for num_threads(NTHREADS) default(none) private(nn) firstprivate(deriv,learningrate,par,alpha,dataset) shared(parl,point)
    #endif
    for (int n = 0; n < npoints; n++) {
        for(int i = 0; i < NPARAMETERS; ++i) {
            alpha = (double)learningrate * (n + 1) / npoints;
            parl[i] = par[i] - alpha * deriv[i];
        }
        arrayToStruct(parl, &nn);
        point[n] = lossFunction(&nn, dataset);
    }
    // find minimum
    double min = point[0];
    int imin = 0;
    for (int n = 0; n < npoints; n++) {
        if(min > point[n]) {
            min = point[n];
            imin = n;
        }
    }
    alpha = (double)learningrate * (imin + 1) / npoints;
    return alpha;
}

// train the network (gradient descent method)
void train1(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    bool optimized = false;
    double h = 0.005;

    // optimize the cost function
    while (!optimized) {
        double lf = lossFunction(nn, dataset);
        if(lf < accuracy) {
            optimized = true;
        } else {
            struct NN nnhi;

            // first derivatives
            double deriv[NPARAMETERS], tempi;

            double par[NPARAMETERS];
            structToArray(nn, par);

            // calculate the derivatives
            #ifdef PARALLEL3
                #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h,dataset,par,lf) shared(deriv) private(tempi,nnhi)
            #endif
            for(int i = 0; i < NPARAMETERS; ++i) {
                tempi = par[i];
                par[i] = par[i] + h;
                arrayToStruct(par, &nnhi);
                deriv[i] = (lossFunction(&nnhi, dataset) - lf) / h;
                par[i] = tempi;
            }

            double alpha = learningrate;;
            #ifdef LINESEARCH
                // if there is only a small change in the lossfunction during
                // two subsequent iterations, use a linesearch to optimize instead.
                if(fabs(lf - save) < THRESHOLDL) {
                    alpha = linesearch(par, deriv, dataset, NPOINTS, learningrate);
                }
                save = lf;
            #endif

            #ifdef ADAPTIVLEARNING
                // if there is only a small change in the lossfunction during
                // two subsequent iterations, increase the learning rate, else decrease it
                if(fabs(lf - save) < THRESHOLDD) nadapt++;
                if(fabs(lf - save) > THRESHOLDU) nadapt--;
                if (nadapt < -NADAPTEND) nadapt = -NADAPTEND;
                if (nadapt >  NADAPTEND) nadapt =  NADAPTEND;
                alpha = alpha * pow(1 + RINCREASE, nadapt);
                save = lf;
            #endif

            #ifdef PARALLEL3
                #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(deriv,alpha) shared(par)
            #endif
            for(int i = 0; i < NPARAMETERS; ++i) {
                par[i] = par[i] - alpha * deriv[i];
            }

            arrayToStruct(par, nn);
        }
    }
}

// train the network (quasi newtonian method)
void train2(struct NN *nn, const struct DataSet dataset[], const double accuracy, const double learningrate) {
    double h = 0.005;

    // optimize the cost function
    while (!(lossFunction(nn, dataset) < accuracy)) {
        struct NN nnhi, nnhj, nnhij;

        // first derivatives
        double deriv[NPARAMETERS];
        // second derivatives
        double hess[NPARAMETERS][NPARAMETERS];

        double par[NPARAMETERS], tempi, tempj;
        structToArray(nn, par);

        // calculate the derivatives
        #ifdef PARALLEL3
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(h, nn, dataset) shared(deriv, hess, par) private(tempi, tempj, nnhi, nnhj, nnhij)
        #endif
        for(int i = 0; i < NPARAMETERS; ++i) {
            tempi = par[i];
            par[i] = par[i] + h;
            arrayToStruct(par, &nnhi);
            deriv[i] = (lossFunction(&nnhi, dataset) - lossFunction(nn, dataset)) / h;
            par[i] = tempi;

            for(int j=i; j<NPARAMETERS; ++j) {
                tempj = par[j];
                par[j] = par[j] + h;
                arrayToStruct(par, &nnhj);
                par[j] = tempj;

                tempi = par[i];
                par[i] = par[i] + h;
                tempj = par[j];
                par[j] = par[j] + h;
                arrayToStruct(par, &nnhij);
                par[i] = tempi;
                par[j] = tempj;

                hess[i][j] = (lossFunction(&nnhij, dataset) - lossFunction(&nnhi, dataset) - lossFunction(&nnhj, dataset) + lossFunction(nn, dataset)) / pow(h, 2.0);
                hess[j][i] = hess[i][j];
            }
        }

        #ifdef PARALLEL3
            #pragma omp parallel for num_threads(NTHREADS) default(none) firstprivate(deriv,learningrate) shared(par)
        #endif
        for(int i = 0; i < NPARAMETERS; ++i) {
            par[i] = par[i] - learningrate * deriv[i];
        }

        arrayToStruct(par, nn);
    }
}

void snapshot(struct NN *nn) {
    FILE *f1 = fopen("nn.dat", "w");
    if(f1 == NULL){
        printf("Error opening file!\n");
        exit(1);
    }

    for(int i = 0; i < NINPUTS; ++i) {
        fprintf(f1, "w%i %f\n", i, nn->ilayer[i].weight);
    }
    for(int i = 0; i < NINPUTS; ++i) {
        fprintf(f1, "t%i %f\n", i, nn->ilayer[i].theta);
    }
    for (int n = 0; n < NNEURONS; n++) {
        for(int i = 0; i < NINPUTS; ++i) {
            fprintf(f1, "w%i%i %f\n", n, i, nn->neuron[n].weight[i]);
        }
    }
    for (int n = 0; n < NNEURONS; n++) {
        fprintf(f1, "t%i %f\n",n, nn->neuron[n].theta);
    }
    for (int n = 0; n < NOUTPUTS; n++) {
        for(int i = 0; i < NNEURONS; ++i) {
            fprintf(f1, "w%i%i %f\n",n,i,nn->olayer[n].weight[i]);
        }
    }
    for (int n = 0; n < NOUTPUTS; n++) {
        fprintf(f1, "t%i %f\n",n,nn->olayer[n].theta);
    }
    fclose(f1);
}

void load(struct NN *nn) {
    FILE *f1 = fopen("nn.dat", "r");
    if(f1 == NULL){
        printf("File does not exist.\n");
        return;
    }

    char buff[255];
    char key1[255], key2[255];
    double val;

    for(int i = 0; i < NINPUTS; ++i) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "w", i);
        if(strcmp(key1, key2)==0) {
            nn->ilayer[i].weight = val;
        }
    }
    for(int i = 0; i < NINPUTS; ++i) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "t", i);
        if(strcmp(key1, key2)==0) {
            nn->ilayer[i].theta = val;
        }
    }
    for (int n = 0; n < NNEURONS; n++) {
        for(int i = 0; i < NINPUTS; ++i) {
            fscanf(f1, "%s", buff);
            strncpy(key1, buff, 255);
            fscanf(f1, "%s", buff);
            sscanf(buff, "%lf", &val);
            sprintf(key2, "%s%d%d", "w", n, i);
            if(strcmp(key1, key2)==0) {
                nn->neuron[n].weight[i] = val;
            }
        }
    }
    for (int n = 0; n < NNEURONS; n++) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "t", n);
        if(strcmp(key1, key2)==0) {
            nn->neuron[n].theta = val;
        }
    }
    for (int n = 0; n < NOUTPUTS; n++) {
        for(int i = 0; i < NNEURONS; ++i) {
            fscanf(f1, "%s", buff);
            strncpy(key1, buff, 255);
            fscanf(f1, "%s", buff);
            sscanf(buff, "%lf", &val);
            sprintf(key2, "%s%d%d", "w", n, i);
            if(strcmp(key1, key2)==0) {
                nn->olayer[n].weight[i] = val;
            }
        }
    }
    for (int n = 0; n < NOUTPUTS; n++) {
        fscanf(f1, "%s", buff);
        strncpy(key1, buff, 255);
        fscanf(f1, "%s", buff);
        sscanf(buff, "%lf", &val);
        sprintf(key2, "%s%d", "t", n);
        if(strcmp(key1, key2)==0) {
            nn->olayer[n].theta = val;
        }
    }

    fclose(f1);
}
