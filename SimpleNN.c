#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include "NN.h"

void test1() {
    printf("Test 1:\n");
    // initialize random numbers
    srand((unsigned int)time(NULL));
    
    // initialize NN
    struct NN nn;
    initNN(&nn);
    
    // data sets
    struct DataSet dataset[NDATASETS];
    dataset[0].xx[0] = 0;
    dataset[0].xx[1] = 0;
    dataset[0].xx[2] = 0;
    dataset[0].xx[3] = 0;
    dataset[0].yy[0] = 0;
    dataset[0].yy[1] = 0;
    dataset[0].yy[2] = 0;
    
    dataset[1].xx[0] = 0;
    dataset[1].xx[1] = 1;
    dataset[1].xx[2] = 0;
    dataset[1].xx[3] = 1;
    dataset[1].yy[0] = 0;
    dataset[1].yy[1] = 1;
    dataset[1].yy[2] = 0;

    dataset[2].xx[0] = 1;
    dataset[2].xx[1] = 0;
    dataset[2].xx[2] = 1;
    dataset[2].xx[3] = 0;
    dataset[2].yy[0] = 1;
    dataset[2].yy[1] = 0;
    dataset[2].yy[2] = 0;

    dataset[3].xx[0] = 1;
    dataset[3].xx[1] = 1;
    dataset[3].xx[2] = 1;
    dataset[3].xx[3] = 1;
    dataset[3].yy[0] = 1;
    dataset[3].yy[1] = 1;
    dataset[3].yy[2] = 0;
    
    // train the network
    train1(&nn, dataset, 0.001, 15);

    // test the network
    double xx[NINPUTS];
    xx[0] = 0;
    xx[1] = 0;
    xx[2] = 0;
    xx[3] = 0;

    // calculate output
    calculateNN(xx, &nn);

    printf("o1 = %f\n",nn.olayer[0].output);
    printf("o1 = %f\n",nn.olayer[1].output);
    printf("o2 = %f\n\n",nn.olayer[2].output);
    
    xx[0] = 1;
    xx[1] = 1;
    xx[2] = 1;
    xx[3] = 1;

    // calculate output
    calculateNN(xx, &nn);

    printf("o1 = %f\n",nn.olayer[0].output);
    printf("o1 = %f\n",nn.olayer[1].output);
    printf("o2 = %f\n\n",nn.olayer[2].output);
}

int main() {
    test1();
    
#ifdef WINDOWS
	printf("Press Any Key to Continue\n");
	getchar();
#endif // WINDOWS

	return 0;
}
