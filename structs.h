#ifndef STRUCTURES_FILE
#define STRUCTURES_FILE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define IMAGE_SIZE 3072 /*32x32x3*/
#define LABEL_SIZE 1

/*Necessary Structs*/
typedef struct{
    int height;
    int width;
    int depth;
    float*** data;
} Matrix3D;

typedef struct{
    int height;
    int width;
    float** data;
} Matrix;

typedef struct{
    Matrix* matrices;
    int count;
} Tensor;

typedef struct{
    Matrix3D* matrices;
    int count;
} Tensor3D;

typedef struct{
    unsigned char label;
    unsigned char pixels[IMAGE_SIZE];
} Image;

typedef struct{
    Image* images;
    int count;
} Image_Dataset;

typedef struct{
    char** class_names;
    int count;
} Class_Names;

typedef struct{
    int input_size;
    int output_size; /*Is 10 due to 10 classes*/
    float** weights;
    float* biases;
    float l1_reg;
    float* input;
} Dense;

typedef struct{
    float lr; /*Learning Rate*/
    float beta1;
    float beta2;
    float epsilon;
    int t; /*Step Counter*/
    float** m; /*First Moment*/
    float** v; /*Second Moment*/
} Adam_Optimizer;
#endif
