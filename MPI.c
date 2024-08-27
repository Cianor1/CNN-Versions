#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "MPI_Operations.h"

#define IMAGE_SIZE 3072 /*32x32x3*/
#define SUB_IMAGE_SIZE 768
#define LABEL_SIZE 1
#define IMAGE_DIM 32
#define SUBIMAGE_DIM 16
#define SUB_IMAGE_DIM2 8

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int myid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs != 4 && nprocs != 16) { /*Can use 16 processors, but a lot of changes are needed across the program*/
        if (myid == 0) {
            fprintf(stderr, "Error: This program must be run with either 4 or 16 processors. Detected %d processors.\n", nprocs);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    srand(time(NULL) + myid); /*Ensures each process has its own sequence of random numbers*/

    const char* train_files[]={
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
    };

    int num_train_files = sizeof(train_files) / sizeof(train_files[0]);
    int subsetSize = 10000;

    Sub_Image* train_images = (Sub_Image*)malloc(subsetSize * sizeof(Sub_Image)); /*40,000 sub-images*/

    for(int i = 0; i < num_train_files; ++i){
        read_cifar10_in_quadrants(train_files[i], train_images, subsetSize, myid);
    }

    Matrix3D* x_train = (Matrix3D*)malloc(subsetSize * sizeof(Matrix3D)); /*Create the matrices to hold the sub-images*/
    float** y_train = (float**)malloc(subsetSize * sizeof(float*));

    for(int i = 0; i < subsetSize; ++i){
        x_train[i] = convert_Sub_Image_To_Matrix_3D(&train_images[i]); /*Converts the sub-images and performs the normalisation of the pixel values too!*/
        y_train[i] = (float*)calloc(10, sizeof(float));
        y_train[i][train_images[i].label] = 1.0f; /*One-hot encoded*/
    }

    int input_size = SUBIMAGE_DIM * SUBIMAGE_DIM * 3;
    int output_size = 10;
    float l1_reg = 0.001;
    int epochs = 10;
    int batch_size = 8;
    float dropout_rate = 0.5;
    int num_filters = 16;

    Dense dense_layer = create_dense(input_size, output_size, l1_reg);

    Tensor3D filters = create_filters(num_filters, 3, 3);

    /*Timing both the CNN and CNN-DNN*/
    clock_t start_time1 = clock();
    clock_t start_time2 = clock();

    /*Perform the CNN with the gather included*/
    float* local_output = train_CNN(&dense_layer, x_train, y_train, subsetSize, batch_size, epochs, dropout_rate, &filters, myid);

    /*Allocate space on the root processor to gather all outputs*/
    float* gathered_outputs = NULL;
    if(myid == 0){
        gathered_outputs = (float*)malloc(nprocs * dense_layer.output_size * sizeof(float));
    }

    /*Gather all outputs on the root processor*/
    MPI_Gather(local_output, dense_layer.output_size, MPI_FLOAT, gathered_outputs, dense_layer.output_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    clock_t end_time1 = clock();
    double elapsed_time1 = (double)(end_time1 - start_time1) / CLOCKS_PER_SEC;

    if(myid == 0){
        printf("\nCNN Training took %.2f seconds.\n", elapsed_time1);
    }
    int final_vec_size = 10;

    float* final_vec = (float*)malloc(final_vec_size * sizeof(float));

    /*This is tailored to use averages and ANN, but can be adjusted to do the global coarse net. w/ DNN*/
    /*Now we can take our gathered data and perform the global coarse network, or take the average on the nodes*/
    if(myid == 0){
        int no_of_nodes = nprocs;
        for(int j = 0; j < final_vec_size; ++j){
            float sum = 0.0;
            for(int i = 0; i < no_of_nodes; ++i){
                sum += gathered_outputs[i * final_vec_size + j];
            }
            final_vec[j] = sum / no_of_nodes;
        }

        int new_batch_size = 10; /*New batch size for DNN*/
        int new_num_layers = 1; /*Three additional dense layers*/
        Dense* layers[new_num_layers];

        //Dense second_layer = create_dense(dense_layer.output_size, 64, l1_reg);
        //layers[0] = &second_layer;

        //Dense third_layer = create_dense(second_layer.output_size, 32, l1_reg);
        //layers[1] = &third_layer;

        Dense fourth_layer = create_dense(dense_layer.output_size, output_size, l1_reg);
        layers[0] = &fourth_layer;

        train_DNN(layers, new_num_layers, final_vec, y_train, 40, dense_layer.output_size, new_batch_size, epochs);

        /*Freeing the respective layers' weights and biases*/
        //for(int i = 0; i < second_layer.output_size; ++i){
        //    free(second_layer.weights[i]);
        //}

        //for(int i = 0; i < third_layer.output_size; ++i){
        //    free(third_layer.weights[i]);
        //}

        for(int i = 0; i < fourth_layer.output_size; ++i){
            free(fourth_layer.weights[i]);
        }

        //free(second_layer.weights);
        //free(second_layer.biases);
        //free(third_layer.weights);
        //free(third_layer.biases);
        free(fourth_layer.weights);
        free(fourth_layer.biases);
        free(final_vec);
    }

    if(myid == 0){
        free(gathered_outputs);
    }

    clock_t end_time2 = clock();
    double elapsed_time2 = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;

    if(myid == 0){
        printf("\nCNN-DNN Training took %.2f seconds.\n", elapsed_time2);
    }

    for (int i = 0; i < subsetSize; ++i) {
        free_matrix_3D(x_train[i]);
        free(y_train[i]);
    }
    free(x_train);
    free(y_train);
    free_Tensor3D(&filters);
    free(train_images);

    for(int i = 0; i < dense_layer.output_size; ++i){
        free(dense_layer.weights[i]);
    }
    free(dense_layer.weights);
    free(dense_layer.biases);

    MPI_Finalize();

    return 0;
}