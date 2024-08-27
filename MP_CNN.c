#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "Serial_Read.h"
#include "MP_Operations.h"

#define IMAGE_SIZE 3072 /*32x32x3*/
#define LABEL_SIZE 1

int main(){
    srand(time(NULL)); /*For Timing*/

    omp_set_num_threads(4);

    const char* train_files[] = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
    };

    //const char* test_files[] = {"test_batch.bin"};

    int num_train_files = sizeof(train_files) / sizeof(train_files[0]);
    int subsetSize = 10000;

    Image_Dataset train_dataset = read_Cifar10_Multi(train_files, num_train_files, subsetSize);

    //Image_Dataset test_dataset = readCifar10Files(test_files, 1, subsetSize);

    Class_Names class_names = read_class_names("batches.meta.txt");

    printf("Number of images read: %d\n", train_dataset.count);
    printf("Number of class names read: %d\n", class_names.count);

    /*Convert training images to 3D Matrices*/
    Matrix3D* x_train = (Matrix3D*)malloc(train_dataset.count * sizeof(Matrix3D));
    float** y_train = (float**)malloc(train_dataset.count * sizeof(float*));

    for (int i = 0; i < train_dataset.count; ++i){
        x_train[i] = create_matrix_3D(32, 32, 3);
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                for (int c = 0; c < 3; ++c) {
                    x_train[i].data[j][k][c] = train_dataset.images[i].pixels[c * 1024 + j * 32 + k] / 255.0f; /*Normalize the pixels*/
                }
            }
        }
        /*One-hot encode labels*/
        y_train[i] = (float*)calloc(10, sizeof(float));
        y_train[i][train_dataset.images[i].label] = 1.0f;
    }

    /*Key model parameters*/
    int output_size = 10;
    float l1_reg = 0.001;
    int epochs = 10;
    int batch_size = 8;
    float dropout_rate = 0.5;
    int num_filters = 16;

    Dense dense_layer = create_dense(IMAGE_SIZE, output_size, l1_reg);

    Tensor3D filters = create_filters(num_filters, 3, 3);

    double time = omp_get_wtime();

    train_CNN(&dense_layer, x_train, y_train, train_dataset.count, batch_size, epochs, dropout_rate, &filters);

    //evaluate_model(&dense_layer, &test_dataset, output_size, &filters);

    printf("\nTraining took %.2f seconds.\n", omp_get_wtime() - time);

    /*Appropriately freeing the memory*/
    for (int i = 0; i < train_dataset.count; ++i) {
        free_matrix_3D(x_train[i]);
        free(y_train[i]);
    }
    free(x_train);
    free(y_train);
    free_Tensor3D(&filters);
    freeClassNames(&class_names);
    free(train_dataset.images);

    for (int i = 0; i < dense_layer.output_size; ++i) {
        free(dense_layer.weights[i]);
    }
    free(dense_layer.weights);
    free(dense_layer.biases);
    free(dense_layer.input);

    return 0;
}