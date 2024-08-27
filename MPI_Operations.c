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


void read_cifar10_in_quadrants(const char* filename, Sub_Image* sub_images, int num_images, int myid){
    MPI_File file;
    MPI_Offset offset;

    int start_row;
    if(myid < 2){ /*Determine where the starting row for a processor is*/
        start_row = 0;
    } 
    else{
        start_row = SUBIMAGE_DIM;
    }

    int start_col;
    if(myid % 2 == 0){ /*Determine where the starting column for a processor is*/
        start_col = 0;
    } 
    else{
        start_col = SUBIMAGE_DIM;
    }

    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    /*This is set up so each sub-image gets its appropriate label*/
    for(int i = 0; i < num_images; ++i){
        MPI_File_read_at(file, i * (LABEL_SIZE + IMAGE_SIZE), &sub_images[i].label, LABEL_SIZE, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE); /*Reads the label for the sub-image*/

        for(int c = 0; c < 3; ++c){
            for(int row = 0; row < SUBIMAGE_DIM; ++row){
                offset = i * (LABEL_SIZE + IMAGE_SIZE) + LABEL_SIZE + c * IMAGE_DIM * IMAGE_DIM + (start_row + row) * IMAGE_DIM + start_col; /*Calculate the offset*/
                MPI_File_read_at(file, offset, &sub_images[i].sub_image[c][row][0], SUBIMAGE_DIM, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE); /*Read the sub-image*/
            }
        }
    }

    MPI_File_close(&file);
}

/*Function that extends the notion to use 16 processors*/
void read_cifar10_in_smaller_quadrants(const char* filename, Sub_Image* sub_images, int num_images, int myid) {
    MPI_File file;
    MPI_Offset start_offset;

    /*Very Manual calculations here*/
    int start_row;
    if(myid < 4){
        start_row = 0;
    } 
    else if(myid >= 4 && myid < 8){
        start_row = SUB_IMAGE_DIM2;
    }
    else if(myid >= 8 && myid < 12){
        start_row = SUB_IMAGE_DIM2 * 2;
    }
    else{
        start_row = SUB_IMAGE_DIM2 * 3;
    }

    int start_col;
    if(myid == 0 || myid == 4 || myid == 8 || myid == 12){
        start_col = 0;
    } 
    else if(myid == 1 || myid == 5 || myid == 9 || myid == 13){
        start_col = SUB_IMAGE_DIM2;
    }
    else if(myid == 2 || myid == 6 || myid == 10 || myid == 14){
        start_col = SUB_IMAGE_DIM2 * 2;
    }
    else{
        start_col = SUB_IMAGE_DIM2 * 3;
    }

    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    /*This is set up so each sub-image gets its appropriate label*/
    for (int i = 0; i < num_images; ++i) {
        MPI_File_read_at(file, i * (LABEL_SIZE + IMAGE_SIZE), &sub_images[i].label, LABEL_SIZE, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE); /*Reads the label for the sub-image*/

        for (int c = 0; c < 3; ++c) {
            for (int row = 0; row < SUB_IMAGE_DIM2; ++row) {
                start_offset = i * (LABEL_SIZE + IMAGE_SIZE) + LABEL_SIZE + c * IMAGE_DIM * IMAGE_DIM + (start_row + row) * IMAGE_DIM + start_col; /*Calculate the offset*/
                MPI_File_read_at(file, start_offset, &sub_images[i].sub_image[c][row][0], SUB_IMAGE_DIM2, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE); /*Read the sub-image*/
            }
        }
    }

    MPI_File_close(&file);
}

Matrix3D create_matrix_3D(int height, int width, int depth){
    Matrix3D matrix;
    matrix.height = height;
    matrix.width = width;
    matrix.depth = depth;

    matrix.data = (float***)malloc(height * sizeof(float**)); /*Allocate memory for the 3D array*/

    for(int i = 0; i < height; ++i){
        matrix.data[i] = (float**)malloc(width * sizeof(float*)); /*Allocate memory for each 2D matrix*/
        for(int j = 0; j < width; ++j){
            matrix.data[i][j] = (float*)malloc(depth * sizeof(float)); /*Allocate memory for each row*/
        }
    }
    return matrix;
}

Matrix3D convert_Sub_Image_To_Matrix_3D(const Sub_Image* sub_image){ /*Converts the sub-image struct in a 3d matrix struct*/
    Matrix3D matrix = create_matrix_3D(SUBIMAGE_DIM, SUBIMAGE_DIM, 3);
    for(int i = 0; i < SUBIMAGE_DIM; ++i){
        for(int j = 0; j < SUBIMAGE_DIM; ++j){
            for(int c = 0; c < 3; ++c){
                matrix.data[i][j][c] = (float)sub_image->sub_image[c][i][j]/255.0f; /*Normalize the pixel values*/
            }
        }
    }
    return matrix;
}

void free_matrix_3D(Matrix3D matrix){ /*Free the 3D matrix*/
    for(int i = 0; i < matrix.height; ++i){
        for(int j = 0; j < matrix.width; ++j){
            free(matrix.data[i][j]);
        }
        free(matrix.data[i]);
    }
    free(matrix.data);
}

Matrix3D apply_padding(const Matrix3D* input, int pad_height, int pad_width){
    Matrix3D padded;
    int padded_height = input->height + 2 * pad_height;
    int padded_width = input->width + 2 * pad_width;
    int depth = input->depth;
    padded = create_matrix_3D(padded_height, padded_width, depth); /*Create the padded matrix that will be used in convolutions*/

    /*Loop through the padded matrix to find the location in it*/
    for(int i = 0; i < padded_height; ++i){
        for(int j = 0; j < padded_width; ++j){
            for(int k = 0; k < depth; ++k){
                int x = i - pad_height;
                int y = j - pad_width;
                if(x >= 0 && x < input->height && y >= 0 && y < input->width){ /*Ensures we are within the dimensions of the original matrix to:*/
                    padded.data[i][j][k] = input->data[x][y][k]; /*Input the correct values*/
                }
                else{
                    padded.data[i][j][k] = 0.0f; /*Input zeros*/
                }
            }
        }
    }

    return padded;
}

Tensor3D convolve(const Matrix3D* input, const Tensor3D* filters, int stride){
    Matrix3D padded_input = apply_padding(input, 2, 2); /*Padding applied to height and width*/

    int num_filters = filters->count;
    int filter_height = filters->matrices[0].height;
    int filter_width = filters->matrices[0].width;
    int filter_depth = filters->matrices[0].depth;
    int output_height = (padded_input.height - filter_height) / stride + 1; /*(34-3)/2 = 15.5=15*/
    int output_width = (padded_input.width - filter_width) / stride + 1; /*(34-3)/2 = 15.5=15*/
    int output_depth = (padded_input.depth - filter_depth) / stride + 1; /*(3-1)/2 = 1*/

    Tensor3D output; /*Initialize a Tensor3D structure to hold the output matrices (Tensor is a high dimensional array)*/
    output.count = num_filters; /*Number of output matrices = number of filters*/
    output.matrices = (Matrix3D*)malloc(num_filters * sizeof(Matrix3D)); /*Memory for the tensor*/

    if(output.matrices == NULL){
        fprintf(stderr, "Failed to allocate memory for output matrices.\n");
        output.count = 0;
        return output;
    }

    for(int f = 0; f < num_filters; ++f){
        output.matrices[f] = create_matrix_3D(output_height, output_width, output_depth);

        /*Performing the Convolution itself*/
        for(int i = 0; i < output_height; ++i){
            for(int j = 0; j < output_width; ++j){
                for(int k = 0; k < output_depth; ++k){
                    float sum = 0.0; /*Sum for the result at a given position in the output matrix*/
                    for(int m = 0; m < filter_height; ++m){
                        for(int n = 0; n < filter_width; ++n){
                            for(int o = 0; o < filter_depth; ++o){
                                /*Calculating the position in the input matrix*/
                                int x = i * stride + m;
                                int y = j * stride + n;
                                int z = k * stride + o;
                                /*Check if we are within the bounds of the input matrix*/
                                if(x < input->height && y < input->width && z < input->depth){
                                    sum += input->data[x][y][z] * filters->matrices[f].data[m][n][o]; /*Convolutional sum operation*/
                                }
                            }
                        }
                    }
                    output.matrices[f].data[i][j][k] = sum; /*Store the result*/
                }
            }
        }
    }

    return output;
}

float* flatten(const Tensor3D* input, int* flattened_size){
    int size = 0;
    /*To flatten the input matrix you must get the total elements and also we need to account for the tensor*/
    for(int c = 0; c < input->count; ++c){
        size += input->matrices[c].height * input->matrices[c].width * input->matrices[c].depth;
    }
    
    float* output = (float*)malloc(size * sizeof(float)); /*Will hold the flattened output*/
    if(output == NULL){
        fprintf(stderr, "Failed to allocate memory for flattened output.\n");
        *flattened_size = 0;
        return NULL;
    }

    int index = 0;
    for(int c = 0; c < input->count; ++c){
        for(int i = 0; i < input->matrices[c].height; ++i){
            for(int j = 0; j < input->matrices[c].width; ++j){
                for(int k = 0; k < input->matrices[c].depth; ++k){
                    output[index++] = input->matrices[c].data[i][j][k]; /*Fill the flattened matrix with the correct values*/
                }
            }
        }
    }
    
    *flattened_size = size;
    return output;
}

float* softmax(const float* input, int size){
    float* output = (float*)malloc(size * sizeof(float)); /*Will hold softmax output*/
    if (output == NULL) {
        fprintf(stderr, "Failed to allocate memory for output.\n");
        return NULL;
    }

    /*Get the max value of the input*/
    float max_value = input[0];
    for(int i = 1; i < size; ++i){
        if(input[i] > max_value){
            max_value = input[i];
        }
    }

    /*Get the exponent of each input minus the max value*/
    float sum = 0.0;
    for(int i = 0; i < size; ++i){
        sum += expf(input[i] - max_value); /*The max value is subtracted to avoid very large values as the biggest value after the exponent will be 1*/
    }
    
    /*This computes the softmax values*/
    for(int i = 0; i < size; ++i){
        output[i] = expf(input[i] - max_value) / sum;
    }

    return output;
}

Dense create_dense(int input_size, int output_size, float l1_reg){
    Dense layer;
    layer.input_size = input_size; /*Size of what enters the dense layer*/
    layer.output_size = output_size; /*Size of what leaves*/
    layer.l1_reg = l1_reg; /*Set the l1 regularisation*/
    layer.input = (float*)malloc(input_size * sizeof(float)); /*Allocating memory for the input*/

    layer.weights = (float**)malloc(output_size * sizeof(float*)); /*Allocate memory for the weight matrix used in the dense layer*/
    for(int i = 0; i < output_size; ++i){
        layer.weights[i] = (float*)malloc(input_size * sizeof(float)); /*Allocate memory for the rows of the matrix*/
        for(int j = 0; j < input_size; ++j){
            layer.weights[i][j] = ((float)rand() / RAND_MAX) * 0.01; /*Initialising the weights as random values*/
        }
    }
    layer.biases = (float*)malloc(output_size * sizeof(float)); /*Allocate memory for the biases*/
    for(int i = 0; i < output_size; ++i){
        layer.biases[i] = 0.0; /*Initialising the biases to 0*/
    }

    return layer;
}

float* dense_forward(Dense* denseLayer, const float* input){
    for(int i = 0; i < denseLayer->input_size; ++i){
        denseLayer->input[i] = input[i]; /*The dense layer input becomes whatever entered the dense layer*/
    }

    float* output = (float*)malloc(denseLayer->output_size * sizeof(float)); /*Allocate the memory for whatever leaves the dense layer (Will be 10 output neurons)*/
    if (output == NULL) {
        fprintf(stderr, "Failed to allocate memory for output.\n");
        exit(EXIT_FAILURE);
    }
    
    for(int i = 0; i < denseLayer->output_size; ++i){
        output[i] = denseLayer->biases[i]; /*Current neuron starts with the bias value*/
        for(int j = 0; j < denseLayer->input_size; ++j){
            output[i] += denseLayer->weights[i][j] * input[j]; /*Then update the output by adding the weights times the input*/
        }
    }
    return output;
}

float L1_Reg_Loss(const Dense* denseLayer){
    float l1_loss = 0.0;
    for(int i = 0; i < denseLayer->output_size; ++i){
        for(int j = 0; j < denseLayer->input_size; ++j){
            l1_loss += fabsf(denseLayer->weights[i][j]); /*Get the L1 loss by summing the absolute value of the output neurons*/
        }
    }
    return l1_loss * denseLayer->l1_reg; /*Multiply it by the l1 regularisation and return*/
}

Adam_Optimizer create_Adam(float lr, float beta1, float beta2, float epsilon){
    Adam_Optimizer optimizer;
    optimizer.lr = lr; /*Learning rate*/
    optimizer.beta1 = beta1; /*Exponential decay rate of the first moment estimate*/
    optimizer.beta2 = beta2; /*Exponential decay rate of the second moment estimate*/
    optimizer.epsilon = epsilon; /*Prevents division by zero (error)*/
    optimizer.t = 0; /*Itereation step counter*/
    optimizer.m = NULL; /*First moment estimate, calculated in training*/
    optimizer.v = NULL; /*Second moment estimate: same as*/
    return optimizer;
}

void Adam_Update(Adam_Optimizer* optimizer, float** weights, float** grads, int rows, int cols){
    if(optimizer->m == NULL){
        optimizer->m = (float**)malloc(rows * sizeof(float*)); /*Allocate memory for first moment estimates*/
        optimizer->v = (float**)malloc(rows * sizeof(float*)); /*Allocate memory for second moment estimates*/
        for(int i = 0; i < rows; ++i){
            optimizer->m[i] = (float*)calloc(cols, sizeof(float)); /*Allocate memory for each row of first moment estimates*/
            optimizer->v[i] = (float*)calloc(cols, sizeof(float)); /*Allocate memory for each row of second moment estimates*/
        }
    }
    optimizer->t++;
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            optimizer->m[i][j] = optimizer->beta1 * optimizer->m[i][j] + (1 - optimizer->beta1) * grads[i][j]; /*First moment estimate eqn. update w/ gradient*/
            optimizer->v[i][j] = optimizer->beta2 * optimizer->v[i][j] + (1 - optimizer->beta2) * grads[i][j] * grads[i][j]; /*Second moment estimate eqn. update w/ gradient*/
            float m_hat = optimizer->m[i][j] / (1 - powf(optimizer->beta1, optimizer->t)); /*Bias corrected estimators for the first and second moments*/
            float v_hat = optimizer->v[i][j] / (1 - powf(optimizer->beta2, optimizer->t)); /*Bias corrected estimators for the first and second moments*/
            weights[i][j] -= optimizer->lr * m_hat / (sqrtf(v_hat) + optimizer->epsilon); /*Weight update*/
        }
    }
}

float categorical_cross_entropy(const float* predicted, const float* actual, int size){
    float loss = 0.0;
    for(int i = 0; i < size; ++i){
        loss -= actual[i] * logf(predicted[i] + 1e-9); /*actual is the one-hot encoded values and predicted is the predicted probability*/
    }
    return loss;
}

float accuracy(const float* predicted, const float* actual, int size){
    int pred_label = 0;
    int actual_label = 0;
    for(int i = 1; i < size; ++i){
        /*Set pred label to the class with the highest predicted prob.*/
        if(predicted[i] > predicted[pred_label]){
            pred_label = i;
        }
        /*Set actual label to the class with the highest actual prob.*/
        if(actual[i] > actual[actual_label]){
            actual_label = i;
        }
    }

    /*If they match return 1 and if they don't return 0*/
    if(pred_label==actual_label){
        return 1.0f;
    }
    else{
        return 0.0f;
    }
}

void calculate_gradients(Dense* denseLayer, const float* output, const float* target, float** grad_weights, float* grad_biases) {
    for (int i = 0; i < denseLayer->output_size; ++i) {
        float error = output[i] - target[i];
        grad_biases[i] = error;
        for (int j = 0; j < denseLayer->input_size; ++j) {
            grad_weights[i][j] = error * denseLayer->input[j];
        }
    }
}

Tensor3D create_filters(int num_filters, int height, int width) {
    Tensor3D filters;
    filters.count = num_filters;
    filters.matrices = (Matrix3D*)malloc(num_filters * sizeof(Matrix3D)); /*Allocate memory for the filter matrices*/

    if (filters.matrices == NULL) {
        fprintf(stderr, "Failed to allocate memory for filters matrices.\n");
        filters.count = 0;
        return filters;
    }

    for (int i = 0; i < num_filters; ++i) {
        filters.matrices[i] = create_matrix_3D(height, width, 1); /*Creating a 2D matrix*/
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                filters.matrices[i].data[h][w][0] = ((float)rand() / RAND_MAX) * 0.01; /*Setting the values of this matrix at random*/
            }
        }
    }

    return filters;
}

void free_Tensor3D(Tensor3D* tensor){ /*Free the high dimensional matrix*/
    if(tensor != NULL && tensor->matrices != NULL){
        for(int i = 0; i < tensor->count; ++i){
            free_matrix_3D(tensor->matrices[i]);
        }
        free(tensor->matrices);
        tensor->matrices = NULL;
        tensor->count = 0;
    }
}


float* train_CNN(Dense* denseLayer, Matrix3D* x_train, float** y_train, int train_size, int batch_size, int epochs, float dropout_rate, Tensor3D* filters, int myid) {
    Adam_Optimizer optimizer = create_Adam(0.001, 0.9, 0.999, 1e-7);
    int num_batches = train_size / batch_size;

    /*Allocate memory for gradient weights and biases*/
    float** grad_weights = (float**)malloc(denseLayer->output_size * sizeof(float*));
    for (int i = 0; i < denseLayer->output_size; ++i) {
        grad_weights[i] = (float*)malloc(denseLayer->input_size * sizeof(float));
    }
    float* grad_biases = (float*)malloc(denseLayer->output_size * sizeof(float));

    /*Variable to store the output of the last batch*/
    float* final_output = (float*)malloc(denseLayer->output_size * sizeof(float)); /*Stores just one result*/

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0;
        float epoch_accuracy = 0.0;
        for (int batch = 0; batch < num_batches; ++batch) {
            float batch_loss = 0.0;
            float batch_accuracy = 0.0;
            for (int i = 0; i < batch_size; ++i) {
                int idx = batch * batch_size + i;

                Tensor3D convolved_output = convolve(&x_train[idx], filters, 1); /*Apply the convolution*/
                int flattened_size = denseLayer->input_size;

                float* flattened_input = flatten(&convolved_output, &flattened_size); /*Flatten the input*/

                /*Apply dropout*/
                float* dropped_input = (float*)malloc(flattened_size * sizeof(float));
                for (int j = 0; j < flattened_size; ++j) {
                    if ((rand() / (float)RAND_MAX) > dropout_rate) {
                        dropped_input[j] = flattened_input[j] / (1 - dropout_rate);
                    } else {
                        dropped_input[j] = 0;
                    }
                }

                /*Forward pass*/
                float* output = dense_forward(denseLayer, dropped_input);

                /*ReLU*/
                for (int j = 0; j < denseLayer->output_size; ++j) {
                    output[j] = fmaxf(0, output[j]);
                }

                /*Softmax*/
                float* softmax_output = softmax(output, denseLayer->output_size);

                batch_loss += categorical_cross_entropy(softmax_output, y_train[idx], denseLayer->output_size);
                batch_accuracy += accuracy(softmax_output, y_train[idx], denseLayer->output_size);

                /*Backward pass*/
                calculate_gradients(denseLayer, output, y_train[idx], grad_weights, grad_biases);

                /*Copy the outputs of the last batch of the last epoch, essentially taking a sample*/
                if (epoch == epochs - 1 && batch == num_batches - 1 && i == batch_size - 1) {
                    memcpy(final_output, output, denseLayer->output_size * sizeof(float));
                }

                free(output);
                free(softmax_output);
                free(dropped_input);
                free(flattened_input);
                free_Tensor3D(&convolved_output);
            }
            batch_loss /= batch_size;
            batch_accuracy /= batch_size;

            batch_loss += L1_Reg_Loss(denseLayer);

            epoch_loss += batch_loss;
            epoch_accuracy += batch_accuracy;

            /*Update weights using Adam optimizer*/
            Adam_Update(&optimizer, denseLayer->weights, grad_weights, denseLayer->output_size, denseLayer->input_size);
            for (int i = 0; i < denseLayer->output_size; ++i){
                denseLayer->biases[i] -= optimizer.lr * grad_biases[i]; /*Update the biases*/
            }
        }
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;
        if(myid==0){
            printf("Process %d - Epoch %d/%d - Loss: %.4f - Accuracy: %.4f\n", myid, epoch + 1, epochs, epoch_loss, epoch_accuracy);
        }
    }

    for (int i = 0; i < denseLayer->output_size; ++i) {
        free(grad_weights[i]);
    }
    free(grad_weights);
    free(grad_biases);

    return final_output; /*Return the output of the last batch from each processor*/
}

void train_DNN(Dense** layers, int num_layers, float* gathered_data, float** y_train, int train_size, int input_size, int batch_size, int epochs) {
    Adam_Optimizer optimizer = create_Adam(0.001, 0.9, 0.999, 1e-7);
    int num_batches = train_size / batch_size;

    /* Allocate memory for gradient weights and biases for each layer */
    float*** grad_weights = (float***)malloc(num_layers * sizeof(float**));
    float** grad_biases = (float**)malloc(num_layers * sizeof(float*));
    for(int layer = 0; layer < num_layers; ++layer){
        grad_weights[layer] = (float**)malloc(layers[layer]->output_size * sizeof(float*));
        for(int i = 0; i < layers[layer]->output_size; ++i){
            grad_weights[layer][i] = (float*)malloc(layers[layer]->input_size * sizeof(float));
        }
        grad_biases[layer] = (float*)malloc(layers[layer]->output_size * sizeof(float));
    }

    for(int epoch = 0; epoch < epochs; ++epoch){
        float epoch_loss = 0.0;
        float epoch_accuracy = 0.0;
        for(int batch = 0; batch < num_batches; ++batch){
            float batch_loss = 0.0;
            float batch_accuracy = 0.0;
            for(int i = 0; i < batch_size; ++i){
                int idx = batch * batch_size + i;

                /*Gathered Data is the input*/
                float* input = &gathered_data[idx * input_size];
                float* temp_output = NULL;
                float* layer_input = input;

                if(num_layers == 1){
                    /*If one dense layer -> ANN*/
                    temp_output = dense_forward(layers[0], layer_input);

                    /*ReLU and Softmax*/
                    for(int j = 0; j < layers[0]->output_size; ++j){
                        temp_output[j] = fmaxf(0, temp_output[j]);
                    }
                    float* softmax_output = softmax(temp_output, layers[0]->output_size);

                    batch_loss += categorical_cross_entropy(softmax_output, y_train[idx], layers[0]->output_size);
                    batch_accuracy += accuracy(softmax_output, y_train[idx], layers[0]->output_size);

                    /*Backward pass, Compute gradients*/
                    calculate_gradients(layers[0], temp_output, y_train[idx], grad_weights[0], grad_biases[0]);

                    /*Update weights and biases*/
                    Adam_Update(&optimizer, layers[0]->weights, grad_weights[0], layers[0]->output_size, layers[0]->input_size);
                    for(int j = 0; j < layers[0]->output_size; ++j){
                        layers[0]->biases[j] -= optimizer.lr * grad_biases[0][j];
                    }

                    free(softmax_output);
                } 
                else{ /*Else, if there are 2 or more layers -> DNN*/
                    float** layer_outputs = (float**)malloc(num_layers * sizeof(float*));
                    for(int layer = 0; layer < num_layers; ++layer){
                        float* output = dense_forward(layers[layer], layer_input);

                        /*ReLU*/
                        for(int j = 0; j < layers[layer]->output_size; ++j){
                            output[j] = fmaxf(0, output[j]);
                        }

                        /*Store layer output*/
                        layer_outputs[layer] = output;
                        layer_input = output;

                        if(layer == num_layers - 1){ /*If we are on our last layer*/
                            temp_output = output;
                        }
                    }

                    /*Apply Softmax to the output of the last layer*/
                    float* softmax_output = softmax(temp_output, layers[num_layers - 1]->output_size);

                    batch_loss += categorical_cross_entropy(softmax_output, y_train[idx], layers[num_layers - 1]->output_size);
                    batch_accuracy += accuracy(softmax_output, y_train[idx], layers[num_layers - 1]->output_size);

                    /*Backward pass, Compute gradients*/
                    calculate_gradients(layers[num_layers - 1], temp_output, y_train[idx], grad_weights[num_layers - 1], grad_biases[num_layers - 1]);

                    for(int layer = num_layers - 2; layer >= 0; --layer){
                        calculate_gradients(layers[layer], layer_outputs[layer], y_train[idx], grad_weights[layer], grad_biases[layer]);
                    }

                    /*Update weights and biases*/
                    for (int layer = 0; layer < num_layers; ++layer) {
                        Adam_Update(&optimizer, layers[layer]->weights, grad_weights[layer], layers[layer]->output_size, layers[layer]->input_size);
                        for (int j = 0; j < layers[layer]->output_size; ++j) {
                            layers[layer]->biases[j] -= optimizer.lr * grad_biases[layer][j];
                        }
                    }

                    free(softmax_output);
                    for (int layer = 0; layer < num_layers; ++layer) {
                        free(layer_outputs[layer]);
                    }
                    free(layer_outputs);
                }
            }

            batch_loss /= batch_size;
            batch_accuracy /= batch_size;

            batch_loss += L1_Reg_Loss(layers[num_layers - 1]); /*Calculate the L1 Regularisation Loss on the last layer*/

            epoch_loss += batch_loss;
            epoch_accuracy += batch_accuracy;
        }
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;
        printf("Epoch %d/%d - Loss: %.4f - Accuracy: %.4f\n", epoch + 1, epochs, epoch_loss, epoch_accuracy);
    }

    for(int layer = 0; layer < num_layers; ++layer){
        for(int i = 0; i < layers[layer]->output_size; ++i){
            free(grad_weights[layer][i]);
        }
        free(grad_weights[layer]);
        free(grad_biases[layer]);
    }
    free(grad_weights);
    free(grad_biases);
}