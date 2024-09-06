#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "Serial_Operations.h"

// CIFAR-10 constants
#define IMAGE_SIZE 3072  // 32x32 pixels * 3 channels
#define LABEL_SIZE 1

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

void free_matrix_3D(Matrix3D matrix) {
    for (int i = 0; i < matrix.height; ++i) {
        for (int j = 0; j < matrix.width; ++j) {
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

void train_CNN(Dense* denseLayer, Matrix3D* x_train, float** y_train, int train_size, int batch_size, int epochs, float dropout_rate, Tensor3D* filters) {
    Adam_Optimizer optimizer = create_Adam(0.001, 0.9, 0.999, 1e-7);
    int num_batches = train_size / batch_size;

    /*Allocate memory for gradient weights and biases*/
    float** grad_weights = (float**)malloc(denseLayer->output_size * sizeof(float*));
    for (int i = 0; i < denseLayer->output_size; ++i) {
        grad_weights[i] = (float*)malloc(denseLayer->input_size * sizeof(float));
    }
    float* grad_biases = (float*)malloc(denseLayer->output_size * sizeof(float));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0;
        float epoch_accuracy = 0.0;
        for (int batch = 0; batch < num_batches; ++batch) {
            float batch_loss = 0.0;
            float batch_accuracy = 0.0;
            for (int i = 0; i < batch_size; ++i){
                int idx = batch * batch_size + i;

                Tensor3D convolved_output = convolve(&x_train[idx], filters, 1); /*Apply the convolution*/
                /*15x15x1*/
                int flattened_size = denseLayer->input_size;

                float* flattened_input = flatten(&convolved_output, &flattened_size); /*Flatten -> 225xnumber of tensors = 225x16 = 3600x1*/

                /*Apply dropout -> 3600x1*/
                float* dropped_input = (float*)malloc(flattened_size * sizeof(float));
                for (int j = 0; j < flattened_size; ++j) {
                    if ((rand() / (float)RAND_MAX) > dropout_rate) {
                        dropped_input[j] = flattened_input[j] / (1 - dropout_rate);
                    } else {
                        dropped_input[j] = 0;
                    }
                }

                float* output = dense_forward(denseLayer, dropped_input); /*Dense Layer, eventual ouptut will be 10x1*/

                /*Relu*/
                for(int j = 0; j < denseLayer->output_size; ++j){
                    output[j] = fmaxf(0, output[j]);
                }

                /*Softmax*/
                float* softmax_output = softmax(output, denseLayer->output_size);

                batch_loss += categorical_cross_entropy(softmax_output, y_train[idx], denseLayer->output_size);
                batch_accuracy += accuracy(softmax_output, y_train[idx], denseLayer->output_size);

                /*Backward pass*/
                calculate_gradients(denseLayer, output, y_train[idx], grad_weights, grad_biases);

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
            for (int i = 0; i < denseLayer->output_size; ++i) {
                denseLayer->biases[i] -= optimizer.lr * grad_biases[i]; /*Update the biases*/
            }
        }
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;
        printf("Epoch %d/%d - Loss: %.4f - Accuracy: %.4f\n", epoch + 1, epochs, epoch_loss, epoch_accuracy);
    }

    for (int i = 0; i < denseLayer->output_size; ++i) {
        free(grad_weights[i]);
    }
    free(grad_weights);
    free(grad_biases);
}


Matrix create_matrix(int height, int width) {
    Matrix matrix;
    matrix.height = height;
    matrix.width = width;
    matrix.data = (float**)malloc(height * sizeof(float*));
    for (int i = 0; i < height; ++i) {
        matrix.data[i] = (float*)malloc(width * sizeof(float));
        memset(matrix.data[i], 0, width * sizeof(float)); /*Initialize with 0*/
    }
    return matrix;
}

Tensor predict(const Tensor3D* input, Dense* denseLayer, Tensor3D* filters) {
    Tensor output;
    output.count = input->count;
    output.matrices = (Matrix*)malloc(output.count * sizeof(Matrix));

    for (int i = 0; i < output.count; ++i) {
        Tensor3D convolved_output = convolve(&input->matrices[i], filters, 1); /*Conv. Layer*/

        /*Flatten*/
        int flattened_size;
        float* flattened_input = flatten(&convolved_output, &flattened_size);

        /*Forward Pass*/
        float* dense_output = dense_forward(denseLayer, flattened_input);

        /*Dense Layer Output -> Matrix Format*/
        output.matrices[i] = create_matrix(1, denseLayer->output_size);
        for (int j = 0; j < denseLayer->output_size; ++j) {
            output.matrices[i].data[0][j] = dense_output[j];
        }

        free(flattened_input);
        free(dense_output);
        free_Tensor3D(&convolved_output);
    }

    return output;
}

void evaluate_model(Dense* denseLayer, Image_Dataset* test, int num_classes, Tensor3D* filters) {
    int correct_predictions = 0;

    for (int i = 0; i < test->count; ++i) {
        Tensor3D input_tensor;
        input_tensor.count = 1;
        input_tensor.matrices = (Matrix3D*)malloc(sizeof(Matrix3D));
        
        if (input_tensor.matrices == NULL){
            fprintf(stderr, "Failed to allocate memory for input_tensor.matrices\n");
            exit(EXIT_FAILURE);
        }

        input_tensor.matrices[0] = create_matrix_3D(32, 32, 3);
        if (input_tensor.matrices[0].data == NULL) {
            fprintf(stderr, "Failed to allocate memory for input_tensor.matrices[0].data\n");
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 32; ++k) {
                for (int c = 0; c < 3; ++c) {
                    input_tensor.matrices[0].data[j][k][c] = test->images[i].pixels[c * 1024 + j * 32 + k] / 255.0f; /*Normalise the pixel values*/
                }
            }
        }

        /*Perform convolution and dense layer prediction*/
        Tensor predictions = predict(&input_tensor, denseLayer, filters);
        if (predictions.matrices == NULL) {
            fprintf(stderr, "Failed to allocate memory for predictions.matrices\n");
            exit(EXIT_FAILURE);
        }
        float* predicted_probs = predictions.matrices[0].data[0];

        /*Find the index with the highest probability*/
        int predicted_label = 0;
        for (int j = 1; j < num_classes; ++j) {
            if (predicted_probs[j] > predicted_probs[predicted_label]) {
                predicted_label = j;
            }
        }

        /*Check if prediction matches actual*/
        if (predicted_label == test->images[i].label) {
            correct_predictions++;
        }

        // Free memory
        free(predictions.matrices[0].data[0]);
        free(predictions.matrices[0].data);
        free(predictions.matrices);
        free_matrix_3D(input_tensor.matrices[0]);
        free(input_tensor.matrices);
    }

    // Calculate and print accuracy
    float accuracy = (float)correct_predictions / test->count;
    printf("Model Accuracy on Test Dataset: %.4f\n", accuracy);
}