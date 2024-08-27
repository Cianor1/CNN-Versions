#ifndef MPI_FILE
#define MPI_FILE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "structs.h"

#define IMAGE_SIZE 3072 /*32x32x3*/
#define SUB_IMAGE_SIZE 768
#define LABEL_SIZE 1
#define IMAGE_DIM 32
#define SUBIMAGE_DIM 16
#define SUB_IMAGE_DIM2 8

/*Additional Struct for parallel model*/
typedef struct{
    unsigned char label;
    unsigned char sub_image[3][SUBIMAGE_DIM][SUBIMAGE_DIM]; /*3x16x16*/
} Sub_Image;

/**
* @brief  This function will use MPI/IO to read the binary image files into 16x16 sizes
* @param[in] filename - The name of the file to be read
* @param[in] sub_images - The structure to hold the images
* @param[in] num_images - The number of images to be read
* @param[in] myid - The processor id
*/
void read_cifar10_in_quadrants(const char* filename, Sub_Image* sub_images, int num_images, int myid);

/**
* @brief  This function will use MPI/IO to read the binary image files into 8x8 sizes
* @param[in] filename - The name of the file to be read
* @param[in] sub_images - The structure to hold the images
* @param[in] num_images - The number of images to be read
* @param[in] myid - The processor id
*/
void read_cifar10_in_smaller_quadrants(const char* filename, Sub_Image* sub_images, int num_images, int myid);

/**
* @brief  This function will create a 3D matrix
* @param[in] height - The height of the matrix to be created
* @param[in] width - The width of the matrix to be created
* @param[in] depth - The depth of the matrix to be created
*/
Matrix3D create_matrix_3D(int height, int width, int depth);

/**
* @brief  This function will convert the sub image object to a matrix object
* @param[in] sub_img - The sub images to be converted (all of them)
*/
Matrix3D convert_Sub_Image_To_Matrix_3D(const Sub_Image* sub_img);

/**
 * @brief This function will free the matrix created using the above function
 * @param[in] matrix - The matrix that was created and needs to be freed
 */
void free_matrix_3D(Matrix3D matrix);

/**
 * @brief This function applies padding to the height and width of a matrix
 * @param[in] input - The matrix that the padding will be applied to
 * @param[in] pad_height - The amount of padding applied to the height (i.e. rows to be added above and below the og matrix)
 * @param[in] pad_width - The amount of padding applied to the width (i.e. cols. to be added either side of the og matrix)
 */
Matrix3D apply_padding(const Matrix3D* input, int pad_height, int pad_width);

/**
 * @brief This function applies the convolutional operations to our input matrix
 * @param[in] input - The matrix that the convolution will be applied to
 * @param[in] filters - The filter matrices to be applied to the input in the convolution operation
 * @param[in] stride - The amount of values to skip over as the kernel or filter slides across the input matrix
 */
Tensor3D convolve(const Matrix3D* input, const Tensor3D* filters, int stride);

/**
 * @brief This function flattens a 3D input down to a 1D array
 * @param[in] input - The high dimensional matrix input
 * @param[in] flattened_size - The
 */
float* flatten(const Tensor3D* input, int* flattened_size);

/**
 * @brief Applies softmax to the output of the dense layer
 * @param[in] input - The 1D array to be worked on
 * @param[in] size - The length of this 1D array
 */
float* softmax(const float* input, int size);

/**
 * @brief Initialises and returns a dnese layer
 * @param[in] input_size - Size of the input entering the dense layer
 * @param[in] output_size - Size of the output exiting the dense layer
 * @param[in] l1_reg - The l1 regularisation applied
 */
Dense create_dense(int input_size, int output_size, float l1_reg);

/**
 * @brief Peforms the forward pass through the dense layer
 * @param[in] denseLayer - The created dense layer to be used
 * @param[in] input - The 1D array to be passed through
 */
float* dense_forward(Dense* denseLayer, const float* input);

/**
 * @brief Calculates the L1 Regularisation Loss from the Weight matrix of the dense layer
 * @param[in] denseLayer - The dense layer that was created
 */
float L1_Reg_Loss(const Dense* denseLayer);

/**
 * @brief Creates the key parameters for the Adam Optimizer
 * @param[in] lr - Learning Rate
 * @param[in] beta1 - Exponential decay rate of the first moment estimate
 * @param[in] beta2 - Exponential decay rate of the second moment estimate
 * @param[in] epsilon - Ensures no division by zero
 */
Adam_Optimizer create_Adam(float lr, float beta1, float beta2, float epsilon);

/**
 * @brief Performs the Adam Update to get the weights
 * @param[in] optimizer - The adam optimizer created and used here with its parameters defined
 * @param[in] weights - The weights from the dense layer
 * @param[in] grads - The gradients from the dense layer
 * @param[in] rows - The number of rows in the weight matrix
 * @param[in] cols - The number of columns in the weight matrix
 */
void Adam_Update(Adam_Optimizer* optimizer, float** weights, float** grads, int rows, int cols);

/**
 * @brief Calculates the Categorical Cross Entropy Loss from the predicted and actual classification
 * @param[in] predicted - The predicted probability of an image being from each class
 * @param[in] actual - The one-hot encoded true classification of an image
 * @param[in] size - The size of the classification array
 */
float categorical_cross_entropy(const float* predicted, const float* actual, int size);

/**
 * @brief Calculates the accuracy of the model, returns 1 if a correct prediction, 0 otherwise. In training the total number of correct classification is divided by total number to get accuracy
 * @param[in] predicted - The predicted class, the one with the highest prob is set to 1
 * @param[in] actual - The actual class, one-hot encoded
 * @param[in] size - The size of the array put in
 */
float accuracy(const float* predicted, const float* actual, int size);

/**
 * @brief Caluclates the gradient weights and biases from the dense layer
 * @param[in] denseLayer - The layer that was created
 * @param[in] output - The output of the dense layer
 * @param[in] target - The target classification
 * @param[in] grad_weights - Updated here
 * @param[in] grad_biases - Updated here
 */
void calculate_gradients(Dense* denseLayer, const float* output, const float* target, float** grad_weights, float* grad_biases);

/**
 * @brief Creates the 2D kernels or filters to be used in the concolution process
 * @param[in] num_filters - The number of kernels applied
 * @param[in] height - The height of the filter matrix
 * @param[in] width - The width of the filter matrix
 */
Tensor3D create_filters(int num_filters, int height, int width);

/**
 * @brief Frees the high-dimensional matrix when needed
 * @param[in] tensor - The high-dimensional matrix to be freed
 */
void free_Tensor3D(Tensor3D* tensor);

/**
 * @brief Trains the Convolutional Neural Network and prints results and returns the outputs of the dense layer
 * @param[in] denseLayer - The dense layer that is created in the main
 * @param[in] x_train - The image matrices
 * @param[in] y_train - The one-hot encoded classifications of the images
 * @param[in] train_size - The number of images to be trained
 * @param[in] input_size - The size of the given image
 * @param[in] batch_size - The size of batches for batch processing
 * @param[in] epochs - The number of cycles of training
 * @param[in] dropout_rate - The fraction of neurons to be dropped
 * @param[in] filters - The kernels applied in the convolutions
 * @param[in] myid - The processor id
 */
float* train_CNN(Dense* denseLayer, Matrix3D* x_train, float** y_train, int train_size, int batch_size, int epochs, float dropout_rate, Tensor3D* filters, int myid);

/**
 * @brief Trains the Deep/Artificial Neural Network and prints results
 * @param[in] layers - The dense layer(s) that process the output from the CNN
 * @param[in] num_layers - The image matrices
 * @param[in] gathered_data - The output nodes to be processed
 * @param[in] y_train - The one-hot encoded classifications of the images
 * @param[in] train_size - The number of output nodes to be processed
 * @param[in] input_size - The size of the input for each dense layer
 * @param[in] batch_size - The size of batches for batch processing
 * @param[in] epochs - The number of cycles of training
 * @param[in] dropout_rate - The fraction of neurons to be dropped
 * @param[in] myid - The process id
 * @param[in] nprocs - The number of processes
 */
void train_DNN(Dense** layers, int num_layers, float* gathered_data, float** y_train, int train_size, int input_size, int batch_size, int epochs);
#endif