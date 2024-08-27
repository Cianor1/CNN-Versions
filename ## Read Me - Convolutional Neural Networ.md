## Read Me - Convolutional Neural Network Versions in C

# Description
These functions all carry out some form of neural network in C. Batch Processing is used with Adam Optimiser to update the weights and biases. structs.h contains most of the necessary structs for this project.

# CNN.c
- Carries out a CNN in serial
- All relvant reading functions are found in Serial_Read
- All other relevant Operations are found in Serial_Operations
- To execute simply enter ./CNN

# MP_CNN.c
- Carries out a CNN in parallel using OpenMP clauses
- The read functions are again in Serial_Read
- The rest of the Operations are found in MP_Operations
- To execute, enter ./MP_CNN

# SIMD_CNN.c
- Carries out a CNN in parallel using OpenMP clauses with SIMD applied where relevant
- The read functions are again in Serial_Read
- The rest of the Operations are found in SIMD_Operations
- To execute, enter ./SIMD_CNN

# MPI.c
- Carries out a CNN-DNN architecture in parallel using MPI/IO
- All operations are found in MPI_Operations
- To execute, enter mpirun -np 4 ./MPI
- Will not run on any other set of processors

# Usage
- To compile, type 'make' into the terminal.
- To execute CNN.c, type './CNN' into the terminal
- To execute MP_CNN.c, type './MP_CNN' into the terminal
- To execute SIMD_CNN.c, type './SIMD_CNN' into the terminal
- To execute MPI.c, type mpirun -np 4 ./MPI into the terminal, note, this function includes the requirement for nprocs to be 4 otherwise an error will be returned.