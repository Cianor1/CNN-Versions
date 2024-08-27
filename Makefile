# Compiler
CC = gcc
MPICC = mpicc

# Compiler flags
CFLAGS = -Wall -Wextra

# Libraries
LIBS = -lm

# Source files for each executable
SRC1 = CNN.c Serial_Read.c Serial_Operations.c
SRC2 = MP_CNN.c MP_Operations.c Serial_Read.c
SRC3 = MPI.c MPI_Operations.c
SRC4 = SIMD_CNN.c SIMD_Operations.c Serial_Read.c

# Object files for each executable
CNN_OBJ = $(SRC1:.c=.o)
MP_CNN_OBJ = $(SRC2:.c=.o)
MPI_OBJ = $(SRC3:.c=.o)
SIMD_OBJ = $(SRC4:.c=.o)

# Executables
EXECS = CNN MP_CNN MPI SIMD_CNN

# Compilation flags for each executable
CNN_FLAGS = $(CFLAGS)
MP_CNN_FLAGS = $(CFLAGS) -fopenmp
MPI_FLAGS = $(CFLAGS)
SIMD_CNN_FLAGS = $(CFLAGS) -fopenmp

# Targets
all: $(EXECS)

# Rule to build each executable
CNN: $(CNN_OBJ)
	$(CC) $(CNN_FLAGS) $(CNN_OBJ) -o $@ $(LIBS)

MP_CNN: $(MP_CNN_OBJ)
	$(CC) $(MP_CNN_FLAGS) $(MP_CNN_OBJ) -o $@ $(LIBS) -fopenmp

SIMD_CNN: $(SIMD_OBJ)
	$(CC) $(SIMD_CNN_FLAGS) $(SIMD_OBJ) -o $@ $(LIBS) -fopenmp

MPI: $(MPI_OBJ)
	$(MPICC) $(MPI_FLAGS) $(MPI_OBJ) -o $@ $(LIBS)

# Generic rule to build object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Specific rules for object files that need different compilation flags
Serial_Read.o: Serial_Read.c Serial_Read.h
	$(CC) $(CFLAGS) -c Serial_Read.c -o $@

Serial_Operations.o: Serial_Operations.c Serial_Operations.h
	$(CC) $(CFLAGS) -c Serial_Operations.c -o $@

MP_Operations.o: MP_Operations.c MP_Operations.h
	$(CC) $(MP_CNN_FLAGS) -c MP_Operations.c -o $@

SIMD_Operations.o: SIMD_Operations.c SIMD_Operations.h
	$(CC) $(SIMD_CNN_FLAGS) -c SIMD_Operations.c -o $@

MPI_Operations.o: MPI_Operations.c MPI_Operations.h
	$(CC) $(MPI_FLAGS) -c MPI_Operations.c -o $@

# Clean
clean:
	rm -f $(CNN_OBJ) $(MP_CNN_OBJ) $(MPI_OBJ) $(SIMD_OBJ) $(EXECS)