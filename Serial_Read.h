#ifndef READ_BIN
#define READ_BIN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "structs.h"

#define IMAGE_SIZE 3072 /*32x32x3*/
#define LABEL_SIZE 1

Image* read_Cifar10(const char* filename, int* numImages);

Image_Dataset read_Cifar10_Multi(const char** filenames, int numFiles, int subsetSize);

Class_Names read_class_names(const char* filename);

void freeClassNames(Class_Names* classNames);

#endif
