#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "Serial_Read.h"

#define IMAGE_SIZE 3072 /*32x32x3*/
#define LABEL_SIZE 1

Image* read_Cifar10(const char* filename, int* num_images){
    FILE* file = fopen(filename, "rb"); /*"rb" ("Read Binary") required when reading a binary file*/
    if(!file){
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END); /*Set the file pointer to the end of the file*/
    long file_size = ftell(file); /*Use the location of the file pointer to get the file size*/
    fseek(file, 0, SEEK_SET); /*Resets the local of the file pointer to the start of the file*/

    *num_images = file_size / (LABEL_SIZE + IMAGE_SIZE); /*Get the number of images*/
    Image* images = (Image*)malloc(*num_images * sizeof(Image)); /*Allocates memory for an array of image structures*/
    if(!images){
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(file);
        return NULL;
    }
    /*Read each image in the file, fread is used as it is useful for binary reads*/
    for(int i = 0; i < *num_images; ++i){
        fread(&images[i].label, LABEL_SIZE, 1, file); /*Read label*/
        fread(images[i].pixels, IMAGE_SIZE, 1, file); /*Read actual image pixels*/
    }

    fclose(file);
    return images;
}

Image_Dataset read_Cifar10_Multi(const char** filenames, int num_files, int subset_size){
    Image_Dataset dataset; /*New structure to hold the images from the various image files*/
    dataset.count = 0;
    dataset.images = NULL; /*Initialise as a NULL (just good practice)*/

    for(int i = 0; i < num_files; ++i){/*Iterate through the image files*/
        int numImages;
        Image* images = read_Cifar10(filenames[i], &numImages); /*Each image structure which will make up the imagedataset structure*/

        if(images){/*Read is successful*/
            dataset.images = (Image*)realloc(dataset.images, (dataset.count + numImages) * sizeof(Image)); /*Now allocate memory so the dataset will hold all the images from the different image structures*/
            memcpy(&dataset.images[dataset.count], images, numImages * sizeof(Image)); /*Copy these image structs in*/
            dataset.count += numImages; /*Update the count*/
            free(images);
        }
    }

    /*Shuffling and subsetting*/
    if(subset_size < dataset.count){
        srand(time(NULL));
        for(int i = dataset.count - 1; i > 0; --i){
            int j = rand() % (i + 1); /*Gets a random index to perform the shuffle*/
            /*Essentially swaps current image with a random*/
            Image temp = dataset.images[i];
            dataset.images[i] = dataset.images[j];
            dataset.images[j] = temp;
        }
        /*Then Resize to the subset size*/
        dataset.count = subset_size;
        dataset.images = (Image*)realloc(dataset.images, subset_size * sizeof(Image));
    }

    return dataset;
}

Class_Names read_class_names(const char* filename){
    Class_Names class_names; /*Initialise the class name structure to hold class names*/
    class_names.class_names = NULL;
    class_names.count = 0;

    FILE* file = fopen(filename, "r"); /*Plain read as it is not a binary file*/
    if(!file){
        fprintf(stderr, "Error opening file: %s\n", filename);
        return class_names;
    }

    char line[100]; /*Buffer to hold each line read from the file, not actually needed to be 100, but again, good practice*/

    while(fgets(line, sizeof(line), file)){ /*While reading each line of the file*/
        class_names.class_names = (char**)realloc(class_names.class_names, (class_names.count + 1) * sizeof(char*)); /*Add space for a new class name on the next line*/
        class_names.class_names[class_names.count] = strdup(line); /*Copy the name and store it in the class_names array*/
        class_names.count++; /*Increment the count*/
    }

    fclose(file);
    return class_names;
}

void freeClassNames(Class_Names* class_names) {
    for (int i = 0; i < class_names->count; ++i) {
        free(class_names->class_names[i]); /*Free each individual class name*/
    }
    free(class_names->class_names); /*Free the space used for the names*/
}
