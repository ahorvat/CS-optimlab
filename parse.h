/*
 * parse.h
 */

#include "data.h"
//dataset 5 ext repeated 
#define FEATURE_LENGTH	5700
#define ROWS 463

//dataset 3
//#define FEATURE_LENGTH  1988	
//#define ROWS		7

//typedef double data_t;

#define DATASET	"dataset_5_ext_repeated.csv"

//#define DATASET	"dataset_3.csv"

int parse_csv(char* dataset_name, data_t features[ROWS][FEATURE_LENGTH], char metadata[ROWS][2][20]);

