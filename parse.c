#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "parse.h"

int parse_csv(char* dataset_name, data_t features[ROWS][FEATURE_LENGTH], char metadata[ROWS][2][20]) {
 char *line;
 char *record;
 char buffer[1000000];
 int row_nb=0, col_nb=0;
 int i,j;
 FILE *dataset;


   printf(" Parsing %s ... \n ", dataset_name);
   dataset = fopen(dataset_name,"r");
   if(dataset==NULL){
        printf("Error opening file %s\n",dataset_name);
        return -1;
   }

   //skip first line with columns title
   line=fgets(buffer,sizeof(buffer),dataset);
   while((line=fgets(buffer,sizeof(buffer),dataset))!=NULL){
         record=strtok(line,",");
         //store author data
         strcpy(metadata[row_nb][0],record);
         //store title data
         record=strtok(NULL,",");
         strcpy(metadata[row_nb][1],record);
         record=strtok(NULL,",");
         col_nb=0;
         while(record!=NULL)
	 {
	      data_t tmp = strtod(record,NULL); 
              features[row_nb][col_nb]=tmp;
              col_nb++;
              record=strtok(NULL,",");
         }
         row_nb++;
   }
   printf("%d %d \n", row_nb+1, col_nb);
   fclose(dataset);
}
