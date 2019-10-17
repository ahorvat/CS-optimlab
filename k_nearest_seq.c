#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "simpletimer.h"
#include "parse.h"
#include "vec.h"

typedef data_t (*(*classifying_funct)(unsigned int lookFor, unsigned int* found));

data_t features[ROWS][FEATURE_LENGTH] __attribute__((aligned(32)));
data_t timer_ref_MD,timer_ref_ED,timer_ref_CS;
data_t timer_opt_MD,timer_opt_ED,timer_opt_CS;

data_t abs_diff(data_t x, data_t y){
    data_t diff = x-y;
    return fabs(diff);
}

data_t mult(data_t x,data_t y){
    data_t m = x*y;
    return m;
}

data_t manhattan_distance(data_t *x, data_t *y, int length){
    data_t distance=0;
    int i =0;
    for(i=0;i<length;i++){
        distance+=abs_diff(x[i],y[i]);
    }
    return distance;
}

data_t squared_eucledean_distance(data_t *x,data_t *y, int length){
	data_t distance=0;
	int i = 0;
	for(i=0;i<length;i++){
		distance+= mult(abs_diff(x[i],y[i]),abs_diff(x[i],y[i]));
	}
	return distance;
}

data_t norm(data_t *x, int length){
    data_t n = 0;
    int i=0;
    for (i=0;i<length;i++){
        n += mult(x[i],x[i]);
    }
    n = sqrt(n);
    return n;
}

data_t cosine_similarity(data_t *x, data_t *y, int length){
    data_t sim=0;
    int i=0;
    for(i=0;i<length;i++){
        sim += mult(x[i],y[i]);
    }
    sim = sim / mult(norm(x,FEATURE_LENGTH),norm(y,FEATURE_LENGTH));
    return sim;
}

//Don't touch this function
data_t *ref_classify_ED(unsigned int lookFor, unsigned int *found) {
    data_t *result =(data_t*)malloc(sizeof(data_t)*(ROWS-1));
    struct timeval stv, etv;
    int i,closest_point=0;
    data_t min_distance,current_distance;
	timer_start(&stv);
	min_distance = squared_eucledean_distance(features[lookFor],features[0],FEATURE_LENGTH);
    	result[0] = min_distance;
	for(i=1;i<ROWS-1;i++){
		current_distance = squared_eucledean_distance(features[lookFor],features[i],FEATURE_LENGTH);
        result[i]=current_distance;
		if(current_distance<min_distance){
			min_distance=current_distance;
			closest_point=i;
		}
	}
    timer_ref_ED = timer_end(stv);
    printf("Calculation using reference ED took: %10.6f \n", timer_ref_ED);
    *found = closest_point;
    return result;
}

//Modify this function
data_t *opt_classify_ED(unsigned int lookFor, unsigned int *found) {

    // data_t *result = (data_t*)calloc(sizeof(data_t), (ROWS - 1));
    data_t *result = (data_t*) malloc(sizeof(data_t)*(ROWS-1));

    struct timeval stv, etv;
    int i,closest_point = 0;

	timer_start(&stv);

    //FROM HERE
    data_t min_distance, current_distance0, current_distance1, current_distance2, current_distance3;
    data_t *lookfor, *y_vec0, *y_vec1, *y_vec2, *y_vec3;
    data_t curr_x, curr_y0, curr_y1, curr_y2, curr_y3;
    data_t diff0, diff1, diff2, diff3;

    // remove function, excessive loop and repeated indexing
    min_distance = 100000000000.0;
	// min_distance = squared_eucledean_distance(features[lookFor],features[0],FEATURE_LENGTH);
    // result[0] = min_distance;

    lookfor = features[lookFor];

    int ROW_step_size = 4;

    //  calculate limit of looping
    int limit = ROWS - (ROWS % ROW_step_size);
	for (i = ROW_step_size-1; i < limit; i+= ROW_step_size) {

        // printf("current i: %d/%d (%d, %d, %d, %d)\n", i, ROWS, (i-3), (i-2), (i-1), i);
		// current_distance = opt_squared_eucledean_distance(lookfor, features[i], FEATURE_LENGTH);
        y_vec0 = features[i-3];
        y_vec1 = features[i-2];
        y_vec2 = features[i-1];
        y_vec3 = features[i];

        current_distance0 = 0;
        current_distance1 = 0;
        current_distance2 = 0;
        current_distance3 = 0;

    	for (int j = 0; j < FEATURE_LENGTH; j++) {

            // distance+= mult(abs_diff(x[i],y[i]),abs_diff(x[i],y[i]));
            curr_x = lookfor[j];

            curr_y0 = y_vec0[j];
            curr_y1 = y_vec1[j];
            curr_y2 = y_vec2[j];
            curr_y3 = y_vec3[j];

            diff0 = fabs(curr_x - curr_y0);
            diff1 = fabs(curr_x - curr_y1);
            diff2 = fabs(curr_x - curr_y2);
            diff3 = fabs(curr_x - curr_y3);

            current_distance0 += (diff0 * diff0);
            current_distance1 += (diff1 * diff1);
            current_distance2 += (diff2 * diff2);
            current_distance3 += (diff3 * diff3);
    	}

        // printf("%f\n", current_distance0);

        result[i-3] = current_distance0;
        result[i-2] = current_distance1;
        result[i-1] = current_distance2;
        result[i] = current_distance3;

        // printf("%f\n", current_distance);

		if (current_distance0 < min_distance) {
			min_distance = current_distance0;
			closest_point = i-3;
		}

        if (current_distance1 < min_distance) {
			min_distance = current_distance1;
			closest_point = i-2;
		}

        if (current_distance2 < min_distance) {
			min_distance = current_distance2;
			closest_point = i-1;
		}

        if (current_distance3 < min_distance) {
			min_distance = current_distance3;
			closest_point = i;
		}
	}

    // printf("remainder!\n");

    // remainder loop
	for (i = limit; i < ROWS - 1; i++) {

		// current_distance = opt_squared_eucledean_distance(lookfor, features[i], FEATURE_LENGTH);
        // printf("current i: %d/%d\n", i, ROWS);
        y_vec0 = features[i];

        current_distance0 = 0;

    	for (int j = 0; j < FEATURE_LENGTH; j++) {

            // distance+= mult(abs_diff(x[i],y[i]),abs_diff(x[i],y[i]));
            curr_x = lookfor[j];

            curr_y0 = y_vec0[j];

            diff0 = fabs(curr_x - curr_y0);
            current_distance0 += (diff0 * diff0);

    	}

        result[i] = current_distance0;

        // printf("%f\n", current_distance);

		if (current_distance0 < min_distance) {
			min_distance = current_distance0;
			closest_point = i;
		}
	}
    //TO HERE
    timer_opt_ED = timer_end(stv);
    printf("Calculation using optimized ED took: %10.6f \n", timer_opt_ED);
    printf("min_dist: %d", closest_point);
    *found = closest_point;
    return result;
}

//Don't touch this function
data_t *ref_classify_CS(unsigned int lookFor, unsigned int* found) {
    data_t *result =(data_t*)malloc(sizeof(data_t)*(ROWS-1));
    struct timeval stv, etv;
    int i,closest_point=0;
    data_t min_distance,current_distance;

	timer_start(&stv);
	min_distance = cosine_similarity(features[lookFor],features[0],FEATURE_LENGTH);
    	result[0] = min_distance;
	for(i=1;i<ROWS-1;i++){
		current_distance = cosine_similarity(features[lookFor],features[i],FEATURE_LENGTH);
        	result[i]=current_distance;
		if(current_distance>min_distance){
			min_distance=current_distance;
			closest_point=i;
		}
	}
    timer_ref_CS = timer_end(stv);
    printf("Calculation using reference CS took: %10.6f \n", timer_ref_CS);
    *found = closest_point;
    return result;
}

//Modify this function
data_t *opt_classify_CS(unsigned int lookFor, unsigned int *found) {
    data_t *result =(data_t*)malloc(sizeof(data_t)*(ROWS-1));
    struct timeval stv, etv;
    int i,closest_point=0;

    timer_start(&stv);

    //MODIFY FROM HERE
    data_t min_distance;
    data_t *lookfor, *y_vec0, *y_vec1, *y_vec2, *y_vec3;
    data_t curr_x, curr_y0, curr_y1, curr_y2, curr_y3;
    data_t norm_x, norm_y0, norm_y1, norm_y2, norm_y3;
    data_t sim0, sim1, sim2, sim3;

    // remove function, excessive loop and repeated indexing
    min_distance = 0.000000000000000001;

	// min_distance = squared_eucledean_distance(features[lookFor],features[0],FEATURE_LENGTH);
    // result[0] = min_distance;

    lookfor = features[lookFor];

    int ROW_step_size = 4;

    // calculate norm_x only once.
    norm_x = 0;

    for (int i = 0; i < FEATURE_LENGTH; i++) {
        curr_x = lookfor[i];
        norm_x += (curr_x * curr_x);
    }

    norm_x = sqrt(norm_x);

    // calculate limit of looping
    int limit = ROWS - (ROWS % ROW_step_size);
	for (i = ROW_step_size-1; i < limit; i+= ROW_step_size) {
        // current_distance = cosine_similarity(features[lookFor],features[i],FEATURE_LENGTH);

        y_vec0 = features[i-3];
        y_vec1 = features[i-2];
        y_vec2 = features[i-1];
        y_vec3 = features[i];

        sim0 = 0;
        sim1 = 0;
        sim2 = 0;
        sim3 = 0;
        norm_y0 = 0;
        norm_y1 = 0;
        norm_y2 = 0;
        norm_y3 = 0;

    	for (int j = 0; j < FEATURE_LENGTH; j++) {

            curr_x = lookfor[j];

            curr_y0 = y_vec0[j];
            curr_y1 = y_vec1[j];
            curr_y2 = y_vec2[j];
            curr_y3 = y_vec3[j];

            sim0 += curr_x * curr_y0;
            sim1 += curr_x * curr_y1;
            sim2 += curr_x * curr_y2;
            sim3 += curr_x * curr_y3;

            norm_y0 += curr_y0 * curr_y0;
            norm_y1 += curr_y1 * curr_y1;
            norm_y2 += curr_y2 * curr_y2;
            norm_y3 += curr_y3 * curr_y3;
        }

        // sim = sim / mult(norm( x, FEATURE_LENGTH),norm(y, FEATURE_LENGTH));
        sim0 = sim0 / (norm_x * sqrt(norm_y0));
        sim1 = sim1 / (norm_x * sqrt(norm_y1));
        sim2 = sim2 / (norm_x * sqrt(norm_y2));
        sim3 = sim3 / (norm_x * sqrt(norm_y3));

        result[i-3] = sim0;
        result[i-2] = sim1;
        result[i-1] = sim2;
        result[i] = sim3;

        if (sim0 > min_distance) {
            min_distance = sim0;
            closest_point = i-3;
        }

        if (sim1 > min_distance) {
            min_distance = sim1;
            closest_point = i-2;
        }

        if (sim2 > min_distance) {
            min_distance = sim2;
            closest_point = i-1;
        }

        if (sim3 > min_distance) {
            min_distance = sim3;
            closest_point = i;
        }
    }

    // int limit = 0;
    // remainder loop
	for (i = limit; i < ROWS - 1; i++) {
        // current_distance = cosine_similarity(features[lookFor],features[i],FEATURE_LENGTH);

        y_vec0 = features[i];

        sim0 = 0;
        norm_y0 = 0;

    	for (int j = 0; j < FEATURE_LENGTH; j++) {

            curr_x = lookfor[j];

            curr_y0 = y_vec0[j];

            sim0 += curr_x * curr_y0;

            norm_y0 += curr_y0 * curr_y0;
    	}

        sim0 = sim0 / (norm_x * sqrt(norm_y0));

        result[i] = sim0;

		if (sim0 > min_distance) {
			min_distance = sim0;
			closest_point = i;
		}
	}

    //TO HERE
    timer_opt_CS = timer_end(stv);
    printf("Calculation using optimized CS took: %10.6f \n", timer_opt_CS);
    *found = closest_point;
    return result;
}

int check_correctness(classifying_funct a, classifying_funct b, unsigned int lookFor, unsigned int *found) {
    unsigned int i, a_found, b_found;
    data_t *a_res = a(lookFor, &a_found);
    data_t *b_res = b(lookFor, &b_found);

    // changed for allowing for pertubations
    data_t epsilon = 0.00001;

    for(i = 0; i < ROWS - 1; i++){


        if (fabs(a_res[i] - b_res[i]) > epsilon) {
            printf("ref= %f, opt= %f\n", a_res[i], b_res[i]);
            // returning 0 means = wrong
            return 0;
        }
    }

    if (fabs(a_found - b_found) > epsilon) {
        printf("Found Result is wrong --> ref: %f, opt= %f\n", a_found, b_found);
        return 0;
    }


    *found=a_found;

    return 1;
}

data_t *ref_classify_MD(unsigned int lookFor, unsigned int *found) {
    data_t *result =(data_t*)malloc(sizeof(data_t)*(ROWS-1));
    struct timeval stv, etv;
    int i,closest_point=0;
    data_t min_distance,current_distance;

	timer_start(&stv);
	min_distance = manhattan_distance(features[lookFor],features[0],FEATURE_LENGTH);
    	result[0] = min_distance;
	for(i=1;i<ROWS-1;i++){
		current_distance = manhattan_distance(features[lookFor],features[i],FEATURE_LENGTH);
        	result[i]=current_distance;
		if(current_distance<min_distance){
			min_distance=current_distance;
			closest_point=i;
		}
	}
    timer_ref_MD = timer_end(stv);
    printf("Calculation using reference MD took: %10.6f \n", timer_ref_MD);
    *found=closest_point;
    return result;
}


//NO NEED to modify this function!
data_t *opt_classify_MD(unsigned int lookFor, unsigned int *found) {
    data_t *result =(data_t*)malloc(sizeof(data_t)*(ROWS-1));
    struct timeval stv, etv;
    int i,closest_point=0;
    data_t min_distance,current_distance;

        timer_start(&stv);
        min_distance = manhattan_distance(features[lookFor],features[0],FEATURE_LENGTH);
    	result[0] = min_distance;
        for(i=1;i<ROWS-1;i++){
                current_distance = manhattan_distance(features[lookFor],features[i],FEATURE_LENGTH);
                result[i]=current_distance;
                if(current_distance<min_distance){
                        min_distance=current_distance;
                        closest_point=i;
                }
        }
    timer_opt_MD = timer_end(stv);
    printf("Calculation using optimized MD took: %10.6f \n", timer_opt_MD);
    *found = closest_point;
    return result;
}

int main(int argc, char **argv){
	char* dataset_name=DATASET;
	int i,j;
        struct timeval stv, etv;
	unsigned int lookFor=ROWS-1, located;
	//PARSE CSV

	//holds the information regarding author and title
	char metadata[ROWS][2][20];

	timer_start(&stv);
	parse_csv(dataset_name, features, metadata);
	printf("Parsing took %9.6f s \n\n", timer_end(stv));

    printf("Classifying using MD:");
    printf("<Record %d, author =\"%s\", title=\"%s\">\n",lookFor,metadata[lookFor][0],metadata[lookFor][1]);
    if(check_correctness(ref_classify_MD,opt_classify_MD, lookFor, &located)){
        printf("opt_classify_MD is correct, speedup: %10.6f\n\n",timer_ref_MD/timer_opt_MD);
    }
    else
        printf("opt_classify_MD is incorrect! \n"); // , speedup: %10.6f\n\n",timer_ref_MD/timer_opt_MD);
    printf("Best match: ");
    printf("<Record %d, author =\"%s\", title=\"%s\">\n\n",located,metadata[located][0],metadata[located][1]);

    printf("Classifying using ED:");
    printf("<Record %d, author =\"%s\", title=\"%s\">\n",lookFor,metadata[lookFor][0],metadata[lookFor][1]);
    if(check_correctness(ref_classify_ED,opt_classify_ED, lookFor, &located)) {
        printf("opt_classify_ED is correct, speedup: %10.6f\n\n",timer_ref_ED/timer_opt_ED);
    }
    else
        printf("opt_classify_ED id incorrect!\n\n");
    printf("Best match: ");
    printf("<Record %d, author =\"%s\", title=\"%s\">\n\n",located,metadata[located][0],metadata[located][1]);

    printf("Classifying using CS (cosine similarity):");
    printf("<Record %d, author =\"%s\", title=\"%s\">\n",lookFor,metadata[lookFor][0],metadata[lookFor][1]);
    if(check_correctness(ref_classify_CS,opt_classify_CS, lookFor, &located)) {
        printf("opt_classify_CS is correct, speedup: %10.6f\n\n",timer_ref_CS/timer_opt_CS);
    }
    else
        printf("opt_classify_CS id incorrect!\n\n");
    printf("Best match: ");
    printf("<Record %d, author =\"%s\", title=\"%s\">\n\n",located,metadata[located][0],metadata[located][1]);

}
