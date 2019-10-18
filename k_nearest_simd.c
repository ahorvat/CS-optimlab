#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <math.h>
#include <sys/time.h>
#include "simpletimer.h"
#include "parse.h"
#include "vec.h"

#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
/* Number of bytes in a vector */
/* Check the extensions of your machine to decide!
 * E.g. SSE4 = 128 bits, AVX = 256 bits*/

#define VBYTES 16    // DAS4 = SSE4.2 = 128 bits

/* Number of elements in a vector */
#define VSIZE VBYTES/sizeof(data_t)

/* Vector data type */
typedef data_t (*(*classifying_funct)(unsigned int lookFor, unsigned int* found));
typedef data_t vec_t __attribute__ ((vector_size(VBYTES)));

typedef union {
    vec_t v;
    data_t d[VSIZE];
} pack_t;

data_t features[ROWS][FEATURE_LENGTH] __attribute__((aligned(32)));
data_t timer_ref_MD,timer_ref_ED,timer_ref_CS;
data_t timer_opt_MD,timer_opt_ED,timer_opt_CS;

data_t  abs_diff(data_t x, data_t y){
    data_t diff = x-y;
    return fabs(diff);
}

vec_t simd_abs_diff(vec_t x, vec_t y) {
    pack_t temp;
    int i;

    temp.v = x-y;
    for (i = 0; i < VSIZE; i++){
      temp.d[i] = fabs(temp.d[i]);
    }

    return temp.v;
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

inline __m128d abs_pd(__m128d x) {
             __m128d sign_mask = _mm_set1_pd(-0.); // -0. = 1 << 63
            return _mm_andnot_pd(sign_mask, x); // !sign_mask & x
}

inline __m256d abs256_pd(__m256d x) {
            __m256d sign_mask = _mm256_set1_pd(-0.);
            return _mm256_andnot_pd(sign_mask,x);
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
    data_t *result =(data_t*)malloc(sizeof(data_t)*(ROWS-1));
    struct timeval stv, etv;
    int i,closest_point=0;
    data_t min_distance, current_distance0, current_distance1, current_distance2, current_distance3;

	timer_start(&stv);

    //FROM HERE
    data_t *lookfor, *y_vec0, *y_vec1, *y_vec2, *y_vec3;
    __m256d simd_x, simd_y0, simd_y1, simd_y2, simd_y3;
    __m256d diff0, diff1, diff2, diff3;
    __m256d simd_distance0, simd_distance1, simd_distance2, simd_distance3;
    data_t* v_distance0, *v_distance1, *v_distance2, *v_distance3;

    min_distance = 100000000000.0;

    lookfor = features[lookFor];

    int ROW_step_size = 4;
    int mm256_size = 4;

    __m256d zero = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

    //  calculate limit of looping
    int limit = ROWS - (ROWS % ROW_step_size);
	for (i = (ROW_step_size - 1); i < limit; i += ROW_step_size) {

        y_vec0 = features[i-3];
        y_vec1 = features[i-2];
        y_vec2 = features[i-1];
        y_vec3 = features[i];

        simd_distance0 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_distance1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_distance2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_distance3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

        // current_distance = squared_eucledean_distance(lookfor,features[i],FEATURE_LENGTH);

        for (int j = 0; j < FEATURE_LENGTH; j+= mm256_size) {

            // distance+= mult(abs_diff(x[i],y[i]),abs_diff(x[i],y[i]));

            simd_x = _mm256_load_pd(lookfor + j);
            simd_y0 = _mm256_load_pd(y_vec0 + j);
            simd_y1 = _mm256_load_pd(y_vec1 + j);
            simd_y2 = _mm256_load_pd(y_vec2 + j);
            simd_y3 = _mm256_load_pd(y_vec3 + j);

            diff0 = abs256_pd(_mm256_sub_pd(simd_x, simd_y0));
            diff1 = abs256_pd(_mm256_sub_pd(simd_x, simd_y1));
            diff2 = abs256_pd(_mm256_sub_pd(simd_x, simd_y2));
            diff3 = abs256_pd(_mm256_sub_pd(simd_x, simd_y3));

            simd_distance0 = _mm256_add_pd(simd_distance0, _mm256_mul_pd(diff0, diff0));
            simd_distance1 = _mm256_add_pd(simd_distance1, _mm256_mul_pd(diff1, diff1));
            simd_distance2 = _mm256_add_pd(simd_distance2, _mm256_mul_pd(diff2, diff2));
            simd_distance3 = _mm256_add_pd(simd_distance3, _mm256_mul_pd(diff3, diff3));
    	}

        simd_distance0 = _mm256_hadd_pd(simd_distance0, zero); // do some computations
        simd_distance1 = _mm256_hadd_pd(simd_distance1, zero);
        simd_distance2 = _mm256_hadd_pd(simd_distance2, zero);
        simd_distance3 = _mm256_hadd_pd(simd_distance3, zero);

        // convert to floats
        v_distance0 = (data_t*) &simd_distance0;
        current_distance0 = v_distance0[0] + v_distance0[2];

        v_distance1 = (double*) &simd_distance1;
        current_distance1 = v_distance1[0]+v_distance1[2];

        v_distance2 = (double*) &simd_distance2;
        current_distance2 = v_distance2[0] + v_distance2[2];

        v_distance3 = (double*) &simd_distance3;
        current_distance3 = v_distance3[0] + v_distance3[2];

        // store result

        result[i-3] = current_distance0;
        result[i-2] = current_distance1;
        result[i-1] = current_distance2;
        result[i] = current_distance3;

        // printf("dist: %f\n", current_distance2);
        // printf("dist: %f\n", current_distance3);


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

    // remainder loop
    for (i = limit; i < ROWS - 1; i++) {
        // current_distance = squared_eucledean_distance(lookfor,features[i],FEATURE_LENGTH);

        y_vec0 = features[i];

        simd_distance0 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

        // current_distance = squared_eucledean_distance(lookfor,features[i],FEATURE_LENGTH);

        for (int j = 0; j < FEATURE_LENGTH; j += mm256_size) {

            simd_x = _mm256_load_pd(lookfor + j);
            simd_y0 = _mm256_load_pd(y_vec0 + j);

            diff0 = abs256_pd(_mm256_sub_pd(simd_x, simd_y0));

            simd_distance0 = _mm256_add_pd(simd_distance0, _mm256_mul_pd(diff0, diff0));
    	}

        // do some computations
        simd_distance0 = _mm256_hadd_pd(simd_distance0, zero);

        // convert to floats
        v_distance0 = (data_t*) &simd_distance0;
        current_distance0 = v_distance0[0] + v_distance0[2];

        // store result
        result[i] = current_distance0;

		if (current_distance0 < min_distance) {
			min_distance = current_distance0;
			closest_point = i;
		}
    }

    //TO HERE
    timer_opt_ED = timer_end(stv);
    printf("Calculation using optimized ED took: %10.6f \n", timer_opt_ED);
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
    data_t min_distance,current_distance;

    timer_start(&stv);

    //MODIFY FROM HERE
    data_t *lookfor, *y_vec0, *y_vec1, *y_vec2, *y_vec3;
    __m256d simd_norm_x, simd_norm_y0, simd_norm_y1, simd_norm_y2, simd_norm_y3;
    __m256d simd_sim0, simd_sim1, simd_sim2, simd_sim3;
    __m256d simd_x, simd_y0, simd_y1, simd_y2, simd_y3;
    __m256d simd_norm_xy0, simd_norm_xy1, simd_norm_xy2, simd_norm_xy3;

    data_t *norm_vec_x, *norm_vec_y0, *norm_vec_y1, *norm_vec_y2, *norm_vec_y3;
    data_t *vec_sim0, *vec_sim1, *vec_sim2, *vec_sim3;
    data_t norm_x, norm_y0, norm_y1, norm_y2, norm_y3;
    data_t sim0, sim1, sim2, sim3;


    // small value to start comparing with
    min_distance = 0.000000000000000001;

    lookfor = features[lookFor]; // avoid repeated indexing

    // calculate norm_x only once (and not repeatedly).
    simd_norm_x = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d zero = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

    int ROW_step_size = 4;
    int mm256_size = 4;

    // calculate norm x only once.
    for (int i = 0; i < FEATURE_LENGTH; i += mm256_size) {

        simd_x = _mm256_load_pd(lookfor + i);
        simd_norm_x = _mm256_add_pd(simd_norm_x, _mm256_mul_pd(simd_x, simd_x));
    }

    simd_norm_x = _mm256_hadd_pd(simd_norm_x, zero);
    // simd_norm_x = _mm256_sqrt_pd(simd_norm_x);

    //  // do some computations
    norm_vec_x = (data_t*) &simd_norm_x;
    norm_x = norm_vec_x[0] + norm_vec_x[2];
    norm_x = sqrt(norm_x);

    // calculate limit of looping
    int limit = ROWS - (ROWS % ROW_step_size);

	for (i = ROW_step_size-1; i < limit; i+= ROW_step_size) {
        // current_distance = cosine_similarity(features[lookFor],features[i],FEATURE_LENGTH);

        y_vec0 = features[i-3];
        y_vec1 = features[i-2];
        y_vec2 = features[i-1];
        y_vec3 = features[i];

        simd_sim0 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_sim1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_sim2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_sim3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

        simd_norm_y0 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_norm_y1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_norm_y2 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_norm_y3 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

    	for (int j = 0; j < FEATURE_LENGTH; j+= mm256_size) {

            simd_x = _mm256_load_pd(lookfor + j);
            simd_y0 = _mm256_load_pd(y_vec0 + j);
            simd_y1 = _mm256_load_pd(y_vec1 + j);
            simd_y2 = _mm256_load_pd(y_vec2 + j);
            simd_y3 = _mm256_load_pd(y_vec3 + j);

            simd_sim0 = _mm256_add_pd(simd_sim0, _mm256_mul_pd(simd_x, simd_y0));
            simd_sim1 = _mm256_add_pd(simd_sim1, _mm256_mul_pd(simd_x, simd_y1));
            simd_sim2 = _mm256_add_pd(simd_sim2, _mm256_mul_pd(simd_x, simd_y2));
            simd_sim3 = _mm256_add_pd(simd_sim3, _mm256_mul_pd(simd_x, simd_y3));

            simd_norm_y0 = _mm256_add_pd(simd_norm_y0, _mm256_mul_pd(simd_y0, simd_y0));
            simd_norm_y1 = _mm256_add_pd(simd_norm_y1, _mm256_mul_pd(simd_y1, simd_y1));
            simd_norm_y2 = _mm256_add_pd(simd_norm_y2, _mm256_mul_pd(simd_y2, simd_y2));
            simd_norm_y3 = _mm256_add_pd(simd_norm_y3, _mm256_mul_pd(simd_y3, simd_y3));

        }

        // convert simd back to normal data_t.
        // see below:
        simd_norm_y0 = _mm256_hadd_pd(simd_norm_y0, zero); // do some computations
        norm_vec_y0 = (data_t*) &simd_norm_y0;
        norm_y0 = sqrt(norm_vec_y0[0] + norm_vec_y0[2]);

        simd_norm_y1 = _mm256_hadd_pd(simd_norm_y1, zero); // do some computations
        norm_vec_y1 = (data_t*) &simd_norm_y1;
        norm_y1 = sqrt(norm_vec_y1[0] + norm_vec_y1[2]);

        simd_norm_y2 = _mm256_hadd_pd(simd_norm_y2, zero); // do some computations
        norm_vec_y2 = (data_t*) &simd_norm_y2;
        norm_y2 = sqrt(norm_vec_y2[0] + norm_vec_y2[2]);

        simd_norm_y3 = _mm256_hadd_pd(simd_norm_y3, zero); // do some computations
        norm_vec_y3 = (data_t*) &simd_norm_y3;
        norm_y3 = sqrt(norm_vec_y3[0] + norm_vec_y3[2]);

        simd_sim0 = _mm256_hadd_pd(simd_sim0, zero); // do some computations
        vec_sim0 = (data_t*) &simd_sim0;
        sim0 = vec_sim0[0] + vec_sim0[2];

        simd_sim1 = _mm256_hadd_pd(simd_sim1, zero); // do some computations
        vec_sim1 = (data_t*) &simd_sim1;
        sim1 = vec_sim1[0] + vec_sim1[2];

        simd_sim2 = _mm256_hadd_pd(simd_sim2, zero); // do some computations
        vec_sim2 = (data_t*) &simd_sim2;
        sim2 = vec_sim2[0] + vec_sim2[2];

        simd_sim3 = _mm256_hadd_pd(simd_sim3, zero); // do some computations
        vec_sim3 = (data_t*) &simd_sim3;
        sim3 = vec_sim3[0] + vec_sim3[2];

        // sim = sim / mult(norm( x, FEATURE_LENGTH),norm(y, FEATURE_LENGTH));
        sim0 = sim0 / (norm_x * norm_y0);
        sim1 = sim1 / (norm_x * norm_y1);
        sim2 = sim2 / (norm_x * norm_y2);
        sim3 = sim3 / (norm_x * norm_y3);

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

    // remainder loop
    for (i = limit; i < ROWS - 1; i++) {
        // current_distance = cosine_similarity(features[lookFor],features[i],FEATURE_LENGTH);

        y_vec0 = features[i];
        simd_sim0 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
        simd_norm_y0 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

    	for (int j = 0; j < FEATURE_LENGTH; j+= mm256_size) {

            simd_x = _mm256_load_pd(lookfor + j);
            simd_y0 = _mm256_load_pd(y_vec0 + j);

            simd_sim0 = _mm256_add_pd(simd_sim0, _mm256_mul_pd(simd_x, simd_y0));

            simd_norm_y0 = _mm256_add_pd(simd_norm_y0, _mm256_mul_pd(simd_y0, simd_y0));

        }

        // convert simd back to normal data_t.
        // see below:
        simd_norm_y0 = _mm256_hadd_pd(simd_norm_y0, zero); // do some computations
        norm_vec_y0 = (data_t*) &simd_norm_y0;
        norm_y0 = sqrt(norm_vec_y0[0] + norm_vec_y0[2]);

        simd_sim0 = _mm256_hadd_pd(simd_sim0, zero); // do some computations
        vec_sim0 = (data_t*) &simd_sim0;
        sim0 = vec_sim0[0] + vec_sim0[2];

        // sim = sim / mult(norm(x,FEATURE_LENGTH),norm(y,FEATURE_LENGTH));
        sim0 = sim0 / (norm_x * norm_y0);

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

data_t simd_manhattan_distance_intr(data_t *x, data_t *y, int length){
    int i =0;
    data_t result=0;
    __m128d vx,vy,sub,abs_diff;
    __m128d distance=_mm_set_pd(0.0,0.0);
    __m128d zero= _mm_set_pd(0.0,0.0);
    for(i=0;i<length;i+=VSIZE){
         vx = _mm_load_pd(x+i);
         vy = _mm_load_pd(y+i);
         sub = _mm_sub_pd(vx,vy);
         abs_diff= abs_pd(sub);
         distance =_mm_add_pd(distance,abs_diff);

    }
    distance = _mm_hadd_pd(distance, zero);
    result = _mm_cvtsd_f64(distance);
    while (i < length) {
        result += fabs(*(x+i) - *(y+i));
        i++;
    }
	return result;
}

data_t simd_avx2_manhattan_distance_intr(data_t *x, data_t *y, int length){
    int i=0;
    data_t result=0;
    __m256d vx,vy,sub,abs_diff;
    __m256d distance=_mm256_set_pd(0.0, 0.0, 0.0, 0.0);
    __m256d zero= _mm256_set_pd(0.0,0.0,0.0,0.0);

    for(i=0;i<length;i+=4){
         vx = _mm256_load_pd(x+i);
         vy = _mm256_load_pd(y+i);
         sub = _mm256_sub_pd(vx,vy);
         abs_diff= abs256_pd(sub);
         distance=_mm256_add_pd(distance,abs_diff);
    }
    distance = _mm256_hadd_pd(distance,zero);
    double *v_distance = (double*) &distance;
    result = v_distance[0]+v_distance[2];

    while (i < length) {
        printf("this is used?\n");
        result += fabs(*(x+i) - *(y+i));
        i++;
    }
	return result;
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
    int i,closest_point = 0;
    data_t min_distance,current_distance;

        timer_start(&stv);
        //min_distance = simd_manhattan_distance_intr(features[lookFor],features[0],FEATURE_LENGTH);
        min_distance = simd_avx2_manhattan_distance_intr(features[lookFor],features[0],FEATURE_LENGTH);
    	result[0] = min_distance;
        for(i=1;i<ROWS-1;i++){
                //current_distance =simd_manhattan_distance_intr(features[lookFor],features[i],FEATURE_LENGTH);
                current_distance = simd_avx2_manhattan_distance_intr(features[lookFor],features[i],FEATURE_LENGTH);
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

int check_correctness(classifying_funct a, classifying_funct b, unsigned int lookFor, unsigned int *found) {
    unsigned int r=1, i, a_found, b_found;
    data_t *a_res = a(lookFor, &a_found);
    data_t *b_res = b(lookFor, &b_found);

    // changed for allowing for pertubations
    data_t epsilon = 0.00001;

    for(i = 0; i < ROWS - 1; i++){
        // printf("vector number = %d, reference value = %f, optimized output = %f\n", i, a_res[i], b_res[i]);

        if (fabs(a_res[i] - b_res[i]) > epsilon) {
            printf("WRONG! vector number = %d, reference value = %f, optimized output = %f\n", i, a_res[i], b_res[i]);
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
