#include <pthread.h>

#define NUM_THREADS  4

typedef struct {
        int id;
	int lookFor;
	unsigned int chunk; 
	data_t min_distance; 
	unsigned int located;
	data_t *tempResult;
} parm;


