#include <sys/time.h>
#include <stddef.h>
#include "simpletimer.h"

//struct timespec start_time;

/*
 * Starts the timing. Get a result by calling timer_end afterwards.
 */
void timer_start(struct timeval* stv) 
{
    gettimeofday(stv, NULL);
}

/*
 * Returns the elapsed time since calling timer_start in seconds.
 * Results when calling this without calling timer_end are undefined.
 */
double timer_end(struct timeval stv) 
{
    struct timeval etv;
    double diff;

    gettimeofday(&etv,NULL);
    diff = 1E3*(etv.tv_sec - stv.tv_sec) + 1E-3*(etv.tv_usec-stv.tv_usec);

    return diff;
}


