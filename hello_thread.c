#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <pthread.h>

#define MAX_THREAD 1000

typedef struct {
	int id;
} parm;

void *hello(void *arg)
{
	parm *p=(parm *)arg;
	printf("Hello from node %d\n", p->id);
	return (NULL);
}

void main(int argc, char* argv[]) {
	int n,i;
	pthread_t threads[4];
	parm p[4];

	n=4;

	/* Start up thread */
	for (i=0; i<n; i++)
	{
		p[i].id=i;
		pthread_create(&threads[i], NULL, hello, (void *)&p[i]);
	}

	/* Synchronize the completion of each thread. */

	for (i=0; i<n; i++)
	{
		pthread_join(threads[i],NULL);
	}
}
