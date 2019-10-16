CC		:= gcc -mavx2 -mfma
CFLAGS          := -O1 -w -fno-inline-functions -fno-early-inlining -fno-inline-small-functions -fno-tree-loop-optimize -fno-loop-parallelize-all 
#uncomment the following lines if running on acheron 
#CFLAGS		:= -O1 -w -fno-inline-functions -fno-early-inlining -fno-inline-small-functions -fno-tree-loop-optimize
#CC		:= gcc 
LDFLAGS += -lm -lpthread

all: clean k_nearest k_nearest_seq k_nearest_simd k_nearest_simd_book k_nearest_thread
 
k_nearest: k_nearest.c simpletimer.c parse.c 
	$(CC) -o k_nearest k_nearest.c simpletimer.c parse.c  $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

include Makefile.students.mk

clean:
	rm -f *.o k_nearest k_nearest_seq k_nearest_simd k_nearest_simd_book k_nearest_thread 

