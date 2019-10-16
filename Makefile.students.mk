CFLAGS_seq		:= -O3 -w -fno-inline-functions -fno-early-inlining -fno-inline-small-functions -fno-tree-loop-optimize -mavx2
CFLAGS_simd		:= -O3 -w -fno-inline-functions -fno-early-inlining -fno-inline-small-functions -fno-tree-loop-optimize -mavx2
CFLAGS_thread	:= -O3 -w -fno-inline-functions -fno-early-inlining -fno-inline-small-functions -fno-tree-loop-optimize -mavx2


k_nearest_seq: k_nearest_seq.c simpletimer.c parse.c
	$(CC) -o k_nearest_seq k_nearest_seq.c simpletimer.c parse.c $(INCLUDES) $(LIBS) $(CFLAGS_seq) $(LDFLAGS)

k_nearest_simd: k_nearest_simd.c simpletimer.c parse.c vec.c
	$(CC) -o k_nearest_simd k_nearest_simd.c simpletimer.c parse.c vec.c $(INCLUDES) $(LIBS) $(CFLAGS_simd) $(LDFLAGS)

k_nearest_simd_book: k_nearest_simd_book.c simpletimer.c parse.c vec.c
	$(CC) -o k_nearest_simd_book k_nearest_simd_book.c simpletimer.c parse.c vec.c $(INCLUDES) $(LIBS) $(CFLAGS_simd) $(LDFLAGS)

k_nearest_thread: k_nearest_thread.c simpletimer.c parse.c vec.c
	$(CC) -o k_nearest_thread k_nearest_thread.c simpletimer.c parse.c vec.c $(INCLUDES) $(LIBS) $(CFLAGS_thread) $(LDFLAGS)
