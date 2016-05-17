CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
all: Word2vec
Word2vec: word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
clean:
	rm -rf word2vec