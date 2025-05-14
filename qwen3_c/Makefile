CC = gcc

all: qwen3

qwen3: qwen3.c
	$(CC) -g -Ofast -fopenmp -march=native qwen3.c -lm  -o qwen3

clean:
	rm -f qwen3

