# 变量定义
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp

.PHONY: all
all: infer.c
	$(CC) $(CFLAGS) -o infer infer.c -lm
	$(CC) $(CFLAGS) -o infer_q infer_q.c -lm

# 清理
.PHONY: clean
clean:
	rm -f infer infer_q
