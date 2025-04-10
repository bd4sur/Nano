# 变量定义
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp

# .PHONY: all
# all: nano_infer_ws_server.c nano_infer_shell.c nano_infer.c
# 	$(CC) $(CFLAGS) -o infer infer.c -lm
# 	$(CC) $(CFLAGS) -o infer_q infer_q.c -lm

nano_infer_shell: nano_infer_shell.c nano_infer.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

nano_infer_ws_server: nano_infer_ws_server.c nano_infer.c
	$(CC) $(CFLAGS) $^ -o $@ -lm -lwebsockets

# 清理
.PHONY: clean
clean:
	rm -f nano_infer_shell nano_infer_ws_server
