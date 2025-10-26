# 检测是否是树莓派
IS_RASPBERRY_PI = $(shell cat /proc/cpuinfo 2>/dev/null | grep -i 'raspberry' > /dev/null && echo yes || echo no)

ifeq ($(IS_RASPBERRY_PI),yes)
    CCFLAGS = -O3 -march=native -ffast-math
    LDFLAGS = -lm
else
    CCFLAGS = -O3 -march=native -ffast-math -fopenmp
    LDFLAGS = -lm -fopenmp
endif

all: nanochat

nanochat: main.c oled.c ui.c avltree.c bpe.c hashmap.c trie.c quant.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f nanochat

