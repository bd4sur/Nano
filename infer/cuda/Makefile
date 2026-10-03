# Nano CUDA 推理引擎 Makefile
#
#   目标:
#     ../bin/nano_cuda   - CUDA int8 推理引擎（本工程主体）
#     ../bin/ref_greedy  - CPU 参考（infer.c）贪心解码器，用于逐 token 正确性对照
#
#   用法:
#     make                  构建两者（输出到上层 bin/）
#     make ../bin/nano_cuda 仅构建 CUDA 引擎
#
#   环境变量（调试用）：
#     NC_PRINT_IDS=1  bench 模式逐词元打印 token id（与 ref_greedy 对照）
#     NC_NO_EOS=1     bench 模式忽略 EOS 跑满指定长度（测速用）
#     NC_PROF=1       bench 模式打印逐 kernel 微基准（多轮取最小，抗 DVFS）

NVCC      ?= nvcc
CC        ?= gcc
ARCH      ?= sm_89
NVCCFLAGS ?= -O3 -arch=$(ARCH) --use_fast_math -lineinfo
LDFLAGS   ?= -Xlinker --gc-sections
CFLAGS    ?= -O2 -Wall -ffunction-sections -fdata-sections

all: ../bin/nano_cuda ../bin/ref_greedy

../bin/nano_cuda: nano_cuda.o tokenizer.o utils.o | ../bin
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^

nano_cuda.o: nano_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# tokenizer.c 中 Nano 架构专用函数引用 utils.c 的 Map/Trie（未用，链接期 gc-sections 丢弃），
# 其隐式声明警告属于预期，予以屏蔽
tokenizer.o: tokenizer.c tokenizer.h platform_shim.h
	$(CC) $(CFLAGS) -Wno-implicit-function-declaration -c -o $@ tokenizer.c

# NANO 架构分词器依赖的 Map/Trie 实现（独立编译，仅 libc 依赖）
utils.o: ../utils.c ../utils.h
	$(CC) $(CFLAGS) -I.. -c -o $@ ../utils.c

# CPU 参考实现：直接复用上层目录的原引擎源码
../bin/ref_greedy: ref_greedy.c ../infer.c ../infer.h ../utils.c ../tokenizer.c ../tensor.c ../hal_ram_linux.c | ../bin
	$(CC) -O2 -I.. -o $@ ref_greedy.c ../infer.c ../utils.c ../tokenizer.c ../tensor.c ../hal_ram_linux.c -lm

../bin:
	mkdir -p ../bin

clean:
	rm -f ../bin/nano_cuda ../bin/ref_greedy *.o

.PHONY: all clean
