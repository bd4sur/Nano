# 变量定义
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp -lm

# 目标文件
TARGET = trie

# 源文件和对象文件
SRCS = trie.c
OBJS = $(SRCS:.c=.o)   # 把所有的 .c 文件替换为 .o 文件

# 默认目标
all: $(TARGET)

# 链接目标
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# 编译规则，%.o 是模式匹配，匹配所有的 .c 文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理目标
clean:
	rm -f $(OBJS) $(TARGET)

# PHONY 目标声明，避免与同名文件冲突
.PHONY: all clean
