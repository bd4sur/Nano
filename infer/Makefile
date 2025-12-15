# 检测是否是树莓派
IS_RASPBERRY_PI = $(shell cat /proc/cpuinfo 2>/dev/null | grep -i 'raspberry' > /dev/null && echo yes || echo no)

ifeq ($(IS_RASPBERRY_PI),yes)
    CCFLAGS = -O3 -march=native -ffast-math -Wall
    LDFLAGS = -lm
else
    CCFLAGS = -O3 -march=native -ffast-math -Wall -fopenmp
    LDFLAGS = -lm -fopenmp
endif

BIN_DIR := bin

all: $(BIN_DIR) pod tty cli sort wss

$(BIN_DIR):
	mkdir -p $@

# Nano-Pod：带键盘和屏幕的电子鹦鹉笼
pod: $(BIN_DIR)/nano_pod
$(BIN_DIR)/nano_pod: main.c asr.c tts.c ups.c platform_linux.c display_ssd130x_linux.c keyboard_matrix16.c graphics.c ui.c utils.c prompt.c tokenizer.c tensor.c matmul_pthread.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_POD_RPI5 $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-TTY：适用于文字终端图形交互的终端程序
tty: $(BIN_DIR)/nano_tty
$(BIN_DIR)/nano_tty: main.c platform_linux.c display_ncurses.c keyboard_ncurses.c graphics.c ui.c utils.c prompt.c tokenizer.c tensor.c matmul_pthread.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_TTY $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lncursesw

# Nano-CLI：适用于文字终端命令交互的终端程序
cli: $(BIN_DIR)/nano_cli
$(BIN_DIR)/nano_cli: main_cli.c platform_linux.c utils.c prompt.c tokenizer.c tensor.c matmul_pthread.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_CLI $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-Sort：演示如何用LLM解决排序问题（
sort: $(BIN_DIR)/nano_sort
$(BIN_DIR)/nano_sort: main_sort.c platform_linux.c utils.c prompt.c tokenizer.c tensor.c matmul_pthread.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_SORT $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-WSS：WebSocket服务器
wss: $(BIN_DIR)/nano_wss
$(BIN_DIR)/nano_wss: main_wss.c platform_linux.c utils.c prompt.c tokenizer.c tensor.c matmul_pthread.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_WSS $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lwebsockets


# 仅适用于OpenWrt
install:
	chmod +x $(BIN_DIR)/nano_pod
	cp -pR $(BIN_DIR)/nano_pod /usr/bin/
	grep -qxF "/usr/bin/nano_pod &" /etc/rc.local || sed -i "/^[[:space:]]*exit[[:space:]]0/i /usr/bin/nano_pod &" /etc/rc.local

clean:
	rm -f $(BIN_DIR)/nano_pod
	rm -f $(BIN_DIR)/nano_tty
	rm -f $(BIN_DIR)/nano_cli
	rm -f $(BIN_DIR)/nano_sort
	rm -f $(BIN_DIR)/nano_wss
