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

all: $(BIN_DIR) pod cli wss

$(BIN_DIR):
	mkdir -p $@

# Nano-pod：带键盘和屏幕的电子鹦鹉笼
pod: $(BIN_DIR)/nano_pod
$(BIN_DIR)/nano_pod: main.c display_hal.c graphics.c ui.c utils.c tokenizer.c tensor.c matmul_pthread.c infer.c | $(BIN_DIR)
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-CLI：适用于文字终端交互的终端程序
cli: $(BIN_DIR)/nano_cli
$(BIN_DIR)/nano_cli: main_cli.c utils.c tokenizer.c tensor.c matmul_pthread.c infer.c | $(BIN_DIR)
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-WSS：WebSocket服务器
wss: $(BIN_DIR)/nano_wss
$(BIN_DIR)/nano_wss: main_wss.c utils.c tokenizer.c tensor.c matmul_pthread.c infer.c | $(BIN_DIR)
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lwebsockets


# 仅适用于OpenWrt
install:
	chmod +x $(BIN_DIR)/nano_pod
	cp -pR $(BIN_DIR)/nano_pod /usr/bin/
	grep -qxF "/usr/bin/nano_pod &" /etc/rc.local || sed -i "/^[[:space:]]*exit[[:space:]]0/i /usr/bin/nano_pod &" /etc/rc.local

clean:
	rm -f $(BIN_DIR)/nano_pod
	rm -f $(BIN_DIR)/nano_cli
	rm -f $(BIN_DIR)/nano_wss
