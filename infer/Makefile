# 检测是否是树莓派
IS_RASPBERRY_PI = $(shell cat /proc/cpuinfo 2>/dev/null | grep -i 'raspberry' > /dev/null && echo yes || echo no)

ifeq ($(IS_RASPBERRY_PI),yes)
    CCFLAGS = -O3 -march=native -ffast-math -Wall
    LDFLAGS = -lm
else
    CCFLAGS = -O3 -march=native -ffast-math -Wall -fopenmp
    LDFLAGS = -lm -fopenmp
endif

all: pod cli wss

# Nano-pod：带键盘和屏幕的电子鹦鹉笼
pod: nano_pod
nano_pod: main.c oled.c ui.c utils.c tokenizer.c tensor.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-CLI：适用于文字终端交互的终端程序
cli: nano_cli
nano_cli: main_cli.c utils.c tokenizer.c tensor.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-WSS：WebSocket服务器
wss: nano_wss
nano_wss: main_wss.c utils.c tokenizer.c tensor.c matmul_pthread.c infer.c
	$(CC) $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lwebsockets


# 仅适用于OpenWrt
install:
	chmod +x nano_pod
	cp -pR nano_pod /usr/bin/
	grep -qxF "/usr/bin/nano_pod &" /etc/rc.local || sed -i "/^[[:space:]]*exit[[:space:]]0/i /usr/bin/nano_pod &" /etc/rc.local

clean:
	rm -f nano_pod
	rm -f nano_cli
	rm -f nano_wss
