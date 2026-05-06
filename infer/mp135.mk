# make -f mp135.mk pod -j16

BUILDROOT_HOST = /home/bd4sur/mp135/CoreMP135_buildroot/output/host
SYSROOT = $(BUILDROOT_HOST)/arm-buildroot-linux-gnueabihf/sysroot

CROSS_COMPILE ?= arm-buildroot-linux-gnueabihf-
CC      = $(BUILDROOT_HOST)/bin/$(CROSS_COMPILE)gcc
CXX     = $(BUILDROOT_HOST)/bin/$(CROSS_COMPILE)g++
AR      = $(BUILDROOT_HOST)/bin/$(CROSS_COMPILE)ar
STRIP   = $(BUILDROOT_HOST)/bin/$(CROSS_COMPILE)strip

CCFLAGS = -I$(SYSROOT)/usr/include --sysroot=$(SYSROOT) -O3 -ffast-math -Wall
LDFLAGS = -L$(SYSROOT)/usr/lib --sysroot=$(SYSROOT) -lm

BIN_DIR := bin

all: $(BIN_DIR) pod tty cli sort wss

$(BIN_DIR):
	mkdir -p $@

# Nano-Pod：带键盘和彩色SPI屏幕的电子鹦鹉笼（基于CoreMP135）
pod: $(BIN_DIR)/nano_pod
$(BIN_DIR)/nano_pod: main.c platform_linux.c display_linux_framebuffer.c keyboard_matrix16.c vsop87c_milli.c celestial.c ephemeris.c nongli.c flip.c graphics.c ui.c ui_app.c utils.c tokenizer.c tensor.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_POD_MP135 $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-TTY：适用于文字终端图形交互的终端程序
tty: $(BIN_DIR)/nano_tty
$(BIN_DIR)/nano_tty: main.c platform_linux.c display_ncurses.c keyboard_ncurses.c vsop87c_milli.c celestial.c ephemeris.c nongli.c flip.c graphics.c ui.c ui_app.c utils.c tokenizer.c tensor.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_TTY $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lncursesw

# Nano-CLI：适用于文字终端命令交互的终端程序
cli: $(BIN_DIR)/nano_cli
$(BIN_DIR)/nano_cli: main_cli.c platform_linux.c utils.c tokenizer.c tensor.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_CLI $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-Sort：演示如何用LLM解决排序问题（
sort: $(BIN_DIR)/nano_sort
$(BIN_DIR)/nano_sort: main_sort.c platform_linux.c utils.c tokenizer.c tensor.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_SORT $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-WSS：WebSocket服务器
wss: $(BIN_DIR)/nano_wss
$(BIN_DIR)/nano_wss: main_wss.c platform_linux.c utils.c tokenizer.c tensor.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_WSS $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lwebsockets


clean:
	rm -f $(BIN_DIR)/nano_pod
	rm -f $(BIN_DIR)/nano_tty
	rm -f $(BIN_DIR)/nano_cli
	rm -f $(BIN_DIR)/nano_sort
	rm -f $(BIN_DIR)/nano_wss
