# make -f mp135.mk pod -j16

BUILDROOT_HOST = /home/bd4sur/mp135/CoreMP135_buildroot/output/host
SYSROOT = $(BUILDROOT_HOST)/arm-buildroot-linux-gnueabihf/sysroot

CROSS_COMPILE ?= arm-buildroot-linux-gnueabihf-
CC      = $(BUILDROOT_HOST)/bin/$(CROSS_COMPILE)gcc
CXX     = $(BUILDROOT_HOST)/bin/$(CROSS_COMPILE)g++
AR      = $(BUILDROOT_HOST)/bin/$(CROSS_COMPILE)ar
STRIP   = $(BUILDROOT_HOST)/bin/$(CROSS_COMPILE)strip

ANIMAC_DIR  := vendor/Animac
ANIMAC_INC  := $(ANIMAC_DIR)
ANIMAC_SRCS := $(filter-out $(ANIMAC_DIR)/exclude.c, $(wildcard $(ANIMAC_DIR)/*.c))

CCFLAGS = -I$(SYSROOT)/usr/include --sysroot=$(SYSROOT) -O3 -ffast-math -Wall -I$(ANIMAC_INC) -I$(ANIMAC_DIR)
LDFLAGS = -L$(SYSROOT)/usr/lib --sysroot=$(SYSROOT) -lm

BIN_DIR := bin

all: $(BIN_DIR) pod tty cli sort wss

$(BIN_DIR):
	mkdir -p $@

# Nano-Pod：带键盘和彩色SPI屏幕的电子鹦鹉笼（基于CoreMP135），含触屏（hal_touch_evdev_linux.c）
pod: $(BIN_DIR)/nano_pod
$(BIN_DIR)/nano_pod: $(ANIMAC_SRCS) \
                        main.c \
                        hal_power_coremp135.c \
                        hal_ram_linux.c \
                        hal_fs_linux.c \
                        hal_os_linux.c \
                        hal_misc_linux.c \
                        hal_display_ili9342c_linux.c \
                        hal_touch_evdev_linux.c \
                        hal_key_mp135.c \
                        vsop87c_milli.c \
                        celestial.c \
                        ephemeris.c \
                        nongli.c \
                        flip.c \
                        nano_fft.c \
                        nano_min.c \
                        ofdm_modem.c \
                        graphics.c \
                        gfx_font_12.c \
                        gfx_font_16.c \
                        pinyin_ime.c \
                        ui.c \
                        ui_app.c \
                        ui_almanac.c \
                        ui_calendar.c \
                        ui_cloud.c \
                        ui_dict.c \
                        ui_ebook.c \
                        ui_goldminer.c \
                        ui_icon.c \
                        ui_llm.c \
                        ui_musicbox_mp3.c \
                        ui_musicbox.c \
                        ui_ofdm.c \
                        ui_particlelife.c \
                        ui_pedometer.c \
                        ui_pinyin_ime.c \
                        ui_ripple.c \
                        ui_softkbd.c \
                        ui_spectrogram.c \
                        ui_tetris.c \
                        ui_water.c \
                        utils.c \
                        tokenizer.c \
                        tensor.c \
                        infer.c | $(BIN_DIR)
	$(CC) -DNANO_POD_MP135 $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-TTY：适用于文字终端图形交互的终端程序
tty: $(BIN_DIR)/nano_tty
$(BIN_DIR)/nano_tty: $(ANIMAC_SRCS) \
                        main.c \
                        hal_audio_out_alsa_linux.c \
                        hal_imu_linux.c \
                        hal_audio_in_alsa_linux.c \
                        hal_ram_linux.c \
                        hal_fs_linux.c \
                        hal_os_linux.c \
                        hal_misc_linux.c \
                        hal_display_ncurses_linux.c \
                        hal_touch_evdev_linux.c \
                        hal_key_ncurses_linux.c \
                        vsop87c_milli.c \
                        celestial.c \
                        ephemeris.c \
                        nongli.c \
                        flip.c \
                        nano_fft.c \
                        nano_min.c \
                        ofdm_modem.c \
                        graphics.c \
                        gfx_font_12.c \
                        gfx_font_16.c \
                        pinyin_ime.c \
                        ui.c \
                        ui_app.c \
                        ui_almanac.c \
                        ui_calendar.c \
                        ui_cloud.c \
                        ui_dict.c \
                        ui_ebook.c \
                        ui_goldminer.c \
                        ui_icon.c \
                        ui_llm.c \
                        ui_musicbox_mp3.c \
                        ui_musicbox.c \
                        ui_ofdm.c \
                        ui_particlelife.c \
                        ui_pedometer.c \
                        ui_pinyin_ime.c \
                        ui_ripple.c \
                        ui_softkbd.c \
                        ui_spectrogram.c \
                        ui_tetris.c \
                        ui_water.c \
                        utils.c \
                        tokenizer.c \
                        tensor.c \
                        infer.c | $(BIN_DIR)
	$(CC) -DNANO_TTY $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lncursesw -lasound

# Nano-CLI：适用于文字终端命令交互的终端程序
cli: $(BIN_DIR)/nano_cli
$(BIN_DIR)/nano_cli: main_cli.c hal_ram_linux.c hal_fs_linux.c hal_os_linux.c hal_misc_linux.c utils.c tokenizer.c tensor.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_CLI $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-Sort：演示如何用LLM解决排序问题（
sort: $(BIN_DIR)/nano_sort
$(BIN_DIR)/nano_sort: main_sort.c hal_ram_linux.c hal_fs_linux.c hal_os_linux.c hal_misc_linux.c utils.c tokenizer.c tensor.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_SORT $(CCFLAGS) $^ -o $@ $(LDFLAGS)

# Nano-WSS：WebSocket服务器
wss: $(BIN_DIR)/nano_wss
$(BIN_DIR)/nano_wss: main_wss.c hal_ram_linux.c hal_fs_linux.c hal_os_linux.c hal_misc_linux.c utils.c tokenizer.c tensor.c infer.c | $(BIN_DIR)
	$(CC) -DNANO_WSS $(CCFLAGS) $^ -o $@ $(LDFLAGS) -lwebsockets


clean:
	rm -f $(BIN_DIR)/nano_pod
	rm -f $(BIN_DIR)/nano_tty
	rm -f $(BIN_DIR)/nano_cli
	rm -f $(BIN_DIR)/nano_sort
	rm -f $(BIN_DIR)/nano_wss
