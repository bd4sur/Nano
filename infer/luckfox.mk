# make -f luckfox.mk pod -j16
# Luckfox-Pico-86-Panel（RV1106G3，720x720 屏幕，buildroot/uClibc 系统）
# 与 mp135.mk 的 pod 目标一致，仅将显示 HAL 替换为 framebuffer 实现。

TOOLCHAIN_DIR := toolchains/arm-rockchip830-linux-uclibcgnueabihf
SYSROOT       := $(abspath $(TOOLCHAIN_DIR)/arm-rockchip830-linux-uclibcgnueabihf/sysroot)
# 交叉编译的 tinyalsa 静态库及其头文件（设备 rootfs 中无 tinyalsa，故静态链接）
EXTRA_ROOT    := $(abspath toolchains/sysroot-extra)

CROSS_COMPILE ?= arm-rockchip830-linux-uclibcgnueabihf-
CC      = $(TOOLCHAIN_DIR)/bin/$(CROSS_COMPILE)gcc
CXX     = $(TOOLCHAIN_DIR)/bin/$(CROSS_COMPILE)g++
AR      = $(TOOLCHAIN_DIR)/bin/$(CROSS_COMPILE)ar
STRIP   = $(TOOLCHAIN_DIR)/bin/$(CROSS_COMPILE)strip

ANIMAC_DIR  := vendor/Animac
ANIMAC_INC  := $(ANIMAC_DIR)
ANIMAC_SRCS := $(filter-out $(ANIMAC_DIR)/exclude.c, $(wildcard $(ANIMAC_DIR)/*.c))

# 设备 framebuffer 节点为 /dev/fb0（720x720, 32bpp），覆盖 HAL 默认的 /dev/fb1；
# FB_SWAP_RB：修正面板红蓝互换（详见 hal_display_framebuffer_linux.c 注释），仅本机型定义；
# FB_UPSCALE=2：逻辑 320x240 放大 2 倍为 640x480 居中上屏（左右黑边 40、上下黑边 120）；
# TOUCH_SCALE=2 + 物理量程 720x720：触屏坐标做逆变换，与逻辑坐标对齐
CCFLAGS = -I$(EXTRA_ROOT)/include --sysroot=$(SYSROOT) -O3 -ffast-math -Wall -DFB_DEVICE='"/dev/fb0"' -DFB_SWAP_RB=1 -DFB_UPSCALE=2 -DTOUCH_SCALE=2 -DTOUCH_PHYS_WIDTH=720 -DTOUCH_PHYS_HEIGHT=720 -I$(ANIMAC_INC) -I$(ANIMAC_DIR)
LDFLAGS = -L$(EXTRA_ROOT)/lib --sysroot=$(SYSROOT) -lm

BIN_DIR := bin

all: $(BIN_DIR) pod

$(BIN_DIR):
	mkdir -p $@

# Nano-Pod：电子鹦鹉笼（Luckfox-Pico-86-Panel），显示走 /dev/fb0 framebuffer
pod: $(BIN_DIR)/nano_pod_luckfox
$(BIN_DIR)/nano_pod_luckfox: $(ANIMAC_SRCS) \
                        main.c \
                        hal_fs_linux.c \
                        hal_ram_linux.c \
                        hal_os_linux.c \
                        hal_audio_in_mp135.c \
                        hal_audio_out_mp135.c \
                        hal_display_framebuffer_linux.c \
                        hal_imu_linux.c \
                        hal_key_mp135.c \
                        hal_touch_evdev_linux.c \
                        hal_power_coremp135.c \
                        hal_misc_linux.c \
                        celestial.c \
                        ephemeris.c \
                        flip.c \
                        gfx_font_12.c \
                        gfx_font_16.c \
                        graphics.c \
                        infer.c \
                        nano_fft.c \
                        nano_min.c \
                        nongli.c \
                        ofdm_modem.c \
                        pinyin_ime.c \
                        tensor.c \
                        tokenizer.c \
                        utils.c \
                        vsop87c_milli.c \
                        ui.c \
                        ui_app.c \
                        ui_almanac.c \
                        ui_calendar.c \
                        ui_cloud.c \
                        ui_dict.c \
                        ui_ebook.c \
                        ui_goldminer.c \
                        ui_grid16kbd.c \
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
                        | $(BIN_DIR)
	$(CC) -DNANO_POD_MP135 $(CCFLAGS) $^ -o $@ $(LDFLAGS) -ltinyalsa -ldl -lpthread

clean:
	rm -f $(BIN_DIR)/nano_pod_luckfox
