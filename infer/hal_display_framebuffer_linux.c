#include "hal_display.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <time.h>

// 最高刷新频率限制（Hz），编译时可通过 -DREFRESH_RATE_LIMIT_HZ=30 覆盖。
// 设为 0 或负数则恢复默认值 60Hz。
#ifndef REFRESH_RATE_LIMIT_HZ
#define REFRESH_RATE_LIMIT_HZ 60
#endif
#if REFRESH_RATE_LIMIT_HZ <= 0
#undef REFRESH_RATE_LIMIT_HZ
#define REFRESH_RATE_LIMIT_HZ 60
#endif

// Framebuffer device path. Can be overridden by FRAMEBUFFER environment variable.
#ifndef FB_DEVICE
#define FB_DEVICE "/dev/fb1"
#endif

// FB_SWAP_RB：交换输出像素的 R/B 通道（编译期 -DFB_SWAP_RB=1 启用）。
// 仅用于 Luckfox-Pico-86-Panel（RV1106G3）：其 /dev/fb0 驱动报告标准
// BGRA8888（R@16,G@8,B@0），但下方 32bpp BGRA 快速路径的打包顺序与之相反，
// 导致面板红蓝互换（青→黄、蓝→红）。该机型在 luckfox.mk 中定义此宏，
// 使快速路径按驱动契约打包；其余平台不定义，行为不变。
#ifndef FB_SWAP_RB
#define FB_SWAP_RB 0
#endif

// FB_UPSCALE：逻辑帧缓冲整数倍放大上屏（编译期 -DFB_UPSCALE=N 启用，默认 1）。
// 仅用于 Luckfox-Pico-86-Panel：业务按 320x240 逻辑分辨率渲染，物理放大 2 倍
// （1 逻辑像素 -> 2x2 物理像素）得到 640x480，再居中显示于 720x720 面板
// （左右黑边 40px、上下黑边 120px）。放大路径经 write_pixel 按 px_fmt 打包，
// 颜色天然正确，与 FB_SWAP_RB 无耦合。其余平台不定义，行为不变。
#ifndef FB_UPSCALE
#define FB_UPSCALE 1
#endif

static int fb_fd = -1;
static uint8_t *fb_mmap = NULL;
static uint32_t fb_mmap_size = 0;
static uint32_t fb_line_length = 0;
static uint32_t fb_bpp = 0;
static uint32_t fb_width = 0;
static uint32_t fb_height = 0;


static inline void limit_refresh_rate(void) {
    static struct timespec last_ts = {0, 0};
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (last_ts.tv_sec != 0 || last_ts.tv_nsec != 0) {
        long elapsed_us = (now.tv_sec - last_ts.tv_sec) * 1000000L
                        + (now.tv_nsec - last_ts.tv_nsec) / 1000L;
        long target_interval_us = 1000000L / REFRESH_RATE_LIMIT_HZ;
        if (elapsed_us < target_interval_us) {
            usleep((useconds_t)(target_interval_us - elapsed_us));
        }
    }
    last_ts = now;
}

static inline void sync_and_throttle(void) {
    if (fb_mmap != NULL && fb_mmap_size > 0) {
        msync(fb_mmap, fb_mmap_size, MS_SYNC);
    }
    limit_refresh_rate();
}

// Pixel format info from fb_var_screeninfo
static struct {
    uint8_t r_offset;
    uint8_t g_offset;
    uint8_t b_offset;
    uint8_t r_len;
    uint8_t g_len;
    uint8_t b_len;
} px_fmt;

// Convert RGB888 to RGB565 (little-endian pixel value, no byte-swap)
static inline uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    // return ((uint16_t)(r & 0xF8) << 8) |
    //        ((uint16_t)(g & 0xFC) << 3) |
    //        (b >> 3);
    uint8_t r5 = (r >= 252) ? 31 : (r + 4) >> 3;
    uint8_t g6 = (g >= 254) ? 63 : (g + 2) >> 2;
    uint8_t b5 = (b >= 252) ? 31 : (b + 4) >> 3;
    return (((uint16_t)(r5 << 11)) | ((uint16_t)(g6 << 5)) | ((uint16_t)b5));
}

// Convert 8-bit color channel to framebuffer bit-length.
// We keep the high bits only (e.g. 8->5: c >> 3), which matches
// the reference rgb888_to_rgb565 implementation and avoids the
// uneven quantization steps of the old multiply+divide method.
static inline uint32_t convert_channel(uint8_t c, uint8_t len) {
    if (len >= 8) return c;
    if (len == 0) return 0;
    return c >> (8 - len);
}

// Convert RGB565 to RGB888 channels (reference: graphics.c)
static inline uint8_t RGB565_R(uint16_t c) {
    uint8_t r = (c >> 11) & 0x1F;
    return (r << 3) | (r >> 2);
}
static inline uint8_t RGB565_G(uint16_t c) {
    uint8_t g = (c >> 5) & 0x3F;
    return (g << 2) | (g >> 4);
}
static inline uint8_t RGB565_B(uint16_t c) {
    uint8_t b = c & 0x1F;
    return (b << 3) | (b >> 2);
}

// Write a single RGB888 pixel into framebuffer memory at (x, y)
static inline void write_pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t *dst = fb_mmap + y * fb_line_length + x * (fb_bpp / 8);

    if (fb_bpp == 16) {
        uint16_t pix;
        if (px_fmt.r_offset == 11 && px_fmt.r_len == 5 &&
            px_fmt.g_offset == 5  && px_fmt.g_len == 6 &&
            px_fmt.b_offset == 0  && px_fmt.b_len == 5) {
            pix = rgb888_to_rgb565(r, g, b);
        } else {
            pix = 0;
            pix |= (convert_channel(r, px_fmt.r_len) << px_fmt.r_offset);
            pix |= (convert_channel(g, px_fmt.g_len) << px_fmt.g_offset);
            pix |= (convert_channel(b, px_fmt.b_len) << px_fmt.b_offset);
        }
        dst[0] = pix & 0xFF;
        dst[1] = (pix >> 8) & 0xFF;
    }
    else if (fb_bpp == 24) {
        uint32_t pix = 0;
        pix |= (convert_channel(r, px_fmt.r_len) << px_fmt.r_offset);
        pix |= (convert_channel(g, px_fmt.g_len) << px_fmt.g_offset);
        pix |= (convert_channel(b, px_fmt.b_len) << px_fmt.b_offset);
        dst[0] = pix & 0xFF;
        dst[1] = (pix >> 8) & 0xFF;
        dst[2] = (pix >> 16) & 0xFF;
    }
    else if (fb_bpp == 32) {
        uint32_t pix = 0;
        pix |= (convert_channel(r, px_fmt.r_len) << px_fmt.r_offset);
        pix |= (convert_channel(g, px_fmt.g_len) << px_fmt.g_offset);
        pix |= (convert_channel(b, px_fmt.b_len) << px_fmt.b_offset);
        // For 32bpp, preserve existing alpha/padding if offset >= 24, else set to 0xFF
        if (px_fmt.r_len + px_fmt.r_offset <= 24 &&
            px_fmt.g_len + px_fmt.g_offset <= 24 &&
            px_fmt.b_len + px_fmt.b_offset <= 24) {
            pix |= (0xFF << 24);
        }
        dst[0] = pix & 0xFF;
        dst[1] = (pix >> 8) & 0xFF;
        dst[2] = (pix >> 16) & 0xFF;
        dst[3] = (pix >> 24) & 0xFF;
    }
    else {
        // 8bpp or other: grayscale fallback
        uint8_t gray = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
        dst[0] = gray;
    }
}

void display_hal_refresh(
    uint8_t *frame_buffer_rgb888, uint32_t fb_width_in, uint32_t fb_height_in,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    if (fb_fd < 0 || fb_mmap == NULL || frame_buffer_rgb888 == NULL) {
        return;
    }

    // Clamp view region to logical framebuffer bounds
    if (x0 >= fb_width_in) x0 = fb_width_in - 1;
    if (y0 >= fb_height_in) y0 = fb_height_in - 1;
    if (x0 + view_width > fb_width_in) view_width = fb_width_in - x0;
    if (y0 + view_height > fb_height_in) view_height = fb_height_in - y0;

    if (view_width == 0 || view_height == 0) {
        return;
    }

#if FB_UPSCALE > 1
    // 整数倍放大上屏：完整逻辑帧（fb_width_in x fb_height_in）放大 FB_UPSCALE 倍后居中。
    // 偏移基于完整逻辑帧（而非当前脏区 view）计算，保证局部刷新落在正确物理位置。
    {
        int32_t frame_off_x = ((int32_t)fb_width  - (int32_t)fb_width_in  * FB_UPSCALE) / 2;
        int32_t frame_off_y = ((int32_t)fb_height - (int32_t)fb_height_in * FB_UPSCALE) / 2;
        if (frame_off_x >= 0 && frame_off_y >= 0) {
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = frame_off_y + (int32_t)src_y * FB_UPSCALE;
                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                for (uint32_t x = 0; x < view_width; x++) {
                    int32_t dst_x = frame_off_x + (int32_t)(x0 + x) * FB_UPSCALE;
                    for (int32_t sy = 0; sy < FB_UPSCALE; sy++) {
                        for (int32_t sx = 0; sx < FB_UPSCALE; sx++) {
                            write_pixel((uint32_t)(dst_x + sx), (uint32_t)(dst_y + sy),
                                        src[0], src[1], src[2]);
                        }
                    }
                    src += 3;
                }
            }
            sync_and_throttle();
            return;
        }
        // 放大后超出物理屏幕：回退为原样渲染
    }
#endif

    // Center the view on the physical screen
    int32_t offset_x = 0;
    int32_t offset_y = 0;

    if (view_width < fb_width) {
        offset_x = (fb_width - view_width) / 2;
    }
    if (view_height < fb_height) {
        offset_y = (fb_height - view_height) / 2;
    }

    // Pre-calculate actual copy width to avoid per-pixel boundary checks in inner loops
    uint32_t copy_width = view_width;
    if ((uint32_t)offset_x + copy_width > fb_width) {
        copy_width = fb_width - (uint32_t)offset_x;
    }

    if (fb_bpp == 16) {
        // Fast path: native RGB565 (R5G6B5, little-endian)
        if (px_fmt.r_offset == 11 && px_fmt.r_len == 5 &&
            px_fmt.g_offset == 5  && px_fmt.g_len == 6 &&
            px_fmt.b_offset == 0  && px_fmt.b_len == 5) {
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                uint8_t *dst = fb_mmap + dst_y * fb_line_length + offset_x * 2;
                for (uint32_t x = 0; x < copy_width; x++) {
                    uint16_t pix = rgb888_to_rgb565(src[0], src[1], src[2]);
                    dst[0] = pix & 0xFF;
                    dst[1] = (pix >> 8) & 0xFF;
                    dst += 2;
                    src += 3;
                }
            }
        }
        else {
            // Generic 16bpp: pack according to pixel format
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                uint8_t *dst = fb_mmap + dst_y * fb_line_length + offset_x * 2;
                for (uint32_t x = 0; x < copy_width; x++) {
                    uint16_t pix = 0;
                    pix |= (convert_channel(src[0], px_fmt.r_len) << px_fmt.r_offset);
                    pix |= (convert_channel(src[1], px_fmt.g_len) << px_fmt.g_offset);
                    pix |= (convert_channel(src[2], px_fmt.b_len) << px_fmt.b_offset);
                    dst[0] = pix & 0xFF;
                    dst[1] = (pix >> 8) & 0xFF;
                    dst += 2;
                    src += 3;
                }
            }
        }
    }
    else if (fb_bpp == 24) {
        // Fast path: standard BGR888 (b_offset=0, g_offset=8, r_offset=16)
        if (px_fmt.r_offset == 16 && px_fmt.r_len == 8 &&
            px_fmt.g_offset == 8  && px_fmt.g_len == 8 &&
            px_fmt.b_offset == 0  && px_fmt.b_len == 8) {
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                uint8_t *dst = fb_mmap + dst_y * fb_line_length + offset_x * 3;
                for (uint32_t x = 0; x < copy_width; x++) {
                    dst[0] = src[2]; // B
                    dst[1] = src[1]; // G
                    dst[2] = src[0]; // R
                    dst += 3;
                    src += 3;
                }
            }
        }
        else {
            // Generic 24bpp
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                uint8_t *dst = fb_mmap + dst_y * fb_line_length + offset_x * 3;
                for (uint32_t x = 0; x < copy_width; x++) {
                    uint32_t pix = 0;
                    pix |= (convert_channel(src[0], px_fmt.r_len) << px_fmt.r_offset);
                    pix |= (convert_channel(src[1], px_fmt.g_len) << px_fmt.g_offset);
                    pix |= (convert_channel(src[2], px_fmt.b_len) << px_fmt.b_offset);
                    dst[0] = pix & 0xFF;
                    dst[1] = (pix >> 8) & 0xFF;
                    dst[2] = (pix >> 16) & 0xFF;
                    dst += 3;
                    src += 3;
                }
            }
        }
    }
    else if (fb_bpp == 32) {
        // Fast path: standard BGRA8888 (b=0, g=8, r=16, a=24)
        if (px_fmt.r_offset == 16 && px_fmt.r_len == 8 &&
            px_fmt.g_offset == 8  && px_fmt.g_len == 8 &&
            px_fmt.b_offset == 0  && px_fmt.b_len == 8) {
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                uint8_t *dst = fb_mmap + dst_y * fb_line_length + offset_x * 4;
                for (uint32_t x = 0; x < copy_width; x++) {
#if FB_SWAP_RB
                    // 按驱动契约打包：bits16-23=R, bits8-15=G, bits0-7=B（字节序 B,G,R,A）
                    uint32_t pix = 0xFF000000U |
                                  ((uint32_t)src[0] << 16) |
                                  ((uint32_t)src[1] << 8) |
                                  src[2];
#else
                    uint32_t pix = 0xFF000000U |
                                  ((uint32_t)src[2] << 16) |
                                  ((uint32_t)src[1] << 8) |
                                  src[0];
#endif
                    dst[0] = pix & 0xFF;
                    dst[1] = (pix >> 8) & 0xFF;
                    dst[2] = (pix >> 16) & 0xFF;
                    dst[3] = (pix >> 24) & 0xFF;
                    dst += 4;
                    src += 3;
                }
            }
        }
        // Fast path: standard RGBA8888 (r=24, g=16, b=8, a=0)
        else if (px_fmt.r_offset == 24 && px_fmt.r_len == 8 &&
                 px_fmt.g_offset == 16 && px_fmt.g_len == 8 &&
                 px_fmt.b_offset == 8  && px_fmt.b_len == 8) {
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                uint8_t *dst = fb_mmap + dst_y * fb_line_length + offset_x * 4;
                for (uint32_t x = 0; x < copy_width; x++) {
                    uint32_t pix = ((uint32_t)src[0] << 24) |
                                  ((uint32_t)src[1] << 16) |
                                  ((uint32_t)src[2] << 8) |
                                  0xFF;
                    dst[0] = pix & 0xFF;
                    dst[1] = (pix >> 8) & 0xFF;
                    dst[2] = (pix >> 16) & 0xFF;
                    dst[3] = (pix >> 24) & 0xFF;
                    dst += 4;
                    src += 3;
                }
            }
        }
        else {
            // Generic 32bpp
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                uint8_t *dst = fb_mmap + dst_y * fb_line_length + offset_x * 4;
                for (uint32_t x = 0; x < copy_width; x++) {
                    uint32_t pix = 0;
                    pix |= (convert_channel(src[0], px_fmt.r_len) << px_fmt.r_offset);
                    pix |= (convert_channel(src[1], px_fmt.g_len) << px_fmt.g_offset);
                    pix |= (convert_channel(src[2], px_fmt.b_len) << px_fmt.b_offset);
                    if (px_fmt.r_len + px_fmt.r_offset <= 24 &&
                        px_fmt.g_len + px_fmt.g_offset <= 24 &&
                        px_fmt.b_len + px_fmt.b_offset <= 24) {
                        pix |= (0xFFU << 24);
                    }
                    dst[0] = pix & 0xFF;
                    dst[1] = (pix >> 8) & 0xFF;
                    dst[2] = (pix >> 16) & 0xFF;
                    dst[3] = (pix >> 24) & 0xFF;
                    dst += 4;
                    src += 3;
                }
            }
        }
    }
    else {
        // 8bpp or other: grayscale fallback, row-by-row
        for (uint32_t y = 0; y < view_height; y++) {
            uint32_t src_y = y0 + y;
            int32_t dst_y = offset_y + (int32_t)y;
            if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

            uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
            uint8_t *dst = fb_mmap + dst_y * fb_line_length + offset_x;
            for (uint32_t x = 0; x < copy_width; x++) {
                dst[x] = (uint8_t)((src[0] * 77 + src[1] * 150 + src[2] * 29) >> 8);
                src += 3;
            }
        }
    }

    sync_and_throttle();
}

void display_hal_refresh_rgb565(
    uint16_t *frame_buffer_rgb565, uint32_t fb_width_in, uint32_t fb_height_in,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    if (fb_fd < 0 || fb_mmap == NULL || frame_buffer_rgb565 == NULL) {
        return;
    }

    // Clamp view region to logical framebuffer bounds
    if (x0 >= fb_width_in) x0 = fb_width_in - 1;
    if (y0 >= fb_height_in) y0 = fb_height_in - 1;
    if (x0 + view_width > fb_width_in) view_width = fb_width_in - x0;
    if (y0 + view_height > fb_height_in) view_height = fb_height_in - y0;

    if (view_width == 0 || view_height == 0) {
        return;
    }

    // Center the view on the physical screen
    int32_t offset_x = 0;
    int32_t offset_y = 0;

    if (view_width < fb_width) {
        offset_x = (fb_width - view_width) / 2;
    }
    if (view_height < fb_height) {
        offset_y = (fb_height - view_height) / 2;
    }

    // Pre-calculate actual copy width to avoid per-pixel boundary checks
    uint32_t copy_width = view_width;
    if ((uint32_t)offset_x + copy_width > fb_width) {
        copy_width = fb_width - (uint32_t)offset_x;
    }

    // Fast path: physical framebuffer is native RGB565, memcpy row-by-row
    if (fb_bpp == 16 &&
        px_fmt.r_offset == 11 && px_fmt.r_len == 5 &&
        px_fmt.g_offset == 5  && px_fmt.g_len == 6 &&
        px_fmt.b_offset == 0  && px_fmt.b_len == 5) {
        // Ultra-fast path: single block copy when row pitches match exactly
        if (offset_x == 0 && x0 == 0 &&
            copy_width == fb_width_in &&
            fb_line_length == fb_width_in * sizeof(uint16_t) &&
            offset_y >= 0 && (uint32_t)offset_y + view_height <= fb_height) {
            uint8_t *src_start = (uint8_t *)frame_buffer_rgb565 + y0 * fb_width_in * sizeof(uint16_t);
            uint8_t *dst_start = fb_mmap + offset_y * fb_line_length;
            memcpy(dst_start, src_start, view_height * fb_line_length);
            sync_and_throttle();
            return;
        }

        for (uint32_t y = 0; y < view_height; y++) {
            uint32_t src_y = y0 + y;
            int32_t dst_y = offset_y + (int32_t)y;
            if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

            uint16_t *src_row = frame_buffer_rgb565 + src_y * fb_width_in + x0;
            uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 2;
            memcpy(dst_row, src_row, copy_width * sizeof(uint16_t));
        }
        sync_and_throttle();
        return;
    }

    // Fallback: decompose RGB565 to RGB888 then write through write_pixel
    for (uint32_t y = 0; y < view_height; y++) {
        uint32_t src_y = y0 + y;
        int32_t dst_y = offset_y + (int32_t)y;
        if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

        for (uint32_t x = 0; x < view_width; x++) {
            uint32_t src_x = x0 + x;
            int32_t dst_x = offset_x + (int32_t)x;
            if (dst_x < 0 || dst_x >= (int32_t)fb_width) continue;

            uint16_t c = frame_buffer_rgb565[src_y * fb_width_in + src_x];
            uint8_t r = RGB565_R(c);
            uint8_t g = RGB565_G(c);
            uint8_t b = RGB565_B(c);
            write_pixel((uint32_t)dst_x, (uint32_t)dst_y, r, g, b);
        }
    }

    sync_and_throttle();
}

void display_hal_refresh_rgb565_double(uint16_t *frame_buffer_rgb565_top, uint16_t *frame_buffer_rgb565_bottom,
    uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height) {
    return;
}

void display_hal_init(void) {
    const char *fb_dev = getenv("FRAMEBUFFER");
    if (fb_dev == NULL) {
        fb_dev = FB_DEVICE;
    }

    fb_fd = open(fb_dev, O_RDWR);
    if (fb_fd < 0) {
        printf("Failed to open framebuffer device %s\n", fb_dev);
        return;
    }

    struct fb_fix_screeninfo finfo;
    struct fb_var_screeninfo vinfo;

    if (ioctl(fb_fd, FBIOGET_FSCREENINFO, &finfo) < 0) {
        printf("Failed to get fb fixed screeninfo\n");
        close(fb_fd);
        fb_fd = -1;
        return;
    }

    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo) < 0) {
        printf("Failed to get fb variable screeninfo\n");
        close(fb_fd);
        fb_fd = -1;
        return;
    }

    fb_width = vinfo.xres;
    fb_height = vinfo.yres;
    fb_bpp = vinfo.bits_per_pixel;
    fb_line_length = finfo.line_length;
    fb_mmap_size = finfo.smem_len;

    px_fmt.r_offset = vinfo.red.offset;
    px_fmt.r_len = vinfo.red.length;
    px_fmt.g_offset = vinfo.green.offset;
    px_fmt.g_len = vinfo.green.length;
    px_fmt.b_offset = vinfo.blue.offset;
    px_fmt.b_len = vinfo.blue.length;

    printf("Framebuffer: %s, %dx%d, %dbpp, line_length=%d\n",
           fb_dev, fb_width, fb_height, fb_bpp, fb_line_length);
    printf("Pixel format: R(%d,%d) G(%d,%d) B(%d,%d)\n",
           px_fmt.r_offset, px_fmt.r_len,
           px_fmt.g_offset, px_fmt.g_len,
           px_fmt.b_offset, px_fmt.b_len);

    fb_mmap = (uint8_t *)mmap(NULL, fb_mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (fb_mmap == MAP_FAILED) {
        printf("Failed to mmap framebuffer\n");
        close(fb_fd);
        fb_fd = -1;
        fb_mmap = NULL;
        return;
    }

    // Clear screen to black on init
    memset(fb_mmap, 0, fb_mmap_size);

    // 默认背光（NANO_DEFAULT_BRIGHTNESS；业务层 ui_init 初值与其他平台 display_hal_init 同源）
    display_set_brightness(NANO_DEFAULT_BRIGHTNESS);
}

void display_hal_close(void) {
    if (fb_mmap != NULL) {
        munmap(fb_mmap, fb_mmap_size);
        fb_mmap = NULL;
    }
    if (fb_fd >= 0) {
        close(fb_fd);
        fb_fd = -1;
    }
}

void display_set_brightness(uint8_t value) {
    // Linux 通用背光调节：经 sysfs backlight 接口（pwm-backlight/gpio-backlight 等
    // 内核驱动均可，Luckfox-Pico-86-Panel 的 RV1106 镜像暴露 /sys/class/backlight/
    // backlight，max_brightness=255）。value 0~255 按比例映射到设备量程。
    static char bl_path[160] = "";   // 背光 brightness 节点路径（首次调用时探测）
    static int  bl_max = 255;        // 设备 max_brightness
    static int  bl_probed = 0;

    if (!bl_probed) {
        bl_probed = 1;
        char bl_name[64] = "";
        DIR *d = opendir("/sys/class/backlight");
        if (d != NULL) {
            struct dirent *de;
            while ((de = readdir(d)) != NULL) {
                if (de->d_name[0] == '.') continue;
                snprintf(bl_name, sizeof(bl_name), "%s", de->d_name);
                break; // 取第一个背光设备
            }
            closedir(d);
        }
        if (bl_name[0] != '\0') {
            snprintf(bl_path, sizeof(bl_path),
                     "/sys/class/backlight/%s/brightness", bl_name);
            char maxpath[176];
            snprintf(maxpath, sizeof(maxpath),
                     "/sys/class/backlight/%s/max_brightness", bl_name);
            int fd = open(maxpath, O_RDONLY);
            if (fd >= 0) {
                char buf[16] = {0};
                if (read(fd, buf, sizeof(buf) - 1) > 0) {
                    int m = atoi(buf);
                    if (m > 0) bl_max = m;
                }
                close(fd);
            }
            printf("Backlight: %s, max=%d\n", bl_path, bl_max);
        }
    }
    if (bl_path[0] == '\0') return; // 无背光设备：忽略

    int v = (int)((value * (uint32_t)bl_max) / 255);
    int fd = open(bl_path, O_WRONLY);
    if (fd < 0) return;
    char buf[8];
    int n = snprintf(buf, sizeof(buf), "%d", v);
    if (write(fd, buf, (size_t)n) < 0) { /* 忽略写失败 */ }
    close(fd);
}


void display_sleep(void) {
    return;
}

void display_wakeup(void) {
    return;
}

