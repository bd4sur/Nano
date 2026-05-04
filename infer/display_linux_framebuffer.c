#include "display_hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>

// Framebuffer device path. Can be overridden by FRAMEBUFFER environment variable.
#ifndef FB_DEVICE
#define FB_DEVICE "/dev/fb1"
#endif

static int fb_fd = -1;
static uint8_t *fb_mmap = NULL;
static uint32_t fb_mmap_size = 0;
static uint32_t fb_line_length = 0;
static uint32_t fb_bpp = 0;
static uint32_t fb_width = 0;
static uint32_t fb_height = 0;

// Pixel format info from fb_var_screeninfo
static struct {
    uint8_t r_offset;
    uint8_t g_offset;
    uint8_t b_offset;
    uint8_t r_len;
    uint8_t g_len;
    uint8_t b_len;
} px_fmt;

// Convert 8-bit color channel to framebuffer bit-length
static inline uint32_t convert_channel(uint8_t c, uint8_t len) {
    if (len == 8) return c;
    if (len == 0) return 0;
    // Scale to target bit depth
    return ((uint32_t)c * ((1U << len) - 1) + 127) / 255;
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
        uint16_t pix = 0;
        pix |= (convert_channel(r, px_fmt.r_len) << px_fmt.r_offset);
        pix |= (convert_channel(g, px_fmt.g_len) << px_fmt.g_offset);
        pix |= (convert_channel(b, px_fmt.b_len) << px_fmt.b_offset);
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
        uint16_t *temp_row = (uint16_t *)malloc(copy_width * sizeof(uint16_t));
        if (!temp_row) return;

        // Fast path: native RGB565 (R5G6B5, little-endian)
        if (px_fmt.r_offset == 11 && px_fmt.r_len == 5 &&
            px_fmt.g_offset == 5  && px_fmt.g_len == 6 &&
            px_fmt.b_offset == 0  && px_fmt.b_len == 5) {
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                for (uint32_t x = 0; x < copy_width; x++) {
                    uint8_t r = src[0];
                    uint8_t g = src[1];
                    uint8_t b = src[2];
                    temp_row[x] = ((uint16_t)(r & 0xF8) << 8) |
                                  ((uint16_t)(g & 0xFC) << 3) |
                                  (b >> 3);
                    src += 3;
                }
                uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 2;
                memcpy(dst_row, temp_row, copy_width * sizeof(uint16_t));
            }
        }
        else {
            // Generic 16bpp: pack according to pixel format
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                for (uint32_t x = 0; x < copy_width; x++) {
                    uint16_t pix = 0;
                    pix |= (convert_channel(src[0], px_fmt.r_len) << px_fmt.r_offset);
                    pix |= (convert_channel(src[1], px_fmt.g_len) << px_fmt.g_offset);
                    pix |= (convert_channel(src[2], px_fmt.b_len) << px_fmt.b_offset);
                    temp_row[x] = pix;
                    src += 3;
                }
                uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 2;
                memcpy(dst_row, temp_row, copy_width * sizeof(uint16_t));
            }
        }
        free(temp_row);
    }
    else if (fb_bpp == 24) {
        uint8_t *temp_row = (uint8_t *)malloc(copy_width * 3);
        if (!temp_row) return;

        // Fast path: standard BGR888 (b_offset=0, g_offset=8, r_offset=16)
        if (px_fmt.r_offset == 16 && px_fmt.r_len == 8 &&
            px_fmt.g_offset == 8  && px_fmt.g_len == 8 &&
            px_fmt.b_offset == 0  && px_fmt.b_len == 8) {
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                for (uint32_t x = 0; x < copy_width; x++) {
                    temp_row[x * 3 + 0] = src[2]; // B
                    temp_row[x * 3 + 1] = src[1]; // G
                    temp_row[x * 3 + 2] = src[0]; // R
                    src += 3;
                }
                uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 3;
                memcpy(dst_row, temp_row, copy_width * 3);
            }
        }
        else {
            // Generic 24bpp
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                for (uint32_t x = 0; x < copy_width; x++) {
                    uint32_t pix = 0;
                    pix |= (convert_channel(src[0], px_fmt.r_len) << px_fmt.r_offset);
                    pix |= (convert_channel(src[1], px_fmt.g_len) << px_fmt.g_offset);
                    pix |= (convert_channel(src[2], px_fmt.b_len) << px_fmt.b_offset);
                    temp_row[x * 3 + 0] = pix & 0xFF;
                    temp_row[x * 3 + 1] = (pix >> 8) & 0xFF;
                    temp_row[x * 3 + 2] = (pix >> 16) & 0xFF;
                    src += 3;
                }
                uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 3;
                memcpy(dst_row, temp_row, copy_width * 3);
            }
        }
        free(temp_row);
    }
    else if (fb_bpp == 32) {
        uint32_t *temp_row = (uint32_t *)malloc(copy_width * sizeof(uint32_t));
        if (!temp_row) return;

        // Fast path: standard BGRA8888 (b=0, g=8, r=16, a=24)
        if (px_fmt.r_offset == 16 && px_fmt.r_len == 8 &&
            px_fmt.g_offset == 8  && px_fmt.g_len == 8 &&
            px_fmt.b_offset == 0  && px_fmt.b_len == 8) {
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
                for (uint32_t x = 0; x < copy_width; x++) {
                    temp_row[x] = 0xFF000000U |
                                  ((uint32_t)src[2] << 16) |
                                  ((uint32_t)src[1] << 8) |
                                  src[0];
                    src += 3;
                }
                uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 4;
                memcpy(dst_row, temp_row, copy_width * sizeof(uint32_t));
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
                for (uint32_t x = 0; x < copy_width; x++) {
                    temp_row[x] = ((uint32_t)src[0] << 24) |
                                  ((uint32_t)src[1] << 16) |
                                  ((uint32_t)src[2] << 8) |
                                  0xFF;
                    src += 3;
                }
                uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 4;
                memcpy(dst_row, temp_row, copy_width * sizeof(uint32_t));
            }
        }
        else {
            // Generic 32bpp
            for (uint32_t y = 0; y < view_height; y++) {
                uint32_t src_y = y0 + y;
                int32_t dst_y = offset_y + (int32_t)y;
                if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

                uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
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
                    temp_row[x] = pix;
                    src += 3;
                }
                uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 4;
                memcpy(dst_row, temp_row, copy_width * sizeof(uint32_t));
            }
        }
        free(temp_row);
    }
    else {
        // 8bpp or other: grayscale fallback, row-by-row
        uint8_t *temp_row = (uint8_t *)malloc(copy_width);
        if (!temp_row) return;

        for (uint32_t y = 0; y < view_height; y++) {
            uint32_t src_y = y0 + y;
            int32_t dst_y = offset_y + (int32_t)y;
            if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

            uint8_t *src = frame_buffer_rgb888 + (src_y * fb_width_in + x0) * 3;
            for (uint32_t x = 0; x < copy_width; x++) {
                temp_row[x] = (uint8_t)((src[0] * 77 + src[1] * 150 + src[2] * 29) >> 8);
                src += 3;
            }
            uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x;
            memcpy(dst_row, temp_row, copy_width);
        }
        free(temp_row);
    }
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

    // Fast path: physical framebuffer is native RGB565, memcpy row-by-row
    if (fb_bpp == 16 &&
        px_fmt.r_offset == 11 && px_fmt.r_len == 5 &&
        px_fmt.g_offset == 5  && px_fmt.g_len == 6 &&
        px_fmt.b_offset == 0  && px_fmt.b_len == 5) {
        for (uint32_t y = 0; y < view_height; y++) {
            uint32_t src_y = y0 + y;
            int32_t dst_y = offset_y + (int32_t)y;
            if (dst_y < 0 || dst_y >= (int32_t)fb_height) continue;

            uint16_t *src_row = frame_buffer_rgb565 + src_y * fb_width_in + x0;
            uint8_t *dst_row = fb_mmap + dst_y * fb_line_length + offset_x * 2;
            memcpy(dst_row, src_row, view_width * sizeof(uint16_t));
        }
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
