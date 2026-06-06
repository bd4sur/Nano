#include "graphics.h"

#include "display_hal.h"

#include "glyph.h"


// 图像缓存项结构
typedef struct {
    char path[256];           // 图像路径（key）
    uint8_t *data;            // resize 后的图像数据（RGB888）
    uint32_t width;           // 图像宽度
    uint32_t height;          // 图像高度
    uint8_t valid;            // 此项是否有效
} ImageCacheEntry;

// 图像缓存数组（简单的固定大小缓存，LRU 淘汰）
static ImageCacheEntry s_image_cache[GFX_IMAGE_CACHE_SIZE];
static uint8_t s_cache_initialized = 0;

// 初始化缓存
static void init_image_cache(void) {
    if (s_cache_initialized) return;
    for (int i = 0; i < GFX_IMAGE_CACHE_SIZE; i++) {
        s_image_cache[i].path[0] = '\0';
        s_image_cache[i].data = NULL;
        s_image_cache[i].width = 0;
        s_image_cache[i].height = 0;
        s_image_cache[i].valid = 0;
    }
    s_cache_initialized = 1;
}

// 在缓存中查找图像，返回缓存项指针，未找到返回 NULL
static ImageCacheEntry* find_in_cache(const char *img_path, uint32_t width, uint32_t height) {
    for (int i = 0; i < GFX_IMAGE_CACHE_SIZE; i++) {
        if (s_image_cache[i].valid && 
            strcmp(s_image_cache[i].path, img_path) == 0 &&
            s_image_cache[i].width == width &&
            s_image_cache[i].height == height) {
            return &s_image_cache[i];
        }
    }
    return NULL;
}

// 将图像存入缓存（使用简单的 LRU 策略：将命中项移到最前，新项插入最前，淘汰最后）
static void add_to_cache(const char *img_path, uint8_t *data, uint32_t width, uint32_t height) {
    // 首先检查是否已存在（可能是 force_fetch 情况）
    ImageCacheEntry *existing = find_in_cache(img_path, width, height);
    if (existing) {
        // 更新现有缓存项的数据
        free(existing->data);
        existing->data = data;
        // 移到最前面（LRU）
        ImageCacheEntry temp = *existing;
        int idx = (int)(existing - s_image_cache);
        for (int i = idx; i > 0; i--) {
            s_image_cache[i] = s_image_cache[i - 1];
        }
        s_image_cache[0] = temp;
        return;
    }

    // 淘汰最后一个缓存项（如果已满）
    if (s_image_cache[GFX_IMAGE_CACHE_SIZE - 1].valid) {
        free(s_image_cache[GFX_IMAGE_CACHE_SIZE - 1].data);
    }

    // 将现有项后移
    for (int i = GFX_IMAGE_CACHE_SIZE - 1; i > 0; i--) {
        s_image_cache[i] = s_image_cache[i - 1];
    }

    // 新项插入最前
    s_image_cache[0].valid = 1;
    strncpy(s_image_cache[0].path, img_path, sizeof(s_image_cache[0].path) - 1);
    s_image_cache[0].path[sizeof(s_image_cache[0].path) - 1] = '\0';
    s_image_cache[0].data = data;
    s_image_cache[0].width = width;
    s_image_cache[0].height = height;
}




static inline uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    // 方法1
    // return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);

    // 方法2
    uint8_t r5 = (r >= 252) ? 31 : (r + 4) >> 3;
    uint8_t g6 = (g >= 254) ? 63 : (g + 2) >> 2;
    uint8_t b5 = (b >= 252) ? 31 : (b + 4) >> 3;
    return (r5 << 11) | (g6 << 5) | b5;

    // 方法3
    // uint8_t r5 = (r * 31 + 127) / 255;  // 0→0, 255→31，中间均匀分布
    // uint8_t g6 = (g * 63 + 127) / 255;  // 0→0, 255→63
    // uint8_t b5 = (b * 31 + 127) / 255;
    // return (r5 << 11) | (g6 << 5) | b5;
}

// #define RGB565_R(v) ((uint8_t)(((uint16_t)(v) & 0xF800) >> 8))
// #define RGB565_G(v) ((uint8_t)(((uint16_t)(v) & 0x07E0) >> 3))
// #define RGB565_B(v) ((uint8_t)(((uint16_t)(v) & 0x001F) << 3))

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

/*
static void convert_rgb888_to_rgb565(uint8_t *rgb888, uint16_t *rgb565, int width, int height) {
    uint8_t *p = rgb888;
    uint16_t *out = rgb565;
    for (int i = 0; i < width * height; i++) {
        uint8_t r = p[0];
        uint8_t g = p[1];
        uint8_t b = p[2];
        *out++ = rgb888_to_rgb565(r, g, b);
        p += 3;
    }
}

static void convert_rgb565_to_rgb888(uint16_t *rgb565, uint8_t *rgb888, int width, int height) {
    uint16_t *p = rgb565;
    uint8_t *out = rgb888;
    for (int i = 0; i < width * height; i++) {
        uint16_t v = p[i];
        *out++ = RGB565_R(v);
        *out++ = RGB565_G(v);
        *out++ = RGB565_B(v);
    }
}
*/

void convert_rgb888_to_rgb565_double(Nano_GFX *gfx, uint8_t *rgb888, int32_t width, int32_t height) {
    uint8_t *p = rgb888;
    uint32_t half_height = gfx->height / 2;
    for (int32_t y = 0; y < height; y++) {
        uint16_t *out;
        uint32_t row_offset;
        if (y < (int32_t)half_height) {
            out = gfx->frame_buffer_rgb565_top;
            row_offset = y * gfx->width;
        } else {
            out = gfx->frame_buffer_rgb565_bottom;
            row_offset = (y - half_height) * gfx->width;
        }
        for (int32_t x = 0; x < width; x++) {
            uint8_t r = p[0];
            uint8_t g = p[1];
            uint8_t b = p[2];
            out[row_offset + x] = rgb888_to_rgb565(r, g, b);
            p += 3;
        }
    }
}

void gfx_test(Nano_GFX *gfx) {
    int32_t width = gfx->width;
    int32_t height = gfx->height;
    for (int32_t x = 0; x < width; x++) {
        float t = (float)x / (float)width;
        uint8_t r = (uint8_t)floorf(t * 256.0f);
        uint8_t g = (uint8_t)floorf(t * 256.0f);
        uint8_t b = (uint8_t)floorf(t * 256.0f);
        gfx_draw_line(gfx, x, 0, x, 59, r, g, b, 1);
        gfx_draw_line(gfx, x, 60, x, 119, r, 0, 0, 1);
        gfx_draw_line(gfx, x, 120, x, 179, 0, g, 0, 1);
        gfx_draw_line(gfx, x, 180, x, 239, 0, 0, b, 1);
    }
}

static uint16_t* gfx_rgb565_ptr_double(Nano_GFX *gfx, uint32_t x, uint32_t y, uint32_t *offset) {
    uint32_t half_height = gfx->height / 2;
    if (y < half_height) {
        *offset = y * gfx->width + x;
        return gfx->frame_buffer_rgb565_top;
    } else {
        *offset = (y - half_height) * gfx->width + x;
        return gfx->frame_buffer_rgb565_bottom;
    }
}

static uint16_t* gfx_rgb565_ptr_single(Nano_GFX *gfx, uint32_t x, uint32_t y, uint32_t *offset) {
    *offset = y * gfx->width + x;
    return gfx->frame_buffer_rgb565;
}

void gfx_init(Nano_GFX *gfx, uint32_t width, uint32_t height, uint32_t color_mode) {
    gfx->color_mode = color_mode;
    gfx->width = width;
    gfx->height = height;
    
    if (!gfx->is_double_buffer) {
        if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
            gfx->frame_buffer_rgb888 = (uint8_t *)platform_calloc(width * height * 3, sizeof(uint8_t));
        }
        else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
            gfx->frame_buffer_rgb565 = (uint16_t *)platform_calloc(width * height, sizeof(uint16_t));
            gfx->frame_buffer_rgb565_top = gfx->frame_buffer_rgb565;
            gfx->frame_buffer_rgb565_bottom = gfx->frame_buffer_rgb565;
            gfx->rgb565_access = gfx_rgb565_ptr_single;
            // NOTE 仅供测试
            // gfx->frame_buffer_rgb888 = (uint8_t *)platform_calloc(width * height * 3, sizeof(uint8_t));
        }
        else return;
    } else {
        if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
            gfx->rgb565_access = gfx_rgb565_ptr_double;
        }
    }
    
    display_hal_init();

}


void gfx_close(Nano_GFX *gfx) {
    // display_hal_close();
}


void gfx_refresh(Nano_GFX *gfx) {
    if (gfx->is_double_buffer) {
        display_hal_refresh_rgb565_double(gfx->frame_buffer_rgb565_top, gfx->frame_buffer_rgb565_bottom, gfx->width, gfx->height, 0, 0, gfx->width, gfx->height);
    } else {
        if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
            display_hal_refresh(gfx->frame_buffer_rgb888, gfx->width, gfx->height, 0, 0, gfx->width, gfx->height);
        }
        else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
            display_hal_refresh_rgb565(gfx->frame_buffer_rgb565, gfx->width, gfx->height, 0, 0, gfx->width, gfx->height);
        }
    }
}

// 设置屏幕亮度(0-255)
void gfx_set_brightness(Nano_GFX *gfx, int32_t brightness) {
    display_set_brightness(brightness % 256);
}

// 清屏函数
void gfx_clear(Nano_GFX *gfx) {
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        memset(gfx->frame_buffer_rgb888, 0, gfx->width * gfx->height * 3);
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        if (gfx->is_double_buffer) {
            uint32_t half_height = gfx->height / 2;
            uint32_t top_pixels = gfx->width * half_height;
            uint32_t bottom_pixels = gfx->width * (gfx->height - half_height);
            if (gfx->frame_buffer_rgb565_top) memset(gfx->frame_buffer_rgb565_top, 0, top_pixels * sizeof(uint16_t));
            if (gfx->frame_buffer_rgb565_bottom) memset(gfx->frame_buffer_rgb565_bottom, 0, bottom_pixels * sizeof(uint16_t));
        } else {
            memset(gfx->frame_buffer_rgb565, 0, gfx->width * gfx->height * sizeof(uint16_t));
        }
    }
    gfx_refresh(gfx);
}

// 清屏函数
void gfx_soft_clear(Nano_GFX *gfx) {
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        memset(gfx->frame_buffer_rgb888, 0, gfx->width * gfx->height * 3);
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        if (gfx->is_double_buffer) {
            uint32_t half_height = gfx->height / 2;
            uint32_t top_pixels = gfx->width * half_height;
            uint32_t bottom_pixels = gfx->width * (gfx->height - half_height);
            if (gfx->frame_buffer_rgb565_top) memset(gfx->frame_buffer_rgb565_top, 0, top_pixels * sizeof(uint16_t));
            if (gfx->frame_buffer_rgb565_bottom) memset(gfx->frame_buffer_rgb565_bottom, 0, bottom_pixels * sizeof(uint16_t));
        } else {
            memset(gfx->frame_buffer_rgb565, 0, gfx->width * gfx->height * sizeof(uint16_t));
        }
    }
}

// 用纯白色填充整个屏幕
void gfx_fill_white(Nano_GFX *gfx) {
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        memset(gfx->frame_buffer_rgb888, 255, gfx->width * gfx->height * 3);
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        if (gfx->is_double_buffer) {
            uint32_t half_height = gfx->height / 2;
            uint32_t top_pixels = gfx->width * half_height;
            uint32_t bottom_pixels = gfx->width * (gfx->height - half_height);
            if (gfx->frame_buffer_rgb565_top) memset(gfx->frame_buffer_rgb565_top, 0xFF, top_pixels * sizeof(uint16_t));
            if (gfx->frame_buffer_rgb565_bottom) memset(gfx->frame_buffer_rgb565_bottom, 0xFF, bottom_pixels * sizeof(uint16_t));
        } else {
            memset(gfx->frame_buffer_rgb565, 0xFF, gfx->width * gfx->height * sizeof(uint16_t));
        }
    }
}


void gfx_get_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t *r, uint8_t *g, uint8_t *b) {
    if (x < 0 || y < 0 || x >= gfx->width || y >= gfx->height) {
        *r = 0; *g = 0; *b = 0;
        return;
    }
    else {
        if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
            uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
            uint32_t fb_width = gfx->width;
            uint32_t i = (y * fb_width + x) * 3;
            *r = frame_buffer[i+0];
            *g = frame_buffer[i+1];
            *b = frame_buffer[i+2];
        }
        else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
            uint32_t i;
            uint16_t *frame_buffer = gfx->rgb565_access(gfx, x, y, &i);
            uint16_t v = frame_buffer[i];
            *r = RGB565_R(v);
            *g = RGB565_G(v);
            *b = RGB565_B(v);
        }
    }
}

// 设置像素
inline void gfx_set_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
        uint32_t fb_width = gfx->width;
        uint32_t i = (y * fb_width + x) * 3;
        frame_buffer[ i ] = MIN(255, r);
        frame_buffer[i+1] = MIN(255, g);
        frame_buffer[i+2] = MIN(255, b);
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        uint32_t i;
        uint16_t *frame_buffer = gfx->rgb565_access(gfx, x, y, &i);
        frame_buffer[i] = rgb888_to_rgb565(MIN(255, r), MIN(255, g), MIN(255, b));
    }
}

// 叠加像素
inline void gfx_add_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
        uint32_t fb_width = gfx->width;
        uint32_t i = (y * fb_width + x) * 3;
        frame_buffer[ i ] = MIN(255, frame_buffer[ i ] + r);
        frame_buffer[i+1] = MIN(255, frame_buffer[i+1] + g);
        frame_buffer[i+2] = MIN(255, frame_buffer[i+2] + b);
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        uint32_t i;
        uint16_t *frame_buffer = gfx->rgb565_access(gfx, x, y, &i);
        uint16_t v = frame_buffer[i];
        frame_buffer[i] = rgb888_to_rgb565(
            MIN(255, RGB565_R(v) + r),
            MIN(255, RGB565_G(v) + g),
            MIN(255, RGB565_B(v) + b));
    }
}

// Alpha 混合像素
// a: 0=全透明(保留背景), 255=不透明(完全覆盖为前景色)
inline void gfx_blend_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    if (a == 0) return;
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
        uint32_t fb_width = gfx->width;
        uint32_t i = (y * fb_width + x) * 3;
        if (a == 255) {
            frame_buffer[i]   = r;
            frame_buffer[i+1] = g;
            frame_buffer[i+2] = b;
        } else {
            uint16_t t = a;
            uint16_t inv_t = 255 - a;
            frame_buffer[i]   = (uint8_t)((t * r + inv_t * frame_buffer[i]   + 127) / 255);
            frame_buffer[i+1] = (uint8_t)((t * g + inv_t * frame_buffer[i+1] + 127) / 255);
            frame_buffer[i+2] = (uint8_t)((t * b + inv_t * frame_buffer[i+2] + 127) / 255);
        }
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        uint32_t i;
        uint16_t *frame_buffer = gfx->rgb565_access(gfx, x, y, &i);
        if (a == 255) {
            frame_buffer[i] = rgb888_to_rgb565(r, g, b);
        } else {
            uint16_t v = frame_buffer[i];
            uint16_t t = a;
            uint16_t inv_t = 255 - a;
            uint8_t br = RGB565_R(v);
            uint8_t bg = RGB565_G(v);
            uint8_t bb = RGB565_B(v);
            uint8_t nr = (uint8_t)((t * r + inv_t * br + 127) / 255);
            uint8_t ng = (uint8_t)((t * g + inv_t * bg + 127) / 255);
            uint8_t nb = (uint8_t)((t * b + inv_t * bb + 127) / 255);
            frame_buffer[i] = rgb888_to_rgb565(nr, ng, nb);
        }
    }
}

// 数乘像素
inline void gfx_scale_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y, float k) {
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
        uint32_t fb_width = gfx->width;
        uint32_t i = (y * fb_width + x) * 3;
        frame_buffer[ i ] = MIN(255, (uint8_t)(k * (float)frame_buffer[ i ]));
        frame_buffer[i+1] = MIN(255, (uint8_t)(k * (float)frame_buffer[i+1]));
        frame_buffer[i+2] = MIN(255, (uint8_t)(k * (float)frame_buffer[i+2]));
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        uint32_t i;
        uint16_t *frame_buffer = gfx->rgb565_access(gfx, x, y, &i);
        uint16_t v = frame_buffer[i];
        frame_buffer[i] = rgb888_to_rgb565(
            MIN(255, (uint8_t)(k * (float)(RGB565_R(v)))),
            MIN(255, (uint8_t)(k * (float)(RGB565_G(v)))),
            MIN(255, (uint8_t)(k * (float)(RGB565_B(v)))));
    }
}

// Gamma 校正
void gfx_gamma(Nano_GFX *gfx, float gamma) {
    if (gamma <= 0.0f || gamma == 1.0f) return;

    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
        if (!frame_buffer) return;
        uint32_t n = fb_width * fb_height * 3;
        for (uint32_t i = 0; i < n; i++) {
            float v = frame_buffer[i] / 255.0f;
            float c = powf(v, gamma) * 255.0f + 0.5f;
            frame_buffer[i] = (uint8_t)MAX(0.0f, MIN(255.0f, c));
        }
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        uint16_t *frame_buffer = gfx->frame_buffer_rgb565;
        if (!frame_buffer) return;
        uint32_t n = fb_width * fb_height;
        for (uint32_t i = 0; i < n; i++) {
            uint16_t v = frame_buffer[i];
            float r = RGB565_R(v) / 255.0f;
            float g = RGB565_G(v) / 255.0f;
            float b = RGB565_B(v) / 255.0f;
            uint8_t rc = (uint8_t)(MAX(0.0f, MIN(255.0f, powf(r, gamma) * 255.0f + 0.5f)));
            uint8_t gc = (uint8_t)(MAX(0.0f, MIN(255.0f, powf(g, gamma) * 255.0f + 0.5f)));
            uint8_t bc = (uint8_t)(MAX(0.0f, MIN(255.0f, powf(b, gamma) * 255.0f + 0.5f)));
            frame_buffer[i] = rgb888_to_rgb565(rc, gc, bc);
        }
    }
}

// 反转像素
inline void gfx_reverse_pixel(Nano_GFX *gfx, uint32_t x, uint32_t y) {
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
        uint32_t fb_width = gfx->width;
        uint32_t i = (y * fb_width + x) * 3;
        uint8_t r = frame_buffer[i+0];
        uint8_t g = frame_buffer[i+1];
        uint8_t b = frame_buffer[i+2];
        frame_buffer[ i ] = (r == 0) ? 255 : 0;
        frame_buffer[i+1] = (g == 0) ? 255 : 0;
        frame_buffer[i+2] = (b == 0) ? 255 : 0;
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        uint32_t i;
        uint16_t *frame_buffer = gfx->rgb565_access(gfx, x, y, &i);
        frame_buffer[i] = (frame_buffer[i] == 0) ? 0xFFFF : 0;
    }
}

// 画点
// mode: 0-置黑  1-置色  2-异或  3-加色 >=4-Alpha混合
void gfx_draw_point(Nano_GFX *gfx, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    if (x < 0 || y < 0 || x >= gfx->width || y >= gfx->height) {
        return;
    }

    // 置0
    if (mode == 0) {
        gfx_set_pixel(gfx, x, y, 0, 0, 0);
    }
    // 正常置位
    else if (mode == 1) {
        gfx_set_pixel(gfx, x, y, red, green, blue);
    }
    // 反转
    else if (mode == 2) {
        gfx_reverse_pixel(gfx, x, y);
    }
    // 叠加
    else if (mode == 3) {
        gfx_add_pixel(gfx, x, y, red, green, blue);
    }
    // Alpha混合
    else {
        gfx_blend_pixel(gfx, x, y, red, green, blue, mode);
    }
}

// 画线
// x1,y1:起点坐标
// x2,y2:结束坐标
void gfx_draw_line(Nano_GFX *gfx, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {

    // 垂直线
    if (x1 == x2 && y1 != y2) {
        int32_t delta = y2 - y1;
        for (int32_t y = y1; ((delta >= 0) ? (y <= y2) : (y >= y2)); ((delta >= 0) ? (y++) : (y--))) {
            gfx_draw_point(gfx, x1, y, red, green, blue, mode);
        }
    }
    // 水平线（或一点）
    else if (y1 == y2) {
        int32_t delta = x2 - x1;
        for (int32_t x = x1; ((delta >= 0) ? (x <= x2) : (x >= x2)); ((delta >= 0) ? (x++) : (x--))) {
            gfx_draw_point(gfx, x, y1, red, green, blue, mode);
        }
    }
    // 斜线
    else {
        int xerr = 0, yerr = 0, delta_x, delta_y, distance;
        int incx, incy, uRow, uCol;
        delta_x = x2 - x1; // 计算坐标增量
        delta_y = y2 - y1;
        uRow = x1; // 画线起点坐标
        uCol = y1;

        if (delta_x > 0)
            incx = 1; // 设置单步方向
        else if (delta_x == 0)
            incx = 0; // 垂直线
        else {
            incx = -1;
            delta_x = -delta_x;
        }

        if (delta_y > 0)
            incy = 1;
        else if (delta_y == 0)
            incy = 0; // 水平线
        else {
            incy = -1;
            delta_y = -delta_y;
        }

        if (delta_x > delta_y)
            distance = delta_x; // 选取基本增量坐标轴
        else
            distance = delta_y;

        for (int32_t t = 0; t < distance + 1; t++) {
            gfx_draw_point(gfx, uRow, uCol, red, green, blue, mode);
            xerr += delta_x;
            yerr += delta_y;
            if (xerr > distance) {
                xerr -= distance;
                uRow += incx;
            }
            if (yerr > distance) {
                yerr -= distance;
                uCol += incy;
            }
        }
    }
}


static inline float _fpart(float x) { return x - floorf(x); }
static inline float _rfpart(float x) { return 1.0f - _fpart(x); }

static void gfx_draw_line_xiaolin_wu(Nano_GFX *gfx,
    float x0, float y0, float x1, float y1,
    uint8_t r, uint8_t g, uint8_t b, uint8_t mode
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    int steep = fabsf(y1 - y0) > fabsf(x1 - x0);

    if (steep) {
        float t = x0; x0 = y0; y0 = t;
        t = x1; x1 = y1; y1 = t;
    }

    if (x0 > x1) {
        float t = x0; x0 = x1; x1 = t;
        t = y0; y0 = y1; y1 = t;
    }

    float dx = x1 - x0;
    float dy = y1 - y0;
    float gradient = (dx == 0.0f) ? 0.0f : dy / dx;

    // steep 决定算法坐标系与屏幕坐标系的映射关系，从而确定各自的合法边界
    int32_t x_limit = steep ? fb_height : fb_width;   // 算法 x 方向对应屏幕的宽度/高度
    int32_t y_limit = steep ? fb_width  : fb_height;  // 算法 y 方向对应屏幕的高度/宽度

    // handle first endpoint
    float xend = roundf(x0);
    float yend = y0 + gradient * (xend - x0);
    float xgap = _rfpart(x0 + 0.5f);
    int xpxl1 = (int)xend;
    int ypxl1 = (int)floorf(yend);
    if (xpxl1 >= 0 && xpxl1 < x_limit) {
        if (steep) {
            if (ypxl1 >= 0 && ypxl1 < y_limit) {
                gfx_add_pixel(gfx, ypxl1,   xpxl1, r * _rfpart(yend) * xgap, g * _rfpart(yend) * xgap, b * _rfpart(yend) * xgap);
            }
            if (ypxl1 + 1 >= 0 && ypxl1 + 1 < y_limit) {
                gfx_add_pixel(gfx, ypxl1+1, xpxl1, r * _fpart(yend)  * xgap, g * _fpart(yend)  * xgap, b * _fpart(yend)  * xgap);
            }
        } else {
            if (ypxl1 >= 0 && ypxl1 < y_limit) {
                gfx_add_pixel(gfx, xpxl1, ypxl1,   r * _rfpart(yend) * xgap, g * _rfpart(yend) * xgap, b * _rfpart(yend) * xgap);
            }
            if (ypxl1 + 1 >= 0 && ypxl1 + 1 < y_limit) {
                gfx_add_pixel(gfx, xpxl1, ypxl1+1, r * _fpart(yend)  * xgap, g * _fpart(yend)  * xgap, b * _fpart(yend)  * xgap);
            }
        }
    }
    float intery = yend + gradient;

    // handle second endpoint
    xend = roundf(x1);
    yend = y1 + gradient * (xend - x1);
    xgap = _fpart(x1 + 0.5f);
    int xpxl2 = (int)xend;
    int ypxl2 = (int)floorf(yend);
    if (xpxl2 >= 0 && xpxl2 < x_limit) {
        if (steep) {
            if (ypxl2 >= 0 && ypxl2 < y_limit) {
                gfx_add_pixel(gfx, ypxl2,   xpxl2, r * _rfpart(yend) * xgap, g * _rfpart(yend) * xgap, b * _rfpart(yend) * xgap);
            }
            if (ypxl2 + 1 >= 0 && ypxl2 + 1 < y_limit) {
                gfx_add_pixel(gfx, ypxl2+1, xpxl2, r * _fpart(yend)  * xgap, g * _fpart(yend)  * xgap, b * _fpart(yend)  * xgap);
            }
        } else {
            if (ypxl2 >= 0 && ypxl2 < y_limit) {
                gfx_add_pixel(gfx, xpxl2, ypxl2,   r * _rfpart(yend) * xgap, g * _rfpart(yend) * xgap, b * _rfpart(yend) * xgap);
            }
            if (ypxl2 + 1 >= 0 && ypxl2 + 1 < y_limit) {
                gfx_add_pixel(gfx, xpxl2, ypxl2+1, r * _fpart(yend)  * xgap, g * _fpart(yend)  * xgap, b * _fpart(yend)  * xgap);
            }
        }
    }

    // main loop：先对 x 范围做裁剪，消除循环变量本身的越界
    int x_start = xpxl1 + 1;
    int x_end = xpxl2 - 1;
    if (x_start < 0) x_start = 0;
    if (x_end >= x_limit) x_end = x_limit - 1;

    if (steep) {
        for (int x = x_start; x <= x_end; x++) {
            int y_base = (int)floorf(intery);
            if (y_base >= 0 && y_base < y_limit) {
                gfx_add_pixel(gfx, y_base,   x, r * _rfpart(intery), g * _rfpart(intery), b * _rfpart(intery));
            }
            if (y_base + 1 >= 0 && y_base + 1 < y_limit) {
                gfx_add_pixel(gfx, y_base+1, x, r * _fpart(intery),  g * _fpart(intery),  b * _fpart(intery));
            }
            intery += gradient;
        }
    } else {
        for (int x = x_start; x <= x_end; x++) {
            int y_base = (int)floorf(intery);
            if (y_base >= 0 && y_base < y_limit) {
                gfx_add_pixel(gfx, x, y_base,   r * _rfpart(intery), g * _rfpart(intery), b * _rfpart(intery));
            }
            if (y_base + 1 >= 0 && y_base + 1 < y_limit) {
                gfx_add_pixel(gfx, x, y_base+1, r * _fpart(intery),  g * _fpart(intery),  b * _fpart(intery));
            }
            intery += gradient;
        }
    }
}

void gfx_draw_line_anti_aliasing(Nano_GFX *gfx,
    float x1, float y1, float x2, float y2, float line_width,
    uint8_t r, uint8_t g, uint8_t b, uint8_t mode
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    if (line_width <= 0.0f) return;

    // 确保颜色在 [0, 255] 范围内
    uint8_t cr = MAX(0, MIN(255, r));
    uint8_t cg = MAX(0, MIN(255, g));
    uint8_t cb = MAX(0, MIN(255, b));

    // 线宽为1时，使用吴小林算法以获得更锐利的抗锯齿效果
    if (line_width <= 1.0f) {
        gfx_draw_line_xiaolin_wu(gfx, x1, y1, x2, y2, cr, cg, cb, mode);
        return;
    }

    float dx = x2 - x1;
    float dy = y2 - y1;
    float len_sq = dx * dx + dy * dy;

    // 退化为点：绘制圆形
    if (len_sq == 0.0f) { // TODO 与0比较
        float radius = line_width / 2.0f;
        float r_sq = radius * radius;
        int32_t xMin = MAX(0, (int32_t)floorf(x1 - radius));
        int32_t xMax = MIN(fb_width - 1, (int32_t)ceilf(x1 + radius));
        int32_t yMin = MAX(0, (int32_t)floorf(y1 - radius));
        int32_t yMax = MIN(fb_height - 1, (int32_t)ceilf(y1 + radius));

        for (int32_t y = yMin; y <= yMax; y++) {
            for (int32_t x = xMin; x <= xMax; x++) {
                float dist_sq = (float)((x - x1) * (x - x1) + (y - y1) * (y - y1));
                if (dist_sq <= r_sq) {
                    gfx_add_pixel(gfx, x, y, cr, cg, cb);
                }
            }
        }
        return;
    }

    float len = sqrtf(len_sq);
    float inv_len = 1.0f / len;
    float nx = -dy * inv_len; // 法向量（垂直于线段）
    float ny = dx * inv_len;

    (void)nx; (void)ny;

    float half_w = line_width / 2.0f;

    // 包围盒（含线宽）
    int32_t xMin = MAX(0, (int32_t)floorf(MIN(x1, x2) - half_w));
    int32_t xMax = MIN(fb_width - 1, (int32_t)ceilf(MAX(x1, x2) + half_w));
    int32_t yMin = MAX(0, (int32_t)floorf(MIN(y1, y2) - half_w));
    int32_t yMax = MIN(fb_height - 1, (int32_t)ceilf(MAX(y1, y2) + half_w));

    for (int32_t y = yMin; y <= yMax; y++) {
        for (int32_t x = xMin; x <= xMax; x++) {
            // 计算点 (x, y) 到线段的有符号距离
            int32_t px = x - x1;
            int32_t py = y - y1;

            // 投影长度（参数 t）
            float t = (float)(px * dx + py * dy) / len_sq;
            float closest_x = x1;
            float closest_y = y1;

            if (t < 0.0f) {
                closest_x = x1;
                closest_y = y1;
            }
            else if (t > 1.0f) {
                closest_x = x2;
                closest_y = y2;
            }
            else {
                closest_x = x1 + t * dx;
                closest_y = y1 + t * dy;
            }

            float dist = sqrtf((x - closest_x) * (x - closest_x) + (y - closest_y) * (y - closest_y));

            if (dist > half_w) continue;

            // 抗锯齿：边缘平滑过渡
            float alpha = 1.0f;
            float edge_fade = MIN(1.0f, half_w); // 自适应边缘宽度，防止细线中心像素过淡
            if (dist > half_w - edge_fade) {
                alpha = (half_w - dist) / edge_fade;
                alpha = MAX(0.0f, MIN(1.0f, alpha));
            }
            gfx_add_pixel(gfx, x, y, cr * alpha, cg * alpha, cb * alpha);
        }
    }
}




void gfx_draw_rectangle(Nano_GFX *gfx, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    for (uint32_t y = y0; y < MIN(gfx->height, y0 + height); y++) {
        for (uint32_t x = x0; x < MIN(gfx->width, x0 + width); x++) {
            gfx_draw_point(gfx, x, y, red, green, blue, mode);
        }
    }
}

// x,y:圆心坐标
// r:圆的半径
void gfx_draw_circle(Nano_GFX *gfx, uint32_t cx, uint32_t cy, uint32_t r, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    if (r == 0) {
        gfx_draw_point(gfx, cx, cy, red, green, blue, mode);
        return;
    }

    int32_t x = 0;
    int32_t y = r;
    int32_t d = 3 - 2 * r;  // 更稳健的初始决策参数（Bresenham 形式）

    while (x <= y) {
        // 定义8个对称点
        int32_t points[8][2] = {
            { cx + x, cy + y },
            { cx + y, cy + x },
            { cx - x, cy + y },
            { cx - y, cy + x },
            { cx + x, cy - y },
            { cx + y, cy - x },
            { cx - x, cy - y },
            { cx - y, cy - x }
        };

        for (int i = 0; i < 8; i++) {
            int32_t px = points[i][0];
            int32_t py = points[i][1];
            // 严格裁剪到屏幕范围
            if (px >= 0 && px < gfx->width && py >= 0 && py < gfx->height) {
                gfx_draw_point(gfx, (uint32_t)px, (uint32_t)py, red, green, blue, mode);
            }
        }

        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y--;
        }
        x++;
    }
}

// 画实心圆
// cx,cy:圆心坐标
// r:圆的半径
void gfx_draw_circle_fill(Nano_GFX *gfx, uint32_t cx, uint32_t cy, uint32_t r, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    if (r == 0) {
        gfx_draw_point(gfx, cx, cy, red, green, blue, mode);
        return;
    }

    int32_t r_sq = (int32_t)r * (int32_t)r;
    int32_t cx_i = (int32_t)cx;
    int32_t cy_i = (int32_t)cy;

    for (int32_t y = cy_i - (int32_t)r; y <= cy_i + (int32_t)r; y++) {
        if (y < 0 || y >= (int32_t)gfx->height) continue;
        int32_t dy = y - cy_i;
        int32_t dy_sq = dy * dy;
        for (int32_t x = cx_i - (int32_t)r; x <= cx_i + (int32_t)r; x++) {
            if (x < 0 || x >= (int32_t)gfx->width) continue;
            int32_t dx = x - cx_i;
            if (dx * dx + dy_sq <= r_sq) {
                gfx_draw_point(gfx, (uint32_t)x, (uint32_t)y, red, green, blue, mode);
            }
        }
    }
}

// 显示汉字
// x,y:起点坐标
// mode:0,反色显示;1,正常显示
void gfx_draw_char(
    Nano_GFX *gfx, uint32_t x, uint32_t y, const uint8_t *glyph,
    uint8_t font_width, uint8_t font_height,
    uint8_t red, uint8_t green, uint8_t blue, uint8_t mode
) {
    int32_t row_bytes = (font_height + 8 - 1) / 8;
    int32_t col_bytes = font_width;
    for (int32_t j = 0; j < row_bytes; j++) {
        int32_t bits = (j == (row_bytes-1)) ? (8 - ((8 * row_bytes) % font_height)) : 8;
        for (int32_t i = 0; i < col_bytes; i++) {
            uint8_t g = glyph[j * col_bytes + i];
            for (int32_t b = 0; b < bits; b++) {
                uint32_t px = x + i;
                uint32_t py = y + j*8 + b;
                if (px < 0 || px >= gfx->width || py < 0 || py >= gfx->height) continue;
                // if ((g >> b) & 0x1) {
                //     gfx_set_pixel(gfx, px, py, red, green, blue);
                // }
                // else if (mode == 0) {
                //     gfx_add_pixel(gfx, px, py, red, green, blue);
                // }
                if ((g >> b) & 0x1) {
                    gfx_draw_point(gfx, (x+i), (y+j*8+b), red, green, blue, mode);
                }
                else if (mode == 0) {
                    gfx_draw_point(gfx, (x+i), (y+j*8+b), red, green, blue, !mode);
                }
            }
        }
    }
}


// 绘制一行文本
// mode: 1 - 正显，不修改底色
void gfx_draw_textline(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    uint32_t x_pos = x;
    uint32_t y_pos = y;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        const uint8_t *glyph = gfx_get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            // printf("出现了字库之外的字符！\n");
            glyph = gfx_get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos + font_width >= gfx->width) {
            break;
        }
        gfx_draw_char(gfx, x_pos, y_pos, glyph, font_width, font_height, red, green, blue, mode);
        x_pos += font_width;
    }
}


// 绘制一行文本，mode为1则为正显，为0则为反白
void gfx_draw_textline_centered(Nano_GFX *gfx, wchar_t *line, uint32_t cx, uint32_t cy, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {

    // 第一遍扫描：计算文本渲染长度
    int32_t total_width = 0;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        const uint8_t *glyph = gfx_get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = gfx_get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        total_width += font_width;
    }

    // 第二遍扫描：渲染
    int32_t x_pos = cx - (total_width/2);
    int32_t y_pos = cy - 6;
    if (y_pos < 0 || y_pos + 6 > gfx->height) return;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        const uint8_t *glyph = gfx_get_glyph(gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = gfx_get_glyph(gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos < 0) {
            x_pos += font_width;
            continue;
        }
        else if (x_pos + font_width > gfx->width) {
            break;
        }
        gfx_draw_char(gfx, x_pos, y_pos, glyph, font_width, font_height, red, green, blue, mode);
        x_pos += font_width;
    }
}


void gfx_draw_textline_mini(Nano_GFX *gfx, wchar_t *line, uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    const uint8_t mini_glyph[43][7] = {
        {3, 5, 0x1F, 0x11, 0x1F, 0x00, 0x00}, // 0
        {3, 5, 0x12, 0x1F, 0x10, 0x00, 0x00}, // 1
        {3, 5, 0x1D, 0x15, 0x17, 0x00, 0x00}, // 2
        {3, 5, 0x15, 0x15, 0x1F, 0x00, 0x00}, // 3
        {3, 5, 0x0F, 0x08, 0x1F, 0x00, 0x00}, // 4
        {3, 5, 0x17, 0x15, 0x1D, 0x00, 0x00}, // 5
        {3, 5, 0x1F, 0x15, 0x1D, 0x00, 0x00}, // 6
        {3, 5, 0x01, 0x01, 0x1F, 0x00, 0x00}, // 7
        {3, 5, 0x1F, 0x15, 0x1F, 0x00, 0x00}, // 8
        {3, 5, 0x17, 0x15, 0x1F, 0x00, 0x00}, // 9

        {3, 5, 0x1E, 0x05, 0x1E, 0x00, 0x00}, // A
        {3, 5, 0x1F, 0x15, 0x0A, 0x00, 0x00}, // B
        {3, 5, 0x0E, 0x11, 0x11, 0x00, 0x00}, // C
        {3, 5, 0x1F, 0x11, 0x0E, 0x00, 0x00}, // D
        {3, 5, 0x1F, 0x15, 0x11, 0x00, 0x00}, // E
        {3, 5, 0x1F, 0x05, 0x11, 0x00, 0x00}, // F
        {3, 5, 0x0E, 0x11, 0x1D, 0x00, 0x00}, // G
        {3, 5, 0x1F, 0x04, 0x1F, 0x00, 0x00}, // H
        {3, 5, 0x11, 0x1F, 0x11, 0x00, 0x00}, // I
        {3, 5, 0x08, 0x10, 0x0F, 0x00, 0x00}, // J
        {4, 5, 0x1F, 0x04, 0x0A, 0x11, 0x00}, // K
        {3, 5, 0x1F, 0x10, 0x10, 0x00, 0x00}, // L
        {5, 5, 0x1F, 0x02, 0x04, 0x02, 0x1F}, // M
        {5, 5, 0x1F, 0x02, 0x04, 0x08, 0x1F}, // N
        {3, 5, 0x0E, 0x11, 0x0E, 0x00, 0x00}, // O
        {3, 5, 0x1F, 0x05, 0x02, 0x00, 0x00}, // P
        {5, 5, 0x0E, 0x11, 0x15, 0x09, 0x16}, // Q
        {3, 5, 0x1F, 0x05, 0x1A, 0x00, 0x00}, // R
        {3, 5, 0x12, 0x15, 0x09, 0x00, 0x00}, // S
        {3, 5, 0x01, 0x1F, 0x01, 0x00, 0x00}, // T
        {3, 5, 0x1F, 0x10, 0x1F, 0x00, 0x00}, // U
        {3, 5, 0x0F, 0x10, 0x0F, 0x00, 0x00}, // V
        {5, 5, 0x0F, 0x10, 0x0C, 0x10, 0x0F}, // W
        {3, 5, 0x1B, 0x04, 0x1B, 0x00, 0x00}, // X
        {3, 5, 0x03, 0x0C, 0x03, 0x00, 0x00}, // Y
        {3, 5, 0x19, 0x15, 0x13, 0x00, 0x00}, // Z

        {3, 5, 0x00, 0x00, 0x00, 0x00, 0x00}, // 空格
        {3, 5, 0x10, 0x00, 0x00, 0x00, 0x00}, // .
        {3, 5, 0x04, 0x04, 0x04, 0x00, 0x00}, // -
        {3, 5, 0x00, 0x0A, 0x00, 0x00, 0x00}, // :
        {3, 5, 0x00, 0x0E, 0x11, 0x00, 0x00}, // (
        {3, 5, 0x11, 0x0E, 0x00, 0x00, 0x00}, // )
        {5, 5, 0x13, 0x0B, 0x04, 0x1A, 0x19}  // %
    };

    uint32_t x_pos = x;
    uint32_t y_pos = y;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        wchar_t current_char = line[i];
        uint8_t font_width = 3;
        uint8_t font_height = 5;
        uint8_t *glyph = NULL;
        if (current_char >= L'0' && current_char <= L'9') {
            glyph = (uint8_t *)mini_glyph[current_char - L'0'];
        }
        else if (current_char >= L'A' && current_char <= L'Z') {
            glyph = (uint8_t *)mini_glyph[current_char - L'A' + 10];
        }
        else if (current_char == L' ') {
            glyph = (uint8_t *)mini_glyph[36 + 0];
        }
        else if (current_char == L'.') {
            glyph = (uint8_t *)mini_glyph[36 + 1];
        }
        else if (current_char == L'-') {
            glyph = (uint8_t *)mini_glyph[36 + 2];
        }
        else if (current_char == L':') {
            glyph = (uint8_t *)mini_glyph[36 + 3];
        }
        else if (current_char == L'(') {
            glyph = (uint8_t *)mini_glyph[36 + 4];
        }
        else if (current_char == L')') {
            glyph = (uint8_t *)mini_glyph[36 + 5];
        }
        else if (current_char == L'%') {
            glyph = (uint8_t *)mini_glyph[36 + 6];
        }
        else if (current_char == L'\n') {
            x_pos = x;
            y_pos += font_height + 1;
            continue;
        }
        else {
            // 未知字符，跳过
            x_pos += font_width;
            continue;
        }
        if (x_pos + font_width >= gfx->width) {
            break;
        }
        font_width = glyph[0];
        font_height = glyph[1];
        for (uint32_t i = 0; i < font_width; i++) {
            uint8_t column_data = glyph[2 + i];
            // printf("Column = %d\n", column_data);
            for (uint32_t j = 0; j < font_height; j++) {
                if ((column_data >> j) & 0x01) {
                    gfx_draw_point(gfx, x_pos + i, y_pos + j, red, green, blue, mode);
                }
            }
        }
        x_pos += (font_width + 1); // 字符间隔1像素
    }
}


// 高性能水平线填充辅助函数
// 直接操作帧缓冲区，绕过 gfx_draw_point 的逐像素函数调用开销
static inline void gfx_fill_hline_fast(Nano_GFX *gfx, int32_t x_start, int32_t x_end, int32_t y,
    uint8_t r, uint8_t g, uint8_t b, uint8_t mode) {
    if (x_start > x_end) {
        int32_t t = x_start; x_start = x_end; x_end = t;
    }
    if (x_start >= (int32_t)gfx->width) return;
    if (x_end < 0) return;
    if (y < 0 || y >= (int32_t)gfx->height) return;
    if (x_start < 0) x_start = 0;
    if (x_end >= (int32_t)gfx->width) x_end = (int32_t)gfx->width - 1;
    if (x_start > x_end) return;

    int32_t count = x_end - x_start + 1;

    if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        uint32_t offset;
        uint16_t *fb = gfx->rgb565_access(gfx, (uint32_t)x_start, (uint32_t)y, &offset);

        if (mode == 1) {
            uint16_t color = rgb888_to_rgb565(r, g, b);
            while (count-- > 0) {
                fb[offset++] = color;
            }
        } else if (mode == 0) {
            while (count-- > 0) {
                fb[offset++] = 0;
            }
        } else if (mode == 2) {
            while (count-- > 0) {
                uint16_t v = fb[offset];
                fb[offset++] = (v == 0) ? 0xFFFF : 0;
            }
        } else { // mode == 3
            while (count-- > 0) {
                uint16_t v = fb[offset];
                fb[offset++] = rgb888_to_rgb565(
                    MIN(255, RGB565_R(v) + r),
                    MIN(255, RGB565_G(v) + g),
                    MIN(255, RGB565_B(v) + b));
            }
        }
    } else if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        uint8_t *fb = gfx->frame_buffer_rgb888;
        uint32_t idx = ((uint32_t)y * gfx->width + (uint32_t)x_start) * 3;

        if (mode == 1) {
            while (count-- > 0) {
                fb[idx++] = r;
                fb[idx++] = g;
                fb[idx++] = b;
            }
        } else if (mode == 0) {
            while (count-- > 0) {
                fb[idx++] = 0;
                fb[idx++] = 0;
                fb[idx++] = 0;
            }
        } else if (mode == 2) {
            while (count-- > 0) {
                uint8_t rr = fb[idx];
                uint8_t gg = fb[idx+1];
                uint8_t bb = fb[idx+2];
                fb[idx++] = (rr == 0) ? 255 : 0;
                fb[idx++] = (gg == 0) ? 255 : 0;
                fb[idx++] = (bb == 0) ? 255 : 0;
            }
        } else { // mode == 3
            while (count-- > 0) {
                fb[idx] = MIN(255, fb[idx] + r); idx++;
                fb[idx] = MIN(255, fb[idx] + g); idx++;
                fb[idx] = MIN(255, fb[idx] + b); idx++;
            }
        }
    }
}

// 绘制实心三角形
// 使用整数扫描线算法填充凸三角形（极致优化版）
// is_anti_aliasing: 保留参数但已去除抗锯齿功能（兼容性）
// mode: 0-置黑 1-置色 2-异或 3-加色
void gfx_draw_triangle(
    Nano_GFX *gfx,
    uint32_t x0, uint32_t y0,
    uint32_t x1, uint32_t y1,
    uint32_t x2, uint32_t y2,
    int32_t is_anti_aliasing,
    uint8_t red, uint8_t green, uint8_t blue, uint8_t mode
) {
    (void)is_anti_aliasing; // 保留参数但不再使用

    // 复制顶点并按 y 坐标排序（sy0 <= sy1 <= sy2）
    uint32_t sx0 = x0, sy0 = y0;
    uint32_t sx1 = x1, sy1 = y1;
    uint32_t sx2 = x2, sy2 = y2;

    #define TRI_SWAP(a, b) do { uint32_t _t = (a); (a) = (b); (b) = _t; } while(0)
    if (sy0 > sy1) { TRI_SWAP(sx0, sx1); TRI_SWAP(sy0, sy1); }
    if (sy1 > sy2) { TRI_SWAP(sx1, sx2); TRI_SWAP(sy1, sy2); }
    if (sy0 > sy1) { TRI_SWAP(sx0, sx1); TRI_SWAP(sy0, sy1); }

    // 退化三角形（共水平线或共点）
    if (sy0 == sy2) return;

    // Y 方向裁剪
    if (sy0 >= gfx->height) return;
    uint32_t sy2_orig = sy2;
    if (sy2 >= gfx->height) sy2 = gfx->height - 1;

    // 长边 (sx0,sy0) -> (sx2,sy2_orig)，使用 16.16 定点数
    // x_step = dx / dy，每行 y 增加 1 时 x 的增量
    int32_t long_dy = (int32_t)(sy2_orig - sy0);
    int32_t long_x  = (int32_t)sx0 << 16;
    int32_t long_step = (((int32_t)sx2 - (int32_t)sx0) << 16) / long_dy;

    // 上半部分：y 从 sy0 到 sy1-1
    if (sy0 < sy1) {
        int32_t top_dy = (int32_t)(sy1 - sy0);
        int32_t top_x  = (int32_t)sx0 << 16;
        int32_t top_step = (((int32_t)sx1 - (int32_t)sx0) << 16) / top_dy;

        uint32_t y_end = sy1;
        if (y_end > gfx->height) y_end = gfx->height;

        for (uint32_t y = sy0; y < y_end; y++) {
            int32_t x_left  = top_x >> 16;
            int32_t x_right = long_x >> 16;
            if (x_left > x_right) {
                int32_t t = x_left; x_left = x_right; x_right = t;
            }
            gfx_fill_hline_fast(gfx, x_left, x_right, (int32_t)y, red, green, blue, mode);
            top_x += top_step;
            long_x += long_step;
        }
    }

    // 下半部分：y 从 sy1 到 sy2（包含 sy2，确保平底三角形底边被填充）
    if (sy1 <= sy2) {
        int32_t bot_x = (int32_t)sx1 << 16;
        int32_t bot_step = 0;
        if (sy1 < sy2) {
            int32_t bot_dy = (int32_t)(sy2_orig - sy1);
            bot_step = (((int32_t)sx2 - (int32_t)sx1) << 16) / bot_dy;
        }

        // 将长边定位到 y = sy1 处（与上半部分自然衔接）
        long_x = ((int32_t)sx0 << 16) + long_step * ((int32_t)sy1 - (int32_t)sy0);

        uint32_t y_end = sy2;

        for (uint32_t y = sy1; y <= y_end; y++) {
            int32_t x_left  = bot_x >> 16;
            int32_t x_right = long_x >> 16;
            if (x_left > x_right) {
                int32_t t = x_left; x_left = x_right; x_right = t;
            }
            gfx_fill_hline_fast(gfx, x_left, x_right, (int32_t)y, red, green, blue, mode);
            bot_x += bot_step;
            long_x += long_step;
        }
    }

    #undef TRI_SWAP
}

// 绘制实心六边形
// 使用扫描线算法填充凸六边形
void gfx_draw_hexagon(
    Nano_GFX *gfx,
    uint32_t x1, uint32_t y1,
    uint32_t x2, uint32_t y2,
    uint32_t x3, uint32_t y3,
    uint32_t x4, uint32_t y4,
    uint32_t x5, uint32_t y5,
    uint32_t x6, uint32_t y6,
    uint8_t red, uint8_t green, uint8_t blue, uint8_t mode
) {
    // 将顶点存储到数组中便于处理
    uint32_t x_coords[6] = {x1, x2, x3, x4, x5, x6};
    uint32_t y_coords[6] = {y1, y2, y3, y4, y5, y6};
    
    // 计算六边形的垂直范围（限制在屏幕范围内）
    uint32_t y_min = y1;
    uint32_t y_max = y1;
    for (int i = 1; i < 6; i++) {
        if (y_coords[i] < y_min) y_min = y_coords[i];
        if (y_coords[i] > y_max) y_max = y_coords[i];
    }
    
    // 限制在画布范围内
    if (y_min >= gfx->height) return;
    if (y_max >= gfx->height) y_max = gfx->height - 1;
    
    // 对每一行扫描线，计算与六边形边的交点
    for (uint32_t y = y_min; y <= y_max; y++) {
        int32_t x_intersections[6];
        int num_intersections = 0;
        
        // 检查每条边与当前水平线的交点
        for (int i = 0; i < 6; i++) {
            uint32_t x_start = x_coords[i];
            uint32_t y_start = y_coords[i];
            uint32_t x_end = x_coords[(i + 1) % 6];
            uint32_t y_end = y_coords[(i + 1) % 6];
            
            // 跳过水平边（已在顶点处处理）
            if (y_start == y_end) continue;
            
            // 检查当前扫描线是否与这条边相交
            // 使用严格的包含-排除规则避免顶点重复计数
            int intersect = 0;
            if (y_start < y_end) {
                // 向上边：y_start <= y < y_end
                intersect = (y >= y_start && y < y_end) ? 1 : 0;
            } else {
                // 向下边：y_end < y <= y_start
                intersect = (y > y_end && y <= y_start) ? 1 : 0;
            }
            
            if (intersect) {
                // 线性插值计算交点的x坐标
                int32_t dy = (int32_t)y_end - (int32_t)y_start;
                int32_t dx = (int32_t)x_end - (int32_t)x_start;
                int32_t dy_scan = (int32_t)y - (int32_t)y_start;
                
                // x = x_start + dx * (y - y_start) / dy
                int32_t x_intersect = (int32_t)x_start + (dx * dy_scan) / dy;
                
                if (num_intersections < 6) {
                    x_intersections[num_intersections++] = x_intersect;
                }
            }
        }
        
        // 对交点进行排序（冒泡排序）
        for (int i = 0; i < num_intersections - 1; i++) {
            for (int j = i + 1; j < num_intersections; j++) {
                if (x_intersections[i] > x_intersections[j]) {
                    int32_t temp = x_intersections[i];
                    x_intersections[i] = x_intersections[j];
                    x_intersections[j] = temp;
                }
            }
        }
        
        // 填充交点之间的区域（凸多边形，成对填充）
        for (int i = 0; i < num_intersections; i += 2) {
            if (i + 1 >= num_intersections) break;
            
            int32_t x_start = x_intersections[i];
            int32_t x_end = x_intersections[i + 1];
            
            // 限制在画布范围内
            if (x_end < 0) continue;
            if (x_start >= (int32_t)gfx->width) continue;
            if (x_start < 0) x_start = 0;
            if (x_end >= (int32_t)gfx->width) x_end = gfx->width - 1;
            
            // 绘制扫描线段
            for (int32_t x = x_start; x <= x_end; x++) {
                gfx_draw_point(gfx, (uint32_t)x, y, red, green, blue, mode);
            }
        }
    }
}

const uint8_t *gfx_get_glyph(Nano_GFX *gfx, uint32_t utf32, uint8_t *font_width, uint8_t *font_height) {
    if (utf32 < 32) {
        return NULL;
    }
    else if (utf32 >= 32 && utf32 < 127) {
        *font_width = 6;
        *font_height = 12;
        return ASCII_6_12[utf32 - 32];
    }
    else {
        int32_t index = binary_search(UTF32_LUT_SORTED, UTF32_LUT_INDEXS, GLYPH_CHAR_NUM, utf32);
        if (index >= 0 && index < 7445) {
            *font_width = 12;
            *font_height = 12;
            return GB2312_12_12[index];
        }
        else {
            return NULL;
        }
    }
}









// 抖动：用于缓解低位深屏幕的色带现象

// 8x8 Bayer矩阵（预缩放至0~255范围，避免运行时乘法）
static const uint8_t bayer8x8[64] = {
    0,  192, 48,  240, 12,  204, 60,  252,
    128,64,  176,112,140,76,  224,160,
    32, 224, 16,  208, 44, 236, 28,  220,
    160,96,  144,80,  172,108,252,188,
    8,  200, 56,  248, 4,  196, 52,  244,
    136,72,  184,120,132,68,  216,152,
    40, 232, 24,  216, 36, 228, 20,  212,
    168,104,152,88,  180,116,244,180
};

// 量化误差补偿表（RGB565量化后还原的亮度偏移补偿）
static const int8_t quant_bias[8] = {0, 1, 1, 2, 2, 3, 3, 4}; // 经验值

void gfx_dithering(Nano_GFX *gfx) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    if (fb_width <= 0 || fb_height <= 0) return;

    if (gfx->color_mode == GFX_COLOR_MODE_RGB888) {
        uint8_t *frame_buffer = gfx->frame_buffer_rgb888;
        if (!frame_buffer) return;

        const int32_t stride = fb_width * 3;

        for (int32_t y = 0; y < fb_height; y++) {
            uint8_t *row = frame_buffer + y * stride;
            for (int32_t x = 0; x < fb_width; x++) {
                uint8_t *px = row + x * 3;
                uint8_t r = px[0], g = px[1], b = px[2];

                // 1. 获取Bayer阈值（归一化到0~7范围，匹配5/6bit量化步长）
                uint8_t threshold = bayer8x8[(y & 7) * 8 + (x & 7)] >> 5; // 0~7

                // 2. 应用阈值偏移（模拟误差扩散的视觉效果）
                int32_t r_adj = r + quant_bias[threshold];
                int32_t g_adj = g + quant_bias[threshold];
                int32_t b_adj = b + quant_bias[threshold];

                // 3. 模拟RGB565量化并还原
                uint8_t r5 = (r_adj > 255) ? 31 : (r_adj >> 3);
                uint8_t g6 = (g_adj > 255) ? 63 : (g_adj >> 2);
                uint8_t b5 = (b_adj > 255) ? 31 : (b_adj >> 3);

                px[0] = (r5 << 3) | (r5 >> 2);
                px[1] = (g6 << 2) | (g6 >> 4);
                px[2] = (b5 << 3) | (b5 >> 2);
            }
        }
    }
    else if (gfx->color_mode == GFX_COLOR_MODE_RGB565) {
        if (gfx->is_double_buffer) {
            uint32_t half_height = gfx->height / 2;

            // 上半区
            uint16_t *frame_buffer = gfx->frame_buffer_rgb565_top;
            if (frame_buffer) {
                for (int32_t y = 0; y < (int32_t)half_height; y++) {
                    for (int32_t x = 0; x < (int32_t)fb_width; x++) {
                        uint32_t i = y * fb_width + x;
                        uint16_t v = frame_buffer[i];

                        uint8_t r = RGB565_R(v);
                        uint8_t g = RGB565_G(v);
                        uint8_t b = RGB565_B(v);

                        // 1. 获取Bayer阈值（归一化到0~7范围，匹配5/6bit量化步长）
                        uint8_t threshold = bayer8x8[(y & 7) * 8 + (x & 7)] >> 5; // 0~7

                        // 2. 应用阈值偏移（模拟误差扩散的视觉效果）
                        int32_t r_adj = r + quant_bias[threshold];
                        int32_t g_adj = g + quant_bias[threshold];
                        int32_t b_adj = b + quant_bias[threshold];

                        // 3. 量化到RGB565
                        frame_buffer[i] = rgb888_to_rgb565(
                            (r_adj > 255) ? 255 : (uint8_t)r_adj,
                            (g_adj > 255) ? 255 : (uint8_t)g_adj,
                            (b_adj > 255) ? 255 : (uint8_t)b_adj
                        );
                    }
                }
            }

            // 下半区
            frame_buffer = gfx->frame_buffer_rgb565_bottom;
            if (frame_buffer) {
                for (int32_t y = 0; y < (int32_t)(gfx->height - half_height); y++) {
                    for (int32_t x = 0; x < (int32_t)fb_width; x++) {
                        uint32_t i = y * fb_width + x;
                        uint16_t v = frame_buffer[i];

                        uint8_t r = RGB565_R(v);
                        uint8_t g = RGB565_G(v);
                        uint8_t b = RGB565_B(v);

                        // 1. 获取Bayer阈值（归一化到0~7范围，匹配5/6bit量化步长）
                        uint8_t threshold = bayer8x8[((y + half_height) & 7) * 8 + (x & 7)] >> 5; // 0~7

                        // 2. 应用阈值偏移（模拟误差扩散的视觉效果）
                        int32_t r_adj = r + quant_bias[threshold];
                        int32_t g_adj = g + quant_bias[threshold];
                        int32_t b_adj = b + quant_bias[threshold];

                        // 3. 量化到RGB565
                        frame_buffer[i] = rgb888_to_rgb565(
                            (r_adj > 255) ? 255 : (uint8_t)r_adj,
                            (g_adj > 255) ? 255 : (uint8_t)g_adj,
                            (b_adj > 255) ? 255 : (uint8_t)b_adj
                        );
                    }
                }
            }
        } else {
            uint16_t *frame_buffer = gfx->frame_buffer_rgb565;
            if (!frame_buffer) return;

            for (int32_t y = 0; y < fb_height; y++) {
                for (int32_t x = 0; x < fb_width; x++) {
                    uint32_t i = y * fb_width + x;
                    uint16_t v = frame_buffer[i];

                    uint8_t r = RGB565_R(v);
                    uint8_t g = RGB565_G(v);
                    uint8_t b = RGB565_B(v);

                    // 1. 获取Bayer阈值（归一化到0~7范围，匹配5/6bit量化步长）
                    uint8_t threshold = bayer8x8[(y & 7) * 8 + (x & 7)] >> 5; // 0~7

                    // 2. 应用阈值偏移（模拟误差扩散的视觉效果）
                    int32_t r_adj = r + quant_bias[threshold];
                    int32_t g_adj = g + quant_bias[threshold];
                    int32_t b_adj = b + quant_bias[threshold];

                    // 3. 量化到RGB565
                    frame_buffer[i] = rgb888_to_rgb565(
                        (r_adj > 255) ? 255 : (uint8_t)r_adj,
                        (g_adj > 255) ? 255 : (uint8_t)g_adj,
                        (b_adj > 255) ? 255 : (uint8_t)b_adj
                    );
                }
            }
        }
    }
}



























#define STBI_MALLOC platform_malloc
#define STBI_REALLOC platform_realloc
#define STBI_FREE free
#define STBIR_MALLOC(size,user_data) platform_malloc(size)
#define STBIR_FREE(ptr,user_data) free(ptr)

// 定义宏以生成 stb 库的实现
#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "vendor/stb_image_resize2.h"

#include <string.h>

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#endif





// 从指定文件读取图像并绘制到 frame_buffer（带缓存）
// img_path: 图像文件路径
// x0, y0: 目标区域左上角坐标
// width, height: 目标区域宽高（图像将被缩放到此大小）
// is_force_fetch: 如果非0，忽略缓存，重新读取并resize图像，并更新缓存
void gfx_draw_image(Nano_GFX *gfx, char *img_path, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, uint8_t is_force_fetch) {
    // 初始化缓存
    init_image_cache();
    
    uint8_t *draw_data = NULL;
    
    if (!is_force_fetch) {
        // 尝试从缓存获取
        ImageCacheEntry *cached = find_in_cache(img_path, width, height);
        if (cached != NULL) {
            draw_data = cached->data;
            // 更新 LRU：将此缓存项移到最前面
            int idx = (int)(cached - s_image_cache);
            if (idx > 0) {
                ImageCacheEntry temp = s_image_cache[idx];
                for (int i = idx; i > 0; i--) {
                    s_image_cache[i] = s_image_cache[i - 1];
                }
                s_image_cache[0] = temp;
            }
        }
    }
    
    // 缓存未命中或强制刷新：从文件读取并resize
    if (draw_data == NULL) {
        int img_w, img_h, channels;
        
        // 加载图像文件，获取 RGB 数据（3通道）
        unsigned char *img_data = stbi_load(img_path, &img_w, &img_h, &channels, 3);
        if (img_data == NULL) {
            // 加载失败，直接返回
            return;
        }
        
        // 分配缩放后图像的内存（RGB888 格式，3字节/像素）
        draw_data = (uint8_t *)platform_malloc(width * height * 3);
        if (draw_data == NULL) {
            stbi_image_free(img_data);
            return;
        }
        
        // 缩放图像到目标尺寸
        stbir_resize_uint8_linear(
            img_data, img_w, img_h, 0,           // 输入图像数据、宽、高、行跨度（0表示连续）
            draw_data, (int)width, (int)height, 0,  // 输出图像数据、宽、高、行跨度
            STBIR_RGB                               // 像素布局：RGB
        );
        
        // 释放原始图像数据
        stbi_image_free(img_data);
        
        // 存入缓存（缓存接管内存所有权）
        add_to_cache(img_path, draw_data, width, height);
    }
    
    // 将图像绘制到 frame_buffer
    // 裁剪到 gfx 边界
    uint32_t x_end = (x0 + width > gfx->width) ? gfx->width : x0 + width;
    uint32_t y_end = (y0 + height > gfx->height) ? gfx->height : y0 + height;
    
    for (uint32_t y = y0; y < y_end; y++) {
        for (uint32_t x = x0; x < x_end; x++) {
            // 计算源图像中的像素位置
            uint32_t src_x = x - x0;
            uint32_t src_y = y - y0;
            uint32_t src_idx = (src_y * width + src_x) * 3;
            
            // 写入frame_buffer
            gfx_set_pixel(gfx, x, y, draw_data[src_idx], draw_data[src_idx + 1], draw_data[src_idx + 2]);
        }
    }
}

// 从内存缓冲区读取图像并绘制到 frame_buffer（无缓存，每次重新解码和缩放）
// img_buffer: 包含图像文件数据的内存缓冲区（如 PNG/JPG/BMP 等格式）
// buffer_size: 缓冲区的字节长度
// x0, y0: 目标区域左上角坐标
// width, height: 目标区域宽高（图像将被缩放到此大小）
// 图像格式由 stb_image 自动识别

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
typedef struct {
    Nano_GFX *gfx;
    uint8_t *img_buffer;
    uint32_t buffer_size;
    uint32_t x0;
    uint32_t y0;
    uint32_t width;
    uint32_t height;
    SemaphoreHandle_t sem;
} gfx_draw_image_buffer_params_t;

static void gfx_draw_image_buffer_task(void *param) {
    gfx_draw_image_buffer_params_t *p = (gfx_draw_image_buffer_params_t *)param;

    int img_w, img_h, channels;
    unsigned char *img_data = stbi_load_from_memory(p->img_buffer, (int)p->buffer_size, &img_w, &img_h, &channels, 3);
    if (img_data != NULL) {
        uint8_t *draw_data = img_data;
        uint32_t draw_w = img_w;
        uint32_t draw_h = img_h;

        if (!(p->width == 0 && p->height == 0)) {
            draw_data = (uint8_t *)platform_malloc(p->width * p->height * 3);
            if (draw_data != NULL) {
                stbir_resize_uint8_linear(
                    img_data, img_w, img_h, 0,
                    draw_data, (int)p->width, (int)p->height, 0,
                    STBIR_RGB
                );
                stbi_image_free(img_data);
                draw_w = p->width;
                draw_h = p->height;
            } else {
                stbi_image_free(img_data);
                xSemaphoreGive(p->sem);
                vTaskDelete(NULL);
                return;
            }
        }

        uint32_t x_end = (p->x0 + draw_w > p->gfx->width) ? p->gfx->width : p->x0 + draw_w;
        uint32_t y_end = (p->y0 + draw_h > p->gfx->height) ? p->gfx->height : p->y0 + draw_h;

        for (uint32_t y = p->y0; y < y_end; y++) {
            for (uint32_t x = p->x0; x < x_end; x++) {
                uint32_t src_x = x - p->x0;
                uint32_t src_y = y - p->y0;
                uint32_t src_idx = (src_y * draw_w + src_x) * 3;
                gfx_set_pixel(p->gfx, x, y, draw_data[src_idx], draw_data[src_idx + 1], draw_data[src_idx + 2]);
            }
        }

        if (p->width == 0 && p->height == 0) {
            stbi_image_free(img_data);
        } else {
            free(draw_data);
        }
    }

    xSemaphoreGive(p->sem);
    vTaskDelete(NULL);
}
#endif

void gfx_draw_image_buffer(Nano_GFX *gfx, uint8_t *img_buffer, uint32_t buffer_size, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height) {
    if (img_buffer == NULL || buffer_size == 0) {
        return;
    }

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
    SemaphoreHandle_t sem = xSemaphoreCreateBinary();
    if (sem == NULL) {
        return;
    }

    gfx_draw_image_buffer_params_t params = {
        .gfx = gfx,
        .img_buffer = img_buffer,
        .buffer_size = buffer_size,
        .x0 = x0,
        .y0 = y0,
        .width = width,
        .height = height,
        .sem = sem
    };

    TaskHandle_t taskHandle = NULL;
    BaseType_t result = xTaskCreate(
        gfx_draw_image_buffer_task,
        "gfx_draw_img",
        32768,  // 32KB stack to avoid stack overflow in stb_image/stbir
        &params,
        1,
        &taskHandle
    );

    if (result == pdPASS) {
        xSemaphoreTake(sem, portMAX_DELAY);
    } else {
        printf("gfx_draw_image_buffer: failed to create task\n");
    }

    vSemaphoreDelete(sem);
#else
    int img_w, img_h, channels;

    // 从内存缓冲区加载图像，stb_image 自动识别格式（PNG/JPG/BMP/GIF 等）
    unsigned char *img_data = stbi_load_from_memory(img_buffer, (int)buffer_size, &img_w, &img_h, &channels, 3);
    if (img_data == NULL) {
        return;
    }

    uint8_t *draw_data = img_data;
    uint32_t draw_w = img_w;
    uint32_t draw_h = img_h;

    if (!(width == 0 && height == 0)) {
        // 分配缩放后图像的内存（RGB888 格式，3字节/像素）
        draw_data = (uint8_t *)platform_malloc(width * height * 3);
        if (draw_data == NULL) {
            stbi_image_free(img_data);
            return;
        }

        // 缩放图像到目标尺寸
        stbir_resize_uint8_linear(
            img_data, img_w, img_h, 0,           // 输入图像数据、宽、高、行跨度
            draw_data, (int)width, (int)height, 0,  // 输出图像数据、宽、高、行跨度
            STBIR_RGB                               // 像素布局：RGB
        );

        // 释放原始图像数据
        stbi_image_free(img_data);
        draw_w = width;
        draw_h = height;
    }

    // 将图像绘制到 frame_buffer
    uint32_t x_end = (x0 + draw_w > gfx->width) ? gfx->width : x0 + draw_w;
    uint32_t y_end = (y0 + draw_h > gfx->height) ? gfx->height : y0 + draw_h;

    for (uint32_t y = y0; y < y_end; y++) {
        for (uint32_t x = x0; x < x_end; x++) {
            uint32_t src_x = x - x0;
            uint32_t src_y = y - y0;
            uint32_t src_idx = (src_y * draw_w + src_x) * 3;
            gfx_set_pixel(gfx, x, y, draw_data[src_idx], draw_data[src_idx + 1], draw_data[src_idx + 2]);
        }
    }

    // 释放图像数据
    if (width == 0 && height == 0) {
        stbi_image_free(img_data);
    } else {
        free(draw_data);
    }
#endif
}

// ============================================================================
// 新增：图像解码与绘制拆分函数
// ============================================================================

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
typedef struct {
    uint8_t *img_buffer;
    uint32_t buffer_size;
    uint32_t req_width;
    uint32_t req_height;
    uint8_t *out_rgb888;
    uint32_t *out_width;
    uint32_t *out_height;
    int32_t result;
    SemaphoreHandle_t sem;
} gfx_decode_image_buffer_params_t;

static void gfx_decode_image_buffer_task(void *param) {
    gfx_decode_image_buffer_params_t *p = (gfx_decode_image_buffer_params_t *)param;

    int img_w, img_h, channels;
    unsigned char *img_data = stbi_load_from_memory(p->img_buffer, (int)p->buffer_size, &img_w, &img_h, &channels, 3);
    if (img_data == NULL) {
        p->result = -1;
        xSemaphoreGive(p->sem);
        vTaskDelete(NULL);
        return;
    }

    if (p->req_width == 0 && p->req_height == 0) {
        memcpy(p->out_rgb888, img_data, img_w * img_h * 3);
        *p->out_width = (uint32_t)img_w;
        *p->out_height = (uint32_t)img_h;
    } else {
        stbir_resize_uint8_linear(
            img_data, img_w, img_h, 0,
            p->out_rgb888, (int)p->req_width, (int)p->req_height, 0,
            STBIR_RGB
        );
        *p->out_width = p->req_width;
        *p->out_height = p->req_height;
    }

    stbi_image_free(img_data);
    p->result = 0;
    xSemaphoreGive(p->sem);
    vTaskDelete(NULL);
}
#endif

// 从内存缓冲区解码图像到调用者预先分配的 RGB888 缓冲区
// img_buffer: 图像文件数据（PNG/JPG/BMP等）
// buffer_size: 缓冲区字节长度
// req_width, req_height: 请求的输出宽高。若都为0，则按原始图像尺寸输出（调用者需确保out_rgb888足够大）。
// out_rgb888: 调用者预先分配的 RGB888 缓冲区（3字节/像素）
// out_width, out_height: 返回实际解码后的图像宽高
// 返回: 0 成功，-1 失败
int32_t gfx_decode_image_buffer(uint8_t *img_buffer, uint32_t buffer_size,
                                uint32_t req_width, uint32_t req_height,
                                uint8_t *out_rgb888,
                                uint32_t *out_width, uint32_t *out_height) {
    if (img_buffer == NULL || buffer_size == 0 || out_rgb888 == NULL || out_width == NULL || out_height == NULL) {
        return -1;
    }

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
    SemaphoreHandle_t sem = xSemaphoreCreateBinary();
    if (sem == NULL) {
        return -1;
    }

    gfx_decode_image_buffer_params_t params = {
        .img_buffer = img_buffer,
        .buffer_size = buffer_size,
        .req_width = req_width,
        .req_height = req_height,
        .out_rgb888 = out_rgb888,
        .out_width = out_width,
        .out_height = out_height,
        .result = -1,
        .sem = sem
    };

    TaskHandle_t taskHandle = NULL;
    BaseType_t result = xTaskCreate(
        gfx_decode_image_buffer_task,
        "gfx_dec_img",
        32768,  // 32KB stack to avoid stack overflow in stb_image/stbir
        &params,
        1,
        &taskHandle
    );

    if (result == pdPASS) {
        xSemaphoreTake(sem, portMAX_DELAY);
    } else {
        printf("gfx_decode_image_buffer: failed to create task\n");
    }

    vSemaphoreDelete(sem);
    return params.result;
#else
    int img_w, img_h, channels;
    unsigned char *img_data = stbi_load_from_memory(img_buffer, (int)buffer_size, &img_w, &img_h, &channels, 3);
    if (img_data == NULL) {
        return -1;
    }

    if (req_width == 0 && req_height == 0) {
        memcpy(out_rgb888, img_data, img_w * img_h * 3);
        *out_width = (uint32_t)img_w;
        *out_height = (uint32_t)img_h;
    } else {
        stbir_resize_uint8_linear(
            img_data, img_w, img_h, 0,
            out_rgb888, (int)req_width, (int)req_height, 0,
            STBIR_RGB
        );
        *out_width = req_width;
        *out_height = req_height;
    }

    stbi_image_free(img_data);
    return 0;
#endif
}

// 将 RGB888 图像缓冲区绘制到 NanoGFX
// rgb888_buffer: RGB888 像素数据（3字节/像素，按行优先连续存储）
// img_width, img_height: 图像宽高
// x0, y0: 绘制起始坐标
void gfx_draw_rgb888_buffer(Nano_GFX *gfx, uint8_t *rgb888_buffer,
                            uint32_t img_width, uint32_t img_height,
                            uint32_t x0, uint32_t y0) {
    if (rgb888_buffer == NULL || img_width == 0 || img_height == 0) {
        return;
    }

    uint32_t x_end = (x0 + img_width > gfx->width) ? gfx->width : x0 + img_width;
    uint32_t y_end = (y0 + img_height > gfx->height) ? gfx->height : y0 + img_height;

    for (uint32_t y = y0; y < y_end; y++) {
        for (uint32_t x = x0; x < x_end; x++) {
            uint32_t src_x = x - x0;
            uint32_t src_y = y - y0;
            uint32_t src_idx = (src_y * img_width + src_x) * 3;
            gfx_set_pixel(gfx, x, y, rgb888_buffer[src_idx], rgb888_buffer[src_idx + 1], rgb888_buffer[src_idx + 2]);
        }
    }
}
