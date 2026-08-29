#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ui_icon.h"
#include "platform.h"

// 图标像素缓存（常驻 PSRAM，首次请求时建立，永不释放）
// 容量须覆盖全部调用方的不同路径总数：主菜单16宫格16个 + 黄金矿工精灵9个，留余量取32
#define UI_ICON_CACHE_MAX_ENTRIES (32)
#define UI_ICON_PATH_MAX_LEN      (64)

typedef struct {
    char path[UI_ICON_PATH_MAX_LEN]; // 缓存键：图标文件路径
    uint8_t *rgba;    // 解码后的 RGBA 像素（PSRAM）；NULL 表示读取/解码失败
    int32_t width;
    int32_t height;
    int32_t is_valid; // 1-槽位已占用（含失败结果）；0-空闲
} UI_Icon_Cache_Entry;

static UI_Icon_Cache_Entry s_icon_cache[UI_ICON_CACHE_MAX_ENTRIES];

// 按路径查找缓存槽位。命中返回已占用槽位；未命中返回空闲槽位（无空闲则返回NULL）
static UI_Icon_Cache_Entry *ui_icon_cache_lookup(const char *path) {
    UI_Icon_Cache_Entry *empty = NULL;
    for (int32_t i = 0; i < UI_ICON_CACHE_MAX_ENTRIES; i++) {
        if (!s_icon_cache[i].is_valid) {
            if (empty == NULL) empty = &s_icon_cache[i];
            continue;
        }
        if (strncmp(s_icon_cache[i].path, path, UI_ICON_PATH_MAX_LEN) == 0) {
            return &s_icon_cache[i];
        }
    }
    return empty;
}

// 按路径取缓存槽位（首次请求时现场从SD卡读取解码）。
// 成功返回槽位（entry->rgba 为 NULL 表示读取/解码失败的缓存结果）；缓存满且未命中返回 NULL。
static UI_Icon_Cache_Entry *ui_icon_cache_load(const char *path) {
    UI_Icon_Cache_Entry *entry = ui_icon_cache_lookup(path);
    if (entry == NULL) {
        return NULL;
    }
    // 首次请求该路径：从SD卡读取并解码为 RGBA 像素，缓存于 PSRAM
    if (!entry->is_valid) {
        uint8_t *file_buffer = NULL;
        size_t file_size = 0;
        entry->rgba = NULL;
        if (platform_read_file_to_buffer(path, &file_buffer, &file_size) == 0
            && file_buffer != NULL && file_size > 0) {
            entry->rgba = gfx_decode_image_rgba(file_buffer, (uint32_t)file_size, &entry->width, &entry->height);
        }
        if (file_buffer != NULL) {
            free(file_buffer);
        }
        strncpy(entry->path, path, UI_ICON_PATH_MAX_LEN - 1);
        entry->path[UI_ICON_PATH_MAX_LEN - 1] = '\0';
        entry->is_valid = 1; // 无论成败均占用槽位，避免重复访问SD卡
    }
    return entry;
}

int32_t ui_icon_get_size(const char *path, int32_t *out_width, int32_t *out_height) {
    if (path == NULL) {
        return -1;
    }
    UI_Icon_Cache_Entry *entry = ui_icon_cache_load(path);
    if (entry == NULL || entry->rgba == NULL) {
        return -1;
    }
    if (out_width != NULL)  *out_width = entry->width;
    if (out_height != NULL) *out_height = entry->height;
    return 0;
}

int32_t ui_icon_draw_rotated(Nano_GFX *gfx, const char *path, int32_t pivot_x, int32_t pivot_y, float angle_rad) {
    if (gfx == NULL || path == NULL) {
        return -1;
    }

    UI_Icon_Cache_Entry *entry = ui_icon_cache_load(path);
    if (entry == NULL || entry->rgba == NULL) {
        return -1;
    }

    const int32_t w = entry->width;
    const int32_t h = entry->height;
    const float s = sinf(angle_rad), c = cosf(angle_rad);

    // 贴图局部坐标系：枢轴 = 上缘中点，局部 x∈[-w/2, w/2)，y∈[0, h)
    // 正映射：screen = pivot + lx*(c, -s) + ly*(s, c)（局部竖直轴对齐旋转方向）
    // 先旋转四角求目标包围盒，并裁剪到屏幕
    const float corner_lx[4] = {-w / 2.0f, w / 2.0f, -w / 2.0f, w / 2.0f};
    const float corner_ly[4] = {0.0f, 0.0f, (float)h, (float)h};
    float min_x = 0.0f, max_x = 0.0f, min_y = 0.0f, max_y = 0.0f;
    for (int32_t i = 0; i < 4; i++) {
        float rx = corner_lx[i] * c + corner_ly[i] * s;
        float ry = -corner_lx[i] * s + corner_ly[i] * c;
        if (i == 0 || rx < min_x) min_x = rx;
        if (i == 0 || rx > max_x) max_x = rx;
        if (i == 0 || ry < min_y) min_y = ry;
        if (i == 0 || ry > max_y) max_y = ry;
    }
    int32_t x0 = pivot_x + (int32_t)min_x;
    int32_t x1 = pivot_x + (int32_t)max_x + 1;
    int32_t y0 = pivot_y + (int32_t)min_y;
    int32_t y1 = pivot_y + (int32_t)max_y + 1;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (int32_t)gfx->width)  x1 = (int32_t)gfx->width;
    if (y1 > (int32_t)gfx->height) y1 = (int32_t)gfx->height;

    // 逆映射逐像素采样（最近邻）+ alpha 混合
    for (int32_t y = y0; y < y1; y++) {
        float dy = (float)(y - pivot_y);
        for (int32_t x = x0; x < x1; x++) {
            float dx = (float)(x - pivot_x);
            // 逆旋转：lx = dx*c - dy*s；ly = dx*s + dy*c
            int32_t sx = (int32_t)(dx * c - dy * s + w / 2.0f);
            int32_t sy = (int32_t)(dx * s + dy * c);
            if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
            uint32_t src_idx = ((uint32_t)sy * (uint32_t)w + (uint32_t)sx) * 4;
            if (entry->rgba[src_idx + 3] == 0) continue;
            gfx_blend_pixel(gfx, (uint32_t)x, (uint32_t)y,
                entry->rgba[src_idx], entry->rgba[src_idx + 1], entry->rgba[src_idx + 2], entry->rgba[src_idx + 3]);
        }
    }
    return 0;
}

int32_t ui_icon_draw_centered(Nano_GFX *gfx, const char *path, int32_t cx, int32_t cy) {
    if (gfx == NULL || path == NULL) {
        return -1;
    }

    UI_Icon_Cache_Entry *entry = ui_icon_cache_load(path);
    if (entry == NULL) {
        return -1; // 缓存满且未命中：放弃绘制
    }

    if (entry->rgba == NULL) {
        return -1; // 缓存的失败结果：不绘制
    }

    // 用缓存像素混合绘制（与 gfx_draw_image_buffer 一致：alpha 混合，右/下边界裁剪）
    int32_t x0 = cx - entry->width / 2;
    int32_t y0 = cy - entry->height / 2;
    int32_t x_end = (x0 + entry->width > (int32_t)gfx->width) ? (int32_t)gfx->width : x0 + entry->width;
    int32_t y_end = (y0 + entry->height > (int32_t)gfx->height) ? (int32_t)gfx->height : y0 + entry->height;
    for (int32_t y = y0; y < y_end; y++) {
        for (int32_t x = x0; x < x_end; x++) {
            uint32_t src_idx = ((uint32_t)(y - y0) * (uint32_t)entry->width + (uint32_t)(x - x0)) * 4;
            gfx_blend_pixel(gfx, (uint32_t)x, (uint32_t)y,
                entry->rgba[src_idx], entry->rgba[src_idx + 1], entry->rgba[src_idx + 2], entry->rgba[src_idx + 3]);
        }
    }
    return 0;
}
