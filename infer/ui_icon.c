#include <stdlib.h>
#include <string.h>

#include "ui_icon.h"
#include "platform.h"

// 图标像素缓存（常驻 PSRAM，首次请求时建立，永不释放）
#define UI_ICON_CACHE_MAX_ENTRIES (16)
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

void ui_icon_draw_centered(Nano_GFX *gfx, const char *path, int32_t cx, int32_t cy) {
    if (gfx == NULL || path == NULL) {
        return;
    }

    UI_Icon_Cache_Entry *entry = ui_icon_cache_lookup(path);
    if (entry == NULL) {
        return; // 缓存满且未命中：放弃绘制
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

    if (entry->rgba == NULL) {
        return; // 缓存的失败结果：不绘制
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
}
