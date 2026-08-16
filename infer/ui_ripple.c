#include <stdio.h>
#include <string.h>

#include "ui_ripple.h"

#include "graphics.h"
#include "input_device.h"
#include "platform.h"
#include "touch.h"

// ===============================================================================
// 实现选择：1-定点（默认，ESP32 无 FPU 优化）；0-浮点（JS 原版直译，可读性参考）
// ===============================================================================
#define WR_USE_FIXED_POINT (1)

// ===============================================================================
// 物理常数（统一以浮点形式给出，定点版换算为移位运算）
// ===============================================================================
#define WR_WIDTH            (SCREEN_WIDTH)      // 渲染宽度（像素）
#define WR_HEIGHT           (SCREEN_HEIGHT)     // 渲染高度（像素）
#define WR_DROP_RADIUS      (5.0f)              // 波源半径大小（全分辨率等效）
#define WR_ATTENUATION      (5.0f)              // 衰减级别（2 的幂指数）
#define WR_MAX_AMPLITUDE    (1024.0f)           // 最大振幅
#define WR_SOURCE_AMPLITUDE (512.0f)            // 震源振幅

// 定点版换算：衰减 = 右移 5 位；位移采样中的 /1024 = 右移 10 位
#define WR_ATTENUATION_SHIFT  ((int32_t)WR_ATTENUATION)                 // 5
#define WR_AMPLITUDE_SHIFT    (10)                                      // log2(WR_MAX_AMPLITUDE)
#define WR_SOURCE_AMPLITUDE_I ((int32_t)WR_SOURCE_AMPLITUDE)            // 512
#define WR_MAX_AMPLITUDE_I    ((int32_t)WR_MAX_AMPLITUDE)               // 1024

// ===============================================================================
// 波场几何：半分辨率模拟（性能关键）
//
// 全分辨率波场（320x240）步进每帧需约 1.5MB PSRAM 流量，实测只有个位数帧率；
// 改为半分辨率（160x120）后计算量降为 1/4，且振幅场两页（~78KB）与 last_map
//（~38KB）可整体放入内部 RAM（带宽为 PSRAM 数倍、无 cache miss），步进耗时从
// 约 100ms 降到几毫秒。波纹折射采样本身有低通性质，视觉损失很小。
// 渲染仍为全分辨率：每个模拟格写 2x2 帧缓冲块，从全分辨率纹理位移采样 4 次。
//
// 振幅场两页缓冲，每页上下各一行零填充（免边界特判），与 JS 原版一致。
// ===============================================================================
#define WR_SIM_W    (WR_WIDTH / 2)                  // 波场宽度（模拟格）
#define WR_SIM_H    (WR_HEIGHT / 2)                 // 波场高度（模拟格）
#define WR_PAGE     (WR_SIM_W * (WR_SIM_H + 2))     // 单页大小（元素数）
#define WR_MAP_SIZE (2 * WR_PAGE)                   // 振幅场总大小（元素数）
#define WR_SIM_DROP_RADIUS ((int32_t)(WR_DROP_RADIUS * 0.6f))  // 半分辨率波源半径 3

#define WR_IMAGE_PATH (PLATFORM_ROOT_DIR "/wp.png")

// ===============================================================================
// 定点实现（int16 振幅场 + 内部 RAM + 移位运算 + 行指针直写帧缓冲）
// ===============================================================================
#if WR_USE_FIXED_POINT

typedef struct {
    uint16_t *texture;      // 背景纹理 RGB565（WR_WIDTH*WR_HEIGHT，PSRAM）
    int16_t  *ripple_map;   // 振幅场两页缓冲（WR_MAP_SIZE，内部 RAM 优先，零初始化）
    int16_t  *last_map;     // 上一帧逐格显示的位移量（WR_SIM_W*WR_SIM_H，内部 RAM 优先）
    int32_t   old_index;    // 旧页基址（含上填充行偏移）
    int32_t   new_index;    // 新页基址（含上填充行偏移）
    int32_t   load_failed;  // 图像加载/解码失败标志
    int32_t   error_drawn;  // 错误画面已绘制标志
} WR_State;

static WR_State s_wr;

// 在触点 (cx, cy)（全分辨率坐标）处激发波源（向当前 old 页振幅场累加震源振幅，含边界钳制）
static void wr_disturb(int32_t cx, int32_t cy) {
    if (s_wr.ripple_map == NULL) return;
    int32_t scx = cx >> 1;      // 全分辨率坐标 -> 半分辨率模拟格坐标
    int32_t scy = cy >> 1;
    int32_t x0 = scx - WR_SIM_DROP_RADIUS; if (x0 < 0) x0 = 0;
    int32_t y0 = scy - WR_SIM_DROP_RADIUS; if (y0 < 0) y0 = 0;
    int32_t x1 = scx + WR_SIM_DROP_RADIUS; if (x1 > WR_SIM_W - 1) x1 = WR_SIM_W - 1;
    int32_t y1 = scy + WR_SIM_DROP_RADIUS; if (y1 > WR_SIM_H - 1) y1 = WR_SIM_H - 1;
    for (int32_t y = y0; y <= y1; y++) {
        int16_t *row = &s_wr.ripple_map[s_wr.old_index + y * WR_SIM_W];
        for (int32_t x = x0; x <= x1; x++) {
            row[x] += WR_SOURCE_AMPLITUDE_I;
        }
    }
}

// 波场步进一帧 + 位移采样渲染
static void wr_render(Nano_GFX *gfx) {
    int16_t  *map     = s_wr.ripple_map;
    int16_t  *last    = s_wr.last_map;
    uint16_t *texture = s_wr.texture;

    // 交换新旧页（与 JS 原版的 old_index/new_index 交互一致）
    int32_t old_index = s_wr.new_index;
    int32_t new_index = s_wr.old_index;
    s_wr.old_index = old_index;
    s_wr.new_index = new_index;

    int32_t i = 0;                      // 模拟格线性索引
    int32_t map_index = old_index;      // 振幅场索引
    for (int32_t sy = 0; sy < WR_SIM_H; sy++) {
        // 本模拟格对应的 2x2 帧缓冲块的首行行指针（每行一次函数指针调用，兼容单/双缓冲；
        // 上下两行相邻且同处一个半屏——半屏分界 120 为偶数，故块内两行必在同一缓冲）
        uint32_t fb_off = 0;
        uint16_t *fb_row = gfx->rgb565_access(gfx, 0, (uint32_t)(2 * sy), &fb_off);
        fb_row += fb_off;
        for (int32_t sx = 0; sx < WR_SIM_W; sx++) {
            int32_t top    = map[map_index - WR_SIM_W];                     // 上边相邻点（首行读到零填充行）
            int32_t bottom = map[map_index + WR_SIM_W];                     // 下边相邻点（末行读到零填充行）
            int32_t left   = (sx > 0)           ? map[map_index - 1] : 0;   // 左边相邻点
            int32_t right  = (sx < WR_SIM_W - 1) ? map[map_index + 1] : 0;  // 右边相邻点

            // 当前模拟格下一时刻的振幅
            int32_t amp = ((top + bottom + left + right) >> 1) - map[new_index + i];
            amp -= amp >> WR_ATTENUATION_SHIFT;                             // 衰减（算术右移）
            map[new_index + i] = (int16_t)amp;

            // 位移量；与上一帧相同则跳过整个 2x2 块（JS 原版 last_map 优化）
            int32_t disp = WR_MAX_AMPLITUDE_I - amp;
            if (last[i] != (int16_t)disp) {
                last[i] = (int16_t)disp;
                // 2x2 帧缓冲块逐像素位移采样（全分辨率纹理）
                for (int32_t oy = 0; oy < 2; oy++) {
                    int32_t y = 2 * sy + oy;
                    int32_t dy = (((y - (WR_HEIGHT / 2)) * disp) >> WR_AMPLITUDE_SHIFT) + (WR_HEIGHT / 2);
                    if (dy < 0) dy = 0; else if (dy > WR_HEIGHT - 1) dy = WR_HEIGHT - 1;
                    const uint16_t *tex_row = &texture[dy * WR_WIDTH];
                    uint16_t *out_row = &fb_row[oy * WR_WIDTH + 2 * sx];
                    for (int32_t ox = 0; ox < 2; ox++) {
                        int32_t x = 2 * sx + ox;
                        int32_t dx = (((x - (WR_WIDTH / 2)) * disp) >> WR_AMPLITUDE_SHIFT) + (WR_WIDTH / 2);
                        if (dx < 0) dx = 0; else if (dx > WR_WIDTH - 1) dx = WR_WIDTH - 1;
                        out_row[ox] = tex_row[dx];
                    }
                }
            }
            i++;
            map_index++;
        }
    }
}

// ===============================================================================
// 浮点实现（JS 原版直译，可读性参考；几何与定点版一致：半分辨率波场）
// ===============================================================================
#else

typedef struct {
    uint16_t *texture;      // 背景纹理 RGB565（WR_WIDTH*WR_HEIGHT，PSRAM）
    float    *ripple_map;   // 振幅场两页缓冲（WR_MAP_SIZE，内部 RAM 优先，零初始化）
    float    *last_map;     // 上一帧逐格显示的位移量（WR_SIM_W*WR_SIM_H，内部 RAM 优先）
    int32_t   old_index;    // 旧页基址（含上填充行偏移）
    int32_t   new_index;    // 新页基址（含上填充行偏移）
    int32_t   load_failed;  // 图像加载/解码失败标志
    int32_t   error_drawn;  // 错误画面已绘制标志
} WR_State;

static WR_State s_wr;

// 在触点 (cx, cy)（全分辨率坐标）处激发波源（向当前 old 页振幅场累加震源振幅，含边界钳制）
static void wr_disturb(int32_t cx, int32_t cy) {
    if (s_wr.ripple_map == NULL) return;
    int32_t scx = cx >> 1;      // 全分辨率坐标 -> 半分辨率模拟格坐标
    int32_t scy = cy >> 1;
    int32_t x0 = scx - WR_SIM_DROP_RADIUS; if (x0 < 0) x0 = 0;
    int32_t y0 = scy - WR_SIM_DROP_RADIUS; if (y0 < 0) y0 = 0;
    int32_t x1 = scx + WR_SIM_DROP_RADIUS; if (x1 > WR_SIM_W - 1) x1 = WR_SIM_W - 1;
    int32_t y1 = scy + WR_SIM_DROP_RADIUS; if (y1 > WR_SIM_H - 1) y1 = WR_SIM_H - 1;
    for (int32_t y = y0; y <= y1; y++) {
        float *row = &s_wr.ripple_map[s_wr.old_index + y * WR_SIM_W];
        for (int32_t x = x0; x <= x1; x++) {
            row[x] += WR_SOURCE_AMPLITUDE;
        }
    }
}

// 波场步进一帧 + 位移采样渲染
static void wr_render(Nano_GFX *gfx) {
    float    *map     = s_wr.ripple_map;
    float    *last    = s_wr.last_map;
    uint16_t *texture = s_wr.texture;

    int32_t half_width  = WR_WIDTH  / 2;
    int32_t half_height = WR_HEIGHT / 2;

    // 交换新旧页（与 JS 原版的 old_index/new_index 交互一致）
    int32_t old_index = s_wr.new_index;
    int32_t new_index = s_wr.old_index;
    s_wr.old_index = old_index;
    s_wr.new_index = new_index;

    int32_t i = 0;                      // 模拟格线性索引
    int32_t map_index = old_index;      // 振幅场索引
    for (int32_t sy = 0; sy < WR_SIM_H; sy++) {
        for (int32_t sx = 0; sx < WR_SIM_W; sx++) {
            float top    = map[map_index - WR_SIM_W];                       // 上边相邻点（首行读到零填充行）
            float bottom = map[map_index + WR_SIM_W];                       // 下边相邻点（末行读到零填充行）
            float left   = (sx > 0)            ? map[map_index - 1] : 0.0f; // 左边相邻点
            float right  = (sx < WR_SIM_W - 1) ? map[map_index + 1] : 0.0f; // 右边相邻点

            // 当前模拟格下一时刻的振幅
            float amp = (top + bottom + left + right) * 0.5f - map[new_index + i];
            amp -= amp / (1 << WR_ATTENUATION_SHIFT);                       // 衰减
            map[new_index + i] = amp;

            // 位移量；与上一帧相同则跳过整个 2x2 块（JS 原版 last_map 优化）
            float disp = WR_MAX_AMPLITUDE - amp;
            if (last[i] != disp) {
                last[i] = disp;
                // 2x2 帧缓冲块逐像素位移采样（全分辨率纹理）
                for (int32_t oy = 0; oy < 2; oy++) {
                    int32_t y = 2 * sy + oy;
                    int32_t dy = (int32_t)(((y - half_height) * disp) / WR_MAX_AMPLITUDE) + half_height;
                    if (dy < 0) dy = 0; else if (dy > WR_HEIGHT - 1) dy = WR_HEIGHT - 1;
                    for (int32_t ox = 0; ox < 2; ox++) {
                        int32_t x = 2 * sx + ox;
                        int32_t dx = (int32_t)(((x - half_width) * disp) / WR_MAX_AMPLITUDE) + half_width;
                        if (dx < 0) dx = 0; else if (dx > WR_WIDTH - 1) dx = WR_WIDTH - 1;
                        uint32_t fb_off = 0;
                        uint16_t *fb = gfx->rgb565_access(gfx, (uint32_t)x, (uint32_t)y, &fb_off);
                        fb[fb_off] = texture[dy * WR_WIDTH + dx];
                    }
                }
            }
            i++;
            map_index++;
        }
    }
}

#endif // WR_USE_FIXED_POINT

// ===============================================================================
// 游戏接口（两种实现共用）
// ===============================================================================

// 释放全部内存（纹理 / 振幅场 / last_map）
static void wr_free_all(void) {
    if (s_wr.texture != NULL)    { free(s_wr.texture);    s_wr.texture = NULL; }
    if (s_wr.ripple_map != NULL) { free(s_wr.ripple_map); s_wr.ripple_map = NULL; }
    if (s_wr.last_map != NULL)   { free(s_wr.last_map);   s_wr.last_map = NULL; }
}

int32_t ui_ripple_init(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    memset(&s_wr, 0, sizeof(s_wr));
    s_wr.old_index = WR_SIM_W;                      // 旧页基址（跳过上填充行）
    s_wr.new_index = WR_SIM_W * (WR_SIM_H + 3);     // 新页基址（另一页 + 上填充行）

    // ---- 从 SD 卡读取 /wp.png 并解码缩放为 320x240 纹理（PSRAM） ----
    uint8_t *file_buffer = NULL;
    size_t   file_size = 0;
    uint8_t *rgb888 = NULL;
    do {
        if (platform_read_file_to_buffer(WR_IMAGE_PATH, &file_buffer, &file_size) != 0) {
            printf("ui_ripple: read %s failed\n", WR_IMAGE_PATH);
            break;
        }
        rgb888 = (uint8_t *)platform_malloc(WR_WIDTH * WR_HEIGHT * 3);
        s_wr.texture = (uint16_t *)platform_malloc(WR_WIDTH * WR_HEIGHT * 2);
        if (rgb888 == NULL || s_wr.texture == NULL) break;
        uint32_t img_w = 0, img_h = 0;
        // 解码并缩放（内部在 32KB 栈临时任务中执行 stb_image/stbir，渲染任务可安全调用）
        if (gfx_decode_image_buffer(file_buffer, (uint32_t)file_size,
                                    WR_WIDTH, WR_HEIGHT, rgb888, &img_w, &img_h) != 0) {
            printf("ui_ripple: decode %s failed\n", WR_IMAGE_PATH);
            break;
        }
        // RGB888 -> RGB565
        for (int32_t i = 0; i < WR_WIDTH * WR_HEIGHT; i++) {
            uint8_t r = rgb888[i * 3 + 0];
            uint8_t g = rgb888[i * 3 + 1];
            uint8_t b = rgb888[i * 3 + 2];
            s_wr.texture[i] = (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
        }
    } while (0);
    if (file_buffer != NULL) free(file_buffer);
    if (rgb888 != NULL) free(rgb888);

    // ---- 振幅场与 last_map（内部 RAM 优先，失败回退 PSRAM；
    //      零初始化：等价于 JS 原版的 init；last_map 为 0 使首帧全量绘制） ----
#if WR_USE_FIXED_POINT
    if (s_wr.texture != NULL) {
        s_wr.ripple_map = (int16_t *)platform_calloc_internal(WR_MAP_SIZE, sizeof(int16_t));
        if (s_wr.ripple_map == NULL) s_wr.ripple_map = (int16_t *)platform_calloc(WR_MAP_SIZE, sizeof(int16_t));
    }
    if (s_wr.ripple_map != NULL) {
        s_wr.last_map = (int16_t *)platform_calloc_internal(WR_SIM_W * WR_SIM_H, sizeof(int16_t));
        if (s_wr.last_map == NULL) s_wr.last_map = (int16_t *)platform_calloc(WR_SIM_W * WR_SIM_H, sizeof(int16_t));
    }
#else
    if (s_wr.texture != NULL) {
        s_wr.ripple_map = (float *)platform_calloc_internal(WR_MAP_SIZE, sizeof(float));
        if (s_wr.ripple_map == NULL) s_wr.ripple_map = (float *)platform_calloc(WR_MAP_SIZE, sizeof(float));
    }
    if (s_wr.ripple_map != NULL) {
        s_wr.last_map = (float *)platform_calloc_internal(WR_SIM_W * WR_SIM_H, sizeof(float));
        if (s_wr.last_map == NULL) s_wr.last_map = (float *)platform_calloc(WR_SIM_W * WR_SIM_H, sizeof(float));
    }
#endif

    if (s_wr.texture == NULL || s_wr.ripple_map == NULL || s_wr.last_map == NULL) {
        wr_free_all();
        s_wr.load_failed = 1;   // 进入错误画面模式（按 A 返回），不静默退回菜单
    }

    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);
    return 0;
}

int32_t ui_ripple_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 按A键(ESC)返回小游戏菜单
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_GAME_MENU;
        return 0;
    }
    return 0;
}

int32_t ui_ripple_render_frame(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    Nano_GFX *gfx = global_state->gfx;

    // 图像加载失败：黑底红字提示（只画一次），等待 A 键返回
    if (s_wr.load_failed) {
        if (!s_wr.error_drawn) {
            gfx_soft_clear(gfx);
            gfx_font_draw_text(gfx, GFX_FONT_ALPHA_16, L"水波", 6, 2, 255, 255, 255, 1);
            gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"读取 /wp.png 失败", 6, 100, 255, 80, 80, 1);
            gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"请将图片放入SD卡根目录", 6, 120, 180, 180, 180, 1);
            gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"A返回", 6, 220, 180, 180, 180, 1);
            gfx_refresh(gfx);
            s_wr.error_drawn = 1;
        }
        return -1;
    }
    if (s_wr.texture == NULL || s_wr.ripple_map == NULL) return -1;

    // ---- 触摸激发水波纹（按住拖动持续激发，等价于原网页 click + mousemove） ----
    int32_t touch_x = 0, touch_y = 0, is_pressed = 0;
    if (touch_read(&touch_x, &touch_y, &is_pressed) == 0 && is_pressed) {
        if (touch_x >= 0 && touch_x < WR_WIDTH && touch_y >= 0 && touch_y < WR_HEIGHT) {
            wr_disturb(touch_x, touch_y);
        }
    }

    // ---- 波场步进 + 渲染一帧 ----
    wr_render(gfx);
    gfx_refresh(gfx);
    return 0;
}

void ui_ripple_on_exit(void) {
    wr_free_all();
}
