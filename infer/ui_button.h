#ifndef __NANO_UI_BUTTON_H__
#define __NANO_UI_BUTTON_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <wchar.h>

#include "graphics.h"
#include "ui.h"

// ===============================================================================
// 通用按钮控件
//
// 设计目标：
//   - 把「背景 + 边框 + 文字/图标 + 按下态 + 命中矩形 + 文字位移」收敛到一处；
//   - 不接管复杂页面自己的滚动/拖动/长按状态机：既提供无状态 draw/hit，
//     也提供可选的简单 tracker + group event；
//   - 默认保留「按下时文字下沉 1px」的拟物反馈（press_text_dy = 1）。
//
// 重要约定：
//   - 本控件只画到 Nano_GFX，不调用 gfx_refresh；刷新策略由宿主决定。
//   - 文字/图标位移会改变像素包围盒：按下态变化时宿主必须重绘整个按钮矩形
//     （建议外扩 1px），不能只重绘文字区域，否则会留下残影。
//   - 半透明按钮（FLIP/玲珑仪）必须先由宿主重绘按钮下方背景，再调用本控件；
//     直接叠画会重复混合 alpha。
//   - 主菜单应用入口宫格不属于本控件的适用范围（它是专用 tile/grid 控件）。
// ===============================================================================

#define UI_BTN_ID_NONE (-1)

// 按下时文字/图标的默认位移：文字下沉 1px，图标不动。
#define UI_BTN_PRESS_TEXT_DY_DEFAULT (1)

typedef struct {
    uint8_t r, g, b;
} Ui_RGB;

typedef enum {
    UI_BTN_VIS_NORMAL = 0,
    UI_BTN_VIS_PRESSED,
    UI_BTN_VIS_SELECTED,
    UI_BTN_VIS_SELECTED_PRESSED,
    UI_BTN_VIS_DISABLED,
    UI_BTN_VIS_COUNT
} Ui_Button_Visual;

typedef struct {
    Ui_RGB bg[UI_BTN_VIS_COUNT];
    Ui_RGB fg[UI_BTN_VIS_COUNT];
    Ui_RGB border[UI_BTN_VIS_COUNT];

    // 1 = 不透明置色；>=4 = alpha 混合；0 = 不画（配合 flags）
    uint8_t bg_alpha[UI_BTN_VIS_COUNT];
    uint8_t border_alpha[UI_BTN_VIS_COUNT];

    uint8_t shadow_alpha;
    int8_t  shadow_dy;
    Ui_RGB  shadow;
} Ui_Button_Theme;

typedef enum {
    UI_BTN_STYLE_RECT = 0,   // 无圆角矩形文字按钮（输入页脚、黄金矿工返回）
    UI_BTN_STYLE_ROUNDED,    // 圆角矩形（水池、玲珑仪、步进器）
    UI_BTN_STYLE_PILL,       // 胶囊（确认框、页眉动作、寻呼机按钮）
    UI_BTN_STYLE_CIRCLE,     // 圆形（音乐盒播放/切歌）
    UI_BTN_STYLE_KEY,        // 键帽（软键盘、16 键、电子书跳页）
    UI_BTN_STYLE_CARD,       // 大卡片（寻呼机模式卡）
    UI_BTN_STYLE_TEXT        // 纯文字热区（页眉返回、页脚软键）
} Ui_Button_Style;

#define UI_BTN_FLAG_DRAW_BG       (1u << 0)
#define UI_BTN_FLAG_DRAW_BORDER   (1u << 1)
#define UI_BTN_FLAG_DOUBLE_BORDER (1u << 2)
#define UI_BTN_FLAG_SHADOW        (1u << 3)
#define UI_BTN_FLAG_HIT_CIRCLE    (1u << 4)

typedef struct Ui_Button Ui_Button;

typedef void (*Ui_Button_Content_Draw)(
    Nano_GFX *gfx, const Ui_Button *b,
    int32_t base_cx, int32_t base_cy,
    Ui_Button_Visual vis, void *user
);

struct Ui_Button {
    int32_t x, y, w, h;
    int16_t radius;          // PILL/CIRCLE 可忽略；<0 表示 h/2
    uint8_t style;
    uint8_t flags;
    int32_t id;

    const wchar_t *label;    // 借用指针，不复制
    const wchar_t *value;    // 双行按钮第二行，可为 NULL
    uint32_t font_id;
    uint32_t value_font_id;
    int16_t  label_dy;       // 相对中心，例如双行按钮 -9
    int16_t  value_dy;       // 相对中心，例如双行按钮 +9

    // ---- 拟物按压位移 ----
    int8_t press_text_dx;    // 默认 0
    int8_t press_text_dy;    // 默认 1
    int8_t press_icon_dx;    // 默认 0
    int8_t press_icon_dy;    // 默认 0；纯图标按钮可设 1

    int16_t hit_pad;         // 矩形额外命中余量
    int16_t hit_radius_pad;  // 圆形额外命中余量

    uint16_t repeat_delay_ms;
    uint16_t repeat_interval_ms;

    uint8_t visible;
    uint8_t enabled;
    uint8_t selected;
    uint8_t quiet;           // 1 = 画弱化色但仍可命中
    uint8_t text_mode;       // 文字绘制 mode；默认 1（不透明置色）

    const Ui_Button_Theme *theme; // 可选，覆盖 group theme
    Ui_Button_Content_Draw content_draw;
    void *content_user;
};

typedef struct {
    int32_t pressed_id;
    int32_t active;
    int32_t dragging;
    int32_t down_x, down_y;
    int32_t drag_threshold;
    uint64_t press_ts;
    uint64_t repeat_next_ts;

    uint8_t press_visual;
    uint8_t repeat_started;
    uint8_t follow_press;    // 1 = 拖出按钮立即弹起；0 = 保持按下到松手
    uint8_t require_up_edge; // 1 = 必须 UP 边沿结算
} Ui_Button_Tracker;

typedef struct {
    Ui_Button *items;
    int32_t count;
    const Ui_Button_Theme *theme;
    int32_t clip_x0, clip_y0, clip_x1, clip_y1; // 可选裁剪区；x1<=x0 表示不裁剪
} Ui_Button_Group;

typedef struct {
    int32_t pressed_id;
    int32_t clicked_id;
    int32_t repeated_id;
    int32_t cancelled_id;
    uint8_t visual_changed;
} Ui_Button_Result;

void ui_button_theme_default(Ui_Button_Theme *t, int32_t is_light);

void ui_button_init(Ui_Button *b, int32_t id,
                    int32_t x, int32_t y, int32_t w, int32_t h);
void ui_button_set_text(Ui_Button *b, const wchar_t *label, const wchar_t *value);
void ui_button_set_press_offset(Ui_Button *b, int8_t text_dx, int8_t text_dy);
void ui_button_set_icon_press_offset(Ui_Button *b, int8_t icon_dx, int8_t icon_dy);

int32_t ui_button_hit(const Ui_Button *b, int32_t x, int32_t y);
int32_t ui_button_group_hit(const Ui_Button_Group *g, int32_t x, int32_t y);

Ui_Button_Visual ui_button_visual(const Ui_Button *b, const Ui_Button_Tracker *t);

int32_t ui_button_label_cx(const Ui_Button *b, Ui_Button_Visual vis);
int32_t ui_button_label_cy(const Ui_Button *b, Ui_Button_Visual vis);
int32_t ui_button_icon_cx(const Ui_Button *b, Ui_Button_Visual vis);
int32_t ui_button_icon_cy(const Ui_Button *b, Ui_Button_Visual vis);

void ui_button_draw(Nano_GFX *gfx, const Ui_Button *b,
                    const Ui_Button_Theme *theme, Ui_Button_Visual vis);

void ui_button_group_draw(Nano_GFX *gfx, const Ui_Button_Group *g,
                          const Ui_Button_Tracker *t);

void ui_button_tracker_reset(Ui_Button_Tracker *t);

Ui_Button_Result ui_button_group_event(Ui_Button_Group *g,
                                       Ui_Button_Tracker *t,
                                       const Key_Event *ke,
                                       uint64_t now_ms);

#ifdef __cplusplus
}
#endif

#endif
