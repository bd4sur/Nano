#include "ui_button.h"

#include <stdlib.h>
#include <string.h>

// ===============================================================================
// 默认主题
//
// 只提供中性底色，模块应通过 ui_button_theme_default() 取默认值后覆盖自己的
// accent / selected / pressed 色。这里不试图统一全项目配色。
// ===============================================================================

static const Ui_Button_Theme s_theme_dark = {
    .bg = {
        {37, 46, 60},   // NORMAL
        {58, 74, 96},   // PRESSED
        {16, 72, 176},  // SELECTED
        {52, 122, 228}, // SELECTED_PRESSED
        {28, 30, 34},   // DISABLED
    },
    .fg = {
        {237, 242, 249},
        {237, 242, 249},
        {255, 255, 255},
        {255, 255, 255},
        {110, 118, 130},
    },
    .border = {
        {70, 78, 92},
        {70, 78, 92},
        {70, 78, 92},
        {70, 78, 92},
        {52, 56, 64},
    },
    .bg_alpha = {1, 1, 1, 1, 1},
    .border_alpha = {1, 1, 1, 1, 1},
    .shadow_alpha = 0,
    .shadow_dy = 1,
    .shadow = {0, 0, 0},
};

static const Ui_Button_Theme s_theme_light = {
    .bg = {
        {238, 243, 250},
        {212, 226, 242},
        {17, 85, 238},
        {88, 142, 252},
        {224, 230, 234},
    },
    .fg = {
        {24, 32, 46},
        {24, 32, 46},
        {255, 255, 255},
        {255, 255, 255},
        {150, 158, 168},
    },
    .border = {
        {200, 210, 224},
        {200, 210, 224},
        {200, 210, 224},
        {200, 210, 224},
        {210, 216, 224},
    },
    .bg_alpha = {1, 1, 1, 1, 1},
    .border_alpha = {1, 1, 1, 1, 1},
    .shadow_alpha = 0,
    .shadow_dy = 1,
    .shadow = {0, 0, 0},
};

void ui_button_theme_default(Ui_Button_Theme *t, int32_t is_light) {
    if (t == NULL) return;
    *t = is_light ? s_theme_light : s_theme_dark;
}

// ===============================================================================
// 几何与状态
// ===============================================================================

static int32_t ui_button_min(int32_t a, int32_t b) {
    return (a < b) ? a : b;
}

static int32_t ui_button_abs(int32_t v) {
    return (v < 0) ? -v : v;
}

static int32_t ui_button_radius(const Ui_Button *b) {
    int32_t r = b->radius;
    switch (b->style) {
        case UI_BTN_STYLE_RECT:
        case UI_BTN_STYLE_TEXT:
            return 0;
        case UI_BTN_STYLE_PILL:
            return b->h / 2;
        case UI_BTN_STYLE_CIRCLE:
            return ui_button_min(b->w, b->h) / 2;
        default:
            if (r < 0) return b->h / 2;
            return r;
    }
}

static int32_t ui_button_is_pressed_visual(Ui_Button_Visual vis) {
    return vis == UI_BTN_VIS_PRESSED || vis == UI_BTN_VIS_SELECTED_PRESSED;
}

static int32_t ui_button_in_group_clip(const Ui_Button_Group *g, int32_t x, int32_t y) {
    if (g == NULL) return 1;
    if (g->clip_x1 <= g->clip_x0 || g->clip_y1 <= g->clip_y0) return 1;
    return (x >= g->clip_x0 && x < g->clip_x1 && y >= g->clip_y0 && y < g->clip_y1);
}

Ui_Button_Visual ui_button_visual(const Ui_Button *b, const Ui_Button_Tracker *t) {
    if (b == NULL) return UI_BTN_VIS_DISABLED;
    if (!b->enabled) return UI_BTN_VIS_DISABLED;

    int32_t pressed = 0;
    if (t != NULL && t->press_visual && t->pressed_id == b->id) {
        pressed = 1;
    }
    if (pressed && b->selected) return UI_BTN_VIS_SELECTED_PRESSED;
    if (pressed) return UI_BTN_VIS_PRESSED;
    if (b->selected) return UI_BTN_VIS_SELECTED;
    return UI_BTN_VIS_NORMAL;
}

int32_t ui_button_label_cx(const Ui_Button *b, Ui_Button_Visual vis) {
    int32_t off = ui_button_is_pressed_visual(vis) ? b->press_text_dx : 0;
    return b->x + b->w / 2 + off;
}

int32_t ui_button_label_cy(const Ui_Button *b, Ui_Button_Visual vis) {
    int32_t off = ui_button_is_pressed_visual(vis) ? b->press_text_dy : 0;
    return b->y + b->h / 2 + off;
}

int32_t ui_button_icon_cx(const Ui_Button *b, Ui_Button_Visual vis) {
    int32_t off = ui_button_is_pressed_visual(vis) ? b->press_icon_dx : 0;
    return b->x + b->w / 2 + off;
}

int32_t ui_button_icon_cy(const Ui_Button *b, Ui_Button_Visual vis) {
    int32_t off = ui_button_is_pressed_visual(vis) ? b->press_icon_dy : 0;
    return b->y + b->h / 2 + off;
}

int32_t ui_button_hit(const Ui_Button *b, int32_t x, int32_t y) {
    if (b == NULL || !b->visible || !b->enabled) return 0;

    if (b->style == UI_BTN_STYLE_CIRCLE || (b->flags & UI_BTN_FLAG_HIT_CIRCLE)) {
        int32_t cx = b->x + b->w / 2;
        int32_t cy = b->y + b->h / 2;
        int32_t r = ui_button_min(b->w, b->h) / 2 + b->hit_radius_pad;
        if (r < 1) r = 1;
        int32_t dx = x - cx;
        int32_t dy = y - cy;
        return (dx * dx + dy * dy <= r * r) ? 1 : 0;
    }

    int32_t pad = b->hit_pad;
    int32_t x0 = b->x - pad;
    int32_t y0 = b->y - pad;
    int32_t x1 = b->x + b->w + pad;
    int32_t y1 = b->y + b->h + pad;
    return (x >= x0 && x < x1 && y >= y0 && y < y1) ? 1 : 0;
}

int32_t ui_button_group_hit(const Ui_Button_Group *g, int32_t x, int32_t y) {
    if (g == NULL || g->items == NULL) return UI_BTN_ID_NONE;
    if (!ui_button_in_group_clip(g, x, y)) return UI_BTN_ID_NONE;
    for (int32_t i = 0; i < g->count; i++) {
        const Ui_Button *b = &g->items[i];
        if (ui_button_hit(b, x, y)) return b->id;
    }
    return UI_BTN_ID_NONE;
}

// ===============================================================================
// 初始化与简单 setter
// ===============================================================================

void ui_button_init(Ui_Button *b, int32_t id,
                    int32_t x, int32_t y, int32_t w, int32_t h) {
    if (b == NULL) return;
    memset(b, 0, sizeof(*b));
    b->id = id;
    b->x = x;
    b->y = y;
    b->w = w;
    b->h = h;
    b->style = UI_BTN_STYLE_RECT;
    b->flags = UI_BTN_FLAG_DRAW_BG;
    b->radius = 0;
    b->font_id = GFX_FONT_ALPHA_12;
    b->value_font_id = GFX_FONT_ALPHA_12;
    b->label_dy = 0;
    b->value_dy = 0;
    b->press_text_dx = 0;
    b->press_text_dy = UI_BTN_PRESS_TEXT_DY_DEFAULT;
    b->press_icon_dx = 0;
    b->press_icon_dy = 0;
    b->hit_pad = 0;
    b->hit_radius_pad = 0;
    b->repeat_delay_ms = 0;
    b->repeat_interval_ms = 0;
    b->visible = 1;
    b->enabled = 1;
    b->selected = 0;
    b->quiet = 0;
    b->text_mode = 1;
    b->theme = NULL;
    b->content_draw = NULL;
    b->content_user = NULL;
}

void ui_button_set_text(Ui_Button *b, const wchar_t *label, const wchar_t *value) {
    if (b == NULL) return;
    b->label = label;
    b->value = value;
}

void ui_button_set_press_offset(Ui_Button *b, int8_t text_dx, int8_t text_dy) {
    if (b == NULL) return;
    b->press_text_dx = text_dx;
    b->press_text_dy = text_dy;
}

void ui_button_set_icon_press_offset(Ui_Button *b, int8_t icon_dx, int8_t icon_dy) {
    if (b == NULL) return;
    b->press_icon_dx = icon_dx;
    b->press_icon_dy = icon_dy;
}

// ===============================================================================
// 绘制
// ===============================================================================

static void ui_button_draw_rrect(Nano_GFX *gfx, int32_t x, int32_t y, int32_t w, int32_t h,
                                 int32_t r, const Ui_RGB *c, uint8_t mode) {
    if (gfx == NULL || c == NULL || w <= 0 || h <= 0) return;
    int32_t rm = ui_button_min(w, h) / 2;
    if (r < 0) r = rm;
    if (r > rm) r = rm;
    gfx_draw_rounded_rectangle(gfx, x, y, w, h, r, r, r, r, c->r, c->g, c->b, mode);
}

static uint8_t ui_button_alpha_mode(uint8_t a) {
    if (a <= 1) return 1;
    if (a < 4) return 4;
    return a;
}

static void ui_button_draw_default_text(Nano_GFX *gfx, const Ui_Button *b,
                                        const Ui_Button_Theme *theme,
                                        Ui_Button_Visual vis) {
    if (b->label == NULL) return;

    Ui_RGB fg = theme->fg[vis];
    if (b->quiet) fg = theme->fg[UI_BTN_VIS_DISABLED];

    int32_t cx = ui_button_label_cx(b, vis);
    int32_t cy = ui_button_label_cy(b, vis);
    uint8_t mode = (b->text_mode != 0) ? b->text_mode : 1;

    if (b->value != NULL) {
        gfx_font_draw_text_centered(gfx, b->font_id, (wchar_t *)b->label,
            cx, cy + b->label_dy, fg.r, fg.g, fg.b, mode);
        gfx_font_draw_text_centered(gfx, b->value_font_id, (wchar_t *)b->value,
            cx, cy + b->value_dy, fg.r, fg.g, fg.b, mode);
    }
    else {
        gfx_font_draw_text_centered(gfx, b->font_id, (wchar_t *)b->label,
            cx, cy + b->label_dy, fg.r, fg.g, fg.b, mode);
    }
}

void ui_button_draw(Nano_GFX *gfx, const Ui_Button *b,
                    const Ui_Button_Theme *theme, Ui_Button_Visual vis) {
    if (gfx == NULL || b == NULL || !b->visible || b->w <= 0 || b->h <= 0) return;
    if (theme == NULL) theme = &s_theme_dark;

    int32_t idx = (int32_t)vis;
    if (idx < 0) idx = 0;
    if (idx >= UI_BTN_VIS_COUNT) idx = UI_BTN_VIS_NORMAL;

    int32_t r = ui_button_radius(b);
    int32_t draw_bg = ((b->flags & UI_BTN_FLAG_DRAW_BG) != 0) && (b->style != UI_BTN_STYLE_TEXT);
    int32_t draw_border = ((b->flags & UI_BTN_FLAG_DRAW_BORDER) != 0) && (b->style != UI_BTN_STYLE_TEXT);
    int32_t border_w = (b->flags & UI_BTN_FLAG_DOUBLE_BORDER) ? 2 : 1;

    if ((b->flags & UI_BTN_FLAG_SHADOW) && theme->shadow_alpha > 0) {
        ui_button_draw_rrect(gfx, b->x, b->y + theme->shadow_dy, b->w, b->h, r,
                              &theme->shadow, ui_button_alpha_mode(theme->shadow_alpha));
    }

    if (draw_border && theme->border_alpha[idx] != 0) {
        ui_button_draw_rrect(gfx, b->x, b->y, b->w, b->h, r,
                             &theme->border[idx], ui_button_alpha_mode(theme->border_alpha[idx]));
        if (draw_bg && theme->bg_alpha[idx] != 0) {
            ui_button_draw_rrect(gfx, b->x + border_w, b->y + border_w,
                                 b->w - 2 * border_w, b->h - 2 * border_w,
                                 (r > border_w) ? (r - border_w) : 0,
                                 &theme->bg[idx], ui_button_alpha_mode(theme->bg_alpha[idx]));
        }
    }
    else if (draw_bg && theme->bg_alpha[idx] != 0) {
        ui_button_draw_rrect(gfx, b->x, b->y, b->w, b->h, r,
                             &theme->bg[idx], ui_button_alpha_mode(theme->bg_alpha[idx]));
    }

    int32_t base_cx = b->x + b->w / 2;
    int32_t base_cy = b->y + b->h / 2;

    if (b->content_draw != NULL) {
        b->content_draw(gfx, b, base_cx, base_cy, vis, b->content_user);
    }
    else {
        ui_button_draw_default_text(gfx, b, theme, vis);
    }
}

void ui_button_group_draw(Nano_GFX *gfx, const Ui_Button_Group *g,
                          const Ui_Button_Tracker *t) {
    if (gfx == NULL || g == NULL || g->items == NULL) return;
    for (int32_t i = 0; i < g->count; i++) {
        const Ui_Button *b = &g->items[i];
        if (!b->visible) continue;
        const Ui_Button_Theme *th = (b->theme != NULL) ? b->theme : g->theme;
        ui_button_draw(gfx, b, th, ui_button_visual(b, t));
    }
}

// ===============================================================================
// 简单 tracker
// ===============================================================================

void ui_button_tracker_reset(Ui_Button_Tracker *t) {
    if (t == NULL) return;
    memset(t, 0, sizeof(*t));
    t->pressed_id = UI_BTN_ID_NONE;
    t->require_up_edge = 1;
}

static const Ui_Button *ui_button_find(const Ui_Button_Group *g, int32_t id) {
    if (g == NULL || g->items == NULL) return NULL;
    for (int32_t i = 0; i < g->count; i++) {
        if (g->items[i].id == id) return &g->items[i];
    }
    return NULL;
}

Ui_Button_Result ui_button_group_event(Ui_Button_Group *g,
                                       Ui_Button_Tracker *t,
                                       const Key_Event *ke,
                                       uint64_t now_ms) {
    Ui_Button_Result r;
    memset(&r, 0, sizeof(r));
    r.pressed_id = UI_BTN_ID_NONE;
    r.clicked_id = UI_BTN_ID_NONE;
    r.repeated_id = UI_BTN_ID_NONE;
    r.cancelled_id = UI_BTN_ID_NONE;

    if (g == NULL || t == NULL || ke == NULL) return r;

    int32_t old_pressed = t->pressed_id;
    uint8_t old_visual = t->press_visual;

    int32_t down = (ke->touch_edge & TOUCH_EDGE_DOWN) != 0;
    int32_t up = (ke->touch_edge & TOUCH_EDGE_UP) != 0;
    int32_t touching = ke->is_touching != 0;

    // 按下
    if (down || (touching && !t->active)) {
        t->active = 1;
        t->dragging = 0;
        t->repeat_started = 0;
        t->repeat_next_ts = 0;
        t->down_x = down ? ke->touch_down_x : ke->touch_x;
        t->down_y = down ? ke->touch_down_y : ke->touch_y;
        t->pressed_id = ui_button_group_hit(g, t->down_x, t->down_y);
        t->press_visual = (t->pressed_id != UI_BTN_ID_NONE) ? 1 : 0;
        t->press_ts = now_ms;
        r.pressed_id = t->pressed_id;
    }
    // 按住
    else if (touching && t->active) {
        if (t->drag_threshold > 0) {
            int32_t dx = ui_button_abs(ke->touch_x - t->down_x);
            int32_t dy = ui_button_abs(ke->touch_y - t->down_y);
            if (dx > t->drag_threshold || dy > t->drag_threshold) {
                t->dragging = 1;
                if (t->follow_press) t->press_visual = 0;
            }
        }

        if (t->follow_press && !t->dragging) {
            int32_t cur = ui_button_group_hit(g, ke->touch_x, ke->touch_y);
            t->press_visual = (cur == t->pressed_id && t->pressed_id != UI_BTN_ID_NONE) ? 1 : 0;
        }

        if (t->pressed_id != UI_BTN_ID_NONE && !t->dragging) {
            const Ui_Button *b = ui_button_find(g, t->pressed_id);
            if (b != NULL && b->repeat_interval_ms > 0 &&
                now_ms - t->press_ts >= b->repeat_delay_ms) {
                if (t->repeat_next_ts == 0) t->repeat_next_ts = now_ms;
                if (now_ms >= t->repeat_next_ts) {
                    r.repeated_id = t->pressed_id;
                    t->repeat_started = 1;
                    t->repeat_next_ts = now_ms + b->repeat_interval_ms;
                }
            }
        }
    }

    // 松手（或兼容电平兜底）
    if (up || (!touching && t->active && !t->require_up_edge)) {
        if (t->pressed_id != UI_BTN_ID_NONE) {
            if (!t->dragging) {
                int32_t up_id = ui_button_group_hit(g, ke->touch_x, ke->touch_y);
                if (up_id == t->pressed_id && !t->repeat_started) {
                    r.clicked_id = t->pressed_id;
                }
                else {
                    r.cancelled_id = t->pressed_id;
                }
            }
            else {
                r.cancelled_id = t->pressed_id;
            }
        }
        t->active = 0;
        t->pressed_id = UI_BTN_ID_NONE;
        t->press_visual = 0;
        t->repeat_next_ts = 0;
    }

    if (old_pressed != t->pressed_id || old_visual != t->press_visual) {
        r.visual_changed = 1;
    }
    return r;
}
