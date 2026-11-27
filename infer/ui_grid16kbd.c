#include <stdio.h>
#include <string.h>

#include "ui_grid16kbd.h"
#include "ui_button.h"

// ===============================================================================
// 16键虚拟键盘（硬件无关实现）
//
// 文本输入控件"16键"模式的显式虚拟键盘：屏幕下半屏绘制 4x4 圆角按钮，
// 点按产出 NANO_KEY_* 键码（键码的唯一触屏来源，触屏本身不派生任何按键事件）。
// 机制与触屏软键盘（ui_softkbd.c）同范式：Core1 轮询产键、Core0 绘制、
// 跨任务共享状态经 volatile 传递（单一写者）。
// ===============================================================================

// 按钮间距（px）：按钮以 4x4 均布于键盘区域，间隙处不命中（落在间隙的按压被吞掉）。
// 间距与键盘边缘间距统一为 2px；纵向 150-5*2=140 恰好整除（行高35），
// 横向 320-5*2=310 不能被 4 整除（77.5），余数 2px 分配给前 2 列（78px）、后 2 列（77px），
// 由 ui_grid16kbd_col_geometry 统一计算（绘制与命中判定共用，保证严格一致）。
#define UI_GRID16KBD_GAP      (2)
#define UI_GRID16KBD_RADIUS   (4)  // 按钮圆角半径
#define UI_GRID16KBD_CELL_H   ((UI_GRID16KBD_HEIGHT - (UI_GRID16KBD_ROWS + 1) * UI_GRID16KBD_GAP) / UI_GRID16KBD_ROWS)

// 逐列几何：列 col 的左缘 x 与宽度 w（横向余数分配给前列，间距/边距严格 UI_GRID16KBD_GAP）
static void ui_grid16kbd_col_geometry(int32_t col, int32_t *out_x, int32_t *out_w) {
    int32_t content = SCREEN_WIDTH - (UI_GRID16KBD_COLS + 1) * UI_GRID16KBD_GAP; // 310
    int32_t base = content / UI_GRID16KBD_COLS; // 77
    int32_t rem  = content % UI_GRID16KBD_COLS; // 2
    int32_t x = UI_GRID16KBD_GAP;
    for (int32_t c = 0; c < col; c++) {
        x += base + ((c < rem) ? 1 : 0) + UI_GRID16KBD_GAP;
    }
    *out_x = x;
    *out_w = base + ((col < rem) ? 1 : 0);
}

// 键盘配色（RGB888，与触屏软键盘同色系）
#define UI_GRID16KBD_COLOR_BORDER    46, 46, 50    // 按钮间隙（键盘区域底色）
#define UI_GRID16KBD_COLOR_KEY       28, 28, 32    // 普通键底色
#define UI_GRID16KBD_COLOR_CTRL_ON   16, 48, 160   // 全局Ctrl激活时 Ctrl 键底色
#define UI_GRID16KBD_COLOR_PRESSED   96, 96, 104   // 按住高亮底色
#define UI_GRID16KBD_COLOR_LABEL     240, 240, 240 // 键名文字颜色

// 键码表（4x4，布局与实体十六键一致）
static const uint8_t S_GRID16KBD_KEYCODE[UI_GRID16KBD_ROWS][UI_GRID16KBD_COLS] = {
    {NANO_KEY_1,    NANO_KEY_2, NANO_KEY_3,     NANO_KEY_esc},
    {NANO_KEY_4,    NANO_KEY_5, NANO_KEY_6,     NANO_KEY_shift},
    {NANO_KEY_7,    NANO_KEY_8, NANO_KEY_9,     NANO_KEY_ctrl},
    {NANO_KEY_left, NANO_KEY_0, NANO_KEY_right, NANO_KEY_enter},
};

// 按钮内容（同已移除的"九键按键提示遮罩"）：每格 {第一行, 第二行}；
// 第二行为 NULL 表示单行（16px 字体），否则双行（12px 字体）。
static const wchar_t *S_GRID16KBD_LABEL_NORMAL[4][4][2] = {
    {{L"1", L"符号"}, {L"2", L"ABC"}, {L"3", L"DEF"}, {L"退格", NULL}},
    {{L"4", L"GHI"},  {L"5", L"JKL"}, {L"6", L"MNO"}, {L"输入法", NULL}},
    {{L"7", L"PQRS"}, {L"8", L"TUV"}, {L"9", L"WXYZ"}, {L"Ctrl", NULL}},
    {{L"←", NULL},     {L"0", NULL}, {L"→", NULL},    {L"确认", NULL}},
};
static const wchar_t *S_GRID16KBD_LABEL_CTRL[4][4][2] = {
    {{L"符号", NULL},      {L"2", L"ABC"},     {L"3", L"DEF"}, {L"退出", NULL}},
    {{L"4", L"GHI"},      {L"5", L"JKL"},     {L"6", L"MNO"}, {L"输入法", NULL}},
    {{L"7", L"PQRS"},     {L"8", L"TUV"}, {L"9", L"WXYZ"}, {L"[Ctrl]", NULL}},
    {{L"↑", NULL},        {L"键盘", NULL},  {L"↓", NULL},  {L"换行", NULL}},
};

// 跨任务共享状态（单一写者，故仅需 volatile）：
//   s_visible     ：写-渲染任务（show/hide），读-轮询任务（poll）与布局计算
//   s_press_row/col：写-轮询任务，读-渲染任务（按住高亮）
//   s_dirty       ：写-轮询任务，读/清-渲染任务
static volatile uint8_t s_visible = 0;
static volatile int8_t  s_press_row = -1;
static volatile int8_t  s_press_col = -1;
static volatile uint8_t s_dirty = 0;
static volatile uint8_t s_claimed = 0;  // 当前触摸是否落在键盘区域内
static uint8_t s_prev_pressed = 0;      // 上一轮询的触摸状态（用于按下沿检测）
static uint8_t s_held_code = NANO_KEY_IDLE; // 按住期间锁存的键码（按下沿解析一次，保证按住期间键码不变）

void ui_grid16kbd_init() {
    s_visible = 0;
    s_press_row = -1;
    s_press_col = -1;
    s_dirty = 0;
    s_claimed = 0;
    s_prev_pressed = 0;
    s_held_code = NANO_KEY_IDLE;
}

uint8_t ui_grid16kbd_is_visible() {
    return s_visible;
}

void ui_grid16kbd_show() {
    s_visible = 1;
}

void ui_grid16kbd_hide() {
    s_visible = 0;
    s_press_row = -1;
    s_press_col = -1;
}

int32_t ui_grid16kbd_height() {
    return (s_visible) ? UI_GRID16KBD_HEIGHT : 0;
}

uint8_t ui_grid16kbd_touch_claimed() {
    return s_claimed;
}

uint8_t ui_grid16kbd_take_dirty() {
    uint8_t d = s_dirty;
    s_dirty = 0;
    return d;
}

// 命中判定：严格的按钮矩形包含（按钮之间的间隙不命中）。
// 命中返回1并输出行列；未命中（含间隙）返回0。
static int32_t ui_grid16kbd_hit_test(int32_t x, int32_t y, int32_t *out_row, int32_t *out_col) {
    int32_t kbd_y = SCREEN_HEIGHT - UI_GRID16KBD_HEIGHT;
    if (x < UI_GRID16KBD_GAP || x >= SCREEN_WIDTH - UI_GRID16KBD_GAP) return 0;
    if (y < kbd_y + UI_GRID16KBD_GAP || y >= SCREEN_HEIGHT - UI_GRID16KBD_GAP) return 0;

    // 列：逐列几何（宽度不完全均分，余数分配给前列，见 ui_grid16kbd_col_geometry）
    int32_t col = -1;
    for (int32_t c = 0; c < UI_GRID16KBD_COLS; c++) {
        int32_t cx = 0, cw = 0;
        ui_grid16kbd_col_geometry(c, &cx, &cw);
        if (x >= cx && x < cx + cw) { col = c; break; }
        if (x < cx + cw + UI_GRID16KBD_GAP) return 0; // 落在该列之后的间隙
    }
    if (col < 0) return 0;

    // 行：严格均分（CELL_H 整除）
    int32_t pitch_y = UI_GRID16KBD_CELL_H + UI_GRID16KBD_GAP;
    int32_t row = (y - kbd_y - UI_GRID16KBD_GAP) / pitch_y;
    if (row < 0 || row >= UI_GRID16KBD_ROWS) return 0;
    // 排除按钮间隙
    if ((y - kbd_y - UI_GRID16KBD_GAP) % pitch_y >= UI_GRID16KBD_CELL_H) return 0;

    *out_row = row;
    *out_col = col;
    return 1;
}

uint8_t ui_grid16kbd_poll(int32_t x, int32_t y, int32_t is_pressed) {
    s_claimed = 0;

    // 松开：清除按住高亮与锁存键码，准备下一次按下沿
    if (!is_pressed) {
        s_prev_pressed = 0;
        s_held_code = NANO_KEY_IDLE;
        if (s_press_row >= 0) {
            s_press_row = -1;
            s_press_col = -1;
            s_dirty = 1;
        }
        return NANO_KEY_IDLE;
    }

    // 键盘区域外：不接管（滑出键盘时取消按住，允许滑入键盘时重新触发按下沿）
    int32_t kbd_y = SCREEN_HEIGHT - UI_GRID16KBD_HEIGHT;
    if (y < kbd_y) {
        s_prev_pressed = 0;
        s_held_code = NANO_KEY_IDLE;
        return NANO_KEY_IDLE;
    }
    s_claimed = 1;

    // 按住中：持续上报锁存键码（交给框架原生的长按/连发机制）
    if (s_prev_pressed) {
        return s_held_code;
    }
    s_prev_pressed = 1;

    int32_t row = -1, col = -1;
    if (!ui_grid16kbd_hit_test(x, y, &row, &col)) {
        s_held_code = NANO_KEY_IDLE;
        return NANO_KEY_IDLE; // 落在按钮间隙：吞掉即可
    }

    s_press_row = (int8_t)row;
    s_press_col = (int8_t)col;
    s_dirty = 1;

    s_held_code = S_GRID16KBD_KEYCODE[row][col];
    return s_held_code;
}

typedef struct {
    const wchar_t *line0;
    const wchar_t *line1;
    int32_t off;
} Grid16_Content;

static void grid16_key_content(Nano_GFX *gfx, const Ui_Button *b,
                               int32_t base_cx, int32_t base_cy,
                               Ui_Button_Visual vis, void *user) {
    (void)base_cx;
    (void)base_cy;
    const Grid16_Content *c = (const Grid16_Content *)user;
    int32_t cx = b->x + b->w / 2;
    int32_t cy = ui_button_label_cy(b, vis);
    if (c->line1 != NULL) {
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)c->line0,
                                    cx, cy - c->off, UI_GRID16KBD_COLOR_LABEL, 1);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)c->line1,
                                    cx, cy + c->off, UI_GRID16KBD_COLOR_LABEL, 1);
    }
    else {
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_16, (wchar_t *)c->line0,
                                    cx, cy, UI_GRID16KBD_COLOR_LABEL, 1);
    }
}

void ui_grid16kbd_draw(Nano_GFX *gfx, uint8_t is_ctrl_active) {
    if (!s_visible) return;

    int32_t kbd_y = SCREEN_HEIGHT - UI_GRID16KBD_HEIGHT;

    // 先以间隙色填充整个键盘区域，各按钮向内绘制，形成均匀间距
    gfx_draw_rectangle(gfx, 0, kbd_y, SCREEN_WIDTH, UI_GRID16KBD_HEIGHT, UI_GRID16KBD_COLOR_BORDER, 1);

    const wchar_t *(*grid)[4][2] = (is_ctrl_active) ? S_GRID16KBD_LABEL_CTRL : S_GRID16KBD_LABEL_NORMAL;

    Ui_Button_Theme theme;
    memset(&theme, 0, sizeof(theme));
    for (int32_t i = 0; i < UI_BTN_VIS_COUNT; i++) {
        theme.bg[i] = (Ui_RGB){28, 28, 32};
        theme.fg[i] = (Ui_RGB){240, 240, 240};
        theme.bg_alpha[i] = 1;
        theme.border_alpha[i] = 0;
    }
    theme.bg[UI_BTN_VIS_PRESSED] = (Ui_RGB){96, 96, 104};
    theme.bg[UI_BTN_VIS_SELECTED] = (Ui_RGB){16, 48, 160};

    for (int32_t r = 0; r < UI_GRID16KBD_ROWS; r++) {
        for (int32_t c = 0; c < UI_GRID16KBD_COLS; c++) {
            int32_t x = 0, cell_w = 0;
            ui_grid16kbd_col_geometry(c, &x, &cell_w);
            int32_t y = kbd_y + UI_GRID16KBD_GAP + r * (UI_GRID16KBD_CELL_H + UI_GRID16KBD_GAP);

            const wchar_t *line0 = grid[r][c][0];
            const wchar_t *line1 = grid[r][c][1];
            int32_t selected = (S_GRID16KBD_KEYCODE[r][c] == NANO_KEY_ctrl && is_ctrl_active) ? 1 : 0;
            int32_t pressed = (r == s_press_row && c == s_press_col) ? 1 : 0;

            Ui_Button b;
            ui_button_init(&b, r * UI_GRID16KBD_COLS + c, x, y, cell_w, UI_GRID16KBD_CELL_H);
            b.style = UI_BTN_STYLE_ROUNDED;
            b.radius = UI_GRID16KBD_RADIUS;
            b.flags = UI_BTN_FLAG_DRAW_BG;
            b.font_id = (line1 != NULL) ? GFX_FONT_ALPHA_12 : GFX_FONT_ALPHA_16;
            ui_button_set_text(&b, line0, NULL);

            Grid16_Content content;
            content.line0 = line0;
            content.line1 = line1;
            content.off = UI_GRID16KBD_CELL_H / 4;
            b.content_draw = grid16_key_content;
            b.content_user = &content;

            Ui_Button_Visual vis = pressed ? UI_BTN_VIS_PRESSED
                                  : (selected ? UI_BTN_VIS_SELECTED : UI_BTN_VIS_NORMAL);
            ui_button_draw(gfx, &b, &theme, vis);
        }
    }
}
