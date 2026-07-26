#include <stdio.h>

#include "ui_softkbd.h"

// ===============================================================================
// 触屏软键盘（硬件无关实现）
//
// 布局与 main.cpp-ref 一致（4行x12列，对应 input_device.h 注释中的48键物理键盘），
// 键码映射到本项目的 NANO_KEY_* 定义：
//   - 可打印字符：键码即ASCII码（NANO_KEY_a == 'a' 等），分 base/shift/sym 三层；
//   - 功能键：ESC/BS/ENT/SP/方向键映射到 NANO_KEY_esc/backspace/enter/space/up等；
//   - SFT/SYM 为粘滞修饰键（参照 main.cpp-ref 的 sticky 机制），切换层状态；
//   - CTRL/ALT 直接产生 NANO_KEY_ctrl / NANO_KEY_alt 键事件，复用本项目UI框架
//     原生的Ctrl机制（is_ctrl_enabled，如 Ctrl+D 换行、Ctrl+B 帮助）。
// ===============================================================================

// 命中判定阈值（分轴归一化欧氏距离）：取最近键心，距离不超过阈值时命中。
// 0.707 时命中区域恰好密铺键盘（Voronoi 划分）；0.75 在键盘外轮廓外保留一段容忍带。
#define UI_SOFTKBD_HIT_THRESHOLD (0.75f)

// 边缘列判定补偿（px）：电容屏边缘效应使贴边框按压的触点质心系统性向屏幕内侧偏移
// （实测：最左列按压 Ctrl 误触发其右侧 Z，偏移明显大于内部键）。窄键（约26.7px）下
// 该物理偏差会越过键心中点判定边界，故将最左/最右两条列判定边界向外扩本值。
// 内部列边界保持键心中点不变。实测可微调。
#define UI_SOFTKBD_EDGE_BIAS_X (8)
// 最底行垂直方向边缘补偿（px）：>0 时，最底行与上一行的判定边界上移本值。默认0（不补偿）
#define UI_SOFTKBD_EDGE_BIAS_Y (0)

// 粘滞修饰键
#define UI_SOFTKBD_MOD_NONE  (0)
#define UI_SOFTKBD_MOD_SHIFT (1)
#define UI_SOFTKBD_MOD_SYM   (2)

// 键盘配色（RGB888）
#define UI_SOFTKBD_COLOR_BORDER    46, 46, 50    // 键间边界（键盘区域底色）
#define UI_SOFTKBD_COLOR_KEY       28, 28, 32    // 普通键底色
#define UI_SOFTKBD_COLOR_MOD_ON    16, 48, 160   // 粘滞修饰键激活底色
#define UI_SOFTKBD_COLOR_PRESSED   96, 96, 104   // 按下瞬间高亮底色
#define UI_SOFTKBD_COLOR_LABEL     240, 240, 240 // 键名文字颜色

typedef struct UI_Softkbd_Key {
    wchar_t label[5]; // 键名（如 L"!1"、L"ESC"）
    uint8_t base;     // 默认层键码（NANO_KEY_*）
    uint8_t shift;    // Shift层键码
    uint8_t sym;      // Sym层键码
    uint8_t mod;      // 粘滞修饰键：UI_SOFTKBD_MOD_*，0表示普通键
} UI_Softkbd_Key;

// 物理键盘布局（16*3=48键），见 input_device.h 头部注释：
//   ESC    !1    @2    #3    $4    %5    ^6    &7    *8    (9    )0    BS
//   ALT    `Q    ~W    {E    }R    [T    ]Y    +U    -I    _O    =P    SYM
//   SFT     A     S    |D    \F    :G    ;H    /J    ,K    .L     ↑    ENT
//   CTRL    Z     X    <C    >V    ?B    'N    "M    SP     ←     ↓     →
static UI_Softkbd_Key S_SOFTKBD[UI_SOFTKBD_ROWS][UI_SOFTKBD_COLS] = {
    {
        {L"ESC",  NANO_KEY_esc,       NANO_KEY_esc,       NANO_KEY_esc,       UI_SOFTKBD_MOD_NONE},
        {L"!1",   NANO_KEY_1,         NANO_KEY_bang,      NANO_KEY_bang,      UI_SOFTKBD_MOD_NONE},
        {L"@2",   NANO_KEY_2,         NANO_KEY_at,        NANO_KEY_at,        UI_SOFTKBD_MOD_NONE},
        {L"#3",   NANO_KEY_3,         NANO_KEY_hash,      NANO_KEY_hash,      UI_SOFTKBD_MOD_NONE},
        {L"$4",   NANO_KEY_4,         NANO_KEY_dollar,    NANO_KEY_dollar,    UI_SOFTKBD_MOD_NONE},
        {L"%5",   NANO_KEY_5,         NANO_KEY_percent,   NANO_KEY_percent,   UI_SOFTKBD_MOD_NONE},
        {L"^6",   NANO_KEY_6,         NANO_KEY_caret,     NANO_KEY_caret,     UI_SOFTKBD_MOD_NONE},
        {L"&7",   NANO_KEY_7,         NANO_KEY_and,       NANO_KEY_and,       UI_SOFTKBD_MOD_NONE},
        {L"*8",   NANO_KEY_8,         NANO_KEY_star,      NANO_KEY_star,      UI_SOFTKBD_MOD_NONE},
        {L"(9",   NANO_KEY_9,         NANO_KEY_parenl,    NANO_KEY_parenl,    UI_SOFTKBD_MOD_NONE},
        {L")0",   NANO_KEY_0,         NANO_KEY_parenr,    NANO_KEY_parenr,    UI_SOFTKBD_MOD_NONE},
        {L"BS",   NANO_KEY_backspace, NANO_KEY_backspace, NANO_KEY_backspace, UI_SOFTKBD_MOD_NONE},
    },
    {
        {L"ALT",  NANO_KEY_alt,  NANO_KEY_alt,      NANO_KEY_alt,      UI_SOFTKBD_MOD_NONE},
        {L"`Q",   NANO_KEY_q,    NANO_KEY_Q,        NANO_KEY_backtick, UI_SOFTKBD_MOD_NONE},
        {L"~W",   NANO_KEY_w,    NANO_KEY_W,        NANO_KEY_tilde,    UI_SOFTKBD_MOD_NONE},
        {L"{E",   NANO_KEY_e,    NANO_KEY_E,        NANO_KEY_bracel,   UI_SOFTKBD_MOD_NONE},
        {L"}R",   NANO_KEY_r,    NANO_KEY_R,        NANO_KEY_bracer,   UI_SOFTKBD_MOD_NONE},
        {L"[T",   NANO_KEY_t,    NANO_KEY_T,        NANO_KEY_bracketl, UI_SOFTKBD_MOD_NONE},
        {L"]Y",   NANO_KEY_y,    NANO_KEY_Y,        NANO_KEY_bracketr, UI_SOFTKBD_MOD_NONE},
        {L"+U",   NANO_KEY_u,    NANO_KEY_U,        NANO_KEY_plus,     UI_SOFTKBD_MOD_NONE},
        {L"-I",   NANO_KEY_i,    NANO_KEY_I,        NANO_KEY_dash,     UI_SOFTKBD_MOD_NONE},
        {L"_O",   NANO_KEY_o,    NANO_KEY_O,        NANO_KEY_underscore, UI_SOFTKBD_MOD_NONE},
        {L"=P",   NANO_KEY_p,    NANO_KEY_P,        NANO_KEY_eq,       UI_SOFTKBD_MOD_NONE},
        {L"SYM",  0,             0,                 0,                 UI_SOFTKBD_MOD_SYM},
    },
    {
        {L"SFT",  0,             0,                 0,                 UI_SOFTKBD_MOD_SHIFT},
        {L"A",    NANO_KEY_a,    NANO_KEY_A,        NANO_KEY_a,        UI_SOFTKBD_MOD_NONE},
        {L"S",    NANO_KEY_s,    NANO_KEY_S,        NANO_KEY_s,        UI_SOFTKBD_MOD_NONE},
        {L"|D",   NANO_KEY_d,    NANO_KEY_D,        NANO_KEY_pipe,     UI_SOFTKBD_MOD_NONE},
        {L"\\F",  NANO_KEY_f,    NANO_KEY_F,        NANO_KEY_backslash, UI_SOFTKBD_MOD_NONE},
        {L":G",   NANO_KEY_g,    NANO_KEY_G,        NANO_KEY_colon,    UI_SOFTKBD_MOD_NONE},
        {L";H",   NANO_KEY_h,    NANO_KEY_H,        NANO_KEY_semicolon, UI_SOFTKBD_MOD_NONE},
        {L"/J",   NANO_KEY_j,    NANO_KEY_J,        NANO_KEY_slash,    UI_SOFTKBD_MOD_NONE},
        {L",K",   NANO_KEY_k,    NANO_KEY_K,        NANO_KEY_comma,    UI_SOFTKBD_MOD_NONE},
        {L".L",   NANO_KEY_l,    NANO_KEY_L,        NANO_KEY_dot,      UI_SOFTKBD_MOD_NONE},
        {L"↑",   NANO_KEY_up,    NANO_KEY_up,       NANO_KEY_up,       UI_SOFTKBD_MOD_NONE},
        {L"ENT",  NANO_KEY_enter, NANO_KEY_enter,   NANO_KEY_enter,    UI_SOFTKBD_MOD_NONE},
    },
    {
        {L"CTRL", NANO_KEY_ctrl,  NANO_KEY_ctrl,    NANO_KEY_ctrl,     UI_SOFTKBD_MOD_NONE},
        {L"Z",    NANO_KEY_z,     NANO_KEY_Z,       NANO_KEY_z,        UI_SOFTKBD_MOD_NONE},
        {L"X",    NANO_KEY_x,     NANO_KEY_X,       NANO_KEY_x,        UI_SOFTKBD_MOD_NONE},
        {L"<C",   NANO_KEY_c,     NANO_KEY_C,       NANO_KEY_lt,       UI_SOFTKBD_MOD_NONE},
        {L">V",   NANO_KEY_v,     NANO_KEY_V,       NANO_KEY_gt,       UI_SOFTKBD_MOD_NONE},
        {L"?B",   NANO_KEY_b,     NANO_KEY_B,       NANO_KEY_ques,     UI_SOFTKBD_MOD_NONE},
        {L"'N",   NANO_KEY_n,     NANO_KEY_N,       NANO_KEY_quote1,   UI_SOFTKBD_MOD_NONE},
        {L"\"M",  NANO_KEY_m,     NANO_KEY_M,       NANO_KEY_quote2,   UI_SOFTKBD_MOD_NONE},
        {L"SP",   NANO_KEY_space, NANO_KEY_space,   NANO_KEY_space,    UI_SOFTKBD_MOD_NONE},
        {L"←",   NANO_KEY_left,  NANO_KEY_left,    NANO_KEY_left,     UI_SOFTKBD_MOD_NONE},
        {L"↓",   NANO_KEY_down,  NANO_KEY_down,    NANO_KEY_down,     UI_SOFTKBD_MOD_NONE},
        {L"→",   NANO_KEY_right, NANO_KEY_right,   NANO_KEY_right,    UI_SOFTKBD_MOD_NONE},
    },
};

// 跨任务共享状态（单一写者，故仅需 volatile）：
//   s_visible    ：写-渲染任务（show/hide），读-轮询任务（poll）与布局计算
//   s_sticky     ：写-轮询任务（poll），读-渲染任务（draw）
//   s_press_row/col：写-轮询任务，读-渲染任务（按下高亮）
//   s_dirty      ：写-轮询任务，读/清-渲染任务
static volatile uint8_t s_visible = 0;
static volatile uint8_t s_toggle_request = 0; // 切换显隐请求（写-轮询任务，读/清-渲染任务）
static volatile uint8_t s_sticky = UI_SOFTKBD_MOD_NONE;
static volatile int8_t  s_press_row = -1;
static volatile int8_t  s_press_col = -1;
static volatile uint8_t s_dirty = 0;
static volatile uint8_t s_claimed = 0;  // 当前触摸是否落在键盘区域内
static uint8_t s_prev_pressed = 0;      // 上一轮询的触摸状态（用于按下沿检测）
static uint8_t s_held_code = NANO_KEY_IDLE; // 按住期间锁存的键码（按下沿解析一次，保证按住期间键码不变）
static uint8_t s_held_no_repeat = 0;    // 当前按住的键不可重复（Ctrl/SFT/Alt/SYM/Esc）：仅按下沿一次性上报

void ui_softkbd_init() {
    touch_init();
    s_visible = 0;
    s_toggle_request = 0;
    s_sticky = UI_SOFTKBD_MOD_NONE;
    s_press_row = -1;
    s_press_col = -1;
    s_dirty = 0;
    s_claimed = 0;
    s_prev_pressed = 0;
    s_held_code = NANO_KEY_IDLE;
    s_held_no_repeat = 0;
}

uint8_t ui_softkbd_is_visible() {
    return s_visible;
}

void ui_softkbd_request_toggle() {
    s_toggle_request = 1;
}

uint8_t ui_softkbd_take_toggle_request() {
    uint8_t r = s_toggle_request;
    s_toggle_request = 0;
    return r;
}

void ui_softkbd_show() {
    s_visible = 1;
}

void ui_softkbd_hide() {
    s_visible = 0;
    s_sticky = UI_SOFTKBD_MOD_NONE;
    s_press_row = -1;
    s_press_col = -1;
}

int32_t ui_softkbd_height() {
    return (s_visible) ? UI_SOFTKBD_HEIGHT : 0;
}

uint8_t ui_softkbd_touch_claimed() {
    return s_claimed;
}

uint8_t ui_softkbd_take_dirty() {
    uint8_t d = s_dirty;
    s_dirty = 0;
    return d;
}

// 命中判定：
//   - 行（y）：取最近行几何中心（与 main.cpp-ref 一致，键盘上沿之外保留容忍带）；
//   - 列（x）：显式判定边界表——内部边界取相邻键心中点，最左/最右边界按
//     UI_SOFTKBD_EDGE_BIAS_X 向外扩（补偿电容屏边缘效应，见宏定义处注释）；
//   - 最后按所判定键的几何中心做阈值判定（分轴归一化欧氏距离），保持外轮廓容忍带语义。
static int32_t ui_softkbd_hit_test(int32_t x, int32_t y, int32_t *out_row, int32_t *out_col) {
    int32_t kbd_y = SCREEN_HEIGHT - UI_SOFTKBD_HEIGHT;
    float key_w = (float)SCREEN_WIDTH / UI_SOFTKBD_COLS;
    float key_h = (float)UI_SOFTKBD_HEIGHT / UI_SOFTKBD_ROWS;

    // 行判定：最近行几何中心
    int32_t best_r = -1;
    float best_dy2 = 1e30f;
    for (int32_t r = 0; r < UI_SOFTKBD_ROWS; r++) {
        float cy = kbd_y + (r + 0.5f) * key_h;
        float dy = (y - cy) / key_h;
        float dy2 = dy * dy;
        if (dy2 < best_dy2) {
            best_dy2 = dy2;
            best_r = r;
        }
    }
    if (best_r < 0) return 0;

    // 最底行垂直边缘补偿：最底行与上一行的判定边界上移 BIAS_Y（默认0，不补偿）
    if (UI_SOFTKBD_EDGE_BIAS_Y > 0 && best_r == UI_SOFTKBD_ROWS - 2) {
        float row_boundary = kbd_y + (UI_SOFTKBD_ROWS - 1) * key_h - UI_SOFTKBD_EDGE_BIAS_Y;
        if (y >= row_boundary) best_r = UI_SOFTKBD_ROWS - 1;
    }

    // 列判定：显式判定边界表。第 c 条边界为 col c 与 col c+1 的分界，
    // 取相邻键心中点 (c+1)*key_w；最左边界右移 BIAS_X、最右边界左移 BIAS_X。
    int32_t col = 0;
    for (int32_t c = 0; c < UI_SOFTKBD_COLS - 1; c++) {
        float boundary = (c + 1) * key_w;
        if (c == 0)                    boundary += UI_SOFTKBD_EDGE_BIAS_X; // 最左边界右移
        if (c == UI_SOFTKBD_COLS - 2)  boundary -= UI_SOFTKBD_EDGE_BIAS_X; // 最右边界左移
        if ((float)x < boundary) break;
        col = c + 1;
    }

    // 阈值判定：触点到所判定键几何中心的分轴归一化欧氏距离不超过阈值（比较平方，避免开方）
    float cx = (col + 0.5f) * key_w;
    float cy = kbd_y + (best_r + 0.5f) * key_h;
    float dx = (x - cx) / key_w;
    float dy = (y - cy) / key_h;
    float d = dx * dx + dy * dy;
    if (d > UI_SOFTKBD_HIT_THRESHOLD * UI_SOFTKBD_HIT_THRESHOLD) return 0;

    *out_row = best_r;
    *out_col = col;
    return 1;
}

uint8_t ui_softkbd_poll() {
    int32_t x = 0, y = 0, is_pressed = 0;
    touch_read(&x, &y, &is_pressed);

    s_claimed = 0;

    // 松开：清除按下高亮与锁存键码，准备下一次按下沿
    if (!is_pressed) {
        s_prev_pressed = 0;
        s_held_code = NANO_KEY_IDLE;
        s_held_no_repeat = 0;
        if (s_press_row >= 0) {
            s_press_row = -1;
            s_press_col = -1;
            s_dirty = 1;
        }
        return NANO_KEY_IDLE;
    }

    // 键盘区域：键盘上沿之外再预留一段容忍带（同 main.cpp-ref），带内触点交给命中判定
    int32_t kbd_y = SCREEN_HEIGHT - UI_SOFTKBD_HEIGHT;
    int32_t band = (int32_t)(UI_SOFTKBD_HIT_THRESHOLD * UI_SOFTKBD_HEIGHT / UI_SOFTKBD_ROWS);
    if (y < kbd_y - band) {
        s_prev_pressed = 0; // 键盘区域外：不接管，允许滑入键盘时重新触发按下沿
        s_held_code = NANO_KEY_IDLE; // 滑出键盘：取消按住（停止重复触发）
        s_held_no_repeat = 0;
        return NANO_KEY_IDLE;
    }
    s_claimed = 1;

    // 按住中：可重复键持续上报锁存键码（交给框架原生的长按/连发机制，与4x4网格键一致）；
    // 不可重复键（Ctrl/SFT/Alt/SYM/Esc）仅按下沿一次性上报，按住期间不产生键码
    if (s_prev_pressed) {
        return (s_held_no_repeat) ? NANO_KEY_IDLE : s_held_code;
    }
    s_prev_pressed = 1;

    int32_t row = -1, col = -1;
    if (!ui_softkbd_hit_test(x, y, &row, &col)) {
        return NANO_KEY_IDLE; // 容忍带内但未命中任何键：吞掉即可
    }

    s_press_row = (int8_t)row;
    s_press_col = (int8_t)col;
    s_dirty = 1;

    const UI_Softkbd_Key *k = &S_SOFTKBD[row][col];

    // SFT：按下沿切换大写粘滞态，同时产生一次 NANO_KEY_shift 传递给UI框架（两者耦合：
    // 框架对软键盘来源的Shift不切换输入模式（软键盘模式下用 Ctrl+空格 切换），仅在Ctrl激活时显示帮助；
    // 软键盘粘滞态决定下一键是否取Shift层）。不可重复。
    if (k->mod == UI_SOFTKBD_MOD_SHIFT) {
        s_sticky = (s_sticky == k->mod) ? UI_SOFTKBD_MOD_NONE : k->mod;
        s_held_code = NANO_KEY_shift;
        s_held_no_repeat = 1;
        return s_held_code;
    }

    // SYM：点按切换符号层粘滞状态，不产生键码。不可重复。
    if (k->mod == UI_SOFTKBD_MOD_SYM) {
        s_sticky = (s_sticky == k->mod) ? UI_SOFTKBD_MOD_NONE : k->mod;
        s_held_code = NANO_KEY_IDLE;
        s_held_no_repeat = 1;
        return NANO_KEY_IDLE;
    }

    // 普通键：按当前粘滞状态决定键码，并锁存（按住期间重复上报本键码）
    uint8_t code = k->base;
    if      (s_sticky == UI_SOFTKBD_MOD_SHIFT) code = k->shift;
    else if (s_sticky == UI_SOFTKBD_MOD_SYM)   code = k->sym;
    s_sticky = UI_SOFTKBD_MOD_NONE; // 粘滞状态在决定键码后释放
    s_dirty = 1;

    s_held_code = code;
    // Ctrl/Alt/Esc 按住不重复（仅按下沿一次性），其余键均可重复
    s_held_no_repeat = (code == NANO_KEY_ctrl || code == NANO_KEY_alt || code == NANO_KEY_esc) ? 1 : 0;

    return code;
}

void ui_softkbd_draw(Nano_GFX *gfx, uint8_t is_ctrl_active) {
    if (!s_visible) return;

    int32_t kbd_y = SCREEN_HEIGHT - UI_SOFTKBD_HEIGHT;
    int32_t key_h = UI_SOFTKBD_HEIGHT / UI_SOFTKBD_ROWS;

    // 先以边界色填充整个键盘区域，各键向内缩1px绘制，形成1px边界（同 main.cpp-ref）
    gfx_draw_rectangle(gfx, 0, kbd_y, SCREEN_WIDTH, UI_SOFTKBD_HEIGHT, UI_SOFTKBD_COLOR_BORDER, 1);

    for (int32_t r = 0; r < UI_SOFTKBD_ROWS; r++) {
        for (int32_t c = 0; c < UI_SOFTKBD_COLS; c++) {
            const UI_Softkbd_Key *k = &S_SOFTKBD[r][c];
            int32_t x0 = c * SCREEN_WIDTH / UI_SOFTKBD_COLS;
            int32_t x1 = (c + 1) * SCREEN_WIDTH / UI_SOFTKBD_COLS;
            int32_t y0 = kbd_y + r * key_h;

            uint8_t bg_R = 0, bg_G = 0, bg_B = 0;
            if ((k->mod != UI_SOFTKBD_MOD_NONE && s_sticky == k->mod) ||
                (k->base == NANO_KEY_ctrl && is_ctrl_active)) {
                bg_R = 16; bg_G = 48; bg_B = 160;  // 粘滞修饰键/Ctrl全局激活：高亮
            }
            else {
                bg_R = 28; bg_G = 28; bg_B = 32;   // 普通键底色
            }
            if (r == s_press_row && c == s_press_col) {
                bg_R = 96; bg_G = 96; bg_B = 104;  // 按下瞬间高亮
            }

            // 右侧与下侧各留1px给边界色
            gfx_draw_rectangle(gfx, x0, y0, x1 - x0 - 1, key_h - 1, bg_R, bg_G, bg_B, 1);

            gfx_font_draw_text_centered(gfx, GFX_FONT_BITMAP_12, k->label,
                (x0 + x1) / 2, y0 + key_h / 2, UI_SOFTKBD_COLOR_LABEL, 1);
        }
    }
}
