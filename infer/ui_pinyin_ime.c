#include <stdio.h>

#include "ui_pinyin_ime.h"
#include "ui_softkbd.h"
#include "ui_color.h"

// 拼音-汉字查表（pinyin_ime.c 为自动生成的表，内容不可手改；作为独立编译单元参与链接）
#include "pinyin_ime.h"

// ===============================================================================
// 全键盘拼音输入法（移植自 main.cpp-ref 的 PinyinIME）
// ===============================================================================

#define UI_IME_CANDIDATES_PER_PAGE (5)    // 每页候选个数（数字键1~5选取）
#define UI_IME_MAX_PINYIN_LEN      (6)    // 单音节拼音最大长度（如 zhuang）
#define UI_IME_MAX_CANDIDATES      (4096) // 候选字缓冲区容量（同 main.cpp-ref）

// 符号候选列表（与 ui.c 九键输入法的符号表一致）
static wchar_t S_IME_SYMBOLS[55] = L"，。、？！：；“”‘’（）《》…—～·【】 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

static uint8_t  s_ime_state = UI_IME_STATE_IDLE;
static char     s_pinyin[UI_IME_MAX_PINYIN_LEN + 2] = {0};
static uint32_t s_pinyin_len = 0;
// 候选字缓冲区：首次使用时分配于PSRAM（避免16KB常驻内部DRAM）
static uint32_t *s_candidates = NULL;
static uint32_t s_candidate_num = 0;
static uint32_t s_page = 0;

// 确保候选字缓冲区已分配（懒分配，只在拼音输入法实际使用时才占用内存）
static void ui_ime_ensure_candidates_buffer() {
    if (!s_candidates) {
        s_candidates = (uint32_t *)platform_calloc(UI_IME_MAX_CANDIDATES, sizeof(uint32_t));
    }
}

void ui_pinyin_ime_reset() {
    s_ime_state = UI_IME_STATE_IDLE;
    s_pinyin[0] = 0;
    s_pinyin_len = 0;
    s_candidate_num = 0;
    s_page = 0;
}

uint8_t ui_pinyin_ime_is_composing() {
    return (s_ime_state == UI_IME_STATE_SELECTING || s_ime_state == UI_IME_STATE_SYMBOL) ? 1 : 0;
}

// 半角符号转全角（同 main.cpp-ref 的 half_to_full）
static const wchar_t *ui_ime_half_to_full(uint8_t c) {
    switch (c) {
        case '!':  return L"！";
        case '^':  return L"……";
        case '(':  return L"（";
        case ')':  return L"）";
        case '[':  return L"【";
        case ']':  return L"】";
        case '\\': return L"、";
        case ';':  return L"；";
        case ':':  return L"：";
        case ',':  return L"，";
        case '.':  return L"。";
        default:   return NULL;
    }
}

static uint32_t ui_ime_total_candidates() {
    if (s_ime_state == UI_IME_STATE_SYMBOL) return 54;
    return s_candidate_num;
}

static uint32_t ui_ime_candidate_at(uint32_t idx) {
    if (s_ime_state == UI_IME_STATE_SYMBOL) return (uint32_t)S_IME_SYMBOLS[idx];
    return s_candidates[idx];
}

static void ui_ime_refresh_candidates() {
    ui_ime_ensure_candidates_buffer();
    if (!s_candidates) { // 分配失败：清空候选，避免空指针
        s_candidate_num = 0;
        s_page = 0;
        return;
    }
    s_candidate_num = (uint32_t)pinyin_to_hanzi(s_pinyin, s_candidates);
    if (s_candidate_num > UI_IME_MAX_CANDIDATES) s_candidate_num = UI_IME_MAX_CANDIDATES;
    s_page = 0;
}

uint8_t ui_pinyin_ime_handle_key(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    uint8_t c = key_event->key_code;

    // Ctrl+BS：进入/退出符号选择状态（对应 main.cpp-ref 的 Fn+Bksp）；其余Ctrl组合键不接管
    if (global_state->is_ctrl_enabled == 1) {
        if (c == NANO_KEY_backspace) {
            global_state->is_ctrl_enabled = 0;
            if (s_ime_state == UI_IME_STATE_SYMBOL) {
                ui_pinyin_ime_reset();
                ui_draw_input_buffer(key_event, global_state, input_state); // 恢复默认页脚
            }
            else {
                s_ime_state = UI_IME_STATE_SYMBOL;
                s_page = 0;
                ui_pinyin_ime_draw_bar(global_state);
                gfx_refresh(global_state->gfx);
            }
            return 1;
        }
        return 0;
    }

    // 选字/选符号状态
    if (s_ime_state == UI_IME_STATE_SELECTING || s_ime_state == UI_IME_STATE_SYMBOL) {

        // 数字键：选字/选符号（1~7）；选字状态下其余数字键忽略，避免误输入
        if (c >= NANO_KEY_0 && c <= NANO_KEY_9) {
            int32_t idx = (c == NANO_KEY_0) ? 9 : (c - NANO_KEY_1); // 按键0对应第10个
            uint32_t base = s_page * UI_IME_CANDIDATES_PER_PAGE;
            uint32_t total = ui_ime_total_candidates();
            if (idx >= 0 && idx < UI_IME_CANDIDATES_PER_PAGE && base + (uint32_t)idx < total) {
                insert_char(input_state, (wchar_t)ui_ime_candidate_at(base + (uint32_t)idx));
                ui_pinyin_ime_reset();
                ui_draw_input_buffer(key_event, global_state, input_state);
            }
            return 1;
        }

        // 左右方向键：翻页
        if (c == NANO_KEY_left) {
            if (s_page > 0) s_page--;
            ui_pinyin_ime_draw_bar(global_state);
            gfx_refresh(global_state->gfx);
            return 1;
        }
        if (c == NANO_KEY_right) {
            uint32_t total = ui_ime_total_candidates();
            uint32_t total_pages = (total + UI_IME_CANDIDATES_PER_PAGE - 1) / UI_IME_CANDIDATES_PER_PAGE;
            if (s_page + 1 < total_pages) s_page++;
            ui_pinyin_ime_draw_bar(global_state);
            gfx_refresh(global_state->gfx);
            return 1;
        }

        // 退格键：删除一个拼音字母；删空则回到空闲状态（仅SELECTING，同 main.cpp-ref）
        if (c == NANO_KEY_backspace && s_ime_state == UI_IME_STATE_SELECTING) {
            if (s_pinyin_len > 0) s_pinyin[--s_pinyin_len] = 0;
            if (s_pinyin_len == 0) {
                ui_pinyin_ime_reset();
                ui_draw_input_buffer(key_event, global_state, input_state); // 恢复默认页脚
            }
            else {
                ui_ime_refresh_candidates();
                ui_pinyin_ime_draw_bar(global_state);
                gfx_refresh(global_state->gfx);
            }
            return 1;
        }

        // Esc：取消拼音/符号输入
        if (c == NANO_KEY_esc) {
            ui_pinyin_ime_reset();
            ui_draw_input_buffer(key_event, global_state, input_state);
            return 1;
        }

        // Enter：把已输入的拼音字母原样上屏（再次 Enter 才提交）
        if (c == NANO_KEY_enter && s_ime_state == UI_IME_STATE_SELECTING) {
            for (uint32_t i = 0; i < s_pinyin_len; i++) {
                insert_char(input_state, (wchar_t)s_pinyin[i]);
            }
            ui_pinyin_ime_reset();
            ui_draw_input_buffer(key_event, global_state, input_state);
            return 1;
        }

        // 字母键：继续输入拼音（大写同样按拼音处理）
        if ((c >= NANO_KEY_a && c <= NANO_KEY_z) || (c >= NANO_KEY_A && c <= NANO_KEY_Z)) {
            char letter = (c >= NANO_KEY_A && c <= NANO_KEY_Z) ? (char)(c - NANO_KEY_A + 'a') : (char)c;
            if (s_ime_state == UI_IME_STATE_SELECTING && s_pinyin_len < UI_IME_MAX_PINYIN_LEN) {
                s_pinyin[s_pinyin_len++] = letter;
                s_pinyin[s_pinyin_len] = 0;
                ui_ime_refresh_candidates();
                ui_pinyin_ime_draw_bar(global_state);
                gfx_refresh(global_state->gfx);
            }
            return 1;
        }

        // 半角符号转全角（直接上屏，不打断拼音组字，同 main.cpp-ref）
        if (c >= NANO_KEY_space && c <= NANO_KEY_tilde) {
            const wchar_t *full = ui_ime_half_to_full(c);
            if (full) {
                for (uint32_t i = 0; i < wcslen(full); i++) insert_char(input_state, full[i]);
            }
            else {
                insert_char(input_state, (wchar_t)c);
            }
            ui_draw_input_buffer(key_event, global_state, input_state);
            return 1;
        }

        return 0;
    }

    // 空闲状态
    // 字母键：开始输入拼音
    if ((c >= NANO_KEY_a && c <= NANO_KEY_z) || (c >= NANO_KEY_A && c <= NANO_KEY_Z)) {
        char letter = (c >= NANO_KEY_A && c <= NANO_KEY_Z) ? (char)(c - NANO_KEY_A + 'a') : (char)c;
        s_pinyin[0] = letter;
        s_pinyin[1] = 0;
        s_pinyin_len = 1;
        s_ime_state = UI_IME_STATE_SELECTING;
        ui_ime_refresh_candidates();
        ui_pinyin_ime_draw_bar(global_state);
        gfx_refresh(global_state->gfx);
        return 1;
    }

    // 半角符号转全角
    if (c >= NANO_KEY_space && c <= NANO_KEY_tilde && ui_ime_half_to_full(c)) {
        const wchar_t *full = ui_ime_half_to_full(c);
        for (uint32_t i = 0; i < wcslen(full); i++) insert_char(input_state, full[i]);
        ui_draw_input_buffer(key_event, global_state, input_state);
        return 1;
    }

    // 其余键不接管：数字直插、Enter提交、Esc删除/返回、方向键移动光标、BS删除
    return 0;
}

void ui_pinyin_ime_draw_bar(Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    int32_t bar_height = line_height + 1; // 与页脚同高
    int32_t bar_y = (int32_t)gfx->height - ui_softkbd_height() - bar_height;

    uint8_t bg_R, bg_G, bg_B, fg_R, fg_G, fg_B;
    if (global_state->ui_color_style == UI_COLOR_DARK) {
        bg_R = 15;  bg_G = 16;  bg_B = 17;
        fg_R = 188; fg_G = 188; fg_B = 188;
    }
    else {
        bg_R = 224; bg_G = 230; bg_B = 234;
        fg_R = 90;  fg_G = 98;  fg_B = 106;
    }

    // 以页脚底色填充底栏区域
    gfx_draw_rectangle(gfx, 0, bar_y, gfx->width, bar_height, bg_R, bg_G, bg_B, 1);

    int32_t y_top = bar_y;
    const int32_t x0 = 2; // 底栏内容左缘

    // 字号基准：以全角字符“一”的实际渲染宽度确定1个全角宽度，半角宽度为其一半
    //   （如12px字体：全角12、半角固定6，与具体字符的实际渲染宽度无关）；
    //   排版一律按像素定位，不使用空格做朴素对齐。
    int32_t full_width = gfx_font_char_advance(font_id, (uint32_t)L'一');
    int32_t half_width = full_width / 2;

    // 布局（各段宽度固定，与拼音长度、有无翻页符号无关）：
    //   [拼音区 7个半角] [左翻页符号区 3个半角] [候选列表，起点固定为左数第11个半角宽度位置] [> 右对齐]

    // 拼音串（符号选择状态下不显示）：左对齐绘制在固定7个半角宽度的拼音区内
    if (s_ime_state == UI_IME_STATE_SELECTING && s_pinyin_len > 0) {
        wchar_t text[8];
        for (uint32_t i = 0; i < s_pinyin_len; i++) text[i] = (wchar_t)s_pinyin[i];
        text[s_pinyin_len] = 0;
        gfx_font_draw_text(gfx, font_id, text, x0, y_top, fg_R, fg_G, fg_B, 1);
    }

    // 左翻页符号区：固定预留3个半角宽度（无翻页符号时保留空位）
    if (s_page > 0) {
        gfx_font_draw_text(gfx, font_id, L"<", x0 + (7 + 1) * half_width, y_top, fg_R, fg_G, fg_B, 1);
    }

    // 候选字/符号列表：起始位置固定（x0 + 10个半角宽度，即左数第11个半角宽度位置）；
    // 编号与候选字紧贴（无间隔点，如“1的”），候选之间间隔2个半角字符宽；
    // 编号1~7逐页显示（超宽截断，为下一页指示预留空间）
    int32_t x = x0 + 10 * half_width;
    uint32_t total = ui_ime_total_candidates();
    uint32_t base = s_page * UI_IME_CANDIDATES_PER_PAGE;
    for (uint32_t i = 0; i < UI_IME_CANDIDATES_PER_PAGE && base + i < total; i++) {
        uint32_t index_ch = (uint32_t)(L'1' + i);
        uint32_t cand_ch = ui_ime_candidate_at(base + i);
        int32_t index_w = gfx_font_char_advance(font_id, index_ch);
        int32_t cand_w = gfx_font_char_advance(font_id, cand_ch);
        if (x + index_w + cand_w >= (int32_t)gfx->width - 12) break;
        gfx_font_draw_char(gfx, font_id, index_ch, x, y_top, fg_R, fg_G, fg_B, 1);
        x += index_w;
        gfx_font_draw_char(gfx, font_id, cand_ch, x, y_top, fg_R, fg_G, fg_B, 1);
        x += cand_w + 2 * half_width;
    }

    // 下一页指示（右对齐）
    if (base + UI_IME_CANDIDATES_PER_PAGE < total) {
        gfx_font_draw_text(gfx, font_id, L">", gfx->width - 9, y_top, fg_R, fg_G, fg_B, 1);
    }
}
