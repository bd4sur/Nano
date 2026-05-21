#include <stdio.h>
#include <time.h>

#include "graphics.h"
#include "input_device.h"
#include "ui.h"

#include "platform.h"

#include "ui_color.h"
#include "ui_pinyin_lut.h"

// 全局色彩变量（用于调节UI配色风格）

static uint8_t S_UI_COLOR_HEADER_TEXT[3]   = {255, 255, 255};
static uint8_t S_UI_COLOR_FOOTER_BG[3]     = {224, 230, 234};
static uint8_t S_UI_COLOR_FOOTER_TEXT[3]   = {90 , 98 , 106};

static uint8_t S_UI_COLOR_IME_HELP_BG[3]   = {222, 222, 222};
static uint8_t S_UI_COLOR_IME_HELP_TEXT[3] = {0  , 0  , 0  };


// 符号列表
static wchar_t ime_symbols[55] = L"，。、？！：；“”‘’（）《》…—～·【】 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
// 按键对应的字母列表
static wchar_t ime_alphabet[10][32] = {L"0", L" 1.,:?!-/+_=&\"*", L"abcABC2", L"defDEF3", L"ghiGHI4", L"jklJKL5", L"mnoMNO6", L"pqrsPRQS7", L"tuvTUV8", L"wxyzWXYZ9"};

// 带四舍五入的整数除法，仅接受正数
static inline uint32_t div_round(uint32_t a, uint32_t b) {
    return (a + b / 2) / b;
}

void get_candidate_hanzi_list(Widget_Input_State *input_state) {
    unsigned int candidate_index[500];
    int candidate_count = 0;
    for(int i = 0; i < IME_HANZI_NUM; i++) {
        if(KEYS_LIST[i] == input_state->pinyin_keys) {
            candidate_index[candidate_count++] = i;
        }
    }

    memset(input_state->candidates, 0, sizeof(input_state->candidates));

    if(candidate_count == 0) {
        input_state->candidate_num = 0;
    }
    else {
        for(int i = 0; i < candidate_count; i++) {
            input_state->candidates[i] = UTF32_LIST[candidate_index[i]];
        }
        input_state->candidate_num = candidate_count;
    }
}


void candidate_paging(Widget_Input_State *input_state) {
    input_state->candidate_page_num = input_state->candidate_num / MAX_CANDIDATE_NUM_PER_PAGE + ((input_state->candidate_num % MAX_CANDIDATE_NUM_PER_PAGE) ? 1 : 0);
    memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));
    uint32_t pos = 0;
    for (uint32_t i = 0; i < input_state->candidate_page_num; i++) {
        for (uint32_t j = 0; j < MAX_CANDIDATE_NUM_PER_PAGE; j++) {
            input_state->candidate_pages[i][j] = (pos < input_state->candidate_num) ? input_state->candidates[pos] : 0; // 选字时，选到0就意味着越界了
            pos++;
        }
    }
}

// 在文本框的光标位置之后插入一个字符
void insert_char(Widget_Input_State *input_state, wchar_t new_char) {
    if (input_state->textarea.length + 1 > UI_STR_BUF_MAX_LENGTH) {
        return;
    }

    input_state->textarea.text[input_state->textarea.length + 1] = L'\0';

    for (uint32_t i = input_state->textarea.length; i >= input_state->cursor_pos + 2; i--) {
        input_state->textarea.text[i] = input_state->textarea.text[i-1];
    }
    input_state->textarea.text[input_state->cursor_pos + 1] = new_char;

    input_state->cursor_pos++;
    input_state->textarea.length++;
}

// 删除光标位置的字符（即光标竖线左边的一个字符）
void delete_char(Widget_Input_State *input_state) {
    if (input_state->textarea.length <= 0 || input_state->cursor_pos < 0) {
        return;
    }

    for (uint32_t i = input_state->cursor_pos; i < input_state->textarea.length; i++) {
        input_state->textarea.text[i] = input_state->textarea.text[i+1];
    }
    input_state->textarea.text[input_state->textarea.length - 1] = L'\0';

    input_state->cursor_pos--;
    input_state->textarea.length--;
}




static int hex_char_to_int(uint32_t c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/**
 * 尝试从指定位置解析颜色标签 [#RRGGBB]
 * 
 * @param text 文本数组
 * @param pos 当前扫描位置
 * @param max_pos 最大扫描位置（包含）
 * @param r 返回红色分量
 * @param g 返回绿色分量  
 * @param b 返回蓝色分量
 * @return 成功返回消耗的字符数（9），失败返回0
 */
static int parse_color_tag(wchar_t *text, int pos, int max_pos, uint32_t *style_code) {
    // 检查是否有足够的字符: '[' '#' R R G G B B ']' 共9个字符
    if (pos + 8 > max_pos) return 0;
    
    // 状态检查：必须是 [#RRGGBB] 格式
    if ((uint32_t)text[pos] != (uint32_t)'[') return 0;
    if ((uint32_t)text[pos + 1] != (uint32_t)'#') return 0;
    if ((uint32_t)text[pos + 8] != (uint32_t)']') return 0;
    
    // 验证并解析6位十六进制颜色值
    int color_val = 0;
    for (int i = 2; i <= 7; i++) {
        int hex_val = hex_char_to_int(text[pos + i]);
        if (hex_val < 0) return 0;  // 发现非十六进制字符，解析失败
        color_val = (color_val << 4) | hex_val;
    }

    *style_code = color_val & 0x00ffffff;
    
    // 提取RGB分量 (RRGGBB -> R, G, B)
    // *r = (color_val >> 16) & 0xFF;
    // *g = (color_val >> 8) & 0xFF;
    // *b = color_val & 0xFF;
    
    return 9;  // 成功消耗9个字符
}


// 排版-折行（高代价）：计算全部文本的length(char_count)、line_num(break_count)、break_pos
//    同时解析文本中的样式控制标签
void typeset_line_breaks(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state) {
    int32_t break_count = 0;
    int32_t line_x_pos = 0;
    int32_t char_count = 0;
    int32_t text_len = wcslen(textarea_state->text);
    uint32_t style_code = 0x00000000;

    // 默认样式与全局的色彩风格有关
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        style_code = 0x00000000;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        style_code = 0x00ffffff;
    }

    // 首字符强制折行
    textarea_state->break_pos[break_count] = 0;  // 记录断行位置
    break_count++;

    for (int32_t i = 0; i < text_len; i++) {
        // 调用 parse_color_tag 检测颜色标签
        int consumed = parse_color_tag(textarea_state->text, char_count, text_len - 1, &style_code);
        textarea_state->style[char_count] = style_code;

        if (consumed > 0) {
            // 将格式标签的style的最高位置为1，代表渲染时忽略
            for (int32_t k = 0; k < consumed; k++) {
                textarea_state->style[i + k] = (textarea_state->style[i + k] | 0x80000000);
            }
            // 是颜色标签：计入总长度，但跳过排版计算（不占宽、不换行）
            i += (consumed - 1);  // 跳过整个标签（-1是因为for循环会执行i++）
            char_count += consumed;
            continue;  // 直接进入下一次循环，不执行下方的宽度计算
        }

        wchar_t ch = textarea_state->text[i];
        int32_t char_width = (ch < 127) ? ((ch == '\n') ? 0 : FONT_WIDTH_HALF) : FONT_WIDTH_FULL;

        // 折行判断（当前行已满）
        if (line_x_pos + char_width >= textarea_state->width) {
            textarea_state->break_pos[break_count] = i;  // 记录断行位置
            break_count++;
            line_x_pos = 0;
        }
        else if (ch == '\n') {
            textarea_state->break_pos[break_count] = i + 1;
            break_count++;
            line_x_pos = 0;
        }

        line_x_pos += char_width;
        char_count++;
    }

    textarea_state->line_num = (break_count <= 0) ? 1 : break_count;
    textarea_state->length = char_count;
}


// 排版-视口（低代价）：给定起始行号和视口宽高，计算视口内文本的index和最大能容纳的行数
void typeset_view_range(Widget_Textarea_State *textarea_state) {
    int32_t view_height = textarea_state->height;
    //   NOTE 考虑到行间距为1，且末行以下无间距，因此分子加1以去除末行无间距的影响。
    //        例如，高度为64的屏幕，实际可容纳(64+1)/(12+1)=5行。
    int32_t max_view_lines = (view_height + 1) / (FONT_HEIGHT + 1);
    int32_t _line_num = textarea_state->line_num;

    textarea_state->view_lines = max_view_lines;

    int32_t start_line = textarea_state->current_line;

    // 对start_line的检查和标准化
    if (start_line < 0) {
        // start_line小于0，解释为将文字末行卷动到视图的某一行。例如：-1代表将文字末行卷动到视图的倒数1行、-max_view_lines代表将文字末行卷动到视图的第1行。
        //   若start_line小于-max_view_lines，则等效于-max_view_lines，保证文字内容不会卷到视图以外。
        if (-start_line <= max_view_lines) {
            if (_line_num >= max_view_lines) {
                start_line = _line_num - 1 - start_line - max_view_lines;
            }
            else {
                start_line = 0;
            }
        }
        else {
            start_line = _line_num - 1;
        }
    }
    else if (start_line >= _line_num) {
        // start_line超过了末行，则对文本行数取模后滚动
        start_line = start_line % _line_num;
    }

    // 情况1：start_line介于首行（0）和（使得末行进入可见区域以下1行的位置），即视图内不包含末行
    if (start_line < _line_num - max_view_lines) {
        textarea_state->view_start_pos = textarea_state->break_pos[start_line];
        textarea_state->view_end_pos = textarea_state->break_pos[start_line + max_view_lines] - 1;
    }
    // 情况2：start_line等于或超过了（使得末行恰好位于可见区域底行的位置），但尚未超出末行，也就是末行位于视图内
    //        若文本行数不大于视图行数，则一定满足此条件。
    else if (start_line >= _line_num - max_view_lines && start_line < _line_num) {
        textarea_state->view_start_pos = textarea_state->break_pos[start_line];
        textarea_state->view_end_pos = textarea_state->length - 1;
    }
}


void ui_draw_text_block(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state) {
    int x_pos = textarea_state->x;
    int y_pos = textarea_state->y;
    
    // 当前绘制颜色
    uint8_t current_r = 0;
    uint8_t current_g = 0;
    uint8_t current_b = 0;

    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        current_r = 0; current_g = 0; current_b = 0;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        current_r = 255; current_g = 255; current_b = 255;
    }

    for (int i = textarea_state->view_start_pos; i <= textarea_state->view_end_pos; i++) {
        // 首先检查这一位是不是格式控制标签的字符
        if (textarea_state->style[i] & 0x80000000) {
            continue;
        }
        uint32_t current_char = textarea_state->text[i];
        if (!current_char) break;
        uint8_t font_width = FONT_WIDTH_FULL;
        uint8_t font_height = FONT_HEIGHT;
        if (current_char == '\n') {
            x_pos = textarea_state->x;
            if(i > 0) y_pos += (font_height + 1);
            continue;
        }
        const uint8_t *glyph = gfx_get_glyph(global_state->gfx, current_char, &font_width, &font_height);
        if (!glyph) {
            // printf("出现了字库之外的字符[%d]\n", current_char);
            glyph = gfx_get_glyph(global_state->gfx, 12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos + font_width >= textarea_state->x + textarea_state->width) {
            y_pos += (font_height + 1);
            x_pos = textarea_state->x;
        }
        // 使用当前颜色绘制字符
        uint32_t style_code = textarea_state->style[i];
        current_r = (style_code >> 16) & 0xFF;
        current_g = (style_code >> 8) & 0xFF;
        current_b = style_code & 0xFF;
        gfx_draw_char(global_state->gfx, x_pos, y_pos, glyph, font_width, font_height, current_r, current_g, current_b, 1);
        x_pos += font_width;
    }
}

// 绘制滚动条
//   line_num - 文本总行数
//   current_line - 当前在屏幕顶端的是哪一行
//   view_lines - 屏幕最多容纳几行
void ui_draw_scroll_bar(Key_Event *key_event, Global_State *global_state, int32_t current_line, int32_t line_num, int32_t view_lines, int32_t x, int32_t y, int32_t width, int32_t height) {

    // 对current_line的检查和标准化
    if (current_line < 0) {
        // current_line小于0，解释为将文字末行卷动到视图的某一行。例如：-1代表将文字末行卷动到视图的倒数1行、-max_view_lines代表将文字末行卷动到视图的第1行。
        //   若current_line小于-max_view_lines，则等效于-max_view_lines，保证文字内容不会卷到视图以外。
        if (-current_line <= view_lines) {
            if (line_num >= view_lines) {
                current_line = line_num - 1 - current_line - view_lines;
            }
            else {
                current_line = 0;
            }
        }
        else {
            current_line = line_num - 1;
        }
    }
    else if (current_line >= line_num) {
        // current_line超过了末行，则对文本行数取模后滚动
        current_line = (line_num <= 0) ? 0 : current_line % line_num;
    }

    uint8_t scroll_bar_bg_R = 0, scroll_bar_bg_G = 0, scroll_bar_bg_B = 0;
    uint8_t scroll_bar_fg_R = 0, scroll_bar_fg_G = 0, scroll_bar_fg_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        scroll_bar_bg_R = 200; scroll_bar_bg_G = 200; scroll_bar_bg_B = 200;
        scroll_bar_fg_R = 33; scroll_bar_fg_G = 33; scroll_bar_fg_B = 33;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        scroll_bar_bg_R = 128; scroll_bar_bg_G = 128; scroll_bar_bg_B = 128;
        scroll_bar_fg_R = 255; scroll_bar_fg_G = 255; scroll_bar_fg_B = 255;
    }

    for (int n = y; n < y + height; n++) {
        gfx_draw_point(global_state->gfx, x + width - 1, n, scroll_bar_bg_R, scroll_bar_bg_G, scroll_bar_bg_B, 1);
    }

    line_num = (line_num <= 0) ? 1 : line_num;

    // 如果总行数装不满视图，则滚动条长度等于视图高度height
    int32_t bar_height = (line_num < view_lines) ? (height) : div_round((view_lines * height), line_num);
    bar_height = (bar_height < 3) ? 3 : bar_height; // 滚动条高度不小于3px

    // 滚动条顶部y坐标
    int32_t y_0 = y + div_round(current_line * height, line_num);
    y_0 = (y_0 >= y + height - 3 - 1) ? (y + height - 3 - 1) : y_0; // 滚动条顶部限位（不低于底部上方3px）

    gfx_draw_line(global_state->gfx, x + width - 1, y_0, x + width - 1, (y_0 + bar_height), scroll_bar_fg_R, scroll_bar_fg_G, scroll_bar_fg_B, 1);
    gfx_draw_line(global_state->gfx, x + width - 2, y_0, x + width - 2, (y_0 + bar_height), scroll_bar_fg_R, scroll_bar_fg_G, scroll_bar_fg_B, 1);
}









void ui_draw_header(Key_Event *key_event, Global_State *global_state, wchar_t *text, int32_t is_center) {
    const int header_height = 14;
    const uint8_t header_bgcolor[42] = {17,85,238,33,101,239,44,114,241,53,126,242,60,137,243,66,146,245,72,155,246,77,163,247,82,171,249,86,178,250,91,185,251,95,192,252,98,198,254,102,204,255};
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        for (int i = 0; i < header_height; i++) {
            gfx_draw_line(global_state->gfx, 0, i, global_state->gfx->width - 1, i, header_bgcolor[i*3+0], header_bgcolor[i*3+1], header_bgcolor[i*3+2], 1);
        }
        S_UI_COLOR_HEADER_TEXT[0] = 255;
        S_UI_COLOR_HEADER_TEXT[1] = 255;
        S_UI_COLOR_HEADER_TEXT[2] = 255;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_draw_rectangle(global_state->gfx, 0, 0, global_state->gfx->width, header_height, 15, 16, 17, 1);
        S_UI_COLOR_HEADER_TEXT[0] = 255;
        S_UI_COLOR_HEADER_TEXT[1] = 255;
        S_UI_COLOR_HEADER_TEXT[2] = 255;
    }
    if (is_center) {
        gfx_draw_textline_centered(global_state->gfx, text, global_state->gfx->width / 2, 7, S_UI_COLOR_HEADER_TEXT[0], S_UI_COLOR_HEADER_TEXT[1], S_UI_COLOR_HEADER_TEXT[2], 1);
    }
    else {
        gfx_draw_textline(global_state->gfx, text, 0, 1, S_UI_COLOR_HEADER_TEXT[0], S_UI_COLOR_HEADER_TEXT[1], S_UI_COLOR_HEADER_TEXT[2], 1);
    }
}

void ui_draw_footer(Key_Event *key_event, Global_State *global_state, wchar_t *text, int32_t is_center) {
    const int footer_height = 14;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_draw_rectangle(global_state->gfx, 0, global_state->gfx->height - footer_height, global_state->gfx->width, footer_height, S_UI_COLOR_FOOTER_BG[0], S_UI_COLOR_FOOTER_BG[1], S_UI_COLOR_FOOTER_BG[2], 1);
        S_UI_COLOR_FOOTER_TEXT[0] = 90;
        S_UI_COLOR_FOOTER_TEXT[1] = 98;
        S_UI_COLOR_FOOTER_TEXT[2] = 106;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_draw_rectangle(global_state->gfx, 0, global_state->gfx->height - footer_height, global_state->gfx->width, footer_height, 15, 16, 17, 1);
        S_UI_COLOR_FOOTER_TEXT[0] = 255;
        S_UI_COLOR_FOOTER_TEXT[1] = 255;
        S_UI_COLOR_FOOTER_TEXT[2] = 255;
    }
    if (is_center) {
        gfx_draw_textline_centered(global_state->gfx, text, global_state->gfx->width / 2, global_state->gfx->height - footer_height + 7, S_UI_COLOR_FOOTER_TEXT[0], S_UI_COLOR_FOOTER_TEXT[1], S_UI_COLOR_FOOTER_TEXT[2], 1);
    }
    else {
        gfx_draw_textline(global_state->gfx, text, 0, global_state->gfx->height - footer_height + 1, S_UI_COLOR_FOOTER_TEXT[0], S_UI_COLOR_FOOTER_TEXT[1], S_UI_COLOR_FOOTER_TEXT[2], 1);
    }
}








void ui_widget_textarea_init(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state,
    uint32_t max_len
) {
    textarea_state->state = 0;
    textarea_state->x = 0;
    textarea_state->y = 14;
    textarea_state->width = global_state->gfx->width;
    textarea_state->height = global_state->gfx->height - 14 - 14; // 减去header和footer
    textarea_state->length = 0;
    textarea_state->line_num = 0;
    textarea_state->view_lines = 0;
    textarea_state->view_start_pos = 0;
    textarea_state->view_end_pos = 0;
    textarea_state->current_line = 0;
    textarea_state->is_show_scroll_bar = 1;
    textarea_state->is_modified = 1;
    textarea_state->text = (wchar_t*)platform_calloc(max_len, sizeof(wchar_t));
    textarea_state->style = (uint32_t*)platform_calloc(max_len, sizeof(uint32_t));
    textarea_state->break_pos = (int32_t*)platform_calloc(max_len, sizeof(int32_t));
}

void ui_widget_textarea_set(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state,
    wchar_t *text, int32_t current_line, int32_t is_show_scroll_bar) {
    textarea_state->is_modified = 1;
    textarea_state->current_line = current_line;
    textarea_state->is_show_scroll_bar = is_show_scroll_bar;
    wcscpy(textarea_state->text, text);
}

void ui_widget_textarea_draw(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state) {
    if (textarea_state->is_modified) {
        typeset_line_breaks(key_event, global_state, textarea_state);
    }
    typeset_view_range(textarea_state);

    uint8_t textarea_bg_R = 0, textarea_bg_G = 0, textarea_bg_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        textarea_bg_R = 255; textarea_bg_G = 255; textarea_bg_B = 255;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        textarea_bg_R = 6; textarea_bg_G = 6; textarea_bg_B = 6;
    }

    if (global_state->is_full_refresh) {
        // gfx_soft_clear(global_state->gfx);
        gfx_draw_rectangle(global_state->gfx, textarea_state->x, textarea_state->y, textarea_state->width, textarea_state->height, textarea_bg_R, textarea_bg_G, textarea_bg_B, 1);
    }

    ui_draw_text_block(key_event, global_state, textarea_state);

    if (textarea_state->is_show_scroll_bar) {
        ui_draw_scroll_bar(
            key_event, global_state,
            textarea_state->current_line, textarea_state->line_num, textarea_state->view_lines,
            textarea_state->x, textarea_state->y, textarea_state->width, textarea_state->height);
    }

    if (global_state->is_full_refresh) {
        gfx_refresh(global_state->gfx);
    }
}

// 通用的文本框卷行事件处理
int32_t ui_widget_textarea_event_handler(
    Key_Event *ke, Global_State *gs, Widget_Textarea_State *ts,
    int32_t prev_focus_state, int32_t current_focus_state
) {
    // 短按A键：回到上一个焦点
    if (ke->key_edge == -1 && ke->key_code == KEYCODE_NUM_A) {
        return prev_focus_state;
    }

    // 长+短按*键：推理结果向上翻一行。如果翻到顶，则回到最后一行。
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == KEYCODE_NUM_STAR) {
        if (ts->current_line <= 0) { // 卷到顶
            ts->current_line = ts->line_num - ts->view_lines;
        }
        else {
            ts->current_line--;
        }

        ts->is_modified = 0;
        ui_widget_textarea_draw(ke, gs, ts);
        ts->is_modified = 1;

        return current_focus_state;
    }

    // 长+短按#键：推理结果向下翻一行。如果翻到底，则回到第一行。
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == KEYCODE_NUM_HASH) {
        if (ts->current_line >= (ts->line_num - ts->view_lines)) { // 卷到底
            ts->current_line = 0;
        }
        else {
            ts->current_line++;
        }

        ts->is_modified = 0;
        ui_widget_textarea_draw(ke, gs, ts);
        ts->is_modified = 1;

        return current_focus_state;
    }

    return current_focus_state;
}







void ui_widget_input_init(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    Widget_Textarea_State *ta = &(input_state->textarea);

    ui_widget_textarea_init(key_event, global_state, ta, UI_STR_BUF_MAX_LENGTH);

    ta->state = 0;
    ta->x = 0;
    ta->y = 14;
    ta->width = global_state->gfx->width;
    ta->height = global_state->gfx->height - 14 * 2; // 减去header和footer NOTE 详见结构体定义处的说明
    ta->length = 0;
    ta->is_show_scroll_bar = 1;

    input_state->cursor_pos = -1;
    input_state->ime_mode_flag = IME_MODE_HANZI;
    input_state->pinyin_keys = 0;
    input_state->candidate_num = 0;
    input_state->candidate_page_num = 0;
    input_state->current_page = 0;
    input_state->alphabet_click_timestamp = 0;
    input_state->alphabet_is_counting_down = 0;
    input_state->alphabet_current_key = 255;
    input_state->alphabet_index = 0;

    // 初始化各个数组
    memset(input_state->candidates, 0, sizeof(input_state->candidates));
    memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));

    ui_draw_input_buffer(key_event, global_state, input_state);
}

void ui_widget_input_refresh(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    input_state->cursor_pos = input_state->textarea.length - 1;
    ui_draw_input_buffer(key_event, global_state, input_state);
}

// 绘制文本输入操作说明
static void ui_draw_input_help(Key_Event *key_event, Global_State *global_state) {
    gfx_draw_rectangle(global_state->gfx, 3, 3, global_state->gfx->width - 6, global_state->gfx->height - 6, S_UI_COLOR_IME_HELP_BG[0], S_UI_COLOR_IME_HELP_BG[1], S_UI_COLOR_IME_HELP_BG[2], 3);
    gfx_draw_textline_centered(global_state->gfx, L"文本输入操作说明", global_state->gfx->width/2, 5+6, 0, 0, 222, 1);
    gfx_draw_textline_centered(global_state->gfx, L"A-退格/返回  B-切换汉英数",   global_state->gfx->width/2, 5+6+(12+1)*1, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    gfx_draw_textline_centered(global_state->gfx, L"C-第二功能  D-输入/提交",    global_state->gfx->width/2, 5+6+(12+1)*2, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    gfx_draw_textline_centered(global_state->gfx, L"按住0选择符号 左右键移动光标",  global_state->gfx->width/2, 5+6+(12+1)*3, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    gfx_draw_textline_centered(global_state->gfx, L"按住D语音输入 Ctrl+D 换行",    global_state->gfx->width/2, 5+6+(12+1)*4, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    gfx_draw_textline_centered(global_state->gfx, L"Ctrl+1 切换思考模式",          global_state->gfx->width/2, 5+6+(12+1)*5, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    gfx_draw_textline_centered(global_state->gfx, L"Ctrl+A 放弃输入并返回",        global_state->gfx->width/2, 5+6+(12+1)*6, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);

    gfx_refresh(global_state->gfx);
}

int32_t ui_widget_input_event_handler(
    Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state,
    int32_t prev_focus_state, int32_t current_focus_state, int32_t next_focus_state
) {

    uint8_t countdown_fg_R = 0, countdown_fg_G = 0, countdown_fg_B = 0;
    uint8_t countdown_bg_R = 0, countdown_bg_G = 0, countdown_bg_B = 0;
    uint8_t candidate0_bg_R = 0, candidate0_bg_G = 0, candidate0_bg_B = 0; // 未选中的候选字母
    uint8_t candidate0_fg_R = 0, candidate0_fg_G = 0, candidate0_fg_B = 0;
    uint8_t candidate1_bg_R = 0, candidate1_bg_G = 0, candidate1_bg_B = 0; // 选中的候选字母
    uint8_t candidate1_fg_R = 0, candidate1_fg_G = 0, candidate1_fg_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        countdown_fg_R = 0x11; countdown_fg_G = 0x55; countdown_fg_B = 0xee;
        countdown_bg_R = 0xff; countdown_bg_G = 0xff; countdown_bg_B = 0xff;
        candidate0_fg_R = 0x00; candidate0_fg_G = 0x00; candidate0_fg_B = 0x00;
        candidate0_bg_R = 0xee; candidate0_bg_G = 0xee; candidate0_bg_B = 0xee;
        candidate1_fg_R = 0xff; candidate1_fg_G = 0xff; candidate1_fg_B = 0xff;
        candidate1_bg_R = 0x00; candidate1_bg_G = 0x00; candidate1_bg_B = 0xff;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        countdown_fg_R = 0x66; countdown_fg_G = 0xcc; countdown_fg_B = 0xff;
        countdown_bg_R = 0x00; countdown_bg_G = 0x00; countdown_bg_B = 0x00;
        candidate0_fg_R = 0x00; candidate0_fg_G = 0x00; candidate0_fg_B = 0x00;
        candidate0_bg_R = 0xee; candidate0_bg_G = 0xee; candidate0_bg_B = 0xee;
        candidate1_fg_R = 0xff; candidate1_fg_G = 0xff; candidate1_fg_B = 0xff;
        candidate1_bg_R = 0x00; candidate1_bg_G = 0x00; candidate1_bg_B = 0xff;
    }

    int32_t state = input_state->state;

    int32_t ta_height = input_state->textarea.height;
    int32_t ta_y = input_state->textarea.y;

    // 定时器触发：字母输入的倒计时进度条
    if (input_state->ime_mode_flag == IME_MODE_ALPHABET && input_state->alphabet_is_counting_down == 1) {
        uint64_t ctimestamp = global_state->timestamp;
        // 倒计时进行中，绘制进度条
        if (ctimestamp - input_state->alphabet_click_timestamp <= ALPHABET_COUNTDOWN_MS) {
            uint32_t x_pos = (ALPHABET_COUNTDOWN_MS - ctimestamp + input_state->alphabet_click_timestamp) * global_state->gfx->width / ALPHABET_COUNTDOWN_MS;
            gfx_draw_line(global_state->gfx, 0, (ta_y + ta_height - 2), x_pos, (ta_y + ta_height - 2), countdown_fg_R, countdown_fg_G, countdown_fg_B, 1);
            gfx_draw_line(global_state->gfx, 0, (ta_y + ta_height - 1), x_pos, (ta_y + ta_height - 1), countdown_fg_R, countdown_fg_G, countdown_fg_B, 1);
            gfx_draw_line(global_state->gfx, x_pos + 1, (ta_y + ta_height - 2), (global_state->gfx->width - 1), (ta_y + ta_height - 2), countdown_bg_R, countdown_bg_G, countdown_bg_B, 1);
            gfx_draw_line(global_state->gfx, x_pos + 1, (ta_y + ta_height - 1), (global_state->gfx->width - 1), (ta_y + ta_height - 1), countdown_bg_R, countdown_bg_G, countdown_bg_B, 1);
            gfx_refresh(global_state->gfx);
            // gfx_draw_line(global_state->gfx, 0, (ta_y + ta_height - 2), (global_state->gfx->width - 1), (ta_y + ta_height - 2), 0, 0, 0, 1);
            // gfx_draw_line(global_state->gfx, 0, (ta_y + ta_height - 1), (global_state->gfx->width - 1), (ta_y + ta_height - 1), 0, 0, 0, 1);
            input_state->state = 0;
        }
        // 倒计时结束，提交当前选中的字母，清除进度条
        else {
            input_state->alphabet_is_counting_down = 0;

            // 清除进度条
            gfx_draw_line(global_state->gfx, 0, (ta_y + ta_height - 1), (global_state->gfx->width - 1), (ta_y + ta_height - 1), countdown_bg_R, countdown_bg_G, countdown_bg_B, 1);
            gfx_refresh(global_state->gfx);

            // 将当前选中的字母加入输入缓冲区
            uint32_t ch = ime_alphabet[(int)(input_state->alphabet_current_key)][input_state->alphabet_index];
            if (ch) {
                insert_char(input_state, ch);
            }
            else {
                printf("选定了列表之外的字母，忽略。\n");
            }

            ui_draw_input_buffer(key_event, global_state, input_state);

            input_state->alphabet_current_key = 255;
            input_state->alphabet_index = 0;
            input_state->state = 0;
        }
    }

    if (state == 0) {

        // 长按0：输入符号
        if (key_event->key_edge == -2 && key_event->key_code == 0) {
            memset(input_state->candidates, 0, sizeof(input_state->candidates));

            input_state->candidate_num = 54;
            for (int i = 0; i < input_state->candidate_num; i++) {
                input_state->candidates[i] = (uint32_t)ime_symbols[i];
            }

            candidate_paging(input_state);

            ui_draw_input_symbol(key_event, global_state, input_state);

            input_state->current_page = 0;
            input_state->state = 3;
        }

        // 短按0：数字输入模式下是直接输入0，其余模式无动作
        else if (key_event->key_edge == -1 && key_event->key_code == 0) {
            if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
                // input_state->text[(input_state->length)++] = L'0';
                // input_state->cursor_pos++;
                insert_char(input_state, L'0');
                ui_draw_input_buffer(key_event, global_state, input_state);
                input_state->state = 0;
            }
        }

        // 短按1-9：输入拼音/字母/数字，根据输入模式标志，转向不同的状态
        else if (key_event->key_edge == -1 && (key_event->key_code >= 1 && key_event->key_code <= 9)) {
            // Ctrl+1：切换思考模式/非思考模式
            if (global_state->is_ctrl_enabled == 1 && key_event->key_code == 1) {
                global_state->is_ctrl_enabled = 0;
                global_state->is_thinking_enabled = 1 - global_state->is_thinking_enabled;
                ui_draw_input_buffer(key_event, global_state, input_state);
            }

            else if (input_state->ime_mode_flag == IME_MODE_HANZI) {
                if (key_event->key_code >= 2 && key_event->key_code <= 9) { // 仅响应按键2-9；1无动作
                    input_state->state = 1;
                    ui_widget_input_event_handler(
                        key_event, global_state, input_state,
                        prev_focus_state, current_focus_state, next_focus_state);
                }
            }
            else if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
                // input_state->text[(input_state->length)++] = L'0' + key_event->key_code;
                // input_state->cursor_pos++;
                insert_char(input_state, (wchar_t)(L'0' + key_event->key_code));
                ui_draw_input_buffer(key_event, global_state, input_state);
                input_state->state = 0;
            }
            else if (input_state->ime_mode_flag == IME_MODE_ALPHABET) {
                // 如果按键按下时，不是字母切换状态，则开始循环切换，并开始倒计时。
                if (input_state->alphabet_is_counting_down == 0) {
                    input_state->alphabet_is_counting_down = 1;
                    input_state->alphabet_click_timestamp = global_state->timestamp;
                    input_state->alphabet_current_key = key_event->key_code;
                    input_state->alphabet_index = 0;
                }
                // 如果按键按下时，倒计时尚未结束，则切换到下一个字母。
                else {
                    input_state->alphabet_is_counting_down = 1;
                    input_state->alphabet_click_timestamp = global_state->timestamp;
                    input_state->alphabet_current_key = key_event->key_code;
                    input_state->alphabet_index = (input_state->alphabet_index + 1) % wcslen(ime_alphabet[(int)(key_event->key_code)]);
                }

                // 在屏幕上循环显示当前选中的字母
                wchar_t letter[2];
                uint32_t x_pos = 1;
                uint32_t y_pos = ta_y + ta_height - FONT_HEIGHT - 3;
                for (int i = 0; i < wcslen(ime_alphabet[(int)(key_event->key_code)]); i++) {
                    letter[0] = ime_alphabet[(int)(key_event->key_code)][i]; letter[1] = 0;
                    if (i == input_state->alphabet_index) {
                        gfx_draw_rectangle(global_state->gfx, x_pos-1, y_pos, FONT_WIDTH_HALF+1, FONT_HEIGHT, candidate1_bg_R, candidate1_bg_G, candidate1_bg_B, 1);
                        gfx_draw_textline(global_state->gfx, letter, x_pos, y_pos, candidate1_fg_R, candidate1_fg_G, candidate1_fg_B, 1);
                    }
                    else {
                        gfx_draw_rectangle(global_state->gfx, x_pos-1, y_pos, FONT_WIDTH_HALF+1, FONT_HEIGHT, candidate0_bg_R, candidate0_bg_G, candidate0_bg_B, 1);
                        gfx_draw_textline(global_state->gfx, letter, x_pos, y_pos, candidate0_fg_R, candidate0_fg_G, candidate0_fg_B, 1);
                    }
                    x_pos += 8;
                }

                input_state->state = 0;
            }
        }

        // 长+短按A键：删除一个字符，或返回上一个状态，取决于缓冲区状态和Ctrl状态
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
            input_state->state = 0;
            // 如果缓冲区非空且非Ctrl状态，则删除一个字符
            if (global_state->is_ctrl_enabled == 0 && input_state->textarea.length >= 1) {
                // input_state->text[--(input_state->length)] = 0;
                // input_state->cursor_pos--;
                delete_char(input_state);
                ui_draw_input_buffer(key_event, global_state, input_state);
            }
            // 如果缓冲区空，或者是Ctrl状态，则清空缓冲区，回到上一个状态
            else {
                // 重置Ctrl状态
                if (global_state->is_ctrl_enabled == 1) {
                    global_state->is_ctrl_enabled = 0;
                }
                ui_widget_input_init(key_event, global_state, input_state);
                return prev_focus_state;
            }
        }

        // 长+短按B键：依次切换汉-英-数输入模式 / 或Ctrl显示帮助
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 11) {
            // 如果非Ctrl状态，则依次切换汉-英-数输入模式
            if (global_state->is_ctrl_enabled == 0) {
                input_state->ime_mode_flag = (input_state->ime_mode_flag + 1) % 3;
                ui_draw_input_buffer(key_event, global_state, input_state);
                input_state->state = 0;
            }
            // 如果Ctrl，则显示帮助文本
            else {
                // 重置Ctrl状态
                global_state->is_ctrl_enabled = 0;
                ui_draw_input_help(key_event, global_state);
                input_state->state = 9;
            }
        }

        // 短按C键：切换全局Ctrl键状态
        else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_C) {
            global_state->is_ctrl_enabled = 1 - global_state->is_ctrl_enabled;
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->state = 0;
        }

        // 短按D键：进入下一个状态；或者Ctrl状态下 输入一个换行符
        else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_D) {
            if (global_state->is_ctrl_enabled == 1) {
                global_state->is_ctrl_enabled = 0;
                insert_char(input_state, L'\n');
                ui_draw_input_buffer(key_event, global_state, input_state);
            }
            else {
                input_state->state = 0;
                return next_focus_state;
            }
        }

        // 长+短按*键：光标向左移动
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
            if (input_state->cursor_pos > -1) {
                input_state->cursor_pos--;
            }
            else {
                input_state->cursor_pos = -1;
            }
            ui_draw_input_buffer(key_event, global_state, input_state);
        }

        // 长+短按#键：光标向右移动
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
            if (input_state->cursor_pos < input_state->textarea.length - 1) {
                input_state->cursor_pos++;
            }
            else {
                input_state->cursor_pos = input_state->textarea.length - 1;
            }
            ui_draw_input_buffer(key_event, global_state, input_state);
        }

        // 无按键：光标闪烁
        else {
            if (global_state->timer % 120 == 0) {
                ui_draw_input_cursor(key_event, global_state, input_state);
                gfx_refresh(global_state->gfx);
            }
        }
    }

    else if (state == 1) {
        // 短按D键：开始选字
        if (key_event->key_edge == -1 && key_event->key_code == 13) {
            if (input_state->candidate_num > 0) {
                ui_draw_input_pinyin(key_event, global_state, input_state, 1);
                input_state->state = 2;
            }
        }

        // 短按A键：取消输入拼音，清除已输入的所有按键，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }

        // 短按2-9键：继续输入拼音
        else if (key_event->key_edge == -1 && (key_event->key_code >= 2 && key_event->key_code <= 9)) {
            input_state->pinyin_keys *= 10;
            input_state->pinyin_keys += (uint32_t)(key_event->key_code);

            memset(input_state->candidates, 0, sizeof(input_state->candidates));
            memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));

            get_candidate_hanzi_list(input_state);

            if (input_state->candidate_num > 0) { // 如果当前键码有对应的候选字
                // 候选字列表分页
                candidate_paging(input_state);
                ui_draw_input_pinyin(key_event, global_state, input_state, 0);
            }
            else {
                ui_draw_input_pinyin(key_event, global_state, input_state, 0);
            }

            input_state->state = 1;
        }
    }

    else if (state == 2) {
        // 短按0-9键：从候选字列表中选定一个字，选定后转到初始状态
        if (key_event->key_edge == -1 && (key_event->key_code >= 0 && key_event->key_code <= 9)) {
            uint32_t index = (key_event->key_code == 0) ? 9 : (key_event->key_code - 1); // 按键0对应9
            // 将选中的字加入输入缓冲区
            uint32_t ch = input_state->candidate_pages[input_state->current_page][index];
            if (ch) {
                // input_state->text[(input_state->length)++] = ch;
                // input_state->cursor_pos++;
                insert_char(input_state, ch);
            }
            else {
                printf("选定了列表之外的字，忽略。\n");
            }

            ui_draw_input_buffer(key_event, global_state, input_state);

            memset(input_state->candidates, 0, sizeof(input_state->candidates));
            memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));
            input_state->current_page = 0;

            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }

        // 长+短按*键：候选字翻页到上一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
            if(input_state->current_page > 0) {
                input_state->current_page--;
                ui_draw_input_pinyin(key_event, global_state, input_state, 1);
            }
            input_state->state = 2;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                ui_draw_input_pinyin(key_event, global_state, input_state, 1);
            }
            input_state->state = 2;
        }

        // 短按A键：取消选择，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }
    }

    else if (state == 3) {
        // 短按0-9键：从符号列表中选定一个符号，选定后转到初始状态
        if (key_event->key_edge == -1 && (key_event->key_code >= 0 && key_event->key_code <= 9)) {
            uint32_t index = (key_event->key_code == 0) ? 9 : (key_event->key_code - 1); // 按键0对应9
            // 将选中的符号加入输入缓冲区
            uint32_t ch = input_state->candidate_pages[input_state->current_page][index];
            if (ch) {
                // input_state->text[(input_state->length)++] = ch;
                // input_state->cursor_pos++;
                insert_char(input_state, ch);
            }
            else {
                printf("选定了列表之外的符号，忽略。\n");
            }
            ui_draw_input_buffer(key_event, global_state, input_state);

            memset(input_state->candidates, 0, sizeof(input_state->candidates));
            memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));
            input_state->current_page = 0;

            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }

        // 长+短按*键：候选字翻页到上一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
            if(input_state->current_page > 0) {
                input_state->current_page--;
                ui_draw_input_symbol(key_event, global_state, input_state);
            }
            input_state->state = 3;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                ui_draw_input_symbol(key_event, global_state, input_state);
            }
            input_state->state = 3;
        }

        // 短按A键：取消选择，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }
    }

    // 特殊状态：显示使用说明
    else if (state == 9) {
        // 按任意键返回状态0
        if ((key_event->key_edge < 0) && key_event->key_code != KEYCODE_NUM_IDLE) {
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->state = 0;
        }
    }

    return current_focus_state;
}




void ui_widget_menu_init(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    menu_state->x = 0;
    menu_state->y = 14;
    menu_state->zindex = 0;
    menu_state->width = global_state->gfx->width;
    menu_state->height = global_state->gfx->height - 14 - 14;
    menu_state->current_item_index = 0;
    menu_state->first_item_intex = 0;
    uint32_t max_items_per_page = (menu_state->height - FONT_HEIGHT + 1) / (FONT_HEIGHT + 1);
    menu_state->items_per_page = (menu_state->item_num > max_items_per_page) ? max_items_per_page : menu_state->item_num;

    ui_widget_menu_draw(key_event, global_state, menu_state);
}

void ui_widget_menu_refresh(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    ui_widget_menu_draw(key_event, global_state, menu_state);
}

void ui_widget_menu_draw(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {

    uint32_t x_indent = 6;

    // 清除背景
    gfx_draw_rectangle(global_state->gfx, menu_state->x, menu_state->y, menu_state->width, menu_state->height, 255, 255, 255, 1);

    // 菜单首行：标题和选项数
    // gfx_draw_textline(global_state->gfx, menu_state->title, x_indent, 0, 0, 255, 255, 1);
    // wchar_t item_counter[13];
    // swprintf(item_counter, 13, L"%d/%d", menu_state->current_item_index + 1, menu_state->item_num);
    // int32_t iclen = wcslen(item_counter);
    // gfx_draw_textline(global_state->gfx, item_counter, (global_state->gfx->width-2) - iclen * 6, 0, 255, 255, 0, 1);

    uint32_t y_pos = menu_state->y + 1;
    uint8_t is_highlight = 0;
    for (uint32_t i = menu_state->first_item_intex; i < menu_state->item_num; i++) {
        if (i == menu_state->first_item_intex + menu_state->items_per_page) {
            break;
        }
        if (i != menu_state->current_item_index) {
            is_highlight = 0;
        }
        else {
            is_highlight = 1;
        }
        // 绘制高亮底色
        if (is_highlight) {
            for (uint32_t j = y_pos - 1; j < y_pos + FONT_HEIGHT; j++) {
                gfx_draw_line(global_state->gfx, menu_state->x, j, menu_state->x + menu_state->width, j, 222, 222, 222, 1);
            }
        }
        // 绘制文字
        gfx_draw_textline(global_state->gfx, menu_state->items[i], menu_state->x + x_indent, y_pos, 0, 0, 0, 1);

        y_pos += (FONT_HEIGHT + 1);
    }

    // 菜单的滚动条
    ui_draw_scroll_bar(
        key_event, global_state,
        menu_state->first_item_intex, menu_state->item_num, menu_state->items_per_page,
        menu_state->x, menu_state->y, menu_state->width, menu_state->height);

    // NOTE 因fb_draw_textline会额外给文字上方增加一行，因此这个横线在菜单文字绘制之后再绘制
    // gfx_draw_line(global_state->gfx, 0, 12, global_state->gfx->width, 12, 128, 128, 128, 1);

    gfx_refresh(global_state->gfx);
}


// 通用的菜单事件处理+回调注册
int32_t ui_widget_menu_event_handler(
    Key_Event *ke, Global_State *gs, Widget_Menu_State *ms,
    int32_t (*menu_item_action_callback)(Key_Event*, Global_State*, Widget_Menu_State*), int32_t prev_focus_state, int32_t current_focus_state
) {
    // 短按1-9数字键：直接选中屏幕上显示的那页的相对第几项
    // NOTE 从1开始
    // if (ke->key_edge == -1 && (ke->key_code >= KEYCODE_NUM_1 && ke->key_code <= KEYCODE_NUM_9)) {
    //     if (ke->key_code <= ms->items_per_page) {
    //         ms->current_item_index = ms->first_item_intex + (uint32_t)(ke->key_code) - 1;
    //         return menu_item_action_callback(ke, gs, ms);
    //     }
    // }
    // 短按A键：返回上一个焦点状态
    if (ke->key_edge == -1 && ke->key_code == KEYCODE_NUM_A) {
        return prev_focus_state;
    }
    // 短按D键：执行菜单项对应的功能
    else if (ke->key_edge == -1 && ke->key_code == KEYCODE_NUM_D) {
        return menu_item_action_callback(ke, gs, ms);
    }
    // 长+短按*键：光标向上移动
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == KEYCODE_NUM_STAR) {
        if (ms->first_item_intex == 0 && ms->current_item_index == 0) {
            ms->first_item_intex = ms->item_num - ms->items_per_page;
            ms->current_item_index = ms->item_num - 1;
        }
        else if (ms->current_item_index == ms->first_item_intex) {
            ms->first_item_intex--;
            ms->current_item_index--;
        }
        else {
            ms->current_item_index--;
        }

        ui_widget_menu_draw(ke, gs, ms);

        return current_focus_state;
    }
    // 长+短按#键：光标向下移动
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == KEYCODE_NUM_HASH) {
        if (ms->first_item_intex == ms->item_num - ms->items_per_page && ms->current_item_index == ms->item_num - 1) {
            ms->first_item_intex = 0;
            ms->current_item_index = 0;
        }
        else if (ms->current_item_index == ms->first_item_intex + ms->items_per_page - 1) {
            ms->first_item_intex++;
            ms->current_item_index++;
        }
        else {
            ms->current_item_index++;
        }

        ui_widget_menu_draw(ke, gs, ms);

        return current_focus_state;
    }

    return current_focus_state;
}













void ui_draw_input_buffer(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {

    Widget_Textarea_State *ta = &(input_state->textarea);

    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    // 底部
    ui_draw_footer(key_event, global_state, L"Ctrl+B 使用说明", 1);

    // 顶部
    ui_draw_header(key_event, global_state, L"", 0);
    gfx_draw_textline(global_state->gfx, L"请输入", 0, 1, 255, 255, 255, 1);
    // 显示思考模式启用状态
    if (global_state->is_thinking_enabled == 1) {
        gfx_draw_textline(global_state->gfx, L"Ψ", global_state->gfx->width - 8*FONT_WIDTH_HALF - 1, 1, 0, 255, 255, 1);
    }
    // 显示Ctrl激活状态
    if (global_state->is_ctrl_enabled == 1) {
        gfx_draw_textline(global_state->gfx, L"◆", global_state->gfx->width - 6*FONT_WIDTH_HALF - 1, 1, 255, 255, 255, 1);
    }
    // 显示输入状态
    if (input_state->ime_mode_flag == IME_MODE_HANZI) {
        gfx_draw_textline(global_state->gfx, L"[汉]", global_state->gfx->width - 4*FONT_WIDTH_HALF - 1, 1, 255, 255, 0, 1);
    }
    else if (input_state->ime_mode_flag == IME_MODE_ALPHABET) {
        gfx_draw_textline(global_state->gfx, L"[En]", global_state->gfx->width - 4*FONT_WIDTH_HALF - 1, 1, 255, 255, 0, 1);
    }
    else if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
        gfx_draw_textline(global_state->gfx, L"[数]", global_state->gfx->width - 4*FONT_WIDTH_HALF - 1, 1, 255, 255, 0, 1);
    }


    // 第一次排版：用于判断光标是否在视图内部
    // ta->current_line = 0;
    typeset_line_breaks(key_event, global_state, ta);
    typeset_view_range(ta);

    // 如果光标不在当前视图范围内
    if (input_state->cursor_pos < ta->view_start_pos || input_state->cursor_pos > ta->view_end_pos) {
        uint32_t cursor_line = 0;
        // 寻找当前光标所在的行
        for (int32_t i = 0; i < ta->line_num; i++) {
            int32_t a = ta->break_pos[i];
            int32_t b = (i == ta->line_num - 1) ? ta->length : ta->break_pos[i+1];
            if (input_state->cursor_pos >= a && input_state->cursor_pos < b) {
                cursor_line = i;
            }
        }

        // 如果光标在当前视图上方，则将current_line设为当前光标所在的行
        if (input_state->cursor_pos < ta->view_start_pos) {
            ta->current_line = cursor_line;
        }
        // 如果光标在当前视图下方，则将current_line设为当前光标所在行上方view_lines行（即，使得光标所在行位于视图的末行）
        //   逻辑上，如果出现这种情况，一定有 line_num > view_lines
        else {
            ta->current_line = cursor_line - ta->view_lines + 1;
        }
        // 重新排版
        typeset_line_breaks(key_event, global_state, ta);
        typeset_view_range(ta);
    }

    // 绘制文本
    ui_draw_text_block(key_event, global_state, ta);

    // 绘制滚动条
    if (ta->is_show_scroll_bar) {
        ui_draw_scroll_bar(
            key_event, global_state,
            ta->current_line, ta->line_num, ta->view_lines,
            ta->x, ta->y, ta->width, ta->height);
    }

    // 绘制光标
    ui_draw_input_cursor(key_event, global_state, input_state);

    gfx_refresh(global_state->gfx);
}


void ui_draw_input_cursor(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    Widget_Textarea_State *ta = &(input_state->textarea);
    // 绘制光标：光标位置在cursor_pos所指字符的右外边缘
    int32_t char_index = 0;
    int32_t break_count = 0;
    int32_t line_x_pos = ta->x;
    if (input_state->cursor_pos >= 0) {
        for (char_index = ta->view_start_pos; char_index <= ta->view_end_pos; char_index++) {
            wchar_t ch = ta->text[char_index];
            int32_t char_width = (ch < 127) ? ((ch == '\n') ? 0 : 6) : 12;
            if (line_x_pos + char_width >= ta->x + ta->width || ch == '\n') {
                break_count++;
                line_x_pos = ta->x;
            }
            line_x_pos += char_width;
            if (input_state->cursor_pos == char_index) break;
        }
    }

    uint8_t cursor_R = 0, cursor_G = 0, cursor_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        cursor_R = 0; cursor_G = 0; cursor_B = 64;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        cursor_R = 255; cursor_G = 255; cursor_B = 255;
    }

    uint32_t x = line_x_pos;
    uint32_t y = ta->y + 13 * break_count; // 12x12字模底部本来就有1px的空白，加上行间距1px，所以每行的起始位置是13的倍数
    gfx_draw_line(global_state->gfx, x, y-1, x, y+12, cursor_R, cursor_G, cursor_B, 2);
    gfx_draw_line(global_state->gfx, x+1, y-1, x+1, y+12, cursor_R, cursor_G, cursor_B, 2);
}

void ui_draw_input_pinyin(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state, uint32_t is_picking) {
    // gfx_soft_clear(global_state->gfx);
    // 计算候选列表长度
    uint32_t count = 0;
    wchar_t cc[MAX_CANDIDATE_NUM_PER_PAGE + 1];
    wchar_t cindex[21] = L"1 2 3 4 5 6 7 8 9 0 ";

    for(int j = 0; j < MAX_CANDIDATE_NUM_PER_PAGE; j++) {
        wchar_t ch = input_state->candidate_pages[input_state->current_page][j];
        if (!ch) break;
        cc[j] = ch;
        count++;
    }
    cc[count] = 0;
    cindex[count << 1] = 0;

    uint32_t x_offset = 1;
    uint32_t y_offset = input_state->textarea.y + input_state->textarea.height - FONT_HEIGHT*3 - 1;

    // 清空输入法显示区域
    gfx_draw_rectangle(global_state->gfx,
        input_state->textarea.x, y_offset-1,
        input_state->textarea.width, input_state->textarea.y + input_state->textarea.height - y_offset + 1 + 1,
        232, 235, 243, 1);

    wchar_t buf[30];
    if (is_picking) {
        swprintf(buf, 30, L"PY[%-6d]   (%2d/%2d)", input_state->pinyin_keys, (input_state->current_page+1), input_state->candidate_page_num);
        gfx_draw_textline(global_state->gfx, buf, x_offset, y_offset + 0, 17, 85, 238, 1);
        gfx_draw_textline(global_state->gfx, cindex, x_offset, y_offset + 13, 128, 128, 128, 1);
    }
    else {
        swprintf(buf, 30, L"PY[%-6d]", input_state->pinyin_keys);
        gfx_draw_textline(global_state->gfx, buf, x_offset, y_offset + 0, 17, 85, 238, 1);
    }
    if (input_state->candidate_num > 0) {
        gfx_draw_textline(global_state->gfx, cc, x_offset, y_offset + 26, 0, 0, 0, 1);
    }
    else {
        gfx_draw_textline(global_state->gfx, L"(无候选字)", x_offset, y_offset + 26, 128, 128, 128, 1);
    }

    gfx_draw_line(global_state->gfx, input_state->textarea.x, y_offset-2, input_state->textarea.x + input_state->textarea.width, y_offset-2, 232, 235, 243, 1);

    gfx_refresh(global_state->gfx);
}

void ui_draw_input_symbol(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    // gfx_soft_clear(global_state->gfx);
    // 计算候选列表长度
    uint32_t count = 0;
    uint32_t list_char_width = 0;
    wchar_t cc[21];
    wchar_t cindex[21] = L"1 2 3 4 5 6 7 8 9 0 ";

    for(int j = 0; j < 10; j++) {
        wchar_t ch = input_state->candidate_pages[input_state->current_page][j];
        if (!ch) {
            break;
        }
        else if (ch < 127) {
            cc[list_char_width++] = ch;
            cc[list_char_width++] = L' ';
        }
        else {
            cc[list_char_width++] = ch;
        }
        count++;
    }
    cc[list_char_width] = 0;
    cindex[count << 1] = 0;

    uint32_t x_offset = 1;
    uint32_t y_offset = input_state->textarea.y + input_state->textarea.height - FONT_HEIGHT*3 - 1;

    // 清空输入法显示区域
    gfx_draw_rectangle(global_state->gfx,
        input_state->textarea.x, y_offset-1,
        input_state->textarea.width, input_state->textarea.y + input_state->textarea.height - y_offset + 1 + 1,
        232, 235, 243, 1);

    wchar_t text[30];

    swprintf(text, 30, L"Symbols      (%2d/%2d)", (input_state->current_page+1), input_state->candidate_page_num);
    gfx_draw_textline(global_state->gfx, text, x_offset, y_offset + 0, 17, 85, 238, 1);
    gfx_draw_textline(global_state->gfx, cindex, x_offset, y_offset + 13, 128, 128, 128, 1);

    if (input_state->candidate_num > 0) {
        gfx_draw_textline(global_state->gfx, cc, x_offset, y_offset + 26, 0, 0, 0, 1);
    }
    else {
        gfx_draw_textline(global_state->gfx, L"(无候选符号)", x_offset, y_offset + 26, 128, 128, 128, 1);
    }

    gfx_draw_line(global_state->gfx, input_state->textarea.x, y_offset-2, input_state->textarea.x + input_state->textarea.width, y_offset-2, 232, 235, 243, 1);

    gfx_refresh(global_state->gfx);
}










// ===============================================================================
// 七段码
// ===============================================================================

/* 笔画长度 l 与粗细 w，可自定义。整体尺寸由二者决定：
   宽度 = l + 2*w, 高度 = 2*l + 3*w */
#define SEG_LENGTH       16.0f
#define SEG_THICKNESS    5.0f
#define CFG_DIGIT_W      (SEG_LENGTH + 2.0f * SEG_THICKNESS)
#define CFG_DIGIT_H      (2.0f * SEG_LENGTH + 3.0f * SEG_THICKNESS)
#define CFG_DIGIT_GAP    6.0f

/* ============================================================
   静态常量数组: 10个数字 x 7个段 (1=点亮, 0=熄灭)
   段索引: 0=上, 1=右上, 2=右下, 3=下, 4=左下, 5=左上, 6=中
   ============================================================ */
static const int g_digit_map[10][7] = {
    {1,1,1,1,1,1,0}, /* 0 */
    {0,1,1,0,0,0,0}, /* 1 */
    {1,1,0,1,1,0,1}, /* 2 */
    {1,1,1,1,0,0,1}, /* 3 */
    {0,1,1,0,0,1,1}, /* 4 */
    {1,0,1,1,0,1,1}, /* 5 */
    {1,0,1,1,1,1,1}, /* 6 */
    {1,1,1,0,0,0,0}, /* 7 */
    {1,1,1,1,1,1,1}, /* 8 */
    {1,1,1,1,0,1,1}  /* 9 */
};

static void draw_seg_rect(Nano_GFX *gfx, float x, float y, float w, float h, int32_t is_shadow, int32_t is_on, uint8_t red, uint8_t green, uint8_t blue) {
    uint32_t rx = (uint32_t)x;
    uint32_t ry = (uint32_t)y;
    uint32_t rw = (uint32_t)w;
    uint32_t rh = (uint32_t)h;
    if (rw == 0) rw = 1;
    if (rh == 0) rh = 1;

    // 判断是横画还是竖画
    int32_t is_heng = (rw > rh) ? 1 : 0;

    if (!is_on) {
        return;
    }

    if (is_heng) {
        int32_t thickness = rh;
        for (int32_t x = 1; x <= thickness/2; x++) {
            int32_t xx1 = rx - x;
            int32_t xx2 = rx + rw - 1 + x;
            int32_t y1 = ry + (thickness/2) - (thickness - 2 * x) / 2;
            int32_t y2 = ry + (thickness/2) + (thickness - 2 * x) / 2;
            gfx_draw_line(gfx, xx1, y1, xx1, y2, red, green, blue, 1);
            gfx_draw_line(gfx, xx2, y1, xx2, y2, red, green, blue, 1);
            if (is_shadow) {
                gfx_draw_point(gfx, xx2, y2+1, 127, 127, 127, 1);
            }
        }
        if (is_shadow) {
            gfx_draw_line(gfx, rx, ry+rh, rx+rw-1, ry+rh, 127, 127, 127, 1);
        }
    }
    else {
        int32_t thickness = rw;
        for (int32_t y = 1; y <= thickness/2; y++) {
            int32_t yy1 = ry - y;
            int32_t yy2 = ry + rh - 1 + y;
            int32_t x1 = rx + (thickness/2) - (thickness - 2 * y) / 2;
            int32_t x2 = rx + (thickness/2) + (thickness - 2 * y) / 2;
            gfx_draw_line(gfx, x1, yy1, x2, yy1, red, green, blue, 1);
            gfx_draw_line(gfx, x1, yy2, x2, yy2, red, green, blue, 1);
            if (is_shadow) {
                gfx_draw_point(gfx, x2+1, yy2, 127, 127, 127, 1);
            }
        }
        if (is_shadow) {
            gfx_draw_line(gfx, rx+rw, ry, rx+rw, ry+rh-1, 127, 127, 127, 1);
        }
    }
    gfx_draw_rectangle(gfx, rx, ry, rw, rh, red, green, blue, 1);

}

/* 绘制单个数字 (0-9)
   use_rect 参数已弃用，保留仅为兼容现有调用签名 */
void ui_draw_7seg_digit(
    Nano_GFX *gfx, int num, float ox, float oy,
    float seg_length, float seg_thickness, int32_t is_shadow,
    uint8_t red, uint8_t green, uint8_t blue,
    float *digit_width, float *digit_height
) {
    float l = seg_length;
    float w = seg_thickness;

    *digit_width = seg_length + 2.0f * seg_thickness;
    *digit_height = 2.0f * seg_length + 3.0f * seg_thickness;

    /* 各段矩形坐标与尺寸 (x, y, width, height)
       横画: l=width, w=height;  竖画: l=height, w=width
       角点相接关系:
       B0=D1, C0=A5, C1=A6, D2=B6, C2=A3, D3=B4, A4=C6, B5=D6 */
    float seg_x[7], seg_y[7], seg_w[7], seg_h[7];

    /* 0: 上横 */
    seg_x[0] = ox + w;     seg_y[0] = oy;
    seg_w[0] = l;          seg_h[0] = w;

    /* 1: 右上竖 */
    seg_x[1] = ox + w + l; seg_y[1] = oy + w;
    seg_w[1] = w;          seg_h[1] = l;

    /* 2: 右下竖 */
    seg_x[2] = ox + w + l; seg_y[2] = oy + w + l + w;
    seg_w[2] = w;          seg_h[2] = l;

    /* 3: 下横 */
    seg_x[3] = ox + w;     seg_y[3] = oy + w + l + w + l;
    seg_w[3] = l;          seg_h[3] = w;

    /* 4: 左下竖 */
    seg_x[4] = ox;         seg_y[4] = oy + w + l + w;
    seg_w[4] = w;          seg_h[4] = l;

    /* 5: 左上竖 */
    seg_x[5] = ox;         seg_y[5] = oy + w;
    seg_w[5] = w;          seg_h[5] = l;

    /* 6: 中横 */
    seg_x[6] = ox + w;     seg_y[6] = oy + w + l;
    seg_w[6] = l;          seg_h[6] = w;

    for (int i = 0; i < 7; i++) {
        draw_seg_rect(gfx, seg_x[i], seg_y[i], seg_w[i], seg_h[i], is_shadow, g_digit_map[num][i], red, green, blue);
    }
}

/* 绘制时间分隔符 (两个实心方块) */
void ui_draw_7seg_colon(
    Nano_GFX *gfx, float ox, float oy,
    float seg_length, float seg_thickness, int32_t is_shadow,
    uint8_t red, uint8_t green, uint8_t blue,
    float *digit_width, float *digit_height
) {
    *digit_height = 2.0f * seg_length + 3.0f * seg_thickness;
    *digit_width = (seg_length + 2.0f * seg_thickness) / 2.0f;
    float h = (*digit_height);

    /* 计算上下圆点中心 Y */
    float cx = ox + (*digit_width) / 2.0f;
    float cy1 = oy + h * 0.25f;
    float cy2 = oy + h * 0.75f;

    /* 上圆点 */
    uint32_t x0 = (uint32_t)(cx - seg_thickness/2);
    uint32_t y1 = (uint32_t)(cy1 - seg_thickness/2);
    uint32_t y2 = (uint32_t)(cy2 - seg_thickness/2);
    gfx_draw_rectangle(gfx, x0, y1, seg_thickness, seg_thickness, red, green, blue, 1);
    if (is_shadow) {
        gfx_draw_line(gfx, x0, y1+seg_thickness-1, x0+seg_thickness-1, y1+seg_thickness-1, 127, 127, 127, 1);
        gfx_draw_line(gfx, x0+seg_thickness-1, y1, x0+seg_thickness-1, y1+seg_thickness-1, 127, 127, 127, 1);
    }

    /* 下圆点 */
    gfx_draw_rectangle(gfx, x0, y2, seg_thickness, seg_thickness, red, green, blue, 1);
    if (is_shadow) {
        gfx_draw_line(gfx, x0, y2+seg_thickness-1, x0+seg_thickness-1, y2+seg_thickness-1, 127, 127, 127, 1);
        gfx_draw_line(gfx, x0+seg_thickness-1, y2, x0+seg_thickness-1, y2+seg_thickness-1, 127, 127, 127, 1);
    }
}

void ui_draw_7seg_string(
    Key_Event *key_event, Global_State *global_state,
    int32_t xx, int32_t yy, wchar_t *text,
    uint8_t red, uint8_t green, uint8_t blue,
    float seg_length, float seg_thickness, float digit_gap, int32_t is_shadow,
    int32_t *text_width, int32_t *text_height
) {
    float digit_width = 0.0f;
    float digit_height = 0.0f;
    float x = xx;
    int32_t len = wcslen(text);
    for (int32_t i = 0; i < len; i++) {
        // 检查字符范围
        wchar_t ch = text[i];
        if (ch >= L'0' && ch <= L'9') {
            int32_t num = (uint32_t)ch - (uint32_t)(L'0');
            ui_draw_7seg_digit(global_state->gfx, num, x, yy, seg_length, seg_thickness, is_shadow, red, green, blue, &digit_width, &digit_height);
            x += digit_width + digit_gap;
        }
        else if (ch == L':') {
            ui_draw_7seg_colon(global_state->gfx, x, yy, seg_length, seg_thickness, is_shadow, red, green, blue, &digit_width, &digit_height);
            x += digit_width + digit_gap;
        }
    }
    *text_width = (int32_t)roundf(x - xx);
    *text_height = (int32_t)roundf(digit_height);
}


