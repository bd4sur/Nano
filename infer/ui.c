#include <stdio.h>
#include <time.h>

#include "graphics.h"
#include "hal_key.h"
#include "ui.h"
#include "ui_softkbd.h"
#include "ui_grid16kbd.h"
#include "ui_pinyin_ime.h"

#include "platform.h"

#include "ui_color.h"
#include "ui_pinyin_lut.h"

// 全局色彩变量（用于调节UI配色风格）

static uint8_t S_UI_COLOR_HEADER_TEXT[3]   = {255, 255, 255};
static uint8_t S_UI_COLOR_FOOTER_BG[3]     = {224, 230, 234};
static uint8_t S_UI_COLOR_FOOTER_TEXT[3]   = {90 , 98 , 106};

static uint8_t S_UI_COLOR_IME_HELP_BG[3]   = {222, 222, 222};
static uint8_t S_UI_COLOR_IME_HELP_TEXT[3] = {0  , 0  , 0  };

// 输入法候选列表（候选字/候选符号）颜色，随全局颜色风格切换（由 ui_ime_candidate_color_apply 应用）
static uint8_t S_UI_COLOR_IME_CANDIDATE_BG[3]     = {232, 235, 243}; // 候选列表底色
static uint8_t S_UI_COLOR_IME_CANDIDATE_TEXT[3]   = {0  , 0  , 0  }; // 候选字/候选符号文字
static uint8_t S_UI_COLOR_IME_CANDIDATE_INDEX[3]  = {128, 128, 128}; // 候选序号（灰，两种风格相同）
static uint8_t S_UI_COLOR_IME_CANDIDATE_PINYIN[3] = {17 , 85 , 238}; // 拼音行（蓝，两种风格相同）

// 按全局颜色风格应用输入法候选列表配色：亮色保持默认；暗色改深灰底+白色候选字，
// 序号灰与拼音蓝保持原值（均已参数化，可直接改上方默认值或此处的暗色值）
static void ui_ime_candidate_color_apply(int32_t ui_color_style) {
    if (ui_color_style == UI_COLOR_DARK) {
        S_UI_COLOR_IME_CANDIDATE_BG[0]   = 45 ; S_UI_COLOR_IME_CANDIDATE_BG[1]   = 48 ; S_UI_COLOR_IME_CANDIDATE_BG[2]   = 54 ;
        S_UI_COLOR_IME_CANDIDATE_TEXT[0] = 255; S_UI_COLOR_IME_CANDIDATE_TEXT[1] = 255; S_UI_COLOR_IME_CANDIDATE_TEXT[2] = 255;
    }
    else {
        S_UI_COLOR_IME_CANDIDATE_BG[0]   = 232; S_UI_COLOR_IME_CANDIDATE_BG[1]   = 235; S_UI_COLOR_IME_CANDIDATE_BG[2]   = 243;
        S_UI_COLOR_IME_CANDIDATE_TEXT[0] = 0  ; S_UI_COLOR_IME_CANDIDATE_TEXT[1] = 0  ; S_UI_COLOR_IME_CANDIDATE_TEXT[2] = 0  ;
    }
}

// 页脚（底栏）底色（与 ui_draw_footer 一致）：亮色取 S_UI_COLOR_FOOTER_BG，暗色为 (15,16,17)。
// 16键输入法候选条绘制于页脚带内，底色与页脚保持一致
static void ui_footer_bg_color(int32_t ui_color_style, uint8_t *r, uint8_t *g, uint8_t *b) {
    if (ui_color_style == UI_COLOR_DARK) { *r = 15; *g = 16; *b = 17; }
    else { *r = S_UI_COLOR_FOOTER_BG[0]; *g = S_UI_COLOR_FOOTER_BG[1]; *b = S_UI_COLOR_FOOTER_BG[2]; }
}


// 符号列表
static wchar_t ime_symbols[55] = L"，。、？！：；“”‘’（）《》…—～·【】 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
// 按键对应的字母列表
static wchar_t ime_alphabet[10][32] = {L"0", L" 1.,:?!-/+_=&\"*", L"abcABC2", L"defDEF3", L"ghiGHI4", L"jklJKL5", L"mnoMNO6", L"pqrsPRQS7", L"tuvTUV8", L"wxyzWXYZ9"};

// 带四舍五入的整数除法，仅接受正数
static inline uint32_t div_round(uint32_t a, uint32_t b) {
    return (a + b / 2) / b;
}

void get_candidate_hanzi_list(Widget_Input_State *input_state) {
    // 候选数量钳制在 candidates[] 容量（MAX_CANDIDATE_NUM）内：越界写入会覆盖结构体内
    // 紧随其后的 candidate_num/candidate_pages 字段，导致分页数据损坏、每页显示数量错乱
    unsigned int candidate_index[MAX_CANDIDATE_NUM];
    int candidate_count = 0;
    for(int i = 0; i < IME_HANZI_NUM; i++) {
        if(KEYS_LIST[i] == input_state->pinyin_keys && candidate_count < MAX_CANDIDATE_NUM) {
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


// 分页一致性纪律：每页候选个数由宏 MAX_CANDIDATE_NUM_PER_PAGE 唯一定义；
// 分页填充（candidate_paging）、候选条显示（ui_draw_input_pinyin/symbol）与
// 选字索引（state 2/3 的数字键 1~5）全部以该宏为同一步进，
// 显示侧按分页数学直接推导本页个数（ui_candidate_page_item_count），不再按“非零哨兵”扫描计数。
void candidate_paging(Widget_Input_State *input_state) {
    if (input_state->candidate_num > MAX_CANDIDATE_NUM) input_state->candidate_num = MAX_CANDIDATE_NUM;
    input_state->candidate_page_num = input_state->candidate_num / MAX_CANDIDATE_NUM_PER_PAGE + ((input_state->candidate_num % MAX_CANDIDATE_NUM_PER_PAGE) ? 1 : 0);
    if (input_state->candidate_page_num > MAX_CANDIDATE_PAGE_NUM) input_state->candidate_page_num = MAX_CANDIDATE_PAGE_NUM;
    memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));
    uint32_t pos = 0;
    for (uint32_t i = 0; i < input_state->candidate_page_num; i++) {
        for (uint32_t j = 0; j < MAX_CANDIDATE_NUM_PER_PAGE; j++) {
            input_state->candidate_pages[i][j] = (pos < input_state->candidate_num) ? input_state->candidates[pos] : 0; // 选字时，选到0就意味着越界了
            pos++;
        }
    }
}

// 当前页的候选个数：由分页数学直接推导（与 candidate_paging 的填充完全一致），
// 供候选条显示使用，保证“实际分页”与“每页显示数量”严格一致
static inline uint32_t ui_candidate_page_item_count(Widget_Input_State *input_state) {
    uint32_t base = input_state->current_page * MAX_CANDIDATE_NUM_PER_PAGE;
    if (base >= input_state->candidate_num) return 0;
    uint32_t rest = input_state->candidate_num - base;
    return (rest > MAX_CANDIDATE_NUM_PER_PAGE) ? MAX_CANDIDATE_NUM_PER_PAGE : rest;
}

// 在文本框的光标位置之后插入一个字符
void insert_char(Widget_Input_State *input_state, wchar_t new_char) {
    // text 缓冲区容量为 UI_STR_BUF_MAX_LENGTH 个 wchar_t（含结尾 L'\0'），
    // 插入后需保证 text[length+1] 不越界，即 length+1 <= UI_STR_BUF_MAX_LENGTH - 1
    if (input_state->textarea.length + 1 >= UI_STR_BUF_MAX_LENGTH) {
        return;
    }

    input_state->desired_x = -1; // 内容变化，重置上下移动的目标x

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

    input_state->desired_x = -1; // 内容变化，重置上下移动的目标x

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
        // 逐字符实际宽度（'\n' 不占宽）；缺字按回退字符宽度计算，与绘制时一致
        int32_t char_width = (ch == '\n') ? 0 : gfx_font_char_advance(global_state->ui_font, (uint32_t)ch);

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
//   line_height - 当前字体的行高（同一套字体行高固定，由 gfx_font_line_height 给出）
void typeset_view_range(Widget_Textarea_State *textarea_state, int32_t line_height) {
    int32_t view_height = textarea_state->height;
    // 视口整行数（排版语义，下取整）：光标跟随/翻页/滚动钳制等排版逻辑一律以此为准。
    // 视口底部的不完整行不属于排版整行数，由本函数末尾的扩展在【绘制端】多覆盖一行
    //（workaround 位于绘制端而非排版端，见 AGENTS.md 第九节）
    int32_t max_view_lines = view_height / line_height;
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

    // 注：视口底部不完整行由 ui_draw_text_block 在【绘制端】多绘制一行处理
    //（绘制循环不受 view_end_pos 限制，绘制到行顶触及文本区底缘为止），本函数保持纯排版语义
}


// font_id: 文本字体（GFX_FONT_*）。同一字体行高固定；每个字符的实际宽度不定，
// 折行与绘制均按逐字符实际宽度处理（gfx_font_char_advance / gfx_font_draw_char，
// 二者对缺字的回退策略一致）；基线对齐由字体接口内部完成。
void ui_draw_text_block(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state, uint32_t font_id) {
    // 像素级连续滚动：首行按亚行偏移上移，超出文本区部分经裁剪隐藏
    int x_pos = textarea_state->x;
    int y_pos = textarea_state->y - textarea_state->scroll_sub_offset;
    gfx_set_clip(global_state->gfx, textarea_state->x, textarea_state->y, textarea_state->width, textarea_state->height);

    // 行高：由字体决定（同一套字体行高固定）
    int32_t line_height = gfx_font_line_height(font_id);

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

    // 绘制端“多绘制一行”：循环不受排版窗口下界（view_end_pos）限制，绘制到文本末尾
    // 或行顶触及文本区底缘为止——视口底部永远有一行被截断显示（哪怕只剩 1px 可见），
    // 排版语义（max_view_lines/view_lines）不受此影响
    for (int i = textarea_state->view_start_pos; i <= textarea_state->length - 1; i++) {
        if (y_pos >= textarea_state->y + textarea_state->height) break; // 后续行已完全在裁剪区外
        // 首先检查这一位是不是格式控制标签的字符
        if (textarea_state->style[i] & 0x80000000) {
            continue;
        }
        uint32_t current_char = textarea_state->text[i];
        if (!current_char) break;
        if (current_char == '\n') {
            x_pos = textarea_state->x;
            if(i > 0) y_pos += line_height;
            continue;
        }
        // 使用当前颜色绘制字符
        uint32_t style_code = textarea_state->style[i];
        current_r = (style_code >> 16) & 0xFF;
        current_g = (style_code >> 8) & 0xFF;
        current_b = style_code & 0xFF;

        // 逐字符实际宽度（缺字时为回退字符的宽度，与 gfx_font_draw_char 一致）
        int32_t char_width = gfx_font_char_advance(font_id, current_char);
        if (x_pos + char_width >= textarea_state->x + textarea_state->width) {
            y_pos += line_height;
            x_pos = textarea_state->x;
        }
        x_pos += gfx_font_draw_char(global_state->gfx, font_id, current_char, x_pos, y_pos, current_r, current_g, current_b, 1);
    }
    gfx_reset_clip(global_state->gfx); // 恢复整屏裁剪，避免泄漏到后续帧的其它绘制
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

    int32_t scroll_bar_x_offset = 0; // 亮色暗色模式，滚动条有不同的横向偏移。亮色模式下，滚动条需要左移1px，以避免与黑色的屏幕外框连在一起看不出来。

    uint8_t scroll_bar_bg_R = 0, scroll_bar_bg_G = 0, scroll_bar_bg_B = 0;
    uint8_t scroll_bar_fg_R = 0, scroll_bar_fg_G = 0, scroll_bar_fg_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        scroll_bar_x_offset = -1;
        scroll_bar_bg_R = 222; scroll_bar_bg_G = 222; scroll_bar_bg_B = 222;
        scroll_bar_fg_R = 17; scroll_bar_fg_G = 85; scroll_bar_fg_B = 238;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        scroll_bar_x_offset = 0;
        scroll_bar_bg_R = 66; scroll_bar_bg_G = 66; scroll_bar_bg_B = 66;
        scroll_bar_fg_R = 102; scroll_bar_fg_G = 204; scroll_bar_fg_B = 255;
    }

    // for (int n = y; n < y + height; n++) {
    //     gfx_draw_point(global_state->gfx, x + width - 1, n, scroll_bar_bg_R, scroll_bar_bg_G, scroll_bar_bg_B, 1);
    // }
    gfx_draw_line(global_state->gfx, x + width - 1 + scroll_bar_x_offset, y, x + width - 1 + scroll_bar_x_offset, (y + height - 1), scroll_bar_bg_R, scroll_bar_bg_G, scroll_bar_bg_B, 1);
    gfx_draw_line(global_state->gfx, x + width - 2 + scroll_bar_x_offset, y, x + width - 2 + scroll_bar_x_offset, (y + height - 1), scroll_bar_bg_R, scroll_bar_bg_G, scroll_bar_bg_B, 1);

    line_num = (line_num <= 0) ? 1 : line_num;

    // 如果总行数装不满视图，则滚动条长度等于视图高度height
    int32_t bar_height = (line_num < view_lines) ? (height) : div_round((view_lines * height), line_num);
    bar_height = (bar_height < 3) ? 3 : bar_height; // 滚动条高度不小于3px

    // 滚动条顶部y坐标
    int32_t y_0 = y + div_round(current_line * height, line_num);
    y_0 = (y_0 >= y + height - 3 - 1) ? (y + height - 3 - 1) : y_0; // 滚动条顶部限位（不低于底部上方3px）

    gfx_draw_line(global_state->gfx, x + width - 1 + scroll_bar_x_offset, y_0, x + width - 1 + scroll_bar_x_offset, (y_0 + bar_height), scroll_bar_fg_R, scroll_bar_fg_G, scroll_bar_fg_B, 1);
    gfx_draw_line(global_state->gfx, x + width - 2 + scroll_bar_x_offset, y_0, x + width - 2 + scroll_bar_x_offset, (y_0 + bar_height), scroll_bar_fg_R, scroll_bar_fg_G, scroll_bar_fg_B, 1);
}









void ui_draw_header(Key_Event *key_event, Global_State *global_state, wchar_t *text, int32_t is_center) {
    // 标准页眉：高度为 1.5 倍字体行高（与菜单控件一致）；文本在页眉带内垂直居中
    ui_draw_header_ex(key_event, global_state, text, is_center, ui_std_header_height(global_state->ui_font));
}

// 计算渐变色：起止两点色彩沿行的线性内插（内插数量 = 渐变区行数，
// 首行恰为起始色、末行恰为终止色；行号越界钳制，单行退化为起始色）
static void ui_gradient_row_color(int32_t row, int32_t rows,
    uint8_t top_R, uint8_t top_G, uint8_t top_B,
    uint8_t bottom_R, uint8_t bottom_G, uint8_t bottom_B,
    uint8_t *out_R, uint8_t *out_G, uint8_t *out_B
) {
    if (rows < 2) rows = 2; // 防除零
    if (row < 0) row = 0;
    if (row > rows - 1) row = rows - 1;
    *out_R = (uint8_t)((int32_t)top_R + ((int32_t)bottom_R - (int32_t)top_R) * row / (rows - 1));
    *out_G = (uint8_t)((int32_t)top_G + ((int32_t)bottom_G - (int32_t)top_G) * row / (rows - 1));
    *out_B = (uint8_t)((int32_t)top_B + ((int32_t)bottom_B - (int32_t)top_B) * row / (rows - 1));
}

// 页眉底色填充（[x0, x1) × [0, header_height)）：亮色为计算渐变（上 102,204,255 → 下 17,85,238）、
// 暗色为纯色（15,16,17）。供 ui_draw_header_ex 与各处页眉标签的自清洁重绘复用。
static void ui_header_bg_fill(Global_State *global_state, int32_t x0, int32_t x1, int32_t header_height) {
    for (int32_t y = 0; y < header_height; y++) {
        uint8_t row_R, row_G, row_B;
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            ui_gradient_row_color(y, header_height, 102, 204, 255, 17, 85, 238, &row_R, &row_G, &row_B);
        }
        else {
            row_R = 15; row_G = 16; row_B = 17;
        }
        gfx_draw_line(global_state->gfx, (uint32_t)x0, (uint32_t)y, (uint32_t)(x1 - 1), (uint32_t)y, row_R, row_G, row_B, 1);
    }
}

void ui_draw_header_ex(Key_Event *key_event, Global_State *global_state, wchar_t *text, int32_t is_center, int32_t header_height) {
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        ui_header_bg_fill(global_state, 0, (int32_t)global_state->gfx->width, header_height);
        S_UI_COLOR_HEADER_TEXT[0] = 255;
        S_UI_COLOR_HEADER_TEXT[1] = 255;
        S_UI_COLOR_HEADER_TEXT[2] = 255;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        ui_header_bg_fill(global_state, 0, (int32_t)global_state->gfx->width, header_height);
        S_UI_COLOR_HEADER_TEXT[0] = 188;
        S_UI_COLOR_HEADER_TEXT[1] = 188;
        S_UI_COLOR_HEADER_TEXT[2] = 188;
    }
    if (is_center) {
        gfx_font_draw_text_centered(global_state->gfx, font_id, text, global_state->gfx->width / 2, header_height / 2, S_UI_COLOR_HEADER_TEXT[0], S_UI_COLOR_HEADER_TEXT[1], S_UI_COLOR_HEADER_TEXT[2], 1);
    }
    else {
        gfx_font_draw_text(global_state->gfx, font_id, text, 0, header_height / 2 - line_height / 2, S_UI_COLOR_HEADER_TEXT[0], S_UI_COLOR_HEADER_TEXT[1], S_UI_COLOR_HEADER_TEXT[2], 1);
    }
}

void ui_draw_footer(Key_Event *key_event, Global_State *global_state, wchar_t *text, int32_t is_center) {
    // 页脚高度跟随当前字体行高（行高 + 1px 边距）
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    const int footer_height = line_height + 1;
    // 触屏软键盘与16键虚拟键盘显示时，页脚上移为键盘让出空间（键盘均隐藏时两者高度为0，行为不变）
    const int32_t footer_bottom = global_state->gfx->height - ui_softkbd_height() - ui_grid16kbd_height();
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_draw_rectangle(global_state->gfx, 0, footer_bottom - footer_height, global_state->gfx->width, footer_height, S_UI_COLOR_FOOTER_BG[0], S_UI_COLOR_FOOTER_BG[1], S_UI_COLOR_FOOTER_BG[2], 1);
        S_UI_COLOR_FOOTER_TEXT[0] = 90;
        S_UI_COLOR_FOOTER_TEXT[1] = 98;
        S_UI_COLOR_FOOTER_TEXT[2] = 106;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_draw_rectangle(global_state->gfx, 0, footer_bottom - footer_height, global_state->gfx->width, footer_height, 15, 16, 17, 1);
        S_UI_COLOR_FOOTER_TEXT[0] = 188;
        S_UI_COLOR_FOOTER_TEXT[1] = 188;
        S_UI_COLOR_FOOTER_TEXT[2] = 188;
    }
    if (is_center) {
        gfx_font_draw_text_centered(global_state->gfx, font_id, text, global_state->gfx->width / 2, footer_bottom - footer_height + footer_height / 2, S_UI_COLOR_FOOTER_TEXT[0], S_UI_COLOR_FOOTER_TEXT[1], S_UI_COLOR_FOOTER_TEXT[2], 1);
    }
    else {
        gfx_font_draw_text(global_state->gfx, font_id, text, 0, footer_bottom - footer_height + footer_height / 2 - line_height / 2, S_UI_COLOR_FOOTER_TEXT[0], S_UI_COLOR_FOOTER_TEXT[1], S_UI_COLOR_FOOTER_TEXT[2], 1);
    }
}

// 绘制软按键提示区页脚（类似早期手机屏幕底部的软按键提示）：
// 页脚作为触屏十六宫格最底部一行4个按键（*、0、#、D）在当前功能状态下的功能提示。
// 4个提示字符串的中心在横向上与底部4个格子的中点对齐（横向4等分，与 input_device 的
// 4x4宫格映射一致），纵向与 ui_draw_footer 一致（页脚高度跟随当前字体行高，文本在页脚带内
// 垂直居中，行高差异由 gfx_font_draw_text_centered 的行框居中语义吸收）。
// 传 NULL 或空字符串表示对应按键无功能（留空）。
void ui_draw_footer_softkeys(
    Key_Event *key_event, Global_State *global_state,
    wchar_t *text_key_left, wchar_t *text_key_0, wchar_t *text_key_right, wchar_t *text_key_enter
) {
    // 页脚高度跟随当前字体行高（行高 + 1px 边距）
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    const int footer_height = line_height + 1;
    // 触屏软键盘与16键虚拟键盘显示时，页脚上移为键盘让出空间（与 ui_draw_footer 一致）
    const int32_t footer_bottom = global_state->gfx->height - ui_softkbd_height() - ui_grid16kbd_height();
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_draw_rectangle(global_state->gfx, 0, footer_bottom - footer_height, global_state->gfx->width, footer_height, S_UI_COLOR_FOOTER_BG[0], S_UI_COLOR_FOOTER_BG[1], S_UI_COLOR_FOOTER_BG[2], 1);
        S_UI_COLOR_FOOTER_TEXT[0] = 90;
        S_UI_COLOR_FOOTER_TEXT[1] = 98;
        S_UI_COLOR_FOOTER_TEXT[2] = 106;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_draw_rectangle(global_state->gfx, 0, footer_bottom - footer_height, global_state->gfx->width, footer_height, 15, 16, 17, 1);
        S_UI_COLOR_FOOTER_TEXT[0] = 188;
        S_UI_COLOR_FOOTER_TEXT[1] = 188;
        S_UI_COLOR_FOOTER_TEXT[2] = 188;
    }
    const wchar_t *texts[4] = {text_key_left, text_key_0, text_key_right, text_key_enter};
    int32_t cell_width = global_state->gfx->width / 4;
    int32_t cy = footer_bottom - footer_height + footer_height / 2;
    for (int32_t i = 0; i < 4; i++) {
        if (texts[i] != NULL && texts[i][0] != L'\0') {
            gfx_font_draw_text_centered(global_state->gfx, font_id, (wchar_t *)texts[i],
                cell_width * i + cell_width / 2, cy,
                S_UI_COLOR_FOOTER_TEXT[0], S_UI_COLOR_FOOTER_TEXT[1], S_UI_COLOR_FOOTER_TEXT[2], 1);
        }
    }
}








void ui_widget_textarea_init(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state,
    uint32_t max_len
) {
    // 文本区位于页眉与页脚之间：页眉高度为 1.5 倍字体行高（与菜单控件一致），页脚为行高 + 1px
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    int32_t header_height = ui_std_header_height(global_state->ui_font);
    textarea_state->state = 0;
    textarea_state->x = 0;
    textarea_state->y = header_height;
    textarea_state->width = global_state->gfx->width;
    textarea_state->height = global_state->gfx->height - ui_softkbd_height() - ui_grid16kbd_height() - header_height - (line_height + 1); // 减去header和footer，并为触屏软键盘与16键虚拟键盘让出空间
    textarea_state->length = 0;
    textarea_state->line_num = 0;
    textarea_state->view_lines = 0;
    textarea_state->view_start_pos = 0;
    textarea_state->view_end_pos = 0;
    textarea_state->current_line = 0;
    textarea_state->is_show_scroll_bar = 1;
    textarea_state->is_modified = 1;
    // 像素级连续滚动与触屏交互状态（见 AGENTS.md 第九节）
    textarea_state->scroll_sub_offset = 0;
    textarea_state->touch_active = 0;
    textarea_state->touch_is_dragging = 0;
    textarea_state->touch_start_x = 0;
    textarea_state->touch_start_y = 0;
    textarea_state->touch_anchor_scroll_px = 0;
    textarea_state->touch_track_scroll = 0;
    textarea_state->touch_track_ts = 0;
    textarea_state->touch_track_vel = 0.0f;
    textarea_state->fling_velocity = 0.0f;
    textarea_state->fling_scroll_px = 0.0f;
    textarea_state->fling_last_timestamp = 0;
    // 重新init时先释放旧缓冲区，避免覆盖式分配造成泄漏（初次init时为NULL，free(NULL)安全）
    if (textarea_state->text)      free(textarea_state->text);
    if (textarea_state->style)     free(textarea_state->style);
    if (textarea_state->break_pos) free(textarea_state->break_pos);
    textarea_state->text = (wchar_t*)platform_calloc(max_len, sizeof(wchar_t));
    textarea_state->style = (uint32_t*)platform_calloc(max_len, sizeof(uint32_t));
    textarea_state->break_pos = (int32_t*)platform_calloc(max_len, sizeof(int32_t));
}

void ui_widget_textarea_set(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state,
    wchar_t *text, int32_t current_line, int32_t is_show_scroll_bar) {
    textarea_state->is_modified = 1;
    textarea_state->current_line = current_line;
    textarea_state->scroll_sub_offset = 0; // 整行路径：吸附回整行（像素滚动不变量，见 AGENTS.md 第九节）
    textarea_state->is_show_scroll_bar = is_show_scroll_bar;
    // text缓冲区容量为 UI_STR_BUF_MAX_LENGTH 个 wchar_t（含结尾 L'\0'），截断拷贝防堆溢出
    wcsncpy(textarea_state->text, text, UI_STR_BUF_MAX_LENGTH - 1);
    textarea_state->text[UI_STR_BUF_MAX_LENGTH - 1] = L'\0';
}

void ui_widget_textarea_draw(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state) {
    // “返回”标签的生命周期与页眉一致：仅在内容重排（is_modified，即宿主整体重绘、页眉同步重绘）
    // 时随页眉绘制一次；滚动重绘（事件处理器以 is_modified==0 包裹的本函数调用）不触碰页眉带，
    // 避免抗锯齿文字边缘被反复混合而模糊（页眉标签现作为页眉右侧文本经 ui_draw_header_full 固有绘制）
    int32_t redraw_chrome = textarea_state->is_modified;
    if (textarea_state->is_modified) {
        typeset_line_breaks(key_event, global_state, textarea_state);
    }
    typeset_view_range(textarea_state, gfx_font_line_height(global_state->ui_font));

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

    ui_draw_text_block(key_event, global_state, textarea_state, global_state->ui_font);

    if (textarea_state->is_show_scroll_bar) {
        // 像素单位（滚动位置/内容高度/可视高度，随触点平滑移动；与菜单控件同一用法）
        int32_t line_height = gfx_font_line_height(global_state->ui_font);
        ui_draw_scroll_bar(
            key_event, global_state,
            textarea_state->current_line * line_height + textarea_state->scroll_sub_offset,
            textarea_state->line_num * line_height, textarea_state->height,
            textarea_state->x, textarea_state->y, textarea_state->width, textarea_state->height);
    }

    // “返回”软按钮：仅随内容重排（页眉重绘）时绘制，滚动重绘不重复混合（见函数头注释）。
    // 即页眉右侧文本 "返回 "（页眉固有侧文本机制，见 ui_draw_header_side_text）
    if (redraw_chrome) {
        ui_draw_header_side_text(key_event, global_state, ui_std_header_height(global_state->ui_font), NULL, L"返回 ");
    }

    if (global_state->is_full_refresh) {
        gfx_refresh(global_state->gfx);
    }
}

// 通用的文本框卷行事件处理

// 页眉左/右侧文本：12px 抗锯齿、页眉带内垂直居中、页眉文字同色；
// 各自区域先按页眉底色回填（自清洁，可独立按需更新），NULL 或空串表示该侧不绘制
void ui_draw_header_side_text(Key_Event *key_event, Global_State *global_state, int32_t header_height,
    wchar_t *left_text, wchar_t *right_text
) {
    (void)key_event;
    const uint32_t font_id = GFX_FONT_ALPHA_12;
    int32_t label_y = (header_height - gfx_font_line_height(font_id)) / 2;
    if (label_y < 0) label_y = 0;
    int32_t w = (int32_t)global_state->gfx->width;
    if (left_text != NULL && left_text[0] != L'\0') {
        int32_t lw = gfx_font_measure_text(font_id, left_text);
        ui_header_bg_fill(global_state, 0, lw, header_height);
        gfx_font_draw_text(global_state->gfx, font_id, left_text, 0, label_y,
            S_UI_COLOR_HEADER_TEXT[0], S_UI_COLOR_HEADER_TEXT[1], S_UI_COLOR_HEADER_TEXT[2], 1);
    }
    if (right_text != NULL && right_text[0] != L'\0') {
        int32_t lw = gfx_font_measure_text(font_id, right_text);
        ui_header_bg_fill(global_state, w - lw, w, header_height);
        gfx_font_draw_text(global_state->gfx, font_id, right_text, w - lw, label_y,
            S_UI_COLOR_HEADER_TEXT[0], S_UI_COLOR_HEADER_TEXT[1], S_UI_COLOR_HEADER_TEXT[2], 1);
    }
}

// 页眉完整绘制：底色 + 标题 + 可选左/右侧文本（标签作为页眉固有部分，与页眉同时机绘制）
void ui_draw_header_full(Key_Event *key_event, Global_State *global_state, wchar_t *title, int32_t is_center,
    int32_t header_height, wchar_t *left_text, wchar_t *right_text
) {
    ui_draw_header_ex(key_event, global_state, title, is_center, header_height);
    ui_draw_header_side_text(key_event, global_state, header_height, left_text, right_text);
}

// 文本框触屏手势机（像素级连续滚动；认 touch_edge 边沿事件，电平兜底；与菜单控件同范式）
int32_t ui_widget_textarea_touch_handler(Key_Event *ke, Global_State *gs, Widget_Textarea_State *ts) {
    int32_t line_height = gfx_font_line_height(gs->ui_font);
    int32_t max_scroll_px = ts->line_num * line_height - ts->height;
    if (max_scroll_px < 0) max_scroll_px = 0;

    // 松手惯性滚动（fling）：动画激活期间每帧推进一次；任意新触摸或按键边沿立即终止
    if (ts->fling_velocity != 0.0f) {
        if (ke->is_touching || ke->touch_edge != 0 || ke->key_edge != 0) {
            ts->fling_velocity = 0.0f;
        }
        else {
            uint64_t now = get_timestamp_in_ms();
            float dt_s = (float)(now - ts->fling_last_timestamp) / 1000.0f;
            ts->fling_last_timestamp = now;
            if (dt_s > 0.05f) dt_s = 0.05f;
            if (dt_s > 0.0f) {
                float scroll = ts->fling_scroll_px + ts->fling_velocity * dt_s;
                float decel = 2000.0f * dt_s; // 线性减速度 ~2000px/s²（与菜单一致）
                if (ts->fling_velocity > 0.0f) {
                    ts->fling_velocity -= decel;
                    if (ts->fling_velocity < 0.0f) ts->fling_velocity = 0.0f;
                }
                else {
                    ts->fling_velocity += decel;
                    if (ts->fling_velocity > 0.0f) ts->fling_velocity = 0.0f;
                }
                if (scroll <= 0.0f) { scroll = 0.0f; ts->fling_velocity = 0.0f; }
                if (scroll >= (float)max_scroll_px) { scroll = (float)max_scroll_px; ts->fling_velocity = 0.0f; }
                ts->fling_scroll_px = scroll;
                int32_t new_scroll_px = (int32_t)scroll;
                int32_t new_line = new_scroll_px / line_height;
                int32_t new_sub = new_scroll_px % line_height;
                if (new_line != ts->current_line || new_sub != ts->scroll_sub_offset) {
                    ts->current_line = new_line;
                    ts->scroll_sub_offset = new_sub;
                    return 1;
                }
            }
        }
    }

    // 触摸序列开始：DOWN 边沿（可靠事件）或电平兜底；起点须在文本区内才激活
    if (((ke->touch_edge & TOUCH_EDGE_DOWN) || ke->is_touching) && !ts->touch_active) {
        int32_t sx = (ke->touch_edge & TOUCH_EDGE_DOWN) ? ke->touch_down_x : ke->touch_x;
        int32_t sy = (ke->touch_edge & TOUCH_EDGE_DOWN) ? ke->touch_down_y : ke->touch_y;
        if (sx < ts->x || sx >= ts->x + ts->width || sy < ts->y || sy >= ts->y + ts->height) {
            return 0; // 起点在文本区外：不接管，交由调用方（如热点按钮）处理
        }
        ts->touch_active = 1;
        ts->touch_is_dragging = 0;
        ts->touch_start_x = sx;
        ts->touch_start_y = sy;
        ts->touch_anchor_scroll_px = ts->current_line * line_height + ts->scroll_sub_offset;
        ts->touch_track_scroll = ts->touch_anchor_scroll_px;
        ts->touch_track_ts = get_timestamp_in_ms();
        ts->touch_track_vel = 0.0f;
        return 1;
    }
    // 拖动跟踪（电平驱动，逐帧最新坐标）：1:1 跟手，钳制不回绕
    else if (ke->is_touching && ts->touch_active) {
        int32_t dy = ke->touch_y - ts->touch_start_y; // >0：手指下滑
        if (!ts->touch_is_dragging && (dy > line_height / 2 || dy < -(line_height / 2))) {
            ts->touch_is_dragging = 1;
        }
        if (ts->touch_is_dragging && max_scroll_px > 0) {
            int32_t new_scroll_px = ts->touch_anchor_scroll_px - dy;
            if (new_scroll_px < 0) new_scroll_px = 0;
            if (new_scroll_px > max_scroll_px) new_scroll_px = max_scroll_px;
            int32_t new_line = new_scroll_px / line_height;
            int32_t new_sub = new_scroll_px % line_height;
            int32_t changed = (new_line != ts->current_line || new_sub != ts->scroll_sub_offset);
            ts->current_line = new_line;
            ts->scroll_sub_offset = new_sub;
            // 拖动速度采样：指数平滑（0.6/0.4），供松手惯性初速度估算
            uint64_t now = get_timestamp_in_ms();
            uint32_t dt_ms = (uint32_t)(now - ts->touch_track_ts);
            if (dt_ms > 0) {
                float v_inst = (float)(new_scroll_px - ts->touch_track_scroll) * 1000.0f / (float)dt_ms;
                ts->touch_track_vel = ts->touch_track_vel * 0.6f + v_inst * 0.4f;
                ts->touch_track_scroll = new_scroll_px;
                ts->touch_track_ts = now;
            }
            if (changed) return 1;
        }
        return 1; // 序列进行中（未构成拖动也视为活动，防止下游误消费同一次触摸）
    }
    // 触摸序列结束：UP 边沿（可靠事件）或电平兜底；构成拖动则启动惯性，否则上报点按
    else if (((ke->touch_edge & TOUCH_EDGE_UP) || !ke->is_touching) && ts->touch_active) {
        ts->touch_active = 0;
        if (ts->touch_is_dragging) {
            // 松手前手指已停顿（>100ms 无新采样）则速度作废，不启动惯性
            uint64_t now = get_timestamp_in_ms();
            float v0 = (now - ts->touch_track_ts <= 100) ? ts->touch_track_vel : 0.0f;
            if (v0 > 4000.0f) v0 = 4000.0f;    // 触点抖动限速
            if (v0 < -4000.0f) v0 = -4000.0f;
            if (v0 > 50.0f || v0 < -50.0f) {   // 低于阈值（50px/s）不动画
                ts->fling_velocity = v0;
                ts->fling_scroll_px = (float)(ts->current_line * line_height + ts->scroll_sub_offset);
                ts->fling_last_timestamp = now;
            }
            return 1;
        }
        return 2; // 点按
    }
    return 0;
}

// 点按命中测试（实现见 ui.h 声明注释）
int32_t ui_widget_textarea_char_index_at(Key_Event *ke, Global_State *gs, Widget_Textarea_State *ts,
    int32_t px, int32_t py
) {
    (void)ke;
    if (px < ts->x || px >= ts->x + ts->width || py < ts->y || py >= ts->y + ts->height) return -1;
    if (ts->line_num <= 0 || ts->length <= 0) return 0;
    uint32_t font_id = gs->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    // 由像素坐标反推行号：scroll_px = current_line*line_height + sub
    int32_t scroll_px = ts->current_line * line_height + ts->scroll_sub_offset;
    int32_t line = (py - ts->y + scroll_px) / line_height;
    if (line < 0) line = 0;
    if (line >= ts->line_num) line = ts->line_num - 1;
    int32_t line_start = ts->break_pos[line];
    int32_t line_end = (line + 1 <= ts->line_num - 1) ? (ts->break_pos[line + 1] - 1) : (ts->length - 1);
    // 行内逐字符累加实际宽度（与 ui_draw_text_block 的折行/绘制逻辑一致）
    int32_t x = ts->x;
    for (int32_t i = line_start; i <= line_end && i < ts->length; i++) {
        if (ts->style[i] & 0x80000000) continue; // 格式控制标记字符不占位
        wchar_t ch = ts->text[i];
        if (ch == L'\0') break;
        if (ch == L'\n') return i; // 显式换行：槽位在换行符之前
        int32_t cw = gfx_font_char_advance(font_id, (uint32_t)ch);
        if (px < x + cw / 2) return i;     // 落在字符左半：光标在该字符左侧 → 槽位 i
        x += cw;
        if (px < x) return i + 1;          // 落在字符右半：槽位 i+1
    }
    // 行尾之后：槽位为行末（钳制到文本长度）
    int32_t slot = line_end + 1;
    if (slot > ts->length) slot = ts->length;
    return slot;
}

int32_t ui_widget_textarea_event_handler(
    Key_Event *ke, Global_State *gs, Widget_Textarea_State *ts,
    int32_t prev_focus_state, int32_t current_focus_state
) {
    // “返回”软按钮：页眉最右侧 1/4 热点（松手沿 + 按下点坐标；页眉带在文本区之外，手势机不会抢占）
    if ((ke->touch_edge & TOUCH_EDGE_UP)
        && ke->touch_down_y >= 0 && ke->touch_down_y < ts->y
        && ke->touch_down_x >= UI_BACK_HOTSPOT_X0((int32_t)gs->gfx->width)) {
        return prev_focus_state;
    }

    // 触屏：滑动=像素级连续滚动（含松手惯性）；点按在显示控件无动作（同样消费，防止下游误用）
    int32_t touch_result = ui_widget_textarea_touch_handler(ke, gs, ts);
    if (touch_result != 0) {
        if (touch_result == 1) {
            ts->is_modified = 0;
            ui_widget_textarea_draw(ke, gs, ts);
            ts->is_modified = 1;
        }
        return current_focus_state;
    }

    // 仅响应硬按键事件（软按键——触屏宫格映射/软键盘派生——一律不响应；触屏滚动/返回见上方）
    if (ke->key_code != NANO_KEY_IDLE && ke->is_soft_key != 0) {
        return current_focus_state;
    }

    // 短按A键：回到上一个焦点
    if (ke->key_edge == -1 && ke->key_code == NANO_KEY_esc) {
        return prev_focus_state;
    }

    // 长+短按*键：推理结果向上翻一行。如果翻到顶，则回到最后一行。
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == NANO_KEY_left) {
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
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == NANO_KEY_right) {
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







void ui_widget_input_init(
    Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state,
    wchar_t *title_text
) {
    Widget_Textarea_State *ta = &(input_state->textarea);

    ui_widget_textarea_init(key_event, global_state, ta, UI_STR_BUF_MAX_LENGTH);

    // 进入控件时收起16键虚拟键盘（与 grid16 旧模式标志等价的清理；键盘为控件固有功能，随控件进入而复位）
    ui_grid16kbd_hide();

    // 文本区位于页眉与页脚之间：页眉高度为 1.5 倍字体行高（与菜单控件一致），页脚为行高 + 1px
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    int32_t header_height = ui_std_header_height(global_state->ui_font);
    ta->state = 0;
    ta->x = 0;
    ta->y = header_height;
    ta->width = global_state->gfx->width;
    ta->height = global_state->gfx->height - ui_softkbd_height() - ui_grid16kbd_height() - header_height - (line_height + 1); // 减去header和footer NOTE 详见结构体定义处的说明；并为触屏软键盘与16键虚拟键盘让出空间
    ta->length = 0;
    ta->is_show_scroll_bar = 1;

    input_state->cursor_pos = -1;
    input_state->desired_x = -1;
    input_state->ime_mode_flag = IME_MODE_HANZI;
    input_state->pinyin_keys = 0;
    input_state->candidate_num = 0;
    input_state->candidate_page_num = 0;
    input_state->current_page = 0;
    input_state->alphabet_click_timestamp = 0;
    input_state->alphabet_is_counting_down = 0;
    input_state->alphabet_current_key = 255;
    input_state->alphabet_index = 0;
    input_state->title_text = title_text;
    // 触屏交互（见 AGENTS.md 第九节）
    input_state->softkey_swallow_until = 0;
    input_state->drawn_cursor_pos = -2; // 强制首次绘制时光标跟随

    // 初始化各个数组
    memset(input_state->candidates, 0, sizeof(input_state->candidates));
    memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));

    ui_draw_input_buffer(key_event, global_state, input_state);
}

void ui_widget_input_refresh(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    input_state->cursor_pos = input_state->textarea.length - 1;
    input_state->desired_x = -1;
    ui_draw_input_buffer(key_event, global_state, input_state);
}

// 切换触屏软键盘显隐，并重新布局为键盘让出/恢复空间（文本输入控件固有功能：
// 供 Ctrl+0 组合键与页脚 [键盘] 热点调用）。与16键虚拟键盘互斥：呼出软键盘前收起16键键盘。
void ui_widget_input_toggle_softkbd(Key_Event *key_event, Global_State *global_state) {
    if (ui_softkbd_is_visible()) {
        ui_softkbd_hide();
    }
    else {
        if (ui_grid16kbd_is_visible()) ui_grid16kbd_hide(); // 与16键虚拟键盘互斥
        ui_softkbd_show();
    }
    ui_pinyin_ime_reset(); // 键盘显隐切换时，放弃进行中的拼音组字
    // 重新布局：文本区高度扣除软键盘与16键键盘高度（均隐藏时两者高度为0，布局复原）
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    int32_t header_height = ui_std_header_height(global_state->ui_font);
    global_state->w_input_main->textarea.height = global_state->gfx->height - ui_softkbd_height() - ui_grid16kbd_height() - header_height - (line_height + 1);
    global_state->w_input_main->textarea.is_modified = 1;
    ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
}

// 切换16键虚拟键盘显隐，并重新布局为键盘让出/恢复空间（文本输入控件固有功能：
// 供页脚 [16键] 热点调用）。与触屏软键盘互斥：呼出16键键盘前收起软键盘。
void ui_widget_input_toggle_grid16(Key_Event *key_event, Global_State *global_state) {
    if (ui_grid16kbd_is_visible()) {
        ui_grid16kbd_hide();
    }
    else {
        if (ui_softkbd_is_visible()) ui_softkbd_hide(); // 与触屏软键盘互斥
        ui_grid16kbd_show();
    }
    ui_pinyin_ime_reset(); // 键盘显隐切换时，放弃进行中的拼音组字
    // 重新布局：文本区高度扣除软键盘与16键键盘高度（均隐藏时两者高度为0，布局复原）
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    int32_t header_height = ui_std_header_height(global_state->ui_font);
    global_state->w_input_main->textarea.height = global_state->gfx->height - ui_softkbd_height() - ui_grid16kbd_height() - header_height - (line_height + 1);
    global_state->w_input_main->textarea.is_modified = 1;
    ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
}

// 绘制文本输入操作说明
static void ui_draw_input_help(Key_Event *key_event, Global_State *global_state) {
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    int32_t cx = global_state->gfx->width / 2;
    int32_t cy = 5 + 6;
    // 触屏软键盘与16键虚拟键盘显示时，帮助页为其让出空间
    gfx_draw_rectangle(global_state->gfx, 3, 3, global_state->gfx->width - 6, global_state->gfx->height - ui_softkbd_height() - ui_grid16kbd_height() - 6, S_UI_COLOR_IME_HELP_BG[0], S_UI_COLOR_IME_HELP_BG[1], S_UI_COLOR_IME_HELP_BG[2], 3);
    gfx_font_draw_text_centered(global_state->gfx, font_id, L"文本输入操作说明", cx, cy, 0, 0, 222, 1);
    cy += line_height;
    gfx_font_draw_text_centered(global_state->gfx, font_id, L"A-退格/返回  B-切换汉英数",   cx, cy, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    cy += line_height;
    gfx_font_draw_text_centered(global_state->gfx, font_id, L"C-第二功能  D-输入/提交",    cx, cy, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    cy += line_height;
    gfx_font_draw_text_centered(global_state->gfx, font_id, L"Ctrl+1选择符号 左右键移动光标",  cx, cy, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    cy += line_height;
    gfx_font_draw_text_centered(global_state->gfx, font_id, L"按住D语音输入 Ctrl+D 换行",    cx, cy, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    cy += line_height;
    gfx_font_draw_text_centered(global_state->gfx, font_id, L"Ctrl+2 切换思考模式",          cx, cy, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);
    cy += line_height;
    gfx_font_draw_text_centered(global_state->gfx, font_id, L"Ctrl+A 放弃输入并返回",        cx, cy, S_UI_COLOR_IME_HELP_TEXT[0], S_UI_COLOR_IME_HELP_TEXT[1], S_UI_COLOR_IME_HELP_TEXT[2], 1);

    // 触屏软键盘与16键虚拟键盘：帮助页为其让出空间，同帧重绘
    //（Ctrl 状态可能刚被消费型组合键复位，键盘文案/高亮需同步刷新）
    if (ui_softkbd_height() > 0) {
        ui_softkbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
    }
    if (ui_grid16kbd_height() > 0) {
        ui_grid16kbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
    }

    gfx_refresh(global_state->gfx);
}


// 离开文本输入控件时的清理（控件两个退出分支 return prev/next_focus_state 处调用）：
// 收起软键盘与16键虚拟键盘并恢复布局（两者均为控件固有功能，随控件退出而关闭）
static void ui_widget_input_on_leave(Global_State *global_state, Widget_Input_State *input_state) {
    if (ui_softkbd_is_visible() || ui_grid16kbd_is_visible()) {
        ui_softkbd_hide();
        ui_grid16kbd_hide();
        ui_pinyin_ime_reset();
        int32_t line_height = gfx_font_line_height(global_state->ui_font);
        input_state->textarea.height = global_state->gfx->height - ui_std_header_height(global_state->ui_font) - (line_height + 1);
        input_state->textarea.is_modified = 1;
    }
}

// 页眉“返回”按钮的善后清理（与 A 键退出路径的区别：【不修改文本输入缓冲区】）：
// 输入法状态全部复位为初始状态（控件状态机、九键拼音/符号候选、英文字母倒计时、
// 汉英数输入模式、全键盘拼音组字、全局 Ctrl 状态），并收起软键盘/16键键盘、
// 恢复文本区与页脚布局（页脚内容随返回后下一状态的整体重绘恢复）——
// 保证全局单例 w_input_main 经返回按钮退出后，下次重入仍是干净的初始状态
static void ui_widget_input_back_cleanup(Global_State *global_state, Widget_Input_State *input_state) {
    input_state->state = 0;                     // 控件状态机（组字/选字/选符/帮助）回初始
    input_state->pinyin_keys = 0;               // 九键拼音按键序列
    input_state->candidate_num = 0;             // 九键拼音/符号候选
    input_state->candidate_page_num = 0;
    input_state->current_page = 0;
    memset(input_state->candidates, 0, sizeof(input_state->candidates));
    memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));
    input_state->alphabet_is_counting_down = 0; // 英文字母输入的倒计时
    input_state->alphabet_current_key = 255;
    input_state->alphabet_index = 0;
    input_state->ime_mode_flag = IME_MODE_HANZI; // 汉英数输入模式回初始
    ui_pinyin_ime_reset();                       // 全键盘拼音组字
    if (global_state->is_ctrl_enabled == 1) {
        global_state->is_ctrl_enabled = 0;       // 全局 Ctrl 状态
    }
    ui_widget_input_on_leave(global_state, input_state);
}

// 光标上下移动：在视觉行（'\n'硬换行 + 按宽度软折行）之间移动，参照 main.cpp-ref 的
//   move_up/move_down 逻辑（visual_pos → 保持目标列 → index_from_visual 反查落点）。
//   本项目按像素宽度折行（比例字体），故“目标列”以视觉x偏移（desired_x）保持；
//   折行判定与 ui_draw_input_cursor 完全一致（宽度超限或'\n'换行，不计颜色标签）。
//   direction: -1 上移一个视觉行，+1 下移一个视觉行。已到首/末视觉行则不动作。
static void ui_widget_input_move_cursor_vertical(Global_State *global_state, Widget_Input_State *input_state, int32_t direction) {
    Widget_Textarea_State *ta = &input_state->textarea;
    uint32_t font_id = global_state->ui_font;
    int32_t len = ta->length;
    if (len <= 0) return;

    // 光标槽位：位于 text[s-1] 与 text[s] 之间（s == cursor_pos + 1，取值 0..len）
    int32_t s = input_state->cursor_pos + 1;

    // 第1遍：确定光标所在视觉行的起点槽位 cur_start、下一视觉行首字符下标 next_start、
    //        上一视觉行起点槽位 prev_start，以及光标的视觉x（槽位在换行判定之前属于当前行）
    int32_t prev_start = -1;
    int32_t cur_start = 0;
    int32_t next_start = len;
    int32_t target_x = 0;
    {
        int32_t line_start = 0;
        int32_t line_x = 0;
        for (int32_t i = 0; i <= len; i++) {
            if (i == s) target_x = line_x;
            int32_t is_end = (i >= len);
            wchar_t ch = is_end ? L'\n' : ta->text[i];
            int32_t char_width = (ch == L'\n') ? 0 : gfx_font_char_advance(font_id, (uint32_t)ch);
            if (is_end || line_x + char_width >= ta->x + ta->width || ch == L'\n') {
                // 视觉行结束：本行槽位范围为 [line_start, i]，下一视觉行从 char（i 或 i+1）开始
                if (s <= i) {
                    cur_start = line_start;
                    next_start = (ch == L'\n') ? (i + 1) : i;
                    break;
                }
                prev_start = line_start;
                line_start = (ch == L'\n') ? (i + 1) : i;
                line_x = 0;
            }
            line_x += char_width;
        }
    }

    // 上下移动保持目标x（参照 main.cpp-ref 的 desired_col）
    if (input_state->desired_x < 0) {
        input_state->desired_x = target_x;
    }
    else {
        target_x = input_state->desired_x;
    }

    // 确定目标视觉行的槽位范围 [dst_start, dst_end]
    int32_t dst_start, dst_end;
    if (direction < 0) {
        if (cur_start <= 0) return; // 已在首个视觉行
        dst_start = prev_start;
        // 上一视觉行的末槽位：若当前行起于'\n'之后，则上一行末槽位在'\n'之前
        dst_end = (ta->text[cur_start - 1] == L'\n') ? (cur_start - 1) : cur_start;
    }
    else {
        if (next_start >= len) {
            // 文本以'\n'结尾时，其下还有一个空视觉行（仅含槽位len）
            if (ta->text[len - 1] == L'\n' && s < len) {
                input_state->cursor_pos = len - 1;
            }
            return;
        }
        dst_start = next_start;
        // 求下一视觉行的末槽位
        dst_end = len;
        int32_t line_x = 0;
        for (int32_t i = dst_start; i < len; i++) {
            wchar_t ch = ta->text[i];
            int32_t char_width = (ch == L'\n') ? 0 : gfx_font_char_advance(font_id, (uint32_t)ch);
            if (line_x + char_width >= ta->x + ta->width || ch == L'\n') {
                dst_end = i;
                break;
            }
            line_x += char_width;
        }
    }

    // 目标行若以软折行接续上一行（首字符是宽度折行而来），槽位 dst_start 在视觉上属于
    // 上一行末尾（与 ui_draw_input_cursor 的归属一致），最小落点须取其后一个槽位，
    // 否则光标会落到一个归属相邻行的槽位上，导致连续上下移动时振荡/跳行
    // （与 main.cpp-ref 的 index_from_visual 跳过折行边界的行为一致）。
    int32_t dst_min = dst_start;
    if (dst_start > 0 && dst_start < len && ta->text[dst_start - 1] != L'\n') {
        dst_min = dst_start + 1;
    }

    // 在目标视觉行 [dst_start, dst_end] 内，取与 target_x 最接近的槽位（不小于 dst_min）
    int32_t new_s = dst_start;
    int32_t x = 0;
    for (int32_t j = dst_start; j < dst_end; j++) {
        wchar_t ch = ta->text[j];
        int32_t char_width = (ch == L'\n') ? 0 : gfx_font_char_advance(font_id, (uint32_t)ch);
        if (x + char_width > target_x) {
            // 当前槽位(x)与下一槽位(x+char_width)中取与目标更近者
            if ((x + char_width - target_x) <= (target_x - x)) {
                new_s = j + 1;
            }
            break;
        }
        x += char_width;
        new_s = j + 1;
    }
    if (new_s < dst_min) new_s = dst_min;

    input_state->cursor_pos = new_s - 1;
}

// ===============================================================================
// 垂直滑动手势跟踪器（通用解释器，见 ui.h）
// ===============================================================================

void ui_swipe_tracker_init(UI_Swipe_Tracker *tracker) {
    tracker->active = 0;
    tracker->start_y = 0;
    tracker->min_y = 0;
    tracker->max_y = 0;
}

int8_t ui_swipe_tracker_feed(UI_Swipe_Tracker *tracker, int32_t is_touching, int32_t touch_y, int32_t confirm_px) {
    if (is_touching) {
        if (!tracker->active) {
            tracker->active = 1;
            tracker->start_y = touch_y;
            tracker->min_y = touch_y;
            tracker->max_y = touch_y;
        }
        else {
            if (touch_y < tracker->min_y) tracker->min_y = touch_y;
            if (touch_y > tracker->max_y) tracker->max_y = touch_y;
        }
        return NANO_TOUCH_GESTURE_NONE;
    }
    // 松手确认：垂直位移跨越阈值即按方向返回手势
    if (!tracker->active) return NANO_TOUCH_GESTURE_NONE;
    tracker->active = 0;
    if (tracker->start_y - tracker->min_y > confirm_px) return NANO_TOUCH_GESTURE_SWIPE_UP;
    if (tracker->max_y - tracker->start_y > confirm_px) return NANO_TOUCH_GESTURE_SWIPE_DOWN;
    return NANO_TOUCH_GESTURE_NONE;
}

int32_t ui_swipe_tracker_displacement(const UI_Swipe_Tracker *tracker, int8_t direction) {
    if (!tracker->active) return 0;
    if (direction == NANO_TOUCH_GESTURE_SWIPE_UP)   return tracker->start_y - tracker->min_y;
    if (direction == NANO_TOUCH_GESTURE_SWIPE_DOWN) return tracker->max_y - tracker->start_y;
    return 0;
}

// 软键盘手势的跟踪器已随像素级滚动改造退役（见 AGENTS.md 第九节）：
// 原“上滑呼出/下滑收起软键盘”手势由页脚热点虚拟按钮替代

int32_t ui_widget_input_event_handler(
    Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state,
    int32_t prev_focus_state, int32_t current_focus_state, int32_t next_focus_state
) {

    // 触屏交互（像素级连续滚动改造，见 AGENTS.md 第九节）：
    //  - 页眉“返回”软按钮（页眉带右 1/4，认松手沿+按下点坐标）：放弃输入并返回上一状态；
    //  - 热点虚拟按钮（页脚带左/右 1/4，认松手沿+按下点坐标）：
    //    [键盘] 呼出/收起软键盘（等价 Ctrl+0）；[16键] 呼出/收起16键虚拟键盘。
    //    热点动作后 150ms 内吞掉同一次触摸经宫格映射产生的残留软按键（范式同 ui_calendar）。
    //  - 16键虚拟键盘可见（ui_grid16kbd）：触屏点按键盘按钮=九键软按键（键码由事件层
    //    ui_grid16kbd_poll 产生，见 ui_app.c get_input_event）；
    //  - 滑动=像素滚动、点按=光标定位（委托 textarea 手势机）：任意键盘显隐状态下均可用——
    //    手势机仅当触摸序列起点在文本区内才激活，键盘区域落在文本区之外不受影响。
    //    宿主状态的全屏宫格映射已在事件层抑制（ui_app_state_hosts_input_widget），
    //    宫格软按键门控仅为兜底，软键盘键码（is_softkbd）与硬按键照常。
    {
        // 页眉“返回”软按钮：页眉带最右侧 1/4 热区（松手沿 + 按下点坐标，与文本显示控件
        // ui_widget_textarea_event_handler 同范式）：做完整善后清理（输入法状态/键盘/布局复位，
        // 但不修改文本输入缓冲区，见 ui_widget_input_back_cleanup）后返回上一状态；
        // 页眉带在文本区之外，手势机不会抢占
        if ((key_event->touch_edge & TOUCH_EDGE_UP)
            && key_event->touch_down_y >= 0 && key_event->touch_down_y < ui_std_header_height(global_state->ui_font)
            && key_event->touch_down_x >= UI_BACK_HOTSPOT_X0((int32_t)global_state->gfx->width)) {
            ui_widget_input_back_cleanup(global_state, input_state);
            return prev_focus_state;
        }
        // 热点虚拟按钮（页脚带 = 底部 ui_draw_footer 区域，随软键盘/16键键盘显隐上移；
        // 页脚带被占用时热点停用防误触：全键盘拼音组词、16键输入法组字/选字/选符 state 1/2/3，
        // 或英文字母指示器显示期间（倒计时进行中））
        if ((key_event->touch_edge & TOUCH_EDGE_UP)
            && !(ui_softkbd_is_visible() && ui_pinyin_ime_is_composing())
            && input_state->state != 1 && input_state->state != 2 && input_state->state != 3
            && !(input_state->ime_mode_flag == IME_MODE_ALPHABET && input_state->alphabet_is_counting_down == 1)) {
            int32_t footer_height = gfx_font_line_height(global_state->ui_font) + 1;
            int32_t footer_top = (int32_t)global_state->gfx->height - (int32_t)ui_softkbd_height() - (int32_t)ui_grid16kbd_height() - footer_height;
            int32_t tx = key_event->touch_down_x;
            int32_t ty = key_event->touch_down_y;
            if (ty >= footer_top && ty < footer_top + footer_height) {
                if (tx < (int32_t)global_state->gfx->width / 4) {          // [键盘]
                    ui_widget_input_toggle_softkbd(key_event, global_state);
                    input_state->softkey_swallow_until = get_timestamp_in_ms() + 150;
                    return current_focus_state;
                }
                else if (tx >= (int32_t)global_state->gfx->width * 3 / 4) { // [16键]
                    ui_widget_input_toggle_grid16(key_event, global_state);
                    input_state->softkey_swallow_until = get_timestamp_in_ms() + 150;
                    return current_focus_state;
                }
            }
        }
        // 宫格软按键门控（兜底；宿主状态的全屏宫格映射已在事件层抑制）：
        // 16键键盘隐藏时全丢（防止点按定位光标时打入杂字）；
        // 键盘可见时热点动作后的短暂窗口内也丢（吞残留）
        if (key_event->key_code != NANO_KEY_IDLE && key_event->is_soft_key && !key_event->is_softkbd) {
            if (!ui_grid16kbd_is_visible() || get_timestamp_in_ms() < input_state->softkey_swallow_until) {
                key_event->key_code = NANO_KEY_IDLE;
                key_event->key_edge = 0;
            }
        }
        // 滑动=像素滚动 / 点按=光标定位（委托 textarea 手势机）：任意键盘显隐状态下均运行；
        // 手势机仅当触摸序列起点在文本区内才激活，键盘区域（软键盘/16键键盘）落在文本区之外
        {
            int32_t touch_result = ui_widget_textarea_touch_handler(key_event, global_state, &input_state->textarea);
            if (touch_result == 1) {
                ui_draw_input_buffer(key_event, global_state, input_state);
                return current_focus_state;
            }
            else if (touch_result == 2) {
                // 点按定位光标（命中辅助返回光标槽位 s，cursor_pos = s - 1）
                int32_t slot = ui_widget_textarea_char_index_at(key_event, global_state,
                    &input_state->textarea, key_event->touch_down_x, key_event->touch_down_y);
                if (slot >= 0) {
                    input_state->cursor_pos = slot - 1;
                    input_state->desired_x = -1;
                    ui_draw_input_buffer(key_event, global_state, input_state);
                }
                return current_focus_state;
            }
        }
    }
    // 软键盘自身状态变化（粘滞修饰键、按下高亮）时，补画键盘并刷新
    if (ui_softkbd_is_visible() && ui_softkbd_take_dirty()) {
        ui_softkbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
        gfx_refresh(global_state->gfx);
    }
    // 16键虚拟键盘自身状态变化（按住高亮）时，补画键盘并刷新
    if (ui_grid16kbd_is_visible() && ui_grid16kbd_take_dirty()) {
        ui_grid16kbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
        gfx_refresh(global_state->gfx);
    }

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

    // 英文字母输入法的字母指示器与倒计时进度条均位于页脚（底栏）带：
    // 带顶 = 屏底 - 软键盘/16键键盘高度 - 带高，进度条绘制于带底沿 2px
    int32_t alpha_band_height = gfx_font_line_height(global_state->ui_font) + 1;
    int32_t alpha_band_bottom = (int32_t)global_state->gfx->height - (int32_t)ui_softkbd_height() - (int32_t)ui_grid16kbd_height();

    // 定时器触发：字母输入的倒计时进度条
    if (input_state->ime_mode_flag == IME_MODE_ALPHABET && input_state->alphabet_is_counting_down == 1) {
        uint64_t ctimestamp = global_state->timestamp;
        // 倒计时进行中，绘制进度条
        if (ctimestamp - input_state->alphabet_click_timestamp <= ALPHABET_COUNTDOWN_MS) {
            uint32_t x_pos = (ALPHABET_COUNTDOWN_MS - ctimestamp + input_state->alphabet_click_timestamp) * global_state->gfx->width / ALPHABET_COUNTDOWN_MS;
            gfx_draw_line(global_state->gfx, 0, (alpha_band_bottom - 2), x_pos, (alpha_band_bottom - 2), countdown_fg_R, countdown_fg_G, countdown_fg_B, 1);
            gfx_draw_line(global_state->gfx, 0, (alpha_band_bottom - 1), x_pos, (alpha_band_bottom - 1), countdown_fg_R, countdown_fg_G, countdown_fg_B, 1);
            gfx_draw_line(global_state->gfx, x_pos + 1, (alpha_band_bottom - 2), (global_state->gfx->width - 1), (alpha_band_bottom - 2), countdown_bg_R, countdown_bg_G, countdown_bg_B, 1);
            gfx_draw_line(global_state->gfx, x_pos + 1, (alpha_band_bottom - 1), (global_state->gfx->width - 1), (alpha_band_bottom - 1), countdown_bg_R, countdown_bg_G, countdown_bg_B, 1);
            gfx_refresh(global_state->gfx);
            input_state->state = 0;
        }
        // 倒计时结束，提交当前选中的字母，清除进度条
        else {
            input_state->alphabet_is_counting_down = 0;

            // 清除进度条
            gfx_draw_line(global_state->gfx, 0, (alpha_band_bottom - 2), (global_state->gfx->width - 1), (alpha_band_bottom - 2), countdown_bg_R, countdown_bg_G, countdown_bg_B, 1);
            gfx_draw_line(global_state->gfx, 0, (alpha_band_bottom - 1), (global_state->gfx->width - 1), (alpha_band_bottom - 1), countdown_bg_R, countdown_bg_G, countdown_bg_B, 1);
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

        // 触屏软键盘 + 汉字输入模式：全键盘拼音输入法（拼音串与候选字显示在底栏，见 ui_pinyin_ime.c）
        if (key_event->is_softkbd == 1 && input_state->ime_mode_flag == IME_MODE_HANZI &&
            (key_event->key_edge == -1 || key_event->key_edge == -2) &&
            ui_pinyin_ime_handle_key(key_event, global_state, input_state) == 1) {
            input_state->state = 0;
        }

        // 触屏软键盘的直接按键：可打印ASCII直接插入缓冲区，绕过九键输入法（Ctrl状态下不接管，交给Ctrl组合键分支）
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->is_softkbd == 1 &&
            global_state->is_ctrl_enabled == 0 &&
            key_event->key_code >= NANO_KEY_space && key_event->key_code <= NANO_KEY_tilde) {
            insert_char(input_state, (wchar_t)(key_event->key_code));
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->state = 0;
        }

        // 退格键（触屏软键盘BS）：始终删除一个字符，不触发返回
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_backspace) {
            delete_char(input_state);
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->state = 0;
        }

        // Ctrl+空格（触屏软键盘）：依次切换汉-英-数输入模式
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->is_softkbd == 1 &&
            global_state->is_ctrl_enabled == 1 && key_event->key_code == NANO_KEY_space) {
            global_state->is_ctrl_enabled = 0;
            input_state->ime_mode_flag = (input_state->ime_mode_flag + 1) % 3;
            ui_pinyin_ime_reset(); // 切换输入模式时，放弃进行中的拼音组字
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->state = 0;
        }

        // Ctrl+1：输入符号（消费型组合键，用后清除Ctrl状态）
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_1 && global_state->is_ctrl_enabled == 1) {
            global_state->is_ctrl_enabled = 0;
            memset(input_state->candidates, 0, sizeof(input_state->candidates));

            input_state->candidate_num = 54;
            for (int i = 0; i < input_state->candidate_num; i++) {
                input_state->candidates[i] = (uint32_t)ime_symbols[i];
            }

            candidate_paging(input_state);

            // 先整体重绘（页眉 ◆ 图标与 16 键键盘文案/高亮随 Ctrl 复位同步刷新），再叠加符号候选条
            ui_draw_input_buffer(key_event, global_state, input_state);
            ui_draw_input_symbol(key_event, global_state, input_state);
            gfx_refresh(global_state->gfx);

            input_state->current_page = 0;
            input_state->state = 3;
        }

        // Ctrl+0：呼出/关闭触屏软键盘（消费型组合键；文本输入控件固有功能）
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_0 && global_state->is_ctrl_enabled == 1) {
            global_state->is_ctrl_enabled = 0;
            ui_widget_input_toggle_softkbd(key_event, global_state);
            input_state->state = 0;
        }

        // 短按0：数字输入模式下是直接输入0，其余模式无动作
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_0) {
            if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
                // input_state->text[(input_state->length)++] = L'0';
                // input_state->cursor_pos++;
                insert_char(input_state, L'0');
                ui_draw_input_buffer(key_event, global_state, input_state);
                input_state->state = 0;
            }
        }

        // 短按1-9：输入拼音/字母/数字，根据输入模式标志，转向不同的状态
        else if (key_event->key_edge == -1 && (key_event->key_code >= NANO_KEY_1 && key_event->key_code <= NANO_KEY_9)) {
            // Ctrl+2：切换思考模式/非思考模式
            if (global_state->is_ctrl_enabled == 1 && key_event->key_code == NANO_KEY_2) {
                global_state->is_ctrl_enabled = 0;
                global_state->is_thinking_enabled = 1 - global_state->is_thinking_enabled;
                ui_draw_input_buffer(key_event, global_state, input_state);
            }

            else if (input_state->ime_mode_flag == IME_MODE_HANZI) {
                if (key_event->key_code >= NANO_KEY_2 && key_event->key_code <= NANO_KEY_9) { // 仅响应按键2-9；1无动作
                    input_state->state = 1;
                    ui_widget_input_event_handler(
                        key_event, global_state, input_state,
                        prev_focus_state, current_focus_state, next_focus_state);
                }
            }
            else if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
                // input_state->text[(input_state->length)++] = (wchar_t)(key_event->key_code);
                // input_state->cursor_pos++;
                insert_char(input_state, (wchar_t)(key_event->key_code));
                ui_draw_input_buffer(key_event, global_state, input_state);
                input_state->state = 0;
            }
            else if (input_state->ime_mode_flag == IME_MODE_ALPHABET) {
                // 如果按键按下时，不是字母切换状态，则开始循环切换，并开始倒计时。
                if (input_state->alphabet_is_counting_down == 0) {
                    input_state->alphabet_is_counting_down = 1;
                    input_state->alphabet_click_timestamp = global_state->timestamp;
                    input_state->alphabet_current_key = key_event->key_code - '0';
                    input_state->alphabet_index = 0;
                }
                // 如果按键按下时，倒计时尚未结束，则切换到下一个字母。
                else {
                    input_state->alphabet_is_counting_down = 1;
                    input_state->alphabet_click_timestamp = global_state->timestamp;
                    input_state->alphabet_current_key = key_event->key_code - '0';
                    input_state->alphabet_index = (input_state->alphabet_index + 1) % wcslen(ime_alphabet[(int)(key_event->key_code - '0')]);
                }

                // 在页脚（底栏）带内循环显示当前选中的字母（每个字母的占位宽度按当前字体逐字符实际宽度计算）：
                // 显示前先清空页脚带（含页脚文本与 [键盘]/[16键] 热点标签，底色与页脚一致）；
                // 倒计时结束提交字母后 ui_draw_input_buffer 整体重绘，恢复页脚全部内容
                wchar_t letter[2];
                uint32_t font_id = global_state->ui_font;
                int32_t line_height = gfx_font_line_height(font_id);
                int32_t band_height = line_height + 1;
                int32_t band_top = (int32_t)global_state->gfx->height - (int32_t)ui_softkbd_height() - (int32_t)ui_grid16kbd_height() - band_height;
                uint8_t band_bg_R, band_bg_G, band_bg_B;
                ui_footer_bg_color(global_state->ui_color_style, &band_bg_R, &band_bg_G, &band_bg_B);
                gfx_draw_rectangle(global_state->gfx, 0, band_top, global_state->gfx->width, band_height, band_bg_R, band_bg_G, band_bg_B, 1);
                int32_t x_pos = 1;
                int32_t y_pos = band_top;
                for (int i = 0; i < wcslen(ime_alphabet[(int)(key_event->key_code - '0')]); i++) {
                    letter[0] = ime_alphabet[(int)(key_event->key_code - '0')][i]; letter[1] = 0;
                    int32_t char_width = gfx_font_char_advance(font_id, (uint32_t)letter[0]);
                    if (i == input_state->alphabet_index) {
                        gfx_draw_rectangle(global_state->gfx, x_pos-1, y_pos, char_width+1, line_height-1, candidate1_bg_R, candidate1_bg_G, candidate1_bg_B, 1);
                        gfx_font_draw_text(global_state->gfx, font_id, letter, x_pos, y_pos, candidate1_fg_R, candidate1_fg_G, candidate1_fg_B, 1);
                    }
                    else {
                        gfx_draw_rectangle(global_state->gfx, x_pos-1, y_pos, char_width+1, line_height-1, candidate0_bg_R, candidate0_bg_G, candidate0_bg_B, 1);
                        gfx_font_draw_text(global_state->gfx, font_id, letter, x_pos, y_pos, candidate0_fg_R, candidate0_fg_G, candidate0_fg_B, 1);
                    }
                    x_pos += char_width + 2;
                }

                input_state->state = 0;
            }
        }

        // 长+短按A键：删除一个字符，或返回上一个状态，取决于缓冲区状态和Ctrl状态
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
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
                ui_widget_input_init(key_event, global_state, input_state, input_state->title_text);
                ui_widget_input_on_leave(global_state, input_state); // 离开输入控件：遮罩+软键盘清理
                return prev_focus_state;
            }
        }

        // 长+短按B键：依次切换汉-英-数输入模式 / 或Ctrl显示帮助
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_shift) {
            // 如果非Ctrl状态，则依次切换汉-英-数输入模式
            if (global_state->is_ctrl_enabled == 0) {
                // 触屏软键盘的SFT键仅用于大写粘滞（键码与粘滞耦合传递，见ui_softkbd.c），
                // 不切换输入模式；软键盘模式下请使用 Ctrl+空格 切换输入模式
                if (key_event->is_softkbd == 0) {
                    input_state->ime_mode_flag = (input_state->ime_mode_flag + 1) % 3;
                    ui_pinyin_ime_reset(); // 切换输入模式时，放弃进行中的拼音组字
                    ui_draw_input_buffer(key_event, global_state, input_state);
                }
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
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_ctrl) {
            global_state->is_ctrl_enabled = 1 - global_state->is_ctrl_enabled;
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->state = 0;
        }

        // 短按D键：进入下一个状态；或者Ctrl状态下 输入一个换行符
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            if (global_state->is_ctrl_enabled == 1) {
                global_state->is_ctrl_enabled = 0;
                insert_char(input_state, L'\n');
                ui_draw_input_buffer(key_event, global_state, input_state);
            }
            else {
                input_state->state = 0;
                ui_widget_input_on_leave(global_state, input_state); // 离开输入控件：遮罩+软键盘清理
                return next_focus_state;
            }
        }

        // 长+短按*键：光标向左移动（Ctrl+*：光标向上移动一个视觉行，消费型组合键）
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
            if (global_state->is_ctrl_enabled == 1) {
                global_state->is_ctrl_enabled = 0; // 消费型组合键：用后清除Ctrl状态
                ui_widget_input_move_cursor_vertical(global_state, input_state, -1);
            }
            else {
                if (input_state->cursor_pos > -1) {
                    input_state->cursor_pos--;
                }
                else {
                    input_state->cursor_pos = -1;
                }
                input_state->desired_x = -1; // 左右移动后，上下移动以新光标位置重新取目标x
            }
            ui_draw_input_buffer(key_event, global_state, input_state);
        }

        // 长+短按#键：光标向右移动（Ctrl+#：光标向下移动一个视觉行，消费型组合键）
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
            if (global_state->is_ctrl_enabled == 1) {
                global_state->is_ctrl_enabled = 0; // 消费型组合键：用后清除Ctrl状态
                ui_widget_input_move_cursor_vertical(global_state, input_state, 1);
            }
            else {
                if (input_state->cursor_pos < input_state->textarea.length - 1) {
                    input_state->cursor_pos++;
                }
                else {
                    input_state->cursor_pos = input_state->textarea.length - 1;
                }
                input_state->desired_x = -1; // 左右移动后，上下移动以新光标位置重新取目标x
            }
            ui_draw_input_buffer(key_event, global_state, input_state);
        }

        // 长+短按↑键：光标向上移动一个视觉行
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_up) {
            ui_widget_input_move_cursor_vertical(global_state, input_state, -1);
            ui_draw_input_buffer(key_event, global_state, input_state);
        }

        // 长+短按↓键：光标向下移动一个视觉行
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_down) {
            ui_widget_input_move_cursor_vertical(global_state, input_state, 1);
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
        if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            if (input_state->candidate_num > 0) {
                ui_draw_input_pinyin(key_event, global_state, input_state, 1);
                gfx_refresh(global_state->gfx);
                input_state->state = 2;
            }
        }

        // 短按A键（退格）：删除一个已输入的数字键并刷新候选条；删空则取消输入、回到初始状态
        //（先复位状态再整体重绘，避免页脚带候选条残留）
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            input_state->pinyin_keys /= 10; // 删除最后一个数字键（数字为 2-9，无前导零问题）
            if (input_state->pinyin_keys == 0) {
                input_state->current_page = 0;
                input_state->state = 0;
                ui_draw_input_buffer(key_event, global_state, input_state);
            }
            else {
                memset(input_state->candidates, 0, sizeof(input_state->candidates));
                memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));
                get_candidate_hanzi_list(input_state);
                candidate_paging(input_state);
                input_state->current_page = 0;
                ui_draw_input_pinyin(key_event, global_state, input_state, 0);
                gfx_refresh(global_state->gfx);
                input_state->state = 1;
            }
        }

        // 短按2-9键：继续输入拼音
        else if (key_event->key_edge == -1 && (key_event->key_code >= NANO_KEY_2 && key_event->key_code <= NANO_KEY_9)) {
            input_state->pinyin_keys *= 10;
            input_state->pinyin_keys += (uint32_t)(key_event->key_code - '0');

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
            gfx_refresh(global_state->gfx);

            input_state->state = 1;
        }
    }

    else if (state == 2) {
        // 短按数字键：从候选字列表中选定一个字（编号 1~N 对应每页 N=MAX_CANDIDATE_NUM_PER_PAGE 个候选，
        // 选取区间直接由该宏界定，与分页/显示严格同步），选定后转到初始状态；
        // 其余数字键忽略，保持选字状态（与全键盘拼音输入法一致）
        if (key_event->key_edge == -1 && key_event->key_code >= NANO_KEY_1
            && key_event->key_code < NANO_KEY_1 + MAX_CANDIDATE_NUM_PER_PAGE) {
            uint32_t index = key_event->key_code - '1';
            // 将选中的字加入输入缓冲区
            uint32_t ch = input_state->candidate_pages[input_state->current_page][index];
            if (ch) {
                insert_char(input_state, ch);

                // 先复位状态与候选数据再整体重绘（顺序不可颠倒：否则 ui_draw_input_buffer
                // 会在 state==2 下把候选条重画进页脚带，选字后候选条残留）
                memset(input_state->candidates, 0, sizeof(input_state->candidates));
                memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));
                input_state->current_page = 0;
                input_state->pinyin_keys = 0;
                input_state->state = 0;
                ui_draw_input_buffer(key_event, global_state, input_state);
            }
            // 选到本页空槽位（候选不足5个）：忽略，保持选字状态
        }

        // 长+短按*键：候选字翻页到上一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
            if(input_state->current_page > 0) {
                input_state->current_page--;
                ui_draw_input_pinyin(key_event, global_state, input_state, 1);
                gfx_refresh(global_state->gfx);
            }
            input_state->state = 2;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                ui_draw_input_pinyin(key_event, global_state, input_state, 1);
                gfx_refresh(global_state->gfx);
            }
            input_state->state = 2;
        }

        // 短按A键（退格）：取消选字，回到组字状态（按键序列与候选保留，候选条去除序号）
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            input_state->state = 1;
            ui_draw_input_pinyin(key_event, global_state, input_state, 0);
            gfx_refresh(global_state->gfx);
        }
    }

    else if (state == 3) {
        // 短按数字键：从符号列表中选定一个符号（编号 1~N 对应每页 N=MAX_CANDIDATE_NUM_PER_PAGE 个候选，
        // 选取区间直接由该宏界定，与分页/显示严格同步），选定后转到初始状态；
        // 其余数字键忽略，保持选符状态（与全键盘拼音输入法一致）
        if (key_event->key_edge == -1 && key_event->key_code >= NANO_KEY_1
            && key_event->key_code < NANO_KEY_1 + MAX_CANDIDATE_NUM_PER_PAGE) {
            uint32_t index = key_event->key_code - '1';
            // 将选中的符号加入输入缓冲区
            uint32_t ch = input_state->candidate_pages[input_state->current_page][index];
            if (ch) {
                insert_char(input_state, ch);

                // 先复位状态与候选数据再整体重绘（顺序不可颠倒，同上）
                memset(input_state->candidates, 0, sizeof(input_state->candidates));
                memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));
                input_state->current_page = 0;
                input_state->pinyin_keys = 0;
                input_state->state = 0;
                ui_draw_input_buffer(key_event, global_state, input_state);
            }
            // 选到本页空槽位（候选不足5个）：忽略，保持选符状态
        }

        // 长+短按*键：候选字翻页到上一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
            if(input_state->current_page > 0) {
                input_state->current_page--;
                ui_draw_input_symbol(key_event, global_state, input_state);
                gfx_refresh(global_state->gfx);
            }
            input_state->state = 3;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                ui_draw_input_symbol(key_event, global_state, input_state);
                gfx_refresh(global_state->gfx);
            }
            input_state->state = 3;
        }

        // 短按A键：取消选择，回到初始状态（先复位状态再整体重绘，避免候选条残留）
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
            ui_draw_input_buffer(key_event, global_state, input_state);
        }
    }

    // 特殊状态：显示使用说明
    else if (state == 9) {
        // 按任意键返回状态0
        if ((key_event->key_edge < 0) && key_event->key_code != NANO_KEY_IDLE) {
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->state = 0;
        }
    }

    return current_focus_state;
}




// 顶栏“返回”标签：作为页眉右侧文本经 ui_draw_header_full 固有绘制（见 ui_widget_menu_refresh），
// 同时作为触屏退出热区提示（顶栏最右侧 1/4 点击退出，见事件处理器）；菜单区滚动重绘
// （ui_widget_menu_draw）不触碰页眉带，避免抗锯齿边缘被反复混合而模糊。

void ui_widget_menu_init(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    // 菜单位于页眉与屏幕底边之间（页脚底栏已取消，菜单有效高度填满原页脚区域）。
    // 页眉高度与条目行高与所使用字体行高成倍数关系（文字在页眉/条目内纵向居中）。
    // 密集列表界面（如电子词典候选）可在本函数返回后覆写 header_height/item_height
    // 并自行修正 y/height（见 ui_dict.c）。
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    menu_state->header_height = line_height * 3 / 2;
    menu_state->x = 0;
    menu_state->y = menu_state->header_height;
    menu_state->zindex = 0;
    menu_state->width = global_state->gfx->width;
    menu_state->height = global_state->gfx->height - ui_softkbd_height() - ui_grid16kbd_height() - menu_state->header_height; // 减去页眉，并为触屏软键盘与16键虚拟键盘让出空间
    menu_state->current_item_index = 0;
    menu_state->first_item_intex = 0;
    // 条目行高布局策略：先以基础行高（字体行高的硬编码倍率）估算每页可容纳的条目数，
    // 再把行高微调为 菜单区高度/条目数（整除向下取整，剩余不足条目数像素，分摊后每行
    // 至多差1px），使条目恰好撑满一页，页底不再遗留大块空白
    int32_t base_item_height = line_height * 3 / 2;
    uint32_t max_items_per_page = menu_state->height / base_item_height;
    if (max_items_per_page < 1) max_items_per_page = 1;
    menu_state->item_height = (menu_state->height - 1) / (int32_t)max_items_per_page; // 条目自 y+1 起绘，预留的 1px 从可填充高度中扣除
    menu_state->items_per_page = (menu_state->item_num > max_items_per_page) ? max_items_per_page : menu_state->item_num;

    // 触屏交互状态复位（菜单场景互斥，进入时重新初始化）
    menu_state->touch_active = 0;
    menu_state->touch_is_dragging = 0;
    menu_state->touch_start_x = 0;
    menu_state->touch_start_y = 0;
    menu_state->touch_anchor_scroll_px = 0;
    menu_state->scroll_sub_offset = 0;
    // 拖动速度采样与惯性滚动状态复位
    menu_state->touch_track_scroll = 0;
    menu_state->touch_track_ts = 0;
    menu_state->touch_track_vel = 0.0f;
    menu_state->fling_velocity = 0.0f;
    menu_state->fling_scroll_px = 0.0f;
    menu_state->fling_last_timestamp = 0;
    // 碰撞回弹状态复位
    menu_state->bounce_offset_px = 0.0f;
    menu_state->bounce_velocity = 0.0f;
    menu_state->bounce_last_timestamp = 0;

    // 注意：此处不再立即绘制。此前末尾调用 ui_widget_menu_draw（内含 gfx_refresh）会导致
    // 进入菜单状态时先刷出菜单区（页眉尚未绘制，残留旧画面）、下一拍状态初始化分支
    // 再补画页眉，形成两阶段断续感。现全部 4 个调用点（model/game/ebook/ofdm 菜单）
    // 均在随后的状态初始化分支统一“页眉+菜单”画齐后一次刷屏（ui_widget_menu_refresh）。
    (void)key_event;
}

void ui_widget_menu_refresh(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    // 菜单控件自绘页眉（若干倍字体行高，标题居中），“返回”标签作为页眉右侧文本固有绘制：
    // 本函数的全部调用点均为状态进入/整体重绘分支，标签与页眉同生命周期。调用点先画的标准高度
    // 页眉会被此处完整覆盖（同为置色模式，无残影）；随后在 ui_widget_menu_draw 的 gfx_refresh
    // 中同帧推屏，不会被滚动重绘反复混合。
    ui_draw_header_full(key_event, global_state, (wchar_t *)menu_state->title, 1, menu_state->header_height, NULL, L"返回 ");
    ui_widget_menu_draw(key_event, global_state, menu_state); // 内含 gfx_refresh 统一推屏
}

void ui_widget_menu_draw(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {

    uint32_t x_indent = 6;

    // 配色随全局色彩风格（ui_color.h）：亮色保持原配色；
    // 暗色：背景纯黑(#000000)、文字白(#ffffff)、选中条高亮底色 #003399（文字仍白）
    uint8_t bg_r = 255, bg_g = 255, bg_b = 255;  // 菜单背景
    uint8_t hl_r = 222, hl_g = 222, hl_b = 222;  // 选中条高亮底色
    uint8_t fg_r = 0,   fg_g = 0,   fg_b = 0;    // 文字
    if (global_state->ui_color_style == UI_COLOR_DARK) {
        bg_r = 0x00; bg_g = 0x00; bg_b = 0x00;
        hl_r = 0x00; hl_g = 0x33; hl_b = 0x99;
        fg_r = 0xff; fg_g = 0xff; fg_b = 0xff;
    }

    // 清除背景
    gfx_draw_rectangle(global_state->gfx, menu_state->x, menu_state->y, menu_state->width, menu_state->height, bg_r, bg_g, bg_b, 1);

    // 裁剪到菜单区：亚行滚动时顶/底条目部分可见，防止文字与高亮块画进页眉/页脚
    gfx_set_clip(global_state->gfx, menu_state->x, menu_state->y, menu_state->width, menu_state->height);

    // 菜单首行：标题和选项数
    // gfx_draw_textline(global_state->gfx, menu_state->title, x_indent, 0, 0, 255, 255, 1);
    // wchar_t item_counter[13];
    // swprintf(item_counter, 13, L"%d/%d", menu_state->current_item_index + 1, menu_state->item_num);
    // int32_t iclen = wcslen(item_counter);
    // gfx_draw_textline(global_state->gfx, item_counter, (global_state->gfx->width-2) - iclen * 6, 0, 255, 255, 0, 1);

    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    int32_t item_height = menu_state->item_height; // 条目行高（默认 1.5 倍字体行高）
    // 像素级滚动：首条目向上平移 scroll_sub_offset（亚行偏移），循环按可见像素区终止，
    // 顶/底各可能出现一个被裁剪的部分可见条目（最多比整行时多画 1 个）；
    // bounce_offset_px 为到顶/到底碰撞回弹的纯视觉位移（顶端下拉为正、底端上拉为负，
    // 拉出的空白区由上方背景清除与裁剪矩形兜底），逻辑滚动位置不含回弹
    int32_t y_pos = (int32_t)menu_state->y + 1 - menu_state->scroll_sub_offset
                  + (int32_t)menu_state->bounce_offset_px;
    int32_t y_end = (int32_t)menu_state->y + menu_state->height;
    uint8_t is_highlight = 0;
    for (uint32_t i = menu_state->first_item_intex; i < menu_state->item_num; i++) {
        if (y_pos >= y_end) {
            break;
        }
        if (i != menu_state->current_item_index) {
            is_highlight = 0;
        }
        else {
            is_highlight = 1;
        }
        // 绘制高亮底色（覆盖整个条目行高）
        if (is_highlight) {
            for (int32_t j = y_pos; j < y_pos + item_height; j++) {
                gfx_draw_line(global_state->gfx, menu_state->x, j, menu_state->x + menu_state->width, j, hl_r, hl_g, hl_b, 1);
            }
        }
        // 绘制文字（行顶偏移使文字在条目行内纵向居中）
        gfx_font_draw_text(global_state->gfx, font_id, (wchar_t *)menu_state->items[i], menu_state->x + x_indent,
            y_pos + (item_height - line_height) / 2, fg_r, fg_g, fg_b, 1);

        y_pos += item_height;
    }

    // 菜单的滚动条（像素单位：滚动位置/内容高度/可视高度，随触点平滑移动）
    int32_t content_px = (int32_t)menu_state->item_num * item_height;
    int32_t view_px = (int32_t)menu_state->height - 1; // 条目自 y+1 起绘，可视高度少 1px
    int32_t scroll_px = (int32_t)menu_state->first_item_intex * item_height + menu_state->scroll_sub_offset;
    ui_draw_scroll_bar(
        key_event, global_state,
        scroll_px, content_px, view_px,
        menu_state->x, menu_state->y, menu_state->width, menu_state->height);

    // NOTE 因fb_draw_textline会额外给文字上方增加一行，因此这个横线在菜单文字绘制之后再绘制
    // gfx_draw_line(global_state->gfx, 0, 12, global_state->gfx->width, 12, 128, 128, 128, 1);

    gfx_reset_clip(global_state->gfx); // 恢复整屏裁剪，避免泄漏到后续帧的其它绘制
    gfx_refresh(global_state->gfx);
}


// 碰撞回弹物理常数（真机手感可调）
#define UI_MENU_BOUNCE_RUBBER_DIV  (2.0f)    // 橡皮筋阻尼：过界位移除以该系数
#define UI_MENU_BOUNCE_MAX_RATIO   (4)       // 最大回弹行程 = 菜单高度 / 该值
#define UI_MENU_BOUNCE_SPRING_K    (120.0f)  // 弹簧刚度（1/s²；ω≈11rad/s，偏软、回摆可见）
#define UI_MENU_BOUNCE_SPRING_C    (12.0f)   // 弹簧阻尼（1/s，ζ≈0.55，带一次可见回摆）
#define UI_MENU_BOUNCE_FLING_GAIN  (13.0f)   // 惯性撞边剩余速度开方映射系数：
                                             // v0 = 13·√|v剩余|——小剩余速度也有可见过冲
                                             // （剩余 500px/s→约 20px；4000px/s→约 48px 顶到上限），
                                             // 线性映射在典型剩余速度（几百 px/s）下过冲仅数像素、肉眼不可见
#define UI_MENU_BOUNCE_SETTLE_X    (0.5f)    // 收敛判定：位移阈值（px）
#define UI_MENU_BOUNCE_SETTLE_V    (10.0f)   // 收敛判定：速度阈值（px/s）

// 通用的菜单事件处理+回调注册
int32_t ui_widget_menu_event_handler(
    Key_Event *ke, Global_State *gs, Widget_Menu_State *ms,
    int32_t (*menu_item_action_callback)(Key_Event*, Global_State*, Widget_Menu_State*), int32_t prev_focus_state, int32_t current_focus_state
) {
    int32_t item_height = ms->item_height; // 条目行高（默认 1.5 倍字体行高，见 ui_widget_menu_init）

    // ========================================================================
    // 松手惯性滚动（fling）：动画激活期间本 handler 每帧推进一次；
    // 任意新触摸或按键边沿立即终止动画（随后的触摸/按键逻辑以当前 scroll_px 接管）
    // ========================================================================
    if (ms->fling_velocity != 0.0f) {
        if (ke->is_touching || ke->touch_edge != 0 || ke->key_edge != 0) {
            ms->fling_velocity = 0.0f;
        }
        else {
            uint64_t now = get_timestamp_in_ms();
            float dt_s = (float)(now - ms->fling_last_timestamp) / 1000.0f;
            ms->fling_last_timestamp = now;
            if (dt_s > 0.05f) dt_s = 0.05f; // 帧间隔异常（阻塞/掉帧）时限幅，避免跳动
            if (dt_s > 0.0f) {
                int32_t max_scroll_px = (int32_t)ms->item_num * item_height - ((int32_t)ms->height - 1);
                if (max_scroll_px < 0) max_scroll_px = 0;
                float scroll = ms->fling_scroll_px + ms->fling_velocity * dt_s;
                // 线性减速度 ~2000px/s²，速度归零即停
                float decel = 2000.0f * dt_s;
                if (ms->fling_velocity > 0.0f) {
                    ms->fling_velocity -= decel;
                    if (ms->fling_velocity < 0.0f) ms->fling_velocity = 0.0f;
                }
                else {
                    ms->fling_velocity += decel;
                    if (ms->fling_velocity > 0.0f) ms->fling_velocity = 0.0f;
                }
                // 撞边回弹：越界且有剩余速度 → 开方映射为弹簧初速度移交回弹动画（过冲→回稳）；
                // 无剩余速度即停（现状）。顶端：vel<0 → bounce_v>0（内容下拉过冲）；
                // 底端：vel>0 → bounce_v<0（内容上提过冲）
                if (scroll <= 0.0f) {
                    scroll = 0.0f;
                    if (ms->fling_velocity < -50.0f) {
                        ms->bounce_offset_px = 0.0f;
                        ms->bounce_velocity = sqrtf(-ms->fling_velocity) * UI_MENU_BOUNCE_FLING_GAIN;
                        ms->bounce_last_timestamp = now;
                    }
                    ms->fling_velocity = 0.0f;
                }
                if (scroll >= (float)max_scroll_px) {
                    scroll = (float)max_scroll_px;
                    if (ms->fling_velocity > 50.0f) {
                        ms->bounce_offset_px = 0.0f;
                        ms->bounce_velocity = -sqrtf(ms->fling_velocity) * UI_MENU_BOUNCE_FLING_GAIN;
                        ms->bounce_last_timestamp = now;
                    }
                    ms->fling_velocity = 0.0f;
                }
                ms->fling_scroll_px = scroll;
                int32_t new_scroll_px = (int32_t)scroll;
                int32_t new_first = new_scroll_px / item_height;
                int32_t new_sub = new_scroll_px % item_height;
                if (new_first != ms->first_item_intex || new_sub != ms->scroll_sub_offset) {
                    ms->first_item_intex = new_first;
                    ms->scroll_sub_offset = new_sub;
                    // 与触屏拖动一致：不做高亮钳制，允许高亮条目滚出可视区
                    ui_widget_menu_draw(ke, gs, ms);
                }
            }
        }
    }

    // ========================================================================
    // 碰撞回弹弹簧动画（与 fling 互斥：fling 撞边即终止并移交本动画）：每帧弹簧积分回位。
    // 手指驱动期间（拖动过界橡皮筋，touch_active==1）本块整体跳过——offset 由拖动分支逐帧
    // 赋值，不得在此吸附/积分（实测：误吸附+拖动置回逐帧交替曾致高频抖动）；取消只认新触摸
    // 序列的 DOWN 沿与按键（电平完底：is_touching 但 touch_active==0，防 DOWN 丢失），
    // 吸附时重绘一帧消除残留位移，随后的事件照常下落处理
    // ========================================================================
    if ((ms->bounce_offset_px != 0.0f || ms->bounce_velocity != 0.0f) && !ms->touch_active) {
        if ((ke->touch_edge & TOUCH_EDGE_DOWN) || ke->key_edge != 0
            || (ke->is_touching && !ms->touch_active)) {
            ms->bounce_offset_px = 0.0f;
            ms->bounce_velocity = 0.0f;
            ui_widget_menu_draw(ke, gs, ms);
        }
        else {
            uint64_t now = get_timestamp_in_ms();
            float dt_s = (float)(now - ms->bounce_last_timestamp) / 1000.0f;
            ms->bounce_last_timestamp = now;
            if (dt_s > 0.05f) dt_s = 0.05f; // 帧间隔异常限幅
            if (dt_s > 0.0f) {
                float x = ms->bounce_offset_px;
                float v = ms->bounce_velocity;
                float a = -UI_MENU_BOUNCE_SPRING_K * x - UI_MENU_BOUNCE_SPRING_C * v;
                v += a * dt_s;
                x += v * dt_s;
                float bounce_cap = (float)(ms->height / UI_MENU_BOUNCE_MAX_RATIO);
                if (x > bounce_cap)  { x = bounce_cap;  v = 0.0f; }
                if (x < -bounce_cap) { x = -bounce_cap; v = 0.0f; }
                // 收敛判定：位移与速度均低于阈值 → 归零停动画
                if ((x > -UI_MENU_BOUNCE_SETTLE_X && x < UI_MENU_BOUNCE_SETTLE_X)
                    && (v > -UI_MENU_BOUNCE_SETTLE_V && v < UI_MENU_BOUNCE_SETTLE_V)) {
                    x = 0.0f; v = 0.0f;
                }
                ms->bounce_offset_px = x;
                ms->bounce_velocity = v;
                ui_widget_menu_draw(ke, gs, ms);
            }
        }
    }

    // ========================================================================
    // 触屏交互（触屏事件队列改造后：序列开始/结束认 touch_edge 边沿事件——生产端
    // 高频检测、可靠投递，亚帧短点按不再湮灭；电平条件为投递失败极端情况的兜底。
    // 拖动跟踪仍由 is_touching 电平逐帧驱动（轨迹走快照最新坐标）。无触屏设备
    // 恒为 0，本块不产生任何行为，菜单仍纯按键操作）
    //   - 拖动屏幕：列表以像素级精度随手指连续滚动（scroll_px = first_item_intex *
    //     item_height + scroll_sub_offset，1:1 跟手）；
    //   - 点击条目：选中并执行（同 Enter）；
    //   - 点击顶栏最右侧 1/4：退出菜单（同 Esc）。
    // 菜单激活期间输入层不再生成宫格软按键（ui_app.c get_input_event 按状态抑制），
    // 触屏流是唯一输入通道，松手帧触发动作不会遗留按键事件泄漏到下一状态。
    // ========================================================================
    // 触摸序列开始：DOWN 边沿（可靠事件，按下点坐标取 touch_down_*）或电平兜底
    if (((ke->touch_edge & TOUCH_EDGE_DOWN) || ke->is_touching) && !ms->touch_active) {
        // 触摸序列开始：锚定像素级滚动位置与起点坐标，复位拖动速度采样
        ms->touch_active = 1;
        ms->touch_is_dragging = 0;
        if (ke->touch_edge & TOUCH_EDGE_DOWN) {
            ms->touch_start_x = ke->touch_down_x;
            ms->touch_start_y = ke->touch_down_y;
        }
        else {
            ms->touch_start_x = ke->touch_x;
            ms->touch_start_y = ke->touch_y;
        }
        ms->touch_anchor_scroll_px = (int32_t)ms->first_item_intex * item_height + ms->scroll_sub_offset;
        ms->touch_track_scroll = ms->touch_anchor_scroll_px;
        ms->touch_track_ts = get_timestamp_in_ms();
        ms->touch_track_vel = 0.0f;
    }
    else if (ke->is_touching && ms->touch_active) {
            int32_t dy = ke->touch_y - ms->touch_start_y; // >0：手指下滑
            if (!ms->touch_is_dragging && (dy > item_height / 2 || dy < -(item_height / 2))) {
                ms->touch_is_dragging = 1;
            }
            // 像素级最大滚动位置：内容总高 - 可视高（条目自 y+1 起绘，可视区少 1px）；
            // 内容不足一屏时为 0（不可滚），与原 item_num > items_per_page 判定等价
            int32_t max_scroll_px = (int32_t)ms->item_num * item_height - ((int32_t)ms->height - 1);
            if (max_scroll_px < 0) max_scroll_px = 0;
            if (ms->touch_is_dragging) {
                // 手指下滑 → 内容下移 → 滚动位置前移；相对锚点按像素计算，1:1 跟手无累计误差。
                // 碰撞回弹：过界部分按 1/RUBBER_DIV 折算为橡皮筋视觉位移（含上限），
                // 逻辑滚动位置仍钳在合法边界；不可滚动（max=0）的菜单也给纯回弹反馈
                int32_t raw_scroll_px = ms->touch_anchor_scroll_px - dy;
                float bounce = 0.0f;
                int32_t new_scroll_px = raw_scroll_px;
                if (raw_scroll_px < 0) {
                    new_scroll_px = 0;
                    bounce = (float)(-raw_scroll_px) / UI_MENU_BOUNCE_RUBBER_DIV;
                }
                else if (raw_scroll_px > max_scroll_px) {
                    new_scroll_px = max_scroll_px;
                    bounce = -(float)(raw_scroll_px - max_scroll_px) / UI_MENU_BOUNCE_RUBBER_DIV;
                }
                float bounce_cap = (float)(ms->height / UI_MENU_BOUNCE_MAX_RATIO);
                if (bounce > bounce_cap) bounce = bounce_cap;
                if (bounce < -bounce_cap) bounce = -bounce_cap;
                int32_t new_first = new_scroll_px / item_height;
                int32_t new_sub = new_scroll_px % item_height;
                if (new_first != ms->first_item_intex || new_sub != ms->scroll_sub_offset
                    || bounce != ms->bounce_offset_px) {
                    ms->first_item_intex = new_first;
                    ms->scroll_sub_offset = new_sub;
                    ms->bounce_offset_px = bounce;
                    ms->bounce_velocity = 0.0f; // 手指驱动期间弹簧速度清零
                    // 触屏拖动不做高亮钳制：高亮条目跟随其原条目，允许滚出可视区
                    // （按键导航时会在按键分支开头钳制回窗口，见下方）
                    ui_widget_menu_draw(ke, gs, ms);
                }
                // 拖动速度采样：指数平滑（0.6/0.4），基于原始位置（过界/回区全程速度连续），
                // 供松手惯性初速度估算
                uint64_t now = get_timestamp_in_ms();
                uint32_t dt_ms = (uint32_t)(now - ms->touch_track_ts);
                if (dt_ms > 0) {
                    float v_inst = (float)(raw_scroll_px - ms->touch_track_scroll) * 1000.0f / (float)dt_ms;
                    ms->touch_track_vel = ms->touch_track_vel * 0.6f + v_inst * 0.4f;
                    ms->touch_track_scroll = raw_scroll_px;
                    ms->touch_track_ts = now;
                }
            }
    }
    // 触摸序列结束：UP 边沿（可靠事件；同帧 DOWN+UP 的亚帧点按亦走到这里）或电平兜底
    if (((ke->touch_edge & TOUCH_EDGE_UP) || !ke->is_touching) && ms->touch_active) {
        // 构成拖动则按松手速度启动惯性滚动，否则按点击处理
        ms->touch_active = 0;
        if (ms->touch_is_dragging) {
            if (ms->bounce_offset_px != 0.0f) {
                // 过界松手：不启动惯性，弹簧回位（从当前位移、零初速释放；顶端下拉为正）
                ms->bounce_velocity = 0.0f;
                ms->bounce_last_timestamp = get_timestamp_in_ms();
            }
            else {
                // 松手前手指已停顿（>100ms 无新采样）则速度作废，不启动惯性
                uint64_t now = get_timestamp_in_ms();
                float v0 = (now - ms->touch_track_ts <= 100) ? ms->touch_track_vel : 0.0f;
                if (v0 > 4000.0f) v0 = 4000.0f;    // 触点抖动限速
                if (v0 < -4000.0f) v0 = -4000.0f;
                if (v0 > 50.0f || v0 < -50.0f) {   // 低于阈值（50px/s）不动画
                    ms->fling_velocity = v0;
                    ms->fling_scroll_px = (float)((int32_t)ms->first_item_intex * item_height + ms->scroll_sub_offset);
                    ms->fling_last_timestamp = now;
                }
            }
        }
        else if (!ms->touch_is_dragging) {
            if (ms->touch_start_y < ms->y && ms->touch_start_x >= ms->x + ms->width * 3 / 4) {
                // 点击顶栏最右侧 1/4：退出菜单
                return prev_focus_state;
            }
            else if (ms->touch_start_y >= ms->y && ms->touch_start_y < ms->y + ms->height) {
                // 点击菜单项：选中并执行（行号补偿亚行滚动偏移；sub_offset 非 0 时底部
                // 第 items_per_page+1 行部分可见，同样允许点选，故上界取等号）
                int32_t row = (ms->touch_start_y - (ms->y + 1) + ms->scroll_sub_offset) / item_height;
                if (row >= 0 && row <= ms->items_per_page) {
                    int32_t tapped_item_index = ms->first_item_intex + row;
                    if (tapped_item_index < ms->item_num) {
                        ms->current_item_index = tapped_item_index;
                        // 先刷新一帧使被点击项高亮可见（视觉反馈），再执行菜单动作
                        ui_widget_menu_draw(ke, gs, ms);
                        return menu_item_action_callback(ke, gs, ms);
                    }
                }
            }
        }
    }

    // 软硬按键仲裁：宫格映射软按键不作为键码采纳（菜单状态下输入层已抑制其生成，
    // 此处为防御性兜底——如状态切换瞬间跨核可见性延迟产生的零星软按键）；
    // 触屏软键盘键码（is_softkbd==1，如词典候选菜单的软键盘方向键导航）与
    // 实体按键（is_soft_key==0）照常走下方按键逻辑。
    if (ke->key_code != NANO_KEY_IDLE && ke->is_soft_key && !ke->is_softkbd) {
        return current_focus_state;
    }

    // 短按1-9数字键：直接选中屏幕上显示的那页的相对第几项
    // NOTE 从1开始
    // if (ke->key_edge == -1 && (ke->key_code >= NANO_KEY_1 && ke->key_code <= NANO_KEY_9)) {
    //     if ((ke->key_code - '0') <= ms->items_per_page) {
    //         ms->current_item_index = ms->first_item_intex + (uint32_t)(ke->key_code - '0') - 1;
    //         return menu_item_action_callback(ke, gs, ms);
    //     }
    // }
    // 短按A键：返回上一个焦点状态
    if (ke->key_edge == -1 && ke->key_code == NANO_KEY_esc) {
        return prev_focus_state;
    }
    // 短按D键：执行菜单项对应的功能
    else if (ke->key_edge == -1 && ke->key_code == NANO_KEY_enter) {
        // 触屏拖动可能将高亮条目滚出可视区：此时 Enter 先将其钳制回窗口并重绘
        // （揭示选中项）而不执行，避免误触发不可见条目；再次按下 Enter 才执行
        if (ms->current_item_index < ms->first_item_intex || ms->current_item_index > ms->first_item_intex + ms->items_per_page - 1) {
            if (ms->current_item_index < ms->first_item_intex) ms->current_item_index = ms->first_item_intex;
            if (ms->current_item_index > ms->first_item_intex + ms->items_per_page - 1) ms->current_item_index = ms->first_item_intex + ms->items_per_page - 1;
            ui_widget_menu_draw(ke, gs, ms);
            return current_focus_state;
        }
        return menu_item_action_callback(ke, gs, ms);
    }
    // 长+短按*键/上键：光标向上移动（上键功能同左键）
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && (ke->key_code == NANO_KEY_left || ke->key_code == NANO_KEY_up)) {
        // 触屏拖动允许高亮条目滚出可视区；按键导航须先将其钳制回窗口，保证高亮始终可见
        if (ms->current_item_index < ms->first_item_intex) ms->current_item_index = ms->first_item_intex;
        if (ms->current_item_index > ms->first_item_intex + ms->items_per_page - 1) ms->current_item_index = ms->first_item_intex + ms->items_per_page - 1;
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
        ms->scroll_sub_offset = 0; // 按键导航吸附回整行

        ui_widget_menu_draw(ke, gs, ms);

        return current_focus_state;
    }
    // 长+短按#键/下键：光标向下移动（下键功能同右键）
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && (ke->key_code == NANO_KEY_right || ke->key_code == NANO_KEY_down)) {
        // 触屏拖动允许高亮条目滚出可视区；按键导航须先将其钳制回窗口，保证高亮始终可见
        if (ms->current_item_index < ms->first_item_intex) ms->current_item_index = ms->first_item_intex;
        if (ms->current_item_index > ms->first_item_intex + ms->items_per_page - 1) ms->current_item_index = ms->first_item_intex + ms->items_per_page - 1;
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
        ms->scroll_sub_offset = 0; // 按键导航吸附回整行

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

    // 底部：触屏软键盘激活且全键盘拼音输入法正在组词时，底栏显示其拼音串与候选字；
    // 16键输入法组字/选字（state 1/2）/选符（state 3）时，底栏显示16键候选条（与全键盘一致，
    // 候选条直接绘制在页脚带内）；否则显示默认页脚
    if (ui_softkbd_height() > 0 && ui_pinyin_ime_is_composing()) {
        ui_pinyin_ime_draw_bar(global_state);
    }
    else if (input_state->state == 1 || input_state->state == 2) {
        ui_draw_input_pinyin(key_event, global_state, input_state, (input_state->state == 2) ? 1 : 0);
    }
    else if (input_state->state == 3) {
        ui_draw_input_symbol(key_event, global_state, input_state);
    }
    else {
        ui_draw_footer(key_event, global_state, L"Ctrl+Shift 使用说明", 1);
    }

    // 顶部：标题居中；右侧“返回”标签作为页眉固有部分经 ui_draw_header_full 随页眉同时机绘制
    //（与菜单控件同范式——避免抗锯齿标签被冗余重绘；点按命中见事件处理器页眉右 1/4 热区）
    ui_draw_header_full(key_event, global_state, input_state->title_text, 1,
        ui_std_header_height(global_state->ui_font), NULL, L"返回 ");
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    int32_t header_text_y = (ui_std_header_height(global_state->ui_font) - line_height) / 2; // 状态图标在页眉内垂直居中

    // 页脚热点虚拟按钮（见 AGENTS.md 第九节）：左 [键盘] 呼出/收起软键盘，右 [16键] 呼出/收起16键虚拟键盘；
    // 命中判定见 ui_widget_input_event_handler 触屏分支（页脚带左/右 1/4）。
    // 页脚带被候选条占用时（全键盘拼音组词 / 16键输入法 state 1/2/3）不绘制热点标签
    if (!(ui_softkbd_height() > 0 && ui_pinyin_ime_is_composing())
        && input_state->state != 1 && input_state->state != 2 && input_state->state != 3)
    {
        const int32_t footer_height = line_height + 1;
        const int32_t footer_bottom = (int32_t)global_state->gfx->height - (int32_t)ui_softkbd_height() - (int32_t)ui_grid16kbd_height();
        const int32_t footer_text_y = footer_bottom - footer_height + footer_height / 2 - line_height / 2;
        uint8_t btn_R = 102, btn_G = 204, btn_B = 255; // 暗色：滚动条前景同色系
        if (global_state->ui_color_style == UI_COLOR_LIGHT) { btn_R = 17; btn_G = 85; btn_B = 238; }
        gfx_font_draw_text(global_state->gfx, font_id, L"[键盘]", 2, footer_text_y, btn_R, btn_G, btn_B, 1);
        wchar_t *grid16_label = L"[16键]";
        int32_t grid16_w = gfx_font_measure_text(font_id, grid16_label);
        if (ui_grid16kbd_is_visible()) {
            gfx_font_draw_text(global_state->gfx, font_id, grid16_label, (int32_t)global_state->gfx->width - 2 - grid16_w, footer_text_y, 255, 255, 0, 1);
        }
        else {
            gfx_font_draw_text(global_state->gfx, font_id, grid16_label, (int32_t)global_state->gfx->width - 2 - grid16_w, footer_text_y, btn_R, btn_G, btn_B, 1);
        }
    }

    // 左上角状态图标：自页眉左缘起从左往右按各字符串的实际渲染宽度紧凑排列（均垂直居中）
    int32_t left_x = 1;
    // 显示输入状态
    wchar_t *ime_tag = NULL;
    if (input_state->ime_mode_flag == IME_MODE_HANZI)         ime_tag = L"[汉]";
    else if (input_state->ime_mode_flag == IME_MODE_ALPHABET) ime_tag = L"[En]";
    else if (input_state->ime_mode_flag == IME_MODE_NUMBER)   ime_tag = L"[数]";
    if (ime_tag) {
        gfx_font_draw_text(global_state->gfx, font_id, ime_tag, left_x, header_text_y, 255, 255, 0, 1);
        left_x += gfx_font_measure_text(font_id, ime_tag) + 1;
    }
    // 显示Ctrl激活状态
    if (global_state->is_ctrl_enabled == 1) {
        gfx_font_draw_text(global_state->gfx, font_id, L"◆", left_x, header_text_y, 255, 255, 255, 1);
        left_x += gfx_font_measure_text(font_id, L"◆") + 1;
    }
    // 显示思考模式启用状态
    if (global_state->is_thinking_enabled == 1) {
        gfx_font_draw_text(global_state->gfx, font_id, L"Ψ", left_x, header_text_y, 0, 255, 255, 1);
    }


    // 第一次排版：用于判断光标是否在视图内部
    // ta->current_line = 0;
    typeset_line_breaks(key_event, global_state, ta);
    typeset_view_range(ta, gfx_font_line_height(global_state->ui_font));

    // 计算光标的视觉行号（与 ui_draw_input_cursor 的折行/归属逻辑一致：
    // 折行判定发生在字符绘制之前，故光标位于 '\n' 或折行点上时归属于下一行）
    int32_t cursor_line = 0;
    {
        int32_t line_x = ta->x;
        for (int32_t i = 0; i <= input_state->cursor_pos && i < ta->length; i++) {
            wchar_t ch = ta->text[i];
            int32_t char_width = (ch == '\n') ? 0 : gfx_font_char_advance(global_state->ui_font, (uint32_t)ch);
            if (line_x + char_width >= ta->x + ta->width || ch == '\n') {
                cursor_line++;
                line_x = ta->x;
            }
            line_x += char_width;
        }
    }

    // 如果光标的视觉行不在当前视图范围内，则滚动视图跟随。
    // 仅在光标位置发生变化时跟随（触屏手动滚动后光标未变，不把视图拉回；
    // 打字/按键移动光标/控制台追加输出等改变光标时照常跟随）
    if (input_state->cursor_pos != input_state->drawn_cursor_pos) {
        if (cursor_line < ta->current_line) {
            // 光标在当前视图上方：卷到光标所在行
            ta->current_line = cursor_line;
            ta->scroll_sub_offset = 0; // 整行路径：吸附回整行
            typeset_view_range(ta, gfx_font_line_height(global_state->ui_font));
        }
        else if (cursor_line > ta->current_line + ta->view_lines - 1) {
            // 光标在当前视图下方：卷到使光标所在行位于视图末行
            //   逻辑上，如果出现这种情况，一定有 line_num > view_lines
            ta->current_line = cursor_line - ta->view_lines + 1;
            ta->scroll_sub_offset = 0; // 整行路径：吸附回整行
            typeset_view_range(ta, gfx_font_line_height(global_state->ui_font));
        }
    }
    input_state->drawn_cursor_pos = input_state->cursor_pos;

    // 绘制文本
    ui_draw_text_block(key_event, global_state, ta, global_state->ui_font);

    // 绘制滚动条（像素单位：滚动位置/内容高度/可视高度）
    if (ta->is_show_scroll_bar) {
        ui_draw_scroll_bar(
            key_event, global_state,
            ta->current_line * line_height + ta->scroll_sub_offset,
            ta->line_num * line_height, ta->height,
            ta->x, ta->y, ta->width, ta->height);
    }

    // 绘制光标
    ui_draw_input_cursor(key_event, global_state, input_state);

    // 触屏软键盘：可见时绘制在屏幕底部（与文本同帧推出，避免闪烁与二次刷新）
    // CTRL键高亮与全局Ctrl激活状态（is_ctrl_enabled）联动
    if (ui_softkbd_height() > 0) {
        ui_softkbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
    }

    // 16键虚拟键盘：可见时绘制在屏幕底部（与触屏软键盘互斥；Ctrl键高亮与全局Ctrl状态联动，同上）
    if (ui_grid16kbd_height() > 0) {
        ui_grid16kbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
    }

    gfx_refresh(global_state->gfx);
}


void ui_draw_input_cursor(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    Widget_Textarea_State *ta = &(input_state->textarea);
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id); // 同一字体行高固定
    // 绘制光标：光标位置在cursor_pos所指字符的右外边缘
    //   横坐标逐字符按实际宽度累加（与 ui_draw_text_block 的折行/绘制逻辑一致）
    int32_t char_index = 0;
    int32_t break_count = 0;
    int32_t line_x_pos = ta->x;

    // 视口首行起始边界处理：当光标位于视口首行起始位置之前的 '\n' 上时，
    // 该 '\n' 在排版（break_pos）上属于上一行末尾，但按绘制归属（见上方注释）
    // 光标应渲染在视口首行行首。若不特判，绘制循环将永远命中不到它，
    // 光标会错误地落到视口最底行（空行场景必然触发，因为空行唯一内容就是 '\n'）。
    int32_t cursor_at_view_start_boundary =
        (input_state->cursor_pos >= 0 &&
         input_state->cursor_pos == ta->view_start_pos - 1 &&
         ta->text[input_state->cursor_pos] == L'\n');

    if (input_state->cursor_pos >= 0 && !cursor_at_view_start_boundary) {
        for (char_index = ta->view_start_pos; char_index <= ta->view_end_pos; char_index++) {
            wchar_t ch = ta->text[char_index];
            int32_t char_width = (ch == '\n') ? 0 : gfx_font_char_advance(font_id, (uint32_t)ch);
            if (line_x_pos + char_width >= ta->x + ta->width || ch == '\n') {
                break_count++;
                line_x_pos = ta->x;
            }
            line_x_pos += char_width;
            if (input_state->cursor_pos == char_index) break;
        }
        // 触屏手动滚动后光标可能滚出视口（光标跟随仅在光标变化时触发）：不在视口内则不绘制
        if (input_state->cursor_pos != char_index) {
            return;
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
    // 亚行偏移：与 ui_draw_text_block 的首行上移保持一致，并裁剪到文本区
    uint32_t y = ta->y + line_height * break_count - ta->scroll_sub_offset;
    gfx_set_clip(global_state->gfx, ta->x, ta->y, ta->width, ta->height);
    gfx_draw_line(global_state->gfx, x, y-1, x, y + line_height - 1, cursor_R, cursor_G, cursor_B, 2);
    gfx_draw_line(global_state->gfx, x+1, y-1, x+1, y + line_height - 1, cursor_R, cursor_G, cursor_B, 2);
    gfx_reset_clip(global_state->gfx);
}

// 16键拼音候选条（单行，绘制于页脚（底栏）带内，位置与布局参照全键盘拼音候选条
// ui_pinyin_ime_draw_bar）：
//   [数字按键序列区 7个半角] [左翻页符号区 3个半角] [候选列表，起点固定为左数第11个半角宽度位置] [> 右对齐]
// 与全键盘的时序差异：16键先输按键序列、按 Enter 才进入选字状态（is_picking）；
// 选字前每个候选字之前预留 1 个半角空白（为候选序号预留），进入选字后以序号填充该空白。
// 本函数只写帧缓冲、不刷新屏幕，由调用方统一 gfx_refresh（同 ui_pinyin_ime_draw_bar）。
void ui_draw_input_pinyin(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state, uint32_t is_picking) {
    ui_ime_candidate_color_apply(global_state->ui_color_style);
    // 本页候选个数：由分页数学直接推导（与 candidate_paging 填充严格一致，见该函数处注释）
    uint32_t count = ui_candidate_page_item_count(input_state);

    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    // 页脚（底栏）带：底边 = 屏底 - 软键盘/16键键盘高度，带高 = 行高 + 1（同 ui_draw_footer）
    int32_t band_height = line_height + 1;
    int32_t y_top = (int32_t)global_state->gfx->height - (int32_t)ui_softkbd_height() - (int32_t)ui_grid16kbd_height() - band_height;

    // 清空候选条区域（整个页脚带；底色与页脚一致）
    uint8_t band_bg_R, band_bg_G, band_bg_B;
    ui_footer_bg_color(global_state->ui_color_style, &band_bg_R, &band_bg_G, &band_bg_B);
    gfx_draw_rectangle(global_state->gfx,
        0, y_top,
        global_state->gfx->width, band_height,
        band_bg_R, band_bg_G, band_bg_B, 1);

    // 字号基准：以全角字符“一”的实际渲染宽度确定1个全角宽度，半角宽度为其一半
    //   （与全键盘拼音候选条一致）；排版一律按像素定位，不使用空格做朴素对齐。
    int32_t full_width = gfx_font_char_advance(font_id, (uint32_t)L'一'); // 1个全角宽度
    int32_t half_width = full_width / 2;
    const int32_t x0 = 1; // 候选条内容左缘

    // 数字按键序列：左对齐绘制在固定7个半角宽度的序列区内
    wchar_t buf[30];
    swprintf(buf, 30, L"%u", input_state->pinyin_keys);
    gfx_font_draw_text(global_state->gfx, font_id, buf, x0, y_top, S_UI_COLOR_IME_CANDIDATE_PINYIN[0], S_UI_COLOR_IME_CANDIDATE_PINYIN[1], S_UI_COLOR_IME_CANDIDATE_PINYIN[2], 1);

    // 左翻页符号区：固定预留3个半角宽度（无翻页符号时保留空位）
    if (input_state->current_page > 0) {
        gfx_font_draw_text(global_state->gfx, font_id, L"<", x0 + (7 + 1) * half_width, y_top, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }

    // 候选字列表：编号与候选字紧贴（如“1的”），候选之间间隔2个半角字符宽；
    // 严格按本页个数（count，≤ MAX_CANDIDATE_NUM_PER_PAGE）绘制，不多不少、无宽度截断；
    // 选字前（is_picking==0）编号位置不绘制、仅预留1个半角空白
    int32_t x = x0 + 10 * half_width;
    if (input_state->candidate_num > 0) {
        for (uint32_t j = 0; j < count; j++) {
            uint32_t index_ch = (uint32_t)(L'1' + j);
            uint32_t cand_ch = input_state->candidate_pages[input_state->current_page][j];
            int32_t index_w = half_width; // 编号/预留空白宽度（1个半角）
            int32_t cand_w = gfx_font_char_advance(font_id, cand_ch);
            if (is_picking) {
                gfx_font_draw_char(global_state->gfx, font_id, index_ch, x, y_top, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
            }
            x += index_w;
            gfx_font_draw_char(global_state->gfx, font_id, cand_ch, x, y_top, S_UI_COLOR_IME_CANDIDATE_TEXT[0], S_UI_COLOR_IME_CANDIDATE_TEXT[1], S_UI_COLOR_IME_CANDIDATE_TEXT[2], 1);
            x += cand_w + 2 * half_width;
        }
    }
    else {
        gfx_font_draw_text(global_state->gfx, font_id, L"(无候选字)", x, y_top, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }

    // 下一页指示（右对齐）
    if (input_state->current_page + 1 < input_state->candidate_page_num) {
        gfx_font_draw_text(global_state->gfx, font_id, L">", global_state->gfx->width - 9, y_top, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }
}

// 16键符号候选条（单行，绘制于页脚（底栏）带内，布局同 ui_draw_input_pinyin，但无按键序列区）：
// 符号输入由 Ctrl+1 直接进入选符状态（无 Enter 步骤），序号恒显示。
// 本函数只写帧缓冲、不刷新屏幕，由调用方统一 gfx_refresh（同 ui_pinyin_ime_draw_bar）。
void ui_draw_input_symbol(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    ui_ime_candidate_color_apply(global_state->ui_color_style);
    // 本页候选个数：由分页数学直接推导（与 candidate_paging 填充严格一致，见该函数处注释）
    uint32_t count = ui_candidate_page_item_count(input_state);

    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    // 页脚（底栏）带：底边 = 屏底 - 软键盘/16键键盘高度，带高 = 行高 + 1（同 ui_draw_footer）
    int32_t band_height = line_height + 1;
    int32_t y_top = (int32_t)global_state->gfx->height - (int32_t)ui_softkbd_height() - (int32_t)ui_grid16kbd_height() - band_height;

    // 清空候选条区域（整个页脚带；底色与页脚一致）
    uint8_t band_bg_R, band_bg_G, band_bg_B;
    ui_footer_bg_color(global_state->ui_color_style, &band_bg_R, &band_bg_G, &band_bg_B);
    gfx_draw_rectangle(global_state->gfx,
        0, y_top,
        global_state->gfx->width, band_height,
        band_bg_R, band_bg_G, band_bg_B, 1);

    // 字号基准：以全角字符“一”的实际渲染宽度确定1个全角宽度，半角宽度为其一半（与拼音候选条一致）
    int32_t half_width = gfx_font_char_advance(font_id, (uint32_t)L'一') / 2;
    const int32_t x0 = 1; // 候选条内容左缘

    // 左翻页符号区：固定预留3个半角宽度（无翻页符号时保留空位；序列区留空，与拼音候选条布局对齐）
    if (input_state->current_page > 0) {
        gfx_font_draw_text(global_state->gfx, font_id, L"<", x0 + (7 + 1) * half_width, y_top, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }

    // 候选符号列表：编号与候选符号紧贴（如“1，”），候选之间间隔2个半角字符宽；
    // 严格按本页个数（count，≤ MAX_CANDIDATE_NUM_PER_PAGE）绘制，不多不少、无宽度截断；
    // 符号输入直接进入选符状态，序号恒显示
    int32_t x = x0 + 10 * half_width;
    if (input_state->candidate_num > 0) {
        for (uint32_t j = 0; j < count; j++) {
            uint32_t index_ch = (uint32_t)(L'1' + j);
            uint32_t cand_ch = input_state->candidate_pages[input_state->current_page][j];
            int32_t index_w = gfx_font_char_advance(font_id, index_ch);
            int32_t cand_w = gfx_font_char_advance(font_id, cand_ch);
            gfx_font_draw_char(global_state->gfx, font_id, index_ch, x, y_top, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
            x += index_w;
            gfx_font_draw_char(global_state->gfx, font_id, cand_ch, x, y_top, S_UI_COLOR_IME_CANDIDATE_TEXT[0], S_UI_COLOR_IME_CANDIDATE_TEXT[1], S_UI_COLOR_IME_CANDIDATE_TEXT[2], 1);
            x += cand_w + 2 * half_width;
        }
    }
    else {
        gfx_font_draw_text(global_state->gfx, font_id, L"(无候选符号)", x, y_top, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }

    // 下一页指示（右对齐）
    if (input_state->current_page + 1 < input_state->candidate_page_num) {
        gfx_font_draw_text(global_state->gfx, font_id, L">", global_state->gfx->width - 9, y_top, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }
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

// 预计算七段码字符串的渲染宽高（不做实际渲染）。
// 纯几何计算（无需 gfx 与上下文），宽度推算与 ui_draw_7seg_string 的步进逻辑完全一致，
// 供实际绘制前计算布局参数（如居中、右对齐、外框尺寸等）。
void ui_measure_7seg_string(
    wchar_t *text,
    float seg_length, float seg_thickness, float digit_gap,
    int32_t *text_width, int32_t *text_height
) {
    float digit_w = seg_length + 2.0f * seg_thickness; // 数字宽度（与 ui_draw_7seg_digit 一致）
    float colon_w = digit_w / 2.0f;                    // 冒号宽度（与 ui_draw_7seg_colon 一致）
    float x = 0.0f;
    int32_t len = wcslen(text);
    for (int32_t i = 0; i < len; i++) {
        wchar_t ch = text[i];
        if (ch >= L'0' && ch <= L'9') {
            x += digit_w + digit_gap;
        }
        else if (ch == L':') {
            x += colon_w + digit_gap;
        }
    }
    *text_width = (int32_t)roundf(x);
    *text_height = (int32_t)roundf(2.0f * seg_length + 3.0f * seg_thickness);
}

// 以 (cx, cy) 为中心绘制七段码字符串（先经 ui_measure_7seg_string 测量宽高，再换算左上角）
void ui_draw_7seg_string_centered(
    Key_Event *key_event, Global_State *global_state,
    int32_t cx, int32_t cy, wchar_t *text,
    uint8_t red, uint8_t green, uint8_t blue,
    float seg_length, float seg_thickness, float digit_gap, int32_t is_shadow,
    int32_t *text_width, int32_t *text_height
) {
    int32_t w = 0, h = 0;
    ui_measure_7seg_string(text, seg_length, seg_thickness, digit_gap, &w, &h);
    ui_draw_7seg_string(key_event, global_state, cx - w / 2, cy - h / 2, text,
        red, green, blue, seg_length, seg_thickness, digit_gap, is_shadow,
        text_width, text_height);
}


