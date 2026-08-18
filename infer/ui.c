#include <stdio.h>
#include <time.h>

#include "graphics.h"
#include "input_device.h"
#include "ui.h"
#include "ui_softkbd.h"
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

// 九键按键提示遮罩：文本输入控件状态下，任何触屏动作即显示，无触屏若干秒后消失。
// 遮罩为 4x4 宫格（与触屏虚拟按键布局一致），内容随 Ctrl 激活态切换。
// 显示时长由全局设置 Global_State.ime_hint_timeout_s 控制（0=关闭，可选 0/3/6 秒，系统设置中切换）。
#define IME_HINT_MASK_ALPHA       (59)    // 遮罩层不透明度（gfx mode>=4 即 alpha）
#define IME_HINT_GRID_LINE_ALPHA  (100)   // 宫格分割线不透明度
static int32_t  ime_hint_mask_armed = 0;           // 遮罩标志：1=生效（刷新钩子已注册）
static Global_State *ime_hint_gs = NULL;           // 钩子回调内取全局状态（is_ctrl_enabled/颜色风格/时间戳）
static uint8_t *ime_hint_backup = NULL;            // 干净帧快照（PSRAM，大小由 gfx_frame_snapshot_bytes 给出）
static int32_t  ime_hint_hook_backed_up = 0;       // 本轮推帧是否已快照（前后置钩子配对依据）


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
    //   NOTE 考虑到行末以下无间距，因此分子加1以去除末行无间距的影响。
    //        例如，高度为64的屏幕，使用行高13的字体时，实际可容纳(64+1)/13=5行。
    int32_t max_view_lines = (view_height + 1) / line_height;
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


// font_id: 文本字体（GFX_FONT_*）。同一字体行高固定；每个字符的实际宽度不定，
// 折行与绘制均按逐字符实际宽度处理（gfx_font_char_advance / gfx_font_draw_char，
// 二者对缺字的回退策略一致）；基线对齐由字体接口内部完成。
void ui_draw_text_block(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state, uint32_t font_id) {
    int x_pos = textarea_state->x;
    int y_pos = textarea_state->y;

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

    for (int i = textarea_state->view_start_pos; i <= textarea_state->view_end_pos; i++) {
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
    // 页眉高度跟随当前字体行高（行高 + 1px 边距）
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    const int header_height = line_height + 1;
    // 浅色模式页眉的渐变蓝色表（每3个字节为一行的RGB），共24级，支持页眉高度最高24px。
    //   前14级为原有配色：R/G通道步进量呈 ease-out 衰减（R约16→4、G约16→6，B收敛至255）。
    //   后10级按同一衰减趋势外插（R的Δ: 4,3,3,2,2,2,1,1,1,1；G的Δ: 6,5,5,4,4,3,3,2,2,2；B保持255封顶），
    //   使渐变压轴到浅蓝 (122,240,255)，更大页眉高度也能保持平滑无跳变。
    #define UI_HEADER_GRADIENT_STEPS (24)
    const uint8_t header_bgcolor[UI_HEADER_GRADIENT_STEPS * 3] = {
        17,85,238, 33,101,239, 44,114,241, 53,126,242, 60,137,243, 66,146,245, 72,155,246,
        77,163,247, 82,171,249, 86,178,250, 91,185,251, 95,192,252, 98,198,254, 102,204,255,
        // ---- 以下为按 ease-out 趋势外插的扩展级 ----
        106,210,255, 109,215,255, 112,220,255, 114,224,255, 116,228,255,
        118,231,255, 119,234,255, 120,236,255, 121,238,255, 122,240,255
    };
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        for (int i = 0; i < header_height; i++) {
            // 页眉高度超过渐变表级数时，钳制到末级颜色（最浅的蓝色）
            int32_t step = (i < UI_HEADER_GRADIENT_STEPS) ? i : (UI_HEADER_GRADIENT_STEPS - 1);
            gfx_draw_line(global_state->gfx, 0, i, global_state->gfx->width - 1, i, header_bgcolor[step*3+0], header_bgcolor[step*3+1], header_bgcolor[step*3+2], 1);
        }
        S_UI_COLOR_HEADER_TEXT[0] = 255;
        S_UI_COLOR_HEADER_TEXT[1] = 255;
        S_UI_COLOR_HEADER_TEXT[2] = 255;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_draw_rectangle(global_state->gfx, 0, 0, global_state->gfx->width, header_height, 15, 16, 17, 1);
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
    // 触屏软键盘显示时，页脚上移为键盘让出空间（键盘隐藏时 ui_softkbd_height() 为0，行为不变）
    const int32_t footer_bottom = global_state->gfx->height - ui_softkbd_height();
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
    // 触屏软键盘显示时，页脚上移为键盘让出空间（与 ui_draw_footer 一致）
    const int32_t footer_bottom = global_state->gfx->height - ui_softkbd_height();
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
    // 文本区位于页眉与页脚之间，页眉/页脚高度跟随当前字体行高（行高 + 1px 边距）
    int32_t header_height = gfx_font_line_height(global_state->ui_font) + 1;
    textarea_state->state = 0;
    textarea_state->x = 0;
    textarea_state->y = header_height;
    textarea_state->width = global_state->gfx->width;
    textarea_state->height = global_state->gfx->height - ui_softkbd_height() - header_height * 2; // 减去header和footer，并为触屏软键盘让出空间
    textarea_state->length = 0;
    textarea_state->line_num = 0;
    textarea_state->view_lines = 0;
    textarea_state->view_start_pos = 0;
    textarea_state->view_end_pos = 0;
    textarea_state->current_line = 0;
    textarea_state->is_show_scroll_bar = 1;
    textarea_state->is_modified = 1;
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
    textarea_state->is_show_scroll_bar = is_show_scroll_bar;
    // text缓冲区容量为 UI_STR_BUF_MAX_LENGTH 个 wchar_t（含结尾 L'\0'），截断拷贝防堆溢出
    wcsncpy(textarea_state->text, text, UI_STR_BUF_MAX_LENGTH - 1);
    textarea_state->text[UI_STR_BUF_MAX_LENGTH - 1] = L'\0';
}

void ui_widget_textarea_draw(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state) {
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

    // 文本区位于页眉与页脚之间，页眉/页脚高度跟随当前字体行高
    int32_t header_height = gfx_font_line_height(global_state->ui_font) + 1;
    ta->state = 0;
    ta->x = 0;
    ta->y = header_height;
    ta->width = global_state->gfx->width;
    ta->height = global_state->gfx->height - ui_softkbd_height() - header_height * 2; // 减去header和footer NOTE 详见结构体定义处的说明；并为触屏软键盘让出空间
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

    // 初始化各个数组
    memset(input_state->candidates, 0, sizeof(input_state->candidates));
    memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));

    // 清零触屏时间戳基线：九键按键提示遮罩仅响应本控件呈现之后的新触屏（同 ui_widget_input_refresh）
    global_state->last_touch_timestamp = 0;

    ui_draw_input_buffer(key_event, global_state, input_state);
}

void ui_widget_input_refresh(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    input_state->cursor_pos = input_state->textarea.length - 1;
    input_state->desired_x = -1;
    // 清零触屏时间戳基线：九键按键提示遮罩仅响应本控件呈现之后的新触屏，
    // 避免“用于进入本状态的那次触摸”（触屏宫格即按键）在控件出现后立即误触发遮罩；
    // 若手指仍按住不放，Core1 会在 1-2ms 内重新锁存，持续触摸的提示不受影响
    global_state->last_touch_timestamp = 0;
    ui_draw_input_buffer(key_event, global_state, input_state);
}

// 切换触屏软键盘显隐，并重新布局为键盘让出/恢复空间（文本输入控件固有功能：
// 供 Ctrl+0 组合键与上滑/下滑手势调用；软键盘启用时关闭输入法提示遮罩，避免干扰软键盘）
void ui_widget_input_toggle_softkbd(Key_Event *key_event, Global_State *global_state) {
    if (ui_softkbd_is_visible()) ui_softkbd_hide();
    else                         ui_softkbd_show();
    ui_ime_hint_mask_set_enabled(!ui_softkbd_is_visible());
    ui_pinyin_ime_reset(); // 键盘显隐切换时，放弃进行中的拼音组字
    // 重新布局：文本区高度扣除软键盘高度（隐藏时 ui_softkbd_height() 为0，布局复原）
    int32_t header_height = gfx_font_line_height(global_state->ui_font) + 1;
    global_state->w_input_main->textarea.height = global_state->gfx->height - ui_softkbd_height() - header_height * 2;
    global_state->w_input_main->textarea.is_modified = 1;
    ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
}

// 绘制文本输入操作说明
static void ui_draw_input_help(Key_Event *key_event, Global_State *global_state) {
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    int32_t cx = global_state->gfx->width / 2;
    int32_t cy = 5 + 6;
    // 触屏软键盘显示时，帮助页为其让出空间
    gfx_draw_rectangle(global_state->gfx, 3, 3, global_state->gfx->width - 6, global_state->gfx->height - ui_softkbd_height() - 6, S_UI_COLOR_IME_HELP_BG[0], S_UI_COLOR_IME_HELP_BG[1], S_UI_COLOR_IME_HELP_BG[2], 3);
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

    gfx_refresh(global_state->gfx);
}

// 九键按键提示遮罩内容（4x4 宫格，与触屏虚拟按键布局一致）。
// 每格 {第一行, 第二行}；第二行为 NULL 表示单行（16px 字体），否则双行（12px 字体）。
static const wchar_t *ime_hint_mask_grid_normal[4][4][2] = {
    {{L"1", L"符号"}, {L"2", L"ABC"}, {L"3", L"DEF"}, {L"退格", NULL}},
    {{L"4", L"GHI"},  {L"5", L"JKL"}, {L"6", L"MNO"}, {L"输入法", NULL}},
    {{L"7", L"PQRS"}, {L"8", L"TUV"}, {L"9", L"WXYZ"}, {L"Ctrl", NULL}},
    {{L"←", NULL},     {L"0", L"TUV"}, {L"→", NULL},    {L"确认", NULL}},
};
static const wchar_t *ime_hint_mask_grid_ctrl[4][4][2] = {
    {{L"符号", NULL},      {L"思考模式", NULL}, {L"3", L"DEF"}, {L"退出", NULL}},
    {{L"4", L"GHI"},      {L"5", L"JKL"}, {L"6", L"MNO"}, {L"帮助", NULL}},
    {{L"7", L"PQRS"},     {L"8", L"TUV"}, {L"9", L"WXYZ"}, {L"[Ctrl]", NULL}},
    {{L"↑", NULL},        {L"键盘", NULL},  {L"↓", NULL},  {L"换行", NULL}},
};
#define IME_HINT_MASK_CTRL_HIGHLIGHT_ROW (2) // Ctrl 激活态下高亮的格子：[Ctrl]
#define IME_HINT_MASK_CTRL_HIGHLIGHT_COL (3)

// 绘制九键按键提示遮罩（叠加在当前帧缓冲之上，调用方负责 gfx_refresh）。
// 色彩随全局颜色风格：暗色模式下为亮色遮罩+白色文字；亮色模式下为暗色遮罩+灰色文字。
static void ui_draw_ime_hint_mask(Nano_GFX *gfx, int32_t is_ctrl_enabled, int32_t ui_color_style) {
    uint8_t mask_R, mask_G, mask_B, text_R, text_G, text_B;
    if (ui_color_style == UI_COLOR_DARK) {
        mask_R = 255; mask_G = 255; mask_B = 255;  // 亮色遮罩
        text_R = 255; text_G = 255; text_B = 255;  // 白色文字
    }
    else {
        mask_R = 0;   mask_G = 0;   mask_B = 0;    // 暗色遮罩
        text_R = 128; text_G = 128; text_B = 128;  // 灰色文字
    }

    int32_t screen_w = gfx->width;
    int32_t screen_h = gfx->height;
    int32_t cell_w = screen_w / 4;
    int32_t cell_h = screen_h / 4;

    // 全屏半透明遮罩
    gfx_draw_rectangle(gfx, 0, 0, screen_w, screen_h, mask_R, mask_G, mask_B, IME_HINT_MASK_ALPHA);

    // 宫格分割线
    for (int32_t i = 1; i < 4; i++) {
        gfx_draw_line(gfx, i * cell_w, 0, i * cell_w, screen_h - 1, text_R, text_G, text_B, IME_HINT_GRID_LINE_ALPHA);
        gfx_draw_line(gfx, 0, i * cell_h, screen_w - 1, i * cell_h, text_R, text_G, text_B, IME_HINT_GRID_LINE_ALPHA);
    }

    // Ctrl 激活态：高亮 [Ctrl] 格（叠加一层文字色 + 实色边框）
    if (is_ctrl_enabled) {
        int32_t hx0 = IME_HINT_MASK_CTRL_HIGHLIGHT_COL * cell_w;
        int32_t hy0 = IME_HINT_MASK_CTRL_HIGHLIGHT_ROW * cell_h;
        gfx_draw_rectangle(gfx, hx0, hy0, cell_w, cell_h, text_R, text_G, text_B, 64);
        for (int32_t d = 0; d < 2; d++) {
            gfx_draw_line(gfx, hx0 + d, hy0 + d, hx0 + cell_w - 1 - d, hy0 + d, text_R, text_G, text_B, 1);
            gfx_draw_line(gfx, hx0 + d, hy0 + cell_h - 1 - d, hx0 + cell_w - 1 - d, hy0 + cell_h - 1 - d, text_R, text_G, text_B, 1);
            gfx_draw_line(gfx, hx0 + d, hy0 + d, hx0 + d, hy0 + cell_h - 1 - d, text_R, text_G, text_B, 1);
            gfx_draw_line(gfx, hx0 + cell_w - 1 - d, hy0 + d, hx0 + cell_w - 1 - d, hy0 + cell_h - 1 - d, text_R, text_G, text_B, 1);
        }
    }

    // 逐格绘制文字：双行 12px、单行 16px，均在格内居中
    const wchar_t *(*grid)[4][2] = is_ctrl_enabled ? ime_hint_mask_grid_ctrl : ime_hint_mask_grid_normal;
    for (int32_t row = 0; row < 4; row++) {
        for (int32_t col = 0; col < 4; col++) {
            int32_t cx = col * cell_w + cell_w / 2;
            int32_t cy = row * cell_h + cell_h / 2;
            const wchar_t *line0 = grid[row][col][0];
            const wchar_t *line1 = grid[row][col][1];
            if (line1 != NULL) {
                gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)line0, cx, cy - 8, text_R, text_G, text_B, 1);
                gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)line1, cx, cy + 8, text_R, text_G, text_B, 1);
            }
            else {
                gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_16, (wchar_t *)line0, cx, cy, text_R, text_G, text_B, 1);
            }
        }
    }
}

// 解除遮罩标志：注销刷新钩子并释放干净帧快照
static void ui_ime_hint_mask_disarm(void) {
    gfx_set_refresh_hook(NULL, NULL);
    ime_hint_mask_armed = 0;
    if (ime_hint_backup != NULL) { free(ime_hint_backup); ime_hint_backup = NULL; }
}

// 遮罩机制外部开关（默认启用；软键盘显示时由上层调用 ui_ime_hint_mask_set_enabled 关闭）
static int32_t ime_hint_mask_enabled = 1;

void ui_ime_hint_mask_set_enabled(int32_t enabled) {
    ime_hint_mask_enabled = (enabled != 0) ? 1 : 0;
    if (!ime_hint_mask_enabled && ime_hint_mask_armed) {
        // 关闭机制时立即解除已激活的遮罩（帧缓冲已被后置钩子恢复为干净底图，无需重绘）
        ui_ime_hint_mask_disarm();
    }
}

// gfx_refresh 前置钩子：推帧前备份干净帧并叠加遮罩。
// 遮罩为 alpha 叠加绘制，帧缓冲在帧间持久存在，若直接叠加会逐帧累积饱和；
// 因此先备份干净帧、推帧后由后置钩子原样恢复，使遮罩只存在于“送往屏幕的那一帧”，
// 与正常 GUI 刷新严格同步且完全不干扰输入控件的交互与各分支绘制逻辑。
static void ui_ime_hint_pre_refresh_hook(Nano_GFX *gfx) {
    ime_hint_hook_backed_up = 0;
    // 仅叠加到注册时的主 UI 帧缓冲实例
    if (gfx != ime_hint_gs->gfx) {
        return;
    }
    // 超过设定时长无触屏（或时长设置为0=关闭）自动解除遮罩标志：本次推帧即为干净帧
    // （触屏时间戳由 Core1 高频锁存于 Global_State.last_touch_timestamp）
    uint64_t timeout_ms = (uint64_t)ime_hint_gs->ime_hint_timeout_s * 1000ULL;
    if (timeout_ms == 0 || ime_hint_gs->last_touch_timestamp == 0 ||
        (ime_hint_gs->timestamp - ime_hint_gs->last_touch_timestamp) >= timeout_ms) {
        ui_ime_hint_mask_disarm();
        return;
    }
    // 快照干净帧（帧缓冲布局由图形层封装）→ 叠加遮罩
    gfx_frame_snapshot(gfx, ime_hint_backup);
    ime_hint_hook_backed_up = 1;
    ui_draw_ime_hint_mask(gfx, ime_hint_gs->is_ctrl_enabled, ime_hint_gs->ui_color_style);
}

// gfx_refresh 后置钩子：推帧后恢复干净帧缓冲（与前置钩子配对）
static void ui_ime_hint_post_refresh_hook(Nano_GFX *gfx) {
    if (!ime_hint_hook_backed_up) {
        return;
    }
    ime_hint_hook_backed_up = 0;
    gfx_frame_restore(gfx, ime_hint_backup);
}

// 离开文本输入控件时的清理（控件两个退出分支 return prev/next_focus_state 处调用）：
// 清除按键提示遮罩标志；收起软键盘并恢复布局（软键盘为控件固有功能，随控件退出而关闭）
static void ui_widget_input_on_leave(Global_State *global_state, Widget_Input_State *input_state) {
    ui_ime_hint_mask_disarm();
    if (ui_softkbd_is_visible()) {
        ui_softkbd_hide();
        ui_ime_hint_mask_set_enabled(1);
        ui_pinyin_ime_reset();
        int32_t header_height = gfx_font_line_height(global_state->ui_font) + 1;
        input_state->textarea.height = global_state->gfx->height - header_height * 2;
        input_state->textarea.is_modified = 1;
    }
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

int32_t ui_widget_input_event_handler(
    Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state,
    int32_t prev_focus_state, int32_t current_focus_state, int32_t next_focus_state
) {

    // 九键按键提示遮罩（全局标志：触屏置位、3 秒无触屏清除）。
    // 触屏时间戳由 Core1 的 get_key_event 以 1-2ms 周期高频锁存（global_state->last_touch_timestamp，
    // 短按不遗漏）；本处于处理器开头只做判定与置位/清除，置位后本帧的正常 UI 刷新推帧前
    // 即被 gfx 刷新钩子叠加遮罩、推帧后恢复帧缓冲（遮罩与正常 GUI 刷新严格同步）。
    {
        uint64_t timeout_ms = (uint64_t)global_state->ime_hint_timeout_s * 1000ULL;
        int32_t ime_hint_active = (ime_hint_mask_enabled != 0) && (timeout_ms > 0) &&
            (global_state->last_touch_timestamp != 0) &&
            ((global_state->timestamp - global_state->last_touch_timestamp) < timeout_ms);
        if (ime_hint_active && !ime_hint_mask_armed && gfx_frame_snapshot_bytes(global_state->gfx) > 0) {
            // 置位遮罩标志：分配干净帧快照缓冲（PSRAM，大小由图形层接口给出）并注册刷新钩子
            ime_hint_backup = (uint8_t *)platform_malloc(gfx_frame_snapshot_bytes(global_state->gfx));
            if (ime_hint_backup != NULL) {
                ime_hint_gs = global_state;
                ime_hint_mask_armed = 1;
                gfx_set_refresh_hook(ui_ime_hint_pre_refresh_hook, ui_ime_hint_post_refresh_hook);
                gfx_refresh(global_state->gfx); // 立即推一帧（钩子叠加遮罩），保证触摸即显
            }
        }
        else if (!ime_hint_active && ime_hint_mask_armed) {
            // 超过设定时长无触屏（或机制被外部开关关闭、时长设置为0）：清除遮罩标志并立即推一帧干净画面
            // （帧缓冲在每次推帧后均被后置钩子恢复为干净底图，此处无需重绘控件）
            ui_ime_hint_mask_disarm();
            gfx_refresh(global_state->gfx);
        }
    }

    // 触屏软键盘（文本输入控件固有功能）：
    // 上滑/下滑手势请求切换显隐（Core1手势识别，见 ui_app.c get_key_event；任何输入状态均在此消费）
    if (ui_softkbd_take_toggle_request()) {
        ui_widget_input_toggle_softkbd(key_event, global_state);
    }
    // 软键盘自身状态变化（粘滞修饰键、按下高亮）时，补画键盘并刷新
    if (ui_softkbd_is_visible() && ui_softkbd_take_dirty()) {
        ui_softkbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
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

            ui_draw_input_symbol(key_event, global_state, input_state);

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

                // 在屏幕上循环显示当前选中的字母（每个字母的占位宽度按当前字体逐字符实际宽度计算）
                wchar_t letter[2];
                uint32_t font_id = global_state->ui_font;
                int32_t line_height = gfx_font_line_height(font_id);
                int32_t x_pos = 1;
                int32_t y_pos = ta_y + ta_height - line_height - 2;
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
                input_state->state = 2;
            }
        }

        // 短按A键：取消输入拼音，清除已输入的所有按键，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
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

            input_state->state = 1;
        }
    }

    else if (state == 2) {
        // 短按0-9键：从候选字列表中选定一个字，选定后转到初始状态
        if (key_event->key_edge == -1 && (key_event->key_code >= NANO_KEY_0 && key_event->key_code <= NANO_KEY_9)) {
            uint32_t index = (key_event->key_code == NANO_KEY_0) ? 9 : ((key_event->key_code - '0') - 1); // 按键0对应9
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
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
            if(input_state->current_page > 0) {
                input_state->current_page--;
                ui_draw_input_pinyin(key_event, global_state, input_state, 1);
            }
            input_state->state = 2;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                ui_draw_input_pinyin(key_event, global_state, input_state, 1);
            }
            input_state->state = 2;
        }

        // 短按A键：取消选择，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }
    }

    else if (state == 3) {
        // 短按0-9键：从符号列表中选定一个符号，选定后转到初始状态
        if (key_event->key_edge == -1 && (key_event->key_code >= NANO_KEY_0 && key_event->key_code <= NANO_KEY_9)) {
            uint32_t index = (key_event->key_code == NANO_KEY_0) ? 9 : ((key_event->key_code - '0') - 1); // 按键0对应9
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
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
            if(input_state->current_page > 0) {
                input_state->current_page--;
                ui_draw_input_symbol(key_event, global_state, input_state);
            }
            input_state->state = 3;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                ui_draw_input_symbol(key_event, global_state, input_state);
            }
            input_state->state = 3;
        }

        // 短按A键：取消选择，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            ui_draw_input_buffer(key_event, global_state, input_state);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
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




void ui_widget_menu_init(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    // 菜单位于页眉与页脚之间，页眉/页脚高度与条目行高均跟随当前字体行高
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    int32_t header_height = line_height + 1;
    menu_state->x = 0;
    menu_state->y = header_height;
    menu_state->zindex = 0;
    menu_state->width = global_state->gfx->width;
    menu_state->height = global_state->gfx->height - ui_softkbd_height() - header_height * 2; // 减去header和footer，并为触屏软键盘让出空间
    menu_state->current_item_index = 0;
    menu_state->first_item_intex = 0;
    uint32_t max_items_per_page = (menu_state->height - line_height + 2) / line_height;
    menu_state->items_per_page = (menu_state->item_num > max_items_per_page) ? max_items_per_page : menu_state->item_num;

    // 注意：此处不再立即绘制。此前末尾调用 ui_widget_menu_draw（内含 gfx_refresh）会导致
    // 进入菜单状态时先刷出菜单区（页眉/页脚尚未绘制，残留旧画面）、下一拍状态初始化分支
    // 再补画页眉页脚，形成两阶段断续感。现全部 4 个调用点（model/game/ebook/ofdm 菜单）
    // 均在随后的状态初始化分支统一“页眉页脚+菜单”画齐后一次刷屏（ui_widget_menu_refresh）。
    (void)key_event;
}

void ui_widget_menu_refresh(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    ui_widget_menu_draw(key_event, global_state, menu_state);
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

    // 菜单首行：标题和选项数
    // gfx_draw_textline(global_state->gfx, menu_state->title, x_indent, 0, 0, 255, 255, 1);
    // wchar_t item_counter[13];
    // swprintf(item_counter, 13, L"%d/%d", menu_state->current_item_index + 1, menu_state->item_num);
    // int32_t iclen = wcslen(item_counter);
    // gfx_draw_textline(global_state->gfx, item_counter, (global_state->gfx->width-2) - iclen * 6, 0, 255, 255, 0, 1);

    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
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
            for (uint32_t j = y_pos - 1; j < y_pos + line_height - 1; j++) {
                gfx_draw_line(global_state->gfx, menu_state->x, j, menu_state->x + menu_state->width, j, hl_r, hl_g, hl_b, 1);
            }
        }
        // 绘制文字
        gfx_font_draw_text(global_state->gfx, font_id, (wchar_t *)menu_state->items[i], menu_state->x + x_indent, y_pos, fg_r, fg_g, fg_b, 1);

        y_pos += line_height;
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
        return menu_item_action_callback(ke, gs, ms);
    }
    // 长+短按*键/上键：光标向上移动（上键功能同左键）
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && (ke->key_code == NANO_KEY_left || ke->key_code == NANO_KEY_up)) {
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
    // 长+短按#键/下键：光标向下移动（下键功能同右键）
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && (ke->key_code == NANO_KEY_right || ke->key_code == NANO_KEY_down)) {
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

    uint8_t title_text_R = 0xff;
    uint8_t title_text_G = 0xff;
    uint8_t title_text_B = 0xff;

    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
        title_text_R = 0xff;
        title_text_G = 0xff;
        title_text_B = 0xff;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
        title_text_R = 0x60;
        title_text_G = 0x60;
        title_text_B = 0x60;
    }

    // 底部：触屏软键盘激活且全键盘拼音输入法正在组词时，底栏显示拼音串与候选字；否则显示默认页脚
    if (ui_softkbd_height() > 0 && ui_pinyin_ime_is_composing()) {
        ui_pinyin_ime_draw_bar(global_state);
    }
    else {
        ui_draw_footer(key_event, global_state, L"Ctrl+Shift 使用说明", 1);
    }

    // 顶部
    ui_draw_header(key_event, global_state, L"", 0);
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    int32_t header_text_y = (line_height + 1) / 2 - line_height / 2; // 文本在页眉内垂直居中
    gfx_font_draw_text(global_state->gfx, font_id, input_state->title_text, 0, header_text_y, title_text_R, title_text_G, title_text_B, 1);

    // 右上角状态图标：从右往左按各字符串的实际渲染宽度紧凑排列
    int32_t right_x = (int32_t)global_state->gfx->width - 1;
    // 显示输入状态
    wchar_t *ime_tag = NULL;
    if (input_state->ime_mode_flag == IME_MODE_HANZI)         ime_tag = L"[汉]";
    else if (input_state->ime_mode_flag == IME_MODE_ALPHABET) ime_tag = L"[En]";
    else if (input_state->ime_mode_flag == IME_MODE_NUMBER)   ime_tag = L"[数]";
    if (ime_tag) {
        right_x -= gfx_font_measure_text(font_id, ime_tag);
        gfx_font_draw_text(global_state->gfx, font_id, ime_tag, right_x, header_text_y, 255, 255, 0, 1);
        right_x -= 1;
    }
    // 显示Ctrl激活状态
    if (global_state->is_ctrl_enabled == 1) {
        right_x -= gfx_font_measure_text(font_id, L"◆");
        gfx_font_draw_text(global_state->gfx, font_id, L"◆", right_x, header_text_y, 255, 255, 255, 1);
        right_x -= 1;
    }
    // 显示思考模式启用状态
    if (global_state->is_thinking_enabled == 1) {
        right_x -= gfx_font_measure_text(font_id, L"Ψ");
        gfx_font_draw_text(global_state->gfx, font_id, L"Ψ", right_x, header_text_y, 0, 255, 255, 1);
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

    // 如果光标的视觉行不在当前视图范围内，则滚动视图跟随
    if (cursor_line < ta->current_line) {
        // 光标在当前视图上方：卷到光标所在行
        ta->current_line = cursor_line;
        typeset_view_range(ta, gfx_font_line_height(global_state->ui_font));
    }
    else if (cursor_line > ta->current_line + ta->view_lines - 1) {
        // 光标在当前视图下方：卷到使光标所在行位于视图末行
        //   逻辑上，如果出现这种情况，一定有 line_num > view_lines
        ta->current_line = cursor_line - ta->view_lines + 1;
        typeset_view_range(ta, gfx_font_line_height(global_state->ui_font));
    }

    // 绘制文本
    ui_draw_text_block(key_event, global_state, ta, global_state->ui_font);

    // 绘制滚动条
    if (ta->is_show_scroll_bar) {
        ui_draw_scroll_bar(
            key_event, global_state,
            ta->current_line, ta->line_num, ta->view_lines,
            ta->x, ta->y, ta->width, ta->height);
    }

    // 绘制光标
    ui_draw_input_cursor(key_event, global_state, input_state);

    // 触屏软键盘：可见时绘制在屏幕底部（与文本同帧推出，避免闪烁与二次刷新）
    // CTRL键高亮与全局Ctrl激活状态（is_ctrl_enabled）联动
    if (ui_softkbd_height() > 0) {
        ui_softkbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
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
    }

    uint8_t cursor_R = 0, cursor_G = 0, cursor_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        cursor_R = 0; cursor_G = 0; cursor_B = 64;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        cursor_R = 255; cursor_G = 255; cursor_B = 255;
    }

    uint32_t x = line_x_pos;
    uint32_t y = ta->y + line_height * break_count; // 每行的起始位置是行高的倍数
    gfx_draw_line(global_state->gfx, x, y-1, x, y + line_height - 1, cursor_R, cursor_G, cursor_B, 2);
    gfx_draw_line(global_state->gfx, x+1, y-1, x+1, y + line_height - 1, cursor_R, cursor_G, cursor_B, 2);
}

void ui_draw_input_pinyin(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state, uint32_t is_picking) {
    // gfx_soft_clear(global_state->gfx);
    ui_ime_candidate_color_apply(global_state->ui_color_style);
    // 计算候选列表长度
    uint32_t count = 0;

    for(int j = 0; j < MAX_CANDIDATE_NUM_PER_PAGE; j++) {
        if (!input_state->candidate_pages[input_state->current_page][j]) break;
        count++;
    }

    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    uint32_t x_offset = 1;
    uint32_t y_offset = input_state->textarea.y + input_state->textarea.height - line_height*3 - 1;

    // 清空输入法显示区域
    gfx_draw_rectangle(global_state->gfx,
        input_state->textarea.x, y_offset-1,
        input_state->textarea.width, input_state->textarea.y + input_state->textarea.height - y_offset + 1 + 1,
        S_UI_COLOR_IME_CANDIDATE_BG[0], S_UI_COLOR_IME_CANDIDATE_BG[1], S_UI_COLOR_IME_CANDIDATE_BG[2], 1);

    // 候选序号与候选字的排版：逐字按字符实际渲染宽度定位绘制，每字占1个全角宽度、靠左对齐。
    //   （原先通过空格分隔实现对齐，仅适用于定宽点阵字体；抗锯齿比例字体下空格偏窄会错位）
    int32_t full_width = gfx_font_char_advance(font_id, (uint32_t)L'一'); // 1个全角宽度

    wchar_t buf[30];
    if (is_picking) {
        swprintf(buf, 30, L"PY[%-6d]   (%2d/%2d)", input_state->pinyin_keys, (input_state->current_page+1), input_state->candidate_page_num);
        gfx_font_draw_text(global_state->gfx, font_id, buf, x_offset, y_offset + 0, S_UI_COLOR_IME_CANDIDATE_PINYIN[0], S_UI_COLOR_IME_CANDIDATE_PINYIN[1], S_UI_COLOR_IME_CANDIDATE_PINYIN[2], 1);
        // 候选序号（1~9,0）：逐字靠左绘制
        for (uint32_t j = 0; j < count; j++) {
            gfx_font_draw_char(global_state->gfx, font_id, (j == 9) ? (uint32_t)L'0' : (uint32_t)(L'1' + j),
                x_offset + j * full_width, y_offset + line_height, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
        }
    }
    else {
        swprintf(buf, 30, L"PY[%-6d]", input_state->pinyin_keys);
        gfx_font_draw_text(global_state->gfx, font_id, buf, x_offset, y_offset + 0, S_UI_COLOR_IME_CANDIDATE_PINYIN[0], S_UI_COLOR_IME_CANDIDATE_PINYIN[1], S_UI_COLOR_IME_CANDIDATE_PINYIN[2], 1);
    }
    if (input_state->candidate_num > 0) {
        // 候选字：逐字靠左绘制，每字占1个全角宽度
        for (uint32_t j = 0; j < count; j++) {
            wchar_t ch = input_state->candidate_pages[input_state->current_page][j];
            gfx_font_draw_char(global_state->gfx, font_id, (uint32_t)ch,
                x_offset + j * full_width, y_offset + 2*line_height, S_UI_COLOR_IME_CANDIDATE_TEXT[0], S_UI_COLOR_IME_CANDIDATE_TEXT[1], S_UI_COLOR_IME_CANDIDATE_TEXT[2], 1);
        }
    }
    else {
        gfx_font_draw_text(global_state->gfx, font_id, L"(无候选字)", x_offset, y_offset + 2*line_height, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }

    gfx_draw_line(global_state->gfx, input_state->textarea.x, y_offset-2, input_state->textarea.x + input_state->textarea.width, y_offset-2, S_UI_COLOR_IME_CANDIDATE_BG[0], S_UI_COLOR_IME_CANDIDATE_BG[1], S_UI_COLOR_IME_CANDIDATE_BG[2], 1);

    gfx_refresh(global_state->gfx);
}

void ui_draw_input_symbol(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    // gfx_soft_clear(global_state->gfx);
    ui_ime_candidate_color_apply(global_state->ui_color_style);
    // 计算候选列表长度
    uint32_t count = 0;

    for(int j = 0; j < 10; j++) {
        if (!input_state->candidate_pages[input_state->current_page][j]) break;
        count++;
    }

    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    uint32_t x_offset = 1;
    uint32_t y_offset = input_state->textarea.y + input_state->textarea.height - line_height*3 - 1;

    // 清空输入法显示区域
    gfx_draw_rectangle(global_state->gfx,
        input_state->textarea.x, y_offset-1,
        input_state->textarea.width, input_state->textarea.y + input_state->textarea.height - y_offset + 1 + 1,
        S_UI_COLOR_IME_CANDIDATE_BG[0], S_UI_COLOR_IME_CANDIDATE_BG[1], S_UI_COLOR_IME_CANDIDATE_BG[2], 1);

    // 候选序号与候选符号的排版：逐字按字符实际渲染宽度定位绘制，每字占1个全角宽度、靠左对齐。
    //   （原先半角符号补空格实现对齐，仅适用于定宽点阵字体；抗锯齿比例字体下空格偏窄会错位）
    int32_t full_width = gfx_font_char_advance(font_id, (uint32_t)L'一'); // 1个全角宽度

    wchar_t text[30];

    swprintf(text, 30, L"Symbols      (%2d/%2d)", (input_state->current_page+1), input_state->candidate_page_num);
    gfx_font_draw_text(global_state->gfx, font_id, text, x_offset, y_offset + 0, S_UI_COLOR_IME_CANDIDATE_PINYIN[0], S_UI_COLOR_IME_CANDIDATE_PINYIN[1], S_UI_COLOR_IME_CANDIDATE_PINYIN[2], 1);
    // 候选序号（1~9,0）：逐字靠左绘制
    for (uint32_t j = 0; j < count; j++) {
        gfx_font_draw_char(global_state->gfx, font_id, (j == 9) ? (uint32_t)L'0' : (uint32_t)(L'1' + j),
            x_offset + j * full_width, y_offset + line_height, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }

    if (input_state->candidate_num > 0) {
        // 候选符号：逐字靠左绘制，每字占1个全角宽度（不论全角半角）
        for (uint32_t j = 0; j < count; j++) {
            wchar_t ch = input_state->candidate_pages[input_state->current_page][j];
            gfx_font_draw_char(global_state->gfx, font_id, (uint32_t)ch,
                x_offset + j * full_width, y_offset + 2*line_height, S_UI_COLOR_IME_CANDIDATE_TEXT[0], S_UI_COLOR_IME_CANDIDATE_TEXT[1], S_UI_COLOR_IME_CANDIDATE_TEXT[2], 1);
        }
    }
    else {
        gfx_font_draw_text(global_state->gfx, font_id, L"(无候选符号)", x_offset, y_offset + 2*line_height, S_UI_COLOR_IME_CANDIDATE_INDEX[0], S_UI_COLOR_IME_CANDIDATE_INDEX[1], S_UI_COLOR_IME_CANDIDATE_INDEX[2], 1);
    }

    gfx_draw_line(global_state->gfx, input_state->textarea.x, y_offset-2, input_state->textarea.x + input_state->textarea.width, y_offset-2, S_UI_COLOR_IME_CANDIDATE_BG[0], S_UI_COLOR_IME_CANDIDATE_BG[1], S_UI_COLOR_IME_CANDIDATE_BG[2], 1);

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


