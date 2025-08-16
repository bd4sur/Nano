#include <stdio.h>
#include <time.h>

#include "ui.h"
#include "oledfont.h"
#include "oled.h"



void text_typeset(
    uint32_t *text,          // in  待排版的文本
    int32_t view_width,      // in  视图宽度
    int32_t view_height,     // in  视图高度
    int32_t start_line,      // in  从哪行开始显示（用于滚动）
    int32_t *length,         // out 文本长度（字符数）
    int32_t *break_pos,      // out 折行位置（每行第一个字符的index）数组
    int32_t *view_lines,     // out 可见行数
    int32_t *view_start_pos, // out 可见区域第一个字符的index
    int32_t *view_end_pos    // out 可见区域最后一个字符的index
) {
    int32_t char_count = 0;
    int32_t break_count = 0;
    int32_t line_x_pos = 0;
    for (char_count = 0; char_count < wcslen(text); char_count++) {
        wchar_t ch = text[char_count];
        int32_t char_width = (ch < 127) ? ((ch == '\n') ? 0 : FONT_WIDTH_HALF) : FONT_WIDTH_FULL;
        if (char_count == 0 || line_x_pos + char_width >= view_width) {
            break_pos[break_count] = char_count;
            break_count++;
            line_x_pos = 0;
        }
        else if (ch == '\n') {
            break_pos[break_count] = char_count + 1;
            break_count++;
            line_x_pos = 0;
        }
        line_x_pos += char_width;
    }

    // 计算当前视图最大能容纳的行数。
    //   NOTE 考虑到行间距为1，且末行以下无间距，因此分子加1以去除末行无间距的影响。
    //        例如，高度为64的屏幕，实际可容纳(64+1)/(12+1)=5行。
    int32_t max_view_lines = (view_height + 1) / (FONT_HEIGHT + 1);

    int32_t _view_lines = break_count;
    *view_lines =_view_lines;
    *length = char_count;

    // 对start_line的检查和标准化
    if (start_line < 0) {
        // start_line小于0，解释为将文字末行卷动到视图的某一行。例如：-1代表将文字末行卷动到视图的倒数1行、-max_view_lines代表将文字末行卷动到视图的第1行。
        //   若start_line小于-max_view_lines，则等效于-max_view_lines，保证文字内容不会卷到视图以外。
        if (-start_line <= max_view_lines) {
            if (_view_lines >= max_view_lines) {
                start_line = _view_lines - 1 - start_line - max_view_lines;
            }
            else {
                start_line = 0;
            }
        }
        else {
            start_line = _view_lines - 1;
        }
    }
    else if (start_line >= _view_lines) {
        // start_line超过了末行，则对文本行数取模后滚动
        start_line = start_line % _view_lines;
    }

    // 情况1：start_line介于首行（0）和（使得末行进入可见区域以下1行的位置），即视图内不包含末行
    if (start_line < _view_lines - max_view_lines) {
        *view_start_pos = break_pos[start_line];
        *view_end_pos = break_pos[start_line + max_view_lines] - 1;
    }
    // 情况2：start_line等于或超过了（使得末行恰好位于可见区域底行的位置），但尚未超出末行，也就是末行位于视图内
    //        若文本行数不大于视图行数，则一定满足此条件。
    else if (start_line >= _view_lines - max_view_lines && start_line < _view_lines) {
        *view_start_pos = break_pos[start_line];
        *view_end_pos = char_count - 1;
    }
}




// 渲染一行文本，mode为1则为正显，为0则为反白
void render_line(wchar_t *line, uint32_t x, uint32_t y, uint8_t mode) {
    uint32_t x_pos = x;
    uint32_t y_pos = y;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            printf("出现了字库之外的字符！\n");
            break;
        }
        if (x_pos + font_width >= 128) {
            break;
        }
        OLED_ShowChar(x_pos, y_pos, glyph, font_width, font_height, (mode % 2));
        x_pos += font_width;
    }
}

// 返回值：文本折行后的行数（含换行符）
int32_t render_text(wchar_t *text, int32_t start_line) {
    int32_t length = 0;
    int32_t break_pos[STRING_BUFFER_LENGTH];
    int32_t view_lines = 0;
    int32_t view_start_pos = 0;
    int32_t view_end_pos = 0;

    text_typeset(text, 128, 64, start_line, &length, break_pos, &view_lines, &view_start_pos, &view_end_pos);

    int x_pos = 0;
    int y_pos = 0;
    // for (int i = 0; i < wcslen(wrapped_clipped); i++) {
    for (int i = view_start_pos; i <= view_end_pos; i++) {
        uint32_t current_char = text[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        if (current_char == '\n') {
            x_pos = 0;
            if(i > 0) y_pos += (font_height + 1);
            continue;
        }
        uint8_t *glyph = get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            printf("出现了字库之外的字符[%d]\n", current_char);
            glyph = get_glyph(12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos + font_width >= 128) {
            y_pos += (font_height + 1);
            x_pos = 0;
        }
        OLED_ShowChar(x_pos, y_pos, glyph, font_width, font_height, 1);
        x_pos += font_width;
    }

    // free(wrapped);
    // free(wrapped_clipped);

    return view_lines;
}



void show_splash_screen() {
    OLED_SoftClear();

    time_t rawtime;
    struct tm *timeinfo;
    char datetime_string_buffer[80];
    wchar_t datetime_wcs_buffer[80];

    time(&rawtime); // 获取当前时间戳
    timeinfo = localtime(&rawtime); // 转换为本地时间
    strftime(datetime_string_buffer, sizeof(datetime_string_buffer), "%Y-%m-%d %H:%M:%S", timeinfo); // 格式化输出
    mbstowcs(datetime_wcs_buffer, datetime_string_buffer, 80);

    for (int i = 1; i <= 14; i++) {
        OLED_DrawLine(1, i, 127, i, 1);
    }

    render_line(L"Project MARGA!", 24, 2, 0);
    render_line(L"完全离线电子鹦鹉", 16, 20, 1);
    // render_line(L"自研Nano模型强力驱动", 4, 34, 1);
    render_line(datetime_wcs_buffer, 8, 34, 1);
    render_line(L"(c) 2025 BD4SUR", 18, 50, 1);

    OLED_DrawLine(0, 0, 127, 0, 1);
    OLED_DrawLine(0, 15, 127, 15, 1);
    OLED_DrawLine(0, 0, 0, 63, 1);
    OLED_DrawLine(127, 0, 127, 63, 1);
    OLED_DrawLine(0, 63, 127, 63, 1);

    OLED_Refresh();
}

void show_main_menu() {
    OLED_SoftClear();
    render_text(L"1.人类的本质\n2.电子鹦鹉\n3.设置", 0);
    OLED_Refresh();
}

int32_t render_input_buffer(uint32_t *input_buffer, uint32_t ime_mode_flag, int32_t cursor_pos) {
    OLED_SoftClear();
    wchar_t text[INPUT_BUFFER_LENGTH] = L"请输入问题：     [";
    if (ime_mode_flag == IME_MODE_HANZI) {
        wcscat(text, L"汉]\n");
    }
    else if (ime_mode_flag == IME_MODE_ALPHABET) {
        wcscat(text, L"En]\n");
    }
    else if (ime_mode_flag == IME_MODE_NUMBER) {
        wcscat(text, L"数]\n");
    }
    wcscat(text, input_buffer);
    // if (cursor_pos) wcscat(text, L"_");
    render_text(text, 0);

    // 绘制光标

    int32_t char_count = 0;
    int32_t break_count = 0;
    int32_t line_x_pos = 0;
    for (char_count = 0; char_count < wcslen(input_buffer); char_count++) {
        wchar_t ch = input_buffer[char_count];
        int32_t char_width = (ch < 127) ? ((ch == '\n') ? 0 : 6) : 12;
        if (line_x_pos + char_width >= 128 || ch == '\n') {
            break_count++;
            line_x_pos = 0;
        }
        line_x_pos += char_width;
        if (cursor_pos == char_count) break;
    }

    uint8_t x = line_x_pos;
    uint8_t y = (uint8_t)(13 * (break_count + 1)); // 12x12字模底部本来就有1px的空白，加上行间距1px，所以每行的起始位置是13的倍数
    OLED_DrawLine(x, y-1, x, y+12, 1);
    OLED_DrawLine(x+1, y-1, x+1, y+12, 1);

    OLED_Refresh();

    return char_count;
}

void render_pinyin_input(uint32_t **candidate_pages, uint32_t pinyin_keys, uint32_t current_page, uint32_t candidate_page_num, uint32_t is_picking) {
    OLED_SoftClear();
    // 计算候选列表长度
    uint32_t candidate_num = 0;
    wchar_t cc[11];
    wchar_t cindex[21] = L"1 2 3 4 5 6 7 8 9 0 ";
    if (candidate_pages) {
        for(int j = 0; j < 10; j++) {
            wchar_t ch = candidate_pages[current_page][j];
            if (!ch) break;
            cc[j] = ch;
            candidate_num++;
        }
        cc[candidate_num] = 0;
        cindex[candidate_num << 1] = 0;
    }

    wchar_t text[INPUT_BUFFER_LENGTH];
    if (is_picking) {
        swprintf(text, INPUT_BUFFER_LENGTH, L" \n\nPY[%-6d]   (%2d/%2d)\n", pinyin_keys, (current_page+1), candidate_page_num);
        wcscat(text, cindex);
        wcscat(text, L"\n");
    }
    else {
        swprintf(text, INPUT_BUFFER_LENGTH, L" \n\nPY[%-6d]\n\n", pinyin_keys);
    }
    if (candidate_pages) {
        wcscat(text, cc);
    }
    else {
        wcscat(text, L"(无候选字)");
    }
    render_text(text, 0);
    OLED_Refresh();
}

void render_symbol_input(uint32_t **candidate_pages, uint32_t current_page, uint32_t candidate_page_num) {
    OLED_SoftClear();
    // 计算候选列表长度
    uint32_t candidate_num = 0;
    uint32_t list_char_width = 0;
    wchar_t cc[21];
    wchar_t cindex[21] = L"1 2 3 4 5 6 7 8 9 0 ";
    if (candidate_pages) {
        for(int j = 0; j < 10; j++) {
            wchar_t ch = candidate_pages[current_page][j];
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
            candidate_num++;
        }
        cc[list_char_width] = 0;
        cindex[candidate_num << 1] = 0;
    }

    wchar_t text[INPUT_BUFFER_LENGTH];
    swprintf(text, INPUT_BUFFER_LENGTH, L" \n\nSymbols      (%2d/%2d)\n", (current_page+1), candidate_page_num);
    wcscat(text, cindex);
    wcscat(text, L"\n");

    if (candidate_pages) {
        wcscat(text, cc);
    }
    else {
        wcscat(text, L"(无候选符号)");
    }
    render_text(text, 0);
    OLED_Refresh();
}

void render_scroll_bar(int32_t line_num, int32_t current_line) {
    for (int y = 0; y < 64; y++) {
        OLED_DrawPoint(127, y, !(y % 3));
    }
    uint8_t bar_height = (uint8_t)((5 * 64) / line_num);
    uint8_t y_0 = (uint8_t)((current_line * 64) / line_num);
    OLED_DrawLine(127, y_0, 127, y_0 + bar_height, 1);
}

uint32_t *refresh_input_buffer(uint32_t *input_buffer, int32_t *input_counter) {
    if (input_counter) *input_counter = 0;
    uint32_t *new_input_buffer = (uint32_t *)realloc(input_buffer, INPUT_BUFFER_LENGTH * sizeof(uint32_t));
    if (!new_input_buffer) {
        free(input_buffer);
        return (uint32_t *)calloc(INPUT_BUFFER_LENGTH, sizeof(uint32_t));
    }
    else {
        for (uint32_t i = 0; i < INPUT_BUFFER_LENGTH; i++) new_input_buffer[i] = 0;
        return new_input_buffer;
    }
}
