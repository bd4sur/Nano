#include <stdio.h>
#include <time.h>

#include "ui.h"
#include "pinyin.h"
#include "oledfont.h"
#include "oled.h"


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
    if (input_state->length + 1 > INPUT_BUFFER_LENGTH) {
        return;
    }

    input_state->text[input_state->length + 1] = L'\0';

    for (uint32_t i = input_state->length; i >= input_state->cursor_pos + 2; i--) {
        input_state->text[i] = input_state->text[i-1];
    }
    input_state->text[input_state->cursor_pos + 1] = new_char;

    input_state->cursor_pos++;
    input_state->length++;
}

// 删除光标位置的字符（即光标竖线左边的一个字符）
void delete_char(Widget_Input_State *input_state) {
    if (input_state->length <= 0 || input_state->cursor_pos < 0) {
        return;
    }

    for (uint32_t i = input_state->cursor_pos; i < input_state->length; i++) {
        input_state->text[i] = input_state->text[i+1];
    }
    input_state->text[input_state->length - 1] = L'\0';

    input_state->cursor_pos--;
    input_state->length--;
}



void text_typeset(
    uint32_t *text,          // in  待排版的文本
    int32_t view_width,      // in  视图宽度
    int32_t view_height,     // in  视图高度
    int32_t start_line,      // in  从哪行开始显示（用于滚动）
    int32_t *length,         // out 文本长度（字符数）
    int32_t *break_pos,      // out 折行位置（每行第一个字符的index）数组
    int32_t *line_num,     // out 可见行数
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

    int32_t _line_num = break_count;
    *line_num = break_count;
    *view_lines = max_view_lines;
    *length = char_count;

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
        *view_start_pos = break_pos[start_line];
        *view_end_pos = break_pos[start_line + max_view_lines] - 1;
    }
    // 情况2：start_line等于或超过了（使得末行恰好位于可见区域底行的位置），但尚未超出末行，也就是末行位于视图内
    //        若文本行数不大于视图行数，则一定满足此条件。
    else if (start_line >= _line_num - max_view_lines && start_line < _line_num) {
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
        // NOTE 反色显示时，在每个字符场面额外补充一条线，避免菜单中高亮区域看起来顶格
        OLED_DrawLine(x_pos, y_pos - 1, x_pos+font_width-1, y_pos - 1, 1 - (mode % 2));
        OLED_ShowChar(x_pos, y_pos, glyph, font_width, font_height, (mode % 2));
        x_pos += font_width;
    }
}

// 返回值：文本折行后的行数（含换行符）
void render_text(wchar_t *text, int32_t start_line, int32_t x_offset, int32_t y_offset, int32_t width, int32_t height) {
    int32_t length = 0;
    int32_t break_pos[STRING_BUFFER_LENGTH];
    int32_t line_num = 0;
    int32_t view_lines = 0;
    int32_t view_start_pos = 0;
    int32_t view_end_pos = 0;

    text_typeset(text, width, height, start_line, &length, break_pos, &line_num, &view_lines, &view_start_pos, &view_end_pos);

    int x_pos = x_offset;
    int y_pos = y_offset;

    for (int i = view_start_pos; i <= view_end_pos; i++) {
        uint32_t current_char = text[i];
        if (!current_char) break;
        uint8_t font_width = FONT_WIDTH_FULL;
        uint8_t font_height = FONT_HEIGHT;
        if (current_char == '\n') {
            x_pos = x_offset;
            if(i > 0) y_pos += (font_height + 1);
            continue;
        }
        uint8_t *glyph = get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            printf("出现了字库之外的字符[%d]\n", current_char);
            glyph = get_glyph(12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos + font_width >= x_offset + width) {
            y_pos += (font_height + 1);
            x_pos = x_offset;
        }
        OLED_ShowChar(x_pos, y_pos, glyph, font_width, font_height, 1);
        x_pos += font_width;
    }

    // free(wrapped);
    // free(wrapped_clipped);
}

// 绘制滚动条
//   line_num - 文本总行数
//   current_line - 当前在屏幕顶端的是哪一行
//   view_lines - 屏幕最多容纳几行
void render_scroll_bar(int32_t current_line, int32_t line_num, int32_t view_lines, int32_t x, int32_t y, int32_t width, int32_t height) {

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
        current_line = current_line % line_num;
    }

    for (int n = y; n < y + height; n++) {
        OLED_DrawPoint(x + width - 1, n, !(n % 3));
    }
    // 如果总行数装不满视图，则滚动条长度等于视图高度height
    uint8_t bar_height = (line_num < view_lines) ? (uint8_t)(height) : (uint8_t)((view_lines * height) / line_num);
    // 进度条高度不小于3px
    bar_height = (bar_height < 3) ? 3 : bar_height;
    uint8_t y_0 = (uint8_t)(y + (current_line * height) / line_num);
    OLED_DrawLine(x + width - 1, y_0, x + width - 1, y_0 + bar_height + 1, 1);
    OLED_DrawLine(x + width - 2, y_0, x + width - 2, y_0 + bar_height + 1, 1);
}









void show_splash_screen(Key_Event *key_event, Global_State *global_state) {
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

    render_line(L"Project Nano", 28, 2, 0);
    render_line(L"语音对话电子鹦鹉", 16, 20, 1);
    // render_line(L"自研Nano模型强力驱动", 4, 34, 1);
    render_line(datetime_wcs_buffer, 8, 34, 1);
    render_line(L"(c) 2025 BD4SUR", 18, 50, 1);

    OLED_DrawLine(0, 0, 127, 0, 1);
    OLED_DrawLine(0, 15, 127, 15, 1);
    OLED_DrawLine(0, 0, 0, 63, 1);
    OLED_DrawLine(127, 0, 127, 63, 1);
    OLED_DrawLine(0, 63, 127, 63, 1);

    // 检查ASR服务状态，如果ASR服务未启动，则在屏幕左上角画一个闪烁的点，表示ASR服务启动中
    if (global_state->is_asr_server_up < 1) {
        uint8_t v = (uint8_t)((global_state->timer >> 2) & 0x1);
        OLED_DrawLine(4, 6, 7, 6, v);
        OLED_DrawLine(4, 7, 7, 7, v);
        OLED_DrawLine(4, 8, 7, 8, v);
        OLED_DrawLine(4, 9, 7, 9, v);
    }

    // 绘制电池电量
    OLED_DrawLine(112, 4, 125, 4, 0);
    OLED_DrawLine(125, 4, 125, 11, 0);
    OLED_DrawLine(112, 11, 125, 11, 0);
    OLED_DrawLine(112, 4, 112, 11, 0);
    OLED_DrawLine(111, 6, 111, 9, 0);

    int32_t soc_bar_length = (int32_t)(10.0f * ((float)global_state->ups_soc / 100.0f));
    soc_bar_length = (soc_bar_length > 9) ? 9 : soc_bar_length;
    OLED_DrawLine(123 - soc_bar_length, 6, 123, 6, 0);
    OLED_DrawLine(123 - soc_bar_length, 7, 123, 7, 0);
    OLED_DrawLine(123 - soc_bar_length, 8, 123, 8, 0);
    OLED_DrawLine(123 - soc_bar_length, 9, 123, 9, 0);

    OLED_Refresh();
}






void draw_textarea(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state) {
    text_typeset(
        textarea_state->text,
        textarea_state->width,
        textarea_state->height,
        textarea_state->current_line,
        &(textarea_state->length),
        textarea_state->break_pos,
        &(textarea_state->line_num),
        &(textarea_state->view_lines),
        &(textarea_state->view_start_pos),
        &(textarea_state->view_end_pos)
    );

    if (global_state->is_full_refresh) {
        OLED_SoftClear();
    }

    render_text(
        textarea_state->text, textarea_state->current_line,
        textarea_state->x, textarea_state->y, textarea_state->width, textarea_state->height);

    if (textarea_state->is_show_scroll_bar) {
        render_scroll_bar(
            textarea_state->current_line, textarea_state->line_num, textarea_state->view_lines,
            textarea_state->x, textarea_state->y, textarea_state->width, textarea_state->height);
    }

    if (global_state->is_full_refresh) {
        OLED_Refresh();
    }
}

void init_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    input_state->state = 0;

    input_state->x = 0;
    input_state->y = 13;
    input_state->width = 128;
    input_state->height = 51; // NOTE 详见结构体定义处的说明
    // input_state->text[INPUT_BUFFER_LENGTH];
    input_state->length = 0;
    // input_state->break_pos[INPUT_BUFFER_LENGTH];
    input_state->line_num = 0;
    input_state->view_lines = 0;
    input_state->view_start_pos = 0;
    input_state->view_end_pos = 0;
    input_state->current_line = 0;
    input_state->is_show_scroll_bar = 1;

    input_state->cursor_pos = -1;
    input_state->ime_mode_flag = IME_MODE_HANZI;
    input_state->pinyin_keys = 0;
    input_state->candidate_num = 0;
    input_state->candidate_page_num = 0;
    input_state->current_page = 0;
    input_state->alphabet_countdown = -1;
    input_state->alphabet_current_key = 255;
    input_state->alphabet_index = 0;

    // 初始化各个数组
    wcscpy(input_state->text, L"");
    memset(input_state->break_pos, 0, sizeof(input_state->break_pos));
    memset(input_state->candidates, 0, sizeof(input_state->candidates));
    memset(input_state->candidate_pages, 0, sizeof(input_state->candidate_pages));

    render_input_buffer(key_event, global_state, input_state);
}

void refresh_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    render_input_buffer(key_event, global_state, input_state);
}

void draw_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {

    int32_t state = input_state->state;

    // 定时器触发：字母输入的倒计时进度条
    if (input_state->ime_mode_flag == IME_MODE_ALPHABET) {
        // 倒计时进行中，绘制进度条
        if (input_state->alphabet_countdown > 0) {
            input_state->alphabet_countdown--;
            uint8_t x_pos = (uint8_t)(input_state->alphabet_countdown * 128 / ALPHABET_COUNTDOWN_MAX);
            OLED_DrawLine(0, 63, x_pos, 63, 1);
            OLED_DrawLine(x_pos + 1, 63, 127, 63, 0);
            OLED_Refresh();
            input_state->state = 0;
        }
        // 倒计时结束，提交当前选中的字母，清除进度条
        else if (input_state->alphabet_countdown == 0) {
            // 清除进度条
            input_state->alphabet_countdown--;
            OLED_DrawLine(0, 63, 127, 63, 0);
            OLED_Refresh();

            // 将当前选中的字母加入输入缓冲区
            uint32_t ch = ime_alphabet[(int)(input_state->alphabet_current_key)][input_state->alphabet_index];
            if (ch) {
                // input_state->text[input_state->length++] = ch;
                // input_state->cursor_pos++;
                insert_char(input_state, ch);
            }
            else {
                printf("选定了列表之外的字母，忽略。\n");
            }

            render_input_buffer(key_event, global_state, input_state);

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

            render_symbol_input(input_state);

            input_state->current_page = 0;
            input_state->state = 3;
        }

        // 短按0：数字输入模式下是直接输入0，其余模式无动作
        else if (key_event->key_edge == -1 && key_event->key_code == 0) {
            if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
                // input_state->text[(input_state->length)++] = L'0';
                // input_state->cursor_pos++;
                insert_char(input_state, L'0');
                render_input_buffer(key_event, global_state, input_state);
                input_state->state = 0;
            }
        }

        // 短按1-9：输入拼音/字母/数字，根据输入模式标志，转向不同的状态
        else if (key_event->key_edge == -1 && (key_event->key_code >= 1 && key_event->key_code <= 9)) {
            if (input_state->ime_mode_flag == IME_MODE_HANZI) {
                if (key_event->key_code >= 2 && key_event->key_code <= 9) { // 仅响应按键2-9；1无动作
                    input_state->state = 1;
                    // goto STATE_1;
                    draw_input(key_event, global_state, input_state);
                }
            }
            else if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
                // input_state->text[(input_state->length)++] = L'0' + key_event->key_code;
                // input_state->cursor_pos++;
                insert_char(input_state, (wchar_t)(L'0' + key_event->key_code));
                render_input_buffer(key_event, global_state, input_state);
                input_state->state = 0;
            }
            else if (input_state->ime_mode_flag == IME_MODE_ALPHABET) {
                // 如果按键按下时，不是字母切换状态，则开始循环切换，并开始倒计时。
                if (input_state->alphabet_countdown == -1) {
                    input_state->alphabet_countdown = ALPHABET_COUNTDOWN_MAX;
                    input_state->alphabet_current_key = key_event->key_code;
                    input_state->alphabet_index = 0;
                }
                // 如果按键按下时，倒计时尚未结束，则切换到下一个字母。
                else if (input_state->alphabet_countdown > 0) {
                    input_state->alphabet_countdown = ALPHABET_COUNTDOWN_MAX;
                    input_state->alphabet_current_key = key_event->key_code;
                    input_state->alphabet_index = (input_state->alphabet_index + 1) % wcslen(ime_alphabet[(int)(key_event->key_code)]);
                }

                // 在屏幕上循环显示当前选中的字母
                wchar_t letter[2];
                uint32_t x_pos = 1;
                for (int i = 0; i < wcslen(ime_alphabet[(int)(key_event->key_code)]); i++) {
                    letter[0] = ime_alphabet[(int)(key_event->key_code)][i]; letter[1] = 0;
                    render_line(letter, x_pos, 50, (i != input_state->alphabet_index));
                    x_pos += 8;
                }

                input_state->state = 0;
            }
        }

        // 长+短按A键：删除一个字符；如果输入缓冲区为空，则回到主菜单（在焦点转换部分处理）
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
            if (input_state->length >= 1) {
                // input_state->text[--(input_state->length)] = 0;
                // input_state->cursor_pos--;
                delete_char(input_state);
                render_input_buffer(key_event, global_state, input_state);
            }
            input_state->state = 0;
        }

        // 长+短按B键：依次切换汉-英-数输入模式
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 11) {
            input_state->ime_mode_flag = (input_state->ime_mode_flag + 1) % 3;
            render_input_buffer(key_event, global_state, input_state);
            input_state->state = 0;
        }

        // 长+短按*键：光标向左移动
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
            if (input_state->cursor_pos > -1) {
                input_state->cursor_pos--;
            }
            else {
                input_state->cursor_pos = -1;
            }
            render_input_buffer(key_event, global_state, input_state);
        }

        // 长+短按#键：光标向右移动
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
            if (input_state->cursor_pos < input_state->length - 1) {
                input_state->cursor_pos++;
            }
            else {
                input_state->cursor_pos = input_state->length - 1;
            }
            render_input_buffer(key_event, global_state, input_state);
        }

        // 无按键：光标闪烁
        else {
            if (global_state->timer % 120 == 0) {
                render_cursor(key_event, global_state, input_state);
                OLED_Refresh();
            }
        }
    }

    else if (state == 1) {
        // 短按D键：开始选字
        if (key_event->key_edge == -1 && key_event->key_code == 13) {
            if (input_state->candidate_num > 0) {
                render_pinyin_input(input_state, 1);
                input_state->state = 2;
            }
        }

        // 短按A键：取消输入拼音，清除已输入的所有按键，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            render_input_buffer(key_event, global_state, input_state);
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
                render_pinyin_input(input_state, 0);
            }
            else {
                render_pinyin_input(input_state, 0);
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

            render_input_buffer(key_event, global_state, input_state);

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
                render_pinyin_input(input_state, 1);
            }
            input_state->state = 2;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                render_pinyin_input(input_state, 1);
            }
            input_state->state = 2;
        }

        // 短按A键：取消选择，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            render_input_buffer(key_event, global_state, input_state);
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
            render_input_buffer(key_event, global_state, input_state);

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
                render_symbol_input(input_state);
            }
            input_state->state = 3;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                render_symbol_input(input_state);
            }
            input_state->state = 3;
        }

        // 短按A键：取消选择，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            render_input_buffer(key_event, global_state, input_state);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }
    }
}




void init_menu(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    menu_state->current_item_intex = 0;
    menu_state->first_item_intex = 0;
    menu_state->items_per_page = 4;

    draw_menu(key_event, global_state, menu_state);
}

void refresh_menu(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    draw_menu(key_event, global_state, menu_state);
}

void draw_menu(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {

    uint32_t x_indent = 6;

    OLED_SoftClear();

    render_line(menu_state->title, x_indent, 0, 1);
    wchar_t item_counter[13];
    swprintf(item_counter, 13, L"%d/%d", menu_state->current_item_intex + 1, menu_state->item_num);
    int32_t iclen = wcslen(item_counter);
    render_line(item_counter, 126 - iclen * 6, 0, 1);

    uint32_t y_pos = 13;
    uint8_t is_highlight = 0;
    for (uint32_t i = menu_state->first_item_intex; i < menu_state->item_num; i++) {
        if (i == menu_state->first_item_intex + menu_state->items_per_page) {
            break;
        }
        if (i != menu_state->current_item_intex) {
            is_highlight = 0;
        }
        else {
            is_highlight = 1;
        }
        render_line(menu_state->items[i], x_indent, y_pos, (1 - is_highlight));
        y_pos += (FONT_HEIGHT + 1);
    }

    // NOTE 因render_line会额外给文字上方增加一行，因此这个横线在菜单文字绘制之后再绘制
    OLED_DrawLine(0, 12, 128, 12, 1);

    OLED_Refresh();
}














void render_input_buffer(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {

    OLED_SoftClear();

    wchar_t prompt[INPUT_BUFFER_LENGTH] = L"请输入           [";
    if (input_state->ime_mode_flag == IME_MODE_HANZI) {
        wcscat(prompt, L"汉]\n");
    }
    else if (input_state->ime_mode_flag == IME_MODE_ALPHABET) {
        wcscat(prompt, L"En]\n");
    }
    else if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
        wcscat(prompt, L"数]\n");
    }
    render_line(prompt, 0, 0, 0);

    // 绘制右侧未填满的两列像素
    OLED_DrawLine(126, 0, 126, 12, 1);
    OLED_DrawLine(127, 0, 127, 12, 1);

    // 第一次排版：用于判断光标是否在视图内部
    // input_state->current_line = 0;
    text_typeset(
        input_state->text,
        input_state->width,
        input_state->height,
        input_state->current_line,
        &(input_state->length),
        input_state->break_pos,
        &(input_state->line_num),
        &(input_state->view_lines),
        &(input_state->view_start_pos),
        &(input_state->view_end_pos)
    );

    // 如果光标不在当前视图范围内
    if (input_state->cursor_pos < input_state->view_start_pos || input_state->cursor_pos > input_state->view_end_pos) {
        uint32_t cursor_line = 0;
        // 寻找当前光标所在的行
        for (int32_t i = 0; i < input_state->line_num; i++) {
            int32_t a = input_state->break_pos[i];
            int32_t b = (i == input_state->line_num - 1) ? input_state->length : input_state->break_pos[i+1];
            if (input_state->cursor_pos >= a && input_state->cursor_pos < b) {
                cursor_line = i;
            }
        }

        // 如果光标在当前视图上方，则将current_line设为当前光标所在的行
        if (input_state->cursor_pos < input_state->view_start_pos) {
            input_state->current_line = cursor_line;
        }
        // 如果光标在当前视图下方，则将current_line设为当前光标所在行上方view_lines行（即，使得光标所在行位于视图的末行）
        //   逻辑上，如果出现这种情况，一定有 line_num > view_lines
        else {
            input_state->current_line = cursor_line - input_state->view_lines + 1;
        }
        // 重新排版
        text_typeset(
            input_state->text,
            input_state->width,
            input_state->height,
            input_state->current_line,
            &(input_state->length),
            input_state->break_pos,
            &(input_state->line_num),
            &(input_state->view_lines),
            &(input_state->view_start_pos),
            &(input_state->view_end_pos)
        );
    }

    // 绘制文本
    render_text(input_state->text, input_state->current_line, input_state->x, input_state->y, input_state->width, input_state->height);

    // 绘制滚动条
    if (input_state->is_show_scroll_bar) {
        render_scroll_bar(
            input_state->current_line, input_state->line_num, input_state->view_lines,
            input_state->x, input_state->y, input_state->width, input_state->height);
    }

    // 绘制光标
    render_cursor(key_event, global_state, input_state);

    OLED_Refresh();
}


void render_cursor(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    // 绘制光标：光标位置在cursor_pos所指字符的右外边缘
    int32_t char_index = 0;
    int32_t break_count = 0;
    int32_t line_x_pos = input_state->x;
    if (input_state->cursor_pos >= 0) {
        for (char_index = input_state->view_start_pos; char_index <= input_state->view_end_pos; char_index++) {
            wchar_t ch = input_state->text[char_index];
            int32_t char_width = (ch < 127) ? ((ch == '\n') ? 0 : 6) : 12;
            if (line_x_pos + char_width >= input_state->x + input_state->width || ch == '\n') {
                break_count++;
                line_x_pos = input_state->x;
            }
            line_x_pos += char_width;
            if (input_state->cursor_pos == char_index) break;
        }
    }

    uint8_t x = line_x_pos;
    uint8_t y = (uint8_t)(input_state->y + 13 * break_count); // 12x12字模底部本来就有1px的空白，加上行间距1px，所以每行的起始位置是13的倍数
    OLED_DrawLine(x, y-1, x, y+12, 2);
    OLED_DrawLine(x+1, y-1, x+1, y+12, 2);
}

void render_pinyin_input(Widget_Input_State *input_state, uint32_t is_picking) {
    OLED_SoftClear();
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


    wchar_t buf[INPUT_BUFFER_LENGTH];
    if (is_picking) {
        swprintf(buf, INPUT_BUFFER_LENGTH, L" \nPY[%-6d]   (%2d/%2d)\n", input_state->pinyin_keys, (input_state->current_page+1), input_state->candidate_page_num);
        wcscat(buf, cindex);
        wcscat(buf, L"\n");
    }
    else {
        swprintf(buf, INPUT_BUFFER_LENGTH, L" \nPY[%-6d]\n\n", input_state->pinyin_keys);
    }
    if (input_state->candidate_num > 0) {
        wcscat(buf, cc);
    }
    else {
        wcscat(buf, L"(无候选字)");
    }
    render_text(buf, 0, input_state->x, input_state->y, input_state->width, input_state->height);
    OLED_Refresh();
}

void render_symbol_input(Widget_Input_State *input_state) {
    OLED_SoftClear();
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

    wchar_t text[INPUT_BUFFER_LENGTH];
    swprintf(text, INPUT_BUFFER_LENGTH, L" \nSymbols      (%2d/%2d)\n", (input_state->current_page+1), input_state->candidate_page_num);
    wcscat(text, cindex);
    wcscat(text, L"\n");

    if (input_state->candidate_num > 0) {
        wcscat(text, cc);
    }
    else {
        wcscat(text, L"(无候选符号)");
    }
    render_text(text, 0, input_state->x, input_state->y, input_state->width, input_state->height);
    OLED_Refresh();
}
