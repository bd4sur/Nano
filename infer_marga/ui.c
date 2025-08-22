#include <stdio.h>
#include <time.h>

#include "ui.h"
#include "pinyin.h"
#include "oledfont.h"
#include "oled.h"


unsigned int *candidate_hanzi_list(unsigned int keys, uint32_t *candidate_num) {
    unsigned int candidate_index[500];
    int candidate_count = 0;
    for(int i = 0; i < IME_HANZI_NUM; i++) {
        if(KEYS_LIST[i] == keys) {
            candidate_index[candidate_count++] = i;
        }
    }
    if(candidate_count == 0) {
        return NULL;
    }
    else {
        unsigned int *candidates = (unsigned int *)calloc(candidate_count, sizeof(unsigned int));
        for(int i = 0; i < candidate_count; i++) {
            candidates[i] = UTF32_LIST[candidate_index[i]];
        }
        *candidate_num = candidate_count;
        return candidates;
    }
}

uint32_t **candidate_paging(uint32_t *candidates, uint32_t candidate_num, uint32_t page_length, uint32_t *pages) {
    if(!pages) return NULL;
    *pages = candidate_num / page_length + ((candidate_num % page_length) ? 1 : 0);
    uint32_t page_num = *pages;
    uint32_t **candidate_pages = (uint32_t **)calloc(page_num, sizeof(uint32_t *));
    uint32_t pos = 0;
    for (uint32_t i = 0; i < page_num; i++) {
        uint32_t *page = (uint32_t *)calloc(page_length, sizeof(uint32_t));
        for (uint32_t j = 0; j < page_length; j++) {
            page[j] = (pos < candidate_num) ? candidates[pos] : 0; // 选字时，选到0就意味着越界了
            pos++;
        }
        candidate_pages[i] = page;
    }
    return candidate_pages;
}

void free_candidate_pages(uint32_t **candidate_pages, uint32_t pages) {
    if (!candidate_pages) return;
    for (uint32_t i = 0; i < pages; i++) {
        free(candidate_pages[i]);
    }
    free(candidate_pages);
}




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

int32_t get_view_lines(wchar_t *text) {
    int32_t length = 0;
    int32_t break_pos[STRING_BUFFER_LENGTH];
    int32_t view_lines = 0;
    int32_t view_start_pos = 0;
    int32_t view_end_pos = 0;
    text_typeset(text, 128, 64, 0, &length, break_pos, &view_lines, &view_start_pos, &view_end_pos);
    return view_lines;
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

// 绘制滚动条
void render_scroll_bar(int32_t line_num, int32_t current_line) {
    for (int y = 0; y < 64; y++) {
        OLED_DrawPoint(127, y, !(y % 3));
    }
    uint8_t bar_height = (uint8_t)((5 * 64) / line_num);
    uint8_t y_0 = (uint8_t)((current_line * 64) / line_num);
    OLED_DrawLine(127, y_0, 127, y_0 + bar_height, 1);
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

    render_line(L"Project MARGA!", 24, 2, 0);
    render_line(L"完全离线电子鹦鹉", 16, 20, 1);
    // render_line(L"自研Nano模型强力驱动", 4, 34, 1);
    render_line(datetime_wcs_buffer, 8, 34, 1);
    render_line(L"(c) 2025 BD4SUR", 18, 50, 1);

    OLED_DrawLine(0, 0, 127, 0, 1);
    OLED_DrawLine(0, 15, 127, 15, 1);
    OLED_DrawLine(0, 0, 0, 63, 1);
    OLED_DrawLine(127, 0, 127, 64, 1); // NOTE 如果横坐标设为63则会漏掉最后一个点
    OLED_DrawLine(0, 63, 127, 63, 1);

    // 检查ASR服务状态，如果ASR服务未启动，则在屏幕左上角画一个小点，表示ASR服务启动中
    if (global_state->is_asr_server_up < 1) {
        int32_t v = ((global_state->timer >> 2) % 2);
        OLED_DrawLine(1, 1, 2, 1, v);
        OLED_DrawLine(1, 1, 1, 2, v);
        OLED_DrawLine(1, 2, 2, 2, v);
        OLED_DrawLine(2, 1, 2, 3, v); // NOTE 如果横坐标设为63则会漏掉最后一个点
    }

    OLED_Refresh();
}






void show_main_menu(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state) {
    OLED_SoftClear();
    render_text(L"1.人类的本质\n2.电子鹦鹉\n3.选择语言模型\n4.设置\n5.安全关机", 0);
    OLED_Refresh();
}





void draw_textarea(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state) {
    OLED_SoftClear();
    textarea_state->line_num = render_text(textarea_state->text, textarea_state->current_line);
    if (textarea_state->is_show_scroll_bar) {
        render_scroll_bar(textarea_state->line_num, textarea_state->current_line);
    }
    OLED_Refresh();
}




void init_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {
    input_state->state = 0;
    input_state->ime_mode_flag = IME_MODE_HANZI;
    input_state->pinyin_keys = 0;
    input_state->candidates = NULL;
    input_state->candidate_num = 0;
    input_state->candidate_pages = NULL;
    input_state->candidate_page_num = 0;
    input_state->current_page = 0;
    input_state->input_buffer = refresh_input_buffer(input_state->input_buffer, &(input_state->input_counter));
    input_state->cursor_pos = input_state->input_counter;
    input_state->alphabet_countdown = -1;
    input_state->alphabet_current_key = 255;
    input_state->alphabet_index = 0;

    render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);
}

void draw_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state) {

    int32_t state = input_state->state;
    printf("State = %d  |  Key_code = %d  |  Key_edge = %d\n", state, key_event->key_code, key_event->key_edge);

    // 定时器触发：更新进度条
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
                input_state->input_buffer[input_state->input_counter++] = ch;
            }
            else {
                printf("选定了列表之外的字母，忽略。\n");
            }

            render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);

            input_state->alphabet_current_key = 255;
            input_state->alphabet_index = 0;
            input_state->state = 0;
        }
    }

    if (state == 0) {

        // 长按0：输入符号
        if (key_event->key_edge == -2 && key_event->key_code == 0) {
            input_state->candidates = (uint32_t *)calloc(54, sizeof(uint32_t));
            for (int i = 0; i < 54; i++) input_state->candidates[i] = (uint32_t)ime_symbols[i];
            input_state->candidate_pages = candidate_paging(input_state->candidates, 54, 10, &(input_state->candidate_page_num));
            render_symbol_input(input_state->candidate_pages, input_state->current_page, input_state->candidate_page_num);
            input_state->current_page = 0;
            input_state->state = 3;
        }

        // 短按0：数字输入模式下是直接输入0，其余模式无动作
        else if (key_event->key_edge == -1 && key_event->key_code == 0) {
            if (input_state->ime_mode_flag == IME_MODE_NUMBER) {
                input_state->input_buffer[(input_state->input_counter)++] = L'0';
                render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);
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
                input_state->input_buffer[(input_state->input_counter)++] = L'0' + key_event->key_code;
                render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);
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
            if (input_state->input_counter >= 1) {
                input_state->input_buffer[--(input_state->input_counter)] = 0;
                render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);
            }
            input_state->state = 0;
        }

        // 长+短按B键：依次切换汉-英-数输入模式
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 11) {
            input_state->ime_mode_flag = (input_state->ime_mode_flag + 1) % 3;
            render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);
            input_state->state = 0;
        }

        // 长+短按*键：光标向左移动
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
            
        }

        // 长+短按#键：光标向右移动
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {

        }
    }

    else if (state == 1) {
        // 短按D键：开始选字
        if (key_event->key_edge == -1 && key_event->key_code == 13) {
            if (input_state->candidate_pages) {
                render_pinyin_input(input_state->candidate_pages, input_state->pinyin_keys, input_state->current_page, input_state->candidate_page_num, 1);
                input_state->state = 2;
            }
        }

        // 短按A键：取消输入拼音，清除已输入的所有按键，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }

        // 短按2-9键：继续输入拼音
        else if (key_event->key_edge == -1 && (key_event->key_code >= 2 && key_event->key_code <= 9)) {
            input_state->pinyin_keys *= 10;
            input_state->pinyin_keys += (uint32_t)(key_event->key_code);

            if (input_state->candidates) {
                free(input_state->candidates);
                input_state->candidates = NULL;
            }
            free_candidate_pages(input_state->candidate_pages, input_state->candidate_page_num);
            input_state->candidate_pages = NULL;

            input_state->candidates = candidate_hanzi_list(input_state->pinyin_keys, &(input_state->candidate_num));

            if (input_state->candidates) { // 如果当前键码有对应的候选字
                // 候选字列表分页
                input_state->candidate_pages = candidate_paging(input_state->candidates, input_state->candidate_num, 10, &(input_state->candidate_page_num));
                render_pinyin_input(input_state->candidate_pages, input_state->pinyin_keys, input_state->current_page, input_state->candidate_page_num, 0);
            }
            else {
                render_pinyin_input(NULL, input_state->pinyin_keys, 0, 0, 0);
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
                input_state->input_buffer[(input_state->input_counter)++] = ch;
            }
            else {
                printf("选定了列表之外的字，忽略。\n");
            }

            render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);

            free(input_state->candidates);
            input_state->candidates = NULL;
            free_candidate_pages(input_state->candidate_pages, input_state->candidate_page_num);
            input_state->candidate_pages = NULL;
            input_state->current_page = 0;

            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }

        // 长+短按*键：候选字翻页到上一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
            if(input_state->current_page > 0) {
                input_state->current_page--;
                render_pinyin_input(input_state->candidate_pages, input_state->pinyin_keys, input_state->current_page, input_state->candidate_page_num, 1);
            }
            input_state->state = 2;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                render_pinyin_input(input_state->candidate_pages, input_state->pinyin_keys, input_state->current_page, input_state->candidate_page_num, 1);
            }
            input_state->state = 2;
        }

        // 短按A键：取消选择，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);
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
                input_state->input_buffer[(input_state->input_counter)++] = ch;
            }
            else {
                printf("选定了列表之外的符号，忽略。\n");
            }
            render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);

            free(input_state->candidates);
            input_state->candidates = NULL;
            free_candidate_pages(input_state->candidate_pages, input_state->candidate_page_num);
            input_state->candidate_pages = NULL;
            input_state->current_page = 0;

            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }

        // 长+短按*键：候选字翻页到上一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
            if(input_state->current_page > 0) {
                input_state->current_page--;
                render_symbol_input(input_state->candidate_pages, input_state->current_page, input_state->candidate_page_num);
            }
            input_state->state = 3;
        }

        // 长+短按#键：候选字翻页到下一页
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
            if(input_state->current_page < input_state->candidate_page_num - 1) {
                input_state->current_page++;
                render_symbol_input(input_state->candidate_pages, input_state->current_page, input_state->candidate_page_num);
            }
            input_state->state = 3;
        }

        // 短按A键：取消选择，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == 10) {
            render_input_buffer(input_state->input_buffer, input_state->ime_mode_flag, -1);
            input_state->current_page = 0;
            input_state->pinyin_keys = 0;
            input_state->state = 0;
        }
    }
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
