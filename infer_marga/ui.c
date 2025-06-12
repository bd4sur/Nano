#include <stdio.h>
#include <time.h>

#include "ui.h"
#include "oled.h"

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

    render_line(L"Project MARGA!", 24, 2, 1);
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

void render_input_buffer(uint32_t *input_buffer, uint32_t ime_mode_flag, uint32_t is_show_cursor) {
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
    if (is_show_cursor) wcscat(text, L"_");
    render_text(text, 0);
    OLED_Refresh();
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
