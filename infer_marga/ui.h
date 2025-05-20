#ifndef __NANO_UI_H__
#define __NANO_UI_H__

#include <stdint.h>

#define INPUT_BUFFER_LENGTH  (4096)
#define OUTPUT_BUFFER_LENGTH (32768)

#define IME_MODE_HANZI    (0)
#define IME_MODE_ALPHABET (1)
#define IME_MODE_NUMBER   (2)

void show_splash_screen();
void render_input_buffer(uint32_t *input_buffer, uint32_t ime_mode_flag, uint32_t is_show_cursor);
void render_pinyin_input(uint32_t **candidate_pages, uint32_t pinyin_keys, uint32_t current_page, uint32_t candidate_page_num, uint32_t is_picking);
void render_symbol_input(uint32_t **candidate_pages, uint32_t current_page, uint32_t candidate_page_num);
void render_scroll_bar(int32_t line_num, int32_t current_line);
uint32_t *refresh_input_buffer(uint32_t *input_buffer, int32_t *input_counter);

#endif
