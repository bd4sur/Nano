#ifndef __NANO_UI_H__
#define __NANO_UI_H__

#include <wchar.h>
#include <stdint.h>

#define INPUT_BUFFER_LENGTH  (65536)
#define OUTPUT_BUFFER_LENGTH (65536)

#define IME_MODE_HANZI    (0)
#define IME_MODE_ALPHABET (1)
#define IME_MODE_NUMBER   (2)

#define ALPHABET_COUNTDOWN_MAX (30)
#define LONG_PRESS_THRESHOLD (360)


typedef struct {
    int32_t timer;
    int32_t focus;
    int32_t is_asr_server_up;
    int32_t is_recording; // 录音状态
    time_t asr_start_timestamp; // 录音起始的时间戳

} Global_State;

typedef struct {
    uint8_t  prev_key;   // 上一次按键的键值
    uint8_t  key_code;   // 大于等于16为没有任何按键，0-15为按键
    int8_t   key_edge;   // 0：松开  1：上升沿  -1：下降沿(短按结束)  -2：下降沿(长按结束)
    uint32_t key_timer;  // 按下状态的计时器
    uint8_t  key_mask;   // 长按超时后，键盘软复位标记。此时虽然物理上依然按键，只要软复位标记为1，则认为是无按键，无论是边沿还是按住都不触发。直到物理按键松开后，软复位标记清0。
    uint8_t  key_repeat; // 触发一次长按后，只要不松手，该标记置1，直到物理按键松开后置0。若该标记为1，则在按住时触发连续重复动作。
} Key_Event;

typedef struct {
    int32_t state;
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    wchar_t *text;
    int32_t line_num;
    int32_t current_line;
    int32_t is_show_scroll_bar; // 是否显示滚动条：0不显示 1显示
} Widget_Textarea_State;

typedef struct {
    int32_t state; // 内部状态
    uint32_t ime_mode_flag; // 汉英数输入模式标志 0汉字 1英文 2数字
    uint32_t pinyin_keys;   // 单字拼音键码暂存
    // 候选字翻页相关
    uint32_t *candidates;
    uint32_t candidate_num;
    uint32_t **candidate_pages;
    uint32_t candidate_page_num;
    uint32_t current_page;
    // 全局文字输入缓冲
    uint32_t *input_buffer; // 文字输入缓冲区
    int32_t input_counter;
    int32_t cursor_pos; // 光标位置
    // 英文字母输入模式的倒计时
    int32_t alphabet_countdown; // 从ALPHABET_COUNTDOWN_MAX开始，每轮主循环后倒数减1，减到0时清除进度条，减到-1意味着英文字母输入状态结束
    uint8_t alphabet_current_key;
    uint32_t alphabet_index;
} Widget_Input_State;

typedef struct {
    int32_t state;
} Widget_Menu_State;


// 符号列表
static wchar_t ime_symbols[55] = L"，。、？！：；“”‘’（）《》…—～·【】 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
// 按键对应的字母列表
static wchar_t ime_alphabet[10][32] = {L"", L" .,;:?!-/+_=&\"*", L"abcABC", L"defDEF", L"ghiGHI", L"jklJKL", L"mnoMNO", L"pqrsPRQS", L"tuvTUV", L"wxyzWXYZ"};


int32_t get_view_lines(wchar_t *text);

void render_line(wchar_t *line, uint32_t x, uint32_t y, uint8_t mode);
int32_t render_text(wchar_t *text, int32_t start_line);

void show_splash_screen(Key_Event *key_event, Global_State *global_state);
void show_main_menu(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state);

void draw_textarea(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state);

void init_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);
void draw_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);

int32_t render_input_buffer(uint32_t *input_buffer, uint32_t ime_mode_flag, int32_t cursor_pos);
void render_pinyin_input(uint32_t **candidate_pages, uint32_t pinyin_keys, uint32_t current_page, uint32_t candidate_page_num, uint32_t is_picking);
void render_symbol_input(uint32_t **candidate_pages, uint32_t current_page, uint32_t candidate_page_num);
void render_scroll_bar(int32_t line_num, int32_t current_line);
uint32_t *refresh_input_buffer(uint32_t *input_buffer, int32_t *input_counter);

#endif
