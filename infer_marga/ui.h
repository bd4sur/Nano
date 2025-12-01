#ifndef __NANO_UI_H__
#define __NANO_UI_H__

#include <wchar.h>
#include <stdint.h>

#include "infer.h"

#define INPUT_BUFFER_LENGTH  (65536)
#define OUTPUT_BUFFER_LENGTH (65536)

#define IME_MODE_HANZI    (0)
#define IME_MODE_ALPHABET (1)
#define IME_MODE_NUMBER   (2)

#define ALPHABET_COUNTDOWN_MAX (30)
#define LONG_PRESS_THRESHOLD (360)

#define MAX_CANDIDATE_NUM (1024)     // 候选字最大数量
#define MAX_CANDIDATE_PAGE_NUM (108) // 候选字最大分页数
#define MAX_CANDIDATE_NUM_PER_PAGE (10) // 每页最多有几个候选字（每页10个字）

#define MAX_MENU_ITEMS (1024)
#define MAX_MENU_ITEM_LEN (24)

typedef struct {
    int32_t timer;
    int32_t focus;
    Nano_Session *llm_session; // LLM一轮对话状态
    int32_t llm_status; // LLM推理状态
    int32_t is_asr_server_up;
    int32_t is_recording; // 录音状态
    time_t asr_start_timestamp; // 录音起始的时间戳

    int32_t ups_voltage; // UPS电压
    int32_t ups_soc; // UPS电量

    int32_t is_full_refresh; // 作为所有绘制函数的一个参数，用于控制是否整帧刷新。默认为1。0-禁用函数内的clear-refresh，1-启用函数内的clear-refresh

    int32_t refresh_ratio; // LLM推理过程中，屏幕刷新的分频系数，也就是每几次推理刷新一次屏幕

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
    int32_t y; // NOTE 设置文本框高度时，按照除末行外，每行margin-bottom:1px来计算。例如，如果希望恰好显示4行，则高度应为13*3+12=51px。
    int32_t width;
    int32_t height;
    wchar_t text[INPUT_BUFFER_LENGTH];
    int32_t length;
    int32_t break_pos[INPUT_BUFFER_LENGTH];
    int32_t line_num;
    int32_t view_lines;
    int32_t view_start_pos;
    int32_t view_end_pos;
    int32_t current_line;
    int32_t is_show_scroll_bar; // 是否显示滚动条：0不显示 1显示
} Widget_Textarea_State;

typedef struct {
    // 以下实际上是继承 Widget_Textarea_State
    int32_t state; // 内部状态
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    wchar_t text[INPUT_BUFFER_LENGTH];
    int32_t length;
    int32_t break_pos[INPUT_BUFFER_LENGTH];
    int32_t line_num;
    int32_t view_lines;
    int32_t view_start_pos;
    int32_t view_end_pos;
    int32_t current_line;
    int32_t is_show_scroll_bar;

    // 以下是Widget_Input_State独有的

    int32_t cursor_pos;           // 光标位置
    uint32_t ime_mode_flag;       // 汉英数输入模式标志 0汉字 1英文 2数字
    uint32_t pinyin_keys;         // 单字拼音键码暂存
    // 候选字翻页相关
    uint32_t candidates[MAX_CANDIDATE_NUM]; // 全部候选字/符号
    uint32_t candidate_num;       // 候选字总数
    uint32_t candidate_pages[MAX_CANDIDATE_PAGE_NUM][MAX_CANDIDATE_NUM_PER_PAGE]; // 候选字分页
    uint32_t candidate_page_num;  // 总的候选字分页数
    uint32_t current_page;        // 当前显示的候选字页标号
    // 英文字母输入模式的倒计时
    int32_t alphabet_countdown;   // 从ALPHABET_COUNTDOWN_MAX开始，每轮主循环后倒数减1，减到0时清除进度条，减到-1意味着英文字母输入状态结束
    uint8_t alphabet_current_key; // 当前选中的字母按键
    uint32_t alphabet_index;
} Widget_Input_State;

typedef struct {
    int32_t current_item_intex; // 当前选中（高亮）的条目的标号（注意：选中条目不一定在显示的页面范围内）
    int32_t first_item_intex; // 当前页面显示的第一个条目的标号
    int32_t item_num; // 菜单条目数
    int32_t items_per_page; // 每页容纳的条目数
    wchar_t title[MAX_MENU_ITEM_LEN]; // 菜单标题
    wchar_t items[MAX_MENU_ITEMS][MAX_MENU_ITEM_LEN]; // 条目标题
} Widget_Menu_State;


// 符号列表
static wchar_t ime_symbols[55] = L"，。、？！：；“”‘’（）《》…—～·【】 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
// 按键对应的字母列表
static wchar_t ime_alphabet[10][32] = {L"0", L" 1.,:?!-/+_=&\"*", L"abcABC2", L"defDEF3", L"ghiGHI4", L"jklJKL5", L"mnoMNO6", L"pqrsPRQS7", L"tuvTUV8", L"wxyzWXYZ9"};


void render_line(wchar_t *line, uint32_t x, uint32_t y, uint8_t mode);

void render_text(
    wchar_t *text, int32_t start_line, int32_t length, int32_t *break_pos, int32_t line_num,
    int32_t x_offset, int32_t y_offset, int32_t width, int32_t height, int32_t do_typeset);

void show_splash_screen(Key_Event *key_event, Global_State *global_state);

void draw_textarea(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state);

void init_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);
void refresh_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);
void draw_input(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);

void init_menu(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state);
void refresh_menu(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state);
void draw_menu(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state);

void render_input_buffer(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);
void render_cursor(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);
void render_pinyin_input(Widget_Input_State *input_state, uint32_t is_picking);
void render_symbol_input(Widget_Input_State *input_state);

void render_scroll_bar(int32_t line_num, int32_t current_line, int32_t view_lines, int32_t x, int32_t y, int32_t width, int32_t height);

#endif
