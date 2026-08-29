#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ui_ebook.h"
#include "ui_color.h"
#include "ui_softkbd.h"
#include "hal_key.h"
#include "platform.h"

#define EBOOK_READ_CHUNK      (2048)
#define EBOOK_MAX_GOTO_DIGITS (6)

// ===============================================================================
// 文件列表（菜单数据，由本模块持有后端存储，菜单控件仅借用指针）
// ===============================================================================

static char    **s_list_mb = NULL;   // UTF-8 完整路径（带前导'/'）
static wchar_t **s_list_w = NULL;    // 显示名（文件名部分）
static const wchar_t **s_items = NULL; // 菜单借用指针表
static int32_t   s_list_count = 0;

// ===============================================================================
// 当前打开的书
// ===============================================================================

static int32_t   s_book_open = 0;
static wchar_t   s_book_title[64];
static uint32_t *s_page_offsets = NULL; // PSRAM：每页起始字节偏移
static int32_t   s_page_count = 0;      // 总页数
static int32_t   s_page_cap = 0;
static int32_t   s_view_lines = 0;      // 每页行数（一屏）
static int32_t   s_ta_width = 0;        // 折行宽度（px）
static int32_t   s_total_lines = 0;     // 全文总行数（预扫描统计）
static int32_t   s_buf_start_line = 0;  // 缓冲区首行的全局行号（滑动窗口起点）

// “跳转到页”模态框
static int32_t   s_goto_active = 0;
static char      s_goto_digits[EBOOK_MAX_GOTO_DIGITS + 1];
static int32_t   s_goto_len = 0;

static uint8_t   s_rbuf[EBOOK_READ_CHUNK];

static int32_t   s_scroll_last_page = -1; // 滚动帧轻量渲染的页码跟踪（-1=首次强制更新）

// UTF-8 增量解码状态（逐字节喂入，跨块保持）
static uint32_t  s_dec_cp;
static int32_t   s_dec_need;

// 喂入一个字节；完成一个码点时返回1（*cp_out 有效），否则返回0
static int32_t utf8_feed(uint8_t b, uint32_t *cp_out) {
    if (s_dec_need == 0) {
        if (b < 0x80)                    { *cp_out = b; return 1; }
        else if ((b & 0xE0) == 0xC0)     { s_dec_cp = b & 0x1F; s_dec_need = 1; }
        else if ((b & 0xF0) == 0xE0)     { s_dec_cp = b & 0x0F; s_dec_need = 2; }
        else if ((b & 0xF8) == 0xF0)     { s_dec_cp = b & 0x07; s_dec_need = 3; }
        else                             { *cp_out = b; return 1; } // 非法字节按单字节处理
        return 0;
    }
    if ((b & 0xC0) == 0x80) {
        s_dec_cp = (s_dec_cp << 6) | (b & 0x3F);
        s_dec_need--;
        if (s_dec_need == 0) { *cp_out = s_dec_cp; return 1; }
        return 0;
    }
    s_dec_need = 0; // 序列损坏：丢弃前导字节，按新起始字节重新处理
    return utf8_feed(b, cp_out);
}

// ===============================================================================
// 文件列表构建/销毁
// ===============================================================================

static void ebook_free_list(void) {
    if (s_list_mb != NULL) {
        for (int32_t i = 0; i < s_list_count; i++) {
            if (s_list_mb[i] != NULL) free(s_list_mb[i]);
        }
        free(s_list_mb);
        s_list_mb = NULL;
    }
    if (s_list_w != NULL) {
        for (int32_t i = 0; i < s_list_count; i++) {
            if (s_list_w[i] != NULL) free(s_list_w[i]);
        }
        free(s_list_w);
        s_list_w = NULL;
    }
    if (s_items != NULL) {
        free((void *)s_items);
        s_items = NULL;
    }
    s_list_count = 0;
}

int32_t ui_ebook_menu_init(Key_Event *key_event, Global_State *global_state) {
    ebook_free_list();

    int32_t total = list_files(PLATFORM_ROOT_DIR "/ebook", NULL);
    if (total > 0) {
        char **names = (char **)platform_calloc((size_t)total, sizeof(char *));
        s_list_mb = (char **)platform_calloc((size_t)total, sizeof(char *));
        s_list_w  = (wchar_t **)platform_calloc((size_t)total, sizeof(wchar_t *));
        s_items   = (const wchar_t **)platform_calloc((size_t)total, sizeof(wchar_t *));
        if (names != NULL && s_list_mb != NULL && s_list_w != NULL && s_items != NULL
            && list_files(PLATFORM_ROOT_DIR "/ebook", names) >= 0) {
            for (int32_t i = 0; i < total; i++) {
                if (names[i] == NULL) continue;
                // 规范为带前缀的完整路径
                char path[160];
                snprintf(path, sizeof(path), PLATFORM_ROOT_DIR "/ebook/%s", names[i]);
                free(names[i]);
                // 仅保留文件（非目录）
                if (platform_is_directory(path)) continue;
                size_t plen = strlen(path);
                s_list_mb[s_list_count] = (char *)platform_malloc(plen + 1);
                s_list_w[s_list_count]  = (wchar_t *)platform_calloc(plen + 1, sizeof(wchar_t));
                if (s_list_mb[s_list_count] == NULL || s_list_w[s_list_count] == NULL) {
                    if (s_list_mb[s_list_count] != NULL) free(s_list_mb[s_list_count]);
                    if (s_list_w[s_list_count]  != NULL) free(s_list_w[s_list_count]);
                    continue;
                }
                memcpy(s_list_mb[s_list_count], path, plen + 1);
                // 显示名：只取最后一段文件名，UTF-8 转宽字符
                const char *disp = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
                const uint8_t *p = (const uint8_t *)disp;
                int32_t wl = 0;
                s_dec_need = 0;
                while (*p != 0 && wl < (int32_t)plen - 1) {
                    uint32_t cp;
                    if (utf8_feed(*p, &cp)) s_list_w[s_list_count][wl++] = (wchar_t)cp;
                    p++;
                }
                s_list_w[s_list_count][wl] = L'\0';
                s_list_count++;
            }
        }
        if (names != NULL) free(names);
    }

    // 按路径字符串升序（插入排序，UTF-8 字节序即码点序）
    for (int32_t i = 1; i < s_list_count; i++) {
        char *key_mb = s_list_mb[i];
        wchar_t *key_w = s_list_w[i];
        int32_t j = i - 1;
        while (j >= 0 && strcmp(s_list_mb[j], key_mb) > 0) {
            s_list_mb[j + 1] = s_list_mb[j];
            s_list_w[j + 1] = s_list_w[j];
            j--;
        }
        s_list_mb[j + 1] = key_mb;
        s_list_w[j + 1] = key_w;
    }
    for (int32_t i = 0; i < s_list_count; i++) {
        s_items[i] = s_list_w[i];
    }

    global_state->w_menu_main->title = L"电子书";
    global_state->w_menu_main->items = s_items;
    global_state->w_menu_main->item_num = s_list_count;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
    return 0;
}

// ===============================================================================
// 打开/关闭
// ===============================================================================

void ui_ebook_close(void) {
    if (s_book_open) {
        platform_file_close();
    }
    s_book_open = 0;
    if (s_page_offsets != NULL) {
        free(s_page_offsets);
        s_page_offsets = NULL;
    }
    s_page_count = 0;
    s_page_cap = 0;
    s_total_lines = 0;
    s_buf_start_line = 0;
    s_goto_active = 0;
}

// 预扫描全文：按与 typeset_line_breaks 一致的折行规则，计算每页（view_lines 行）的起始字节偏移
// 前向声明（定义见下文）
static void ebook_load_window(Key_Event *key_event, Global_State *global_state, int32_t start_line);

static int32_t ebook_scan_pages(Key_Event *key_event, Global_State *global_state) {
    s_page_cap = 256;
    s_page_offsets = (uint32_t *)platform_calloc((size_t)s_page_cap, sizeof(uint32_t));
    if (s_page_offsets == NULL) {
        return -1;
    }
    s_page_offsets[0] = 0;
    s_page_count = 1;

    int32_t line_x = 0;
    int32_t lines_in_page = 0;
    int32_t char_start = 0;
    uint32_t pos = 0;
    s_dec_need = 0;

    // 打开进度条：总量为文件大小，每消费一块按字节数更新（时间节流，避免刷屏拖慢扫描）
    uint32_t file_size = platform_file_size();
    uint64_t last_bar_ts = 0;

    platform_file_seek(0);
    while (1) {
        int32_t n = platform_file_read(s_rbuf, EBOOK_READ_CHUNK);
        if (n <= 0) break;

        if (file_size > 0) {
            uint64_t now = get_timestamp_in_ms();
            if (now - last_bar_ts > 100) {
                last_bar_ts = now;
                Nano_GFX *gfx = global_state->gfx;
                int32_t w = (int32_t)((uint64_t)pos * gfx->width / file_size);
                gfx_draw_rectangle(gfx, 0, gfx->height - 4, gfx->width, 4, 30, 30, 36, 1);
                gfx_draw_rectangle(gfx, 0, gfx->height - 4, (uint32_t)w, 4, 0x00, 0xaa, 0xff, 1);
                gfx_refresh(gfx);
            }
        }

        for (int32_t i = 0; i < n; i++) {
            uint8_t b = s_rbuf[i];
            if (s_dec_need == 0) char_start = (int32_t)pos;
            pos++;
            uint32_t cp;
            if (!utf8_feed(b, &cp)) continue;
            if (cp == '\r') continue;

            int32_t cw = (cp == '\n') ? 0 : gfx_font_char_advance(global_state->ui_font, cp);
            int32_t new_page_at = -1;
            // 折行判断与 typeset_line_breaks 一致：先判软折行，再判硬换行
            if (line_x + cw >= s_ta_width) {
                lines_in_page++;
                s_total_lines++;
                if (lines_in_page == s_view_lines) new_page_at = char_start;
                line_x = 0;
            }
            else if (cp == '\n') {
                lines_in_page++;
                s_total_lines++;
                if (lines_in_page == s_view_lines) new_page_at = (int32_t)pos;
                line_x = 0;
            }
            line_x += cw;

            if (new_page_at >= 0) {
                if (s_page_count >= s_page_cap) {
                    // 容量倍增（4字节/页，远低于4MB PSRAM预算上限）
                    int32_t new_cap = s_page_cap * 2;
                    uint32_t *np = (uint32_t *)platform_realloc(s_page_offsets, (size_t)new_cap * sizeof(uint32_t));
                    if (np != NULL) {
                        s_page_offsets = np;
                        s_page_cap = new_cap;
                    }
                }
                if (s_page_count < s_page_cap) {
                    s_page_offsets[s_page_count++] = (uint32_t)new_page_at;
                }
                lines_in_page = 0;
            }
        }
    }
    s_total_lines++; // 折行事件数 + 1（首行）= 全文总行数
    return 0;
}

int32_t ui_ebook_open(Key_Event *key_event, Global_State *global_state, const char *path_mb) {
    ui_ebook_close(); // 防御：关闭上一本

    // 重置文本控件几何（LLM观测模式会修改 x/width；阅读恢复页眉页脚间全宽布局；
    // 页眉 1.5 倍行高 + 页脚行高+1，与 ui_widget_textarea_init 的标准布局一致）
    Widget_Textarea_State *ta = global_state->w_textarea_main;
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    int32_t header_height = ui_std_header_height(global_state->ui_font);
    ta->x = 0;
    ta->y = header_height;
    ta->width = global_state->gfx->width;
    ta->height = global_state->gfx->height - ui_softkbd_height() - header_height - (line_height + 1);
    ta->is_show_scroll_bar = 0; // 关闭控件自带滚动条（缓冲区内行位置），改由本模块绘制总进度条
    s_ta_width = ta->width;
    s_view_lines = ta->height / line_height; // 排版语义页容量（下取整），底部不完整行由 typeset 扩展在绘制端覆盖
    if (s_view_lines <= 0) s_view_lines = 1;

    // 提示正在打开（预扫描大文件需要数秒）：蓝底白字，置于屏幕顶部中央（同“正在计算”横幅）
    gfx_draw_rectangle(global_state->gfx, global_state->gfx->width / 2 - 70, 0, 140, 14, 0x11, 0x55, 0xee, 1);
    gfx_draw_textline_centered(global_state->gfx, L"正在打开，请稍候...", global_state->gfx->width / 2, 7, 255, 255, 255, 1);
    gfx_refresh(global_state->gfx);

    if (platform_file_open(path_mb) != 0) {
        return -1;
    }
    s_book_open = 1;

    // 标题：仅文件名（最后一个路径分隔符 '/' 之后的部分）
    const char *fname = strrchr(path_mb, '/');
    const uint8_t *p = (const uint8_t *)((fname != NULL) ? (fname + 1) : path_mb);
    int32_t tl = 0;
    s_dec_need = 0;
    while (*p != 0 && tl < 63) {
        uint32_t cp;
        if (utf8_feed(*p, &cp)) s_book_title[tl++] = (wchar_t)cp;
        p++;
    }
    s_book_title[tl] = L'\0';

    if (ebook_scan_pages(key_event, global_state) != 0 || s_page_count <= 0) {
        ui_ebook_close();
        return -2;
    }
    // 换入第一个窗口（否则初次进入阅读状态时控件内仍是旧内容，需翻页才会触发载入）
    ebook_load_window(key_event, global_state, 0);
    return 0;
}

// 按需换入滑动窗口：从文件的第 start_line 行（全局行号，0起）开始解码，
// 尽量填满文本控件缓冲区（容量 UI_STR_BUF_MAX_LENGTH，通常容纳1.2~2.4页），
// 使滚行跨页时页间断续处已在缓冲区内，从SD卡取数对用户无感知。
static void ebook_load_window(Key_Event *key_event, Global_State *global_state, int32_t start_line) {
    if (!s_book_open || s_page_count <= 0) {
        return;
    }
    if (start_line < 0) start_line = 0;
    if (s_total_lines > 0 && start_line >= s_total_lines) start_line = s_total_lines - 1;

    // 起点所在页：seek 到页首后向前跳过 start_line % view_lines 个行首
    int32_t page = start_line / s_view_lines;
    int32_t skip = start_line % s_view_lines;
    uint32_t fend = (page + 1 < s_page_count) ? s_page_offsets[page + 1] : 0xFFFFFFFFUL;

    Widget_Textarea_State *ta = global_state->w_textarea_main;
    platform_file_seek(s_page_offsets[page]);
    uint32_t pos = s_page_offsets[page];
    int32_t out_len = 0;
    int32_t line_x = 0;
    int32_t skipped = 0;
    int32_t collecting = (skip == 0) ? 1 : 0;
    s_dec_need = 0;

    while (out_len < UI_STR_BUF_MAX_LENGTH - 1) {
        int32_t n = platform_file_read(s_rbuf, EBOOK_READ_CHUNK);
        if (n <= 0) break;
        for (int32_t i = 0; i < n && out_len < UI_STR_BUF_MAX_LENGTH - 1; i++) {
            uint8_t b = s_rbuf[i];
            pos++;
            uint32_t cp;
            if (!utf8_feed(b, &cp)) continue;
            if (cp == '\r') continue;

            if (!collecting) {
                if (pos > fend) break; // 跳过阶段不越出本页（防御：末页行数不足时止于EOF）
                int32_t cw = (cp == '\n') ? 0 : gfx_font_char_advance(global_state->ui_font, cp);
                // 折行判断与 typeset_line_breaks 一致：先判软折行，再判硬换行
                if (line_x + cw >= s_ta_width) {
                    if (++skipped == skip) collecting = 1; // 新行从当前字符开始（当前字符也要收）
                    line_x = 0;
                }
                else if (cp == '\n') {
                    if (++skipped == skip) {
                        collecting = 1; // 新行从换行符之后开始（'\n'本身不收）
                        line_x = 0;
                        continue;
                    }
                    line_x = 0;
                }
                line_x += cw;
                if (!collecting) continue;
            }
            ta->text[out_len++] = (wchar_t)cp;
        }
    }
    ta->text[out_len] = L'\0';
    ta->length = out_len;
    ta->current_line = 0;
    ta->scroll_sub_offset = 0; // 换窗整行跳转：吸附回整行（像素滚动不变量）
    ta->is_modified = 1;
    s_buf_start_line = start_line;
    // 立即排版，使 line_num/view_lines 可供滑动判定使用
    typeset_line_breaks(key_event, global_state, ta);
}

// ===============================================================================
// 阅读渲染/事件
// ===============================================================================

// 总进度滚动条：显示当前页在全部页中的位置（替代文本控件自带的“缓冲区内行位置”滚动条）。
// 显示风格复用 ui.c 的 ui_draw_scroll_bar（配色/2px轨道+2px滑块/横向偏移/最小高度与限位均一致）：
// 将“页”映射为其“行”语义——全部页视为 line_num 行、视口视为 1 行、当前页视为 current_line。
static void ui_ebook_draw_progress_bar(Key_Event *key_event, Global_State *global_state) {
    Widget_Textarea_State *ta = global_state->w_textarea_main;
    int32_t pages = (s_page_count <= 0) ? 1 : s_page_count;
    // 以视口首行所在页表示总进度
    int32_t cur_page = (s_buf_start_line + ta->current_line) / ((s_view_lines > 0) ? s_view_lines : 1);
    ui_draw_scroll_bar(key_event, global_state, cur_page, pages, 1, ta->x, ta->y, ta->width, ta->height);
}

// ===============================================================================
// “跳转到页”模态框 + 触屏数字键盘
// ===============================================================================

// 布局：模态框上移（为下方数字键盘留空间）；键盘 3行4列，尽量填满可用空间并留边距/间距
#define EBOOK_GOTO_MODAL_X   (60)
#define EBOOK_GOTO_MODAL_Y   (8)
#define EBOOK_GOTO_MODAL_W   (200)
#define EBOOK_GOTO_MODAL_H   (44)
#define EBOOK_GOTO_PAD_X0    (8)
#define EBOOK_GOTO_PAD_Y0    (56)
#define EBOOK_GOTO_PAD_X1    (312)
#define EBOOK_GOTO_PAD_Y1    (232)
#define EBOOK_GOTO_PAD_GAP   (8)
#define EBOOK_GOTO_PAD_COLS  (4)
#define EBOOK_GOTO_PAD_ROWS  (3)
#define EBOOK_GOTO_PAD_CELL_W ((EBOOK_GOTO_PAD_X1 - EBOOK_GOTO_PAD_X0 - (EBOOK_GOTO_PAD_COLS - 1) * EBOOK_GOTO_PAD_GAP) / EBOOK_GOTO_PAD_COLS)
#define EBOOK_GOTO_PAD_CELL_H ((EBOOK_GOTO_PAD_Y1 - EBOOK_GOTO_PAD_Y0 - (EBOOK_GOTO_PAD_ROWS - 1) * EBOOK_GOTO_PAD_GAP) / EBOOK_GOTO_PAD_ROWS)

// 跳页确认（硬按键 D 与触屏“确认”按钮共用）
static void ui_ebook_goto_confirm(Key_Event *key_event, Global_State *global_state) {
    s_goto_digits[s_goto_len] = '\0';
    int32_t p = (s_goto_len > 0) ? atoi(s_goto_digits) : 0;
    s_goto_active = 0;
    if (p >= 1) {
        // 跳页：窗口起点对齐到目标页首行，视口置于缓冲区顶部
        ebook_load_window(key_event, global_state, (p - 1) * s_view_lines);
    }
}

// 绘制“跳转到页”模态框与触屏数字键盘（纯触屏设备上数字输入的唯一渠道；
// 布局 3行4列：1 2 3 退格 / 4 5 6 0 / 7 8 9 确认）
static void ui_ebook_goto_modal_render(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    Nano_GFX *gfx = global_state->gfx;
    // 模态框（上移）
    gfx_draw_rectangle(gfx, EBOOK_GOTO_MODAL_X, EBOOK_GOTO_MODAL_Y, EBOOK_GOTO_MODAL_W, EBOOK_GOTO_MODAL_H, 20, 20, 28, 1);
    gfx_draw_rectangle(gfx, EBOOK_GOTO_MODAL_X, EBOOK_GOTO_MODAL_Y, EBOOK_GOTO_MODAL_W, 2, 90, 90, 110, 1);
    gfx_draw_rectangle(gfx, EBOOK_GOTO_MODAL_X, EBOOK_GOTO_MODAL_Y + EBOOK_GOTO_MODAL_H - 2, EBOOK_GOTO_MODAL_W, 2, 90, 90, 110, 1);
    wchar_t buf[32];
    wchar_t digits_w[EBOOK_MAX_GOTO_DIGITS + 2];
    for (int32_t i = 0; i < s_goto_len; i++) digits_w[i] = (wchar_t)s_goto_digits[i];
    digits_w[s_goto_len] = L'\0';
    swprintf(buf, 32, L"跳转到页： %ls_", digits_w);
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, buf, gfx->width / 2, EBOOK_GOTO_MODAL_Y + EBOOK_GOTO_MODAL_H / 2, 255, 255, 255, 1);

    // 触屏数字键盘
    static const wchar_t *pad_label[EBOOK_GOTO_PAD_ROWS][EBOOK_GOTO_PAD_COLS] = {
        {L"1", L"2", L"3", L"退格"},
        {L"4", L"5", L"6", L"0"},
        {L"7", L"8", L"9", L"确认"},
    };
    for (int32_t row = 0; row < EBOOK_GOTO_PAD_ROWS; row++) {
        for (int32_t col = 0; col < EBOOK_GOTO_PAD_COLS; col++) {
            int32_t x = EBOOK_GOTO_PAD_X0 + col * (EBOOK_GOTO_PAD_CELL_W + EBOOK_GOTO_PAD_GAP);
            int32_t y = EBOOK_GOTO_PAD_Y0 + row * (EBOOK_GOTO_PAD_CELL_H + EBOOK_GOTO_PAD_GAP);
            gfx_draw_rectangle(gfx, (uint32_t)x, (uint32_t)y, EBOOK_GOTO_PAD_CELL_W, EBOOK_GOTO_PAD_CELL_H, 32, 32, 44, 1);
            gfx_draw_rectangle(gfx, (uint32_t)x, (uint32_t)y, EBOOK_GOTO_PAD_CELL_W, EBOOK_GOTO_PAD_CELL_H, 90, 90, 110, 0);
            gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_16, pad_label[row][col],
                x + EBOOK_GOTO_PAD_CELL_W / 2, y + EBOOK_GOTO_PAD_CELL_H / 2, 230, 230, 230, 1);
        }
    }
}

int32_t ui_ebook_reading_render(Key_Event *key_event, Global_State *global_state) {
    // 页眉：标题居中、页码为左侧文本（随页更新）、“返回”为右侧文本——全部作为页眉固有部分绘制
    wchar_t page_info[24];
    {
        int32_t cur_page = (s_buf_start_line + global_state->w_textarea_main->current_line)
            / ((s_view_lines > 0) ? s_view_lines : 1) + 1;
        swprintf(page_info, 24, L"%d/%d", cur_page, s_page_count);
        s_scroll_last_page = cur_page; // 与滚动帧轻量渲染的页码跟踪同步（避免重复局部更新）
    }
    ui_draw_header_full(key_event, global_state, s_book_title, 1,
        ui_std_header_height(global_state->ui_font), page_info, L"返回 ");

    // 页脚 4 个触屏软按键（与底部十六宫格 4 格对齐）：上页/下页/跳页/书签（书签为占位符，暂不实现）
    ui_draw_footer_softkeys(key_event, global_state, L"上页", L"下页", L"跳页", L"书签");

    ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

    // 总进度滚动条（绘制于文本控件之后，避免被其背景清除覆盖）
    ui_ebook_draw_progress_bar(key_event, global_state);
    gfx_refresh(global_state->gfx);

    // “跳转到页”模态框 + 触屏数字键盘
    if (s_goto_active) {
        ui_ebook_goto_modal_render(key_event, global_state);
        gfx_refresh(global_state->gfx);
    }
    return 0;
}

// 轻量滚动渲染（拖动/滚行帧专用，性能专题修复 2026-08）：滚动中页眉标题/页脚软按键/
// 缓冲区排版均不变，只重绘文本区本体与总进度条；页码仅跨页变化时经页眉左侧文本
// 局部自清洁更新。否则每帧全量 typeset_line_breaks（4096 wchar）+ 页眉页脚重绘会把
// 脏区扩满全屏、推帧退化为全屏传输——此为拖动不跟手的根因（对齐自述的滚动帧成本）。
static void ui_ebook_reading_render_scroll(Key_Event *key_event, Global_State *global_state) {
    Widget_Textarea_State *ta = global_state->w_textarea_main;

    // 缓冲内容未变：以 is_modified=0 包裹跳过全量重排版（与通用事件处理器同一约定）
    ta->is_modified = 0;
    ui_widget_textarea_draw(key_event, global_state, ta);
    ta->is_modified = 1;

    // 页码仅跨页变化时经页眉左侧文本局部更新（自清洁回填，不触碰标题与“返回”）
    int32_t cur_page = (s_buf_start_line + ta->current_line) / ((s_view_lines > 0) ? s_view_lines : 1) + 1;
    if (cur_page != s_scroll_last_page) {
        s_scroll_last_page = cur_page;
        wchar_t page_info[24];
        swprintf(page_info, 24, L"%d/%d", cur_page, s_page_count);
        ui_draw_header_side_text(key_event, global_state,
            ui_std_header_height(global_state->ui_font), page_info, NULL);
    }

    // 总进度滚动条（绘制于文本控件之后，避免被其背景清除覆盖）
    ui_ebook_draw_progress_bar(key_event, global_state);
    gfx_refresh(global_state->gfx);
}

// 上一页（窗口起点对齐上一页页首）
static void ui_ebook_prev_page(Key_Event *key_event, Global_State *global_state) {
    int32_t cur_page = (s_buf_start_line + global_state->w_textarea_main->current_line) / s_view_lines;
    if (cur_page > 0) {
        ebook_load_window(key_event, global_state, (cur_page - 1) * s_view_lines);
    }
    ui_ebook_reading_render(key_event, global_state);
}

// 下一页（窗口起点对齐下一页页首）
static void ui_ebook_next_page(Key_Event *key_event, Global_State *global_state) {
    int32_t cur_page = (s_buf_start_line + global_state->w_textarea_main->current_line) / s_view_lines;
    if (cur_page < s_page_count - 1) {
        ebook_load_window(key_event, global_state, (cur_page + 1) * s_view_lines);
    }
    ui_ebook_reading_render(key_event, global_state);
}

int32_t ui_ebook_reading_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 上页/下页触屏按钮的按住反复触发状态（500ms 缓冲：按住不足 500ms 松手=单次触发；
    // 按住满 500ms 后，以最大可能频率（每帧一次）反复触发，直到松手）
    static int32_t  s_pad_hold = 0;      // 0-无 1-按住上页 2-按住下页
    static uint64_t s_pad_hold_ts = 0;   // 按下时刻（ms）
    static int32_t  s_pad_repeating = 0; // 1-已进入反复触发

    int32_t band_h = gfx_font_line_height(global_state->ui_font) + 1; // 页脚带高度
    int32_t header_h = ui_std_header_height(global_state->ui_font);   // 页眉带高度（1.5 倍行高）
    int32_t footer_top = (int32_t)global_state->gfx->height - (int32_t)ui_softkbd_height() - band_h;

    // 按下沿：命中上页/下页按钮 → 开始按住跟踪
    if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
        int32_t tx = key_event->touch_down_x;
        int32_t ty = key_event->touch_down_y;
        if (!s_goto_active && ty >= footer_top && ty < footer_top + band_h) {
            int32_t quarter = tx / ((int32_t)global_state->gfx->width / 4);
            if (quarter == 0 || quarter == 1) {
                s_pad_hold = quarter + 1;
                s_pad_hold_ts = global_state->timestamp;
                s_pad_repeating = 0;
            }
        }
    }
    // 按住跟踪中：松手结算/计时/反复触发（本分支独占本次触摸序列）
    if (s_pad_hold != 0) {
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            // 松手只认 UP 边沿事件（可靠投递，不湮灭），不得以触屏电平作为松手判据：
            // Core1 在松手轮询中先写快照(is_touching=0)再经历按键反馈阻塞（misc_led_blink
            // 同步阻塞 ~10ms）才将 UP 事件入队——窗口内 Core0 空转帧看到“电平0+无边沿”，
            // 若按电平清除按住状态，姗姗来迟的 UP 将无人受理，短按必丢（2026-08 实测故障）。
            if (!s_pad_repeating) {
                if (s_pad_hold == 1) ui_ebook_prev_page(key_event, global_state);
                else                 ui_ebook_next_page(key_event, global_state);
            }
            s_pad_hold = 0;
            s_pad_repeating = 0;
        }
        else if (key_event->is_touching) {
            // 按住中：满 500ms 进入反复触发（缓冲期内不响应）
            if (!s_pad_repeating
                && global_state->timestamp - s_pad_hold_ts >= 500) {
                s_pad_repeating = 1;
            }
            if (s_pad_repeating) {
                if (s_pad_hold == 1) ui_ebook_prev_page(key_event, global_state);
                else                 ui_ebook_next_page(key_event, global_state);
            }
        }
        // 其余情形（电平已松但 UP 尚未到达）：保持跟踪不做任何事，等待 UP 事件；
        // 即使 UP 丢失，残留状态也无害——触发只认 UP 边沿、重复只在按住时进行、
        // 下一次 DOWN 会重新武装（覆盖）本状态
        return 0;
    }

    // 触屏拖动像素级连续滚动（全局行空间 + 滑动窗口融合）：
    // 按下沿在文本区内锚定全局滚动位置（px），拖动逐帧 1:1 跟手并拆回行号/亚行偏移，
    // 视口首行越出缓冲窗口时按需从 SD 卡滑动重载（与按键滚行同一窗口语义）；
    // 不启用惯性（跨窗加载需读 SD，惯性会放大抖动）。
    static int32_t s_eb_drag_active = 0;     // 1-正在跟踪一次文本区拖动
    static int32_t s_eb_drag_start_y = 0;    // 按下点 y
    static int32_t s_eb_drag_anchor_px = 0;  // 按下时的全局滚动位置（px）
    {
        int32_t line_height = gfx_font_line_height(global_state->ui_font);
        Widget_Textarea_State *ta = global_state->w_textarea_main;
        if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
            int32_t tx = key_event->touch_down_x;
            int32_t ty = key_event->touch_down_y;
            if (!s_goto_active && tx >= ta->x && tx < ta->x + ta->width
                && ty >= ta->y && ty < ta->y + ta->height) {
                s_eb_drag_active = 1;
                s_eb_drag_start_y = ty;
                s_eb_drag_anchor_px = (s_buf_start_line + ta->current_line) * line_height
                                    + ta->scroll_sub_offset;
            }
        }
        if (s_eb_drag_active) {
            if ((key_event->touch_edge & TOUCH_EDGE_UP) || !key_event->is_touching) {
                s_eb_drag_active = 0; // 序列结束（松手边沿为准，电平兜底）
            }
            else if (key_event->is_touching) {
                int32_t max_px = s_total_lines * line_height - ta->height;
                if (max_px < 0) max_px = 0;
                int32_t new_px = s_eb_drag_anchor_px - (key_event->touch_y - s_eb_drag_start_y);
                if (new_px < 0) new_px = 0;
                if (new_px > max_px) new_px = max_px;
                int32_t top_line = new_px / line_height;
                int32_t new_sub = new_px % line_height;
                if (top_line != s_buf_start_line + ta->current_line
                    || new_sub != ta->scroll_sub_offset) {
                    // 视口越出缓冲窗口：按需重载（向前预留约一页上文；向后窗口起点即视口首行）
                    if (top_line < s_buf_start_line) {
                        int32_t new_start = top_line - s_view_lines;
                        if (new_start < 0) new_start = 0;
                        ebook_load_window(key_event, global_state, new_start);
                    }
                    else if (top_line + s_view_lines > s_buf_start_line + ta->line_num) {
                        ebook_load_window(key_event, global_state, top_line);
                    }
                    ta->current_line = top_line - s_buf_start_line;
                    ta->scroll_sub_offset = new_sub;
                    ui_ebook_reading_render_scroll(key_event, global_state); // 轻量滚动渲染（不重排版/不重绘页眉页脚）
                }
            }
            return 0;
        }
    }

    // 触屏（松手沿 + 按下点坐标；本状态在 ui_app_state_is_menu 抑制表内，无宫格软按键干扰）：
    // 页眉最右侧“返回”软按钮；页脚 4 软按键：上页/下页（按住反复触发，见上）/跳页/书签（占位）
    if (key_event->touch_edge & TOUCH_EDGE_UP) {
        int32_t tx = key_event->touch_down_x;
        int32_t ty = key_event->touch_down_y;
        // 跳页模态框激活：触屏数字键盘命中判定（退格/确认/数字）；键盘区域外松手沿视为取消
        //（模态框数字输入仅认硬按键与虚拟键盘，纯触屏设备上须有触摸逃生通道）
        if (s_goto_active) {
            if (ty >= EBOOK_GOTO_PAD_Y0 && ty < EBOOK_GOTO_PAD_Y1
                && tx >= EBOOK_GOTO_PAD_X0 && tx < EBOOK_GOTO_PAD_X1) {
                int32_t pitch_x = EBOOK_GOTO_PAD_CELL_W + EBOOK_GOTO_PAD_GAP;
                int32_t pitch_y = EBOOK_GOTO_PAD_CELL_H + EBOOK_GOTO_PAD_GAP;
                int32_t col = (tx - EBOOK_GOTO_PAD_X0) / pitch_x;
                int32_t row = (ty - EBOOK_GOTO_PAD_Y0) / pitch_y;
                int32_t in_cell = ((tx - EBOOK_GOTO_PAD_X0) % pitch_x < EBOOK_GOTO_PAD_CELL_W)
                               && ((ty - EBOOK_GOTO_PAD_Y0) % pitch_y < EBOOK_GOTO_PAD_CELL_H);
                if (in_cell && col >= 0 && col < EBOOK_GOTO_PAD_COLS && row >= 0 && row < EBOOK_GOTO_PAD_ROWS) {
                    if (row == 0 && col == 3) {          // 退格
                        if (s_goto_len > 0) s_goto_len--;
                    }
                    else if (row == 2 && col == 3) {     // 确认
                        ui_ebook_goto_confirm(key_event, global_state);
                    }
                    else {                               // 数字（第二行第4列为 0）
                        if (s_goto_len < EBOOK_MAX_GOTO_DIGITS) {
                            char d = (row == 1 && col == 3) ? '0' : (char)('1' + row * 3 + col);
                            s_goto_digits[s_goto_len++] = d;
                        }
                    }
                }
                // 命中键盘区域（含按钮间隙）：不取消模态框
                ui_ebook_reading_render(key_event, global_state);
                return 0;
            }
            s_goto_active = 0;
            ui_ebook_reading_render(key_event, global_state);
            return 0;
        }
        if (ty >= 0 && ty < header_h && tx >= UI_BACK_HOTSPOT_X0((int32_t)global_state->gfx->width)) {
            ui_ebook_close();
            global_state->STATE = STATE_EBOOK;
            return 0;
        }
        if (ty >= footer_top && ty < footer_top + band_h) {
            int32_t quarter = tx / ((int32_t)global_state->gfx->width / 4);
            // quarter 0/1（上页/下页）由上方按住跟踪分支处理，此处不再响应
            if (quarter == 2) {
                s_goto_active = 1;
                s_goto_len = 0;
                ui_ebook_reading_render(key_event, global_state);
            }
            // quarter == 3：书签（占位符，暂不实现）
            return 0;
        }
    }

    if (key_event->key_edge != -1 && key_event->key_edge != -2) {
        return 0;
    }

    // 过渡期：全部按键响应仅认硬按键（软按键——触屏宫格映射/软键盘派生——一律不响应）
    if (key_event->is_soft_key != 0) {
        return 0;
    }

    // 模态框激活：数字输入页码，D确认，←删位，A取消
    if (s_goto_active) {
        if (key_event->key_code >= NANO_KEY_0 && key_event->key_code <= NANO_KEY_9 && key_event->key_edge == -1) {
            if (s_goto_len < EBOOK_MAX_GOTO_DIGITS) {
                s_goto_digits[s_goto_len++] = (char)key_event->key_code;
            }
        }
        else if (key_event->key_code == NANO_KEY_left && key_event->key_edge == -1) {
            if (s_goto_len > 0) s_goto_len--;
        }
        else if (key_event->key_code == NANO_KEY_enter && key_event->key_edge == -1) {
            ui_ebook_goto_confirm(key_event, global_state);
        }
        else if (key_event->key_code == NANO_KEY_esc) {
            s_goto_active = 0;
        }
        ui_ebook_reading_render(key_event, global_state);
        return 0;
    }

    // A(ESC)：关闭本书，返回文件菜单
    if (key_event->key_code == NANO_KEY_esc && key_event->key_edge == -1) {
        ui_ebook_close();
        global_state->STATE = STATE_EBOOK;
        return 0;
    }
    // C(Ctrl)：弹出“跳转到页”模态框
    if (key_event->key_code == NANO_KEY_ctrl && key_event->key_edge == -1) {
        s_goto_active = 1;
        s_goto_len = 0;
        ui_ebook_reading_render(key_event, global_state);
        return 0;
    }
    // ←/→：逐行滚行，与分页取数融合——缓冲区是以视口为中心的滑动窗口，
    // 滚行越出窗口时按需从SD卡滑动重载（页间断续处已在缓冲区内，取数无感知）
    if (key_event->key_code == NANO_KEY_left || key_event->key_code == NANO_KEY_right) {
        Widget_Textarea_State *ta = global_state->w_textarea_main;
        int32_t max_top = (s_total_lines > s_view_lines) ? (s_total_lines - s_view_lines) : 0;
        int32_t top = s_buf_start_line + ta->current_line; // 视口首行的全局行号
        if (key_event->key_code == NANO_KEY_left) {
            top = (top <= 0) ? max_top : (top - 1); // 上滚一行，到顶卷回末页
        }
        else {
            top = (top >= max_top) ? 0 : (top + 1); // 下滚一行，到底卷回首頁
        }
        // 视口越出滑动窗口：按需重载（向前预留约一页上文；向后窗口起点即视口首行）
        if (top < s_buf_start_line) {
            int32_t new_start = top - s_view_lines;
            if (new_start < 0) new_start = 0;
            ebook_load_window(key_event, global_state, new_start);
        }
        else if (top + s_view_lines > s_buf_start_line + ta->line_num) {
            ebook_load_window(key_event, global_state, top);
        }
        ta->current_line = top - s_buf_start_line;
        ta->scroll_sub_offset = 0; // 换窗整行跳转：吸附回整行
        ui_ebook_reading_render_scroll(key_event, global_state); // 轻量滚动渲染（不重排版/不重绘页眉页脚）
        return 0;
    }
    // 4：上一页（窗口起点对齐上一页页首）
    if (key_event->key_code == NANO_KEY_4) {
        ui_ebook_prev_page(key_event, global_state);
        return 0;
    }
    // 6：下一页（窗口起点对齐下一页页首）
    if (key_event->key_code == NANO_KEY_6) {
        ui_ebook_next_page(key_event, global_state);
        return 0;
    }
    return 0;
}

int32_t ui_ebook_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    int32_t idx = ms->current_item_index;
    if (idx < 0 || idx >= s_list_count) {
        return STATE_EBOOK; // 空列表或越界：留在菜单
    }
    if (ui_ebook_open(ke, gs, s_list_mb[idx]) != 0) {
        return STATE_EBOOK; // 打开失败：留在菜单
    }
    return STATE_EBOOK_READING;
}
