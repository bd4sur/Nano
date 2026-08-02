#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ui_ebook.h"
#include "ui_color.h"
#include "ui_softkbd.h"
#include "input_device.h"
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

    int32_t total = list_files("/", NULL);
    if (total > 0) {
        char **names = (char **)platform_calloc((size_t)total, sizeof(char *));
        s_list_mb = (char **)platform_calloc((size_t)total, sizeof(char *));
        s_list_w  = (wchar_t **)platform_calloc((size_t)total, sizeof(wchar_t *));
        s_items   = (const wchar_t **)platform_calloc((size_t)total, sizeof(wchar_t *));
        if (names != NULL && s_list_mb != NULL && s_list_w != NULL && s_items != NULL
            && list_files("/", names) >= 0) {
            for (int32_t i = 0; i < total; i++) {
                if (names[i] == NULL) continue;
                // 规范为带前导'/'的完整路径
                char path[160];
                if (names[i][0] != '/') snprintf(path, sizeof(path), "/%s", names[i]);
                else                    strncpy(path, names[i], sizeof(path) - 1);
                path[sizeof(path) - 1] = '\0';
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
                // 显示名：去掉前导'/'，UTF-8 转宽字符
                const uint8_t *p = (const uint8_t *)(path + 1);
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

    // 重置文本控件几何（LLM观测模式会修改 x/width；阅读恢复页眉页脚间全宽布局）
    Widget_Textarea_State *ta = global_state->w_textarea_main;
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    int32_t bar_height = line_height + 1;
    ta->x = 0;
    ta->y = bar_height;
    ta->width = global_state->gfx->width;
    ta->height = global_state->gfx->height - ui_softkbd_height() - bar_height * 2;
    ta->is_show_scroll_bar = 0; // 关闭控件自带滚动条（缓冲区内行位置），改由本模块绘制总进度条
    s_ta_width = ta->width;
    s_view_lines = (ta->height + 1) / line_height;
    if (s_view_lines <= 0) s_view_lines = 1;

    // 提示正在打开（预扫描大文件需要数秒）：蓝底白字，置于屏幕顶部中央（同“正在计算”横幅）
    gfx_draw_rectangle(global_state->gfx, global_state->gfx->width / 2 - 70, 0, 140, 14, 0x11, 0x55, 0xee, 1);
    gfx_draw_textline_centered(global_state->gfx, L"正在打开，请稍候...", global_state->gfx->width / 2, 7, 255, 255, 255, 1);
    gfx_refresh(global_state->gfx);

    if (platform_file_open(path_mb) != 0) {
        return -1;
    }
    s_book_open = 1;

    // 标题：文件名（去掉前导'/'）
    const uint8_t *p = (const uint8_t *)path_mb;
    while (*p == '/') p++;
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
    ta->is_modified = 1;
    s_buf_start_line = start_line;
    // 立即排版，使 line_num/view_lines 可供滑动判定使用
    typeset_line_breaks(key_event, global_state, ta);
}

// ===============================================================================
// 阅读渲染/事件
// ===============================================================================

// 总进度滚动条：显示当前页在全部页中的位置（替代文本控件自带的“缓冲区内行位置”滚动条）
static void ui_ebook_draw_progress_bar(Key_Event *key_event, Global_State *global_state) {
    Widget_Textarea_State *ta = global_state->w_textarea_main;
    uint8_t bg_R = 128, bg_G = 128, bg_B = 128;
    uint8_t fg_R = 255, fg_G = 255, fg_B = 255;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        bg_R = 200; bg_G = 200; bg_B = 200;
        fg_R = 33;  fg_G = 33;  fg_B = 33;
    }
    int32_t x = ta->x + ta->width - 1;
    int32_t y = ta->y;
    int32_t h = ta->height;
    // 轨道
    gfx_draw_line(global_state->gfx, x, y, x, y + h, bg_R, bg_G, bg_B, 1);
    int32_t pages = (s_page_count <= 0) ? 1 : s_page_count;
    int32_t bar_h = h / pages;
    if (bar_h < 3) bar_h = 3;
    if (bar_h > h) bar_h = h;
    int32_t y0 = y;
    if (pages > 1) {
        // 以视口首行所在页表示总进度
        int32_t cur_page = (s_buf_start_line + ta->current_line) / ((s_view_lines > 0) ? s_view_lines : 1);
        y0 = y + cur_page * (h - bar_h) / (pages - 1);
    }
    gfx_draw_line(global_state->gfx, x, y0, x, y0 + bar_h, fg_R, fg_G, fg_B, 1);
    gfx_draw_line(global_state->gfx, x - 1, y0, x - 1, y0 + bar_h, fg_R, fg_G, fg_B, 1);
}

int32_t ui_ebook_reading_render(Key_Event *key_event, Global_State *global_state) {
    ui_draw_header(key_event, global_state, s_book_title, 1);

    wchar_t footer[48];
    int32_t cur_page = (s_buf_start_line + global_state->w_textarea_main->current_line)
        / ((s_view_lines > 0) ? s_view_lines : 1) + 1;
    swprintf(footer, 48, L"第%d/%d页 4/6翻页 C跳页 A返回", cur_page, s_page_count);
    ui_draw_footer(key_event, global_state, footer, 1);

    ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

    // 总进度滚动条（绘制于文本控件之后，避免被其背景清除覆盖）
    ui_ebook_draw_progress_bar(key_event, global_state);
    gfx_refresh(global_state->gfx);

    // “跳转到页”模态框
    if (s_goto_active) {
        Nano_GFX *gfx = global_state->gfx;
        gfx_draw_rectangle(gfx, 60, 96, 200, 44, 20, 20, 28, 1);
        gfx_draw_rectangle(gfx, 60, 96, 200, 2, 90, 90, 110, 1);
        gfx_draw_rectangle(gfx, 60, 138, 200, 2, 90, 90, 110, 1);
        wchar_t buf[32];
        wchar_t digits_w[EBOOK_MAX_GOTO_DIGITS + 2];
        for (int32_t i = 0; i < s_goto_len; i++) digits_w[i] = (wchar_t)s_goto_digits[i];
        digits_w[s_goto_len] = L'\0';
        swprintf(buf, 32, L"跳转到页： %ls_", digits_w);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, buf, gfx->width / 2, 118, 255, 255, 255, 1);
        gfx_refresh(gfx);
    }
    return 0;
}

int32_t ui_ebook_reading_event_handler(Key_Event *key_event, Global_State *global_state) {
    if (key_event->key_edge != -1 && key_event->key_edge != -2) {
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
            s_goto_digits[s_goto_len] = '\0';
            int32_t p = (s_goto_len > 0) ? atoi(s_goto_digits) : 0;
            s_goto_active = 0;
            if (p >= 1) {
                // 跳页：窗口起点对齐到目标页首行，视口置于缓冲区顶部
                ebook_load_window(key_event, global_state, (p - 1) * s_view_lines);
            }
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
        ui_ebook_reading_render(key_event, global_state);
        return 0;
    }
    // 4：上一页（窗口起点对齐上一页页首）
    if (key_event->key_code == NANO_KEY_4) {
        int32_t cur_page = (s_buf_start_line + global_state->w_textarea_main->current_line) / s_view_lines;
        if (cur_page > 0) {
            ebook_load_window(key_event, global_state, (cur_page - 1) * s_view_lines);
        }
        ui_ebook_reading_render(key_event, global_state);
        return 0;
    }
    // 6：下一页（窗口起点对齐下一页页首）
    if (key_event->key_code == NANO_KEY_6) {
        int32_t cur_page = (s_buf_start_line + global_state->w_textarea_main->current_line) / s_view_lines;
        if (cur_page < s_page_count - 1) {
            ebook_load_window(key_event, global_state, (cur_page + 1) * s_view_lines);
        }
        ui_ebook_reading_render(key_event, global_state);
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
