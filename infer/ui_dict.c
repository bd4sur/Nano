// ui_dict.c - 英汉电子词典（硬件无关核心 + 查询/详情状态驱动，见 ui_dict.h 头注释）
#include <stdio.h>
#include <string.h>

#include "ui_dict.h"
#include "ui_color.h"

// ===============================================================================
// 配置
// ===============================================================================
#define UI_DICT_CSV_PATH        "/dict/ecdict.csv" // 英汉词典CSV（word,phonetic,definition）
#define UI_DICT_IDX_PATH        "/dict/ecdict.idx" // 设备端生成的二进制索引（RAM映像）
#define UI_DICT_DIR_PATH        "/dict"            // 词典目录（写索引前确保存在）
#define UI_DICT_IDX_MAGIC       (0x4E444354)  // "NDCT"
#define UI_DICT_MATCH_MAX       (64)          // 候选词条数上限
#define UI_DICT_WORD_WCHAR_MAX  (48)          // 候选表单个词条宽字符缓冲长度
#define UI_DICT_WORD_CHAR_MAX   (64)          // 索引词条UTF-8长度上限（超出截断）
#define UI_DICT_ENTRY_READ_MAX  (3200)        // 详情单条记录读取上限（超出截断）
#define UI_DICT_SCAN_CHUNK      (4096)        // 索引构建扫描块大小

// ===============================================================================
// 索引映像（/dict.idx 文件布局 == RAM布局，加载零修复）
// ===============================================================================
typedef struct {
    uint32_t magic;
    uint32_t csv_size;     // 指纹：CSV文件大小
    uint32_t csv_hash;     // 指纹：CSV前4KB的djb2
    uint32_t item_num;     // 条目数N
    uint32_t pool_bytes;   // 词条池字节数P
} Dict_Idx_Header;

typedef struct {
    uint32_t word_pool_ofs; // 词条在池内偏移（UTF-8，\0结尾）
    uint32_t file_ofs;      // 记录在CSV文件中的字节偏移
} Dict_Index_Item;

static uint8_t *s_idx_blob = NULL;       // 整个索引映像（header + items + pool，PSRAM）
static Dict_Index_Item *s_items = NULL;  // = blob + sizeof(Dict_Idx_Header)
static char *s_pool = NULL;              // = blob + sizeof(header) + N*sizeof(item)
static uint32_t s_item_num = 0;

// ===============================================================================
// 词典模块状态
// ===============================================================================
static wchar_t s_prefix[UI_DICT_WORD_WCHAR_MAX];         // 查询前缀（宽字符）
static int32_t s_prefix_len = 0;
static char    s_prefix_utf8[UI_DICT_WORD_CHAR_MAX * 2]; // 查询前缀（UTF-8，供二分）

static int32_t s_match_num = 0;                          // 真实匹配数（不含提示行）
static int32_t s_match_index[UI_DICT_MATCH_MAX];         // 匹配项 → 索引下标
static wchar_t *s_match_wbuf = NULL;                     // 候选词宽字符缓冲（PSRAM，MATCH_MAX*WORD_WCHAR_MAX）
static const wchar_t *s_match_items[UI_DICT_MATCH_MAX];  // 菜单条目指针表（借用式）

static const wchar_t S_HINT_EMPTY[] = L"（输入前缀以检索）";
static const wchar_t S_HINT_NOMATCH[] = L"（无匹配词条）";

static int32_t s_detail_match_pos = 0;   // 详情当前词条在匹配表中的位置
static wchar_t *s_detail_buf = NULL;     // 详情内容拼装缓冲（PSRAM，进入时申请/退出释放；
                                         // 不得放静态区——内部RAM需为启动时DMA帧缓冲保留连续大块）
static uint8_t *s_fetch_buf = NULL;      // 单条记录读取缓冲（PSRAM，生命周期同上）
static char *s_entry_buf = NULL;         // 详情取词输出缓冲（PSRAM，64词+256音标+3200释义，生命周期同上）
static wchar_t *s_entry_wbuf = NULL;     // 详情宽字符转换缓冲（PSRAM，48+256+3200 个wchar_t，生命周期同上；
                                         // 不得放局部数组——3200个wchar_t会撑爆渲染任务12KB栈）

static uint32_t s_prev_ui_font = GFX_FONT_ALPHA_16; // 进入词典前的ui_font（退出时恢复）

// ===============================================================================
// 工具：ASCII大小写不敏感比较（词典词条为英文，不处理多字节大小写）
// ===============================================================================
static char ci_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
}
// 全串比较：word 与 key 的大小写不敏感 strcmp
static int ci_cmp(const char *word, const char *key) {
    while (*word && *key) {
        char a = ci_lower(*word), b = ci_lower(*key);
        if (a != b) return (a < b) ? -1 : 1;
        word++; key++;
    }
    if (*word) return 1;
    if (*key) return -1;
    return 0;
}
// 前缀匹配：word 是否以 prefix 开头（大小写不敏感）
static int32_t ci_prefix_match(const char *word, const char *prefix) {
    while (*prefix) {
        if (!*word || ci_lower(*word) != ci_lower(*prefix)) return 0;
        word++; prefix++;
    }
    return 1;
}
static uint32_t djb2(const uint8_t *data, size_t len) {
    uint32_t h = 5381;
    for (size_t i = 0; i < len; i++) h = ((h << 5) + h) + data[i];
    return h;
}

// ===============================================================================
// CSV 引号状态机（构建与取词共用）：
// 字段仅在被引号包裹时允许包含逗号/换行；引号字段内 "" 转义为 "。
// ===============================================================================
typedef struct {
    int32_t field_idx;      // 当前字段序号（0=word 1=phonetic 2=definition）
    int32_t in_quoted;      // 当前字段为引号字段
    int32_t after_quote;    // 引号字段中刚遇到一个 "（等待判定转义或结束）
} Csv_Scan_State;

static void csv_scan_reset(Csv_Scan_State *st) {
    st->field_idx = 0;
    st->in_quoted = 0;
    st->after_quote = 0;
}

// 逐字节喂入。当应输出字符（属于当前字段内容）时返回1（*out 为字符），否则返回0。
// 字段结束（逗号）时返回2；记录结束（换行）时返回3。
static int csv_scan_feed(Csv_Scan_State *st, char c, char *out) {
    if (st->after_quote) {
        // 引号字段内遇到 " 之后：再一个 " 是转义；逗号结束字段；换行结束记录；其它视为字段外字符
        st->after_quote = 0;
        if (c == '"') { *out = '"'; return 1; }
        if (c == ',') { st->in_quoted = 0; st->field_idx++; return 2; }
        if (c == '\n' || c == '\r') { st->in_quoted = 0; return 3; }
        return 0; // 引号字段后的空白等杂散字符，忽略
    }
    if (st->in_quoted) {
        if (c == '"') { st->after_quote = 1; return 0; }
        *out = c; return 1; // 引号字段内一切字符原样（含逗号/换行）
    }
    // 字段起始或非引号字段中
    if (c == '"') { st->in_quoted = 1; return 0; }
    if (c == ',') { st->field_idx++; return 2; }
    if (c == '\n' || c == '\r') { return 3; }
    *out = c; return 1;
}

// ===============================================================================
// 索引构建（设备端，两遍扫描 + 精确分配 + 进度显示）
//   Pass 1 仅计数（条目数/词条池字节数/有序性），随后一次性精确申请最终映像，
//   Pass 2 直接填充该映像——内存峰值仅为 blob 本身（≈2.3MB）+ 4KB 扫描块，
//   适配 4MB PSRAM 机型（旧单遍倍增扩容+末尾组装方案峰值约 6.3MB，4MB 板必然失败）；
//   代价是扫描两遍（构建仅在首次/指纹失效时运行一次，慢一点可接受）。
// ===============================================================================
static const char *s_qsort_pool = NULL; // qsort比较器用（模块单线程）
static int dict_idx_cmp(const void *pa, const void *pb) {
    const Dict_Index_Item *a = (const Dict_Index_Item *)pa;
    const Dict_Index_Item *b = (const Dict_Index_Item *)pb;
    return ci_cmp(s_qsort_pool + a->word_pool_ofs, s_qsort_pool + b->word_pool_ofs);
}

// 构建进度显示（定义见下文 UI 辅助一节）
static void ui_dict_draw_progress(Global_State *gs, const wchar_t *title, int32_t percent);

#define DICT_SCAN_COUNT (0) // 计数模式：只统计条目数与词条池字节数、校验有序性
#define DICT_SCAN_FILL  (1) // 填充模式：直接写入精确分配的 blob 区域

typedef struct {
    int32_t mode;
    // COUNT 输出（FILL 时作为防御上限）
    uint32_t item_count;
    uint32_t pool_bytes;
    int32_t is_sorted;
    // FILL 目标与进度
    Dict_Index_Item *items;
    char *pool;
    uint32_t items_filled;
    uint32_t pool_filled;
    int32_t overflow; // 防御：FILL 超出 COUNT 计数（同一确定性解析器，理论上不会发生）
} Dict_Scan_Ctx;

// 单遍顺序扫描 CSV（引号状态机；BOM 跳过、空行与表头跳过）。返回0成功；FILL 越界返回-2。
static int32_t dict_scan_csv(Global_State *gs, uint32_t csv_size, Dict_Scan_Ctx *ctx,
    const wchar_t *progress_title, int32_t pct_base, int32_t pct_range) {
    uint8_t *chunk = (uint8_t *)platform_malloc(UI_DICT_SCAN_CHUNK);
    if (!chunk) return -1;

    Csv_Scan_State st;
    csv_scan_reset(&st);
    char word[UI_DICT_WORD_CHAR_MAX]; uint32_t word_len = 0;       // 字段0（索引词条）
    char phon[16];                     uint32_t phon_len = 0;      // 字段1（仅用于表头识别）
    uint32_t record_ofs = 0;                                       // 当前记录起始文件偏移
    uint32_t record_seq = 0;                                       // 已提交记录数（含被跳过的表头）
    char prev_word[UI_DICT_WORD_CHAR_MAX] = {0};

    // UTF-8 BOM（EF BB BF）跳过：部分编辑器导出的CSV带BOM，否则首条记录词条会混入BOM字节
    uint32_t ofs = 0;
    {
        uint8_t bom[3] = {0};
        platform_file_seek(0);
        if (platform_file_read(bom, 3) == 3 && bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF) {
            ofs = 3;
        }
    }
    record_ofs = ofs;

    uint32_t progress_mark = 0;
    int32_t done = 0, ret = 0;
    while (!done && ret == 0) {
        platform_file_seek(ofs);
        int32_t n = platform_file_read(chunk, UI_DICT_SCAN_CHUNK);
        if (n <= 0) break;
        for (int32_t i = 0; i < n; i++) {
            char c = (char)chunk[i], out = 0;
            int r = csv_scan_feed(&st, c, &out);
            if (r == 1) {
                if (st.field_idx == 0 && word_len < UI_DICT_WORD_CHAR_MAX - 1) word[word_len++] = out;
                else if (st.field_idx == 1 && phon_len < sizeof(phon) - 1) phon[phon_len++] = out;
            }
            else if (r == 3) {
                // 记录结束：提交（跳过空行与CSV表头）
                word[word_len] = '\0'; phon[phon_len] = '\0';
                int32_t is_blank = (word_len == 0 && st.field_idx == 0);
                int32_t is_header = (record_seq == 0 && strcmp(word, "word") == 0 && strcmp(phon, "phonetic") == 0);
                if (!is_blank && !is_header && word_len > 0) {
                    if (ctx->mode == DICT_SCAN_COUNT) {
                        if (ctx->is_sorted && ctx->item_count > 0 && ci_cmp(prev_word, word) > 0) ctx->is_sorted = 0;
                        strcpy(prev_word, word);
                        ctx->item_count++;
                        ctx->pool_bytes += word_len + 1;
                    }
                    else {
                        if (ctx->items_filled >= ctx->item_count ||
                            ctx->pool_filled + word_len + 1 > ctx->pool_bytes) {
                            ctx->overflow = 1;
                            ret = -2;
                            break;
                        }
                        ctx->items[ctx->items_filled].word_pool_ofs = ctx->pool_filled;
                        ctx->items[ctx->items_filled].file_ofs = record_ofs;
                        ctx->items_filled++;
                        memcpy(ctx->pool + ctx->pool_filled, word, word_len + 1);
                        ctx->pool_filled += word_len + 1;
                    }
                }
                record_seq++;
                // 复位，准备下一条记录
                csv_scan_reset(&st);
                word_len = 0; phon_len = 0;
                record_ofs = ofs + (uint32_t)i + 1;
            }
        }
        ofs += (uint32_t)n;
        if (ofs - progress_mark >= 256 * 1024) {
            progress_mark = ofs;
            ui_dict_draw_progress(gs, progress_title, pct_base + (int32_t)((uint64_t)ofs * pct_range / csv_size));
        }
        if ((uint32_t)n < UI_DICT_SCAN_CHUNK) done = 1; // 读到文件尾
    }
    free(chunk);
    return ret;
}

// 构建进度显示（简单文本+进度条，风格随全局色彩）
static void ui_dict_draw_progress(Global_State *gs, const wchar_t *title, int32_t percent) {
    if (gs->ui_color_style == UI_COLOR_LIGHT) { gfx_fill_white(gs->gfx); }
    else                                      { gfx_soft_clear(gs->gfx); }
    uint8_t fg_R = 0, fg_G = 0, fg_B = 0;
    if (gs->ui_color_style == UI_COLOR_LIGHT) { fg_R = 0; fg_G = 0; fg_B = 0; }
    else                                      { fg_R = 255; fg_G = 255; fg_B = 255; }
    int32_t cx = gs->gfx->width / 2, cy = gs->gfx->height / 2;
    gfx_font_draw_text_centered(gs->gfx, GFX_FONT_ALPHA_16, (wchar_t *)title, cx, cy - 24, fg_R, fg_G, fg_B, 1);
    wchar_t buf[32];
    swprintf(buf, 32, L"%d%%", percent);
    gfx_font_draw_text_centered(gs->gfx, GFX_FONT_ALPHA_16, buf, cx, cy + 8, fg_R, fg_G, fg_B, 1);
    int32_t bar_w = gs->gfx->width - 60;
    gfx_draw_rectangle(gs->gfx, 30, cy + 32, bar_w, 4, 128, 128, 128, 1);
    gfx_draw_rectangle(gs->gfx, 30, cy + 32, bar_w * percent / 100, 4, 17, 85, 238, 1);
    gfx_refresh(gs->gfx);
}

// 从 s_idx_blob 建立运行时指针；调用前 blob 已就绪（加载或刚构建）
static void dict_bind_blob(void) {
    Dict_Idx_Header *h = (Dict_Idx_Header *)s_idx_blob;
    s_item_num = h->item_num;
    s_items = (Dict_Index_Item *)(s_idx_blob + sizeof(Dict_Idx_Header));
    s_pool = (char *)(s_idx_blob + sizeof(Dict_Idx_Header) + (size_t)h->item_num * sizeof(Dict_Index_Item));
}

// 计算CSV指纹（文件已以 platform_file_open 打开）
static uint32_t dict_csv_hash(uint32_t csv_size) {
    uint8_t buf[4096];
    uint32_t n = (csv_size < sizeof(buf)) ? csv_size : (uint32_t)sizeof(buf);
    platform_file_seek(0);
    int32_t got = platform_file_read(buf, n);
    return (got > 0) ? djb2(buf, (size_t)got) : 0;
}

// 校验已载入的索引与当前CSV指纹是否一致（0一致）
static int32_t dict_index_validate(Global_State *gs, uint32_t csv_size) {
    Dict_Idx_Header *h = (Dict_Idx_Header *)s_idx_blob;
    if (h->magic != UI_DICT_IDX_MAGIC || h->item_num == 0) return -1;
    if (h->csv_size != csv_size) return -1;
    if (h->csv_hash != dict_csv_hash(csv_size)) return -1;
    return 0;
}

// 两遍扫描构建索引（含排序兜底、写盘、进度显示）。返回0成功。
static int32_t dict_index_build(Key_Event *key_event, Global_State *gs, uint32_t csv_size) {
    // Pass 1：计数（仅 4KB 扫描块，无大块分配）
    Dict_Scan_Ctx cnt;
    memset(&cnt, 0, sizeof(cnt));
    cnt.mode = DICT_SCAN_COUNT;
    cnt.is_sorted = 1;
    if (dict_scan_csv(gs, csv_size, &cnt, L"正在分析词典(1/2)…", 0, 45) != 0) return -1;
    if (cnt.item_count == 0) return -1;

    // 精确分配最终映像（即索引文件布局；构建期唯一大块，≈2.3MB）
    size_t total = sizeof(Dict_Idx_Header) + (size_t)cnt.item_count * sizeof(Dict_Index_Item) + cnt.pool_bytes;
    s_idx_blob = (uint8_t *)platform_malloc(total);
    if (!s_idx_blob) return -1;

    // Pass 2：直接填充 blob（容量精确已知，带防御性越界检查）
    Dict_Scan_Ctx fill;
    memset(&fill, 0, sizeof(fill));
    fill.mode = DICT_SCAN_FILL;
    fill.item_count = cnt.item_count;  // 作为防御上限
    fill.pool_bytes = cnt.pool_bytes;
    fill.items = (Dict_Index_Item *)(s_idx_blob + sizeof(Dict_Idx_Header));
    fill.pool = (char *)(s_idx_blob + sizeof(Dict_Idx_Header) + (size_t)cnt.item_count * sizeof(Dict_Index_Item));
    if (dict_scan_csv(gs, csv_size, &fill, L"正在构建索引(2/2)…", 45, 45) != 0 ||
        fill.items_filled != cnt.item_count || fill.pool_filled != cnt.pool_bytes) {
        free(s_idx_blob);
        s_idx_blob = NULL;
        return -1;
    }

    // 排序兜底：CSV未按词条排序时，对 blob 内索引数组原地排序
    if (!cnt.is_sorted) {
        ui_dict_draw_progress(gs, L"索引排序中…", 90);
        s_qsort_pool = fill.pool;
        qsort(fill.items, fill.items_filled, sizeof(Dict_Index_Item), dict_idx_cmp);
    }

    // 头部
    Dict_Idx_Header *h = (Dict_Idx_Header *)s_idx_blob;
    h->magic = UI_DICT_IDX_MAGIC;
    h->csv_size = csv_size;
    h->csv_hash = dict_csv_hash(csv_size);
    h->item_num = fill.items_filled;
    h->pool_bytes = fill.pool_filled;
    dict_bind_blob();

    // 写盘（失败不致命：本次仍可用内存索引，下次启动重建）
    ui_dict_draw_progress(gs, L"正在保存索引…", 100);
    platform_mkdir(UI_DICT_DIR_PATH); // SD 的 open(FILE_WRITE) 不会自动创建父目录
    platform_write_buffer_to_file(UI_DICT_IDX_PATH, s_idx_blob, total);
    return 0;
}

// 尝试从 /dict.idx 加载索引并校验指纹。返回0成功。
static int32_t dict_index_load(Key_Event *key_event, Global_State *gs, uint32_t csv_size) {
    size_t blob_size = 0;
    s_idx_blob = NULL;
    if (platform_read_file_to_buffer(UI_DICT_IDX_PATH, &s_idx_blob, &blob_size) != 0) {
        return -1;
    }
    Dict_Idx_Header *h = (Dict_Idx_Header *)s_idx_blob;
    size_t expect = sizeof(Dict_Idx_Header) + (size_t)h->item_num * sizeof(Dict_Index_Item) + h->pool_bytes;
    if (blob_size != expect || dict_index_validate(gs, csv_size) != 0) {
        free(s_idx_blob);
        s_idx_blob = NULL;
        return -1;
    }
    dict_bind_blob();
    return 0;
}

// ===============================================================================
// 查询核心：前缀二分 + 候选枚举
// ===============================================================================
static int32_t dict_lower_bound(const char *prefix) {
    int32_t lo = 0, hi = (int32_t)s_item_num; // [lo, hi)
    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        if (ci_cmp(s_pool + s_items[mid].word_pool_ofs, prefix) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// 取单条记录并拆出三列（去引号/还原转义）。返回0成功。
// 调用前提：s_fetch_buf 已分配（ui_dict_enter）；模块单线程运行，无重入
static int32_t dict_fetch_entry(uint32_t idx,
    char *word_out, uint32_t wcap, char *phon_out, uint32_t pcap, char *def_out, uint32_t dcap) {
    uint8_t *buf = s_fetch_buf;
    platform_file_seek(s_items[idx].file_ofs);
    int32_t n = platform_file_read(buf, UI_DICT_ENTRY_READ_MAX);
    if (n <= 0) return -1;
    Csv_Scan_State st;
    csv_scan_reset(&st);
    uint32_t wl = 0, pl = 0, dl = 0;
    for (int32_t i = 0; i < n; i++) {
        char out = 0;
        int r = csv_scan_feed(&st, (char)buf[i], &out);
        if (r == 1) {
            if (st.field_idx == 0 && wl < wcap - 1) word_out[wl++] = out;
            else if (st.field_idx == 1 && pl < pcap - 1) phon_out[pl++] = out;
            else if (st.field_idx >= 2 && dl < dcap - 1) def_out[dl++] = out; // 第3列起全部并入释义
        }
        else if (r == 3) break; // 记录结束
    }
    word_out[wl] = '\0'; phon_out[pl] = '\0'; def_out[dl] = '\0';
    return 0;
}

// ===============================================================================
// UI辅助
// ===============================================================================

// 错误画面（文件缺失/索引构建失败等）：显示后由调用方决定是否停留
static void ui_dict_draw_error(Global_State *gs, const wchar_t *line1, const wchar_t *line2) {
    if (gs->ui_color_style == UI_COLOR_LIGHT) { gfx_fill_white(gs->gfx); }
    else                                      { gfx_soft_clear(gs->gfx); }
    uint8_t fg_R = 0, fg_G = 0, fg_B = 0;
    if (gs->ui_color_style == UI_COLOR_LIGHT) { fg_R = 0; fg_G = 0; fg_B = 0; }
    else                                      { fg_R = 255; fg_G = 255; fg_B = 255; }
    int32_t cx = gs->gfx->width / 2, cy = gs->gfx->height / 2;
    gfx_font_draw_text_centered(gs->gfx, GFX_FONT_ALPHA_16, (wchar_t *)line1, cx, cy - 12, fg_R, fg_G, fg_B, 1);
    gfx_font_draw_text_centered(gs->gfx, GFX_FONT_ALPHA_16, (wchar_t *)line2, cx, cy + 12, fg_R, fg_G, fg_B, 1);
    gfx_refresh(gs->gfx);
}

// 查询前缀行（header之下固定一行高，字体跟随 ui_font=GFX_FONT_ALPHA_12）
static void ui_dict_draw_query_line(Key_Event *key_event, Global_State *gs) {
    uint32_t font_id = gs->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    int32_t y = line_height + 1; // header 高度
    // 查询前缀文字颜色随全局色彩风格：暗色黄（#ffff00）、亮色蓝（#1155ee）
    uint8_t bg_R = 255, bg_G = 255, bg_B = 255, fg_R = 17, fg_G = 85, fg_B = 238;
    if (gs->ui_color_style == UI_COLOR_DARK) {
        bg_R = 0; bg_G = 0; bg_B = 0;
        fg_R = 255; fg_G = 255; fg_B = 0;
    }
    gfx_draw_rectangle(gs->gfx, 0, y, gs->gfx->width, line_height, bg_R, bg_G, bg_B, 1);
    wchar_t buf[UI_DICT_WORD_WCHAR_MAX + 2];
    wcscpy(buf, s_prefix);
    wcscat(buf, L"_"); // 简易光标
    gfx_font_draw_text(gs->gfx, font_id, buf, 6, y, fg_R, fg_G, fg_B, 1);
}

// 菜单布局微调：为查询前缀行让出一行（ui_widget_menu_init 已扣除软键盘高度与页眉页脚），
// 并按当前 item_num 重算 items_per_page（与 ui_widget_menu_init 内公式一致；不修改通用组件）
static void ui_dict_menu_relayout(Key_Event *key_event, Global_State *gs) {
    Widget_Menu_State *ms = gs->w_menu_main;
    int32_t line_height = gfx_font_line_height(gs->ui_font);
    ms->current_item_index = 0;
    ms->first_item_intex = 0;
    // 每页条目数按像素精确撑满菜单区域：首行顶为 y+1、每行占 line_height，
    // 第 n 行底为 y + n*line_height，故 n_max = height / line_height（比通用公式多利用余量行）
    uint32_t max_items = (uint32_t)ms->height / (uint32_t)line_height;
    ms->items_per_page = (ms->item_num > (int32_t)max_items) ? (int32_t)max_items : ms->item_num;
    if (ms->items_per_page < 1) ms->items_per_page = 1;
}

// 按当前前缀重建候选表（item_num 恒 >=1：空前缀/零匹配显示提示行，复用菜单绘制与高亮）
static void ui_dict_rebuild_matches(Key_Event *key_event, Global_State *gs) {
    Widget_Menu_State *ms = gs->w_menu_main;
    s_match_num = 0;
    if (s_prefix_len > 0) {
        _wcstombs(s_prefix_utf8, s_prefix, sizeof(s_prefix_utf8) - 1);
        int32_t lb = dict_lower_bound(s_prefix_utf8);
        for (int32_t i = lb; i < (int32_t)s_item_num && s_match_num < UI_DICT_MATCH_MAX; i++) {
            const char *word = s_pool + s_items[i].word_pool_ofs;
            if (!ci_prefix_match(word, s_prefix_utf8)) break;
            s_match_index[s_match_num] = i;
            _mbstowcs(s_match_wbuf + s_match_num * UI_DICT_WORD_WCHAR_MAX, word, UI_DICT_WORD_WCHAR_MAX - 1);
            s_match_items[s_match_num] = s_match_wbuf + s_match_num * UI_DICT_WORD_WCHAR_MAX;
            s_match_num++;
        }
    }
    if (s_prefix_len == 0) {
        ms->item_num = 1;
        s_match_items[0] = S_HINT_EMPTY;
    }
    else if (s_match_num == 0) {
        ms->item_num = 1;
        s_match_items[0] = S_HINT_NOMATCH;
    }
    else {
        ms->item_num = s_match_num;
    }
    ui_dict_menu_relayout(key_event, gs);
}

// 查询界面全量重绘：header/查询行/软键盘先入帧缓冲，菜单绘制（仅清自身区域）末尾统一推帧
static void ui_dict_query_redraw(Key_Event *key_event, Global_State *gs) {
    ui_draw_header(key_event, gs, L"电子词典", 1);
    ui_dict_draw_query_line(key_event, gs);
    ui_softkbd_draw(gs->gfx, (uint8_t)gs->is_ctrl_enabled);
    ui_widget_menu_draw(key_event, gs, gs->w_menu_main); // 末尾自带 gfx_refresh
}

// ===============================================================================
// 进入 / 退出
// ===============================================================================
int32_t ui_dict_enter(Key_Event *key_event, Global_State *global_state) {
    // 打开CSV并保持单句柄（详情取词经随机访问API使用，退出时关闭）
    if (platform_file_open(UI_DICT_CSV_PATH) != 0) {
        ui_dict_draw_error(global_state, L"未找到词典文件", L"请将 ecdict.csv 放入SD卡 /dict 目录");
        return -1;
    }
    uint32_t csv_size = platform_file_size();
    if (csv_size == 0) {
        ui_dict_draw_error(global_state, L"词典文件为空", L"/dict/ecdict.csv");
        platform_file_close();
        return -1;
    }

    // 顶部提示（仿 gfx_draw_busy / 电子书“正在打开，请稍候”）：加载与构建均需耗时，
    // 进入即给出即时反馈；构建分支随后以进度画面接管
    gfx_draw_rectangle(global_state->gfx, global_state->gfx->width / 2 - 90, 0, 180, 14, 0x11, 0x55, 0xee, 1);
    gfx_draw_textline_centered(global_state->gfx, L"正在构建索引，请稍候……", global_state->gfx->width / 2, 7, 255, 255, 255, 1);
    gfx_refresh(global_state->gfx);

    // 索引：加载校验失败则重建（带进度）
    if (dict_index_load(key_event, global_state, csv_size) != 0) {
        if (dict_index_build(key_event, global_state, csv_size) != 0) {
            ui_dict_draw_error(global_state, L"词典索引构建失败", L"内存不足或文件损坏");
            platform_file_close();
            return -1;
        }
    }

    // 候选词宽字符缓冲、详情拼装缓冲、记录读取缓冲（均PSRAM）
    s_match_wbuf = (wchar_t *)platform_malloc(UI_DICT_MATCH_MAX * UI_DICT_WORD_WCHAR_MAX * sizeof(wchar_t));
    s_detail_buf = (wchar_t *)platform_malloc(UI_STR_BUF_MAX_LENGTH * sizeof(wchar_t));
    s_fetch_buf = (uint8_t *)platform_malloc(UI_DICT_ENTRY_READ_MAX);
    s_entry_buf = (char *)platform_malloc(UI_DICT_WORD_CHAR_MAX + 256 + UI_DICT_ENTRY_READ_MAX);
    s_entry_wbuf = (wchar_t *)platform_malloc((UI_DICT_WORD_WCHAR_MAX + 256 + UI_DICT_ENTRY_READ_MAX) * sizeof(wchar_t));
    if (!s_match_wbuf || !s_detail_buf || !s_fetch_buf || !s_entry_buf || !s_entry_wbuf) {
        ui_dict_draw_error(global_state, L"内存不足", L"PSRAM不足");
        if (s_match_wbuf) { free(s_match_wbuf); s_match_wbuf = NULL; }
        if (s_detail_buf) { free(s_detail_buf); s_detail_buf = NULL; }
        if (s_fetch_buf)  { free(s_fetch_buf);  s_fetch_buf = NULL; }
        if (s_entry_buf)  { free(s_entry_buf);  s_entry_buf = NULL; }
        if (s_entry_wbuf) { free(s_entry_wbuf); s_entry_wbuf = NULL; }
        free(s_idx_blob); s_idx_blob = NULL;
        platform_file_close();
        return -1;
    }

    // 查询界面字体统一 GFX_FONT_ALPHA_12（详情与退出时恢复；参照 Animac 控制台先例）
    s_prev_ui_font = global_state->ui_font;
    global_state->ui_font = GFX_FONT_ALPHA_12;

    // 固定显示软键盘（无呼出/收起逻辑；词典状态不加入手势门控列表）
    ui_softkbd_show();
    ui_ime_hint_mask_set_enabled(0);

    // 查询现场初始化
    s_prefix[0] = L'\0';
    s_prefix_len = 0;
    s_detail_match_pos = 0;

    // 菜单初始化（条目指针表借用本模块存储），并为查询行让位
    Widget_Menu_State *ms = global_state->w_menu_main;
    ms->title = L"电子词典";
    ms->items = s_match_items;
    ms->item_num = 1;
    ui_widget_menu_init(key_event, global_state, ms);
    // 布局修正（不修改通用组件）：本界面菜单上方是查询前缀行、下方紧贴软键盘（无页脚），
    // 收回 ui_widget_menu_init 预留的页脚高度，使菜单恰好撑满查询行与软键盘之间的区域
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    ms->y += line_height;
    ms->height = (global_state->gfx->height - ui_softkbd_height()) - ms->y;

    ui_dict_rebuild_matches(key_event, global_state);
    ui_dict_query_redraw(key_event, global_state);
    return 0;
}

void ui_dict_exit(Key_Event *key_event, Global_State *global_state) {
    global_state->ui_font = s_prev_ui_font; // 恢复全局字体
    ui_softkbd_hide();
    ui_ime_hint_mask_set_enabled(1);
    platform_file_close();
    if (s_idx_blob)   { free(s_idx_blob);   s_idx_blob = NULL; }
    if (s_match_wbuf) { free(s_match_wbuf); s_match_wbuf = NULL; }
    if (s_detail_buf) { free(s_detail_buf); s_detail_buf = NULL; }
    if (s_fetch_buf)  { free(s_fetch_buf);  s_fetch_buf = NULL; }
    if (s_entry_buf)  { free(s_entry_buf);  s_entry_buf = NULL; }
    if (s_entry_wbuf) { free(s_entry_wbuf); s_entry_wbuf = NULL; }
    s_items = NULL; s_pool = NULL; s_item_num = 0;
}

// ===============================================================================
// 详情
// ===============================================================================

// 取匹配表第 pos 个词条并拼装详情内容（词条 #66ccff、音标与释义原样），刷新详情界面
static void ui_dict_detail_open(Key_Event *key_event, Global_State *gs) {
    // 取词输出缓冲（PSRAM，ui_dict_enter 已分配；模块单线程运行，无重入）
    char *word = s_entry_buf;
    char *phon = s_entry_buf + UI_DICT_WORD_CHAR_MAX;
    char *def  = s_entry_buf + UI_DICT_WORD_CHAR_MAX + 256;
    word[0] = phon[0] = def[0] = '\0';
    // 注意容量必须显式给出：word/phon/def 是指针，sizeof 得到的是指针大小
    dict_fetch_entry((uint32_t)s_match_index[s_detail_match_pos],
        word, UI_DICT_WORD_CHAR_MAX, phon, 256, def, UI_DICT_ENTRY_READ_MAX);

    // 颜色复位标签随全局色彩风格（参照 STATE_README 先例）
    const wchar_t *reset = (gs->ui_color_style == UI_COLOR_LIGHT) ? L"[#000000]" : L"[#ffffff]";
    wchar_t *w_word = s_entry_wbuf;
    wchar_t *w_phon = s_entry_wbuf + UI_DICT_WORD_WCHAR_MAX;
    wchar_t *w_def  = s_entry_wbuf + UI_DICT_WORD_WCHAR_MAX + 256;
    _mbstowcs(w_word, word, UI_DICT_WORD_WCHAR_MAX - 1);
    _mbstowcs(w_phon, phon, 255);
    _mbstowcs(w_def, def, UI_DICT_ENTRY_READ_MAX - 1);

    s_detail_buf[0] = L'\0';
    wcscat(s_detail_buf, L"[#66ccff]");
    wcscat(s_detail_buf, w_word);
    wcscat(s_detail_buf, reset);
    wcscat(s_detail_buf, L"\n");
    wcscat(s_detail_buf, w_phon);
    wcscat(s_detail_buf, L"\n\n");
    wcscat(s_detail_buf, w_def);

    ui_draw_header(key_event, gs, L"电子词典", 1);
    ui_draw_footer_softkeys(key_event, gs, L"上一个", L"", L"下一个", L"返回");
    ui_widget_textarea_set(key_event, gs, gs->w_textarea_main, s_detail_buf, 0, 1);
    ui_widget_textarea_draw(key_event, gs, gs->w_textarea_main);
}

// 从详情返回查询界面（查询现场保留：前缀/候选表/菜单选中项均在模块状态中）
static int32_t ui_dict_detail_back(Key_Event *key_event, Global_State *gs, int32_t query_state) {
    gs->ui_font = GFX_FONT_ALPHA_12;
    ui_softkbd_show();
    ui_dict_query_redraw(key_event, gs);
    return query_state;
}

int32_t ui_dict_detail_event(Key_Event *key_event, Global_State *gs,
    int32_t query_state, int32_t detail_state) {
    // 长+短按*键：候选表内上一个词条（端点钳制）
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
        if (s_detail_match_pos > 0) {
            s_detail_match_pos--;
            ui_dict_detail_open(key_event, gs);
        }
    }
    // 长+短按#键：候选表内下一个词条（端点钳制）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
        if (s_detail_match_pos < s_match_num - 1) {
            s_detail_match_pos++;
            ui_dict_detail_open(key_event, gs);
        }
    }
    // 长+短按4键：释义向上滚一行（卷到顶则回到底部；与通用文本框卷行逻辑一致）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_4) {
        Widget_Textarea_State *ts = gs->w_textarea_main;
        if (ts->current_line <= 0) ts->current_line = ts->line_num - ts->view_lines;
        else                       ts->current_line--;
        ts->is_modified = 0;
        ui_widget_textarea_draw(key_event, gs, ts);
        ts->is_modified = 1;
    }
    // 长+短按6键：释义向下滚一行（卷到底则回到顶部）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_6) {
        Widget_Textarea_State *ts = gs->w_textarea_main;
        if (ts->current_line >= (ts->line_num - ts->view_lines)) ts->current_line = 0;
        else                                                     ts->current_line++;
        ts->is_modified = 0;
        ui_widget_textarea_draw(key_event, gs, ts);
        ts->is_modified = 1;
    }
    // 短按A/D键：返回查询界面
    else if (key_event->key_edge == -1 &&
        (key_event->key_code == NANO_KEY_esc || key_event->key_code == NANO_KEY_enter)) {
        return ui_dict_detail_back(key_event, gs, query_state);
    }
    return detail_state;
}

// ===============================================================================
// 查询
// ===============================================================================

// 菜单动作回调：实际不会触发（D/ENT 在 ui_dict_query_event 中先于菜单处理器被拦截），
// 仅为满足 ui_widget_menu_event_handler 的回调签名
static int32_t ui_dict_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    return gs->STATE;
}

int32_t ui_dict_query_event(Key_Event *key_event, Global_State *global_state,
    int32_t main_menu_state, int32_t query_state, int32_t detail_state) {
    // 软键盘自身状态变化（粘滞修饰键、按下高亮）时，补画键盘并刷新
    if (ui_softkbd_is_visible() && ui_softkbd_take_dirty()) {
        ui_softkbd_draw(global_state->gfx, (uint8_t)global_state->is_ctrl_enabled);
        gfx_refresh(global_state->gfx);
    }

    if (key_event->key_edge != -1 && key_event->key_edge != -2) {
        return query_state;
    }

    // 可打印字符（软键盘直键，含 SFT/SYM 层；不限制内容，检索不出即零匹配）
    if (key_event->key_code >= NANO_KEY_space && key_event->key_code <= NANO_KEY_tilde) {
        if (s_prefix_len < UI_DICT_WORD_WCHAR_MAX - 2) {
            s_prefix[s_prefix_len++] = (wchar_t)key_event->key_code;
            s_prefix[s_prefix_len] = L'\0';
            ui_dict_rebuild_matches(key_event, global_state);
            ui_dict_query_redraw(key_event, global_state);
        }
        return query_state;
    }

    // 退格（软键盘 BS 键）
    if (key_event->key_code == NANO_KEY_backspace) {
        if (s_prefix_len > 0) {
            s_prefix[--s_prefix_len] = L'\0';
            ui_dict_rebuild_matches(key_event, global_state);
            ui_dict_query_redraw(key_event, global_state);
        }
        return query_state;
    }

    // A键（软键盘或宫格 Esc）：前缀非空删末字符，空则退出词典
    if (key_event->key_code == NANO_KEY_esc) {
        if (s_prefix_len > 0) {
            s_prefix[--s_prefix_len] = L'\0';
            ui_dict_rebuild_matches(key_event, global_state);
            ui_dict_query_redraw(key_event, global_state);
            return query_state;
        }
        ui_dict_exit(key_event, global_state);
        return main_menu_state;
    }

    // 方向键（软键盘 ←/→/↑/↓）：候选菜单导航（有真实匹配时；菜单控件内 ↑同←、↓同→）
    if ((key_event->key_code == NANO_KEY_left || key_event->key_code == NANO_KEY_right ||
         key_event->key_code == NANO_KEY_up   || key_event->key_code == NANO_KEY_down) && s_match_num > 0) {
        return ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main,
            ui_dict_menu_item_action, main_menu_state, query_state);
    }

    // D/ENT：选中词条进入详情（仅真实匹配可选中）
    if (key_event->key_code == NANO_KEY_enter && s_match_num > 0) {
        s_detail_match_pos = global_state->w_menu_main->current_item_index;
        if (s_detail_match_pos >= s_match_num) s_detail_match_pos = s_match_num - 1;
        global_state->ui_font = GFX_FONT_ALPHA_16; // 详情界面字体
        ui_softkbd_hide();
        ui_dict_detail_open(key_event, global_state);
        return detail_state;
    }

    return query_state;
}
