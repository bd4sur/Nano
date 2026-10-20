#include <wchar.h>
#include <wctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "am_lexer.h"
#include "am_highlight.h"


/* ===== ANSI转义序列（Linux/macOS/现代终端） ===== */
static const char* ANSI_COLORS[COLOR_COUNT] = {
    [COLOR_DEFAULT]    = "\033[0m",
    [COLOR_KEYWORD]    = "\033[0m\033[1;35m",
    [COLOR_BOOLEAN]    = "\033[0m\033[1;36m",
    [COLOR_SPECIAL]    = "\033[0m\033[34m",
    [COLOR_NUMBER]     = "\033[0m\033[32m",
    [COLOR_STRING]     = "\033[0m\033[1;33m",
    [COLOR_SYMBOL]     = "\033[0m\033[1;31m",
    [COLOR_QUOTE]      = "\033[0m\033[1;31m",
    [COLOR_BRACKET]    = "\033[0m\033[37m",
    [COLOR_IDENTIFIER] = "\033[0m\033[1;37m",
    [COLOR_COMMENT]    = "\033[0m\033[1;32m",
    [COLOR_ERROR]      = "\033[0m\033[38m",
};


/* ===== 平台无关的颜色控制 ===== */
static void set_color(ColorScheme scheme) {
    // 检查终端是否支持颜色（可选优化）
    if(isatty(fileno(stdout))) {
        fputs(ANSI_COLORS[scheme], stdout);
    }
}

static void reset_color(void) {
    set_color(COLOR_DEFAULT);
}

/* ===== 辅助：判断字符是否属于注释（; 到行尾）===== */
// 由于Lexer会跳过注释不生成Token，这里需要手动检测注释区域用于高亮
static inline int in_comment(wchar_t *code, int32_t pos) {
    if(code[pos] == L';' && (pos == 0 || code[pos-1] != L'\\')) {
        return 1;
    }
    // 向前查找：当前位置是否在 ; 之后、换行之前
    int32_t i = pos - 1;
    while(i >= 0 && code[i] != L'\n' && code[i] != L'\r') {
        if(code[i] == L';' && (i == 0 || code[i-1] != L'\\')) {
            return 1;
        }
        i--;
    }
    return 0;
}

/* ===== Token类型 → 颜色映射 ===== */
static ColorScheme token_type_to_color(int32_t type) {
    switch(type) {
        case 3:  return COLOR_KEYWORD;    // AM_TOKEN_TYPE_KEYWORD
        case 4:  return COLOR_BOOLEAN;    // AM_TOKEN_TYPE_BOOLEAN
        case 5:  return COLOR_SPECIAL;    // AM_TOKEN_TYPE_UNDEFINED
        case 6:  return COLOR_SPECIAL;    // AM_TOKEN_TYPE_NULL
        case 7:  return COLOR_NUMBER;     // AM_TOKEN_TYPE_NUMBER
        case 8:  return COLOR_SYMBOL;     // AM_TOKEN_TYPE_SYMBOL
        case 9:  return COLOR_IDENTIFIER; // AM_TOKEN_TYPE_IDENTIFIER
        case 10: return COLOR_STRING;     // AM_TOKEN_TYPE_STRING
        case 11: return COLOR_QUOTE;      // AM_TOKEN_TYPE_QUOTE
        case 12: return COLOR_QUOTE;      // AM_TOKEN_TYPE_QUOTE
        case 13: return COLOR_QUOTE;      // AM_TOKEN_TYPE_QUOTE
        case 14: return COLOR_QUOTE;      // AM_TOKEN_TYPE_UNQUOTE_SPLICING
        case 1:  // fallthrough
        case 2:  return COLOR_BRACKET;    // AM_TOKEN_TYPE_LB/RB
        case 99: return COLOR_ERROR;      // AM_TOKEN_TYPE_UNEXPECTED
        default: return COLOR_DEFAULT;
    }
}

/* ===== 主函数：语法高亮输出 ===== */
int32_t am_highlight_code(wchar_t *code, am_token_t *tokens, int32_t count,
                          int32_t start_line, int32_t max_lines) {
    if(!code) return 0;

    int32_t pos = 0;              // 当前字符位置
    int32_t tok_idx = 0;          // 当前token索引
    int32_t line = 1, col = 0;    // 当前行列
    int32_t output_lines = 0;     // 已输出行数
    ColorScheme current_color = COLOR_DEFAULT;

    // 提前退出检查
    if(start_line > 1) {
        // 快速跳过前面的行
        while(code[pos] && line < start_line) {
            if(code[pos] == L'\n') { line++; col = 0; }
            else if(code[pos] == L'\r') {
                if(code[pos+1] != L'\n') { line++; col = 0; }
            }
            else { col++; }
            pos++;
        }
        if(!code[pos]) return 0;  // 已到达文件末尾
    }

    reset_color();  // 确保起始为默认颜色

    while(code[pos]) {
        while(tok_idx < count && pos >= (int32_t)(tokens[tok_idx].index + tokens[tok_idx].length)) {
            tok_idx++;
        }

        // 检查是否需要切换颜色：当前位置是否是某个真实token的起始
        ColorScheme target_color = COLOR_DEFAULT;

        // 优先检查注释（注释未被lexer输出为token，需特殊处理）
        if(in_comment(code, pos)) {
            target_color = COLOR_COMMENT;
        }
        // 检查是否匹配当前token的起始位置
        else if(tok_idx < count) {
            int32_t start = tokens[tok_idx].index;
            int32_t end = start + tokens[tok_idx].length;
            if(pos >= start && pos < end) {
                target_color = token_type_to_color(tokens[tok_idx].type);
            }
        }

        // 颜色变化时更新终端状态
        if(target_color != current_color) {
            set_color(target_color);
            current_color = target_color;
        }

        // 输出当前字符
        wchar_t c = code[pos];
        if(c == L'\n' || c == L'\r') {
            // 处理换行
            printf("%lc", c);
            if(c == L'\r' && code[pos+1] == L'\n') {
                printf("%lc", code[++pos]);
            }
            line++;
            col = 0;
            output_lines++;

            // 检查是否达到输出行数限制
            if(max_lines > 0 && output_lines >= max_lines) {
                reset_color();
                return output_lines;
            }

            // 换行后重置颜色（避免颜色污染下一行）
            reset_color();
            current_color = COLOR_DEFAULT;
        }
        else {
            printf("%lc", c);
            col++;
        }

        pos++;
    }

    // 文件末尾：确保颜色重置
    reset_color();
    printf("\n");

    return output_lines + (line > start_line ? 1 : 0);
}

/* ===== 便捷封装：默认参数版本 ===== */
void am_print_highlighted(wchar_t *code, am_token_t *tokens, int32_t count) {
    am_highlight_code(code, tokens, count, 1, 0);
}
