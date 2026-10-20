#ifndef __AM_HIGHLIGHT_H__
#define __AM_HIGHLIGHT_H__

#include <stdint.h>
#include <wchar.h>

#include "am_lexer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===== 颜色主题配置（可自定义）===== */
typedef enum {
    COLOR_DEFAULT = 0,
    COLOR_KEYWORD,      // define, if, lambda 等
    COLOR_BOOLEAN,      // #t, #f
    COLOR_SPECIAL,      // #null, #undefined
    COLOR_NUMBER,       // 数字字面值
    COLOR_STRING,       // "hello"
    COLOR_SYMBOL,       // 'symbol, `, , 等
    COLOR_QUOTE,        // 'symbol, `, , 等
    COLOR_BRACKET,      // ( ) [ ] { }
    COLOR_IDENTIFIER,   // 普通变量/标识符
    COLOR_COMMENT,      // ; 注释
    COLOR_ERROR,        // 意外token
    COLOR_COUNT
} ColorScheme;


/* ===== 主函数：语法高亮输出 ===== */
/**
 * 基于Lexer输出的tokens，对原始code进行终端语法高亮
 * @param code   原始源代码（宽字符串）
 * @param tokens Lexer生成的token数组（按源码顺序排列）
 * @param count  token数量
 * @param start_line 起始行号（用于分页输出，默认1）
 * @param max_lines 最大输出行数（0表示无限制）
 * @return 实际输出的行数
 */
int32_t am_highlight_code(wchar_t *code, am_token_t *tokens, int32_t count,
                          int32_t start_line, int32_t max_lines);

/* ===== 便捷封装：默认参数版本 ===== */
void am_print_highlighted(wchar_t *code, am_token_t *tokens, int32_t count);

#ifdef __cplusplus
}
#endif

#endif
