#ifndef __AM_LEXER_H__
#define __AM_LEXER_H__

#include <stdint.h>
#include <wchar.h>
#include <wctype.h>
#include <string.h>

#include "am_object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Token类型定义 ===== */
#define AM_TOKEN_TYPE_DELIMITER  (0)   // 分隔符: 空白字符
#define AM_TOKEN_TYPE_LB         (1)   // 左括号: ( [ {
#define AM_TOKEN_TYPE_RB         (2)   // 右括号: ) ] }
#define AM_TOKEN_TYPE_KEYWORD    (3)   // 关键字: define if cond while lambda begin 等
#define AM_TOKEN_TYPE_BOOLEAN    (4)   // 字面值：#t #f
#define AM_TOKEN_TYPE_UNDEFINED  (5)   // 字面值：#undefined
#define AM_TOKEN_TYPE_NULL       (6)   // 字面值：#null
#define AM_TOKEN_TYPE_NUMBER     (7)   // 字面值：数字，如 -3.14 +12.3 2e-5 等
#define AM_TOKEN_TYPE_SYMBOL     (8)   // 字面值：符号，即单撇号开头的符号，如 'symbol 等
#define AM_TOKEN_TYPE_IDENTIFIER (9)   // 标识符（变量、运算符等）
#define AM_TOKEN_TYPE_STRING     (10)  // 字符串: "hello"
#define AM_TOKEN_TYPE_QUOTE      (11)  // 出现在括号前面的单引号'
#define AM_TOKEN_TYPE_QUASIQUOTE (12)  // 反引号`
#define AM_TOKEN_TYPE_UNQUOTE    (13)  // 逗号,
#define AM_TOKEN_TYPE_UNQUOTE_SPLICING (14)  // 逗号-at ,@
#define AM_TOKEN_TYPE_UNEXPECTED (99)  // 意料之外的token

typedef struct am_token_t {
    am_object_t base;

    size_t  index;   // token首字符在code中的偏移
    size_t  length;  // token长度(字符数)
    int32_t type;    // token类型
    int32_t line;    // 行号(从1开始)
    int32_t column;  // 列号(从0开始)
    size_t  id;      // 如果是 AM_TOKEN_TYPE_SYMBOL 或 AM_TOKEN_TYPE_IDENTIFIER，记录其在编译时分配到的 am_symbol_t 或 am_varid_t
} am_token_t;

// 关键字
#define AM_KEYWORDS_NUM (29)
extern const wchar_t* AM_KEYWORDS[];


/* ===== 通用辅助函数 ===== */

// 定界符判断
static inline int is_delimiter(wchar_t c) {
    return c == L'(' || c == L')' || c == L'[' || c == L']' ||
           c == L'{' || c == L'}' || c == L'"' || c == L'`' || c == L',';
}

static inline int is_whitespace(wchar_t c) {
    return c == L' ' || c == L'\t' || c == L'\n' || c == L'\r';
}

static inline int is_escaped(wchar_t *code, int32_t pos) {
    if(pos <= 0) return 0;
    int32_t count = 0, i = pos - 1;
    while(i >= 0 && code[i--] == L'\\') count++;
    return (count & 1);
}

// 增强：数字字符判断（支持科学计数法）
static inline int is_num_char(wchar_t c) {
    return iswdigit(c) || c == L'.' || c == L'e' || c == L'E' ||
           c == L'+' || c == L'-';
}

/* ===== 主Lexer函数 ===== */

// 对 code 进行词法分析，结果写入 tokens 数组。
// 返回 token 数量；出错返回 -1。
int32_t am_lexer(wchar_t *code, am_token_t *tokens);

// 安全获取 token 文本（处理虚拟 token）。
// 注意：返回指向静态缓冲区的指针，非线程安全。
const wchar_t* token_text(am_token_t *tok, wchar_t *code);

#ifdef __cplusplus
}
#endif

#endif
