#ifndef __AM_JS2SCM_H__
#define __AM_JS2SCM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <wchar.h>

/* 将 JavaScript 代码机械翻译成非标准 Scheme 子集。
 * 输入：宽字符 JS 源码。
 * 输出：宽字符 Scheme 源码；失败或在翻译过程中发生词法/语法错误时返回 NULL。
 * 返回的指针由调用者使用 free() 释放。 */
wchar_t *am_js_to_scheme(const wchar_t *js_source);

// JS 翻译器最近一次词法/语法错误消息（UTF-32），无错误时为空字符串；
// am_js_to_scheme 每次进入翻译时清空。供上层（REPL）在翻译失败（返回 NULL）时取用。
const wchar_t *am_js_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
