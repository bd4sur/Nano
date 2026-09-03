#ifndef __NANO_WFMT_H__
#define __NANO_WFMT_H__

// ===============================================================================
// uClibc 兼容层：可移植宽格式输出（仅 uClibc 构建启用，其余平台零影响）
// uClibc 的 swprintf/vswprintf 内部先把宽格式串经 wcsrtombs 转成多字节再解析；
// 在无 locale（__UCLIBC_HAS_LOCALE__ 未定义）或 C locale 下，格式串中的非 ASCII
// 字符（如中文）转换失败，输出退化为 "Invalid wide format string."。
// nano_swprintf/nano_vswprintf（实现于 utils.c）直接逐字符解析宽格式串，
// 非 ASCII 字面量原样透传，与 locale 无关；数值/浮点子格式委托窄 snprintf
// （ASCII，无 locale 依赖），格式化结果与各平台原生一致。
// 通过宏接管 swprintf/vswprintf 调用；glibc/newlib（ESP32、Linux PC、树莓派、
// STM32MP135 等）上不定义宏，代码路径完全不变。
// ===============================================================================

#include <stddef.h>
#include <stdarg.h>
#include <wchar.h>

#if defined(__UCLIBC__)
int nano_swprintf(wchar_t *s, size_t n, const wchar_t *fmt, ...);
int nano_vswprintf(wchar_t *s, size_t n, const wchar_t *fmt, va_list arg);
#define swprintf  nano_swprintf
#define vswprintf nano_vswprintf
#endif

#endif // __NANO_WFMT_H__
