// Luckfox / ILP32 String ABI 适配。
//
// vendor/Animac/am_native_String.c 保持硬件无关且逐字不改；本翻译单元仅在
// luckfox.mk 中替代该文件参与编译。这里仅适配其三个 swprintf 调用，
// 不改变其他平台或其他 %lu/%ld 调用的语义。

#include <stdint.h>

#include "nano_wfmt.h"

static int luckfox_string_swprintf(wchar_t *s, size_t n, const wchar_t *fmt, ...) {
    va_list ap;
    int result;

    va_start(ap, fmt);
    if (wcscmp(fmt, L"%lu") == 0) {
        // 原实参为 uint64_t：ILP32 上是 unsigned long long，不是 unsigned long。
        uint64_t value = va_arg(ap, uint64_t);
        result = nano_swprintf(s, n, L"%llu", (unsigned long long)value);
    }
    else if (wcscmp(fmt, L"%ld") == 0) {
        // 原实参为 int64_t：ILP32 上是 long long，不是 long。
        int64_t value = va_arg(ap, int64_t);
        result = nano_swprintf(s, n, L"%lld", (long long)value);
    }
    else {
        result = nano_vswprintf(s, n, fmt, ap);
    }
    va_end(ap);
    return result;
}

#undef swprintf
#define swprintf luckfox_string_swprintf
#include "vendor/Animac/am_native_String.c"
#undef swprintf
