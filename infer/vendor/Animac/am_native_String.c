#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "am_native_String.h"


// ===============================================================================
// 内部辅助函数
// ===============================================================================


// 从操作数栈中弹出一个数值，统一转换为 float。
// 成功返回 true，失败返回 false。
static bool native_pop_number(am_process_t *proc, am_float_t *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;

    if (am_value_is_float(v)) {
        *out = am_value_to_float(v);
        return true;
    }
    if (am_value_is_int(v)) {
        *out = (am_float_t)am_value_to_int(v);
        return true;
    }
    if (am_value_is_uint(v)) {
        *out = (am_float_t)am_value_to_uint(v);
        return true;
    }
    return false;
}


// 从操作数栈中弹出一个字符串对象。
// 成功返回 true，失败返回 false。
static bool native_pop_wstring(am_process_t *proc, am_wstring_t **out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return false;

    *out = (am_wstring_t *)obj;
    return true;
}


// 将 float 结果压回操作数栈；若结果为 NaN，则压入 null。
static int32_t native_push_float_or_null(am_process_t *proc, am_float_t result) {
    if (isnan(result)) {
        if (am_process_push_operand(proc, AM_VALUE_NULL) != 0) return -1;
    }
    else {
        if (am_process_push_operand(proc, am_make_value_of_float(result)) != 0) return -1;
    }
    am_process_step(proc);
    return 0;
}


// 从 wchar_t 缓冲区创建字符串对象并压回操作数栈。
// 成功返回 0，失败返回 -1。
static int32_t native_push_wstring_buf(am_process_t *proc, const wchar_t *buf, size_t len) {
    am_handle_t hd = am_process_make_wstring_handle(proc, buf, len);
    if (hd == AM_HANDLE_NULL) return -1;

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// 将 wstring 对象的内容复制到堆分配器新分配的 wchar_t 缓冲区中。
// 调用者负责使用 proc->heap_alloc 释放返回的缓冲区。
static wchar_t *wstring_content_to_buffer(am_process_t *proc, am_wstring_t *ws) {
    if (!ws || ws->length == 0) return NULL;

    wchar_t *buf = (wchar_t *)am_malloc(proc->heap_alloc, (ws->length + 1) * sizeof(wchar_t));
    if (!buf) return NULL;

    for (size_t i = 0; i < ws->length; i++) {
        buf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    }
    buf[ws->length] = L'\0';
    return buf;
}


// ===============================================================================
// Native 函数实现
// ===============================================================================

// (String.length str:String) : Number
int32_t am_native_String_length(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_wstring_t *ws;
    if (!native_pop_wstring(proc, &ws)) return -1;
    return native_push_float_or_null(proc, (am_float_t)ws->length);
}


// (String.atom_to_string x:Boolean|Number|Symbol) : String
int32_t am_native_String_atom_to_string(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return -1;

    wchar_t buf[128];

    if (am_value_is_boolean(v)) {
        const wchar_t *s = am_value_to_boolean(v) ? L"#t" : L"#f";
        return native_push_wstring_buf(proc, s, wcslen(s));
    }

    if (am_value_is_null(v)) {
        return native_push_wstring_buf(proc, L"#null", 5);
    }

    if (am_value_is_undefined(v)) {
        return native_push_wstring_buf(proc, L"#undefined", 10);
    }

    if (am_value_is_uint(v)) {
        am_uint_t n = am_runtime_number_to_uint(v);
        swprintf(buf, 128, L"%lu", (uint64_t)n);
        return native_push_wstring_buf(proc, buf, wcslen(buf));
    }

    if (am_value_is_int(v)) {
        am_int_t n = am_runtime_number_to_int(v);
        swprintf(buf, 128, L"%ld", (int64_t)n);
        return native_push_wstring_buf(proc, buf, wcslen(buf));
    }

    if (am_value_is_float(v)) {
        am_float_t f = am_runtime_number_to_float(v);
        swprintf(buf, 128, L"%g", (double)f);
        return native_push_wstring_buf(proc, buf, wcslen(buf));
    }

    if (am_value_is_symbol(v)) {
        am_symbol_t sym = am_value_to_symbol(v);
        wchar_t *s = am_vocab_get(proc->vm_alloc, proc->symbol_vocab, &sym);
        if (!s) return -1;
        return native_push_wstring_buf(proc, s, wcslen(s));
    }

    if (am_value_is_wchar(v)) {
        buf[0] = (wchar_t)am_value_to_wchar(v);
        buf[1] = L'\0';
        return native_push_wstring_buf(proc, buf, 1);
    }

    // 对已存在的字符串对象返回副本；其余类型返回空字符串。
    if (am_value_is_handle(v)) {
        am_handle_t hd = am_value_to_handle(v);
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (am_value_is_ptr(obj_val)) {
            am_object_t *obj = am_value_to_ptr(obj_val);
            if (obj->type == AM_OBJECT_TYPE_WSTRING) {
                am_wstring_t *ws = (am_wstring_t *)obj;
                wchar_t *content = wstring_content_to_buffer(proc, ws);
                if (ws->length > 0 && !content) return -1;
                int32_t ret = native_push_wstring_buf(proc, content, ws->length);
                if (content) am_free(proc->heap_alloc, content);
                return ret;
            }
        }
    }

    return native_push_wstring_buf(proc, L"", 0);
}


// (String.concat str1:String str2:String) : String
int32_t am_native_String_concat(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_wstring_t *ws2;
    am_wstring_t *ws1;
    if (!native_pop_wstring(proc, &ws2)) return -1;
    if (!native_pop_wstring(proc, &ws1)) return -1;

    size_t len = ws1->length + ws2->length;
    if (len == 0) return native_push_wstring_buf(proc, L"", 0);

    wchar_t *buf = (wchar_t *)am_malloc(proc->heap_alloc, (len + 1) * sizeof(wchar_t));
    if (!buf) return -1;

    for (size_t i = 0; i < ws1->length; i++) {
        buf[i] = (wchar_t)am_value_to_wchar(ws1->content[i]);
    }
    for (size_t i = 0; i < ws2->length; i++) {
        buf[ws1->length + i] = (wchar_t)am_value_to_wchar(ws2->content[i]);
    }
    buf[len] = L'\0';

    int32_t ret = native_push_wstring_buf(proc, buf, len);
    am_free(proc->heap_alloc, buf);
    return ret;
}


// (String.charCodeAt index:Number str:String) : Number
int32_t am_native_String_charCodeAt(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_wstring_t *ws;
    am_float_t idx_f;

    // 参数退栈顺序与参数列表顺序相反：先 str，后 index
    if (!native_pop_wstring(proc, &ws)) return -1;
    if (!native_pop_number(proc, &idx_f)) return -1;

    if (isnan(idx_f)) return native_push_float_or_null(proc, 0.0);

    am_int_t idx = (am_int_t)idx_f;
    if (idx < 0 || (size_t)idx >= ws->length) return native_push_float_or_null(proc, 0.0);

    return native_push_float_or_null(proc, (am_float_t)am_value_to_wchar(ws->content[idx]));
}


// (String.fromCharCode charcode:Number) : String
int32_t am_native_String_fromCharCode(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t code;
    if (!native_pop_number(proc, &code)) return -1;

    // NaN 输入返回空字符串（保持返回类型为 String）
    if (isnan(code)) return native_push_wstring_buf(proc, L"", 0);

    wchar_t buf[2] = { (wchar_t)(am_wchar_t)code, L'\0' };
    return native_push_wstring_buf(proc, buf, 1);
}


// (String.slice str:String start:Number end:Number) : String
int32_t am_native_String_slice(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_wstring_t *ws;
    am_float_t start_f, end_f;

    // 参数退栈顺序与参数列表顺序相反：先 end，再 start，最后 str
    if (!native_pop_number(proc, &end_f)) return -1;
    if (!native_pop_number(proc, &start_f)) return -1;
    if (!native_pop_wstring(proc, &ws)) return -1;

    if (isnan(start_f) || isnan(end_f)) return native_push_wstring_buf(proc, L"", 0);

    am_int_t len = (am_int_t)ws->length;
    am_int_t start = (am_int_t)start_f;
    am_int_t end = (am_int_t)end_f;

    // 负数索引从末尾开始计算
    if (start < 0) start = len + start;
    if (end < 0) end = len + end;

    // 限制在合法范围内
    if (start < 0) start = 0;
    if (end < 0) end = 0;
    if (start > len) start = len;
    if (end > len) end = len;
    if (start > end) start = end;

    size_t sublen = (size_t)(end - start);
    if (sublen == 0) return native_push_wstring_buf(proc, L"", 0);

    wchar_t *buf = (wchar_t *)am_malloc(proc->heap_alloc, (sublen + 1) * sizeof(wchar_t));
    if (!buf) return -1;

    for (size_t i = 0; i < sublen; i++) {
        buf[i] = (wchar_t)am_value_to_wchar(ws->content[start + i]);
    }
    buf[sublen] = L'\0';

    int32_t ret = native_push_wstring_buf(proc, buf, sublen);
    am_free(proc->heap_alloc, buf);
    return ret;
}


// (String.equals str1:String str2:String) : Boolean
int32_t am_native_String_equals(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_wstring_t *ws2;
    am_wstring_t *ws1;
    if (!native_pop_wstring(proc, &ws2)) return -1;
    if (!native_pop_wstring(proc, &ws1)) return -1;

    bool equal = (ws1->length == ws2->length);
    if (equal) {
        for (size_t i = 0; i < ws1->length; i++) {
            if (ws1->content[i] != ws2->content[i]) {
                equal = false;
                break;
            }
        }
    }

    if (am_process_push_operand(proc, equal ? AM_VALUE_TRUE : AM_VALUE_FALSE) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (String.charAt str:String index:Number) : String
int32_t am_native_String_charAt(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_wstring_t *ws;
    am_float_t idx_f;

    if (!native_pop_number(proc, &idx_f)) return -1;
    if (!native_pop_wstring(proc, &ws)) return -1;

    if (isnan(idx_f)) return native_push_wstring_buf(proc, L"", 0);

    am_int_t idx = (am_int_t)idx_f;
    if (idx < 0 || (size_t)idx >= ws->length) return native_push_wstring_buf(proc, L"", 0);

    wchar_t buf[2] = { (wchar_t)am_value_to_wchar(ws->content[idx]), L'\0' };
    return native_push_wstring_buf(proc, buf, 1);
}


// (String.parseNumber x:String) : Number|#undefined
int32_t am_native_String_parseNumber(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_wstring_t *ws;
    if (!native_pop_wstring(proc, &ws)) return -1;

    if (ws->length == 0) {
        if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    wchar_t *buf = wstring_content_to_buffer(proc, ws);
    if (!buf) return -1;

    wchar_t *endptr = NULL;
    double val = wcstod(buf, &endptr);
    am_free(proc->heap_alloc, buf);

    // 必须完整解析整个字符串
    if (endptr == buf || *endptr != L'\0') {
        if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    if (isnan(val)) {
        if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    if (am_process_push_operand(proc, am_make_value_of_float((am_float_t)val)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


static const am_native_func_entry_t am_native_String_funcs[] = {
    { L"length",         am_native_String_length },
    { L"atom_to_string", am_native_String_atom_to_string },
    { L"concat",         am_native_String_concat },
    { L"charCodeAt",     am_native_String_charCodeAt },
    { L"fromCharCode",   am_native_String_fromCharCode },
    { L"slice",          am_native_String_slice },
    { L"equals",         am_native_String_equals },
    { L"charAt",         am_native_String_charAt },
    { L"parseNumber",    am_native_String_parseNumber },
};

const am_native_lib_entry_t am_native_String_lib = {
    L"String",
    am_native_String_funcs,
    sizeof(am_native_String_funcs) / sizeof(am_native_String_funcs[0])
};
