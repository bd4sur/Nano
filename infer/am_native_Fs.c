////////////////////////////////////////////////////////////////////////////
//  Native Library : Fs —— 设计说明见 am_native_Fs.h
//
//  分层：本文件只做「Scheme 值 ↔ UTF-8 字节」与「资源根路径拼接」，真正的文件
//  操作全部交给 hal_fs（platform_read/write/is_directory/mkdir/file_* / list_files），
//  因此三个平台（桌面 / WASM 的 MEMFS / ESP32 的 SD）行为一致，无需平台分支。
//
//  文本编码：Scheme 字符串是 Unicode 码点序列（am_wstring_t），落盘一律 UTF-8；
//  读写转换用项目自带的 _wcstombs/_mbstowcs（utils.c，locale 无关、对坏字节宽容）。
////////////////////////////////////////////////////////////////////////////

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "am_native_Fs.h"
#include "hal_fs.h"
#include "hal_ram.h"
#include "platform.h"
#include "utils.h"

#define FS_PATH_MAX (256)

// 写入成功后的可选落盘钩子（弱符号：未提供实现的平台为空操作）。
// WASM 目标由 hal_fs_persist_wasm.c 实现，把文件同步进浏览器 IndexedDB。
#if defined(__GNUC__)
extern void am_native_fs_on_write(const char *path, const uint8_t *data, size_t size) __attribute__((weak));
#endif

// ===============================================================================
// 内部辅助
// ===============================================================================

// 从操作数栈弹出一个字符串对象（模式与 vendor 的 am_native_String.c 一致）。
static bool fs_pop_wstring(am_process_t *proc, am_wstring_t **out) {
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

// 把 Scheme 字符串按 UTF-8 编码写入 out（容量 cap，含结尾 NUL）。
// 返回写入的字节数；空间不足或参数非法返回 (size_t)-1。
static size_t fs_wstring_to_utf8(am_wstring_t *ws, char *out, size_t cap) {
    if (!ws || !out || cap < 2) return (size_t)-1;
    if (ws->length + 1 > cap) return (size_t)-1; // UTF-8 最短也有 1 字节/码点

    size_t n = 0;
    for (size_t i = 0; i < ws->length; i++) {
        uint32_t cp = (uint32_t)am_value_to_wchar(ws->content[i]);
        size_t need = (cp < 0x80) ? 1 : (cp < 0x800) ? 2 : (cp < 0x10000) ? 3 : 4;
        if (n + need + 1 > cap) return (size_t)-1;
        if (need == 1) {
            out[n++] = (char)cp;
        }
        else if (need == 2) {
            out[n++] = (char)(0xC0 | (cp >> 6));
            out[n++] = (char)(0x80 | (cp & 0x3F));
        }
        else if (need == 3) {
            out[n++] = (char)(0xE0 | (cp >> 12));
            out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[n++] = (char)(0x80 | (cp & 0x3F));
        }
        else {
            out[n++] = (char)(0xF0 | (cp >> 18));
            out[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
            out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[n++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    out[n] = '\0';
    return n;
}

// 把相对路径解析为 HAL 路径（见头文件的路径规则）。
// 返回 true 表示解析成功；路径非法（空、超长、含 ".."）返回 false。
static bool fs_resolve(am_wstring_t *ws, char *out) {
    char raw[FS_PATH_MAX];

    if (fs_wstring_to_utf8(ws, raw, sizeof raw) == (size_t)-1) return false;
    if (raw[0] == '\0') return false;
    if (strstr(raw, "..") != NULL) return false; // 不允许穿越出资源根

    if (raw[0] == '/') {
        memcpy(out, raw, strlen(raw) + 1);
        return true;
    }
    int n = snprintf(out, FS_PATH_MAX, "%s/%s", PLATFORM_ROOT_DIR, raw);
    return n > 0 && n < FS_PATH_MAX;
}

// 逐级创建父目录（例如 "data/a/b.txt" 会创建 <root>/data 与 <root>/data/a）。
// 已存在的目录由 platform_mkdir 视为成功，故可无脑逐级调用。
static void fs_make_parents(const char *path) {
    char buf[FS_PATH_MAX];
    size_t len = strlen(path);

    if (len + 1 > sizeof buf) return;
    memcpy(buf, path, len + 1);

    for (size_t i = 1; i < len; i++) {
        if (buf[i] == '/') {
            buf[i] = '\0';
            platform_mkdir(buf);
            buf[i] = '/';
        }
    }
}

// 判断普通文件是否存在（借 hal_fs 的全局单句柄短暂探测，见头文件说明）。
static int32_t fs_file_exists(const char *path) {
    if (platform_file_open(path) != 0) return 0;
    platform_file_close();
    return 1;
}

// 整文件读为裸字节（platform_malloc 分配，调用方 free）。
// 返回缓冲区指针；文件不存在/读失败返回 NULL（*out_size 置 0）。
// 空文件返回非 NULL 的最小缓冲（size=0），以便与"读失败"区分。
static uint8_t *fs_read_raw(const char *path, size_t *out_size) {
    uint32_t size;
    uint8_t *buf;

    *out_size = 0;
    if (platform_file_open(path) != 0) return NULL;

    size = platform_file_size();
    if (size > AM_NATIVE_FS_READ_MAX) {
        platform_file_close();
        return NULL;
    }

    buf = (uint8_t *)platform_malloc(size > 0 ? size : 1);
    if (!buf) {
        platform_file_close();
        return NULL;
    }

    size_t got = 0;
    while (got < size) {
        int32_t r = platform_file_read(buf + got, size - got);
        if (r <= 0) break; // 读失败或提前 EOF：按实际读到的字节数处理
        got += (size_t)r;
    }
    platform_file_close();

    *out_size = got;
    return buf;
}

// 写入裸字节（含"自动建父目录"与落盘钩子）。成功 true。
static bool fs_write_raw(const char *path, const uint8_t *data, size_t size) {
    fs_make_parents(path);
    if (platform_write_buffer_to_file(path, data, size) != 0) return false;
#if defined(__GNUC__)
    if (am_native_fs_on_write) am_native_fs_on_write(path, data, size);
#endif
    return true;
}

// ---------------------------------------------------------------------------
// 值型收尾（契约 N1 + N6）：压 1 个值 + step 1 次。全部过程共用单一出口。
// ---------------------------------------------------------------------------
static int32_t fs_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t fs_push_bool(am_process_t *proc, bool b) {
    return fs_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

static int32_t fs_push_null(am_process_t *proc) {
    return fs_push(proc, AM_VALUE_NULL);
}

// 由 UTF-8 缓冲区构造 Scheme 字符串并压栈（失败压 #null）
static int32_t fs_push_string(am_process_t *proc, const char *utf8, size_t len) {
    // 最坏情况：每个字节一个码点
    wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (len + 1) * sizeof(wchar_t));
    if (!wbuf) return fs_push_null(proc);

    uint32_t wlen = _mbstowcs(wbuf, utf8, (uint32_t)len);
    am_handle_t hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
    am_free(proc->heap_alloc, wbuf);
    if (hd == AM_HANDLE_NULL) return fs_push_null(proc);

    return fs_push(proc, am_make_value_of_handle(hd));
}

// ===============================================================================
// Native 函数实现
// ===============================================================================

int32_t am_native_Fs_exists(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    bool ok = false;
    (void)rt;

    if (!fs_pop_wstring(proc, &ws)) return -1;
    if (fs_resolve(ws, path)) {
        ok = (platform_is_directory(path) != 0) || (fs_file_exists(path) != 0);
    }
    return fs_push_bool(proc, ok);
}

int32_t am_native_Fs_is_dir(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    bool ok = false;
    (void)rt;

    if (!fs_pop_wstring(proc, &ws)) return -1;
    if (fs_resolve(ws, path)) ok = (platform_is_directory(path) != 0);
    return fs_push_bool(proc, ok);
}

int32_t am_native_Fs_size(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    (void)rt;

    if (!fs_pop_wstring(proc, &ws)) return -1;
    if (!fs_resolve(ws, path)) return fs_push_null(proc);

    if (platform_file_open(path) != 0) return fs_push_null(proc);
    uint32_t size = platform_file_size();
    platform_file_close();
    return fs_push(proc, am_make_value_of_int((am_int_t)size));
}

int32_t am_native_Fs_list(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    char **names = NULL;
    int32_t n;
    (void)rt;

    if (!fs_pop_wstring(proc, &ws)) return -1;
    if (!fs_resolve(ws, path)) return fs_push_null(proc);

    n = list_files(path, NULL);
    if (n < 0) return fs_push_null(proc);

    if (n > 0) {
        names = (char **)platform_calloc((size_t)n, sizeof(char *));
        if (!names) return fs_push_null(proc);
        n = list_files(path, names); // 重新枚举并填充（hal_fs 契约：两次调用之间目录应稳定）
        if (n < 0) {
            free(names);
            return fs_push_null(proc);
        }
    }

    am_list_t *lst = am_list_create(proc->heap_alloc, n > 0 ? (size_t)n : 1,
                                    AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!lst) {
        if (names) free(names);
        return fs_push_null(proc);
    }

    for (int32_t i = 0; i < n; i++) {
        size_t blen = strlen(names[i]);
        wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (blen + 1) * sizeof(wchar_t));
        am_handle_t hd = AM_HANDLE_NULL;
        if (wbuf) {
            uint32_t wlen = _mbstowcs(wbuf, names[i], (uint32_t)blen);
            hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
            am_free(proc->heap_alloc, wbuf);
        }
        if (hd == AM_HANDLE_NULL) {
            for (int32_t j = 0; j < n; j++) free(names[j]);
            free(names);
            am_list_destroy(proc->heap_alloc, lst);
            return fs_push_null(proc);
        }
        am_list_t *grown = am_list_push(proc->heap_alloc, lst, am_make_value_of_handle(hd));
        if (!grown) {
            for (int32_t j = 0; j < n; j++) free(names[j]);
            free(names);
            am_list_destroy(proc->heap_alloc, lst);
            return fs_push_null(proc);
        }
        lst = grown;
    }

    if (names) {
        for (int32_t i = 0; i < n; i++) free(names[i]);
        free(names);
    }

    am_handle_t lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, lst);
        return fs_push_null(proc);
    }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return fs_push_null(proc);
    }
    return fs_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Fs_mkdir(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    bool ok = false;
    (void)rt;

    if (!fs_pop_wstring(proc, &ws)) return -1;
    if (fs_resolve(ws, path)) {
        fs_make_parents(path);
        ok = (platform_mkdir(path) == 0);
    }
    return fs_push_bool(proc, ok);
}

int32_t am_native_Fs_read(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    uint8_t *buf;
    size_t size = 0;
    (void)rt;

    if (!fs_pop_wstring(proc, &ws)) return -1;
    if (!fs_resolve(ws, path)) return fs_push_null(proc);

    buf = fs_read_raw(path, &size);
    if (!buf) return fs_push_null(proc);

    int32_t ret = fs_push_string(proc, (const char *)buf, size);
    free(buf);
    return ret;
}

int32_t am_native_Fs_write(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws_text = NULL;
    am_wstring_t *ws_path = NULL;
    char path[FS_PATH_MAX];
    char *utf8 = NULL;
    bool ok = false;
    (void)rt;

    // 参数从右往左弹：先 text，再 path
    if (!fs_pop_wstring(proc, &ws_text)) return -1;
    if (!fs_pop_wstring(proc, &ws_path)) return -1;

    size_t cap = ws_text->length * 4 + 1;
    utf8 = (char *)platform_malloc(cap);
    if (utf8) {
        size_t len = fs_wstring_to_utf8(ws_text, utf8, cap);
        if (len != (size_t)-1 && fs_resolve(ws_path, path)) {
            ok = fs_write_raw(path, (const uint8_t *)utf8, len);
        }
        free(utf8);
    }
    return fs_push_bool(proc, ok);
}

int32_t am_native_Fs_append(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws_text = NULL;
    am_wstring_t *ws_path = NULL;
    char path[FS_PATH_MAX];
    char *utf8 = NULL;
    bool ok = false;
    (void)rt;

    if (!fs_pop_wstring(proc, &ws_text)) return -1;
    if (!fs_pop_wstring(proc, &ws_path)) return -1;

    size_t cap = ws_text->length * 4 + 1;
    utf8 = (char *)platform_malloc(cap);
    if (utf8) {
        size_t len = fs_wstring_to_utf8(ws_text, utf8, cap);
        if (len != (size_t)-1 && fs_resolve(ws_path, path)) {
            size_t old_size = 0;
            uint8_t *old = fs_read_raw(path, &old_size); // 不存在/读失败 → NULL/0，等价于空文件

            if (old_size + len <= AM_NATIVE_FS_READ_MAX) {
                size_t total = old_size + len;
                uint8_t *merged = (uint8_t *)platform_malloc(total > 0 ? total : 1);
                if (merged) {
                    if (old_size > 0 && old) memcpy(merged, old, old_size);
                    if (len > 0) memcpy(merged + old_size, utf8, len);
                    ok = fs_write_raw(path, merged, total);
                    free(merged);
                }
            }
            if (old) free(old);
        }
        free(utf8);
    }
    return fs_push_bool(proc, ok);
}

// ===============================================================================
// 库注册表
// ===============================================================================

static const am_native_func_entry_t am_native_Fs_funcs[] = {
    { L"exists", am_native_Fs_exists },
    { L"is_dir", am_native_Fs_is_dir },
    { L"size",   am_native_Fs_size },
    { L"list",   am_native_Fs_list },
    { L"mkdir",  am_native_Fs_mkdir },
    { L"read",   am_native_Fs_read },
    { L"write",  am_native_Fs_write },
    { L"append", am_native_Fs_append },
};

const am_native_lib_entry_t am_native_Fs_lib = {
    L"Fs",
    am_native_Fs_funcs,
    sizeof(am_native_Fs_funcs) / sizeof(am_native_Fs_funcs[0])
};