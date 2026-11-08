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

// 写入成功后、删除成功后、改名前后的可选落盘钩子（弱符号：未提供实现的平台为空操作）。
// WASM 目标由 hal_fs_persist_wasm.c 实现，把文件同步进浏览器 IndexedDB。
#if defined(__GNUC__)
extern void am_native_fs_on_write(const char *path, const uint8_t *data, size_t size) __attribute__((weak));
extern void am_native_fs_on_remove(const char *path) __attribute__((weak));
#endif

// ===============================================================================
// 内部辅助
// ===============================================================================

// 从操作数栈弹出一个数值，统一转换为 double（模式与 am_native_Gfx.c 一致）。
static bool fs_pop_number(am_process_t *proc, double *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;

    if (am_value_is_float(v)) { *out = (double)am_value_to_float(v); return true; }
    if (am_value_is_int(v))   { *out = (double)am_value_to_int(v);   return true; }
    if (am_value_is_uint(v))  { *out = (double)am_value_to_uint(v);  return true; }
    return false;
}

// 弹出数值并截断为 int32（NaN 视为失败）。
static bool fs_pop_int(am_process_t *proc, int32_t *out) {
    double f;
    if (!fs_pop_number(proc, &f)) return false;
    if (f != f) return false; // NaN
    if (f > 2147483647.0)  f = 2147483647.0;
    if (f < -2147483648.0) f = -2147483648.0;
    *out = (int32_t)f;
    return true;
}

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
// cap 允许为 1 —— 那是**空串**的唯一合法容量（只需留一个 NUL）。此前这里写成
// cap<2 报错，导致 (Fs.write path "") 永远失败；文件管理器「新建空文件」需要它。
static size_t fs_wstring_to_utf8(am_wstring_t *ws, char *out, size_t cap) {
    if (!ws || !out || cap < 1) return (size_t)-1;
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
bool am_native_fs_resolve(am_wstring_t *ws, char *out) {
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

// 是否是「业务资源根本身」——删除/改名必须拒绝（否则整个资源目录会被干掉）。
// fs_resolve 把 "." 解析成 "<root>/."、把 "" 拒绝掉，因此这三种写法都要认。
static bool fs_is_root_path(const char *path) {
    size_t rl = strlen(PLATFORM_ROOT_DIR);

    if (strncmp(path, PLATFORM_ROOT_DIR, rl) != 0) return false;
    const char *rest = path + rl; // "" / "/" / "/."
    if (rest[0] == '\0') return true;
    if (strcmp(rest, "/") == 0) return true;
    if (strcmp(rest, "/.") == 0) return true;
    return false;
}

// 通知落盘层「旧路径已消失」（仅当平台提供了钩子；桌面/掌机为空操作）
static void fs_notify_remove(const char *path) {
#if defined(__GNUC__)
    if (am_native_fs_on_remove) am_native_fs_on_remove(path);
#else
    (void)path;
#endif
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

// 目录树改名后的「落盘搬运」：把新树里的每个文件重新 put，并把旧路径的键删掉。
// 为什么需要：WASM 的 IndexedDB 是按路径存的键值库，MEMFS 里改了名并不会影响它；
// 不搬的话，刷新页面后旧路径的文件会从 store 里重新活过来、新路径反而消失。
// 只在平台提供了钩子时执行（桌面/掌机零开销）；深度与文件数都有硬上限，
// 超出部分静默跳过——宁可少同步几个文件，也不要把改名操作卡住。
#define FS_PERSIST_MAX_DEPTH (8)
#define FS_PERSIST_MAX_FILES (512)

static int32_t s_persist_files = 0;

static void fs_persist_move_tree(const char *old_path, const char *new_path, int32_t depth) {
#if defined(__GNUC__)
    if (!am_native_fs_on_write) return;
    if (depth > FS_PERSIST_MAX_DEPTH) return;

    int32_t n = list_files(new_path, NULL);
    if (n <= 0) return;

    char **names = (char **)platform_calloc((size_t)n, sizeof(char *));
    if (!names) return;
    n = list_files(new_path, names);
    if (n < 0) {
        free(names);
        return;
    }

    for (int32_t i = 0; i < n; i++) {
        char child_new[FS_PATH_MAX];
        char child_old[FS_PATH_MAX];
        int32_t wn = snprintf(child_new, sizeof child_new, "%s/%s", new_path, names[i]);
        int32_t wo = snprintf(child_old, sizeof child_old, "%s/%s", old_path, names[i]);
        if (wn <= 0 || wn >= (int32_t)sizeof child_new || wo <= 0 || wo >= (int32_t)sizeof child_old) continue;

        if (platform_is_directory(child_new)) {
            fs_persist_move_tree(child_old, child_new, depth + 1);
            continue;
        }
        if (s_persist_files >= FS_PERSIST_MAX_FILES) continue;

        size_t sz = 0;
        uint8_t *buf = fs_read_raw(child_new, &sz);
        if (!buf) continue; // 空文件/过大/读失败：跳过（不影响改名本身）
        s_persist_files++;
        am_native_fs_on_write(child_new, buf, sz);
        fs_notify_remove(child_old);
        free(buf);
    }

    for (int32_t i = 0; i < n; i++) free(names[i]);
    free(names);
#else
    (void)old_path;
    (void)new_path;
    (void)depth;
#endif
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

// 由 UTF-8 缓冲区构造 Scheme 字符串把柄（失败返回 AM_HANDLE_NULL）。
static am_handle_t fs_make_string_handle(am_process_t *proc, const char *utf8, size_t len) {
    // 最坏情况：每个字节一个码点
    wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (len + 1) * sizeof(wchar_t));
    if (!wbuf) return AM_HANDLE_NULL;

    uint32_t wlen = _mbstowcs(wbuf, utf8, (uint32_t)len);
    am_handle_t hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
    am_free(proc->heap_alloc, wbuf);
    return hd;
}

// 由 UTF-8 缓冲区构造 Scheme 字符串并压栈（失败压 #null）
static int32_t fs_push_string(am_process_t *proc, const char *utf8, size_t len) {
    am_handle_t hd = fs_make_string_handle(proc, utf8, len);
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
    if (am_native_fs_resolve(ws, path)) {
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
    if (am_native_fs_resolve(ws, path)) ok = (platform_is_directory(path) != 0);
    return fs_push_bool(proc, ok);
}

int32_t am_native_Fs_size(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    (void)rt;

    if (!fs_pop_wstring(proc, &ws)) return -1;
    if (!am_native_fs_resolve(ws, path)) return fs_push_null(proc);

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
    if (!am_native_fs_resolve(ws, path)) return fs_push_null(proc);

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
    if (am_native_fs_resolve(ws, path)) {
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
    if (!am_native_fs_resolve(ws, path)) return fs_push_null(proc);

    buf = fs_read_raw(path, &size);
    if (!buf) return fs_push_null(proc);

    int32_t ret = fs_push_string(proc, (const char *)buf, size);
    free(buf);
    return ret;
}

// (Fs.read_range path:String offset:Number maxbytes:Number) : (文本 下一偏移) | #null
//
// 分段读取（大文件流式处理的底座，见 app/lib/md_stream.scm）：
//   · offset 为**字节**偏移，从文件头算起；maxbytes 是**期望**上限（钳到 64KB）；
//   · 返回值第一个元素是从 offset 起的文本，第二个元素是**下一偏移**（offset + 实际
//     消费字节数），调用方用它推进；offset >= 文件大小 时返回空串（=EOF）；
//   · **不切断 UTF-8 码点**：非 EOF 情况下实际消费量回退到最后一个完整码点的边界，
//     因此相邻两次读取拼起来恰是原文件的 UTF-8 分解（Scheme 侧因此不会看到半个字）；
//   · 路径不存在/非法/读失败 → #null（与 Fs.read 的软失败约定一致）。
int32_t am_native_Fs_read_range(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    int32_t off = 0;
    int32_t want = 0;
    uint8_t *buf = NULL;
    size_t got = 0;
    size_t take = 0;
    size_t cap = 0;
    uint32_t size = 0;
    (void)rt;

    // 参数从右往左弹：maxbytes → offset → path（与求值顺序相反）
    if (!fs_pop_int(proc, &want)) return -1;
    if (!fs_pop_int(proc, &off)) return -1;
    if (!fs_pop_wstring(proc, &ws)) return -1;

    if (off < 0 || want <= 0) return fs_push_null(proc);
    if (want > (int32_t)AM_NATIVE_FS_RANGE_MAX) want = (int32_t)AM_NATIVE_FS_RANGE_MAX;
    if (!am_native_fs_resolve(ws, path)) return fs_push_null(proc);

    if (platform_file_open(path) != 0) return fs_push_null(proc);
    size = platform_file_size();
    if ((uint32_t)off > size) {
        platform_file_close();
        return fs_push_null(proc);
    }
    if (platform_file_seek((uint32_t)off) != 0) {
        platform_file_close();
        return fs_push_null(proc);
    }

    // 多读 4 字节：保证至少能凑出一个完整码点（见下面 take 的算法）
    cap = (size_t)want + 4u;
    buf = (uint8_t *)platform_malloc(cap);
    if (!buf) {
        platform_file_close();
        return fs_push_null(proc);
    }
    while (got < cap) {
        int32_t r = platform_file_read(buf + got, cap - got);
        if (r <= 0) break; // 读失败或提前 EOF：按实际读到的字节数处理
        got += (size_t)r;
    }
    platform_file_close();

    bool eof = ((uint32_t)off + (uint32_t)got >= size);
    // offset 允许落在码点中间（调用方从文件尾倒推、按固定步长跳读等场景）：
    // 前移到下一个码点起始字节再算，保证返回的必然是完整码点序列。
    size_t skip = 0;
    while (skip < got && skip < 3 && (buf[skip] & 0xC0) == 0x80) skip++;

    const uint8_t *d = buf + skip;
    size_t gn = got - skip;
    uint32_t base = (uint32_t)off + (uint32_t)skip;
    take = eof ? gn : (gn < (size_t)want ? gn : (size_t)want);
    // 非 EOF 时保证返回的是**完整码点**的前缀：定位 take 处最后一个码点的起始字节，
    // 若该码点被截断，则把它整体留给下一次读取；take 退到 0 时至少返回一个完整码点
    // （因此返回字节数可能比 maxbytes 多 1..3，属预期）。
    if (!eof && take > 0) {
        size_t p = take;
        while (p > 0 && (d[p - 1] & 0xC0) == 0x80) p--; // p-1 = 最后那个码点的起始字节
        if (p > 0) {
            size_t lead = d[p - 1];
            size_t clen = (lead < 0x80) ? 1
                        : ((lead & 0xE0) == 0xC0) ? 2
                        : ((lead & 0xF0) == 0xE0) ? 3 : 4;
            if ((p - 1) + clen > take) take = p - 1; // 该码点不完整 → 让给下一次
        }
        if (take == 0) {
            size_t lead = d[0];
            size_t clen = (lead < 0x80) ? 1
                        : ((lead & 0xE0) == 0xC0) ? 2
                        : ((lead & 0xF0) == 0xE0) ? 3 : 4;
            take = (gn < clen) ? gn : clen;   // 极端小 maxbytes：至少 1 个码点
        }
    }

    am_handle_t shd = fs_make_string_handle(proc, (const char *)d, take);
    free(buf);
    if (shd == AM_HANDLE_NULL) return fs_push_null(proc);

    am_list_t *lst = am_list_create(proc->heap_alloc, 2, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!lst) {
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, shd);
        return fs_push_null(proc);
    }
    am_list_t *grown = am_list_push(proc->heap_alloc, lst, am_make_value_of_handle(shd));
    if (!grown) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, shd);
        return fs_push_null(proc);
    }
    lst = grown;
    grown = am_list_push(proc->heap_alloc, lst,
                         am_make_value_of_uint((am_uint_t)(base + (uint32_t)take)));
    if (!grown) {
        am_list_destroy(proc->heap_alloc, lst);
        return fs_push_null(proc);
    }
    lst = grown;

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
        if (len != (size_t)-1 && am_native_fs_resolve(ws_path, path)) {
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
        if (len != (size_t)-1 && am_native_fs_resolve(ws_path, path)) {
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

int32_t am_native_Fs_remove(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[FS_PATH_MAX];
    bool ok = false;
    (void)rt;

    if (!fs_pop_wstring(proc, &ws)) return -1;
    if (am_native_fs_resolve(ws, path) && !fs_is_root_path(path)) {
        // 非空目录会失败（POSIX remove / SD.rmdir 都是如此），由 App 自己递归清空
        ok = (platform_remove_path(path) == 0);
        if (ok) fs_notify_remove(path);
    }
    return fs_push_bool(proc, ok);
}

int32_t am_native_Fs_rename(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws_new = NULL;
    am_wstring_t *ws_old = NULL;
    char old_path[FS_PATH_MAX];
    char new_path[FS_PATH_MAX];
    bool ok = false;
    (void)rt;

    // 参数从右往左弹：先 new，再 old
    if (!fs_pop_wstring(proc, &ws_new)) return -1;
    if (!fs_pop_wstring(proc, &ws_old)) return -1;

    if (am_native_fs_resolve(ws_old, old_path) && am_native_fs_resolve(ws_new, new_path)
        && !fs_is_root_path(old_path) && !fs_is_root_path(new_path)) {
        // 源必须存在；目标必须不存在（不静默覆盖——POSIX rename 会覆盖同名文件）
        bool src_is_dir = (platform_is_directory(old_path) != 0);
        bool src_exists = src_is_dir || (fs_file_exists(old_path) != 0);
        bool dst_busy = (platform_is_directory(new_path) != 0) || (fs_file_exists(new_path) != 0);
        if (src_exists && !dst_busy) {
            // 普通文件：先把字节读出来，改名成功后按新路径重新落盘（WASM 的 IndexedDB）
            size_t sz = 0;
            uint8_t *buf = src_is_dir ? NULL : fs_read_raw(old_path, &sz);

#if defined(__GNUC__)
            bool persist = (am_native_fs_on_write != NULL) || (am_native_fs_on_remove != NULL);
#else
            bool persist = false;
#endif

            ok = (platform_rename_path(old_path, new_path) == 0);
            if (ok && persist) {
                if (src_is_dir) {
                    s_persist_files = 0;
                    fs_persist_move_tree(old_path, new_path, 0);
                }
                else {
#if defined(__GNUC__)
                    if (buf && am_native_fs_on_write) am_native_fs_on_write(new_path, buf, sz);
#endif
                    fs_notify_remove(old_path);
                }
            }
            if (buf) free(buf);
        }
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
    { L"read_range", am_native_Fs_read_range },
    { L"write",  am_native_Fs_write },
    { L"append", am_native_Fs_append },
    { L"remove", am_native_Fs_remove },
    { L"rename", am_native_Fs_rename },
};

const am_native_lib_entry_t am_native_Fs_lib = {
    L"Fs",
    am_native_Fs_funcs,
    sizeof(am_native_Fs_funcs) / sizeof(am_native_Fs_funcs[0])
};