////////////////////////////////////////////////////////////////////////////
//  Native Library : Gfx —— Nano 图形层的灵机（Animac）本地库封装
//
//  栈纪律（遵循上游“编译期单值栈纪律”契约，见解释器文档 doc/todo/stack_bal.md §11）：
//    - 值型函数（width/height）：弹出全部参数后向操作数栈压入恰好 1 个值（N1）；
//    - 语句型函数（clear/fill_white/set_pixel/draw_*/fill_*/refresh）：弹全部
//      参数后压入 1 个 #undefined（N2）——编译器会在语句位置把它弹出，因此这类
//      函数既可写在语句位置，也可以合法地出现在任何要求“恰好 1 个值”的表达式
//      位置（其值为 #undefined）。
//  任何 native 函数都不得压 0 个或 2 个以上值，否则会破坏编译器的栈平衡假设。
//  同步上游时必须复核本文件与 N1–N10 契约的一致性。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <wchar.h>

#include "am_native_Gfx.h"
#include "am_native_Fs.h"   // 路径解析复用 Fs 本地库的规则


// 绘图目标（宿主经 am_native_gfx_bind 注入）
static Nano_GFX *s_gfx = NULL;

// 当前字模（0=12px 点阵 / 1=12px 抗锯齿 / 2=16px 抗锯齿），由 (Gfx.set_font id) 切换。
// 默认沿用历史行为：12px 抗锯齿。
static uint32_t s_font_id = 1;

void am_native_gfx_bind(Nano_GFX *gfx) {
    s_gfx = gfx;
}

// 取当前绘图目标；未绑定时向虚拟机报错并返回 NULL。
static Nano_GFX *native_gfx(am_runtime_t *rt) {
    if (s_gfx == NULL && rt != NULL) {
        am_runtime_error(rt, L"[Gfx] 未绑定绘图目标（宿主未调用 am_native_gfx_bind）\n");
    }
    return s_gfx;
}


// ===============================================================================
// 操作数栈辅助
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

// 弹出数值并截断为 int32。
static bool native_pop_int(am_process_t *proc, int32_t *out) {
    am_float_t f;
    if (!native_pop_number(proc, &f)) return false;
    if (isnan(f)) return false;
    if (f > 2147483647.0)  f = 2147483647.0;
    if (f < -2147483648.0) f = -2147483648.0;
    *out = (int32_t)f;
    return true;
}

// 弹出数值并钳位为 [0,255] 的 uint8 颜色分量。
static bool native_pop_color(am_process_t *proc, uint8_t *out) {
    am_float_t f;
    if (!native_pop_number(proc, &f)) return false;
    if (isnan(f)) return false;
    if (f < 0.0)   f = 0.0;
    if (f > 255.0) f = 255.0;
    *out = (uint8_t)f;
    return true;
}

// 语句型收尾：压 1 个 #undefined 再推进 PC（契约 N2；与 Table.set 等语句型
// native 同范式）。编译器在语句位置会把这个值弹出，因此栈平衡始终成立。
static int32_t native_statement_done(am_process_t *proc) {
    if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// 软失败收尾：压 #null 并推进 PC（与 Fs 本地库的软失败约定一致）。
static int32_t native_push_null(am_process_t *proc) {
    if (am_process_push_operand(proc, AM_VALUE_NULL) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// 压入一个两元素数值列表 (a b) 并推进 PC（图片尺寸用）。
static int32_t native_push_pair(am_process_t *proc, am_float_t a, am_float_t b) {
    am_list_t *lst = am_list_create(proc->heap_alloc, 2, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    am_list_t *grown;
    am_handle_t lhd;
    if (lst == NULL) return native_push_null(proc);
    grown = am_list_push(proc->heap_alloc, lst, am_make_value_of_float(a));
    if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return native_push_null(proc); }
    lst = grown;
    grown = am_list_push(proc->heap_alloc, lst, am_make_value_of_float(b));
    if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return native_push_null(proc); }
    lst = grown;

    lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return native_push_null(proc); }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return native_push_null(proc);
    }
    if (am_process_push_operand(proc, am_make_value_of_handle(lhd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// 把 Scheme 路径解析为 HAL 路径（规则与 Fs 本地库完全一致：相对业务资源根、
// 以 '/' 开头视为绝对路径、拒绝含 ".." 的路径）。成功返回 true。
static bool native_resolve_path(am_wstring_t *ws, char *out, size_t cap) {
    if (ws == NULL || out == NULL) return false;
    return am_native_fs_resolve(ws, out) && strlen(out) < cap;
}

// 数值收尾：压入数值并推进 PC。
static int32_t native_push_number(am_process_t *proc, am_float_t result) {
    if (am_process_push_operand(proc, am_make_value_of_float(result)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// 从操作数栈中弹出一个字符串对象。
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

// 将 wstring 对象的内容转换为以 L'\0' 结尾的 wchar_t 缓冲区。
// 缓冲区由进程堆分配器分配，调用者负责 am_free(proc->heap_alloc, buf)。
static wchar_t *wstring_to_buffer(am_process_t *proc, am_wstring_t *ws) {
    if (ws == NULL || ws->length == 0) return NULL;

    wchar_t *buf = (wchar_t *)am_malloc(proc->heap_alloc, (ws->length + 1) * sizeof(wchar_t));
    if (buf == NULL) return NULL;

    for (size_t i = 0; i < ws->length; i++) {
        buf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    }
    buf[ws->length] = L'\0';
    return buf;
}


// ===============================================================================
// Native 函数实现
// ===============================================================================

int32_t am_native_Gfx_width(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    (void)proc;
    return native_push_number(proc, (am_float_t)g->width);
}


int32_t am_native_Gfx_height(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    (void)proc;
    return native_push_number(proc, (am_float_t)g->height);
}


int32_t am_native_Gfx_clear(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    (void)proc;
    gfx_soft_clear(g);
    return native_statement_done(proc);
}


int32_t am_native_Gfx_fill_white(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    (void)proc;
    gfx_fill_white(g);
    return native_statement_done(proc);
}


int32_t am_native_Gfx_set_pixel(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    uint8_t b, gc, r;
    int32_t x, y;
    if (!native_pop_color(proc, &b))  return -1;
    if (!native_pop_color(proc, &gc)) return -1;
    if (!native_pop_color(proc, &r))  return -1;
    if (!native_pop_int(proc, &y))    return -1;
    if (!native_pop_int(proc, &x))    return -1;
    // gfx_set_pixel 不做越界裁剪，此处显式判界
    if (x >= 0 && y >= 0 && x < (int32_t)g->width && y < (int32_t)g->height) {
        gfx_set_pixel(g, (uint32_t)x, (uint32_t)y, r, gc, b);
    }
    return native_statement_done(proc);
}


int32_t am_native_Gfx_draw_point(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    uint8_t b, gc, r;
    int32_t x, y;
    if (!native_pop_color(proc, &b))  return -1;
    if (!native_pop_color(proc, &gc)) return -1;
    if (!native_pop_color(proc, &r))  return -1;
    if (!native_pop_int(proc, &y))    return -1;
    if (!native_pop_int(proc, &x))    return -1;
    gfx_draw_point(g, (uint32_t)x, (uint32_t)y, r, gc, b, 1); // 内部含越界/裁剪判界
    return native_statement_done(proc);
}


int32_t am_native_Gfx_draw_line(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    uint8_t b, gc, r;
    int32_t x2, y2, x1, y1;
    if (!native_pop_color(proc, &b))  return -1;
    if (!native_pop_color(proc, &gc)) return -1;
    if (!native_pop_color(proc, &r))  return -1;
    if (!native_pop_int(proc, &y2))   return -1;
    if (!native_pop_int(proc, &x2))   return -1;
    if (!native_pop_int(proc, &y1))   return -1;
    if (!native_pop_int(proc, &x1))   return -1;
    gfx_draw_line(g, (uint32_t)x1, (uint32_t)y1, (uint32_t)x2, (uint32_t)y2, r, gc, b, 1);
    return native_statement_done(proc);
}


int32_t am_native_Gfx_fill_rect(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    uint8_t b, gc, r;
    int32_t h, w, y, x;
    if (!native_pop_color(proc, &b))  return -1;
    if (!native_pop_color(proc, &gc)) return -1;
    if (!native_pop_color(proc, &r))  return -1;
    if (!native_pop_int(proc, &h))    return -1;
    if (!native_pop_int(proc, &w))    return -1;
    if (!native_pop_int(proc, &y))    return -1;
    if (!native_pop_int(proc, &x))    return -1;
    if (w > 0 && h > 0) {
        gfx_draw_rectangle(g, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h, r, gc, b, 1);
    }
    return native_statement_done(proc);
}


int32_t am_native_Gfx_draw_rect(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    uint8_t b, gc, r;
    int32_t h, w, y, x;
    if (!native_pop_color(proc, &b))  return -1;
    if (!native_pop_color(proc, &gc)) return -1;
    if (!native_pop_color(proc, &r))  return -1;
    if (!native_pop_int(proc, &h))    return -1;
    if (!native_pop_int(proc, &w))    return -1;
    if (!native_pop_int(proc, &y))    return -1;
    if (!native_pop_int(proc, &x))    return -1;
    // 1px 边框：四条边各画一条线（gfx_draw_line 自含越界裁剪）
    if (w > 0 && h > 0) {
        int32_t rx1 = x + w - 1;
        int32_t ry1 = y + h - 1;
        gfx_draw_line(g, (uint32_t)x,    (uint32_t)y,    (uint32_t)rx1, (uint32_t)y,    r, gc, b, 1);
        gfx_draw_line(g, (uint32_t)x,    (uint32_t)ry1,  (uint32_t)rx1, (uint32_t)ry1,  r, gc, b, 1);
        gfx_draw_line(g, (uint32_t)x,    (uint32_t)y,    (uint32_t)x,   (uint32_t)ry1,  r, gc, b, 1);
        gfx_draw_line(g, (uint32_t)rx1,  (uint32_t)y,    (uint32_t)rx1, (uint32_t)ry1,  r, gc, b, 1);
    }
    return native_statement_done(proc);
}


int32_t am_native_Gfx_draw_circle(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    uint8_t b, gc, r;
    int32_t rad, cy, cx;
    if (!native_pop_color(proc, &b))  return -1;
    if (!native_pop_color(proc, &gc)) return -1;
    if (!native_pop_color(proc, &r))  return -1;
    if (!native_pop_int(proc, &rad))  return -1;
    if (!native_pop_int(proc, &cy))   return -1;
    if (!native_pop_int(proc, &cx))   return -1;
    if (rad > 0) {
        gfx_draw_circle(g, (uint32_t)cx, (uint32_t)cy, (uint32_t)rad, r, gc, b, 1);
    }
    return native_statement_done(proc);
}


int32_t am_native_Gfx_fill_circle(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    uint8_t b, gc, r;
    int32_t rad, cy, cx;
    if (!native_pop_color(proc, &b))  return -1;
    if (!native_pop_color(proc, &gc)) return -1;
    if (!native_pop_color(proc, &r))  return -1;
    if (!native_pop_int(proc, &rad))  return -1;
    if (!native_pop_int(proc, &cy))   return -1;
    if (!native_pop_int(proc, &cx))   return -1;
    if (rad > 0) {
        gfx_draw_circle_fill(g, (uint32_t)cx, (uint32_t)cy, (uint32_t)rad, r, gc, b, 1);
    }
    return native_statement_done(proc);
}


int32_t am_native_Gfx_draw_text(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    uint8_t b, gc, r;
    int32_t y, x;
    am_wstring_t *ws;
    if (!native_pop_color(proc, &b))  return -1;
    if (!native_pop_color(proc, &gc)) return -1;
    if (!native_pop_color(proc, &r))  return -1;
    if (!native_pop_int(proc, &y))    return -1;
    if (!native_pop_int(proc, &x))    return -1;
    if (!native_pop_wstring(proc, &ws)) return -1;

    wchar_t *text = wstring_to_buffer(proc, ws);
    if (text != NULL) {
        gfx_font_draw_text(g, s_font_id, text, x, y, r, gc, b, 1);
        am_free(proc->heap_alloc, text);
    }
    return native_statement_done(proc);
}


int32_t am_native_Gfx_refresh(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    (void)proc;
    gfx_refresh(g);
    return native_statement_done(proc);
}


// ===============================================================================
// 库注册表
// ===============================================================================

// (Gfx.image_size path:String) : (宽 高) | #null
// 只读文件头取原始尺寸（stbi_info），不解码像素 —— 供排版阶段算缩放与行高。
int32_t am_native_Gfx_image_size(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[256];
    uint32_t w = 0, h = 0;
    (void)rt;

    if (!native_pop_wstring(proc, &ws)) return -1;
    if (!native_resolve_path(ws, path, sizeof path)) return native_push_null(proc);
    if (gfx_image_info(path, &w, &h) != 0) return native_push_null(proc);
    return native_push_pair(proc, (am_float_t)w, (am_float_t)h);
}

// (Gfx.draw_image path:String x:Number y:Number w:Number h:Number) : Boolean
// 从文件读图、缩放到 (w,h) 并绘制（宿主侧带 LRU 缓存；像素不进解释器堆）。
// 失败（路径非法/文件缺失/解码失败/尺寸非法）返回 #f —— 调用方据此画占位框。
int32_t am_native_Gfx_draw_image(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    am_wstring_t *ws = NULL;
    char path[256];
    int32_t h = 0, w = 0, y = 0, x = 0;
    if (g == NULL) return -1;

    if (!native_pop_int(proc, &h)) return -1;
    if (!native_pop_int(proc, &w)) return -1;
    if (!native_pop_int(proc, &y)) return -1;
    if (!native_pop_int(proc, &x)) return -1;
    if (!native_pop_wstring(proc, &ws)) return -1;

    if (w <= 0 || h <= 0) {
        if (am_process_push_operand(proc, AM_VALUE_FALSE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }
    if (!native_resolve_path(ws, path, sizeof path)) {
        if (am_process_push_operand(proc, AM_VALUE_FALSE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }
    if (gfx_draw_image(g, path, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h, 0) != 0) {
        if (am_process_push_operand(proc, AM_VALUE_FALSE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }
    if (am_process_push_operand(proc, AM_VALUE_TRUE) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// (Gfx.set_font id:Number) : Boolean
// —— 选择 Gfx.draw_text 使用的字模：0 = 12px 二值点阵、1 = 12px 抗锯齿、2 = 16px 抗锯齿。
//    非法 id 返回 #f（软失败，不报错）。字模影响**行高与步进**，调用方切换后应按
//    (Gfx.font_info id) 重新排版（见 app/markdown.scm 的 apply_font）。
int32_t am_native_Gfx_set_font(am_runtime_t *rt, am_process_t *proc) {
    int32_t id = 0;
    (void)rt;
    if (!native_pop_int(proc, &id)) return -1;
    if (id < 0 || id > 2) {
        if (am_process_push_operand(proc, AM_VALUE_FALSE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }
    s_font_id = (uint32_t)id;
    if (am_process_push_operand(proc, AM_VALUE_TRUE) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// (Gfx.font_info id:Number) : List | #null
// —— 该字模的 (半角步进 全角步进 行高 基线)；id 非法返回 #null。
//    排版端拿它算折行宽度与行高（见 md_layout 的 set_font_metrics）。
int32_t am_native_Gfx_font_info(am_runtime_t *rt, am_process_t *proc) {
    int32_t id = 0;
    am_list_t *lst;
    am_handle_t lhd;
    (void)rt;
    if (!native_pop_int(proc, &id)) return -1;
    if (id < 0 || id > 2) return native_push_null(proc);

    lst = am_list_create(proc->heap_alloc, 4, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (lst == NULL) return native_push_null(proc);
    am_list_t *grown = am_list_push(proc->heap_alloc, lst,
        am_make_value_of_float((am_float_t)gfx_font_char_advance((uint32_t)id, (uint32_t)'i')));
    if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return native_push_null(proc); }
    lst = grown;
    grown = am_list_push(proc->heap_alloc, lst,
        am_make_value_of_float((am_float_t)gfx_font_char_advance((uint32_t)id, (uint32_t)0x4E2D)));
    if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return native_push_null(proc); }
    lst = grown;
    grown = am_list_push(proc->heap_alloc, lst,
        am_make_value_of_float((am_float_t)gfx_font_line_height((uint32_t)id)));
    if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return native_push_null(proc); }
    lst = grown;
    grown = am_list_push(proc->heap_alloc, lst,
        am_make_value_of_float((am_float_t)gfx_font_baseline((uint32_t)id)));
    if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return native_push_null(proc); }
    lst = grown;

    lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return native_push_null(proc); }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return native_push_null(proc);
    }
    if (am_process_push_operand(proc, am_make_value_of_handle(lhd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static const am_native_func_entry_t am_native_Gfx_funcs[] = {
    { L"width",       am_native_Gfx_width },
    { L"height",      am_native_Gfx_height },
    { L"clear",       am_native_Gfx_clear },
    { L"fill_white",  am_native_Gfx_fill_white },
    { L"set_pixel",   am_native_Gfx_set_pixel },
    { L"draw_point",  am_native_Gfx_draw_point },
    { L"draw_line",   am_native_Gfx_draw_line },
    { L"fill_rect",   am_native_Gfx_fill_rect },
    { L"draw_rect",   am_native_Gfx_draw_rect },
    { L"draw_circle", am_native_Gfx_draw_circle },
    { L"fill_circle", am_native_Gfx_fill_circle },
    { L"draw_text",   am_native_Gfx_draw_text },
    { L"set_font",    am_native_Gfx_set_font },
    { L"font_info",   am_native_Gfx_font_info },
    { L"image_size",  am_native_Gfx_image_size },
    { L"draw_image",  am_native_Gfx_draw_image },
    { L"refresh",     am_native_Gfx_refresh },
};

const am_native_lib_entry_t am_native_Gfx_lib = {
    L"Gfx",
    am_native_Gfx_funcs,
    sizeof(am_native_Gfx_funcs) / sizeof(am_native_Gfx_funcs[0])
};
