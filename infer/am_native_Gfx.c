////////////////////////////////////////////////////////////////////////////
//  Native Library : Gfx —— Nano 图形层的灵机（Animac）本地库封装
//
//  栈纪律（与本项目 vendored 解释器的语句型内建 display/newline 一致）：
//    - 语句型函数（clear/fill_white/set_pixel/draw_*/fill_*/refresh）：
//      弹出全部参数后【不向操作数栈压值】，仅推进 PC——只能用于语句位置；
//    - 值型函数（width/height）：向操作数栈压入恰好 1 个值。
//  注意：上游新版编译器采用“单值栈纪律”（语句压 #undefined 后由编译器弹
//  出），届时语句型函数需改为压入 #undefined（同步自 upstream 时注意）。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <wchar.h>

#include "am_native_Gfx.h"


// 绘图目标（宿主经 am_native_gfx_bind 注入）
static Nano_GFX *s_gfx = NULL;

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

// 语句型收尾：不向操作数栈压值，仅推进 PC（与 display/newline 同语义）。
static int32_t native_statement_done(am_process_t *proc) {
    am_process_step(proc);
    return 0;
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
        gfx_font_draw_text(g, GFX_FONT_ALPHA_12, text, x, y, r, gc, b, 1);
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
    { L"refresh",     am_native_Gfx_refresh },
};

const am_native_lib_entry_t am_native_Gfx_lib = {
    L"Gfx",
    am_native_Gfx_funcs,
    sizeof(am_native_Gfx_funcs) / sizeof(am_native_Gfx_funcs[0])
};
