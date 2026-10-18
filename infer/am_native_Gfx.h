#ifndef __AM_NATIVE_GFX_H__
#define __AM_NATIVE_GFX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

#include "graphics.h"


////////////////////////////////////////////////////////////////////////////
//  Native Library : Gfx
//
//  将 Nano 图形层（Nano_GFX）封装为灵机（Animac）本地库，供 Scheme 程序
//  直接绘图。宿主须在创建解释器运行时后、加载进程前调用
//  am_native_gfx_bind() 绑定目标帧缓冲；未绑定时所有绘图函数报运行时错误。
//
//  Scheme 侧用法：源码中声明 (native Gfx) 后以 Gfx.函数名 调用。
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Gfx_lib;

// 绑定绘图目标帧缓冲（传 NULL 解除绑定）。宿主在解释器进程运行期间
// 必须保证 gfx 指向有效的、已完成 gfx_init 的 Nano_GFX。
void am_native_gfx_bind(Nano_GFX *gfx);

// (Gfx.width) : Number —— 画布宽度（像素）
int32_t am_native_Gfx_width(am_runtime_t *rt, am_process_t *proc);

// (Gfx.height) : Number —— 画布高度（像素）
int32_t am_native_Gfx_height(am_runtime_t *rt, am_process_t *proc);

// (Gfx.clear) : #undefined —— 清屏为黑
int32_t am_native_Gfx_clear(am_runtime_t *rt, am_process_t *proc);

// (Gfx.fill_white) : #undefined —— 清屏为白
int32_t am_native_Gfx_fill_white(am_runtime_t *rt, am_process_t *proc);

// (Gfx.set_pixel x:Number y:Number r:Number g:Number b:Number) : #undefined
// —— 直写像素颜色（自动越界裁剪）
int32_t am_native_Gfx_set_pixel(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_point x:Number y:Number r:Number g:Number b:Number) : #undefined
// —— 画点（置色语义，同 mode 1）
int32_t am_native_Gfx_draw_point(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_line x1:Number y1:Number x2:Number y2:Number r:Number g:Number b:Number) : #undefined
// —— 画线段
int32_t am_native_Gfx_draw_line(am_runtime_t *rt, am_process_t *proc);

// (Gfx.fill_rect x:Number y:Number w:Number h:Number r:Number g:Number b:Number) : #undefined
// —— 实心矩形
int32_t am_native_Gfx_fill_rect(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_rect x:Number y:Number w:Number h:Number r:Number g:Number b:Number) : #undefined
// —— 空心矩形（1px 边框）
int32_t am_native_Gfx_draw_rect(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_circle cx:Number cy:Number r:Number rr:Number gg:Number bb:Number) : #undefined
// —— 空心圆
int32_t am_native_Gfx_draw_circle(am_runtime_t *rt, am_process_t *proc);

// (Gfx.fill_circle cx:Number cy:Number r:Number rr:Number gg:Number bb:Number) : #undefined
// —— 实心圆
int32_t am_native_Gfx_fill_circle(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_text str:String x:Number y:Number r:Number g:Number b:Number) : #undefined
// —— 绘制一行文本（12px 平滑字模；(x,y) 为行框左上角）
int32_t am_native_Gfx_draw_text(am_runtime_t *rt, am_process_t *proc);

// (Gfx.refresh) : #undefined —— 推帧上屏（脏区局部刷新）
int32_t am_native_Gfx_refresh(am_runtime_t *rt, am_process_t *proc);


#ifdef __cplusplus
}
#endif

#endif
