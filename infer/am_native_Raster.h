////////////////////////////////////////////////////////////////////////////
//  Native Library : Raster —— 水池 3D 软光栅**后端**（薄 C / 重 Scheme）
//
//  设计要点（与 (native Water) 的"整机封装"版对照）：
//
//    (native Water)   ：C 持有渲染器 + 相机 + 物理 + 输入状态机 + HUD；
//                       Scheme 只有 3 行骨架（App/waterpool_native.scm，97 行）。
//    (native Raster)  ：C 只持有**逐像素/逐格点/逐三角形内层循环**与它们必需的数据
//                       （帧缓冲、z-buffer、波场双页、焦散纹理、已上传的网格）；
//                       几何、相机、绘制顺序、剔除策略、光照、物理、交互、HUD 全在
//                       Scheme（App/waterpool_raster.scm，约 700 行）。
//
//  边界为什么不能再细（实测，见 am_native_Raster.c 头部的分段归因表）：
//  水面 pass 光是"逐顶点 + 逐三角形 setup"就要 816 ms/帧、焦散约 900 ms、圆球 86 ms
//  —— 已超 10fps 预算（100ms）的十几倍，所以"Scheme 里逐三角形发 draw call"不成立。
//  反过来再粗一档就退化成 Water 版（连相机/输入/HUD 都进 C）。
//
//  栈纪律（遵循上游"编译期单值栈纪律"契约 doc/todo/stack_bal.md §11）：
//    值型函数（init）压 1 个值；语句型函数压 1 个 #undefined。
//    任何 native 函数都不得压 0 个或 2 个以上值。
//
//  坐标/网格约定：
//    · 网格顶点为 5 个 float：(x, y, z, u, v)。
//    · 池壁/圆球： (x,y,z) 是三维坐标（池壁为世界坐标；圆球为单位球，按 center/radius
//      缩放）。u,v 未用。
//    · 水面：沿用原版 WT_GeoV 布局——(x, y) 是**平面坐标 (x, z)**，z 未用，y 由波场
//      高度替代（与 ui_water.c 的 wt_water_vert_world 一致）。
////////////////////////////////////////////////////////////////////////////

#ifndef __AM_NATIVE_RASTER_H__
#define __AM_NATIVE_RASTER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

#include "graphics.h"

// 由 am_native_Raster.c 定义
extern const am_native_lib_entry_t am_native_Raster_lib;

// 绑定绘图目标帧缓冲（传 NULL 解除绑定）——与 (native Gfx) 同一块 Nano_GFX。
void am_native_raster_bind(Nano_GFX *gfx);

// (Raster.init) : Number —— 按 Gfx 尺寸分配 帧缓冲/z-buffer/波场/焦散纹理；成功 1 / 失败 0
int32_t am_native_Raster_init(am_runtime_t *rt, am_process_t *proc);

// (Raster.mesh id verts tris) : #undefined —— 上传一个网格
//   verts：每 5 个数一个顶点 (x, y, z, u, v)（见文件头约定）
//   tris ：每 3 个数一个三角形（顶点下标）
int32_t am_native_Raster_mesh(am_runtime_t *rt, am_process_t *proc);

// (Raster.begin) : #undefined —— 清帧缓冲（黑）+ 深度（远）
int32_t am_native_Raster_begin(am_runtime_t *rt, am_process_t *proc);

// (Raster.draw mesh kind mvp params) : #undefined
//   mvp   ：16 元列表（行主序 P·MV）
//   params：11 元列表
//           (eye.x eye.y eye.z  light.x light.y light.z  ball.x ball.y ball.z radius cull)
//   kind  ：0 池壁 / 1 水面（水上）/ 2 水面（水下）/ 3 圆球
//   cull  ：0 不剔除 / 1 只留屏幕带符号面积<0 / 2 只留>=0
int32_t am_native_Raster_draw(am_runtime_t *rt, am_process_t *proc);

// (Raster.sim_move ox oy oz cx cy cz radius) : #undefined
//   —— 圆球排开/回填水体（逐格点；写另一页并交换）
int32_t am_native_Raster_sim_move(am_runtime_t *rt, am_process_t *proc);

// (Raster.sim_step) : #undefined —— 波场推进一拍（逐格点双页乒乓）
int32_t am_native_Raster_sim_step(am_runtime_t *rt, am_process_t *proc);

// (Raster.sim_normals) : #undefined —— 由高度场重算法线（逐格点）
int32_t am_native_Raster_sim_normals(am_runtime_t *rt, am_process_t *proc);

// (Raster.caustics mesh light.x light.y light.z ball.x ball.y ball.z radius) : #undefined
//   —— 焦散 pass：把水面网格沿折射光投影到池底，渲进 256² 焦散纹理
int32_t am_native_Raster_caustics(am_runtime_t *rt, am_process_t *proc);

// (Raster.blit) : #undefined —— 后端 RGB565 帧缓冲 → 屏幕帧缓冲
int32_t am_native_Raster_blit(am_runtime_t *rt, am_process_t *proc);

// (Raster.dispose) : #undefined —— 释放全部缓冲与已上传网格（幂等）
int32_t am_native_Raster_dispose(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif
