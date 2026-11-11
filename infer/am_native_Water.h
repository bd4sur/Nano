////////////////////////////////////////////////////////////////////////////
//  Native Library : Water —— Nano 水池仿真渲染器的灵机（Animac）本地库封装
//
//  为什么需要本库（profile 依据见 doc/AGENTS.md 2026-11-11「水池（本地库版）」）：
//  水池的参考实现 infer/ui_water.c 是一台纯软件光栅化 + 逐像素光线追迹渲染器
//  （水面/池壁/圆球三个 pass、焦散投影 pass、128² 波场双页乒乓、320×240 z-buffer）。
//  逐函数直译到灵机解释器的版本（app/waterpool.scm）实测**每帧约 2.2 亿条 IL 指令**，
//  宿主每帧只给 312500 条 → 浏览器里约 12 秒/帧。逐段归因（x86 宿主，稳态）：
//
//      wt_update_caustics   2050 ms/frame  (18%)
//      wt_render_cube       3200 ms/frame  (28%)
//      wt_render_water      5200 ms/frame  (46%)
//      wt_render_sphere      170 ms/frame  ( 1.5%)
//      wt_move_sphere+sim+normals  320 ms  ( 3%)
//      wt_blit(76800×set_pixel) 140 ms/frame (1.2%)
//      wt_compose_camera / HUD    <  2 ms   ( ~0%)
//
//  即：**98% 的代价是「逐像素/逐格点」的数值内层循环**，交互与 HUD 可忽略。因此
//  抽象边界划在「渲染器的全部数值内核 + 它的状态」与「App 生命周期/退出策略」之间：
//  本库持有 WT_State（波场/焦散/网格/z-buffer/相机/圆球/控件状态）并把
//  「输入 → 物理 → 仿真 → 焦散 → 光栅化 → 上屏 → 控件栏」整帧做成**一次本地调用**，
//  Scheme 侧只保留 App 的骨架：装退出库、每拍调一次 (Water.tick)、按返回码走
//  (EX.do_exit)、退出时 (Water.dispose)。
//
//  与参照实现的关系：本库的渲染内核是 infer/ui_water.c 的**逐行副本**（只把设备侧的
//  Global_State/Key_Event 接线换成 Animac 本地库操作数栈 ABI）。ui_water.c 本身一字未改，
//  C 版水池 App（STATE_WATER）行为不变；两侧因此**逐像素一致**（同一份 float32 公式、
//  同一网格、同一剔除/深度/混合次序）。验证方法见 doc/AGENTS.md（water_ref 参照台）。
//
//  栈纪律（遵循上游“编译期单值栈纪律”契约 doc/todo/stack_bal.md §11）：
//    · 值型函数（init/tick/probe）弹完参数后向操作数栈压入**恰好 1 个值**；
//    · 语句型函数（dispose）压入 1 个 #undefined。
//  任何 native 函数都不得压 0 个或 2 个以上值，否则会破坏编译器的栈平衡假设。
//
//  Scheme 侧用法：
//
//      (native Water)
//      (Water.init)                 ; 分配 + 建网格 + 复位默认场景（重复调用会先释放）
//      (Water.tick down x y dt)     ; 一整拍：返回 0 继续 / 1「返回」按钮已松手
//      (Water.probe)                ; 诊断：9 元数值列表（给回归脚本用）
//      (Water.dispose)              ; 释放全部缓冲（退出/重启用）
//
//  宿主须在创建解释器运行时后、加载进程前调用 am_native_water_bind() 绑定目标
//  帧缓冲（与 (native Gfx) 同一块 Nano_GFX）；未绑定时所有函数报运行时错误。
////////////////////////////////////////////////////////////////////////////

#ifndef __AM_NATIVE_WATER_H__
#define __AM_NATIVE_WATER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

#include "graphics.h"

// 由本文件定义
extern const am_native_lib_entry_t am_native_Water_lib;

// 绑定绘图目标帧缓冲（传 NULL 解除绑定）。宿主在解释器进程运行期间必须保证
// gfx 指向有效的、已完成 gfx_init 的 Nano_GFX。
void am_native_water_bind(Nano_GFX *gfx);

// (Water.init) : Number —— 分配缓冲、建网格、复位默认场景；成功 1，失败 0（并报错）
int32_t am_native_Water_init(am_runtime_t *rt, am_process_t *proc);

// (Water.tick down:Number x:Number y:Number dt:Number) : Number
//   —— 推进并绘制一整帧（输入 → 圆球物理 → 波场 → 法线 → 焦散 → 相机 →
//      光栅化 → blit → 控件栏 + 标题 → gfx_refresh）。
//   down 为当前触屏电平（1 按下 / 0 抬起），(x,y) 为当前触屏坐标，dt 为秒。
//   返回 1 表示「返回」控件在本次松手沿被点按（App 应走 (EX.do_exit)），否则 0。
int32_t am_native_Water_tick(am_runtime_t *rt, am_process_t *proc);

// (Water.probe) : List —— 9 元数值列表，供回归脚本断言内部状态：
//   (ready paused useSpherePhysics mode angleX angleY center.x center.y center.z)
int32_t am_native_Water_probe(am_runtime_t *rt, am_process_t *proc);

// (Water.dispose) : #undefined —— 释放全部缓冲并复位状态（幂等）
int32_t am_native_Water_dispose(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif
