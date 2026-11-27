////////////////////////////////////////////////////////////////////////////
//  Native Library : Touch —— 灵机（Animac）触屏本地库
//
//  为什么需要本库：
//    GUI 层的触屏消费走 ui_app.c 的 get_input_event（电平 + 边沿），
//    而"灵机应用程序"（Scheme App）需要【自己的一段触屏输入】——画笔、拖拽、
//    虚拟摇杆等。此前 Scheme 侧只能看 Gfx（画布）与 Time/Diag（时钟/自观测），
//    完全没有输入通道，交互式 App 无法编写。本库把 HAL 触屏（hal_touch.h）以
//    只读方式暴露给 Scheme，使 App 可以逐帧取触点，并在 Scheme 内自行解释手势。
//
//  设计要点：
//    1) 只读、旁路：不消费、不改变 HAL 状态，宿主 GUI 的输入管线完全不受影响
//       （与 ui_app.c 各读各的同一份电平，互不干扰）。
//    2) 电平模型：与 get_input_event 同一哲学——每次调用采样一次 HAL 电平，
//       边沿判定由调用方（或本库的 take_press/take_release 锁存）完成。
//       HAL 只给电平不给事件，因此**同一帧内的按下+松开（亚帧点按）可能被漏检**；
//       人手点按远长于一帧（~16ms），实际使用无影响。
//    3) 坐标缓存：hal_touch.h 约定"未按下时不保证坐标有效"，故本库缓存最后一次
//       按下期间的有效坐标，松开后 x/y 仍返回该值（调试与"最后落点"语义都靠它）。
//    4) 触屏不可用（无触屏硬件 / 端口打开失败）时：is_available 返回 #f，
//       is_pressed 恒 #f，x/y 恒 0，take_* 恒 #f——不报错，App 可据此降级。
//    5) **例外：滚轮。** Touch.wheel/wheel_enable 不是触屏电平（滚轮没有“触点”），
//       而是宿主转交的帧累加像素增量；开启后页面不再把滚轮合成为手指，
//       详见下方 wheel_enable 的条目注释与 hal_touch.h。
//
//  Scheme 侧用法：源码中声明 (native Touch) 后以 Touch.函数名 调用。
//
//  栈纪律（上游本地库契约 N1/N2，见解释器文档 doc/todo/stack_bal.md §11）：
//  所有函数都是值型——弹完参数后向操作数栈压入恰好 1 个值（本库全部零参数，
//  故净效应恒为 +1），随后调用一次 am_process_step。
////////////////////////////////////////////////////////////////////////////

#ifndef __AM_NATIVE_TOUCH_H__
#define __AM_NATIVE_TOUCH_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

// 由 infer/am_native_Touch.c 定义
extern const am_native_lib_entry_t am_native_Touch_lib;

// (Touch.is_available) : Boolean —— 宿主是否提供可用的触屏（HAL 探测成功）
int32_t am_native_Touch_is_available(am_runtime_t *rt, am_process_t *proc);

// (Touch.is_pressed) : Boolean —— 当前是否正在触摸
int32_t am_native_Touch_is_pressed(am_runtime_t *rt, am_process_t *proc);

// (Touch.x) : Number —— 触点 x 像素坐标（未触摸时为最后一次有效值；从未触摸为 0）
int32_t am_native_Touch_x(am_runtime_t *rt, am_process_t *proc);

// (Touch.y) : Number —— 触点 y 像素坐标（同上）
int32_t am_native_Touch_y(am_runtime_t *rt, am_process_t *proc);

// (Touch.take_press) : Boolean —— 自上次调用以来是否出现过"按下"沿（读后清零）
int32_t am_native_Touch_take_press(am_runtime_t *rt, am_process_t *proc);

// (Touch.take_release) : Boolean —— 自上次调用以来是否出现过"松开"沿（读后清零）
int32_t am_native_Touch_take_release(am_runtime_t *rt, am_process_t *proc);

// (Touch.wheel_available) : Boolean —— 宿主是否提供原生滚轮通道（当前仅 WASM GUI）
int32_t am_native_Touch_wheel_available(am_runtime_t *rt, am_process_t *proc);

// (Touch.wheel_enable) : Boolean —— 声明"本 App 自己消费滚轮"，返回通道是否可用
//
// 为何需要一条专门的滚轮通道（而不是把滚轮当手指）：
//   宿主 GUI 的滚动控件直接吃 key_event->wheel_dy（按像素直接落位，见 ui_wheel_scroll_clamp），
//   而 Scheme App 只能看触屏电平。页面过去对 App 运行态把滚轮**合成为一根手指**，
//   那条路有三个结构性缺陷，在长文档上表现为"滚不动 / 一抖一抖 / 偶尔跳一下"：
//     1) 合成手指走不出画布：一格 28px，中心起手只有 119px 行程 → 每 4 格就要松手重按；
//     2) 重按与上一段的松手落在同一采样间隔里（App 的拍子是几十 ms），App 看不到
//        松开沿，只看到触点从画布顶部"瞬移"回中心 —— 一段凭空出现的反向位移；
//     3) 起手点的位移（中心 → 首格位置）发生在同一拍内，App 的 press 沿只能拿到
//        末端坐标，第一格的位移被整段丢掉。
//   启用本通道后：页面不再合成手指，滚轮走与 GUI 相同的原生通道（逐帧平滑、
//   不受画布高度限制、与触摸状态机完全解耦），App 用 (Touch.wheel) 取像素增量。
//   幂等：重复调用无副作用（宿主切换实例后会清掉认领，App 应在每拍重新声明）。
int32_t am_native_Touch_wheel_enable(am_runtime_t *rt, am_process_t *proc);

// (Touch.wheel) : Number —— 取走自上次调用以来累计的滚轮增量（帧缓冲像素，
//   正 = 内容上滚，与 (Key)/触屏拖动无任何关系）；读后清零。未 (wheel_enable)
//   过时恒为 0（滚轮仍走合成手指那条旧路）。
int32_t am_native_Touch_wheel(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 宿主服务接口（Scheme 侧不可见）
// ---------------------------------------------------------------------------

// 丢弃一段**在途手势**：置位后直到采样到"未按下"为止，本库一律按"没按"上报
// （is_pressed 恒 #f，两个沿都不产生），触点坐标停在最后一次有效值上。
//
// 为什么需要（宿主切换 App 时的经典误触）：
//   拉起 / 返回 / 恢复实例都发生在用户**手指还按着**的时刻（点列表项、点退出确认框
//   的「确认」）。此时前台一换，新 App 的第一拍就会看到 is_pressed=1 而它自己的
//   "上一拍电平"是 0 —— 派生出一个凭空出现的按下沿，紧接着的松手就成了一次点击
//   （实测症状：确认退出后 filemgr 立刻打开了一个文件）。这与应用列表的
//   s_list_touch_guard 是同一类问题的两个面：那边守 Key_Event，这边守 (native Touch)。
//   语义上等价于"这一下不属于你，等用户松手后的下一次按下才算"。
//
// 兜底：若 HAL 一直报按下（触摸卡住），3 秒（墙钟）后自动解除，
// 不会把 App 的输入永久冻住。
void am_native_Touch_drop_inflight(void);

// 宿主服务：把本帧的滚轮增量转交本库（Scheme 侧由 (Touch.wheel) 取走）。
// ui_scheme_app.c 每帧从 key_event->wheel_dy 转一次（那里已经是 get_input_event
// 从 HAL 取走的帧累加值）；非 WASM GUI 平台恒为 0，照转无害。
void am_native_Touch_set_wheel(int32_t dy);

// 宿主服务：清空滚轮暂存并交还原生通道的认领（新实例冷启动时调，
// 避免上一个 App 的认领/积压带到下一个 App）。
void am_native_Touch_reset_wheel(void);

#ifdef __cplusplus
}
#endif

#endif