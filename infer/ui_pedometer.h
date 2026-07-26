#ifndef __NANO_UI_PEDOMETER_H__
#define __NANO_UI_PEDOMETER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 计步器（时域峰值计数为主，频域周期性校验）
//
// 算法流水线（时域宽条件候选点 + 频域频谱平坦度交叉验证）：
//   1. 100Hz 三轴加速度 → 合成幅值 → DC-blocker 高通去重力 → EMA 低通平滑；
//   2. 时域候选点：超过自适应阈值（滑动2s窗 均值+0.3×标准差）且满足最小间隔
//      250ms 即记为步伐候选点（宽条件，不做过多筛选）；
//   3. 频域分析：每1s对最近2.56s（256点）去直流信号做FFT，以频谱平坦度
//      （SFM=几何均值/算术均值）区分无规律抖动与节奏明确的周期性运动，
//      并提取主频；候选速率与主频在相对容差内一致则确认候选点计入总步数，
//      否则丢弃；
//   4. 显示：总步数、步频、状态（静止/行走/未确认）、实时波形与峰值标记。
// ===============================================================================

int32_t ui_pedometer_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_pedometer_render_frame(Key_Event *key_event, Global_State *global_state);
int32_t ui_pedometer_deinit(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
