#ifndef __NANO_UI_SPECTROGRAM_H__
#define __NANO_UI_SPECTROGRAM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 音频频谱仪（STFT声谱图，从下往上滚动显示）
//
// 原理：从内置麦克风读取一帧音频（1024采样@44.1kHz），加 Hann 窗后做 1024 点
// 实数 FFT（自实现基2迭代FFT，无外部库依赖），取前 320 个频点的幅度（对数刻度），
// 按热力调色板映射为颜色，作为一行绘制到屏幕底部；每帧将帧缓冲整体上移，
// 形成自下而上滚动的声谱图（瀑布图）。
// ===============================================================================

// 初始化频谱仪（构建窗函数/twiddle表/调色板，并初始化麦克风）
int32_t ui_spectrogram_init(Key_Event *key_event, Global_State *global_state);

// 采集一帧音频并渲染一帧声谱图（滚动+新行+刷新）
int32_t ui_spectrogram_render_frame(Key_Event *key_event, Global_State *global_state);

// 退出频谱仪（关闭麦克风并恢复扬声器）
int32_t ui_spectrogram_deinit(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
