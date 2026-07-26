#ifndef __NANO_MIC_H__
#define __NANO_MIC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

// 麦克风HAL：硬件无关的音频采集接口。
// 各平台在对应的 mic_<platform>.cpp 中实现本接口（如 mic_m5core2.cpp）。
// 注意：麦克风与扬声器通常共用同一I2S外设，mic_init/mic_close 负责
// 在其间切换（进入采集时关闭扬声器，结束时恢复）。

// 初始化麦克风（配置I2S采集通道；会关闭扬声器）
int32_t mic_init();

// 读取一帧音频采样（阻塞至数据就绪或超时）
//   buffer  ：输出缓冲（int16 采样，有符号16bit）
//   samples ：期望读取的采样点数
// 返回值：实际读取的采样点数（<0 表示错误）
int32_t mic_read(int16_t *buffer, uint32_t samples);

// 关闭麦克风（释放I2S，并恢复扬声器供按键音等使用）
int32_t mic_close();

#ifdef __cplusplus
}
#endif

#endif
