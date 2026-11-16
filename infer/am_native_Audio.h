#ifndef __AM_NATIVE_AUDIO_H__
#define __AM_NATIVE_AUDIO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Audio —— 灵机（Animac）音频**输入**（采集 + 频谱分析）本地库
//
//  为什么需要本库：
//    C 版频谱仪（infer/ui_spectrogram.c）把「开麦 → 读一帧 → 加窗 → FFT →
//    对数幅度映射 → 逐像素上屏」全写在 GUI 状态机里；灵机（Scheme）侧此前只有
//    Gfx/Touch/Key/Fs/Music（后者只出不进），**没有任何音频输入通路**，于是
//    声谱图、示波器、音高计这类 App 一个都写不出来。本库把 HAL 麦克风
//    （hal_audio_in.h）与频谱分析下沉成一台**分析引擎**，App 只负责界面与交互。
//
//  为什么 FFT 不放在 Scheme 侧（与 (native Music) 把解码放在 C 侧同一理由）：
//    1024 点基 2 FFT 是 5120 个蝶形运算，每个蝶形含复乘 + 两次加减 + 旋转因子
//    访存。在灵机解释器里实测约 3×10⁵ VM 指令/帧量级，@44.1kHz 每 23ms 就要出
//    一列（43 列/秒），即 >10⁷ VM 指令/秒——比解释器在 wasm32 上的能力高一个
//    量级，声谱图会掉到个位数 fps。因此时域→频域整条流水线留在 C 侧（复用
//    nano_fft.c，与 C 版频谱仪**同一份**旋转因子/加窗/映射代码），跨 FFI 边界
//    只传标量（每频点一个 0~255 强度、RMS、原始采样按点取用），**从不把整块
//    PCM/频谱列表塞进解释器堆**（1024 采样的列表 = 每帧上千个堆对象）。
//
//  与 C 版频谱仪逐项对齐（ui_spectrogram.c）：
//    · FFT 点数 1024、Hann 窗（周期型 w[n]=0.5(1-cos(2πn/N))）、基 2 迭代 FFT；
//    · 幅度映射 v = (log10(mag+1)·80 − 基准) · 255 / 动态范围，钳到 0~255；
//    · 频点从 bin 1 起（跳过直流），可绘制频点数 = 511（= fft/2 − 1）；
//    · 基准电平/动态范围的默认值、步长与上下限由 **App** 按 C 版策略施加
//      （本库只做宽松钳位：基准 0~1000、动态范围 1~4096），库保持通用。
//
//  采样率：由 (Audio.open rate) 指定（声谱图用 44100，与 C 版一致）。WASM 目标上
//    hal_audio_in_wasm.c 以 `new AudioContext({sampleRate})` 建立采集上下文，浏览器
//    负责把设备流重采样到该速率；个别浏览器拒绝指定速率时会退让为设备默认速率
//    （C 侧无从得知，频率轴会整体偏移——已知限制，见 hal_audio_in_wasm.c 注释）。
//
//  Scheme 接口（全部值型：弹完参数后压入恰好 1 个值并 step 一次，契约 N1/N6）：
//
//    (Audio.is_available)     : Boolean        —— 宿主是否有可用麦克风（探测一次后缓存）
//    (Audio.open rate)        : Boolean        —— 打开麦克风并准备分析缓冲；重复调用先关
//    (Audio.close)            : Boolean        —— 关麦、释放分析缓冲（幂等）
//    (Audio.info)             : List           —— (采样率 FFT点数 可绘制频点数)；未打开也返回
//    (Audio.read)             : Number         —— 采集一列：1=有新数据 0=数据不足 −1=硬错误
//    (Audio.bin i)            : Number         —— 最新一列第 i 个频点的强度 0~255（越界取 0）
//    (Audio.sample k)         : Number         —— 最新一帧第 k 个原始采样（int16 值；越界取 0）
//    (Audio.rms)              : Number         —— 最新一帧的时域电平 0~255（−60dBFS…0dBFS）
//    (Audio.level)            : List           —— 当前的 (基准 动态范围)
//    (Audio.set_level b r)    : List           —— 设置基准/动态范围，返回生效后的 (基准 动态范围)
//
//  软失败约定（manual §1.1.1）：没有麦克风、未 (Audio.open)、数据还没到位时一律
//  不报错——is_available 为 #f、read 为 0、bin/sample/rms 为 0，App 据此降级。
//
//  (Audio.read) 的取数语义：**一次调用只取一帧**（1024 采样 ≈23ms），返回 1 表示
//  本列数据已就绪；不足一帧时返回 0 且不消费任何采样（WASM HAL 非阻塞，ALSA 则
//  阻塞至就绪或 100ms 超时）。
//  为什么不在库内"多取只留最新"：声谱图要把**每一帧**都画成一列（丢帧等于丢信息，
//  短促的声音会被整段跳过）。要跟上音频，App 每拍循环调用若干次即可（每拍上限
//  见 app/spectrogram.scm 的 COL_MAX）——浏览器里 AudioWorklet 每 2048 采样投递一次，
//  于是典型情况是"一拍 0 次 / 一拍 2 次"交替，恰好把 43 列/秒全部画出来。
//
//  全局单例：分析引擎是进程级 static（与 (native Music) 的播放引擎同一形态），
//  同一时刻只有一个 App 能用麦克风；App 退出前务必 (Audio.close)（EX.on_exit）。
//
//  Scheme 侧用法（最小骨架）：
//      (native Audio)
//      (Audio.close)                        ; 进入 App 时清一次残留状态
//      (if (Audio.open 44100) #f #f)
//      (set! t (System.set_interval 16 (lambda ()
//                (if (== (Audio.read) 1) (draw_column) #f))))
//      (Audio.close)                        ; 退出 App 前必须调用
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Audio_lib;

// (Audio.is_available) : Boolean —— 麦克风可用性（未探测过则尝试打开一次再关闭）
int32_t am_native_Audio_is_available(am_runtime_t *rt, am_process_t *proc);

// (Audio.open rate:Number) : Boolean
int32_t am_native_Audio_open(am_runtime_t *rt, am_process_t *proc);

// (Audio.close) : Boolean
int32_t am_native_Audio_close(am_runtime_t *rt, am_process_t *proc);

// (Audio.info) : List —— (采样率 FFT点数 可绘制频点数)
int32_t am_native_Audio_info(am_runtime_t *rt, am_process_t *proc);

// (Audio.read) : Number —— 1 / 0 / -1
int32_t am_native_Audio_read(am_runtime_t *rt, am_process_t *proc);

// (Audio.bin index:Number) : Number
int32_t am_native_Audio_bin(am_runtime_t *rt, am_process_t *proc);

// (Audio.sample index:Number) : Number
int32_t am_native_Audio_sample(am_runtime_t *rt, am_process_t *proc);

// (Audio.rms) : Number
int32_t am_native_Audio_rms(am_runtime_t *rt, am_process_t *proc);

// (Audio.level) : List —— (基准 动态范围)
int32_t am_native_Audio_level(am_runtime_t *rt, am_process_t *proc);

// (Audio.set_level base:Number range:Number) : List
int32_t am_native_Audio_set_level(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif
