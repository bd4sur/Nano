#ifndef __AM_NATIVE_BIRD_H__
#define __AM_NATIVE_BIRD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Bird —— 灵机（Animac）鸟鸣合成器本地库
//
//  为什么需要本库：
//    本 App（app/bird.scm「百鸟园」）要做的是**实时合成**鸟叫，不是播放录音。合成
//    内核是一个二阶非线性 ODE（Mindlin–Laje 低维鸣管模型）：
//
//        dx/dt = y
//        dy/dt = -αγ² - βγ²x - γ²x³ + γ²x² - γx²y - γxy
//
//    每个采样点要积分 20~45 步，外加按 (α,β) 查双线性标定表反解 β、二阶带通声道
//    滤波、包络/颤音/调幅、双声门与谐波支路。实测单声部 ≈ 2×10³ flops/采样
//    （44.1kHz 下 4 声部量级 10⁸ flops/s）——比解释器在 wasm32 上的能力高三个数量级以上
//    （对照 (native Audio) 里 1024 点 FFT 的估算），因此**整条合成流水线必须在 C 侧**，
//    Scheme 只拿"音高/包络/事件"这类标量去画界面。这与 (native Music) 把解码放在 C 侧、
//    (native Audio) 把 FFT 放在 C 侧是同一条职责边界。
//
//  算法出处（只参照算法，数据自己造）：
//    参照 https://github.com/emwstudio/esp32-perch 的鸟叫合成算法（该项目把
//    sha5b/M5Stack-Lyrebird 的 AudioWorklet 合成器移植到 ESP32-S3）：
//      · Mindlin–Laje 鸣管振荡器（上式的半隐式欧拉积分）；
//      · 离线"标定表"把 (α,β) 映射到基频与峰峰振幅——音高与响度解耦的关键；
//      · 音高轮廓（contour）+ 起振/保持/衰减包络 + 颤音(vibrato) + 调幅(AM)
//        + 按音高跟踪的声道带通(tract) + 失谐第二声门(two) + 倍频谐波支路；
//      · 多个"个体"（voicing：音高/时长/响度/AM 的微差）组成合唱与应答。
//    **不搬运**它的数据：它携带 2423 种鸟的 12724 个音节与 CC BY-NC-SA 的标定表；
//    本库的标定表由 infer/tools/gen_bird_calib.py 自己测（见 bird_calib.h 文件头），
//    鸟种/曲目是 12 种常见鸟的手工编写参数（见下方"鸟种数据"）。
//
//  与参考实现的工程差异（都是为"在解释器 App 里跑"服务的）：
//    1. **固定步长比**：每个声部按 gamma 取整"每采样积分步数"，使 γ·dt≈0.05 恒定，
//       于是同一张标定表对所有音高都成立（参考实现按阈值取步数，音高与响度会随
//       步数漂移，靠额外增益补偿）。
//    2. **无量纲表**：表里存 phi = f0/γ 而不是 Hz，省掉参考实现里的 gammaK 换算。
//    3. **节拍在 C 侧**：乐句的推进按**采样计数**做（不是按 UI 帧），所以颤音般的
//       短促乐句不会因为 App 主循环慢而走音；App 每拍只调 (Bird.pump) 把音频喂出去。
//    4. **无声也能跑**：没有音频设备时（打桩平台 / 浏览器自动播放被挂起）转入
//       "静音走带"，按墙钟照常合成并丢弃采样——界面上的星图与电平表照样有内容，
//       与 (native Music) 的静音走带同一策略。
//
//  Scheme 接口（全部值型：弹完参数后压入恰好 1 个值并 step 一次，契约 N1/N6）：
//
//    (Bird.count)          : Number          —— 鸟种数
//    (Bird.name i)         : String|#null    —— 中文名（i 越界 → #null）
//    (Bird.latin i)        : String|#null    —— 拉丁学名
//    (Bird.mnemonic i)     : String|#null    —— 拟声词（如"叽叽—嘿"）
//    (Bird.voicing i)      : List|#null      —— (音节数 曲目数 最低音 最高音 Hz)
//    (Bird.select i)       : Number          —— 选种，返回生效序号（越界保持原样）
//    (Bird.selected)       : Number          —— 当前鸟种序号
//    (Bird.mode)           : Number          —— 0=独唱 1=应答 2=齐鸣
//    (Bird.set_mode m)     : Number          —— 设模式（返回生效值）
//    (Bird.play)           : Boolean         —— 开始鸣唱（幂等；已播放返回 #t）
//    (Bird.stop)           : Boolean         —— 停止并清空所有声部
//    (Bird.is_playing)     : Boolean
//    (Bird.pause)          : Boolean         —— 暂停（返回 #t）；(Bird.resume) 恢复
//    (Bird.pump)           : Number          —— **每拍调用**：合成并喂音频，
//                                               返回本拍送入设备的采样数（0=队列满/未播放）
//    (Bird.chirp)          : Boolean         —— 立刻叫一声（触屏点按试听，独立于播放）
//    (Bird.voices)         : Number          —— 当前在发声的声部数（0~8）
//    (Bird.vf0 i)          : Number          —— 第 i 个声部的当前基频（Hz，0=空闲）
//    (Bird.venv i)         : Number          —— 第 i 个声部的包络（0~100）
//    (Bird.vtag i)         : Number          —— 第 i 个声部的个体标签（0~3，配色用）
//    (Bird.level)          : Number          —— 峰值电平 0~255（衰减峰值，电平表用）
//    (Bird.pitch)          : Number          —— 音高缩放百分数（-50~50）
//    (Bird.set_pitch pct)  : Number          —— 微调音高（模拟器旋钮），返回生效值
//    (Bird.tempo)          : Number          —— 速度缩放百分数（-50~50）
//    (Bird.set_tempo pct)  : Number          —— 微调语速，返回生效值
//    (Bird.volume)         : Number          —— 音量 0~128
//    (Bird.set_volume v)   : Number          —— 设音量（钳到 0~128），返回生效值
//    (Bird.rate)           : Number          —— 合成采样率（固定 22050）
//
//  全局单例：合成引擎是进程级 static（同 (native Music)/(native Audio) 形态）。
//  同一时刻只有一个 App 能用它；App 退出前务必 (Bird.stop)（EX.on_exit），否则
//  扬声器设备不会释放（下一次进入 App 时 play 会重新 init，行为不变但设备不优雅）。
//
//  Scheme 侧用法（最小骨架）：
//      (native Bird)
//      (Bird.select 0)
//      (Bird.play)
//      (set! t (System.set_interval 16 (lambda () (Bird.pump) (draw))))
//      (Bird.stop)                       ; 退出前
////////////////////////////////////////////////////////////////////////////

// 输出采样率（固定；hal_audio_out 会按需重采样）
#define AM_NATIVE_BIRD_RATE 22050

// 由本文件定义
extern const am_native_lib_entry_t am_native_Bird_lib;

int32_t am_native_Bird_count(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_name(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_latin(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_mnemonic(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_voicing(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_select(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_selected(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_mode(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_set_mode(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_play(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_stop(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_is_playing(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_pause(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_resume(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_pump(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_chirp(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_voices(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_vf0(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_venv(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_vtag(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_level(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_pitch(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_set_pitch(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_tempo(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_set_tempo(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_volume(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_set_volume(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Bird_rate(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif
