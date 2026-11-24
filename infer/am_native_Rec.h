#ifndef __AM_NATIVE_REC_H__
#define __AM_NATIVE_REC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Rec —— 灵机（Animac）**录音机**本地库（采集 + MP3 编码 + 落盘）
//
//  为什么需要本库：
//    (native Audio) 只把麦克风采样交付给 Scheme（频谱分析用），没有任何"存下来"
//    的通路；而 (native Fs) 刻意只做**文本**文件的整读整写（见 am_native_Fs.h），
//    MP3 是二进制码流，用 Fs 存不了（UTF-8 解码会把码流改坏）。于是"录音 + 压缩 +
//    存盘"整条链路只能下沉到 C 侧——与 (native Music) 把解码放在 C 侧同理。
//
//  编码器：Aqua MP3 编码器（vendor/aqua_mp3/，从 /home/bd4sur/Aqua 的 aqua-c/src
//    原样拷贝，只取编码端，未做任何算法改动）。格式固定为
//      MPEG-1 Layer III · 44100 Hz · **单声道** · **320 kbps CBR**
//    （320kbps 是 MPEG-1 允许的最高码率；一帧 1152 采样 → 1044/1045 字节，
//      ≈ 26.12ms 音频 / 40KB·s⁻¹）。采样率与声道数是常量而非参数：CBR 320 +
//      单声道是需求给定的目标格式，暴露可调参数只会让"能不能播、多大"变得不确定。
//
//  实时性（这是本库设计的核心，也是"为什么不用 Fs 攒完再写"的原因）：
//    编码一帧在 wasm32（浏览器）上实测约 1ms（node/V8，x86 宿主；见
//    doc/AGENTS.md），而一帧代表 26ms 音频——编码本身有 >20× 的余量。
//    (Rec.pump) 每次调用**只做一小块**：读一帧 → 编码 → 立即追加写入文件
//    （fwrite + 库缓冲，每 8 帧 ≈209ms fflush 一次），因此录音过程中数据一直在落盘，
//    不需要把整段 PCM 或整段 MP3 攒在内存里（1 分钟 320kbps = 2.4MB，攒着不放对 MCU
//    是灾难）；进程被强杀时最多丢最后不足 0.2 秒。
//    宿主慢于音频时 pump 返回 0（数据还没到），App 下一拍再来——不会丢样、不会
//    阻塞 UI（WASM 的 mic_read 非阻塞，见 hal_audio_in_wasm.c 的语义约定）。
//
//  文件落点（重要）：
//    · 存到业务资源根的 `music/` 目录，文件名 `rec-YYYYMMDD-HHMMSS.mp3`（本地时间；
//      重名时追加 -1/-2…）。放 music/ 是为了**音乐盒直接可见可播放**（(Music.list)
//      枚举的正是 music/ 下的 .wav/.mp3）。
//    · 目录不存在时自动创建（platform_mkdir）。
//    · 收尾（(Rec.stop)）时若 WASM 目标链入了 hal_fs_persist_wasm.c，则整份文件经
//      am_native_fs_on_write 弱钩子写进 nano-store（IndexedDB），刷新页面后录音
//      仍由 loader.js 挂载回来——与 (Fs.write) 的持久化通路是同一条。
//      （弱符号：其它平台不链接该文件，零开销。）
//
//  全局单例：编码器状态、输出文件、跨帧 PCM 都是进程级 static（与 (native Audio) /
//    (native Music) 同一形态）。同一时刻只有一个录音在进行；换 App 前必须
//    (Rec.stop)（App 在 EX.on_exit 里做），否则文件不会收尾（fclose 前的字节仍在
//    库缓冲里，且空文件不会被清理）。
//
//  Scheme 接口（全部值型：弹完参数后压入恰好 1 个值并 step 一次，契约 N1/N6）：
//
//    (Rec.info)        : List      —— (采样率 声道数 码率kbps 帧采样数)；
//                                    恒为 (44100 1 320 1152)，供 App 显示/自检
//    (Rec.start)       : String|#null
//                                  —— 开麦 + 新建文件并开始录音。返回**资源根相对路径**
//                                    （如 "music/rec-20261124-121530.mp3"）；已在录音中、
//                                    或麦克风打不开（无 getUserMedia / 无 ALSA 设备）时
//                                    返回 #null（软失败，App 据此显示"麦克风不可用"）。
//    (Rec.pump n)      : Number    —— 推进至多 n 帧（n 钳到 1..8）：返回**本拍编成的帧数**
//                                    （>=0）；返回 -1 表示麦克风硬错误（权限被拒/采集中断），
//                                    库已自行收尾（关麦 + 关文件 + 丢弃空文件），App 只需
//                                    切回"未录音"画面。返回 0 = 数据还没到（不丢样）。
//    (Rec.stop)        : List      —— 收尾并落盘，返回 (毫秒 帧数 字节数 路径)；
//                                    路径为相对路径字符串（一帧都没编成时为 #null，
//                                    此时空文件已被删除）。未在录音时返回 (0 0 0 #null)。
//    (Rec.stats)       : List      —— (状态毫秒 帧数 字节数)；状态 1=录音中 0=空闲
//                                    （毫秒 = 帧数 × 1152 × 1000 / 44100，按实际编成的
//                                    帧数计，是**音频时长**而不是墙钟时长）
//    (Rec.level)       : Number    —— 最近一帧的时域电平 0~255（−60dBFS…0dBFS，与
//                                    (Audio.rms) 同一映射），供电平表用；无数据时为 0
//    (Rec.wave k)      : Number    —— 最近一帧的第 k 个原始采样（int16，越界取 0），供波形用
//
//  软失败约定（manual §1.1.1）：没有麦克风、尚未开始、数据没到位时一律不报错，
//  以 #null / 0 表示，App 据此降级。
////////////////////////////////////////////////////////////////////////////

// 由 infer/am_native_Rec.c 定义
extern const am_native_lib_entry_t am_native_Rec_lib;

// (Rec.info) : List —— (采样率 声道数 码率kbps 帧采样数)
int32_t am_native_Rec_info(am_runtime_t *rt, am_process_t *proc);

// (Rec.start) : String | #null —— 开始录音；返回资源根相对路径
int32_t am_native_Rec_start(am_runtime_t *rt, am_process_t *proc);

// (Rec.pump n:Number) : Number —— 推进至多 n 帧；返回本拍帧数，-1 = 硬错误（已收尾）
int32_t am_native_Rec_pump(am_runtime_t *rt, am_process_t *proc);

// (Rec.stop) : List —— 收尾落盘 → (毫秒 帧数 字节数 路径)
int32_t am_native_Rec_stop(am_runtime_t *rt, am_process_t *proc);

// (Rec.stats) : List —— (状态 毫秒 帧数 字节数)
int32_t am_native_Rec_stats(am_runtime_t *rt, am_process_t *proc);

// (Rec.level) : Number —— 最近一帧电平 0~255
int32_t am_native_Rec_level(am_runtime_t *rt, am_process_t *proc);

// (Rec.wave k:Number) : Number —— 最近一帧第 k 个采样
int32_t am_native_Rec_wave(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif
