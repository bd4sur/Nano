#ifndef __AM_NATIVE_MUSIC_H__
#define __AM_NATIVE_MUSIC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Music —— 灵机（Animac）音乐播放引擎本地库
//
//  为什么需要本库：
//    C 版音乐盒（infer/ui_musicbox.c）把「文件枚举 + WAV/MP3 解码 + 随机定位 +
//    向扬声器 HAL 流式喂入」全部写在 GUI 状态机里；灵机（Scheme）侧此前只有
//    Gfx/Touch/Key/Fs 这类"画"与"读"的能力，**没有任何音频通路**，于是音乐盒
//    这类 App 根本写不出来。本库把这些能力下沉成一台**播放引擎**，App 只负责
//    界面与交互（选曲、进度条、音量、播放/暂停/切曲），一行解码也不用写。
//
//  为什么不做成"把 PCM 块交给 Scheme"：
//    一块 4096 采样的 int16 PCM 若以列表形式进解释器堆，每块要造 4096 个堆对象，
//    每秒十块就是四万个对象——内存与 GC 都会崩。因此 PCM **从不跨 FFI 边界**：
//    解码、混单声道、双缓冲、背压喂入全在 C 侧完成（范式与 ui_musicbox.c 的
//    播放器段落一致），Scheme 只拿到"进度/时长/状态"这类标量。
//
//  职责边界（重要，App 侧务必按此协作）：
//    · 本库**不自己装定时器**：喂数据与状态推进都由 App 每拍调 (Music.pump) 驱动，
//      与 (native Touch)/(native Key) 的"App 自己的轮询点"哲学一致。
//    · 本库**不自动切曲**：读到文件尾时 (Music.pump) 返回 1，由 App 决定下一首是谁
//      （列表循环、随机、停住都行）。
//    · 本库**不画任何东西**：界面 100% 在 Scheme 侧。
//
//  解码支持（与 C 版音乐盒一致）：
//    · WAV：RIFF chunk 遍历解析（不假设 44 字节固定头），PCM 8/16 bit、单/双声道；
//    · MP3：vendor/minimp3.h 流式逐帧解码（实现单元 ui_musicbox_mp3.c，本库复用其
//      外部 scratch 注入 minimp3_set_scratch()，**故该编译单元必须一同链接**）。
//    两种格式统一解码为 int16 单声道（立体声取两声道平均）后送 hal_audio_out。
//
//  随机定位（(Music.seek ratio)）：
//    WAV 直接 seek 到 data 区内的采样帧边界；MP3 定位到近似字节偏移后重置解码器、
//    由解码器自行重新同步帧头（与 C 版同一策略：按字节比例近似，足够做进度条拖动）。
//    定位后立即重新预填播放队列，无需 App 干预。
//
//  无音频设备的降级（"静音走带"）：
//    hal_audio_out_init 失败时（打桩平台 / 无 Web Audio 的浏览器 / node 回归台）
//    引擎转入**静音走带**模式：照常解码、按墙钟节奏推进进度、丢弃 PCM。于是
//    进度条、时长、切曲、EOF 这些逻辑在没有声卡的环境里也能被完整验证——
//    端到端回归正是这么跑的。
//
//  WASM 目标的资源按需挂载：
//    music/ 体积大（~90MB），WASM-GUI 把它列为 manifest 的 p4 层，不随启动挂载。
//    (Music.list) 在音乐目录为空时会经 EM_JS 请求页面把该层拉进 MEMFS
//    （页面侧函数 nanoRequestMusicTier，未提供时安全空操作）。因此 App 的首个列表
//    可能为空：应在"列表为空"时按几百毫秒的节奏重试，并提示"正在加载音乐资源"。
//
//  Scheme 接口（全部值型：弹完参数后压入恰好 1 个值并 step 一次，契约 N1/N6）：
//
//    (Music.list)             : List             —— music/ 目录下 .wav/.mp3 的**文件名**
//                                                   （不含路径，按名字升序；空目录返回 '()）
//    (Music.open path)        : Boolean          —— 打开并开始播放。path 遵循 Fs 的路径
//                                                   规则（相对业务资源根 / 以 '/' 开头为绝对
//                                                   路径），故写 "music/xx.mp3" 即可
//    (Music.close)            : Boolean          —— 停止播放、关文件、释放缓冲（幂等）
//    (Music.pump)             : Number           —— 每拍调用：喂队列/推进静音走带
//                                                   0=正常 1=曲目结束(EOF) 2=未打开或已暂停
//                                                   -1=错误（解码/投放失败）
//    (Music.play)             : Boolean          —— 恢复播放（返回"当前是否在播放"）
//    (Music.pause)            : Boolean          —— 暂停（返回"当前是否在播放"，即 #f）
//    (Music.toggle)           : Boolean          —— 播放/暂停切换（返回切换后是否在播放）
//    (Music.is_playing)       : Boolean
//    (Music.seek ratio)       : Boolean          —— 定位到 0.0~1.0（拖动进度条松手时调用）
//    (Music.progress)         : Number           —— 已播放比例 0.0~1.0
//    (Music.elapsed_ms)       : Number
//    (Music.duration_ms)      : Number           —— WAV 精确；MP3 按首帧码率估算（VBR 近似）
//    (Music.info)             : List | #null     —— (采样率 声道数 位深 码率kbps)；
//                                                   未打开返回 #null。位深仅 WAV 有意义
//                                                   （MP3 恒 0），码率仅 MP3 有意义（WAV 恒 0）
//    (Music.format)           : String | #null   —— "WAV" / "MP3"；未打开返回 #null
//    (Music.volume)           : Number           —— 当前播放音量 0~128
//    (Music.set_volume v)     : Number           —— 设置音量（钳到 0~128），返回生效值
//
//  音量域说明：音乐盒内部域为 0~128（与 C 版 MUSICBOX_VOLUME_MAX 一致：音乐内容动态
//  范围大，主音量 255 的增益容易削顶）。首次打开时本库以**主音量**（clamp 到 128）
//  作为初值，之后由 App 经 set_volume 调节，不回写主音量。
//
//  与其它本地库的共享资源（易错点）：
//    · hal_fs 的随机访问 API 是**全局单句柄**：本库从 (Music.open) 到 (Music.close)
//      一直持有它。播放期间 App **不要**再调 (Fs.read)/(Fs.size)/... —— 会把句柄换掉，
//      表现为播放卡住或进度乱跳（列表页读目录 (Fs.list) 也应在播放前完成）。
//    · minimp3 的外部 scratch 也是**进程级单例**（ui_musicbox_mp3.c），本库每次
//      (Music.open) 都会重新注入自己的缓冲，退出时置空。
//
//  Scheme 侧用法（最小骨架）：
//      (native Music)
//      (Music.close)                       ; 进入 App 时清一次残留状态
//      (if (Music.open "music/a.mp3") #f #f)
//      (set! t (System.set_interval 33 (lambda () (Music.pump) ...)))
//      (Music.close)                       ; 退出 App 前务必调用（停机外设）
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Music_lib;

// (Music.list) : List —— music/ 目录下的 .wav/.mp3 文件名（升序）
int32_t am_native_Music_list(am_runtime_t *rt, am_process_t *proc);

// (Music.open path:String) : Boolean —— 打开并开始播放
int32_t am_native_Music_open(am_runtime_t *rt, am_process_t *proc);

// (Music.close) : Boolean —— 停止并释放（幂等）
int32_t am_native_Music_close(am_runtime_t *rt, am_process_t *proc);

// (Music.pump) : Number —— 每拍喂数据（0/1/2/-1）
int32_t am_native_Music_pump(am_runtime_t *rt, am_process_t *proc);

// (Music.play) : Boolean
int32_t am_native_Music_play(am_runtime_t *rt, am_process_t *proc);

// (Music.pause) : Boolean
int32_t am_native_Music_pause(am_runtime_t *rt, am_process_t *proc);

// (Music.toggle) : Boolean
int32_t am_native_Music_toggle(am_runtime_t *rt, am_process_t *proc);

// (Music.is_playing) : Boolean
int32_t am_native_Music_is_playing(am_runtime_t *rt, am_process_t *proc);

// (Music.seek ratio:Number) : Boolean
int32_t am_native_Music_seek(am_runtime_t *rt, am_process_t *proc);

// (Music.progress) : Number
int32_t am_native_Music_progress(am_runtime_t *rt, am_process_t *proc);

// (Music.elapsed_ms) : Number
int32_t am_native_Music_elapsed_ms(am_runtime_t *rt, am_process_t *proc);

// (Music.duration_ms) : Number
int32_t am_native_Music_duration_ms(am_runtime_t *rt, am_process_t *proc);

// (Music.info) : List | #null
int32_t am_native_Music_info(am_runtime_t *rt, am_process_t *proc);

// (Music.format) : String | #null
int32_t am_native_Music_format(am_runtime_t *rt, am_process_t *proc);

// (Music.volume) : Number
int32_t am_native_Music_volume(am_runtime_t *rt, am_process_t *proc);

// (Music.set_volume v:Number) : Number
int32_t am_native_Music_set_volume(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif
