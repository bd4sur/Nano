// hal_audio_out_wasm.c —— Nano-WASM-GUI 扬声器 HAL（Web Audio 实现）
//
// 通路：C 侧 int16 单声道 PCM 块 ──EM_JS──▶ AudioBuffer(1ch, n, srcRate)
//        ──AudioBufferSourceNode.start(nextTime)──▶ GainNode ──▶ destination
//
// 设计要点（与各平台 HAL 的语义约定对照）：
//
//   1. 「播放队列」不落在数组里，而是落在 **AudioContext 的时间轴**上：
//        audio_out_queue_free() = (nextTime - ctx.currentTime) < LOOKAHEAD
//      稳态下每接受一块就推进 n/rate 秒（音乐盒 4096@44.1k≈93ms；OFDM 一帧
//      31680@48k≈660ms），于是生产速率自动等于消费速率——既不饥饿，也不会
//      因为 UI 主循环跑 60fps 而把音频灌成快放。这一点是打桩版之外最容易写错
//      的地方（`queue_free` 恒真 = 5.6 倍速播放）。
//
//   2. enqueue 是**拷贝语义**（与 hal_audio_out_alsa_linux.c 同）：调用方仍按
//      乒乓双缓冲轮换，但块内容在 enqueue 内即被复制进 AudioBuffer，故其
//      「须保持有效至播完」的纪律在本平台是安全超集。
//
//   3. 采样率交给浏览器：AudioBuffer 按源采样率建立，AudioBufferSourceNode
//      自动重采样到 AudioContext 的实际速率；两者相等时不做重采样（直通）。
//      因此 C 侧「切歌先 audio_out_close() 再 audio_out_init(新采样率)」的既有
//      逻辑无需任何改动即可播 8k~96k 的任意 WAV/MP3，不必像 ALSA 那样协商设备。
//
//   4. 无 SharedArrayBuffer / AudioWorklet / COOP-COEP 依赖：全部在主线程调度，
//      保住本项目「单线程、任意静态 HTTP 服务可托管」的既有部署形态。
//
//   5. 页面未提供 JS 胶水（node 端回归台、probe）时所有接口安全失败，行为与
//      原打桩版一致（init=-1 / queue_free=0 / enqueue=-1），各调用点的降级路径不变。
//
//   6. 已知边界：音频喂入由渲染循环驱动（每帧最多一块），浏览器切到后台时
//      rAF 暂停 → 前瞻窗口（LOOKAHEAD≈0.3s）耗尽后停声。这是「用主循环喂音频」
//      的固有代价，与 ESP32 上由 RTOS 任务喂 I2S 的形态不同；后台播放需
//      AudioWorklet+SharedArrayBuffer 或整曲预调度，不在本实现范围内。

#include "hal_audio_out.h"

#include "platform.h"

#include <emscripten.h>

// ---- 页面上屏/播放胶水（见 webgui/index.html 的「音频子系统」一节） ----
// 统一用 typeof 守卫：无 JS 胶水的环境（node 回归台）返回失败而不是抛异常。

EM_JS(int32_t, nano_ao_init_js, (int32_t rate, int32_t volume), {
    if (typeof nanoAudioOutInit !== "function") return -1;
    try { return nanoAudioOutInit(rate, volume) ? 0 : -1; } catch (e) { return -1; }
});

EM_JS(int32_t, nano_ao_free_js, (void), {
    if (typeof nanoAudioOutFree !== "function") return 0;
    try { return nanoAudioOutFree() ? 1 : 0; } catch (e) { return 0; }
});

EM_JS(int32_t, nano_ao_push_js, (const int16_t *pcm, int32_t samples, int32_t rate), {
    if (typeof nanoAudioOutPush !== "function") return -1;
    try { return nanoAudioOutPush(pcm, samples, rate) ? 0 : -1; } catch (e) { return -1; }
});

EM_JS(void, nano_ao_stop_js, (void), {
    if (typeof nanoAudioOutStop !== "function") return;
    try { nanoAudioOutStop(); } catch (e) { /* 忽略 */ }
});

EM_JS(void, nano_ao_volume_js, (int32_t volume), {
    if (typeof nanoAudioOutSetVolume !== "function") return;
    try { nanoAudioOutSetVolume(volume); } catch (e) { /* 忽略 */ }
});

// ---- 状态 ----

static uint8_t  s_volume        = 16;  // 当前播放音量（init/set_volume 设置）
static uint8_t  s_prev_volume   = 16;  // init 时保存，close 时恢复（对齐 ESP32/ALSA 语义）
static uint8_t  s_master_volume = 16;  // 系统主音量（系统设置读写）
static uint32_t s_src_rate      = 44100;
static int32_t  s_inited        = 0;

int32_t audio_out_init(uint32_t sample_rate, uint8_t volume) {
    // 允许重复 init（音乐盒切歌时采样率可能变化）：先停掉已排队的源
    if (s_inited) nano_ao_stop_js();

    s_prev_volume = s_volume;   // 保存"进入本次播放前的音量"，供 close 恢复
    s_volume = volume;
    s_src_rate = sample_rate;
    s_inited = (nano_ao_init_js((int32_t)sample_rate, (int32_t)volume) == 0) ? 1 : 0;
    return s_inited ? 0 : -1;
}

int32_t audio_out_queue_free(void) {
    if (!s_inited) return 0;
    return nano_ao_free_js() ? 1 : 0;
}

int32_t audio_out_enqueue(const int16_t *pcm, uint32_t samples) {
    if (!s_inited || !pcm || samples == 0) return -1;
    return nano_ao_push_js(pcm, (int32_t)samples, (int32_t)s_src_rate);
}

void audio_out_stop(void) {
    nano_ao_stop_js(); // 清空已调度的源（暂停/切曲/EOF 用；幂等）
}

void audio_out_set_volume(uint8_t volume) {
    s_volume = volume;
    if (s_inited) nano_ao_volume_js((int32_t)volume);
}

void audio_out_set_master_volume(uint8_t volume) {
    s_master_volume = volume;
    s_volume = volume; // 对齐 ALSA/Linux 版：主音量变化立即作用于当前播放增益
    if (s_inited) nano_ao_volume_js((int32_t)volume);
}

uint8_t audio_out_get_master_volume(void) {
    return s_master_volume;
}

// platform.h 声明的全局主音量接口（其余平台由各自 audio_out HAL 实现，对齐之）
void platform_set_master_volume(uint8_t volume) {
    audio_out_set_master_volume(volume);
}

uint8_t platform_get_master_volume(void) {
    return audio_out_get_master_volume();
}

void audio_out_close(void) {
    audio_out_stop();
    s_volume = s_prev_volume; // 恢复进入前的音量（对齐 ESP32 语义）
    s_inited = 0;
}
