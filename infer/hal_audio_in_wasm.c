// hal_audio_in_wasm.c —— Nano-WASM-GUI 麦克风 HAL（getUserMedia + Web Audio 采集）
//
// 通路：getUserMedia(单声道/关 AEC/关 AGC/关降噪) ──▶ AudioWorkletProcessor
//        （无 AudioWorklet 时退化为 ScriptProcessorNode）──port.postMessage──▶
//        JS 侧 int16 环形缓冲 ──EM_JS──▶ mic_read() 拷入 C 缓冲
//
// 与各平台 HAL 的语义差异（重要，调用点已按此约定工作）：
//
//   · mic_init 是**乐观返回**：浏览器采集权限是异步授权的（Promise），无法在同步
//     接口里等待。因此只要当前环境具备 getUserMedia 就返回 0，采样在授权完成后
//     陆续到达；不具备（node 回归台、无 API 的浏览器）则返回 -1，调用点按
//     「麦克风初始化失败」降级。
//   · mic_read **不阻塞**：单线程 WASM 里阻塞等待会冻结整个 UI。约定：
//       返回 n == samples ：已拷出恰好一帧；
//       返回 0            ：环内不足一帧，**未消费任何数据**（调用方跳过本帧再来）；
//       返回 -1           ：硬错误（权限被拒 / 采集中断）。
//     「不足一帧就不消费」是关键：频谱仪与 OFDM 都对返回值做 `< samples` 的跳过
//     处理，若返回半帧并被丢弃，等于持续丢样。
//   · 采集速率：AudioContext 以请求的 sampleRate 建立（频谱仪 44100 / OFDM 48000），
//     由浏览器把设备流重采样到该速率，与 C 侧假设严格一致（否则频谱轴与 OFDM
//     时序会整体偏移）。个别浏览器拒绝指定采样率时退让为设备默认速率并上报状态。
//   · 关闭 AEC/AGC/降噪：回声消除会把 OFDM 波形当"回声"抹掉，自动增益会破坏
//     幅度信息，降噪会把窄带子载波当噪声。三者必须显式关闭。
//   · 需要安全上下文（https 或 localhost），与 DeviceMotion 的要求一致。
//
// 已知边界：麦克风采集在页面隐藏时仍可继续（AudioWorklet 跑在音频线程，不受
// rAF 节流影响），但 C 侧消费由渲染循环驱动，隐藏期间环会溢满并丢弃最旧数据；
// 返回前台后从最新数据继续（频谱仪/接收机语义上等价于"暂停后从当前时刻继续"）。

#include "hal_audio_in.h"

#include "platform.h"

#include <emscripten.h>

EM_JS(int32_t, nano_mic_start_js, (int32_t rate), {
    if (typeof nanoMicStart !== "function") return -1;
    try { return nanoMicStart(rate) ? 0 : -1; } catch (e) { return -1; }
});

EM_JS(int32_t, nano_mic_read_js, (int16_t *dst, int32_t samples), {
    if (typeof nanoMicRead !== "function") return -1;
    try { return nanoMicRead(dst, samples); } catch (e) { return -1; }
});

EM_JS(void, nano_mic_stop_js, (void), {
    if (typeof nanoMicStop !== "function") return;
    try { nanoMicStop(); } catch (e) { /* 忽略 */ }
});

static int32_t s_inited = 0;

int32_t mic_init(uint32_t sample_rate, uint8_t restore_volume) {
    (void)restore_volume; // 浏览器里扬声器与麦克风可共存，无需"关闭扬声器再恢复"这套 I2S 切换
    nano_mic_stop_js();
    s_inited = (nano_mic_start_js((int32_t)sample_rate) == 0) ? 1 : 0;
    return s_inited ? 0 : -1;
}

int32_t mic_read(int16_t *buffer, uint32_t samples) {
    if (!s_inited || !buffer || samples == 0) return -1;
    return nano_mic_read_js(buffer, (int32_t)samples);
}

int32_t mic_close() {
    nano_mic_stop_js();
    s_inited = 0;
    return 0;
}
