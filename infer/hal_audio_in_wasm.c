// hal_audio_in_wasm.c —— Nano-WASM-GUI 麦克风 HAL 打桩
//
// WASM 平台无麦克风采集通路，全部接口打桩失败。
// 调用点均有失败处理：ui_ofdm RX 判 mic_init 返回值（s_mic_ok=0 静默），
// ui_spectrogram 的 mic_read 返回不足一帧则跳过本帧（画面停住但可正常退出）。

#include "hal_audio_in.h"

#include "platform.h"

int32_t mic_init(uint32_t sample_rate, uint8_t restore_volume) {
    (void)sample_rate;
    (void)restore_volume;
    return -1;
}

int32_t mic_read(int16_t *buffer, uint32_t samples) {
    (void)buffer;
    (void)samples;
    return -1;
}

int32_t mic_close() {
    return 0;
}
