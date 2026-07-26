#include "platform.h"
#include "mic.h"

#include <Arduino.h>
#include "M5Unified.h"
#include <driver/i2s_std.h>
#include <driver/i2s_pdm.h>

// M5Core2 内置硅麦（PDM）HAL实现——使用 ESP-IDF 新版 I2S 驱动（i2s_std）。
//
// 注意（重要）：本设备上绝不能使用 legacy I2S 驱动（driver/i2s.h）！
// legacy I2S 会链接 adc_i2s_deprecated → adc_legacy（旧版ADC驱动），
// 与 M5Unified Power_Class 已使用的 esp_adc（新版ADC驱动）冲突，
// 导致启动期 check_adc_oneshot_driver_conflict 断言重启。
// 新版 I2S 驱动无此问题（M5Unified 的麦克风/扬声器同样基于 i2s_std）。
//
// 硬件资源：I2S_NUM_0（与 M5Unified 扬声器共用，采集前关闭扬声器、结束后恢复）。
// 引脚（参照 M5Core2 FactoryTest）：CLK(WS)=GPIO0, DATA_IN=GPIO34。

#define MIC_I2S_PORT     (I2S_NUM_0)
#define MIC_PIN_CLK      (0)
#define MIC_PIN_DATA_IN  (34)
#define MIC_SAMPLE_RATE  (44100)

static i2s_chan_handle_t s_rx_handle = NULL;

int32_t mic_init() {
    // 扬声器与麦克风共用 I2S_NUM_0：先让出
    M5.Speaker.end();

    // 若上次未正常关闭，先清理残留通道
    if (s_rx_handle) {
        i2s_channel_disable(s_rx_handle);
        i2s_del_channel(s_rx_handle);
        s_rx_handle = NULL;
    }

    // 创建RX通道（新版驱动）
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(MIC_I2S_PORT, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = 128;
    esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &s_rx_handle);
    if (err != ESP_OK) return -1;

    // PDM RX 模式：单声道、右槽、16bit、44.1kHz
    i2s_pdm_rx_config_t pdm_cfg;
    memset(&pdm_cfg, 0, sizeof(pdm_cfg));
    pdm_cfg.clk_cfg.clk_src = I2S_CLK_SRC_PLL_160M;
    pdm_cfg.clk_cfg.sample_rate_hz = MIC_SAMPLE_RATE;
    pdm_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_128;
    pdm_cfg.slot_cfg.data_bit_width = I2S_DATA_BIT_WIDTH_16BIT;
    pdm_cfg.slot_cfg.slot_bit_width = I2S_SLOT_BIT_WIDTH_16BIT;
    pdm_cfg.slot_cfg.slot_mode = I2S_SLOT_MODE_MONO;
    pdm_cfg.slot_cfg.slot_mask = I2S_PDM_SLOT_RIGHT;
    pdm_cfg.gpio_cfg.clk = (gpio_num_t)MIC_PIN_CLK;
    pdm_cfg.gpio_cfg.din = (gpio_num_t)MIC_PIN_DATA_IN;

    err = i2s_channel_init_pdm_rx_mode(s_rx_handle, &pdm_cfg);
    if (err != ESP_OK) return -2;

    err = i2s_channel_enable(s_rx_handle);
    if (err != ESP_OK) return -3;

    return 0;
}

int32_t mic_read(int16_t *buffer, uint32_t samples) {
    if (!s_rx_handle) return -1;
    size_t bytes_read = 0;
    esp_err_t err = i2s_channel_read(s_rx_handle, buffer, samples * sizeof(int16_t),
                                     &bytes_read, (100 / portTICK_RATE_MS));
    if (err != ESP_OK) return -2;
    return (int32_t)(bytes_read / sizeof(int16_t));
}

int32_t mic_close() {
    if (s_rx_handle) {
        i2s_channel_disable(s_rx_handle);
        i2s_del_channel(s_rx_handle);
        s_rx_handle = NULL;
    }
    // 恢复扬声器（按键音等；音量与 setup() 中一致）
    M5.Speaker.begin();
    M5.Speaker.setVolume(12);
    return 0;
}
