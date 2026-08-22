#include "platform.h"
#include "hal_audio_in.h"

#include <Arduino.h>
#include "M5Unified.h"
#include <driver/i2s_std.h>

// mic_init 时注入的“恢复音量”（系统主音量）：mic_close 恢复扬声器时使用。
// 由调用方从 Global_State.volume 传入，mic HAL 不反向依赖扬声器 HAL（audio_out）。
static uint8_t s_restore_volume = 16; // 与 ui_init 的 volume 初值一致

// 内置麦克风 HAL 实现——ESP-IDF 新版 I2S 驱动（i2s_std），按 NANO_PLATFORM_* 分平台：
// - NANO_PLATFORM_M5CORE2 ：PDM 硅麦，I2S_NUM_0（引脚 CLK=GPIO0, DIN=GPIO34）
// - NANO_PLATFORM_M5CORES3：ES7210 I2S ADC（标准 I2S 模式），I2S_NUM_1
//
// 注意（重要）：本系列设备上绝不能使用 legacy I2S 驱动（driver/i2s.h）！
// legacy I2S 会链接 adc_i2s_deprecated → adc_legacy（旧版ADC驱动），
// 与 M5Unified Power_Class 已使用的 esp_adc（新版ADC驱动）冲突，
// 导致启动期 check_adc_oneshot_driver_conflict 断言重启。
// 新版 I2S 驱动无此问题（M5Unified 的麦克风/扬声器同样基于 i2s_std）。
//
// 硬件资源：麦克风与扬声器共用同一 I2S 端口（Core2: I2S_NUM_0 / CoreS3: I2S_NUM_1），
// 采集前 M5.Speaker.end() 让出、结束后 M5.Speaker.begin() 恢复。

#if defined(NANO_PLATFORM_M5CORES3)

// CoreS3 麦克风为 ES7210（4 通道 I2S ADC，I2C 地址 0x40，挂在内部 I2C 总线上），
// 使用时需先经 I2C 写入初始化序列；数据走标准 I2S（非 PDM）。
// 引脚与 I2C 初始化序列均参照 M5Unified（_microphone_enabled_cb_cores3）。
//
// 时钟说明：ES7210 要求 MCLK = 256 × fs（mclk_multiple = I2S_MCLK_MULTIPLE_256）。
// M5Unified 注释指出 CoreS3 麦克风 MCLK/BCK 分频比须 ≥ 8 否则精度下降：
// 本实现单声道 16bit 时 BCK = 16 × fs，MCLK/BCK = 16 ≥ 8，满足要求。
// 双麦（左右声道各一只）取左声道（slot_mask = I2S_STD_SLOT_LEFT）。

#define ES7210_I2C_ADDR (0x40)

static i2s_chan_handle_t s_rx_handle = NULL;

static void es7210_write_reg(uint8_t reg, uint8_t value) {
    M5.In_I2C.writeRegister(ES7210_I2C_ADDR, reg, &value, 1, 400000);
}

// ES7210 初始化序列（抄自 M5Unified _microphone_enabled_cb_cores3）
// REG06 取值教训（2026-08-01）：Everest 官方系数表对 MCLK/LRCK=256 的纸面配对是
// REG06=0x04（DLL_POWER_DOWN），但真机实测该值会导致采集质量随时间渐进劣化
//（SC 精化度量/训练相关度逐帧下降，重进接收即恢复——mic_init 重写寄存器复位的特征），
// 推测本机 MCLK 来自 ESP32-S3 分数分频器（边沿抖动大），DLL 实为清洗该抖动所需；
// M5Unified 的 REG06=0x00（DLL 使能）在 128fs 与 256fs MCLK 下均实测稳定，故保持 0x00。
static void es7210_init(void) {
    static const uint8_t init_seq[][2] = {
        { 0x00, 0x41 }, // RESET_CTL
        { 0x01, 0x1f }, // CLK_ON_OFF
        { 0x06, 0x00 }, // DIGITAL_PDN（DLL 保持使能：勿改 0x04，见上方教训）
        { 0x07, 0x20 }, // ADC_OSR
        { 0x08, 0x10 }, // MODE_CFG
        { 0x09, 0x30 }, // TCT0_CHPINI
        { 0x0A, 0x30 }, // TCT1_CHPINI
        { 0x20, 0x0a }, // ADC34_HPF2
        { 0x21, 0x2a }, // ADC34_HPF1
        { 0x22, 0x0a }, // ADC12_HPF2
        { 0x23, 0x2a }, // ADC12_HPF1
        { 0x02, 0xC1 },
        { 0x04, 0x01 },
        { 0x05, 0x00 },
        { 0x11, 0x60 },
        { 0x40, 0x42 }, // ANALOG_SYS
        { 0x41, 0x70 }, // MICBIAS12
        { 0x42, 0x70 }, // MICBIAS34
        { 0x43, 0x1B }, // MIC1_GAIN
        { 0x44, 0x1B }, // MIC2_GAIN
        { 0x45, 0x00 }, // MIC3_GAIN
        { 0x46, 0x00 }, // MIC4_GAIN
        { 0x47, 0x00 }, // MIC1_LP
        { 0x48, 0x00 }, // MIC2_LP
        { 0x49, 0x00 }, // MIC3_LP
        { 0x4A, 0x00 }, // MIC4_LP
        { 0x4B, 0x00 }, // MIC12_PDN
        { 0x4C, 0xFF }, // MIC34_PDN
        { 0x01, 0x14 }, // CLK_ON_OFF
    };
    es7210_write_reg(0x00, 0xFF); // RESET_CTL
    for (size_t i = 0; i < sizeof(init_seq) / sizeof(init_seq[0]); i++) {
        es7210_write_reg(init_seq[i][0], init_seq[i][1]);
    }
}

int32_t mic_init(uint32_t sample_rate, uint8_t restore_volume) {
    s_restore_volume = restore_volume;
    // 扬声器与麦克风共用 I2S_NUM_1：先让出
    M5.Speaker.end();

    // 初始化 ES7210（I2C 寄存器配置）
    es7210_init();

    // 若上次未正常关闭，先清理残留通道
    if (s_rx_handle) {
        i2s_channel_disable(s_rx_handle);
        i2s_del_channel(s_rx_handle);
        s_rx_handle = NULL;
    }

    // 创建RX通道（新版驱动）
    // DMA 缓冲 = desc_num × frame_num 采样。OFDM 寻呼机等连续采样业务要求数据流严格连续：
    // 上层读取消耗不及时（如全量刷帧 ~30ms）时，DMA 缓冲不足会导致 I2S 接收溢出丢样。
    // 8 × 512 = 4096 采样（@48kHz ≈ 85ms），足以覆盖 UI 重绘造成的消费间隙。
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG((i2s_port_t)MIC_I2S_PORT, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = 512;
    esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &s_rx_handle);
    if (err != ESP_OK) return -1;

    // 标准 I2S（Philips）RX 模式：单声道、左槽、16bit，MCLK = 256 × fs
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = (gpio_num_t)MIC_PIN_MCLK,
            .bclk = (gpio_num_t)MIC_PIN_BCLK,
            .ws   = (gpio_num_t)MIC_PIN_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)MIC_PIN_DATA_IN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256; // ES7210 要求 MCLK = 256fs
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;        // 双麦取左声道

    err = i2s_channel_init_std_mode(s_rx_handle, &std_cfg);
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
    // 恢复扬声器（按键音等；恢复为 mic_init 注入的主音量）
    M5.Speaker.begin();
    M5.Speaker.setVolume(s_restore_volume);
    return 0;
}

#else // NANO_PLATFORM_M5CORE2

#include <driver/i2s_pdm.h>

static i2s_chan_handle_t s_rx_handle = NULL;

int32_t mic_init(uint32_t sample_rate, uint8_t restore_volume) {
    s_restore_volume = restore_volume;
    // 扬声器与麦克风共用 I2S_NUM_0：先让出
    M5.Speaker.end();

    // 若上次未正常关闭，先清理残留通道
    if (s_rx_handle) {
        i2s_channel_disable(s_rx_handle);
        i2s_del_channel(s_rx_handle);
        s_rx_handle = NULL;
    }

    // 创建RX通道（新版驱动）
    // DMA 缓冲 = desc_num × frame_num 采样。OFDM 寻呼机等连续采样业务要求数据流严格连续：
    // 上层读取消耗不及时（如全量刷帧 ~30ms）时，DMA 缓冲不足会导致 I2S 接收溢出丢样。
    // 8 × 512 = 4096 采样（@48kHz ≈ 85ms），足以覆盖 UI 重绘造成的消费间隙。
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG((i2s_port_t)MIC_I2S_PORT, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = 512;
    esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &s_rx_handle);
    if (err != ESP_OK) return -1;

    // PDM RX 模式：单声道、右槽、16bit
    i2s_pdm_rx_config_t pdm_cfg;
    memset(&pdm_cfg, 0, sizeof(pdm_cfg));
    pdm_cfg.clk_cfg.clk_src = I2S_CLK_SRC_PLL_160M;
    pdm_cfg.clk_cfg.sample_rate_hz = sample_rate;
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
    // 恢复扬声器（按键音等；恢复为 mic_init 注入的主音量）
    M5.Speaker.begin();
    M5.Speaker.setVolume(s_restore_volume);
    return 0;
}

#endif
