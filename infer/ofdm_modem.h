#ifndef __NANO_OFDM_MODEM_H__
#define __NANO_OFDM_MODEM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// ============================================================================
// OFDM Modem Core（C 语言移植）/ BD4SUR 2026-07
// 移植自 ofdm/ofdm-modem-worklet.js（Web Audio AudioWorklet 参考实现），
// 物理层/帧协议参数与 JS 版完全一致，空口互通。
//
// 硬件无关：不依赖任何 Arduino/平台头文件，浮点采样（float，标度[-1,1]）进/出；
// 实例内存由调用方注入分配器（ESP32 侧传 platform_calloc/free 走 PSRAM）。
//
// 物理层：定长中等帧广播（类DRM）。每个数据 packet 独立封装为一个物理帧，
// 帧结构固定为 22 槽（≈660ms）：
//   [SC粗同步前导][训练A][训练B][5×数据][训练][5×数据][训练][5×数据][训练][1×数据]
// 调制/解调采用基带复FFT制式：发射=基带复IFFT → ×8多相内插 → IQ混频上变频；
// 接收=IQ混频下变频 → 多相抽取 → 基带复FFT读 bin。混频本振为相位累加器DDS。
// ============================================================================

// ---------------- 系统参数（与 JS 版一致，请勿单独修改） ----------------
#define OFDM_SAMPLE_RATE        (48000)
#define OFDM_BASE_FREQ          (46.875f)   // 子载波间隔(Hz)
#define OFDM_CARRIER_NUMBER     (64)        // 正交子载波数
#define OFDM_CARRIER_FREQ       (4000)      // 通带中心频率(Hz)（频带 2500~5500Hz，避开 <2kHz 的 1/f 噪声区）
#define OFDM_SYMBOL_LENGTH      (1024)      // 全速率符号长度（采样点）
#define OFDM_BB_FFT_LEN         (128)       // 基带复FFT点数
#define OFDM_DECIM              (8)         // 抽取/内插因子（基带 6kHz）
#define OFDM_CP_BB              (52)        // 基带循环前缀长度
#define OFDM_CP_LENGTH          (416)       // 全速率循环前缀长度
#define OFDM_GROSS_SYMBOL_LENGTH (1440)     // 槽长（符号+CP）
#define OFDM_SLOT_BB            (180)       // 基带槽长
#define OFDM_FRAME_DATA_SYMBOLS (16)        // 每帧数据符号数
#define OFDM_FRAME_SLOTS        (22)        // 每帧槽数（含SC前导）
#define OFDM_FRAME_LENGTH       (31680)     // 帧长（采样点）= 22 × 1440
#define OFDM_PILOT_NUMBER       (8)         // 散布导频点数/符号
#define OFDM_DATA_CARRIERS      (56)        // 数据子载波数/符号
#define OFDM_BYTES_PER_SYMBOL   (14)        // QAM4 → 14字节/符号
#define OFDM_PKT_HEADER_LEN     (8)         // magic(6)+len(1)+seq(1)
#define OFDM_PKT_WIRE_LEN       (224)       // 线字节/帧
#define OFDM_PKT_RS_BLOCKS      (7)         // RS(32,16)块数
#define OFDM_PKT_UNCODED_LEN    (112)       // 编码前字节数
#define OFDM_PKT_PAYLOAD_MAX    (104)       // 净荷上限/帧

// 内存分配器注入：ESP32 侧传 platform_calloc/free（PSRAM），宿主机传 calloc/free
typedef void *(*ofdm_alloc_fn)(size_t size);
typedef void (*ofdm_free_fn)(void *ptr);

// ============================================================================
// 码表生命周期管理
// 码表（DDS LUT/抗混叠FIR/训练与导频图案/训练与SC前导时域模板/GF与RS表，约24KB）
// 在使用 modem 前由调用方初始化（内存由调用方注入，如 PSRAM），使用完毕释放。
// ============================================================================

// 初始化并生成全部码表（幂等，重复调用安全）。返回 0=成功，<0=内存不足。
int32_t ofdm_tables_init(ofdm_alloc_fn alloc);

// 释放码表内存（幂等）
void ofdm_tables_free(ofdm_free_fn dealloc);

// ============================================================================
// 发射机（流式逐帧渲染，天然支持任意长文本与循环播放）
// ============================================================================

typedef struct OFDM_TX OFDM_TX;

// 由 UTF-8 净荷创建发射机（内部完成 packet 切分/RS编码/交织/加扰）。
// 失败（内存不足等）返回 NULL。
OFDM_TX *ofdm_tx_create(ofdm_alloc_fn alloc, ofdm_free_fn dealloc,
                        const uint8_t *payload_utf8, uint32_t payload_len);

// 渲染下一个物理帧到 out（OFDM_FRAME_LENGTH 个 float，逐帧双区峰值归一化到 0.9）。
// 返回 0=成功产出一帧；1=全部帧已渲染完（本次未产出，需 ofdm_tx_rewind 后再渲染）。
int32_t ofdm_tx_render_frame(OFDM_TX *tx, float *out);

// 回到第 0 帧（循环播放用）
void ofdm_tx_rewind(OFDM_TX *tx);

// 总帧数（= packet 数）
uint32_t ofdm_tx_frame_count(const OFDM_TX *tx);

void ofdm_tx_destroy(OFDM_TX *tx);

// ============================================================================
// 接收机（流式状态机："sync" 搜索SC前导 ⇄ "frame" 帧内逐槽接收）
// ============================================================================

typedef struct OFDM_RX OFDM_RX;

// 每成功解出一个 packet（magic 校验通过）回调其净荷（UTF-8 字节流，非 NUL 结尾）
typedef void (*ofdm_rx_text_cb)(const uint8_t *bytes, uint32_t len, void *user);
// 事件日志（ASCII）：粗同步锁定、CFO/SFO 估计、坏帧、失锁回退等
typedef void (*ofdm_rx_log_cb)(const char *msg, void *user);

// 创建接收机。两个回调均可为 NULL。失败返回 NULL。
OFDM_RX *ofdm_rx_create(ofdm_alloc_fn alloc, ofdm_free_fn dealloc,
                        ofdm_rx_text_cb text_cb, ofdm_rx_log_cb log_cb, void *user);

// 喂入采样流（任意块长；内部按 ≤1 槽分片驱动状态机，调用方无需关心块长；
// 定容环形缓冲满时覆盖最旧采样，O(1) 写入/丢弃，稳态零堆分配）
void ofdm_rx_feed(OFDM_RX *rx, const float *samples, uint32_t n);

// 复位到同步搜索状态（清空缓冲与统计）
void ofdm_rx_reset(OFDM_RX *rx);

void ofdm_rx_destroy(OFDM_RX *rx);

// ---------------- 遥测与可视化钩子（为后续频谱/星座图等可视化预留） ----------------

typedef struct {
    int32_t  locked;         // 1=帧内接收中（已锁定）/ 0=搜索前导中
    uint32_t frames_ok;      // 成功解出帧数
    uint32_t frames_bad;     // 坏帧数（magic 不符）
    uint32_t rs_corrected;   // RS 累计纠正字节数
    uint32_t rs_fail_blocks; // RS 累计失败块数
    float    cfo_hz;         // 最近一次 CFO 估计（Hz）
    float    sfo_ppm;        // 最近一次 SFO 估计（ppm）
    float    sc_metric;      // sync 状态下当前 SC 自相关最佳度量（-1=尚无候选；调试用）
    uint32_t buf_len;        // 帧接收环形缓冲当前占用采样数（容量 BUF_RING_CAP=65536，调试用）
    uint32_t det_len;        // SC 检测历史环形缓冲当前占用采样数（容量 DET_RING_CAP=16384，调试用）
    uint64_t feed_abs;       // 累计喂入采样数（调试用）
} OFDM_RX_Stat;

void ofdm_rx_get_stat(OFDM_RX *rx, OFDM_RX_Stat *stat);

// 取最近一次数据符号的时域块（OFDM_SYMBOL_LENGTH 点）与均衡后 IQ（OFDM_CARRIER_NUMBER 点）。
// viz_seq 每处理一个数据符号递增，调用方据此判断是否有新数据。
// 返回 0=有效；<0=尚无数据符号。指针指向 RX 内部缓冲，下次 ofdm_rx_feed 后失效。
int32_t ofdm_rx_get_viz(OFDM_RX *rx, const float **wave,
                        const float **iq_i, const float **iq_q, uint32_t *viz_seq);

#ifdef __cplusplus
}
#endif

#endif
