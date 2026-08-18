//
// nano_min.h - 极致小内存 LLM 推理引擎（以文件系统为内存）
//
//   基于 infer/infer.c 的 Q80 模型前向推理流程重新实现，支持 NANO / QWEN2 / QWEN3 架构。
//   设计目标：模型推理相关 RAM 占用 < 1MB，速度不计。
//
//   内存策略：
//     - 权重：不整体载入，经模型文件块缓存按大块读取（NM_CACHE_BLOCK_BYTES×NM_CACHE_SLOTS
//       字节的 FIFO 块缓存，默认 256KB×24=6MB PSRAM；可用编译宏覆盖），大幅减少 SD 随机
//       seek；块分配失败自动减半槽位重试，仍失败回退到逐行随机读取；
//     - KV-Cache 与 logits：写入工作文件，随机读/写；
//     - 采样：对 logits 做流式多遍扫描（top-p 用定长堆，精确条件见 nm_sample 注释）；
//     - 词表：NANO 小词表紧凑驻留 RAM；QWEN BPE 大词表只留文件，
//       首次打开时自动生成 BPE 索引文件 <model>.bpeidx（可预先拷贝到 SD 卡）。
//
//   平台无关性：引擎仅依赖 platform.h 的抽象（内存 platform_calloc/malloc、
//   模型文件随机读 platform_file_*、fs_init 由应用调用）；KV/logits/索引文件的
//   随机读写由引擎内部按平台宏选择实现：
//     - 普通 Linux：POSIX open/pread/pwrite；
//     - ESP32（ESP32/ARDUINO_ARCH_ESP32/ESP_PLATFORM/NANO_ESP32_*）：
//       经 Arduino SD 库（实现在 nano_min_esp32.cpp，C++ 垫片；部分内核未把
//       SD 挂载点注册进 VFS，stdio/POSIX 路径不可用），文件系统位于 SD 卡。
//
//   构建（WSL2，在 infer 目录下）：
//     gcc -DNANO_CLI -O2 -Wall -o bin/nano_min main_nano_min.c nano_min.c platform_linux.c utils.c -lm -pthread
//

#ifndef __NANO_MIN_H__
#define __NANO_MIN_H__

#include <stdint.h>
#include <stddef.h>
#include <wchar.h>

#define NM_ARCH_NANO  (0u)
#define NM_ARCH_QWEN2 (2u)
#define NM_ARCH_QWEN3 (3u)

typedef struct NM_Engine NM_Engine;

// 打开模型文件并初始化引擎（调用前须先 fs_init()）。
//   model_path  : 模型文件路径（Q80 量化的 Nano/Qwen 模型）
//   work_path   : 工作文件路径（KV-Cache + logits，会被创建/截断；
//                 大小约 n_layer*max_seq_len*kv_dim*2*4 + vocab_size*4 字节）
//   max_seq_len : 最大序列长度（<= 模型 block_size）
// 对于 QWEN 模型，若 <model_path>.bpeidx 不存在，将自动生成（构建过程允许使用较多内存，
// 属于离线预处理；也可在 PC 上生成后拷贝到 SD 卡）。
NM_Engine *nm_open(const char *model_path, const char *work_path, uint32_t max_seq_len);
void       nm_close(NM_Engine *e);

// 打印模型配置与内存占用统计
void nm_print_info(NM_Engine *e);

uint32_t nm_get_arch(NM_Engine *e);

// NANO 分词：贪心最长匹配（与原引擎 tokenizer.c tokenize() 行为一致）
uint32_t nm_encode(NM_Engine *e, const wchar_t *text, uint32_t *out_ids, uint32_t max_ids);
// QWEN BPE 分词：输入 UTF-8 字节串（与原引擎 encode_bpe 行为一致，不含聊天模板）
uint32_t nm_encode_bpe(NM_Engine *e, const char *text, uint32_t *out_ids, uint32_t max_ids);

// 取 token 对应文本：NANO 返回宽字符串；QWEN 返回 UTF-8 字节串（拷贝进 buf，返回 buf）
const wchar_t *nm_token_str(NM_Engine *e, uint32_t id);
const char    *nm_bpe_token_str(NM_Engine *e, uint32_t id, char *buf, uint32_t buf_size);

// 是否为结束 token（与原引擎一致：NANO 为 0/3；QWEN 为 151643/151645）
int nm_is_eos(NM_Engine *e, uint32_t token);

// 设置采样参数
void nm_set_sampler(NM_Engine *e, float repetition_penalty, float temperature, float top_p, uint64_t rng_seed);

// 执行一步前向推理（token 位于位置 pos），logits 写入工作文件
void nm_forward(NM_Engine *e, uint32_t token, uint32_t pos);

// 在工作文件中的 logits 上采样；seen_ids/n_seen 为已生成序列（用于复读惩罚，可为 NULL）
uint32_t nm_sample(NM_Engine *e, const uint32_t *seen_ids, uint32_t n_seen);

// 引擎自身统计的动态内存占用（字节）
size_t nm_ram_bytes(NM_Engine *e);

// 调试输出（printf 风格，低频关键行为进度）：
//   - 设备端（ESP32）由 nano_min_esp32.cpp 实现为 Serial.printf；
//   - 宿主机测试程序可实现为 printf 以观察；
//   - 未提供实现时为 no-op（weak 默认），不会破坏任何宿主构建。
void nm_dbg(const char *fmt, ...);

// 调试：打印当前 logits 的 top1/top2 及差值（扫描工作文件，O(1) 内存）
void nm_debug_top2(NM_Engine *e);

#endif
