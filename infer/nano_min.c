//
// nano_min.c - 极致小内存 LLM 推理引擎（以文件系统为内存）
//
//   参照 infer/infer.c 的 Q80 前向推理流程实现，支持 NANO / QWEN2 / QWEN3 架构，区别：
//     - 权重永不整体进入 RAM：经块缓存按大块读取（见 nm_model_read，默认缓存 6MB PSRAM，
//       大幅减少 SD 随机 seek；缓存不可用时回退按行随机读取）；
//     - KV-Cache 与 logits 驻留工作文件：随机读/写（经本文件内的 nm_file_* 抽象）；
//     - QWEN BPE 大词表只留文件：经预生成的 <model>.bpeidx 索引文件做折半查找；
//     - 采样对 logits 文件做流式扫描，top-p 用定长堆，无需 logits/排序缓冲驻留 RAM。
//

#include "nano_min.h"

#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>

// weak 默认空实现：设备端由 nano_min_esp32.cpp 提供强实现（Serial.printf），
// 宿主测试程序可实现为 printf；未提供时静默丢弃，宿主构建不被破坏。
__attribute__((weak)) void nm_dbg(const char *fmt, ...) { (void)fmt; }

#define NM_MAGIC_0 (1111766099u)
#define NM_MAGIC_1 (1431456845u)

#define NM_QUANT_Q80  (0x80u)

#define NM_HEADER_BYTES (256u)

#define NM_BPE_IDX_MAGIC (0x4D424549u) // "MBEI"
#define NM_BPE_IDX_VERSION (1u)

#define NM_TOPK (512u)   // top-p 采样堆大小（见 nm_sample 注释）
#define NM_LBUF (1024u)  // logits 流式读/写块大小（float 数）

// 模型文件块缓存：权重读取由“逐行随机 seek+小读”改为“按大块缓存进 PSRAM，命中直接拷贝”。
// 默认 256KB × 24 = 6MB（8MB PSRAM 设备尽量多用）；块大小与槽位可经编译宏覆盖。
// 分配失败时槽位自动减半重试，直至 1 槽；仍失败则回退直接逐行读。
#ifndef NM_CACHE_BLOCK_BYTES
#define NM_CACHE_BLOCK_BYTES (262144)
#endif
#ifndef NM_CACHE_SLOTS
#define NM_CACHE_SLOTS (24)
#endif

// ===============================================================================
// 随机访问文件抽象（nm_file_*）：多句柄、支持随机读/写
//
//   为什么不复用 platform.h 的 platform_file_*：
//     platform_file_* 是“全局单句柄 + 只读”抽象，而本引擎需要在保持模型文件
//     （占用 platform_file_* 全局句柄）打开的同时，对工作文件（KV-Cache、logits）
//     和 BPE 索引文件做并发随机读/写，故在此补充同风格的最小抽象并按平台宏选择实现。
// ===============================================================================

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM) \
    || defined(NANO_ESP32_S3) || defined(NANO_ESP32_P4)
    #define NM_PLATFORM_ESP32 (1)
#endif

typedef struct NM_File NM_File;
struct NM_File {
#if defined(NM_PLATFORM_ESP32)
    void *impl;            // ESP32：SD 卡 File 对象（由 nano_min_esp32.cpp 实现）
#else
    int fd;                // 普通 Linux：POSIX fd
#endif
};

#if defined(NM_PLATFORM_ESP32)

// ESP32：经 Arduino SD 库访问（C++，实现在 nano_min_esp32.cpp）。
// 不能走 stdio/POSIX 的 "/sdcard" VFS 路径：部分 Arduino-ESP32 内核未将 SD 挂载点注册进 VFS，
// fopen("/sdcard/...") 会报 ENOENT；设备上唯一可靠的途径是 SD 库 API 本身。
// 路径直接使用业务路径（SD 根相对，如 "/llm/nano_min_work.tmp"），不添加任何前缀。
void   *nm_sd_open(const char *path, int32_t rw); // rw=0 只读；rw=1 读写（不存在则创建，不截断）
int32_t nm_sd_pread (void *f, void *buf, uint32_t size, uint64_t offset);
int32_t nm_sd_pwrite(void *f, const void *buf, uint32_t size, uint64_t offset);
void    nm_sd_close(void *f);
int32_t nm_sd_remove(const char *path);
int32_t nm_sd_exists(const char *path);

static NM_File *nm_wrap_impl(void *impl) {
    if (!impl) return NULL;
    NM_File *f = (NM_File *)platform_malloc_internal(sizeof(NM_File));
    if (!f) { nm_sd_close(impl); return NULL; }
    f->impl = impl;
    return f;
}

static NM_File *nm_fopen_read(const char *path) {
    return nm_wrap_impl(nm_sd_open(path, 0));
}

static NM_File *nm_fopen_rw(const char *path, uint64_t total_bytes) {
    (void)total_bytes; // FatFS 随写入自动扩展
    return nm_wrap_impl(nm_sd_open(path, 1));
}

static int32_t nm_fread_at(NM_File *f, void *buf, uint32_t size, uint64_t offset) {
    return nm_sd_pread(f->impl, buf, size, offset);
}

static int32_t nm_fwrite_at(NM_File *f, const void *buf, uint32_t size, uint64_t offset) {
    return nm_sd_pwrite(f->impl, buf, size, offset);
}

static void nm_fclose(NM_File *f) {
    if (!f) return;
    nm_sd_close(f->impl);
    free(f);
}

static int32_t nm_fremove(const char *path) {
    return nm_sd_remove(path);
}

static int32_t nm_fexists(const char *path) {
    return nm_sd_exists(path);
}

#else // 普通 Linux：POSIX 实现

#include <fcntl.h>
#include <unistd.h>

static NM_File *nm_falloc(int fd) {
    NM_File *f = (NM_File *)platform_malloc_internal(sizeof(NM_File));
    if (!f) { close(fd); return NULL; }
    f->fd = fd;
    return f;
}

static NM_File *nm_fopen_read(const char *path) {
    int fd = open(path, O_RDONLY);
    return (fd < 0) ? NULL : nm_falloc(fd);
}

static NM_File *nm_fopen_rw(const char *path, uint64_t total_bytes) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return NULL;
    if (ftruncate(fd, (off_t)total_bytes) != 0) { /* 预分配失败不致命，随写入扩展 */ }
    return nm_falloc(fd);
}

static int32_t nm_fread_at(NM_File *f, void *buf, uint32_t size, uint64_t offset) {
    uint8_t *p = (uint8_t *)buf;
    uint32_t done = 0;
    while (done < size) {
        ssize_t r = pread(f->fd, p + done, size - done, (off_t)(offset + done));
        if (r <= 0) return -1;
        done += (uint32_t)r;
    }
    return 0;
}

static int32_t nm_fwrite_at(NM_File *f, const void *buf, uint32_t size, uint64_t offset) {
    const uint8_t *p = (const uint8_t *)buf;
    uint32_t done = 0;
    while (done < size) {
        ssize_t r = pwrite(f->fd, p + done, size - done, (off_t)(offset + done));
        if (r <= 0) return -1;
        done += (uint32_t)r;
    }
    return 0;
}

static void nm_fclose(NM_File *f) {
    if (!f) return;
    close(f->fd);
    free(f);
}

static int32_t nm_fremove(const char *path) {
    return (unlink(path) == 0) ? 0 : -1;
}

static int32_t nm_fexists(const char *path) {
    return (access(path, F_OK) == 0) ? 1 : 0;
}

#endif

// ===============================================================================
// 数据结构
// ===============================================================================

typedef struct {
    float    prob;
    uint32_t index;
} NM_ProbIndex;

struct NM_Engine {
    // 模型配置
    uint32_t arch;
    uint32_t block_size;
    uint32_t vocab_size;   // 分类器/logits 维度（模型配置）
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_kv_head;
    uint32_t n_hidden;
    uint32_t head_dim;
    uint32_t is_shared_classifier;
    uint32_t quant_type;
    uint32_t group_size;

    uint32_t q_dim;   // NANO/QWEN2: n_embd;  QWEN3: head_dim * n_head
    uint32_t kv_dim;  // NANO/QWEN2: n_embd*n_kv_head/n_head;  QWEN3: head_dim * n_kv_head
    uint32_t kv_mul;  // = n_head / n_kv_head
    uint32_t max_seq_len;

    // 文件句柄：模型经 platform_file_* 全局句柄；work/索引经 nm_file_*
    NM_File *work_file;   // KV-Cache + logits 工作文件
    NM_File *bpeidx_file; // QWEN BPE 索引文件（仅 QWEN）
    char *work_path_owned;
    char *bpeidx_path_owned;

    // 权重在模型文件中的字节偏移
    uint64_t off_rms_attn;   // (n_layer, n_embd) f32
    uint64_t off_rms_ffn;    // (n_layer, n_embd) f32
    uint64_t off_rms_final;  // (n_embd,) f32
    uint64_t off_emb_q;      // (vocab, n_embd) i8
    uint64_t off_emb_s;      // (vocab, n_embd/gs) f32
    uint64_t off_wq;         // 每层: (q_dim,  n_embd)    i8 + scales
    uint64_t off_wk;         // 每层: (kv_dim, n_embd)    i8 + scales
    uint64_t off_wv;         // 每层: (kv_dim, n_embd)    i8 + scales
    uint64_t off_wo;         // 每层: (n_embd, q_dim)     i8 + scales
    uint64_t off_w1;         // 每层: (n_hidden, n_embd)  i8 + scales
    uint64_t off_w2;         // 每层: (n_embd, n_hidden)  i8 + scales
    uint64_t off_w3;         // 每层: (n_hidden, n_embd)  i8 + scales
    uint64_t off_q_norm;     // QWEN3: (n_layer, head_dim) f32
    uint64_t off_k_norm;     // QWEN3: (n_layer, head_dim) f32
    uint64_t off_freq_re;    // (block_size, head_dim/2) f32（QWEN3 不使用，改为即时计算）
    uint64_t off_freq_im;
    uint64_t off_cls_q;      // 非共享分类器: (vocab, n_embd) i8 + scales
    uint64_t off_cls_s;
    // 每层张量（含 scale）的字节步长
    uint64_t stride_wq, stride_wk, stride_wv, stride_wo, stride_w1, stride_w2, stride_w3;

    // 工作文件布局: [k 区][v 区][logits 区]
    uint64_t kv_row_bytes;
    uint64_t kv_v_base;
    uint64_t logits_base;
    uint64_t work_total_bytes;

    // NANO 紧凑词表
    uint32_t nano_vocab_tokens;
    uint32_t *tok_off;   // (vocab,) 每个 token 在 tok_str 中的起始下标
    wchar_t  *tok_str;   // 所有 token 字符串拼接，每个 token 以 0 结尾
    uint32_t  max_tok_len;

    // QWEN BPE 索引（仅元信息驻留 RAM，内容全部在索引文件中）
    uint32_t bpe_count;        // 词表条目数
    uint32_t bpe_max_tok_len;  // 最长 token 字节数
    uint64_t bpe_sorted_off;   // 索引文件中“按字符串排序记录”区偏移
    uint64_t bpe_idtab_off;    // 索引文件中“按 id 记录”区偏移

    // 激活值与临时缓冲（RAM）
    float *x, *xb, *xb2, *q, *k, *v, *xba, *hb, *hb2;
    float *att;       // (n_head, max_seq_len)
    float *norm_w;    // (n_embd,)
    float *freq_re, *freq_im; // (head_dim/2,)
    float *krow, *vrow;       // (kv_dim,)
    float *lbuf;      // (NM_LBUF,) logits 流式读写块
    int8_t *wrow_q; float *wrow_s; // 权重行缓冲: (max_n,) i8 + (max_n/gs,) f32
    int8_t *xq_q;   float *xq_s;   // (n_embd,) 量化激活
    int8_t *xaq_q;  float *xaq_s;  // (q_dim,)
    int8_t *hq_q;   float *hq_s;   // (n_hidden,)
    float    *heap_val; uint32_t *heap_id; // (NM_TOPK,) top-p 采样堆
    char     *bpe_buf;  // (2*bpe_max_tok_len+3,) BPE 编码暂存
    char     *bpe_tmp;  // (bpe_max_tok_len+1,) 折半查找读入暂存（与 bpe_buf 分离，避免键值混叠）

    // 采样参数
    float rep_penalty;
    float temperature;
    float top_p;
    uint64_t rng;

    size_t ram_bytes; // 动态内存统计
};

// ---------------------------------------------------------------------------
// 基础工具（内存与文件访问一律经平台抽象）
// ---------------------------------------------------------------------------

static void *nm_alloc(NM_Engine *e, size_t bytes) {
    void *p = platform_calloc(1, bytes);
    if (!p) { fprintf(stderr, "nano_min: alloc failed (%zu bytes)\n", bytes); exit(EXIT_FAILURE); }
    e->ram_bytes += bytes;
    return p;
}

static char *nm_strdup_platform(const char *s) {
    size_t len = strlen(s);
    char *p = (char *)platform_malloc(len + 1);
    if (!p) { fprintf(stderr, "nano_min: alloc failed\n"); exit(EXIT_FAILURE); }
    memcpy(p, s, len + 1);
    return p;
}

// 模型文件块缓存状态（engine 生命周期内全局单例；nano_min 一次只跑一个引擎）
static uint8_t  *g_cache_data = NULL;   // (slots × block) 权重块缓冲（PSRAM）
static uint64_t  *g_cache_base = NULL;  // (slots,) 每槽块基址；~0ULL = 无效
static uint32_t  *g_cache_len  = NULL;  // (slots,) 每槽实际加载字节数（文件末尾块可不满）
static uint32_t   g_cache_slots = 0;
static uint32_t   g_cache_block = 0;
static uint32_t   g_cache_head  = 0;    // FIFO 替换指针

static void nm_model_cache_free(void) {
    if (g_cache_data) { free(g_cache_data); g_cache_data = NULL; }
    if (g_cache_base) { free(g_cache_base); g_cache_base = NULL; }
    if (g_cache_len)  { free(g_cache_len);  g_cache_len  = NULL; }
    g_cache_slots = g_cache_block = g_cache_head = 0;
}

// 返回 0 成功；失败返回 -1（不退出，调用方回退到直接读路径）。
// 大块数据缓冲区在 PSRAM 分配：从 NM_CACHE_SLOTS 槽位起，分配失败逐级减半重试，
// 尽最大可能用满可用 PSRAM 以换取更少 seek。
static int nm_model_cache_init(void) {
    if (g_cache_data) return 0;
    g_cache_block = (uint32_t)NM_CACHE_BLOCK_BYTES;
    uint32_t slots = (uint32_t)NM_CACHE_SLOTS;
    if (slots == 0 || g_cache_block == 0 || (uint64_t)slots * g_cache_block > 0xFFFFFFFFull) return -1;
    for (;;) {
        g_cache_data = (uint8_t *)platform_malloc((size_t)slots * g_cache_block);
        if (g_cache_data) { g_cache_slots = slots; break; }
        if (slots == 1) return -1;   // 减到头仍失败 → 直接读路径
        slots = (slots + 1) / 2;     // 槽位减半重试，尽最大可能用满 PSRAM
    }
    g_cache_base = (uint64_t *)platform_calloc_internal(g_cache_slots, sizeof(uint64_t));
    g_cache_len  = (uint32_t *)platform_calloc_internal(g_cache_slots, sizeof(uint32_t));
    if (!g_cache_base || !g_cache_len) {
        nm_model_cache_free();
        return -1;
    }
    for (uint32_t i = 0; i < g_cache_slots; i++) g_cache_base[i] = ~0ULL;
    g_cache_head = 0;
    fprintf(stderr, "nano_min: model block cache = %u KB (%u × %u B)\n",
            g_cache_slots * g_cache_block / 1024, g_cache_slots, g_cache_block);
    return 0;
}

// 定位（或加载）块 block_off；返回槽下标。加载 = 一次 seek + 大块顺序读。
static int32_t nm_block_find_or_load(uint64_t block_off) {
    for (uint32_t i = 0; i < g_cache_slots; i++) {
        if (g_cache_base[i] == block_off) return (int32_t)i;
    }
    uint32_t slot = g_cache_head;
    g_cache_head = (g_cache_head + 1) % g_cache_slots;
    g_cache_base[slot] = block_off;
    if (block_off > 0xFFFFFFFFull) {
        fprintf(stderr, "nano_min: model offset beyond 4GiB\n"); exit(EXIT_FAILURE);
    }
    if (platform_file_seek((uint32_t)block_off) != 0) {
        fprintf(stderr, "nano_min: seek failed @%llu\n", (unsigned long long)block_off); exit(EXIT_FAILURE);
    }
    uint8_t *dst = g_cache_data + (size_t)slot * g_cache_block;
    uint32_t done = 0;
    while (done < g_cache_block) {
        int32_t r = platform_file_read(dst + done, g_cache_block - done);
        if (r <= 0) break; // EOF：文件末尾不满一块
        done += (uint32_t)r;
    }
    if (done == 0) {
        fprintf(stderr, "nano_min: block read failed @%llu\n", (unsigned long long)block_off); exit(EXIT_FAILURE);
    }
    g_cache_len[slot] = done;
    return (int32_t)slot;
}

// 从模型文件随机读取 size 字节到 buffer（优先命中块缓存，其次整块缓存后拷贝）。
// 注意：platform_file_seek 的偏移为 uint32_t，故模型文件须小于 4GiB。
static void nm_model_read(void *buf, uint32_t size, uint64_t offset) {
    if (offset > 0xFFFFFFFFull) {
        fprintf(stderr, "nano_min: model offset beyond 4GiB\n"); exit(EXIT_FAILURE);
    }
    if (!g_cache_data) {
        // 缓存不可用时的直接读路径（与原实现一致）
        if (platform_file_seek((uint32_t)offset) != 0) {
            fprintf(stderr, "nano_min: seek failed @%llu\n", (unsigned long long)offset); exit(EXIT_FAILURE);
        }
        uint8_t *p = (uint8_t *)buf;
        uint32_t done = 0;
        while (done < size) {
            int32_t r = platform_file_read(p + done, size - done);
            if (r <= 0) {
                fprintf(stderr, "nano_min: read failed @%llu\n", (unsigned long long)offset); exit(EXIT_FAILURE);
            }
            done += (uint32_t)r;
        }
        return;
    }
    uint8_t *p = (uint8_t *)buf;
    while (size > 0) {
        uint64_t block_off = offset & ~((uint64_t)g_cache_block - 1);
        int32_t slot = nm_block_find_or_load(block_off);
        uint32_t in_block = (uint32_t)(offset - block_off);
        uint32_t loaded = g_cache_len[slot];
        if (in_block >= loaded) {
            fprintf(stderr, "nano_min: read beyond cached block @%llu\n", (unsigned long long)offset);
            exit(EXIT_FAILURE);
        }
        uint32_t take = loaded - in_block;
        if (take > size) take = size;
        memcpy(p, g_cache_data + (size_t)slot * g_cache_block + in_block, take);
        p += take; offset += take; size -= take;
    }
}

static void nm_work_read(NM_Engine *e, void *buf, uint32_t size, uint64_t offset) {
    if (nm_fread_at(e->work_file, buf, size, offset) != 0) {
        fprintf(stderr, "nano_min: work read failed @%llu size=%u errno=%d\n", (unsigned long long)offset, size, errno); exit(EXIT_FAILURE);
    }
}

static void nm_work_write(NM_Engine *e, const void *buf, uint32_t size, uint64_t offset) {
    if (nm_fwrite_at(e->work_file, buf, size, offset) != 0) {
        fprintf(stderr, "nano_min: work write failed @%llu\n", (unsigned long long)offset); exit(EXIT_FAILURE);
    }
}

static void nm_bpeidx_read(NM_Engine *e, void *buf, uint32_t size, uint64_t offset) {
    if (nm_fread_at(e->bpeidx_file, buf, size, offset) != 0) {
        fprintf(stderr, "nano_min: bpeidx read failed @%llu\n", (unsigned long long)offset); exit(EXIT_FAILURE);
    }
}

static uint32_t nm_header_u32(const uint8_t *h, uint32_t idx) {
    uint32_t v; memcpy(&v, h + idx * 4, 4); return v;
}

// ===============================================================================
// NANO 紧凑词表（两遍流式解析，词表字段永不整体进入 RAM）
// ===============================================================================

static int nm_load_nano_vocab(NM_Engine *e, uint32_t tok_field_bytes) {
    uint32_t vocab_in_file = 0;
    nm_model_read(&vocab_in_file, 4, NM_HEADER_BYTES + 4);
    if (vocab_in_file != e->vocab_size) {
        fprintf(stderr, "nano_min: vocab size mismatch (%u vs %u)\n", vocab_in_file, e->vocab_size);
        return -1;
    }

    // 第一遍：统计总字符数与最长 token（每个 token 只读 8 字节头）
    uint32_t total_chars = 0;
    {
        uint64_t p = NM_HEADER_BYTES + 8;
        uint64_t end = NM_HEADER_BYTES + tok_field_bytes;
        uint32_t max_len = 0;
        while (p < end) {
            uint32_t rec[2];
            nm_model_read(rec, 8, p);
            uint32_t len = rec[0] & 0xff;
            p += 8 + (uint64_t)len * 4;
            total_chars += len + 1;
            if (len > max_len) max_len = len;
        }
        e->max_tok_len = max_len;
        e->nano_vocab_tokens = e->vocab_size;
        e->tok_off = (uint32_t *)nm_alloc(e, (size_t)e->vocab_size * sizeof(uint32_t));
        e->tok_str = (wchar_t  *)nm_alloc(e, (size_t)total_chars * sizeof(wchar_t));
    }
    // 第二遍：逐 token 读取并填充紧凑词表
    {
        uint64_t p = NM_HEADER_BYTES + 8;
        uint64_t end = NM_HEADER_BYTES + tok_field_bytes;
        uint32_t char_pos = 0;
        while (p < end) {
            uint32_t rec[2];
            nm_model_read(rec, 8, p); p += 8;
            uint32_t len = rec[0] & 0xff;
            uint32_t tok_id = rec[1];
            e->tok_off[tok_id] = char_pos;
            for (uint32_t i = 0; i < len; i++) {
                uint32_t ch;
                nm_model_read(&ch, 4, p); p += 4;
                e->tok_str[char_pos++] = (wchar_t)ch;
            }
            e->tok_str[char_pos++] = 0;
        }
    }
    return 0;
}

// ===============================================================================
// QWEN BPE 索引文件（词表只留文件；索引含按字符串排序区与按 id 区）
//
//   索引文件布局（小端）：
//     [0]  u32 magic = "MBEI"
//     [4]  u32 version
//     [8]  u32 count                词表条目数
//     [12] u32 max_tok_len          最长 token 字节数
//     [16] u64 sorted_off           按字符串排序记录区偏移
//     [24] u64 idtab_off            按 id 记录区偏移
//     排序记录区: count × { u32 model_off; u32 len; u32 id; f32 score }（按 strcmp 升序）
//     id 记录区:  count × { u32 model_off; u32 len }（按下标=token id）
//   其中 model_off 为 token 字节串在模型文件中的绝对偏移。
// ===============================================================================

// 供 qsort 使用的全局上下文（构建期单线程）
static const uint8_t *g_bpe_blob;

typedef struct {
    uint32_t id;
    uint32_t len;
    uint32_t model_off;
    uint32_t blob_off;
    float    score;
} NM_BpeRec;

static int nm_bpe_rec_cmp(const void *a, const void *b) {
    const NM_BpeRec *ra = (const NM_BpeRec *)a, *rb = (const NM_BpeRec *)b;
    return strcmp((const char *)g_bpe_blob + ra->blob_off, (const char *)g_bpe_blob + rb->blob_off);
}

// 生成 BPE 索引文件（离线预处理：允许使用较多内存，推理阶段不使用该路径）
// 返回 0 成功，-1 失败（由调用方传播，避免 exit 导致整机重启）
static int nm_bpe_index_build(NM_Engine *e, const char *idx_path, uint32_t tok_field_bytes) {
    fprintf(stderr, "nano_min: building BPE index %s ...\n", idx_path);
    fflush(stderr);

    uint64_t p = NM_HEADER_BYTES + 8;
    uint64_t end = NM_HEADER_BYTES + tok_field_bytes;

    // 第一遍：统计条目数与字节总量
    uint32_t count = 0;
    uint64_t total_bytes = 0;
    uint32_t max_tok_len = 0;
    {
        uint64_t q = p;
        while (q < end) {
            uint32_t hdr[2]; // f32 score, u32 len
            nm_model_read(hdr, 8, q);
            uint32_t len = hdr[1];
            q += 8 + len;
            count++;
            total_bytes += len;
            if (len > max_tok_len) max_tok_len = len;
        }
    }

    // 第二遍：读入记录数组与字符串堆（构建期瞬态内存）
    NM_BpeRec *recs = (NM_BpeRec *)platform_malloc((size_t)count * sizeof(NM_BpeRec));
    uint8_t   *blob = (uint8_t   *)platform_malloc(total_bytes + count);
    if (!recs || !blob) {
        fprintf(stderr, "nano_min: bpe build alloc failed (recs=%zuKB blob=%zuKB)\n",
                (size_t)count * sizeof(NM_BpeRec) / 1024, (total_bytes + count) / 1024);
        free(recs); free(blob);
        return -1;
    }
    {
        uint64_t q = p;
        uint64_t blob_pos = 0;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t hdr[2];
            nm_model_read(hdr, 8, q); q += 8;
            float score; memcpy(&score, &hdr[0], 4);
            uint32_t len = hdr[1];
            recs[i].id = i;
            recs[i].len = len;
            recs[i].model_off = (uint32_t)q;
            recs[i].blob_off = (uint32_t)blob_pos;
            recs[i].score = score;
            nm_model_read(blob + blob_pos, len, q); q += len;
            blob_pos += len;
            blob[blob_pos++] = 0;
        }
    }

    // 按字符串排序
    g_bpe_blob = blob;
    qsort(recs, count, sizeof(NM_BpeRec), nm_bpe_rec_cmp);

    // 写索引文件
    NM_File *idx = nm_fopen_rw(idx_path, 0);
    if (!idx) {
        fprintf(stderr, "nano_min: cannot create %s\n", idx_path);
        free(recs); free(blob); g_bpe_blob = NULL;
        return -1;
    }

    uint64_t sorted_off = 32;
    uint64_t idtab_off  = sorted_off + (uint64_t)count * 16;
    {
        uint32_t hdr32[4] = { NM_BPE_IDX_MAGIC, NM_BPE_IDX_VERSION, count, max_tok_len };
        uint64_t hdr64[2] = { sorted_off, idtab_off };
        nm_fwrite_at(idx, hdr32, 16, 0);
        nm_fwrite_at(idx, hdr64, 16, 16);
    }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t rec[3] = { recs[i].model_off, recs[i].len, recs[i].id };
        nm_fwrite_at(idx, rec, 12, sorted_off + (uint64_t)i * 16);
        nm_fwrite_at(idx, &recs[i].score, 4, sorted_off + (uint64_t)i * 16 + 12);
    }
    // id 记录区：建立 id -> 排序后位置 的逆映射
    {
        uint32_t *inv = (uint32_t *)platform_malloc((size_t)count * sizeof(uint32_t));
        if (!inv) {
            fprintf(stderr, "nano_min: bpe build alloc failed\n");
            nm_fclose(idx); nm_fremove(idx_path); // 不留半成品索引，下次重建
            free(recs); free(blob); g_bpe_blob = NULL;
            return -1;
        }
        for (uint32_t i = 0; i < count; i++) inv[recs[i].id] = i;
        for (uint32_t id = 0; id < count; id++) {
            uint32_t rec[2] = { recs[inv[id]].model_off, recs[inv[id]].len };
            nm_fwrite_at(idx, rec, 8, idtab_off + (uint64_t)id * 8);
        }
        free(inv);
    }
    nm_fclose(idx);

    free(recs);
    free(blob);
    g_bpe_blob = NULL;

    fprintf(stderr, "nano_min: BPE index built (%u tokens).\n", count);
    return 0;
}

// 打开（缺失时构建）BPE 索引文件；返回 0 成功，-1 失败
static int nm_bpe_index_open(NM_Engine *e, const char *model_path, uint32_t tok_field_bytes) {
    // 派生索引文件路径：<model_path>.bpeidx
    size_t len = strlen(model_path);
    e->bpeidx_path_owned = (char *)platform_malloc(len + 8);
    if (!e->bpeidx_path_owned) { fprintf(stderr, "nano_min: alloc failed\n"); return -1; }
    memcpy(e->bpeidx_path_owned, model_path, len);
    memcpy(e->bpeidx_path_owned + len, ".bpeidx", 8);

    if (!nm_fexists(e->bpeidx_path_owned)) {
        if (nm_bpe_index_build(e, e->bpeidx_path_owned, tok_field_bytes) != 0) return -1;
    }

    e->bpeidx_file = nm_fopen_read(e->bpeidx_path_owned);
    if (!e->bpeidx_file) { fprintf(stderr, "nano_min: cannot open %s\n", e->bpeidx_path_owned); return -1; }

    uint32_t hdr32[4];
    uint64_t hdr64[2];
    nm_bpeidx_read(e, hdr32, 16, 0);
    nm_bpeidx_read(e, hdr64, 16, 16);
    if (hdr32[0] != NM_BPE_IDX_MAGIC || hdr32[1] != NM_BPE_IDX_VERSION) {
        fprintf(stderr, "nano_min: bad BPE index file\n"); return -1;
    }
    e->bpe_count       = hdr32[2];
    e->bpe_max_tok_len = hdr32[3];
    e->bpe_sorted_off  = hdr64[0];
    e->bpe_idtab_off   = hdr64[1];

    e->bpe_buf = (char *)nm_alloc(e, (size_t)e->bpe_max_tok_len * 2 + 3);
    e->bpe_tmp = (char *)nm_alloc(e, (size_t)e->bpe_max_tok_len + 1);
    return 0;
}

// 从模型文件读取 token 字节串（经索引记录的 model_off）
static void nm_bpe_read_token_bytes(NM_Engine *e, uint32_t model_off, uint32_t len, char *buf) {
    nm_model_read(buf, len, model_off);
    buf[len] = 0;
}

// 按 id 取 token 字节串
const char *nm_bpe_token_str(NM_Engine *e, uint32_t id, char *buf, uint32_t buf_size) {
    if (id >= e->bpe_count) { if (buf_size) buf[0] = 0; return buf; }
    uint32_t rec[2];
    nm_bpeidx_read(e, rec, 8, e->bpe_idtab_off + (uint64_t)id * 8);
    if (rec[1] + 1 > buf_size) { if (buf_size) buf[0] = 0; return buf; }
    nm_bpe_read_token_bytes(e, rec[0], rec[1], buf);
    return buf;
}

// 按字符串折半查找（在索引文件的排序记录区上做文件折半）。命中返回 id 并带出 score；未命中返回 -1。
static int32_t nm_bpe_lookup(NM_Engine *e, const char *key, float *score_out) {
    uint32_t lo = 0, hi = e->bpe_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        uint32_t rec[3];
        nm_bpeidx_read(e, rec, 12, e->bpe_sorted_off + (uint64_t)mid * 16);
        // 读入 token 字节串并与 key 比较（key 可能位于 bpe_buf，必须用无符号字节序语义）
        char *tok = e->bpe_tmp;
        nm_bpe_read_token_bytes(e, rec[0], rec[1], tok);
        int cmp = strcmp(key, tok);
        if (cmp == 0) {
            if (score_out) {
                nm_bpeidx_read(e, score_out, 4, e->bpe_sorted_off + (uint64_t)mid * 16 + 12);
            }
            return (int32_t)rec[2];
        }
        if (cmp < 0) hi = mid; else lo = mid + 1;
    }
    return -1;
}

// QWEN BPE 编码（与 tokenizer.c encode_bpe 行为一致：UTF-8 码点成组 -> 查表/字节回退 -> 按 score 归并）
uint32_t nm_encode_bpe(NM_Engine *e, const char *text, uint32_t *tokens, uint32_t max_tokens) {
    if (!text) return 0;

    char *str_buffer = e->bpe_buf; // (2*max_tok_len + 3)
    size_t str_len = 0;
    uint32_t n_tokens = 0;

    // 处理原始 UTF-8 字节序列
    for (const char *c = text; *c != '\0' && n_tokens < max_tokens; c++) {
        // 非延续字节（ASCII 或起始字节）：重置缓冲区
        if (((uint8_t)*c & 0xC0) != 0x80) str_len = 0;

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // 下一字节是延续字节：继续累积当前码点
        if (((uint8_t)*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;

        int32_t id = nm_bpe_lookup(e, str_buffer, NULL);
        if (id >= 0) {
            tokens[n_tokens++] = (uint32_t)id;
        }
        else {
            // 字节回退：前 3 个词元是 <unk> <s> </s>，单字节词元从下标 3 开始
            for (size_t i = 0; i < str_len && n_tokens < max_tokens; i++) {
                tokens[n_tokens++] = (uint8_t)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    // 按 score 归并最佳相邻对
    char *tok_a = (char *)platform_malloc(e->bpe_max_tok_len + 1);
    char *tok_b = (char *)platform_malloc(e->bpe_max_tok_len + 1);
    if (!tok_a || !tok_b) { fprintf(stderr, "nano_min: alloc failed\n"); exit(EXIT_FAILURE); }

    while (n_tokens >= 2) {
        float best_score = -1e10f;
        int32_t best_id = -1;
        uint32_t best_idx = 0;

        for (uint32_t i = 0; i + 1 < n_tokens; i++) {
            uint32_t la = 0, lb = 0;
            {
                uint32_t rec[2];
                nm_bpeidx_read(e, rec, 8, e->bpe_idtab_off + (uint64_t)tokens[i] * 8);
                nm_bpe_read_token_bytes(e, rec[0], rec[1], tok_a); la = rec[1];
                nm_bpeidx_read(e, rec, 8, e->bpe_idtab_off + (uint64_t)tokens[i + 1] * 8);
                nm_bpe_read_token_bytes(e, rec[0], rec[1], tok_b); lb = rec[1];
            }
            if (la + lb > e->bpe_max_tok_len * 2) continue;
            memcpy(str_buffer, tok_a, la);
            memcpy(str_buffer + la, tok_b, lb + 1); // 含 NUL

            float score = 0.0f;
            int32_t id = nm_bpe_lookup(e, str_buffer, &score);
            if (id >= 0 && score > best_score) {
                best_score = score;
                best_id = id;
                best_idx = i;
            }
        }

        if (best_id < 0) break;

        tokens[best_idx] = (uint32_t)best_id;
        for (uint32_t i = best_idx + 1; i + 1 < n_tokens; i++) tokens[i] = tokens[i + 1];
        n_tokens--;
    }

    free(tok_a);
    free(tok_b);
    return n_tokens;
}

// ===============================================================================
// 打开 / 关闭
// ===============================================================================

NM_Engine *nm_open(const char *model_path, const char *work_path, uint32_t max_seq_len) {
    NM_Engine *e = (NM_Engine *)platform_calloc(1, sizeof(NM_Engine));
    if (!e) { fprintf(stderr, "nano_min: alloc failed\n"); return NULL; }

    if (platform_file_open(model_path) != 0) {
        fprintf(stderr, "nano_min: cannot open model %s\n", model_path);
        free(e);
        return NULL;
    }

    // ---- 文件头 ----
    uint8_t header[NM_HEADER_BYTES];
    nm_model_read(header, NM_HEADER_BYTES, 0);

    if (nm_header_u32(header, 0) != NM_MAGIC_0 || nm_header_u32(header, 1) != NM_MAGIC_1) {
        fprintf(stderr, "nano_min: bad magic number\n"); goto fail;
    }

    e->arch                 = nm_header_u32(header, 4);
    e->block_size           = nm_header_u32(header, 6);
    e->vocab_size           = nm_header_u32(header, 7);
    e->n_layer              = nm_header_u32(header, 8);
    e->n_embd               = nm_header_u32(header, 9);
    e->n_head               = nm_header_u32(header, 10);
    e->n_kv_head            = nm_header_u32(header, 11);
    e->n_hidden             = nm_header_u32(header, 12);
    e->is_shared_classifier = nm_header_u32(header, 13);
    e->head_dim             = nm_header_u32(header, 14);
    e->quant_type           = nm_header_u32(header, 15);
    e->group_size           = nm_header_u32(header, 16);

    if (e->arch != NM_ARCH_NANO && e->arch != NM_ARCH_QWEN2 && e->arch != NM_ARCH_QWEN3) {
        fprintf(stderr, "nano_min: unsupported arch %u\n", e->arch); goto fail;
    }
    if (e->quant_type != NM_QUANT_Q80) {
        fprintf(stderr, "nano_min: only Q80 quant is supported (quant=0x%x)\n", e->quant_type);
        goto fail;
    }
    if (max_seq_len == 0 || max_seq_len > e->block_size) max_seq_len = e->block_size;
    e->max_seq_len = max_seq_len;

    if (e->arch == NM_ARCH_QWEN3) {
        e->q_dim  = e->head_dim * e->n_head;
        e->kv_dim = e->head_dim * e->n_kv_head;
    }
    else {
        e->head_dim = e->n_embd / e->n_head;
        e->q_dim    = e->n_embd;
        e->kv_dim   = e->head_dim * e->n_kv_head;
    }
    e->kv_mul = e->n_head / e->n_kv_head;

    uint32_t gs = e->group_size;
    uint64_t n_layer = e->n_layer, n_embd = e->n_embd, vocab = e->vocab_size;
    uint64_t q_dim = e->q_dim, kv_dim = e->kv_dim, n_hidden = e->n_hidden;

    // ---- 词表（先于块缓存分配！） ----
    // QWEN 的 BPE 索引构建需瞬态持有大内存（Qwen3 词表 15 万条：recs ~3MB + 字符串堆 ~1MB），
    // 若先分配 6MB 块缓存，8MB PSRAM 余量不足会导致构建失败；
    // 构建只需 nm_model_read 的直接读路径（缓存未初始化时自动回退），构建完成后索引走文件。
    uint32_t tok_field_bytes = 0;
    nm_model_read(&tok_field_bytes, 4, NM_HEADER_BYTES);

    if (e->arch == NM_ARCH_NANO) {
        if (nm_load_nano_vocab(e, tok_field_bytes) != 0) goto fail;
    }
    else {
        if (nm_bpe_index_open(e, model_path, tok_field_bytes) != 0) goto fail;
    }

    // 权重读取块缓存（分配失败仅回退直接读，不致命）
    if (nm_model_cache_init() != 0) {
        fprintf(stderr, "nano_min: model block cache unavailable, fallback to row reads\n");
    }

    // ---- 权重偏移（与 infer.c memory_map_params 的 Q80 布局一致） ----
    uint64_t base = NM_HEADER_BYTES + (uint64_t)tok_field_bytes;

    e->off_rms_attn  = base;                          base += n_layer * n_embd * 4;
    e->off_rms_ffn   = base;                          base += n_layer * n_embd * 4;
    e->off_rms_final = base;                          base += n_embd * 4;

    e->off_emb_q = base;                              base += vocab * n_embd;
    e->off_emb_s = base;                              base += vocab * n_embd / gs * 4;

    e->stride_wq = q_dim * n_embd    + q_dim * n_embd    / gs * 4;
    e->stride_wk = kv_dim * n_embd   + kv_dim * n_embd   / gs * 4;
    e->stride_wv = kv_dim * n_embd   + kv_dim * n_embd   / gs * 4;
    e->stride_wo = n_embd * q_dim    + n_embd * q_dim    / gs * 4;
    e->stride_w1 = n_hidden * n_embd + n_hidden * n_embd / gs * 4;
    e->stride_w2 = n_embd * n_hidden + n_embd * n_hidden / gs * 4;
    e->stride_w3 = n_hidden * n_embd + n_hidden * n_embd / gs * 4;

    e->off_wq = base; base += n_layer * e->stride_wq;
    e->off_wk = base; base += n_layer * e->stride_wk;
    e->off_wv = base; base += n_layer * e->stride_wv;
    e->off_wo = base; base += n_layer * e->stride_wo;
    e->off_w1 = base; base += n_layer * e->stride_w1;
    e->off_w2 = base; base += n_layer * e->stride_w2;
    e->off_w3 = base; base += n_layer * e->stride_w3;

    if (e->arch == NM_ARCH_QWEN2) {
        base += n_layer * (q_dim + kv_dim + kv_dim) * 4; // bq/bk/bv（原引擎未参与计算，仅跳过）
    }
    else if (e->arch == NM_ARCH_QWEN3) {
        e->off_q_norm = base; base += n_layer * e->head_dim * 4;
        e->off_k_norm = base; base += n_layer * e->head_dim * 4;
    }

    e->off_freq_re = base; base += (uint64_t)e->block_size * (e->head_dim / 2) * 4;
    e->off_freq_im = base; base += (uint64_t)e->block_size * (e->head_dim / 2) * 4;

    // 分类器：is_shared_classifier=1 时与嵌入共享；否则文件中还有一份 Q80 张量
    if (!e->is_shared_classifier) {
        e->off_cls_q = base; base += vocab * n_embd;
        e->off_cls_s = base; base += vocab * n_embd / gs * 4;
    }
    else {
        e->off_cls_q = e->off_emb_q;
        e->off_cls_s = e->off_emb_s;
    }

    // ---- 工作文件（KV-Cache + logits） ----
    e->kv_row_bytes = (uint64_t)e->kv_dim * 4;
    e->kv_v_base    = (uint64_t)e->n_layer * e->max_seq_len * e->kv_row_bytes;
    e->logits_base  = e->kv_v_base * 2;
    e->work_total_bytes = e->logits_base + (uint64_t)e->vocab_size * 4;

    e->work_path_owned = nm_strdup_platform(work_path);
    e->work_file = nm_fopen_rw(work_path, e->work_total_bytes);
    if (!e->work_file) { fprintf(stderr, "nano_min: cannot open work file %s\n", work_path); goto fail; }

    // ---- 激活值与临时缓冲 ----
    uint32_t max_n = e->n_embd;
    if (e->n_hidden > max_n) max_n = e->n_hidden;
    if (e->q_dim    > max_n) max_n = e->q_dim;

    e->x       = (float *)nm_alloc(e, n_embd * 4);
    e->xb      = (float *)nm_alloc(e, n_embd * 4);
    e->xb2     = (float *)nm_alloc(e, n_embd * 4);
    e->q       = (float *)nm_alloc(e, q_dim * 4);
    e->k       = (float *)nm_alloc(e, kv_dim * 4);
    e->v       = (float *)nm_alloc(e, kv_dim * 4);
    e->xba     = (float *)nm_alloc(e, q_dim * 4);
    e->hb      = (float *)nm_alloc(e, n_hidden * 4);
    e->hb2     = (float *)nm_alloc(e, n_hidden * 4);
    e->att     = (float *)nm_alloc(e, (size_t)e->n_head * e->max_seq_len * 4);
    e->norm_w  = (float *)nm_alloc(e, n_embd * 4);
    e->freq_re = (float *)nm_alloc(e, (e->head_dim / 2) * 4);
    e->freq_im = (float *)nm_alloc(e, (e->head_dim / 2) * 4);
    e->krow    = (float *)nm_alloc(e, kv_dim * 4);
    e->vrow    = (float *)nm_alloc(e, kv_dim * 4);
    e->lbuf    = (float *)nm_alloc(e, NM_LBUF * 4);
    e->wrow_q  = (int8_t *)nm_alloc(e, max_n);
    e->wrow_s  = (float *)nm_alloc(e, (max_n / gs) * 4);
    e->xq_q    = (int8_t *)nm_alloc(e, n_embd);
    e->xq_s    = (float *)nm_alloc(e, (n_embd / gs) * 4);
    e->xaq_q   = (int8_t *)nm_alloc(e, q_dim);
    e->xaq_s   = (float *)nm_alloc(e, (q_dim / gs) * 4);
    e->hq_q    = (int8_t *)nm_alloc(e, n_hidden);
    e->hq_s    = (float *)nm_alloc(e, (n_hidden / gs) * 4);
    e->heap_val = (float   *)nm_alloc(e, NM_TOPK * 4);
    e->heap_id  = (uint32_t*)nm_alloc(e, NM_TOPK * 4);

    // 默认采样参数
    nm_set_sampler(e, 1.0f, 0.7f, 0.8f, 42);

    const char *arch_name = (e->arch == NM_ARCH_NANO) ? "NANO"
                          : (e->arch == NM_ARCH_QWEN2) ? "QWEN2" : "QWEN3";
    nm_dbg("[nm] open %s arch=%s layers=%u embd=%u vocab=%u cache=%uKB(%ux%uB) work=%.1fMB ram=%.1fKB\n",
           model_path, arch_name, e->n_layer, e->n_embd, e->vocab_size,
           g_cache_slots * g_cache_block / 1024, g_cache_slots, g_cache_block,
           (double)e->work_total_bytes / 1048576.0, (double)e->ram_bytes / 1024.0);
    return e;

fail:
    // 打开失败：nm_close 对部分初始化状态是安全的（e 由 platform_calloc 零初始化，
    // 各指针/句柄字段为空时自动跳过），避免 exit 导致整机重启
    nm_close(e);
    return NULL;
}

void nm_close(NM_Engine *e) {
    if (!e) return;
    platform_file_close(); // 模型文件（全局句柄）
    nm_model_cache_free(); // 释放模型块缓存
    if (e->work_file) nm_fclose(e->work_file);
    if (e->bpeidx_file) nm_fclose(e->bpeidx_file);
    // 注意：工作文件是引擎自身创建的，随引擎关闭而删除（BPE 索引文件保留复用）
    if (e->work_path_owned) { nm_fremove(e->work_path_owned); free(e->work_path_owned); }
    if (e->bpeidx_path_owned) free(e->bpeidx_path_owned);
    if (e->tok_off) free(e->tok_off);
    if (e->tok_str) free(e->tok_str);
    if (e->bpe_buf) free(e->bpe_buf);
    if (e->bpe_tmp) free(e->bpe_tmp);
    free(e->x); free(e->xb); free(e->xb2); free(e->q); free(e->k); free(e->v);
    free(e->xba); free(e->hb); free(e->hb2); free(e->att);
    free(e->norm_w); free(e->freq_re); free(e->freq_im); free(e->krow); free(e->vrow);
    free(e->lbuf); free(e->wrow_q); free(e->wrow_s);
    free(e->xq_q); free(e->xq_s); free(e->xaq_q); free(e->xaq_s); free(e->hq_q); free(e->hq_s);
    free(e->heap_val); free(e->heap_id);
    free(e);
}

size_t nm_ram_bytes(NM_Engine *e) { return e ? e->ram_bytes : 0; }

uint32_t nm_get_arch(NM_Engine *e) { return e->arch; }

int nm_is_eos(NM_Engine *e, uint32_t token) {
    if (e->arch == NM_ARCH_NANO) return (token == 0 || token == 3);
    return (token == 151643 || token == 151645); // QWEN: <|endoftext|> / <|im_end|>
}

void nm_print_info(NM_Engine *e) {
    const char *arch_name = (e->arch == NM_ARCH_NANO) ? "NANO" : (e->arch == NM_ARCH_QWEN2) ? "QWEN2" : "QWEN3";
    printf("  arch = %s, quant = Q80, group_size = %u\n", arch_name, e->group_size);
    printf("  block_size = %u, vocab_size = %u, n_layer = %u\n", e->block_size, e->vocab_size, e->n_layer);
    printf("  n_embd = %u, n_head = %u, n_kv_head = %u, n_hidden = %u, head_dim = %u\n",
           e->n_embd, e->n_head, e->n_kv_head, e->n_hidden, e->head_dim);
    printf("  max_seq_len = %u\n", e->max_seq_len);
    if (e->arch != NM_ARCH_NANO) {
        printf("  BPE 词表 %u 条（内容在索引文件，不占 RAM）\n", e->bpe_count);
    }
    printf("  工作文件占用 = %.1f MB（KV-Cache + logits，位于文件系统，不占 RAM）\n",
           (double)e->work_total_bytes / 1048576.0);
    printf("  引擎动态内存(RAM) = %.1f KB\n", (double)e->ram_bytes / 1024.0);
}

// ===============================================================================
// NANO 分词（贪心最长匹配，与原引擎 tokenizer.c tokenize() 行为一致）
// ===============================================================================

uint32_t nm_encode(NM_Engine *e, const wchar_t *text, uint32_t *out_ids, uint32_t max_ids) {
    size_t len = wcslen(text);
    size_t pos = 0;
    uint32_t n = 0;
    while (pos < len && n < max_ids) {
        uint32_t best_len = 0, best_id = 0;
        size_t limit = len - pos;
        if (limit > e->max_tok_len) limit = e->max_tok_len;
        for (uint32_t id = 0; id < e->nano_vocab_tokens; id++) {
            const wchar_t *tok = e->tok_str + e->tok_off[id];
            size_t tl = wcslen(tok);
            if (tl == 0 || tl > limit || tl <= best_len) continue;
            if (wcsncmp(text + pos, tok, tl) == 0) { best_len = (uint32_t)tl; best_id = id; }
        }
        if (best_len == 0) { pos++; continue; } // 未登录字符：跳过
        out_ids[n++] = best_id;
        pos += best_len;
    }
    return n;
}

const wchar_t *nm_token_str(NM_Engine *e, uint32_t id) {
    if (id >= e->nano_vocab_tokens) return L"?";
    return e->tok_str + e->tok_off[id];
}

// ===============================================================================
// 算子（与 infer.c 一致）
// ===============================================================================

static void nm_rmsnorm(float *o, const float *x, const float *w, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t j = 0; j < n; j++) ss += x[j] * x[j];
    ss /= n; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
    for (uint32_t j = 0; j < n; j++) o[j] = w[j] * (ss * x[j]);
}

static void nm_softmax(float *x, uint32_t n) {
    float max_val = x[0];
    for (uint32_t i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (uint32_t i = 0; i < n; i++) x[i] /= sum;
}

static void nm_quantize(int8_t *qq, float *qs, const float *x, uint32_t n, uint32_t gs) {
    for (uint32_t g = 0; g < n / gs; g++) {
        float wmax = 0.0f;
        for (uint32_t i = 0; i < gs; i++) {
            float v = fabsf(x[g * gs + i]);
            if (v > wmax) wmax = v;
        }
        float scale = wmax / 127.0f;
        qs[g] = scale;
        for (uint32_t i = 0; i < gs; i++) {
            qq[g * gs + i] = (int8_t)roundf(x[g * gs + i] / scale);
        }
    }
}

// 流式 Q80 矩阵-向量乘：W(d,n) @ xq(n,) -> 回调输出每行结果
// 权重张量（单层）布局: [q: d*n i8][s: d*(n/gs) f32]，逐行随机读取，权重永不整体进入 RAM。
static void nm_matvec_q80(NM_Engine *e, const int8_t *xq, const float *xs,
                          uint64_t tensor_off, uint32_t n, uint32_t d,
                          void (*emit)(NM_Engine *, uint32_t, float)) {
    uint32_t gs = e->group_size;
    uint64_t s_base = tensor_off + (uint64_t)d * n;
    uint32_t n_groups = n / gs;
    for (uint32_t i = 0; i < d; i++) {
        nm_model_read(e->wrow_q, n, tensor_off + (uint64_t)i * n);
        nm_model_read(e->wrow_s, n_groups * 4, s_base + (uint64_t)i * n_groups * 4);
        float val = 0.0f;
        int32_t ival = 0;
        for (uint32_t j = 0; j < n; j += gs) {
            for (uint32_t g = 0; g < gs; g++) {
                ival += (int32_t)xq[j + g] * (int32_t)e->wrow_q[j + g];
            }
            val += (float)ival * e->wrow_s[j / gs] * xs[j / gs];
            ival = 0;
        }
        emit(e, i, val);
    }
}

// emit 目标：RAM 缓冲（激活值）
typedef struct { float *out; } NM_EmitRamCtx;
static NM_EmitRamCtx g_emit_ram;
static void nm_emit_ram(NM_Engine *e, uint32_t i, float val) { (void)e; g_emit_ram.out[i] = val; }

// emit 目标：工作文件 logits 区（经 NM_LBUF 块缓冲）
typedef struct { uint32_t count; } NM_EmitLogitsCtx;
static NM_EmitLogitsCtx g_emit_logits;
static void nm_emit_logits(NM_Engine *e, uint32_t i, float val) {
    e->lbuf[i % NM_LBUF] = val;
    if (i % NM_LBUF == NM_LBUF - 1) {
        nm_work_write(e, e->lbuf, NM_LBUF * 4, e->logits_base + (uint64_t)(i / NM_LBUF) * NM_LBUF * 4);
    }
    g_emit_logits.count = i + 1;
}
static void nm_emit_logits_flush(NM_Engine *e) {
    uint32_t rem = g_emit_logits.count % NM_LBUF;
    if (rem) {
        nm_work_write(e, e->lbuf, rem * 4, e->logits_base + (uint64_t)(g_emit_logits.count / NM_LBUF) * NM_LBUF * 4);
    }
}

static void nm_matvec_to_ram(NM_Engine *e, float *xout, const int8_t *xq, const float *xs,
                             uint64_t tensor_off, uint32_t n, uint32_t d) {
    g_emit_ram.out = xout;
    nm_matvec_q80(e, xq, xs, tensor_off, n, d, nm_emit_ram);
}

static void nm_matvec_to_logits(NM_Engine *e, const int8_t *xq, const float *xs,
                                uint64_t tensor_off, uint32_t n, uint32_t d) {
    g_emit_logits.count = 0;
    nm_matvec_q80(e, xq, xs, tensor_off, n, d, nm_emit_logits);
    nm_emit_logits_flush(e);
}

// NANO/QWEN2：相邻成对旋转
static void nm_rope(float *head, uint32_t head_dim, const float *fre, const float *fim) {
    for (uint32_t i = 0; i < head_dim; i += 2) {
        float v0 = head[i], v1 = head[i + 1];
        float fcr = fre[i / 2], fci = fim[i / 2];
        head[i]     = v0 * fcr - v1 * fci;
        head[i + 1] = v0 * fci + v1 * fcr;
    }
}

// QWEN3：前后半对半旋转
static void nm_rope_qwen3(float *head, uint32_t head_dim, const float *fre, const float *fim) {
    for (uint32_t i = 0; i < head_dim / 2; i++) {
        float fcr = fre[i], fci = fim[i];
        float v0 = head[i];
        float v1 = head[i + head_dim / 2];
        head[       i        ] = v0 * fcr - v1 * fci;
        head[i + head_dim / 2] = v1 * fcr + v0 * fci;
    }
}

// ===============================================================================
// 前向推理（logits 写入工作文件）
// ===============================================================================

void nm_forward(NM_Engine *e, uint32_t token, uint32_t pos) {
    // 低频进度：每 token 一行（推理本身远慢于串口开销，不影响性能）
    nm_dbg("[nm] fwd tok=%u pos=%u\n", token, pos);

    uint32_t n_embd  = e->n_embd;
    uint32_t kv_dim  = e->kv_dim;
    uint32_t q_dim   = e->q_dim;
    uint32_t n_head  = e->n_head;
    uint32_t hd      = e->head_dim;
    uint32_t n_hidden= e->n_hidden;
    uint32_t gs      = e->group_size;

    // 嵌入：按 token 按需读取一行量化权重并反量化（不缓存整个嵌入表）
    nm_model_read(e->wrow_q, n_embd, e->off_emb_q + (uint64_t)token * n_embd);
    nm_model_read(e->wrow_s, (n_embd / gs) * 4, e->off_emb_s + (uint64_t)token * (n_embd / gs) * 4);
    for (uint32_t i = 0; i < n_embd; i++) e->x[i] = (float)e->wrow_q[i] * e->wrow_s[i / gs];

    // RoPE 系数（仅与 pos 有关，各层共享）
    if (e->arch == NM_ARCH_QWEN3) {
        // 即时计算（theta = 1e6，与 infer.c 对 QWEN3 的处理一致）
        for (uint32_t i = 0; i < hd / 2; i++) {
            float freq = 1.0f / powf(1000000.0f, (float)(i * 2) / (float)hd);
            e->freq_re[i] = cosf(pos * freq);
            e->freq_im[i] = sinf(pos * freq);
        }
    }
    else {
        nm_model_read(e->freq_re, (hd / 2) * 4, e->off_freq_re + (uint64_t)pos * (hd / 2) * 4);
        nm_model_read(e->freq_im, (hd / 2) * 4, e->off_freq_im + (uint64_t)pos * (hd / 2) * 4);
    }

    for (uint32_t l = 0; l < e->n_layer; l++) {

        // attention rmsnorm（权重按层读取）
        nm_model_read(e->norm_w, n_embd * 4, e->off_rms_attn + (uint64_t)l * n_embd * 4);
        nm_rmsnorm(e->xb, e->x, e->norm_w, n_embd);

        // QKV
        nm_quantize(e->xq_q, e->xq_s, e->xb, n_embd, gs);
        nm_matvec_to_ram(e, e->q, e->xq_q, e->xq_s, e->off_wq + (uint64_t)l * e->stride_wq, n_embd, q_dim);
        nm_matvec_to_ram(e, e->k, e->xq_q, e->xq_s, e->off_wk + (uint64_t)l * e->stride_wk, n_embd, kv_dim);
        nm_matvec_to_ram(e, e->v, e->xq_q, e->xq_s, e->off_wv + (uint64_t)l * e->stride_wv, n_embd, kv_dim);

        if (e->arch == NM_ARCH_QWEN3) {
            // q/k 逐头 rmsnorm（QWEN3 特有）
            nm_model_read(e->norm_w, hd * 4, e->off_q_norm + (uint64_t)l * hd * 4);
            for (uint32_t h = 0; h < n_head; h++) nm_rmsnorm(e->q + h * hd, e->q + h * hd, e->norm_w, hd);
            nm_model_read(e->norm_w, hd * 4, e->off_k_norm + (uint64_t)l * hd * 4);
            for (uint32_t h = 0; h < e->n_kv_head; h++) nm_rmsnorm(e->k + h * hd, e->k + h * hd, e->norm_w, hd);
            for (uint32_t h = 0; h < n_head; h++) nm_rope_qwen3(e->q + h * hd, hd, e->freq_re, e->freq_im);
            for (uint32_t h = 0; h < e->n_kv_head; h++) nm_rope_qwen3(e->k + h * hd, hd, e->freq_re, e->freq_im);
        }
        else {
            for (uint32_t h = 0; h < n_head; h++) nm_rope(e->q + h * hd, hd, e->freq_re, e->freq_im);
            for (uint32_t m = 0; m < e->n_kv_head; m++) nm_rope(e->k + m * hd, hd, e->freq_re, e->freq_im);
        }

        // 写入 KV-Cache（工作文件）
        uint64_t kv_off = ((uint64_t)l * e->max_seq_len + pos) * e->kv_row_bytes;
        nm_work_write(e, e->k, e->kv_row_bytes, kv_off);
        nm_work_write(e, e->v, e->kv_row_bytes, e->kv_v_base + kv_off);

        // 多头注意力：第一遍算分数（逐 t 从文件读 k 行）
        for (uint32_t t = 0; t <= pos; t++) {
            nm_work_read(e, e->krow, e->kv_row_bytes,
                         ((uint64_t)l * e->max_seq_len + t) * e->kv_row_bytes);
            for (uint32_t h = 0; h < n_head; h++) {
                const float *qh = e->q + h * hd;
                const float *kh = e->krow + (h / e->kv_mul) * hd;
                float score = 0.0f;
                for (uint32_t i = 0; i < hd; i++) score += qh[i] * kh[i];
                e->att[h * e->max_seq_len + t] = score / sqrtf((float)hd);
            }
        }
        for (uint32_t h = 0; h < n_head; h++) nm_softmax(e->att + h * e->max_seq_len, pos + 1);

        // 第二遍加权求和（逐 t 从文件读 v 行）
        memset(e->xba, 0, q_dim * 4);
        for (uint32_t t = 0; t <= pos; t++) {
            nm_work_read(e, e->vrow, e->kv_row_bytes,
                         e->kv_v_base + ((uint64_t)l * e->max_seq_len + t) * e->kv_row_bytes);
            for (uint32_t h = 0; h < n_head; h++) {
                float a = e->att[h * e->max_seq_len + t];
                const float *vh = e->vrow + (h / e->kv_mul) * hd;
                float *xba = e->xba + h * hd;
                for (uint32_t i = 0; i < hd; i++) xba[i] += a * vh[i];
            }
        }

        // 输出投影 + 残差
        nm_quantize(e->xaq_q, e->xaq_s, e->xba, q_dim, gs);
        nm_matvec_to_ram(e, e->xb2, e->xaq_q, e->xaq_s, e->off_wo + (uint64_t)l * e->stride_wo, q_dim, n_embd);
        for (uint32_t i = 0; i < n_embd; i++) e->x[i] += e->xb2[i];

        // FFN rmsnorm
        nm_model_read(e->norm_w, n_embd * 4, e->off_rms_ffn + (uint64_t)l * n_embd * 4);
        nm_rmsnorm(e->xb, e->x, e->norm_w, n_embd);

        // W1, W3
        nm_quantize(e->xq_q, e->xq_s, e->xb, n_embd, gs);
        nm_matvec_to_ram(e, e->hb,  e->xq_q, e->xq_s, e->off_w1 + (uint64_t)l * e->stride_w1, n_embd, n_hidden);
        nm_matvec_to_ram(e, e->hb2, e->xq_q, e->xq_s, e->off_w3 + (uint64_t)l * e->stride_w3, n_embd, n_hidden);

        // SwiGLU
        for (uint32_t i = 0; i < n_hidden; i++) {
            float val = e->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            e->hb[i] = val * e->hb2[i];
        }

        // W2 + 残差
        nm_quantize(e->hq_q, e->hq_s, e->hb, n_hidden, gs);
        nm_matvec_to_ram(e, e->xb, e->hq_q, e->hq_s, e->off_w2 + (uint64_t)l * e->stride_w2, n_hidden, n_embd);
        for (uint32_t i = 0; i < n_embd; i++) e->x[i] += e->xb[i];
    }

    // final rmsnorm + 分类器（逐行流式读取，logits 写入工作文件）
    nm_model_read(e->norm_w, n_embd * 4, e->off_rms_final);
    nm_rmsnorm(e->x, e->x, e->norm_w, n_embd);

    nm_quantize(e->xq_q, e->xq_s, e->x, n_embd, gs);
    nm_matvec_to_logits(e, e->xq_q, e->xq_s, e->off_cls_q, n_embd, e->vocab_size);
}

// ===============================================================================
// 流式采样（logits 在工作文件中，多遍扫描，O(1) 额外内存 + 定长堆）
// ===============================================================================

void nm_set_sampler(NM_Engine *e, float repetition_penalty, float temperature, float top_p, uint64_t rng_seed) {
    e->rep_penalty = repetition_penalty;
    e->temperature = temperature;
    e->top_p = top_p;
    e->rng = rng_seed ? rng_seed : 1;
}

static float nm_random_f32(uint64_t *state) {
    // xorshift64*
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return (float)((x * 2685821657736338717ULL) >> 11) * (1.0f / 9007199254740992.0f);
}

static int nm_u32_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int nm_probindex_cmp(const void *a, const void *b) {
    // 用于堆数组降序排序（heap_val/heap_id 联动由调用方处理，这里比较 (val,id) 对）
    const NM_ProbIndex *pa = (const NM_ProbIndex *)a, *pb = (const NM_ProbIndex *)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

void nm_debug_top2(NM_Engine *e) {
    float top1 = -1e30f, top2 = -1e30f;
    uint32_t id1 = 0, id2 = 0;
    uint32_t done = 0;
    while (done < e->vocab_size) {
        uint32_t chunk = (e->vocab_size - done < NM_LBUF) ? (e->vocab_size - done) : NM_LBUF;
        nm_work_read(e, e->lbuf, chunk * 4, e->logits_base + (uint64_t)done * 4);
        for (uint32_t i = 0; i < chunk; i++) {
            if (e->lbuf[i] > top1) { top2 = top1; id2 = id1; top1 = e->lbuf[i]; id1 = done + i; }
            else if (e->lbuf[i] > top2) { top2 = e->lbuf[i]; id2 = done + i; }
        }
        done += chunk;
    }
    fprintf(stderr, "[top1=%u(%.4f) top2=%u(%.4f) margin=%.6f]", id1, top1, id2, top2, top1 - top2);
}

uint32_t nm_sample(NM_Engine *e, const uint32_t *seen_ids, uint32_t n_seen) {
    uint32_t vocab = e->vocab_size;

    // 复读惩罚（对每个出现过的 token 只惩罚一次，与原引擎 tokenset 行为一致）：
    // 直接在工作文件的 logits 区上读-改-写
    if (e->rep_penalty != 1.0f && seen_ids && n_seen > 0) {
        uint32_t *sorted = (uint32_t *)platform_malloc(n_seen * sizeof(uint32_t));
        if (sorted) {
            memcpy(sorted, seen_ids, n_seen * sizeof(uint32_t));
            qsort(sorted, n_seen, sizeof(uint32_t), nm_u32_cmp);
            uint32_t prev = UINT32_MAX;
            for (uint32_t i = 0; i < n_seen; i++) {
                if (sorted[i] != prev && sorted[i] < vocab) {
                    float lg;
                    nm_work_read(e, &lg, 4, e->logits_base + (uint64_t)sorted[i] * 4);
                    lg /= e->rep_penalty;
                    nm_work_write(e, &lg, 4, e->logits_base + (uint64_t)sorted[i] * 4);
                    prev = sorted[i];
                }
            }
            free(sorted);
        }
    }

    // 第一遍：求最大值与 argmax
    float max_val = 0.0f;
    uint32_t argmax = 0;
    {
        int32_t first = 1;
        uint32_t done = 0;
        while (done < vocab) {
            uint32_t chunk = (vocab - done < NM_LBUF) ? (vocab - done) : NM_LBUF;
            nm_work_read(e, e->lbuf, chunk * 4, e->logits_base + (uint64_t)done * 4);
            for (uint32_t i = 0; i < chunk; i++) {
                if (first || e->lbuf[i] > max_val) { max_val = e->lbuf[i]; argmax = done + i; first = 0; }
            }
            done += chunk;
        }
    }

    // 温度为 0：贪心
    if (e->temperature == 0.0f) return argmax;

    float inv_t = 1.0f / e->temperature;
    float coin = nm_random_f32(&e->rng);

    int32_t use_top_p = (e->top_p > 0.0f && e->top_p < 1.0f);

    // 第二遍：softmax 归一化项 sum = Σ exp((l-max)/T)；
    //         top-p 时同时维护 exp 值最大的前 NM_TOPK 个（定长堆）
    float sum_exp = 0.0f;
    uint32_t heap_n = 0;
    uint32_t heap_min_idx = 0;
    {
        uint32_t done = 0;
        while (done < vocab) {
            uint32_t chunk = (vocab - done < NM_LBUF) ? (vocab - done) : NM_LBUF;
            nm_work_read(e, e->lbuf, chunk * 4, e->logits_base + (uint64_t)done * 4);
            for (uint32_t i = 0; i < chunk; i++) {
                float pv = expf((e->lbuf[i] - max_val) * inv_t);
                sum_exp += pv;
                if (use_top_p) {
                    if (heap_n < NM_TOPK) {
                        e->heap_val[heap_n] = pv; e->heap_id[heap_n] = done + i;
                        if (e->heap_val[heap_n] < e->heap_val[heap_min_idx]) heap_min_idx = heap_n;
                        heap_n++;
                    }
                    else if (pv > e->heap_val[heap_min_idx]) {
                        e->heap_val[heap_min_idx] = pv; e->heap_id[heap_min_idx] = done + i;
                        // 重新找堆内最小
                        for (uint32_t m = 0; m < NM_TOPK; m++) {
                            if (e->heap_val[m] < e->heap_val[heap_min_idx]) heap_min_idx = m;
                        }
                    }
                }
            }
            done += chunk;
        }
    }

    if (use_top_p) {
        // 堆内按 prob 降序排序（排序缓冲复用 lbuf 之后的小内存：NM_TOPK×8B）
        NM_ProbIndex *pi = (NM_ProbIndex *)platform_malloc(NM_TOPK * sizeof(NM_ProbIndex));
        if (!pi) { fprintf(stderr, "nano_min: alloc failed\n"); exit(EXIT_FAILURE); }
        float heap_sum = 0.0f;
        for (uint32_t i = 0; i < heap_n; i++) {
            pi[i].prob = e->heap_val[i];
            pi[i].index = e->heap_id[i];
            heap_sum += e->heap_val[i];
        }
        qsort(pi, heap_n, sizeof(NM_ProbIndex), nm_probindex_cmp);

        // 核集合 = 累积概率质量最先达到 top_p 的最高概率词元集合。
        // 若堆内总质量已 ≥ top_p × sum_exp，则核集合必含于堆内，结果为精确 top-p；
        // 否则（重尾分布，罕见）退化为在堆内归一化采样（近似）。
        float threshold = e->top_p * sum_exp;
        float r, cdf;
        if (heap_sum >= threshold) {
            float cumulative = 0.0f;
            uint32_t last_idx = heap_n - 1;
            for (uint32_t i = 0; i < heap_n; i++) {
                cumulative += pi[i].prob;
                if (cumulative > threshold) { last_idx = i; break; }
            }
            r = coin * cumulative;
            cdf = 0.0f;
            uint32_t result = pi[last_idx].index;
            for (uint32_t i = 0; i <= last_idx; i++) {
                cdf += pi[i].prob;
                if (r < cdf) { result = pi[i].index; break; }
            }
            free(pi);
            return result;
        }
        else {
            r = coin * heap_sum;
            cdf = 0.0f;
            uint32_t result = pi[heap_n - 1].index;
            for (uint32_t i = 0; i < heap_n; i++) {
                cdf += pi[i].prob;
                if (r < cdf) { result = pi[i].index; break; }
            }
            free(pi);
            return result;
        }
    }

    // 多项式采样：第三遍扫描累积分布
    {
        float r = coin * sum_exp;
        float cdf = 0.0f;
        uint32_t done = 0;
        while (done < vocab) {
            uint32_t chunk = (vocab - done < NM_LBUF) ? (vocab - done) : NM_LBUF;
            nm_work_read(e, e->lbuf, chunk * 4, e->logits_base + (uint64_t)done * 4);
            for (uint32_t i = 0; i < chunk; i++) {
                cdf += expf((e->lbuf[i] - max_val) * inv_t);
                if (r < cdf) return done + i;
            }
            done += chunk;
        }
        return vocab - 1; // 舍入误差兜底
    }
}
