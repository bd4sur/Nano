//
// nano_cuda.cu - Nano 语言模型 CUDA 推理引擎（极简高效版）
//
//   BD4SUR 2026-10
//   参照 infer.c 从零实现的 CUDA 推理引擎。
//
//   支持范围：
//     - 模型架构：Qwen3（LLM_ARCH_QWEN3，head_dim=128，QK-norm，半区旋转 RoPE）
//       与自研 Nano（LLM_ARCH_NANO，head_dim 32/48，文件表交织 RoPE，宽字符 trie 词表）
//     - 已验证：qwen3-0b6 / qwen3-1b7 / qwen3-4b-instruct-2507 / nano-168m / nano-56m(-base)
//     - LoRA 外挂（仅 NANO 架构，fp32 低秩分支，-lora 传入，语义同 infer.c）
//     - 量化：仅 int8（Q80，group_size=128），与 infer.c 的 Q80 文件格式完全兼容
//     - 权重布局：与 infer.c memory_map_params 解析顺序一致
//
//   精度扩展口（当前仅实例化 Q80）：
//     - WeightType 枚举 + gemv 分发器 gemv_dispatch()：新增精度时，
//       实现对应的 gemv 内核 / WeightView，并在分发器中注册即可；
//       前向流程（forward_launch）不感知具体精度。
//     - KV cache 当前为 fp32，可仿照权重的做法替换为 fp16/int8 视图。
//
//   性能要点：
//     - GEMV 使用 dp4a（int8×int8→int32）指令，权重与激活均按 128 分组量化
//     - QKV 与 W1/W3 按层拼接，减少 kernel 数与 weight 指针切换
//     - RMSNorm+量化、SwiGLU+量化、Attention+输出量化 均为融合 kernel
//     - 整个 decode step 被捕获为 CUDA Graph，逐步重放，消除 launch 开销
//     - prefill 阶段逐步异步入队（无逐 step 同步），流水线执行
//
//   实测性能（RTX 4070 Laptop / sm_89，贪心解码，稳态 TPS）：
//     - qwen3-0b6-q80 ：约 255-330
//     - qwen3-1b7-q80 ：约 115
//     - qwen3-4b-q80  ：约 51-55
//     - nano-168m-q80 ：约 605
//     - nano-56m-q80  ：约 1235（+LoRA 约 958）
//   正确性：与 infer.c（CPU 参考，同模型同 prompt 贪心解码）对照：qwen3-1b7/4b 与
//           nano-168m 前 48/32 token 完全一致；其余前 11-44 一致（其后为浮点规约
//           顺序差异下的 argmax 近平局翻转，两侧输出均连贯，属预期）。

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <float.h>
#include <cmath>
#include <clocale>
#include <cwchar>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

#include <cub/cub.cuh>

#include "tokenizer.h"
#include "../utils.h" // NANO 架构分词器需要 Map/Trie（链接上层 utils.c）

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}

// ===============================================================================
// 模型配置与常量
// ===============================================================================

#define QUANT_TYPE_Q80 (0x80)
#define LLM_ARCH_QWEN3 (3)
#define LLM_ARCH_NANO  (0)

#define GROUP_SIZE   (128)   // Q80 量化分组大小（本引擎固定）
#define HEAD_DIM     (128)   // Qwen3 head_dim（qknorm/rope/attention 融合内核固定）

// Qwen3 特殊词元
#define QWEN_EOS_1   (151643)
#define QWEN_EOS_2   (151645)

typedef struct {
    uint32_t block_size;
    uint32_t vocab_size;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_kv_head;
    uint32_t n_hidden;
    uint32_t is_shared_classifier;
    uint32_t head_dim;
    uint32_t arch;
    uint32_t quant_type;
    uint32_t group_size;
} ModelConfig;

// ===============================================================================
// 精度抽象：权重视图（为其他精度预留的扩展口）
// ===============================================================================

typedef enum {
    WEIGHT_Q80 = 0,   // int8 值 + 每 GROUP_SIZE 一个 fp32 scale
    // WEIGHT_F32,    // 预留：fp32 权重
    // WEIGHT_F16,    // 预留：fp16 权重
} WeightType;

// Q80 权重视图：一段 int8 值 + 一段 fp32 scale（每组 GROUP_SIZE 个元素一个 scale）
typedef struct {
    const int8_t *q;
    const float  *s;
} Q80View;

// 统一权重张量描述：type 决定如何解释 view。新增精度时在 gemv_dispatch 中分发。
typedef struct {
    WeightType type;
    Q80View q80;   // type == WEIGHT_Q80 时有效
    int n;         // 输入维（列数，= 每行元素数）
    int d;         // 输出维（行数）
} WeightMat;

// ===============================================================================
// 设备端权重与运行状态
// ===============================================================================

typedef struct {
    // 词嵌入（vocab, embd）；共享分类器时同时充当 lm_head
    int8_t *emb_q;  float *emb_s;
    // 非共享分类器（vocab, embd），共享时为 NULL
    int8_t *cls_q;  float *cls_s;

    float *rms_attn;   // (L, embd)
    float *rms_ffn;    // (L, embd)
    float *rms_final;  // (embd,)

    int8_t *wqkv_q;   float *wqkv_s;  // (L, q_dim+2*kv_dim, embd)  按层拼接 q|k|v
    int8_t *wo_q;     float *wo_s;    // (L, embd, q_dim)
    int8_t *w13_q;    float *w13_s;   // (L, 2*n_hidden, embd)      按层拼接 w1|w3
    int8_t *w2_q;     float *w2_s;    // (L, embd, n_hidden)

    float *q_norm;    // (L, head_dim)（QWEN3 专用，NANO 为 NULL）
    float *k_norm;    // (L, head_dim)（QWEN3 专用，NANO 为 NULL）
    float *inv_freq;  // (head_dim/2,) RoPE 逆频率（QWEN3 运行时计算）
    float *fcr_tab;   // (block_size, head_dim/2) RoPE 余弦表（NANO 从文件读取）
    float *fci_tab;   // (block_size, head_dim/2) RoPE 正弦表（NANO 从文件读取）
} DevWeights;

// LoRA 外挂权重（仅 NANO 架构）：fp32 低秩分解，wq/wk/wv/wo 各一对 A(down)/B(up)
typedef struct {
    int rank;
    float alpha;
    float *wq_a, *wq_b;  // (L, rank, E) (L, E, rank)
    float *wk_a, *wk_b;  // (L, rank, E) (L, KV, rank)
    float *wv_a, *wv_b;  // (L, rank, E) (L, KV, rank)
    float *wo_a, *wo_b;  // (L, rank, Q) (L, E, rank)
} DevLora;

typedef struct {
    int   *d_step;     // [2] = {token, pos}，每步由主机写入
    int   *d_out_ids;  // (max_seq+1,) 序列词元（复读惩罚用），由 graph 内首节点写入
    float *x;          // (embd,) 残差流
    int8_t *xq;        // (embd,) 量化激活
    float *xs;         // (embd/GS,)
    float *qkv;        // (q_dim+2*kv_dim,) q|k|v 暂存
    float *xb_f;       // (embd,) attn-rmsnorm 浮点输出（LoRA 输入，use_lora 时分配）
    float *xba_f;      // (q_dim,) 注意力浮点输出（NANO 架构使用，量化前）
    float *k_cache;    // (L, max_seq, kv_dim)
    float *v_cache;
    float *att;        // (n_head, max_seq) 注意力分数暂存
    int8_t *xba_q;     // (q_dim,) 注意力输出（量化）
    float *xba_s;      // (q_dim/GS,)
    float *h13;        // (2*n_hidden,) w1|w3 输出
    int8_t *hq;        // (n_hidden,) SwiGLU 输出（量化）
    float *hs;         // (n_hidden/GS,)
    float *logits;     // (vocab,)
    int   *d_next;     // 采样/argmax 输出的下一个词元
    float *pval;       // argmax/max 规约部分和（512,）
    int   *pidx;
    float *d_red;      // [2] = {max, sum} 采样用规约标量
    int   *d_ids;      // (vocab,) 0..V-1，排序的 value 数组
    float *d_sorted_probs;  int *d_sorted_ids;
    void  *cub_temp;   size_t cub_temp_bytes;
    float *d_probs;    // (vocab,) exp 后的未归一化概率
} RunState;

// ===============================================================================
// 设备端工具：块内规约
// ===============================================================================

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}

// 块内求和规约。red 为 32 个 float 的共享数组。所有线程须一致调用。
__device__ float block_reduce_sum(float v, float *red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int nw = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        v = (lane < nw) ? red[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    float out = red[0];
    __syncthreads();
    return out;
}

__device__ float block_reduce_max(float v, float *red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int nw = (blockDim.x + 31) >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        v = (lane < nw) ? red[lane] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    float out = red[0];
    __syncthreads();
    return out;
}

// 4 个 int8 打包为一个 int32（little-endian，与 dp4a 的字节序约定一致）
__device__ __forceinline__ int pack_i8x4(int q0, int q1, int q2, int q3) {
    return (q0 & 0xff) | ((q1 & 0xff) << 8) | ((q2 & 0xff) << 16) | ((q3 & 0xff) << 24);
}

// ===============================================================================
// 融合 kernel：RMSNorm + Q80 量化
//   输入 x (E,)、权重 w (E,)；输出量化激活 xq 与分组 scale xs。
//   256 线程；每个 warp 处理一个 128 元素分组（E 须为 128 的倍数）。
// ===============================================================================
//   xf 非空时同时写出归一化后的浮点值（LoRA 分支需要浮点激活）
__global__ void rmsnorm_quant_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                     int8_t *__restrict__ xq, float *__restrict__ xs, int E,
                                     float *__restrict__ xf) {
    __shared__ float red[32];
    int tid = threadIdx.x;
    int lane = tid & 31, warp = tid >> 5;

    // 求平方和
    float ss = 0.0f;
    for (int i = tid * 4; i < E; i += blockDim.x * 4) {
        float4 v = *(const float4 *)(x + i);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = block_reduce_sum(ss, red);
    float inv_rms = rsqrtf(ss / (float)E + 1e-5f);

    // 每个 warp 处理一个分组：lane 处理 4 个连续元素
    int groups = E / GROUP_SIZE;
    int warps = blockDim.x >> 5;
    for (int g = warp; g < groups; g += warps) {
        int i = g * GROUP_SIZE + lane * 4;
        float4 v = *(const float4 *)(x + i);
        float4 wv = *(const float4 *)(w + i);
        float n0 = v.x * inv_rms * wv.x;
        float n1 = v.y * inv_rms * wv.y;
        float n2 = v.z * inv_rms * wv.z;
        float n3 = v.w * inv_rms * wv.w;
        if (xf) { float4 o = make_float4(n0, n1, n2, n3); *(float4 *)(xf + i) = o; }
        float amax = fmaxf(fmaxf(fabsf(n0), fabsf(n1)), fmaxf(fabsf(n2), fabsf(n3)));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax; // amax>0；为零时整组元素均为0，q=0 即可（防御）
        int q0 = (amax > 0.0f) ? __float2int_rn(n0 * inv_scale) : 0;
        int q1 = (amax > 0.0f) ? __float2int_rn(n1 * inv_scale) : 0;
        int q2 = (amax > 0.0f) ? __float2int_rn(n2 * inv_scale) : 0;
        int q3 = (amax > 0.0f) ? __float2int_rn(n3 * inv_scale) : 0;
        *(int *)(xq + i) = pack_i8x4(q0, q1, q2, q3);
        if (lane == 0) xs[g] = scale;
    }
}

// ===============================================================================
// 核心 kernel：Q80 GEMV（W(d,n) @ x(n,) -> out(d,)），dp4a 加速
//   - 权重与激活均为 int8 + 每 128 元素一个 fp32 scale（对称量化）
//   - 每个 warp 负责一行；lane 跨步读取 16 字节（int4），warp 每轮覆盖 512 字节
//   - 每轮在组内用 dp4a 做 int32 精确累加，然后乘分组 scale 浮点累加
//   - N 为编译期模板参数（1024/2048/3072），循环完全展开
//   - residual 非空时：out[row] = residual[row] + dot（残差融合）
// ===============================================================================

// 核心 kernel：Q80 GEMV（W(d,n) @ x(n,) -> out(d,)），dp4a 加速
//   - 权重与激活均为 int8 + 每 128 元素一个 fp32 scale（对称量化）
//   - 每 warp 负责 RPW 行；lane 跨步读取 16 字节（int4），dp4a 组内 int32 精确累加
//   - 调优参数（RTX 4070 Laptop 冷缓存实测选定）：
//       RPW   每 warp 行数（提高 ILP）
//       SPLIT split-N 路数；>1 时以 atomicAdd 累加，out 须已含残差或预先清零
//       LDCS  权重经 __ldcs 流式加载（只读一次，避免污染 L2）
//       BT    块线程数
template <int N, int RPW, int SPLIT, int LDCS, int BT>
__global__ void gemv_q80_kernel(const int8_t *__restrict__ wq, const float *__restrict__ ws,
                                const int8_t *__restrict__ xq, const float *__restrict__ xs,
                                float *__restrict__ out, float *__restrict__ residual, int d) {
    constexpr int NC = (N + SPLIT - 1) / SPLIT;  // 每 warp 负责的列数
    constexpr int R = (NC + 511) / 512;          // 512B 轮数
    constexpr int NGRP = N / GROUP_SIZE;

    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int part = blockIdx.y;
    int row0 = (blockIdx.x * (BT / 32) + warp) * RPW;
    if (row0 >= d) return;
    int col0 = part * NC;

    float acc[RPW];
    #pragma unroll
    for (int i = 0; i < RPW; i++) acc[i] = 0.0f;

    #pragma unroll
    for (int r = 0; r < R; r++) {
        int off_local = r * 512 + lane * 16;
        if (off_local >= NC) break;
        int off = col0 + off_local;
        int g = off >> 7; // 16 | GROUP_SIZE，每个 int4 完整落在一个分组内
        int4 x4 = *(const int4 *)(xq + off);
        float xs_g = xs[g];
        #pragma unroll
        for (int i = 0; i < RPW; i++) {
            int row = row0 + i;
            if (row >= d) break;
            const int8_t *wrow = wq + (size_t)row * N;
            const float *srow = ws + (size_t)row * NGRP;
            int4 w4 = LDCS ? __ldcs((const int4 *)(wrow + off)) : *(const int4 *)(wrow + off);
            int iv = __dp4a(w4.x, x4.x, 0);
            iv = __dp4a(w4.y, x4.y, iv);
            iv = __dp4a(w4.z, x4.z, iv);
            iv = __dp4a(w4.w, x4.w, iv);
            acc[i] = fmaf((float)iv, srow[g] * xs_g, acc[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < RPW; i++) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[i] += __shfl_down_sync(0xffffffffu, acc[i], o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < RPW; i++) {
            int row = row0 + i;
            if (row >= d) break;
            if (SPLIT > 1) atomicAdd(out + row, acc[i]);             // out 已含残差/已清零
            else out[row] = (residual ? residual[row] : 0.0f) + acc[i];
        }
    }
}

// 调用点形状标识：每种层矩阵使用实测最优的调优参数组合（按 n 进一步分派）
typedef enum {
    GEMV_GENERIC = 0,  // 保守默认（warp/row，无 split/ldcs）
    GEMV_QKV,          // (q+2k)xE    ：split2 + ldcs（调用方须先清零 out）
    GEMV_WO,           // E x q_dim   ：深 split + ldcs + atomicAdd（out 已含残差）
    GEMV_W13,          // 2H x E      ：128 线程块 + ldcs
    GEMV_W2,           // E x H       ：split + ldcs + atomicAdd（out 已含残差）
    GEMV_CLS,          // vocab x E   ：2row + ldcs
} GemvKind;

#define GEMV_LAUNCH(NN, RPW, SP, LD, BT, GY) \
    gemv_q80_kernel<NN, RPW, SP, LD, BT><<<dim3(((unsigned)w->d + (BT) / 32 * (RPW) - 1) / ((BT) / 32 * (RPW)), (GY)), (BT), 0, stream>>>( \
        w->q80.q, w->q80.s, xq, xs, out, residual, w->d)

// 精度分发器：前向流程只面对 WeightMat，不感知具体精度（扩展口）
//   各调用点的调优参数在 RTX 4070 Laptop 上按 0.6B 形状实测选定；新形状（1.7B/4B/NANO）
//   使用同族启发式配置（split 数按矩阵规模），如更换 GPU 可重新微调。
static void gemv_dispatch(const WeightMat *w, GemvKind kind, const int8_t *xq, const float *xs,
                          float *out, float *residual, cudaStream_t stream) {
    if (w->type != WEIGHT_Q80) {
        fprintf(stderr, "gemv_dispatch: unsupported weight type %d\n", (int)w->type);
        exit(EXIT_FAILURE);
    }
    switch (kind) {
        case GEMV_QKV:
            if (w->n == 512)  { GEMV_LAUNCH(512, 1, 1, 1, 256, 1); return; }
            if (w->n == 768)  { GEMV_LAUNCH(768, 1, 2, 1, 256, 2); return; }
            if (w->n == 1024) { GEMV_LAUNCH(1024, 1, 2, 1, 256, 2); return; }
            if (w->n == 2048) { GEMV_LAUNCH(2048, 1, 2, 1, 256, 2); return; }
            if (w->n == 2560) { GEMV_LAUNCH(2560, 1, 2, 1, 256, 2); return; }
            break;
        case GEMV_WO:
            if (w->n == 512)  { GEMV_LAUNCH(512, 1, 2, 1, 128, 2); return; }
            if (w->n == 768)  { GEMV_LAUNCH(768, 1, 4, 1, 128, 4); return; }
            if (w->n == 2048) { GEMV_LAUNCH(2048, 1, 8, 1, 128, 8); return; }
            if (w->n == 4096) { GEMV_LAUNCH(4096, 1, 4, 1, 128, 4); return; }
            break;
        case GEMV_W13:
            if (w->n == 512)  { GEMV_LAUNCH(512, 1, 1, 1, 128, 1); return; }
            if (w->n == 768)  { GEMV_LAUNCH(768, 1, 1, 1, 128, 1); return; }
            if (w->n == 1024) { GEMV_LAUNCH(1024, 1, 1, 1, 128, 1); return; }
            if (w->n == 2048) { GEMV_LAUNCH(2048, 1, 1, 1, 128, 1); return; }
            if (w->n == 2560) { GEMV_LAUNCH(2560, 1, 1, 1, 128, 1); return; }
            break;
        case GEMV_W2:
            if (w->n == 1408) { GEMV_LAUNCH(1408, 1, 2, 1, 256, 2); return; }
            if (w->n == 2048) { GEMV_LAUNCH(2048, 1, 4, 1, 256, 4); return; }
            if (w->n == 3072) { GEMV_LAUNCH(3072, 1, 3, 1, 256, 3); return; }
            if (w->n == 6144) { GEMV_LAUNCH(6144, 1, 3, 1, 256, 3); return; }
            if (w->n == 9728) { GEMV_LAUNCH(9728, 1, 2, 1, 256, 2); return; }
            break;
        case GEMV_CLS:
            if (w->n == 512)  { GEMV_LAUNCH(512, 2, 1, 1, 256, 1); return; }
            if (w->n == 768)  { GEMV_LAUNCH(768, 2, 1, 1, 256, 1); return; }
            if (w->n == 1024) { GEMV_LAUNCH(1024, 2, 1, 1, 256, 1); return; }
            if (w->n == 2048) { GEMV_LAUNCH(2048, 2, 1, 1, 256, 1); return; }
            if (w->n == 2560) { GEMV_LAUNCH(2560, 2, 1, 1, 256, 1); return; }
            break;
        case GEMV_GENERIC:
        default:
            break;
    }
    switch (w->n) {
        case 1024: GEMV_LAUNCH(1024, 1, 1, 0, 256, 1); return;
        case 2048: GEMV_LAUNCH(2048, 1, 1, 0, 256, 1); return;
        case 2560: GEMV_LAUNCH(2560, 1, 1, 0, 256, 1); return;
        case 3072: GEMV_LAUNCH(3072, 1, 1, 0, 256, 1); return;
        case 4096: GEMV_LAUNCH(4096, 1, 1, 0, 256, 1); return;
        case 6144: GEMV_LAUNCH(6144, 1, 1, 0, 256, 1); return;
        case 9728: GEMV_LAUNCH(9728, 1, 1, 0, 256, 1); return;
        default:
            fprintf(stderr, "gemv_dispatch: unsupported n=%d for Q80\n", w->n);
            exit(EXIT_FAILURE);
    }
}

// ===============================================================================
// kernel：词元 id 写入序列（复读惩罚的历史依据）
// ===============================================================================

__global__ void write_id_kernel(const int *__restrict__ d_step, int *__restrict__ d_out_ids) {
    d_out_ids[d_step[1]] = d_step[0];
}

// ===============================================================================
// kernel：嵌入查表（Q80 反量化一行）
// ===============================================================================

__global__ void embed_kernel(const int8_t *__restrict__ eq, const float *__restrict__ es,
                             const int *__restrict__ d_step, float *__restrict__ x, int E) {
    int tok = d_step[0];
    const int8_t *row = eq + (size_t)tok * E;
    const float *srow = es + (size_t)tok * (E / GROUP_SIZE);
    int i = threadIdx.x * 4;
    float s = srow[i >> 7];
    x[i + 0] = (float)row[i + 0] * s;
    x[i + 1] = (float)row[i + 1] * s;
    x[i + 2] = (float)row[i + 2] * s;
    x[i + 3] = (float)row[i + 3] * s;
}

// ===============================================================================
// 融合 kernel：QK-norm + RoPE + KV 写入 + 单 token 多头注意力 + 输出 Q80 量化
//   每个 block 处理一个 q 头（128 线程 = head_dim）：
//     1. 对本 q 头做 QK-norm + RoPE（结果置于共享内存，供分数计算）
//     2. 对所属 kv 头做 K-norm + RoPE 并写入 k_cache[pos]（两个 q 头块各写一次，
//        幂等同值写入）；V 行拷贝进 v_cache[pos]（同理冗余）
//     3. 因果自注意力（分数 -> 稳定 softmax -> 加权和）
//     4. 对本头 128 维输出做分组量化
//   数值与原 qknorm_rope_kernel + attention 分离实现完全一致（相同的规约顺序）。
// ===============================================================================

__global__ void attention_kernel(float *__restrict__ qkv,
                                 float *__restrict__ kc, float *__restrict__ vc,
                                 const float *__restrict__ q_norm, const float *__restrict__ k_norm,
                                 const float *__restrict__ inv_freq,
                                 float *__restrict__ att,
                                 int8_t *__restrict__ xba_q, float *__restrict__ xba_s,
                                 const int *__restrict__ d_pos,
                                 int max_seq, int n_head, int n_kv_head, int kv_mul) {
    constexpr int HD = HEAD_DIM;
    __shared__ __align__(16) float sq[HD];
    __shared__ __align__(16) float sk[HD];
    __shared__ float red[32];

    int h = blockIdx.x, tid = threadIdx.x;
    int pos = d_pos[0];
    int kvh = h / kv_mul;
    int kv_dim = n_kv_head * HD;

    // ---- 1. q 头：QK-norm + RoPE（qk-norm 权重按层传入，调用侧已加层偏移） ----
    float qv = qkv[h * HD + tid];
    float ssq = block_reduce_sum(qv * qv, red);
    sq[tid] = qv * rsqrtf(ssq / (float)HD + 1e-5f) * q_norm[tid];
    __syncthreads();
    if (tid < HD / 2) {
        float fci, fcr;
        sincosf((float)pos * inv_freq[tid], &fci, &fcr);
        float v0 = sq[tid], v1 = sq[tid + HD / 2];
        sq[tid]          = v0 * fcr - v1 * fci;
        sq[tid + HD / 2] = v1 * fcr + v0 * fci;
    }

    // ---- 2. k 头：K-norm + RoPE -> k_cache[pos]；v 行 -> v_cache[pos] ----
    float kval = qkv[(n_head + kvh) * HD + tid];
    float ssk = block_reduce_sum(kval * kval, red);
    sk[tid] = kval * rsqrtf(ssk / (float)HD + 1e-5f) * k_norm[tid];
    __syncthreads();
    float *krow = kc + (size_t)pos * kv_dim + kvh * HD;
    if (tid < HD / 2) {
        float fci, fcr;
        sincosf((float)pos * inv_freq[tid], &fci, &fcr);
        float v0 = sk[tid], v1 = sk[tid + HD / 2];
        krow[tid]          = v0 * fcr - v1 * fci;
        krow[tid + HD / 2] = v1 * fcr + v0 * fci;
    }
    vc[(size_t)pos * kv_dim + kvh * HD + tid] = qkv[(n_head + n_kv_head + kvh) * HD + tid];
    __syncthreads(); // 本块随后即读 cache[pos] 行（其它线程写入），须同步

    // ---- 3. 因果自注意力 ----
    float *att_h = att + (size_t)h * max_seq;
    const float att_scale = 0.08838834764831845f; // 1/sqrt(128)

    // 分数：q · k_t（线程各负责若干 t；行内 32 个 float4 保持 ILP，行由 L1 复用）
    const float4 *q4 = (const float4 *)sq;
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float4 *k4 = (const float4 *)(kc + (size_t)t * kv_dim + kvh * HD);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < HD / 4; j++) {
            float4 qvv = q4[j];
            float4 kvv = k4[j];
            a0 = fmaf(qvv.x, kvv.x, a0);
            a1 = fmaf(qvv.y, kvv.y, a1);
            a2 = fmaf(qvv.z, kvv.z, a2);
            a3 = fmaf(qvv.w, kvv.w, a3);
        }
        att_h[t] = ((a0 + a1) + (a2 + a3)) * att_scale;
    }
    __syncthreads();

    // softmax（数值稳定）：max -> exp -> sum
    float m = -FLT_MAX;
    for (int t = tid; t <= pos; t += blockDim.x) m = fmaxf(m, att_h[t]);
    m = block_reduce_max(m, red);

    float s = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        float e = expf(att_h[t] - m);
        att_h[t] = e;
        s += e;
    }
    s = block_reduce_sum(s, red);
    float inv_sum = 1.0f / s;

    // 加权和：线程 tid 负责输出维 tid；4 路展开保持 4 条独立加载/FMA 链（ILP）
    float acc = 0.0f;
    const float *vcol = vc + kvh * HD + tid;
    {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int t = 0;
        for (; t + 4 <= pos + 1; t += 4) {
            a0 = fmaf(att_h[t + 0], vcol[(size_t)(t + 0) * kv_dim], a0);
            a1 = fmaf(att_h[t + 1], vcol[(size_t)(t + 1) * kv_dim], a1);
            a2 = fmaf(att_h[t + 2], vcol[(size_t)(t + 2) * kv_dim], a2);
            a3 = fmaf(att_h[t + 3], vcol[(size_t)(t + 3) * kv_dim], a3);
        }
        for (; t <= pos; t++) a0 = fmaf(att_h[t], vcol[(size_t)t * kv_dim], a0);
        acc = ((a0 + a1) + (a2 + a3)) * inv_sum;
    }

    // ---- 4. 输出量化（本头 128 维为一组） ----
    float amax = block_reduce_max(fabsf(acc), red);
    float scale = amax / 127.0f;
    xba_q[h * HD + tid] = (int8_t)((amax > 0.0f) ? __float2int_rn(acc * (127.0f / amax)) : 0);
    if (tid == 0) xba_s[h] = scale;
}

// ===============================================================================
// 融合 kernel（NANO 架构）：RoPE（文件系数表、相邻对交织）+ KV 写入 +
//   单 token 多头注意力（输出浮点 xba，量化由独立 quantize_kernel 完成：
//   NANO 的 head_dim(32/48) 不等于 group_size(128)，量化分组跨越头边界，
//   无法在头内完成，与 infer.c 的全 q_dim 分组量化保持一致）。
//   每个 block 处理一个 q 头（HD 线程 = head_dim，HD 为模板参数）。
// ===============================================================================

template <int HD>
__global__ void attention_nano_kernel(float *__restrict__ qkv,
                                      float *__restrict__ kc, float *__restrict__ vc,
                                      const float *__restrict__ fcr_tab, const float *__restrict__ fci_tab,
                                      float *__restrict__ att,
                                      float *__restrict__ xba,
                                      const int *__restrict__ d_pos,
                                      int max_seq, int n_head, int n_kv_head, int kv_mul) {
    __shared__ __align__(16) float sq[HD];
    __shared__ __align__(16) float sk[HD];
    __shared__ float red[32];

    int h = blockIdx.x, tid = threadIdx.x;
    int pos = d_pos[0];
    int kvh = h / kv_mul;
    int kv_dim = n_kv_head * HD;
    constexpr int HF = HD / 2;

    // ---- q/k 头 RoPE（相邻对 (2i,2i+1) 交织，系数取自文件表）----
    const float *fcr_row = fcr_tab + (size_t)pos * HF;
    const float *fci_row = fci_tab + (size_t)pos * HF;
    sq[tid] = qkv[h * HD + tid];
    sk[tid] = qkv[(n_head + kvh) * HD + tid];
    __syncthreads();
    float *krow = kc + (size_t)pos * kv_dim + kvh * HD;
    if (tid < HF) {
        float fcr = fcr_row[tid], fci = fci_row[tid];
        float q0 = sq[2 * tid], q1 = sq[2 * tid + 1];
        float k0 = sk[2 * tid], k1 = sk[2 * tid + 1];
        sq[2 * tid]     = q0 * fcr - q1 * fci;
        sq[2 * tid + 1] = q0 * fci + q1 * fcr;
        krow[2 * tid]     = k0 * fcr - k1 * fci;
        krow[2 * tid + 1] = k0 * fci + k1 * fcr;
    }
    vc[(size_t)pos * kv_dim + kvh * HD + tid] = qkv[(n_head + n_kv_head + kvh) * HD + tid];
    __syncthreads();

    // ---- 因果自注意力（结构同 Qwen3 版）----
    float *att_h = att + (size_t)h * max_seq;
    const float att_scale = 1.0f / sqrtf((float)HD);

    const float4 *q4 = (const float4 *)sq;
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float4 *k4 = (const float4 *)(kc + (size_t)t * kv_dim + kvh * HD);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < HD / 4; j++) {
            float4 qvv = q4[j];
            float4 kvv = k4[j];
            a0 = fmaf(qvv.x, kvv.x, a0);
            a1 = fmaf(qvv.y, kvv.y, a1);
            a2 = fmaf(qvv.z, kvv.z, a2);
            a3 = fmaf(qvv.w, kvv.w, a3);
        }
        att_h[t] = ((a0 + a1) + (a2 + a3)) * att_scale;
    }
    __syncthreads();

    float m = -FLT_MAX;
    for (int t = tid; t <= pos; t += blockDim.x) m = fmaxf(m, att_h[t]);
    m = block_reduce_max(m, red);

    float s = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        float e = expf(att_h[t] - m);
        att_h[t] = e;
        s += e;
    }
    s = block_reduce_sum(s, red);
    float inv_sum = 1.0f / s;

    float acc = 0.0f;
    const float *vcol = vc + kvh * HD + tid;
    {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int t = 0;
        for (; t + 4 <= pos + 1; t += 4) {
            a0 = fmaf(att_h[t + 0], vcol[(size_t)(t + 0) * kv_dim], a0);
            a1 = fmaf(att_h[t + 1], vcol[(size_t)(t + 1) * kv_dim], a1);
            a2 = fmaf(att_h[t + 2], vcol[(size_t)(t + 2) * kv_dim], a2);
            a3 = fmaf(att_h[t + 3], vcol[(size_t)(t + 3) * kv_dim], a3);
        }
        for (; t <= pos; t++) a0 = fmaf(att_h[t], vcol[(size_t)t * kv_dim], a0);
        acc = ((a0 + a1) + (a2 + a3)) * inv_sum;
    }

    // ---- 输出浮点 xba（量化由 quantize_kernel 按 128 分组完成）----
    xba[h * HD + tid] = acc;
}

// ===============================================================================
// kernel：独立 Q80 量化（x(n,) -> xq/xs，n 为 128 的倍数；warp 负责一个分组）
// ===============================================================================

__global__ void quantize_kernel(const float *__restrict__ x,
                                int8_t *__restrict__ xq, float *__restrict__ xs, int n) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int groups = n / GROUP_SIZE;
    for (int g = warp; g < groups; g += warps) {
        int i = g * GROUP_SIZE + lane * 4;
        float4 v = *(const float4 *)(x + i);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        int q0 = (amax > 0.0f) ? __float2int_rn(v.x * inv_scale) : 0;
        int q1 = (amax > 0.0f) ? __float2int_rn(v.y * inv_scale) : 0;
        int q2 = (amax > 0.0f) ? __float2int_rn(v.z * inv_scale) : 0;
        int q3 = (amax > 0.0f) ? __float2int_rn(v.w * inv_scale) : 0;
        *(int *)(xq + i) = pack_i8x4(q0, q1, q2, q3);
        if (lane == 0) xs[g] = scale;
    }
}

__global__ void swiglu_quant_kernel(const float *__restrict__ h13,
                                    int8_t *__restrict__ hq, float *__restrict__ hs, int H) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int groups = H / GROUP_SIZE;
    for (int g = warp; g < groups; g += warps) {
        int i = g * GROUP_SIZE + lane * 4;
        float4 a = *(const float4 *)(h13 + i);
        float4 b = *(const float4 *)(h13 + H + i);
        float v0 = a.x / (1.0f + expf(-a.x)) * b.x;
        float v1 = a.y / (1.0f + expf(-a.y)) * b.y;
        float v2 = a.z / (1.0f + expf(-a.z)) * b.z;
        float v3 = a.w / (1.0f + expf(-a.w)) * b.w;
        float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        int q0 = (amax > 0.0f) ? __float2int_rn(v0 * inv_scale) : 0;
        int q1 = (amax > 0.0f) ? __float2int_rn(v1 * inv_scale) : 0;
        int q2 = (amax > 0.0f) ? __float2int_rn(v2 * inv_scale) : 0;
        int q3 = (amax > 0.0f) ? __float2int_rn(v3 * inv_scale) : 0;
        *(int *)(hq + i) = pack_i8x4(q0, q1, q2, q3);
        if (lane == 0) hs[g] = scale;
    }
}
// ===============================================================================
// LoRA kernels（仅 NANO 架构，fp32 低秩分支，与 infer.c 语义一致）
//   lora_qkv_kernel：q0/k0/v0 = A_q/k/v @ xb（E -> rank，warp 每个输出一行的点积），
//                    q1/k1/v1 = B_q/k/v @ z（rank -> E/KV/KV），乘 alpha/rank 后累加进 qkv 暂存。
//   lora_o_kernel  ：o0 = A_o @ xba（Q -> rank），o1 = B_o @ o0（rank -> E），乘 scale 累加进 x。
//   均为单块 kernel；rank <= 32（中间值置于共享内存）。
// ===============================================================================

#define LORA_MAX_RANK (32)

__global__ void lora_qkv_kernel(const float *__restrict__ xb,
                                const float *__restrict__ wa_q, const float *__restrict__ wb_q,
                                const float *__restrict__ wa_k, const float *__restrict__ wb_k,
                                const float *__restrict__ wa_v, const float *__restrict__ wb_v,
                                float *__restrict__ qkv,
                                int E, int KV, int rank, float scale) {
    __shared__ __align__(16) float z[3 * LORA_MAX_RANK];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;

    // down：3*rank 个 rank 维输出，每 warp 负责一个（对 E 做点积）
    const float *amats[3] = { wa_q, wa_k, wa_v };
    for (int o = warp; o < 3 * rank; o += nw) {
        const float *a = amats[o / rank] + (size_t)(o % rank) * E;
        float acc = 0.0f;
        for (int i = lane; i < E; i += 32) acc = fmaf(a[i], xb[i], acc);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) z[o] = acc;
    }
    __syncthreads();

    // up：q1(E) k1(KV) v1(KV)，线程各负责一个输出（对 rank 做点积）后累加进 qkv 暂存
    const float *bmats[3] = { wb_q, wb_k, wb_v };
    for (int o = threadIdx.x; o < E + 2 * KV; o += blockDim.x) {
        int m = (o < E) ? 0 : ((o < E + KV) ? 1 : 2);
        int i = (m == 0) ? o : ((m == 1) ? o - E : o - E - KV);
        const float *b = bmats[m] + (size_t)i * rank;
        const float *zrow = z + m * rank;
        float acc = 0.0f;
        for (int r = 0; r < rank; r++) acc = fmaf(b[r], zrow[r], acc);
        int off = (m == 0) ? i : (E + ((m == 1) ? 0 : KV) + i);
        qkv[off] += scale * acc;
    }
}

__global__ void lora_o_kernel(const float *__restrict__ xba,
                              const float *__restrict__ wa, const float *__restrict__ wb,
                              float *__restrict__ x,
                              int Q, int E, int rank, float scale) {
    __shared__ __align__(16) float z[LORA_MAX_RANK];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;

    for (int o = warp; o < rank; o += nw) {
        const float *a = wa + (size_t)o * Q;
        float acc = 0.0f;
        for (int i = lane; i < Q; i += 32) acc = fmaf(a[i], xba[i], acc);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) z[o] = acc;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float *b = wb + (size_t)i * rank;
        float acc = 0.0f;
        for (int r = 0; r < rank; r++) acc = fmaf(b[r], z[r], acc);
        x[i] += scale * acc;
    }
}


// ===============================================================================
// kernel：复读惩罚（对历史词元的 logits 除以惩罚因子）
// ===============================================================================

__global__ void rep_penalty_kernel(float *__restrict__ logits, const int *__restrict__ ids,
                                   const int *__restrict__ d_pos, float penalty) {
    int pos = d_pos[0];
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pos; i += stride) {
        logits[ids[i]] /= penalty;
    }
}

// ===============================================================================
// kernel：argmax（单内核）与 max 规约（采样 softmax 用）
// ===============================================================================

#define REDUCE_BLOCKS (512)


// 单内核 argmax：1 块 1024 线程，跨步扫描 + 块内 (max, 最小下标) 联合规约
__global__ void argmax_one_kernel(const float *__restrict__ x, int n, int *__restrict__ out) {
    float best = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __shared__ float sval[32];
    __shared__ int sidx[32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, best, o);
        int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) { sval[w] = best; sidx[w] = bi; }
    __syncthreads();
    if (w == 0) {
        int nw = (blockDim.x + 31) >> 5;
        best = (lane < nw) ? sval[lane] : -FLT_MAX;
        bi = (lane < nw) ? sidx[lane] : 0x7fffffff;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, best, o);
            int oi = __shfl_down_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) out[0] = bi;
    }
}

__global__ void max_p1_kernel(const float *__restrict__ x, int n, float *__restrict__ pval) {
    float best = -FLT_MAX;
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) best = fmaxf(best, x[i]);
    __shared__ float red[32];
    best = block_reduce_max(best, red);
    if (threadIdx.x == 0) pval[blockIdx.x] = best;
}

__global__ void max_p2_kernel(const float *__restrict__ pval, int n, float *__restrict__ out) {
    float best = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) best = fmaxf(best, pval[i]);
    __shared__ float red[32];
    best = block_reduce_max(best, red);
    if (threadIdx.x == 0) out[0] = best;
}

// exp((x-max)*inv_temp) 并按块求部分和
__global__ void exp_sum_kernel(const float *__restrict__ x, float *__restrict__ probs,
                               float *__restrict__ psum, const float *__restrict__ d_max,
                               float inv_temp, int n) {
    float m = d_max[0];
    float s = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float p = expf((x[i] - m) * inv_temp);
        probs[i] = p;
        s += p;
    }
    __shared__ float red[32];
    s = block_reduce_sum(s, red);
    if (threadIdx.x == 0) psum[blockIdx.x] = s;
}

__global__ void sum_p2_kernel(const float *__restrict__ psum, int n, float *__restrict__ out) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += psum[i];
    __shared__ float red[32];
    s = block_reduce_sum(s, red);
    if (threadIdx.x == 0) out[0] = s;
}

__global__ void iota_kernel(int *__restrict__ ids, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) ids[i] = i;
}

// ===============================================================================
// 引擎上下文
// ===============================================================================

typedef struct {
    ModelConfig cfg;
    int max_seq_len;
    int q_dim;      // n_head * head_dim
    int kv_dim;     // n_kv_head * head_dim
    int kv_mul;     // n_head / n_kv_head

    DevWeights w;
    RunState s;

    // CUDA Graph（decode step 全量捕获）
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;

    // 主机侧固定内存
    int *h_step;         // [2] pinned：{token, pos}
    int *h_step_stage;   // pinned：prefill 流水线暂存 [2*max_seq]
    int *h_next;         // pinned
    float *h_red;        // pinned [2] = {max, sum}
    float *h_sorted_probs; // pinned (vocab,)
    int *h_sorted_ids;     // pinned (vocab,)

    float rep_penalty;
    float temperature;
    float top_p;
    uint64_t rng_state;

    int use_lora;        // 是否启用 LoRA（仅 NANO 架构）
    DevLora lora;        // LoRA 外挂权重
    uint32_t eos1, eos2; // 结束词元（随架构：QWEN3=151643/151645，NANO=0/3）
    const char *lora_path;

    uint8_t *file_buffer;
    size_t file_size;
} CudaEngine;

// ===============================================================================
// 前向：单 token decode step 的全部 kernel 发射（供 graph 捕获 / 直接执行）
// ===============================================================================

static void forward_launch(CudaEngine *e, cudaStream_t stream) {
    ModelConfig *c = &e->cfg;
    DevWeights *w = &e->w;
    RunState *s = &e->s;

    int E = (int)c->n_embd;
    int QK = e->q_dim + 2 * e->kv_dim;
    int H2 = 2 * (int)c->n_hidden;
    int V = (int)c->vocab_size;
    int *d_pos = e->s.d_step + 1;
    int is_nano = (c->arch == LLM_ARCH_NANO);

    write_id_kernel<<<1, 1, 0, stream>>>(s->d_step, s->d_out_ids);
    embed_kernel<<<1, E / 4, 0, stream>>>(w->emb_q, w->emb_s, s->d_step, s->x, E);

    WeightMat mat;
    mat.type = WEIGHT_Q80;

    for (uint32_t l = 0; l < c->n_layer; l++) {
        // attention rmsnorm + 量化
        rmsnorm_quant_kernel<<<1, 256, 0, stream>>>(s->x, w->rms_attn + l * E, s->xq, s->xs, E, e->use_lora ? s->xb_f : NULL);

        // QKV
        mat.q80.q = w->wqkv_q + (size_t)l * QK * E;
        mat.n = E; mat.d = QK;
        mat.q80.s = w->wqkv_s + (size_t)l * QK * (E / GROUP_SIZE);
        cudaMemsetAsync(s->qkv, 0, (size_t)QK * sizeof(float), stream); // split2 以 atomicAdd 累加，须先清零
        gemv_dispatch(&mat, GEMV_QKV, s->xq, s->xs, s->qkv, NULL, stream);
        // 注意力（QWEN3：融合 QK-norm+RoPE+输出量化；NANO：文件表 RoPE+浮点 xba 输出）
        if (!is_nano) {
            attention_kernel<<<c->n_head, HEAD_DIM, 0, stream>>>(
                s->qkv, s->k_cache + (size_t)l * e->max_seq_len * e->kv_dim,
                s->v_cache + (size_t)l * e->max_seq_len * e->kv_dim,
                w->q_norm + l * HEAD_DIM, w->k_norm + l * HEAD_DIM, w->inv_freq,
                s->att, s->xba_q, s->xba_s, d_pos, e->max_seq_len,
                (int)c->n_head, (int)c->n_kv_head, e->kv_mul);
        }
        else {
            if (e->use_lora) {
                lora_qkv_kernel<<<1, 256, 0, stream>>>(
                    s->xb_f,
                    e->lora.wq_a + (size_t)l * e->lora.rank * E, e->lora.wq_b + (size_t)l * E * e->lora.rank,
                    e->lora.wk_a + (size_t)l * e->lora.rank * E, e->lora.wk_b + (size_t)l * e->kv_dim * e->lora.rank,
                    e->lora.wv_a + (size_t)l * e->lora.rank * E, e->lora.wv_b + (size_t)l * e->kv_dim * e->lora.rank,
                    s->qkv, E, e->kv_dim, e->lora.rank, e->lora.alpha / (float)e->lora.rank);
            }
            int hd = (int)c->head_dim;
            if (hd == 32) {
                attention_nano_kernel<32><<<c->n_head, 32, 0, stream>>>(
                    s->qkv, s->k_cache + (size_t)l * e->max_seq_len * e->kv_dim,
                    s->v_cache + (size_t)l * e->max_seq_len * e->kv_dim,
                    w->fcr_tab, w->fci_tab, s->att, s->xba_f, d_pos, e->max_seq_len,
                    (int)c->n_head, (int)c->n_kv_head, e->kv_mul);
            }
            else if (hd == 48) {
                attention_nano_kernel<48><<<c->n_head, 48, 0, stream>>>(
                    s->qkv, s->k_cache + (size_t)l * e->max_seq_len * e->kv_dim,
                    s->v_cache + (size_t)l * e->max_seq_len * e->kv_dim,
                    w->fcr_tab, w->fci_tab, s->att, s->xba_f, d_pos, e->max_seq_len,
                    (int)c->n_head, (int)c->n_kv_head, e->kv_mul);
            }
            quantize_kernel<<<1, 256, 0, stream>>>(s->xba_f, s->xba_q, s->xba_s, e->q_dim);
        }

        // 输出投影 + 残差
        mat.q80.q = w->wo_q + (size_t)l * E * e->q_dim;
        mat.q80.s = w->wo_s + (size_t)l * E * (e->q_dim / GROUP_SIZE);
        mat.n = e->q_dim; mat.d = E;
        gemv_dispatch(&mat, GEMV_WO, s->xba_q, s->xba_s, s->x, s->x, stream);
        if (is_nano && e->use_lora) {
            lora_o_kernel<<<1, 256, 0, stream>>>(
                s->xba_f, e->lora.wo_a + (size_t)l * e->lora.rank * e->q_dim,
                e->lora.wo_b + (size_t)l * E * e->lora.rank,
                s->x, e->q_dim, E, e->lora.rank, e->lora.alpha / (float)e->lora.rank);
        }
        // ffn rmsnorm + 量化
        rmsnorm_quant_kernel<<<1, 256, 0, stream>>>(s->x, w->rms_ffn + l * E, s->xq, s->xs, E, NULL);

        // W1|W3
        mat.q80.q = w->w13_q + (size_t)l * H2 * E;
        mat.q80.s = w->w13_s + (size_t)l * H2 * (E / GROUP_SIZE);
        mat.n = E; mat.d = H2;
        gemv_dispatch(&mat, GEMV_W13, s->xq, s->xs, s->h13, NULL, stream);

        // SwiGLU + 量化
        swiglu_quant_kernel<<<1, 256, 0, stream>>>(s->h13, s->hq, s->hs, (int)c->n_hidden);

        // W2 + 残差
        mat.q80.q = w->w2_q + (size_t)l * E * (int)c->n_hidden;
        mat.q80.s = w->w2_s + (size_t)l * E * ((int)c->n_hidden / GROUP_SIZE);
        mat.n = (int)c->n_hidden; mat.d = E;
        gemv_dispatch(&mat, GEMV_W2, s->hq, s->hs, s->x, s->x, stream);
    }

    // 最终 rmsnorm + 量化
    rmsnorm_quant_kernel<<<1, 256, 0, stream>>>(s->x, w->rms_final, s->xq, s->xs, E, NULL);

    // 分类器
    mat.q80.q = w->cls_q ? w->cls_q : w->emb_q;
    mat.q80.s = w->cls_s ? w->cls_s : w->emb_s;
    mat.n = E; mat.d = V;
    gemv_dispatch(&mat, GEMV_CLS, s->xq, s->xs, s->logits, NULL, stream);

    // 复读惩罚（penalty==1.0 时在捕获期剔除该节点）
    if (e->rep_penalty != 1.0f) {
        rep_penalty_kernel<<<4, 256, 0, stream>>>(s->logits, s->d_out_ids, d_pos, e->rep_penalty);
    }
}

// ===============================================================================
// 采样
// ===============================================================================

// argmax（GPU）：结果写入 d_next
static void launch_argmax(CudaEngine *e) {
    int V = (int)e->cfg.vocab_size;
    argmax_one_kernel<<<1, 1024, 0, e->stream>>>(e->s.logits, V, e->s.d_next);
}

// 温度 + top-p 采样（GPU softmax + 排序，主机截断）：结果写入 h_next
static void launch_sampling(CudaEngine *e) {
    int V = (int)e->cfg.vocab_size;
    RunState *s = &e->s;
    float inv_temp = 1.0f / e->temperature;
    max_p1_kernel<<<REDUCE_BLOCKS, 256, 0, e->stream>>>(s->logits, V, s->pval);
    max_p2_kernel<<<1, 256, 0, e->stream>>>(s->pval, REDUCE_BLOCKS, s->d_red);
    exp_sum_kernel<<<REDUCE_BLOCKS, 256, 0, e->stream>>>(s->logits, s->d_probs, s->pval, s->d_red, inv_temp, V);
    sum_p2_kernel<<<1, 256, 0, e->stream>>>(s->pval, REDUCE_BLOCKS, s->d_red + 1);
    cub::DeviceRadixSort::SortPairsDescending(s->cub_temp, s->cub_temp_bytes,
                                              s->d_probs, s->d_sorted_probs,
                                              s->d_ids, s->d_sorted_ids,
                                              V, 0, 32, e->stream);
}

static uint32_t rng_next_u32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * 2685821657736338717ull) >> 32);
}

static float rng_next_f32(uint64_t *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

// 主机端 top-p 截断 + 轮盘（输入为 GPU 排序后的未归一化概率，取前 n_copied 个）
//   返回采样词元 id；若 n_copied 个词元未覆盖 top_p 概率质量（尾部不足），返回 -1，
//   调用方需全量拷贝后重试（top_p 采样语义与原 infer.c sample_top_p 一致）。
static int sample_top_p_host(CudaEngine *e, int n_copied, float coin) {
    double total = (double)e->h_red[1];
    double threshold = (double)e->top_p * total;
    double cum = 0.0;
    int last = -1;
    for (int i = 0; i < n_copied; i++) {
        cum += (double)e->h_sorted_probs[i];
        last = i;
        if (cum > threshold) break;
    }
    if (cum <= threshold && n_copied < (int)e->cfg.vocab_size) return -1; // 截断点不在前 n_copied 内
    double r = (double)coin * cum;
    double cdf = 0.0;
    for (int i = 0; i <= last; i++) {
        cdf += (double)e->h_sorted_probs[i];
        if (r < cdf) return e->h_sorted_ids[i];
    }
    return e->h_sorted_ids[last];
}

// 每步采样：graph 之后在流上追加采样 kernel，同步取回下一个词元。
//   temp==0 走 GPU argmax；temp>0 走 GPU softmax + cub 排序 + 主机 top-p 截断。
//   top-p 只拷回概率最高的前 SAMPLE_TOPK 个（覆盖 top_p 质量的实际充分条件）；
//   极端平坦分布下不足覆盖时回退为全量拷贝（与原实现语义一致）。
#define SAMPLE_TOPK (4096)
static int engine_sample(CudaEngine *e) {
    RunState *s = &e->s;
    int V = (int)e->cfg.vocab_size;
    if (e->temperature == 0.0f) {
        launch_argmax(e);
        CUDA_CHECK(cudaMemcpyAsync(e->h_next, s->d_next, sizeof(int), cudaMemcpyDeviceToHost, e->stream));
        CUDA_CHECK(cudaStreamSynchronize(e->stream));
        return e->h_next[0];
    }
    launch_sampling(e);
    CUDA_CHECK(cudaMemcpyAsync(e->h_red, s->d_red, 2 * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
    int n_copy = (V < SAMPLE_TOPK) ? V : SAMPLE_TOPK;
    CUDA_CHECK(cudaMemcpyAsync(e->h_sorted_probs, s->d_sorted_probs, (size_t)n_copy * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
    CUDA_CHECK(cudaMemcpyAsync(e->h_sorted_ids, s->d_sorted_ids, (size_t)n_copy * sizeof(int), cudaMemcpyDeviceToHost, e->stream));
    CUDA_CHECK(cudaStreamSynchronize(e->stream));
    float coin = rng_next_f32(&e->rng_state);
    int tok = sample_top_p_host(e, n_copy, coin);
    if (tok < 0) { // 罕见回退：全量拷贝后重采样（同一枚 coin，语义一致）
        CUDA_CHECK(cudaMemcpyAsync(e->h_sorted_probs, s->d_sorted_probs, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
        CUDA_CHECK(cudaMemcpyAsync(e->h_sorted_ids, s->d_sorted_ids, (size_t)V * sizeof(int), cudaMemcpyDeviceToHost, e->stream));
        CUDA_CHECK(cudaStreamSynchronize(e->stream));
        tok = sample_top_p_host(e, V, coin);
    }
    return tok;
}

// ===============================================================================
// 模型加载
// ===============================================================================

// 读取 Q80 张量（n 个，每个 size_each 元素），推进指针
static const uint8_t *walk_q80(const uint8_t *p, int n, int size_each, int gs) {
    return p + (size_t)n * ((size_t)size_each + (size_t)(size_each / gs) * sizeof(float));
}

static void engine_load_lora(CudaEngine *e, const char *lora_path);
static void engine_load(CudaEngine *e, const char *model_path, int max_seq_len) {
    // 读文件（mmap）
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "无法打开模型文件 %s\n", model_path); exit(EXIT_FAILURE); }
    e->file_size = (size_t)lseek(fd, 0, SEEK_END);
    uint8_t *buf = (uint8_t *)mmap(NULL, e->file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (buf == MAP_FAILED) { fprintf(stderr, "mmap 失败\n"); exit(EXIT_FAILURE); }
    e->file_buffer = buf;

    // 解析文件头（与 infer.c parse_model_file 一致）
    uint32_t *header = (uint32_t *)buf;
    ModelConfig *c = &e->cfg;
    c->block_size      = header[6];
    c->vocab_size      = header[7];
    c->n_layer         = header[8];
    c->n_embd          = header[9];
    c->n_head          = header[10];
    c->n_kv_head       = header[11];
    c->n_hidden        = header[12];
    c->is_shared_classifier = header[13];
    c->head_dim        = header[14];
    c->arch            = header[4];
    c->quant_type      = header[15];
    c->group_size      = header[16];

    // 结构约束检查
    if (c->arch != LLM_ARCH_QWEN3 && c->arch != LLM_ARCH_NANO) {
        fprintf(stderr, "仅支持 Qwen3/Nano 架构（arch=%u）\n", c->arch); exit(EXIT_FAILURE);
    }
    int is_nano = (c->arch == LLM_ARCH_NANO);
    if (c->n_head % c->n_kv_head != 0) {
        fprintf(stderr, "n_head(%u) 须为 n_kv_head(%u) 的整数倍\n", c->n_head, c->n_kv_head); exit(EXIT_FAILURE);
    }
    if (c->quant_type != QUANT_TYPE_Q80) { fprintf(stderr, "仅支持 Q80 量化（quant=%u）\n", c->quant_type); exit(EXIT_FAILURE); }
    if (c->group_size != GROUP_SIZE) { fprintf(stderr, "仅支持 group_size=%d（实际 %u）\n", GROUP_SIZE, c->group_size); exit(EXIT_FAILURE); }
    if (!is_nano && c->head_dim != HEAD_DIM) { fprintf(stderr, "Qwen3 仅支持 head_dim=%d（实际 %u）\n", HEAD_DIM, c->head_dim); exit(EXIT_FAILURE); }
    if (c->n_embd % GROUP_SIZE || (c->n_head * c->head_dim) % GROUP_SIZE || c->n_hidden % GROUP_SIZE) {
        fprintf(stderr, "维度须为 %d 的倍数（embd=%u q_dim=%u hidden=%u）\n",
                GROUP_SIZE, c->n_embd, c->n_head * c->head_dim, c->n_hidden); exit(EXIT_FAILURE);
    }
    if (is_nano && c->head_dim != 32 && c->head_dim != 48) {
        fprintf(stderr, "NANO 架构当前支持 head_dim 32/48（实际 %u）\n", c->head_dim); exit(EXIT_FAILURE);
    }
    if ((int)c->block_size < max_seq_len) {
        fprintf(stderr, "[CUDA] max_seq_len(%d) 超过模型 block_size(%u)，已截断\n", max_seq_len, c->block_size);
        max_seq_len = (int)c->block_size;
    }
    if (!c->is_shared_classifier) {
        fprintf(stderr, "仅支持共享分类器的模型\n"); exit(EXIT_FAILURE);
    }
    if (e->use_lora && !is_nano) {
        fprintf(stderr, "LoRA 仅支持 NANO 架构模型\n"); exit(EXIT_FAILURE);
    }
    e->eos1 = is_nano ? 0 : QWEN_EOS_1;
    e->eos2 = is_nano ? 3 : QWEN_EOS_2;

    e->max_seq_len = max_seq_len;
    e->q_dim = (int)(c->n_head * c->head_dim);
    e->kv_dim = (int)(c->n_kv_head * c->head_dim);
    e->kv_mul = (int)(c->n_head / c->n_kv_head);

    int E = (int)c->n_embd, L = (int)c->n_layer, V = (int)c->vocab_size;
    int Q = e->q_dim, KV = e->kv_dim, H = (int)c->n_hidden;
    int GS = GROUP_SIZE;

    // 词表/分词器偏移
    uint32_t tokenizer_field_bytes = *(uint32_t *)(buf + 256);
    const uint8_t *pp = buf + 256 + tokenizer_field_bytes;

    // ---------------- 解析参数区（与 memory_map_params 顺序一致） ----------------
    const float *f_rms_attn  = (const float *)pp;  pp += (size_t)L * E * 4;
    const float *f_rms_ffn   = (const float *)pp;  pp += (size_t)L * E * 4;
    const float *f_rms_final = (const float *)pp;  pp += (size_t)E * 4;

    const uint8_t *q_emb = pp;  pp = walk_q80(pp, 1, V * E, GS);
    const int8_t *emb_q = (const int8_t *)q_emb;
    const float *emb_s = (const float *)(q_emb + (size_t)V * E);

    const uint8_t *p_wq = pp;  pp = walk_q80(pp, L, Q * E, GS);
    const uint8_t *p_wk = pp;  pp = walk_q80(pp, L, KV * E, GS);
    const uint8_t *p_wv = pp;  pp = walk_q80(pp, L, KV * E, GS);
    const uint8_t *p_wo = pp;  pp = walk_q80(pp, L, E * Q, GS);
    const uint8_t *p_w1 = pp;  pp = walk_q80(pp, L, H * E, GS);
    const uint8_t *p_w2 = pp;  pp = walk_q80(pp, L, E * H, GS);
    const uint8_t *p_w3 = pp;  pp = walk_q80(pp, L, H * E, GS);

    // QWEN3：QK-norm 权重；文件其后的 RoPE 表跳过（运行时自行计算）
    // NANO ：无 QK-norm；文件其后即为 RoPE 频率表（读取到设备）
    const float *f_q_norm = NULL, *f_k_norm = NULL, *f_fcr = NULL, *f_fci = NULL;
    if (!is_nano) {
        f_q_norm = (const float *)pp;  pp += (size_t)L * HEAD_DIM * 4;
        f_k_norm = (const float *)pp;  pp += (size_t)L * HEAD_DIM * 4;
    }
    else {
        f_fcr = (const float *)pp;  pp += (size_t)c->block_size * (c->head_dim / 2) * 4;
        f_fci = (const float *)pp;  pp += (size_t)c->block_size * (c->head_dim / 2) * 4;
    }
    if ((size_t)(pp - buf) > e->file_size) { fprintf(stderr, "模型文件损坏：参数区越界\n"); exit(EXIT_FAILURE); }
    // ---------------- 设备内存分配 ----------------
    DevWeights *w = &e->w;
    RunState *s = &e->s;
    memset(w, 0, sizeof(*w));
    memset(s, 0, sizeof(*s));

    size_t wqkv_qb = (size_t)L * (Q + 2 * KV) * E;
    size_t wqkv_sb = (size_t)L * (Q + 2 * KV) * (E / GS) * 4;
    size_t wo_qb = (size_t)L * E * Q,          wo_sb = (size_t)L * E * (Q / GS) * 4;
    size_t w13_qb = (size_t)L * 2 * H * E,     w13_sb = (size_t)L * 2 * H * (E / GS) * 4;
    size_t w2_qb = (size_t)L * E * H,          w2_sb = (size_t)L * E * (H / GS) * 4;
    size_t emb_qb = (size_t)V * E,             emb_sb = (size_t)V * (E / GS) * 4;

    CUDA_CHECK(cudaMalloc(&w->emb_q, emb_qb));  CUDA_CHECK(cudaMalloc(&w->emb_s, emb_sb));
    CUDA_CHECK(cudaMalloc(&w->rms_attn, (size_t)L * E * 4));
    CUDA_CHECK(cudaMalloc(&w->rms_ffn, (size_t)L * E * 4));
    CUDA_CHECK(cudaMalloc(&w->rms_final, (size_t)E * 4));
    CUDA_CHECK(cudaMalloc(&w->wqkv_q, wqkv_qb));  CUDA_CHECK(cudaMalloc(&w->wqkv_s, wqkv_sb));
    CUDA_CHECK(cudaMalloc(&w->wo_q, wo_qb));      CUDA_CHECK(cudaMalloc(&w->wo_s, wo_sb));
    CUDA_CHECK(cudaMalloc(&w->w13_q, w13_qb));    CUDA_CHECK(cudaMalloc(&w->w13_s, w13_sb));
    CUDA_CHECK(cudaMalloc(&w->w2_q, w2_qb));      CUDA_CHECK(cudaMalloc(&w->w2_s, w2_sb));
    if (!is_nano) {
        CUDA_CHECK(cudaMalloc(&w->q_norm, (size_t)L * HEAD_DIM * 4));
        CUDA_CHECK(cudaMalloc(&w->k_norm, (size_t)L * HEAD_DIM * 4));
        CUDA_CHECK(cudaMalloc(&w->inv_freq, HEAD_DIM / 2 * 4));
    }
    else {
        CUDA_CHECK(cudaMalloc(&w->fcr_tab, (size_t)c->block_size * (c->head_dim / 2) * 4));
        CUDA_CHECK(cudaMalloc(&w->fci_tab, (size_t)c->block_size * (c->head_dim / 2) * 4));
        CUDA_CHECK(cudaMalloc(&s->xba_f, Q * 4));
        if (e->use_lora) CUDA_CHECK(cudaMalloc(&s->xb_f, E * 4));
    }

    size_t kv_bytes = (size_t)L * max_seq_len * KV * 4;
    CUDA_CHECK(cudaMalloc(&s->k_cache, kv_bytes));
    CUDA_CHECK(cudaMalloc(&s->v_cache, kv_bytes));
    CUDA_CHECK(cudaMalloc(&s->d_step, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->d_out_ids, (max_seq_len + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->x, E * 4));
    CUDA_CHECK(cudaMalloc(&s->xq, E));        CUDA_CHECK(cudaMalloc(&s->xs, E / GS * 4));
    CUDA_CHECK(cudaMalloc(&s->qkv, (Q + 2 * KV) * 4));
    CUDA_CHECK(cudaMalloc(&s->att, (size_t)c->n_head * max_seq_len * 4));
    CUDA_CHECK(cudaMalloc(&s->xba_q, Q));     CUDA_CHECK(cudaMalloc(&s->xba_s, Q / GS * 4));
    CUDA_CHECK(cudaMalloc(&s->h13, 2 * H * 4));
    CUDA_CHECK(cudaMalloc(&s->hq, H));        CUDA_CHECK(cudaMalloc(&s->hs, H / GS * 4));
    CUDA_CHECK(cudaMalloc(&s->logits, V * 4));
    CUDA_CHECK(cudaMalloc(&s->d_next, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s->pval, REDUCE_BLOCKS * 4));
    CUDA_CHECK(cudaMalloc(&s->pidx, REDUCE_BLOCKS * 4));
    CUDA_CHECK(cudaMalloc(&s->d_red, 2 * 4));
    CUDA_CHECK(cudaMalloc(&s->d_ids, V * 4));
    CUDA_CHECK(cudaMalloc(&s->d_probs, V * 4));
    CUDA_CHECK(cudaMalloc(&s->d_sorted_probs, V * 4));
    CUDA_CHECK(cudaMalloc(&s->d_sorted_ids, V * 4));

    // ---------------- 权重上传（按层拼接 QKV 与 W1|W3） ----------------
    CUDA_CHECK(cudaMemcpy(w->emb_q, emb_q, emb_qb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w->emb_s, emb_s, emb_sb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w->rms_attn, f_rms_attn, (size_t)L * E * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w->rms_ffn, f_rms_ffn, (size_t)L * E * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w->rms_final, f_rms_final, (size_t)E * 4, cudaMemcpyHostToDevice));
    if (!is_nano) {
        CUDA_CHECK(cudaMemcpy(w->q_norm, f_q_norm, (size_t)L * HEAD_DIM * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->k_norm, f_k_norm, (size_t)L * HEAD_DIM * 4, cudaMemcpyHostToDevice));
    }
    else {
        CUDA_CHECK(cudaMemcpy(w->fcr_tab, f_fcr, (size_t)c->block_size * (c->head_dim / 2) * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->fci_tab, f_fci, (size_t)c->block_size * (c->head_dim / 2) * 4, cudaMemcpyHostToDevice));
    }

    for (int l = 0; l < L; l++) {
        const int8_t *lq = (const int8_t *)(p_wq + (size_t)l * ((size_t)Q * E + (size_t)(Q * E / GS) * 4));
        const float  *ls = (const float *)(p_wq + (size_t)l * ((size_t)Q * E + (size_t)(Q * E / GS) * 4) + (size_t)Q * E);
        CUDA_CHECK(cudaMemcpy(w->wqkv_q + (size_t)l * (Q + 2 * KV) * E, lq, (size_t)Q * E, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->wqkv_s + (size_t)l * (Q + 2 * KV) * (E / GS), ls, (size_t)Q * (E / GS) * 4, cudaMemcpyHostToDevice));

        const int8_t *kq = (const int8_t *)(p_wk + (size_t)l * ((size_t)KV * E + (size_t)(KV * E / GS) * 4));
        const float  *ks = (const float *)(p_wk + (size_t)l * ((size_t)KV * E + (size_t)(KV * E / GS) * 4) + (size_t)KV * E);
        CUDA_CHECK(cudaMemcpy(w->wqkv_q + ((size_t)l * (Q + 2 * KV) + Q) * E, kq, (size_t)KV * E, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->wqkv_s + ((size_t)l * (Q + 2 * KV) + Q) * (E / GS), ks, (size_t)KV * (E / GS) * 4, cudaMemcpyHostToDevice));

        const int8_t *vq = (const int8_t *)(p_wv + (size_t)l * ((size_t)KV * E + (size_t)(KV * E / GS) * 4));
        const float  *vs = (const float *)(p_wv + (size_t)l * ((size_t)KV * E + (size_t)(KV * E / GS) * 4) + (size_t)KV * E);
        CUDA_CHECK(cudaMemcpy(w->wqkv_q + ((size_t)l * (Q + 2 * KV) + Q + KV) * E, vq, (size_t)KV * E, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->wqkv_s + ((size_t)l * (Q + 2 * KV) + Q + KV) * (E / GS), vs, (size_t)KV * (E / GS) * 4, cudaMemcpyHostToDevice));

        const int8_t *oq = (const int8_t *)(p_wo + (size_t)l * ((size_t)E * Q + (size_t)(E * Q / GS) * 4));
        const float  *os = (const float *)(p_wo + (size_t)l * ((size_t)E * Q + (size_t)(E * Q / GS) * 4) + (size_t)E * Q);
        CUDA_CHECK(cudaMemcpy(w->wo_q + (size_t)l * E * Q, oq, (size_t)E * Q, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->wo_s + (size_t)l * E * (Q / GS), os, (size_t)E * (Q / GS) * 4, cudaMemcpyHostToDevice));

        const int8_t *u1 = (const int8_t *)(p_w1 + (size_t)l * ((size_t)H * E + (size_t)(H * E / GS) * 4));
        const float  *s1 = (const float *)(p_w1 + (size_t)l * ((size_t)H * E + (size_t)(H * E / GS) * 4) + (size_t)H * E);
        CUDA_CHECK(cudaMemcpy(w->w13_q + (size_t)l * 2 * H * E, u1, (size_t)H * E, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->w13_s + (size_t)l * 2 * H * (E / GS), s1, (size_t)H * (E / GS) * 4, cudaMemcpyHostToDevice));

        const int8_t *u3 = (const int8_t *)(p_w3 + (size_t)l * ((size_t)H * E + (size_t)(H * E / GS) * 4));
        const float  *s3 = (const float *)(p_w3 + (size_t)l * ((size_t)H * E + (size_t)(H * E / GS) * 4) + (size_t)H * E);
        CUDA_CHECK(cudaMemcpy(w->w13_q + ((size_t)l * 2 + 1) * H * E, u3, (size_t)H * E, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->w13_s + ((size_t)l * 2 + 1) * H * (E / GS), s3, (size_t)H * (E / GS) * 4, cudaMemcpyHostToDevice));

        const int8_t *u2 = (const int8_t *)(p_w2 + (size_t)l * ((size_t)E * H + (size_t)(E * H / GS) * 4));
        const float  *s2 = (const float *)(p_w2 + (size_t)l * ((size_t)E * H + (size_t)(E * H / GS) * 4) + (size_t)E * H);
        if (getenv("NC_DEBUG")) { fprintf(stderr, "l=%d w2_q=%p u2=%p p_w2=%p size=%zu\n", l, (void*)w->w2_q, (void*)u2, (void*)p_w2, (size_t)E * H); }
        CUDA_CHECK(cudaMemcpy(w->w2_q + (size_t)l * E * H, u2, (size_t)E * H, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w->w2_s + (size_t)l * E * (H / GS), s2, (size_t)E * (H / GS) * 4, cudaMemcpyHostToDevice));
    }

    // RoPE 逆频率（QWEN3 专用）：inv_freq[i] = 1000000^(-2i/head_dim)
    if (!is_nano) {
        float h_if[HEAD_DIM / 2];
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            h_if[i] = powf(1000000.0f, -(float)(i * 2) / (float)HEAD_DIM);
        }
        CUDA_CHECK(cudaMemcpy(w->inv_freq, h_if, sizeof(h_if), cudaMemcpyHostToDevice));
    }

    // 词元 id 数组（排序 value）
    iota_kernel<<<(V + 255) / 256, 256>>>(s->d_ids, V);

    // cub 临时存储
    s->cub_temp = NULL; s->cub_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(NULL, s->cub_temp_bytes,
                                                         s->d_probs, s->d_sorted_probs,
                                                         s->d_ids, s->d_sorted_ids, V));
    CUDA_CHECK(cudaMalloc(&s->cub_temp, s->cub_temp_bytes));

    // 主机固定内存
    CUDA_CHECK(cudaHostAlloc(&e->h_step, 2 * sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&e->h_step_stage, 2 * sizeof(int) * max_seq_len, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&e->h_next, sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&e->h_red, 2 * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&e->h_sorted_probs, V * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&e->h_sorted_ids, V * sizeof(int), cudaHostAllocDefault));

    CUDA_CHECK(cudaStreamCreate(&e->stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (e->use_lora) engine_load_lora(e, e->lora_path);

    // ---------------- 捕获 CUDA Graph ----------------
    cudaStream_t cap;
    CUDA_CHECK(cudaStreamCreate(&cap));
    CUDA_CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeGlobal));
    forward_launch(e, cap);
    CUDA_CHECK(cudaStreamEndCapture(cap, &e->graph));
    CUDA_CHECK(cudaGraphInstantiate(&e->graph_exec, e->graph, NULL, NULL, 0));
    CUDA_CHECK(cudaStreamDestroy(cap));

    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    fprintf(stderr, "[CUDA] 显存占用 %.1f / %.1f MiB\n", (total_b - free_b) / 1048576.0, total_b / 1048576.0);
}

static void engine_free(CudaEngine *e) {
    cudaGraphExecDestroy(e->graph_exec);
    cudaGraphDestroy(e->graph);
    cudaStreamDestroy(e->stream);
    cudaFreeHost(e->h_step);
    cudaFreeHost(e->h_step_stage);
    cudaFreeHost(e->h_next);
    cudaFreeHost(e->h_red);
    cudaFreeHost(e->h_sorted_probs);
    cudaFreeHost(e->h_sorted_ids);
    munmap(e->file_buffer, e->file_size);
    cudaDeviceReset();
}

// 解析并上传 LoRA 外挂权重（仅 NANO 架构；布局与 infer.c parse_lora_file 一致）
static void engine_load_lora(CudaEngine *e, const char *lora_path) {
    FILE *f = fopen(lora_path, "rb");
    if (!f) { fprintf(stderr, "无法打开 LoRA 文件 %s\n", lora_path); exit(EXIT_FAILURE); }
    fseek(f, 0, SEEK_END);
    size_t fsize = (size_t)ftell(f);
    rewind(f);
    uint8_t *lb = (uint8_t *)malloc(fsize);
    if (!lb || fread(lb, 1, fsize, f) != fsize) { fprintf(stderr, "读取 LoRA 文件失败\n"); exit(EXIT_FAILURE); }
    fclose(f);

    uint32_t *header = (uint32_t *)lb;
    int rank       = (int)header[6];
    float alpha    = (float)header[7];
    ModelConfig *c = &e->cfg;
    if ((int)header[8] != (int)c->n_layer || (int)header[9] != (int)c->n_embd ||
        (int)header[10] != (int)c->n_head || (int)header[11] != (int)c->n_kv_head ||
        (int)header[12] != (int)c->n_hidden) {
        fprintf(stderr, "LoRA 与基座模型维度不匹配\n"); exit(EXIT_FAILURE);
    }
    if (rank <= 0 || rank > LORA_MAX_RANK) {
        fprintf(stderr, "LoRA rank=%d 超出支持范围（1..%d）\n", rank, LORA_MAX_RANK); exit(EXIT_FAILURE);
    }

    int L = (int)c->n_layer, E = (int)c->n_embd, KV = e->kv_dim, Q = e->q_dim;
    DevLora *a = &e->lora;
    a->rank = rank;
    a->alpha = alpha;

    const float *lp = (const float *)(lb + 256);
    size_t qa = (size_t)L * rank * E * 4, qb = (size_t)L * E * rank * 4;
    size_t kb = (size_t)L * KV * rank * 4, oa = (size_t)L * rank * Q * 4;
    CUDA_CHECK(cudaMalloc(&a->wq_a, qa)); CUDA_CHECK(cudaMemcpy(a->wq_a, lp, qa, cudaMemcpyHostToDevice)); lp += (size_t)L * rank * E;
    CUDA_CHECK(cudaMalloc(&a->wq_b, qb)); CUDA_CHECK(cudaMemcpy(a->wq_b, lp, qb, cudaMemcpyHostToDevice)); lp += (size_t)L * E * rank;
    CUDA_CHECK(cudaMalloc(&a->wk_a, qa)); CUDA_CHECK(cudaMemcpy(a->wk_a, lp, qa, cudaMemcpyHostToDevice)); lp += (size_t)L * rank * E;
    CUDA_CHECK(cudaMalloc(&a->wk_b, kb)); CUDA_CHECK(cudaMemcpy(a->wk_b, lp, kb, cudaMemcpyHostToDevice)); lp += (size_t)L * KV * rank;
    CUDA_CHECK(cudaMalloc(&a->wv_a, qa)); CUDA_CHECK(cudaMemcpy(a->wv_a, lp, qa, cudaMemcpyHostToDevice)); lp += (size_t)L * rank * E;
    CUDA_CHECK(cudaMalloc(&a->wv_b, kb)); CUDA_CHECK(cudaMemcpy(a->wv_b, lp, kb, cudaMemcpyHostToDevice)); lp += (size_t)L * KV * rank;
    CUDA_CHECK(cudaMalloc(&a->wo_a, oa)); CUDA_CHECK(cudaMemcpy(a->wo_a, lp, oa, cudaMemcpyHostToDevice)); lp += (size_t)L * rank * Q;
    CUDA_CHECK(cudaMalloc(&a->wo_b, qb)); CUDA_CHECK(cudaMemcpy(a->wo_b, lp, qb, cudaMemcpyHostToDevice));

    free(lb);
    fprintf(stderr, "[CUDA] LoRA 已加载：rank=%d alpha=%.1f（scale=%.3f）\n", rank, alpha, alpha / rank);
}

// ===============================================================================
// 会话：prefill + decode
// ===============================================================================

typedef struct {
    uint32_t n_prompt;
    uint32_t n_generated;
    uint64_t t_prefill_ms;
    uint64_t t_decode_ms;
    uint64_t t_steady_ms;      // 跳过前 steady_skip 个 decode token 后的稳态耗时
    uint32_t n_steady;
    int stopped_by_eos;
} SessionStats;

// 执行一次完整会话。prompt_tokens 已套好模板；逐词元回调 on_token（可为 NULL）。
static void engine_run_session(CudaEngine *e, Tokenizer *tk,
                               const uint32_t *prompt_tokens, uint32_t n_prompt,
                               uint32_t max_new_tokens,
                               void (*on_token)(uint32_t id, void *env), void *env,
                               SessionStats *stats) {
    RunState *s = &e->s;
    int max_pos = e->max_seq_len - 1;

    int no_eos = (getenv("NC_NO_EOS") != NULL);
    // ---------- prefill：逐步异步入队，无需逐步同步 ----------
    uint64_t t0 = now_ms();
    for (uint32_t pos = 0; pos + 1 < n_prompt; pos++) {
        e->h_step_stage[2 * pos + 0] = (int)prompt_tokens[pos];
        e->h_step_stage[2 * pos + 1] = (int)pos;
        CUDA_CHECK(cudaMemcpyAsync(s->d_step, e->h_step_stage + 2 * pos, 2 * sizeof(int),
                                   cudaMemcpyHostToDevice, e->stream));
        CUDA_CHECK(cudaGraphLaunch(e->graph_exec, e->stream));
    }
    // 最后一个 prompt 词元：其前向产生首个生成词元的 logits
    uint32_t pos = n_prompt - 1;
    e->h_step[0] = (int)prompt_tokens[pos];
    e->h_step[1] = (int)pos;
    CUDA_CHECK(cudaMemcpyAsync(s->d_step, e->h_step, 2 * sizeof(int), cudaMemcpyHostToDevice, e->stream));
    CUDA_CHECK(cudaGraphLaunch(e->graph_exec, e->stream));

    // 采样第一个生成词元
    int next = engine_sample(e);
    stats->t_prefill_ms = now_ms() - t0;
    // ---------- decode ----------
    uint64_t td0 = now_ms();
    uint32_t n_gen = 0;
    uint64_t t_mark = td0;
    while (1) {
        if (!no_eos && (next == (int)e->eos1 || next == (int)e->eos2)) { stats->stopped_by_eos = 1; break; }
        if (on_token) on_token((uint32_t)next, env);
        n_gen++;
        if (n_gen == 33) t_mark = now_ms(); // 前 32 token 视为升频预热期
        if (max_new_tokens > 0 && n_gen >= max_new_tokens) break;
        if ((int)pos + 1 >= max_pos) break;

        pos++;
        e->h_step[0] = next;
        e->h_step[1] = (int)pos;
        CUDA_CHECK(cudaMemcpyAsync(s->d_step, e->h_step, 2 * sizeof(int), cudaMemcpyHostToDevice, e->stream));
        CUDA_CHECK(cudaGraphLaunch(e->graph_exec, e->stream));
        next = engine_sample(e);
    }
    uint64_t t_end = now_ms();
    stats->t_decode_ms = t_end - td0;
    stats->t_steady_ms = (n_gen > 32) ? (t_end - t_mark) : stats->t_decode_ms;
    stats->n_steady = (n_gen > 32) ? (n_gen - 32) : n_gen;
    stats->n_prompt = n_prompt;
    stats->n_generated = n_gen;
}

// ===============================================================================
// 终端交互（参照 main_cli.c）
// ===============================================================================

#define MAX_PROMPT_BUFFER_LENGTH (16384)
#define INITIAL_CAPACITY 8
#define LINE_BUFFER_SIZE 1024

static char **readlines(int *line_count) {
    if (!line_count) return NULL;
    clearerr(stdin);
    *line_count = 0;
    int capacity = INITIAL_CAPACITY;
    char **lines = (char **)malloc(capacity * sizeof(char *));
    if (!lines) return NULL;

    char buffer[LINE_BUFFER_SIZE];
    while (fgets(buffer, sizeof(buffer), stdin) != NULL) {
        if (*line_count >= capacity) {
            capacity *= 2;
            char **new_lines = (char **)realloc(lines, capacity * sizeof(char *));
            if (!new_lines) {
                for (int i = 0; i < *line_count; i++) free(lines[i]);
                free(lines);
                return NULL;
            }
            lines = new_lines;
        }
        size_t len = strlen(buffer);
        lines[*line_count] = (char *)malloc((len + 1) * sizeof(char));
        if (!lines[*line_count]) {
            for (int i = 0; i < *line_count; i++) free(lines[i]);
            free(lines);
            return NULL;
        }
        strcpy(lines[*line_count], buffer);
        (*line_count)++;
    }
    if (*line_count == 0) {
        free(lines);
        lines = NULL;
    }
    return lines;
}

static void freelines(char **lines, int line_count) {
    if (!lines) return;
    for (int i = 0; i < line_count; i++) free(lines[i]);
    free(lines);
}

// NANO 架构词表解析（与 infer.c parse_model_file 的 NANO 分支一致）：
//   逐条读取 (header, token_id, ucs32...) 词元，构建 unicode map 与多字词元 trie
static void build_nano_tokenizer(Tokenizer *tk, uint8_t *buffer, uint32_t tokenizer_field_bytes) {
    uint32_t *vocab_ptr = (uint32_t *)buffer + 1; // 跳过 tokenizer_field_bytes 字段

    uint32_t byte_count = 0;
    uint32_t char_count = 0;

    tk->vocab_size = *vocab_ptr; vocab_ptr++;

    tk->token_list        = (wchar_t **)calloc(tk->vocab_size, sizeof(wchar_t *));
    tk->unicode_charset   = (wchar_t  *)calloc(tk->vocab_size, sizeof(wchar_t));
    tk->unicode_to_id_map = new_map(tk->vocab_size);
    tk->token_to_id_map   = new_map(tk->vocab_size);
    tk->vocab_trie        = new_trie(tk->vocab_size, 0);

    while (byte_count < tokenizer_field_bytes - 8) { // 不含 field_bytes 与 vocab_size 的 8 字节
        uint32_t token_header = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);
        uint32_t token_id     = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);

        // 小端序：MSB(reserved_1 reserved_0 is_special token_length)LSB
        uint32_t token_length = (token_header & 0x000000ff);

        wchar_t *token = (wchar_t *)calloc(token_length + 1, sizeof(wchar_t));
        if (token_length == 1) {
            tk->unicode_charset[char_count] = *vocab_ptr;
            map_set(tk->unicode_to_id_map, *vocab_ptr, token_id);
            char_count++;
        }
        for (uint32_t i = 0; i < token_length; i++) {
            token[i] = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);
        }
        token[token_length] = 0;
        tk->token_list[token_id] = token;
    }

    for (uint32_t i = 0; i < tk->vocab_size; i++) {
        wchar_t *utoken = tk->token_list[i];
        uint32_t len = (uint32_t)wcslen(utoken);
        if (len > 1) {
            uint32_t *ids = string_to_ids(tk->unicode_to_id_map, utoken);
            add_token(tk->vocab_trie, ids, len, i);
            free(ids);
        }
    }
}

// 输出回调环境：按架构选择 BPE 字节串或 NANO 宽字符解码
typedef struct {
    Tokenizer *tk;
    int is_nano;
} PrintEnv;

static void print_token_cb(uint32_t id, void *env) {
    PrintEnv *pe = (PrintEnv *)env;
    if (pe->is_nano) {
        wchar_t *w = decode_nano(pe->tk, &id, 1);
        printf("%ls", w);
        free(w);
    }
    else {
        printf("%s", pe->tk->vocab[id]);
    }
    fflush(stdout);
}

static void print_token_id_cb(uint32_t id, void *env) {
    (void)env;
    printf("%u\n", id);
    fflush(stdout);
}

// 按架构编码 prompt：QWEN3 用 ChatML 模板；NANO 用 instruct/response 标记 + trie 编码
static uint32_t *encode_prompt(CudaEngine *e, Tokenizer *tk, wchar_t *input, uint32_t *n_out) {
    if (e->cfg.arch == LLM_ARCH_QWEN3) {
        return apply_qwen_chat_template(tk, input, n_out, 1);
    }
    size_t in_len = wcslen(input);
    wchar_t *prompt = (wchar_t *)malloc((in_len + 64) * sizeof(wchar_t));
    wcscpy(prompt, L"<|instruct_mark|>");
    wcscat(prompt, input);
    wcscat(prompt, L"<|response_mark|>");
    uint32_t *ids = encode_nano(tk, prompt, n_out);
    free(prompt);
    return ids;
}


// ===============================================================================
// 微基准：逐 kernel 计时（NC_PROF=1 时在 bench 模式中启用）
// ===============================================================================

static double time_op(CudaEngine *e, int iters, void (*launch)(CudaEngine *, cudaStream_t)) {
    // 分 6 批测量取最小批均值：抗 DVFS 时钟波动（最小值对应最高时钟状态）
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    double best = 1e30;
    for (int b = 0; b < 6; b++) {
        launch(e, e->stream); // warm
        CUDA_CHECK(cudaStreamSynchronize(e->stream));
        cudaEventRecord(ev0, e->stream);
        for (int i = 0; i < iters; i++) launch(e, e->stream);
        cudaEventRecord(ev1, e->stream);
        CUDA_CHECK(cudaStreamSynchronize(e->stream));
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        double us = (double)ms * 1000.0 / iters;
        if (us < best) best = us;
    }
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return best; // us
}

#define PROF_LAUNCH(name, ...) \
    static void launch_##name(CudaEngine *e, cudaStream_t st) { __VA_ARGS__ }
PROF_LAUNCH(embed, {
    embed_kernel<<<1, e->cfg.n_embd / 4, 0, st>>>(e->w.emb_q, e->w.emb_s, e->s.d_step, e->s.x, e->cfg.n_embd);
})
PROF_LAUNCH(rmsnorm, {
    rmsnorm_quant_kernel<<<1, 256, 0, st>>>(e->s.x, e->w.rms_attn, e->s.xq, e->s.xs, e->cfg.n_embd, NULL);
})
PROF_LAUNCH(attn, {
    attention_kernel<<<e->cfg.n_head, HEAD_DIM, 0, st>>>(
        e->s.qkv, e->s.k_cache, e->s.v_cache, e->w.q_norm, e->w.k_norm, e->w.inv_freq,
        e->s.att, e->s.xba_q, e->s.xba_s,
        e->s.d_step + 1, e->max_seq_len, (int)e->cfg.n_head, (int)e->cfg.n_kv_head, e->kv_mul);
})
PROF_LAUNCH(gemv_qkv, {
    static int li = 0; li = (li + 1) % (int)e->cfg.n_layer;
    size_t qk = (size_t)(e->q_dim + 2 * e->kv_dim);
    WeightMat m = { WEIGHT_Q80, { e->w.wqkv_q + li * qk * e->cfg.n_embd, e->w.wqkv_s + li * qk * (e->cfg.n_embd / GROUP_SIZE) }, (int)e->cfg.n_embd, (int)qk };
    cudaMemsetAsync(e->s.qkv, 0, qk * sizeof(float), st);
    gemv_dispatch(&m, GEMV_QKV, e->s.xq, e->s.xs, e->s.qkv, NULL, st);
})
PROF_LAUNCH(gemv_wo, {
    static int li = 0; li = (li + 1) % (int)e->cfg.n_layer;
    WeightMat m = { WEIGHT_Q80, { e->w.wo_q + (size_t)li * e->cfg.n_embd * e->q_dim, e->w.wo_s + (size_t)li * e->cfg.n_embd * (e->q_dim / GROUP_SIZE) }, e->q_dim, (int)e->cfg.n_embd };
    gemv_dispatch(&m, GEMV_WO, e->s.xba_q, e->s.xba_s, e->s.x, e->s.x, st);
})
PROF_LAUNCH(gemv_w13, {
    static int li = 0; li = (li + 1) % (int)e->cfg.n_layer;
    WeightMat m = { WEIGHT_Q80, { e->w.w13_q + (size_t)li * 2 * e->cfg.n_hidden * e->cfg.n_embd, e->w.w13_s + (size_t)li * 2 * e->cfg.n_hidden * (e->cfg.n_embd / GROUP_SIZE) }, (int)e->cfg.n_embd, 2 * (int)e->cfg.n_hidden };
    gemv_dispatch(&m, GEMV_W13, e->s.xq, e->s.xs, e->s.h13, NULL, st);
})
PROF_LAUNCH(swiglu, {
    swiglu_quant_kernel<<<1, 256, 0, st>>>(e->s.h13, e->s.hq, e->s.hs, e->cfg.n_hidden);
})
PROF_LAUNCH(gemv_w2, {
    static int li = 0; li = (li + 1) % (int)e->cfg.n_layer;
    WeightMat m = { WEIGHT_Q80, { e->w.w2_q + (size_t)li * e->cfg.n_embd * e->cfg.n_hidden, e->w.w2_s + (size_t)li * e->cfg.n_embd * (e->cfg.n_hidden / GROUP_SIZE) }, (int)e->cfg.n_hidden, (int)e->cfg.n_embd };
    gemv_dispatch(&m, GEMV_W2, e->s.hq, e->s.hs, e->s.x, e->s.x, st);
})
PROF_LAUNCH(gemv_cls, {
    WeightMat m = { WEIGHT_Q80, { e->w.emb_q, e->w.emb_s }, (int)e->cfg.n_embd, (int)e->cfg.vocab_size };
    gemv_dispatch(&m, GEMV_CLS, e->s.xq, e->s.xs, e->s.logits, NULL, st);
})
PROF_LAUNCH(argmax, {
    argmax_one_kernel<<<1, 1024, 0, st>>>(e->s.logits, (int)e->cfg.vocab_size, e->s.d_next);
})
PROF_LAUNCH(graph, {
    cudaGraphLaunch(e->graph_exec, st);
})

static void profile_kernels(CudaEngine *e) {
    e->h_step[0] = 97072; e->h_step[1] = 512; // 非零 pos，使 attention 有实际工作量
    CUDA_CHECK(cudaMemcpy(e->s.d_step, e->h_step, 2 * sizeof(int), cudaMemcpyHostToDevice));
    int L = (int)e->cfg.n_layer;

    double t_graph = time_op(e, 50, launch_graph);
    printf("[prof] graph replay          : %8.1f us\n", t_graph);
    struct { const char *name; void (*fn)(CudaEngine *, cudaStream_t); int count; } ops[] = {
        { "embed          ", launch_embed, 1 },
        { "rmsnorm_quant  ", launch_rmsnorm, 2 * L + 1 },
        { "gemv_qkv  n=1024", launch_gemv_qkv, L },
        { "attention      ", launch_attn, L },
        { "gemv_wo   n=2048", launch_gemv_wo, L },
        { "gemv_w13 n=1024 ", launch_gemv_w13, L },
        { "swiglu_quant   ", launch_swiglu, L },
        { "gemv_w2   n=3072", launch_gemv_w2, L },
        { "gemv_cls  n=1024", launch_gemv_cls, 1 },
        { "argmax         ", launch_argmax, 1 },
    };
    double sum = 0.0;
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++) {
        double us = time_op(e, 200, ops[i].fn);
        printf("[prof] %s : %7.1f us x %2d = %8.1f us\n", ops[i].name, us, ops[i].count, us * ops[i].count);
        sum += us * ops[i].count;
    }
    printf("[prof] kernel 合计: %.1f us（graph 开销约 %.1f us）\n", sum, t_graph - sum);
}

int main(int argc, char **argv) {
    if (!setlocale(LC_CTYPE, "")) return -1;

    const char *model_path = "/home/bd4sur/ai/_model/Nano/qwen3-0b6-q80.bin";
    int max_seq_len = 2048;
    float temperature = 0.7f;
    float top_p = 0.8f;
    float rep_penalty = 1.0f;
    uint64_t seed = now_ms();
    uint32_t max_new = 0;      // 0 = 不限（到 EOS 或序列上限）
    int bench_tokens = 0;      // >0 时进入基准模式
    const char *lora_path = NULL;

    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-l") && i + 1 < argc) max_seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) rep_penalty = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-bench") && i + 1 < argc) bench_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-lora") && i + 1 < argc) lora_path = argv[++i];
        else {
            printf("用法: %s [-m/--model 模型路径] [-lora LoRA路径] [-l max_seq_len] [-t temperature] [-p top_p]\n"
                   "          [-r rep_penalty] [-s seed] [-n max_new_tokens] [-bench N]\n", argv[0]);
            return 0;
        }
    }

    printf("Nano CUDA Inference Engine (Qwen3 / Nano / Q80)\n\n");
    printf("Using model: %s\n", model_path);

    CudaEngine engine;
    memset(&engine, 0, sizeof(engine));
    engine.rep_penalty = rep_penalty;
    engine.temperature = temperature;
    engine.rng_state = seed;
    engine.use_lora = (lora_path != NULL);
    engine.lora_path = lora_path;

    engine_load(&engine, model_path, max_seq_len);

    ModelConfig *c = &engine.cfg;
    printf("  block_size = %u\n  vocab_size = %u\n  n_layer = %u\n  n_embd = %u\n"
           "  n_head = %u\n  n_kv_head = %u\n  n_hidden = %u\n  is_shared_classifier = %u\n"
           "  head_dim = %u\n  arch = %u\n  quant_type = 0x%x\n  group_size = %u\n",
           c->block_size, c->vocab_size, c->n_layer, c->n_embd, c->n_head, c->n_kv_head,
           c->n_hidden, c->is_shared_classifier, c->head_dim, c->arch, c->quant_type, c->group_size);
    printf("  max_seq_len = %d  temperature = %.2f  top_p = %.2f  rep_penalty = %.2f  seed = %llu\n",
           max_seq_len, temperature, top_p, rep_penalty, (unsigned long long)seed);

    // 分词器（QWEN3 用 BPE；NANO 用宽字符 trie 词表）
    Tokenizer *tk = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    int is_nano = (engine.cfg.arch == LLM_ARCH_NANO);
    if (!is_nano) {
        build_bpe_tokenizer(tk, engine.file_buffer + 256, 151669);
    }
    else {
        uint32_t tokenizer_field_bytes = *(uint32_t *)(engine.file_buffer + 256);
        build_nano_tokenizer(tk, engine.file_buffer + 256, tokenizer_field_bytes);
    }
    PrintEnv penv = { tk, is_nano };
    if (bench_tokens > 0) {
        if (getenv("NC_PROF")) { profile_kernels(&engine); }
        // ---------- 基准模式：固定 prompt，打印统计 ----------
        wchar_t wprompt[MAX_PROMPT_BUFFER_LENGTH];
        mbstowcs(wprompt, "请你介绍一下你自己。", MAX_PROMPT_BUFFER_LENGTH);
        uint32_t n_prompt = 0;
        uint32_t *prompt_tokens = encode_prompt(&engine, tk, wprompt, &n_prompt);
        SessionStats stats = {0};
        printf("\n[bench] prompt_tokens = %u, gen %d tokens, temperature = %.2f\n\n",
               n_prompt, bench_tokens, temperature);
        engine_run_session(&engine, tk, prompt_tokens, n_prompt, (uint32_t)bench_tokens,
                           getenv("NC_PRINT_IDS") ? print_token_id_cb : print_token_cb, &penv, &stats);
        printf("\n\n[bench] prefill: %u tokens / %llu ms = %.1f tok/s\n",
               stats.n_prompt, (unsigned long long)stats.t_prefill_ms,
               stats.t_prefill_ms ? (double)stats.n_prompt / (double)stats.t_prefill_ms * 1000.0 : 0.0);
        printf("[bench] decode:  %u tokens / %llu ms = %.1f tok/s\n",
               stats.n_generated, (unsigned long long)stats.t_decode_ms,
               stats.t_decode_ms ? (double)stats.n_generated / (double)stats.t_decode_ms * 1000.0 : 0.0);
        printf("[bench] steady:  %u tokens / %llu ms = %.1f tok/s（跳过前 32 token）\n",
               stats.n_steady, (unsigned long long)stats.t_steady_ms,
               stats.t_steady_ms ? (double)stats.n_steady / (double)stats.t_steady_ms * 1000.0 : 0.0);
        free(prompt_tokens);
        if (is_nano) free_tokenizer(tk); else free_bpe_tokenizer(tk);
        free(tk);
        engine_free(&engine);
        return 0;
    }

    printf("\n请输入问题，过程中可按Enter换行；输入完成请按Ctrl+D提交。在空行按Ctrl+D退出。\n\n");

    while (1) {
        wchar_t input_text[MAX_PROMPT_BUFFER_LENGTH] = L"";

        printf("\x1b[32;1mHomo:\x1b[0m ");
        fflush(stdout);

        int line_count = 0;
        char **lines = readlines(&line_count);
        if (!lines && line_count == 0) break; // 空输入直接 EOF：退出

        for (int i = 0; i < line_count; i++) {
            wchar_t wcline[MAX_PROMPT_BUFFER_LENGTH];
            mbstowcs(wcline, lines[i], MAX_PROMPT_BUFFER_LENGTH);
            wcscat(input_text, wcline);
        }
        freelines(lines, line_count);

        uint32_t n_prompt = 0;
        uint32_t *prompt_tokens = encode_prompt(&engine, tk, input_text, &n_prompt);
        if ((int)n_prompt >= max_seq_len - 2) {
            printf("输入过长（%u tokens），请缩短。\n\n", n_prompt);
            free(prompt_tokens);
            continue;
        }

        printf("\n\x1b[34;1mNano:\x1b[0m ");
        fflush(stdout);

        SessionStats stats = {0};
        engine_run_session(&engine, tk, prompt_tokens, n_prompt, max_new, print_token_cb, &penv, &stats);

        double prefill_tps = stats.t_prefill_ms ? (double)stats.n_prompt / (double)stats.t_prefill_ms * 1000.0 : 0.0;
        double decode_tps = stats.t_decode_ms ? (double)stats.n_generated / (double)stats.t_decode_ms * 1000.0 : 0.0;
        printf("\n\n[prefill %.1f tok/s | decode %.1f tok/s | %u tokens]\n\n",
               prefill_tps, decode_tps, stats.n_generated);

        free(prompt_tokens);
    }

    if (is_nano) free_tokenizer(tk); else free_bpe_tokenizer(tk);
    free(tk);
    engine_free(&engine);
    printf("Bye.\n");
    return 0;
}
