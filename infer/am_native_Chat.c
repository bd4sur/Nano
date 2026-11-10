////////////////////////////////////////////////////////////////////////////
//  Native Library : Chat —— 设计说明见 am_native_Chat.h
//
//  本文件只做四件事：
//    1) 把鹦鹉笼的 C 推理引擎（infer.c / tensor.c / tokenizer.c）包成
//       「一拍一步」的会话状态机（begin / step / end），供 Scheme 每帧推进；
//    2) 把模型文件、分词器、超参数、采样统计（进度 / TPS）以值的形式交给 Scheme；
//    3) 把引擎的观测钩子（Nano_Observation）与输出 logits 转成可视化数据
//       （层/相位、top-K 候选文本与采样概率），并直接用 C 图形层画前向框图；
//    4) 严格守本地库栈纪律（值型：压 1 值 + step 1 次）。
//
//  **不修改任何既有 C 实现**：只调用 infer.h / tokenizer.h 的公开接口，并给
//  Nano_Context 的 observation / observation_env 两个钩子字段赋值（infer.c 就是
//  为此暴露的，见 ENABLE_NANO_OBSERVATION）。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "am_native_Chat.h"
#include "am_host.h"

#include "platform.h"

#include "infer.h"
#include "tokenizer.h"

// ===============================================================================
// 模型候选表（白名单 + 文件存在性）
//  与 ui_llm.c 的 preset_model_configs 保持同一组采样参数，但**只列端侧可用的
//  Nano 系列**：Qwen 大模型在浏览器 WASM 里跑不动（单线程 + 单精度浮点），列出来
//  只会让用户点了白等。参数逐项对齐 preset 表（repetition_penalty / temperature /
//  top_p / max_seq_len），改那里就要同步这里。
// ===============================================================================

typedef struct {
    const wchar_t *name;        // 显示名
    const char    *file;        // 绝对路径（MODEL_ROOT_DIR 之下）
    const char    *rel;         // 相对业务资源根的路径（给 Scheme 显示）
    float    repetition_penalty;
    float    temperature;
    float    top_p;
    uint32_t top_k;
    uint32_t max_seq_len;
} chat_model_t;

static const chat_model_t s_models[] = {
    { L"Nano-168M", MODEL_ROOT_DIR "/nano-168m-q80.bin", "llm/nano-168m-q80.bin", 1.05f, 1.0f, 0.5f, 0, 512 },
    { L"Nano-56M",  MODEL_ROOT_DIR "/nano-56m-q80.bin",  "llm/nano-56m-q80.bin",  1.05f, 1.0f, 0.5f, 0, 512 },
};
#define CHAT_MODEL_NUM ((int32_t)(sizeof(s_models) / sizeof(s_models[0])))

// ===============================================================================
// 模块状态（虚拟机全局单例，与 C 版鹦鹉笼的引擎单例语义一致）
// ===============================================================================

static Nano_Context *s_ctx = NULL;        // 引擎上下文（模型 + 分词器 + 采样器）
static Nano_Session *s_session = NULL;    // 当前生成会话
static int32_t       s_model_sel = -1;    // 已载入的 s_models 下标（-1=未载入）

static int32_t s_menu_map[AM_NATIVE_CHAT_MODEL_MAX]; // (Chat.models) 下标 → s_models 下标
static int32_t s_menu_num = 0;                       // 可用模型数

// 观测钩子给出的"现在算到哪了"
static int32_t s_obs_layer = -1;
static int32_t s_obs_phase = 0;

// 一步前向内部的**相位轨迹**：infer.c 在每个子步骤入口回调一次（层 × 相位），
// 顺序即执行顺序。Scheme 侧据此把"这一步算了哪些部件"逐帧回放出来——
// 引擎一次前向是原子的（不能中途交还控制权），因此只能事后回放，这与
// app/llm.scm「一相位一帧」的观感一致，且本地库本身不做任何绘制。
#define CHAT_TRACE_MAX (512)          // 单步上限（Nano/Qwen 单步最多约 4×n_layer+4）
static uint16_t s_trace[CHAT_TRACE_MAX];   // ((layer+1) << 8) | phase
static int32_t  s_trace_n = 0;

// 输出分布 top-K 缓存（每次 (Chat.step) 后重算）
static uint32_t s_top_id[AM_NATIVE_CHAT_TOP_K];
static float    s_top_prob[AM_NATIVE_CHAT_TOP_K];
static int32_t  s_top_num = 0;
static int32_t  s_pick = -1;              // 最近采样的词元（预填充阶段 -1）
static float    s_pick_prob = 0.0f;       // 该词元在本步输出分布下的概率

// 采样统计（整数毫秒口径，避开 wasm32 的数值精度坑）
static uint64_t s_t_begin_ms = 0;
static uint64_t s_t_prefill_end_ms = 0;
static uint64_t s_t_last_ms = 0;
static int32_t  s_last_state = 0;

// ===============================================================================
// 小工具
// ===============================================================================

static uint64_t chat_now_ms(void) {
    return (uint64_t)am_current_timestamp_in_ms();
}

// ---- 操作数栈辅助（与 am_native_Gfx.c 同一套） ------------------------------

static bool chat_pop_number(am_process_t *proc, am_float_t *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (am_value_is_float(v)) { *out = am_value_to_float(v); return true; }
    if (am_value_is_int(v))   { *out = (am_float_t)am_value_to_int(v); return true; }
    if (am_value_is_uint(v))  { *out = (am_float_t)am_value_to_uint(v); return true; }
    return false;
}

static bool chat_pop_int(am_process_t *proc, int32_t *out) {
    am_float_t f;
    if (!chat_pop_number(proc, &f)) return false;
    if (isnan(f)) return false;
    if (f > 2147483647.0)  f = 2147483647.0;
    if (f < -2147483648.0) f = -2147483648.0;
    *out = (int32_t)f;
    return true;
}

// 弹出一个 Scheme 字符串并转成 UTF-8（buf 至少 cap 字节；失败时置空串）
static bool chat_pop_utf8(am_process_t *proc, char *buf, size_t cap) {
    am_value_t v = am_process_pop_operand(proc);

    buf[0] = '\0';
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return false;

    am_wstring_t *ws = (am_wstring_t *)obj;
    wchar_t wbuf[4096];
    size_t n = ws->length;
    if (n > sizeof(wbuf) / sizeof(wbuf[0]) - 1) n = sizeof(wbuf) / sizeof(wbuf[0]) - 1;
    for (size_t i = 0; i < n; i++) wbuf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    wbuf[n] = L'\0';

    am_wcstombs(buf, wbuf, (uint32_t)cap);
    return true;
}

// 值型收尾（契约 N1 + N6）：压 1 个值 + step 1 次。
static int32_t chat_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t chat_push_bool(am_process_t *proc, bool b) {
    return chat_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

static int32_t chat_push_int(am_process_t *proc, int64_t x) {
    return chat_push(proc, am_make_value_of_int((am_int_t)x));
}

static int32_t chat_push_null(am_process_t *proc) {
    return chat_push(proc, AM_VALUE_NULL);
}

// 由 UTF-8 构造 Scheme 字符串把柄（失败返回 AM_HANDLE_NULL）
static am_handle_t chat_make_string(am_process_t *proc, const char *utf8, size_t len) {
    wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (len + 1) * sizeof(wchar_t));
    if (!wbuf) return AM_HANDLE_NULL;
    uint32_t wlen = am_mbstowcs(wbuf, utf8, (uint32_t)len);
    am_handle_t hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
    am_free(proc->heap_alloc, wbuf);
    return hd;
}

// 压入一个字符串（失败压 #null）
static int32_t chat_push_string(am_process_t *proc, const char *utf8) {
    am_handle_t hd = chat_make_string(proc, utf8, strlen(utf8));
    if (hd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(hd));
}

// 新建一个空列表（返回列表对象，失败 NULL）
static am_list_t *chat_list_new(am_process_t *proc, size_t cap) {
    return am_list_create(proc->heap_alloc, cap > 0 ? cap : 1, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
}

// 往列表追加一个已构造的值（失败返回 NULL 并销毁列表）
static am_list_t *chat_list_add(am_process_t *proc, am_list_t *lst, am_value_t v) {
    am_list_t *grown = am_list_push(proc->heap_alloc, lst, v);
    if (!grown) am_list_destroy(proc->heap_alloc, lst);
    return grown;
}

// 把列表对象装箱成堆对象并把柄（失败返回 AM_HANDLE_NULL 并销毁列表）
static am_handle_t chat_list_box(am_process_t *proc, am_list_t *lst) {
    am_handle_t lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, lst);
        return AM_HANDLE_NULL;
    }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return AM_HANDLE_NULL;
    }
    return lhd;
}

static bool chat_file_exists(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return false;
    fclose(fp);
    return true;
}

// 当前会话已生成的文本（宽字符 → UTF-8），并剔除结尾的特殊词元标记。
// 说明：预填充阶段引擎的 output_text 是"已读入的提示词"，不是生成内容，故返回空串。
static void chat_output_utf8(char *out, size_t cap) {
    out[0] = '\0';
    if (s_session == NULL || s_ctx == NULL) return;
    if (s_session->output_text == NULL || s_session->output_count == 0) return;

    size_t n = wcslen(s_session->output_text);
    wchar_t *tmp = (wchar_t *)malloc((n + 1) * sizeof(wchar_t));
    if (!tmp) return;
    wmemcpy(tmp, s_session->output_text, n + 1);

    // 剔除结尾的结束/填充标记（Nano：<|eos|> / <|padding|>；Qwen：<|im_end|> 等）
    static const wchar_t *tail[] = { L"<|eos|>", L"<|padding|>", L"<|im_end|>", L"<|endoftext|>" };
    int32_t changed = 1;
    while (changed) {
        changed = 0;
        n = wcslen(tmp);
        for (size_t k = 0; k < sizeof(tail) / sizeof(tail[0]); k++) {
            size_t tl = wcslen(tail[k]);
            if (n >= tl && wmemcmp(tmp + (n - tl), tail[k], tl) == 0) {
                tmp[n - tl] = L'\0';
                changed = 1;
                break;
            }
        }
    }

    am_wcstombs(out, tmp, (uint32_t)cap);
    free(tmp);
}

// 取模型超参数（供内部使用；未载入返回 false）
static bool chat_config_of(LLM_Config *cfg) {
    if (s_ctx == NULL || s_ctx->llm == NULL) return false;
    *cfg = s_ctx->llm->config;
    return true;
}

// ===============================================================================
// 观测钩子（infer.c 在每个子步骤处回调）与输出分布 top-K
// ===============================================================================

static void chat_observation(Nano_Observation obs, void *env) {
    (void)env;
    // 采样钩子只带相位（layer = -1 是它的约定值，不代表"回到词嵌入"）：
    // 保留上一次的层号，界面上的"当前层"才不会在最后一个相位跳回嵌入层。
    if (obs.phase != NANO_LLM_PHASE_SAMPLE || s_obs_layer == -1) s_obs_layer = obs.layer;
    s_obs_phase = obs.phase;

    if (s_trace_n < CHAT_TRACE_MAX) {
        int32_t lay = obs.layer + 1;                 // -1（嵌入）→ 0
        if (lay < 0) lay = 0;
        if (lay > 255) lay = 255;
        s_trace[s_trace_n++] = (uint16_t)(((uint32_t)lay << 8) | (uint32_t)(obs.phase & 0xFF));
    }
}

// 本步输出分布的归一化参数（一次算好，top-K 与"采样词元概率"共用）
static int32_t s_dist_is_prob = 0;     // 1 = logits 里已是概率（解码步 + 温度采样）
static float   s_dist_temp = 1.0f;
static float   s_dist_maxlogit = 0.0f;
static float   s_dist_inv_sum = 1.0f;

// 某个词元在本步输出分布下的概率
static float chat_prob_of(uint32_t id) {
    if (s_ctx == NULL || s_ctx->llm == NULL) return 0.0f;
    if (id >= s_ctx->llm->config.vocab_size) return 0.0f;
    float v = s_ctx->llm->state.logits[id];
    if (s_dist_is_prob) return v;
    float p = expf((v - s_dist_maxlogit) / s_dist_temp) * s_dist_inv_sum;
    return isfinite(p) ? p : 0.0f;
}

// 重算输出分布 top-K 并缓存归一化参数。
//   解码步（温度采样）：generate_next_token 已就地做过「复读惩罚 → 除温度 → softmax」，
//                       logits 里就是**采样分布本身**，直接用；
//                       （温度为 0 的贪心路径不做 softmax，需自行归一化）
//   预填充步：logits 是原始分数，按温度做 softmax 后再取概率。
static void chat_update_topk(bool logits_are_probs) {
    LLM *llm = s_ctx->llm;
    uint32_t vnum = llm->config.vocab_size;
    float *logits = llm->state.logits;
    if (!logits || vnum == 0) {
        s_top_num = 0;
        return;
    }

    s_dist_is_prob = (logits_are_probs && s_ctx->sampler->temperature > 0.0f) ? 1 : 0;
    s_dist_temp = s_ctx->sampler->temperature;
    if (s_dist_temp <= 0.0f) s_dist_temp = 1.0f;
    s_dist_maxlogit = 0.0f;
    s_dist_inv_sum = 1.0f;

    if (!s_dist_is_prob) {
        s_dist_maxlogit = logits[0];
        for (uint32_t i = 1; i < vnum; i++) if (logits[i] > s_dist_maxlogit) s_dist_maxlogit = logits[i];
        float sum = 0.0f;
        for (uint32_t i = 0; i < vnum; i++) sum += expf((logits[i] - s_dist_maxlogit) / s_dist_temp);
        s_dist_inv_sum = (sum > 0.0f && isfinite(sum)) ? (1.0f / sum) : 1.0f;
    }

    // 定长插入排序（K=6，O(V·K) 但只有几次浮点比较，远低于一次前向）
    s_top_num = 0;
    for (uint32_t i = 0; i < vnum; i++) {
        float p = chat_prob_of(i);
        if (!isfinite(p)) continue;
        if (s_top_num >= AM_NATIVE_CHAT_TOP_K && p <= s_top_prob[s_top_num - 1]) continue;
        int32_t pos = (s_top_num < AM_NATIVE_CHAT_TOP_K) ? s_top_num : (AM_NATIVE_CHAT_TOP_K - 1);
        while (pos > 0 && s_top_prob[pos - 1] < p) {
            s_top_prob[pos] = s_top_prob[pos - 1];
            s_top_id[pos] = s_top_id[pos - 1];
            pos--;
        }
        s_top_prob[pos] = p;
        s_top_id[pos] = i;
        if (s_top_num < AM_NATIVE_CHAT_TOP_K) s_top_num++;
    }
}

// 词元编号 → UTF-8 文本（引擎按架构选择分词器；buf 由调用方提供）
static bool chat_token_utf8(uint32_t id, char *buf, size_t cap) {
    if (s_ctx == NULL) return false;
    buf[0] = '\0';
    wchar_t *wtext = NULL;

    if (s_ctx->llm->arch == LLM_ARCH_NANO) {
        wtext = decode_nano(s_ctx->tokenizer, &id, 1);
    }
    else if (s_ctx->llm->arch == LLM_ARCH_QWEN2 || s_ctx->llm->arch == LLM_ARCH_QWEN3 ||
             s_ctx->llm->arch == LLM_ARCH_QWEN35) {
        wtext = decode_bpe(s_ctx->tokenizer, &id, 1);
    }
    if (wtext == NULL) return false;

    am_wcstombs(buf, wtext, (uint32_t)cap);
    free(wtext);
    return true;
}

// ===============================================================================
// 模型与引擎
// ===============================================================================

// 释放会话与模型（幂等；宿主与 (Chat.unload) 共用）
void am_native_chat_release(void) {
    if (s_session) {
        llm_session_free(s_session);
        s_session = NULL;
    }
    if (s_ctx) {
        llm_context_free(s_ctx);
        s_ctx = NULL;
    }
    s_model_sel = -1;
    s_top_num = 0;
    s_pick = -1;
    s_obs_layer = -1;
    s_obs_phase = 0;
    s_trace_n = 0;
    s_last_state = 0;
    s_t_begin_ms = 0;
    s_t_prefill_end_ms = 0;
    s_t_last_ms = 0;
}

int32_t am_native_Chat_models(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;

    s_menu_num = 0;
    am_list_t *lst = chat_list_new(proc, CHAT_MODEL_NUM);
    if (!lst) return chat_push_null(proc);

    for (int32_t i = 0; i < CHAT_MODEL_NUM; i++) {
        // 白名单 + 文件存在性：浏览器目标模型未预载时不列出（Scheme 据此提示）
        if (!chat_file_exists(s_models[i].file)) continue;

        am_list_t *item = chat_list_new(proc, 2);
        if (!item) {
            am_list_destroy(proc->heap_alloc, lst);
            return chat_push_null(proc);
        }
        am_handle_t nhd = am_process_make_wstring_handle(proc, s_models[i].name, wcslen(s_models[i].name));
        if (nhd == AM_HANDLE_NULL) {
            am_list_destroy(proc->heap_alloc, item);
            am_list_destroy(proc->heap_alloc, lst);
            return chat_push_null(proc);
        }
        item = chat_list_add(proc, item, am_make_value_of_handle(nhd));
        if (!item) {
            am_list_destroy(proc->heap_alloc, lst);
            return chat_push_null(proc);
        }
        am_handle_t phd = chat_make_string(proc, s_models[i].rel, strlen(s_models[i].rel));
        if (phd == AM_HANDLE_NULL) {
            am_list_destroy(proc->heap_alloc, item);
            am_list_destroy(proc->heap_alloc, lst);
            return chat_push_null(proc);
        }
        item = chat_list_add(proc, item, am_make_value_of_handle(phd));
        if (!item) {
            am_list_destroy(proc->heap_alloc, lst);
            return chat_push_null(proc);
        }
        am_handle_t ihd = chat_list_box(proc, item);
        if (ihd == AM_HANDLE_NULL) {
            am_list_destroy(proc->heap_alloc, lst);
            return chat_push_null(proc);
        }
        s_menu_map[s_menu_num++] = i;
        lst = chat_list_add(proc, lst, am_make_value_of_handle(ihd));
        if (!lst) return chat_push_null(proc);
    }

    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Chat_load(am_runtime_t *rt, am_process_t *proc) {
    int32_t idx = -1;
    (void)rt;

    if (!chat_pop_int(proc, &idx)) return chat_push_int(proc, -1);
    if (idx < 0 || idx >= s_menu_num) {
        // 菜单未建立（未先调 (Chat.models)）：按 s_models 直取，便于调试
        if (idx < 0 || idx >= CHAT_MODEL_NUM) return chat_push_int(proc, -1);
        if (!chat_file_exists(s_models[idx].file)) return chat_push_int(proc, -1);
        s_menu_map[idx] = idx;
        s_menu_num = (s_menu_num > idx + 1) ? s_menu_num : idx + 1;
    }

    const chat_model_t *m = &s_models[s_menu_map[idx]];
    if (!chat_file_exists(m->file)) return chat_push_int(proc, -1);

    am_native_chat_release();

    // 种子取宿主墙钟（与 C 版鹦鹉笼 Math.random 同一哲学：每次载入不同生成流）
    uint64_t seed = chat_now_ms() * 2654435761ull + 12345ull;
    s_ctx = llm_context_init((char *)m->file, NULL, m->max_seq_len,
                             m->repetition_penalty, m->temperature, m->top_p,
                             m->top_k, seed);
    if (s_ctx == NULL) return chat_push_int(proc, -2);

    // 挂观测钩子（infer.c 的前向过程按相位回调；NULL 会导致解引用，必须挂上）
    s_ctx->observation = chat_observation;
    s_ctx->observation_env = NULL;

    s_model_sel = s_menu_map[idx];
    s_obs_layer = -1;
    s_obs_phase = 0;
    s_top_num = 0;
    s_pick = -1;
    s_last_state = 0;
    printf("[chat] 模型已载入：%ls（层 %u / 词表 %u / 最大序列 %u）\n",
           m->name, s_ctx->llm->config.n_layer, s_ctx->llm->config.vocab_size, m->max_seq_len);
    return chat_push_int(proc, 0);
}

int32_t am_native_Chat_unload(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_native_chat_release();
    return chat_push_bool(proc, true);
}

int32_t am_native_Chat_loaded(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return chat_push_bool(proc, s_ctx != NULL);
}

int32_t am_native_Chat_config(am_runtime_t *rt, am_process_t *proc) {
    LLM_Config cfg;
    (void)rt;

    if (!chat_config_of(&cfg)) return chat_push_null(proc);

    am_list_t *lst = chat_list_new(proc, 10);
    if (!lst) return chat_push_null(proc);
    const int64_t vals[10] = {
        (int64_t)s_ctx->llm->arch,
        (int64_t)cfg.block_size,
        (int64_t)cfg.vocab_size,
        (int64_t)cfg.n_layer,
        (int64_t)cfg.n_embd,
        (int64_t)cfg.n_head,
        (int64_t)cfg.n_kv_head,
        (int64_t)cfg.n_hidden,
        (int64_t)(cfg.head_dim ? cfg.head_dim : (cfg.n_embd / (cfg.n_head ? cfg.n_head : 1))),
        (int64_t)s_ctx->max_seq_len,
    };
    for (int32_t i = 0; i < 10; i++) {
        lst = chat_list_add(proc, lst, am_make_value_of_int((am_int_t)vals[i]));
        if (!lst) return chat_push_null(proc);
    }
    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Chat_set_seed(am_runtime_t *rt, am_process_t *proc) {
    am_float_t seed = 0.0f;
    (void)rt;

    if (!chat_pop_number(proc, &seed)) return chat_push_bool(proc, false);
    if (s_ctx == NULL || s_ctx->sampler == NULL) return chat_push_bool(proc, false);

    uint64_t v = (uint64_t)(seed < 0.0 ? 0.0 : seed);
    s_ctx->sampler->rng_state = v * 2654435761ull + 1ull;
    s_ctx->random_seed = v;
    return chat_push_bool(proc, true);
}

// ===============================================================================
// 分词器
// ===============================================================================

int32_t am_native_Chat_encode(am_runtime_t *rt, am_process_t *proc) {
    char text[2048];
    (void)rt;

    if (!chat_pop_utf8(proc, text, sizeof(text))) return chat_push_null(proc);
    if (s_ctx == NULL) return chat_push_null(proc);

    wchar_t wtext[2048];
    am_mbstowcs(wtext, text, (uint32_t)sizeof(wtext) / sizeof(wtext[0]));

    uint32_t n = 0;
    uint32_t *ids = NULL;
    if (s_ctx->llm->arch == LLM_ARCH_NANO) {
        ids = encode_nano(s_ctx->tokenizer, wtext, &n);
    }
    else {
        // encode_bpe 要求调用方预分配上界缓冲（不做 NULL 探测）：BPE 最坏情况是
        // 每个字节一个词元，+4 兜住空串/结尾
        size_t cap = strlen(text) + 4;
        ids = (uint32_t *)malloc(cap * sizeof(uint32_t));
        if (ids) {
            uint32_t m = (uint32_t)cap;
            encode_bpe(s_ctx->tokenizer, text, ids, &m);
            n = m;
        }
    }
    if (ids == NULL) return chat_push_null(proc);

    am_list_t *lst = chat_list_new(proc, n);
    if (!lst) {
        free(ids);
        return chat_push_null(proc);
    }
    for (uint32_t i = 0; i < n; i++) {
        lst = chat_list_add(proc, lst, am_make_value_of_uint((am_uint_t)ids[i]));
        if (!lst) {
            free(ids);
            return chat_push_null(proc);
        }
    }
    free(ids);

    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Chat_token_text(am_runtime_t *rt, am_process_t *proc) {
    int32_t id = 0;
    char buf[256];
    (void)rt;

    if (!chat_pop_int(proc, &id)) return chat_push_null(proc);
    if (id < 0) return chat_push_string(proc, "");
    if (!chat_token_utf8((uint32_t)id, buf, sizeof(buf))) return chat_push_null(proc);
    return chat_push_string(proc, buf);
}

// ===============================================================================
// 推理内核（一拍一步）
// ===============================================================================

int32_t am_native_Chat_begin(am_runtime_t *rt, am_process_t *proc) {
    char user[4096];
    (void)rt;

    if (!chat_pop_utf8(proc, user, sizeof(user))) return chat_push_int(proc, 0);
    if (s_ctx == NULL) return chat_push_int(proc, 0);

    if (s_session) {
        llm_session_free(s_session);
        s_session = NULL;
    }

    // prompt 组装：与 ui_llm.c 的 STATE_LLM_ON_INFER 初始化逐条对齐——
    //   Nano：<|instruct_mark|>用户输入<|response_mark|>
    //   Qwen：原文直传，聊天模板由 llm_session_init 内部套（apply_qwen*_chat_template）
    wchar_t wuser[4096];
    am_mbstowcs(wuser, user, (uint32_t)sizeof(wuser) / sizeof(wuser[0]));

    uint32_t max_seq = s_ctx->max_seq_len;
    wchar_t *prompt = (wchar_t *)calloc((size_t)max_seq + 64, sizeof(wchar_t));
    if (!prompt) return chat_push_int(proc, -1);

    if (s_ctx->llm->arch == LLM_ARCH_NANO) {
        wcscpy(prompt, L"<|instruct_mark|>");
        wcsncat(prompt, wuser, (size_t)max_seq - 32);
        wcscat(prompt, L"<|response_mark|>");
    }
    else {
        wcsncpy(prompt, wuser, (size_t)max_seq - 1);
        prompt[max_seq - 1] = L'\0';
    }

    s_session = llm_session_init(s_ctx, prompt, max_seq, 1);
    free(prompt);
    if (s_session == NULL) return chat_push_int(proc, -1);

    // 兜底：提示词元数必须落在会话缓冲容量内（infer.c 的 output_ids 为 max_seq_len+1，
    // 超长提示词会越界写）。本库已按字符数截断过用户文本，这里只做最后一道防线。
    if (s_session->num_prompt_tokens == 0 || s_session->num_prompt_tokens > max_seq) {
        llm_session_free(s_session);
        s_session = NULL;
        return chat_push_int(proc, -1);
    }

    s_t_begin_ms = chat_now_ms();
    s_t_prefill_end_ms = 0;
    s_t_last_ms = s_t_begin_ms;
    s_last_state = 0;
    s_pick = -1;
    s_top_num = 0;
    s_obs_layer = -1;
    s_obs_phase = 0;

    return chat_push_int(proc, (int64_t)s_session->num_prompt_tokens);
}

int32_t am_native_Chat_step(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;

    if (s_ctx == NULL || s_session == NULL) {
        s_last_state = 0;
        return chat_push_int(proc, 0);
    }

    // 本步是解码步吗？（llm_session_step 内部按 pos 判定，这里提前算同一条件，
    // 用于决定 logits 是否已是"采样分布"）
    int32_t will_decode = ((int32_t)s_session->pos >= (int32_t)s_session->num_prompt_tokens - 1) ? 1 : 0;

    s_trace_n = 0;   // 本步的相位轨迹（由 chat_observation 在钩子里追加）
    int32_t st = llm_session_step(s_ctx, s_session);
    uint64_t now = chat_now_ms();
    s_t_last_ms = now;
    s_last_state = st;

    if (will_decode && s_session->output_count > 0) {
        if (s_t_prefill_end_ms == 0) s_t_prefill_end_ms = now;
        s_pick = (int32_t)s_session->next_token;
        chat_update_topk(true);
        s_pick_prob = chat_prob_of((uint32_t)s_pick);
        if (s_session->output_count == 1) {
            // 首个解码步的诊断行（端到端测试据此断言"确实在采样"，也便于排查采样异常）
            printf("[chat] first decode pick=%u top:", (unsigned)s_session->next_token);
            for (int32_t q = 0; q < s_top_num; q++) printf(" %u=%.4f", s_top_id[q], s_top_prob[q]);
            printf("\n");
        }
    }
    else {
        s_pick = -1;
        chat_update_topk(false);
    }

    return chat_push_int(proc, (int64_t)st);
}

int32_t am_native_Chat_end(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (s_session) {
        llm_session_free(s_session);
        s_session = NULL;
    }
    s_pick = -1;
    s_last_state = 0;
    return chat_push_bool(proc, true);
}

int32_t am_native_Chat_output(am_runtime_t *rt, am_process_t *proc) {
    char buf[8192];
    (void)rt;
    chat_output_utf8(buf, sizeof(buf));
    return chat_push_string(proc, buf);
}

int32_t am_native_Chat_info(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;

    if (s_ctx == NULL || s_session == NULL) {
        // 无会话：返回 (0 0 0 0 0 0 0 -1)，调用方据此画"空闲"状态
        am_list_t *lst = chat_list_new(proc, 8);
        if (!lst) return chat_push_null(proc);
        for (int32_t i = 0; i < 7; i++) {
            lst = chat_list_add(proc, lst, am_make_value_of_int((am_int_t)0));
            if (!lst) return chat_push_null(proc);
        }
        lst = chat_list_add(proc, lst, am_make_value_of_int((am_int_t)-1));
        if (!lst) return chat_push_null(proc);
        am_handle_t lhd = chat_list_box(proc, lst);
        if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
        return chat_push(proc, am_make_value_of_handle(lhd));
    }

    uint64_t prefill_ms = 0;
    uint64_t decode_ms = 0;
    uint64_t tps = 0;

    if (s_t_prefill_end_ms > 0) {
        prefill_ms = s_t_prefill_end_ms - s_t_begin_ms;
        if (s_t_last_ms > s_t_prefill_end_ms) decode_ms = s_t_last_ms - s_t_prefill_end_ms;
        if (decode_ms > 0) tps = ((uint64_t)s_session->output_count * 1000ull) / decode_ms;
    }
    else if (s_t_last_ms > s_t_begin_ms) {
        // 预填充进行中：进度条用的耗时（不是最终值）
        prefill_ms = s_t_last_ms - s_t_begin_ms;
    }

    const int64_t vals[8] = {
        (int64_t)s_session->pos,
        (int64_t)s_session->num_prompt_tokens,
        (int64_t)s_session->output_count,
        (int64_t)tps,
        (int64_t)prefill_ms,
        (int64_t)decode_ms,
        (int64_t)s_last_state,
        (int64_t)((s_session->output_count > 0) ? s_pick : -1),
    };

    am_list_t *lst = chat_list_new(proc, 8);
    if (!lst) return chat_push_null(proc);
    for (int32_t i = 0; i < 8; i++) {
        lst = chat_list_add(proc, lst, am_make_value_of_int((am_int_t)vals[i]));
        if (!lst) return chat_push_null(proc);
    }
    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

// ===============================================================================
// 可视化
// ===============================================================================

int32_t am_native_Chat_obs(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_list_t *lst = chat_list_new(proc, 2);
    if (!lst) return chat_push_null(proc);
    lst = chat_list_add(proc, lst, am_make_value_of_int((am_int_t)s_obs_layer));
    if (!lst) return chat_push_null(proc);
    lst = chat_list_add(proc, lst, am_make_value_of_int((am_int_t)s_obs_phase));
    if (!lst) return chat_push_null(proc);
    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Chat_top_ids(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_list_t *lst = chat_list_new(proc, AM_NATIVE_CHAT_TOP_K);
    if (!lst) return chat_push_null(proc);
    for (int32_t i = 0; i < s_top_num; i++) {
        lst = chat_list_add(proc, lst, am_make_value_of_uint((am_uint_t)s_top_id[i]));
        if (!lst) return chat_push_null(proc);
    }
    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Chat_top_texts(am_runtime_t *rt, am_process_t *proc) {
    char buf[256];
    (void)rt;

    am_list_t *lst = chat_list_new(proc, AM_NATIVE_CHAT_TOP_K);
    if (!lst) return chat_push_null(proc);
    for (int32_t i = 0; i < s_top_num; i++) {
        if (!chat_token_utf8(s_top_id[i], buf, sizeof(buf))) buf[0] = '\0';
        am_handle_t hd = chat_make_string(proc, buf, strlen(buf));
        if (hd == AM_HANDLE_NULL) {
            am_list_destroy(proc->heap_alloc, lst);
            return chat_push_null(proc);
        }
        lst = chat_list_add(proc, lst, am_make_value_of_handle(hd));
        if (!lst) return chat_push_null(proc);
    }
    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Chat_trace_len(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return chat_push_int(proc, (int64_t)s_trace_n);
}

int32_t am_native_Chat_trace_at(am_runtime_t *rt, am_process_t *proc) {
    int32_t i = 0;
    (void)rt;

    if (!chat_pop_int(proc, &i)) return chat_push_null(proc);
    if (i < 0 || i >= s_trace_n) return chat_push_null(proc);

    int32_t lay = (int32_t)((s_trace[i] >> 8) & 0xFF) - 1;   // 还原成 -1/0../n_layer
    int32_t ph = (int32_t)(s_trace[i] & 0xFF);
    am_list_t *lst = chat_list_new(proc, 2);
    if (!lst) return chat_push_null(proc);
    lst = chat_list_add(proc, lst, am_make_value_of_int((am_int_t)lay));
    if (!lst) return chat_push_null(proc);
    lst = chat_list_add(proc, lst, am_make_value_of_int((am_int_t)ph));
    if (!lst) return chat_push_null(proc);
    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Chat_pick_prob(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    // 采样选中的词元**可能不在 top-K 里**（核采样从截断后的全词表抽），此时可视化
    // 需要单独补一列画出它，故这里给出它在本步分布下的概率
    if (s_ctx == NULL || s_session == NULL || s_session->output_count == 0 || s_pick < 0) {
        return chat_push(proc, am_make_value_of_float((am_float_t)0.0));
    }
    return chat_push(proc, am_make_value_of_float((am_float_t)s_pick_prob));
}

int32_t am_native_Chat_top_probs(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_list_t *lst = chat_list_new(proc, AM_NATIVE_CHAT_TOP_K);
    if (!lst) return chat_push_null(proc);
    for (int32_t i = 0; i < s_top_num; i++) {
        lst = chat_list_add(proc, lst, am_make_value_of_float((am_float_t)s_top_prob[i]));
        if (!lst) return chat_push_null(proc);
    }
    am_handle_t lhd = chat_list_box(proc, lst);
    if (lhd == AM_HANDLE_NULL) return chat_push_null(proc);
    return chat_push(proc, am_make_value_of_handle(lhd));
}

// ===============================================================================
// 库注册表
// ===============================================================================

static const am_native_func_entry_t am_native_Chat_funcs[] = {
    { L"models",       am_native_Chat_models },
    { L"load",         am_native_Chat_load },
    { L"unload",       am_native_Chat_unload },
    { L"loaded",       am_native_Chat_loaded },
    { L"config",       am_native_Chat_config },
    { L"set_seed",     am_native_Chat_set_seed },
    { L"encode",       am_native_Chat_encode },
    { L"token_text",   am_native_Chat_token_text },
    { L"begin",        am_native_Chat_begin },
    { L"step",         am_native_Chat_step },
    { L"end",          am_native_Chat_end },
    { L"output",       am_native_Chat_output },
    { L"info",         am_native_Chat_info },
    { L"obs",          am_native_Chat_obs },
    { L"trace_len",    am_native_Chat_trace_len },
    { L"trace_at",     am_native_Chat_trace_at },
    { L"top_ids",      am_native_Chat_top_ids },
    { L"top_texts",    am_native_Chat_top_texts },
    { L"top_probs",    am_native_Chat_top_probs },
    { L"pick_prob",    am_native_Chat_pick_prob },
};

const am_native_lib_entry_t am_native_Chat_lib = {
    L"Chat",
    am_native_Chat_funcs,
    sizeof(am_native_Chat_funcs) / sizeof(am_native_Chat_funcs[0])
};
