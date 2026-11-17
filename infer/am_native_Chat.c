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
#include "am_native_Fs.h"   // am_native_fs_resolve：与 (native Fs) 完全一致的路径规则

#include "platform.h"

#include "infer.h"
#include "tokenizer.h"

// ===============================================================================
// 模型加载配置（由 App 以**值**的形式传入）
//
//   C 版鹦鹉笼把「模型名 / 模型文件 / LoRA 文件 / 采样参数 / 上下文长度」硬编码在
//   ui_llm.c 的 preset_model_configs 里，模型菜单只是这张表的视图。灵机 App 反过来：
//   **配置是 App 的数据**（app/nanochat.scm 的 model_defs），本地库只提供「按参数
//   载入」这一个动作。好处：加模型 / 调采样参数是纯 Scheme 改动，C 侧零改动；
//   也符合本库「只封装引擎，界面与业务数据都在 Scheme」的分层。
//
//   路径规则与 (native Fs) 完全一致（am_native_fs_resolve）：相对业务资源根
//   PLATFORM_ROOT_DIR（桌面/WASM = /home/bd4sur/linglong，掌机 = SD 卡根）。
// ===============================================================================

#define CHAT_PATH_MAX (256)          // am_native_fs_resolve 要求的输出缓冲 ≥ 256 字节

// (Chat.load) 的配置列表**必需的**字段数（其后还可带一个可选的「思考模式」标记，
// 见 chat_load_cfg_t.is_thinking_model）
#define CHAT_CFG_FIELDS (7)

// Qwen3/Qwen3.5 思考模式的**两套采样参数**：逐项对齐 ui_llm.c 的
// qwen3_infer_args_thinking / qwen3_infer_args_no_thinking。C 版在每次建会话时按
// 开关就地改写 ctx->sampler（采样参数是 ctx 级而非会话级，这是既有取舍），本库同构。
#define CHAT_QWEN_THINK_TEMPERATURE    (0.6f)
#define CHAT_QWEN_THINK_TOP_P          (0.95f)
#define CHAT_QWEN_NO_THINK_TEMPERATURE (0.7f)
#define CHAT_QWEN_NO_THINK_TOP_P       (0.8f)

typedef struct {
    char     model_path[CHAT_PATH_MAX];  // 模型文件（已解析为宿主路径）
    char     lora_path[CHAT_PATH_MAX];   // LoRA 文件；空串 = 不挂 LoRA
    uint32_t max_seq_len;                // 上下文长度（KV-Cache 与生成长度上限）
    float    repetition_penalty;
    float    temperature;
    float    top_p;
    uint32_t top_k;
    int32_t  is_thinking_model;          // 该模型是否有思考模式（配置第 8 项，可选）
} chat_load_cfg_t;

// ===============================================================================
// 模块状态（虚拟机全局单例，与 C 版鹦鹉笼的引擎单例语义一致）
// ===============================================================================

static Nano_Context *s_ctx = NULL;        // 引擎上下文（模型 + 分词器 + 采样器）
static Nano_Session *s_session = NULL;    // 当前生成会话

// 思考模式（Qwen3/Qwen3.5 等可切换思考的模型；Nano 架构忽略）
//   s_think_model  ：本模型是否有思考模式（载入时由配置第 8 项给出）
//   s_think_enabled：用户开关（(Chat.set_thinking)，载入时复位为 1，与 C 版
//                    ui_llm_init_config 的 is_thinking_enabled 初值一致）
static int32_t s_think_model = 0;
static int32_t s_think_enabled = 1;

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

// 弹出一个**列表**对象（不复制，仅返回对象指针）
static bool chat_pop_list(am_process_t *proc, am_list_t **out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return false;

    *out = (am_list_t *)obj;
    return true;
}

// 列表第 i 项当数值取（任意数值表示；越界/非数值 → false）
static bool chat_list_number(am_process_t *proc, am_list_t *lst, size_t i, am_float_t *out) {
    am_value_t v = am_list_get(proc->heap_alloc, lst, i);
    if (am_value_is_float(v)) { *out = am_value_to_float(v); return true; }
    if (am_value_is_int(v))   { *out = (am_float_t)am_value_to_int(v); return true; }
    if (am_value_is_uint(v))  { *out = (am_float_t)am_value_to_uint(v); return true; }
    return false;
}

// 列表第 i 项当字符串对象取（越界/非字符串 → false）
static bool chat_list_wstring(am_process_t *proc, am_list_t *lst, size_t i, am_wstring_t **out) {
    am_value_t v = am_list_get(proc->heap_alloc, lst, i);
    if (!am_value_is_handle(v)) return false;

    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, am_value_to_handle(v));
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return false;

    *out = (am_wstring_t *)obj;
    return true;
}

// 宽字符串对象 → UTF-8（路径用；buf 至少 CHAT_PATH_MAX 字节）
static void chat_wstring_path(am_wstring_t *ws, char *buf, size_t cap) {
    wchar_t wbuf[CHAT_PATH_MAX];
    size_t n = ws->length;
    if (n > CHAT_PATH_MAX - 1) n = CHAT_PATH_MAX - 1;
    for (size_t i = 0; i < n; i++) wbuf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    wbuf[n] = L'\0';
    am_wcstombs(buf, wbuf, (uint32_t)cap);
}

// 浮点 → 无符号整数（负值/NaN 归 0；超上界钳到上界）
//   注意上界用 double 字面量比较：4294967295.0f 会被舍入成 2^32，float→uint32 转换
//   在 f == 2^32 时是未定义行为。
static uint32_t chat_float_to_uint(am_float_t f) {
    if (!(f > 0.0f)) return 0;
    if (f > 4294967295.0) return 4294967295u;
    return (uint32_t)f;
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

// 解析 (Chat.load) 的配置列表：
//   (模型文件 LoRA文件 上下文 复读惩罚 温度 top_p top_k)
// 失败（类型/长度不符、路径非法）返回 false，调用方按软失败处理。
static bool chat_parse_cfg(am_process_t *proc, chat_load_cfg_t *cfg) {
    am_list_t *lst = NULL;
    am_wstring_t *ws = NULL;
    am_float_t f = 0.0f;
    char raw[CHAT_PATH_MAX];

    memset(cfg, 0, sizeof(*cfg));
    if (!chat_pop_list(proc, &lst)) return false;
    if (lst->length < CHAT_CFG_FIELDS) return false;

    // [0] 模型文件（相对业务资源根，或 '/' 开头的宿主绝对路径）
    if (!chat_list_wstring(proc, lst, 0, &ws)) return false;
    if (!am_native_fs_resolve(ws, cfg->model_path)) return false;

    // [1] LoRA 文件（空串 = 不挂 LoRA）
    if (!chat_list_wstring(proc, lst, 1, &ws)) return false;
    chat_wstring_path(ws, raw, sizeof(raw));
    if (raw[0] != '\0') {
        if (!am_native_fs_resolve(ws, cfg->lora_path)) return false;
    }

    // [2] 上下文长度（词元数）
    if (!chat_list_number(proc, lst, 2, &f)) return false;
    cfg->max_seq_len = chat_float_to_uint(f);

    // [3] 复读惩罚
    if (!chat_list_number(proc, lst, 3, &f)) return false;
    cfg->repetition_penalty = (float)f;

    // [4] 温度
    if (!chat_list_number(proc, lst, 4, &f)) return false;
    cfg->temperature = (float)f;

    // [5] top-p
    if (!chat_list_number(proc, lst, 5, &f)) return false;
    cfg->top_p = (float)f;

    // [6] top-k（0 = 不做 top-k 截断，由 top-p 核采样决定）
    if (!chat_list_number(proc, lst, 6, &f)) return false;
    cfg->top_k = chat_float_to_uint(f);

    // [7]（可选）该模型是否有思考模式；缺省 0 = 普通模型
    cfg->is_thinking_model = 0;
    if (lst->length > CHAT_CFG_FIELDS) {
        if (chat_list_number(proc, lst, 7, &f)) {
            cfg->is_thinking_model = (f != 0.0f) ? 1 : 0;
        }
    }

    // 采样参数兜底：NaN / 非法值会让 softmax 输出全 NaN（生成一串垃圾词元），
    // 因此在这里收敛到「不做特殊处理」的默认值（与 build_sampler 的语义一致）。
    //   温度 0 是**合法**值（贪心采样），只拦负值与非有限值。
    if (!isfinite(cfg->repetition_penalty) || cfg->repetition_penalty <= 0.0f) cfg->repetition_penalty = 1.0f;
    if (!isfinite(cfg->temperature) || cfg->temperature < 0.0f) cfg->temperature = 1.0f;
    if (!isfinite(cfg->top_p) || cfg->top_p <= 0.0f || cfg->top_p > 1.0f) cfg->top_p = 1.0f;

    return true;
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
    s_think_model = 0;
    s_think_enabled = 1;
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

// 按 App 给出的配置载入模型（阻塞：解析权重 + 反量化 + 分配 KV-Cache）。
//   配置来源与字段顺序见本文件顶部 chat_load_cfg_t 的说明。
//   返回 0 成功 / -1 配置非法或文件缺失 / -2 引擎初始化失败。
int32_t am_native_Chat_load(am_runtime_t *rt, am_process_t *proc) {
    chat_load_cfg_t cfg;
    (void)rt;

    if (!chat_parse_cfg(proc, &cfg)) {
        printf("[chat] load: 配置列表非法（应为 (模型文件 LoRA文件 上下文 复读惩罚 温度 top_p top_k)）\n");
        return chat_push_int(proc, -1);
    }
    if (cfg.max_seq_len == 0) {
        printf("[chat] load: 上下文长度必须 > 0\n");
        return chat_push_int(proc, -1);
    }
    // 文件缺失绝不进 load_llm / load_lora：二者的失败路径是 exit(EXIT_FAILURE)，
    // 会中止整个进程（WASM 里就是整个运行时）。必须在此拦下。
    if (!chat_file_exists(cfg.model_path)) {
        printf("[chat] load: 模型文件缺失 %s\n", cfg.model_path);
        return chat_push_int(proc, -1);
    }
    if (cfg.lora_path[0] != '\0' && !chat_file_exists(cfg.lora_path)) {
        printf("[chat] load: LoRA 文件缺失 %s\n", cfg.lora_path);
        return chat_push_int(proc, -1);
    }

    am_native_chat_release();

    // 种子取宿主墙钟（与 C 版鹦鹉笼 Math.random 同一哲学：每次载入不同生成流）
    uint64_t seed = chat_now_ms() * 2654435761ull + 12345ull;
    s_ctx = llm_context_init(cfg.model_path,
                             (cfg.lora_path[0] != '\0') ? cfg.lora_path : NULL,
                             cfg.max_seq_len,
                             cfg.repetition_penalty, cfg.temperature, cfg.top_p,
                             cfg.top_k, seed);
    if (s_ctx == NULL) return chat_push_int(proc, -2);

    // 挂观测钩子（infer.c 的前向过程按相位回调；NULL 会导致解引用，必须挂上）
    s_ctx->observation = chat_observation;
    s_ctx->observation_env = NULL;

    s_think_model = cfg.is_thinking_model;
    s_think_enabled = 1;   // 与 C 版全局默认一致（载入后由 App 再同步用户选择）
    s_obs_layer = -1;
    s_obs_phase = 0;
    s_top_num = 0;
    s_pick = -1;
    s_last_state = 0;
    printf("[chat] 模型已载入：%s（架构 %u / 层 %u / 词表 %u / 上下文 %u / 复读 %.2f / 温度 %.2f / top-p %.2f / top-k %u / 思考 %d / LoRA %s）\n",
           cfg.model_path, s_ctx->llm->arch, s_ctx->llm->config.n_layer, s_ctx->llm->config.vocab_size,
           cfg.max_seq_len, cfg.repetition_penalty, cfg.temperature, cfg.top_p, cfg.top_k,
           cfg.is_thinking_model, (cfg.lora_path[0] != '\0') ? cfg.lora_path : "(无)");
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

// (Chat.set_thinking flag:Number) : Boolean —— 开关思考模式（仅对有思考模式的模型有意义）
//   与 C 版 ui_llm.c 的 is_thinking_enabled 同义：真正生效在**下一次 (Chat.begin)**——
//   关闭时追加 " /no_think"、切到无思考采样参数、并让会话模板插入空思考块。
//   未载入模型返回 #f；对无思考模式的模型（含 Nano）设置成功但不起作用（#t）。
int32_t am_native_Chat_set_thinking(am_runtime_t *rt, am_process_t *proc) {
    am_float_t flag = 0.0f;
    (void)rt;

    if (!chat_pop_number(proc, &flag)) return chat_push_bool(proc, false);
    if (s_ctx == NULL) return chat_push_bool(proc, false);

    s_think_enabled = (flag != 0.0f) ? 1 : 0;
    return chat_push_bool(proc, true);
}

// (Chat.thinking) : Number —— 思考模式的当前状态（供界面/测试查询）
//   -1 = 当前模型没有思考模式（或未载入）；0 = 关闭；1 = 开启
int32_t am_native_Chat_thinking(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (s_ctx == NULL || s_think_model == 0) return chat_push_int(proc, -1);
    return chat_push_int(proc, (int64_t)s_think_enabled);
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

    // prompt 组装 + 思考模式处理：与 ui_llm.c 的 STATE_LLM_ON_INFER 初始化逐条对齐——
    //   Nano：<|instruct_mark|>用户输入<|response_mark|>（无思考模式概念）
    //   Qwen：原文直传，聊天模板由 llm_session_init 内部套（apply_qwen*_chat_template）；
    //         思考模型再按开关追加 " /no_think" 并切换采样参数（见下）
    wchar_t wuser[4096];
    am_mbstowcs(wuser, user, (uint32_t)sizeof(wuser) / sizeof(wuser[0]));

    uint32_t max_seq = s_ctx->max_seq_len;
    wchar_t *prompt = (wchar_t *)calloc((size_t)max_seq + 64, sizeof(wchar_t));
    if (!prompt) return chat_push_int(proc, -1);

    // 传给 llm_session_init 的 enable_thinking（决定模板里是否插入空思考块）
    int32_t think = 1;
    // 诊断用：本次是否真的加了 " /no_think" 标记
    int32_t no_think_tag = 0;

    if (s_ctx->llm->arch == LLM_ARCH_NANO) {
        wcscpy(prompt, L"<|instruct_mark|>");
        wcsncat(prompt, wuser, (size_t)max_seq - 32);
        wcscat(prompt, L"<|response_mark|>");
    }
    else {
        if (s_think_model != 0) {
            if (s_think_enabled != 0) {
                // 思考开：用思考模式的采样参数（与 ui_llm.c 一致，就地改写 ctx 的采样器）
                s_ctx->sampler->temperature = CHAT_QWEN_THINK_TEMPERATURE;
                s_ctx->sampler->top_p = CHAT_QWEN_THINK_TOP_P;
                wcsncpy(prompt, wuser, (size_t)max_seq - 1);
                prompt[max_seq - 1] = L'\0';
                think = 1;
            }
            else {
                // 思考关：①用户文本末尾追加 " /no_think"（Qwen 的软开关）；
                //           ②切到无思考采样参数；③enable_thinking=0 让模板插入
                //           空思考块，模型因此直接作答而不吐思考内容。
                wcsncpy(prompt, wuser, (size_t)max_seq - 16);
                prompt[max_seq - 16] = L'\0';
                wcscat(prompt, L" /no_think");
                s_ctx->sampler->temperature = CHAT_QWEN_NO_THINK_TEMPERATURE;
                s_ctx->sampler->top_p = CHAT_QWEN_NO_THINK_TOP_P;
                think = 0;
                no_think_tag = 1;
            }
        }
        else {
            // 非思考模型（如 Qwen3-4B-Inst）：无论如何都不加 /no_think 标记；统一把
            // 思考标记打开，避免次元编码器输出多余的空思考占位词元（同 ui_llm.c）。
            wcsncpy(prompt, wuser, (size_t)max_seq - 1);
            prompt[max_seq - 1] = L'\0';
            think = 1;
        }
    }

    s_session = llm_session_init(s_ctx, prompt, max_seq, think);
    free(prompt);
    if (s_session == NULL) return chat_push_int(proc, -1);

    printf("[chat] begin think=%d no_think_tag=%d thinking_model=%d (temp=%.2f top-p=%.2f)\n",
           think, no_think_tag, s_think_model,
           s_ctx->sampler->temperature, s_ctx->sampler->top_p);

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
    { L"load",         am_native_Chat_load },
    { L"unload",       am_native_Chat_unload },
    { L"loaded",       am_native_Chat_loaded },
    { L"set_thinking", am_native_Chat_set_thinking },
    { L"thinking",     am_native_Chat_thinking },
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
