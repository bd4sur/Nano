// ===============================================================================
// 鹦鹉笼：端侧大语言模型推理及其可视化（自 ui_app.c 提取的独立模块，逻辑原样保留）
//   小鹦鹉笼（nano_min 极小内存引擎）已并入本模块（见下方“引擎适配层”隔离区块）：
//   二者共用同一套 UI 状态（STATE_MODEL_MENU / STATE_LLM_INPUT / STATE_LLM_ON_INFER /
//   STATE_LLM_AFTER_INFER），仅在 STATE_LLM_ON_INFER 内按引擎分发 会话建立/单步/销毁 三个钩子。
//   远端模型（云端 OpenAI 兼容流式 API）同样以引擎适配层形式并入（见“远端模型引擎适配层”
//   隔离区块）：推理后端为 hal_net 网络抽象层之上的 SSE 流式对话，交互与端侧模型一致。
// ===============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <math.h>

#include "graphics.h"
#include "hal_key.h"
#include "ui.h"
#include "ui_color.h"
#include "platform.h"

#include "infer.h"
#include "tokenizer.h"
#include "hal_net.h"  // 远端模型：网络通信 HAL（见本文件下方“远端模型引擎适配层”）

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    #include "model/nano_psycho_230k_q80.h"
#endif

#ifdef TTS_ENABLED
    #include "tts.h"
#endif

#include "ui_app.h"
#include "ui_llm.h"
#include "nano_min.h"  // 小鹦鹉笼引擎适配层（nano_min 极小内存引擎，见本文件下方隔离区块）

// ===============================================================================
// 核心业务：电子鹦鹉
// ===============================================================================

typedef struct {
    wchar_t *model_name;
    int32_t is_thinking_model;
    char *model_path;
    char *lora_path;
    float repetition_penalty;
    float temperature;
    float top_p;
    uint32_t top_k;
    uint32_t max_seq_len;
} Model_Config;

#define MODEL_CONFIG_ENTRY(name, is_think, m_path, l_path, rep_pen, temp, top_p_val, top_k_val, max_seq) \
    { \
        .model_name = (name), \
        .is_thinking_model = (is_think), \
        .model_path = (m_path), \
        .lora_path = (l_path), \
        .repetition_penalty = (rep_pen), \
        .temperature = (temp), \
        .top_p = (top_p_val), \
        .top_k = (top_k_val), \
        .max_seq_len = (max_seq) \
    }


static const Model_Config preset_model_configs[] = {
    MODEL_CONFIG_ENTRY(L"Nano-168M", 0, MODEL_ROOT_DIR "/nano-168m-q80.bin", NULL, 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Nano-56M", 0, MODEL_ROOT_DIR "/nano-56m-q80.bin", NULL, 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Nano-56M-Neko", 0, MODEL_ROOT_DIR "/nano-56m-base-q80.bin", MODEL_ROOT_DIR "/nano-56m-lora-neko.bin", 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Qwen3-0.6B", 1, MODEL_ROOT_DIR "/qwen3-0b6-q80.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-1.7B", 1, MODEL_ROOT_DIR "/qwen3-1b7-q80.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-4B-Inst", 0, MODEL_ROOT_DIR "/qwen3-4b-instruct-2507-q80.bin", NULL, 1.0f, 0.7f, 0.8f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-4B-Think", 1, MODEL_ROOT_DIR "/qwen3-4b-thinking-2507-q80.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Nano-168M-Q4KS", 0, MODEL_ROOT_DIR "/nano-168m-q4ks.bin", NULL, 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Nano-56M-Q4KS", 0, MODEL_ROOT_DIR "/nano-56m-q4ks.bin", NULL, 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Qwen3-0.6B-Q4KS", 1, MODEL_ROOT_DIR "/qwen3-0b6-q4ks.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-1.7B-Q4KS", 1, MODEL_ROOT_DIR "/qwen3-1b7-q4ks.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-4B-Inst-Q4KS", 0, MODEL_ROOT_DIR "/qwen3-4b-instruct-2507-q4ks.bin", NULL, 1.0f, 0.7f, 0.8f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-4B-Think-Q4KS", 1, MODEL_ROOT_DIR "/qwen3-4b-thinking-2507-q4ks.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768)
};

// Qwen3思考模式和非思考模式的参数不同：分别是temperature和top-p
static const float qwen3_infer_args_thinking[2] = {0.6f, 0.95f};
static const float qwen3_infer_args_no_thinking[2] = {0.7f, 0.8f};


// ===============================================================================
// 小鹦鹉笼引擎适配层（nano_min 极小内存引擎，自原 ui_nanochat 模块并入）
//   与 infer.c 标准引擎路径完全隔离：本区块的一切均为 static，仅通过下列统一钩子
//   被模型菜单与 STATE_LLM_ON_INFER 调用（会话视图统一挂到 gs->llm_session）：
//     llm_nm_model_num/name/enter（合并模型菜单）/ llm_nm_release（引擎释放，幂等）
//     llm_nm_is_active / llm_nm_session_begin/step/free（会话钩子，语义对齐
//     infer.c 的 llm_session_init/step/free）
//   模型与工作文件均位于 PLATFORM_ROOT_DIR "/llm" 目录：
//     - nano-168m-q80.bin / qwen3-0b6-q80.bin
//     - qwen3-0b6-q80.bin.bpeidx（BPE 索引，缺失时自动生成；建议在 PC 上生成后拷贝到 SD 卡）
//     - nano_min_work.tmp（KV-Cache + logits 工作文件，退出时删除）
// ===============================================================================

// 静态条目表容量上限（供合并模型菜单的定长数组使用）
#define LLM_NM_MODEL_NUM_MAX (8)

typedef struct {
    const wchar_t *name;        // 菜单显示名（兼作界面标题），带 [轻] 前缀
    const char    *model_file;  // 模型文件路径
    float rep_penalty, temperature, top_p; // 采样参数（对齐 preset_model_configs）
    uint32_t max_seq_len;
} LLM_NM_Model_Preset;

static const LLM_NM_Model_Preset s_llm_nm_models[] = {
    { L"[轻] Nano-168M-Q80",  PLATFORM_ROOT_DIR "/llm/nano-168m-q80.bin", 1.05f, 1.0f, 0.5f,  512 },
    { L"[轻] Qwen3-0.6B-Q80", PLATFORM_ROOT_DIR "/llm/qwen3-0b6-q80.bin", 1.0f,  0.6f, 0.95f, 512 },
};
#define LLM_NM_MODEL_NUM ((int32_t)(sizeof(s_llm_nm_models) / sizeof(s_llm_nm_models[0])))

#define LLM_NM_WORK_PATH PLATFORM_ROOT_DIR "/llm/nano_min_work.tmp"

// 引擎实例与模块内部状态（静态，避免侵入 Global_State）
static NM_Engine *s_nm_engine = NULL;
static uint32_t   s_nm_arch = NM_ARCH_NANO;
static const wchar_t *s_nm_model_name = L"";
static uint32_t   s_nm_max_seq_len = 512;

// 会话缓冲（首次加载模型时分配，退出功能时释放）
static uint32_t *s_nm_ids = NULL;        // (max_seq_len+1,) token 序列
static wchar_t  *s_nm_out_text = NULL;   // (UI_STR_BUF_MAX_LENGTH,) 当前输出文本
static char     *s_nm_out_bytes = NULL;  // (UI_STR_BUF_MAX_LENGTH*4,) QWEN 输出 UTF-8 累积

// 会话状态
static uint32_t s_nm_n_prompt = 0;
static uint32_t s_nm_pos = 0;
static uint32_t s_nm_token = 0;
static uint32_t s_nm_out_len = 0;
static uint32_t s_nm_out_bytes_len = 0;

// 释放引擎与会话缓冲（选中其他模型 / 退出模型菜单时经 ui_llm_unload_model 调用；幂等）
static void llm_nm_release(void) {
    if (s_nm_engine) { nm_close(s_nm_engine); s_nm_engine = NULL; }
    if (s_nm_ids)       { free(s_nm_ids);       s_nm_ids = NULL; }
    if (s_nm_out_text)  { free(s_nm_out_text);  s_nm_out_text = NULL; }
    if (s_nm_out_bytes) { free(s_nm_out_bytes); s_nm_out_bytes = NULL; }
}

// 当前是否由 nano_min 引擎承载对话（即选中的是 [轻] 模型）：1-是，0-否
static int32_t llm_nm_is_active(void) {
    return (s_nm_engine != NULL) ? 1 : 0;
}

// 生成一个 token 后，将其文本追加到输出缓冲（QWEN 先累积 UTF-8 字节再整体转换，
// 容忍跨 token 的不完整 UTF-8 序列——与 main_cli.c 的说明一致）
static void llm_nm_append_output(uint32_t tok) {
    if (s_nm_arch == NM_ARCH_NANO) {
        const wchar_t *s = nm_token_str(s_nm_engine, tok);
        size_t l = wcslen(s);
        if (s_nm_out_len + l < (uint32_t)UI_STR_BUF_MAX_LENGTH - 1) {
            wcscpy(s_nm_out_text + s_nm_out_len, s);
            s_nm_out_len += l;
        }
    }
    else {
        char buf[300];
        nm_bpe_token_str(s_nm_engine, tok, buf, sizeof(buf));
        size_t l = strlen(buf);
        if (s_nm_out_bytes_len + l < (uint32_t)UI_STR_BUF_MAX_LENGTH * 4 - 1) {
            memcpy(s_nm_out_bytes + s_nm_out_bytes_len, buf, l);
            s_nm_out_bytes_len += l;
            s_nm_out_bytes[s_nm_out_bytes_len] = 0;
            _mbstowcs(s_nm_out_text, s_nm_out_bytes, UI_STR_BUF_MAX_LENGTH - 1);
            s_nm_out_len = wcslen(s_nm_out_text);
        }
    }
}

// 模型数量与显示名（供合并模型菜单使用）
static int32_t llm_nm_model_num(void) {
    return LLM_NM_MODEL_NUM;
}

static const wchar_t *llm_nm_model_name(int32_t idx) {
    if (idx < 0 || idx >= LLM_NM_MODEL_NUM) return L"";
    return s_llm_nm_models[idx].name;
}

// 加载所选模型并初始化输入控件；返回 0 成功（调用方转 STATE_LLM_INPUT），-1 失败（已显示错误提示）
static int32_t llm_nm_model_enter(Key_Event *ke, Global_State *gs, int32_t idx) {
    if (idx < 0 || idx >= LLM_NM_MODEL_NUM) return -1;
    const LLM_NM_Model_Preset *m = &s_llm_nm_models[idx];

    // 加载提示（nm_open 为阻塞式，模型在 SD 卡上时可能耗时较长）
    ui_draw_header(ke, gs, L"模型加载中...", 1);
    gfx_refresh(gs->gfx);

    // 预检模型文件是否存在（nano_min 打开失败会直接退出进程，此处先拦截）
    if (platform_file_open(m->model_file) != 0) {
        platform_file_close();
        ui_draw_header(ke, gs, L"模型文件缺失", 1);
        gfx_refresh(gs->gfx);
        sleep_in_ms(1500);
        return -1;
    }
    platform_file_close();

    // 释放上一个引擎（若存在），再加载新模型
    if (s_nm_engine) { nm_close(s_nm_engine); s_nm_engine = NULL; }
    s_nm_engine = nm_open(m->model_file, LLM_NM_WORK_PATH, m->max_seq_len);
    if (!s_nm_engine) {
        // 加载失败（引擎内部已打印原因，如 BPE 索引构建内存不足/文件损坏）；
        // 显示错误并返回菜单，不再 exit 重启整机
        ui_draw_header(ke, gs, L"模型加载失败", 1);
        gfx_refresh(gs->gfx);
        sleep_in_ms(1500);
        return -1;
    }
    nm_set_sampler(s_nm_engine, m->rep_penalty, m->temperature, m->top_p, gs->timestamp);
    s_nm_arch = nm_get_arch(s_nm_engine);
    s_nm_max_seq_len = m->max_seq_len;
    s_nm_model_name = m->name;

    // 会话缓冲（首次分配，之后复用）
    if (!s_nm_ids) {
        s_nm_ids       = (uint32_t *)platform_calloc(s_nm_max_seq_len + 1, sizeof(uint32_t));
        s_nm_out_text  = (wchar_t  *)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
        s_nm_out_bytes = (char     *)platform_calloc(UI_STR_BUF_MAX_LENGTH * 4, sizeof(char));
        if (!s_nm_ids || !s_nm_out_text || !s_nm_out_bytes) {
            llm_nm_release();
            return -1;
        }
    }

    // 填充共享 UI 读取的全局模型信息（输入框/结果页标题、页脚统计、长度上限）
    gs->llm_model_name = (wchar_t *)m->name;
    gs->llm_is_thinking_model = 0;
    gs->llm_max_seq_len = m->max_seq_len;

    // nano_min 引擎无层级观测能力：若之前观测模式收窄过主文本框，恢复全宽
    gs->w_textarea_main->x = 0;
    gs->w_textarea_main->width = gs->gfx->width;

    // 初始化输入控件（复用全局输入控件实例，标题为模型名）
    ui_widget_input_init(ke, gs, gs->w_input_main, (wchar_t *)s_nm_model_name);
    return 0;
}

// 组装 prompt、编码为 token 序列，并建立 Nano_Session 会话视图挂到 gs->llm_session
// （进入 STATE_LLM_ON_INFER 首轮调用一次）；返回 0 成功，-1 失败
static int32_t llm_nm_session_begin(Key_Event *ke, Global_State *gs) {
    (void)ke;
    if (!s_nm_engine) return -1;
    wchar_t *text = gs->w_input_main->textarea.text;

    // 输入为空时随机选用一个预置 prompt（与标准引擎路径一致）
    if (wcslen(text) == 0) {
        set_random_prompt(text, gs->timestamp);
        gs->w_input_main->textarea.length = wcslen(text);
    }

    if (s_nm_arch == NM_ARCH_NANO) {
        // Nano 提示词模板（同标准引擎路径）
        wchar_t *prompt = (wchar_t *)platform_calloc(wcslen(text) + 64, sizeof(wchar_t));
        wcscpy(prompt, L"<|instruct_mark|>");
        wcscat(prompt, text);
        wcscat(prompt, L"<|response_mark|>");
        s_nm_n_prompt = nm_encode(s_nm_engine, prompt, s_nm_ids, s_nm_max_seq_len);
        free(prompt);
    }
    else {
        // ChatML 模板（同 tokenizer.c apply_qwen_chat_template，enable_thinking=1）
        uint32_t n = 0;
        s_nm_ids[n++] = 151644; // <|im_start|>
        s_nm_ids[n++] = 872;    // user
        s_nm_ids[n++] = 198;    // \n
        {
            size_t blen = wcslen(text) * 4 + 1;
            char *bytes = (char *)platform_calloc(blen, sizeof(char));
            _wcstombs(bytes, text, blen);
            n += nm_encode_bpe(s_nm_engine, bytes, s_nm_ids + n, s_nm_max_seq_len - n - 5);
            free(bytes);
        }
        s_nm_ids[n++] = 151645; // <|im_end|>
        s_nm_ids[n++] = 198;    // \n
        s_nm_ids[n++] = 151644; // <|im_start|>
        s_nm_ids[n++] = 77091;  // assistant
        s_nm_ids[n++] = 198;    // \n
        s_nm_n_prompt = n;
    }

    s_nm_pos = 0;
    s_nm_token = s_nm_ids[0];
    s_nm_out_len = 0;       s_nm_out_text[0] = 0;
    s_nm_out_bytes_len = 0; s_nm_out_bytes[0] = 0;

    // 建立 UI 层会话视图（与 infer.c 引擎共用 Nano_Session 结构，UI 回调直接读取）；
    // 各缓冲均为借用指针或 NULL，由 llm_nm_session_free 销毁视图时只释放视图本身
    Nano_Session *session = (Nano_Session *)platform_calloc(1, sizeof(Nano_Session));
    if (!session) return -1;
    session->num_prompt_tokens = s_nm_n_prompt;
    session->max_seq_len = s_nm_max_seq_len;
    session->is_prefilling = 1;
    session->output_text = text; // prefill 阶段回显输入 prompt（借用指针）
    gs->llm_session = session;
    return 0;
}

// 协作式单步推理（每轮主循环推进一步，同标准引擎路径）；返回 LLM 状态码（语义同 infer.h），不做 UI
static int32_t llm_nm_session_step(Key_Event *ke, Global_State *gs) {
    (void)ke;
    Nano_Session *session = gs->llm_session;

    if (s_nm_pos + 1 < s_nm_n_prompt) {
        // Pre-filling
        nm_forward(s_nm_engine, s_nm_ids[s_nm_pos], s_nm_pos);
        s_nm_pos++;
        session->pos = s_nm_pos;
        session->output_text = gs->w_input_main->textarea.text; // 回显输入 prompt（借用指针）
        return LLM_RUNNING_IN_PREFILLING;
    }

    // Decoding
    nm_forward(s_nm_engine, s_nm_token, s_nm_pos);
    uint32_t next_tok = nm_sample(s_nm_engine, s_nm_ids, s_nm_pos + 1);
    s_nm_pos++;
    session->pos = s_nm_pos;
    if (nm_is_eos(s_nm_engine, next_tok) || s_nm_pos >= s_nm_max_seq_len) {
        session->output_text = s_nm_out_text;
        return LLM_STOPPED_NORMALLY;
    }
    s_nm_ids[s_nm_pos] = next_tok;
    s_nm_token = next_tok;
    llm_nm_append_output(next_tok);
    session->output_text = s_nm_out_text;
    return LLM_RUNNING_IN_DECODING;
}

// 销毁会话视图（离开 STATE_LLM_ON_INFER 时调用）：只释放视图本身，
// 引擎与静态会话缓冲保留复用，由 llm_nm_release 统一释放
static void llm_nm_session_free(Global_State *gs) {
    if (gs->llm_session) {
        free(gs->llm_session);
        gs->llm_session = NULL;
    }
}

// ===============================================================================
// 远端模型引擎适配层（云端 OpenAI 兼容流式 API，经 hal_net 网络抽象层）
//   与 infer.c / nano_min 引擎路径完全隔离：本区块的一切均为 static，仅通过下列
//   统一钩子被模型菜单与 STATE_LLM_ON_INFER 调用（会话视图统一挂到 gs->llm_session）：
//     llm_rm_is_active / llm_rm_activate / llm_rm_release（引擎登记与释放，幂等）
//     llm_rm_session_begin/step/free（会话钩子，语义对齐 infer.c 的 llm_session_*）
//
//   与端侧模型的交互完全一致（模型菜单→输入→流式输出→结果页），差异仅在推理
//   后端：本地权重前向计算 → HTTP POST {API}/chat/completions（stream=true）。
//   服务端事件流（SSE）在本适配层解析：delta.reasoning_content（思考）与
//   delta.content（正文）按到达顺序拼接为会话输出文本；思考内容与正文之间以
//   分隔线区隔（对齐结果页既有的 "--------------------" 风格）。
//
//   API 信息（BD4SUR 2026-10；API 为内网自建服务，基址不入源码/文档，避免随仓库泄露内网拓扑）：
//     - 服务：内网自建 OpenAI 兼容 API（SSE 流式）；基址经环境变量 NANO_RM_API_BASE
//       或编译期宏 LLM_RM_API_BASE_BUILD 提供（见下方 llm_rm_api_base）
//     - 模型：GLM-5.3-Flash-EXL3；思考强度：思考模式开（默认）= max（reasoning_effort），
//       思考模式关（输入态 Ctrl+2 切换）= none（经实测校准，该 API 支持
//       none/minimal/low/medium/high/xhigh/max 七档）
//     - WASM（NANO_WASM_GUI）例外：浏览器同源策略下跨源 fetch 被阻（API 不支持
//       CORS 预检），故经测试服务的同源反向代理 /v1 访问（见 tools/https_static_server.py）
// ===============================================================================

// API 基址解析（源码不保存任何内网地址）：
//   WASM（NANO_WASM_GUI）：同源相对路径 /v1（浏览器同源策略下经测试服务反向代理，
//   规避 CORS，见 tools/https_static_server.py）；
//   其余平台（直连）：优先运行时环境变量 NANO_RM_API_BASE（如 http://<API主机>:<端口>/v1），
//   其次编译期宏 LLM_RM_API_BASE_BUILD（由本地私有构建配置注入，不入库）；
//   均未提供时返回 NULL，会话建立按「未配置」失败（结果页显示连接失败）。
static const char *llm_rm_api_base(void) {
#if defined(NANO_WASM_GUI)
    return "/v1";
#else
    const char *env = getenv("NANO_RM_API_BASE");
    if (env != NULL && env[0] != '\0') return env;
#ifdef LLM_RM_API_BASE_BUILD
    return LLM_RM_API_BASE_BUILD;
#else
    return NULL;
#endif
#endif
}
#define LLM_RM_MODEL_ID        "GLM-5.3-Flash-EXL3"
#define LLM_RM_REASONING_EFFORT_MAX  "max"   // 思考模式开（预设规格：思考强度 max）
#define LLM_RM_REASONING_EFFORT_NONE "none" // 思考模式关（Ctrl+2 切换）
#define LLM_RM_MENU_NAME       L"远端模型（云端API）"  // 模型菜单条目名
#define LLM_RM_MODEL_NAME      L"远端模型"             // 输入/推理/结果页标题名
#define LLM_RM_MAX_SEQ_LEN     (8192)                  // 结果页统计显示用上限（远端无本地KV限制）
#define LLM_RM_LINE_BUF_MAX    (64 * 1024)             // SSE 单行累积缓冲（一行即一个JSON对象）

// 引擎登记与会话状态（静态，避免侵入 Global_State）
static Net_Http_Request *s_rm_req = NULL;        // 进行中的流式请求（NULL-无）
static wchar_t   *s_rm_out_text = NULL;          // (UI_STR_BUF_MAX_LENGTH,) 累积输出文本
static char      *s_rm_out_bytes = NULL;         // (UI_STR_BUF_MAX_LENGTH*4,) 累积输出 UTF-8 字节
static uint32_t   s_rm_out_len = 0;              // s_rm_out_text 当前长度
static uint32_t   s_rm_out_bytes_len = 0;        // s_rm_out_bytes 当前长度
static char      *s_rm_line_buf = NULL;          // (LLM_RM_LINE_BUF_MAX,) SSE 行累积缓冲
static uint32_t   s_rm_line_len = 0;
static int32_t    s_rm_line_invalid = 0;         // 当前行超长被丢弃（内容无效）
static uint32_t   s_rm_chunks = 0;               // 已收到的流式块计数（词元数近似，供 TPS 统计）
static int32_t    s_rm_saw_reasoning = 0;        // 本会话已收到思考内容
static int32_t    s_rm_content_started = 0;      // 本会话已收到正文内容
static int32_t    s_rm_done = 0;                 // 已见到 data: [DONE]
static int32_t    s_rm_active = 0;               // 引擎登记标记
static int32_t    s_rm_http_status = 0;          // 最近一次响应的 HTTP 状态码（诊断显示用）

// 极简 JSON 巡游：仅服务于远端 API 的 SSE 数据行解析（结构不符一律安全返回 NULL）

// 跳过一个 JSON 值（字符串/对象/数组/数字/字面量），返回其结束后的首个字符指针
static const char *rm_json_skip_value(const char *p, const char *end) {
    if (p >= end) return NULL;
    if (*p == '"') {
        p++;
        while (p < end) {
            if (*p == '\\') { p += 2; continue; }
            if (*p == '"') return p + 1;
            p++;
        }
        return NULL;
    }
    if (*p == '{' || *p == '[') {
        char open = *p, close = (*p == '{') ? '}' : ']';
        int32_t depth = 0;
        while (p < end) {
            if (*p == '"') {
                p++;
                while (p < end && *p != '"') {
                    if (*p == '\\') p++;
                    p++;
                }
                if (p >= end) return NULL;
                p++;
                continue;
            }
            if (*p == open) depth++;
            else if (*p == close) {
                depth--;
                if (depth == 0) return p + 1;
            }
            p++;
        }
        return NULL;
    }
    while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n' && *p != '\r') p++;
    return p;
}

// 在对象（p 指向 '{'）中查找 key 对应的值，返回指向值首字符的指针；未找到返回 NULL
static const char *rm_json_find_member(const char *p, const char *end, const char *key) {
    if (p >= end || *p != '{') return NULL;
    p++;
    size_t key_len = strlen(key);
    for (;;) {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ',')) p++;
        if (p >= end || *p == '}') return NULL;
        if (*p != '"') return NULL;
        const char *k0 = p + 1;
        const char *k1 = k0;
        while (k1 < end && *k1 != '"') k1++;
        if (k1 >= end) return NULL;
        p = k1 + 1;
        while (p < end && *p != ':') p++;
        if (p >= end) return NULL;
        p++;
        while (p < end && (*p == ' ' || *p == '\t')) p++;
        if ((size_t)(k1 - k0) == key_len && memcmp(k0, key, key_len) == 0) return p;
        p = rm_json_skip_value(p, end);
        if (p == NULL) return NULL;
    }
}

// 将 p 指向的 JSON 字符串（'"' 开头）反序列化为 UTF-8 写入 out（容量 out_cap，含 \uXXXX
// 转义与代理对；\u 输出为 UTF-8）。返回写入字节数（不含末尾 0）；解析失败返回 -1。
static int32_t rm_json_unescape_string(const char *p, const char *end, char *out, size_t out_cap) {
    if (p >= end || *p != '"') return -1;
    p++;
    size_t w = 0;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            p++;
            if (p >= end) return -1;
            char e = *p++;
            switch (e) {
                case '"':  if (w < out_cap) out[w++] = '"';  break;
                case '\\': if (w < out_cap) out[w++] = '\\'; break;
                case '/':  if (w < out_cap) out[w++] = '/';  break;
                case 'b':  if (w < out_cap) out[w++] = '\b'; break;
                case 'f':  if (w < out_cap) out[w++] = '\f'; break;
                case 'n':  if (w < out_cap) out[w++] = '\n'; break;
                case 'r':  if (w < out_cap) out[w++] = '\r'; break;
                case 't':  if (w < out_cap) out[w++] = '\t'; break;
                case 'u': {
                    if (end - p < 4) return -1;
                    uint32_t cp = 0;
                    for (int32_t i = 0; i < 4; i++) {
                        char h = *p++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9')      cp |= (uint32_t)(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (uint32_t)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (uint32_t)(h - 'A' + 10);
                        else return -1;
                    }
                    // 高代理 + \uXXXX 低代理 → 合成增补平面码点
                    if (cp >= 0xD800 && cp <= 0xDBFF && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        uint32_t lo = 0;
                        int32_t ok = 1;
                        for (int32_t i = 0; i < 4; i++) {
                            char h = p[2 + i];
                            lo <<= 4;
                            if (h >= '0' && h <= '9')      lo |= (uint32_t)(h - '0');
                            else if (h >= 'a' && h <= 'f') lo |= (uint32_t)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') lo |= (uint32_t)(h - 'A' + 10);
                            else { ok = 0; break; }
                        }
                        if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            p += 6;
                        }
                    }
                    if (cp < 0x80) { if (w < out_cap) out[w++] = (char)cp; }
                    else if (cp < 0x800) { if (w + 1 < out_cap) { out[w++] = (char)(0xC0 | (cp >> 6)); out[w++] = (char)(0x80 | (cp & 0x3F)); } }
                    else if (cp < 0x10000) { if (w + 2 < out_cap) { out[w++] = (char)(0xE0 | (cp >> 12)); out[w++] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[w++] = (char)(0x80 | (cp & 0x3F)); } }
                    else { if (w + 3 < out_cap) { out[w++] = (char)(0xF0 | (cp >> 18)); out[w++] = (char)(0x80 | ((cp >> 12) & 0x3F)); out[w++] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[w++] = (char)(0x80 | (cp & 0x3F)); } }
                    break;
                }
                default: return -1;
            }
        }
        else {
            if (w < out_cap) out[w++] = *p; // 非 ASCII 原样 UTF-8 字节
            p++;
        }
    }
    if (p >= end) return -1; // 字符串未闭合
    return (int32_t)w;
}

// 找到 UTF-8 字节缓冲中“完整字符前缀”的长度（多字节序列被截断时截到最后一个完整字符）。
// 用于增量转换：网络数据块可能把一个多字节字符从中间切开。
static uint32_t rm_utf8_complete_prefix_len(const char *s, uint32_t len) {
    uint32_t i = len;
    uint32_t back = 0;
    while (back < 4 && i > 0) {
        i--; back++;
        uint8_t b = (uint8_t)s[i];
        if ((b & 0xC0) == 0x80) continue; // continuation 字节，继续回溯
        uint32_t seq_len = 1;
        if      ((b & 0xF8) == 0xF0) seq_len = 4;
        else if ((b & 0xF0) == 0xE0) seq_len = 3;
        else if ((b & 0xE0) == 0xC0) seq_len = 2;
        if (i + seq_len <= len) return len; // 末字符完整
        return i;                            // 末字符不完整，截到它之前
    }
    return len; // 末尾全是 continuation（异常流），原样转换
}

// 追加 UTF-8 文本到输出缓冲（增量转换；空间不足时静默截断，防堆溢出）
static void rm_append_output(const char *utf8, uint32_t len) {
    if (s_rm_out_bytes_len + len < (uint32_t)UI_STR_BUF_MAX_LENGTH * 4 - 4) {
        memcpy(s_rm_out_bytes + s_rm_out_bytes_len, utf8, len);
        s_rm_out_bytes_len += len;
        s_rm_out_bytes[s_rm_out_bytes_len] = 0;
    }
    // 整体重转换（对齐 nano_min 适配层做法；缓冲有界，开销可接受）。
    // 末尾被截断的多字节字符留在字节缓冲，待后续数据到齐再转换，不产生 '?'；
    // 转换上限为 wchar 缓冲容量（纯 ASCII 时字节数可达 wchar 数的 4 倍，必须封顶防溢出）
    uint32_t complete = rm_utf8_complete_prefix_len(s_rm_out_bytes, s_rm_out_bytes_len);
    if (complete > (uint32_t)UI_STR_BUF_MAX_LENGTH - 1) complete = (uint32_t)UI_STR_BUF_MAX_LENGTH - 1;
    if (complete > 0) {
        _mbstowcs(s_rm_out_text, s_rm_out_bytes, complete);
        s_rm_out_len = wcslen(s_rm_out_text);
    }
}

// 处理一个完整的 SSE 数据载荷（data: 之后的部分）
static void rm_process_sse_payload(const char *p, size_t len) {
    if (len == 6 && memcmp(p, "[DONE]", 6) == 0) {
        s_rm_done = 1;
        return;
    }
    const char *end = p + len;
    const char *choices = rm_json_find_member(p, end, "choices");
    if (choices == NULL) return;
    while (choices < end && (*choices == ' ' || *choices == '\t')) choices++;
    if (choices >= end || *choices != '[') return;
    choices++;
    while (choices < end && (*choices == ' ' || *choices == '\t')) choices++;
    const char *delta = rm_json_find_member(choices, end, "delta");
    if (delta == NULL) return;

    char text[8192];
    // 思考内容（reasoning_content 为主，兼容 reasoning 键）
    const char *rv = rm_json_find_member(delta, end, "reasoning_content");
    if (rv == NULL) rv = rm_json_find_member(delta, end, "reasoning");
    if (rv != NULL && rv < end && *rv == '"') {
        int32_t n = rm_json_unescape_string(rv, end, text, sizeof(text) - 1);
        if (n > 0) {
            if (!s_rm_saw_reasoning) s_rm_saw_reasoning = 1;
            s_rm_chunks++; // 每个携带内容的流式块 ≈ 1 词元（TPS 统计用）
            rm_append_output(text, (uint32_t)n);
        }
    }
    // 正文内容
    const char *cv = rm_json_find_member(delta, end, "content");
    if (cv != NULL && cv < end && *cv == '"') {
        int32_t n = rm_json_unescape_string(cv, end, text, sizeof(text) - 1);
        if (n > 0) {
            if (s_rm_saw_reasoning && !s_rm_content_started) {
                rm_append_output("\n--------------------\n", 22); // 思考/正文分隔线（对齐结果页风格）
            }
            s_rm_content_started = 1;
            s_rm_chunks++;
            rm_append_output(text, (uint32_t)n);
        }
    }
}

// 处理一行完整的 SSE 数据（不含行尾换行符）
static void rm_process_sse_line(const char *line, size_t len) {
    if (len > 5 && memcmp(line, "data:", 5) == 0) {
        const char *p = line + 5;
        while (len > (size_t)(p - line) && *p == ' ') p++;
        rm_process_sse_payload(p, len - (size_t)(p - line));
    }
    // 其余行（event:/id:/注释行/空行）忽略
}

// 当前是否由远端引擎承载对话：1-是，0-否
static int32_t llm_rm_is_active(void) {
    return s_rm_active;
}

// 登记远端引擎（菜单选中时调用；会话缓冲在此懒分配）
static void llm_rm_activate(void) {
    s_rm_active = 1;
    if (!s_rm_out_text) {
        s_rm_out_text  = (wchar_t *)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
        s_rm_out_bytes = (char *)platform_calloc(UI_STR_BUF_MAX_LENGTH * 4, sizeof(char));
        s_rm_line_buf  = (char *)platform_calloc(LLM_RM_LINE_BUF_MAX, sizeof(char));
    }
}

// 释放远端引擎与会话遗留资源（选中其他模型 / 退出模型菜单时经 ui_llm_unload_model 调用；幂等）
static void llm_rm_release(void) {
    s_rm_active = 0;
    if (s_rm_req) { net_http_close(s_rm_req); s_rm_req = NULL; }
    if (s_rm_out_text)  { free(s_rm_out_text);  s_rm_out_text = NULL;  }
    if (s_rm_out_bytes) { free(s_rm_out_bytes); s_rm_out_bytes = NULL; }
    if (s_rm_line_buf)  { free(s_rm_line_buf);  s_rm_line_buf = NULL;  }
    s_rm_out_len = 0; s_rm_out_bytes_len = 0;
    s_rm_line_len = 0; s_rm_line_invalid = 0;
    s_rm_chunks = 0; s_rm_saw_reasoning = 0; s_rm_content_started = 0;
    s_rm_done = 0; s_rm_http_status = 0;
}

// 组装请求体并发起流式请求，建立会话视图挂到 gs->llm_session
// （进入 STATE_LLM_ON_INFER 首轮调用一次）；返回 0 成功，-1 失败
static int32_t llm_rm_session_begin(Key_Event *ke, Global_State *gs) {
    (void)ke;
    if (s_rm_active == 0) return -1;
    wchar_t *text = gs->w_input_main->textarea.text;

    // 输入为空时随机选用一个预置 prompt（与端侧引擎路径一致）
    if (wcslen(text) == 0) {
        set_random_prompt(text, gs->timestamp);
        gs->w_input_main->textarea.length = wcslen(text);
    }

    // 关闭遗留请求（上一会话异常退出等；幂等）
    if (s_rm_req) { net_http_close(s_rm_req); s_rm_req = NULL; }

    // 会话缓冲（首次分配，之后复用）
    if (!s_rm_out_text) {
        llm_rm_activate();
        if (!s_rm_out_text || !s_rm_out_bytes || !s_rm_line_buf) return -1;
    }

    // 重置会话状态
    s_rm_out_len = 0;        s_rm_out_text[0] = 0;
    s_rm_out_bytes_len = 0;  s_rm_out_bytes[0] = 0;
    s_rm_line_len = 0;       s_rm_line_invalid = 0;
    s_rm_chunks = 0;         s_rm_saw_reasoning = 0;
    s_rm_content_started = 0; s_rm_done = 0;
    s_rm_http_status = 0;

    // 用户输入 wchar → UTF-8 → JSON 字符串转义
    size_t utf8_cap = wcslen(text) * 4 + 16;
    char *user_utf8 = (char *)platform_calloc(utf8_cap, 1);
    if (!user_utf8) return -1;
    _wcstombs(user_utf8, text, utf8_cap);

    size_t esc_cap = strlen(user_utf8) * 6 + 16;
    char *esc = (char *)platform_calloc(esc_cap, 1);
    if (!esc) { free(user_utf8); return -1; }
    {
        size_t w = 0;
        for (const uint8_t *p = (const uint8_t *)user_utf8; *p; p++) {
            const char *e = NULL;
            switch (*p) {
                case '"':  e = "\\\""; break;
                case '\\': e = "\\\\"; break;
                case '\n': e = "\\n";  break;
                case '\r': e = "\\r";  break;
                case '\t': e = "\\t";  break;
                case '\b': e = "\\b";  break;
                case '\f': e = "\\f";  break;
                default:
                    if (*p < 0x20) {
                        w += (size_t)snprintf(esc + w, esc_cap - w, "\\u%04x", *p);
                    }
                    else if (w + 1 < esc_cap) {
                        esc[w++] = (char)*p;
                    }
                    continue;
            }
            size_t l = strlen(e);
            if (w + l < esc_cap) { memcpy(esc + w, e, l); w += l; }
        }
        esc[w] = 0;
    }

    // 请求体：OpenAI 兼容对话补全（流式）。思考强度：思考模式开（默认）= max（远端模型
    // 预设规格），思考模式关（输入态 Ctrl+2 可切换，页眉 Ψ 标记）= none（实测该 API
    // 支持 none/minimal/low/medium/high/xhigh/max，none 时不产生 reasoning_content）
    size_t body_cap = strlen(esc) + 320;
    char *body = (char *)platform_calloc(body_cap, 1);
    if (!body) { free(esc); free(user_utf8); return -1; }
    snprintf(body, body_cap,
             "{\"model\": \"%s\", \"messages\": [{\"role\": \"user\", \"content\": \"%s\"}],"
             " \"stream\": true, \"reasoning_effort\": \"%s\"}",
             LLM_RM_MODEL_ID, esc, (gs->is_thinking_enabled != 0) ? LLM_RM_REASONING_EFFORT_MAX : LLM_RM_REASONING_EFFORT_NONE);

    // 解析 API 基址并拼接完整 URL（net_http_post_stream 内部深拷贝，此后即可释放本缓冲）
    const char *api_base = llm_rm_api_base();
    if (api_base == NULL) {
        fprintf(stderr, "[llm_rm] API base not configured (set NANO_RM_API_BASE or -DLLM_RM_API_BASE_BUILD)\n");
        free(body); free(esc); free(user_utf8);
        return -1;
    }
    size_t url_cap = strlen(api_base) + 32;
    char *url = (char *)platform_calloc(url_cap, 1);
    if (!url) { free(body); free(esc); free(user_utf8); return -1; }
    snprintf(url, url_cap, "%s/chat/completions", api_base);

    Net_Http_Request_Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.url = url;
    cfg.body = (const uint8_t *)body;
    cfg.body_len = strlen(body);
    cfg.headers[0].key = "Content-Type";        cfg.headers[0].value = "application/json";
    cfg.headers[1].key = "Accept";              cfg.headers[1].value = "text/event-stream";
    cfg.header_num = 2;

    printf("[llm_rm] POST %s (model=%s, body=%zu bytes)\n", cfg.url, LLM_RM_MODEL_ID, cfg.body_len);
    int32_t rc = net_http_post_stream(&cfg, &s_rm_req);
    free(url); free(body); free(esc); free(user_utf8);
    if (rc != 0 || s_rm_req == NULL) {
        s_rm_req = NULL;
        return -1;
    }

    // 建立 UI 层会话视图（与 infer.c/nano_min 引擎共用 Nano_Session 结构，UI 回调直接读取）；
    // output_text 为借用指针（s_rm_out_text），由 llm_rm_session_free 销毁视图时只释放视图本身
    Nano_Session *session = (Nano_Session *)platform_calloc(1, sizeof(Nano_Session));
    if (!session) {
        net_http_close(s_rm_req); s_rm_req = NULL;
        return -1;
    }
    session->num_prompt_tokens = 1; // 远端无 prefill；预取回调永不触发（避免进度条除零）
    session->max_seq_len = gs->llm_max_seq_len;
    session->is_prefilling = 0;
    session->output_text = s_rm_out_text;
    gs->llm_session = session;
    return 0;
}

// 协作式单步推理（每轮主循环拉取一次网络流；返回 LLM 状态码，语义同 infer.h，不做 UI）
static int32_t llm_rm_session_step(Key_Event *ke, Global_State *gs) {
    (void)ke;
    Nano_Session *session = gs->llm_session;
    if (s_rm_req == NULL || session == NULL) return LLM_STOPPED_WITH_ERROR;

    // 非阻塞拉取全部已到达数据并逐行解析 SSE
    uint8_t rbuf[4096];
    int32_t stream_err = 0;
    for (;;) {
        int32_t n = net_http_stream_read(s_rm_req, rbuf, sizeof(rbuf));
        if (n < 0) { stream_err = 1; break; }
        if (n == 0) break;
        for (int32_t i = 0; i < n; i++) {
            if (rbuf[i] == '\n') {
                if (s_rm_line_len > 0 && s_rm_line_invalid == 0) {
                    if (s_rm_line_buf[s_rm_line_len - 1] == '\r') s_rm_line_len--;
                    s_rm_line_buf[s_rm_line_len] = 0;
                    rm_process_sse_line(s_rm_line_buf, s_rm_line_len);
                }
                s_rm_line_len = 0;
                s_rm_line_invalid = 0;
            }
            else if (s_rm_line_len < LLM_RM_LINE_BUF_MAX - 1) {
                s_rm_line_buf[s_rm_line_len++] = (char)rbuf[i];
            }
            else {
                s_rm_line_invalid = 1; // 行超长：丢弃（正常 SSE 行远小于此）
            }
        }
    }

    s_rm_http_status = net_http_status_code(s_rm_req);
    Net_Stream_State st = net_http_stream_state(s_rm_req);
    session->pos = s_rm_chunks;
    session->output_text = s_rm_out_text;

    // 输出缓冲即将写满：主动结束（远端无 max_seq_len 硬约束，以 UI 缓冲容量为限）
    if (s_rm_out_len >= (uint32_t)UI_STR_BUF_MAX_LENGTH - 512) {
        printf("[llm_rm] output buffer full, stop\n");
        return LLM_STOPPED_NORMALLY;
    }

    // 见到 data: [DONE]：无论后续连接状态如何，服务端已正常完成
    if (s_rm_done) {
        return LLM_STOPPED_NORMALLY;
    }
    if (stream_err && st != NET_STREAM_DONE) {
        // 流错误：把诊断信息附到输出尾部（结果页可见），按异常结束处理
        if (s_rm_http_status > 0) {
            char note[64];
            snprintf(note, sizeof(note), "\n[远端模型：HTTP %ld]", (long)s_rm_http_status);
            rm_append_output(note, (uint32_t)strlen(note));
        }
        else {
            const char *note = "\n[远端模型：连接失败]";
            rm_append_output(note, (uint32_t)strlen(note));
        }
        return LLM_STOPPED_WITH_ERROR;
    }
    if (st == NET_STREAM_DONE) {
        return LLM_STOPPED_NORMALLY;
    }

    return LLM_RUNNING_IN_DECODING;
}

// 销毁会话视图（离开 STATE_LLM_ON_INFER 时调用）：关闭请求、释放视图本身；
// 引擎登记与静态会话缓冲保留复用，由 llm_rm_release 统一释放
static void llm_rm_session_free(Global_State *gs) {
    if (s_rm_req) { net_http_close(s_rm_req); s_rm_req = NULL; }
    if (gs->llm_session) {
        free(gs->llm_session);
        gs->llm_session = NULL;
    }
    printf("[llm_rm] session closed (chunks=%u, http_status=%d)\n", s_rm_chunks, s_rm_http_status);
}

// ===============================================================================
// 推理过程 UI 回调（两套引擎共用：经 gs->llm_session 会话视图读取进度与输出）
// ===============================================================================

int32_t on_llm_prefilling(Key_Event *key_event, Global_State *global_state) {
    Nano_Session *session = global_state->llm_session;

    if (session->t_0 == 0) {
        session->t_0 = global_state->timestamp;
    }
    else {
        session->tps = (session->pos - 1) / (float)(global_state->timestamp - session->t_0) * 1000;
    }

    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        wcscpy(global_state->llm_output_of_last_session, L"");
        global_state->tps_of_last_session = session->tps;
        global_state->token_num_of_last_session = session->pos;
        return LLM_STOPPED_IN_PREFILLING;
    }

    // PREFILL_LED_ON

    // 屏幕刷新节流
    if (global_state->timestamp - global_state->llm_refresh_timestamp > (1000 / global_state->llm_refresh_max_fps)) {
        // 临时关闭draw_textarea的gfx_refresh，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
        global_state->is_full_refresh = 0;

        // 清屏
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            gfx_fill_white(global_state->gfx);
        }
        else if (global_state->ui_color_style == UI_COLOR_DARK) {
            gfx_soft_clear(global_state->gfx);
        }

        // 显示界面标题
        wchar_t prefill_title_str[50];
        swprintf(prefill_title_str, 50, L"%ls Reading...", global_state->llm_model_name);
        ui_draw_header(key_event, global_state, prefill_title_str, 1);

        // 显示已经处理的输入prompt（复用主文本框控件）
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, session->output_text, -1, 1);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

        // 进度条
        uint8_t progress_R = 102, progress_G = 204, progress_B = 255;
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            progress_R = 102; progress_G = 204; progress_B = 255;
        }
        else if (global_state->ui_color_style == UI_COLOR_DARK) {
            progress_R = 102; progress_G = 204; progress_B = 255;
        }
        // 进度条位于页脚（底部状态栏）上缘，页脚高度跟随当前字体行高（行高 + 1px 边距）
        uint32_t pg_bottom_y = global_state->gfx->height - (gfx_font_line_height(global_state->ui_font) + 1);
        uint32_t pgpos_x = MIN(global_state->gfx->width - 1, session->pos * global_state->gfx->width / (session->num_prompt_tokens - 1));
        gfx_draw_line(global_state->gfx, 1, (pg_bottom_y - 1), pgpos_x, (pg_bottom_y - 1), progress_R, progress_G, progress_B, 1);
        gfx_draw_line(global_state->gfx, 1, (pg_bottom_y - 2), pgpos_x, (pg_bottom_y - 2), progress_R, progress_G, progress_B, 1);

        // 进度百分比
        wchar_t progress_str[30];
        swprintf(progress_str, 30, L"%d/%d", session->pos, session->num_prompt_tokens);
        ui_draw_footer(key_event, global_state, progress_str, 1);

        gfx_refresh(global_state->gfx);

        // 重新开启整帧绘制，注意这个标记是所有函数共享的全局标记。
        global_state->is_full_refresh = 1;

        global_state->llm_refresh_timestamp = global_state->timestamp;
    }

#ifdef TTS_ENABLED
    reset_tts_split_status();
#endif

    // PREFILL_LED_OFF
    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_llm_decoding(Key_Event *key_event, Global_State *global_state) {
    Nano_Session *session = global_state->llm_session;

    if (session->t_0 == 0) {
        session->t_0 = global_state->timestamp;
    }
    else {
        session->tps = (session->pos - 1) / (float)(global_state->timestamp - session->t_0) * 1000;
    }

    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        if (session->output_text) {
            wcscpy(global_state->llm_output_of_last_session, session->output_text);
        }
        global_state->tps_of_last_session = session->tps;
        global_state->token_num_of_last_session = session->pos;
        return LLM_STOPPED_IN_DECODING;
    }

    // DECODE_LED_ON

    // 屏幕刷新节流
    if (global_state->timestamp - global_state->llm_refresh_timestamp > (1000 / global_state->llm_refresh_max_fps)) {
        // 标题
        wchar_t title_str[50];
        swprintf(title_str, 50, L"%ls Decoding...", global_state->llm_model_name);
        ui_draw_header(key_event, global_state, title_str, 1);

        // 底部
        wchar_t tps_str[50];
        swprintf(tps_str, 50, L"%ls | %d/%d | %.1f词元/秒", global_state->llm_model_name, session->pos, global_state->llm_max_seq_len, session->tps);
        ui_draw_footer(key_event, global_state, tps_str, 1);

        // 刷新输出文本
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, session->output_text, -1, 1);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        global_state->llm_refresh_timestamp = global_state->timestamp;
    }

    // DECODE_LED_OFF

#ifdef TTS_ENABLED
    if (global_state->tts_req_mode > 0) {
        send_tts_request(session->output_text, 0);
    }
#endif

    return LLM_RUNNING_IN_DECODING;
}

int32_t on_llm_finished(Key_Event *key_event, Global_State *global_state) {
    Nano_Session *session = global_state->llm_session;

    session->t_1 = global_state->timestamp;
    session->tps = (session->pos - 1) / (float)(session->t_1 - session->t_0) * 1000;

    if (session->output_text) {
        // 缓冲区容量为 UI_STR_BUF_MAX_LENGTH * 2 个 wchar_t（含结尾 L'\0'），截断拷贝防堆溢出
        wcsncpy(global_state->llm_output_of_last_session, session->output_text, UI_STR_BUF_MAX_LENGTH * 2 - 1);
        global_state->llm_output_of_last_session[UI_STR_BUF_MAX_LENGTH * 2 - 1] = L'\0';
    }

    // 将本轮对话写入日志
    // write_chat_log(LOG_FILE_PATH, global_state->timestamp, session->prompt, global_state->llm_output_of_last_session);

#ifdef TTS_ENABLED
    if (global_state->tts_req_mode > 0) {
        send_tts_request(session->output_text, 1);
    }
    reset_tts_split_status();
#endif

    global_state->tps_of_last_session = session->tps;
    global_state->token_num_of_last_session = session->pos;

    return LLM_STOPPED_NORMALLY;
}


// WASM 版模型菜单：白名单 + 文件存在性动态过滤（外部挂载体系仅含 linglong/llm 的
// Nano-168M-Q80 / Nano-56M-Q80 两个标准模型；不带 [轻] 前缀的 nano_min 技术试验条目
// 不列出；其余 preset 条目（Qwen/Q4KS/LoRA 等）文件未挂载自动隐去——其加载路径
//（infer.c load_llm）缺失时 exit(EXIT_FAILURE) 会中止整个 WASM 运行时，必须隐去）。
// 菜单序号 → preset 表索引的映射由本表维护（init_model_menu 填、action 查）。
#if defined(NANO_WASM_GUI)
#define WASM_MODEL_MENU_MAX (16)
static int32_t s_wasm_mm_index[WASM_MODEL_MENU_MAX]; // 在 preset_model_configs 中的序号
static int32_t s_wasm_mm_num = 0;

// WASM 可选模型白名单（按 MODEL_ROOT_DIR 相对文件名匹配）
static const char *s_wasm_model_whitelist[] = {
    "nano-168m-q80.bin",
    "nano-56m-q80.bin",
};
#define WASM_MODEL_WHITELIST_NUM ((int32_t)(sizeof(s_wasm_model_whitelist) / sizeof(s_wasm_model_whitelist[0])))

// 文件存在性探测（fopen 直探；不占用 platform_file_open 的全局单句柄）
static int32_t wasm_model_file_exists(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) return 0;
    fclose(fp);
    return 1;
}
#endif

void init_model_menu(Key_Event *key_event, Global_State *global_state) {
#if defined(NANO_WASM_GUI)
    static const wchar_t *wasm_items[WASM_MODEL_MENU_MAX + 1]; // +1：末尾追加“远端模型”条目
    s_wasm_mm_num = 0;
    // preset 标准引擎条目：白名单 + 文件存在性双重过滤
    for (size_t i = 0; i < sizeof(preset_model_configs) / sizeof(preset_model_configs[0])
            && s_wasm_mm_num < WASM_MODEL_MENU_MAX; i++) {
        const char *mp = preset_model_configs[i].model_path;
        int32_t whitelisted = 0;
        for (int32_t w = 0; w < WASM_MODEL_WHITELIST_NUM; w++) {
            size_t wl = strlen(s_wasm_model_whitelist[w]);
            size_t pl = strlen(mp);
            if (pl >= wl && strcmp(mp + pl - wl, s_wasm_model_whitelist[w]) == 0) {
                whitelisted = 1;
                break;
            }
        }
        if (!whitelisted || !wasm_model_file_exists(mp)) continue;
        wasm_items[s_wasm_mm_num] = preset_model_configs[i].model_name;
        s_wasm_mm_index[s_wasm_mm_num] = (int32_t)i;
        s_wasm_mm_num++;
    }
    global_state->w_menu_main->title = L"选择语言模型";
    if (s_wasm_mm_num > 0) {
        // 本地模型之后追加“远端模型”条目（无本地文件依赖，始终可用）
        wasm_items[s_wasm_mm_num] = LLM_RM_MENU_NAME;
        global_state->w_menu_main->items = wasm_items;
        global_state->w_menu_main->item_num = s_wasm_mm_num + 1;
    }
    else {
        // 无可用本地模型：仅提供远端模型
        static const wchar_t *wasm_remote_only[] = { LLM_RM_MENU_NAME };
        global_state->w_menu_main->items = wasm_remote_only;
        global_state->w_menu_main->item_num = 1;
    }
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
    return;
#endif
    size_t model_count = sizeof(preset_model_configs) / sizeof(preset_model_configs[0]);
    int32_t nmchat_count = llm_nm_model_num();
    if (nmchat_count > LLM_NM_MODEL_NUM_MAX) nmchat_count = LLM_NM_MODEL_NUM_MAX;
    // 条目字符串借用 preset_model_configs 与小鹦鹉笼预设表的静态存储，控件不复制；
    // 合并菜单 = 大模型（infer.c 引擎） + [轻]小鹦鹉笼模型（nano_min 极小内存引擎） + 远端模型
    static const wchar_t *model_menu_items[sizeof(preset_model_configs) / sizeof(preset_model_configs[0]) + LLM_NM_MODEL_NUM_MAX + 1];
    for (size_t i = 0; i < model_count; i++) {
        model_menu_items[i] = preset_model_configs[i].model_name;
    }
    for (int32_t i = 0; i < nmchat_count; i++) {
        model_menu_items[model_count + i] = llm_nm_model_name(i);
    }
    model_menu_items[model_count + nmchat_count] = LLM_RM_MENU_NAME;
    global_state->w_menu_main->title = L"选择语言模型";
    global_state->w_menu_main->items = model_menu_items;
    global_state->w_menu_main->item_num = (int32_t)model_count + nmchat_count + 1;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
}





void ui_app_llm_model_diagram_draw(Key_Event *key_event, Global_State *global_state, int32_t x0, int32_t y0, int32_t total_layers, Nano_Observation obs) {
    Nano_GFX *gfx = global_state->gfx;
    int32_t layer = obs.layer;
    int32_t phase = obs.phase;

    // 色彩
    uint8_t bg_R = 0x00, bg_G = 0x00, bg_B = 0x00;
    uint8_t line_R = 0x99, line_G = 0x99, line_B = 0x99;
    uint8_t block_R = 0x33, block_G = 0x33, block_B = 0x33;
    uint8_t block_active_R = 0x00, block_active_G = 0xff, block_active_B = 0x00;
    uint8_t text_R = 0xcc, text_G = 0xcc, text_B = 0xcc;
    uint8_t text_active_R = 0xff, text_active_G = 0xff, text_active_B = 0xff;

    // 绘制连线
    gfx_draw_line(gfx, x0+55, y0+14, x0+55, y0+50, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+50, x0+95, y0+50, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+79, x0+95, y0+79, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+50, x0+15, y0+79, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+95, y0+50, x0+95, y0+79, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+55, y0+79, x0+55, y0+133, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+35, y0+133, x0+95, y0+133, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+35, y0+133, x0+35, y0+160, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+95, y0+133, x0+95, y0+203, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+160, x0+55, y0+160, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+203, x0+15, y0+160, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+55, y0+160, x0+55, y0+227, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+203, x0+95, y0+203, line_R, line_G, line_B, 1);

    gfx_draw_line(gfx, x0+36, y0+103, x0+55, y0+103, line_R, line_G, line_B, 1); // Res Branch
    gfx_draw_line(gfx, x0+53, y0+103, x0+57, y0+103, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+55, y0+101, x0+55, y0+105, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+54, y0+103, x0+56, y0+103, line_R+10, line_G+10, line_B+10, 1);
    gfx_draw_line(gfx, x0+55, y0+102, x0+55, y0+104, line_R+10, line_G+10, line_B+10, 1);
    gfx_draw_textline_centered(gfx, L"X1", x0+36-6, y0+103, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+36, y0+20, x0+55, y0+20, line_R, line_G, line_B, 1); // Res Branch
    gfx_draw_line(gfx, x0+53, y0+20, x0+57, y0+20, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+55, y0+20-2, x0+55, y0+20+2, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+54, y0+20, x0+56, y0+20, line_R+10, line_G+10, line_B+10, 1);
    gfx_draw_line(gfx, x0+55, y0+20-1, x0+55, y0+20+1, line_R+10, line_G+10, line_B+10, 1);
    gfx_draw_textline_centered(gfx, L"X2", x0+36-6, y0+20, line_R, line_G, line_B, 1);


    // 绘制方框和文字

    uint8_t bR = block_R, bG = block_G, bB = block_B;
    uint8_t tR = text_R, tG = text_G, tB = text_B;

    // NANO_LLM_PHASE_W2
    if (phase == NANO_LLM_PHASE_W2) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+40, y0+25, 30, 14, bR, bG, bB, 1); // W2
    gfx_draw_textline_centered(gfx, L"W2", x0+40+15, y0+25+7, tR, tG, tB, 1);
    gfx_draw_circle_fill(gfx, x0+55, y0+50, 6, bR, bG, bB, 1); // FFN Hadamard
    gfx_draw_line(gfx, x0+49, y0+44, x0+49+12, y0+44+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+49+12, y0+44, x0+49, y0+44+12, bg_R, bg_G, bg_B, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_W1W3
    if (phase == NANO_LLM_PHASE_W1W3) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+0, y0+61, 30, 14, bR, bG, bB, 1); // W1
    gfx_draw_textline_centered(gfx, L"W1", x0+0+15, y0+61+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+80, y0+43, 30, 14, bR, bG, bB, 1); // SiLU
    gfx_draw_textline_centered(gfx, L"SiLU", x0+80+15, y0+43+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+80, y0+61, 30, 14, bR, bG, bB, 1); // W3
    gfx_draw_textline_centered(gfx, L"W3", x0+80+15, y0+61+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_FFN_NORM
    if (phase == NANO_LLM_PHASE_FFN_NORM) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+40, y0+83, 30, 14, bR, bG, bB, 1); // FFN Norm
    gfx_draw_textline_centered(gfx, L"Norm", x0+40+15, y0+83+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_O
    if (phase == NANO_LLM_PHASE_O) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+40, y0+109, 30, 14, bR, bG, bB, 1); // O
    gfx_draw_textline_centered(gfx, L"O", x0+40+15, y0+109+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_MHA
    if (phase == NANO_LLM_PHASE_MHA) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+29, y0+138, 12, 12, bR, bG, bB, 1); // Mask
    gfx_draw_line(gfx, x0+29, y0+138, x0+29+12, y0+138+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_circle_fill(gfx, x0+55, y0+133, 6, bR, bG, bB, 1); // A*V
    gfx_draw_line(gfx, x0+49, y0+127, x0+49+12, y0+127+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+49+12, y0+127, x0+49, y0+127+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_circle_fill(gfx, x0+35, y0+160, 6, bR, bG, bB, 1); // Q*K
    gfx_draw_line(gfx, x0+29, y0+154, x0+29+12, y0+154+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+29+12, y0+154, x0+29, y0+154+12, bg_R, bg_G, bg_B, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_QK_ROPE
    if (phase == NANO_LLM_PHASE_QK_ROPE) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+0, y0+168, 30, 14, bR, bG, bB, 1); // RoPE Q
    gfx_draw_textline_centered(gfx, L"RoPE", x0+0+15, y0+168+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+40, y0+168, 30, 14, bR, bG, bB, 1); // RoPE K
    gfx_draw_textline_centered(gfx, L"RoPE", x0+40+15, y0+168+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_QKV
    if (phase == NANO_LLM_PHASE_QKV) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+0, y0+185, 30, 14, bR, bG, bB, 1); // Q
    gfx_draw_textline_centered(gfx, L"Q", x0+0+15, y0+185+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+40, y0+185, 30, 14, bR, bG, bB, 1); // K
    gfx_draw_textline_centered(gfx, L"K", x0+40+15, y0+185+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+80, y0+185, 30, 14, bR, bG, bB, 1); // V
    gfx_draw_textline_centered(gfx, L"V", x0+80+15, y0+185+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_ATTN_NORM
    if (phase == NANO_LLM_PHASE_ATTN_NORM) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+40, y0+207, 30, 14, bR, bG, bB, 1); // Attn Norm
    gfx_draw_textline_centered(gfx, L"Norm", x0+40+15, y0+207+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;


    // 绘制模型各层
    int32_t H = gfx->height - 14 - 14;
    int32_t layer_h = floorf(H / ((total_layers+2) * 2.0f));
    int32_t delta_y = (int32_t)floorf((float)(H - layer_h) / (float)((total_layers+2) - 1));
    int32_t y_pos = y0 + 14;
    for (int32_t ll = total_layers+1; ll >= 0; ll--) { // 包含Embd和Cls额外两层
        if ((layer == -1 && ll == 0) || (ll == layer + 1)) {
            bR = 0x00; bG = 0xff; bB = 0xff;
        }
        else {
            bR = block_R; bG = block_G; bB = block_B;
        }
        gfx_draw_rectangle(gfx, x0+120, y_pos, 16, layer_h, bR, bG, bB, 1);

        y_pos += delta_y;
    }

    // 显示top6
    static uint32_t tokens[6];
    Nano_Context *ctx = global_state->llm_ctx;
    if (obs.token_0) {
        tokens[0] = obs.token_0;
        tokens[1] = obs.token_1;
        tokens[2] = obs.token_2;
        tokens[3] = obs.token_3;
        tokens[4] = obs.token_4;
        tokens[5] = obs.token_5;
    }
    for (int32_t i = 0; i < 6; i++) {
        wchar_t *top_token_text = NULL;
        if (ctx->llm->arch == LLM_ARCH_NANO) {
            top_token_text = decode_nano(ctx->tokenizer, tokens + i, 1);
        }
        else if (ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3) {
            top_token_text = decode_bpe(ctx->tokenizer, tokens + i, 1);
        }
        else {
            return;
        }
        gfx_draw_textline(gfx, top_token_text, x0+ 140, y0+140+i*13, 255, 255, 0, 1);
        free(top_token_text);
    }

}


void llm_observation(Nano_Observation obs, void *env) {
    Global_State *global_state = (Global_State*)env;
    if (!global_state->llm_enable_observation) return;

    Nano_GFX *gfx = global_state->gfx;
    int32_t total_layers = global_state->llm_ctx->llm->config.n_layer;

    gfx_draw_rectangle(gfx, 0, 14, gfx->width/2, gfx->height-14-14, 0, 0, 0, 1);
    ui_app_llm_model_diagram_draw(NULL, global_state, 0, 0, total_layers, obs);
    gfx_refresh(gfx);
}

// 模型菜单“远端模型”条目动作：登记远端引擎并进入输入状态（无本地加载过程，
// 交互与端侧模型一致；推理时经 hal_net 调云端 OpenAI 兼容流式 API）
static int32_t ui_llm_remote_model_select(Key_Event *ke, Global_State *gs) {
    ui_llm_unload_model(gs); // 卸载旧模型（含远端会话遗留资源）
    gs->llm_model_name = (wchar_t *)LLM_RM_MODEL_NAME;
    gs->llm_is_thinking_model = 1;
    gs->llm_max_seq_len = LLM_RM_MAX_SEQ_LEN;
    llm_rm_activate();
    ui_widget_input_init(ke, gs, gs->w_input_main, (wchar_t *)LLM_RM_MODEL_NAME);
    return STATE_LLM_INPUT;
}

int32_t model_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
#if defined(NANO_WASM_GUI)
    // WASM：菜单序号经映射表找到 preset 表索引；末尾条目为“远端模型”（空表时唯一条目）
    int32_t mi = ms->current_item_index;
    if (mi == s_wasm_mm_num) {
        return ui_llm_remote_model_select(ke, gs);
    }
    if (mi < 0 || mi >= s_wasm_mm_num) return STATE_MODEL_MENU;
    {
        ui_llm_unload_model(gs); // 卸载旧模型（远端/本地互斥）
        Model_Config mc = preset_model_configs[s_wasm_mm_index[mi]];
        gs->llm_model_name = mc.model_name;
        gs->llm_is_thinking_model = mc.is_thinking_model;
        gs->llm_model_path = mc.model_path;
        gs->llm_lora_path = mc.lora_path;
        gs->llm_repetition_penalty = mc.repetition_penalty;
        gs->llm_temperature = mc.temperature;
        gs->llm_top_p = mc.top_p;
        gs->llm_top_k = mc.top_k;
        gs->llm_max_seq_len = mc.max_seq_len;
        goto wasm_load_preset_model; // 汇入下方原有加载流程
    }
#endif
    int32_t item_index = ms->current_item_index;

    // 选中任一模型前，卸载旧模型（释放 infer.c 上下文 / nano_min 引擎 / 远端引擎），避免并存
    ui_llm_unload_model(gs);

    int32_t model_count = (int32_t)(sizeof(preset_model_configs) / sizeof(preset_model_configs[0]));
    int32_t nm_count = llm_nm_model_num();
    if (nm_count > LLM_NM_MODEL_NUM_MAX) nm_count = LLM_NM_MODEL_NUM_MAX;

    if (item_index == model_count + nm_count) {
        // 远端模型（云端 OpenAI 兼容流式 API）：无本地加载过程，直接进入输入状态
        return ui_llm_remote_model_select(ke, gs);
    }

    if (item_index >= model_count) {
        // [轻] 小鹦鹉笼模型（nano_min 极小内存引擎）：由引擎适配层加载，
        // 成功后与标准引擎路径共用输入状态（STATE_LLM_INPUT）
        if (llm_nm_model_enter(ke, gs, item_index - model_count) == 0) {
            return STATE_LLM_INPUT;
        }
        // 加载失败（错误提示已由其显示）：重绘模型菜单并停留（本状态未离开焦点，需手动重绘）
        ui_draw_header(ke, gs, (wchar_t *)gs->w_menu_main->title, 1);
        ui_widget_menu_refresh(ke, gs, gs->w_menu_main);
        return STATE_MODEL_MENU;
    }

    if (item_index >= 0) {
        Model_Config mc = preset_model_configs[item_index];
        gs->llm_model_name = mc.model_name;
        gs->llm_is_thinking_model = mc.is_thinking_model;
        gs->llm_model_path = mc.model_path;
        gs->llm_lora_path = mc.lora_path;
        gs->llm_repetition_penalty = mc.repetition_penalty;
        gs->llm_temperature = mc.temperature;
        gs->llm_top_p = mc.top_p;
        gs->llm_top_k = mc.top_k;
        gs->llm_max_seq_len = mc.max_seq_len;
    }
    else {
        return STATE_MAIN_MENU;
    }

    wchar_t llm_loading_prompt[88];
#if defined(NANO_WASM_GUI)
wasm_load_preset_model: ; // WASM 大模型路径汇入点（字段已在宏门控分支内设置）
#endif
    swprintf(llm_loading_prompt, 88, L" 正在加载语言模型\n %ls\n 请稍等...", gs->llm_model_name);

    ui_widget_textarea_set(ke, gs, gs->w_textarea_main, llm_loading_prompt, 0, 0);
    ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    gs->llm_ctx = llm_context_init_from_buffer(
        (uint8_t *)psycho_230k_1214_q80,
        gs->llm_max_seq_len,
        1.0,
        1.0,
        0.8,
        20,
        gs->timestamp);
#else
    gs->llm_ctx = llm_context_init(
        gs->llm_model_path,
        gs->llm_lora_path,
        gs->llm_max_seq_len,
        gs->llm_repetition_penalty,
        gs->llm_temperature,
        gs->llm_top_p,
        gs->llm_top_k,
        gs->timestamp);
#endif


    gs->llm_ctx->observation = llm_observation;
    gs->llm_ctx->observation_env = gs; // 模拟闭包：将观测函数的词法环境指向UI全局上下文，这样就可以在观测回调中使用UI的API

    if (gs->llm_enable_observation) {
        gs->w_textarea_main->x = 160;
        gs->w_textarea_main->width = gs->gfx->width - 160;
    }

    // 进入电子鹦鹉
    ui_widget_input_init(ke, gs, gs->w_input_main, gs->llm_model_name);
    return STATE_LLM_INPUT;
}



// 带容量上限的安全追加（剩余空间不足时直接截断，防堆溢出）
static void ui_app_wcscat_bounded(wchar_t *dst, const wchar_t *src, uint32_t cap) {
    size_t cur = wcslen(dst);
    if (cur < cap - 1) {
        wcsncat(dst, src, cap - 1 - cur);
    }
}

// ===============================================================================
// 模块生命周期
// ===============================================================================

// 卸载当前模型（选中其他模型 / 退出模型菜单时调用；幂等）：
// 释放 infer.c 上下文（KV cache/分词器/采样器等常驻PSRAM）与小鹦鹉笼引擎（若已加载）
void ui_llm_unload_model(Global_State *global_state) {
    if (global_state->llm_ctx) {
        llm_context_free(global_state->llm_ctx);
        global_state->llm_ctx = NULL; // 释放后置NULL，避免悬垂指针被二次释放
    }
    llm_nm_release();
    llm_rm_release();
}

// LLM 相关全局字段初始化（由 ui_init 调用）
void ui_llm_init_config(Global_State *global_state) {
    global_state->llm_status = LLM_STOPPED_NORMALLY;
    global_state->llm_model_name = NULL;
    global_state->llm_is_thinking_model = 0;
    global_state->llm_model_path = NULL;
    global_state->llm_lora_path = NULL;
    global_state->llm_repetition_penalty = 1.05f;
    global_state->llm_temperature = 1.0f;
    global_state->llm_top_p = 0.5f;
    global_state->llm_top_k = 0;
    global_state->llm_max_seq_len = 512;
    global_state->is_thinking_enabled = 1;
    global_state->llm_output_of_last_session = (wchar_t*)platform_calloc(UI_STR_BUF_MAX_LENGTH * 2, sizeof(wchar_t));
    global_state->tps_of_last_session = 0.0f;
    global_state->token_num_of_last_session = 0;
    global_state->llm_enable_observation = 0;
    global_state->llm_refresh_max_fps = 10;
    global_state->llm_refresh_timestamp = 0;
}

// LLM 资源释放（由 main_deinit 调用）
void ui_llm_deinit(Global_State *global_state) {
    llm_context_free(global_state->llm_ctx);
    llm_nm_release();
    llm_rm_release();
    free(global_state->llm_output_of_last_session);
}

// ===============================================================================
// 状态机处理器（自 ui_app.c 主状态机提取，逻辑原样保留）
// ===============================================================================

// STATE_LLM_INPUT：文字编辑器状态
int32_t ui_llm_input_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 首次获得焦点：初始化
    if (global_state->PREV_STATE != global_state->STATE) {
        ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
    }
    global_state->PREV_STATE = global_state->STATE;

#ifdef ASR_ENABLED
    // 长按D键：开始PTT
    if (key_event->key_edge == -2 && key_event->key_code == NANO_KEY_enter) {
        return STATE_ASR_RUNNING;
    }
#endif

    return ui_widget_input_event_handler(key_event, global_state, global_state->w_input_main, STATE_MODEL_MENU, STATE_LLM_INPUT, STATE_LLM_ON_INFER);
}

// 按当前引擎销毁对话 session（infer.c 会话 / nano_min 会话视图 / 远端模型会话）
static void ui_llm_session_free(Global_State *global_state) {
    if (llm_nm_is_active()) {
        llm_nm_session_free(global_state);
    }
    else if (llm_rm_is_active()) {
        llm_rm_session_free(global_state);
    }
    else {
        llm_session_free(global_state->llm_session);
        global_state->llm_session = NULL;
    }
}

// STATE_LLM_ON_INFER：语言推理进行中（异步，每个iter结束后会将控制权交还事件循环，
// 而非自行阻塞到最后一个token；实际上就是将generate_sync的while循环打开，置于大的事件循环）
//   小鹦鹉笼（nano_min 极小内存引擎）共用本状态：仅 会话建立/单步/销毁 三个钩子按引擎分发，
//   UI 回调（on_llm_prefilling/on_llm_decoding/on_llm_finished）经 llm_session 会话视图零分支复用
int32_t ui_llm_on_infer_event_handler(Key_Event *key_event, Global_State *global_state) {
    int32_t use_nm_engine = llm_nm_is_active(); // 当前由小鹦鹉笼（nano_min 引擎）承载？
    int32_t use_rm_engine = llm_rm_is_active(); // 当前由远端模型（云端流式 API）承载？

    // 首次获得焦点：初始化对话 session
    if (global_state->PREV_STATE != global_state->STATE) {
        if (use_nm_engine) {
            // 小鹦鹉笼：prompt 模板组装与编码由引擎适配层完成，会话视图挂到 llm_session
            if (llm_nm_session_begin(key_event, global_state) != 0) {
                return STATE_LLM_INPUT;
            }
        }
        else if (use_rm_engine) {
            // 远端模型：请求体组装与流式请求发起由引擎适配层完成，会话视图挂到 llm_session
            if (llm_rm_session_begin(key_event, global_state) != 0) {
                return STATE_LLM_INPUT;
            }
        }
        else {
        wchar_t *prompt = (wchar_t*)platform_calloc(global_state->llm_max_seq_len + 1, sizeof(wchar_t));

        // 如果输入为空，则随机选用一个预置prompt
        if (wcslen(global_state->w_input_main->textarea.text) == 0) {
            set_random_prompt(global_state->w_input_main->textarea.text, global_state->timestamp);
            global_state->w_input_main->textarea.length = wcslen(global_state->w_input_main->textarea.text);
        }

        // 根据模型类型应用prompt模板（NOTE 注意：prompt模板会占用max_seq_len长度）
        if (global_state->llm_ctx->llm->arch == LLM_ARCH_NANO) {
            wcscat(prompt, L"<|instruct_mark|>");
            wcscat(prompt, global_state->w_input_main->textarea.text);
            wcscat(prompt, L"<|response_mark|>");
        }
        else if (global_state->llm_ctx->llm->arch == LLM_ARCH_QWEN2 || global_state->llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
            wcscpy(prompt, global_state->w_input_main->textarea.text);
            // Qwen思考模型：涉及主动添加/no_think标记和生成参数调整
            if (global_state->llm_is_thinking_model != 0) {
                if (global_state->is_thinking_enabled == 0) {
                    wcscat(prompt, L" /no_think");
                    // TODO 采样参数应该是session的参数，而不是ctx的参数
                    global_state->llm_ctx->sampler->temperature = qwen3_infer_args_no_thinking[0];
                    global_state->llm_ctx->sampler->top_p = qwen3_infer_args_no_thinking[1];
                }
                else {
                    global_state->llm_ctx->sampler->temperature = qwen3_infer_args_thinking[0];
                    global_state->llm_ctx->sampler->top_p = qwen3_infer_args_thinking[1];
                }
            }
            // Qwen非思考模型：无论如何都不加/no_think标记；统一将思考标记打开，避免次元编码器输出多余的<think></think>占位词元
            else {
                global_state->is_thinking_enabled = 1;
            }
        }
        else {
            return STATE_SPLASH_SCREEN;
        }

        // 初始化对话session
        global_state->llm_session = llm_session_init(global_state->llm_ctx, prompt, global_state->llm_max_seq_len, global_state->is_thinking_enabled);
        // session内部已复制prompt（infer.c llm_session_init），此处释放原缓冲，避免每轮泄漏
        free(prompt);
        }
    }
    global_state->PREV_STATE = global_state->STATE;

    // 事件循环主体：即同步版本的while(1)的循环体

    global_state->llm_status = (use_nm_engine)
        ? llm_nm_session_step(key_event, global_state)
        : (use_rm_engine)
        ? llm_rm_session_step(key_event, global_state)
        : llm_session_step(global_state->llm_ctx, global_state->llm_session);

    if (global_state->llm_status == LLM_RUNNING_IN_PREFILLING) {
        global_state->llm_status = on_llm_prefilling(key_event, global_state);
        // 外部被动中止
        if (global_state->llm_status == LLM_STOPPED_IN_PREFILLING) {
            ui_llm_session_free(global_state);
            return STATE_LLM_AFTER_INFER;
        }
        else {
            return STATE_LLM_ON_INFER;
        }
    }
    else if (global_state->llm_status == LLM_RUNNING_IN_DECODING) {
        global_state->llm_status = on_llm_decoding(key_event, global_state);
        // 外部被动中止
        if (global_state->llm_status == LLM_STOPPED_IN_DECODING) {
#ifdef TTS_ENABLED
            if (global_state->tts_req_mode > 0) {
                stop_tts();
            }
#endif
            ui_llm_session_free(global_state);
            return STATE_LLM_AFTER_INFER;
        }
        else {
            return STATE_LLM_ON_INFER;
        }
    }
    else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {
        global_state->llm_status = on_llm_finished(key_event, global_state);
        ui_llm_session_free(global_state);
        return STATE_LLM_AFTER_INFER;
    }
    else {
        global_state->llm_status = on_llm_finished(key_event, global_state);
        ui_llm_session_free(global_state);
        return STATE_LLM_AFTER_INFER;
    }
}

// STATE_LLM_AFTER_INFER：推理结束（自然结束或中断），显示推理结果
int32_t ui_llm_after_infer_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 首次获得焦点：初始化
    if (global_state->PREV_STATE != global_state->STATE) {
        // 标题
        ui_draw_header(key_event, global_state, global_state->llm_model_name, 1);

        // 底部
        wchar_t tps_str[50];
        swprintf(tps_str, 50, L"%ls | 已生成%d词元 | %.1f词元/秒", global_state->llm_model_name, global_state->token_num_of_last_session, global_state->llm_max_seq_len, global_state->tps_of_last_session);
        ui_draw_footer(key_event, global_state, tps_str, 1);

        // 计算提示语+生成内容的行数
        wchar_t *prompt_and_output = (wchar_t *)platform_calloc(UI_STR_BUF_MAX_LENGTH * 2, sizeof(wchar_t));
        ui_app_wcscat_bounded(prompt_and_output, L"[#1155ee]Homo:", UI_STR_BUF_MAX_LENGTH * 2);
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            ui_app_wcscat_bounded(prompt_and_output, L"[#000000]\n", UI_STR_BUF_MAX_LENGTH * 2);
        }
        else if (global_state->ui_color_style == UI_COLOR_DARK) {
            ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]\n", UI_STR_BUF_MAX_LENGTH * 2);
        }
        ui_app_wcscat_bounded(prompt_and_output, global_state->w_input_main->textarea.text, UI_STR_BUF_MAX_LENGTH * 2);
        ui_app_wcscat_bounded(prompt_and_output, L"\n--------------------\n[#65bb00]Nano:", UI_STR_BUF_MAX_LENGTH * 2);
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            ui_app_wcscat_bounded(prompt_and_output, L"[#000000]\n", UI_STR_BUF_MAX_LENGTH * 2);
        }
        else if (global_state->ui_color_style == UI_COLOR_DARK) {
            ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]\n", UI_STR_BUF_MAX_LENGTH * 2);
        }
        ui_app_wcscat_bounded(prompt_and_output, global_state->llm_output_of_last_session, UI_STR_BUF_MAX_LENGTH * 2);
        // 推理中止
        if (global_state->llm_status == LLM_STOPPED_IN_PREFILLING || global_state->llm_status == LLM_STOPPED_IN_DECODING) {
            ui_app_wcscat_bounded(prompt_and_output, L"\n\n[#ff0000][Nano:推理中止]", UI_STR_BUF_MAX_LENGTH * 2);
            if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#000000]", UI_STR_BUF_MAX_LENGTH * 2);
            }
            else if (global_state->ui_color_style == UI_COLOR_DARK) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]", UI_STR_BUF_MAX_LENGTH * 2);
            }
        }
        // 推理自然结束
        else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {

        }
        // 推理异常结束
        else {
            ui_app_wcscat_bounded(prompt_and_output, L"\n\n[#ff0000][Nano:推理异常结束]", UI_STR_BUF_MAX_LENGTH * 2);
            if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#000000]", UI_STR_BUF_MAX_LENGTH * 2);
            }
            else if (global_state->ui_color_style == UI_COLOR_DARK) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]", UI_STR_BUF_MAX_LENGTH * 2);
            }
        }
        wchar_t tps_wcstr[50];
        swprintf(tps_wcstr, 50, L"\n\n[#ffc840][%d/%d|%.1fTPS]", global_state->token_num_of_last_session, global_state->llm_max_seq_len, global_state->tps_of_last_session);
        ui_app_wcscat_bounded(prompt_and_output, tps_wcstr, UI_STR_BUF_MAX_LENGTH * 2);
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            ui_app_wcscat_bounded(prompt_and_output, L"[#000000]", UI_STR_BUF_MAX_LENGTH * 2);
        }
        else if (global_state->ui_color_style == UI_COLOR_DARK) {
            ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]", UI_STR_BUF_MAX_LENGTH * 2);
        }

        // 缓冲区与 prompt_and_output 同容量（UI_STR_BUF_MAX_LENGTH * 2），并截断双保险
        wcsncpy(global_state->llm_output_of_last_session, prompt_and_output, UI_STR_BUF_MAX_LENGTH * 2 - 1);
        global_state->llm_output_of_last_session[UI_STR_BUF_MAX_LENGTH * 2 - 1] = L'\0';

        free(prompt_and_output);

        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, global_state->llm_output_of_last_session, -1, 1);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
    }
    global_state->PREV_STATE = global_state->STATE;

    // 短按D键：重新推理。推理完成后，并不清除输入缓冲区，因此再次按D键会重新推理。
    if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
        return STATE_LLM_ON_INFER;
    }
    else {
        // 短按A键：停止TTS
        if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
#ifdef TTS_ENABLED
            if (global_state->tts_req_mode > 0) {
                stop_tts();
            }
#endif
        }
        return ui_widget_textarea_event_handler(key_event, global_state, global_state->w_textarea_main, STATE_LLM_INPUT, STATE_LLM_AFTER_INFER);
    }
}
