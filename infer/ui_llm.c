// ===============================================================================
// 鹦鹉笼：端侧大语言模型推理及其可视化（自 ui_app.c 提取的独立模块，逻辑原样保留）
//   小鹦鹉笼（nano_min 极小内存引擎）已并入本模块（见下方“引擎适配层”隔离区块）：
//   二者共用同一套 UI 状态（STATE_MODEL_MENU / STATE_LLM_INPUT / STATE_LLM_ON_INFER /
//   STATE_LLM_AFTER_INFER），仅在 STATE_LLM_ON_INFER 内按引擎分发 会话建立/单步/销毁 三个钩子。
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


void init_model_menu(Key_Event *key_event, Global_State *global_state) {
    size_t model_count = sizeof(preset_model_configs) / sizeof(preset_model_configs[0]);
    int32_t nmchat_count = llm_nm_model_num();
    if (nmchat_count > LLM_NM_MODEL_NUM_MAX) nmchat_count = LLM_NM_MODEL_NUM_MAX;
    // 条目字符串借用 preset_model_configs 与小鹦鹉笼预设表的静态存储，控件不复制；
    // 合并菜单 = 大模型（infer.c 引擎） + [轻]小鹦鹉笼模型（nano_min 极小内存引擎）
    static const wchar_t *model_menu_items[sizeof(preset_model_configs) / sizeof(preset_model_configs[0]) + LLM_NM_MODEL_NUM_MAX];
    for (size_t i = 0; i < model_count; i++) {
        model_menu_items[i] = preset_model_configs[i].model_name;
    }
    for (int32_t i = 0; i < nmchat_count; i++) {
        model_menu_items[model_count + i] = llm_nm_model_name(i);
    }
    global_state->w_menu_main->title = L"选择语言模型";
    global_state->w_menu_main->items = model_menu_items;
    global_state->w_menu_main->item_num = (int32_t)model_count + nmchat_count;
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

int32_t model_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    int32_t item_index = ms->current_item_index;

    // 选中任一模型前，卸载旧模型（释放 infer.c 上下文 / nano_min 引擎），避免并存
    ui_llm_unload_model(gs);

    int32_t model_count = (int32_t)(sizeof(preset_model_configs) / sizeof(preset_model_configs[0]));

    if (item_index >= model_count) {
        // [轻] 小鹦鹉笼模型（nano_min 极小内存引擎）：由引擎适配层加载，
        // 成功后与标准引擎路径共用输入状态（STATE_LLM_INPUT）
        if (llm_nm_model_enter(ke, gs, item_index - model_count) == 0) {
            return STATE_LLM_INPUT;
        }
        // 加载失败（错误提示已由其显示）：重绘模型菜单并停留（本状态未离开焦点，需手动重绘）
        ui_draw_header(ke, gs, (wchar_t *)gs->w_menu_main->title, 1);
        ui_draw_footer_softkeys(ke, gs, L"↑", L"", L"↓", L"选择");
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

// 按当前引擎销毁对话 session（infer.c 会话 / nano_min 会话视图）
static void ui_llm_session_free(Global_State *global_state) {
    if (llm_nm_is_active()) {
        llm_nm_session_free(global_state);
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

    // 首次获得焦点：初始化对话 session
    if (global_state->PREV_STATE != global_state->STATE) {
        if (use_nm_engine) {
            // 小鹦鹉笼：prompt 模板组装与编码由引擎适配层完成，会话视图挂到 llm_session
            if (llm_nm_session_begin(key_event, global_state) != 0) {
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
