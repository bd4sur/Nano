//
// ui_nanochat.c - 小鹦鹉笼：基于 nano_min 极小内存推理引擎的 LLM 对话功能
//
//   状态流转（对照“鹦鹉笼”的 STATE_MODEL_MENU -> STATE_LLM_INPUT -> STATE_LLM_ON_INFER
//   -> STATE_LLM_AFTER_INFER 流程）：
//     小游戏菜单 -> STATE_NMCHAT_MODEL_MENU（选择模型，选中即加载）
//                -> STATE_NMCHAT_INPUT（文字编辑，D键提交 / A键返回模型菜单）
//                -> STATE_NMCHAT_ON_INFER（协作式单步推理，A键中止）
//                -> STATE_NMCHAT_AFTER_INFER（结果呈现，D键重新推理 / A键返回编辑）
//   模型菜单按A键返回小游戏菜单，并释放引擎与工作文件。
//

#include "ui_nanochat.h"

#include "platform.h"      // PLATFORM_ROOT_DIR / UI_STR_BUF_MAX_LENGTH / fs 抽象
#include "input_device.h"  // NANO_KEY_esc / NANO_KEY_enter
#include "infer.h"         // LLM_RUNNING_IN_* / LLM_STOPPED_* 状态码
#include "ui_color.h"      // UI_COLOR_DARK

#include "nano_min.h"

// ===============================================================================
// 模型预设（模型与工作文件均位于 PLATFORM_ROOT_DIR/llm）
// ===============================================================================

typedef struct {
    const wchar_t *name;        // 菜单显示名（兼作界面标题）
    const char    *model_file;  // 模型文件路径
    float rep_penalty, temperature, top_p; // 采样参数（对齐 ui_app.c preset_model_configs）
    uint32_t max_seq_len;
} NMChat_Model_Preset;

static const NMChat_Model_Preset s_nmchat_models[] = {
    { L"Nano-168M-Q80",  PLATFORM_ROOT_DIR "/llm/nano-168m-q80.bin", 1.05f, 1.0f, 0.5f,  512 },
    { L"Qwen3-0.6B-Q80", PLATFORM_ROOT_DIR "/llm/qwen3-0b6-q80.bin", 1.0f,  0.6f, 0.95f, 512 },
};
#define NMCHAT_MODEL_NUM ((int32_t)(sizeof(s_nmchat_models) / sizeof(s_nmchat_models[0])))

#define NMCHAT_WORK_PATH PLATFORM_ROOT_DIR "/llm/nano_min_work.tmp"

// ===============================================================================
// 模块内部状态（静态，避免侵入 Global_State）
// ===============================================================================

static NM_Engine *s_engine = NULL;
static uint32_t   s_arch = NM_ARCH_NANO;
static const wchar_t *s_model_name = L"";
static uint32_t   s_max_seq_len = 512;

// 会话缓冲（首次加载模型时分配，退出功能时释放）
static uint32_t *s_ids = NULL;        // (max_seq_len+1,) token 序列
static wchar_t  *s_out_text = NULL;   // (UI_STR_BUF_MAX_LENGTH,) 当前输出文本
static char     *s_out_bytes = NULL;  // (UI_STR_BUF_MAX_LENGTH*4,) QWEN 输出 UTF-8 累积
static wchar_t  *s_result = NULL;     // (UI_STR_BUF_MAX_LENGTH*2,) 结果页文本

// 会话状态
static uint32_t s_n_prompt = 0;
static uint32_t s_pos = 0;
static uint32_t s_token = 0;
static uint32_t s_n_gen = 0;
static uint32_t s_out_len = 0;
static uint32_t s_out_bytes_len = 0;
static int32_t  s_status = 0;   // LLM_RUNNING_IN_PREFILLING / LLM_RUNNING_IN_DECODING / LLM_STOPPED_*
static uint64_t s_t0 = 0;
static float    s_tps = 0.0f;

// ===============================================================================
// 内部工具
// ===============================================================================

// 有界宽字符串追加（与 ui_app.c 的 ui_app_wcscat_bounded 语义一致；其为 static 无法复用）
static void nmchat_wcscat(wchar_t *dst, const wchar_t *src, uint32_t cap) {
    size_t dl = wcslen(dst);
    size_t sl = wcslen(src);
    if (dl + sl >= cap) sl = (cap > dl + 1) ? (cap - dl - 1) : 0;
    wcsncpy(dst + dl, src, sl);
    dst[dl + sl] = L'\0';
}

static void nmchat_release(void) {
    if (s_engine) { nm_close(s_engine); s_engine = NULL; }
    if (s_ids)       { free(s_ids);       s_ids = NULL; }
    if (s_out_text)  { free(s_out_text);  s_out_text = NULL; }
    if (s_out_bytes) { free(s_out_bytes); s_out_bytes = NULL; }
    if (s_result)    { free(s_result);    s_result = NULL; }
}

// 生成一个 token 后，将其文本追加到输出缓冲（QWEN 先累积 UTF-8 字节再整体转换，
// 容忍跨 token 的不完整 UTF-8 序列——与 main_cli.c 的说明一致）
static void nmchat_append_output(uint32_t tok) {
    if (s_arch == NM_ARCH_NANO) {
        const wchar_t *s = nm_token_str(s_engine, tok);
        size_t l = wcslen(s);
        if (s_out_len + l < (uint32_t)UI_STR_BUF_MAX_LENGTH - 1) {
            wcscpy(s_out_text + s_out_len, s);
            s_out_len += l;
        }
    }
    else {
        char buf[300];
        nm_bpe_token_str(s_engine, tok, buf, sizeof(buf));
        size_t l = strlen(buf);
        if (s_out_bytes_len + l < (uint32_t)UI_STR_BUF_MAX_LENGTH * 4 - 1) {
            memcpy(s_out_bytes + s_out_bytes_len, buf, l);
            s_out_bytes_len += l;
            s_out_bytes[s_out_bytes_len] = 0;
            _mbstowcs(s_out_text, s_out_bytes, UI_STR_BUF_MAX_LENGTH - 1);
            s_out_len = wcslen(s_out_text);
        }
    }
}

// ===============================================================================
// 模型选择菜单
// ===============================================================================

void ui_nanochat_model_menu_init(Key_Event *key_event, Global_State *global_state) {
    // 条目字符串借用预设表的静态存储，控件不复制
    static const wchar_t *items[NMCHAT_MODEL_NUM];
    for (int32_t i = 0; i < NMCHAT_MODEL_NUM; i++) {
        items[i] = s_nmchat_models[i].name;
    }
    global_state->w_menu_main->title = L"小鹦鹉笼·选择模型";
    global_state->w_menu_main->items = items;
    global_state->w_menu_main->item_num = NMCHAT_MODEL_NUM;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
}

int32_t ui_nanochat_model_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    int32_t idx = ms->current_item_index;
    if (idx < 0 || idx >= NMCHAT_MODEL_NUM) return STATE_NMCHAT_MODEL_MENU;
    const NMChat_Model_Preset *m = &s_nmchat_models[idx];

    // 加载提示（nm_open 为阻塞式，模型在 SD 卡上时可能耗时较长）
    ui_draw_header(ke, gs, L"模型加载中...", 1);
    gfx_refresh(gs->gfx);

    // 预检模型文件是否存在（nano_min 打开失败会直接退出进程，此处先拦截）
    if (platform_file_open(m->model_file) != 0) {
        platform_file_close();
        ui_draw_header(ke, gs, L"模型文件缺失", 1);
        gfx_refresh(gs->gfx);
        sleep_in_ms(1500);
        // 恢复菜单界面
        ui_draw_header(ke, gs, (wchar_t *)gs->w_menu_main->title, 1);
        ui_widget_menu_refresh(ke, gs, gs->w_menu_main);
        return STATE_NMCHAT_MODEL_MENU;
    }
    platform_file_close();

    // 释放上一个引擎（若存在），再加载新模型
    if (s_engine) { nm_close(s_engine); s_engine = NULL; }
    s_engine = nm_open(m->model_file, NMCHAT_WORK_PATH, m->max_seq_len);
    nm_set_sampler(s_engine, m->rep_penalty, m->temperature, m->top_p, gs->timestamp);
    s_arch = nm_get_arch(s_engine);
    s_max_seq_len = m->max_seq_len;
    s_model_name = m->name;

    // 会话缓冲（首次分配，之后复用）
    if (!s_ids) {
        s_ids       = (uint32_t *)platform_calloc(s_max_seq_len + 1, sizeof(uint32_t));
        s_out_text  = (wchar_t  *)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
        s_out_bytes = (char     *)platform_calloc(UI_STR_BUF_MAX_LENGTH * 4, sizeof(char));
        s_result    = (wchar_t  *)platform_calloc(UI_STR_BUF_MAX_LENGTH * 2, sizeof(wchar_t));
        if (!s_ids || !s_out_text || !s_out_bytes || !s_result) {
            nmchat_release();
            return STATE_NMCHAT_MODEL_MENU;
        }
    }

    // 初始化输入控件（复用全局输入控件实例，标题为模型名）
    ui_widget_input_init(ke, gs, gs->w_input_main, (wchar_t *)s_model_name);
    return STATE_NMCHAT_INPUT;
}

// ===============================================================================
// 推理会话
// ===============================================================================

// 组装 prompt 并编码为 token 序列（进入 STATE_NMCHAT_ON_INFER 时调用一次）
static void nmchat_session_begin(Key_Event *ke, Global_State *gs) {
    (void)ke;
    wchar_t *text = gs->w_input_main->textarea.text;

    // 输入为空时随机选用一个预置 prompt（与鹦鹉笼一致）
    if (wcslen(text) == 0) {
        set_random_prompt(text, gs->timestamp);
        gs->w_input_main->textarea.length = wcslen(text);
    }

    if (s_arch == NM_ARCH_NANO) {
        // Nano 提示词模板（同鹦鹉笼）
        wchar_t *prompt = (wchar_t *)platform_calloc(wcslen(text) + 64, sizeof(wchar_t));
        wcscpy(prompt, L"<|instruct_mark|>");
        wcscat(prompt, text);
        wcscat(prompt, L"<|response_mark|>");
        s_n_prompt = nm_encode(s_engine, prompt, s_ids, s_max_seq_len);
        free(prompt);
    }
    else {
        // ChatML 模板（同 tokenizer.c apply_qwen_chat_template，enable_thinking=1）
        uint32_t n = 0;
        s_ids[n++] = 151644; // <|im_start|>
        s_ids[n++] = 872;    // user
        s_ids[n++] = 198;    // \n
        {
            size_t blen = wcslen(text) * 4 + 1;
            char *bytes = (char *)platform_calloc(blen, sizeof(char));
            _wcstombs(bytes, text, blen);
            n += nm_encode_bpe(s_engine, bytes, s_ids + n, s_max_seq_len - n - 5);
            free(bytes);
        }
        s_ids[n++] = 151645; // <|im_end|>
        s_ids[n++] = 198;    // \n
        s_ids[n++] = 151644; // <|im_start|>
        s_ids[n++] = 77091;  // assistant
        s_ids[n++] = 198;    // \n
        s_n_prompt = n;
    }

    s_pos = 0;
    s_token = s_ids[0];
    s_n_gen = 0;
    s_out_len = 0;       s_out_text[0] = 0;
    s_out_bytes_len = 0; s_out_bytes[0] = 0;
    s_t0 = 0;
    s_tps = 0.0f;
    s_status = LLM_RUNNING_IN_PREFILLING;
}

// 预填充中的一步渲染（含 A 键中止）；返回 LLM 状态码（语义同 infer.h）
static int32_t nmchat_on_prefilling(Key_Event *ke, Global_State *gs) {
    if (s_t0 == 0) s_t0 = gs->timestamp;
    else s_tps = (float)(s_pos) / (float)(gs->timestamp - s_t0 + 1) * 1000.0f;

    // 长/短按A键中止推理
    if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == NANO_KEY_esc) {
        s_out_text[0] = 0; s_out_len = 0;
        return LLM_STOPPED_IN_PREFILLING;
    }

    // 屏幕刷新节流
    if (gs->timestamp - gs->llm_refresh_timestamp > (1000 / gs->llm_refresh_max_fps)) {
        wchar_t title_str[50];
        swprintf(title_str, 50, L"%ls Reading...", s_model_name);
        ui_draw_header(ke, gs, title_str, 1);

        wchar_t progress_str[30];
        swprintf(progress_str, 30, L"%d/%d", s_pos, s_n_prompt);
        ui_draw_footer(ke, gs, progress_str, 1);

        ui_widget_textarea_set(ke, gs, gs->w_textarea_main, gs->w_input_main->textarea.text, -1, 1);
        ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);

        gs->llm_refresh_timestamp = gs->timestamp;
    }
    return LLM_RUNNING_IN_PREFILLING;
}

// 解码中的一步渲染（含 A 键中止）；返回 LLM 状态码
static int32_t nmchat_on_decoding(Key_Event *ke, Global_State *gs) {
    if (s_t0 == 0) s_t0 = gs->timestamp;
    else s_tps = (float)(s_pos) / (float)(gs->timestamp - s_t0 + 1) * 1000.0f;

    // 长/短按A键中止推理
    if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == NANO_KEY_esc) {
        return LLM_STOPPED_IN_DECODING;
    }

    // 屏幕刷新节流
    if (gs->timestamp - gs->llm_refresh_timestamp > (1000 / gs->llm_refresh_max_fps)) {
        wchar_t title_str[50];
        swprintf(title_str, 50, L"%ls Decoding...", s_model_name);
        ui_draw_header(ke, gs, title_str, 1);

        wchar_t tps_str[50];
        swprintf(tps_str, 50, L"%ls | %d/%d | %.1f词元/秒", s_model_name, s_pos, s_max_seq_len, s_tps);
        ui_draw_footer(ke, gs, tps_str, 1);

        ui_widget_textarea_set(ke, gs, gs->w_textarea_main, s_out_text, -1, 1);
        ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);

        gs->llm_refresh_timestamp = gs->timestamp;
    }
    return LLM_RUNNING_IN_DECODING;
}

// 结果页文本组装（进入 STATE_NMCHAT_AFTER_INFER 时调用一次）
static void nmchat_result_render(Key_Event *ke, Global_State *gs) {
    // 标题
    ui_draw_header(ke, gs, (wchar_t *)s_model_name, 1);

    // 底部
    wchar_t footer_str[50];
    swprintf(footer_str, 50, L"%ls | 已生成%d词元 | %.1f词元/秒", s_model_name, s_n_gen, s_tps);
    ui_draw_footer(ke, gs, footer_str, 1);

    // 提示语 + 生成内容（带颜色标签，同鹦鹉笼）
    const wchar_t *fg_tag = (gs->ui_color_style == UI_COLOR_DARK) ? L"[#ffffff]" : L"[#000000]";
    s_result[0] = 0;
    nmchat_wcscat(s_result, L"[#1155ee]Homo:", UI_STR_BUF_MAX_LENGTH * 2);
    nmchat_wcscat(s_result, fg_tag, UI_STR_BUF_MAX_LENGTH * 2);
    nmchat_wcscat(s_result, L"\n", UI_STR_BUF_MAX_LENGTH * 2);
    nmchat_wcscat(s_result, gs->w_input_main->textarea.text, UI_STR_BUF_MAX_LENGTH * 2);
    nmchat_wcscat(s_result, L"\n--------------------\n[#65bb00]Nano:", UI_STR_BUF_MAX_LENGTH * 2);
    nmchat_wcscat(s_result, fg_tag, UI_STR_BUF_MAX_LENGTH * 2);
    nmchat_wcscat(s_result, L"\n", UI_STR_BUF_MAX_LENGTH * 2);
    nmchat_wcscat(s_result, s_out_text, UI_STR_BUF_MAX_LENGTH * 2);

    if (s_status == LLM_STOPPED_IN_PREFILLING || s_status == LLM_STOPPED_IN_DECODING) {
        nmchat_wcscat(s_result, L"\n\n[#ff0000][推理中止]", UI_STR_BUF_MAX_LENGTH * 2);
        nmchat_wcscat(s_result, fg_tag, UI_STR_BUF_MAX_LENGTH * 2);
    }

    wchar_t tps_wcstr[50];
    swprintf(tps_wcstr, 50, L"\n\n[#ffc840][%d/%d|%.1fTPS]", s_n_gen, s_max_seq_len, s_tps);
    nmchat_wcscat(s_result, tps_wcstr, UI_STR_BUF_MAX_LENGTH * 2);
    nmchat_wcscat(s_result, fg_tag, UI_STR_BUF_MAX_LENGTH * 2);

    ui_widget_textarea_set(ke, gs, gs->w_textarea_main, s_result, -1, 1);
    ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);
}

// ===============================================================================
// 状态总分发
// ===============================================================================

int32_t ui_nanochat_event_handler(Key_Event *key_event, Global_State *global_state) {
    Key_Event *ke = key_event;
    Global_State *gs = global_state;
    int32_t st = gs->STATE;
    int32_t first = (gs->PREV_STATE != st) ? 1 : 0;
    gs->PREV_STATE = st;

    switch (st) {

    /////////////////////////////////////////////
    // 选择模型
    /////////////////////////////////////////////
    case STATE_NMCHAT_MODEL_MENU: {
        if (first) {
            ui_nanochat_model_menu_init(ke, gs);
            ui_draw_header(ke, gs, (wchar_t *)gs->w_menu_main->title, 1);
            ui_draw_footer_softkeys(ke, gs, L"↑", L"", L"↓", L"选择");
            ui_widget_menu_refresh(ke, gs, gs->w_menu_main);
        }
        int32_t next = ui_widget_menu_event_handler(ke, gs, gs->w_menu_main,
                        ui_nanochat_model_menu_item_action, STATE_GAME_MENU, STATE_NMCHAT_MODEL_MENU);
        // 退出模型菜单回到小游戏菜单时，释放引擎与工作文件
        if (next == STATE_GAME_MENU && s_engine != NULL) {
            nmchat_release();
        }
        return next;
    }

    /////////////////////////////////////////////
    // 文字编辑
    /////////////////////////////////////////////
    case STATE_NMCHAT_INPUT: {
        if (first) {
            ui_widget_input_refresh(ke, gs, gs->w_input_main);
        }
        return ui_widget_input_event_handler(ke, gs, gs->w_input_main,
                    STATE_NMCHAT_MODEL_MENU, STATE_NMCHAT_INPUT, STATE_NMCHAT_ON_INFER);
    }

    /////////////////////////////////////////////
    // 推理进行中（协作式单步：每轮主循环推进一步，同鹦鹉笼）
    /////////////////////////////////////////////
    case STATE_NMCHAT_ON_INFER: {
        if (first) {
            nmchat_session_begin(ke, gs);
        }

        if (s_pos + 1 < s_n_prompt) {
            // Pre-filling
            nm_forward(s_engine, s_ids[s_pos], s_pos);
            s_pos++;
            s_status = nmchat_on_prefilling(ke, gs);
            if (s_status == LLM_STOPPED_IN_PREFILLING) {
                return STATE_NMCHAT_AFTER_INFER;
            }
        }
        else {
            // Decoding
            nm_forward(s_engine, s_token, s_pos);
            uint32_t next_tok = nm_sample(s_engine, s_ids, s_pos + 1);
            s_pos++;
            s_n_gen++;
            if (nm_is_eos(s_engine, next_tok) || s_pos >= s_max_seq_len) {
                s_status = LLM_STOPPED_NORMALLY;
                s_tps = (s_t0 > 0) ? (float)(s_pos) / (float)(gs->timestamp - s_t0 + 1) * 1000.0f : 0.0f;
                return STATE_NMCHAT_AFTER_INFER;
            }
            s_ids[s_pos] = next_tok;
            s_token = next_tok;
            nmchat_append_output(next_tok);
            s_status = nmchat_on_decoding(ke, gs);
            if (s_status == LLM_STOPPED_IN_DECODING) {
                return STATE_NMCHAT_AFTER_INFER;
            }
        }
        return STATE_NMCHAT_ON_INFER;
    }

    /////////////////////////////////////////////
    // 结果呈现
    /////////////////////////////////////////////
    case STATE_NMCHAT_AFTER_INFER: {
        if (first) {
            nmchat_result_render(ke, gs);
        }
        // 短按D键：重新推理（输入缓冲区不清除，同鹦鹉笼）
        if (ke->key_edge == -1 && ke->key_code == NANO_KEY_enter) {
            return STATE_NMCHAT_ON_INFER;
        }
        // A键返回编辑 / 翻页（文本框控件处理）
        return ui_widget_textarea_event_handler(ke, gs, gs->w_textarea_main,
                    STATE_NMCHAT_INPUT, STATE_NMCHAT_AFTER_INFER);
    }

    default:
        return STATE_MAIN_MENU;
    }
}
