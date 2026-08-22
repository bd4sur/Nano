#ifndef __NANO_UI_LLM_H__
#define __NANO_UI_LLM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "infer.h"

// ===============================================================================
// 鹦鹉笼：端侧大语言模型推理及其可视化（自 ui_app.c 提取的独立模块）
//   覆盖状态：STATE_MODEL_MENU（模型选择）/ STATE_LLM_INPUT（输入）/
//             STATE_LLM_ON_INFER（推理进行中）/ STATE_LLM_AFTER_INFER（结果展示）
//   小鹦鹉笼（nano_min 极小内存引擎）已并入本模块的引擎适配层，共用以上全部状态，仅推理引擎不同
//   可视化：llm_observation 观测回调（模型层级图 + top6 词元，由“LLM演示”设置开关；
//           仅 infer.c 引擎支持，小鹦鹉笼无此能力）
// ===============================================================================

// LLM 相关全局字段初始化（由 ui_init 调用；分配 llm_output_of_last_session）
void ui_llm_init_config(Global_State *global_state);

// 卸载当前模型（选中其他模型 / 退出模型菜单时调用；幂等：
// 释放 infer.c 上下文与小鹦鹉笼引擎）
void ui_llm_unload_model(Global_State *global_state);

// LLM 资源释放（由 main_deinit 调用）
void ui_llm_deinit(Global_State *global_state);

// 模型菜单（选择语言模型）
void init_model_menu(Key_Event *key_event, Global_State *global_state);
int32_t model_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);

// 推理过程 UI 回调（prefill/decode 中进行帧刷新与中止检测；finished 收尾统计）
int32_t on_llm_prefilling(Key_Event *key_event, Global_State *global_state);
int32_t on_llm_decoding(Key_Event *key_event, Global_State *global_state);
int32_t on_llm_finished(Key_Event *key_event, Global_State *global_state);

// 推理可视化：观测回调（注入 llm_ctx->observation）与模型结构图绘制
void llm_observation(Nano_Observation obs, void *env);
void ui_app_llm_model_diagram_draw(Key_Event *key_event, Global_State *global_state, int32_t x0, int32_t y0, int32_t total_layers, Nano_Observation obs);

// 状态机处理器（处理 STATE_LLM_INPUT / STATE_LLM_ON_INFER / STATE_LLM_AFTER_INFER，返回新状态）
int32_t ui_llm_input_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_llm_on_infer_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_llm_after_infer_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
