//
// ui_nanochat.h - 小鹦鹉笼：基于 nano_min 极小内存推理引擎的 LLM 对话功能
//
//   仿照“鹦鹉笼”（ui_app.c 中的 STATE_LLM_* 流程），但使用 nano_min 引擎：
//   模型权重按需从文件系统读取，KV-Cache 与 logits 驻留工作文件，RAM 占用极小。
//   功能涵盖：模型选择（菜单）-> 文字编辑（输入控件，九键拼音/软键盘）-> 推理结果呈现。
//   不含推理观测可视化功能。
//
//   模型与工作文件均位于 PLATFORM_ROOT_DIR "/llm" 目录：
//     - nano-168m-q80.bin / qwen3-0b6-q80.bin
//     - qwen3-0b6-q80.bin.bpeidx（BPE 索引，缺失时自动生成；建议在 PC 上生成后拷贝到 SD 卡）
//     - nano_min_work.tmp（KV-Cache + logits 工作文件，退出时删除）
//

#ifndef __NANO_UI_NANOCHAT_H__
#define __NANO_UI_NANOCHAT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui_app.h"

// 模型选择菜单初始化（填充全局共享菜单控件 w_menu_main）
void ui_nanochat_model_menu_init(Key_Event *key_event, Global_State *global_state);
// 模型选择菜单动作：加载所选模型并进入输入界面（作为菜单控件的回调）
int32_t ui_nanochat_model_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);
// 小鹦鹉笼全部状态（STATE_NMCHAT_*）的总分发：每轮主循环调用一次，返回下一个状态
int32_t ui_nanochat_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
