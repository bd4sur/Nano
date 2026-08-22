// ui_nanochat.h - 小鹦鹉笼：基于 nano_min 极小内存推理引擎的 LLM 对话功能
//
//   已并入“鹦鹉笼”（ui_llm）：模型选择由鹦鹉笼的统一模型菜单承担（条目带 [轻] 前缀），
//   本模块只保留引擎封装与 输入 -> 推理中 -> 结果页 三状态（STATE_NMCHAT_INPUT/
//   ON_INFER/AFTER_INFER），交互按键与鹦鹉笼一致（D 提交/重新推理、A 中止/返回模型菜单）。
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

// 静态条目表容量上限（供 ui_llm 合并模型菜单的定长数组使用）
#define UI_NANOCHAT_MODEL_NUM_MAX (8)

// 模型数量与显示名（供 ui_llm 合并模型菜单使用；显示名带 [轻] 前缀）
int32_t ui_nanochat_model_num(void);
const wchar_t *ui_nanochat_model_name(int32_t idx);

// 加载所选模型并初始化输入控件（由 ui_llm 的模型菜单动作调用）；
// 返回 0 成功（调用方应转 STATE_NMCHAT_INPUT），-1 失败（模型文件缺失/内存不足，已显示错误提示）
int32_t ui_nanochat_model_enter(Key_Event *ke, Global_State *gs, int32_t idx);

// 释放引擎与会话缓冲（选中其他模型 / 退出鹦鹉笼模型菜单时调用；幂等）
void ui_nanochat_release(void);

// 小鹦鹉笼状态（STATE_NMCHAT_INPUT/ON_INFER/AFTER_INFER）的总分发：每轮主循环调用一次，返回下一个状态
int32_t ui_nanochat_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
