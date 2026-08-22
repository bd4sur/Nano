#ifndef __NANO_UI_PINYIN_IME_H__
#define __NANO_UI_PINYIN_IME_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"
#include "graphics.h"
#include "hal_key.h"
#include "ui.h"

// ===============================================================================
// 全键盘拼音输入法（硬件无关），专用于触屏软键盘激活时的中文输入。
//
// 移植自 main.cpp-ref 的 PinyinIME 实现，交互方式与其保持一致：
//   - 汉字模式下，字母键输入拼音（大写同样按拼音处理），拼音串与候选字显示在底栏；
//   - 数字键 1~N 选字，左右方向键翻页，退格键删除拼音字母；
//   - Esc 取消拼音/符号输入；Enter 将已输入的拼音字母原样上屏（再次 Enter 才提交）；
//   - Ctrl+BS 进入/退出符号选择状态（对应 main.cpp-ref 的 Fn+Bksp）；
//   - 汉字模式下部分半角标点自动转全角（同 main.cpp-ref 的 half_to_full）。
// 查表实现为根目录 pinyin_ime.c（自动生成的拼音-汉字表，内容不可手改，候选按字频预排序）。
// ===============================================================================

#define UI_IME_STATE_IDLE      (0) // 空闲（未在输入拼音）
#define UI_IME_STATE_SELECTING (1) // 拼音输入/选字中
#define UI_IME_STATE_SYMBOL    (2) // 符号选择中

// 重置输入法状态（拼音串、候选字、符号态）
void ui_pinyin_ime_reset();

// 是否正在组词（SELECTING 或 SYMBOL）：是则底栏显示拼音/候选字，否则显示默认页脚
uint8_t ui_pinyin_ime_is_composing();

// 处理触屏软键盘的键事件（仅在汉字输入模式、软键盘激活时被调用）。
// 返回 1 表示该键已被输入法接管，0 表示未接管（调用方走默认处理，如直接插入、提交等）。
uint8_t ui_pinyin_ime_handle_key(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);

// 把拼音串与候选字/符号栏绘制到页脚区域（只写帧缓冲，不刷新屏幕，由调用方统一 gfx_refresh）
void ui_pinyin_ime_draw_bar(Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
