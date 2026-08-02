#ifndef __NANO_UI_OFDM_H__
#define __NANO_UI_OFDM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui_app.h"

// ===============================================================================
// 寻呼机（OFDM 声波数传）UI 模块
// 入口：主菜单 [9] 寻呼机 → STATE_OFDM_MENU（发射/接收/软件环路自测，同一时刻仅一种模式）
// 信号处理核见 ofdm_modem.c（硬件无关；码表在进入本模块时初始化到 PSRAM、退出时释放）；
// 音频收发见 mic.h / audio_out.h。
// ===============================================================================

// 模式菜单（发射/接收/软件环路自测）
void ui_ofdm_menu_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_ofdm_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);
// 退出寻呼机模式菜单（回主菜单）时调用：释放 modem 码表（PSRAM）
void ui_ofdm_menu_on_exit(void);

// STATE_OFDM_TX（文本输入）：首次获焦刷新输入控件；事件委托输入控件（D提交→TXING，A→菜单）
void ui_ofdm_tx_on_enter(Key_Event *key_event, Global_State *global_state);
int32_t ui_ofdm_tx_event(Key_Event *key_event, Global_State *global_state);

// STATE_OFDM_TXING（发射中）：首次获焦调制并启动循环播放；每轮喂扬声器通道；D/A停止
void ui_ofdm_txing_on_enter(Key_Event *key_event, Global_State *global_state);
int32_t ui_ofdm_txing_event(Key_Event *key_event, Global_State *global_state);

// STATE_OFDM_RX（接收中）：首次获焦接管麦克风并创建接收机；每轮采集/解调/限频刷新；A退出
void ui_ofdm_rx_on_enter(Key_Event *key_event, Global_State *global_state);
int32_t ui_ofdm_rx_event(Key_Event *key_event, Global_State *global_state);

// STATE_OFDM_LOOP / STATE_OFDM_LOOPING（软件环路自测：自发自收，不出声）
// 发射机逐帧渲染 → 直接喂给本机接收机 → 显示解调文本与环回校验（测试解调可行性）
void ui_ofdm_loop_on_enter(Key_Event *key_event, Global_State *global_state);
int32_t ui_ofdm_loop_event(Key_Event *key_event, Global_State *global_state);
void ui_ofdm_looping_on_enter(Key_Event *key_event, Global_State *global_state);
int32_t ui_ofdm_looping_event(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
