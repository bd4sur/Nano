#ifndef __NANO_UI_APP_H__
#define __NANO_UI_APP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"


// UI应用层全局状态定义
#define STATE_DEFAULT           (-100)
#define STATE_SPLASH_SCREEN     (-1)
#define STATE_MAIN_MENU         (-2)
#define STATE_EBOOK             (-3)
#define STATE_LLM_INPUT         (0)
#define STATE_MODEL_MENU        (4)
#define STATE_SETTING_MENU      (5)
#define STATE_LLM_ON_INFER      (8)
#define STATE_LLM_AFTER_INFER   (10)
#define STATE_ASR_RUNNING       (21)
#define STATE_GENETIC           (23)
#define STATE_GENETIC_TSP       (24)
#define STATE_README            (25)
// 编号 26 已退役：原 C 版 Bad Apple（STATE_BADAPPLE）已移除，Bad Apple 仅保留
// Scheme 实现（app/badapple.scm，经“灵机应用程序”入口启动）。保留编号空缺避免冲突。
// 编号 27 已退役：原 C 版康威生命游戏（STATE_GAMEOFLIFE）已移除，生命游戏仅保留
// Scheme 实现（app/lib/alife_gol.scm，2026-11 起由 app/emergence.scm 的「生命游戏」
// 页承载）。保留编号空缺避免冲突。
#define STATE_LINGLONG          (28)
#define STATE_FLASHMEMO         (29)
#define STATE_TORCH             (30)
// 编号 32 已退役：原 C 版时光集（STATE_ALBUM）已移除，图片浏览仅保留 Scheme 实现
// （app/album.scm，经“灵机应用程序”入口启动）。保留编号空缺避免冲突。
#define STATE_ASR_SETTING       (33)
#define STATE_SETTING_INPUT     (35)
#define STATE_FLIP              (36)
// 编号 37 已退役：原 C 版计步器（STATE_PEDOMETER）已移除，计步器仅保留
// Scheme 实现（app/pedometer.scm，经“灵机应用程序”入口启动）。保留编号空缺避免冲突。
#define STATE_ANIMAC_INIT       (50)
#define STATE_ANIMAC_CONSOLE    (51)
#define STATE_ANIMAC_RUNNING    (52)
// 编号 53 已退役：原 C 版音频频谱仪（STATE_SPECTROGRAM）已移除，声谱图仅保留
// Scheme 实现（app/spectrogram.scm，经“灵机应用程序”入口启动）。保留编号空缺避免冲突。
#define STATE_GAME_MENU         (54)
#define STATE_GOLDMINER         (55)
// 编号 56 已退役：原 C 版俄罗斯方块（STATE_TETRIS）已移除，俄罗斯方块仅保留
// Scheme 实现（app/tetris.scm，经“灵机应用程序”入口启动）。保留编号空缺避免冲突。
#define STATE_EBOOK_READING     (57)
#define STATE_OFDM_MENU         (58)
#define STATE_OFDM_TX           (59)
#define STATE_OFDM_RX           (60)
#define STATE_OFDM_TXING        (61)
#define STATE_OFDM_LOOP         (62)
#define STATE_OFDM_LOOPING      (63)
#define STATE_MUSICBOX_MENU     (64)
#define STATE_MUSICBOX_PLAYING  (65)
// 编号 66 已退役：原 C 版粒子生命（STATE_PARTICLELIFE）已移除，粒子生命仅保留
// Scheme 实现（app/lib/alife_particle.scm，2026-11 起由 app/emergence.scm 的
// 「粒子生命」页承载）。保留编号空缺避免冲突。
#define STATE_RIPPLE            (67)
#define STATE_DICT_QUERY        (68)
#define STATE_DICT_DETAIL       (69)
// 编号 70 已退役：原 C 版日历（STATE_CALENDAR）已移除，日历仅保留 Scheme 实现
// （app/calendar.scm，经“灵机应用程序”入口启动；黄历逻辑/数据 almanac.h 保留备查）。
// 保留编号空缺避免冲突。
#define STATE_WATER             (71)
#define STATE_CLOUD             (72)
// 编号 73~76 已退役：原“小鹦鹉笼”独立的模型菜单/输入/推理/结果状态已并入鹦鹉笼统一状态
// （STATE_MODEL_MENU / STATE_LLM_INPUT / STATE_LLM_ON_INFER / STATE_LLM_AFTER_INFER），
// 仅推理引擎不同（nano_min，见 ui_llm.c 引擎适配层）；保留编号空缺避免冲突
#define STATE_OFDM_MSG          (81)  // 寻呼机：消息收发（对话式记录 + 接收/发射按钮，收发在本页内完成）
#define STATE_OFDM_MSG_INPUT    (82)  // 寻呼机：消息收发的发报文本输入（输入控件宿主）
#define STATE_LOCK_SCREEN       (77)  // 电源键锁屏（灭屏低功耗；处理见 main_event_handler 入口全局拦截）
#define STATE_ANIMAC_EXIT_CONFIRM (78)  // 电子核桃控制台退出确认模态框（页眉“返回”拦截，见 ui_app.c）
#define STATE_SHUTDOWN          (99)


#define PREFILL_LED_ON  system("echo \"1\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define PREFILL_LED_OFF system("echo \"0\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define DECODE_LED_ON   system("echo \"1\" > /sys/devices/platform/leds/leds/blue:status/brightness");
#define DECODE_LED_OFF  system("echo \"0\" > /sys/devices/platform/leds/leds/blue:status/brightness");




// ===============================================================================
// UI框架：获取输入事件（按键 + 触屏）
// ===============================================================================

void get_input_event(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// UI框架：全局GUI+gfx初始化
// ===============================================================================

void ui_init(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// 核心业务：电子鹦鹉（LLM 推理及其可视化）——已提取至独立模块 ui_llm（ui_llm.h/ui_llm.c），
// 使用方（ui_app.c 等）直接包含 ui_llm.h
// ===============================================================================

// ===============================================================================
// 小游戏菜单
// ===============================================================================

void init_game_menu(Key_Event *key_event, Global_State *global_state);
int32_t game_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);

// ===============================================================================
// 主菜单（表驱动应用宫格）：实现已整体剥离至 ui_main_menu.h / ui_main_menu.c
//（entry / draw / event_handler 三个入口；条目表与点击行为也在那里）
// ===============================================================================


// ===============================================================================
// 开机欢迎画面
// ===============================================================================

void ui_app_splash_render_frame(Key_Event *key_event, Global_State *global_state);

// 退出确认模态框（“是否退出？” 确认/留下；控制台与灵机 App 等场景共用）
//   基础版：固定标题（同历史行为，兼容既有调用点）
//   _msg 版：自定义标题 + 可选正文（正文非空时对话框自动加高一行）——
//   供 (native Ui) 的 (Ui.confirm 标题 正文) 与 (native App) 的退出确认共用，
//   保证“全局确认框”与 App 自弹确认框视觉/命中完全一致。
void    ui_exit_confirm_layout(Global_State *global_state, int32_t *out_dx, int32_t *out_dy,
                               int32_t *out_btn_y, int32_t *out_confirm_x, int32_t *out_stay_x);
void    ui_exit_confirm_draw(Key_Event *key_event, Global_State *global_state);
void    ui_exit_confirm_draw_msg(Key_Event *key_event, Global_State *global_state,
                                 const wchar_t *title, const wchar_t *body);
int32_t ui_exit_confirm_hit(Global_State *global_state, int32_t x, int32_t y); // 1=确认 2=留下 0=未命中
// 带正文版命中判定（with_body 必须与绘制时一致，否则按钮纵向位置不符）
int32_t ui_exit_confirm_hit_msg(Global_State *global_state, int32_t with_body, int32_t x, int32_t y);

// 三按钮变体（灵机 App 的退出确认框）：确认 / 后台 / 留下。
//   with_bg=1 时按钮行变成三枚窄按钮（54px）：「确认」「后台」「留下」
//   ——「后台」= 保留实例、冻结、回应用列表（见 am_native_App.h 的 (App.hide)）。
void    ui_exit_confirm_draw_msg_ex(Key_Event *key_event, Global_State *global_state,
                                    const wchar_t *title, const wchar_t *body, int32_t with_bg);
int32_t ui_exit_confirm_hit_msg_ex(Global_State *global_state, int32_t with_body, int32_t with_bg,
                                   int32_t x, int32_t y);   // 1=确认 2=留下 3=后台 0=未命中

// ---- 模态框按钮的「按下」反馈（2026-11）-------------------------------------
// 模态框是**叠加式**的：只在弹出时画一次，触屏结算发生在**松手沿**，所以过去
// 按下去完全没反馈——手指按住的那一刻看不出“到底选中了哪一枚”。
// 现在由调用方在事件循环里每帧报一次当前触点，绘制端据此把对应按钮画成「按下色」
// （底色调亮一档 + 标签下沉 1px）。
//
// **手势语义与宿主其它按钮一致（app/lib/button.scm 的 (BTN.track) 同款）**：
//   · 按下 → 按住 → 松开，**三下都在同一枚按钮内**才算一次触发；
//   · 按住后把手指拖出按钮 → 立即恢复常态、不再算按下；拖回来又亮；
//   · 按下点不在按钮内（从外面拖进来）→ 不亮，也不会触发。
// 所以高亮的判据就是“现在松手会不会触发”，两者严格同一口径：
//   按下点命中 X 且**当前点**也命中 X。
//
//   touching = 当前是否按着；down_* = 本次手势按下点；cur_* = 当前触点
//   返回 1 = 按下目标变了（调用方应重绘一次模态框；叠加式重绘是幂等的）
int32_t ui_exit_confirm_arm(Global_State *global_state, int32_t with_body, int32_t with_bg,
                            int32_t touching,
                            int32_t down_x, int32_t down_y, int32_t cur_x, int32_t cur_y);
// 清掉按下态（新弹一个模态框时调；否则上一个框的按下色会“带”到新框上）
void    ui_exit_confirm_arm_clear(void);


// ===============================================================================
// 玲珑天象仪：实现已整体剥离至 ui_linglong.h / ui_linglong.c
// （全局资源初始化 ui_linglong_global_init + 状态页 init/render/event 三个入口）
// ===============================================================================


// ===============================================================================
// 设置菜单
// ===============================================================================

// 设置界面已重构为独立模块 ui_settings.c / ui_settings.h（菜单 + 日期/时间/经纬度编辑器）。




// ===============================================================================
// UI主体框架
// ===============================================================================

int32_t main_init(Key_Event *key_event, Global_State *global_state);
int32_t main_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t main_periodic_task(Key_Event *key_event, Global_State *global_state);
int32_t main_deinit(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
