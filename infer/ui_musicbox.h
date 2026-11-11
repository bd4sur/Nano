#ifndef __NANO_UI_MUSICBOX_H__
#define __NANO_UI_MUSICBOX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui_app.h"

// ===============================================================================
// 音乐盒 UI 模块
// 入口：主菜单 [5] 音乐盒 → STATE_MUSICBOX_MENU（SD 卡根目录 music/ 下 WAV/MP3 列表）
//   → 选中曲目 → STATE_MUSICBOX_PLAYING（显式触屏控件播放页）
// 交互：菜单页由通用菜单控件直接消费触屏流；播放页由本模块自绘控件直接消费触屏流——
//   退出 / 上一首 / 播放·暂停 / 下一首 / 音量− / 音量+ / 可拖动进度条，
//   并显示文件名与格式参数（采样率/位深/码率/声道）。
//   STATE_MUSICBOX_MENU 与 STATE_MUSICBOX_PLAYING 均列入 ui_app_state_is_menu 抑制表，
//   十六宫格触屏软按键在音乐盒内整体退场；实体键盘（A/D/←/→/4/6）仅作遗留兜底。
// 解码：WAV（RIFF PCM 8/16bit，自解析）与 MP3（vendor/minimp3.h，实现单元 ui_musicbox_mp3.c）
// 输出：统一解码为 int16 单声道 PCM，经 hal_audio_out.h 流式 HAL 播放（乒乓双缓冲 + 背压喂入，
//   范式同 ui_ofdm.c TX 路径）。进入播放态申请、退出时释放全部内存（见 ui_musicbox.c 注释）。
// ===============================================================================

// 文件列表菜单（复用全局单例 w_menu_main；枚举 SD 根目录 *.wav/*.mp3）
void ui_musicbox_menu_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_musicbox_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);
// 退出音乐盒（回主菜单）时调用：释放文件列表（PSRAM）
void ui_musicbox_menu_on_exit(void);

// STATE_MUSICBOX_PLAYING：首次获焦打开解码器并启动播放；每轮喂扬声器并处理触屏控件/实体键；
// 播放页的“退出”按钮（松手沿）返回文件列表菜单
void ui_musicbox_playing_on_enter(Key_Event *key_event, Global_State *global_state);
int32_t ui_musicbox_playing_event(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
