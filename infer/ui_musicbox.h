#ifndef __NANO_UI_MUSICBOX_H__
#define __NANO_UI_MUSICBOX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui_app.h"

// ===============================================================================
// 音乐盒 UI 模块
// 入口：主菜单 [5] 音乐盒 → STATE_MUSICBOX_MENU（SD 卡根目录 WAV/MP3 文件列表）
//   → 选中曲目 → STATE_MUSICBOX_PLAYING（播放/暂停/音量/上下曲）
// 解码：WAV（RIFF PCM 8/16bit，自解析）与 MP3（vendor/minimp3.h，实现单元 ui_musicbox_mp3.c）
// 输出：统一解码为 int16 单声道 PCM，经 hal_audio_out.h 流式 HAL 播放（乒乓双缓冲 + 背压喂入，
//   范式同 ui_ofdm.c TX 路径）。进入播放态申请、退出时释放全部内存（见 ui_musicbox.c 注释）。
// ===============================================================================

// 文件列表菜单（复用全局单例 w_menu_main；枚举 SD 根目录 *.wav/*.mp3）
void ui_musicbox_menu_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_musicbox_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);
// 退出音乐盒（回主菜单）时调用：释放文件列表（PSRAM）
void ui_musicbox_menu_on_exit(void);

// STATE_MUSICBOX_PLAYING：首次获焦打开解码器并启动播放；每轮喂扬声器并处理按键；
// A 停止并返回文件列表菜单
void ui_musicbox_playing_on_enter(Key_Event *key_event, Global_State *global_state);
int32_t ui_musicbox_playing_event(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
