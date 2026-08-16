#ifndef __NANO_UI_ALMANAC_H__
#define __NANO_UI_ALMANAC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "graphics.h"
#include "platform.h"
#include "almanac.h"

// ===============================================================================
// 黄历（农历择日）模态框
//
// 数据来源：almanac.h（cnlunar 单头库，平台无关）。布局与配色参照 almanac.js
// 配套 HTML 四卡片（按 320x240 + 12px 字体做最小适配，详见 ui_almanac.c 头注释）：
//    卡1 - 八字 + 当日吉凶徽标 / 星期 | 星座 | 宿 | 建除 | 十二神
//    卡2 - 子午流注（当令器官 / 宜 / 忌 + 说明）
//    卡3 - 宜 / 忌
//    卡4 - 九宫飞星 / 胎神 / 彭祖百忌 / 时辰吉凶 + 吉神方位
//
// 内存纪律（AGENTS.md）：145KB cnlunar_result + 64KB 计算工作区全部走
// platform_malloc（PSRAM），进入时申请、关闭时释放；任务栈零大分配
// （cnlunar 内部改用 workspace 版接口，不使用原 145KB 栈上临时结构）。
// ===============================================================================

// 计算指定日期（hour/minute 决定 时柱/子午流注/时辰吉凶）的黄历并持有结果。
// 内部申请并保存 PSRAM 内存。返回 0 成功；负值失败（内存不足/日期越界，
// 失败后进入错误态，draw 显示错误提示；调用方仍应打开模态框）。
int32_t ui_almanac_open(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute);

// 释放结果与工作区（模态框关闭时调用）
void ui_almanac_close(void);

// 当前是否有可绘制的黄历结果
int32_t ui_almanac_is_open(void);

// 绘制整个屏幕的黄历模态框（覆盖日历页；关闭后由调用方重绘日历）
void ui_almanac_draw(Nano_GFX *gfx);

#ifdef __cplusplus
}
#endif

#endif
