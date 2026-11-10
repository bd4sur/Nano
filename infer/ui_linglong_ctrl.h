#ifndef __NANO_UI_LINGLONG_CTRL_H__
#define __NANO_UI_LINGLONG_CTRL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "graphics.h"

// ===============================================================================
// 玲珑天象仪主界面显式触屏控件（硬件无关）
//
// 替代主天空界面原有的全屏 4x4 宫格隐式软按键（“软 key”）：
//   - 顶部靠右一排按钮：控制 / 退出；
//   - “控制”同时切换以下控件的显隐（同显同隐）：
//       · 底部四分之一区域的四个时间控制按钮：
//           <<（快退） 实时（恢复当前时间） 运行/暂停（时间流逝开关） >>（快进）；
//       · 画面中央的 3x3 半透明九宫格，功能与十六宫格软键 1~9 一一对应；
//       · 九宫格左侧竖排三枚设置按钮（与九宫格三行齐平）：投影算法、赤道坐标、
//           地平坐标，按钮上直接显示现值；
//       · 九宫格右侧竖排：顶部“设置”（等价 C 键，打开设置界面），下方两枚
//           大气散射、地景（显示现值）；
//   - “退出”等价于十六宫格 A 键（Esc），退出玲珑仪。
//
// 设计约束（与项目触屏化改造的既有范式一致）：
//   - 只绘制到调用方传入的帧缓冲，不调用 gfx_refresh；
//   - 不直接访问触屏 HAL：触屏样本由事件层（ui_app.c）统一采样后作为参数传入；
//   - 命中判定与绘制共用同一套几何常量，保证严格一致；
//   - 主天空界面与设置界面均使用显式触屏控件；实体硬键在两种界面下均照常工作。
// ===============================================================================

// ---------------------------------------------------------------------------
// 玲珑天象仪设置界面（5 列 x 4 行显式按钮，与控制面板同一圆角半透明样式）
//
// 布局（320x240）：直接叠加在主画面上，不清屏；按钮几何与控制面板一致。
// 第 1 行：云量、云类、云精度、黄道、返回（红色特殊样式）；
// 第 2 行：空、空、天体名称、平滑滤波、跟踪太阳；
// 第 3 行第 2~4 格：观察高度、涌浪波高、涌浪波长；其余为无功能备用按钮。
// 调用方填充 cells 数组；本模块只按固定布局绘制。
// ---------------------------------------------------------------------------
#define UI_LL_SETTING_COLS   (5)
#define UI_LL_SETTING_ROWS   (4)
#define UI_LL_SETTING_COUNT  (UI_LL_SETTING_COLS * UI_LL_SETTING_ROWS)

typedef struct {
    const wchar_t *label;   // NULL 表示无功能备用按钮
    const wchar_t *value;   // 第二行现值；NULL 表示不显示
    uint8_t value_r, value_g, value_b; // 现值文字颜色
    uint8_t is_special;     // 1=返回按钮（特殊配色/单行大字）
} UiLinglongSettingCell;

// 绘制设置界面（不刷新屏幕；调用方先绘制主画面，本函数直接叠加）
void    ui_linglong_ctrl_draw_settings(Nano_GFX *gfx, const UiLinglongSettingCell *cells);

// 设置界面触屏：命中并松手在同一按钮时返回 0..UI_LL_SETTING_COUNT-1；无动作返回 -1。
// 无功能备用按钮也会返回索引，由调用方按索引决定是否合成按键（当前映射为空）。
int32_t ui_linglong_ctrl_settings_handle_touch(int32_t touch_edge, int32_t x, int32_t y,
                                               int32_t down_x, int32_t down_y);

// 清除设置界面的按住高亮（进入/离开设置界面时调用）
void    ui_linglong_ctrl_settings_reset(void);

// 控件动作（ui_linglong_ctrl_handle_touch 的返回值，NONE 表示无动作）
#define UI_LL_CTRL_NONE       (0)
#define UI_LL_CTRL_CONTROLS   (1)  // 同时切换时间控制按钮与视角九宫格的显隐
#define UI_LL_CTRL_SETTINGS   (3)  // 等价于十六宫格 C 键：打开设置界面
#define UI_LL_CTRL_EXIT       (4)  // 等价于十六宫格 A 键：退出玲珑仪
#define UI_LL_CTRL_REWIND     (5)  // 时间快退 <<
#define UI_LL_CTRL_REALTIME   (6)  // 恢复当前时间
#define UI_LL_CTRL_RUNPAUSE   (7)  // 运行/暂停（时间流逝开关）
#define UI_LL_CTRL_FORWARD    (8)  // 时间快进 >>
// 视角九宫格：UI_LL_CTRL_VIEW_1 + (n-1)，n=1..9，对应软键 NANO_KEY_1..9
#define UI_LL_CTRL_VIEW_1     (10)
#define UI_LL_CTRL_VIEW_9     (18)
// 控制面板左侧/右侧的设置按钮（点击后直接改配置，不切换界面）
#define UI_LL_CTRL_PROJECTION   (19)  // 投影算法（鱼眼/透视）
#define UI_LL_CTRL_EQUATORIAL   (20)  // 赤道坐标（关/开）
#define UI_LL_CTRL_HORIZONTAL   (21)  // 地平坐标（关/方位角/坐标圈）
#define UI_LL_CTRL_SKY_MODEL    (22)  // 大气散射（0..5 循环）
#define UI_LL_CTRL_LANDSCAPE    (23)  // 地景（0..3 循环）

// 绘制/命中所需的控件状态（由调用方从玲珑仪当前配置回填；按钮上直接显示现值）
typedef struct {
    int32_t running_state;          // 时光机状态（0-停止 1-运行 2-实时），用于时间按钮高亮
    int32_t speed;                  // 时光机速度（正未来/负过去）
    int32_t projection;             // 投影算法 0-鱼眼 1-透视
    int32_t equatorial_coord;       // 赤道坐标 0-关 1-开
    int32_t horizontal_coord;       // 地平坐标 0-关 1-方位角 2-坐标圈
    int32_t sky_model;              // 大气散射 0..5
    int32_t landscape_index;        // 地景 0..3
} UiLinglongCtrlState;

// 初始化/复位：隐藏控制面板（时间按钮 + 视角九宫格 + 设置按钮）并清除按住高亮
void    ui_linglong_ctrl_init(void);
void    ui_linglong_ctrl_reset(void);

// 触屏事件处理（由玲珑仪事件处理器在主天空界面调用）：
//   touch_edge：TOUCH_EDGE_DOWN / TOUCH_EDGE_UP 位掩码（同帧可叠加）；
//   x,y：当前触点（松开沿为最后有效触点）；down_x,down_y：本次触摸序列按下点。
// 按下沿只锁存高亮，松开沿在按下点与松开点命中同一按钮时返回对应动作。
// 返回 UI_LL_CTRL_*；无动作返回 UI_LL_CTRL_NONE。
int32_t ui_linglong_ctrl_handle_touch(int32_t touch_edge, int32_t x, int32_t y,
                                      int32_t down_x, int32_t down_y);

// 主界面拖动视角（仅在控制面板隐藏、且按下点不在顶部按钮上时启用）：
// 输出自上次查询以来触点的像素增量（帧缓冲坐标）；返回1表示有拖动，0表示无。
// 本模块只负责“是否允许拖动/增量多少”，角度换算由调用方按实际画布尺寸完成，
// 不在此写死屏幕宽高。
int32_t ui_linglong_ctrl_poll_drag(int32_t touch_edge, int32_t is_touching,
                                   int32_t x, int32_t y,
                                   int32_t down_x, int32_t down_y,
                                   int32_t *out_dx, int32_t *out_dy);

// 控件显示状态自上次查询后是否变化（面板显隐或按住高亮变化）：变化过返回1并清除标记。
// 调用方据此决定是否需要重绘（避免点击空白天空也强制重绘整幅星图）。
uint8_t ui_linglong_ctrl_take_dirty(void);

// 绘制控件（不刷新屏幕，由调用方统一 gfx_refresh）。
// st 中的时光机状态用于时间按钮高亮，其余设置项用于按钮上直接显示现值。
void    ui_linglong_ctrl_draw(Nano_GFX *gfx, const UiLinglongCtrlState *st);

#ifdef __cplusplus
}
#endif

#endif
