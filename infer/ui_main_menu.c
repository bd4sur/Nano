// ui_main_menu.c —— 主菜单（STATE_MAIN_MENU）：表驱动的应用宫格
//
// 本模块从 ui_app.c 剥离而来（2026-11-28）：条目表、几何/配色、绘制（左侧 3 列滚动区 +
// 右侧 1 列固定区）、触屏与键盘交互、以及每个入口的点击行为（menu_act_*）全部集中在这里。
//
// 表驱动：每个入口只有三件事——**标题、图标路径、点击行为**——一律由表给出，绘制与点按
// 分发都只认表：
//   · 内置应用 → 静态表 s_app_menu_builtin_items[]（加一行即新增入口；个别入口可用
//     s_app_menu_hidden_actions[] 只屏蔽"上不上菜单"，点击行为与状态机保留）；
//   · 灵机 App → 进入菜单时扫描 app/ 目录（ui_scheme_app 的 catalog），标题取源码元数据
//     ;@title=，图标默认 icon/default.png（icon/<id>.png 存在则用它），点击行为 = 启动该 App。
//   两者在 s_menu_items[] 里拼成同一张运行时表（行优先排布）。
//
// 后台运行角标：灵机 App 有存活实例（(App.hide) / 退出确认框「后台」/ 留在后台冻结）时
// 图标右上角带绿色角标，点它即恢复同一实例。
//
// 交互（全部触屏自绘控件；实体键盘仅作兜底）：
//   · 左侧滚动区：上下拖动按像素跟手 + 松手惯性；短点按执行条目；左缘竖条是滚动进度条；
//   · 右侧固定区：返回 / 设置 / 自述 / 关机（仅短点按）；
//   · 键盘：数字键 1-9 = 表前 9 项；Esc/Shift/Ctrl/Enter = 返回/设置/自述/关机；上/下整行滚动。
//
// 与其它模块的边界：本模块只做"菜单这一页"；二级菜单（小游戏 STATE_GAME_MENU）仍在
// ui_app.c（init_game_menu / game_menu_item_action），入口动作 menu_act_game 调它。
// ===============================================================================

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ui_main_menu.h"

#include "graphics.h"
#include "hal_key.h"
#include "platform.h"
#include "ui.h"
#include "ui_color.h"        // UI_COLOR_LIGHT/DARK：主题分支
#include "ui_dict.h"         // ui_dict_enter：词典入口动作
#include "ui_ebook.h"        // ui_ebook_menu_init：电子书入口动作
#include "ui_icon.h"
#include "ui_llm.h"          // init_model_menu：鹦鹉笼入口的动作
#include "ui_musicbox.h"     // ui_musicbox_menu_init：音乐盒入口动作
#include "ui_ofdm.h"         // ui_ofdm_menu_init：寻呼机入口动作
#include "ui_scheme_app.h"   // 灵机 App 目录扫描 / 启动 / 在途触摸余波抑制

// ===============================================================================
// 主菜单：**表驱动**的应用宫格（左侧滚动区 + 右侧固定区）
//
// 布局（几何基准沿用十六宫格：单格宽度 = 屏宽/4，与旧宫格单格同宽）：
//   +---------------------------+---------+
//   | 番茄表 | 鹦鹉笼 | 玲珑仪  |  返回   |  <- 左侧滚动区：3 列 x 一屏 4 行，按条目表
//   | 电子书 | 小游戏 | 寻呼机  |  设置   |     顺序**从左到右、从上到下**依次填格；
//   | 控制台 | 时光集 | BadApple |  自述  |     内容超过一屏可拖动/惯性滚动
//   | …      |        |         |  关机   |  <- 右侧固定区：1 列 x 4 行（静态）
//   +---------------------------+---------+     单格 = 屏宽/4 x 屏高/4
// 单格内的图标/文字排版与旧十六宫格逐像素一致：图标（36x36）中心位于格子中心
// 偏上 8px，文字（12px alpha，中心对齐）再往下 26px。
//
// 表驱动（2026-11）：每个图标的属性只有三个——**标题、图标路径、点击行为**——
// 一律由表给出，绘制与点按分发都只认表，不再有「画一份、点一份」的重复分支：
//   · 内置应用 → 静态表 s_app_menu_builtin_items[]（C 侧，表里加一行即新增入口；
//     个别入口可用 s_app_menu_hidden_actions[] **屏蔽**——只不上菜单，实现与点击行为
//     都保留，当前屏蔽了与灵机 App 同名的 C 版「音乐盒」「词典」，以及已由灵机 App
//     「电子鹦鹉」取代的 C 版「鹦鹉笼」与二级菜单「小游戏」）；
//   · 灵机 App → 进入菜单时扫描 app/ 目录，逐个读出源码头部元数据 ;@title= 当标题
//     （见 ui_scheme_app.h），图标默认 icon/default.png（若 icon/<id>.png 存在则用它），
//     点击行为 = 启动该 App。新增 App = 往 app/ 丢一个带元数据的 .scm，固件无需改动。
//   两者在 s_menu_items[] 里拼成同一张运行时表（行优先排布），容量
//   UI_APP_MENU_ITEM_MAX = 内置上限 + UI_SCHEME_APP_MAX。
//
// 后台运行：某个灵机 App 还有活着的实例（被 (App.hide)、退出确认框的「后台」或
// "回主菜单"留在后台冻结）时，它的图标右上角带**绿色运行角标**，点它即**恢复**
// 同一实例（解释器池与 App 内状态都还在），而不是冷启动。
//
// 交互：
//   - 左侧：触屏控件。上下拖动按像素跟手滚动，松手带惯性；短点按执行对应条目。
//           滚动区左缘居中的竖条是滚动进度条（低饱和灰，提示本面板可滚动）。
//   - 右侧：静态按钮，仅响应短点按（返回/设置/自述/关机）。
//   - 键盘：数字键 1-9 = 表格前 9 项；Esc/Shift/Ctrl/Enter = 返回/设置/自述/关机；
//           上/下键整行滚动（见 ui_main_menu_event_handler 末段）。
//           触屏不派生任何键码（全屏宫格软键已整体退场，见 get_input_event）。
// ===============================================================================

#define UI_APP_MENU_SCROLL_COLS  (3) // 左侧滚动区列数（＝旧宫格的左 3 列）
#define UI_APP_MENU_VISIBLE_ROWS (4) // 左侧滚动区一屏可见行数（3 列 x 4 行 = 12 格恰好一屏）
#define UI_APP_MENU_FIXED_NUM    (4) // 右侧固定区按钮数（返回/设置/自述/关机）

// 条目表容量：内置表上限 + 灵机 App 扫描上限（见 ui_scheme_app.h 的 UI_SCHEME_APP_MAX）
#define UI_APP_MENU_BUILTIN_MAX   (16)
#define UI_APP_MENU_ITEM_MAX      (UI_APP_MENU_BUILTIN_MAX + UI_SCHEME_APP_MAX)
#define UI_APP_MENU_ICON_PATH_MAX (64)
// 默认图标（条目 icon_path 为空串时使用）
#define UI_APP_MENU_DEFAULT_ICON  PLATFORM_ROOT_DIR "/icon/default.png"

// 触屏：拖动判定阈值（累计纵向位移超过该值即认定为本序列是拖动而非点按）
#define UI_APP_MENU_DRAG_THRESHOLD (12)
// 触屏：惯性滚动减速度（px/s^2，与菜单控件一致）与启动/限速阈值
#define UI_APP_MENU_FLING_DECEL    (2000.0f)
#define UI_APP_MENU_FLING_VMAX     (4000.0f)
#define UI_APP_MENU_FLING_VMIN     (50.0f)

// 滚动进度条：贴在滚动区左缘的 2px 宽竖条，总高 = 屏高/8 且垂直居中（不抢眼，只作「可滚动」
// 的提示）。配色为低饱和灰（亮/暗各一套），不跟随菜单/文本框的蓝系滚动条。
// 水平位置保持旧观感：暗色占 x=2,3，亮色因避开黑色屏框而整体左移 1px 占 x=1,2。
#define UI_APP_MENU_SB_X          (0)
#define UI_APP_MENU_SB_W          (2)
#define UI_APP_MENU_SB_TRACK_DIV  (8) // 滚动条总高 = 屏高 / 该值

// -------------------------------------------------------------------------------
// 条目表：{标题, 图标路径, 点击行为}
// -------------------------------------------------------------------------------
// 点击行为签名：返回切换后的状态（= STATE_MAIN_MENU 表示“不切换/留在本菜单”）。
typedef struct Ui_App_Menu_Item Ui_App_Menu_Item;
typedef int32_t (*Ui_App_Menu_Action)(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item);

struct Ui_App_Menu_Item {
    const wchar_t     *title;                     // 标题（NULL = 空位，只铺底色）
    char               icon_path[UI_APP_MENU_ICON_PATH_MAX]; // 图标路径（空串 = UI_APP_MENU_DEFAULT_ICON）
    Ui_App_Menu_Action action;                    // 点击行为（NULL = 无动作）
    int32_t            app_index;                 // 灵机 App 在扫描表中的下标（-1 = 非灵机 App）
    int32_t            running;                   // 1 = 有存活实例（前台/后台），画运行角标
};

// 内置条目表是**静态常量**（不占运行时缓冲），故用同形但 icon_path 为指针的窄结构
typedef struct {
    const wchar_t     *title;
    const char        *icon_path;
    Ui_App_Menu_Action action;
} Ui_App_Menu_Builtin_Item;

// ---------------- 点击行为实现 ----------------
// 图标 36x36，经 ui_icon_draw_centered 绘制（首次读取解码后常驻缓存）。

// 番茄表（电子沙漏）
static int32_t menu_act_flip(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)ke; (void)gs; (void)item;
    return STATE_FLIP;
}

// 鹦鹉笼（模型菜单）
static int32_t menu_act_model(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)item;
    init_model_menu(ke, gs);
    return STATE_MODEL_MENU;
}

// 玲珑仪
static int32_t menu_act_linglong(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)ke; (void)gs; (void)item;
    return STATE_LINGLONG;
}

// 电子书（文件列表菜单）
static int32_t menu_act_ebook(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)item;
    ui_ebook_menu_init(ke, gs);
    return STATE_EBOOK;
}

// 音乐盒（曲目列表菜单）
static int32_t menu_act_musicbox(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)item;
    ui_musicbox_menu_init(ke, gs);
    return STATE_MUSICBOX_MENU;
}

// 小游戏（二级菜单）
static int32_t menu_act_game(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)item;
    init_game_menu(ke, gs);
    return STATE_GAME_MENU;
}

// 寻呼机（OFDM 收发）
static int32_t menu_act_ofdm(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)item;
    ui_ofdm_menu_init(ke, gs);
    return STATE_OFDM_MENU;
}

// 词典：索引就绪（缺失/失效则带进度构建）后进入查询状态；失败则停留在主菜单
// （错误画面已显示，停留片刻后重绘本菜单）
static int32_t menu_act_dict(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)item;
    if (ui_dict_enter(ke, gs) == 0) {
        return STATE_DICT_QUERY;
    }
    sleep_in_ms(1500);
    ui_main_menu_draw(ke, gs, 1);
    return STATE_MAIN_MENU;
}

// 控制台（Animac Scheme 解释器）
static int32_t menu_act_animac(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)ke; (void)gs; (void)item;
    return STATE_ANIMAC_INIT;
}

// 灵机 App：启动（已有后台实例则恢复）—— 行为完全由表项里的 app_index 决定
static int32_t menu_act_scheme_app(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)ke; (void)gs;
    return ui_scheme_app_catalog_enter(item->app_index);
}

// 右侧固定区：返回 / 设置 / 自述 / 关机
static int32_t menu_act_back(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)ke; (void)gs; (void)item;
    return STATE_SPLASH_SCREEN;
}

static int32_t menu_act_settings(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)ke; (void)gs; (void)item;
    return STATE_SETTING_MENU;
}

static int32_t menu_act_readme(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)ke; (void)gs; (void)item;
    return STATE_README;
}

static int32_t menu_act_poweroff(Key_Event *ke, Global_State *gs, const Ui_App_Menu_Item *item) {
    (void)ke; (void)gs; (void)item;
    return STATE_SHUTDOWN;
}

// -------------------------------------------------------------------------------
// 内置条目表（左侧滚动区）。顺序即宫格填充顺序（从左到右、从上到下）。
// 灵机 App 的图标接在这张表后面，由 menu_table_build() 拼上。
//
// 注意：表里可以保留“已屏蔽”的入口——哪些行不上主菜单由 s_app_menu_hidden_actions[]
// 决定（见下方），这样点击行为与表项都还在，恢复时删一行屏蔽记录即可。
// -------------------------------------------------------------------------------
static const Ui_App_Menu_Builtin_Item s_app_menu_builtin_items[] = {
    { L"沙漏番茄钟", PLATFORM_ROOT_DIR "/icon/fanqie.png",   menu_act_flip     },
    { L"鹦鹉笼", PLATFORM_ROOT_DIR "/icon/nano.png",      menu_act_model    },  // 已屏蔽（见下）
    { L"玲珑天象仪", PLATFORM_ROOT_DIR "/icon/linglong.png",  menu_act_linglong },
    { L"电子书", PLATFORM_ROOT_DIR "/icon/ebook.png",     menu_act_ebook    },
    { L"音乐盒", PLATFORM_ROOT_DIR "/icon/music.png",     menu_act_musicbox },  // 已屏蔽（见下）
    { L"小游戏", PLATFORM_ROOT_DIR "/icon/game.png",      menu_act_game     },  // 已屏蔽（见下）
    { L"声波短消息", PLATFORM_ROOT_DIR "/icon/ptt.png",       menu_act_ofdm     },
    { L"词典",   PLATFORM_ROOT_DIR "/icon/dict.png",      menu_act_dict     },  // 已屏蔽（见下）
    { L"灵机控制台", PLATFORM_ROOT_DIR "/icon/animac.png",    menu_act_animac   },
};
#define UI_APP_MENU_BUILTIN_NUM ((int32_t)(sizeof(s_app_menu_builtin_items) / sizeof(s_app_menu_builtin_items[0])))

// 主菜单里**屏蔽**的内置入口（2026-11）：只不摆进格子，功能实现一律保留。
//   · 「音乐盒」—— 灵机 App 版（app/musicbox.scm，图标沿用 icon/music.png）已经在同一张
//     表里，两个同名格子并存会分不清；
//   · 「词典」  —— 同理，app/dict.scm 是灵机 App 版；
//   · 「鹦鹉笼」—— 灵机 App 版「电子鹦鹉」（app/nanochat.scm）已在同一张表里，
//     且沿用同一图标 icon/nano.png（见下方 s_scheme_app_icon_override），两个同名
//     格子并存会分不清；
//   · 「小游戏」—— 二级菜单里的 C 版游戏入口（黄金矿工/遗传算法/水波/水池）退场，
//     游戏类 App 统一由主菜单的灵机 App 入口直达。
// 屏蔽的只是**菜单入口**：点击行为（menu_act_musicbox / menu_act_dict /
// menu_act_model / menu_act_game）、状态机（STATE_MUSICBOX_MENU /
// STATE_DICT_QUERY / STATE_MODEL_MENU / STATE_GAME_MENU）与 C 侧实现全在，随时可恢复——
// 删掉对应那一行即可。
static const Ui_App_Menu_Action s_app_menu_hidden_actions[] = {
    menu_act_musicbox,
    menu_act_dict,
    menu_act_model,
    menu_act_game,
};

static int32_t menu_builtin_hidden(Ui_App_Menu_Action action) {
    for (int32_t i = 0; i < (int32_t)(sizeof(s_app_menu_hidden_actions) / sizeof(s_app_menu_hidden_actions[0])); i++) {
        if (s_app_menu_hidden_actions[i] == action) {
            return 1;
        }
    }
    return 0;
}

// 右侧固定区条目（自上而下，各占屏高/4）
static const Ui_App_Menu_Builtin_Item s_app_menu_fixed_items[UI_APP_MENU_FIXED_NUM] = {
    { L"返回", PLATFORM_ROOT_DIR "/icon/home.png",     menu_act_back     },
    { L"设置", PLATFORM_ROOT_DIR "/icon/settings.png", menu_act_settings },
    { L"自述", PLATFORM_ROOT_DIR "/icon/readme.png",   menu_act_readme   },
    { L"关机", PLATFORM_ROOT_DIR "/icon/poweroff.png", menu_act_poweroff },
};

// 运行时条目表（内置项 + 灵机 App 项；进入菜单时重建，一次性分配后常驻）
static Ui_App_Menu_Item *s_menu_items = NULL;
static int32_t s_menu_item_num = 0;

typedef struct Ui_App_Menu_Geometry {
    int32_t cell_w;        // 单格宽度（屏宽/4，与旧宫格单格同宽）
    int32_t scroll_cell_h; // 滚动区单格高度（屏高/4，与旧宫格单格同高）
    int32_t fixed_cell_h;  // 固定区单格高度（屏高/4，与旧宫格第 4 列一致）
    int32_t scroll_x0, scroll_y0, scroll_w, scroll_h; // 滚动区视口
    int32_t fixed_x0, fixed_y0, fixed_w, fixed_h;     // 固定区
    int32_t content_h;     // 滚动内容总高度
    int32_t content_rows;  // 滚动内容总行数（含末行空位所在的整行）
    int32_t max_scroll_px; // 最大滚动位置（0 = 内容不足一屏，不可滚）
    int32_t item_num;      // 滚动区条目数
} Ui_App_Menu_Geometry;

static void menu_geometry(Nano_GFX *gfx, Ui_App_Menu_Geometry *g) {
    g->item_num = s_menu_item_num;   // 表驱动：条目数来自运行时条目表
    g->cell_w        = (int32_t)gfx->width / (UI_APP_MENU_SCROLL_COLS + 1); // 屏宽/4
    g->scroll_cell_h = (int32_t)gfx->height / UI_APP_MENU_VISIBLE_ROWS;     // 屏高/4
    g->fixed_cell_h  = (int32_t)gfx->height / UI_APP_MENU_FIXED_NUM;        // 屏高/4

    g->scroll_x0 = 0;
    g->scroll_y0 = 0;
    g->scroll_w  = g->cell_w * UI_APP_MENU_SCROLL_COLS;
    g->scroll_h  = (int32_t)gfx->height;

    g->fixed_x0 = g->scroll_w;
    g->fixed_y0 = 0;
    g->fixed_w  = (int32_t)gfx->width - g->scroll_w;
    g->fixed_h  = (int32_t)gfx->height;

    int32_t content_rows = (g->item_num + UI_APP_MENU_SCROLL_COLS - 1) / UI_APP_MENU_SCROLL_COLS;
    g->content_rows = content_rows;
    g->content_h = content_rows * g->scroll_cell_h;
    g->max_scroll_px = g->content_h - g->scroll_h;
    if (g->max_scroll_px < 0) {
        g->max_scroll_px = 0;
    }
}

typedef struct Ui_App_Menu_Palette {
    uint8_t fixed_bg_r, fixed_bg_g, fixed_bg_b;   // 固定区按钮底色
    uint8_t scroll_bg_r, scroll_bg_g, scroll_bg_b; // 滚动区按钮底色（与固定区不同色）
    uint8_t text_r, text_g, text_b;               // 按钮文字色（两侧一致）
} Ui_App_Menu_Palette;

static void menu_palette(Global_State *global_state, Ui_App_Menu_Palette *p) {
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        // 亮色：固定区纯白；滚动区提亮为很淡的蓝灰（比纯白略沉一点，不形成深色块）
        p->fixed_bg_r = 0xff; p->fixed_bg_g = 0xff; p->fixed_bg_b = 0xff;
        p->scroll_bg_r = 0xf1; p->scroll_bg_g = 0xf5; p->scroll_bg_b = 0xfb;
        p->text_r = 0x66; p->text_g = 0x66; p->text_b = 0x66;
    }
    else {
        // 暗色：右固定区比左滚动区亮一档（仍为同系蓝灰），既保留“右侧固定”的暗示，也不与左区割裂
        p->fixed_bg_r = 34; p->fixed_bg_g = 42; p->fixed_bg_b = 58;
        p->scroll_bg_r = 16; p->scroll_bg_g = 22; p->scroll_bg_b = 36;
        p->text_r = 0xcc; p->text_g = 0xcc; p->text_b = 0xcc;
    }
}

// 滚动/惯性状态（本控件为全局唯一实例，静态持有）
typedef struct Ui_App_Menu_Scroll {
    int32_t scroll_px;   // 逻辑滚动位置（0..max_scroll_px）
    // 触摸序列跟踪
    int32_t touch_active;
    int32_t touch_is_dragging;
    int32_t touch_start_x;
    int32_t touch_start_y;
    int32_t touch_anchor_scroll_px; // 按下时的滚动位置（拖动锚点）
    // 拖动速度采样（松手惯性初速度估算）
    int32_t  touch_track_scroll;
    uint64_t touch_track_ts;
    float    touch_track_vel;
    // 松手惯性滚动
    float    fling_velocity;
    float    fling_scroll_px;
    uint64_t fling_last_timestamp;
} Ui_App_Menu_Scroll;

static Ui_App_Menu_Scroll s_app_menu_scroll;

// -------------------------------------------------------------------------------
// 条目表构建（进入主菜单时调用）
// -------------------------------------------------------------------------------
// 灵机 App 专属图标：默认约定是 icon/<id>.png（存在则自动使用）。个别 App 想沿用已有
// 图标时在这里登记 id → 路径，免得为同一张图再存一份副本（图标缓存也少占一个槽）。
static const struct {
    const char *app_id;
    const char *icon_path;
} s_scheme_app_icon_override[] = {
    // 灵机版音乐盒（app/musicbox.scm）沿用 C 版音乐盒的图标
    { "musicbox", PLATFORM_ROOT_DIR "/icon/music.png" },
    // 灵机版「电子鹦鹉」（app/nanochat.scm）沿用原 C 版鹦鹉笼的图标
    { "nanochat", PLATFORM_ROOT_DIR "/icon/nano.png" },
    // 灵机版「贪吃蛇」（app/snake.scm）沿用已退场的 C 版「小游戏」入口图标
    { "snake", PLATFORM_ROOT_DIR "/icon/game.png" },
};

// 图标选择：① 覆盖表 → ② icon/<id>.png（存在则用）→ ③ 留空（绘制时取默认图标）。
// 只做一次存在性探测（open + close，不读内容）。
static void menu_pick_icon(Ui_App_Menu_Item *item, const char *app_id) {
    char path[UI_APP_MENU_ICON_PATH_MAX];
    item->icon_path[0] = '\0';
    if (app_id == NULL || app_id[0] == '\0') {
        return;
    }
    for (int32_t i = 0; i < (int32_t)(sizeof(s_scheme_app_icon_override) / sizeof(s_scheme_app_icon_override[0])); i++) {
        if (strcmp(s_scheme_app_icon_override[i].app_id, app_id) == 0) {
            strncpy(item->icon_path, s_scheme_app_icon_override[i].icon_path, sizeof(item->icon_path) - 1);
            item->icon_path[sizeof(item->icon_path) - 1] = '\0';
            return;
        }
    }
    snprintf(path, sizeof(path), "%s/icon/%s.png", PLATFORM_ROOT_DIR, app_id);
    if (platform_file_open(path) == 0) {
        platform_file_close();
        strncpy(item->icon_path, path, sizeof(item->icon_path) - 1);
        item->icon_path[sizeof(item->icon_path) - 1] = '\0';
    }
}

// 重建条目表：内置项（静态表，顺序不变）+ 灵机 App（按 id 字典序，见 ui_scheme_app.c 的
// 扫描排序）。行优先填充由绘制侧负责，表自身就是“从左到右、从上到下”的顺序。
static void menu_table_build(void) {
    if (s_menu_items == NULL) {
        s_menu_items = (Ui_App_Menu_Item *)platform_calloc(UI_APP_MENU_ITEM_MAX, sizeof(Ui_App_Menu_Item));
        if (s_menu_items == NULL) {   // 防御（实际不可达）：空表，只剩固定区可用
            s_menu_item_num = 0;
            return;
        }
    }
    memset(s_menu_items, 0, sizeof(Ui_App_Menu_Item) * UI_APP_MENU_ITEM_MAX);

    int32_t n = 0;
    int32_t builtin_added = 0;
    for (int32_t i = 0; i < UI_APP_MENU_BUILTIN_NUM && n < UI_APP_MENU_ITEM_MAX; i++) {
        if (menu_builtin_hidden(s_app_menu_builtin_items[i].action)) {
            continue;   // 被屏蔽的内置入口（如 C 版「音乐盒」「词典」）：不上菜单，实现保留
        }
        Ui_App_Menu_Item *item = &s_menu_items[n++];
        item->title = s_app_menu_builtin_items[i].title;
        strncpy(item->icon_path, s_app_menu_builtin_items[i].icon_path, sizeof(item->icon_path) - 1);
        item->action = s_app_menu_builtin_items[i].action;
        item->app_index = -1;
        item->running = 0;
        builtin_added++;
    }

    // 灵机 App：重新扫描 app/（支持热插拔），标题取源码元数据 ;@title=。
    // 注意：标题指针指向扫描缓冲，下次 refresh 后失效——所以每次进菜单都整表重建。
    int32_t app_num = ui_scheme_app_catalog_refresh();
    for (int32_t i = 0; i < app_num && n < UI_APP_MENU_ITEM_MAX; i++) {
        Ui_App_Menu_Item *item = &s_menu_items[n++];
        item->title = ui_scheme_app_catalog_title(i);
        item->action = menu_act_scheme_app;
        item->app_index = i;
        item->running = ui_scheme_app_catalog_running(i);
        menu_pick_icon(item, ui_scheme_app_catalog_id(i));
    }
    s_menu_item_num = n;
    printf("[main-menu] 条目表：内置 %d 项（屏蔽 %d 项）+ 灵机 App %d 项 = %d 项\n",
           builtin_added, (int)(UI_APP_MENU_BUILTIN_NUM - builtin_added), app_num, s_menu_item_num);
}

// 进入主菜单：重建条目表（含灵机 App 目录扫描与后台运行角标刷新），复位触摸跟踪，
// 并夹住滚动位置（**不强制回顶**：从 App 退出/切后台回来时仍停在刚才那一屏，
// 方便连续操作；滚动位置只在超出新内容高度时才被收拢）。
void ui_main_menu_entry(Global_State *global_state) {
    menu_table_build();

    int32_t keep_scroll_px = s_app_menu_scroll.scroll_px;
    memset(&s_app_menu_scroll, 0, sizeof(s_app_menu_scroll));
    s_app_menu_scroll.scroll_px = keep_scroll_px;

    Ui_App_Menu_Geometry g;
    menu_geometry(global_state->gfx, &g);
    if (s_app_menu_scroll.scroll_px > g.max_scroll_px) {
        s_app_menu_scroll.scroll_px = g.max_scroll_px;
    }
}

// 绘制单格（底色矩形 + 图标 + 文字 + 后台运行角标）。几何与排版与旧十六宫格一致：
//   - 首列左缘、首行上缘各内缩 1px（旧宫格的 bx/by 规则，避免与屏幕边框粘连）；
//   - 图标中心 = 格心偏上 8px；文字中心 = 图标中心再往下 26px（后画，覆盖在图标上）。
// icon_path 为 NULL/空串 = 默认图标 icon/default.png；text 为 NULL = 空位（只铺底色）。
// running = 1 时在图标右上角画一个绿色小圆点（“后台有这个 App 的活实例”）。
// icon_clip 为图标的裁剪矩形 {x, y, w, h}（滚动区传视口，防止半格图标画出界）。
static void ui_main_menu_draw_cell(
    Global_State *global_state,
    int32_t x0, int32_t y0, int32_t w, int32_t h,
    int32_t is_first_col, int32_t is_first_row,
    const wchar_t *text, const char *icon_path, int32_t running,
    uint8_t bg_r, uint8_t bg_g, uint8_t bg_b,
    uint8_t text_r, uint8_t text_g, uint8_t text_b,
    const int32_t *icon_clip
) {
    int32_t bx = (is_first_col) ? 1 : 0;
    int32_t by = (is_first_row) ? 1 : 0;
    gfx_draw_rectangle(global_state->gfx, x0 + bx, y0 + by, w - 1 - bx, h - 1 - by, bg_r, bg_g, bg_b, 1);

    // text == NULL：只铺底色（用于内容末行的空位，保持格子网格连续）
    if (text == NULL) {
        return;
    }

    int32_t cell_cx = x0 + w / 2;
    int32_t cell_cy = y0 + h / 2;
    if (icon_path == NULL || icon_path[0] == '\0') {
        icon_path = UI_APP_MENU_DEFAULT_ICON;
    }

    // 图标（36x36，带 PSRAM 缓存）
    if (icon_clip != NULL) {
        ui_icon_draw_centered_clip(global_state->gfx, icon_path, cell_cx, cell_cy - 8,
            icon_clip[0], icon_clip[1], icon_clip[2], icon_clip[3]);
    }
    else {
        ui_icon_draw_centered(global_state->gfx, icon_path, cell_cx, cell_cy - 8);
    }

    // 后台运行角标：图标（36x36，中心 = 格心偏上 8px）右上角内侧的一个 7x5 绿点。
    // 用 5 条横线拼出一个近似圆点（比实心方块不抢眼），配色跟亮/暗主题。
    if (running) {
        uint8_t dot_r = 0x3d, dot_g = 0xdc, dot_b = 0x6e;
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            dot_r = 0x17; dot_g = 0x8f; dot_b = 0x3a;
        }
        int32_t dot_x = cell_cx + 15;
        int32_t dot_y = cell_cy - 8 - 15;
        for (int32_t dy = -2; dy <= 2; dy++) {
            int32_t half = (dy == 0) ? 3 : ((dy == -2 || dy == 2) ? 1 : 2);
            gfx_draw_line(global_state->gfx, dot_x - half, dot_y + dy, dot_x + half, dot_y + dy,
                dot_r, dot_g, dot_b, 1);
        }
    }

    // 文字（12px alpha，中心对齐）
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, (wchar_t *)text,
        cell_cx, cell_cy - 8 + 18 + 8, text_r, text_g, text_b, 1);
}

// 滚动进度条：贴滚动区左缘的 2px 低饱和灰竖条。总高 = 屏高/8、垂直居中；
// 轨道高固定，滑块高 = 轨道高 * 可视高 / 内容高，位置按滚动像素线性映射；
// 内容不足一屏时滑块占满轨道（此时也不会有滚动，仅作为“此处可滚”的弱提示）。
static void ui_main_menu_draw_scroll_bar(Global_State *global_state, const Ui_App_Menu_Geometry *g) {
    int32_t track_h = g->scroll_h / UI_APP_MENU_SB_TRACK_DIV;
    if (track_h < 8) track_h = 8; // 小屏兼容：太短就看不出比例了
    int32_t track_y = g->scroll_y0 + (g->scroll_h - track_h) / 2; // 垂直居中
    // 水平位置：保持旧观感（亮色避开黑色屏框，整体左移 1px）
    int32_t x = UI_APP_MENU_SB_X + ((global_state->ui_color_style == UI_COLOR_LIGHT) ? 1 : 2);

    uint8_t track_r = 0, track_g = 0, track_b = 0;
    uint8_t thumb_r = 0, thumb_g = 0, thumb_b = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        track_r = 0xd9; track_g = 0xdd; track_b = 0xe3; // 浅灰（微带蓝，跟面板底色同系）
        thumb_r = 0x99; thumb_g = 0x9e; thumb_b = 0xa7; // 中灰
    }
    else {
        track_r = 0x40; track_g = 0x46; track_b = 0x52; // 深灰（在深色面板上提亮一档，可辨识但不抢眼）
        thumb_r = 0x92; thumb_g = 0x99; thumb_b = 0xa4; // 浅中灰
    }

    // 轨道
    for (int32_t i = 0; i < UI_APP_MENU_SB_W; i++) {
        gfx_draw_line(global_state->gfx, x + i, track_y, x + i, track_y + track_h - 1,
            track_r, track_g, track_b, 1);
    }

    // 滑块
    int32_t content_px = (g->content_h < 1) ? 1 : g->content_h;
    int32_t view_px = (g->scroll_h < 1) ? 1 : g->scroll_h;
    int32_t thumb_h = (content_px <= view_px)
        ? track_h
        : ((view_px * track_h + content_px / 2) / content_px);
    if (thumb_h < 4) thumb_h = 4;
    if (thumb_h > track_h) thumb_h = track_h;
    int32_t thumb_y = track_y + ((s_app_menu_scroll.scroll_px * track_h + content_px / 2) / content_px);
    if (thumb_y + thumb_h > track_y + track_h) thumb_y = track_y + track_h - thumb_h;
    if (thumb_y < track_y) thumb_y = track_y;
    for (int32_t i = 0; i < UI_APP_MENU_SB_W; i++) {
        gfx_draw_line(global_state->gfx, x + i, thumb_y, x + i, thumb_y + thumb_h - 1,
            thumb_r, thumb_g, thumb_b, 1);
    }
}

// 绘制左侧滚动区（底色 + 可见格子 + 左缘滚动进度条）。
// 底色每次重绘都重铺（与旧宫格格间缝隙一致：亮色白 / 暗色黑），既兜住格子之间 1px 的缝，
// 又保证拖动/惯性期间只重绘本区时不残留上一帧的格子痕迹。
static void ui_main_menu_draw_scroll_area(
    Global_State *global_state,
    const Ui_App_Menu_Geometry *g, const Ui_App_Menu_Palette *p
) {
    int32_t scroll_px = s_app_menu_scroll.scroll_px;
    int32_t icon_clip[4] = { g->scroll_x0, g->scroll_y0, g->scroll_w, g->scroll_h };

    gfx_set_clip(global_state->gfx, g->scroll_x0, g->scroll_y0, g->scroll_w, g->scroll_h);

    // 底色（同 gfx_fill_white / gfx_soft_clear 的屏色）
    uint8_t bg_r = 0, bg_g = 0, bg_b = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        bg_r = 0xff; bg_g = 0xff; bg_b = 0xff;
    }
    gfx_draw_rectangle(global_state->gfx, g->scroll_x0, g->scroll_y0, g->scroll_w, g->scroll_h,
        bg_r, bg_g, bg_b, 1);

    // 首个可能可见的行：其格顶 = scroll_y0 + row*cell_h - scroll_px（可为负，被裁剪）
    int32_t first_row = scroll_px / g->scroll_cell_h;
    // 多画一行：亚行滚动时上下各有一个部分可见的格子
    for (int32_t row = first_row; row <= first_row + UI_APP_MENU_VISIBLE_ROWS; row++) {
        int32_t y0 = g->scroll_y0 + row * g->scroll_cell_h - scroll_px;
        if (y0 >= g->scroll_y0 + g->scroll_h) {
            break;
        }
        for (int32_t col = 0; col < UI_APP_MENU_SCROLL_COLS; col++) {
            int32_t index = row * UI_APP_MENU_SCROLL_COLS + col;
            const Ui_App_Menu_Item *item = (index < g->item_num) ? &s_menu_items[index] : NULL;
            if (item == NULL) {
                // 内容末行的空位：只铺同色底格（不画图标/文字），避免右侧露出黑色空洞
                if (row < g->content_rows) {
                    ui_main_menu_draw_cell(global_state,
                        g->scroll_x0 + col * g->cell_w, y0, g->cell_w, g->scroll_cell_h,
                        (col == 0), (row == 0),
                        NULL, NULL, 0,
                        p->scroll_bg_r, p->scroll_bg_g, p->scroll_bg_b,
                        p->text_r, p->text_g, p->text_b,
                        icon_clip);
                }
                continue;
            }
            ui_main_menu_draw_cell(global_state,
                g->scroll_x0 + col * g->cell_w, y0, g->cell_w, g->scroll_cell_h,
                (col == 0), (row == 0),
                item->title, item->icon_path, item->running,
                p->scroll_bg_r, p->scroll_bg_g, p->scroll_bg_b,
                p->text_r, p->text_g, p->text_b,
                icon_clip);
        }
    }

    gfx_reset_clip(global_state->gfx);

    // 滚动进度条（贴在滚动区左缘，提示左侧面板可上下拖动）
    ui_main_menu_draw_scroll_bar(global_state, g);
}

// 主菜单绘制：is_full_redraw=1 整屏重绘（清屏 + 滚动区 + 固定区）；
// 0 仅重绘左侧滚动区（拖动/惯性期间逐帧调用，右侧固定区保持不动）。内含推屏。
void ui_main_menu_draw(Key_Event *key_event, Global_State *global_state, int32_t is_full_redraw) {
    Ui_App_Menu_Geometry g;
    Ui_App_Menu_Palette p;
    (void)key_event; // 本控件只消费触屏电平/边沿，绘制不依赖输入事件（保留参数与其它控件同名同形）
    menu_geometry(global_state->gfx, &g);
    menu_palette(global_state, &p);

    if (is_full_redraw) {
        // 清屏
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            gfx_fill_white(global_state->gfx);
        }
        else {
            gfx_soft_clear(global_state->gfx);
        }
    }

    // 左侧滚动区
    ui_main_menu_draw_scroll_area(global_state, &g, &p);

    // 右侧固定区（静态 4 格，仅整屏重绘时绘制）
    if (is_full_redraw) {
        for (int32_t i = 0; i < UI_APP_MENU_FIXED_NUM; i++) {
            ui_main_menu_draw_cell(global_state,
                g.fixed_x0, g.fixed_y0 + i * g.fixed_cell_h, g.fixed_w, g.fixed_cell_h,
                0, (i == 0),
                s_app_menu_fixed_items[i].title, s_app_menu_fixed_items[i].icon_path, 0,
                p.fixed_bg_r, p.fixed_bg_g, p.fixed_bg_b,
                p.text_r, p.text_g, p.text_b,
                NULL);
        }
    }

    gfx_refresh(global_state->gfx);
}

// 执行滚动区第 index 个条目：**只查表并调它的点击行为**，不再有 per-index 分支。
// 返回切换后的状态（= STATE_MAIN_MENU 表示不切换，如词典索引构建失败后停留）。
static int32_t menu_activate_item(Key_Event *key_event, Global_State *global_state, int32_t index) {
    if (index < 0 || index >= s_menu_item_num) {
        return STATE_MAIN_MENU;
    }
    const Ui_App_Menu_Item *item = &s_menu_items[index];
    if (item->action == NULL) {
        return STATE_MAIN_MENU;   // 空位/占位项：无行为
    }
    return item->action(key_event, global_state, item);
}

// 执行右侧固定区第 index 个按钮（返回/设置/自述/关机），同样查表
static int32_t menu_fixed_action(Key_Event *key_event, Global_State *global_state, int32_t index) {
    if (index < 0 || index >= UI_APP_MENU_FIXED_NUM) {
        return STATE_MAIN_MENU;
    }
    const Ui_App_Menu_Builtin_Item *item = &s_app_menu_fixed_items[index];
    return (item->action != NULL) ? item->action(key_event, global_state, NULL) : STATE_MAIN_MENU;
}

void ui_main_menu_event_handler(Key_Event *key_event, Global_State *global_state) {
    Ui_App_Menu_Geometry g;
    menu_geometry(global_state->gfx, &g);

    // 退出/切后台瞬间的在途触摸序列余波：本菜单是 App 退出后的落点，旧序列的触屏电平
    // 与迟到的 UP 边沿若当成一次点按，会直接在宫格里启动 App（见 ui_scheme_app.c 的
    // s_list_touch_guard）。抑制窗口内只剥离触摸字段，按键照常。
    Key_Event ke_filtered = *key_event;
    if (ui_scheme_app_touch_aftermath_filter(&ke_filtered)) {
        key_event = &ke_filtered;
    }

    // ========================================================================
    // 松手惯性滚动（fling）：动画激活期间本 handler 每帧推进一次；
    // 任意新触摸或按键边沿立即终止动画（随后的触摸/按键逻辑以当前滚动位置接管）
    // ========================================================================
    if (s_app_menu_scroll.fling_velocity != 0.0f) {
        if (key_event->is_touching || key_event->touch_edge != 0 || key_event->key_edge != 0
            || key_event->wheel_dy != 0) {
            s_app_menu_scroll.fling_velocity = 0.0f;
        }
        else {
            uint64_t now = get_timestamp_in_ms();
            float dt_s = (float)(now - s_app_menu_scroll.fling_last_timestamp) / 1000.0f;
            s_app_menu_scroll.fling_last_timestamp = now;
            if (dt_s > 0.05f) dt_s = 0.05f; // 帧间隔异常（阻塞/掉帧）时限幅，避免跳动
            if (dt_s > 0.0f) {
                float scroll = s_app_menu_scroll.fling_scroll_px + s_app_menu_scroll.fling_velocity * dt_s;
                // 线性减速度，速度归零即停
                float decel = UI_APP_MENU_FLING_DECEL * dt_s;
                if (s_app_menu_scroll.fling_velocity > 0.0f) {
                    s_app_menu_scroll.fling_velocity -= decel;
                    if (s_app_menu_scroll.fling_velocity < 0.0f) s_app_menu_scroll.fling_velocity = 0.0f;
                }
                else {
                    s_app_menu_scroll.fling_velocity += decel;
                    if (s_app_menu_scroll.fling_velocity > 0.0f) s_app_menu_scroll.fling_velocity = 0.0f;
                }
                if (scroll <= 0.0f) {
                    scroll = 0.0f;
                    s_app_menu_scroll.fling_velocity = 0.0f;
                }
                if (scroll >= (float)g.max_scroll_px) {
                    scroll = (float)g.max_scroll_px;
                    s_app_menu_scroll.fling_velocity = 0.0f;
                }
                s_app_menu_scroll.fling_scroll_px = scroll;
                int32_t new_scroll_px = (int32_t)scroll;
                if (new_scroll_px != s_app_menu_scroll.scroll_px) {
                    s_app_menu_scroll.scroll_px = new_scroll_px;
                    ui_main_menu_draw(key_event, global_state, 0);
                }
            }
        }
    }

    // 滚轮/触控板：直接按像素滚动左侧滚动区（说明见 ui.c ui_wheel_scroll_clamp）。
    // 无锚点、不占触摸状态机，因此一帧可走任意像素，不受画布高度限制。
    if (key_event->wheel_dy != 0) {
        int32_t wdy = key_event->wheel_dy;
        key_event->wheel_dy = 0;
        if (!s_app_menu_scroll.touch_active) {
            int32_t next = ui_wheel_scroll_clamp(s_app_menu_scroll.scroll_px, wdy, g.max_scroll_px);
            if (next != s_app_menu_scroll.scroll_px) {
                s_app_menu_scroll.scroll_px = next;
                s_app_menu_scroll.fling_velocity = 0.0f;
                // 仅重绘左侧滚动区：右侧固定区保持不动
                ui_main_menu_draw(key_event, global_state, 0);
            }
        }
    }

    // ========================================================================
    // 触屏交互（序列开始/结束认 touch_edge 边沿事件，拖动轨迹逐帧读触点快照）：
    //   - 上下拖动：滚动区按像素跟手滚动，松手按速度启动惯性；
    //   - 短点按：执行被点按钮（左滚动区=应用，右固定区=返回/设置/自述/关机）。
    // 动作一律在松手沿触发，故状态切换后下一状态看到的触屏电平为 0，
    // 不会把本次触摸误当作新的点击（见本文件「输入事件处理」的按键/触屏处理原则第 4 条）。
    // ========================================================================
    // 触摸序列开始：DOWN 边沿（按下点坐标取 touch_down_*）或电平兜底
    if (((key_event->touch_edge & TOUCH_EDGE_DOWN) || key_event->is_touching) && !s_app_menu_scroll.touch_active) {
        s_app_menu_scroll.touch_active = 1;
        s_app_menu_scroll.touch_is_dragging = 0;
        if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
            s_app_menu_scroll.touch_start_x = key_event->touch_down_x;
            s_app_menu_scroll.touch_start_y = key_event->touch_down_y;
        }
        else {
            s_app_menu_scroll.touch_start_x = key_event->touch_x;
            s_app_menu_scroll.touch_start_y = key_event->touch_y;
        }
        s_app_menu_scroll.touch_anchor_scroll_px = s_app_menu_scroll.scroll_px;
        s_app_menu_scroll.touch_track_scroll = s_app_menu_scroll.scroll_px;
        s_app_menu_scroll.touch_track_ts = get_timestamp_in_ms();
        s_app_menu_scroll.touch_track_vel = 0.0f;
    }
    else if (key_event->is_touching && s_app_menu_scroll.touch_active) {
        int32_t dy = key_event->touch_y - s_app_menu_scroll.touch_start_y; // >0：手指下滑
        if (!s_app_menu_scroll.touch_is_dragging
            && (dy > UI_APP_MENU_DRAG_THRESHOLD || dy < -UI_APP_MENU_DRAG_THRESHOLD)) {
            s_app_menu_scroll.touch_is_dragging = 1;
        }
        if (s_app_menu_scroll.touch_is_dragging) {
            // 手指下滑 → 内容下移 → 滚动位置前移；相对锚点按像素计算，1:1 跟手无累计误差
            int32_t new_scroll_px = s_app_menu_scroll.touch_anchor_scroll_px - dy;
            if (new_scroll_px < 0) new_scroll_px = 0;
            if (new_scroll_px > g.max_scroll_px) new_scroll_px = g.max_scroll_px;
            if (new_scroll_px != s_app_menu_scroll.scroll_px) {
                s_app_menu_scroll.scroll_px = new_scroll_px;
                // 仅重绘左侧滚动区：右侧固定区保持不动
                ui_main_menu_draw(key_event, global_state, 0);
            }
            // 拖动速度采样：指数平滑（0.6/0.4），供松手惯性初速度估算
            uint64_t now = get_timestamp_in_ms();
            uint32_t dt_ms = (uint32_t)(now - s_app_menu_scroll.touch_track_ts);
            if (dt_ms > 0) {
                float v_inst = (float)(new_scroll_px - s_app_menu_scroll.touch_track_scroll) * 1000.0f / (float)dt_ms;
                s_app_menu_scroll.touch_track_vel = s_app_menu_scroll.touch_track_vel * 0.6f + v_inst * 0.4f;
                s_app_menu_scroll.touch_track_scroll = new_scroll_px;
                s_app_menu_scroll.touch_track_ts = now;
            }
        }
    }
    // 触摸序列结束：UP 边沿（同帧 DOWN+UP 的亚帧点按亦走到这里）或电平兜底
    if (((key_event->touch_edge & TOUCH_EDGE_UP) || !key_event->is_touching) && s_app_menu_scroll.touch_active) {
        s_app_menu_scroll.touch_active = 0;
        if (s_app_menu_scroll.touch_is_dragging) {
            // 构成拖动：松手按速度启动惯性（手指停顿过久则速度作废）
            uint64_t now = get_timestamp_in_ms();
            float v0 = (now - s_app_menu_scroll.touch_track_ts <= 100) ? s_app_menu_scroll.touch_track_vel : 0.0f;
            if (v0 > UI_APP_MENU_FLING_VMAX) v0 = UI_APP_MENU_FLING_VMAX;
            if (v0 < -UI_APP_MENU_FLING_VMAX) v0 = -UI_APP_MENU_FLING_VMAX;
            if (v0 > UI_APP_MENU_FLING_VMIN || v0 < -UI_APP_MENU_FLING_VMIN) {
                s_app_menu_scroll.fling_velocity = v0;
                s_app_menu_scroll.fling_scroll_px = (float)s_app_menu_scroll.scroll_px;
                s_app_menu_scroll.fling_last_timestamp = now;
            }
        }
        else {
            // 点按命中判定：右侧固定区优先（列区不重叠）
            int32_t tx = s_app_menu_scroll.touch_start_x;
            int32_t ty = s_app_menu_scroll.touch_start_y;
            int32_t new_state = STATE_MAIN_MENU;
            if (tx >= g.fixed_x0 && tx < g.fixed_x0 + g.fixed_w && ty >= g.fixed_y0 && ty < g.fixed_y0 + g.fixed_h) {
                int32_t index = (ty - g.fixed_y0) / g.fixed_cell_h;
                if (index >= 0 && index < UI_APP_MENU_FIXED_NUM) {
                    new_state = menu_fixed_action(key_event, global_state, index);
                }
            }
            else if (tx >= g.scroll_x0 && tx < g.scroll_x0 + g.scroll_w
                     && ty >= g.scroll_y0 && ty < g.scroll_y0 + g.scroll_h) {
                // 行号需补偿滚动偏移（滚动位置非整行时，顶/底各有一个部分可见的格子）
                int32_t col = (tx - g.scroll_x0) / g.cell_w;
                int32_t row = (ty - g.scroll_y0 + s_app_menu_scroll.scroll_px) / g.scroll_cell_h;
                int32_t index = row * UI_APP_MENU_SCROLL_COLS + col;
                if (col >= 0 && col < UI_APP_MENU_SCROLL_COLS && index >= 0 && index < g.item_num) {
                    new_state = menu_activate_item(key_event, global_state, index);
                }
            }
            if (new_state != global_state->STATE) {
                global_state->STATE = new_state;
            }
        }
    }

    // ========================================================================
    // 实体键盘（触屏不派生键码，见 get_input_event）：
    // 数字键 1-9 对应条目表前 9 项（即内置应用：番茄表……控制台）；
    // 灵机 App 图标在表格后段，用触屏滚动/点按（或“小游戏→灵机应用程序”列表）。
    // Esc/Shift/Ctrl/Enter 为返回/设置/自述/关机；上/下键为滚动区整行滚动。
    // ========================================================================
    if (key_event->key_edge == -1 || key_event->key_edge == -2) {
        int32_t item_index = -1;
        switch (key_event->key_code) {
            case NANO_KEY_1: item_index = 0; break;
            case NANO_KEY_2: item_index = 1; break;
            case NANO_KEY_3: item_index = 2; break;
            case NANO_KEY_4: item_index = 3; break;
            case NANO_KEY_5: item_index = 4; break;
            case NANO_KEY_6: item_index = 5; break;
            case NANO_KEY_7: item_index = 6; break;
            case NANO_KEY_8: item_index = 7; break;
            case NANO_KEY_9: item_index = 8; break;
            case NANO_KEY_esc:   global_state->STATE = STATE_SPLASH_SCREEN; return;
            case NANO_KEY_shift: global_state->STATE = STATE_SETTING_MENU;  return;
            case NANO_KEY_ctrl:  global_state->STATE = STATE_README;        return;
            case NANO_KEY_enter: global_state->STATE = STATE_SHUTDOWN;      return;
            case NANO_KEY_up:
            case NANO_KEY_down: {
                int32_t step = g.scroll_cell_h;
                int32_t new_scroll_px = s_app_menu_scroll.scroll_px
                    + ((key_event->key_code == NANO_KEY_up) ? -step : step);
                if (new_scroll_px < 0) new_scroll_px = 0;
                if (new_scroll_px > g.max_scroll_px) new_scroll_px = g.max_scroll_px;
                if (new_scroll_px != s_app_menu_scroll.scroll_px) {
                    s_app_menu_scroll.scroll_px = new_scroll_px;
                    s_app_menu_scroll.fling_velocity = 0.0f;
                    ui_main_menu_draw(key_event, global_state, 0);
                }
                return;
            }
            default: return;
        }
        if (item_index >= 0 && item_index < g.item_num) {
            global_state->STATE = menu_activate_item(key_event, global_state, item_index);
        }
    }
}
