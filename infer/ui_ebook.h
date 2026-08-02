#ifndef __NANO_UI_EBOOK_H__
#define __NANO_UI_EBOOK_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 电子书
//
// - 菜单（复用全局单例 w_menu_main）：枚举 SD 卡根目录全部文件（非目录），
//   按字符串升序排列；选中即打开。
// - 阅读（复用全局单例 w_textarea_main）：文件按需分块读取——打开时预扫描全文，
//   按与 typeset_line_breaks 一致的折行规则计算每页（一屏 view_lines 行）的起始
//   字节偏移并存于 PSRAM；翻页/跳页时按偏移从文件读入文本控件缓冲区。
// - 底栏显示 当前页/总页数；右侧滚动条显示当前页在全文中的总进度。
//   ←/→ 逐行滚行（文本控件原生交互），4/6 上/下一页；按 C(Ctrl) 弹出“跳转到页”
//   模态框，数字键输入页码，D(回车)确认，A(ESC)取消；阅读中 A 返回文件菜单。
// ===============================================================================

int32_t ui_ebook_menu_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_ebook_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);

int32_t ui_ebook_open(Key_Event *key_event, Global_State *global_state, const char *path_mb);
void    ui_ebook_close(void);

int32_t ui_ebook_reading_render(Key_Event *key_event, Global_State *global_state);
int32_t ui_ebook_reading_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
