#ifndef __NANO_UI_DICT_H__
#define __NANO_UI_DICT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"
#include "graphics.h"
#include "ui.h"
#include "ui_softkbd.h"

// ===============================================================================
// 英汉电子词典（硬件无关模块，宿主可测）
//
// 数据通路：SD 卡 /dict.csv（word,phonetic,definition 三列 CSV，引号状态机解析）
//   → 首次/指纹失效时设备端构建二进制索引 /dict.idx（含进度显示）
//   → 进入词典时索引整体载入 PSRAM（词条池+定长索引项，前缀二分零 SD 访问）
//   → 详情按索引项中的文件偏移 seek 读取单条记录。
//
// 查询界面（STATE_DICT_QUERY 分支驱动）：header + 固定一行查询前缀 + 候选词条菜单
//   （复用 w_menu_main）+ 固定显示的触屏软键盘（无呼出/收起逻辑）；字体统一
//   GFX_FONT_ALPHA_12（进入时切换 global_state->ui_font，退出/进详情时恢复）。
// 详情界面（STATE_DICT_DETAIL 分支驱动）：复用 w_textarea_main，词条 #66ccff、
//   音标与释义原样（引号转义已还原）；←/→ 候选表内上下词条、4/6 滚行、A/D 返回。
//
// 本模块不依赖 ui_app.h：状态码一律由调用方以参数注入（同通用控件惯例），
// 以便宿主机自测（提供 platform_file_* 等打桩即可驱动核心逻辑）。
// ===============================================================================

// 进入词典：确保索引就绪（缺失/失效则带进度构建）并载入 PSRAM、申请候选缓冲、
// 切换 ui_font 为 GFX_FONT_ALPHA_12、固定显示软键盘。返回 0 成功（调用方据此切换到
// 查询状态）；负值失败（已显示错误画面，调用方应停留在原状态）。
int32_t ui_dict_enter(Key_Event *key_event, Global_State *global_state);

// 退出词典：释放索引与候选缓冲、关闭 CSV、收起软键盘、恢复遮罩开关与 ui_font
void ui_dict_exit(Key_Event *key_event, Global_State *global_state);

// 查询状态事件处理（每帧调用一次），返回新状态码：
// 默认返回 query_state；退出（前缀为空时按A）返回 main_menu_state（已调用 ui_dict_exit）；
// 选中词条（D/ENT）返回 detail_state。
int32_t ui_dict_query_event(Key_Event *key_event, Global_State *global_state,
    int32_t main_menu_state, int32_t query_state, int32_t detail_state);

// 详情状态事件处理（每帧调用一次），返回新状态码：
// 默认返回 detail_state；A/D 返回 query_state（查询现场保留并全量重绘）。
int32_t ui_dict_detail_event(Key_Event *key_event, Global_State *global_state,
    int32_t query_state, int32_t detail_state);

#ifdef __cplusplus
}
#endif

#endif
