#ifndef __NANO_UI_H__
#define __NANO_UI_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <wchar.h>

#include "graphics.h"
#include "utils.h"
#include "ui_layout.h"


#define IME_MODE_HANZI    (0)
#define IME_MODE_ALPHABET (1)
#define IME_MODE_NUMBER   (2)

#define ALPHABET_COUNTDOWN_MS (500)
#define LONG_PRESS_THRESHOLD (360)

#define MAX_CANDIDATE_NUM (256)     // 候选字最大数量
#define MAX_CANDIDATE_PAGE_NUM (26) // 候选字最大分页数
#define MAX_CANDIDATE_NUM_PER_PAGE (10) // 每页最多有几个候选字（每页10个字）

struct Nano_Context;
struct Nano_Session;
struct Linglong_Config;

typedef struct Nano_Context Nano_Context;
typedef struct Nano_Session Nano_Session;

typedef struct Widget_Textarea_State Widget_Textarea_State;
typedef struct Widget_Input_State Widget_Input_State;
typedef struct Widget_Menu_State Widget_Menu_State;
typedef struct Linglong_Config Linglong_Config;

// NOTE 增删字段时，务必修改初始化部分
typedef struct Global_State {
    // gfx对象
    Nano_GFX *gfx;

    // UI组件（指向它们的指针），目前暂且硬编码
    Widget_Textarea_State  *w_textarea_main;

    Widget_Input_State     *w_input_main;

    Widget_Menu_State      *w_menu_main; // 全局唯一菜单实例（各菜单场景互斥，进入时重新初始化）

    // 全局状态
    int32_t STATE; // 当前状态
    int32_t PREV_STATE; // 上一状态

    // 全局通用信息
    uint64_t timestamp; // 物理时间戳（ms）
    uint64_t timestamp_last; // 上一次主循环的物理时间戳（ms），用于统计帧率、节流等用途
    volatile uint64_t last_touch_timestamp; // 最后一次触屏按下的物理时间戳（ms；0=从未触摸）。由 Core1 的 get_input_event 以 1-2ms 周期高频锁存（短按不遗漏），供九键按键提示遮罩在 Core0 渲染侧可靠触发（见 ui.c）
    // 触屏电平共享快照（与 last_touch_timestamp 同一机制）：由 Core1 的 get_input_event 高频锁存，
    // Core0 渲染任务每帧取用并覆盖到 key_event，供业务逻辑统一经 key_event 消费触屏，
    // 避免把高频电平样本刷入事件队列（挤占一次性按键/手势事件，见 linglong_m5core2.ino）
    volatile int32_t touch_x;       // 触点坐标（is_touching==1 时有效）
    volatile int32_t touch_y;
    volatile int32_t is_touching;   // 触屏电平：1-触摸中，0-未触摸
    int32_t year;
    int32_t month;
    int32_t day;
    int32_t hour;
    int32_t minute;
    int32_t second;
    int32_t millisecond;
    float timezone;
    float longitude;
    float latitude;
    int32_t timer; // 主循环计数器：从0开始递增，不与物理时间关联
    int32_t focus;
    int32_t is_ctrl_enabled; // 是否处于Ctrl键的激活状态：1-是，0-否
    int32_t ui_color_style;
    uint32_t ui_font; // UI 文本字体（GFX_FONT_*，见 graphics.h）。默认 0 = GFX_FONT_BITMAP_12（12px 二值点阵）

    // LLM相关
    Nano_Context *llm_ctx;
    Nano_Session *llm_session; // LLM一轮对话状态
    int32_t llm_status; // LLM推理状态
    wchar_t *llm_model_name;
    int32_t llm_is_thinking_model;
    char *llm_model_path;
    char *llm_lora_path;
    float llm_repetition_penalty;
    float llm_temperature;
    float llm_top_p;
    uint32_t llm_top_k;
    uint32_t llm_max_seq_len;
    int32_t is_thinking_enabled; // 是否开启思考模式：1-是，0-否
    wchar_t *llm_output_of_last_session;
    float tps_of_last_session;
    int32_t token_num_of_last_session;
    int32_t llm_enable_observation; // 是否启用推理状态观测

    // ASR相关
    wchar_t *asr_output_buffer;
    int32_t is_auto_submit_after_asr; // ASR结束后立刻提交识别内容到LLM？默认值1（不编辑，立刻提交）
    int32_t is_asr_server_up;
    int32_t is_recording; // 录音状态
    uint64_t asr_start_timestamp; // 录音起始的时间戳

    // TTS相关
    int32_t tts_req_mode; // TTS请求方式：0-关闭（默认值）   1-实时（每生成“一句话”立刻请求TTS）   2-全部生成完成后统一请求TTS

    // IMU相关
    float pitch;
    float roll;
    float yaw;
    float imu_temperature;

    // UPS相关
    int32_t ups_is_charging;
    int32_t ups_voltage; // UPS电压
    int32_t ups_current;
    int32_t ups_soc; // UPS电量

    // 显示相关
    int32_t is_full_refresh; // 作为所有绘制函数的一个参数，用于控制是否整帧刷新。默认为1。0-禁用函数内的clear-refresh，1-启用函数内的clear-refresh
    uint32_t llm_refresh_max_fps; // 设置项：LLM推理过程中屏幕刷新的最高帧率
    uint64_t llm_refresh_timestamp; // LLM推理过程中，上一次刷新屏幕的时间戳。用于控制刷新频率（不高于llm_refresh_max_fps），避免刷新过于频繁，拖累表观TPS（目前LLM推理与屏幕刷新是同步串行的）。
    int32_t brightness; // 屏幕亮度
    int32_t volume;     // 全局主音量（0~255；影响按键音、寻呼机OFDM发射音量、音乐盒初始音量；音乐盒内部调节不回写）
    int32_t auto_shutdown_minutes;   // 自动关机时长设置（分钟；0=关，可选 1/2/3/5/10/20/30/60）
    uint64_t auto_shutdown_deadline; // 自动关机到期时间戳（ms，对 timestamp；0=未启用）
    int32_t ime_hint_timeout_s;      // 九键按键提示遮罩显示时长设置（秒）：0=关闭，可选 0/3/6，默认 3（系统设置中循环切换）
    int32_t key_feedback_mode;       // 按键提示（按键反馈方式）设置：0=无，1=灯光，2=蜂鸣，3=灯光+蜂鸣；默认 1（灯光）（系统设置中循环切换）

    // 玲珑天象仪全局配置
    Linglong_Config *linglong_cfg;
    Nano_GFX *llgfx; // 玲珑仪专用gfx

    // BadApple相关
    uint32_t ba_frame_count;
    uint64_t ba_begin_timestamp;

} Global_State;

// 触屏滑动手势方向（通用）：事件层不做手势识别，由消费者用 UI_Swipe_Tracker
// 从 key_event 的触屏流逐帧自行解释（语义归消费者，如文本输入控件的上滑呼出软键盘）
#define NANO_TOUCH_GESTURE_NONE        (0)
#define NANO_TOUCH_GESTURE_SWIPE_UP    (1)
#define NANO_TOUCH_GESTURE_SWIPE_DOWN  (-1)

// 垂直滑动手势跟踪器（通用解释器）：消费者每帧喂入一次触屏样本（取自 key_event），
// 松手确认后返回手势方向；触摸期间可查询任意方向上的累计位移（"滑动中"判定用）
typedef struct UI_Swipe_Tracker {
    int32_t active;  // 正在跟踪一次触摸序列
    int32_t start_y; // 序列起点y
    int32_t min_y;   // 序列中的最小y（最高点）
    int32_t max_y;   // 序列中的最大y（最低点）
} UI_Swipe_Tracker;

void ui_swipe_tracker_init(UI_Swipe_Tracker *tracker);
// 喂入一帧触屏样本：松手时垂直位移跨越 confirm_px 即返回对应方向手势
// （NANO_TOUCH_GESTURE_SWIPE_UP/DOWN），否则返回 NANO_TOUCH_GESTURE_NONE
int8_t ui_swipe_tracker_feed(UI_Swipe_Tracker *tracker, int32_t is_touching, int32_t touch_y, int32_t confirm_px);
// 当前跟踪序列在指定方向（NANO_TOUCH_GESTURE_SWIPE_UP/DOWN）上的累计位移（px）
int32_t ui_swipe_tracker_displacement(const UI_Swipe_Tracker *tracker, int8_t direction);

typedef struct Key_Event {
    int32_t  event_type; // 事件类型
    int32_t  touch_x;       // 触点坐标（像素；is_touching==1 时有效）
    int32_t  touch_y;
    int32_t  is_touching;   // 触屏电平：1-正在触摸，0-未触摸

    uint8_t  prev_key;   // 上一次按键的键值
    uint8_t  key_code;   // 大于等于16为没有任何按键，0-15为按键
    int8_t   key_edge;   // 0：松开  1：上升沿  -1：下降沿(短按结束)  -2：下降沿(长按结束)
    uint64_t key_timer;  // 按下状态的计时器
    uint8_t  key_mask;   // 长按超时后，键盘软复位标记。此时虽然物理上依然按键，只要软复位标记为1，则认为是无按键，无论是边沿还是按住都不触发。直到物理按键松开后，软复位标记清0。
    uint8_t  key_repeat; // 触发一次长按后，只要不松手，该标记置1，直到物理按键松开后置0。若该标记为1，则在按住时触发连续重复动作。
    uint8_t  is_softkbd; // 本事件是否来自触屏软键盘：1-是（键码为直接键码，不再经过九键输入法），0-否（触屏4x4网格键）
    uint8_t  is_soft_key; // 按键来源：1-触屏派生的软按键（4x4宫格映射或触屏软键盘），0-实体键盘（见 platform.h NANO_HAS_HW_KEYBOARD）。按下时锁存，下降沿事件沿用
    // 触屏边沿事件（触屏事件队列改造，见 AGENTS.md 第八节）：
    // 边沿检测在生产端（Core1 get_input_event，1-2ms 轮询）完成，DOWN/UP 经 event_queue
    // 可靠投递，Core0 每帧排空合并为位掩码——亚帧短点按不再湮灭。
    // 队列中触屏事件的身份标识：key_code == NANO_KEY_IDLE 且 touch_edge != 0。
    uint8_t  touch_edge;    // 触屏边沿位掩码（TOUCH_EDGE_DOWN/TOUCH_EDGE_UP，同帧可叠加），无边沿为 0
    int32_t  touch_down_x;  // 本次触摸序列按下点坐标（触摸期间及 UP 事件时有效）
    int32_t  touch_down_y;
} Key_Event;

// 触屏边沿位掩码（Key_Event.touch_edge）
#define TOUCH_EDGE_DOWN (1) // 按下沿
#define TOUCH_EDGE_UP   (2) // 松开沿

typedef struct Widget_Textarea_State {
    int32_t state;
    int32_t x;
    int32_t y; // NOTE 设置文本框高度时，按照当前字体行高（gfx_font_line_height）来计算。例如，使用默认12px点阵字体（行高13）时，如果希望恰好显示4行，则高度应为13*3+12=51px。
    int32_t width;
    int32_t height;
    wchar_t *text;
    uint32_t *style; // 逐字符样式，MSB-0xXXRRGGBB-LSB。最高位为1代表格式控制标记的字符，渲染时忽略。
    int32_t length;
    int32_t *break_pos;
    int32_t line_num;
    int32_t view_lines;
    int32_t view_start_pos;
    int32_t view_end_pos;
    int32_t current_line;
    int32_t is_show_scroll_bar; // 是否显示滚动条：0不显示 1显示
    int32_t is_modified; // 文本内容是否有修改过？默认1。用于控制是否进行typeset_line_breaks排版

    // 像素级连续滚动（见 AGENTS.md 第九节）：
    // 不变量 scroll_px = current_line * line_height + scroll_sub_offset（行高恒定）。
    // current_line 保留为对外行粒度接口；整行路径（按键/外部直写）改写 current_line 时须将 sub 归零。
    int32_t scroll_sub_offset;      // 亚行滚动偏移（px，∈ [0, line_height)）
    // 触屏交互状态（一次触摸序列的跟踪，与菜单控件同范式；见 ui_widget_textarea_touch_handler）
    int32_t touch_active;           // 1-正在跟踪一次触摸序列
    int32_t touch_is_dragging;      // 1-本序列已构成拖动（位移越阈值），松手时不按点按处理
    int32_t touch_start_x;          // 序列起点坐标（点按判定用）
    int32_t touch_start_y;
    int32_t touch_anchor_scroll_px; // 按下时的像素级滚动位置（拖动锚点）
    int32_t touch_track_scroll;     // 上一采样帧的 scroll_px
    uint64_t touch_track_ts;        // 上一采样帧的时间戳（ms）
    float    touch_track_vel;       // 平滑后的拖动速度（px/s）
    float    fling_velocity;        // 松手惯性速度（px/s；0=无惯性动画）
    float    fling_scroll_px;       // 惯性动画中的浮点滚动位置
    uint64_t fling_last_timestamp;  // 上一动画帧的时间戳（ms）
} Widget_Textarea_State;

typedef struct Widget_Input_State {
    // 继承 Widget_Textarea_State
    Widget_Textarea_State textarea;

    // 以下是Widget_Input_State独有的

    int32_t state;                // 控件状态（不复用textarea内部的状态）
    int32_t cursor_pos;           // 光标位置
    int32_t desired_x;            // 上下移动光标时希望保持的视觉x偏移（px，-1表示无效；左右移动/增删字符时重置）
    uint32_t ime_mode_flag;       // 汉英数输入模式标志 0汉字 1英文 2数字
    uint32_t pinyin_keys;         // 单字拼音键码暂存
    // 候选字翻页相关
    uint32_t candidates[MAX_CANDIDATE_NUM]; // 全部候选字/符号
    uint32_t candidate_num;       // 候选字总数
    uint32_t candidate_pages[MAX_CANDIDATE_PAGE_NUM][MAX_CANDIDATE_NUM_PER_PAGE]; // 候选字分页
    uint32_t candidate_page_num;  // 总的候选字分页数
    uint32_t current_page;        // 当前显示的候选字页标号
    // 英文字母输入模式的倒计时
    uint64_t alphabet_click_timestamp; // 按键时刻的时间戳，用于计算倒计时进度条
    int32_t alphabet_is_counting_down; // 1-正在倒计时；0-不在倒计时
    uint32_t alphabet_index;
    uint8_t alphabet_current_key;      // 当前选中的字母按键
    // 杂项
    wchar_t *title_text;  // 顶部标题
    // 触屏交互（见 AGENTS.md 第九节）：
    int32_t grid16_mode;            // 十六键输入模式：1-触屏点按=宫格软按键（旧行为，供九键打字）；
                                    // 0-点按=光标定位、滑动=像素滚动（默认）。页脚 [16键] 热点切换
    uint64_t softkey_swallow_until; // 热点动作后吞掉宫格残留软按键的截止时间戳（ms；范式同 ui_calendar）
    int32_t drawn_cursor_pos;       // 上次绘制时的光标位置：光标跟随滚动仅在光标变化时触发，
                                    // 避免触屏手动滚动后被光标跟随拉回（初始 -2 强制首次跟随）
} Widget_Input_State;

typedef struct Widget_Menu_State {
    int32_t x;
    int32_t y;
    int32_t zindex;
    int32_t width;
    int32_t height;
    int32_t header_height; // 页眉高度（ui_widget_menu_init 默认为字体行高的若干倍；密集列表界面如词典可覆写）
    int32_t item_height;   // 条目行高（同上，默认字体行高的若干倍，文字在行内纵向居中）
    int32_t current_item_index; // 当前选中（高亮）的条目的标号（注意：选中条目不一定在显示的页面范围内）
    int32_t first_item_intex; // 当前页面显示的第一个条目的标号
    int32_t item_num; // 菜单条目数
    int32_t items_per_page; // 每页容纳的条目数
    const wchar_t *title;   // 菜单标题（借用调用方字符串，不复制；调用方需保证生命周期）
    const wchar_t **items;  // 条目字符串表（借用调用方存储，不复制；调用方需保证生命周期）
    // 触屏交互状态（一次触摸序列的跟踪；见 ui_widget_menu_event_handler）
    // 像素级滚动位置不变量：scroll_px = first_item_intex * item_height + scroll_sub_offset，
    // 拖动期间以像素更新后拆回两者；按键导航等整行路径改写 first_item_intex 时
    // 须同步将 scroll_sub_offset 归零（吸附回整行）。
    int32_t touch_active;           // 1-正在跟踪一次触摸序列
    int32_t touch_is_dragging;      // 1-本序列已构成拖动（累计位移越阈值），松手时不按点击处理
    int32_t touch_start_x;          // 序列起点坐标（点击判定用）
    int32_t touch_start_y;
    int32_t touch_anchor_scroll_px; // 按下时的像素级滚动位置 scroll_px（拖动滚动的锚点）
    int32_t scroll_sub_offset;      // 亚行滚动偏移（px，∈ [0, item_height)）；按键导航时归零
    // 拖动速度采样（松手惯性初速度估算用；按下时复位，拖动帧指数平滑更新）
    int32_t  touch_track_scroll;    // 上一采样帧的 scroll_px
    uint64_t touch_track_ts;        // 上一采样帧的时间戳（ms）
    float    touch_track_vel;       // 平滑后的拖动速度（px/s，>0 表示 scroll_px 增大/内容上移）
    // 松手惯性滚动（fling）：松手时以 touch_track_vel 为初速度启动，handler 每帧
    // 按线性减速度衰减推进，越界/速度归零即停；任意新触摸或按键立即终止动画
    float    fling_velocity;        // 当前惯性速度（px/s；0=无惯性动画）
    float    fling_scroll_px;       // 动画中的浮点滚动位置（保留亚像素，避免逐帧取整损耗）
    uint64_t fling_last_timestamp;  // 上一动画帧的时间戳（ms）
    // 到顶/到底碰撞回弹（overscroll bounce，2026-08）：视觉位移与逻辑滚动位置解耦——
    // bounce 只作绘制端位移（y_pos 附加），first_item_intex/scroll_sub_offset 始终钳在
    // 合法范围，点击命中/Enter 钳制/按键导航零感知。拖动过界为橡皮筋（手指驱动，
    // bounce_velocity=0），松手过界/惯性撞边转为弹簧动画（bounce_velocity 驱动）；
    // 静止时两者恒为 0。与 fling 互斥：fling 撞边即终止并移交弹簧动画。
    float    bounce_offset_px;      // 回弹位移（px；>0=顶端下拉内容下移，<0=底端上拉）
    float    bounce_velocity;       // 回弹弹簧速度（px/s）
    uint64_t bounce_last_timestamp; // 上一回弹动画帧的时间戳（ms）
} Widget_Menu_State;


void ui_draw_header(Key_Event *key_event, Global_State *global_state, wchar_t *text, int32_t is_center);
// 指定高度的页眉绘制（标准页眉为字体行高+1；菜单控件页眉为字体行高的若干倍，见 ui_widget_menu_init）
void ui_draw_header_ex(Key_Event *key_event, Global_State *global_state, wchar_t *text, int32_t is_center, int32_t header_height);
// 页眉完整绘制：底色 + 标题 + 可选左/右侧文本（“返回”等标签作为页眉固有部分，随页眉同时机绘制）。
// 左右侧文本为 12px 抗锯齿、页眉带内垂直居中、页眉文字同色；左侧自 x=0 左对齐，右侧自右缘右对齐
//（尾随空格可作右边距）；传 NULL 或空串表示该侧不绘制。
void ui_draw_header_full(Key_Event *key_event, Global_State *global_state, wchar_t *title, int32_t is_center,
    int32_t header_height, wchar_t *left_text, wchar_t *right_text);
// 仅按需更新页眉左/右侧文本（不重绘标题；各自区域先按页眉底色回填再绘制，自清洁）。
// 典型用途：电子书页眉左侧页码的随页更新。
void ui_draw_header_side_text(Key_Event *key_event, Global_State *global_state, int32_t header_height,
    wchar_t *left_text, wchar_t *right_text);
void ui_draw_footer(Key_Event *key_event, Global_State *global_state, wchar_t *text, int32_t is_center);
// 软按键提示区页脚：4个字符串依次为十六宫格最底部一行 *、0、#、D 四键的功能提示，
// 横向与底部4个格子中点对齐，纵向与 ui_draw_footer 一致；NULL或空串表示该键无功能
void ui_draw_footer_softkeys(Key_Event *key_event, Global_State *global_state,
    wchar_t *text_key_left, wchar_t *text_key_0, wchar_t *text_key_right, wchar_t *text_key_enter);

// font_id: 文本字体（GFX_FONT_*），决定行高、逐字符宽度、基线与渲染方式
void ui_draw_text_block(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state, uint32_t font_id);

// 排版-折行：按当前字体（global_state->ui_font）逐字符实际宽度计算断行位置
void typeset_line_breaks(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state);
// 排版-视口：line_height 为当前字体行高（gfx_font_line_height）
void typeset_view_range(Widget_Textarea_State *textarea_state, int32_t line_height);



void ui_widget_textarea_init(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state,
    uint32_t max_len);
void ui_widget_textarea_set(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state,
    wchar_t *text, int32_t current_line, int32_t is_show_scroll_bar);
void ui_widget_textarea_draw(Key_Event *key_event, Global_State *global_state, Widget_Textarea_State *textarea_state);
int32_t ui_widget_textarea_event_handler(
    Key_Event *ke, Global_State *gs, Widget_Textarea_State *ts,
    int32_t prev_focus_state, int32_t current_focus_state
);

// 文本框触屏手势机（像素级连续滚动，与菜单控件同范式；认 touch_edge 边沿事件，电平兜底）：
//   DOWN 锚定 / 拖动跟手（钳制不回绕）/ UP 启动惯性或判定点按；序列起点须在文本区内才激活。
// 返回值：0-无触屏活动；1-滚动活动（拖动/惯性进行中，调用方应重绘并消费本帧）；2-点按（松手且未构成拖动，
//          按下点坐标取 ke->touch_down_x/y，供调用方做命中判定）
int32_t ui_widget_textarea_touch_handler(Key_Event *ke, Global_State *gs, Widget_Textarea_State *ts);

// 点按命中测试：像素坐标 → 光标槽位 s（0..length，即光标左侧字符数，cursor_pos = s - 1）；
// 未命中（坐标在文本区外或文本为空）返回 -1。调用前须保证排版（break_pos）为最新。
int32_t ui_widget_textarea_char_index_at(Key_Event *ke, Global_State *gs, Widget_Textarea_State *ts,
    int32_t px, int32_t py);

// “返回”软按钮热区左界（页眉最右侧 1/4，与菜单控件约定一致）
#define UI_BACK_HOTSPOT_X0(gfx_width) ((gfx_width) * 3 / 4)

// 标准页眉（顶栏）高度：1.5 倍字体行高（与菜单控件页眉 ui_widget_menu_init 数值一致）
static inline int32_t ui_std_header_height(uint32_t font_id) {
    return gfx_font_line_height(font_id) * 3 / 2;
}

void ui_widget_input_init(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state, wchar_t *title_text);
void ui_widget_input_refresh(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);
// 九键按键提示遮罩外部开关：1=启用机制（默认），0=关闭机制（立即解除已激活的遮罩并禁止触发）。
// 软键盘启用时应由上层关闭本机制，避免遮罩干扰软键盘（见 ui_app.c 软键盘显隐切换处）。
void ui_ime_hint_mask_set_enabled(int32_t enabled);
// 切换触屏软键盘显隐（文本输入控件固有功能，供 Ctrl+0 组合键与上滑/下滑手势调用）：
// 联动按键提示遮罩开关、全键盘拼音组字重置，并重新布局文本区为键盘让出/恢复空间。
void ui_widget_input_toggle_softkbd(Key_Event *key_event, Global_State *global_state);
// 在文本框的光标位置之后插入/删除一个字符（触屏软键盘及其拼音输入法也会调用）
void insert_char(Widget_Input_State *input_state, wchar_t new_char);
void delete_char(Widget_Input_State *input_state);
int32_t ui_widget_input_event_handler(
    Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state,
    int32_t prev_focus_state, int32_t current_focus_state, int32_t next_focus_state
);

void ui_widget_menu_init(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state);
void ui_widget_menu_refresh(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state);
void ui_widget_menu_draw(Key_Event *key_event, Global_State *global_state, Widget_Menu_State *menu_state);
int32_t ui_widget_menu_event_handler(
    Key_Event *ke, Global_State *gs, Widget_Menu_State *ms,
    int32_t (*menu_item_action_callback)(Key_Event*, Global_State*, Widget_Menu_State*), int32_t prev_focus_state, int32_t current_focus_state
);

void ui_draw_input_buffer(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);
void ui_draw_input_cursor(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);
void ui_draw_input_pinyin(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state, uint32_t is_picking);
void ui_draw_input_symbol(Key_Event *key_event, Global_State *global_state, Widget_Input_State *input_state);

void ui_draw_scroll_bar(Key_Event *key_event, Global_State *global_state, int32_t current_line, int32_t line_num, int32_t view_lines, int32_t x, int32_t y, int32_t width, int32_t height);


// ===============================================================================
// 七段码
// ===============================================================================

void ui_draw_7seg_string(
    Key_Event *key_event, Global_State *global_state,
    int32_t xx, int32_t yy, wchar_t *text,
    uint8_t red, uint8_t green, uint8_t blue,
    float seg_length, float seg_thickness, float digit_gap, int32_t is_shadow,
    int32_t *text_width, int32_t *text_height
);

// 预计算七段码字符串的渲染宽高（不做实际渲染，纯几何计算无需上下文），
// 供实际绘制前计算布局参数（如居中、右对齐等）
void ui_measure_7seg_string(
    wchar_t *text,
    float seg_length, float seg_thickness, float digit_gap,
    int32_t *text_width, int32_t *text_height
);

// 以 (cx, cy) 为中心绘制七段码字符串
void ui_draw_7seg_string_centered(
    Key_Event *key_event, Global_State *global_state,
    int32_t cx, int32_t cy, wchar_t *text,
    uint8_t red, uint8_t green, uint8_t blue,
    float seg_length, float seg_thickness, float digit_gap, int32_t is_shadow,
    int32_t *text_width, int32_t *text_height
);


#ifdef __cplusplus
}
#endif

#endif
