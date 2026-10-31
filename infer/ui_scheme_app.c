#include <stdio.h>
#include <string.h>
#include <wchar.h>

#include "ui_scheme_app.h"

#include "graphics.h"
#include "hal_key.h"
#include "hal_fs.h"
#include "platform.h"

#include "animac.h"

#include "am_host.h"

#include "am_native_System.h"
#include "am_native_Math.h"
#include "am_native_String.h"
#include "am_native_Table.h"
#include "am_native_Gfx.h"
#include "am_native_Time.h"
#include "am_native_Diag.h"
#include "am_native_Touch.h"
#include "am_native_Imu.h"
#include "am_native_Key.h"
#include "am_native_Fs.h"
#include "am_native_Ui.h"
#include "am_native_App.h"

// ===============================================================================
// 运行时参数
// ===============================================================================

// 解释器内存池：期望大小、最小可接受大小、自适应申请时留出的安全余量
//（策略与 ui_animac 一致；WASM/桌面平台堆充裕，实际按期望值分配）
// 由 2MB 逐级上调为 12MB：
//   · 4MB：sky3.scm（scatter_model_3 逐行移植）
//   · 8MB：ephemeris.scm（月球 10×60 项级数 + VSOP87 + 星历函数集）
//   · 12MB：nongli.scm + app/lib/ephem.scm（共享星历模块 + 农历年表推算）
// 编译期的 VM 工作区占用远大于运行期堆占用（后者极小）。
// 内存受限宿主：pool_size 按「最大连续块 - 余量」自适应下调（见下），
// 小 App 仍可正常运行；装不下的大 App 会显示「内存不足」错误画面。
// ESP32（M5Core2/CoreS3，8MB PSRAM）：12MB 期望值超过整片 PSRAM，「最大连续块-余量」
// 自适应会把池子撑到数 MB，几乎吃光 PSRAM 并饿死音频/词典/电子书等其他运行时消费者。
// 故 ESP32 的期望值与 Animac 控制台（ui_animac.h，ESP32 验证值）对齐为 2MB，
#if NANO_IS_M5ESP
    #define MB_POOL_SIZE   (2ULL * 1024 * 1024)
#else
    #define MB_POOL_SIZE   (12ULL * 1024 * 1024)
#endif
#define MB_POOL_MIN    (512ULL * 1024)
#define MB_POOL_MARGIN (64ULL * 1024)

// 每帧分配给解释器的时间片（VM指令条数）：
// WASM(桌面浏览器)下一帧约1~3ms，GUI帧率不受影响（时间片耗尽即归还控制权）；
// 演示类 App 的节奏建议在源码内用 System.set_interval 节流。
//
// 2026-10 同步上游 0f5e9d4 时由 250000 上调为 312500（+25%）：
// 上游自 bbf3333 起改用“编译期单值栈纪律”，每个表达式在语句位置多出 pop/push，
// 单位 App 工作的 VM 指令数增加约 24%（实测 mandelbrot 完成所需 120 帧推进量：
// 旧 250000 → 174 次推屏，新 312500 → 175 次推屏；每帧 CPU 时间 5.3ms → 5.6ms）。
// 上调后“每帧 App 推进量”与同步前持平，避免渐进式渲染类 App 观感变慢。
// 若需回退，改回 250000 即可（只影响每帧推进量，不影响正确性）。
#define MB_TIMESLICE_TICKS (312500u)

// 源码大小防御上限（字节）：正常 App 源码几 KB，超限视为异常文件拒绝加载
#define MB_SOURCE_MAX_BYTES (256 * 1024)

// 运行期错误信息显示缓冲（宽字符）
#define MB_ERR_BUF_LEN (128)

// App 数量上限与名称长度上限
#define SCHEME_APP_MAX     (48)   // app/*.scm 上限（当前 33 个；余量留 15 个）
// 2026-11 由 32 上调为 48：沙盒颗粒物理/2048/弹幕陨石等 App 陆续加入后，
// app/ 下的 .scm 已达 33 个 —— 32 的上限会把按名排序**最后**的 App 悄悄丢掉
// （当时是 xray.scm，列表里凭空少一项），且没有任何提示。改动 App 数量时务必
// 核对这里与 browser/regress 的 App 清单。
#define SCHEME_APP_NAME_MAX (48)

// ===============================================================================
// App 描述与扫描结果
// ===============================================================================

typedef struct {
    wchar_t title[SCHEME_APP_NAME_MAX];  // 菜单显示名（文件名去 .scm）
    char    source_rel[192];             // 源码路径（相对解释器工作根，UTF-8）
} Scheme_App_Desc;

// 扫描结果与菜单条目表：主堆（ESP32 上为 PSRAM）进入功能时一次性分配，约 9.3KB。
// 此前为静态 bss，常驻内部 DRAM——Core2 内部 RAM 紧张，曾致开机 DMA 帧缓冲
//（2×75KB，见 linglong_m5core2.ino）分配失败（"Failed to alloc frame buffers!"）
static Scheme_App_Desc *s_apps = NULL;                  // 扫描结果（进入列表菜单时刷新）
static int32_t scheme_app_buffers_init(void);           // 一次性分配（定义见扫描区块前）
static int32_t s_app_count = 0;                 // 有效 App 数
static int32_t s_selected = -1;                 // 运行中的 App 下标（-1 无）
static int32_t s_list_empty = 0;                // app 目录下没有 .scm（菜单显示占位项）

// 「在途触摸序列」余波抑制（退出 App → 回列表时置位，序列结束后自动解除）：
//   退出可能是**在用户手指动作过程中**发生的（点 App 自己的退出按钮 / 点全局确认框的
//   「确认」）：App 侧的 (native Touch) 由电平派生边沿，往往先于宿主输入层判定到"松手"
//   （宿主有 UP 去抖），于是 App 已销毁、宿主仍认为"按住中"。此时列表菜单被初始化，
//   它的电平兜底（`is_touching && !touch_active`）会把这一拍的旧电平当成**新的按下**，
//   紧接着迟到的 UP 边沿就成了一次"点击"，落到某个菜单项上——症状：确认退出后
//   立刻自动进入了列表里被点位置的 App（实测从俄罗斯方块退出会直接启动 gol）。
//   处置：进入列表后的这一小段里剥掉触摸字段（按键照常），序列结束后恢复。
//   （与 ui_app.c 输入原则第 4 条同一精神：状态切换不得继承在途触摸序列。）
static int32_t s_list_touch_guard = 0;
static uint64_t s_list_touch_guard_until_ms = 0;  // 抑制窗口（超时兜底解除，防止永久吞触摸）

// 菜单条目文本（供 w_menu_main->items 借用）与指针表
static wchar_t (*s_item_text)[SCHEME_APP_NAME_MAX] = NULL;
static const wchar_t **s_item_ptrs = NULL;

// ===============================================================================
// 应用实例表（P4：单前台 + N 个后台）
//
// 模型：
//   · 一个 App 被"切后台"时**不销毁**解释器运行时与内存池，只是不再泵它
//     （suspend：宿主不调 am_runtime_event_handler、不渲染，定时器随之停摆）；
//   · 列表里该 App 带「● 运行中」角标，选中即 **resume**（同一个运行时接着跑，
//     App 内的状态与定时器都还在），而不是重新冷启动；
//   · 槽位上限 SCHEME_APP_INST_MAX：满了就按"最久未用"驱逐一个**后台**实例
//     （前台永不驱逐）；ESP32 内部 RAM 紧张，取 1（= 没有后台能力，hide 退化为退出）。
//   · 实例的身份用**源码相对路径**（source_rel）标识，不用 s_apps[] 指针——列表每次
//     进入都会重扫描并复用同一块 s_apps 缓冲，指针会失效。
// ===============================================================================

#if NANO_IS_M5ESP
    #define SCHEME_APP_INST_MAX (1)   // 只有单一实例：hide 退化为退出（内部 RAM 不够）
#else
    #define SCHEME_APP_INST_MAX (3)   // 前台 1 + 后台 2（每实例一个解释器池）
#endif

typedef struct {
    int32_t used;                     // 槽是否被占用
    wchar_t title[SCHEME_APP_NAME_MAX];  // 归属 App 的显示名（拷贝，跨重扫描稳定）
    char    source_rel[192];             // 归属 App 的源码相对路径（身份键）
    am_allocator_pool_t *pool;
    am_allocator_t *vm_alloc;
    am_allocator_t *heap_alloc;
    am_runtime_t *rt;
    am_pid_t pid;
    size_t pool_size_bytes;           // 实际分配的解释器池大小（错误提示用）
    int32_t alive;                    // 运行时是否就绪
    int32_t load_failed;              // 构建/加载失败（显示错误画面，等待退出）
    wchar_t err_load[MB_ERR_BUF_LEN]; // 加载失败原因（文件缺失/过大/内存不足等）
    wchar_t err_buf[MB_ERR_BUF_LEN];  // 最近一条运行期错误（虚拟机错误FIFO排水）
    int32_t err_buf_shown;            // 运行期错误已叠加显示（只提示一次）
    wchar_t *code_w;                  // 源码宽字符串（解释器编译后仍保留，退出时释放）
    int32_t foreground;               // 1=前台（唯一）；0=后台（冻结）
    int32_t suspended;                // 1=被宿主挂起（切后台或锁屏），(App.is_active)=#f
    uint64_t last_active_ms;          // 前台/活跃时刻（LRU 驱逐依据）
} UI_Scheme_App_State;

static UI_Scheme_App_State s_inst[SCHEME_APP_INST_MAX];
static UI_Scheme_App_State *s_app = NULL;   // 指向**前台**实例；无前台时为 NULL

static UI_Scheme_App_State *app_instance_find(const char *source_rel);
static UI_Scheme_App_State *app_instance_alloc(int32_t desc_idx);
static void app_instance_destroy(UI_Scheme_App_State *inst);
static void app_instance_foreground(UI_Scheme_App_State *inst);
static void app_instance_suspend(UI_Scheme_App_State *inst);

// 模态文本输入/输出（本地库 Ui）宿主侧：状态与服务的定义在文件后半部分，
// 此处前置声明供 render_frame / on_exit 引用。
static int32_t s_modal_active = 0;   // 模态框是否在显示
// 长按重复抑制：事件层的长按（-2）在按住期间会**持续重复**（按键连发机制，见
// ui_app.c get_input_event），而模态框用「长按 A/Esc」做取消——若不拦截，同一次长按
// 会连续取消多个模态框、最后一路把 App 也退出。策略：模态框消费掉一个 -2 后置位，
// 其后相同长按的重复事件一律丢弃，直到看到新的按下沿（key_edge == 1）才解除。
static int32_t s_modal_ignore_longpress_repeat = 0;
// 「打开模态框的那次按键」的松开沿抑制：
//   按键边沿由生产端逐事件投递、消费者每帧只取一个，且生产端只投递**下降沿**
//   （-1 短按结束 / -2 长按结束），不投递上升沿。App 往往在吃掉按下沿的那一帧里发起
//   (Ui.input ...)，于是这次按键迟到的 -1 会落进刚打开的模态框，被控件当成"短按 Enter
//   提交"（症状：输入框一出现就自己交了空串，或消息框一闪而过）。
//   判据：打开时向 (native Key) 问「最近一次被取走的按下沿是哪个键、什么时候」——
//   若就在刚才（≤500ms），则丢弃**该键**接下来的第一条 -1/-2 边沿，其余事件照常。
static uint8_t  s_modal_open_key = 0;   // 打开模态框时"正在收尾"的键（0=无）
static uint64_t s_modal_open_key_deadline_ms = 0; // 只在这个时刻之前剔除（毫秒）
#define UI_MODAL_OPEN_KEY_WINDOW_MS (300)
static void scheme_modal_open(Key_Event *key_event, Global_State *global_state);

// 解释器工作目录宽前缀（带尾斜杠；链接器 import 解析基准，本类 App 无 import 仅备用）
static wchar_t s_app_root[128] = L"/";


// ===============================================================================
// 宿主时间函数适配（runtime vtable 要求的带 rt 引参签名，同 am_repl.c）
// ===============================================================================

static am_timestamp_t app_host_now_ms(am_runtime_t *rt) {
    (void)rt;
    return (am_timestamp_t)am_current_timestamp_in_ms();
}

static void app_host_sleep_in_ms(am_runtime_t *rt, am_timestamp_t ms) {
    (void)rt;
    (void)ms;
    // GUI 驱动模型下宿主永不主动睡眠：时间片耗尽即归还控制权，帧间由 GUI 调度
}

static void app_on_tick(am_runtime_t *rt) { (void)rt; }
static void app_on_event(am_runtime_t *rt) { (void)rt; }
static void app_on_halt(am_runtime_t *rt) { (void)rt; }
static void app_on_error(am_runtime_t *rt) { (void)rt; }

static const am_runtime_vtable_t APP_VTABLE = {
    .on_tick = app_on_tick,
    .on_event = app_on_event,
    .on_halt = app_on_halt,
    .on_error = app_on_error,
    .sleep_in_ms = app_host_sleep_in_ms,
    .now_ms = app_host_now_ms,
};


// ===============================================================================
// 扫描：枚举工作根下 app/ 目录的 .scm 文件，填充 s_apps 与菜单条目表
// ===============================================================================

// ASCII 小写比较（后缀过滤用，避免引入 strings.h）
static char ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
}

// 一次性分配扫描/菜单缓冲（本功能全生命周期持有；ESP32 上落在 PSRAM）。
// 返回 0 成功；失败（防御，实际不可达）返回 -1
static int32_t scheme_app_buffers_init(void) {
    if (s_apps) return 0;
    s_apps      = (Scheme_App_Desc *)platform_calloc(SCHEME_APP_MAX, sizeof(Scheme_App_Desc));
    s_item_text = (wchar_t (*)[SCHEME_APP_NAME_MAX])platform_calloc(SCHEME_APP_MAX, SCHEME_APP_NAME_MAX * sizeof(wchar_t));
    s_item_ptrs = (const wchar_t **)platform_calloc(SCHEME_APP_MAX, sizeof(wchar_t *));
    return (s_apps && s_item_text && s_item_ptrs) ? 0 : -1;
}


static void scheme_app_scan(void) {
    s_app_count = 0;
    s_list_empty = 0;

    char dir[256];
    snprintf(dir, sizeof(dir), "%s/app", PLATFORM_ROOT_DIR);  // ESP32 为 /app（SD 根）；桌面/WASM 为资源目录下 app/

    int32_t n = list_files(dir, NULL);
    if (n <= 0) {
        printf("[scheme-app] 扫描 %s：无条目或目录不存在\n", dir);
        s_list_empty = 1;
        return;
    }

    char **names = (char **)platform_calloc((size_t)n, sizeof(char *));
    if (!names) {
        s_list_empty = 1;
        return;
    }
    if (list_files(dir, names) < 0) {
        free(names);
        s_list_empty = 1;
        return;
    }

    for (int32_t i = 0; i < n && s_app_count < SCHEME_APP_MAX; i++) {
        char *name = names[i];
        if (!name) continue;
        size_t len = strlen(name);
        // 过滤：须为 xxx.scm（后缀不区分大小写）
        if (len < 5
            || ascii_lower(name[len - 4]) != '.'
            || ascii_lower(name[len - 3]) != 's'
            || ascii_lower(name[len - 2]) != 'c'
            || ascii_lower(name[len - 1]) != 'm') {
            free(name);
            names[i] = NULL;  // 防止下方统一清理循环二次释放（ESP32 multi_heap 中毒检测会断言）
            continue;
        }

        Scheme_App_Desc *d = &s_apps[s_app_count];
        // 显示名：去掉 .scm 后缀（截断到容量；给「● 运行中」角标留 8 个字符）
        size_t tl = len - 4;
        if (tl > SCHEME_APP_NAME_MAX - 9) tl = SCHEME_APP_NAME_MAX - 9;
        wchar_t tmp[SCHEME_APP_NAME_MAX];
        am_mbstowcs(tmp, name, (uint32_t)tl);
        tmp[tl] = L'\0';
        wcscpy(d->title, tmp);
        // 相对路径：app/xxx.scm
        snprintf(d->source_rel, sizeof(d->source_rel), "app/%s", name);

        // 菜单项：有存活实例（前台或后台冻结）的 App 带「● 运行中」角标
        wcscpy(s_item_text[s_app_count], d->title);
        if (ui_scheme_app_has_live_instance(d->source_rel)) {
            wcscat(s_item_text[s_app_count], L"  ● 运行中");
        }
        s_item_ptrs[s_app_count] = s_item_text[s_app_count];
        s_app_count++;
    }
    for (int32_t i = 0; i < n; i++) {
        if (names[i]) free(names[i]);
    }
    free(names);

    // 按显示名排序（冒泡；同步交换描述与菜单指针，保证两者下标一致）
    for (int32_t i = 0; i + 1 < s_app_count; i++) {
        for (int32_t j = 0; j + 1 < s_app_count - i; j++) {
            if (wcscmp(s_item_ptrs[j], s_item_ptrs[j + 1]) > 0) {
                Scheme_App_Desc td = s_apps[j];
                s_apps[j] = s_apps[j + 1];
                s_apps[j + 1] = td;
                const wchar_t *tp = s_item_ptrs[j];
                s_item_ptrs[j] = s_item_ptrs[j + 1];
                s_item_ptrs[j + 1] = tp;
            }
        }
    }

    if (s_app_count == 0) {
        printf("[scheme-app] %s 下没有 .scm 文件\n", dir);
        s_list_empty = 1;
    }
    else {
        printf("[scheme-app] 扫描到 %d 个 Scheme App：", s_app_count);
        for (int32_t i = 0; i < s_app_count; i++) {
            printf(" %ls", s_apps[i].title);
        }
        printf("\n");
    }
}

// ===============================================================================
// 列表菜单：扫描 + 构建 w_menu_main + 绘制；item_action 选中后转入运行态
// ===============================================================================
int32_t ui_scheme_app_list_init(Key_Event *key_event, Global_State *global_state) {
    Widget_Menu_State *ms = global_state->w_menu_main;
    if (scheme_app_buffers_init() != 0) {
        // 主堆分配失败（防御，实际不可达）：显示占位项，不触碰未分配的缓冲
        static const wchar_t *oom_item = L"（内存不足）";
        static const wchar_t *oom_ptrs[1] = {NULL};
        oom_ptrs[0] = oom_item;
        ms->items = oom_ptrs;
        ms->item_num = 1;
        ms->title = L"灵机应用程序";
        s_list_empty = 1;
        ui_widget_menu_init(key_event, global_state, ms);
        return -1;
    }
    scheme_app_scan();

    if (s_list_empty || s_app_count == 0) {
        // 空目录占位项（不可进入）
        static const wchar_t *empty_item = L"（app 目录下没有 Scheme App）";
        s_item_ptrs[0] = empty_item;
        ms->items = s_item_ptrs;
        ms->item_num = 1;
        s_list_empty = 1;
    }
    else {
        ms->items = s_item_ptrs;
        ms->item_num = s_app_count;
    }
    ms->title = L"灵机应用程序";
    ui_widget_menu_init(key_event, global_state, ms);
    return 0;
}

int32_t ui_scheme_app_list_item_action(Key_Event *key_event, Global_State *global_state,
                                       Widget_Menu_State *menu_state) {
    (void)key_event;
    (void)global_state;
    int32_t idx = menu_state->current_item_index;
    if (s_list_empty || s_app_count == 0 || idx < 0 || idx >= s_app_count) {
        return STATE_SCHEME_APP_LIST;  // 占位项/越界：留在列表
    }
    s_selected = idx;
    return STATE_SCHEME_APP_RUN;
}

int32_t ui_scheme_app_list_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 退出 App 后的在途触摸序列余波：剥掉触摸字段后再交给菜单（见 s_list_touch_guard 注释）
    if (s_list_touch_guard) {
        // 解除条件：① 出现**新的一次按下**（说明旧序列已彻底结束、用户开始新动作）；
        //           ② 抑制窗口到期（兜底，窗口内用户未再触摸）
        if ((key_event->touch_edge & TOUCH_EDGE_DOWN) != 0 || get_timestamp_in_ms() > s_list_touch_guard_until_ms) {
            s_list_touch_guard = 0;
        }
        else {
            Key_Event ke = *key_event;
            ke.is_touching = 0;
            ke.touch_edge = 0;
            // 菜单控件统一处理（上下键/触屏/惯性滚动；A 返回 STATE_GAME_MENU 由 back_state 给出）
            return ui_widget_menu_event_handler(&ke, global_state, global_state->w_menu_main,
                                                ui_scheme_app_list_item_action,
                                                STATE_GAME_MENU, STATE_SCHEME_APP_LIST);
        }
    }
    // 菜单控件统一处理（上下键/触屏/惯性滚动；A 返回 STATE_GAME_MENU 由 back_state 给出）
    return ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main,
                                        ui_scheme_app_list_item_action,
                                        STATE_GAME_MENU, STATE_SCHEME_APP_LIST);
}

// ===============================================================================
// 运行宿主：读取源码 → 解析 → 链接 → 编译 → 加载进程
// ===============================================================================

static void app_load_source(void) {
    // ---- 从业务资源目录读取 Scheme App 源码（UTF-8）----
    wchar_t rel_w[64];
    am_mbstowcs(rel_w, s_app->source_rel, (uint32_t)strlen(s_app->source_rel));
    wchar_t path_w[160];
    swprintf(path_w, sizeof(path_w) / sizeof(wchar_t), L"%ls%ls", s_app_root, rel_w);
    char path_mb[256];
    memset(path_mb, 0, sizeof(path_mb));
    am_wcstombs(path_mb, path_w, (uint32_t)sizeof(path_mb) - 1);

    uint8_t *file_buffer = NULL;
    size_t file_size = 0;
    if (platform_read_file_to_buffer(path_mb, &file_buffer, &file_size) != 0
        || file_buffer == NULL || file_size == 0) {
        swprintf(s_app->err_load, MB_ERR_BUF_LEN, L"未找到 %ls", path_w);
        printf("[scheme-app] 读取 %s 失败\n", path_mb);
        s_app->load_failed = 1;
        return;
    }
    if (file_size > MB_SOURCE_MAX_BYTES) {
        swprintf(s_app->err_load, MB_ERR_BUF_LEN, L"源码过大（%u 字节）", (uint32_t)file_size);
        printf("[scheme-app] 源码过大：%u 字节\n", (uint32_t)file_size);
        free(file_buffer);
        s_app->load_failed = 1;
        return;
    }
    printf("[scheme-app] %ls：%s（%u 字节）\n", s_app->title, path_mb, (uint32_t)file_size);

    // ---- UTF-8 → 宽字符。am_parse 只接受单个顶级表达式，与编辑器模式/
    //      am_repl 同惯例，外套 ((lambda () ...)) 包装。----
    const wchar_t *prefix = L"((lambda () \n";
    const wchar_t *suffix = L"\n))";
    size_t code_cap = wcslen(prefix) + file_size + wcslen(suffix) + 1; // UTF-8最坏每字节一wchar
    wchar_t *code_w = (wchar_t *)platform_malloc(code_cap * sizeof(wchar_t));
    if (code_w == NULL) {
        wcscpy(s_app->err_load, L"内存不足");
        free(file_buffer);
        s_app->load_failed = 1;
        return;
    }
    wcscpy(code_w, prefix);
    am_mbstowcs(code_w + wcslen(prefix), (const char *)file_buffer, (uint32_t)file_size);
    wcscat(code_w, suffix);
    free(file_buffer);   // platform_read_file_to_buffer 的缓冲区由 free() 释放（同 ui_animac .load）
    s_app->code_w = code_w;

    wchar_t mod_name_w[32];
    wcscpy(mod_name_w, rel_w);

    am_ast_t *ast = am_parse(s_app->vm_alloc, code_w, mod_name_w, 0);
    if (!ast) {
        wcscpy(s_app->err_load, L"Scheme 语法解析失败");
        s_app->load_failed = 1;
        return;
    }

    am_ast_t *linked = am_link(ast, s_app_root, am_host_read_source_from_file, NULL);
    if (!linked) {
        am_ast_destroy(ast);
        wcscpy(s_app->err_load, L"模块链接失败");
        s_app->load_failed = 1;
        return;
    }

    am_module_t *mod = am_compile(linked, 0, 0);
    if (!mod) {
        am_ast_destroy(linked);
        // 32 位宿主的 VM 工作区只有池的一半左右；大 App（含共享星历模块）在
        // 内存受限宿主上会因 VM 区不足而编译失败，这里给出可诊断的提示。
        swprintf(s_app->err_load, MB_ERR_BUF_LEN,
                 L"Scheme 编译失败（解释器池 %u KB，可能内存不足）",
                 (uint32_t)(s_app->pool_size_bytes / 1024));
        s_app->load_failed = 1;
        return;
    }

    s_app->pid = am_runtime_load_module(s_app->rt, mod);
    if (s_app->pid == (am_pid_t)-1) {
        if (mod->ilcode) am_free(s_app->vm_alloc, mod->ilcode);
        if (mod->ast) am_ast_destroy(mod->ast);
        am_free(s_app->vm_alloc, mod);
        wcscpy(s_app->err_load, L"进程加载失败（内存不足）");
        s_app->load_failed = 1;
        return;
    }

    // 模块静态数据已复制进进程，释放之（同 ui_animac 编辑器模式）
    if (mod->ilcode) am_free(s_app->vm_alloc, mod->ilcode);
    if (mod->ast) am_ast_destroy(mod->ast);
    am_free(s_app->vm_alloc, mod);
}

// 解释器池上限/最小可接受值（单实例用；每个后台实例各占一份）
static int32_t app_instance_cold_start(Global_State *global_state);

int32_t ui_scheme_app_init(Key_Event *key_event, Global_State *global_state) {
    UI_Scheme_App_State *inst;
    (void)key_event;

    // 选中合法性（列表菜单的 item_action 已记录 s_selected）
    if (s_selected < 0 || s_selected >= s_app_count) {
        printf("[scheme-app] 未选择 App\n");
        return 0;
    }

    // 绘图目标绑定（Gfx 本地库读写 global_state->gfx）——冷启动与恢复都需要
    am_native_gfx_bind(global_state->gfx);

    // ① 同一个 App 已有实例（后台冻结中）→ 直接恢复，不重建解释器
    inst = app_instance_find(s_apps[s_selected].source_rel);
    if (inst != NULL) {
        s_app = inst;
        app_instance_foreground(inst);
        printf("[scheme-app] 恢复后台实例：%ls（内存池 %u 字节仍在）\n",
               inst->title, (uint32_t)inst->pool_size_bytes);
        return 0;
    }

    // ② 冷启动：先挑槽位（必要时驱逐最久未用的后台实例）
    s_app = app_instance_alloc(s_selected);
    if (s_app == NULL) {
        printf("[scheme-app] 实例槽位不足且无可驱逐的后台实例\n");
        return 0;
    }
    return app_instance_cold_start(global_state);
}

// 冷启动当前 s_app 指向的实例：建池 → 建运行时 → 注册本地库 → 读源码加载进程
static int32_t app_instance_cold_start(Global_State *global_state) {

    // 工作目录宽前缀（带尾斜杠）
    _mbstowcs(s_app_root, PLATFORM_ROOT_DIR, 126);
    size_t rl = wcslen(s_app_root);
    if (rl == 0) wcscpy(s_app_root, L"/");
    else if (s_app_root[rl - 1] != L'/') { s_app_root[rl] = L'/'; s_app_root[rl + 1] = L'\0'; }

    // 内存池自适应申请（同 ui_animac 策略：按最大连续块留安全余量）
    size_t largest = platform_get_largest_free_block();
    if (largest < MB_POOL_MIN + MB_POOL_MARGIN) {
        printf("[scheme-app] 内存不足（最大连续块 %u 字节）\n", (uint32_t)largest);
        wcscpy(s_app->err_load, L"内存不足");
        s_app->load_failed = 1;
        return 0;
    }
    size_t pool_size = MB_POOL_SIZE;
    if (pool_size > largest - MB_POOL_MARGIN) {
        pool_size = largest - MB_POOL_MARGIN;
    }
    printf("[scheme-app] 解释器内存池：%u 字节（最大连续块 %u）\n", (uint32_t)pool_size, (uint32_t)largest);
    s_app->pool_size_bytes = pool_size;

    s_app->pool = am_allocator_pool_create(pool_size, &am_host_default_vtable);
    if (!s_app->pool) {
        wcscpy(s_app->err_load, L"内存不足");
        s_app->load_failed = 1;
        return 0;
    }
    s_app->vm_alloc = am_allocator_pool_get_vm(s_app->pool);
    s_app->heap_alloc = am_allocator_pool_get_heap(s_app->pool);

    s_app->rt = am_runtime_create(s_app->vm_alloc, s_app->heap_alloc, s_app_root, &APP_VTABLE);
    if (!s_app->rt) {
        am_allocator_pool_destroy(s_app->pool);
        s_app->pool = NULL;
        wcscpy(s_app->err_load, L"内存不足");
        s_app->load_failed = 1;
        return 0;
    }

    am_runtime_set_default_timeslice(s_app->rt, MB_TIMESLICE_TICKS);
    am_set_runtime_host_context(s_app->rt, s_app);

    // 注册本地库：System/Math/String/Table 为灵机标配；Gfx 与 Time 为本项目新增
    //（Time 用于 32 位宿主上取得不被截断/不失精度的绝对时间，见 am_native_Time.h）
    am_runtime_register_native_lib(s_app->rt, &am_native_System_lib);
    am_runtime_register_native_lib(s_app->rt, &am_native_Math_lib);
    am_runtime_register_native_lib(s_app->rt, &am_native_String_lib);
    am_runtime_register_native_lib(s_app->rt, &am_native_Table_lib);
    am_runtime_register_native_lib(s_app->rt, &am_native_Gfx_lib);
    am_runtime_register_native_lib(s_app->rt, &am_native_Time_lib);
    // Diag：虚拟机自观测（只读栈深/PC/闭包槽位/GC 等），供 app/xray.scm 等仪表盘使用
    am_runtime_register_native_lib(s_app->rt, &am_native_Diag_lib);
    // Touch/Imu：触屏与姿态（重力）输入，供交互式 App 使用（见 am_native_Touch.h / am_native_Imu.h）
    am_runtime_register_native_lib(s_app->rt, &am_native_Touch_lib);
    am_runtime_register_native_lib(s_app->rt, &am_native_Imu_lib);
    // Key/Fs：键盘边沿事件与文本文件读写（见 am_native_Key.h / am_native_Fs.h）
    am_runtime_register_native_lib(s_app->rt, &am_native_Key_lib);
    am_runtime_register_native_lib(s_app->rt, &am_native_Fs_lib);
    // Ui：宿主模态文本输入/输出（借用鹦鹉笼同款文本控件，见 am_native_Ui.h）
    am_runtime_register_native_lib(s_app->rt, &am_native_Ui_lib);
    // App：应用生命周期（退出/返回请求/多任务占位，见 am_native_App.h）
    am_runtime_register_native_lib(s_app->rt, &am_native_App_lib);
    am_native_ui_abort(); // 清掉上一个 App 可能遗留的请求
    am_native_app_reset(s_app->title);
    am_native_Key_reset();

    // 清屏后加载并启动 Scheme 进程（首帧由 App 自行绘制接管）
    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);

    app_load_source();
    if (s_app->load_failed) {
        printf("[scheme-app] %ls 源码构建/加载失败\n", s_app->title);
    }
    else {
        s_app->alive = 1;
    }
    // 冷启动的实例同样是前台（eviction 只驱逐后台实例）
    for (int32_t i = 0; i < SCHEME_APP_INST_MAX; i++) s_inst[i].foreground = 0;
    s_app->foreground = 1;
    s_app->suspended = 0;
    s_app->last_active_ms = get_timestamp_in_ms();
    am_native_app_set_active(1);
    return 0;
}

// ===============================================================================
// 每帧驱动：泵一个时间片 → 排水错误FIFO → 推送脏区
// ===============================================================================

// 把运行时错误FIFO追加进 s_app->err_buf（保留最近内容）
static void app_drain_error_fifo(void) {
    if (!s_app->rt || !s_app->rt->error_fifo) return;
    while (s_app->rt->error_fifo->length > 0) {
        am_value_t v = am_list_shift(s_app->vm_alloc, s_app->rt->error_fifo);
        if (!am_value_is_wchar(v)) continue;
        wchar_t c = (wchar_t)am_value_to_wchar(v);
        size_t len = wcslen(s_app->err_buf);
        if (c == L'\n') {
            // 换行即重置：只保留最近一行错误
            s_app->err_buf[0] = L'\0';
            continue;
        }
        if (len + 1 < MB_ERR_BUF_LEN) {
            s_app->err_buf[len] = c;
            s_app->err_buf[len + 1] = L'\0';
        }
    }
}

int32_t ui_scheme_app_render_frame(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    Nano_GFX *gfx = global_state->gfx;
    if (s_app == NULL) return 0;


    if (s_app->load_failed) {
        // 失败画面：黑底，标题 + 原因 + 提示（只画一次，等 A 退出）
        static int32_t error_drawn = 0;
        if (!error_drawn) {
            gfx_soft_clear(gfx);
            gfx_font_draw_text(gfx, GFX_FONT_ALPHA_16,
                               s_app->title,
                               8, 8, 255, 255, 255, 1);
            gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12,
                               (s_app->err_load[0] != L'\0') ? s_app->err_load : L"Scheme 程序加载失败",
                               8, 100, 255, 80, 80, 1);
            wchar_t hint_rel_w[64];
            am_mbstowcs(hint_rel_w, s_app->source_rel, (uint32_t)strlen(s_app->source_rel));
            wchar_t hint[160];
            swprintf(hint, 160, L"请检查资源目录 %ls", hint_rel_w);
            gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, hint, 8, 120, 180, 180, 180, 1);
            gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"按 A 返回", 8, 200, 180, 180, 180, 1);
            gfx_refresh(gfx);
            error_drawn = 1;
        }
        return 0;
    }

    if (s_app->alive && s_app->rt) {
        // 驱动解释器一个时间片（时间片内自动抢占，帧率不受影响；
        // 源码内的 set_interval 定时器亦由本调用触发）。
        // 推屏时机由 Scheme 经 (Gfx.refresh) 控制：跨多帧的绘制
        // （如逐行渐进、逐代演进）不在中途推屏，避免闪烁。
        am_runtime_event_handler(s_app->rt);
        app_drain_error_fifo();

        // 本帧内 App 可能刚发起模态文本请求（(Ui.input ...) 已把进程挂起）：
        // 立刻打开宿主控件并把画面交给它，后续帧由 scheme_modal_event 服务。
        if (am_native_ui_pending() && !s_modal_active) {
            scheme_modal_open(key_event, global_state);
        }
    }

    // 运行期错误一次性叠加提示（虚拟机报错后进程停机，Scheme 帧不再更新）
    if (s_app->err_buf[0] != L'\0' && !s_app->err_buf_shown) {
        gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, s_app->err_buf, 2, 2, 255, 80, 80, 1);
        gfx_refresh(gfx);
        s_app->err_buf_shown = 1;
    }
    return 0;
}

// ===============================================================================
// 事件：A(ESC) 返回灵机应用程序列表
// ===============================================================================

// ===============================================================================
// 模态文本输入/输出（本地库 (native Ui) 的宿主侧）
//
//  分工：am_native_Ui.c 负责「把 Scheme 进程挂起 / 唤醒」与文案交接；本段落负责
//  **界面**——直接借用宿主现成的文本控件（鹦鹉笼的输入框 ui_widget_input_* 与文本显示
//  控件 ui_widget_textarea_*），因此全键盘拼音 / 16 键九宫格输入法 / 软键盘 / 候选字翻页 /
//  光标与滚动全部天然可用，Scheme 侧一行不用写。
//
//  生命周期：
//    库发起请求（解释器在 render_frame 里跑出一帧）→ 本帧 scheme_modal_open() 打开控件；
//    之后每帧 scheme_modal_event() 把事件喂给控件（此前 ui_app.c 的 4x4 宫格映射会被抑制）；
//    提交/取消 → am_native_ui_resolve() 唤醒进程 → App 自己的下一拍重绘接管画面。
//
//  机制细节（易错）：ui_widget_input_event_handler 用「上一/下一焦点状态」表达退出方向，
//  我们对这两个位置传哨兵值（-1001/-1002），从而在一套控件代码上区分「取消」与「提交」。
// ===============================================================================

#define UI_MODAL_STATE_CANCEL (-1001) // 控件上一焦点态哨兵：取消（页眉返回热区）
#define UI_MODAL_STATE_SUBMIT (-1002) // 控件下一焦点态哨兵：提交（短按 Enter）

static int32_t s_modal_kind = AM_NATIVE_UI_KIND_NONE;
static Widget_Input_State *s_modal_in = NULL;       // 借用 w_input_main
static Widget_Textarea_State *s_modal_ta = NULL;    // 借用 w_textarea_main（只读消息）
static int32_t s_modal_confirm_with_body = 0;       // 确认框是否带正文（几何/命中判定须与绘制一致）
static int32_t s_modal_confirm_with_bg = 0;         // 确认框是否有第三枚「后台」按钮（退出型请求）
static wchar_t s_modal_title[AM_NATIVE_UI_TITLE_MAX];   // 页眉文案（控件长期持有指针）
static wchar_t s_modal_result[AM_NATIVE_UI_BODY_MAX];   // 提交时的文本快照

// 供 ui_app.c 的事件层查询：模态框显示期间抑制触屏 4x4 宫格映射
//（否则拖动滚动/点按定位会不断产生软按键，且右上角宫格会变成 Esc）
int32_t ui_scheme_app_hosts_input_widget(void) {
    return s_modal_active;
}

static void scheme_modal_finish(Global_State *global_state, int32_t submit, Key_Event *key_event) {
    if (!s_modal_active) return;
    s_modal_active = 0;
    if (key_event != NULL && key_event->key_edge == -2) {
        s_modal_ignore_longpress_repeat = 1; // 之后的长按重复事件不再生效（见变量注释）
    }

    // 确认框：把 #t/#f 交回 App。若这是「退出型」确认框（(App.exit) 发起的），
    // 肯定答复会同时置 App 退出意愿 —— 宿主在**本帧泵完之后**销毁实例
    //（见 ui_scheme_app_should_exit），因此 App 还有整整一个时间片做收尾。
    // 这里不做帧缓冲快照恢复：关闭后由 App 自己重绘（零快照方案，见文件头说明）。
    if (s_modal_kind == AM_NATIVE_UI_KIND_CONFIRM) {
        // submit：1=确认 / 0=留下 / 2=后台（见 scheme_modal_event）
        am_native_ui_resolve_confirm(s_app->rt, submit);
        am_native_app_notify_modal_closed();
        if (global_state->gfx != NULL) global_state->is_full_refresh = 1;
        s_modal_in = NULL;
        s_modal_ta = NULL;
        s_modal_confirm_with_body = 0;
        s_modal_confirm_with_bg = 0;
        s_modal_kind = AM_NATIVE_UI_KIND_NONE;
        return;
    }

    // 先唤醒进程：连发提问的 App 会在下一帧重新打开模态框，状态不能残留
    am_native_ui_resolve(s_app->rt, submit ? s_modal_result : NULL);
    am_native_app_notify_modal_closed();

    if (s_modal_kind == AM_NATIVE_UI_KIND_INPUT && s_modal_in != NULL) {
        // 输入法全状态复位 + 双键盘收起 + 布局恢复（不动文本缓冲区）
        ui_widget_input_back_cleanup(global_state, s_modal_in);
    }
    if (key_event != NULL && global_state->gfx != NULL) {
        // 让 App 的下一拍自行重绘（这里只把控制权交回，不画任何 App 内容）
        global_state->is_full_refresh = 1;
    }

    s_modal_in = NULL;
    s_modal_ta = NULL;
    s_modal_kind = AM_NATIVE_UI_KIND_NONE;
}

static void scheme_modal_open(Key_Event *key_event, Global_State *global_state) {
    const wchar_t *title = am_native_ui_title();
    s_modal_kind = am_native_ui_kind();

    if (title != NULL && title[0] != L'\0') {
        wcsncpy(s_modal_title, title, AM_NATIVE_UI_TITLE_MAX - 1);
        s_modal_title[AM_NATIVE_UI_TITLE_MAX - 1] = L'\0';
    }
    else {
        if (s_modal_kind == AM_NATIVE_UI_KIND_CONFIRM) wcscpy(s_modal_title, L"是否退出？");
        else                                            wcscpy(s_modal_title, L"消息");
    }

    s_modal_active = 1;
    s_modal_open_key = 0;
    s_modal_open_key_deadline_ms = 0;
    {
        // 剔除打开本次模态框的那次按键的松开沿（见变量注释）
        uint8_t code = 0;
        uint64_t ts = 0;
        am_native_Key_last_press(&code, &ts);
        if (code != 0 && ts != 0) {
            uint64_t now = (uint64_t)global_state->timestamp;
            if (now >= ts && (now - ts) <= 200) {
                s_modal_open_key = code;
                s_modal_open_key_deadline_ms = now + UI_MODAL_OPEN_KEY_WINDOW_MS;
            }
        }
    }
    am_native_Key_reset(); // 打开模态框时清一次：触发本次输入的那次按键残留在队列里也无害

    // 确认框（(Ui.confirm ...) 与 (App.exit) 共用）：用图形层现成的退出确认框组件绘制，
    // 与 App 自己弹的框、控制台的退出确认框是同一视觉与命中——无需文本控件。
    if (s_modal_kind == AM_NATIVE_UI_KIND_CONFIRM) {
        s_modal_in = NULL;
        s_modal_ta = NULL;
        s_modal_confirm_with_body = (am_native_ui_body()[0] != L'\0');
        // 退出型确认框多一枚「后台」（确认 / 后台 / 留下）——多任务入口就在这里
        s_modal_confirm_with_bg = am_native_ui_confirm_is_exit();
        // 任何模态框开起来都说明 App 还活着在干活：清除可能挂起的“返回请求”，
        // 免得框关掉之后被宿主兜底当成未接管的请求而误杀
        am_native_app_clear_back_request();
        ui_exit_confirm_draw_msg_ex(key_event, global_state, s_modal_title, am_native_ui_body(),
                                    s_modal_confirm_with_bg);
        return;
    }

    if (s_modal_kind == AM_NATIVE_UI_KIND_INPUT) {
        const wchar_t *init = am_native_ui_initial();
        s_modal_in = global_state->w_input_main;
        s_modal_ta = NULL;
        if (s_modal_in == NULL) {
            // 控件未就绪（理论上不可能）：直接按取消结束，避免进程永远挂着
            s_modal_result[0] = L'\0';
            scheme_modal_finish(global_state, 0, NULL);
            return;
        }
        // 控件自绘（页眉标题 + 文本区 + 页脚 + 软键盘状态），预算好标题指针长期有效
        ui_widget_input_init(key_event, global_state, s_modal_in, s_modal_title);
        if (init != NULL && init[0] != L'\0') {
            for (size_t i = 0; init[i] != L'\0'; i++) insert_char(s_modal_in, init[i]);
            s_modal_in->cursor_pos = s_modal_in->textarea.length - 1;
            s_modal_in->desired_x = -1;
            ui_draw_input_buffer(key_event, global_state, s_modal_in);
        }
    }
    else {
        s_modal_in = NULL;
        s_modal_ta = global_state->w_textarea_main;
        if (s_modal_ta == NULL) {
            scheme_modal_finish(global_state, 0, NULL);
            return;
        }
        ui_widget_textarea_reset_geometry(key_event, global_state, s_modal_ta);
        ui_widget_textarea_set(key_event, global_state, s_modal_ta,
                               (wchar_t *)am_native_ui_body(), 0, 1);
        ui_draw_header(key_event, global_state, s_modal_title, 1);
        ui_widget_textarea_draw(key_event, global_state, s_modal_ta);
        ui_draw_footer(key_event, global_state, L"任意键 / 点按返回", 1);
        gfx_refresh(global_state->gfx);
    }
}

static void scheme_modal_event(Key_Event *key_event, Global_State *global_state) {
    if (!s_modal_active) return;

    // 打开模态框那次按键的松开沿：吃掉一次（见 s_modal_open_key 注释）。
    // 一旦出现"别的按键活动"或触屏动作，说明用户已经在模态框里操作了，立刻解除剔除，
    // 免得把用户真正的回车吃掉。
    if (s_modal_open_key != 0 && key_event->key_code != NANO_KEY_IDLE
        && (key_event->key_edge == -1 || key_event->key_edge == -2)) {
        if (key_event->key_code == s_modal_open_key
            && (uint64_t)global_state->timestamp <= s_modal_open_key_deadline_ms) {
            s_modal_open_key = 0;
            return;
        }
        s_modal_open_key = 0; // 是别的键：解除剔除，照常交给控件
    }
    if (s_modal_open_key != 0 && key_event->touch_edge != 0) {
        s_modal_open_key = 0; // 触屏活动同样解除
    }

    if (s_modal_kind == AM_NATIVE_UI_KIND_CONFIRM) {
        /* 确认框交互：
         *   · 触屏——松手沿 + 按下点命中按钮（全局范式，按下沿不结算）；点框外不关闭；
         *   · 键盘——Enter = 确认；其他任意键（含 Esc）= 留下（安全默认：不做破坏性动作）；
         *   · 长按重复（-2 连发）已在 ui_scheme_app_event_handler 入口处拦掉，
         *     否则“长按 A 唤起确认框”会被自己的重复事件连坐关掉。
         */
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            int32_t hit = ui_exit_confirm_hit_msg_ex(global_state, s_modal_confirm_with_body,
                                                    s_modal_confirm_with_bg,
                                                    key_event->touch_down_x, key_event->touch_down_y);
            if (hit == 1)      scheme_modal_finish(global_state, 1, key_event);   // 确认
            else if (hit == 2) scheme_modal_finish(global_state, 0, key_event);   // 留下
            else if (hit == 3) scheme_modal_finish(global_state, 2, key_event);   // 后台
            return;
        }
        if (key_event->key_edge == -1 || key_event->key_edge == -2) {
            scheme_modal_finish(global_state, (key_event->key_code == NANO_KEY_enter) ? 1 : 0, key_event);
        }
        return;
    }

    if (s_modal_kind == AM_NATIVE_UI_KIND_INPUT) {
        // 取消：长按 A/Esc（短按 Esc 是控件自己的退格键，平台惯例）
        if (key_event->key_edge == -2 && key_event->key_code == NANO_KEY_esc) {
            scheme_modal_finish(global_state, 0, key_event);
            return;
        }
        int32_t next = ui_widget_input_event_handler(
            key_event, global_state, s_modal_in,
            UI_MODAL_STATE_CANCEL, global_state->STATE, UI_MODAL_STATE_SUBMIT);

        if (next == UI_MODAL_STATE_SUBMIT) {
            int32_t n = s_modal_in->textarea.length;
            if (n < 0) n = 0;
            if (n > AM_NATIVE_UI_BODY_MAX - 1) n = AM_NATIVE_UI_BODY_MAX - 1;
            for (int32_t i = 0; i < n; i++) s_modal_result[i] = s_modal_in->textarea.text[i];
            s_modal_result[n] = L'\0';
            scheme_modal_finish(global_state, 1, key_event);
        }
        else if (next == UI_MODAL_STATE_CANCEL) {
            scheme_modal_finish(global_state, 0, key_event);
        }
        return;
    }

    // MESSAGE：拖动=滚动、点按或任意键=关闭
    if (s_modal_ta != NULL) {
        int32_t r = ui_widget_textarea_touch_handler(key_event, global_state, s_modal_ta);
        if (r == 1) { // 拖动滚动
            ui_widget_textarea_draw(key_event, global_state, s_modal_ta);
            return;
        }
        if (r == 2) { // 点按关闭
            scheme_modal_finish(global_state, 0, key_event);
            return;
        }
    }
    if (key_event->key_edge == -1 || key_event->key_edge == -2) {
        scheme_modal_finish(global_state, 0, key_event);
    }
}

int32_t ui_scheme_app_event_handler(Key_Event *key_event, Global_State *global_state) {
    if (s_app == NULL) return 0;
    // 长按连发抑制（见 s_modal_ignore_longpress_repeat 注释）：
    // 生产端只投递下降沿（-1 短按结束 / -2 长按结束），故用"出现了一次 -1"表示这次长按
    // 早已结束、用户又按了一下，届时解除抑制。
    if (key_event->key_edge == -1) s_modal_ignore_longpress_repeat = 0;
    if (s_modal_ignore_longpress_repeat && key_event->key_edge == -2) return 0;

    // 模态文本输入/输出期间：事件全部交给宿主控件，"A 返回列表"暂时让位
    if (s_modal_active) {
        scheme_modal_event(key_event, global_state);
        // 模态框期间的按键归**控件**所有，不能渗进 (native Key) 的旁路队列：
        // 否则用户敲的每一次 Enter/退格都会在 App 恢复后被它当成新按键读走
        //（典型症状：关掉消息框后 App 又"自己按了一次 Enter"）。
        am_native_Key_reset();
        return 0;
    }
    // 硬 A/ESC：不再直接退出（见 ui_scheme_app.h 顶部的退出交互说明）。分三种情形：
    //   ① App 加载失败/无运行时（没有 Scheme 代码可接管）→ 保持旧行为：直接回列表；
    //   ② 正常：只置“返回请求”位，由 App 自己消费（app/lib/exit.scm 的 (EX.check)）；
    //      长按（-2）时同时抑制其连发重复，否则后续 -2 会把 App 刚弹开的确认框连坐关掉；
    //   ③ 700ms 无人接管 → ui_scheme_app_should_exit() 兜底退出（= 旧行为，不会把人困在 App 里）。
    // 触屏「退出」按钮：由 App 侧登记的**热区**判定（事件级触摸边沿，与 App 的轮询
    // 节拍无关——慢节奏 App 如星历 1000ms/拍也能可靠点中）。命中 = 与硬 A/ESC 同一条
    // 通道：置"返回请求"，由 App 自己的 (EX.check) 弹全局确认框。
    if ((key_event->touch_edge & TOUCH_EDGE_UP)
        && am_native_app_hit_exit_button(key_event->touch_down_x, key_event->touch_down_y)
        && !am_native_app_exit_confirmed() && !am_native_app_exit_wanted(s_app->rt)) {
        am_native_app_set_back_request(s_app->rt);
        return 0;
    }
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        if (s_app->load_failed || s_app->rt == NULL) {
            global_state->STATE = STATE_SCHEME_APP_LIST;
            return 0;
        }
        // 已在退出流程中（收尾窗口内）：A/ESC 不再有意义，忽略
        if (am_native_app_exit_confirmed() || am_native_app_exit_wanted(s_app->rt)) return 0;
        am_native_app_set_back_request(s_app->rt);
        if (key_event->key_edge == -2) s_modal_ignore_longpress_repeat = 1;
        return 0;
    }
    return 0;
}

// ===============================================================================
// 退出判定（由 ui_app.c 在**泵完本帧之后**调用）
// ===============================================================================

// 返回请求无人接管的兜底窗口（毫秒）：
// 未迁移的 App（不轮询 (App.back_requested)）、或 App 已经停机报错（没代码可消费）时，
// 硬件 A/ESC 不应失去意义——超时后按旧行为退出。
// 取值必须大于**最慢 App 的轮询周期**：星历是 1000ms/拍（其余 ≤100ms），故取 1500ms。
// （原来 700ms 会让星历在"还没来得及弹确认框"时就被兜底退掉。）
#define SCHEME_APP_BACK_FALLBACK_MS (1500)

int32_t ui_scheme_app_should_exit(Global_State *global_state) {
    int32_t ms;
    (void)global_state;

    if (s_app == NULL || s_app->rt == NULL) return 0;

    // 模态框存续期间：App 正在和用户交互（不是卡在收尾），续期收尾窗口、不销毁
    if (s_modal_active) {
        am_native_app_extend_grace(s_app->rt);
        return 0;
    }

    // (App.exit_now)/(App.hide) → 立即销毁；
    // (App.exit) 已获确认但**收尾窗口超时**（App 忘了 exit_now）→ 强制销毁
    if (am_native_app_exit_wanted(s_app->rt)) {
        int32_t grace = am_native_app_exit_grace_ms(s_app->rt);
        if (grace >= AM_NATIVE_APP_EXIT_GRACE_MS) {
            printf("[scheme-app] 退出收尾超时（%d ms 仍未 (App.exit_now)），强制销毁实例\n", grace);
        }
        return 1;
    }

    ms = am_native_app_back_pending_ms(s_app->rt);
    if (ms >= SCHEME_APP_BACK_FALLBACK_MS) {
        printf("[scheme-app] 返回请求 %d ms 无人接管，按旧行为退出\n", ms);
        return 1;
    }
    return 0;
}

// 是否应该**切后台**（保留实例）：(App.hide) 或退出确认框里选了「后台」。
// 单实例平台（ESP32）没有后台能力：hide 退化为退出（不把 PSRAM 长期占住）。
int32_t ui_scheme_app_should_hide(Global_State *global_state) {
    (void)global_state;
    if (s_app == NULL || s_app->rt == NULL) return 0;
    if (s_modal_active) return 0;
    if (!am_native_app_hide_requested()) return 0;
#if SCHEME_APP_INST_MAX <= 1
    printf("[scheme-app] 本平台无后台能力（实例槽位 %d）：切后台按退出处理\n",
           (int32_t)SCHEME_APP_INST_MAX);
    return 0;
#else
    return 1;
#endif
}

// ===============================================================================
// 退出清理：杀进程 → 销毁运行时 → 销毁内存池 → 解绑 Gfx
// ===============================================================================

// ---------------------------------------------------------------------------
// 实例表：查找 / 取槽 / 前台化 / 挂起 / 销毁
// ---------------------------------------------------------------------------

// 按源码相对路径找已有实例（冷启动前用它判断"是不是该恢复"）
static UI_Scheme_App_State *app_instance_find(const char *source_rel) {
    for (int32_t i = 0; i < SCHEME_APP_INST_MAX; i++) {
        if (s_inst[i].used && strcmp(s_inst[i].source_rel, source_rel) == 0) return &s_inst[i];
    }
    return NULL;
}

// 取一个槽：优先空槽；没有空槽时驱逐**最久未用**的后台实例（前台永不驱逐）。
static UI_Scheme_App_State *app_instance_alloc(int32_t desc_idx) {
    UI_Scheme_App_State *victim = NULL;
    int32_t i;

    for (i = 0; i < SCHEME_APP_INST_MAX; i++) {
        if (!s_inst[i].used) { victim = &s_inst[i]; break; }
    }
    if (victim == NULL) {
        // 无空槽：找 last_active_ms 最小的后台实例
        for (i = 0; i < SCHEME_APP_INST_MAX; i++) {
            if (s_inst[i].foreground) continue;
            if (victim == NULL || s_inst[i].last_active_ms < victim->last_active_ms) victim = &s_inst[i];
        }
        if (victim == NULL) return NULL;   // 理论上不可达（唯一前台不驱逐）
        printf("[scheme-app] 实例槽位已满（%d），驱逐后台实例：%ls\n",
               (int32_t)SCHEME_APP_INST_MAX, victim->title);
        app_instance_destroy(victim);
    }

    memset(victim, 0, sizeof(*victim));
    victim->used = 1;
    victim->pid = (am_pid_t)-1;
    wcscpy(victim->title, s_apps[desc_idx].title);
    strncpy(victim->source_rel, s_apps[desc_idx].source_rel, sizeof(victim->source_rel) - 1);
    victim->source_rel[sizeof(victim->source_rel) - 1] = '\0';
    victim->last_active_ms = get_timestamp_in_ms();
    return victim;
}

// 销毁一个实例（kill 进程 → 销毁运行时 → 销毁内存池 → 释放源码缓冲）
static void app_instance_destroy(UI_Scheme_App_State *inst) {
    if (inst == NULL || !inst->used) return;
    if (inst->rt) {
        if (inst->pid != (am_pid_t)-1) am_runtime_kill_process(inst->rt, inst->pid);
        am_runtime_destroy(inst->rt);
    }
    if (inst->pool) am_allocator_pool_destroy(inst->pool);
    if (inst->code_w) free(inst->code_w);
    if (s_app == inst) s_app = NULL;
    memset(inst, 0, sizeof(*inst));
    inst->pid = (am_pid_t)-1;
}

// 把实例提为前台：恢复本地库的"当前 App"上下文、解除挂起、通知 App 自己被恢复了
static void app_instance_foreground(UI_Scheme_App_State *inst) {
    for (int32_t i = 0; i < SCHEME_APP_INST_MAX; i++) s_inst[i].foreground = 0;
    inst->foreground = 1;
    inst->suspended = 0;
    inst->last_active_ms = get_timestamp_in_ms();
    am_native_ui_abort();                  // 上一次前台遗留的模态请求（若有）
    am_native_Key_reset();                 // 后台期间积压的按键不带给 App
    am_native_app_reset(inst->title);      // 本地库的"当前 App"上下文
    am_native_app_set_active(1);
    am_native_app_notify_resumed();         // (App.resumed) → #t 一次（App 据此重绘）
}

// 挂起实例（后台或锁屏）：不再泵它，(App.is_active) 变 #f；运行时与内存池原地保留
static void app_instance_suspend(UI_Scheme_App_State *inst) {
    if (inst == NULL) return;
    inst->suspended = 1;
    inst->foreground = 0;
    am_native_app_set_active(0);
    am_native_ui_abort();   // 挂起期间的模态请求作废（进程随后不再被泵）
}

// 退出当前前台实例（销毁）；供 ui_app.c 在 (App.exit) 后调用
void ui_scheme_app_on_exit(void) {
    UI_Scheme_App_State *inst = s_app;

    // 丢弃可能挂起的模态输入/退出请求（进程随后会被销毁，不能留悬空 PID）
    am_native_ui_abort();
    am_native_app_abort();
    // 退出可能发生在手指动作过程中：列表紧接着收到的"余波"（旧序列的电平与迟到的
    // UP 边沿）必须吞掉，直到用户真正开始新的一次按下或窗口到期（见 s_list_touch_guard）
    s_list_touch_guard = 1;
    s_list_touch_guard_until_ms = get_timestamp_in_ms() + 600;
    s_modal_active = 0;
    s_modal_ignore_longpress_repeat = 0;
    s_modal_open_key = 0;
    s_modal_open_key_deadline_ms = 0;
    s_modal_in = NULL;
    s_modal_ta = NULL;
    s_modal_kind = AM_NATIVE_UI_KIND_NONE;
    s_modal_confirm_with_body = 0;
    s_modal_confirm_with_bg = 0;

    app_instance_destroy(inst);
    am_native_gfx_bind(NULL);
}

// 切后台：保留实例与内存池，只把前台让出去（回应用列表）
void ui_scheme_app_suspend_to_list(void) {
    if (s_app == NULL) return;
    printf("[scheme-app] 切后台：%ls（内存池 %u 字节保留）\n",
           s_app->title, (uint32_t)s_app->pool_size_bytes);
    app_instance_suspend(s_app);
    s_app = NULL;

    // 与"退出"同样的在途触摸余波抑制（切后台也发生在手指动作过程中）
    s_list_touch_guard = 1;
    s_list_touch_guard_until_ms = get_timestamp_in_ms() + 600;
    s_modal_active = 0;
    s_modal_ignore_longpress_repeat = 0;
    s_modal_open_key = 0;
    s_modal_open_key_deadline_ms = 0;
    s_modal_in = NULL;
    s_modal_ta = NULL;
    s_modal_kind = AM_NATIVE_UI_KIND_NONE;
    s_modal_confirm_with_body = 0;
    s_modal_confirm_with_bg = 0;
}

// 锁屏/唤醒用：挂起或恢复**当前前台**实例（不切状态机，由 ui_app.c 的锁屏路径调用）
void ui_scheme_app_suspend_current(void) {
    if (s_app != NULL) app_instance_suspend(s_app);
}

void ui_scheme_app_resume_current(void) {
    if (s_app != NULL) app_instance_foreground(s_app);
}

// 是否有活着的后台实例（列表角标用）
int32_t ui_scheme_app_has_live_instance(const char *source_rel) {
    return app_instance_find(source_rel) != NULL;
}
