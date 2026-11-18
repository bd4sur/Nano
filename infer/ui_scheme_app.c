#include <stdio.h>
#include <stdlib.h>
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
#include "am_native_Chat.h"   // 鹦鹉笼端侧 LLM 推理引擎（(native Chat)）
#include "am_native_Raster.h" // 水池软光栅后端（(native Raster)，见 app/waterpool_raster.scm）
#include "am_native_Music.h"  // 音乐播放引擎（(native Music)，见 app/musicbox.scm）
#include "am_native_Audio.h"  // 音频采集 + 频谱分析（(native Audio)，见 app/spectrogram.scm）

// ===============================================================================
// 运行时参数
// ===============================================================================

// 解释器内存池：期望大小、最小可接受大小、自适应申请时留出的安全余量
//（策略与 ui_animac 一致；WASM/桌面平台堆充裕，实际按期望值分配）
// 由 2MB 逐级上调为 12MB：
//   · 4MB：sky3.scm（scatter_model_3 逐行移植）
//   · 12MB：ephemeris.scm（星历 · 农历单页：月球 10×60 项级数 + VSOP87 + 星历
//          函数集 + app/lib/ephem.scm + 农历年表推算）。合并前是两个 App
//          （ephemeris.scm 8MB、nongli.scm 12MB），合并后实测 12MB 即可
//   · 24MB：markdown.scm（连着 md_parse / md_layout / md_demo / exit 四个模块一起编译，
//          编译期工作区就要 ~13MB；实测 12MB 时连 8KB 的 Markdown 文件都装不下，
//          24MB 能容纳约 32KB 文档——App 侧据此设了 DOC_MAX 上限并给出友好提示）
//   · 32MB：markdown.scm 继续长大（又并进 filepicker / fsx 两个模块，且加了目录功能），
//          24MB 时编译期 VM 区（池的一半）就装不下了；32MB 实测可正常加载
//   · 48MB：markdown.scm 加入**块级数学公式**（复用 mathjax 的 TeX 引擎，连
//          tex_parse / tex_layout / tex_sym / tex_draw / tex_base 五个模块一起编译）
//          后编译期工作区又涨了一截：wasm32 实测 40MB 池仍 link_fail、44MB 可链接，
//          故期望值上调到 48MB（WASM/桌面的线性内存充裕，实测 384MB 从不增长；
//          ESP32 仍是 2MB——本 App 本来也装不下，见下）
//   · 80MB（仅 64 位宿主）：上面 48MB 那条的括号里已写明「64 位宿主 48MB 不够、64MB 够」——
//          wasm32 是 4 字节指针，64 位宿主（PocketTerm35 / RPi5 aarch64、桌面 x86_64）
//          的 AST 节点约两倍大。实测（PocketTerm35）markdown 的 15 模块闭包在链接期的
//          VM 峰值 ≈31.5MB（链接器先**逐模块全部解析**再逐个 merge，约为链接后驻留
//          9.2MB 的 3.4 倍）：56MB / 60MB 池仍 link_fail，64MB 刚好过；48MB 池（VM 区
//          24MB）在解析第 14 个模块 md_stream.scm 时就「VM segregated 分配失败」，
//          宿主只报「模块链接失败」。故 64 位宿主取 80MB（VM 区 40MB，留 8MB 余量）；
//          3 个实例槽位全占也仅 240MB，8GB 的 RPi5 无压力。
// 编译期的 VM 工作区占用远大于运行期堆占用（后者极小）。
// 内存受限宿主：pool_size 按「最大连续块 - 余量」自适应下调（见下），
// 小 App 仍可正常运行；装不下的大 App 会显示「内存不足」错误画面。
// ESP32（M5Core2/CoreS3，8MB PSRAM）：12MB 期望值超过整片 PSRAM，「最大连续块-余量」
// 自适应会把池子撑到数 MB，几乎吃光 PSRAM 并饿死音频/词典/电子书等其他运行时消费者。
// 故 ESP32 的期望值与 Animac 控制台（ui_animac.h，ESP32 验证值）对齐为 2MB，
#if NANO_IS_M5ESP
    #define MB_POOL_SIZE   (2ULL * 1024 * 1024)
#elif defined(__SIZEOF_POINTER__) && (__SIZEOF_POINTER__ == 8)
    // 64 位宿主（PocketTerm35 / 树莓派 aarch64、桌面 x86_64）：见上面「80MB」条
    #define MB_POOL_SIZE   (80ULL * 1024 * 1024)
#else
    // 32 位宿主（wasm32 浏览器 / webgui、Luckfox Pico 等 32 位 Linux）：见上面「48MB」条。
    // 2026-11 调整 48 → 56MB：markdown 加入「行内公式」后（md_parse/md_layout/markdown
    // 各涨 ~1-2KB，md_demo 的示例文档 +7KB 公式语料），wasm32 下 48MB 池**首次**能链接、
    // 但**第二次**冷启动就「源码构建/加载失败」（VM 区的链接期峰值顶到池边，浏览器里
    // 实测：同一页面先开一次、退出再开一次即复现；mathjax 等小 App 不受影响）。
    // 56MB（VM 区 28MB）留出余量，浏览器里连开两次正常；内存受限宿主仍走下面的
    // 「最大连续块 - 余量」自适应下调，装不下的大 App 照旧报「内存不足」。
    #define MB_POOL_SIZE   (56ULL * 1024 * 1024)
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
#define SCHEME_APP_MAX     (UI_SCHEME_APP_MAX)  // app/*.scm 上限（当前 52 个；余量留 76 个）
// 2026-11 历次上调：32 → 48 → 56 → 128。前两次都是被实际数量顶到只剩个位数的余量：
// app/ 下的 .scm 从 33 涨到 47 时 48 只剩 1 个余量、涨到 50 时 56 只剩 6 个（期间
// 另一会话在 app/ 里临时放 6 个草稿 .scm 就顶满了）。**上限越界不会报错**：收集循环
// 一旦装满就停，超出的项按**目录枚举顺序**被静默丢弃（历史上 xray.scm 就这样在列表里
// 凭空消失过一次）。故此处一次给足余量（128 项，约为当前 App 数的 2.5 倍）。
// 一次性缓冲的代价：s_apps(128×384B) + s_item_text(128×48×4B) + 指针表 ≈ 74KB，
// 进入本功能时一次性 calloc（ESP32 落在 PSRAM），见 scheme_app_buffers_init。
// 改动 App 数量时仍应核对这里与 browser/regress 的 App 清单（regress.js 直接从
// manifest 推出清单，不必手工同步）。
#define SCHEME_APP_NAME_MAX (48)

// App id（文件名去 .scm）与源码头部元数据探测窗口
//   id 用于**启动身份**与 (App.launch "id") 的名字匹配，与显示名（元数据 ;@title=）
//   分开存放：显示名是给菜单看的（可以是中文），id 是给代码看的（ASCII）。
#define SCHEME_APP_ID_MAX           (48)
#define SCHEME_APP_META_PROBE_BYTES (512) // 元数据按约定在源码头几行，只读这么多
#define SCHEME_APP_META_TITLE_MAX   (128) // 元数据 title 的 UTF-8 缓冲

// ===============================================================================
// App 描述与扫描结果
// ===============================================================================

typedef struct {
    wchar_t title[SCHEME_APP_NAME_MAX];  // 菜单显示名（源码头部元数据 ;@title=，缺省回退为 id）
    char    id[SCHEME_APP_ID_MAX];       // App id（文件名去 .scm，ASCII）——启动身份 / (App.launch) 名字
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

// 宿主自己发起的“退出当前实例”请求（非 App 侧发起）。目前只有一个来源：
//   **加载失败时按硬 A/ESC**（没有 Scheme 代码可接管返回请求，旧实现直接改 STATE）。
// 走本请求位而不是直接改 STATE，是为了让所有退出都汇到帧末的同一个漏斗
// （ui_scheme_app_perform_exit）——“退出 > 拉起 > 切后台”的排序也只有一处。
static int32_t s_host_exit_request = 0;

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
//   · 返回链（back_rel）：见 UI_Scheme_App_State 的字段说明。多槽平台才有意义
//     （单槽平台拉起时调用者已被销毁，没有“上一 App”可回）。
//     实例数 3 → 返回链最多 2 级（列表 → A → B → C 就占满了；再深会驱逐 A，链在 A 处断）。
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
    int32_t error_drawn;              // 加载失败画面已画过（每实例一份，冷启动清零）
    // ---- 返回链（App → App 的“回上一 App”，2026-11）----
    // “我是被谁拉起的”：本实例退出（销毁）后要回到的 App，同样用**源码相对路径**
    // 做身份键。空串 = 顶层（从应用列表进入），退出回应用列表。
    // 为什么存在**实例**里而不是另立一个栈：链的每个节点天然就是一个实例，回退 = 销毁
    // 当前实例 + 前台化 back_rel 指向的实例 —— 不需要深度上限、不需要驱逐时的双向簿记；
    // 父实例若已被 LRU 驱逐，app_instance_find 自然返回 NULL，降级为回列表即可。
    // 也正因为每次回退都先销毁当前实例，链**必然终止**（即使 A→B→A 绕圈）。
    char    back_rel[192];
} UI_Scheme_App_State;

static UI_Scheme_App_State s_inst[SCHEME_APP_INST_MAX];
static UI_Scheme_App_State *s_app = NULL;   // 指向**前台**实例；无前台时为 NULL

static UI_Scheme_App_State *app_instance_find(const char *source_rel);
static UI_Scheme_App_State *app_instance_alloc(int32_t desc_idx);
static void app_instance_destroy(UI_Scheme_App_State *inst);
static void app_instance_foreground(UI_Scheme_App_State *inst);
static void app_instance_suspend(UI_Scheme_App_State *inst);
static void app_apply_back_hint(UI_Scheme_App_State *inst);

// 名字 → s_apps[] 下标（拉起用；接受 "markdown"/"markdown.scm"/"app/markdown.scm"）
static int32_t app_desc_find_name(const wchar_t *name);
// 宿主服务钩子：给 (native App) 做 (App.launch) 的名字存在性校验
static int32_t app_host_lookup_app(const wchar_t *name);

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

// ---------------- 源码头部元数据读取（;@id= / ;@title= / ;@version=） ----------------
// 约定见 ui_scheme_app.h 文件头：元数据是注释形式，写在源码头几行。只读文件头
// SCHEME_APP_META_PROBE_BYTES 字节（不整文件读取：app/ 下的 .scm 从几十 KB 到
// 几百 KB 不等），逐行识别 ";@key=value"。
// 输出：out_title 为 UTF-8 的 title（未提供则置空串，由调用方回退为 id）。
static void scheme_app_meta_trim_value(const char *begin, const char *end, char *out, size_t cap) {
    // 去掉首尾空白（含 \r：Windows 换行的源码）
    while (begin < end && (*begin == ' ' || *begin == '\t' || *begin == '\r')) begin++;
    while (end > begin && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r')) end--;
    size_t n = (size_t)(end - begin);
    if (n >= cap) n = cap - 1;
    memcpy(out, begin, n);
    out[n] = '\0';
}

// 若 line..eol 这一行是 ";@key=value"，则把 value 拷进 out 并返回 1；否则返回 0
static int32_t scheme_app_meta_match(const char *line, const char *eol, const char *key,
                                     char *out, size_t cap) {
    size_t klen = strlen(key);
    if ((size_t)(eol - line) < 3 + klen + 1) return 0;   // 至少 ";@key=x"
    if (line[0] != ';' || line[1] != '@') return 0;
    if (memcmp(line + 2, key, klen) != 0) return 0;
    if (line[2 + klen] != '=') return 0;
    scheme_app_meta_trim_value(line + 2 + klen + 1, eol, out, cap);
    return 1;
}

// 读取 path 指向的 App 源码头部的标题元数据（UTF-8）。失败/无元数据 → out_title 为空串。
static void scheme_app_read_meta_title(const char *path, char *out_title, size_t title_cap) {
    out_title[0] = '\0';
    if (platform_file_open(path) != 0) {
        return;
    }
    uint32_t fsize = platform_file_size();
    char buf[SCHEME_APP_META_PROBE_BYTES + 1];
    size_t want = (size_t)fsize;
    if (want > SCHEME_APP_META_PROBE_BYTES) want = SCHEME_APP_META_PROBE_BYTES;
    int32_t got = 0;
    if (want > 0) {
        got = platform_file_read((uint8_t *)buf, want);
    }
    platform_file_close();
    if (got <= 0) {
        return;
    }
    buf[got] = '\0';

    const char *p = buf;
    while (*p != '\0') {
        const char *eol = p;
        while (*eol != '\0' && *eol != '\n') eol++;
        if (scheme_app_meta_match(p, eol, "title", out_title, title_cap)) {
            return;
        }
        if (*eol == '\0') break;
        p = eol + 1;
    }
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
        // id：文件名去 .scm（ASCII，启动身份与 (App.launch) 名字）
        size_t idl = len - 4;
        if (idl > SCHEME_APP_ID_MAX - 1) idl = SCHEME_APP_ID_MAX - 1;
        memcpy(d->id, name, idl);
        d->id[idl] = '\0';
        // 相对路径：app/xxx.scm
        snprintf(d->source_rel, sizeof(d->source_rel), "app/%s", name);

        // 显示名：取源码头部元数据 ;@title=（见 ui_scheme_app.h 文件头的约定）。
        // 无元数据的旧 App / 第三方 App 回退为 id，不影响可运行性。
        // 截断到容量并给「● 运行中」角标留 8 个字符（s_item_text 拼角标用）。
        char title_utf8[SCHEME_APP_META_TITLE_MAX];
        char meta_path[320];
        snprintf(meta_path, sizeof(meta_path), "%s/%s", PLATFORM_ROOT_DIR, d->source_rel);
        scheme_app_read_meta_title(meta_path, title_utf8, sizeof(title_utf8));
        if (title_utf8[0] == '\0') {
            memcpy(title_utf8, d->id, idl + 1);
        }
        wchar_t tmp[SCHEME_APP_NAME_MAX];
        size_t tl = strlen(title_utf8);
        if (tl > SCHEME_APP_NAME_MAX - 9) tl = SCHEME_APP_NAME_MAX - 9;  // UTF-8 字节数作上限，保证 wchar 数不超
        am_mbstowcs(tmp, title_utf8, (uint32_t)tl);
        tmp[SCHEME_APP_NAME_MAX - 9] = L'\0';
        wcscpy(d->title, tmp);

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

    // 按 id 排序（ASCII）。菜单顺序因此稳定且与文件名一致：显示名是中文（元数据 title），
    // 按 wcscmp 排会得到无意义的码点序，且改一个标题就会重排整个菜单。
    // 冒泡；同步交换描述与菜单指针，保证两者下标一致。
    for (int32_t i = 0; i + 1 < s_app_count; i++) {
        for (int32_t j = 0; j + 1 < s_app_count - i; j++) {
            if (strcmp(s_apps[j].id, s_apps[j + 1].id) > 0) {
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

// 退出/切后台瞬间的在途触摸序列余波抑制（主菜单与列表菜单共用）：
// 返回 1 表示本帧触摸字段已被剥离（按键字段保留），调用方直接按剥离后的事件处理。
int32_t ui_scheme_app_touch_aftermath_filter(Key_Event *key_event) {
    if (!s_list_touch_guard) {
        return 0;
    }
    // 解除条件：① 出现**新的一次按下**（旧序列已彻底结束、用户开始新动作）；
    //           ② 抑制窗口到期（兜底，窗口内用户未再触摸）
    if ((key_event->touch_edge & TOUCH_EDGE_DOWN) != 0 || get_timestamp_in_ms() > s_list_touch_guard_until_ms) {
        s_list_touch_guard = 0;
        return 0;
    }
    key_event->is_touching = 0;
    key_event->touch_edge = 0;
    return 1;
}

int32_t ui_scheme_app_list_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 退出 App 后的在途触摸序列余波：剥掉触摸字段后再交给菜单（见 s_list_touch_guard 注释）
    Key_Event ke = *key_event;
    if (ui_scheme_app_touch_aftermath_filter(&ke)) {
        key_event = &ke;
    }
    // 菜单控件统一处理（上下键/触屏/惯性滚动；A 返回 STATE_GAME_MENU 由 back_state 给出）
    return ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main,
                                        ui_scheme_app_list_item_action,
                                        STATE_GAME_MENU, STATE_SCHEME_APP_LIST);
}

// ===============================================================================
// IL 缓存（编译产物落盘；等价于 Python 的 __pycache__）
//
// 为什么要有：冷启动的时间几乎全花在「把源码现场编译成 IL」上。WASM 实测 markdown
// （16 个模块 / 432KB / 10.5k 行）：parse 22ms + link 150ms + compile 620ms +
// load 18ms ≈ 810ms，而 App 自己解析首屏文档只要 ~150ms。编译结果本来就是**与本次
// 运行无关**的产物，只是解释器不落盘，于是每次进 App 都重算一遍（黑屏就发生在这段）。
//
// 做法（**不改 vendor**）：只调用它已导出的 am_module_dump / am_module_load
// （序列化 ilcode + AST 静态数据段 + 变量/符号表 + 闭包帧描述符表）。恢复出来的
// am_module_t 与编译产物完全等价，装载仍走 am_runtime_load_module 同一条路。
//   命中实测：load 6ms + 装载 23ms ≈ 29ms（对照 810ms），模块 924KB。
//
// 文件：<资源根>/data/ilcache/<app id>.ilc —— 一 App 一份，就地覆盖，无需淘汰。
//   · data/ 不在资源清单里（build_wasm_gui.sh 的 SKIP_DIRS），服务器永远不会覆盖它；
//   · WASM 目标写完后经 am_native_fs_on_write 弱钩子落进 IndexedDB（与 App 存档
//     同一条通路），刷新页面仍在；没有该钩子的平台是一行空操作。
//
// 失效判据（任一不符即当作未命中，编译成功后重写）：
//   · magic / format / ABI —— ABI 由 MB_ILCACHE_ABI 与几个结构体 sizeof 混成：
//     同步 vendor 或增删本地库注册后把 MB_ILCACHE_ABI +1；
//   · **来源清单**：每个参与编译的 .scm 的（相对路径, 字节数, 内容哈希）。清单由
//     链接器**实际读过**的文件当场记下（替换 am_link 的读源回调），不是自己扫 import
//     ——所以不会漏模块；源码改一个字节就失配（比 Python 的 mtime+size 更严）；
//   · payload 校验和 —— 防止「写一半掉电 / SD 卡拔出」留下的半截文件被当成有效 IL。
//
// 只缓存「编译确实贵」的 App：编译耗时 ≥ MB_ILCACHE_MIN_COMPILE_MS 且产物在
// [MIN,MAX] 字节之间才写盘。52 个 App 各塞一份小缓存没收益，而且 WASM 每次页面
// 加载都要把这些「本机文件」从 IndexedDB 灌回 MEMFS——不能为了 30ms 的 App 付这份钱。
//
// 降级：缓存侧任何失败（目录建不出、写失败、校验不过、内存不足、load 返回 NULL）
// 都只是「当作未命中」，App 照原路径编译，功能与行为不受影响。
// ===============================================================================

#if NANO_IS_M5ESP
    // ESP32：2MB 池下能跑的都是小 App（编译十几毫秒），而 SD 读一遍来源清单
    // 可能比编译本身还慢 → 关掉（零成本、零风险）。
    #define MB_ILCACHE_ENABLE 0
#elif defined(NANO_ILCACHE_OFF)
    #define MB_ILCACHE_ENABLE 0
#else
    #define MB_ILCACHE_ENABLE 1
#endif

#define MB_ILCACHE_DIR            "data/ilcache"
#define MB_ILCACHE_MAGIC          "AMILCH01"   // 8 字节
#define MB_ILCACHE_FORMAT         1u
#define MB_ILCACHE_ABI            1u   // 手工维护：同步 vendor / 改本地库注册表后 +1
#define MB_ILCACHE_HDR            32u  // 头长度（之后紧接来源清单，再之后是 payload）
#define MB_ILCACHE_MAX_MODULES    96u
#define MB_ILCACHE_MAX_PATH       200u
#define MB_ILCACHE_MIN_COMPILE_MS 250u
#define MB_ILCACHE_MIN_PAYLOAD    (128u * 1024u)
#define MB_ILCACHE_MAX_PAYLOAD    (6u * 1024u * 1024u)

#if MB_ILCACHE_ENABLE

// 一条来源记录：相对资源根（不带前导斜杠；验证时用 root + path 复现绝对路径）
typedef struct {
    char     path[MB_ILCACHE_MAX_PATH];
    uint32_t size;
    uint64_t hash;      // FNV-1a 64（文件原始字节，不做 UTF-8 解码）
} ilc_src_t;

// 链接回调的记账本
typedef struct {
    ilc_src_t item[MB_ILCACHE_MAX_MODULES];
    uint32_t  count;
    int32_t   overflow;     // 模块数超上限 → 放弃缓存（宁可不缓存，不可漏模块）
} ilc_ctx_t;

#define ILC_FNV_SEED 1469598103934665603ULL

static uint64_t ilc_fnv1a64(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = ILC_FNV_SEED;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// 头/清单一律按小端手工编解码：宿主从 wasm32 到 aarch64/x86_64 字节序都一样，
// 但结构体对齐/填充不同，所以不用 memcpy 结构体的写法。
static void ilc_put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
}
static void ilc_put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}
static void ilc_put_u64(uint8_t *p, uint64_t v) {
    ilc_put_u32(p, (uint32_t)(v & 0xFFFFFFFFu));
    ilc_put_u32(p + 4, (uint32_t)((v >> 32) & 0xFFFFFFFFu));
}
static uint16_t ilc_get_u16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
static uint32_t ilc_get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t ilc_get_u64(const uint8_t *p) {
    return (uint64_t)ilc_get_u32(p) | ((uint64_t)ilc_get_u32(p + 4) << 32);
}

// ABI 校验字：手工号 + 结构体尺寸 + 几个「哨兵 opcode」的当前值。
// vendor 改了指令/帧描述符布局、或增删重排了 opcode，即使忘了 bump 手工号，
// 这里多半也会因为 sizeof / opcode 编号变了而整体失效（多一层保险）。
static uint32_t ilc_abi(void) {
    uint32_t h = MB_ILCACHE_ABI * 2654435761u;
    h ^= (uint32_t)sizeof(am_instruction_t) * 131u;
    h ^= (uint32_t)sizeof(am_value_t) * 17u;
    h ^= (uint32_t)sizeof(am_frame_slot_t) * 7u;
    h ^= (uint32_t)sizeof(am_module_t) * 3u;
    h ^= (uint32_t)AM_VM_OP_halt;
    h ^= (uint32_t)AM_VM_OP_wind << 8;
    h ^= (uint32_t)AM_VM_OP_dynamicwind_done << 16;
    return h;
}

// 资源根（UTF-8，带尾斜杠）：把链接器给的绝对路径折回「相对资源根」
static const char *ilc_root_mb(void) {
    static char s_root_mb[256];
    if (s_root_mb[0] == '\0') {
        am_wcstombs(s_root_mb, s_app_root, (uint32_t)sizeof(s_root_mb) - 1);
    }
    return s_root_mb;
}

static void ilc_abs_path(const char *rel, char *out, size_t cap) {
    if (rel[0] == '/') {
        snprintf(out, cap, "%s", rel);
    }
    else {
        snprintf(out, cap, "%s%s", ilc_root_mb(), rel);
    }
}

// App id（文件名去 .scm，只留 [0-9A-Za-z_-]）：既是缓存文件名，也是日志里的名字。
// 极端情况下两个文件名只有非 [0-9A-Za-z_-] 字符不同（a-b.scm / a_b.scm）会撞同一个
// 缓存槽——后果只是两者互相覆盖、每次都重编，不影响正确性。
static void ilc_app_id(const char *source_rel, char *out, size_t cap) {
    const char *base = strrchr(source_rel, '/');
    base = (base != NULL) ? base + 1 : source_rel;
    size_t n = strlen(base);
    if (n > 4 && strcmp(base + n - 4, ".scm") == 0) n -= 4;
    if (n >= cap) n = cap - 1;
    for (size_t i = 0; i < n; i++) {
        char c = base[i];
        int32_t ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
                  || (c >= 'A' && c <= 'Z') || c == '_' || c == '-';
        out[i] = ok ? c : '_';
    }
    out[n] = '\0';
}

static void ilc_cache_path(const char *app_id, char *out, size_t cap) {
    snprintf(out, cap, "%s/%s/%s.ilc", PLATFORM_ROOT_DIR, MB_ILCACHE_DIR, app_id);
}

// 往记账本里加一条（按相对路径去重：同一模块被多处 import 只记一次）
static void ilc_note(ilc_ctx_t *ctx, const char *abs_mb, uint32_t size, uint64_t hash) {
    if (ctx == NULL) return;
    const char *root = ilc_root_mb();
    size_t rl = strlen(root);
    const char *rel = abs_mb;
    if (rl > 0 && strncmp(abs_mb, root, rl) == 0) rel = abs_mb + rl;
    // 链接器把 root 与 import 串直接相接，而 root 自带尾斜杠 → 会出现 "root//app/…"，
    // 剥前缀后留下一个前导 '/'。清单里统一用「相对资源根、无前导斜杠」的形式，
    // 验证阶段才能用 root + rel 复现同一个绝对路径。
    while (*rel == '/' || *rel == '\\') rel++;
    if (rel[0] == '\0' || strlen(rel) >= MB_ILCACHE_MAX_PATH) {
        ctx->overflow = 1;
        return;
    }
    for (uint32_t i = 0; i < ctx->count; i++) {
        if (strcmp(ctx->item[i].path, rel) == 0) return;
    }
    if (ctx->count >= MB_ILCACHE_MAX_MODULES) {
        ctx->overflow = 1;
        return;
    }
    ilc_src_t *it = &ctx->item[ctx->count++];
    strcpy(it->path, rel);
    it->size = size;
    it->hash = hash;
}

// am_link 的读源回调：读完**顺手记账**（这是「缓存来源清单」的唯一数据来源）。
// 语义与 vendor 的 am_host_read_source_from_file 一致：返回 am_malloc 的宽字符源码。
static wchar_t *ilc_read_source(am_allocator_t *alloc, const wchar_t *abs_path, void *user_data) {
    if (!alloc || !abs_path) return NULL;
    char mb[384];
    mb[0] = '\0';
    am_wcstombs(mb, abs_path, (uint32_t)sizeof(mb) - 1);
    if (mb[0] == '\0') return NULL;

    uint8_t *bytes = NULL;
    size_t size = 0;
    if (platform_read_file_to_buffer(mb, &bytes, &size) != 0 || bytes == NULL || size == 0) {
        if (bytes) free(bytes);
        return NULL;
    }
    ilc_note((ilc_ctx_t *)user_data, mb, (uint32_t)size, ilc_fnv1a64(bytes, size));

    // UTF-8 → 宽字符（同 am_read_file_to_wchar 的语义：非法字节退化为 '?' 并截断）
    wchar_t *code = (wchar_t *)am_malloc(alloc, (size + 1) * sizeof(wchar_t));
    if (code == NULL) {
        free(bytes);
        return NULL;
    }
    am_mbstowcs(code, (const char *)bytes, (uint32_t)size);
    code[size] = L'\0';
    free(bytes);
    return code;
}

// 校验来源清单：每个模块都按（路径, 字节数, 内容哈希）复核一遍。
// 命中成本 = 读一遍闭包源码（markdown 432KB ≈ 2ms），远小于重新编译。
static int32_t ilc_verify_manifest(const uint8_t *buf, size_t size, uint32_t count, size_t *off_io) {
    size_t off = *off_io;
    for (uint32_t i = 0; i < count; i++) {
        if (off + 14u > size) return 0;
        uint32_t sz = ilc_get_u32(buf + off);
        uint64_t hash = ilc_get_u64(buf + off + 4);
        uint16_t plen = ilc_get_u16(buf + off + 12);
        off += 14u;
        if (plen == 0 || plen >= MB_ILCACHE_MAX_PATH || off + plen > size) return 0;
        char rel[MB_ILCACHE_MAX_PATH];
        memcpy(rel, buf + off, plen);
        rel[plen] = '\0';
        off += plen;

        char abs[384];
        ilc_abs_path(rel, abs, sizeof(abs));
        uint8_t *fb = NULL;
        size_t fs = 0;
        if (platform_read_file_to_buffer(abs, &fb, &fs) != 0 || fb == NULL) {
            if (fb) free(fb);
            return 0;                      // 模块被删/被改名 → 未命中
        }
        uint64_t disk_hash = ilc_fnv1a64(fb, fs);
        int32_t ok = ((uint32_t)fs == sz) && (disk_hash == hash);
        if (!ok) {                         // 模块改过（或被删/改短）→ 未命中
            printf("[scheme-app] ILCACHE stale：来源不符 %s（cache %u/%08x disk %u/%08x）\n",
                   rel, (uint32_t)sz, (uint32_t)hash, (uint32_t)fs, (uint32_t)disk_hash);
            free(fb);
            return 0;
        }
        free(fb);
    }
    *off_io = off;
    return 1;
}

// 命中则返回可直接 am_runtime_load_module 的模块（调用方负责释放），否则 NULL。
static am_module_t *ilc_try_load(const char *app_id) {
    char path[320];
    ilc_cache_path(app_id, path, sizeof(path));

    uint8_t *buf = NULL;
    size_t size = 0;
    if (platform_read_file_to_buffer(path, &buf, &size) != 0 || buf == NULL) {
        printf("[scheme-app] ILCACHE miss %s（无缓存文件 %s）\n", app_id, path);
        return NULL;
    }
    if (size < MB_ILCACHE_HDR || memcmp(buf, MB_ILCACHE_MAGIC, 8) != 0) {
        printf("[scheme-app] ILCACHE stale %s（头不符 size=%u）\n", app_id, (uint32_t)size);
        free(buf);
        return NULL;
    }
    if (ilc_get_u32(buf + 8) != MB_ILCACHE_FORMAT || ilc_get_u32(buf + 12) != ilc_abi()) {
        free(buf);
        printf("[scheme-app] ILCACHE stale %s（format/ABI 不符）\n", app_id);
        return NULL;
    }
    uint32_t count = ilc_get_u32(buf + 16);
    uint32_t payload = ilc_get_u32(buf + 20);
    uint32_t sum = ilc_get_u32(buf + 24);
    if (count == 0 || count > MB_ILCACHE_MAX_MODULES || payload == 0
        || (size_t)MB_ILCACHE_HDR + payload > size) {
        printf("[scheme-app] ILCACHE stale %s（头字段不合理 n=%u payload=%u size=%u）\n",
               app_id, count, payload, (uint32_t)size);
        free(buf);
        return NULL;
    }
    size_t off = MB_ILCACHE_HDR;
    int32_t ok = ilc_verify_manifest(buf, size, count, &off);
    if (!ok) {
        free(buf);
        return NULL;                       // 具体哪一条不符已由 ilc_verify_manifest 打印
    }
    if (off + payload != size) {
        printf("[scheme-app] ILCACHE stale %s（长度不符 off=%u payload=%u size=%u）\n",
               app_id, (uint32_t)off, (uint32_t)payload, (uint32_t)size);
        free(buf);
        return NULL;
    }
    if ((uint32_t)ilc_fnv1a64(buf + off, payload) != sum) {
        free(buf);
        printf("[scheme-app] ILCACHE stale %s（校验和错）\n", app_id);
        return NULL;
    }
    am_module_t *mod = am_module_load(s_app->vm_alloc, s_app->vm_alloc, buf, off);
    free(buf);
    if (mod == NULL) {
        printf("[scheme-app] ILCACHE stale %s（am_module_load 失败）\n", app_id);
    }
    return mod;
}

// WASM 目标的落盘钩子（弱符号：其他平台链接为空，一行空操作）
extern void am_native_fs_on_write(const char *path, const uint8_t *data, size_t size) __attribute__((weak));

// 编译产物写盘（best effort）。compile_ms 只在日志里有意义——是否写盘由调用方判定。
static void ilc_store(const char *app_id, ilc_ctx_t *ctx, am_module_t *mod) {
    if (ctx == NULL || ctx->overflow || ctx->count == 0) return;

    size_t payload = am_module_dump(s_app->vm_alloc, s_app->vm_alloc, mod, NULL, 0);
    if (payload == SIZE_MAX) {
        printf("[scheme-app] ILCACHE skip %s（序列化尺寸计算失败）\n", app_id);
        return;
    }
    if (payload < MB_ILCACHE_MIN_PAYLOAD || payload > MB_ILCACHE_MAX_PAYLOAD) {
        printf("[scheme-app] ILCACHE skip %s（payload %uKB 不在 [%u,%u]KB）\n", app_id,
               (uint32_t)(payload / 1024u), (uint32_t)(MB_ILCACHE_MIN_PAYLOAD / 1024u),
               (uint32_t)(MB_ILCACHE_MAX_PAYLOAD / 1024u));
        return;
    }

    size_t manifest = 0;
    for (uint32_t i = 0; i < ctx->count; i++) manifest += 14u + strlen(ctx->item[i].path);
    size_t total = MB_ILCACHE_HDR + manifest + payload;
    uint8_t *buf = (uint8_t *)malloc(total);
    if (buf == NULL) return;

    memcpy(buf, MB_ILCACHE_MAGIC, 8);
    ilc_put_u32(buf + 8, MB_ILCACHE_FORMAT);
    ilc_put_u32(buf + 12, ilc_abi());
    ilc_put_u32(buf + 16, ctx->count);
    ilc_put_u32(buf + 20, (uint32_t)payload);
    ilc_put_u32(buf + 24, 0);            // 校验和：payload 序列化完再填
    ilc_put_u32(buf + 28, 0);

    size_t off = MB_ILCACHE_HDR;
    for (uint32_t i = 0; i < ctx->count; i++) {
        const ilc_src_t *it = &ctx->item[i];
        uint16_t plen = (uint16_t)strlen(it->path);
        ilc_put_u32(buf + off, it->size);
        ilc_put_u64(buf + off + 4, it->hash);
        ilc_put_u16(buf + off + 12, plen);
        memcpy(buf + off + 14, it->path, plen);
        off += 14u + plen;
    }

    // am_module_dump 的偏移都相对 payload 起点，所以直接把 payload 写在清单之后
    size_t written = am_module_dump(s_app->vm_alloc, s_app->vm_alloc, mod, buf, off);
    if (written != payload) {
        printf("[scheme-app] ILCACHE skip %s（序列化长度不符）\n", app_id);
        free(buf);
        return;
    }
    ilc_put_u32(buf + 24, (uint32_t)ilc_fnv1a64(buf + off, payload));

    char dir[320];
    snprintf(dir, sizeof(dir), "%s/data", PLATFORM_ROOT_DIR);
    platform_mkdir(dir);                              // 已存在即成功；失败交给下面的写盘判定
    snprintf(dir, sizeof(dir), "%s/%s", PLATFORM_ROOT_DIR, MB_ILCACHE_DIR);
    platform_mkdir(dir);

    char path[320];
    ilc_cache_path(app_id, path, sizeof(path));
    if (platform_write_buffer_to_file(path, buf, total) != 0) {
        printf("[scheme-app] ILCACHE write fail %s\n", path);
        free(buf);
        return;
    }
    if (am_native_fs_on_write) am_native_fs_on_write(path, buf, total);   // WASM：进 IndexedDB
    printf("[scheme-app] ILCACHE store %s: modules=%u payload=%uKB total=%uKB\n",
           app_id, ctx->count, (uint32_t)(payload / 1024u), (uint32_t)(total / 1024u));
    free(buf);
}

#endif  // MB_ILCACHE_ENABLE

// ===============================================================================
// 通用「正在启动」画面（所有灵机 App 共用；宿主侧机制，App 一行不用写）
//
// 为什么需要：编译/链接是**一次不可中断的同步调用**，WASM 单线程下这期间浏览器
// 完全无法重绘。旧做法是清成纯黑再进编译，用户盯着黑屏 ~1s，没有任何反馈。
// 现在改成先画这一帧（标题 + 状态 + 忙条）并推屏，再进编译——用户立刻知道
// 「正在启动哪个 App、机器在忙」，而不是怀疑死机。
// 缓存命中时它只闪一下（甚至来不及看到），因此不额外增加观感负担。
// ===============================================================================
static void app_draw_start_screen(Nano_GFX *gfx, const wchar_t *title, int32_t first_time) {
    if (gfx == NULL) return;
    int32_t w = (int32_t)gfx->width;
    int32_t h = (int32_t)gfx->height;
    int32_t cx = w / 2;

    gfx_draw_rectangle(gfx, 0, 0, (uint32_t)w, (uint32_t)h, 14, 17, 22, 1);   // 与阅读页同底色
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_16, (wchar_t *)title,
                               cx, h / 2 - 22, 236, 242, 250, 1);
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12,
                               first_time ? L"正在启动…（首次启动需准备，请稍候）" : L"正在启动…",
                               cx, h / 2 + 2, 150, 168, 190, 1);
    // 忙条：底槽 + 亮段（静态；单线程下无法做动画，但它明确表示「在干活」）
    int32_t bar_w = (w * 2) / 5;
    if (bar_w < 60) bar_w = 60;
    gfx_draw_rectangle(gfx, (uint32_t)(cx - bar_w / 2), (uint32_t)(h / 2 + 24),
                       (uint32_t)bar_w, 3, 40, 48, 62, 1);
    gfx_draw_rectangle(gfx, (uint32_t)(cx - bar_w / 2), (uint32_t)(h / 2 + 24),
                       (uint32_t)(bar_w / 3), 3, 96, 160, 235, 1);
    gfx_refresh(gfx);
}

// ===============================================================================
// 运行宿主：读取源码 → 解析 → 链接 → 编译 → 加载进程
// ===============================================================================

static void app_load_source(Nano_GFX *gfx) {
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

#if MB_ILCACHE_ENABLE
    // 入口文件的哈希必须**现在**算：file_buffer 转成宽字符后就被释放了。
    uint64_t entry_hash = ilc_fnv1a64(file_buffer, file_size);
#else
    uint64_t entry_hash = 0;
#endif

    char app_id[MB_ILCACHE_MAX_PATH];
#if MB_ILCACHE_ENABLE
    ilc_app_id(s_app->source_rel, app_id, sizeof(app_id));

    // ---- ① IL 缓存命中：跳过 parse/link/compile，直接装载序列化产物 ----
    {
        uint64_t t_hit0 = get_timestamp_in_ms();
        am_module_t *cached = ilc_try_load(app_id);
        if (cached != NULL) {
            s_app->pid = am_runtime_load_module(s_app->rt, cached);
            if (cached->ilcode) am_free(s_app->vm_alloc, cached->ilcode);
            if (cached->ast) am_ast_destroy(cached->ast);
            am_free(s_app->vm_alloc, cached);
            free(file_buffer);
            if (s_app->pid != (am_pid_t)-1) {
                printf("[scheme-app] ILCACHE hit %s：装载 %ums（省去编译）\n",
                       app_id, (uint32_t)(get_timestamp_in_ms() - t_hit0));
                return;
            }
            // 装载失败（内存不足等）→ 落到编译路径重试一次
            printf("[scheme-app] ILCACHE hit %s 但装载失败，回退编译\n", app_id);
            s_app->pid = (am_pid_t)-1;
        }
    }
#endif
    (void)app_id;
    (void)entry_hash;

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

#if MB_ILCACHE_ENABLE
    // 未命中：先把「正在启动」画上屏，再做这段会阻塞主线程的编译
    //（首次启动没有缓存，这一帧要顶住几百毫秒到一秒）
    app_draw_start_screen(gfx, s_app->title, 1);
    uint64_t t_compile0 = get_timestamp_in_ms();
    ilc_ctx_t ilc;
    memset(&ilc, 0, sizeof(ilc));
    // 入口文件本身不经过链接器的读源回调，单独记账（清单里永远第 0 条）
    {
        char abs_entry[384];
        ilc_abs_path(s_app->source_rel, abs_entry, sizeof(abs_entry));
        ilc_note(&ilc, abs_entry, (uint32_t)file_size, entry_hash);
    }
#else
    (void)gfx;
#endif

    am_ast_t *ast = am_parse(s_app->vm_alloc, code_w, mod_name_w, 0);
    if (!ast) {
        wcscpy(s_app->err_load, L"Scheme 语法解析失败");
        s_app->load_failed = 1;
        return;
    }

#if MB_ILCACHE_ENABLE
    am_ast_t *linked = am_link(ast, s_app_root, ilc_read_source, &ilc);
#else
    am_ast_t *linked = am_link(ast, s_app_root, am_host_read_source_from_file, NULL);
#endif
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

#if MB_ILCACHE_ENABLE
    // ---- ④ 写缓存（必须在释放 mod 之前：序列化要读它的 ilcode / AST / 帧表）----
    // 只在「这次编译确实贵」时落盘：快 App 不缓存，免得把 IndexedDB 塞满。
    {
        uint64_t compile_ms = get_timestamp_in_ms() - t_compile0;
        if (compile_ms >= MB_ILCACHE_MIN_COMPILE_MS) {
            ilc_store(app_id, &ilc, mod);
            printf("[scheme-app] ILCACHE compile %s：%ums（已写缓存）\n", app_id, (uint32_t)compile_ms);
        }
        else {
            printf("[scheme-app] ILCACHE compile %s：%ums（够快，不缓存）\n", app_id, (uint32_t)compile_ms);
        }
    }
#endif

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
    // Raster 本地库直写同一块帧缓冲（App 自己 (Raster.dispose) 负责释放内部缓冲）
    am_native_raster_bind(global_state->gfx);
    // (App.launch*) 的名字存在性校验钩子（App 库是进程级单例状态，只需注册一次）
    am_native_app_bind_lookup(app_host_lookup_app);

    // ① 同一个 App 已有实例（后台冻结中）→ 直接恢复，不重建解释器
    inst = app_instance_find(s_apps[s_selected].source_rel);
    if (inst != NULL) {
        s_app = inst;
        // **从应用列表进来 = 顶层**：抹掉旧的返回链，退出时回列表（而不是跳回上一个 App）。
        // 规则只有一句：返回链只由「拉起」建立，进列表即复位。（“后台”按钮 = 回列表，
        // 再从列表点进来当然就是一层新的顶层。）
        // 先清 back_rel 再 foreground：foreground 会在 am_native_app_reset 之后据此
        // 重新置“退出后回到谁”的提示（这里就是置空）。
        inst->back_rel[0] = '\0';
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
    // 【注意】增删或重排下面这批注册（以及 IL 缓存里 native 绑定对它们的依赖）后，
    //   必须把文件头的 MB_ILCACHE_ABI +1，否则旧缓存里的 native 绑定会串位。
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
    // Chat：鹦鹉笼端侧 LLM 推理引擎（模型/分词器/推理内核/推理可视化，见 am_native_Chat.h）。
    // 引擎是**虚拟机全局单例**（与 C 版鹦鹉笼一致），实例销毁时由 on_exit 释放。
    am_runtime_register_native_lib(s_app->rt, &am_native_Chat_lib);
    am_native_ui_abort(); // 清掉上一个 App 可能遗留的请求
    am_native_app_reset(s_app->title);
    // 返回链提示（“退出后回到谁”）必须在 reset **之后**重新告知，而且要在
    // app_load_source() 之前——被拉起冷启动时，App 的 boot（(EX.install)）就要用
    // (App.back_to) 决定把自己的退出按钮画成「返回」还是「退出」。
    app_apply_back_hint(s_app);
    am_native_Key_reset();
    // Gfx 的「绘制设置」：字模/裁剪是 am_native_Gfx.c 与 Nano_GFX 上的宿主单例状态，
    // 不随实例销毁而复位。不复位则上一个 App 的 (Gfx.set_font 0) 会被本 App 继承
    // （先跑 gol 再跑 2048 → 2048 的文字变成 12px 点阵，见 am_native_Gfx_reset 的注释）。
    // 只在**冷启动**复位：resume 路径（app_instance_foreground）不复位，否则会打断
    // "实例内设过字模就不再重设"的 App。
    am_native_gfx_reset();
    printf("[scheme-app] Gfx 绘制设置已复位（字模→默认 12px 抗锯齿）\n");
    // Raster：水池的 3D 软光栅**后端**（薄 C / 重 Scheme 版，见 app/waterpool_raster.scm）。
    // 只持帧缓冲/z-buffer/波场/焦散纹理/已上传网格；相机、几何、物理、交互、HUD 在 Scheme。
    am_runtime_register_native_lib(s_app->rt, &am_native_Raster_lib);
    // Music：音乐播放引擎（文件枚举 + WAV/MP3 解码 + 随机定位 + 扬声器喂入，见 am_native_Music.h）。
    // 解码与 PCM 全在 C 侧，Scheme 只拿进度/时长/状态；无音频设备时自动转"静音走带"。
    am_runtime_register_native_lib(s_app->rt, &am_native_Music_lib);
    // Audio：音频**输入**（开麦 + 加窗 + FFT + 对数幅度映射，见 am_native_Audio.h）。
    // 引擎是虚拟机全局单例；FFT 整条流水线在 C 侧（Scheme 里 1024 点 FFT 会掉到个位数 fps），
    // Scheme 只拿每频点 0~255 强度与 RMS，声谱图/示波器类 App 据此绘图。
    am_runtime_register_native_lib(s_app->rt, &am_native_Audio_lib);

    // 清屏后加载并启动 Scheme 进程（首帧由 App 自行绘制接管）。
    // 未命中 IL 缓存时 app_load_source() 会在编译前把「正在启动 <App>…」画上屏
    // （见 app_draw_start_screen）——否则用户看到的是编译期间的一整段纯黑。
    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);

    app_load_source(global_state->gfx);
    if (s_app->load_failed) {
        printf("[scheme-app] %ls 源码构建/加载失败\n", s_app->title);
    }
    else {
        s_app->alive = 1;
    }
    // 冷启动的实例同样是前台（eviction 只驱逐后台实例）
    s_app->error_drawn = 0;   // 本实例的错误画面还没画过（旧实现是函数内 static，第二次加载失败不重画）
    for (int32_t i = 0; i < SCHEME_APP_INST_MAX; i++) s_inst[i].foreground = 0;
    s_app->foreground = 1;
    s_app->suspended = 0;
    s_app->last_active_ms = get_timestamp_in_ms();
    am_native_app_set_active(1);
    // 冷启动同样是在“手指还按着”的时候发生的（点列表项 / 点 filemgr 里的文件行）→
    // 丢弃这段在途手势，否则新 App 的 boot/第一拍会凭空多出一个按下沿（一次误点）。
    am_native_Touch_drop_inflight();
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
            if (s_app->err_buf[0] != L'\0') {
                printf("[scheme-app][error] %ls\n", s_app->err_buf);
                // 运行期错误的现场：出错 PC 附近的 IL 窗口（排查「指令执行异常」最有效）
                am_process_t *p = am_runtime_get_process(s_app->rt, s_app->pid);
                if (p && p->ilcode) {
                    unsigned pc = (unsigned)p->PC;
                    unsigned lo = pc > 8 ? pc - 8 : 0;
                    unsigned hi = pc + 8 < p->ilcode_length ? pc + 8 : (unsigned)p->ilcode_length;
                    printf("[scheme-app][error] PC=%u opstack=%zu fstack=%zu\n", pc,
                           (size_t)(p->opstack_top - p->opstack),
                           (size_t)(p->fstack_top - p->fstack));
                    for (unsigned i = lo; i < hi; i++)
                        printf("   %s%u op=%u operand=%zu\n", i == pc ? ">>" : "  ",
                               i, (unsigned)p->ilcode[i].opcode, (size_t)p->ilcode[i].operand);
                }
            }
            s_app->err_buf[0] = L'\0';
            continue;
        }
        if (len + 1 < MB_ERR_BUF_LEN) {
            s_app->err_buf[len] = c;
            s_app->err_buf[len + 1] = L'\0';
        }
    }
}

// 把用户输出 FIFO（(display ...) / (newline)）排空并打到宿主 stdout。
//
// 为什么需要：灵机程序里的 (display ...) 并不直接写 stdout，而是把字符逐个压进
// rt->output_fifo，由**宿主**负责取出落地（见 animac_core.h 的 am_runtime_output）。
// 本宿主此前只排空了 error_fifo，于是 App 的调试输出在 GUI 目标（掌机/浏览器）上
// 人间蒸发——排查只能靠改 C 代码加 printf。这里统一按 UTF-8 打印，
// 使 (display ...) 在浏览器控制台 / 串口 / 终端里都可见（端到端测试也据此断言）。
// 代价：只有真有输出时才进入本函数，逐字符转换；App 里 display 极少，可忽略。
static void app_drain_output_fifo(void) {
    if (!s_app->rt || !s_app->rt->output_fifo) return;
    char line[256];
    size_t used = 0;
    while (s_app->rt->output_fifo->length > 0) {
        am_value_t v = am_list_shift(s_app->vm_alloc, s_app->rt->output_fifo);
        if (!am_value_is_wchar(v)) continue;
        wchar_t one[2] = { (wchar_t)am_value_to_wchar(v), L'\0' };
        char mb[8];
        uint32_t l = am_wcstombs(mb, one, (uint32_t)sizeof(mb) - 1);
        if (l == 0) continue;
        if (used + l + 1 >= sizeof(line)) {
            line[used] = '\0';
            fputs(line, stdout);
            used = 0;
        }
        memcpy(line + used, mb, l);
        used += l;
    }
    if (used > 0) {
        line[used] = '\0';
        fputs(line, stdout);
        fflush(stdout);
    }
}

int32_t ui_scheme_app_render_frame(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    Nano_GFX *gfx = global_state->gfx;
    if (s_app == NULL) return 0;


    if (s_app->load_failed) {
        // 失败画面：黑底，标题 + 原因 + 提示（每实例只画一次，等 A 退出）
        if (!s_app->error_drawn) {
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
            s_app->error_drawn = 1;
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
        app_drain_output_fifo();

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
            // 加载失败/无运行时：没有 Scheme 代码可以接管返回请求 → 宿主自己退出本实例。
            // 不再直接改 STATE_SCHEME_APP_LIST：走与 (App.exit) 同一个帧末漏斗，
            // 这样“被拉起但装不下”的 App 按 A 也回到**拉起它的 App**（而不是被扔回列表）。
            s_host_exit_request = 1;
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

    if (s_app == NULL) return 0;

    // 宿主自己发起的退出（目前只有“加载失败时按 A”）：取走即清零，立即走退出流程。
    // 放在 rt 判空之前——加载失败的实例本来就可能没有运行时。
    if (s_host_exit_request) {
        s_host_exit_request = 0;
        return 1;
    }

    if (s_app->rt == NULL) return 0;

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

// 按实例的 back_rel 刷新“退出后回到谁”的提示（必须在 am_native_app_reset 之后调）。
// 这一个值同时供两个用途：(App.exit) 的确认框标题「退出并返回 X？」与 (App.back_to)。
// 链空或父实例已不在实例表 → 置空（= 顶层，退出回应用列表）。
// 调用点只有两处——实例上前台的两种方式：app_instance_foreground() 与
// app_instance_cold_start()，且都在自己的 am_native_app_reset() 之后立即调用。
static void app_apply_back_hint(UI_Scheme_App_State *inst) {
    UI_Scheme_App_State *back;
    if (inst == NULL || inst->back_rel[0] == '\0') {
        am_native_app_set_back_hint(NULL);
        return;
    }
    back = app_instance_find(inst->back_rel);
    am_native_app_set_back_hint(back != NULL ? back->title : NULL);
}

// 把实例提为前台：恢复本地库的"当前 App"上下文、解除挂起、通知 App 自己被恢复了
//
// 注意：本函数**不动 back_rel**（返回链由调用的地方决定）：
//   · 从应用列表恢复 → 视为顶层，调用方**先**清空 back_rel 再调本函数（见 ui_scheme_app_init）；
//   · 拉起另一个 App → 调用方（perform_launch）在**本函数之前**写好 back_rel；
//   · 回上一 App   → 父实例的 back_rel 不动（它可能自己也是被拉起的，见 perform_exit）。
// 本函数在 am_native_app_reset 之后自己调 app_apply_back_hint()，把“退出后回到谁”
// 重新告知本地库——App 在 (App.resumed) 那一拍重读 (App.back_to) 就能刷新按钮文案。
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
    // 提示要在 reset 之后（reset 会清空），且在恢复/返回后可能**变**（进顶层变 NULL、
    // 返回上一 App 变成它自己的父）：App 在 (App.resumed) 那拍重读 (App.back_to)。
    app_apply_back_hint(inst);
    // 切换发生在这个 App **看不到的那段时间**里（用户可能正按着手指：点列表项、
    // 点退出确认框的「确认」、点另一个 App 里的文件行）。那段手势不属于本 App，
    // 必须整段丢掉，否则恢复后的第一拍会凭空多出一个按下沿（→ 一次误点）。
    // 见 am_native_Touch.h 的 am_native_Touch_drop_inflight。
    am_native_Touch_drop_inflight();
}

// 挂起实例（后台或锁屏）：不再泵它，(App.is_active) 变 #f；运行时与内存池原地保留
static void app_instance_suspend(UI_Scheme_App_State *inst) {
    if (inst == NULL) return;
    inst->suspended = 1;
    inst->foreground = 0;
    am_native_app_set_active(0);
    am_native_ui_abort();   // 挂起期间的模态请求作废（进程随后不再被泵）
}

// 销毁前台实例的公共收尾（杀进程 → 销毁运行时/内存池 → 解绑 Gfx → 清模态与请求位）。
// 两条出口共用：
//   · 回应用列表（ui_scheme_app_on_exit）—— 额外置列表触摸守卫；
//   · 回上一 App（ui_scheme_app_perform_exit）—— 不需要列表守卫（App 侧由
//     am_native_Touch_drop_inflight 负责在途手势）。
static void app_teardown_foreground(void) {
    UI_Scheme_App_State *inst = s_app;

    // 丢弃可能挂起的模态输入/退出请求（进程随后会被销毁，不能留悬空 PID）
    am_native_ui_abort();
    am_native_app_abort();
    // 释放 (native Chat) 的推理引擎（模型权重 + KV-Cache 可达数百 MB，不随解释器池回收；
    // Scheme 进程已销毁，留着只会挤占内存。恢复后台实例时 App 会按 (Chat.loaded) 重新载入）
    am_native_chat_release();
    s_modal_active = 0;
    s_modal_ignore_longpress_repeat = 0;
    s_modal_open_key = 0;
    s_modal_open_key_deadline_ms = 0;
    s_modal_in = NULL;
    s_modal_ta = NULL;
    s_modal_kind = AM_NATIVE_UI_KIND_NONE;
    s_modal_confirm_with_body = 0;
    s_modal_confirm_with_bg = 0;
    s_host_exit_request = 0;

    app_instance_destroy(inst);
    // 实例已销毁：把绘制设置交还默认值，避免退出后的菜单/列表继承 App 的字模/裁剪
    am_native_gfx_reset();
    am_native_gfx_bind(NULL);
}

// 退出当前前台实例（销毁）；供 ui_app.c 在 (App.exit) 后调用
void ui_scheme_app_on_exit(void) {
    app_teardown_foreground();

    // 退出可能发生在手指动作过程中：列表紧接着收到的"余波"（旧序列的电平与迟到的
    // UP 边沿）必须吞掉，直到用户真正开始新的一次按下或窗口到期（见 s_list_touch_guard）
    s_list_touch_guard = 1;
    s_list_touch_guard_until_ms = get_timestamp_in_ms() + 600;
}

// 结束当前前台实例：有活着的返回链 → 回到上一 App（状态机仍是 STATE_SCHEME_APP_RUN），
// 返回 1；否则销毁并回**主菜单**，返回 0（调用方把 STATE 切到 STATE_MAIN_MENU）。
//
// 顺序很关键：**先**把 back_rel 读出来、**先**解析父实例指针，再销毁当前实例
// （app_instance_destroy 会 memset 本槽）。
int32_t ui_scheme_app_perform_exit(Global_State *global_state) {
    UI_Scheme_App_State *inst = s_app;
    UI_Scheme_App_State *caller = NULL;
    wchar_t inst_title[SCHEME_APP_NAME_MAX];
    char back[192];
    size_t back_len = 0;

    back[0] = '\0';
    inst_title[0] = L'\0';
    if (inst != NULL) {
        // 定长拷贝（不依赖 strncpy 的 NUL 补齐行为，静态分析器也看得懂）
        back_len = strlen(inst->back_rel);
        if (back_len >= sizeof(back)) back_len = sizeof(back) - 1;
        memcpy(back, inst->back_rel, back_len);
        back[back_len] = '\0';
        wcsncpy(inst_title, inst->title, SCHEME_APP_NAME_MAX - 1);
        inst_title[SCHEME_APP_NAME_MAX - 1] = L'\0';
        if (back[0] != '\0') caller = app_instance_find(back);
        if (caller == inst) caller = NULL;   // 防御：链指向自己（不应发生）
    }

    app_teardown_foreground();

    if (caller == NULL) {
        if (back[0] != '\0') {
            printf("[scheme-app] 返回链已断（%s 已不在实例表，可能被驱逐），回应用列表\n", back);
        }
        // 回列表：与 ui_scheme_app_on_exit 同一份在途触摸余波抑制
        s_list_touch_guard = 1;
        s_list_touch_guard_until_ms = get_timestamp_in_ms() + 600;
        return 0;
    }

    // 回上一 App：与“在应用列表里选中 ● 运行中”完全同一条恢复路径（实例/状态全保留）
    s_app = caller;
    am_native_gfx_bind(global_state->gfx);
    am_native_raster_bind(global_state->gfx);
    app_instance_foreground(caller);
    global_state->is_full_refresh = 1;
    printf("[scheme-app] 退出 %ls → 返回 %ls（实例保留，状态不变）\n", inst_title, caller->title);
    return 1;
}

// 切后台：保留实例与内存池，只把前台让出去（回主菜单，该 App 图标带运行角标）
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

// ===============================================================================
// 主菜单表驱动：目录视图（扫描 → 逐项查询 → 启动）
//
// 主菜单把 app/ 下的每个 Scheme App 也当作一个图标，标题取自源码元数据 ;@title=。
// 与列表菜单共用同一份扫描结果（s_apps）：进入主菜单时 refresh 一次，
// 随后逐项取标题/id/后台运行状态。**标题指针在下一次 refresh 后失效**——
// 调用方（ui_app.c 的条目表）因此必须在 refresh 之后立即重建。
// ===============================================================================
int32_t ui_scheme_app_catalog_refresh(void) {
    if (scheme_app_buffers_init() != 0) {
        s_app_count = 0;
        return 0;
    }
    scheme_app_scan();
    return s_app_count;
}

int32_t ui_scheme_app_catalog_count(void) {
    return s_app_count;
}

const wchar_t *ui_scheme_app_catalog_title(int32_t index) {
    if (index < 0 || index >= s_app_count) return L"";
    return s_apps[index].title;
}

const char *ui_scheme_app_catalog_id(int32_t index) {
    if (index < 0 || index >= s_app_count) return "";
    return s_apps[index].id;
}

int32_t ui_scheme_app_catalog_running(int32_t index) {
    if (index < 0 || index >= s_app_count) return 0;
    return ui_scheme_app_has_live_instance(s_apps[index].source_rel);
}

int32_t ui_scheme_app_catalog_enter(int32_t index) {
    if (index < 0 || index >= s_app_count) {
        return STATE_MAIN_MENU;  // 越界（目录刚变）：留在主菜单
    }
    // 与列表菜单选中同一条路径：记下选中项，状态机转到运行态后由 ui_scheme_app_init
    // 决定恢复后台实例还是冷启动
    s_selected = index;
    return STATE_SCHEME_APP_RUN;
}

// ===============================================================================
// 拉起另一个 App（App → App）
//
// 机制说明见 am_native_App.h 与 ui_scheme_app.h。本段只负责“帧末那一下子”：
// 定槽 → 让出前台 → 目标上前台 → 交付启动参数。
// ===============================================================================

// 名字 → s_apps[] 下标。接受三种写法：“markdown” / “markdown.scm” / “app/markdown.scm”。
// 比较对象是列表**显示名**（扫描时已去 .scm 并按 SCHEME_APP_NAME_MAX-9 截断），
// 所以超长文件名需用截断后的名字——这是刻意的：可拉起集合与列表里能点到的集合一致。
static int32_t app_desc_find_name(const wchar_t *name) {
    wchar_t base[SCHEME_APP_NAME_MAX * 2];
    size_t n = 0;
    size_t start = 0;
    size_t m = 0;
    int32_t i;

    if (name == NULL || s_apps == NULL || s_app_count <= 0) return -1;
    n = wcslen(name);
    if (n == 0 || n >= sizeof(base) / sizeof(base[0])) return -1;

    // 取 basename（容忍 "app/xxx.scm" 写法）
    for (size_t k = 0; k < n; k++) {
        if (name[k] == L'/') start = k + 1;
    }
    for (size_t k = start; k < n; k++) base[m++] = name[k];
    base[m] = L'\0';

    // 去 ".scm" 后缀（不区分大小写，与 scanning 时的过滤一致）
    if (m >= 4
        && base[m - 4] == L'.'
        && ascii_lower((char)(base[m - 3] & 0xFF)) == 's'
        && ascii_lower((char)(base[m - 2] & 0xFF)) == 'c'
        && ascii_lower((char)(base[m - 1] & 0xFF)) == 'm') {
        m -= 4;
        base[m] = L'\0';
    }
    if (m == 0) return -1;

    // 名字先按 id（文件名，ASCII）匹配——(App.launch "markdown") 一直用的是文件名，
    // 而显示名现在是元数据 title（可为中文）；再按显示名匹配，方便用中文名拉起。
    char base_mb[SCHEME_APP_ID_MAX * 4];
    am_wcstombs(base_mb, base, (uint32_t)sizeof(base_mb));
    for (i = 0; i < s_app_count; i++) {
        if (strcmp(s_apps[i].id, base_mb) == 0) return i;
        if (wcscmp(s_apps[i].title, base) == 0) return i;
    }
    return -1;
}

static int32_t app_host_lookup_app(const wchar_t *name) {
    return (app_desc_find_name(name) >= 0) ? 1 : 0;
}

int32_t ui_scheme_app_perform_launch(Global_State *global_state) {
    wchar_t name[SCHEME_APP_NAME_MAX * 2];
    wchar_t arg[AM_NATIVE_APP_LAUNCH_ARG_MAX];
    wchar_t caller_title[SCHEME_APP_NAME_MAX];
    char    caller_rel[192];          // 调用者的身份键（返回链用；单槽平台会先被销毁，故提前拷）
    UI_Scheme_App_State *caller = s_app;
    UI_Scheme_App_State *target = NULL;
    int32_t idx;
    int32_t cold = 0;

    // 取请求（取走即清零）。无请求是常态，直接返回。
    if (!am_native_app_launch_take(name, sizeof(name) / sizeof(name[0]),
                                  arg, sizeof(arg) / sizeof(arg[0]))) {
        return 0;
    }
    if (caller == NULL) {
        printf("[scheme-app] 拉起失败：没有前台实例\n");
        return 0;
    }
    wcsncpy(caller_title, caller->title, SCHEME_APP_NAME_MAX - 1);
    caller_title[SCHEME_APP_NAME_MAX - 1] = L'\0';
    {
        size_t rl = strlen(caller->source_rel);
        if (rl >= sizeof(caller_rel)) rl = sizeof(caller_rel) - 1;
        memcpy(caller_rel, caller->source_rel, rl);
        caller_rel[rl] = '\0';
    }

    // ① 名字解析
    idx = app_desc_find_name(name);
    if (idx < 0) {
        printf("[scheme-app] 拉起失败：找不到 App \"%ls\"（app/ 下没有对应的 .scm）\n", name);
        return 0;
    }
    if (strcmp(s_apps[idx].source_rel, caller->source_rel) == 0) {
        printf("[scheme-app] 拉起失败：目标是当前 App 自己（%ls）\n", name);
        return 0;
    }

#if SCHEME_APP_INST_MAX <= 1
    // 单实例平台：没有后台槽位。先销毁调用者再建目标（“拉起即替换”），
    // 否则 app_instance_alloc 会因“唯一槽位是前台、不可驱逐”而失败。
    // 这里绕过了 ui_scheme_app_on_exit，所以把它的两项释放动作补上（模态请求 +
    // (native Chat) 的推理引擎——引擎是 C 侧全局单例，不随解释器池回收）。
    printf("[scheme-app] 本平台无后台能力（实例槽位 %d）：拉起即替换调用者\n",
           (int32_t)SCHEME_APP_INST_MAX);
    // 走统一的实例收尾（ui_abort / app_abort / Chat 引擎释放 / 销毁 / 解绑 Gfx）。
    // 注意：没有后台实例可回，所以本平台**不建返回链**（见下面 ⑤）。
    app_teardown_foreground();
    s_app = NULL;
    s_selected = idx;
    target = app_instance_alloc(idx);
    if (target == NULL) {
        printf("[scheme-app] 拉起失败：实例槽位不可用\n");
        return 0;
    }
    cold = 1;
#else
    // 多实例平台：**先定目标槽位再让出前台**。此刻调用者仍是前台，
    // app_instance_alloc 的 LRU 驱逐只看后台实例，因此绝不可能驱逐调用者自己。
    target = app_instance_find(s_apps[idx].source_rel);
    if (target == NULL) {
        s_selected = idx;
        target = app_instance_alloc(idx);   // 可能驱逐最久未用的后台实例
        if (target == NULL) {
            printf("[scheme-app] 拉起失败：实例槽位不足且无可驱逐的后台实例\n");
            return 0;
        }
        cold = 1;
    }
    s_selected = idx;

    // 让出前台：调用者冻结保留（与 (App.hide) 同一条通道）
    app_instance_suspend(caller);
    s_app = NULL;
#endif

#if SCHEME_APP_INST_MAX > 1
    // ⑤ 建立返回链：目标退出时回到调用者（多槽平台才有意义——调用者此刻是被
    //    app_instance_suspend 冻结在后台的活实例，恢复它就是“回上一 App”）。
    //    **必须在 ③ 之前**：cold_start / foreground 里的 am_native_app_reset 会清掉
    //    “退出后回到谁”的提示，故它们都在 reset 之后自己调 app_apply_back_hint()；
    //    这样冷启动的 boot（(EX.install) 读 (App.back_to)）看到的就是新链。
    //    单槽平台（ESP32）不建链：调用者已被销毁，目标退出仍回应用列表（旧行为）。
    {
        size_t rl = strlen(caller_rel);
        if (rl >= sizeof(target->back_rel)) rl = sizeof(target->back_rel) - 1;
        memcpy(target->back_rel, caller_rel, rl);
        target->back_rel[rl] = '\0';
    }
#endif

    // ③ 目标上前台（冷启动 / 恢复后台实例）
    s_app = target;
    if (cold) {
        // 冷启动需要绑定绘图目标（同 ui_scheme_app_init 的冷启动路径）
        am_native_gfx_bind(global_state->gfx);
        am_native_raster_bind(global_state->gfx);
        app_instance_cold_start(global_state);
    }
    else {
        app_instance_foreground(target);
    }

    // ④ 交付启动参数（必须在 cold_start / foreground 里的 am_native_app_reset 之后，
    //    否则会被 reset 清掉——这是本机制最容易写错的一步）
    am_native_app_set_launch_arg(arg);

    // 恢复路径下画面归新 App，下一帧由它的 (App.resumed) 重绘：先请求一次整屏刷新
    global_state->is_full_refresh = 1;

    printf("[scheme-app] 拉起 %ls → %ls（%s%s%ls）\n",
           caller_title, target->title,
           cold ? "冷启动" : "恢复后台实例",
           (arg[0] != L'\0') ? "，参数 " : "",
           (arg[0] != L'\0') ? arg : L"");
    return 1;
}
