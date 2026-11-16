////////////////////////////////////////////////////////////////////////////
//  Native Library : App —— 设计说明见 am_native_App.h
//
//  本文件只做三件事：
//    1) 维护本 App 的生命周期状态位（退出意愿 / 返回请求 / 模态框关闭通知 / 名字）；
//    2) 把 (App.exit*)/(App.exit_now)/(App.hide) 翻译成宿主可轮询的请求位；
//    3) 把只读信息（name/platform/wanted…）按单值栈纪律压回操作数栈（N1/N6）。
//
//  真正的界面与唤醒动作都在 (native Ui) 一侧：退出确认框就是一个"退出型确认框"
//  （am_native_ui_confirm_begin(..., is_exit=1)），阻塞/唤醒逻辑与 (Ui.confirm ...)
//  完全同源，避免两套模态框机制。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <wchar.h>

#include "am_native_App.h"
#include "am_native_Ui.h"

#include "platform.h"

// ===============================================================================
// 平台标识（编译期决定；纯信息性，供 App 做能力降级）
// ===============================================================================

#if defined(NANO_WASM_GUI)
    #define APP_PLATFORM_NAME L"wasm-gui"
#elif defined(NANO_WEB)
    #define APP_PLATFORM_NAME L"web"
#elif defined(NANO_TTY)
    #define APP_PLATFORM_NAME L"tty"
#elif defined(NANO_PLATFORM_M5CORE2) || defined(NANO_PLATFORM_M5CORES3) || \
      defined(NANO_ESP32_S3) || defined(NANO_ESP32_P4)
    #define APP_PLATFORM_NAME L"m5esp"
#else
    #define APP_PLATFORM_NAME L"linux"
#endif

// App 名缓冲（码点数；宿主传入的显示名，超出截断）
#define APP_NAME_MAX (64)

// ===============================================================================
// 生命周期状态（同一时刻单前台模型；与 ui_scheme_app.c 的 s_app 一一对应）
// ===============================================================================

static int32_t s_exit_wanted = 0;    // 1=宿主应立即在本帧末销毁实例（已收尾完/打桩的 hide）
static int32_t s_exit_confirmed = 0; // 1=用户已确认退出，正在收尾（收尾窗口的起点）
static uint64_t s_exit_confirmed_ms = 0;
static int32_t s_back_pending = 0; // 1=硬 A/ESC 已按下、尚无消费者
static uint64_t s_back_ms = 0;     // 请求时刻（am_runtime_now_ms 时间线）
static int32_t s_modal_closed = 0; // 1=最近一次宿主模态框已关闭（待消费）
static int32_t s_active = 1;       // 1=前台活动；0=后台冻结/锁屏挂起
static int32_t s_resumed = 0;      // 1=刚从后台恢复（待消费）
static int32_t s_hide_pending = 0; // 1=有切后台请求（宿主取走执行）

// 屏幕上「退出」按钮的热区（由 Scheme 侧 app/lib/exit.scm 登记；宿主在触摸边沿上查询）
static int32_t s_btn_x = 0, s_btn_y = 0, s_btn_w = 0, s_btn_h = 0;
static wchar_t s_name[APP_NAME_MAX];

// ---- 拉起请求（App → App）-------------------------------------------------
// 三段式的第一段就在这几个静态量里：native 函数只置位，宿主在帧末取走（见头文件）。
static int32_t s_launch_pending = 0;                      // 1=有未处理的拉起请求
static wchar_t s_launch_name[APP_NAME_MAX];               // 目标 App 名（列表显示名）
static wchar_t s_launch_arg_out[AM_NATIVE_APP_LAUNCH_ARG_MAX]; // 随请求携带的启动参数
// 启动参数 inbox：宿主在 reset 之后写入，App 侧 (App.launch_arg) 取走即清零
static int32_t s_launch_arg_valid = 0;
static wchar_t s_launch_arg_inbox[AM_NATIVE_APP_LAUNCH_ARG_MAX];
// 名字存在性查询钩子（宿主提供；未注册时一律按"受理"处理）
static int32_t (*s_launch_lookup)(const wchar_t *name) = NULL;
// “退出后会回到哪个 App”（宿主登记；空串 = 顶层，退出回应用列表）——只用于确认框标题
static wchar_t s_back_hint[APP_NAME_MAX];

// ===============================================================================
// 内部辅助
// ===============================================================================

// 值型收尾（契约 N1 + N6）：压 1 个值 + step 1 次。
static int32_t app_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// 压一个布尔值
static int32_t app_push_bool(am_process_t *proc, int32_t b) {
    return app_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

// 压一个宽字符串（App 名/平台名；内容固定或由宿主登记，长度有界）
static int32_t app_push_wstring(am_process_t *proc, const wchar_t *str) {
    am_handle_t hd = am_process_make_wstring_handle(proc, str, wcslen(str));
    if (hd == AM_HANDLE_NULL) return -1;
    return app_push(proc, am_make_value_of_handle(hd));
}

// 从操作数栈弹出一个字符串对象并拷贝到 dst（含 NUL；超出 cap-1 截断）。
// 非字符串参数一律按空串处理（软失败，manual §1.1.1）。
// 返回**原始长度**（码点，未截断）——调用方可据此拒绝"放不下"的参数，
// 而不是拿半截字符串去干活（拉起路径必须这样，见 am_native_App_launch_common）。
static size_t app_pop_wstring_copy(am_process_t *proc, wchar_t *dst, size_t cap) {
    am_value_t *content = NULL;
    am_wstring_t *ws = NULL;
    am_value_t v = am_process_pop_operand(proc);

    dst[0] = L'\0';
    if (v == (am_value_t)UINTPTR_MAX) return 0;
    if (!am_value_is_handle(v)) return 0;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return 0;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return 0;

    ws = (am_wstring_t *)obj;
    content = ws->content;
    if (content == NULL) return 0;

    size_t n = ws->length;
    size_t copy = n;
    if (cap == 0) return n;
    if (copy > cap - 1) copy = cap - 1;
    for (size_t i = 0; i < copy; i++) {
        dst[i] = (wchar_t)am_value_to_wchar(content[i]);
    }
    dst[copy] = L'\0';
    return n;
}

// ===============================================================================
// Scheme 可见过程
// ===============================================================================

int32_t am_native_App_exit(am_runtime_t *rt, am_process_t *proc) {
    // 无参数：直接以默认标题发起"退出型确认框"
    wchar_t title[AM_NATIVE_UI_TITLE_MAX];

    am_native_app_clear_back_request();
    // 有返回链时把结果提前告诉用户：“确认”不是回桌面，而是回上一个 App
    // （否则用户会以为"点退出却没有应用列表"）。标题超长时退化为默认文案。
    // 文案长度受限：确认框宽 200px、标题用 16px 字模且超宽会被截成“…”，
    // 故不用全角括号（「」各占 16px，实测会把末尾的“？”截掉——2026-11 截图核对）。
    if (s_back_hint[0] != L'\0') {
        if (swprintf(title, AM_NATIVE_UI_TITLE_MAX, L"退出并返回 %ls？", s_back_hint) < 0) {
            wcscpy(title, L"是否退出？");
        }
    }
    else {
        wcscpy(title, L"是否退出？");
    }
    return am_native_ui_confirm_begin(rt, proc, title, NULL, 1);
}

int32_t am_native_App_exit_with(am_runtime_t *rt, am_process_t *proc) {
    wchar_t title[AM_NATIVE_UI_TITLE_MAX];

    app_pop_wstring_copy(proc, title, AM_NATIVE_UI_TITLE_MAX);
    if (title[0] == L'\0') wcscpy(title, L"是否退出？");

    am_native_app_clear_back_request();
    return am_native_ui_confirm_begin(rt, proc, title, NULL, 1);
}

int32_t am_native_App_exit_now(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    // 宣告“收尾完毕，可以销毁”（两段式协议的第二段）
    am_native_app_clear_back_request();
    s_exit_wanted = 1;
    return app_push_bool(proc, 1);
}

int32_t am_native_App_hide(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    // 切后台：宿主在本帧末冻结实例（保留解释器与内存池），前台回应用列表。
    // App 自己该做的收尾（落盘等）请**先做完**再调用本函数——冻结之后不再有执行机会。
    am_native_app_clear_back_request();
    s_hide_pending = 1;
    return app_push_bool(proc, 1);
}

// 弹一个数值并按像素取整（三种数字表示都接受）。
// **为什么不能只判 am_value_is_int**：非负整数字面量（36、16、2…）在灵机里是
// **无符号整数**，`-` 的结果才是有符号整数。原实现只认 int，于是 App 里写
// (App.set_exit_button 280 2 36 16) 会被静默降级成 (280 0 0 0)——宿主侧热区
// 判定直接失效（按钮画出来却点不动），这正是 app/lib/exit.scm 的宿主热区路径
// 长期"看起来没生效"的原因（它靠自己的电平兜底才勉强可用）。
static int32_t app_pop_pixel(am_process_t *proc, int32_t fallback) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return fallback;
    if (am_value_is_int(v))   return (int32_t)am_value_to_int(v);
    if (am_value_is_uint(v))  return (int32_t)am_value_to_uint(v);
    if (am_value_is_float(v)) {
        am_float_t f = am_value_to_float(v);
        if (isnan(f)) return fallback;
        return (int32_t)(f < 0 ? (f - 0.5) : (f + 0.5));
    }
    return fallback;   // 非数字：调用方按"没有按钮"处理
}

int32_t am_native_App_set_exit_button(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    // 参数从右往左弹：h, w, y, x（非数字按 0 处理 → 视为"没有按钮"）
    int32_t h = app_pop_pixel(proc, 0);
    int32_t w = app_pop_pixel(proc, 0);
    int32_t y = app_pop_pixel(proc, 0);
    int32_t x = app_pop_pixel(proc, 0);

    s_btn_x = x; s_btn_y = y; s_btn_w = w; s_btn_h = h;
    return app_push(proc, AM_VALUE_UNDEFINED);
}

int32_t am_native_App_resumed(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t v = s_resumed;
    s_resumed = 0;
    return app_push_bool(proc, v);
}

int32_t am_native_App_back_requested(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t pending = s_back_pending;
    s_back_pending = 0; // 取走即清零：宿主兜底据"无挂起请求"判定 App 已接管
    return app_push_bool(proc, pending);
}

int32_t am_native_App_modal_closed(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t v = s_modal_closed;
    s_modal_closed = 0;
    return app_push_bool(proc, v);
}

int32_t am_native_App_is_active(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return app_push_bool(proc, (s_active && !s_exit_wanted && !s_exit_confirmed) ? 1 : 0);
}

// 时间片上下限（见 am_native_App.h 的说明）
#define APP_TIMESLICE_MIN (31250u)
#define APP_TIMESLICE_MAX (1250000u)

int32_t am_native_App_set_timeslice(am_runtime_t *rt, am_process_t *proc) {
    int32_t ticks = app_pop_pixel(proc, 0);   // 非数字/缺失 → 0（忽略，保持默认）
    if (ticks > 0) {
        uint32_t t = (uint32_t)ticks;
        if (t < APP_TIMESLICE_MIN) t = APP_TIMESLICE_MIN;
        if (t > APP_TIMESLICE_MAX) t = APP_TIMESLICE_MAX;
        am_runtime_set_default_timeslice(rt, t);
    }
    return app_push(proc, AM_VALUE_UNDEFINED);
}

int32_t am_native_App_name(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return app_push_wstring(proc, s_name);
}

int32_t am_native_App_platform(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return app_push_wstring(proc, APP_PLATFORM_NAME);
}

// ---------------------------------------------------------------------------
// 拉起（三段式的第一段：只校验 + 置位）
// ---------------------------------------------------------------------------

// (App.launch name) / (App.launch_with name arg) 的公共实现。
// 参数从右往左弹：with_arg 时先 arg 再 name。
static int32_t app_launch_common(am_process_t *proc, int32_t with_arg) {
    wchar_t name[APP_NAME_MAX];
    wchar_t arg[AM_NATIVE_APP_LAUNCH_ARG_MAX];
    size_t nlen = 0;
    size_t alen = 0;

    // **先把所有参数弹完再校验**：提前 return 会把未弹的操作数留在栈上，
    // 破坏“单值堆纪律”（N1）并污染后续表达式（这类 bug 极难查）。
    arg[0] = L'\0';
    if (with_arg) {
        alen = app_pop_wstring_copy(proc, arg, AM_NATIVE_APP_LAUNCH_ARG_MAX);
    }
    nlen = app_pop_wstring_copy(proc, name, APP_NAME_MAX);

    // 放不下就拒绝：拿半截路径去打开文件只会失败得更难查（不静默截断）
    if (with_arg && alen >= AM_NATIVE_APP_LAUNCH_ARG_MAX) return app_push_bool(proc, 0);
    if (nlen == 0 || nlen >= APP_NAME_MAX) return app_push_bool(proc, 0);
    // 同一时刻只允许一条在途请求：多发的会静默覆盖上一条（难以排查），故直接拒绝
    if (s_launch_pending) return app_push_bool(proc, 0);
    // 名字存在性：宿主注册了钩子时预先校验（找不到就当场软失败，App 可感知）
    if (s_launch_lookup != NULL && !s_launch_lookup(name)) return app_push_bool(proc, 0);

    wcsncpy(s_launch_name, name, APP_NAME_MAX - 1);
    s_launch_name[APP_NAME_MAX - 1] = L'\0';
    wcsncpy(s_launch_arg_out, arg, AM_NATIVE_APP_LAUNCH_ARG_MAX - 1);
    s_launch_arg_out[AM_NATIVE_APP_LAUNCH_ARG_MAX - 1] = L'\0';
    s_launch_pending = 1;
    return app_push_bool(proc, 1);
}

int32_t am_native_App_launch(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return app_launch_common(proc, 0);
}

int32_t am_native_App_launch_with(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return app_launch_common(proc, 1);
}

int32_t am_native_App_launch_arg(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    // 一次性：取走即清零（语义同 (App.back_requested)，宿主兜底/重复拉起都不会重复交付）
    if (!s_launch_arg_valid) return app_push(proc, AM_VALUE_NULL);
    s_launch_arg_valid = 0;
    return app_push_wstring(proc, s_launch_arg_inbox);
}

int32_t am_native_App_back_to(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    // 只读、**不**取走：App 可能每拍/每次重绘都要看（与 launch_arg 的一次性语义不同）
    if (s_back_hint[0] == L'\0') return app_push(proc, AM_VALUE_NULL);
    return app_push_wstring(proc, s_back_hint);
}

// ===============================================================================
// 宿主服务接口
// ===============================================================================

void am_native_app_reset(const wchar_t *name) {
    s_btn_x = s_btn_y = s_btn_w = s_btn_h = 0;
    s_active = 1;
    s_resumed = 0;
    s_hide_pending = 0;
    s_exit_wanted = 0;
    s_exit_confirmed = 0;
    s_exit_confirmed_ms = 0;
    s_back_pending = 0;
    s_back_ms = 0;
    s_modal_closed = 0;
    // 拉起请求与启动参数都属于"上一个上下文"：换了实例就一律作废
    // （宿主在 reset **之后**才写入本次的启动参数，见 am_native_app_set_launch_arg）
    s_launch_pending = 0;
    s_launch_name[0] = L'\0';
    s_launch_arg_out[0] = L'\0';
    s_launch_arg_valid = 0;
    s_launch_arg_inbox[0] = L'\0';
    // 返回链提示属于"上一个上下文"：前台切换时先清空，宿主要在 reset 之后重新告知
    s_back_hint[0] = L'\0';

    s_name[0] = L'\0';
    if (name != NULL) {
        wcsncpy(s_name, name, APP_NAME_MAX - 1);
        s_name[APP_NAME_MAX - 1] = L'\0';
    }
}

void am_native_app_abort(void) {
    s_btn_x = s_btn_y = s_btn_w = s_btn_h = 0;
    s_hide_pending = 0;
    s_exit_wanted = 0;
    s_exit_confirmed = 0;
    s_exit_confirmed_ms = 0;
    s_back_pending = 0;
    s_back_ms = 0;
    s_modal_closed = 0;
    s_launch_pending = 0;
    s_launch_name[0] = L'\0';
    s_launch_arg_out[0] = L'\0';
    s_launch_arg_valid = 0;
    s_launch_arg_inbox[0] = L'\0';
    s_back_hint[0] = L'\0';   // 实例销毁：返回链提示跟着前台上下文一起作废
}

void am_native_app_set_back_request(am_runtime_t *rt) {
    s_back_pending = 1;
    s_back_ms = (uint64_t)am_runtime_now_ms(rt);
}

int32_t am_native_app_back_pending_ms(am_runtime_t *rt) {
    uint64_t now;
    if (!s_back_pending) return -1;
    now = (uint64_t)am_runtime_now_ms(rt);
    if (now < s_back_ms) return 0;
    if (now - s_back_ms > 60000ULL) return 60000; // 防御：时钟回拨/异常不产生巨大值
    return (int32_t)(now - s_back_ms);
}

void am_native_app_clear_back_request(void) {
    s_back_pending = 0;
    s_back_ms = 0;
}

int32_t am_native_app_exit_confirmed(void) {
    return s_exit_confirmed;
}

int32_t am_native_app_exit_grace_ms(am_runtime_t *rt) {
    uint64_t now;
    if (!s_exit_confirmed) return -1;
    now = (uint64_t)am_runtime_now_ms(rt);
    if (now < s_exit_confirmed_ms) return 0;
    if (now - s_exit_confirmed_ms > 60000ULL) return 60000; // 防御：时钟回拨/异常
    return (int32_t)(now - s_exit_confirmed_ms);
}

void am_native_app_extend_grace(am_runtime_t *rt) {
    if (s_exit_confirmed) s_exit_confirmed_ms = (uint64_t)am_runtime_now_ms(rt);
}

int32_t am_native_app_exit_wanted(am_runtime_t *rt) {
    if (s_exit_wanted) return 1;                       // exit_now / hide：立即销毁
    if (!s_exit_confirmed) return 0;
    // 已确认退出但 App 还没宣告收尾完毕：给满收尾窗口，超时才强制销毁
    return (am_native_app_exit_grace_ms(rt) >= AM_NATIVE_APP_EXIT_GRACE_MS) ? 1 : 0;
}

void am_native_app_clear_exit_wanted(void) {
    s_exit_wanted = 0;
    s_exit_confirmed = 0;
    s_exit_confirmed_ms = 0;
}

void am_native_app_mark_exit_confirmed(am_runtime_t *rt) {
    s_exit_confirmed = 1;
    s_exit_confirmed_ms = (uint64_t)am_runtime_now_ms(rt);
}

void am_native_app_notify_modal_closed(void) {
    s_modal_closed = 1;
}

void am_native_app_set_active(int32_t active) {
    s_active = active ? 1 : 0;
    if (!s_active) {
        // 挂起后这些"待处理"的东西都没有意义了（恢复时由宿主重新通知）
        s_back_pending = 0;
        s_hide_pending = 0;
    }
}

void am_native_app_notify_resumed(void) {
    s_resumed = 1;
    s_modal_closed = 1;   // 恢复后画面归自己了：让 App 补一次重绘（与模态框关闭同一语义）
}

void am_native_app_request_hide(void) {
    s_hide_pending = 1;
}

int32_t am_native_app_hit_exit_button(int32_t x, int32_t y) {
    if (s_btn_w <= 0 || s_btn_h <= 0) return 0;
    if (x < s_btn_x || x >= s_btn_x + s_btn_w) return 0;
    if (y < s_btn_y || y >= s_btn_y + s_btn_h) return 0;
    return 1;
}

int32_t am_native_app_hide_requested(void) {
    int32_t v = s_hide_pending;
    s_hide_pending = 0;   // 取走即清零（宿主据此只冻结一次）
    return v;
}

int32_t am_native_app_launch_take(wchar_t *name, size_t ncap, wchar_t *arg, size_t acap) {
    if (!s_launch_pending) return 0;
    s_launch_pending = 0;   // 取走即清零（宿主据此只切换一次）
    if (name != NULL && ncap > 0) {
        wcsncpy(name, s_launch_name, ncap - 1);
        name[ncap - 1] = L'\0';
    }
    if (arg != NULL && acap > 0) {
        wcsncpy(arg, s_launch_arg_out, acap - 1);
        arg[acap - 1] = L'\0';
    }
    s_launch_name[0] = L'\0';
    s_launch_arg_out[0] = L'\0';
    return 1;
}

void am_native_app_set_launch_arg(const wchar_t *arg) {
    s_launch_arg_valid = 0;
    s_launch_arg_inbox[0] = L'\0';
    if (arg == NULL || arg[0] == L'\0') return;   // 空串 = 无参数（从不交付）
    wcsncpy(s_launch_arg_inbox, arg, AM_NATIVE_APP_LAUNCH_ARG_MAX - 1);
    s_launch_arg_inbox[AM_NATIVE_APP_LAUNCH_ARG_MAX - 1] = L'\0';
    s_launch_arg_valid = 1;
}

void am_native_app_bind_lookup(int32_t (*fn)(const wchar_t *name)) {
    s_launch_lookup = fn;
}

void am_native_app_set_back_hint(const wchar_t *title) {
    s_back_hint[0] = L'\0';
    if (title == NULL || title[0] == L'\0') return;
    wcsncpy(s_back_hint, title, APP_NAME_MAX - 1);
    s_back_hint[APP_NAME_MAX - 1] = L'\0';
}

// ===============================================================================
// 库注册表
// ===============================================================================

static const am_native_func_entry_t am_native_App_funcs[] = {
    { L"exit",            am_native_App_exit },
    { L"exit_with",       am_native_App_exit_with },
    { L"exit_now",        am_native_App_exit_now },
    { L"hide",            am_native_App_hide },
    { L"back_requested",  am_native_App_back_requested },
    { L"modal_closed",    am_native_App_modal_closed },
    { L"is_active",       am_native_App_is_active },
    { L"resumed",         am_native_App_resumed },
    { L"set_exit_button", am_native_App_set_exit_button },
    { L"set_timeslice",   am_native_App_set_timeslice },
    { L"name",            am_native_App_name },
    { L"platform",        am_native_App_platform },
    { L"launch",          am_native_App_launch },
    { L"launch_with",     am_native_App_launch_with },
    { L"launch_arg",      am_native_App_launch_arg },
    { L"back_to",         am_native_App_back_to },
};

const am_native_lib_entry_t am_native_App_lib = {
    L"App",
    am_native_App_funcs,
    sizeof(am_native_App_funcs) / sizeof(am_native_App_funcs[0])
};