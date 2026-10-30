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
static wchar_t s_name[APP_NAME_MAX];

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
static void app_pop_wstring_copy(am_process_t *proc, wchar_t *dst, size_t cap) {
    am_value_t *content = NULL;
    am_wstring_t *ws = NULL;
    am_value_t v = am_process_pop_operand(proc);

    dst[0] = L'\0';
    if (v == (am_value_t)UINTPTR_MAX) return;
    if (!am_value_is_handle(v)) return;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return;

    ws = (am_wstring_t *)obj;
    content = ws->content;
    if (content == NULL) return;

    size_t n = ws->length;
    if (n > cap - 1) n = cap - 1;
    for (size_t i = 0; i < n; i++) {
        dst[i] = (wchar_t)am_value_to_wchar(content[i]);
    }
    dst[n] = L'\0';
}

// ===============================================================================
// Scheme 可见过程
// ===============================================================================

int32_t am_native_App_exit(am_runtime_t *rt, am_process_t *proc) {
    // 无参数：直接以默认标题发起"退出型确认框"
    am_native_app_clear_back_request();
    return am_native_ui_confirm_begin(rt, proc, L"是否退出？", NULL, 1);
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
    // P1 打桩：切后台 = 关闭（多任务落地时只改这里与宿主侧）
    am_native_app_clear_back_request();
    s_exit_wanted = 1;
    return app_push_bool(proc, 1);
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
    return app_push_bool(proc, (s_exit_wanted || s_exit_confirmed) ? 0 : 1);
}

int32_t am_native_App_name(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return app_push_wstring(proc, s_name);
}

int32_t am_native_App_platform(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return app_push_wstring(proc, APP_PLATFORM_NAME);
}

// ===============================================================================
// 宿主服务接口
// ===============================================================================

void am_native_app_reset(const wchar_t *name) {
    s_exit_wanted = 0;
    s_exit_confirmed = 0;
    s_exit_confirmed_ms = 0;
    s_back_pending = 0;
    s_back_ms = 0;
    s_modal_closed = 0;

    s_name[0] = L'\0';
    if (name != NULL) {
        wcsncpy(s_name, name, APP_NAME_MAX - 1);
        s_name[APP_NAME_MAX - 1] = L'\0';
    }
}

void am_native_app_abort(void) {
    s_exit_wanted = 0;
    s_exit_confirmed = 0;
    s_exit_confirmed_ms = 0;
    s_back_pending = 0;
    s_back_ms = 0;
    s_modal_closed = 0;
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
    { L"name",            am_native_App_name },
    { L"platform",        am_native_App_platform },
};

const am_native_lib_entry_t am_native_App_lib = {
    L"App",
    am_native_App_funcs,
    sizeof(am_native_App_funcs) / sizeof(am_native_App_funcs[0])
};