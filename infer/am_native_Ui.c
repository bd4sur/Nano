////////////////////////////////////////////////////////////////////////////
//  Native Library : Ui —— 设计说明见 am_native_Ui.h
//
//  本文件只做三件事：
//    1) 把 Scheme 传进来的标题/初值/正文按码点拷进本模块的静态缓冲（Scheme 的字符串
//       对象随进程堆走，不能长期持有裸指针）；
//    2) 把**进程**置为 BLOCKED 并返回 0（契约 N7 的阻塞型函数：不压值、不 step），
//       由宿主的事件循环接管界面；
//    3) 宿主提交时 am_native_ui_resolve()：压结果值 + step 一次 + 置 READY + 重新入队
//       （与上游 runtime 的队列唤醒 runtime_queue_wake_process 同一套动作）。
//
//  为什么不用现成的队列阻塞（System.read）：队列的唤醒值是队列里的元素，而我们要把
//  「用户敲出来的字符串」交给**同一条指令**继续执行；直接用 BLOCKED + 唤醒更直接，
//  且不占用队列资源、不需要超时。风险点只有一个——唤醒时必须让 PC 越过 callnative
//  指令（am_process_step），本文件在 resolve 里严格照抄队列唤醒的四个动作。
////////////////////////////////////////////////////////////////////////////

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <wchar.h>

#include "am_native_Ui.h"
#include "am_native_App.h"

// ===============================================================================
// 请求状态（同一时刻仅一个）
// ===============================================================================

static int32_t s_kind = AM_NATIVE_UI_KIND_NONE; // 当前请求种类
static int32_t s_confirm_is_exit = 0;           // 当前确认框是否为「退出型」（见 am_native_ui_confirm_begin）
static am_pid_t s_pid = 0;                      // 被阻塞的进程
static wchar_t s_title[AM_NATIVE_UI_TITLE_MAX];
static wchar_t s_initial[AM_NATIVE_UI_BODY_MAX];
static wchar_t s_body[AM_NATIVE_UI_BODY_MAX];

// ===============================================================================
// 内部辅助
// ===============================================================================

// 从操作数栈弹出一个字符串对象并拷贝到 dst（含 NUL；超出 cap-1 截断）。
// 非字符串参数一律按空串处理（软失败，manual §1.1.1），返回 false 但已填好 dst。
static bool ui_pop_wstring_copy(am_process_t *proc, wchar_t *dst, size_t cap) {
    am_wstring_t *ws = NULL;
    am_value_t v = am_process_pop_operand(proc);

    dst[0] = L'\0';
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return false;

    ws = (am_wstring_t *)obj;
    size_t n = ws->length;
    if (n > cap - 1) n = cap - 1;
    for (size_t i = 0; i < n; i++) {
        dst[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    }
    dst[n] = L'\0';
    return true;
}

// 值型收尾（契约 N1 + N6）：压 1 个值 + step 1 次。
static int32_t ui_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// 发起一个阻塞请求。已有请求在等待时软失败：立即返回 #null（INPUT）/#undefined（MESSAGE）。
static int32_t ui_begin(am_runtime_t *rt, am_process_t *proc, int32_t kind) {
    (void)rt;

    if (s_kind != AM_NATIVE_UI_KIND_NONE) {
        // 同时只允许一个模态框（见头文件说明）：后到者按"取消"软失败
        return ui_push(proc, (kind == AM_NATIVE_UI_KIND_INPUT) ? AM_VALUE_NULL : AM_VALUE_UNDEFINED);
    }

    s_kind = kind;
    s_pid = proc->pid;

    // 阻塞：不压值、不 step（N7）。虚拟机因此把本进程移出调度，宿主每帧接管界面。
    am_process_set_state(proc, AM_PROCESS_STATE_BLOCKED);
    return 0;
}

// ===============================================================================
// Scheme 可见过程
// ===============================================================================

int32_t am_native_Ui_input(am_runtime_t *rt, am_process_t *proc) {
    ui_pop_wstring_copy(proc, s_title, AM_NATIVE_UI_TITLE_MAX);
    s_initial[0] = L'\0';
    s_body[0] = L'\0';
    return ui_begin(rt, proc, AM_NATIVE_UI_KIND_INPUT);
}

int32_t am_native_Ui_input_with(am_runtime_t *rt, am_process_t *proc) {
    // 参数从右往左弹：先初值，再标题
    ui_pop_wstring_copy(proc, s_initial, AM_NATIVE_UI_BODY_MAX);
    ui_pop_wstring_copy(proc, s_title, AM_NATIVE_UI_TITLE_MAX);
    s_body[0] = L'\0';
    return ui_begin(rt, proc, AM_NATIVE_UI_KIND_INPUT);
}

int32_t am_native_Ui_message(am_runtime_t *rt, am_process_t *proc) {
    s_title[0] = L'\0';
    s_initial[0] = L'\0';
    ui_pop_wstring_copy(proc, s_body, AM_NATIVE_UI_BODY_MAX);
    return ui_begin(rt, proc, AM_NATIVE_UI_KIND_MESSAGE);
}

int32_t am_native_Ui_message_titled(am_runtime_t *rt, am_process_t *proc) {
    ui_pop_wstring_copy(proc, s_body, AM_NATIVE_UI_BODY_MAX);
    ui_pop_wstring_copy(proc, s_title, AM_NATIVE_UI_TITLE_MAX);
    s_initial[0] = L'\0';
    return ui_begin(rt, proc, AM_NATIVE_UI_KIND_MESSAGE);
}

int32_t am_native_Ui_confirm(am_runtime_t *rt, am_process_t *proc) {
    wchar_t title[AM_NATIVE_UI_TITLE_MAX];
    wchar_t body[AM_NATIVE_UI_BODY_MAX];

    // 参数从右往左弹：先正文，再标题
    ui_pop_wstring_copy(proc, body, AM_NATIVE_UI_BODY_MAX);
    ui_pop_wstring_copy(proc, title, AM_NATIVE_UI_TITLE_MAX);
    return am_native_ui_confirm_begin(rt, proc, title, body, 0);
}

// ===============================================================================
// 宿主服务接口
// ===============================================================================

int32_t am_native_ui_pending(void) {
    return s_kind != AM_NATIVE_UI_KIND_NONE;
}

int32_t am_native_ui_kind(void) {
    return s_kind;
}

const wchar_t *am_native_ui_title(void) {
    return s_title;
}

const wchar_t *am_native_ui_initial(void) {
    return s_initial;
}

const wchar_t *am_native_ui_body(void) {
    return s_body;
}

void am_native_ui_resolve(am_runtime_t *rt, const wchar_t *result) {
    int32_t kind = s_kind;
    am_pid_t pid = s_pid;
    am_process_t *proc;

    if (kind == AM_NATIVE_UI_KIND_NONE) return;

    // 先清状态：唤醒路径可能触发宿主再次进入模态框（App 循环提问），不能残留
    s_kind = AM_NATIVE_UI_KIND_NONE;
    s_pid = 0;
    s_title[0] = s_initial[0] = s_body[0] = L'\0';
    s_confirm_is_exit = 0;

    if (!rt) return;
    proc = am_runtime_get_process(rt, pid);
    if (!proc) return; // 进程已被销毁（App 退出等）：无需唤醒

    am_value_t v = AM_VALUE_UNDEFINED;
    if (kind == AM_NATIVE_UI_KIND_INPUT) {
        if (!result) {
            v = AM_VALUE_NULL; // 取消
        }
        else {
            am_handle_t hd = am_process_make_wstring_handle(proc, result, wcslen(result));
            if (hd == AM_HANDLE_NULL) {
                am_runtime_error(rt, L"[Ui] 唤醒失败：无法为返回值分配字符串\n");
                am_process_set_state(proc, AM_PROCESS_STATE_KILLED);
                return;
            }
            v = am_make_value_of_handle(hd);
        }
    }

    // 与上游 runtime_queue_wake_process 同一套动作：压值 → step 越过 callnative → READY → 重新入队
    if (am_process_push_operand(proc, v) != 0) {
        am_runtime_error(rt, L"[Ui] 唤醒失败：操作数栈压值失败，进程终止\n");
        am_process_set_state(proc, AM_PROCESS_STATE_KILLED);
        return;
    }
    am_process_step(proc);
    am_process_set_state(proc, AM_PROCESS_STATE_READY);

    am_list_t *nq = am_list_push(rt->vm_alloc, rt->process_queue, am_make_value_of_uint((am_uint_t)pid));
    if (!nq) {
        am_process_set_state(proc, AM_PROCESS_STATE_STOPPED);
    }
    else {
        rt->process_queue = nq;
    }
}

void am_native_ui_abort(void) {
    s_kind = AM_NATIVE_UI_KIND_NONE;
    s_pid = 0;
    s_confirm_is_exit = 0;
    s_title[0] = s_initial[0] = s_body[0] = L'\0';
}

// ===============================================================================
// 确认框（(Ui.confirm ...) 与 (native App) 的退出请求共用）
// ===============================================================================

int32_t am_native_ui_confirm_begin(am_runtime_t *rt, am_process_t *proc,
                                   const wchar_t *title, const wchar_t *body, int32_t is_exit) {
    (void)rt;

    if (s_kind != AM_NATIVE_UI_KIND_NONE) {
        // 同时只允许一个模态框：后到者按「取消」软失败（与 ui_begin 同一策略）
        return ui_push(proc, AM_VALUE_FALSE);
    }

    if (title != NULL && title[0] != L'\0') {
        wcsncpy(s_title, title, AM_NATIVE_UI_TITLE_MAX - 1);
        s_title[AM_NATIVE_UI_TITLE_MAX - 1] = L'\0';
    }
    else {
        wcscpy(s_title, L"是否退出？");
    }

    if (body != NULL && body[0] != L'\0') {
        wcsncpy(s_body, body, AM_NATIVE_UI_BODY_MAX - 1);
        s_body[AM_NATIVE_UI_BODY_MAX - 1] = L'\0';
    }
    else {
        s_body[0] = L'\0';
    }
    s_initial[0] = L'\0';

    s_confirm_is_exit = is_exit ? 1 : 0;
    s_kind = AM_NATIVE_UI_KIND_CONFIRM;
    s_pid = proc->pid;

    // 阻塞：不压值、不 step（N7）。虚拟机因此把本进程移出调度，宿主每帧接管界面。
    am_process_set_state(proc, AM_PROCESS_STATE_BLOCKED);
    return 0;
}

int32_t am_native_ui_confirm_is_exit(void) {
    return (s_kind == AM_NATIVE_UI_KIND_CONFIRM) ? s_confirm_is_exit : 0;
}

void am_native_ui_resolve_confirm(am_runtime_t *rt, int32_t yes) {
    am_pid_t pid = s_pid;
    am_process_t *proc;
    int32_t is_exit;

    if (s_kind != AM_NATIVE_UI_KIND_CONFIRM) return;
    is_exit = s_confirm_is_exit;

    // 先清状态（同 am_native_ui_resolve 的理由）
    s_kind = AM_NATIVE_UI_KIND_NONE;
    s_pid = 0;
    s_confirm_is_exit = 0;
    s_title[0] = s_initial[0] = s_body[0] = L'\0';

    // 退出型确认框的「确认」：启动 App 的收尾窗口（两段式协议第一段）。
    // 真正销毁要等 App 调 (App.exit_now) 或收尾窗口超时（见 am_native_App.h）。
    if (yes && is_exit) am_native_app_mark_exit_confirmed(rt);

    if (!rt) return;
    proc = am_runtime_get_process(rt, pid);
    if (!proc) return; // 进程已被销毁（App 退出等）：无需唤醒

    // 与上游 runtime_queue_wake_process 同一套动作：压值 → step 越过 callnative → READY → 重新入队
    if (am_process_push_operand(proc, yes ? AM_VALUE_TRUE : AM_VALUE_FALSE) != 0) {
        am_runtime_error(rt, L"[Ui] 唤醒失败：操作数栈压值失败，进程终止\n");
        am_process_set_state(proc, AM_PROCESS_STATE_KILLED);
        return;
    }
    am_process_step(proc);
    am_process_set_state(proc, AM_PROCESS_STATE_READY);

    am_list_t *nq = am_list_push(rt->vm_alloc, rt->process_queue, am_make_value_of_uint((am_uint_t)pid));
    if (!nq) {
        am_process_set_state(proc, AM_PROCESS_STATE_STOPPED);
    }
    else {
        rt->process_queue = nq;
    }
}

// ===============================================================================
// 库注册表
// ===============================================================================

static const am_native_func_entry_t am_native_Ui_funcs[] = {
    { L"input",          am_native_Ui_input },
    { L"input_with",     am_native_Ui_input_with },
    { L"message",        am_native_Ui_message },
    { L"message_titled", am_native_Ui_message_titled },
    { L"confirm",        am_native_Ui_confirm },
};

const am_native_lib_entry_t am_native_Ui_lib = {
    L"Ui",
    am_native_Ui_funcs,
    sizeof(am_native_Ui_funcs) / sizeof(am_native_Ui_funcs[0])
};