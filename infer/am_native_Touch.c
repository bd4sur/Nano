////////////////////////////////////////////////////////////////////////////
//  Native Library : Touch —— 设计说明见 am_native_Touch.h
//
//  HAL 契约（hal_touch.h）：touch_read(x, y, is_pressed) 每次调用采样一次触屏，
//  返回 0 正常、负值表示无触屏硬件或读取失败；is_pressed=0 时 x/y 不保证有效。
//
//  本文件只做三件事：
//    1) 采样电平并缓存"最后一次有效触点"（HAL 松开时坐标不可靠）；
//    2) 由相邻两次采样的电平变化派生按下/松开沿，锁存到 take_press/take_release；
//    3) 把结果按单值栈纪律压回操作数栈（N1/N6）。
////////////////////////////////////////////////////////////////////////////

#include <stdbool.h>
#include <stdint.h>

#include "am_native_Touch.h"
#include "hal_touch.h"
#include "am_host.h"   // am_current_timestamp_in_ms（丢弃在途手势的兜底窗口）

// 触屏可用性：-1 尚未探测，0 不可用，1 可用（touch_read 返回值缓存）
static int32_t s_available = -1;

// 最后一次按下期间的有效触点（HAL 松开时不保证坐标有效，故由本库缓存）
static int32_t s_last_x = 0;
static int32_t s_last_y = 0;

// 边沿锁存（与 ui_app.c 的 get_input_event 同哲学：生产端只提供电平，
// 边沿由消费者判定；此处消费者是 Scheme App）
static int32_t s_prev_pressed = 0;    // 上一次采样的电平
static int32_t s_press_latched = 0;   // 待取走的"按下"沿
static int32_t s_release_latched = 0; // 待取走的"松开"沿

// 在途手势丢弃（宿主专用，见 am_native_Touch.h 的 am_native_Touch_drop_inflight）：
//   置位后直到采样到"未按下"为止，一律按"没按"上报（电平式与沿式 App 都安全）。
//   兜底用**时钟**（am_current_timestamp_in_ms，与 (native Time) 同一来源）而不是采样次数：
//   一个 App 一拍里可能采样 5~10 次（is_pressed/take_press/take_release/x/y 各一次），
//   计数型兜底会被很快烧完——实测按次数计 180 时，60Hz 下只挡住约 0.6 秒。
#define TOUCH_DROP_MAX_MS (3000)      // 兜底：HAL 卡住时报"按下"也能自动解开
static int32_t s_drop_inflight = 0;   // 1=正在丢弃在途手势
static uint64_t s_drop_until_ms = 0;  // 丢弃窗口的下限（毫秒，同一时钟源）

// 滚轮原生通道的暂存（宿主每帧经 am_native_Touch_set_wheel 交来）
static int32_t s_wheel_pending = 0;

// ---------------------------------------------------------------------------
// 采样：一次 HAL 电平读取 + 坐标缓存 + 边沿锁存
// 返回 1（可用且按下）/ 0（可用但未按下）/-1（触屏不可用）
// ---------------------------------------------------------------------------
static int32_t touch_sample(int32_t *out_x, int32_t *out_y) {
    int32_t x = 0, y = 0, is_pressed = 0;

    if (touch_read(&x, &y, &is_pressed) < 0) {
        // 无触屏硬件/端口不可用：视为恒未按下（不残留上一次的按下状态）
        s_available = 0;
        s_prev_pressed = 0;
        s_press_latched = 0;
        s_release_latched = 0;
        s_drop_inflight = 0;   // 无触屏：丢弃态也无意义（否则触屏恢复后会白丢一段输入）
        s_drop_until_ms = 0;
        if (out_x) *out_x = s_last_x;
        if (out_y) *out_y = s_last_y;
        return -1;
    }
    s_available = 1;

    // 丢弃在途手势：见 am_native_Touch_drop_inflight 的说明。
    // 只在“手指已抬（原始采样为未按下）”或兜底时限到期时解除；期间不更新坐标
    // （那个触点是上一个 App/宿主确认框的，不是本 App 的），也不派生任何沿。
    if (s_drop_inflight) {
        if (!is_pressed || am_current_timestamp_in_ms() >= s_drop_until_ms) {
            s_drop_inflight = 0;
            s_drop_until_ms = 0;
            s_prev_pressed = 0;   // 以"未按下"为新基线，不补发历史沿
        }
        else {
            s_prev_pressed = 0;
            if (out_x) *out_x = s_last_x;
            if (out_y) *out_y = s_last_y;
            return 0;
        }
    }

    if (is_pressed) {
        s_last_x = x;
        s_last_y = y;
        if (!s_prev_pressed) s_press_latched = 1; // 上升沿
        s_prev_pressed = 1;
    }
    else {
        if (s_prev_pressed) s_release_latched = 1; // 下降沿
        s_prev_pressed = 0;
    }

    if (out_x) *out_x = s_last_x;
    if (out_y) *out_y = s_last_y;
    return s_prev_pressed;
}

// ---------------------------------------------------------------------------
// 值型收尾（契约 N1 + N6）：压 1 个值 + step 1 次。全部函数共用单一出口。
// ---------------------------------------------------------------------------
static int32_t touch_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t touch_push_bool(am_process_t *proc, bool b) {
    return touch_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

static int32_t touch_push_int(am_process_t *proc, int32_t n) {
    return touch_push(proc, am_make_value_of_int((am_int_t)n));
}

// ===============================================================================
// Native 函数实现
// ===============================================================================

int32_t am_native_Touch_is_available(am_runtime_t *rt, am_process_t *proc) {
    int32_t x = 0, y = 0;
    (void)rt;
    touch_sample(&x, &y);
    return touch_push_bool(proc, s_available == 1);
}

int32_t am_native_Touch_is_pressed(am_runtime_t *rt, am_process_t *proc) {
    int32_t x = 0, y = 0;
    (void)rt;
    return touch_push_bool(proc, touch_sample(&x, &y) == 1);
}

int32_t am_native_Touch_x(am_runtime_t *rt, am_process_t *proc) {
    int32_t x = 0, y = 0;
    (void)rt;
    touch_sample(&x, &y);
    return touch_push_int(proc, x);
}

int32_t am_native_Touch_y(am_runtime_t *rt, am_process_t *proc) {
    int32_t x = 0, y = 0;
    (void)rt;
    touch_sample(&x, &y);
    return touch_push_int(proc, y);
}

// 取走"按下"沿：锁存清零（一次按下只报一次）。必须先采样，边沿才有机会被发现。
int32_t am_native_Touch_take_press(am_runtime_t *rt, am_process_t *proc) {
    int32_t x = 0, y = 0;
    int32_t had;
    (void)rt;
    touch_sample(&x, &y);
    had = s_press_latched;
    s_press_latched = 0;
    return touch_push_bool(proc, had != 0);
}

// 取走"松开"沿：锁存清零（一次松开只报一次）。
int32_t am_native_Touch_take_release(am_runtime_t *rt, am_process_t *proc) {
    int32_t x = 0, y = 0;
    int32_t had;
    (void)rt;
    touch_sample(&x, &y);
    had = s_release_latched;
    s_release_latched = 0;
    return touch_push_bool(proc, had != 0);
}

// ---------------------------------------------------------------------------
// 滚轮（原生通道，见 am_native_Touch.h 的说明）
//
// 与触屏的三个函数不同，滚轮**不在**本库采样：宿主的输入层（ui_app.c 的
// get_input_event）本来就是唯一消费 hal_touch_wheel_take 的地方，本库再去取就会
// 与它抢（50ms/2ms 心跳随时可能先把增量取走）。所以由宿主每帧把
// key_event->wheel_dy 转交进来（am_native_Touch_set_wheel），本库只做暂存。
// ---------------------------------------------------------------------------

int32_t am_native_Touch_wheel_available(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
#if defined(NANO_WASM_GUI)
    return touch_push_bool(proc, true);
#else
    // 其余平台没有滚轮输入源（掌机/桌面/串口键盘），App 应退回触屏拖动
    return touch_push_bool(proc, false);
#endif
}

int32_t am_native_Touch_wheel_enable(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
#if defined(NANO_WASM_GUI)
    hal_touch_wheel_claim(1);
    return touch_push_bool(proc, true);
#else
    return touch_push_bool(proc, false);
#endif
}

int32_t am_native_Touch_wheel(am_runtime_t *rt, am_process_t *proc) {
    int32_t v = s_wheel_pending;
    (void)rt;
    s_wheel_pending = 0;
    return touch_push_int(proc, v);
}

// 宿主服务：转交本帧的滚轮增量（正 = 内容上滚）
void am_native_Touch_set_wheel(int32_t dy) {
    s_wheel_pending += dy;
}

// 宿主服务：新实例冷启动时清干净（积压增量 + 上一个 App 的认领）
void am_native_Touch_reset_wheel(void) {
    s_wheel_pending = 0;
#if defined(NANO_WASM_GUI)
    hal_touch_wheel_claim(0);
#endif
}

// ===============================================================================
// 宿主服务接口
// ===============================================================================

// 丢弃一段在途手势（宿主在切换 App / 恢复实例时调用；见头文件的说明）
void am_native_Touch_drop_inflight(void) {
    s_drop_inflight = 1;
    s_drop_until_ms = am_current_timestamp_in_ms() + TOUCH_DROP_MAX_MS;
    s_prev_pressed = 0;
    s_press_latched = 0;    // 上一个上下文残留的沿一律作废
    s_release_latched = 0;
}

// ===============================================================================
// 库注册表
// ===============================================================================

static const am_native_func_entry_t am_native_Touch_funcs[] = {
    { L"is_available",  am_native_Touch_is_available },
    { L"is_pressed",    am_native_Touch_is_pressed },
    { L"x",             am_native_Touch_x },
    { L"y",             am_native_Touch_y },
    { L"take_press",    am_native_Touch_take_press },
    { L"take_release",  am_native_Touch_take_release },
    { L"wheel_available", am_native_Touch_wheel_available },
    { L"wheel_enable",    am_native_Touch_wheel_enable },
    { L"wheel",           am_native_Touch_wheel },
};

const am_native_lib_entry_t am_native_Touch_lib = {
    L"Touch",
    am_native_Touch_funcs,
    sizeof(am_native_Touch_funcs) / sizeof(am_native_Touch_funcs[0])
};