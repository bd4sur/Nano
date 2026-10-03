#include "hal_key.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <linux/kd.h>

#include "platform.h"

// ===============================================================================
// Linux 控制台全键盘 HAL（Waveshare PocketTerm35 专用）
//
// 输入链路：外壳 USB HID 全键盘（RP2040）→ 内核 kbd → 前台 VT 行律 → stdin 字节流。
// 本 HAL 参照 hal_key_ncurses_linux.c 在 PC 控制台上的键位处理（nano_tty 范式），
// 但不依赖 ncurses（显示走 framebuffer，stdin 仅用于键盘），自行完成 termios
// 原始模式与转义序列解析。
//
// 键位映射（与 nano_tty 的 PC 控制台习惯一致；掌机为横排数字键故数字键一一对应）：
//   1..9, 0        → NANO_KEY_1..9, 0   （nano_tty 的 789→123 是 PC 小键盘布局
//                                         仿电话矩阵的转置，掌机无小键盘，恒等即可）
//   *              → NANO_KEY_esc       （同 nano_tty）
//   -              → NANO_KEY_shift     （同 nano_tty：切换 汉-英-数 输入模式等）
//   +              → NANO_KEY_ctrl      （同 nano_tty：Ctrl 组合键引导键）
//   Enter          → NANO_KEY_enter     （\r / \n）
//   Backspace/Del  → NANO_KEY_esc       （同 nano_tty KEY_BACKSPACE→esc：
//                                         输入控件内删字、缓冲空时返回）
//   Esc 键         → NANO_KEY_esc       （全键盘有实体 Esc，直接利用）
//   方向键         → NANO_KEY_up/down/left/right（ESC [ A-D 及 ESC O A-D）
//   其余可打印 ASCII → 键码恒等（NANO_KEY 键值即 ASCII；业务层当前不消费实体键
//                      字母，与 nano_tty 行为一致，留作后续扩展）
//   F1-F12/Home/End/PgUp/PgDn/Insert/Tab/Ctrl+字母/Alt+组合/非ASCII UTF-8
//                  → NANO_KEY_IDLE      （业务层未消费，同 nano_tty）
//
// 上报语义：每读到一次按键只上报一个轮询周期，随后回到 IDLE——由 ui_app.c
// get_input_event 的边沿检测形成“短按(-1)”事件；按住不放时内核控制台自动重复
// （kbdrate 默认 250ms/25cps）产生重复字节，表现为连续短按，与 nano_tty 在 PC
// 控制台上的行为完全一致（终端范式下无“长按(-2)”）。
//
// 终端态管理：init 时将 stdin 置为 termios 原始模式（非规范、无回显、无信号、
// 非阻塞）；若 stdin 是 VT（systemd 服务 TTYPath=/dev/tty1 场景），同时将控制台
// 切换为 KD_GRAPHICS，禁止 fbcon 绘制（闪烁光标、内核日志）干扰 framebuffer
// 画面。atexit 与 SIGINT/SIGTERM/SIGHUP/SIGQUIT 时恢复 termios 与 KD_TEXT。
// 非 tty stdin（SSH 管道调试）跳过 termios/KD，仅保留非阻塞读，便于远程冒烟测试。
// ===============================================================================

// 转义序列收集超时（ms）：区分单独按下的 Esc 键与转义序列前缀。
// 仅在实际读到 0x1B 后阻塞等待，无输入的轮询不付出任何代价。
#define ESC_TIMEOUT_MS (30)

// 待上报键码 FIFO 容量（一次 drain 多个字节时逐个上报，防粘贴/连发丢失）
#define KEY_FIFO_CAP (16)

static struct termios s_saved_termios;
static int   s_termios_saved = 0;
static int   s_kd_fd = -1;      // 已切 KD_GRAPHICS 的 VT fd（-1=未切换/非VT）
static int   s_kd_saved_mode = KD_TEXT;
static uint8_t s_key_fifo[KEY_FIFO_CAP];
static int   s_fifo_head = 0;   // 弹出位置
static int   s_fifo_len = 0;    // 有效元素数

static void console_key_cleanup(void) {
    // 恢复 VT 光标显示（与 init 中的 "\033[?25l" 配对）
    if (s_termios_saved) (void)!write(STDOUT_FILENO, "\033[?25h", 6);
    if (s_kd_fd >= 0) {
        ioctl(s_kd_fd, KDSETMODE, s_kd_saved_mode);
        close(s_kd_fd);
        s_kd_fd = -1;
    }
    if (s_termios_saved) {
        tcsetattr(STDIN_FILENO, TCSANOW, &s_saved_termios);
        s_termios_saved = 0;
    }
}

static void console_key_signal_handler(int sig) {
    console_key_cleanup();
    // 恢复默认处理并重新触发，保持原有退出码语义（systemd 据此判定失败/正常）
    signal(sig, SIG_DFL);
    raise(sig);
    _exit(128 + sig);
}

static void fifo_push(uint8_t key) {
    if (key == NANO_KEY_IDLE) return;
    if (s_fifo_len >= KEY_FIFO_CAP) return; // 满则丢弃新键（连发场景下旧键优先）
    s_key_fifo[(s_fifo_head + s_fifo_len) % KEY_FIFO_CAP] = key;
    s_fifo_len++;
}

static uint8_t fifo_pop(void) {
    if (s_fifo_len == 0) return NANO_KEY_IDLE;
    uint8_t key = s_key_fifo[s_fifo_head];
    s_fifo_head = (s_fifo_head + 1) % KEY_FIFO_CAP;
    s_fifo_len--;
    return key;
}

// 非阻塞读一个字节；返回 1 读到，0 无数据/EOF
static int read_byte(uint8_t *out) {
    ssize_t n = read(STDIN_FILENO, out, 1);
    return (n == 1) ? 1 : 0;
}

// 带总时限地等一个字节（转义序列期间用），时限内每 1ms 轮询一次
static int read_byte_wait(uint8_t *out, int timeout_ms) {
    for (int t = 0; t < timeout_ms; t++) {
        if (read_byte(out)) return 1;
        usleep(1000);
    }
    return 0;
}

// 可打印 ASCII 的键码映射（含 nano_tty 的三个功能键约定）
static uint8_t map_printable(uint8_t ch) {
    switch (ch) {
        case '*': return NANO_KEY_esc;
        case '-': return NANO_KEY_shift;
        case '+': return NANO_KEY_ctrl;
        default:  return ch; // NANO_KEY 键值即 ASCII，恒等映射
    }
}

// 解析 CSI（ESC [ ...）序列主体（首字节 '[' 之后的部分），返回键码
static uint8_t parse_csi_body(void) {
    char seq[8];
    int  n = 0;
    uint8_t b = 0;
    // 收集参数字节/中间字节，直至最终字节（0x40-0x7E）或超时/超长
    while (n < (int)sizeof(seq)) {
        if (!read_byte_wait(&b, ESC_TIMEOUT_MS)) break;
        if (b >= 0x40 && b <= 0x7E) { // 最终字节
            seq[n++] = (char)b;
            break;
        }
        seq[n++] = (char)b;
    }
    seq[(n < (int)sizeof(seq)) ? n : (int)sizeof(seq) - 1] = '\0';

    if (n == 1) {
        switch (seq[0]) {
            case 'A': return NANO_KEY_up;
            case 'B': return NANO_KEY_down;
            case 'C': return NANO_KEY_right;
            case 'D': return NANO_KEY_left;
            default:  return NANO_KEY_IDLE; // H/F/Z、带修饰方向键等不消费
        }
    }
    if (n == 2 && seq[1] == '~') {
        switch (seq[0]) {
            case '3': return NANO_KEY_esc;  // Del 键：同 Backspace 语义（nano_tty）
            default:  return NANO_KEY_IDLE; // 2~(Insert)/5~/6~/1~/4~ 等不消费
        }
    }
    // "[A".."[E"(Linux 控制台 F1-F5)、"11~".."24~"(F1-F12) 等：不消费
    return NANO_KEY_IDLE;
}

// 解析 SS3（ESC O X）序列主体
static uint8_t parse_ss3_body(void) {
    uint8_t b = 0;
    if (!read_byte_wait(&b, ESC_TIMEOUT_MS)) return NANO_KEY_IDLE;
    switch (b) {
        case 'A': return NANO_KEY_up;
        case 'B': return NANO_KEY_down;
        case 'C': return NANO_KEY_right;
        case 'D': return NANO_KEY_left;
        default:  return NANO_KEY_IDLE; // P/Q/R/S(F1-F4)、H/F 不消费
    }
}

// 处理一个 ESC 起始的转义序列（或单独的 Esc 键）
static uint8_t parse_escape(void) {
    uint8_t b = 0;
    if (!read_byte_wait(&b, ESC_TIMEOUT_MS)) {
        return NANO_KEY_esc; // 超时无后续字节：单独的 Esc 键
    }
    if (b == '[') return parse_csi_body();
    if (b == 'O') return parse_ss3_body();
    return NANO_KEY_IDLE; // Alt+键等：不消费（同 nano_tty）
}

int32_t input_device_init() {
    // stdin 非阻塞（管道/普通文件亦可，便于 SSH 冒烟测试）
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (flags >= 0) fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    if (isatty(STDIN_FILENO)) {
        // termios 原始模式：无规范缓冲/回显/信号，Ctrl+C 等以字节形式上抛（不消费）
        if (tcgetattr(STDIN_FILENO, &s_saved_termios) == 0) {
            struct termios raw = s_saved_termios;
            cfmakeraw(&raw);
            raw.c_cc[VMIN] = 0;
            raw.c_cc[VTIME] = 0;
            if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0) {
                s_termios_saved = 1;
            }
        }
        // 若是 VT（控制台服务场景）：切 KD_GRAPHICS，禁止 fbcon 绘制干扰 framebuffer
        int kd_mode = 0;
        if (ioctl(STDIN_FILENO, KDGETMODE, &kd_mode) == 0) {
            s_kd_saved_mode = kd_mode;
            if (ioctl(STDIN_FILENO, KDSETMODE, KD_GRAPHICS) == 0) {
                s_kd_fd = dup(STDIN_FILENO);
            }
        }

        // 隐藏 VT 闪烁光标（KDSETMODE 因权限失败时的兜底；转义序列无条件可发）：
        // 服务以普通用户运行于 tty1，fbcon 光标闪烁会周期污染 framebuffer 画面
        (void)!write(STDOUT_FILENO, "\033[?25l", 6);

        atexit(console_key_cleanup);
        signal(SIGINT,  console_key_signal_handler);
        signal(SIGTERM, console_key_signal_handler);
        signal(SIGHUP,  console_key_signal_handler);
        signal(SIGQUIT, console_key_signal_handler);
    }
    return 0;
}

uint8_t input_device_read_key() {
    // FIFO 有待上报键码：每轮询周期弹出一个
    uint8_t queued = fifo_pop();
    if (queued != NANO_KEY_IDLE) return queued;

    // drain 当前 stdin 全部可用字节，解析结果入 FIFO
    uint8_t b = 0;
    while (read_byte(&b)) {
        if (b == 0x1B) {
            fifo_push(parse_escape());
        }
        else if (b == '\r' || b == '\n') {
            fifo_push(NANO_KEY_enter);
        }
        else if (b == 0x7F) {
            fifo_push(NANO_KEY_esc); // Backspace → esc（nano_tty KEY_BACKSPACE 语义）
        }
        else if (b >= 0x20 && b <= 0x7E) {
            fifo_push(map_printable(b));
        }
        // 其余：Ctrl+字母控制字节、Tab、非 ASCII UTF-8 字节 —— 不消费
    }
    return fifo_pop();
}
