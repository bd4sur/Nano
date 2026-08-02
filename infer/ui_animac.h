#ifndef __NANO_UI_ANIMAC_H__
#define __NANO_UI_ANIMAC_H__

#include "ui_app.h"
#include "platform.h"

#include "animac_core.h"

#include "am_repl.h"
#include "am_host.h"

#include "am_native_System.h"
#include "am_native_Math.h"
#include "am_native_String.h"
#include "am_native_Table.h"

#ifdef __cplusplus
extern "C" {
#endif

static Nano_GFX *gfx = NULL;

static am_repl_ctx_t *ctx = NULL;

// 启动画面（进入控制台时显示，也用于 .clear 指令恢复）
#define UI_ANIMAC_STARTUP_MESSAGE \
    L"灵机计算引擎 V2608 | M5Core2(ESP32)\n(c) 2018-2026 BD4SUR\nEnter换行, Ctrl+Enter执行, 上滑或Ctrl+0呼出键盘\nCtrl+V恢复上次输入, .ls列出文件, .load载入, .save保存, .clear清屏重置, .editor/.repl切换模式\n"

// 上一次提交的输入内容（供 Ctrl+V 恢复；空串表示无）
// 以下三个缓冲区分配于 PSRAM（ui_animac_init 分配、ui_animac_close 释放），避免常驻内部 DRAM
static wchar_t *s_animac_last_input = NULL;

// .load 指令待注入输入区的内容（空串表示无）
static wchar_t *s_animac_pending_input = NULL;

// 编辑器模式空闲泵的输出行暂存
static wchar_t *s_animac_pump_lines = NULL;

// 控制台模式：0-REPL模式 1-编辑器模式（编辑器上下文定义见下文编辑器模式区块）
static int32_t s_animac_editor_mode = 0;

// 记录上一次提交的输入（在提交执行前调用）
static void ui_animac_save_last_input(const wchar_t *input) {
    if (s_animac_last_input == NULL) {
        return;
    }
    wcsncpy(s_animac_last_input, input, UI_STR_BUF_MAX_LENGTH - 1);
    s_animac_last_input[UI_STR_BUF_MAX_LENGTH - 1] = L'\0';
}

// 向控制台输出缓冲区追加文本（带边界检查）。
// output 指向容量为 UI_STR_BUF_MAX_LENGTH 个 wchar_t 的缓冲区（含结尾 L'\0'），超长部分截断丢弃。
static void ui_animac_console_append(wchar_t *output, const wchar_t *suffix) {
    size_t cur_len = wcslen(output);
    if (cur_len >= (size_t)(UI_STR_BUF_MAX_LENGTH - 1)) {
        return; // 缓冲区已满，丢弃追加内容
    }
    wcsncat(output, suffix, UI_STR_BUF_MAX_LENGTH - 1 - cur_len);
}

// 提交输出时的预留空间（不足时触发历史清减）
#define UI_ANIMAC_OUTPUT_RESERVE (UI_STR_BUF_MAX_LENGTH / 2)

// 历史清减：控制台文本由 [历史][提示符"> "][当前输入] 组成且只增不减，
// 当尾部空闲空间不足 reserve 个 wchar 时，从头部逐行丢弃最早的历史，直到空间足够或历史耗尽。
// 同步更新输入区起点 *console_text_len 与 textarea.length。
static void ui_animac_console_trim_history(Global_State *global_state, uint32_t *console_text_len, uint32_t reserve) {
    Widget_Textarea_State *ta = &global_state->w_input_main->textarea;
    size_t cur_len = wcslen(ta->text);
    if (cur_len + reserve < (size_t)(UI_STR_BUF_MAX_LENGTH - 1)) {
        return; // 空间充足
    }
    // 历史区为 [0, *console_text_len - 2)，其后是 2 字符提示符"> "
    uint32_t hist_end = (*console_text_len >= 2) ? (*console_text_len - 2) : 0;
    size_t need = cur_len + reserve - (UI_STR_BUF_MAX_LENGTH - 1);
    // 从头逐行累计，直到累计丢弃量覆盖缺口
    uint32_t pos = 0;
    while (pos < hist_end && pos < need) {
        uint32_t nl = pos;
        while (nl < hist_end && ta->text[nl] != L'\n') nl++;
        pos = (nl < hist_end) ? (nl + 1) : hist_end;
    }
    if (pos == 0) {
        return;
    }
    memmove(ta->text, ta->text + pos, (cur_len - pos + 1) * sizeof(wchar_t));
    *console_text_len -= pos;
    ta->length = wcslen(ta->text);
}

// 解释器内存池：期望大小、最小可接受大小、自适应申请时给最大连续块留出的安全余量
#define UI_ANIMAC_POOL_SIZE   (2ULL * 1024 * 1024)
#define UI_ANIMAC_POOL_MIN    (512ULL * 1024)
#define UI_ANIMAC_POOL_MARGIN (64ULL * 1024)

// 前向声明（编辑器模式，定义见下文）
static void ui_animac_editor_stop();
static int32_t ui_animac_editor_start();

// 销毁并重建解释器上下文（ui_animac_init 与 .clear 指令共用）
static int32_t ui_animac_reset_ctx() {
    // 防御：已有解释器上下文时先销毁，避免重复创建造成内存泄漏
    if (ctx) {
        am_repl_ctx_destroy(ctx);
        ctx = NULL;
    }

    // 内存池自适应申请：2MB连续内存在碎片化的堆上可能申请失败（差几十字节也会失败），
    // 按当前最大连续块留出安全余量后申请；低于下限则判定内存不足
    // size_t largest = platform_get_largest_free_block();
    // if (largest < UI_ANIMAC_POOL_MIN + UI_ANIMAC_POOL_MARGIN) {
    //     printf("[Animac] 内存不足，无法创建解释器内存池（最大连续块 %u 字节）\n", (uint32_t)largest);
    //     return -1;
    // }
    size_t pool_size = UI_ANIMAC_POOL_SIZE;
    // if (pool_size > largest - UI_ANIMAC_POOL_MARGIN) {
    //     pool_size = largest - UI_ANIMAC_POOL_MARGIN;
    // }
    // printf("[Animac] 解释器内存池：%u 字节（最大连续块 %u）\n", (uint32_t)pool_size, (uint32_t)largest);

    ctx = am_repl_ctx_create(pool_size);
    if (!ctx) {
        printf("[Animac] 解释器内存池创建失败\n");
        return -2;
    }
    am_repl_ctx_set_js_mode(ctx, 1);

    return 0;
}

int32_t ui_animac_init(Key_Event *key_event, Global_State *global_state) {
    gfx = global_state->gfx;
    // 控制台工作缓冲区分配于 PSRAM（避免约 12KB 常驻内部 DRAM）
    if (s_animac_last_input == NULL)    s_animac_last_input    = (wchar_t *)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
    if (s_animac_pending_input == NULL) s_animac_pending_input = (wchar_t *)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
    if (s_animac_pump_lines == NULL)    s_animac_pump_lines    = (wchar_t *)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
    // 进入控制台固定从REPL模式开始（防御：销毁可能残留的编辑器运行时）
    ui_animac_editor_stop();
    s_animac_editor_mode = 0;
    return ui_animac_reset_ctx();
}

// 销毁解释器上下文，释放内存（退出控制台时调用，供其他内存大户按需使用内存）
int32_t ui_animac_close(Key_Event *key_event, Global_State *global_state) {
    ui_animac_editor_stop();
    s_animac_editor_mode = 0;
    if (ctx) {
        am_repl_ctx_destroy(ctx);
        ctx = NULL;
    }
    if (s_animac_last_input != NULL)    { free(s_animac_last_input);    s_animac_last_input = NULL; }
    if (s_animac_pending_input != NULL) { free(s_animac_pending_input); s_animac_pending_input = NULL; }
    if (s_animac_pump_lines != NULL)    { free(s_animac_pump_lines);    s_animac_pump_lines = NULL; }
    return 0;
}

// 将上一次提交的输入恢复到控制台输入区（console_text_len 为提示符之后输入区的起始位置）。
// 参照 main.cpp-ref-m5tab5 的 Ctrl+V（paste_last_input）功能。
static void ui_animac_restore_last_input(Key_Event *key_event, Global_State *global_state, uint32_t console_text_len) {
    if (s_animac_last_input == NULL || s_animac_last_input[0] == L'\0') {
        return;
    }
    Widget_Textarea_State *ta = &global_state->w_input_main->textarea;
    ta->text[console_text_len] = L'\0'; // 丢弃当前输入区内容
    ui_animac_console_append(ta->text, s_animac_last_input);
    ta->length = wcslen(ta->text);
    ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
}

// 将 .load 载入的内容注入控制台输入区（提示符之后作为可编辑输入）。返回1表示有内容注入。
// 注入前先做历史清减，为载入内容腾出空间。
static int32_t ui_animac_apply_pending_input(Global_State *global_state, uint32_t *console_text_len) {
    if (s_animac_pending_input == NULL || s_animac_pending_input[0] == L'\0') {
        return 0;
    }
    ui_animac_console_trim_history(global_state, console_text_len, (uint32_t)wcslen(s_animac_pending_input));
    Widget_Textarea_State *ta = &global_state->w_input_main->textarea;
    ui_animac_console_append(ta->text, s_animac_pending_input);
    ta->length = wcslen(ta->text);
    s_animac_pending_input[0] = L'\0';
    return 1;
}

// ===============================================================================
// 编辑器模式（参照 main.cpp-ref-m5tab5 的 Editor 模式）
//
// 与 REPL 模式的差异：
//   - 每次提交的输入视为完整代码（JS 经 am_js_to_scheme 翻译），parse/link/compile 后加载进程；
//   - 每次提交前先彻底销毁旧运行时（含未结束的异步定时任务）再从零构建，避免旧任务残留；
//   - 进程的驱动不在提交时阻塞完成，而是由 ui_animac_idle_pump 在控制台空闲期分帧驱动：
//     主进程执行完毕后运行时并不销毁，(System.set_interval ...) 注册的定时任务继续触发。
// ===============================================================================

#define UI_ANIMAC_EDITOR_BUF_LEN (512)

typedef struct {
    am_allocator_pool_t *pool;
    am_allocator_t *vm_alloc;
    am_allocator_t *heap_alloc;
    am_runtime_t *rt;
    am_pid_t pid;
    wchar_t *code_w;            // 当前运行的源码
    int32_t code_w_needs_free;  // code_w 是否需要用 free() 释放（am_js_to_scheme 返回值需要；vm_alloc 分配的随池释放）
    int32_t alive;
    // 输出行缓冲：on_tick 将运行时输出/错误 FIFO 逐字符汇入，成行后由泵搬入控制台
    wchar_t out_buf[UI_ANIMAC_EDITOR_BUF_LEN];
    uint32_t out_len;
    wchar_t err_buf[UI_ANIMAC_EDITOR_BUF_LEN];
    uint32_t err_len;
} UI_Animac_Editor;

static UI_Animac_Editor s_editor;

// 编辑器语言模式：1-JS（经 am_js_to_scheme 翻译） 0-Scheme（手动套 lambda 包装）
static int32_t s_editor_js_mode = 1;

static void ui_animac_editor_buf_putw(wchar_t *buf, uint32_t *len, wchar_t c) {
    if (*len + 1 < UI_ANIMAC_EDITOR_BUF_LEN) {
        buf[(*len)++] = c;
        buf[*len] = L'\0';
    }
}

// 运行时回调：将输出/错误 FIFO 排水到行缓冲（参照 am_repl.c 的 on_tick）
static void ui_animac_editor_on_tick(am_runtime_t *rt) {
    if (!rt || !rt->output_fifo || !rt->error_fifo) return;
    UI_Animac_Editor *ed = (UI_Animac_Editor *)am_get_runtime_host_context(rt);
    if (!ed) return;
    while (rt->output_fifo->length > 0) {
        am_value_t v = am_list_shift(rt->vm_alloc, rt->output_fifo);
        if (am_value_is_wchar(v)) {
            ui_animac_editor_buf_putw(ed->out_buf, &ed->out_len, (wchar_t)am_value_to_wchar(v));
        }
    }
    while (rt->error_fifo->length > 0) {
        am_value_t v = am_list_shift(rt->vm_alloc, rt->error_fifo);
        if (am_value_is_wchar(v)) {
            ui_animac_editor_buf_putw(ed->err_buf, &ed->err_len, (wchar_t)am_value_to_wchar(v));
        }
    }
}

static void ui_animac_editor_on_halt(am_runtime_t *rt) { (void)rt; }
static void ui_animac_editor_on_error(am_runtime_t *rt) { (void)rt; }

// 宿主时间函数适配：实现 runtime vtable 要求的带 rt 引数签名（同 am_repl.c）
static am_timestamp_t ui_animac_host_now_ms(am_runtime_t *rt) {
    (void)rt;
    return (am_timestamp_t)am_current_timestamp_in_ms();
}

static void ui_animac_host_sleep_in_ms(am_runtime_t *rt, am_timestamp_t ms) {
    (void)rt;
    am_sleep_in_ms((uint64_t)ms);
}

static const am_runtime_vtable_t S_ANIMAC_EDITOR_VTABLE = {
    .on_tick = ui_animac_editor_on_tick,
    .on_event = NULL,
    .on_halt = ui_animac_editor_on_halt,
    .on_error = ui_animac_editor_on_error,
    .sleep_in_ms = ui_animac_host_sleep_in_ms,
    .now_ms = ui_animac_host_now_ms,
};

// 彻底销毁编辑器运行时（kill 进程、销毁运行时与内存池、释放源码）
static void ui_animac_editor_stop() {
    if (s_editor.rt) {
        if (s_editor.pid != (am_pid_t)-1) {
            am_runtime_kill_process(s_editor.rt, s_editor.pid);
        }
        am_runtime_destroy(s_editor.rt);
    }
    if (s_editor.pool) {
        am_allocator_pool_destroy(s_editor.pool);
    }
    if (s_editor.code_w) {
        if (s_editor.code_w_needs_free) {
            free(s_editor.code_w);
        }
    }
    memset(&s_editor, 0, sizeof(s_editor));
    s_editor.pid = (am_pid_t)-1;
}

// 从零构建编辑器运行时（内存池自适应申请，同 ui_animac_reset_ctx 策略）
static int32_t ui_animac_editor_start() {
    s_editor.pid = (am_pid_t)-1;

    size_t largest = platform_get_largest_free_block();
    if (largest < UI_ANIMAC_POOL_MIN + UI_ANIMAC_POOL_MARGIN) {
        printf("[Animac] 内存不足，无法创建编辑器内存池（最大连续块 %u 字节）\n", (uint32_t)largest);
        return -1;
    }
    size_t pool_size = UI_ANIMAC_POOL_SIZE;
    if (pool_size > largest - UI_ANIMAC_POOL_MARGIN) {
        pool_size = largest - UI_ANIMAC_POOL_MARGIN;
    }

    s_editor.pool = am_allocator_pool_create(pool_size, &am_host_default_vtable);
    if (!s_editor.pool) {
        return -2;
    }
    s_editor.vm_alloc = am_allocator_pool_get_vm(s_editor.pool);
    s_editor.heap_alloc = am_allocator_pool_get_heap(s_editor.pool);

    // 工作目录固定为 SD 卡根目录
    s_editor.rt = am_runtime_create(s_editor.vm_alloc, s_editor.heap_alloc, L"/", &S_ANIMAC_EDITOR_VTABLE);
    if (!s_editor.rt) {
        am_allocator_pool_destroy(s_editor.pool);
        s_editor.pool = NULL;
        s_editor.vm_alloc = NULL;
        s_editor.heap_alloc = NULL;
        return -3;
    }

    am_runtime_set_default_timeslice(s_editor.rt, 8192);
    am_set_runtime_host_context(s_editor.rt, &s_editor);

    am_runtime_register_native_lib(s_editor.rt, &am_native_System_lib);
    am_runtime_register_native_lib(s_editor.rt, &am_native_Math_lib);
    am_runtime_register_native_lib(s_editor.rt, &am_native_String_lib);
    am_runtime_register_native_lib(s_editor.rt, &am_native_Table_lib);

    s_editor.alive = 1;
    return 0;
}

// 编辑器模式求值：输入视为完整代码，全量编译并加载进程（驱动交给 ui_animac_idle_pump 分帧执行）
static void ui_animac_editor_eval(Key_Event *key_event, Global_State *global_state, wchar_t *input, wchar_t *output) {
    // 每次提交先彻底销毁旧运行时（含未结束的异步定时任务），再从零构建
    ui_animac_editor_stop();
    if (ui_animac_editor_start() != 0) {
        ui_animac_console_append(output, L"\n[editor] 运行时创建失败（内存不足）");
        return;
    }

    if (s_editor_js_mode) {
        // JS 模式：源码翻译为 Scheme（翻译器已产生 ((lambda () ...)) 包装）
        s_editor.code_w = am_js_to_scheme(input);
        s_editor.code_w_needs_free = 1; // am_js_to_scheme 返回的内存由 free() 释放
        if (!s_editor.code_w) {
            ui_animac_console_append(output, L"\n[editor] JS翻译失败");
            return;
        }
    }
    else {
        // Scheme 模式：手动套一层 ((lambda () ...))，由 vm_alloc 分配（随内存池释放）
        const wchar_t *prefix = L"((lambda () \n";
        const wchar_t *suffix = L"\n))";
        size_t code_len = wcslen(prefix) + wcslen(input) + wcslen(suffix);
        s_editor.code_w = (wchar_t *)am_malloc(s_editor.vm_alloc, (code_len + 1) * sizeof(wchar_t));
        s_editor.code_w_needs_free = 0;
        if (!s_editor.code_w) {
            ui_animac_console_append(output, L"\n[editor] 内存不足");
            return;
        }
        wcscpy(s_editor.code_w, prefix);
        wcscat(s_editor.code_w, input);
        wcscat(s_editor.code_w, suffix);
    }

    wchar_t path_w[16];
    wcscpy(path_w, L"__editor__");
    am_ast_t *ast = am_parse(s_editor.vm_alloc, s_editor.code_w, path_w, 0);
    if (!ast) {
        ui_animac_console_append(output, L"\n[editor] 语法解析失败");
        return;
    }

    wchar_t base_dir_w[2] = L"/";
    am_ast_t *linked = am_link(ast, base_dir_w, am_host_read_source_from_file, NULL);
    if (!linked) {
        am_ast_destroy(ast);
        ui_animac_console_append(output, L"\n[editor] 链接失败");
        return;
    }

    am_module_t *mod = am_compile(linked, 0, 0);
    if (!mod) {
        am_ast_destroy(linked);
        ui_animac_console_append(output, L"\n[editor] 编译失败");
        return;
    }

    s_editor.pid = am_runtime_load_module(s_editor.rt, mod);
    if (s_editor.pid == (am_pid_t)-1) {
        if (mod->ilcode) am_free(s_editor.vm_alloc, mod->ilcode);
        if (mod->ast) am_ast_destroy(mod->ast);
        am_free(s_editor.vm_alloc, mod);
        ui_animac_console_append(output, L"\n[editor] 进程加载失败");
        return;
    }

    // 释放模块静态数据（已复制到进程中，后续不再需要）
    if (mod->ilcode) am_free(s_editor.vm_alloc, mod->ilcode);
    if (mod->ast) am_ast_destroy(mod->ast);
    am_free(s_editor.vm_alloc, mod);
}

// 在控制台文本的提示符 "> " 之前插入异步输出（历史区末尾），并同步输入区起点
static void ui_animac_console_insert_history(Global_State *global_state, uint32_t *console_text_len, const wchar_t *text) {
    // 先历史清减，为异步输出腾出空间
    ui_animac_console_trim_history(global_state, console_text_len, (uint32_t)wcslen(text));
    Widget_Textarea_State *ta = &global_state->w_input_main->textarea;
    uint32_t insert_pos = (*console_text_len >= 2) ? (*console_text_len - 2) : 0;
    size_t cur_len = wcslen(ta->text);
    size_t ins_len = wcslen(text);
    if (cur_len + ins_len >= (size_t)(UI_STR_BUF_MAX_LENGTH - 1)) {
        ins_len = (cur_len < (size_t)(UI_STR_BUF_MAX_LENGTH - 1)) ? (UI_STR_BUF_MAX_LENGTH - 1 - cur_len) : 0;
    }
    if (ins_len == 0) {
        return;
    }
    memmove(ta->text + insert_pos + ins_len, ta->text + insert_pos, (cur_len - insert_pos + 1) * sizeof(wchar_t));
    wcsncpy(ta->text + insert_pos, text, ins_len);
    *console_text_len += (uint32_t)ins_len;
    ta->length = wcslen(ta->text);
}

// 编辑器模式空闲泵（控制台每帧调用）：
//   1. 驱动解释器事件循环一个时间片（主进程结束后，set_interval 定时任务仍由此持续触发）；
//   2. 将行缓冲中成行的输出（错误行带 "ERR: " 前缀）插入控制台提示符之前并刷新显示。
static void ui_animac_idle_pump(Key_Event *key_event, Global_State *global_state, uint32_t *console_text_len) {
    if (!s_animac_editor_mode || !s_editor.alive || !s_editor.rt || s_animac_pump_lines == NULL) {
        return;
    }

    int32_t vm_state = am_runtime_event_handler(s_editor.rt);

    wchar_t *lines = s_animac_pump_lines; // 控制台单线程运行，无重入
    lines[0] = L'\0';

    for (int32_t pass = 0; pass < 2; pass++) {
        wchar_t *buf = (pass == 0) ? s_editor.out_buf : s_editor.err_buf;
        uint32_t *len = (pass == 0) ? &s_editor.out_len : &s_editor.err_len;
        const wchar_t *prefix = (pass == 0) ? L"" : L"ERR: ";
        uint32_t start = 0;
        for (uint32_t i = 0; i < *len; i++) {
            if (buf[i] == L'\n') {
                buf[i] = L'\0';
                ui_animac_console_append(lines, prefix);
                ui_animac_console_append(lines, buf + start);
                ui_animac_console_append(lines, L"\n");
                start = i + 1;
            }
        }
        uint32_t rem = *len - start;
        if (rem > 0 && vm_state == AM_VM_STATE_IDLE) {
            // 虚拟机空闲（进程已结束）：flush 没有尾随换行的残余行
            ui_animac_console_append(lines, prefix);
            ui_animac_console_append(lines, buf + start);
            ui_animac_console_append(lines, L"\n");
            rem = 0;
        }
        memmove(buf, buf + start, rem * sizeof(wchar_t));
        *len = rem;
        buf[rem] = L'\0';
    }

    if (lines[0] == L'\0') {
        return;
    }
    ui_animac_console_insert_history(global_state, console_text_len, lines);
    ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
}


// UI层特殊指令（以 . 开头的单行输入）拦截处理。
// 返回1-已在UI层处理完毕；返回0-非UI层指令，交给解释器（如 .help/.js/.reset 等）。
static int32_t ui_animac_handle_ui_command(Key_Event *key_event, Global_State *global_state, wchar_t *input, wchar_t *output) {
    // 仅拦截以 '.' 开头的输入（允许前导空白）
    wchar_t *p = input;
    while (*p == L' ' || *p == L'\t') p++;
    if (*p != L'.') {
        return 0;
    }
    // 多行输入不当作指令处理
    if (wcschr(p, L'\n') != NULL) {
        return 0;
    }

    // 拆分指令与参数
    wchar_t *sp = p;
    while (*sp != L'\0' && *sp != L' ' && *sp != L'\t') sp++;
    size_t cmd_len = (size_t)(sp - p);
    if (cmd_len == 0 || cmd_len >= 16) {
        return 0; // 指令过长，不视为UI指令
    }
    wchar_t cmd[16];
    wcsncpy(cmd, p, cmd_len);
    cmd[cmd_len] = L'\0';
    while (*sp == L' ' || *sp == L'\t') sp++;
    wchar_t *arg = sp;

    // .editor：切换到编辑器模式。先释放REPL全部资源（上下文+内存池+输入历史），再从零构建编辑器运行时
    if (wcscmp(cmd, L".editor") == 0) {
        if (s_animac_editor_mode) {
            ui_animac_console_append(output, L"\n[editor] 已处于编辑器模式");
            return 1;
        }
        if (ctx) {
            am_repl_ctx_destroy(ctx);
            ctx = NULL;
        }
        if (s_animac_last_input != NULL)    s_animac_last_input[0] = L'\0';
        if (s_animac_pending_input != NULL) s_animac_pending_input[0] = L'\0';
        if (ui_animac_editor_start() != 0) {
            ui_animac_console_append(output, L"\n[editor] 运行时创建失败（内存不足）");
            return 1;
        }
        s_animac_editor_mode = 1;
        ui_animac_console_append(output, L"\n[editor] 编辑器模式：每次输入视为完整代码全量执行");
        return 1;
    }

    // .repl：切换回REPL模式。先销毁编辑器运行时，再从零构建REPL上下文
    if (wcscmp(cmd, L".repl") == 0) {
        if (!s_animac_editor_mode) {
            ui_animac_console_append(output, L"\n[repl] 已处于REPL模式");
            return 1;
        }
        ui_animac_editor_stop();
        s_animac_editor_mode = 0;
        if (s_animac_last_input != NULL)    s_animac_last_input[0] = L'\0';
        if (s_animac_pending_input != NULL) s_animac_pending_input[0] = L'\0';
        ui_animac_reset_ctx(); // 忽略失败：ctx 为 NULL 时后续输入会提示解释器不可用
        ui_animac_console_append(output, L"\n[repl] REPL模式");
        return 1;
    }

    // .js / .scm：编辑器模式下切换语言并完全重置（销毁并重建编辑器运行时）；REPL模式交给解释器
    if (wcscmp(cmd, L".js") == 0 || wcscmp(cmd, L".scm") == 0) {
        if (s_animac_editor_mode) {
            s_editor_js_mode = (cmd[1] == L'j') ? 1 : 0;
            ui_animac_editor_stop();
            ui_animac_editor_start();
            ui_animac_console_append(output,
                (cmd[1] == L'j') ? L"\n[editor] JS模式（已重置）" : L"\n[editor] Scheme模式（已重置）");
            return 1;
        }
        return 0;
    }

    // .reset：编辑器模式下彻底销毁并重建运行时；REPL模式下交给解释器自身处理
    if (wcscmp(cmd, L".reset") == 0) {
        if (s_animac_editor_mode) {
            ui_animac_editor_stop();
            ui_animac_editor_start();
            ui_animac_console_append(output, L"\n[editor] 已重置运行时");
            return 1;
        }
        return 0;
    }

    // .clear：清空控制台并彻底重置解释器/编辑器运行时，恢复启动画面
    if (wcscmp(cmd, L".clear") == 0) {
        if (s_animac_editor_mode) {
            ui_animac_editor_stop();
            ui_animac_editor_start();
        }
        else {
            ui_animac_reset_ctx(); // 忽略失败：ctx 为 NULL 时后续输入会提示解释器不可用
        }
        output[0] = L'\0';
        ui_animac_console_append(output, UI_ANIMAC_STARTUP_MESSAGE);
        return 1;
    }

    // .ls：列出SD卡根目录文件
    if (wcscmp(cmd, L".ls") == 0) {
        int32_t count = list_files("/", NULL);
        if (count < 0) {
            ui_animac_console_append(output, L"\n[ls] 无法打开SD卡根目录");
        }
        else if (count == 0) {
            ui_animac_console_append(output, L"\n[ls] （空目录）");
        }
        else {
            char **names = (char **)platform_calloc((size_t)count, sizeof(char *));
            if (names == NULL || list_files("/", names) < 0) {
                ui_animac_console_append(output, L"\n[ls] 内存不足，无法列出文件");
            }
            else {
                for (int32_t i = 0; i < count; i++) {
                    if (names[i] == NULL) continue;
                    size_t n = strlen(names[i]);
                    wchar_t *name_w = (wchar_t *)calloc(n + 1, sizeof(wchar_t));
                    if (name_w != NULL) {
                        am_mbstowcs(name_w, names[i], (uint32_t)n);
                        ui_animac_console_append(output, L"\n");
                        ui_animac_console_append(output, name_w);
                        free(name_w);
                    }
                    free(names[i]);
                }
            }
            if (names != NULL) free(names);
        }
        return 1;
    }

    // .load <路径>：载入SD卡文件内容到输入区（下次提交前可编辑）
    if (wcscmp(cmd, L".load") == 0) {
        if (*arg == L'\0') {
            ui_animac_console_append(output, L"\n[load] 用法：.load /路径/文件名");
            return 1;
        }
        char path_mb[128];
        memset(path_mb, 0, sizeof(path_mb));
        am_wcstombs(path_mb, arg, sizeof(path_mb) - 1);
        uint8_t *file_buffer = NULL;
        size_t file_size = 0;
        if (platform_read_file_to_buffer(path_mb, &file_buffer, &file_size) != 0
            || file_buffer == NULL || file_size == 0) {
            ui_animac_console_append(output, L"\n[load] 无法读取文件：");
            ui_animac_console_append(output, arg);
            return 1;
        }
        // UTF-8 转宽字符存入待注入缓冲区（超出输入区容量的部分截断）
        if (s_animac_pending_input == NULL) {
            free(file_buffer);
            ui_animac_console_append(output, L"\n[load] 内存不足");
            return 1;
        }
        memset(s_animac_pending_input, 0, UI_STR_BUF_MAX_LENGTH * sizeof(wchar_t));
        uint32_t n = (uint32_t)((file_size < UI_STR_BUF_MAX_LENGTH - 1) ? file_size : UI_STR_BUF_MAX_LENGTH - 1);
        am_mbstowcs(s_animac_pending_input, (char *)file_buffer, n);
        free(file_buffer);
        ui_animac_console_append(output, L"\n[load] 已载入 ");
        ui_animac_console_append(output, arg);
        ui_animac_console_append(output, L" 到输入区");
        return 1;
    }

    // .save [路径]：保存控制台历史记录到SD卡（缺省按日期时间生成文件名）
    if (wcscmp(cmd, L".save") == 0) {
        wchar_t path_w[128];
        if (*arg != L'\0') {
            wcsncpy(path_w, arg, 127);
            path_w[127] = L'\0';
        }
        else {
            swprintf(path_w, 128, L"/animac_%04d%02d%02d_%02d%02d%02d.txt",
                global_state->year, global_state->month, global_state->day,
                global_state->hour, global_state->minute, global_state->second);
        }
        char path_mb[384];
        memset(path_mb, 0, sizeof(path_mb));
        am_wcstombs(path_mb, path_w, sizeof(path_mb) - 1);

        // 控制台全文转 UTF-8（每个宽字符至多3字节UTF-8）
        size_t mb_cap = UI_STR_BUF_MAX_LENGTH * 3 + 1;
        char *mb = (char *)platform_malloc(mb_cap);
        if (mb == NULL) {
            ui_animac_console_append(output, L"\n[save] 内存不足");
            return 1;
        }
        memset(mb, 0, mb_cap);
        am_wcstombs(mb, output, (uint32_t)(mb_cap - 1));
        int32_t ret = platform_write_buffer_to_file(path_mb, (uint8_t *)mb, strlen(mb));
        free(mb);
        if (ret != 0) {
            ui_animac_console_append(output, L"\n[save] 写入失败：");
            ui_animac_console_append(output, path_w);
        }
        else {
            ui_animac_console_append(output, L"\n[save] 已保存到 ");
            ui_animac_console_append(output, path_w);
        }
        return 1;
    }

    return 0; // 其他 . 指令交给解释器
}

int32_t ui_animac_exec(
    Key_Event *key_event, Global_State *global_state,
    wchar_t *input, wchar_t *output, uint32_t *console_text_len
) {
    // 历史清减：为本次执行的输出预留空间（不足时从头部丢弃最早的历史行）
    ui_animac_console_trim_history(global_state, console_text_len, UI_ANIMAC_OUTPUT_RESERVE);

    // UI层特殊指令（以 . 开头）：先尝试在UI层拦截处理，未识别则交给解释器。
    // 拦截先于解释器可用性检查，即使解释器不可用，.ls/.load/.save/.clear 等仍可用。
    if (ui_animac_handle_ui_command(key_event, global_state, input, output)) {
        return 0;
    }

    // 编辑器模式：输入视为完整代码全量编译执行；未识别的 . 指令不可用于编辑器模式
    if (s_animac_editor_mode) {
        wchar_t *p = input;
        while (*p == L' ' || *p == L'\t') p++;
        if (*p == L'.') {
            ui_animac_console_append(output, L"\n[editor] 该指令在编辑器模式下不可用");
            return 0;
        }
        ui_animac_editor_eval(key_event, global_state, input, output);
        return 0;
    }

    // 解释器上下文未成功创建（内存不足）时，仅提示，不执行
    if (!ctx) {
        ui_animac_console_append(output, L"[错误] 解释器不可用（内存不足）");
        return -1;
    }

    // wcscat(output, L"\n刚刚输入的是：");
    // wcscat(output, input);
    // wcscat(output, L"\n");

    // 宽字符转 UTF-8 后中文每字占 3 字节，转换缓冲按 3 倍容量准备
    size_t line_cap = UI_STR_BUF_MAX_LENGTH * 3 + 1;
    char *line = (char*)calloc(line_cap, sizeof(char));
    if (line == NULL) {
        ui_animac_console_append(output, L"[错误] 内存不足");
        return -1;
    }
    am_wcstombs(line, input, (uint32_t)line_cap);

    am_repl_result_t res = am_repl_ctx_feed(ctx, line);

    free(line);

    if (res.status == AM_REPL_STATUS_EXIT) {
        ui_animac_console_append(output, L"\n");
        return 0;
    }

    if (res.status == AM_REPL_STATUS_OUTPUT || res.status == AM_REPL_STATUS_ERROR) {
        if (res.output && res.output[0] != '\0') {
            ui_animac_console_append(output, L"\n");
            size_t len = strlen(res.output);
            wchar_t *output_buffer_w = (wchar_t*)calloc(len + 1, sizeof(wchar_t));
            am_mbstowcs(output_buffer_w, res.output, len);
            ui_animac_console_append(output, output_buffer_w);
            free(output_buffer_w);
            return 0;
        }
        if (res.error && res.error[0] != '\0') {
            ui_animac_console_append(output, L"\n");
            size_t len = strlen(res.error);
            wchar_t *output_buffer_w = (wchar_t*)calloc(len + 1, sizeof(wchar_t));
            am_mbstowcs(output_buffer_w, res.error, len);
            ui_animac_console_append(output, output_buffer_w);
            free(output_buffer_w);
            return 0;
        }
    }
    // AM_REPL_STATUS_CONTINUE：无需输出

    ui_animac_console_append(output, L"\n");
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif
