// ============================================================================
// nref.c —— 最小「原生参考台」：真实 vendor 灵机解释器 + 上游本地库（含 LLM）
//
// 用途：跑同一份测试脚本两遍——一遍链接上游 (native LLM) 本地库（--llm），
// 一遍只用纯 Scheme 的 app/lib/llm_pure.scm（不注册 LLM 库），再逐行 diff 两边
// 的输出，以此证明纯 Scheme 库与本地库「完全功能等价」。见 run.py。
//
// 无图形层、无触摸：只注册 System/Math/String/Table/Fs(/LLM)，display 输出打到 stdout。
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <locale.h>
#include <stdint.h>
#include <time.h>

#include "animac.h"
#include "am_host.h"
#include "am_native_System.h"
#include "am_native_Math.h"
#include "am_native_String.h"
#include "am_native_Table.h"
#include "am_native_Fs.h"
#include "am_native_LLM.h"

static am_allocator_t *g_vm, *g_heap;

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)(ts.tv_nsec / 1000000);
}

static void on_tick(am_runtime_t *rt) { (void)rt; }
static void on_event(am_runtime_t *rt) { (void)rt; }
static void on_halt(am_runtime_t *rt) { (void)rt; }
static void on_error(am_runtime_t *rt) { (void)rt; }
static void sleep_in_ms(uint32_t ms) { (void)ms; }

static void drain_out(am_runtime_t *rt) {
    if (!rt->output_fifo) return;
    while (rt->output_fifo->length > 0) {
        am_value_t v = am_list_shift(rt->vm_alloc, rt->output_fifo);
        if (!am_value_is_wchar(v)) continue;
        wchar_t one[2] = { (wchar_t)am_value_to_wchar(v), 0 };
        char mb[8];
        uint32_t l = am_wcstombs(mb, one, sizeof(mb) - 1);
        if (!l) continue;
        mb[l] = 0;
        fputs(mb, stdout);
    }
    fflush(stdout);
}

static void drain_err(am_runtime_t *rt) {
    if (!rt->error_fifo) return;
    while (rt->error_fifo->length > 0) {
        am_value_t v = am_list_shift(rt->vm_alloc, rt->error_fifo);
        if (!am_value_is_wchar(v)) continue;
        wchar_t one[2] = { (wchar_t)am_value_to_wchar(v), 0 };
        char mb[8];
        uint32_t l = am_wcstombs(mb, one, sizeof(mb) - 1);
        if (!l) continue;
        mb[l] = 0;
        fputs(mb, stdout);
    }
    fflush(stdout);
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "");
    if (argc < 3) {
        fprintf(stderr, "usage: %s <script.scm> <module_root> [--llm]\n", argv[0]);
        return 2;
    }
    int with_llm = (argc > 3 && strcmp(argv[3], "--llm") == 0);

    am_allocator_pool_t *pool = am_allocator_pool_create(512ull * 1024 * 1024,
                                                        &am_host_default_vtable);
    if (!pool) { fprintf(stderr, "pool fail\n"); return 3; }
    g_vm = am_allocator_pool_get_vm(pool);
    g_heap = am_allocator_pool_get_heap(pool);

    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "open fail: %s\n", argv[1]); return 3; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;
    size_t wl = mbstowcs(NULL, buf, 0);
    wchar_t *wb = malloc((wl + 1) * sizeof(wchar_t));
    mbstowcs(wb, buf, wl + 1);
    free(buf);

    const wchar_t *pre = L"((lambda () \n", *suf = L"\n))";
    wchar_t *code = (wchar_t *)am_malloc(g_vm, (wcslen(pre) + wl + wcslen(suf) + 2) * sizeof(wchar_t));
    wcscpy(code, pre);
    wcscat(code, wb);
    wcscat(code, suf);
    free(wb);

    wchar_t path_w[512], root_w[512];
    am_mbstowcs(path_w, argv[1], 512);
    am_mbstowcs(root_w, argv[2], 512);

    am_ast_t *ast = am_parse(g_vm, code, path_w, 0);
    if (!ast) { printf("RESULT parse_fail\n"); return 4; }
    am_ast_t *linked = am_link(ast, root_w, am_host_read_source_from_file, NULL);
    if (!linked) { printf("RESULT link_fail\n"); return 4; }
    am_module_t *mod = am_compile(linked, 0, 0);
    if (!mod) { printf("RESULT compile_fail\n"); return 4; }

    am_runtime_vtable_t vt = { .on_tick = on_tick, .on_event = on_event, .on_halt = on_halt,
                               .on_error = on_error, .sleep_in_ms = sleep_in_ms,
                               .now_ms = now_ms };
    am_runtime_t *rt = am_runtime_create(g_vm, g_heap, root_w, &vt);
    if (!rt) { printf("RESULT runtime_create_fail\n"); return 4; }
    am_runtime_set_default_timeslice(rt, 4000000u);
    am_set_runtime_host_context(rt, NULL);
    am_runtime_register_native_lib(rt, &am_native_System_lib);
    am_runtime_register_native_lib(rt, &am_native_Math_lib);
    am_runtime_register_native_lib(rt, &am_native_String_lib);
    am_runtime_register_native_lib(rt, &am_native_Table_lib);
    am_runtime_register_native_lib(rt, &am_native_Fs_lib);
    if (with_llm) am_runtime_register_native_lib(rt, &am_native_LLM_lib);

    am_pid_t pid = am_runtime_load_module(rt, mod);
    if (pid == (am_pid_t)-1) { printf("RESULT load_fail\n"); return 4; }
    am_process_t *proc = am_rumtime_get_process_by_pid(rt, pid);
    am_set_process_host_context(rt, proc, NULL);

    int rounds = 0;
    for (;;) {
        am_runtime_event_handler(rt);
        drain_out(rt);
        drain_err(rt);
        proc = am_rumtime_get_process_by_pid(rt, pid);
        if (proc && (proc->state == AM_PROCESS_STATE_STOPPED ||
                     proc->state == AM_PROCESS_STATE_KILLED)) break;
        if (++rounds > 20000000) { printf("RESULT timeout\n"); return 5; }
    }
    printf("RESULT done\n");
    return 0;
}
