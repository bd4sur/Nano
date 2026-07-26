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

// 向控制台输出缓冲区追加文本（带边界检查）。
// output 指向容量为 UI_STR_BUF_MAX_LENGTH 个 wchar_t 的缓冲区（含结尾 L'\0'），超长部分截断丢弃。
static void ui_animac_console_append(wchar_t *output, const wchar_t *suffix) {
    size_t cur_len = wcslen(output);
    if (cur_len >= (size_t)(UI_STR_BUF_MAX_LENGTH - 1)) {
        return; // 缓冲区已满，丢弃追加内容
    }
    wcsncat(output, suffix, UI_STR_BUF_MAX_LENGTH - 1 - cur_len);
}

// 解释器内存池：期望大小、最小可接受大小、自适应申请时给最大连续块留出的安全余量
#define UI_ANIMAC_POOL_SIZE   (2ULL * 1024 * 1024)
#define UI_ANIMAC_POOL_MIN    (512ULL * 1024)
#define UI_ANIMAC_POOL_MARGIN (64ULL * 1024)

int32_t ui_animac_init(Key_Event *key_event, Global_State *global_state) {
    gfx = global_state->gfx;

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

// 销毁解释器上下文，释放内存（退出控制台时调用，供其他内存大户按需使用内存）
int32_t ui_animac_close(Key_Event *key_event, Global_State *global_state) {
    if (ctx) {
        am_repl_ctx_destroy(ctx);
        ctx = NULL;
    }
    return 0;
}

int32_t ui_animac_exec(
    Key_Event *key_event, Global_State *global_state,
    wchar_t *input, wchar_t *output
) {
    // 解释器上下文未成功创建（内存不足）时，仅提示，不执行
    if (!ctx) {
        ui_animac_console_append(output, L"[错误] 解释器不可用（内存不足）");
        return -1;
    }

    // wcscat(output, L"\n刚刚输入的是：");
    // wcscat(output, input);
    // wcscat(output, L"\n");

    char *line = (char*)calloc(1024, sizeof(char));
    am_wcstombs(line, input, 1024);

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
