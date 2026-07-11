#ifndef __NANO_UI_ANIMAC_H__
#define __NANO_UI_ANIMAC_H__

#include "ui_app.h"
#include "platform.h"

#include "repl.h"

#ifdef __cplusplus
extern "C" {
#endif

static Nano_GFX *gfx = NULL;

static am_repl_ctx_t *ctx = NULL;

int32_t ui_animac_init(Key_Event *key_event, Global_State *global_state) {
    gfx = global_state->gfx;

    ctx = am_repl_ctx_create();
    am_repl_ctx_set_js_mode(ctx, 1);

    return 0;
}

int32_t ui_animac_exec(
    Key_Event *key_event, Global_State *global_state,
    wchar_t *input, wchar_t *output
) {
    // wcscat(output, L"\n刚刚输入的是：");
    // wcscat(output, input);
    // wcscat(output, L"\n");

    char *line = (char*)calloc(1024, sizeof(char));
    _wcstombs(line, input, 1024);

    am_repl_result_t res = am_repl_ctx_feed(ctx, line);

    free(line);

    if (res.status == AM_REPL_STATUS_EXIT) {
        wcscat(output, L"\n");
        return 0;
    }

    if (res.status == AM_REPL_STATUS_OUTPUT || res.status == AM_REPL_STATUS_ERROR) {
        if (res.output && res.output[0] != '\0') {
            wcscat(output, L"\n");
            size_t len = strlen(res.output);
            wchar_t *output_buffer_w = (wchar_t*)calloc(len, sizeof(wchar_t));
            _mbstowcs(output_buffer_w, res.output, len);
            wcscat(output, output_buffer_w);
            free(output_buffer_w);
            return 0;
        }
        if (res.error && res.error[0] != '\0') {
            wcscat(output, L"\n");
            size_t len = strlen(res.error);
            wchar_t *output_buffer_w = (wchar_t*)calloc(len, sizeof(wchar_t));
            _mbstowcs(output_buffer_w, res.error, len);
            wcscat(output, output_buffer_w);
            free(output_buffer_w);
            return 0;
        }
    }
    // AM_REPL_STATUS_CONTINUE：无需输出

    wcscat(output, L"\n");
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif
