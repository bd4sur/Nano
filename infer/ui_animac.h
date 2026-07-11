#ifndef __NANO_UI_ANIMAC_H__
#define __NANO_UI_ANIMAC_H__

#include "ui_app.h"
#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

static Nano_GFX *gfx = NULL;

int32_t ui_animac_init(Key_Event *key_event, Global_State *global_state) {
    gfx = global_state->gfx;
    return 0;
}

int32_t ui_animac_exec(
    Key_Event *key_event, Global_State *global_state,
    wchar_t *input, wchar_t *output
) {
    wcscat(output, L"\n刚刚输入的是：");
    wcscat(output, input);
    wcscat(output, L"\n");

    gfx_draw_textline_centered(gfx, L"Animac 2026", 100, 100, 0x00, 0xff, 0x00, 1);

    return 0;
}

#ifdef __cplusplus
}
#endif

#endif
