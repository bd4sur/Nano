// gfx_font_12.c - 12px alpha 字模（glyph_12.h）的包装编译单元
//
// 自动生成的字模头文件（glyph_12.h / glyph_16.h）内部符号（MCU_FONT_* / mcu_font_*）
// 相互冲突，无法被同一个编译单元同时包含，因此每个字模单独放在一个编译单元中，
// 对外仅暴露本文件中命名唯一的包装函数。
//
// 字体参数（见 glyph_12.h 头注释）：monospace 12px，行高15，基线12（相对行顶）

#include "graphics.h"
#include "glyph_12.h"

uint8_t gfx_font_12_get_glyph(uint32_t codepoint, uint8_t *alpha,
                              uint8_t *w, uint8_t *h,
                              int8_t *x_offset, int8_t *y_offset, uint8_t *x_advance) {
    return mcu_font_get_glyph(codepoint, alpha, w, h, x_offset, y_offset, x_advance);
}
