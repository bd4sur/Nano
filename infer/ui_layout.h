#ifndef __NANO_UI_LAYOUT_H__
#define __NANO_UI_LAYOUT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/******************************************************

网格布局示意（以 4x4 为例）
    +---------+---------+---------+---------+
    |              Padding Top              |
+---0---------1---------2---------3---------4---+
|   | (0,0)   | (1,0)   | (2,0)   | (3,0)   |   |
| L 1---------+---------+---------+---------+ R |
| E | (0,1)   | (1,1)   | (2,1)   | (3,1)   | I |
| F 2---------+---------+---------+---------+ G |
| T | (0,2)   | (1,2)   | (2,2)   | (3,2)   | H |
|   3---------+---------+---------+---------+ T |
|   | (0,3)   | (1,3)   | (2,3)   | (3,3)   |   |
+---4---------+---------+---------+---------+---|
    |            Padding Bottom             |
    +---------+---------+---------+---------+

******************************************************/

// 通用网格布局参数。padding 传 0 即表示无对应边栏（如无顶栏/无底栏/皆无）；
// cell_num_x / cell_num_y 可取任意正值，支持横纵格子数不固定的场景。
typedef struct UI_Grid_Layout {
    int32_t area_width;     // 布局区域总宽度（px，通常取 gfx->width 或 SCREEN_WIDTH）
    int32_t area_height;    // 布局区域总高度（px，通常取 gfx->height 或 SCREEN_HEIGHT）
    int32_t padding_top;    // 顶部留白（px）
    int32_t padding_bottom; // 底部留白（px）
    int32_t padding_left;   // 左侧留白（px）
    int32_t padding_right;  // 右侧留白（px）
    int32_t cell_num_x;     // 横向格子数
    int32_t cell_num_y;     // 纵向格子数
} UI_Grid_Layout;

static inline UI_Grid_Layout ui_grid_layout_make(
    int32_t area_width, int32_t area_height,
    int32_t padding_top, int32_t padding_bottom,
    int32_t padding_left, int32_t padding_right,
    int32_t cell_num_x, int32_t cell_num_y
) {
    UI_Grid_Layout layout;
    layout.area_width    = area_width;
    layout.area_height   = area_height;
    layout.padding_top    = padding_top;
    layout.padding_bottom = padding_bottom;
    layout.padding_left   = padding_left;
    layout.padding_right  = padding_right;
    layout.cell_num_x = cell_num_x;
    layout.cell_num_y = cell_num_y;
    return layout;
}

static inline int32_t ui_grid_cell_width(const UI_Grid_Layout *layout) {
    return (layout->area_width - layout->padding_left - layout->padding_right) / layout->cell_num_x;
}

static inline int32_t ui_grid_cell_height(const UI_Grid_Layout *layout) {
    return (layout->area_height - layout->padding_top - layout->padding_bottom) / layout->cell_num_y;
}

// 注意：col 允许取 0..cell_num_x，row 允许取 0..cell_num_y，
// 以便用 ui_grid_cell_x0(layout, cell_num_x) 求得网格区右边界等用途。
static inline int32_t ui_grid_cell_x0(const UI_Grid_Layout *layout, int32_t col) {
    return layout->padding_left + col * ui_grid_cell_width(layout);
}

static inline int32_t ui_grid_cell_y0(const UI_Grid_Layout *layout, int32_t row) {
    return layout->padding_top + row * ui_grid_cell_height(layout);
}

static inline int32_t ui_grid_cell_center_x(const UI_Grid_Layout *layout, int32_t col) {
    return ui_grid_cell_x0(layout, col) + ui_grid_cell_width(layout) / 2;
}

static inline int32_t ui_grid_cell_center_y(const UI_Grid_Layout *layout, int32_t row) {
    return ui_grid_cell_y0(layout, row) + ui_grid_cell_height(layout) / 2;
}

// 命中测试：(x,y) 是否落在 (col,row) 格子内（左闭右开）
static inline int32_t ui_grid_hit_test(const UI_Grid_Layout *layout, int32_t col, int32_t row, int32_t x, int32_t y) {
    return (x >= ui_grid_cell_x0(layout, col) && x < ui_grid_cell_x0(layout, col + 1) &&
            y >= ui_grid_cell_y0(layout, row) && y < ui_grid_cell_y0(layout, row + 1));
}

#ifdef __cplusplus
}
#endif

#endif
