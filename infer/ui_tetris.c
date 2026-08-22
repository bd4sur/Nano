#include <stdlib.h>
#include <stdio.h>

#include "ui_tetris.h"
#include "hal_key.h"

// ===============================================================================
// 俄罗斯方块
// ===============================================================================

#define TRS_COLS        (10)
#define TRS_ROWS        (20)
#define TRS_CELL        (11)                       // 格子边长（px）
#define TRS_FIELD_X     (108)                      // 场地左上角
#define TRS_FIELD_Y     (12)
#define TRS_DT_MAX      (0.05f)                    // 单帧最大步长（秒）

// 下落间隔（ms）：随关卡递减，下限100ms
#define TRS_FALL_MS(level) ((800 - 70 * ((level) - 1)) > 100 ? (800 - 70 * ((level) - 1)) : 100)

// 7 种方块的基础形态（4x4 网格中的 4 个格子坐标）
static const int8_t S_TRS_SHAPES[7][4][2] = {
    {{0,1},{1,1},{2,1},{3,1}}, // 0-I
    {{1,1},{2,1},{1,2},{2,2}}, // 1-O
    {{1,1},{0,2},{1,2},{2,2}}, // 2-T
    {{1,1},{2,1},{0,2},{1,2}}, // 3-S
    {{0,1},{1,1},{1,2},{2,2}}, // 4-Z
    {{0,1},{0,2},{1,2},{2,2}}, // 5-J
    {{2,1},{0,2},{1,2},{2,2}}, // 6-L
};

// 方块颜色（索引 1..7 对应 field 中的非零值）
static const uint8_t S_TRS_COLORS[7][3] = {
    {  0, 220, 220}, // I 青
    {240, 220,  40}, // O 黄
    {170,  80, 220}, // T 紫
    { 60, 200,  80}, // S 绿
    {230,  60,  60}, // Z 红
    { 70, 110, 230}, // J 蓝
    {240, 150,  40}, // L 橙
};

typedef struct {
    uint8_t field[TRS_ROWS][TRS_COLS]; // 0-空，1..7-方块颜色索引
    int32_t piece;      // 当前方块类型 0..6
    int32_t rot;        // 当前旋转态 0..3
    int32_t px, py;     // 当前方块 4x4 包围盒左上角在场地中的坐标
    int32_t next_piece;
    int32_t score;
    int32_t lines;
    int32_t level;
    float   fall_acc;   // 重力累计（ms）
    int32_t game_over;
    uint64_t last_ts;
} Tetris_State;

static Tetris_State s_trs;

// 取方块在 rot 旋转态下的 4 个格子坐标（4x4 网格内，顺时针旋转 rot 次：(x,y)->(3-y,x)）
static void trs_get_cells(int32_t piece, int32_t rot, int8_t out[4][2]) {
    for (int32_t i = 0; i < 4; i++) {
        int8_t x = S_TRS_SHAPES[piece][i][0];
        int8_t y = S_TRS_SHAPES[piece][i][1];
        for (int32_t r = 0; r < rot; r++) {
            int8_t t = x;
            x = 3 - y;
            y = t;
        }
        out[i][0] = x;
        out[i][1] = y;
    }
}

// 碰撞检测：包围盒位于 (px,py) 时是否与边界/已锁定方块冲突（py<0 的格子视为合法）
static int32_t trs_collide(int32_t piece, int32_t rot, int32_t px, int32_t py) {
    int8_t cells[4][2];
    trs_get_cells(piece, rot, cells);
    for (int32_t i = 0; i < 4; i++) {
        int32_t x = px + cells[i][0];
        int32_t y = py + cells[i][1];
        if (x < 0 || x >= TRS_COLS || y >= TRS_ROWS) return 1;
        if (y >= 0 && s_trs.field[y][x] != 0) return 1;
    }
    return 0;
}

// 生成新方块；无法入场则游戏结束
static void trs_spawn() {
    s_trs.piece = s_trs.next_piece;
    s_trs.next_piece = rand() % 7;
    s_trs.rot = 0;
    s_trs.px = 3;
    s_trs.py = -1;
    if (trs_collide(s_trs.piece, s_trs.rot, s_trs.px, s_trs.py)) {
        s_trs.game_over = 1;
    }
}

// 锁定当前方块并消行计分
static void trs_lock_and_clear() {
    int8_t cells[4][2];
    trs_get_cells(s_trs.piece, s_trs.rot, cells);
    for (int32_t i = 0; i < 4; i++) {
        int32_t x = s_trs.px + cells[i][0];
        int32_t y = s_trs.py + cells[i][1];
        if (y >= 0 && y < TRS_ROWS && x >= 0 && x < TRS_COLS) {
            s_trs.field[y][x] = (uint8_t)(s_trs.piece + 1);
        }
    }

    // 消行：统计满行并整体下移
    int32_t cleared = 0;
    for (int32_t y = TRS_ROWS - 1; y >= 0; y--) {
        int32_t full = 1;
        for (int32_t x = 0; x < TRS_COLS; x++) {
            if (s_trs.field[y][x] == 0) { full = 0; break; }
        }
        if (full) {
            cleared++;
            for (int32_t yy = y; yy > 0; yy--) {
                for (int32_t x = 0; x < TRS_COLS; x++) {
                    s_trs.field[yy][x] = s_trs.field[yy - 1][x];
                }
            }
            for (int32_t x = 0; x < TRS_COLS; x++) {
                s_trs.field[0][x] = 0;
            }
            y++; // 本行被上方内容填充，需重新检查
        }
    }

    static const int32_t score_table[4] = {100, 300, 500, 800};
    if (cleared > 0) {
        s_trs.score += score_table[cleared - 1] * s_trs.level;
        s_trs.lines += cleared;
        s_trs.level = s_trs.lines / 10 + 1;
    }

    trs_spawn();
}

// 下落一格；无法下落则锁定
static void trs_step_down() {
    if (!trs_collide(s_trs.piece, s_trs.rot, s_trs.px, s_trs.py + 1)) {
        s_trs.py++;
    }
    else {
        trs_lock_and_clear();
    }
}

// ===============================================================================
// 游戏接口
// ===============================================================================

int32_t ui_tetris_init(Key_Event *key_event, Global_State *global_state) {
    for (int32_t y = 0; y < TRS_ROWS; y++) {
        for (int32_t x = 0; x < TRS_COLS; x++) {
            s_trs.field[y][x] = 0;
        }
    }
    s_trs.score = 0;
    s_trs.lines = 0;
    s_trs.level = 1;
    s_trs.fall_acc = 0.0f;
    s_trs.game_over = 0;
    srand((uint32_t)(global_state->timestamp ^ 0xA5A5));
    s_trs.next_piece = rand() % 7;
    trs_spawn();
    s_trs.last_ts = global_state->timestamp;

    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);
    return 0;
}

int32_t ui_tetris_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 按A键(ESC)返回小游戏菜单
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_GAME_MENU;
        return 0;
    }

    if (s_trs.game_over) {
        return 0;
    }

    if (key_event->key_edge != -1 && key_event->key_edge != -2) {
        return 0;
    }

    if (key_event->key_code == NANO_KEY_left) {
        if (!trs_collide(s_trs.piece, s_trs.rot, s_trs.px - 1, s_trs.py)) s_trs.px--;
    }
    else if (key_event->key_code == NANO_KEY_right) {
        if (!trs_collide(s_trs.piece, s_trs.rot, s_trs.px + 1, s_trs.py)) s_trs.px++;
    }
    else if (key_event->key_code == NANO_KEY_1) {
        // 旋转（顺时针），带简单踢墙：依次尝试 原位/左1/右1/左2/右2
        static const int8_t kicks[5] = {0, -1, 1, -2, 2};
        int32_t new_rot = (s_trs.rot + 1) % 4;
        for (int32_t k = 0; k < 5; k++) {
            if (!trs_collide(s_trs.piece, new_rot, s_trs.px + kicks[k], s_trs.py)) {
                s_trs.rot = new_rot;
                s_trs.px += kicks[k];
                break;
            }
        }
    }
    else if (key_event->key_code == NANO_KEY_2) {
        trs_step_down(); // 加速下落（按住时由长按重复事件连续触发）
        s_trs.fall_acc = 0.0f;
    }
    else if (key_event->key_code == NANO_KEY_enter) {
        // 直接落底并锁定
        while (!trs_collide(s_trs.piece, s_trs.rot, s_trs.px, s_trs.py + 1)) {
            s_trs.py++;
        }
        trs_lock_and_clear();
        s_trs.fall_acc = 0.0f;
    }
    return 0;
}

// 绘制一个场地格子（填色 + 暗色描边，留出 1px 间隙形成网格感）
static void trs_draw_cell(Nano_GFX *gfx, int32_t fx, int32_t fy, uint8_t color_idx) {
    const uint8_t *c = S_TRS_COLORS[color_idx - 1];
    int32_t x = TRS_FIELD_X + fx * TRS_CELL;
    int32_t y = TRS_FIELD_Y + fy * TRS_CELL;
    gfx_draw_rectangle(gfx, x, y, TRS_CELL - 1, TRS_CELL - 1, c[0], c[1], c[2], 1);
    gfx_draw_rectangle(gfx, x, y + TRS_CELL - 3, TRS_CELL - 1, 2, c[0] / 2, c[1] / 2, c[2] / 2, 1);
}

int32_t ui_tetris_render_frame(Key_Event *key_event, Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;

    // 帧步长（ms），钳制上限防卡顿跳变
    float dt_ms = (float)(global_state->timestamp - s_trs.last_ts);
    if (dt_ms < 0.0f) dt_ms = 0.0f;
    if (dt_ms > TRS_DT_MAX * 1000.0f) dt_ms = TRS_DT_MAX * 1000.0f;
    s_trs.last_ts = global_state->timestamp;

    // 重力
    if (!s_trs.game_over) {
        s_trs.fall_acc += dt_ms;
        if (s_trs.fall_acc >= (float)TRS_FALL_MS(s_trs.level)) {
            s_trs.fall_acc = 0.0f;
            trs_step_down();
        }
    }

    // ---------------- 渲染 ----------------
    gfx_soft_clear(gfx);

    // 场地背景与边框
    gfx_draw_rectangle(gfx, TRS_FIELD_X - 2, TRS_FIELD_Y - 2, TRS_COLS * TRS_CELL + 4, TRS_ROWS * TRS_CELL + 4, 40, 40, 48, 1);
    gfx_draw_rectangle(gfx, TRS_FIELD_X - 1, TRS_FIELD_Y - 1, TRS_COLS * TRS_CELL + 2, TRS_ROWS * TRS_CELL + 2, 18, 18, 24, 1);

    // 已锁定方块
    for (int32_t y = 0; y < TRS_ROWS; y++) {
        for (int32_t x = 0; x < TRS_COLS; x++) {
            if (s_trs.field[y][x] != 0) {
                trs_draw_cell(gfx, x, y, s_trs.field[y][x]);
            }
        }
    }

    // 当前方块
    if (!s_trs.game_over) {
        int8_t cells[4][2];
        trs_get_cells(s_trs.piece, s_trs.rot, cells);
        for (int32_t i = 0; i < 4; i++) {
            int32_t x = s_trs.px + cells[i][0];
            int32_t y = s_trs.py + cells[i][1];
            if (y >= 0) {
                trs_draw_cell(gfx, x, y, (uint8_t)(s_trs.piece + 1));
            }
        }
    }

    // 左侧面板：标题/得分/行数/关卡/下一个
    wchar_t buf[32];
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"俄罗斯方块", 8, 14, 255, 255, 255, 1);
    swprintf(buf, 32, L"得分 %d", s_trs.score);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, buf, 8, 44, 220, 220, 220, 1);
    swprintf(buf, 32, L"行数 %d", s_trs.lines);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, buf, 8, 62, 220, 220, 220, 1);
    swprintf(buf, 32, L"关卡 %d", s_trs.level);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, buf, 8, 80, 220, 220, 220, 1);

    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"下一个:", 8, 112, 180, 180, 180, 1);
    int8_t ncells[4][2];
    trs_get_cells(s_trs.next_piece, 0, ncells);
    const uint8_t *nc = S_TRS_COLORS[s_trs.next_piece];
    for (int32_t i = 0; i < 4; i++) {
        gfx_draw_rectangle(gfx, 12 + ncells[i][0] * 8, 130 + ncells[i][1] * 8, 7, 7, nc[0], nc[1], nc[2], 1);
    }

    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"1旋转 2速降", 8, 200, 150, 150, 150, 1);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"D落底 A返回", 8, 216, 150, 150, 150, 1);

    // 游戏结束遮罩
    if (s_trs.game_over) {
        gfx_draw_rectangle(gfx, TRS_FIELD_X + 2, TRS_FIELD_Y + 80, TRS_COLS * TRS_CELL - 4, 60, 0, 0, 0, 1);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, L"游戏结束", TRS_FIELD_X + TRS_COLS * TRS_CELL / 2, TRS_FIELD_Y + 96, 255, 80, 80, 1);
        swprintf(buf, 32, L"得分 %d", s_trs.score);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, buf, TRS_FIELD_X + TRS_COLS * TRS_CELL / 2, TRS_FIELD_Y + 116, 255, 255, 255, 1);
    }

    gfx_refresh(gfx);
    return 0;
}
