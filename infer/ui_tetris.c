#include <stdlib.h>
#include <stdio.h>

#include "ui_tetris.h"
#include "hal_key.h"

// ===============================================================================
// 俄罗斯方块
// ===============================================================================

#define TRS_COLS        (10)
#define TRS_ROWS        (20)
// （原 TRS_CELL/TRS_FIELD_X/TRS_FIELD_Y 为 320x240 硬编码，已运行化为下方静态量，
//    见 trs_layout()：按实际逻辑屏幕尺寸布局并填满整屏，格子保持方形）
#define TRS_DT_MAX      (0.05f)                    // 单帧最大步长（秒）

// 运行时布局（trs_layout 每帧/进入时按 gfx 尺寸计算；命中判定与绘制共用）
static int32_t s_top_h;         // 顶部虚拟按键栏高（速降/退出）
static int32_t s_bot_h;         // 底部虚拟按键栏高（左移/旋转/右移）
static int32_t s_cell;          // 场地格子边长（px，方形）
static int32_t s_field_x, s_field_y; // 场地左上角
static int32_t s_scr_w, s_scr_h;
static int32_t s_panel_y[5];    // 侧栏各行 y：标题/得分/行数/关卡/下一个
static int32_t s_preview_cell;  // “下一个”预览格子边长

// 布局：底部按键栏(H/8) 预留；场地顶部伸展到屏幕上沿(y=0)、横向居中，
// 格子边长=(H−底栏)/20（保持方形）；顶部「速降/退出」按键以覆盖层形式绘于场上方两角。
// 左右各留 ≥96px 侧栏（窄屏兜底约束）；主区尽可能填满。
static void trs_layout(Nano_GFX *gfx) {
    int32_t W = (int32_t)gfx->width, H = (int32_t)gfx->height;
    s_scr_w = W; s_scr_h = H;
    s_top_h = H / 10;
    s_bot_h = H / 8;
    int32_t cell = (H - s_bot_h) / TRS_ROWS;
    int32_t cell_wmax = (W - 192) / TRS_COLS;   // 双侧栏最小宽度约束
    if (cell > cell_wmax) cell = cell_wmax;
    if (cell < 1) cell = 1;
    s_cell = cell;
    s_field_y = 0;                              // 顶部伸展到屏幕上沿
    s_field_x = (W - TRS_COLS * cell) / 2;      // 横向居中
    static const float S_LINE_FR[5] = { 0.02f, 0.16f, 0.30f, 0.44f, 0.60f };
    for (int32_t i = 0; i < 5; i++) s_panel_y[i] = (int32_t)((H - s_bot_h) * S_LINE_FR[i]);
    s_preview_cell = (cell * 2 / 3 > 5) ? cell * 2 / 3 : 5;
}

// 虚拟按键命中判定（与 trs_draw_vkeys 共用几何）：1左移 2旋转 3右移 4速降 5退出 0未命中
static int32_t trs_vkey_hit(int32_t x, int32_t y) {
    if (y < s_top_h) {
        if (x < s_scr_w / 4) return 4;              // 左上角：速降
        if (x >= s_scr_w - s_scr_w / 4) return 5;   // 右上角：退出
        return 0;
    }
    if (y >= s_scr_h - s_bot_h) {
        int32_t t = x * 3 / s_scr_w;                // 底部三等分
        return 1 + t;
    }
    return 0;
}

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
    int32_t exit_confirm; // 退出确认模态框激活（与电子核桃控制台同款）
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

// 旋转（顺时针），带简单踢墙：依次尝试 原位/左1/右1/左2/右2（硬键与虚拟按键共用）
static void trs_rotate(void) {
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

// 绘制虚拟按键（几何与 trs_vkey_hit 严格一致）：
// 顶部左「速降」右「退出」；底部三等分「左移」「旋转」「右移」
static void trs_draw_vkeys(Nano_GFX *gfx) {
    struct { int32_t x, y, w, h; const wchar_t *label; uint8_t r, g, b; } btns[5];
    int32_t W = s_scr_w;
    int32_t tbw = W / 4;
    btns[0] = (typeof(btns[0])){ 0,           0, tbw,     s_top_h, L"速降", 46, 46, 50 };
    btns[1] = (typeof(btns[0])){ W - tbw,     0, tbw,     s_top_h, L"退出", 90, 40, 40 };
    int32_t bw = W / 3;
    btns[2] = (typeof(btns[0])){ 0,           s_scr_h - s_bot_h, bw,             s_bot_h, L"左移", 46, 46, 50 };
    btns[3] = (typeof(btns[0])){ bw,          s_scr_h - s_bot_h, bw,             s_bot_h, L"旋转", 46, 46, 50 };
    btns[4] = (typeof(btns[0])){ bw * 2,      s_scr_h - s_bot_h, W - bw * 2,     s_bot_h, L"右移", 46, 46, 50 };
    for (int32_t i = 0; i < 5; i++) {
        gfx_draw_rounded_rectangle(gfx, btns[i].x + 2, btns[i].y + 2, btns[i].w - 4, btns[i].h - 4,
            6, 6, 6, 6, btns[i].r, btns[i].g, btns[i].b, 1);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_16, (wchar_t *)btns[i].label,
            btns[i].x + btns[i].w / 2, btns[i].y + btns[i].h / 2, 240, 240, 240, 1);
    }
}

// 虚拟按键动作（左移/右移/速降；短按单次与长按反复两路径共用）
static void trs_vkey_action(int32_t hit) {
    switch (hit) {
        case 1: // 左移
            if (!s_trs.game_over && !trs_collide(s_trs.piece, s_trs.rot, s_trs.px - 1, s_trs.py)) s_trs.px--;
            break;
        case 3: // 右移
            if (!s_trs.game_over && !trs_collide(s_trs.piece, s_trs.rot, s_trs.px + 1, s_trs.py)) s_trs.px++;
            break;
        case 4: // 速降
            if (!s_trs.game_over) { trs_step_down(); s_trs.fall_acc = 0.0f; }
            break;
        default: break;
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
    s_trs.exit_confirm = 0;
    srand((uint32_t)(global_state->timestamp ^ 0xA5A5));
    s_trs.next_piece = rand() % 7;
    trs_spawn();
    s_trs.last_ts = global_state->timestamp;
    trs_layout(global_state->gfx); // 布局静态量即刻可用（首帧前命中判定有效）

    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);
    return 0;
}

int32_t ui_tetris_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 退出确认模态框激活期间：只消费模态框事件，游戏暂停（渲染侧同步停重力）
    if (s_trs.exit_confirm) {
        // 触屏按钮（松手沿 + 按下点命中，全局范式）
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            int32_t hit = ui_exit_confirm_hit(global_state, key_event->touch_down_x, key_event->touch_down_y);
            if (hit == 1) {      // 确认退出 → 小游戏菜单
                s_trs.exit_confirm = 0;
                global_state->STATE = STATE_GAME_MENU;
            }
            else if (hit == 2) { // 留下：关闭模态框（下一帧整帧重绘游戏画面）
                s_trs.exit_confirm = 0;
            }
        }
        // 硬按 A 键等价于“留下”（软按键不响应）
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc
                 && key_event->is_soft_key == 0) {
            s_trs.exit_confirm = 0;
        }
        return 0;
    }

    // 触屏虚拟按键：认 touch_edge 边沿（按下沿武装、松手沿结算，命中判定用按下点坐标）。
    // 本状态已列入 ui_app_state_is_menu 抑制表，输入层不再生成本次触摸的宫格软按键。
    //
    // 左移/右移/速降的短按单次 + 长按反复触发（范式同电子书上页/下页按钮）：
    // 按住不足 500ms 松手=单次触发；按住满 500ms 后以每帧一次的频率反复触发直到松手。
    // 松手只认 UP 边沿事件（可靠投递），不得以触屏电平作为松手判据（快照与 UP 入队
    // 存在窗口期，按电平清除会丢短按——见 ui_ebook.c 注释的实测故障）。
    static int32_t  s_vk_hold = 0;        // 0-无，否则为按住跟踪中的键号（1左移/3右移/4速降）
    static uint64_t s_vk_hold_ts = 0;     // 按下时刻（ms）
    static int32_t  s_vk_repeating = 0;   // 1-已进入反复触发

    // 按下沿：命中左移/右移/速降 → 开始按住跟踪（不在按下时触发，结算见下）
    if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
        int32_t hit = trs_vkey_hit(key_event->touch_down_x, key_event->touch_down_y);
        if (hit == 1 || hit == 3 || hit == 4) {
            s_vk_hold = hit;
            s_vk_hold_ts = global_state->timestamp;
            s_vk_repeating = 0;
        }
    }
    // 按住跟踪中：松手结算/计时/反复触发（本分支独占本次触摸序列）
    if (s_vk_hold != 0) {
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            if (!s_vk_repeating) trs_vkey_action(s_vk_hold); // 短按：单次
            s_vk_hold = 0;
            s_vk_repeating = 0;
        }
        else if (key_event->is_touching) {
            if (!s_vk_repeating && global_state->timestamp - s_vk_hold_ts >= 500) {
                s_vk_repeating = 1;
            }
            if (s_vk_repeating) trs_vkey_action(s_vk_hold);  // 长按：每帧反复
        }
        // 其余情形（电平已松但 UP 尚未到达）：保持跟踪等待 UP；即使 UP 丢失，
        // 残留状态也无害——触发只认 UP 边沿、重复只在按住时进行、下次 DOWN 重新武装
        return 0;
    }

    // 其余触屏键（旋转/退出）：松手沿单次触发
    if (key_event->touch_edge & TOUCH_EDGE_UP) {
        switch (trs_vkey_hit(key_event->touch_down_x, key_event->touch_down_y)) {
            case 2: // 旋转（单次，不参与长按反复）
                if (!s_trs.game_over) trs_rotate();
                break;
            case 5: // 退出：确认模态框
                s_trs.exit_confirm = 1;
                break;
            default: break;
        }
        return 0;
    }

    // 硬按键（保留原按键键值；软按键/触屏派生事件不响应——见 ui_app.c 触屏化原则 1）
    if (key_event->is_soft_key) return 0;

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
        trs_rotate();
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
    int32_t x = s_field_x + fx * s_cell;
    int32_t y = s_field_y + fy * s_cell;
    gfx_draw_rectangle(gfx, x, y, s_cell - 1, s_cell - 1, c[0], c[1], c[2], 1);
    gfx_draw_rectangle(gfx, x, y + s_cell - 3, s_cell - 1, 2, c[0] / 2, c[1] / 2, c[2] / 2, 1);
}

int32_t ui_tetris_render_frame(Key_Event *key_event, Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    trs_layout(gfx); // 布局随实际逻辑屏尺寸（虚拟按键命中判定与绘制共用）

    // 帧步长（ms），钳制上限防卡顿跳变；模态框存续期间暂停重力
    float dt_ms = (float)(global_state->timestamp - s_trs.last_ts);
    if (dt_ms < 0.0f) dt_ms = 0.0f;
    if (dt_ms > TRS_DT_MAX * 1000.0f) dt_ms = TRS_DT_MAX * 1000.0f;
    s_trs.last_ts = global_state->timestamp;

    // 重力
    if (!s_trs.game_over && !s_trs.exit_confirm) {
        s_trs.fall_acc += dt_ms;
        if (s_trs.fall_acc >= (float)TRS_FALL_MS(s_trs.level)) {
            s_trs.fall_acc = 0.0f;
            trs_step_down();
        }
    }

    // ---------------- 渲染 ----------------
    gfx_soft_clear(gfx);

    // 场地背景与边框（场地顶部抵屏幕上沿：上边无边框，左右/底边裁剪到屏内）
    {
        int32_t fw = TRS_COLS * s_cell, fh = TRS_ROWS * s_cell;
        int32_t bx0 = s_field_x - 2, by0 = (s_field_y - 2 > 0) ? s_field_y - 2 : 0;
        gfx_draw_rectangle(gfx, bx0, by0, fw + 4, s_field_y + fh + 2 - by0, 40, 40, 48, 1);
        bx0 = s_field_x - 1; by0 = (s_field_y - 1 > 0) ? s_field_y - 1 : 0;
        gfx_draw_rectangle(gfx, bx0, by0, fw + 2, s_field_y + fh + 1 - by0, 18, 18, 24, 1);
    }

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

    // 左侧面板：标题/得分/行数/关卡/下一个（行位随主区高度分布）
    wchar_t buf[32];
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"俄罗斯方块", 8, s_panel_y[0], 255, 255, 255, 1);
    swprintf(buf, 32, L"得分 %d", s_trs.score);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, buf, 8, s_panel_y[1], 220, 220, 220, 1);
    swprintf(buf, 32, L"行数 %d", s_trs.lines);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, buf, 8, s_panel_y[2], 220, 220, 220, 1);
    swprintf(buf, 32, L"关卡 %d", s_trs.level);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, buf, 8, s_panel_y[3], 220, 220, 220, 1);

    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"下一个:", 8, s_panel_y[4], 180, 180, 180, 1);
    int8_t ncells[4][2];
    trs_get_cells(s_trs.next_piece, 0, ncells);
    const uint8_t *nc = S_TRS_COLORS[s_trs.next_piece];
    int32_t pv_y = s_panel_y[4] + 20;
    for (int32_t i = 0; i < 4; i++) {
        gfx_draw_rectangle(gfx, 12 + ncells[i][0] * s_preview_cell, pv_y + ncells[i][1] * s_preview_cell,
            s_preview_cell - 1, s_preview_cell - 1, nc[0], nc[1], nc[2], 1);
    }

    // 虚拟按键（顶部速降/退出 + 底部左移/旋转/右移）
    trs_draw_vkeys(gfx);

    // 游戏结束遮罩（场地内居中）
    if (s_trs.game_over) {
        int32_t ov_h = s_cell * 6;
        int32_t ov_y = s_field_y + (TRS_ROWS * s_cell - ov_h) * 2 / 5;
        gfx_draw_rectangle(gfx, s_field_x + 2, ov_y, TRS_COLS * s_cell - 4, ov_h, 0, 0, 0, 1);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, L"游戏结束",
            s_field_x + TRS_COLS * s_cell / 2, ov_y + ov_h / 2 - 10, 255, 80, 80, 1);
        swprintf(buf, 32, L"得分 %d", s_trs.score);
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, buf,
            s_field_x + TRS_COLS * s_cell / 2, ov_y + ov_h / 2 + 12, 255, 255, 255, 1);
    }

    // 退出确认模态框（叠加于游戏画面之上；函数内部自带推帧）
    if (s_trs.exit_confirm) {
        ui_exit_confirm_draw(key_event, global_state);
    }

    gfx_refresh(gfx);
    return 0;
}
