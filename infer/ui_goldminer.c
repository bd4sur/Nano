#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "ui_goldminer.h"
#include "ui_icon.h"
#include "input_device.h"

// ===============================================================================
// 黄金矿工
// ===============================================================================

// ========== 贴图预留设计 ==========
// 精灵表：当前全部为 NULL —— 使用基本图形绘制原型；
// 后期将对应路径改为贴图文件（如 "/icon/gm_gold_big.png"）即可启用贴图，
// 贴图经 ui_icon_draw_centered 带 PSRAM 缓存绘制（贴图中心点对齐实体中心）。
typedef enum {
    GM_SPRITE_MINER = 0,
    GM_SPRITE_HOOK,
    GM_SPRITE_GOLD_BIG,
    GM_SPRITE_GOLD_SMALL,
    GM_SPRITE_ROCK_BIG,
    GM_SPRITE_ROCK_SMALL,
    GM_SPRITE_DIAMOND,
    GM_SPRITE_NUM
} GM_Sprite_Id;

static const char *S_GM_SPRITE_PATH[GM_SPRITE_NUM] = {
    NULL, // GM_SPRITE_MINER      例如 "/icon/gm_miner.png"
    NULL, // GM_SPRITE_HOOK       例如 "/icon/gm_hook.png"
    "/icon/default.png", // GM_SPRITE_GOLD_BIG   例如 "/icon/gm_gold_big.png"
    NULL, // GM_SPRITE_GOLD_SMALL 例如 "/icon/gm_gold_small.png"
    NULL, // GM_SPRITE_ROCK_BIG   例如 "/icon/gm_rock_big.png"
    NULL, // GM_SPRITE_ROCK_SMALL 例如 "/icon/gm_rock_small.png"
    NULL, // GM_SPRITE_DIAMOND    例如 "/icon/gm_diamond.png"
};

// ========== 场景常量 ==========
#define GM_GROUND_Y      (56)      // 地表线 y（以上为天空/矿区台面）
#define GM_PIVOT_X       (160)     // 钩子摆动支点
#define GM_PIVOT_Y       (50)
#define GM_ROPE_MIN      (16.0f)   // 绳长下限（收回完成位置）
#define GM_SWING_MAX_RAD (1.30f)   // 最大摆角（约75°）
#define GM_SWING_SPEED   (2.2f)    // 摆动角速度（rad/s，随关卡略增）
#define GM_EXTEND_SPEED  (300.0f)  // 出钩速度（px/s）
#define GM_RETRACT_SPEED (240.0f)  // 空钩回收速度（px/s）；抓到物体后除以重量
#define GM_DT_MAX        (0.05f)   // 单帧最大步长（秒），防卡顿跳变
#define GM_MAX_ITEMS     (24)

// ========== 物品 ==========
typedef enum {
    GM_ITEM_GOLD_BIG = 0,
    GM_ITEM_GOLD_SMALL,
    GM_ITEM_ROCK_BIG,
    GM_ITEM_ROCK_SMALL,
    GM_ITEM_DIAMOND,
    GM_ITEM_TYPE_NUM
} GM_Item_Type;

typedef struct {
    int32_t value;   // 分值
    float   weight;  // 重量（回收速度 = GM_RETRACT_SPEED / weight）
    float   radius;  // 碰撞/绘制半径
    uint8_t cr, cg, cb;
} GM_Item_Proto;

static const GM_Item_Proto S_GM_ITEM_PROTO[GM_ITEM_TYPE_NUM] = {
    {500, 2.5f, 13.0f, 255, 200,  30}, // GM_ITEM_GOLD_BIG
    {200, 1.2f,  8.0f, 255, 215,  60}, // GM_ITEM_GOLD_SMALL
    { 80, 3.5f, 12.0f, 150, 150, 160}, // GM_ITEM_ROCK_BIG
    { 40, 1.8f,  7.0f, 170, 170, 180}, // GM_ITEM_ROCK_SMALL
    {600, 0.8f,  6.0f, 120, 220, 255}, // GM_ITEM_DIAMOND
};

typedef struct {
    int32_t active;
    GM_Item_Type type;
    float x, y; // 中心坐标
} GM_Item;

// ========== 游戏状态 ==========
typedef enum {
    GM_PHASE_SWING = 0, // 摆动待发射
    GM_PHASE_EXTEND,    // 出钩
    GM_PHASE_RETRACT    // 回收
} GM_Phase;

typedef struct {
    GM_Phase phase;
    float angle_phase;  // 摆动相位
    float hook_angle;   // 当前钩绳与竖直方向夹角（发射后固定）
    float rope_len;
    float hook_x, hook_y; // 钩尖坐标
    int32_t grabbed;    // 抓到的物品索引（-1 无）
    int32_t score;
    int32_t level;
    int32_t items_left;
    uint64_t last_ts;
    GM_Item items[GM_MAX_ITEMS];
} GM_State;

static GM_State s_gm;

// ===============================================================================
// 精灵绘制（贴图预留）
// ===============================================================================

// 按精灵ID绘制：配置了贴图路径则走贴图（带缓存），否则绘制基本图形原型。
// (cx, cy) 为精灵中心。
static void gm_draw_sprite(Nano_GFX *gfx, GM_Sprite_Id id, int32_t cx, int32_t cy) {
    if (S_GM_SPRITE_PATH[id] != NULL) {
        ui_icon_draw_centered(gfx, S_GM_SPRITE_PATH[id], cx, cy);
        return;
    }
    switch (id) {
        case GM_SPRITE_MINER:
            // 原型：头（圆）+ 身体（矩形）
            gfx_draw_circle_fill(gfx, (uint32_t)cx, (uint32_t)(cy - 12), 6, 240, 200, 160, 1);
            gfx_draw_rectangle(gfx, (uint32_t)(cx - 8), (uint32_t)(cy - 6), 16, 14, 200, 60, 60, 1);
            break;
        case GM_SPRITE_HOOK:
            // 原型：V 形钩爪
            gfx_draw_line(gfx, (uint32_t)cx, (uint32_t)cy, (uint32_t)(cx - 6), (uint32_t)(cy + 7), 210, 210, 210, 1);
            gfx_draw_line(gfx, (uint32_t)cx, (uint32_t)cy, (uint32_t)(cx + 6), (uint32_t)(cy + 7), 210, 210, 210, 1);
            break;
        case GM_SPRITE_GOLD_BIG:
        case GM_SPRITE_GOLD_SMALL:
        case GM_SPRITE_ROCK_BIG:
        case GM_SPRITE_ROCK_SMALL: {
            const GM_Item_Proto *p = &S_GM_ITEM_PROTO[id - GM_SPRITE_GOLD_BIG];
            gfx_draw_circle_fill(gfx, (uint32_t)cx, (uint32_t)cy, (uint32_t)p->radius, p->cr, p->cg, p->cb, 1);
            gfx_draw_circle(gfx, (uint32_t)cx, (uint32_t)cy, (uint32_t)p->radius, p->cr * 3 / 4, p->cg * 3 / 4, p->cb * 3 / 4, 1);
            break;
        }
        case GM_SPRITE_DIAMOND:
            gfx_draw_triangle(gfx, (uint32_t)cx, (uint32_t)(cy - 7), (uint32_t)(cx - 7), (uint32_t)(cy + 6), (uint32_t)(cx + 7), (uint32_t)(cy + 6), 0, 120, 220, 255, 1);
            break;
        default:
            break;
    }
}

// ===============================================================================
// 关卡生成
// ===============================================================================

static void gm_generate_level(GM_State *s) {
    // 各类型数量（随关卡递增，封顶到 GM_MAX_ITEMS）
    int32_t counts[GM_ITEM_TYPE_NUM];
    counts[GM_ITEM_GOLD_BIG]   = 2;
    counts[GM_ITEM_GOLD_SMALL] = 3 + ((s->level - 1 > 3) ? 3 : s->level - 1);
    counts[GM_ITEM_ROCK_BIG]   = 1;
    counts[GM_ITEM_ROCK_SMALL] = 2 + s->level / 2;
    counts[GM_ITEM_DIAMOND]    = (s->level >= 2) ? 1 : 0;

    int32_t n = 0;
    for (int32_t t = 0; t < GM_ITEM_TYPE_NUM; t++) {
        for (int32_t k = 0; k < counts[t] && n < GM_MAX_ITEMS; k++) {
            float r = S_GM_ITEM_PROTO[t].radius;
            // 随机放置，拒绝与已放置物体重叠（最多尝试50次）
            for (int32_t attempt = 0; attempt < 50; attempt++) {
                float x = 20.0f + (float)(rand() % 280);
                float y = (float)(GM_GROUND_Y + 26) + (float)(rand() % (240 - GM_GROUND_Y - 52));
                int32_t overlap = 0;
                for (int32_t i = 0; i < n; i++) {
                    float dx = s->items[i].x - x, dy = s->items[i].y - y;
                    float rr = S_GM_ITEM_PROTO[s->items[i].type].radius + r + 8.0f;
                    if (dx * dx + dy * dy < rr * rr) { overlap = 1; break; }
                }
                if (!overlap) {
                    s->items[n].active = 1;
                    s->items[n].type = (GM_Item_Type)t;
                    s->items[n].x = x;
                    s->items[n].y = y;
                    n++;
                    break;
                }
            }
        }
    }
    s->items_left = n;
}

// ===============================================================================
// 游戏接口
// ===============================================================================

int32_t ui_goldminer_init(Key_Event *key_event, Global_State *global_state) {
    s_gm.phase = GM_PHASE_SWING;
    s_gm.angle_phase = 0.0f;
    s_gm.hook_angle = 0.0f;
    s_gm.rope_len = GM_ROPE_MIN;
    s_gm.hook_x = GM_PIVOT_X;
    s_gm.hook_y = GM_PIVOT_Y + GM_ROPE_MIN;
    s_gm.grabbed = -1;
    s_gm.score = 0;
    s_gm.level = 1;
    srand((uint32_t)(global_state->timestamp ^ 0x5A5A));
    gm_generate_level(&s_gm);
    s_gm.last_ts = global_state->timestamp;

    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);
    return 0;
}

int32_t ui_goldminer_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 按A键(ESC)返回主菜单
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_MAIN_MENU;
        return 0;
    }
    // 按D键(回车)或2键：摆动状态下发射钩子
    if ((key_event->key_edge == -1 || key_event->key_edge == -2)
        && (key_event->key_code == NANO_KEY_enter || key_event->key_code == NANO_KEY_2)
        && s_gm.phase == GM_PHASE_SWING) {
        s_gm.phase = GM_PHASE_EXTEND;
    }
    return 0;
}

int32_t ui_goldminer_render_frame(Key_Event *key_event, Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;

    // 帧步长（秒），钳制上限防卡顿跳变
    float dt = (float)(global_state->timestamp - s_gm.last_ts) / 1000.0f;
    if (dt < 0.0f) dt = 0.0f;
    if (dt > GM_DT_MAX) dt = GM_DT_MAX;
    s_gm.last_ts = global_state->timestamp;

    // ---------------- 逻辑更新 ----------------
    if (s_gm.phase == GM_PHASE_SWING) {
        s_gm.angle_phase += (GM_SWING_SPEED + 0.1f * (float)(s_gm.level - 1)) * dt;
        s_gm.hook_angle = GM_SWING_MAX_RAD * sinf(s_gm.angle_phase);
        s_gm.rope_len = GM_ROPE_MIN;
    }
    else if (s_gm.phase == GM_PHASE_EXTEND) {
        s_gm.rope_len += GM_EXTEND_SPEED * dt;
    }
    else if (s_gm.phase == GM_PHASE_RETRACT) {
        float speed = GM_RETRACT_SPEED;
        if (s_gm.grabbed >= 0) {
            speed /= S_GM_ITEM_PROTO[s_gm.items[s_gm.grabbed].type].weight;
        }
        s_gm.rope_len -= speed * dt;
        if (s_gm.rope_len <= GM_ROPE_MIN) {
            s_gm.rope_len = GM_ROPE_MIN;
            // 收回完成：结算抓到的物品
            if (s_gm.grabbed >= 0) {
                s_gm.score += S_GM_ITEM_PROTO[s_gm.items[s_gm.grabbed].type].value;
                s_gm.items[s_gm.grabbed].active = 0;
                s_gm.grabbed = -1;
                s_gm.items_left--;
                // 清空全部物品：进入下一关
                if (s_gm.items_left <= 0) {
                    s_gm.level++;
                    gm_generate_level(&s_gm);
                }
            }
            s_gm.phase = GM_PHASE_SWING;
        }
    }

    // 钩尖坐标
    s_gm.hook_x = (float)GM_PIVOT_X + s_gm.rope_len * sinf(s_gm.hook_angle);
    s_gm.hook_y = (float)GM_PIVOT_Y + s_gm.rope_len * cosf(s_gm.hook_angle);

    // 出钩：边界与抓取检测
    if (s_gm.phase == GM_PHASE_EXTEND) {
        if (s_gm.hook_x <= 3.0f || s_gm.hook_x >= 317.0f || s_gm.hook_y >= 236.0f) {
            s_gm.phase = GM_PHASE_RETRACT;
        }
        else {
            for (int32_t i = 0; i < GM_MAX_ITEMS; i++) {
                if (!s_gm.items[i].active) continue;
                float dx = s_gm.items[i].x - s_gm.hook_x;
                float dy = s_gm.items[i].y - s_gm.hook_y;
                float rr = S_GM_ITEM_PROTO[s_gm.items[i].type].radius + 4.0f;
                if (dx * dx + dy * dy <= rr * rr) {
                    s_gm.grabbed = i;
                    s_gm.phase = GM_PHASE_RETRACT;
                    break;
                }
            }
        }
    }

    // 抓到的物品跟随钩尖
    if (s_gm.grabbed >= 0) {
        s_gm.items[s_gm.grabbed].x = s_gm.hook_x;
        s_gm.items[s_gm.grabbed].y = s_gm.hook_y + 6.0f;
    }

    // ---------------- 渲染 ----------------
    gfx_soft_clear(gfx);

    // 背景：天空 + 矿区台面 + 地下
    gfx_draw_rectangle(gfx, 0, 0, gfx->width, GM_GROUND_Y, 25, 45, 75, 1);
    gfx_draw_rectangle(gfx, 0, GM_GROUND_Y, gfx->width, gfx->height - GM_GROUND_Y, 110, 75, 40, 1);
    gfx_draw_line(gfx, 0, GM_GROUND_Y, gfx->width, GM_GROUND_Y, 200, 160, 110, 1);

    // 顶栏信息
    wchar_t hud[64];
    swprintf(hud, 64, L"得分 %d  第%d关  剩余 %d", s_gm.score, s_gm.level, s_gm.items_left);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, hud, 6, 2, 255, 255, 255, 1);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"A返回 D放钩", 236, 2, 180, 180, 180, 1);

    // 物品
    for (int32_t i = 0; i < GM_MAX_ITEMS; i++) {
        if (!s_gm.items[i].active) continue;
        gm_draw_sprite(gfx, (GM_Sprite_Id)(GM_SPRITE_GOLD_BIG + s_gm.items[i].type),
            (int32_t)s_gm.items[i].x, (int32_t)s_gm.items[i].y);
    }

    // 矿工
    gm_draw_sprite(gfx, GM_SPRITE_MINER, GM_PIVOT_X, GM_PIVOT_Y - 4);

    // 钩绳（吴小林抗锯齿算法）+ 钩爪
    gfx_draw_line_anti_aliasing(gfx, (float)GM_PIVOT_X, (float)GM_PIVOT_Y, s_gm.hook_x, s_gm.hook_y, 1.0f, 230, 230, 230, 1);
    gm_draw_sprite(gfx, GM_SPRITE_HOOK, (int32_t)s_gm.hook_x, (int32_t)s_gm.hook_y);

    gfx_refresh(gfx);
    return 0;
}
