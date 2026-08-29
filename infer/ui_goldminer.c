#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "ui_goldminer.h"
#include "ui_icon.h"
#include "hal_key.h"

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
    GM_SPRITE_GOLD_1,
    GM_SPRITE_GOLD_2,
    GM_SPRITE_GOLD_3,
    GM_SPRITE_ROCK,
    GM_SPRITE_DIAMOND,
    GM_SPRITE_BONUS_1,
    GM_SPRITE_BONUS_2,
    GM_SPRITE_NUM
} GM_Sprite_Id;

static const char *S_GM_SPRITE_PATH[GM_SPRITE_NUM] = {
    PLATFORM_ROOT_DIR "/icon/gm_miner.png",   // GM_SPRITE_MINER
    PLATFORM_ROOT_DIR "/icon/gm_hook.png",    // GM_SPRITE_HOOK
    PLATFORM_ROOT_DIR "/icon/gm_gold_1.png",  // GM_SPRITE_GOLD_1
    PLATFORM_ROOT_DIR "/icon/gm_gold_2.png",  // GM_SPRITE_GOLD_2
    PLATFORM_ROOT_DIR "/icon/gm_gold_3.png",  // GM_SPRITE_GOLD_3
    PLATFORM_ROOT_DIR "/icon/gm_rock.png",    // GM_SPRITE_ROCK
    PLATFORM_ROOT_DIR "/icon/gm_diamond.png", // GM_SPRITE_DIAMOND
    PLATFORM_ROOT_DIR "/icon/gm_bonus_1.png", // GM_SPRITE_BONUS_1
    PLATFORM_ROOT_DIR "/icon/gm_bonus_2.png", // GM_SPRITE_BONUS_2
};

// ========== 场景常量 ==========
#define GM_GROUND_Y      (70)      // 地表线 y（以上为天空/矿区台面）
#define GM_PIVOT_X       (160)     // 钩子摆动支点
#define GM_PIVOT_Y       (66)
#define GM_ROPE_MIN      (16.0f)   // 绳长下限（收回完成位置）
#define GM_SWING_MAX_RAD (1.30f)   // 最大摆角（约75°）
#define GM_SWING_SPEED   (2.2f)    // 摆动角速度（rad/s，随关卡略增）
#define GM_EXTEND_SPEED  (300.0f)  // 出钩速度（px/s）
#define GM_RETRACT_SPEED (240.0f)  // 空钩回收速度（px/s）；抓到物体后除以重量
#define GM_DT_MAX        (0.05f)   // 单帧最大步长（秒），防卡顿跳变
#define GM_MAX_ITEMS     (24)

// 返回虚拟按钮（右上角）：白色半透明方底 + "返回" 文字
#define GM_BACK_BTN_X      (256)
#define GM_BACK_BTN_Y      (0)
#define GM_BACK_BTN_W      (64)
#define GM_BACK_BTN_H      (40)
#define GM_BACK_BTN_ALPHA  (80)   // 白底不透明度（gfx mode>=4 即 alpha）

// ========== 物品 ==========
// 物品类型顺序必须与 GM_SPRITE_GOLD_1 起的精灵顺序一一对应
//（渲染时经 GM_SPRITE_GOLD_1 + type 直接映射精灵ID）
typedef enum {
    GM_ITEM_GOLD_1 = 0,
    GM_ITEM_GOLD_2,
    GM_ITEM_GOLD_3,
    GM_ITEM_ROCK,
    GM_ITEM_DIAMOND,
    GM_ITEM_BONUS_1,
    GM_ITEM_BONUS_2,
    GM_ITEM_TYPE_NUM
} GM_Item_Type;

typedef struct {
    int32_t value;   // 分值
    float   weight;  // 重量（回收速度 = GM_RETRACT_SPEED / weight）
    float   radius;  // 原型图形绘制半径（碰撞/命中判定以精灵实际宽高为准）
    uint8_t cr, cg, cb;
} GM_Item_Proto;

static const GM_Item_Proto S_GM_ITEM_PROTO[GM_ITEM_TYPE_NUM] = {
    { 500, 2.5f, 13.0f, 255, 200,  30}, // GM_ITEM_GOLD_1 大金块
    { 300, 1.8f, 10.0f, 255, 208,  45}, // GM_ITEM_GOLD_2 中金块
    { 200, 1.2f,  8.0f, 255, 215,  60}, // GM_ITEM_GOLD_3 小金块
    {  80, 3.5f, 12.0f, 150, 150, 160}, // GM_ITEM_ROCK    岩石
    { 600, 0.8f,  6.0f, 120, 220, 255}, // GM_ITEM_DIAMOND 钻石
    { 350, 0.6f,  7.0f, 120, 230, 120}, // GM_ITEM_BONUS_1 福袋（轻、中分）
    {1000, 0.5f,  7.0f, 230,  90, 220}, // GM_ITEM_BONUS_2 宝箱（轻、高分）
};

typedef struct {
    int32_t active;
    GM_Item_Type type;
    float x, y;   // 中心坐标
    float w, h;   // 精灵实际宽高（贴图尺寸；无贴图时为原型包围盒）
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

// 按精灵ID绘制：配置了贴图路径则优先走贴图（带缓存），
// 贴图不可用（路径为 NULL 或读取/解码失败）时回退绘制基本图形原型。
// (cx, cy) 为精灵中心。
static void gm_draw_sprite(Nano_GFX *gfx, GM_Sprite_Id id, int32_t cx, int32_t cy) {
    if (S_GM_SPRITE_PATH[id] != NULL
        && ui_icon_draw_centered(gfx, S_GM_SPRITE_PATH[id], cx, cy) == 0) {
        return;
    }
    switch (id) {
        case GM_SPRITE_MINER:
            // 原型：头（圆）+ 身体（矩形）
            gfx_draw_circle_fill(gfx, (uint32_t)cx, (uint32_t)(cy - 12), 6, 240, 200, 160, 1);
            gfx_draw_rectangle(gfx, (uint32_t)(cx - 8), (uint32_t)(cy - 6), 16, 14, 200, 60, 60, 1);
            break;
        case GM_SPRITE_HOOK:
            // 原型：V 形钩爪（顶点位于上缘横向中点，与贴图锚点位置一致）
            gfx_draw_line(gfx, (uint32_t)cx, (uint32_t)(cy - 4), (uint32_t)(cx - 6), (uint32_t)(cy + 3), 210, 210, 210, 1);
            gfx_draw_line(gfx, (uint32_t)cx, (uint32_t)(cy - 4), (uint32_t)(cx + 6), (uint32_t)(cy + 3), 210, 210, 210, 1);
            break;
        case GM_SPRITE_GOLD_1:
        case GM_SPRITE_GOLD_2:
        case GM_SPRITE_GOLD_3:
        case GM_SPRITE_ROCK: {
            const GM_Item_Proto *p = &S_GM_ITEM_PROTO[id - GM_SPRITE_GOLD_1];
            gfx_draw_circle_fill(gfx, (uint32_t)cx, (uint32_t)cy, (uint32_t)p->radius, p->cr, p->cg, p->cb, 1);
            gfx_draw_circle(gfx, (uint32_t)cx, (uint32_t)cy, (uint32_t)p->radius, p->cr * 3 / 4, p->cg * 3 / 4, p->cb * 3 / 4, 1);
            break;
        }
        case GM_SPRITE_DIAMOND:
            gfx_draw_triangle(gfx, (uint32_t)cx, (uint32_t)(cy - 7), (uint32_t)(cx - 7), (uint32_t)(cy + 6), (uint32_t)(cx + 7), (uint32_t)(cy + 6), 0, 120, 220, 255, 1);
            break;
        case GM_SPRITE_BONUS_1:
        case GM_SPRITE_BONUS_2: {
            // 原型：方块 + 深色描边（福袋/宝箱）
            const GM_Item_Proto *p = &S_GM_ITEM_PROTO[id - GM_SPRITE_GOLD_1];
            uint32_t r = (uint32_t)p->radius;
            gfx_draw_rectangle(gfx, (uint32_t)(cx - (int32_t)r), (uint32_t)(cy - (int32_t)r), r * 2, r * 2, p->cr, p->cg, p->cb, 1);
            gfx_draw_rectangle(gfx, (uint32_t)(cx - (int32_t)r), (uint32_t)(cy - (int32_t)r), r * 2, r * 2, p->cr * 3 / 4, p->cg * 3 / 4, p->cb * 3 / 4, 0);
            break;
        }
        default:
            break;
    }
}

// 取精灵的实际尺寸（像素）：贴图可用时为贴图实际宽高，否则回退为原型图形的包围盒。
static void gm_sprite_size(GM_Sprite_Id id, float *out_w, float *out_h) {
    int32_t w = 0, h = 0;
    if (S_GM_SPRITE_PATH[id] != NULL
        && ui_icon_get_size(S_GM_SPRITE_PATH[id], &w, &h) == 0 && w > 0 && h > 0) {
        *out_w = (float)w;
        *out_h = (float)h;
        return;
    }
    switch (id) {
        case GM_SPRITE_MINER: *out_w = 16.0f; *out_h = 26.0f; break; // 头圆+身体矩形
        case GM_SPRITE_HOOK:  *out_w = 12.0f; *out_h = 8.0f;  break; // V 形钩爪
        default: {
            float r = S_GM_ITEM_PROTO[id - GM_SPRITE_GOLD_1].radius;
            *out_w = r * 2.0f;
            *out_h = r * 2.0f;
            break;
        }
    }
}

// 精灵尺寸缓存：进入游戏时一次性填充（同时触发全部贴图的SD读取/解码），
// 之后关卡生成与渲染全程查表，不再访问 ui_icon 缓存与 SD 卡
static float s_gm_sprite_w[GM_SPRITE_NUM];
static float s_gm_sprite_h[GM_SPRITE_NUM];

static void gm_sprite_size_cache_load(void) {
    for (int32_t id = 0; id < GM_SPRITE_NUM; id++) {
        gm_sprite_size((GM_Sprite_Id)id, &s_gm_sprite_w[id], &s_gm_sprite_h[id]);
    }
}

// 按精灵ID绕枢轴旋转绘制：贴图可用时 ui_icon_draw_rotated（枢轴=上缘中点），
// 否则回退为原型图形绕枢轴旋转。(pivot_x, pivot_y) 为枢轴（如绳端锚点）。
static void gm_draw_sprite_rotated(Nano_GFX *gfx, GM_Sprite_Id id, int32_t pivot_x, int32_t pivot_y, float angle_rad) {
    if (S_GM_SPRITE_PATH[id] != NULL
        && ui_icon_draw_rotated(gfx, S_GM_SPRITE_PATH[id], pivot_x, pivot_y, angle_rad) == 0) {
        return;
    }
    const float s = sinf(angle_rad), c = cosf(angle_rad);
    if (id == GM_SPRITE_HOOK) {
        // 原型：V 形钩爪，顶点在枢轴，两端点局部坐标 (±6, 7) 绕枢轴旋转
        int32_t x1 = pivot_x + (int32_t)(-6.0f * c + 7.0f * s);
        int32_t y1 = pivot_y + (int32_t)( 6.0f * s + 7.0f * c);
        int32_t x2 = pivot_x + (int32_t)( 6.0f * c + 7.0f * s);
        int32_t y2 = pivot_y + (int32_t)(-6.0f * s + 7.0f * c);
        gfx_draw_line(gfx, (uint32_t)pivot_x, (uint32_t)pivot_y, (uint32_t)x1, (uint32_t)y1, 210, 210, 210, 1);
        gfx_draw_line(gfx, (uint32_t)pivot_x, (uint32_t)pivot_y, (uint32_t)x2, (uint32_t)y2, 210, 210, 210, 1);
        return;
    }
    // 其他精灵：中心 = 枢轴沿旋转后的竖直轴偏移半个精灵高度
    float h = s_gm_sprite_h[id];
    gm_draw_sprite(gfx, id,
        pivot_x + (int32_t)(h / 2.0f * s),
        pivot_y + (int32_t)(h / 2.0f * c));
}

// ===============================================================================
// 关卡生成
// ===============================================================================

static void gm_generate_level(GM_State *s) {
    // 各类型数量（大金块/中金块每关有且仅有1个，其余随关卡递增，封顶到 GM_MAX_ITEMS）
    int32_t counts[GM_ITEM_TYPE_NUM];
    counts[GM_ITEM_GOLD_1]  = 1;
    counts[GM_ITEM_GOLD_2]  = 1;
    counts[GM_ITEM_GOLD_3]  = 3 + ((s->level - 1 > 3) ? 3 : s->level - 1);
    counts[GM_ITEM_ROCK]    = 3 + s->level / 2;
    counts[GM_ITEM_DIAMOND] = (s->level >= 2) ? 1 : 0;
    counts[GM_ITEM_BONUS_1] = 1;
    counts[GM_ITEM_BONUS_2] = (s->level >= 3) ? 1 : 0;

    int32_t n = 0;
    for (int32_t t = 0; t < GM_ITEM_TYPE_NUM; t++) {
        // 该类型精灵的实际宽高（查尺寸缓存表）
        float w = s_gm_sprite_w[GM_SPRITE_GOLD_1 + t];
        float h = s_gm_sprite_h[GM_SPRITE_GOLD_1 + t];
        // 随机放置范围：保证整个精灵不出屏幕、整体位于地下
        int32_t x_range = (int32_t)(320.0f - w - 12.0f);
        int32_t y_range = (int32_t)(240.0f - (float)GM_GROUND_Y - h - 24.0f);
        if (x_range < 1) x_range = 1;
        if (y_range < 1) y_range = 1;
        for (int32_t k = 0; k < counts[t] && n < GM_MAX_ITEMS; k++) {
            // 随机放置，拒绝与已放置物体重叠（最多尝试50次）
            for (int32_t attempt = 0; attempt < 50; attempt++) {
                float x = 6.0f + w / 2.0f + (float)(rand() % x_range);
                float y = (float)GM_GROUND_Y + 12.0f + h / 2.0f + (float)(rand() % y_range);
                int32_t overlap = 0;
                for (int32_t i = 0; i < n; i++) {
                    // 按双方精灵实际宽高做椭圆重叠判定（外扩 8px 间距）
                    float ox = (s->items[i].w + w) / 2.0f + 8.0f;
                    float oy = (s->items[i].h + h) / 2.0f + 8.0f;
                    float nx = (s->items[i].x - x) / ox;
                    float ny = (s->items[i].y - y) / oy;
                    if (nx * nx + ny * ny < 1.0f) { overlap = 1; break; }
                }
                if (!overlap) {
                    s->items[n].active = 1;
                    s->items[n].type = (GM_Item_Type)t;
                    s->items[n].x = x;
                    s->items[n].y = y;
                    s->items[n].w = w;
                    s->items[n].h = h;
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
    gm_sprite_size_cache_load(); // 进入游戏时一次性缓存全部精灵尺寸（并触发贴图加载）
    gm_generate_level(&s_gm);
    s_gm.last_ts = global_state->timestamp;

    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);
    return 0;
}

int32_t ui_goldminer_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 触屏：认 touch_edge 边沿事件（生产端高频检测+队列可靠投递，亚帧点按不湮灭，
    // 见 AGENTS.md 第八节）。松手沿触发动作，命中判定用按下点坐标 touch_down_x/y——
    // 状态切换发生在手指抬起之后，本次触摸序列不会泄漏到下一状态。
    // 本状态已列入 ui_app_state_is_menu 抑制表，输入层不再生成本次触摸的宫格软按键。
    if (key_event->touch_edge & TOUCH_EDGE_UP) {
        int32_t tx = key_event->touch_down_x;
        int32_t ty = key_event->touch_down_y;
        if (tx >= GM_BACK_BTN_X && tx < GM_BACK_BTN_X + GM_BACK_BTN_W
            && ty >= GM_BACK_BTN_Y && ty < GM_BACK_BTN_Y + GM_BACK_BTN_H) {
            global_state->STATE = STATE_GAME_MENU;
            return 0;
        }
        if (s_gm.phase == GM_PHASE_SWING) {
            s_gm.phase = GM_PHASE_EXTEND;
        }
    }

    // 硬按键（软按键/触屏派生事件不响应）
    if (key_event->is_soft_key == 0) {
        // 按A键(ESC)返回小游戏菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_GAME_MENU;
            return 0;
        }
        // 按D键(回车)或2键：摆动状态下发射钩子
        if ((key_event->key_edge == -1 || key_event->key_edge == -2)
            && (key_event->key_code == NANO_KEY_enter || key_event->key_code == NANO_KEY_2)
            && s_gm.phase == GM_PHASE_SWING) {
            s_gm.phase = GM_PHASE_EXTEND;
        }
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
                // 命中判定：钩尖进入以精灵中心为圆心的中心内接圆
                //（直径 = 精灵宽高中的较小者）即判定为命中
                float dx = s_gm.items[i].x - s_gm.hook_x;
                float dy = s_gm.items[i].y - s_gm.hook_y;
                float rr = ((s_gm.items[i].w < s_gm.items[i].h) ? s_gm.items[i].w : s_gm.items[i].h) / 2.0f;
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

    // 返回虚拟按钮：白色半透明方底 + "返回" 文字
    gfx_draw_rectangle(gfx, GM_BACK_BTN_X, GM_BACK_BTN_Y, GM_BACK_BTN_W, GM_BACK_BTN_H, 255, 255, 255, GM_BACK_BTN_ALPHA);
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, L"返回",
        GM_BACK_BTN_X + GM_BACK_BTN_W / 2, GM_BACK_BTN_Y + GM_BACK_BTN_H / 2, 255, 255, 255, 1);

    // 物品
    for (int32_t i = 0; i < GM_MAX_ITEMS; i++) {
        if (!s_gm.items[i].active) continue;
        gm_draw_sprite(gfx, (GM_Sprite_Id)(GM_SPRITE_GOLD_1 + s_gm.items[i].type),
            (int32_t)s_gm.items[i].x, (int32_t)s_gm.items[i].y);
    }

    // 矿工
    gm_draw_sprite(gfx, GM_SPRITE_MINER, GM_PIVOT_X + 20, GM_PIVOT_Y - 20);

    // 钩绳（吴小林抗锯齿算法）+ 钩爪
    // 钩子以绳端锚点 (hook_x, hook_y)（上缘横向中点）为枢轴，随绳角旋转，
    // 使贴图竖直轴线始终与绳子平行
    gfx_draw_line_anti_aliasing(gfx, (float)GM_PIVOT_X, (float)GM_PIVOT_Y, s_gm.hook_x, s_gm.hook_y, 1.0f, 230, 230, 230, 1);
    gm_draw_sprite_rotated(gfx, GM_SPRITE_HOOK, (int32_t)s_gm.hook_x, (int32_t)s_gm.hook_y, s_gm.hook_angle);

    gfx_refresh(gfx);
    return 0;
}
