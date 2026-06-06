#ifndef __NANO_UI_TSP_H__
#define __NANO_UI_TSP_H__

#include "ui_app.h"

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================================
// TSP 遗传算法：34 个中国主要城市
// ===============================================================================

static const float tsp_map[34][2] = {
    {116.46f, 39.92f},   // 0  北京
    {117.20f, 39.13f},   // 1  天津
    {121.48f, 31.22f},   // 2  上海
    {106.54f, 29.59f},   // 3  重庆
    {91.11f,  29.97f},   // 4  拉萨
    {87.68f,  43.77f},   // 5  乌鲁木齐
    {106.27f, 38.47f},   // 6  银川
    {111.65f, 40.82f},   // 7  呼和浩特
    {108.33f, 22.84f},   // 8  南宁
    {126.63f, 45.75f},   // 9  哈尔滨
    {125.35f, 43.88f},   // 10 长春
    {123.38f, 41.80f},   // 11 沈阳
    {114.48f, 38.03f},   // 12 石家庄
    {112.53f, 37.87f},   // 13 太原
    {101.74f, 36.56f},   // 14 西宁
    {117.00f, 36.65f},   // 15 济南
    {113.60f, 34.76f},   // 16 郑州
    {118.78f, 32.04f},   // 17 南京
    {117.27f, 31.86f},   // 18 合肥
    {120.19f, 30.26f},   // 19 杭州
    {119.30f, 26.08f},   // 20 福州
    {115.89f, 28.68f},   // 21 南昌
    {113.00f, 28.21f},   // 22 长沙
    {114.31f, 30.52f},   // 23 武汉
    {113.23f, 23.16f},   // 24 广州
    {121.50f, 25.05f},   // 25 台北
    {110.35f, 20.02f},   // 26 海口
    {103.73f, 36.03f},   // 27 兰州
    {108.95f, 34.27f},   // 28 西安
    {104.06f, 30.67f},   // 29 成都
    {106.71f, 26.57f},   // 30 贵阳
    {102.73f, 25.04f},   // 31 昆明
    {114.10f, 22.20f},   // 32 香港
    {113.33f, 22.13f}    // 33 澳门
};

static wchar_t *tsp_city_names[34] = {
    L"PEK", L"TSN", L"SHA", L"CKG",
    L"LXA", L"URC", L"INC", L"HET",
    L"NNG", L"HRB", L"CGQ", L"SHE",
    L"SJW", L"TYN", L"XNN", L"TNA",
    L"CGO", L"NKG", L"HFE", L"HGH",
    L"FOC", L"KHN", L"CSX", L"WUH",
    L"CAN", L"TPE", L"HAK", L"LHW",
    L"XIY", L"TFU", L"KWE", L"KMG",
    L"HKG", L"MFM"
};

#define TSP_MUTATION_PROB  (0.01f)
#define TSP_CROSSOVER_PROB (0.2f)
#define TSP_CITY_NUM       (34)
#define TSP_POP_SIZE       (1000)
#define TSP_HISTORY_SIZE   (300)

// 屏幕布局（固定 320 x 240）
#define TSP_SCREEN_W       (320)
#define TSP_SCREEN_H       (240)
#define TSP_HEADER_H       (14)
#define TSP_MAP_TOP        (14)
#define TSP_MAP_BOTTOM     (210)
#define TSP_MAP_H          (196)
#define TSP_CURVE_TOP      (216)
#define TSP_CURVE_BOTTOM   (238)
#define TSP_CURVE_H        (22)

typedef struct {
    int gene[TSP_CITY_NUM];
} TSP_Individual;

typedef struct {
    TSP_Individual population[TSP_POP_SIZE];
    float fitness[TSP_POP_SIZE];
    float total_fitness;
    int best_id;
    float best_fitness;
} TSP_Eden;

static TSP_Eden s_tsp_eden;
static TSP_Individual s_tsp_new_population[TSP_POP_SIZE];
static uint64_t s_tsp_rng_state = 0;
static uint32_t s_tsp_generation = 0;
static int32_t s_tsp_initialized = 0;
static float s_tsp_best_distance = 0.0f;
static float s_tsp_history[TSP_HISTORY_SIZE];
static int s_tsp_history_count = 0;

static inline float tsp_random_float(void) {
    return random_f32(&s_tsp_rng_state);
}

static inline int tsp_random_int(int n) {
    return (int)(random_u32(&s_tsp_rng_state) % (uint32_t)n);
}

static void tsp_shuffle(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }
    for (int i = n - 1; i > 0; i--) {
        int j = tsp_random_int(i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

static float tsp_distance(int a, int b) {
    float dx = tsp_map[a][0] - tsp_map[b][0];
    float dy = tsp_map[a][1] - tsp_map[b][1];
    return sqrtf(dx * dx + dy * dy);
}

static float tsp_path_length(const int *path) {
    float sum = 0.0f;
    for (int i = 1; i < TSP_CITY_NUM; i++) {
        int prev = path[i - 1];
        int curr = path[i];
        // 加入一些偏好约束
        // 乌市-拉萨优先
        if ((prev == 4 && curr == 5) || (prev == 5 && curr == 4)) {
            sum -= 0.1f;
        }
        // 乌市-西宁优先
        else if ((prev == 14 && curr == 5) || (prev == 5 && curr == 14)) {
            sum -= 0.1f;
        }
        // 乌市-哈市排除
        else if ((prev == 9 && curr == 5) || (prev == 5 && curr == 9)) {
            sum += 100000.0f;
        }
        else {
            sum += tsp_distance(prev, curr);
        }
    }
    sum += tsp_distance(path[0], path[TSP_CITY_NUM - 1]);
    return sum;
}

static float tsp_individual_decode(const TSP_Individual *ind) {
    float plen = tsp_path_length(ind->gene);
    if (plen <= 0.0f) {
        return 0.0f;
    }
    return (float)TSP_CITY_NUM / plen;
}

static void tsp_individual_mutate(TSP_Individual *ind) {
    int p1 = tsp_random_int(TSP_CITY_NUM);
    int p2 = tsp_random_int(TSP_CITY_NUM);
    int tmp = ind->gene[p1];
    ind->gene[p1] = ind->gene[p2];
    ind->gene[p2] = tmp;
}

static void tsp_individual_crossover(const TSP_Individual *a, const TSP_Individual *b, TSP_Individual *child) {
    int p1 = tsp_random_int(TSP_CITY_NUM);
    int p2 = tsp_random_int(TSP_CITY_NUM);
    int pos1 = (p1 < p2) ? p1 : p2;
    int pos2 = (p1 < p2) ? p2 : p1;

    int used[TSP_CITY_NUM];
    for (int i = 0; i < TSP_CITY_NUM; i++) {
        used[i] = 0;
        child->gene[i] = -1;
    }

    for (int i = pos1; i <= pos2; i++) {
        child->gene[i] = a->gene[i];
        used[a->gene[i]] = 1;
    }

    int b_idx = 0;
    for (int i = 0; i < TSP_CITY_NUM; i++) {
        if (child->gene[i] == -1) {
            while (b_idx < TSP_CITY_NUM && used[b->gene[b_idx]]) {
                b_idx++;
            }
            if (b_idx < TSP_CITY_NUM) {
                child->gene[i] = b->gene[b_idx];
                used[b->gene[b_idx]] = 1;
                b_idx++;
            }
        }
    }
}

static void tsp_eden_evaluate(TSP_Eden *eden) {
    float max_value = -1e30f;
    int max_index = 0;
    float total = 0.0f;
    for (int i = 0; i < TSP_POP_SIZE; i++) {
        float fit = tsp_individual_decode(&eden->population[i]);
        eden->fitness[i] = fit;
        total += fit;
        if (fit >= max_value) {
            max_value = fit;
            max_index = i;
        }
    }
    eden->total_fitness = total;
    eden->best_id = max_index;
    eden->best_fitness = max_value;
}

static int tsp_eden_roulette_wheel(TSP_Eden *eden) {
    float pointer = tsp_random_float();
    float position = 0.0f;
    for (int i = 0; i < TSP_POP_SIZE; i++) {
        position += eden->fitness[i] / eden->total_fitness;
        if (pointer <= position) {
            return i;
        }
    }
    return TSP_POP_SIZE - 1;
}

static void tsp_eden_next_generation(TSP_Eden *eden) {
    s_tsp_new_population[0] = eden->population[eden->best_id];

    int count = 1;
    while (count < TSP_POP_SIZE) {
        int p1 = tsp_eden_roulette_wheel(eden);
        TSP_Individual child;
        if (tsp_random_float() <= TSP_CROSSOVER_PROB) {
            int p2 = tsp_eden_roulette_wheel(eden);
            tsp_individual_crossover(&eden->population[p1], &eden->population[p2], &child);
        } else {
            child = eden->population[p1];
        }
        if (tsp_random_float() <= TSP_MUTATION_PROB) {
            tsp_individual_mutate(&child);
        }
        s_tsp_new_population[count++] = child;
    }

    for (int i = 0; i < TSP_POP_SIZE; i++) {
        eden->population[i] = s_tsp_new_population[i];
    }
}

static void tsp_eden_evolve(TSP_Eden *eden) {
    tsp_eden_evaluate(eden);
    tsp_eden_next_generation(eden);
}

static void tsp_eden_init(TSP_Eden *eden) {
    for (int i = 0; i < TSP_POP_SIZE; i++) {
        tsp_shuffle(eden->population[i].gene, TSP_CITY_NUM);
    }
    eden->total_fitness = 0.0f;
    eden->best_id = 0;
    eden->best_fitness = 0.0f;
    tsp_eden_evaluate(eden);
}

static void tsp_lonlat_to_screen_f(float lon, float lat, float *sx, float *sy) {
    int margin = 6;
    float lon_min = 85.0f;
    float lon_max = 130.0f;
    float lat_min = 18.0f;
    float lat_max = 48.0f;
    int map_w = TSP_SCREEN_W - margin * 2;
    int map_h = TSP_MAP_BOTTOM - TSP_MAP_TOP;
    if (map_w < 1) map_w = 1;
    if (map_h < 1) map_h = 1;
    *sx = (float)margin + (lon - lon_min) * (float)map_w / (lon_max - lon_min);
    *sy = (float)TSP_MAP_TOP + (lat_max - lat) * (float)map_h / (lat_max - lat_min);
}

// ===============================================================================
// UI 接口
// ===============================================================================

void ui_app_tsp_init(Key_Event *key_event, Global_State *global_state) {
    // if (s_tsp_initialized) {
    //     return;
    // }
    (void)key_event;
    s_tsp_rng_state = global_state->timestamp;
    s_tsp_generation = 0;
    s_tsp_best_distance = 0.0f;
    s_tsp_history_count = 0;
    for (int i = 0; i < TSP_HISTORY_SIZE; i++) {
        s_tsp_history[i] = 0.0f;
    }

    tsp_eden_init(&s_tsp_eden);
    tsp_eden_evolve(&s_tsp_eden);

    s_tsp_generation = 1;
    s_tsp_initialized = 1;
}

void ui_app_tsp_refresh(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    if (!s_tsp_initialized) {
        ui_app_tsp_init(key_event, global_state);
    }

    tsp_eden_evolve(&s_tsp_eden);

    s_tsp_best_distance = (float)TSP_CITY_NUM / s_tsp_eden.best_fitness;

    // 记录历史
    s_tsp_history[s_tsp_history_count % TSP_HISTORY_SIZE] = s_tsp_best_distance;
    s_tsp_history_count++;

    gfx_soft_clear(global_state->gfx);

    // 状态栏
    gfx_draw_rectangle(global_state->gfx, 0, 0, TSP_SCREEN_W, TSP_HEADER_H, 39, 39, 39, 3);
    wchar_t header[64];
    swprintf(header, 64, L"TSP | 距离:%.1f | 代数:%u", s_tsp_best_distance, s_tsp_generation);
    gfx_draw_textline(global_state->gfx, header, 2, 1, 255, 255, 255, 1);

    // 缓存城市屏幕坐标（浮点 + 整数四舍五入）
    float city_fx[TSP_CITY_NUM];
    float city_fy[TSP_CITY_NUM];
    int city_sx[TSP_CITY_NUM];
    int city_sy[TSP_CITY_NUM];
    for (int i = 0; i < TSP_CITY_NUM; i++) {
        tsp_lonlat_to_screen_f(tsp_map[i][0], tsp_map[i][1], &city_fx[i], &city_fy[i]);
        city_sx[i] = (int)(city_fx[i] + 0.5f);
        city_sy[i] = (int)(city_fy[i] + 0.5f);
        if (city_sx[i] < 0) city_sx[i] = 0;
        if (city_sx[i] >= TSP_SCREEN_W) city_sx[i] = TSP_SCREEN_W - 1;
        if (city_sy[i] < TSP_MAP_TOP) city_sy[i] = TSP_MAP_TOP;
        if (city_sy[i] >= TSP_MAP_BOTTOM) city_sy[i] = TSP_MAP_BOTTOM - 1;
    }

    const int *best_gene = s_tsp_eden.population[s_tsp_eden.best_id].gene;

    // 绘制城市点（红色）
    for (int i = 0; i < TSP_CITY_NUM; i++) {
        gfx_draw_circle_fill(global_state->gfx, (uint32_t)city_sx[i], (uint32_t)city_sy[i], 2, 255, 0, 0, 1);
    }

    // 绘制城市名称（白色）
    for (int i = 0; i < TSP_CITY_NUM; i++) {
        int tx = city_sx[i] + 3;
        int ty = city_sy[i] + 3;
        gfx_draw_textline_mini(global_state->gfx, tsp_city_names[i], tx, ty, 255, 255, 255, 127);
    }

    // 绘制路径（蓝色）——使用浮点坐标 + 抗锯齿画线，避免整数截断导致的水平/垂直线
    for (int i = 1; i < TSP_CITY_NUM; i++) {
        int from = best_gene[i - 1];
        int to = best_gene[i];
        gfx_draw_line_anti_aliasing(global_state->gfx,
            city_fx[from], city_fy[from], city_fx[to], city_fy[to],
            1.0f, 0, 0, 255, 1);
    }
    int last = best_gene[TSP_CITY_NUM - 1];
    int first = best_gene[0];
    gfx_draw_line_anti_aliasing(global_state->gfx,
        city_fx[last], city_fy[last], city_fx[first], city_fy[first],
        1.0f, 0, 0, 255, 1);

    // 绘制距离变化曲线
    int valid_count = (s_tsp_history_count < TSP_HISTORY_SIZE) ? s_tsp_history_count : TSP_HISTORY_SIZE;
    if (valid_count > 0) {
        // 曲线背景
        gfx_draw_rectangle(global_state->gfx, 0, TSP_CURVE_TOP, TSP_SCREEN_W, TSP_CURVE_H, 20, 20, 20, 3);

        // 计算纵轴范围
        float hmin = 1e30f, hmax = 0.0f;
        int start_idx = (s_tsp_history_count <= TSP_HISTORY_SIZE) ? 0 : (s_tsp_history_count % TSP_HISTORY_SIZE);
        for (int i = 0; i < valid_count; i++) {
            int idx = (start_idx + i) % TSP_HISTORY_SIZE;
            float v = s_tsp_history[idx];
            if (v < hmin) hmin = v;
            if (v > hmax) hmax = v;
        }
        if (hmax <= hmin) {
            hmax = hmin + 1.0f;
        }

        // 画折线
        if (valid_count > 1) {
            for (int i = 1; i < valid_count; i++) {
                int idx0 = (start_idx + i - 1) % TSP_HISTORY_SIZE;
                int idx1 = (start_idx + i) % TSP_HISTORY_SIZE;
                float px0 = (float)(i - 1) * (float)TSP_SCREEN_W / (float)(valid_count - 1);
                float px1 = (float)i * (float)TSP_SCREEN_W / (float)(valid_count - 1);
                float py0 = (float)TSP_CURVE_TOP + (hmax - s_tsp_history[idx0]) * (float)TSP_CURVE_H / (hmax - hmin);
                float py1 = (float)TSP_CURVE_TOP + (hmax - s_tsp_history[idx1]) * (float)TSP_CURVE_H / (hmax - hmin);
                // 限制在曲线区域内
                if (py0 < TSP_CURVE_TOP) py0 = TSP_CURVE_TOP;
                if (py0 >= TSP_CURVE_BOTTOM) py0 = TSP_CURVE_BOTTOM - 1;
                if (py1 < TSP_CURVE_TOP) py1 = TSP_CURVE_TOP;
                if (py1 >= TSP_CURVE_BOTTOM) py1 = TSP_CURVE_BOTTOM - 1;
                gfx_draw_line_anti_aliasing(global_state->gfx, px0, py0, px1, py1, 1.0f, 0, 255, 0, 1);
            }
        }

        // 当前数值
        wchar_t dist_text[32];
        swprintf(dist_text, 32, L"%.1f", s_tsp_best_distance);
        gfx_draw_textline_mini(global_state->gfx, dist_text, 4, TSP_CURVE_TOP + 2, 255, 255, 255, 1);
    }

    gfx_refresh(global_state->gfx);
    s_tsp_generation += 1;
}

#ifdef __cplusplus
}
#endif

#endif
