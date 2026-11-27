// ui_linglong.c —— 玲珑天象仪（STATE_LINGLONG 的状态页）
//
// 本模块从 ui_app.c 剥离而来（2026-11-28）：状态页面、星图渲染驱动、时间机器（时光机）、
// 视角漫游、设置界面、触屏控件编排与实体键盘兜底，全部集中在这里。
//
// 分层：
//   celestial.c / celestial_ocean.c   星图与海洋天空渲染（render_sky / linglong_ocean_*）
//   ephemeris.c                        日月位置、月相、日出日落（纯计算）
//   ui_linglong_ctrl.c                 叠加在主界面上的触屏控件（顶部按钮栏 / 时间按钮 /
//                                      视角九宫格 / 设置按钮 / 5x4 设置界面）——只绘制与命中，
//                                      动作语义由本模块解释
//   ui_linglong.c（本文件）            状态页主体：配置读写、渲染帧、事件处理（触屏 + 键盘）、
//                                      时光机与漫游状态机、设置界面的按键合成
//   ui_app.c                           状态机：STATE_LINGLONG 分支只调三个入口 + 全局资源初始化
//
// 渲染走专用画布 global_state->llgfx（ESP32 上为独立 RGB888 缓冲，最后转 RGB565 拷进
// 主帧缓冲；其它平台直接复用主帧缓冲）。
// ===============================================================================

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "ui_linglong.h"

#include "graphics.h"
#include "hal_display.h"   // display_hal_get_color_depth：低色深时的抖动
#include "hal_key.h"
#include "platform.h"
#include "cloud.h"   // linglong_cloud_coverage_level_num/name：云量档位名（设置界面显示）
#include "ui_color.h"
#include "ui_linglong_ctrl.h"

#include "celestial.h"        // render_sky / linglong_free / fisheye_*
#include "celestial_ocean.h"  // linglong_ocean_*
#include "ephemeris.h"        // 日月位置 / 月相 / 日出日落

#ifdef IMU_ENABLED
    #include "hal_imu.h"      // imu_read_angle / imu_calib：姿态跟随与手动控制切换
#endif

// ===============================================================================
// 玲珑天象仪
// ===============================================================================

static uint64_t linglong_first_call_timestamp = 0;
static uint32_t linglong_last_day = 0;
static uint32_t linglong_sunrise_time[2] = {0, 0}; // hour, minute
static uint32_t linglong_sunset_time[2] = {0, 0}; // hour, minute

static int32_t linglong_timemachine_running_state = 2; // 0-停止；1-时光机运行；2-实时
static int32_t linglong_timemachine_speed = 0; // 时光机速度，正数为未来，负数为过去，单位秒
static uint64_t linglong_timemachine_start_timestamp = 0;
static int32_t linglong_paused_resume_state = 2; // 暂停前的运行状态（1-时光机；2-实时），供“运行/暂停”按钮恢复

static int32_t linglong_state = 0; // 玲珑仪UI状态

#define LL_STATE_SKY (0)
#define LL_STATE_SETTING (1)
#define LL_STATE_SETTING_CALLBACK (2)


void ui_linglong_init(Key_Event *key_event, Global_State *global_state) {
    linglong_state = LL_STATE_SKY;
    linglong_paused_resume_state = 2;
    ui_linglong_ctrl_init();
}

// 大气散射档位循环：屏蔽「简化模型(1)」「一次散射(2)」——两者分别被二次散射(3)与
// 海洋天空(5)完全覆盖（观感更差且没有独占能力），保留渲染实现但不再出现在选项里。
// 顺序：关(0) → 二次散射(3) → 体积云(4) → 海洋天空(5) → 关。
// 当前值若落在被屏蔽档位（手改配置/旧存档），下一次点击落到二次散射。
static int32_t ui_linglong_next_sky_model(int32_t cur) {
    static const int32_t kOrder[] = { 0, 3, 4, 5 };
    const int32_t n = (int32_t)(sizeof(kOrder) / sizeof(kOrder[0]));
    for (int32_t i = 0; i < n; i++) {
        if (kOrder[i] == cur) return kOrder[(i + 1) % n];
    }
    return 3;
}

// 云参数现值名称与控制（设置界面第 1 行前三格；原主控制面板左列）
static const wchar_t *ui_linglong_cloud_layer_name(Linglong_Config *cfg) {
    int32_t n = 0;
    const Linglong_Cloud_Option *opt = linglong_cloud_layer_options(&n);
    if (n <= 0) return L"关";
    for (int32_t i = 0; i < n; i++) {
        if (opt[i].value == cfg->cloud_layer_mask) return opt[i].name;
    }
    return opt[0].name;
}

static const wchar_t *ui_linglong_cloud_precision_name(Linglong_Config *cfg) {
    int32_t n = 0;
    const Linglong_Cloud_Option *opt = linglong_cloud_precision_options(&n);
    int32_t idx = cfg->cloud_precision_level;
    if (idx < 0 || idx >= n) idx = 0;
    return (n > 0) ? opt[idx].name : L"中";
}

static void ui_linglong_cloud_next_coverage(Global_State *global_state) {
    int32_t n = linglong_cloud_coverage_level_num();
    if (n < 1) n = 1;
    global_state->linglong_cfg->cloud_coverage_level =
        (global_state->linglong_cfg->cloud_coverage_level + 1) % n;
}

static void ui_linglong_cloud_next_layer(Global_State *global_state) {
    int32_t n = 0;
    const Linglong_Cloud_Option *opt = linglong_cloud_layer_options(&n);
    int32_t idx = 0;
    for (int32_t i = 0; i < n; i++) {
        if (opt[i].value == global_state->linglong_cfg->cloud_layer_mask) { idx = i; break; }
    }
    idx = (n > 0) ? (idx + 1) % n : 0;
    if (n > 0) global_state->linglong_cfg->cloud_layer_mask = opt[idx].value;
}

static void ui_linglong_cloud_next_precision(Global_State *global_state) {
    int32_t n = 0;
    (void)linglong_cloud_precision_options(&n);
    if (n < 1) n = 1;
    global_state->linglong_cfg->cloud_precision_level =
        (global_state->linglong_cfg->cloud_precision_level + 1) % n;
}

// 设置界面剩余功能格映射：返回现有处理链使用的键码（调用前会置 Ctrl 激活）。
// 第 1 行前三格（云量/云类/云精度）由调用方直接循环配置，不合成按键。
static int32_t ui_linglong_setting_cell_key(int32_t cell) {
    switch (cell) {
        case 3:  return NANO_KEY_4;    // 黄道
        case 4:  return NANO_KEY_ctrl; // 返回
        case 7:  return NANO_KEY_5;    // 天体名称
        case 8:  return NANO_KEY_9;    // 平滑滤波
        case 9:  return NANO_KEY_0;    // 跟踪太阳
        default: return NANO_KEY_IDLE;
    }
}

// 填设置界面 5x4 单元格：标签 / 现值 / 现值颜色；返回按钮特殊样式，其余空位保留备用。
static void ui_linglong_setting_fill_cells(Global_State *global_state, UiLinglongSettingCell *cells) {
    Linglong_Config *cfg = global_state->linglong_cfg;

    for (int32_t i = 0; i < UI_LL_SETTING_COUNT; i++) {
        cells[i].label = NULL;
        cells[i].value = NULL;
        cells[i].value_r = 224;
        cells[i].value_g = 224;
        cells[i].value_b = 224;
        cells[i].is_special = 0;
    }

    // 第 1 行：云量、云类、云精度、黄道、返回
    // 云参数只在 sky_model==4（体积云）时影响画面；其它档位现值画灰但可预调
    // （与旧控制面板云参数按钮“不吃参数时画暗”的语义一致）。
    const int32_t cloud_active = (cfg->sky_model == 4);
    const uint8_t cloud_r = cloud_active ? 0 : 168;
    const uint8_t cloud_g = cloud_active ? 255 : 176;
    const uint8_t cloud_b = cloud_active ? 255 : 190;

    cells[0].label = L"云量";
    cells[0].value = linglong_cloud_coverage_level_name(cfg->cloud_coverage_level);
    cells[0].value_r = cloud_r; cells[0].value_g = cloud_g; cells[0].value_b = cloud_b;

    cells[1].label = L"云类";
    cells[1].value = ui_linglong_cloud_layer_name(cfg);
    cells[1].value_r = cloud_r; cells[1].value_g = cloud_g; cells[1].value_b = cloud_b;

    cells[2].label = L"云精度";
    cells[2].value = ui_linglong_cloud_precision_name(cfg);
    cells[2].value_r = cloud_r; cells[2].value_g = cloud_g; cells[2].value_b = cloud_b;

    cells[3].label = L"黄道";
    if (cfg->enable_ecliptic_circle == 0) {
        cells[3].value = L"关";
        cells[3].value_r = 222; cells[3].value_g = 0; cells[3].value_b = 0;
    }
    else {
        cells[3].value = L"开";
        cells[3].value_r = 0; cells[3].value_g = 255; cells[3].value_b = 0;
    }

    cells[4].label = L"返回";
    cells[4].is_special = 1;

    // 第 2 行：第 1 格镜头光晕、第 2 格星芒（大气类视觉开关；两者默认“光晕开、星芒关”）
    cells[5].label = L"镜头光晕";
    if (cfg->enable_lens_flare == 0) {
        cells[5].value = L"关";
        cells[5].value_r = 222; cells[5].value_g = 0; cells[5].value_b = 0;
    }
    else {
        cells[5].value = L"开";
        cells[5].value_r = 0; cells[5].value_g = 255; cells[5].value_b = 0;
    }

    cells[6].label = L"星芒";
    if (cfg->enable_star_burst == 0) {
        cells[6].value = L"关";
        cells[6].value_r = 222; cells[6].value_g = 0; cells[6].value_b = 0;
    }
    else {
        cells[6].value = L"开";
        cells[6].value_r = 0; cells[6].value_g = 255; cells[6].value_b = 0;
    }

    cells[7].label = L"天体名称";
    switch (cfg->enable_star_name) {
        case 0:
            cells[7].value = L"关";
            cells[7].value_r = 222; cells[7].value_g = 0; cells[7].value_b = 0;
            break;
        case 1:
            cells[7].value = L"仅恒星";
            cells[7].value_r = 0; cells[7].value_g = 255; cells[7].value_b = 255;
            break;
        case 2:
            cells[7].value = L"仅行星";
            cells[7].value_r = 0; cells[7].value_g = 255; cells[7].value_b = 255;
            break;
        case 3:
            cells[7].value = L"全部显示";
            cells[7].value_r = 0; cells[7].value_g = 255; cells[7].value_b = 255;
            break;
        default: break;
    }

    cells[8].label = L"平滑滤波";
    if (cfg->enable_opt_bilinear == 0) {
        cells[8].value = L"关";
        cells[8].value_r = 222; cells[8].value_g = 0; cells[8].value_b = 0;
    }
    else {
        cells[8].value = L"开";
        cells[8].value_r = 0; cells[8].value_g = 255; cells[8].value_b = 0;
    }

    cells[9].label = L"跟踪太阳";
    if (cfg->enable_tracking_sun == 0) {
        cells[9].value = L"关";
        cells[9].value_r = 222; cells[9].value_g = 0; cells[9].value_b = 0;
    }
    else {
        cells[9].value = L"开";
        cells[9].value_r = 0; cells[9].value_g = 255; cells[9].value_b = 0;
    }

    // 第 3 行第 2~4 格：海洋参数（观察高度 / 涌浪波高 / 涌浪波长）。
    // 观察高度对 sky_model==4（体积云相机海拔）/5 或 landscape_index==3 都有效；
    // 涌浪波高/波长只对海洋有效。不影响的档位现值画灰但可预调。
    {
        const int32_t height_active = (cfg->sky_model == 4 || cfg->sky_model == 5 ||
                                       cfg->landscape_index == 3);
        const int32_t ocean_active  = (cfg->sky_model == 5 || cfg->landscape_index == 3);
        int32_t n = 0;
        const Linglong_Ocean_Option *opt;
        struct { int32_t cell; int32_t level; const Linglong_Ocean_Option *(*get)(int32_t *); } m[3] = {
            { 11, cfg->ocean_height_level,    linglong_ocean_height_options },
            { 12, cfg->ocean_swell_hs_level,  linglong_ocean_swell_hs_options },
            { 13, cfg->ocean_swell_len_level, linglong_ocean_swell_len_options },
        };
        static const wchar_t *labels[3] = { L"观察高度", L"涌浪波高", L"涌浪波长" };
        static wchar_t s_alt_text[24];
        for (int32_t i = 0; i < 3; i++) {
            opt = m[i].get(&n);
            if (opt == NULL || n <= 0) continue;
            int32_t lv = m[i].level;
            if (lv < 0 || lv >= n) lv = 0;
            cells[m[i].cell].label = labels[i];
            if (m[i].cell == 11) {
                // 观察高度显示真实海拔（漫游时是连续量），档位只作为快捷跳转
                const double alt = (double)cfg->obs_y;
                if (alt < 1000.0) swprintf(s_alt_text, 24, L"%.0fm", alt);
                else swprintf(s_alt_text, 24, L"%.1fkm", alt / 1000.0);
                cells[m[i].cell].value = s_alt_text;
            }
            else {
                cells[m[i].cell].value = opt[lv].name;
            }
            if ((m[i].cell == 11) ? height_active : ocean_active) {
                cells[m[i].cell].value_r = 0; cells[m[i].cell].value_g = 200; cells[m[i].cell].value_b = 255;
            }
            else {
                cells[m[i].cell].value_r = 168; cells[m[i].cell].value_g = 176; cells[m[i].cell].value_b = 190;
            }
        }
    }

    // 第 3 行首格：岛屿地形开关（landscape_index==3 时生效）
    {
        static wchar_t s_terr_prog[24];
        const int32_t terr_active = (cfg->landscape_index == 3);
        const uint8_t tr = terr_active ? 0 : 168;
        const uint8_t tg = terr_active ? 200 : 176;
        const uint8_t tb = terr_active ? 255 : 190;

        cells[10].label = L"岛屿地形";
        cells[10].value = (cfg->terrain_enable) ? L"开" : L"关";
        if (cfg->terrain_enable) { cells[10].value_r = 0; cells[10].value_g = 255; cells[10].value_b = 0; }
        else { cells[10].value_r = 222; cells[10].value_g = 0; cells[10].value_b = 0; }
        if (!terr_active) { cells[10].value_r = tr; cells[10].value_g = tg; cells[10].value_b = tb; }

        cells[14].label = L"地形精度";
        {
            static const wchar_t *names[3] = { L"省电", L"标准", L"精细" };
            int32_t lv = cfg->terrain_quality;
            if (lv < 0 || lv > 2) lv = 1;
            cells[14].value = names[lv];
        }
        cells[14].value_r = tr; cells[14].value_g = tg; cells[14].value_b = tb;

        cells[15].label = L"换一座岛";
        {
            // 生成中显示进度，就绪后显示当前种子的低 4 位（便于复现同一组岛）
            int32_t prog = linglong_ocean_terrain_progress();
            if (prog < 100) {
                swprintf(s_terr_prog, 24, L"%d%%", prog);
            }
            else {
                swprintf(s_terr_prog, 24, L"#%04u", (unsigned)(cfg->terrain_seed % 10000u));
            }
            cells[15].value = s_terr_prog;
        }
        cells[15].value_r = tr; cells[15].value_g = tg; cells[15].value_b = tb;

        cells[16].label = L"地形网格";
        cells[16].value = (cfg->terrain_res >= 1024) ? L"1024" : L"512";
        cells[16].value_r = tr; cells[16].value_g = tg; cells[16].value_b = tb;

        cells[17].label = L"漫游速度";
        {
            static const wchar_t *spd[3] = { L"10m/s", L"100m/s", L"1km/s" };
            int32_t lv = cfg->roam_speed_level;
            if (lv < 0 || lv > 2) lv = 1;
            cells[17].value = spd[lv];
        }
        cells[17].value_r = 0; cells[17].value_g = 200; cells[17].value_b = 255;

        cells[18].label = L"观察位置";
        {
            static wchar_t s_pos[32];
            swprintf(s_pos, 32, L"%+.1f/%+.1fkm", (double)cfg->obs_x / 1000.0, (double)cfg->obs_z / 1000.0);
            cells[18].value = s_pos;
        }
        cells[18].value_r = 200; cells[18].value_g = 220; cells[18].value_b = 240;

        cells[19].label = L"漫游";
        cells[19].value = (cfg->roam_enable) ? L"开" : L"关";
        if (cfg->roam_enable) { cells[19].value_r = 0; cells[19].value_g = 255; cells[19].value_b = 0; }
        else { cells[19].value_r = 222; cells[19].value_g = 0; cells[19].value_b = 0; }
    }
}

void ui_linglong_setting_draw(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    // 不清屏：设置界面直接叠加在每帧新绘制的天象仪主界面上（与控制面板同一做法），
    // 5x4 按钮使用与控制按钮一致的圆角半透明样式。
    UiLinglongSettingCell cells[UI_LL_SETTING_COUNT];
    ui_linglong_setting_fill_cells(global_state, cells);
    ui_linglong_ctrl_draw_settings(global_state->llgfx, cells);
}




// ===============================================================================
// 漫游（观察者水平/垂直移动）
// ===============================================================================
// 方向来源有两路，最终合成一个位掩码后按 dt 积分（与帧率无关）：
//   · 触屏移动 pad（控制面板底部，长按连续移动）——每帧直接问当前按住的方向；
//   · 物理键/键盘（上/下/左/右、W/A/S/D、Q/E）——按键事件是边沿触发（松手沿 -1、
//     长按重复 -2、按住 0），这里把“最近一次事件”当作按住状态，180 ms 无事件即释放，
//     因而对任意重复频率都稳定，不会因长按重复而超速。
static int32_t  s_roam_key_mask = 0;
static uint64_t s_roam_key_ts[6] = { 0, 0, 0, 0, 0, 0 };
static uint64_t s_roam_last_ms = 0;

static void ui_linglong_roam_key_hold(int32_t bit) {
    s_roam_key_mask |= bit;
    int32_t idx = 0;
    while ((1 << idx) != bit && idx < 6) idx++;
    if (idx < 6) s_roam_key_ts[idx] = get_timestamp_in_ms();
}

static int32_t ui_linglong_roam_key_mask(void) {
    const uint64_t now = get_timestamp_in_ms();
    int32_t mask = 0;
    for (int32_t i = 0; i < 6; i++) {
        if (s_roam_key_mask & (1 << i)) {
            if (now - s_roam_key_ts[i] > 180) s_roam_key_mask &= ~(1 << i);
            else mask |= (1 << i);
        }
    }
    return mask;
}

// 每帧推进观察者位置（在 render_sky 之前调用）
static void ui_linglong_roam_step(Key_Event *key_event, Global_State *global_state) {
    Linglong_Config *c = global_state->linglong_cfg;
    const uint64_t now = get_timestamp_in_ms();
    double dt = (s_roam_last_ms == 0) ? 0.0 : (double)(now - s_roam_last_ms) / 1000.0;
    s_roam_last_ms = now;
    if (dt < 0.0) dt = 0.0;
    if (dt > 0.1) dt = 0.1;

    // 触屏 pad（只有漫游开启且面板可见才有值）
    int32_t mask = ui_linglong_ctrl_roam_held(
        key_event->is_touching, key_event->touch_x, key_event->touch_y);
    // 物理键（同样只在漫游开启时生效；否则方向键仍是时光机快退/快进）
    if (c->roam_enable) mask |= ui_linglong_roam_key_mask();
    if (mask == 0 || dt <= 0.0) return;

    static const float S_SPEED[3] = { 10.0f, 100.0f, 1000.0f };
    int32_t lv = c->roam_speed_level;
    if (lv < 0 || lv > 2) lv = 1;
    const double speed = (double)S_SPEED[lv];

    // 前进方向 = 视线方向的水平投影（罗盘方位：北 = +z，东 = +x）
    const double azi = (double)c->view_azi * M_PI / 180.0;
    const double fx = sin(azi), fz = cos(azi);
    const double rx = cos(azi), rz = -sin(azi);
    double mx = 0.0, mz = 0.0;
    if (mask & UI_LL_ROAM_FWD)   { mx += fx; mz += fz; }
    if (mask & UI_LL_ROAM_BACK)  { mx -= fx; mz -= fz; }
    if (mask & UI_LL_ROAM_RIGHT) { mx += rx; mz += rz; }
    if (mask & UI_LL_ROAM_LEFT)  { mx -= rx; mz -= rz; }
    const double ml = sqrt(mx * mx + mz * mz);
    if (ml > 1e-6) { mx /= ml; mz /= ml; }

    c->obs_x += (float)(mx * speed * dt);
    c->obs_z += (float)(mz * speed * dt);
    if (mask & UI_LL_ROAM_UP)   c->obs_y += (float)(speed * dt);
    if (mask & UI_LL_ROAM_DOWN) c->obs_y -= (float)(speed * dt);
    if (c->obs_y < 1.5f) c->obs_y = 1.5f;
    c->obs_y = (float)MIN(c->obs_y, 20000.0f);
    // 不钻地：地形就绪时抬到地面以上（与 celestial_ocean 内部的钳制一致，
    // 这里先夹一次是为了让“观察高度”显示与渲染同帧一致）
    if (c->terrain_enable && linglong_ocean_terrain_ready()) {
        double g = linglong_ocean_terrain_height((double)c->obs_x, (double)c->obs_z);
        if ((double)c->obs_y < g + 1.5) c->obs_y = (float)(g + 1.5);
    }
}

// 玲珑仪主天空界面渲染。
// 每帧总是完整重绘（没有“暂停就只画一次”的跳过逻辑）：实时海洋的波场、体积云的
// 飘移都按墙钟推进（get_timestamp_in_ms），与模拟时间是否流动无关——时钟暂停时画面
// 仍在持续变化，跳过重绘会把它们冻住；太阳移动时还需要按帧重建大气 LUT。
// 故本函数无返回值，调用方每帧都会得到一帧全新画面，可安全叠加半透明控件
// （不会出现 alpha 逐帧累积加深）。
void ui_linglong_draw_full(Key_Event *key_event, Global_State *global_state) {

    Linglong_Config *llcfg = global_state->linglong_cfg;

    // FPS统计
    static uint64_t fps_last_timestamp = 0;
    static uint32_t fps_frame_count = 0;
    static float fps_display_value = 0.0f;

    fps_frame_count++;
    if (fps_last_timestamp == 0) {
        fps_last_timestamp = global_state->timestamp;
    }
    else if (global_state->timestamp - fps_last_timestamp >= 1000) {
        fps_display_value = fps_frame_count * 1000.0f / (float)(global_state->timestamp - fps_last_timestamp);
        fps_frame_count = 0;
        fps_last_timestamp = global_state->timestamp;
    }

    time_t ts = (time_t)(global_state->timestamp / 1000);

    if (linglong_timemachine_running_state == 1) {
        linglong_timemachine_start_timestamp += (linglong_timemachine_speed * 1000);
        ts = (time_t)(linglong_timemachine_start_timestamp / 1000);
        struct tm *timeinfo = localtime(&ts); // 转换为本地时间

        llcfg->second = timeinfo->tm_sec;
        llcfg->minute = timeinfo->tm_min;
        llcfg->hour = timeinfo->tm_hour;
        llcfg->day = timeinfo->tm_mday;
        llcfg->month = timeinfo->tm_mon + 1;
        llcfg->year = timeinfo->tm_year + 1900;
    }
    else if (linglong_timemachine_running_state == 2) {
        ts = (time_t)(global_state->timestamp / 1000);
        struct tm *timeinfo = localtime(&ts);
        llcfg->second = timeinfo->tm_sec;
        llcfg->minute = timeinfo->tm_min;
        llcfg->hour = timeinfo->tm_hour;
        llcfg->day = timeinfo->tm_mday;
        llcfg->month = timeinfo->tm_mon + 1;
        llcfg->year = timeinfo->tm_year + 1900;
    }


    if (llcfg->enable_imu) {
        llcfg->view_alt  = global_state->pitch;
        llcfg->view_azi  = global_state->yaw + 180.0f;
        llcfg->view_roll = global_state->roll;
    }

    ui_linglong_roam_step(key_event, global_state);

    gfx_soft_clear(global_state->llgfx);

    render_sky(global_state->llgfx,
        MIN(global_state->llgfx->width, global_state->llgfx->height) / 2, global_state->llgfx->width / 2, global_state->llgfx->height / 2,
        llcfg->view_alt, llcfg->view_azi, llcfg->view_roll, llcfg->view_f,
        // 2026, 3, 24, 18, 10, 0, 8.0, 119.0, 31.0,
        llcfg->year, llcfg->month, llcfg->day, llcfg->hour, llcfg->minute, llcfg->second, llcfg->timezone, llcfg->longitude, llcfg->latitude,
        llcfg->downsampling_factor,
        llcfg->enable_opt_sym,
        llcfg->enable_opt_lut,
        llcfg->enable_opt_bilinear,
        llcfg->projection,
        llcfg->sky_model,
        llcfg->landscape_index,
        llcfg->enable_equatorial_coord,
        llcfg->enable_horizontal_coord,
        llcfg->enable_star_burst,
        llcfg->enable_lens_flare,
        llcfg->enable_star_name,
        llcfg->enable_planet,
        llcfg->enable_ecliptic_circle,
        llcfg->enable_att_indicator,
        llcfg->enable_tracking_sun,
        llcfg->cloud_coverage_level,
        llcfg->cloud_layer_mask,
        llcfg->cloud_precision_level,
        llcfg->ocean_swell_hs_level,
        llcfg->ocean_swell_len_level,
        llcfg->obs_x, llcfg->obs_y, llcfg->obs_z,
        llcfg->terrain_enable, llcfg->terrain_quality, llcfg->terrain_res, llcfg->terrain_seed
    );

    // 观察者位置回读：海洋模块内部做了水平周期回绕与“不钻地”钳制，
    // 把归一化后的值写回配置（否则漫游跨过世界边界时界面数值会越滚越大）。
    {
        double ox = 0.0, oy = 0.0, oz = 0.0;
        linglong_ocean_get_observer(&ox, &oy, &oz);
        llcfg->obs_x = (float)ox;
        llcfg->obs_y = (float)oy;
        llcfg->obs_z = (float)oz;
    }

#if defined(NANO_POD_POCKETTERM35)
    // PocketTerm35 的 HDMI 屏经 32bpp framebuffer 输出 8bit/通道时无需 565 抖动
    //（gfx_dithering 会预量化到 565 色阶，反而成为色深瓶颈）；fb 为 16bpp 时保留
    if (display_hal_get_color_depth() <= 16) gfx_dithering(global_state->llgfx);
#else
    gfx_dithering(global_state->llgfx);
#endif
    // gfx_gamma(global_state->llgfx, 1.3f);

    // 显示FPS
    wchar_t fps_str[16];
    swprintf(fps_str, 16, L"FPS=%.1f", fps_display_value);
    gfx_font_draw_text(global_state->llgfx, GFX_FONT_ALPHA_12, fps_str, 1, 0, 0, 255, 0, 1);
}

void ui_linglong_draw_lite(
    Key_Event *key_event, Global_State *global_state,
    int32_t x, int32_t y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double longitude, double latitude, double timezone
) {
    uint8_t BG_R = 255, BG_G = 255, BG_B = 255;
    uint8_t COORD_R = 222, COORD_G = 222, COORD_B = 222;
    uint8_t NSWE_R = 255, NSWE_G = 0, NSWE_B = 0;
    uint8_t DATETIME_R = 0, DATETIME_G = 0, DATETIME_B = 255;
    uint8_t TEXT_R = 0, TEXT_G = 0, TEXT_B = 0;
    uint8_t SUN_R = 255, SUN_G = 0, SUN_B = 0;
    uint8_t MOON_R = 255, MOON_G = 0, MOON_B = 255;

    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        // 同初始化
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        BG_R = 0, BG_G = 0, BG_B = 0;
        COORD_R = 66, COORD_G = 66, COORD_B = 66;
        NSWE_R = 255, NSWE_G = 0, NSWE_B = 0;
        DATETIME_R = 0, DATETIME_G = 255, DATETIME_B = 255;
        TEXT_R = 255, TEXT_G = 255, TEXT_B = 255;
        SUN_R = 255, SUN_G = 255, SUN_B = 0;
        MOON_R = 222, MOON_G = 222, MOON_B = 0;
    }

    gfx_draw_rectangle(global_state->gfx, x, y, 128, 64, BG_R, BG_G, BG_B, 1);

    gfx_draw_circle(global_state->gfx, x+64, y+32, 30,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_circle(global_state->gfx, x+64, y+32, 20,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_circle(global_state->gfx, x+64, y+32, 10,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_line(global_state->gfx,   x+32, y+32, x+96, y+32, COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_line(global_state->gfx,   x+64, y+0, x+64, y+64,  COORD_R, COORD_G, COORD_B, 1);

    gfx_draw_rectangle(global_state->gfx, x+62-1, y+0,    5+2, 5+1, BG_R, BG_G, BG_B, 1); // N背景
    gfx_draw_rectangle(global_state->gfx, x+63-1, y+59-1, 3+2, 5+1, BG_R, BG_G, BG_B, 1); // S背景
    gfx_draw_rectangle(global_state->gfx, x+32-1, y+30-1, 5+2, 5+2, BG_R, BG_G, BG_B, 1); // W背景
    gfx_draw_rectangle(global_state->gfx, x+93-1, y+30-1, 3+2, 5+2, BG_R, BG_G, BG_B, 1); // E背景

    // 方位文字和周围的边框
    gfx_draw_textline_mini(global_state->gfx, L"N", x+62, y+0,  NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+61, y+2,  BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+67, y+2,  BG_R, BG_G, BG_B, 1);  gfx_draw_point(global_state->gfx, x+64, y+5, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"S", x+63, y+59, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+62, y+62, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+66, y+62, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+64, y+58, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"W", x+32, y+30, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+34, y+29, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+37, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+34, y+35, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"E", x+93, y+30, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+92, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+96, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+94, y+29, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+94, y+35, BG_R, BG_G, BG_B, 1);

    gfx_draw_line(global_state->gfx, x+0, y+43, x+30, y+43, COORD_R, COORD_G, COORD_B, 1);

    wchar_t timestr[30];
    swprintf(timestr, 30, L"%04d-%02d-%02d\n%02d:%02d:%02d", year, month, day, hour, minute, second);
    gfx_draw_textline_mini(global_state->gfx, timestr, x+0, y+0, DATETIME_R, DATETIME_G, DATETIME_B, 1);

    double altitude_moon = 0.0;
    double azimuth_moon = 0.0;

    where_is_the_moon(year, month, day, hour, minute, second, timezone, longitude, latitude, &azimuth_moon, &altitude_moon);
    double i_deg = moon_phase(year, month, day, hour, minute, second, timezone);
    double moon_k = (1.0 + cos(i_deg / 180.0 * M_PI)) / 2.0;

    wchar_t coordstr_moon[30];
    swprintf(coordstr_moon, 30, L"MOON\nP:%d%%\nA:%.1f\nE:%.1f", (int32_t)(moon_k * 100.0), azimuth_moon, altitude_moon);
    gfx_draw_textline_mini(global_state->gfx, coordstr_moon, x+0, y+18, TEXT_R, TEXT_G, TEXT_B, 1);

    double x_moon = 64 + (90.0 - altitude_moon) * 32.0 / 90.0 * sin(azimuth_moon / 180.0 * M_PI);
    double y_moon = 32 - (90.0 - altitude_moon) * 32.0 / 90.0 * cos(azimuth_moon / 180.0 * M_PI);

    if (x_moon >= 32 && x_moon <= 96 && y_moon >= 0 && y_moon <= 64) {
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
    }

    double altitude_sun = 0.0;
    double azimuth_sun = 0.0;

    where_is_the_sun(year, month, day, hour, minute, second, +8.0, longitude, latitude, &azimuth_sun, &altitude_sun);

    wchar_t coordstr_sun[30];
    swprintf(coordstr_sun, 30, L"SUN\nA:%.1f\nE:%.1f", azimuth_sun, altitude_sun);
    gfx_draw_textline_mini(global_state->gfx, coordstr_sun, x+0, y+46, TEXT_R, TEXT_G, TEXT_B, 1);

    double x_sun = 64 + (90.0 - altitude_sun) * 32.0 / 90.0 * sin(azimuth_sun / 180.0 * M_PI);
    double y_sun = 32 - (90.0 - altitude_sun) * 32.0 / 90.0 * cos(azimuth_sun / 180.0 * M_PI);

    if (x_sun >= 32 && x_sun <= 96 && y_sun >= 0 && y_sun <= 64) {
        gfx_draw_circle(global_state->gfx, x+(int)x_sun, y+(int)y_sun, 2, SUN_R, SUN_G, SUN_B, 1);
    }


    // 二分搜索日出日落时间
    if (linglong_first_call_timestamp == 0 || linglong_last_day != day) { // 只在首次调用和当天日期变化时计算
        linglong_first_call_timestamp = global_state->timestamp;
        linglong_last_day = day;

        int32_t sunrise_min = find_sunrise(year, month, day, timezone, longitude, latitude);
        if (sunrise_min != -1) {
            linglong_sunrise_time[0] = sunrise_min / 60;
            linglong_sunrise_time[1] = sunrise_min % 60;
        }
        int32_t sunset_min = find_sunset(year, month, day, timezone, longitude, latitude);
        if (sunset_min != -1) {
            linglong_sunset_time[0] = sunset_min / 60;
            linglong_sunset_time[1] = sunset_min % 60;
        }
    }
    wchar_t risefall_time[60];
    swprintf(risefall_time, 60, L"R:%02d:%02d\nS:%02d:%02d", linglong_sunrise_time[0], linglong_sunrise_time[1], linglong_sunset_time[0], linglong_sunset_time[1]);
    gfx_draw_textline_mini(global_state->gfx, risefall_time, x+98, y+0, TEXT_R, TEXT_G, TEXT_B, 1);

    gfx_draw_textline_mini(global_state->gfx, L"    BD4SUR\n 2011-2026", x+86, y+53, TEXT_R, TEXT_G, TEXT_B, 1);
}

void ui_linglong_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 天空每帧都完整重绘；设置界面与主控制面板都是叠加在其上的半透明控件，
    // 因此不会出现 alpha 逐帧累积加深。
    ui_linglong_draw_full(key_event, global_state);

    if (linglong_state == LL_STATE_SETTING || linglong_state == LL_STATE_SETTING_CALLBACK) {
        ui_linglong_setting_draw(key_event, global_state);
    }
    gfx_font_draw_text(global_state->llgfx, GFX_FONT_ALPHA_12, (wchar_t *)L"玲珑天象仪 V" NANO_VERSION,
        1, global_state->llgfx->height - gfx_font_line_height(GFX_FONT_ALPHA_12), 255, 255, 255, 200);

    wchar_t timestr[30];
    swprintf(timestr, 30, L"%ls %04d-%02d-%02d %02d:%02d:%02d",
        (linglong_timemachine_running_state == 0) ? L"  " :
            (((linglong_timemachine_running_state == 1) && (linglong_timemachine_speed > 0)) ? L">>" :
            (((linglong_timemachine_running_state == 1) && (linglong_timemachine_speed < 0)) ? L"<<" : L" >")),
        global_state->linglong_cfg->year, global_state->linglong_cfg->month, global_state->linglong_cfg->day, global_state->linglong_cfg->hour, global_state->linglong_cfg->minute, global_state->linglong_cfg->second);
    int32_t timestr_x = global_state->llgfx->width - gfx_font_measure_text(GFX_FONT_ALPHA_12, timestr) - 1;
    if (timestr_x < 0) timestr_x = 0;
    gfx_font_draw_text(global_state->llgfx, GFX_FONT_ALPHA_12, timestr,
        timestr_x, global_state->llgfx->height - gfx_font_line_height(GFX_FONT_ALPHA_12), 255, 255, 255, 1);

    // 主天空界面：叠加显式触屏控件（顶部按钮栏 + 可选的时间/视角面板）。
    // 设置界面（LL_STATE_SETTING*）由 5x4 显式触屏按钮承载（见 ui_linglong_ctrl_settings_*），
    // 这里不绘制主界面控件。
    if (linglong_state == LL_STATE_SKY) {
        UiLinglongCtrlState ctrl_state;
        ctrl_state.running_state = linglong_timemachine_running_state;
        ctrl_state.speed = linglong_timemachine_speed;
        ctrl_state.projection = global_state->linglong_cfg->projection;
        ctrl_state.equatorial_coord = global_state->linglong_cfg->enable_equatorial_coord;
        ctrl_state.horizontal_coord = global_state->linglong_cfg->enable_horizontal_coord;
        ctrl_state.sky_model = global_state->linglong_cfg->sky_model;
        ctrl_state.landscape_index = global_state->linglong_cfg->landscape_index;
        ctrl_state.roam_enable = global_state->linglong_cfg->roam_enable;
        ctrl_state.roam_speed_level = global_state->linglong_cfg->roam_speed_level;
        ui_linglong_ctrl_draw(global_state->llgfx, &ctrl_state);
    }

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    convert_rgb888_to_rgb565_double(global_state->gfx, global_state->llgfx->frame_buffer_rgb888, global_state->llgfx->width, global_state->llgfx->height);
    gfx_refresh(global_state->gfx);
#else
    // convert_rgb888_to_rgb565_double(global_state->gfx, global_state->llgfx->frame_buffer_rgb888, global_state->llgfx->width, global_state->llgfx->height);
    gfx_refresh(global_state->llgfx);
#endif

}


void ui_linglong_toggle_timemachine(Key_Event *key_event, Global_State *global_state) {
    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = 1;
    }
    else {
        linglong_timemachine_running_state = 0;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

void ui_linglong_set_timemachine_speed(Key_Event *key_event, Global_State *global_state, int32_t speed) {
    linglong_timemachine_speed = speed;
    switch (linglong_timemachine_running_state) {
        case 0: linglong_timemachine_running_state = 1; break;
        case 1: linglong_timemachine_running_state = 0; break;
        case 2: linglong_timemachine_running_state = 1; break;
        default: linglong_timemachine_running_state = 0; break;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

void ui_linglong_set_realtime(Key_Event *key_event, Global_State *global_state) {
    Linglong_Config *llcfg = global_state->linglong_cfg;
    time_t ts = (time_t)(global_state->timestamp / 1000);
    struct tm *timeinfo = localtime(&ts); // 转换为本地时间
    llcfg->second = timeinfo->tm_sec;
    llcfg->minute = timeinfo->tm_min;
    llcfg->hour = timeinfo->tm_hour;
    llcfg->day = timeinfo->tm_mday;
    llcfg->month = timeinfo->tm_mon + 1;
    llcfg->year = timeinfo->tm_year + 1900;

    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = 2;
    }
    else {
        linglong_timemachine_running_state = 0;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

// 显式触屏控件：恢复当前时间（始终回到实时，不做“再按一次暂停”的开关语义）
static void ui_linglong_control_realtime(Global_State *global_state) {
    Linglong_Config *llcfg = global_state->linglong_cfg;
    time_t ts = (time_t)(global_state->timestamp / 1000);
    struct tm *timeinfo = localtime(&ts);
    llcfg->second = timeinfo->tm_sec;
    llcfg->minute = timeinfo->tm_min;
    llcfg->hour   = timeinfo->tm_hour;
    llcfg->day    = timeinfo->tm_mday;
    llcfg->month  = timeinfo->tm_mon + 1;
    llcfg->year   = timeinfo->tm_year + 1900;

    linglong_timemachine_running_state = 2;
    linglong_paused_resume_state = 2;
}

// 显式触屏控件：运行/暂停（暂停记住暂停前的模式，恢复时回到该模式）
static void ui_linglong_control_runpause(Global_State *global_state) {
    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = linglong_paused_resume_state;
        // 兜底：只可能恢复到“时光机”或“实时”
        if (linglong_timemachine_running_state != 1) {
            linglong_timemachine_running_state = 2;
        }
    }
    else {
        linglong_paused_resume_state = linglong_timemachine_running_state;
        linglong_timemachine_running_state = 0;
    }
}

// ===============================================================================
// 视角按钮的“按住反复触发”
// ===============================================================================
// 视角九宫格的 8 枚方向/缩放按钮（正中“归中”除外）支持按住反复触发，交互与漫游移动 pad
// 一致：按住不放即连续转动/缩放视角。触发节奏取键盘自动重复的两段式——
//   · 短按（< 初始延时）：只由松手沿的单次点击完成（与改动前完全一致，拖动离手不触发）；
//   · 长按：超过初始延时后按固定间隔重复同一步进（5° / 0.1f，约 70°/s），松手即停。
// 状态按“当前按住的按钮”跟踪：按下点移到别的按钮、面板隐藏或松手都会重新计时/停掉，
// 因而换按钮不会有上一枚按钮的重复惯性（状态无需显式复位：每帧轮询电平自行收敛）。
#define LL_VIEW_REPEAT_DELAY_MS     (320)
#define LL_VIEW_REPEAT_INTERVAL_MS  (70)

static int32_t  s_view_repeat_action = UI_LL_CTRL_NONE;
static uint64_t s_view_repeat_next_ms = 0;

// 每帧调用一次；返回本帧应重复触发的视角动作（UI_LL_CTRL_VIEW_n），否则 NONE
static int32_t ui_linglong_view_repeat(int32_t is_touching, int32_t x, int32_t y) {
    int32_t held = ui_linglong_ctrl_view_held(is_touching, x, y);
    if (held == UI_LL_CTRL_NONE) {
        s_view_repeat_action = UI_LL_CTRL_NONE; // 松手/移出按钮/面板隐藏：停掉重复
        return UI_LL_CTRL_NONE;
    }
    const uint64_t now = get_timestamp_in_ms();
    if (held != s_view_repeat_action) {
        s_view_repeat_action = held;                          // 新的按住：先等初始延时
        s_view_repeat_next_ms = now + LL_VIEW_REPEAT_DELAY_MS;
        return UI_LL_CTRL_NONE;
    }
    if (now < s_view_repeat_next_ms) return UI_LL_CTRL_NONE;
    s_view_repeat_next_ms = now + LL_VIEW_REPEAT_INTERVAL_MS;
    return held;
}

void ui_linglong_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 获取机器姿态（欧拉角）
#ifdef IMU_ENABLED

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)

#else
    if (global_state->linglong_cfg->enable_imu) {
        imu_read_angle(&(global_state->pitch), &(global_state->roll), &(global_state->yaw));
        printf("俯仰=%-10.2f    滚转=%-10.2f    航向=%-10.2f\n", global_state->pitch, global_state->roll, global_state->yaw);
    }
#endif

#endif

    // 主天空界面显式触屏控件（已无启动画面停留期，进入即刻接管触屏）。
    // 设置界面另有 5x4 显式按钮，见下方独立分支。
    if (linglong_state == LL_STATE_SKY) {
        // 触屏拖动变换视角：仅在控制面板隐藏时启用（控制模块内部判断），
        // 按下点落在顶部按钮上时不启动拖动。像素增量按实际画布尺寸归一化：
        // 横/纵向拖满整屏分别对应 180° 方位/俯仰变化，不写死屏幕宽高。
        // 方向为“抓取天空”：画面随手指同向移动——右拖天空右移（视角左转）、
        // 下拖天空下移（视角抬头）；即与十六键数字键的“相机转向”方向相反。
        int32_t drag_dx = 0, drag_dy = 0;
        if (ui_linglong_ctrl_poll_drag(
                key_event->touch_edge, key_event->is_touching,
                key_event->touch_x, key_event->touch_y,
                key_event->touch_down_x, key_event->touch_down_y,
                &drag_dx, &drag_dy)) {
            Linglong_Config *llcfg = global_state->linglong_cfg;
            int32_t canvas_w = (int32_t)global_state->llgfx->width;
            int32_t canvas_h = (int32_t)global_state->llgfx->height;
            if (canvas_w > 0 && canvas_h > 0) {
                llcfg->enable_imu = 0; // 手动拖动，关闭IMU
                llcfg->view_azi -= (float)drag_dx * 180.0f / (float)canvas_w;
                while (llcfg->view_azi >= 360.0f) llcfg->view_azi -= 360.0f;
                while (llcfg->view_azi < 0.0f) llcfg->view_azi += 360.0f;
                llcfg->view_alt += (float)drag_dy * 180.0f / (float)canvas_h;
                if (llcfg->view_alt > 90.0f) llcfg->view_alt = 90.0f;
                if (llcfg->view_alt < -90.0f) llcfg->view_alt = -90.0f;
            }
        }

        int32_t action = ui_linglong_ctrl_handle_touch(
            key_event->touch_edge, key_event->touch_x, key_event->touch_y,
            key_event->touch_down_x, key_event->touch_down_y);

        // 视角方向按钮的按住重复：逐帧轮询触点（与漫游 pad 同一范式，与触屏边沿无关）
        int32_t repeat_action = ui_linglong_view_repeat(
            key_event->is_touching, key_event->touch_x, key_event->touch_y);

        // 控件显隐/按住高亮变化无需特殊处理：玲珑仪每帧都完整重绘（暂停也不例外），
        // 控件状态每帧都会被重画在新画面上；这里只把脏标记消费掉，避免它一直挂着。
        (void)ui_linglong_ctrl_take_dirty();

        if (action != UI_LL_CTRL_NONE) {
            int32_t synth_key = NANO_KEY_IDLE;

            switch (action) {
                case UI_LL_CTRL_SETTINGS:
                    // 控制面板右上角“设置”：打开 5x4 显式触屏设置界面
                    global_state->is_ctrl_enabled = 1;
                    linglong_state = LL_STATE_SETTING;
                    ui_linglong_ctrl_reset();
                    ui_linglong_ctrl_settings_reset();
                    return;

                case UI_LL_CTRL_EXIT:
                    // 显式控件的“退出”按钮：退出玲珑仪（实体硬 A 键同语义）
                    global_state->is_ctrl_enabled = 0;
                    global_state->STATE = STATE_MAIN_MENU;
                    return;

                case UI_LL_CTRL_REWIND:
                    // 沿用十六键“*”键行为：设置快退速度并切换启停（再点一次即暂停）
                    ui_linglong_set_timemachine_speed(key_event, global_state, -120);
                    linglong_paused_resume_state = 1;
                    break;
                case UI_LL_CTRL_REALTIME:
                    ui_linglong_control_realtime(global_state);
                    break;
                case UI_LL_CTRL_RUNPAUSE:
                    ui_linglong_control_runpause(global_state);
                    break;
                case UI_LL_CTRL_FORWARD:
                    // 沿用十六键“#”键行为：设置快进速度并切换启停（再点一次即暂停）
                    ui_linglong_set_timemachine_speed(key_event, global_state, 120);
                    linglong_paused_resume_state = 1;
                    break;

                case UI_LL_CTRL_PROJECTION:
                    // 投影算法：鱼眼 ↔ 透视
                    global_state->linglong_cfg->projection =
                        (global_state->linglong_cfg->projection == 0) ? 1 : 0;
                    break;
                case UI_LL_CTRL_EQUATORIAL:
                    // 赤道坐标：关 → 开 → 关
                    global_state->linglong_cfg->enable_equatorial_coord =
                        (global_state->linglong_cfg->enable_equatorial_coord + 1) % 2;
                    break;
                case UI_LL_CTRL_HORIZONTAL:
                    // 地平坐标：关 → 方位角 → 坐标圈 → 关
                    global_state->linglong_cfg->enable_horizontal_coord =
                        (global_state->linglong_cfg->enable_horizontal_coord + 1) % 3;
                    break;
                case UI_LL_CTRL_SKY_MODEL:
                    // 大气散射：关/二次散射/体积云/海洋天空 循环（简化模型、一次散射已屏蔽）
                    global_state->linglong_cfg->sky_model =
                        ui_linglong_next_sky_model(global_state->linglong_cfg->sky_model);
                    break;
                case UI_LL_CTRL_LANDSCAPE:
                    // 地景：0..3 循环
                    global_state->linglong_cfg->landscape_index =
                        (global_state->linglong_cfg->landscape_index + 1) % 4;
                    break;
                case UI_LL_CTRL_ROAM:
                    // 漫游模式开关（顶部第三枚按钮）：开启后底部行换成 6 键移动 pad
                    global_state->linglong_cfg->roam_enable =
                        (global_state->linglong_cfg->roam_enable != 0) ? 0 : 1;
                    if (global_state->linglong_cfg->roam_enable) {
                        s_roam_last_ms = 0;   // 避免用上一段漫游的 dt
                    }
                    break;
                case UI_LL_CTRL_MOVE_FWD:
                case UI_LL_CTRL_MOVE_BACK:
                case UI_LL_CTRL_MOVE_LEFT:
                case UI_LL_CTRL_MOVE_RIGHT:
                case UI_LL_CTRL_MOVE_UP:
                case UI_LL_CTRL_MOVE_DOWN:
                    // 移动 pad 是“按住连续移动”（ui_linglong_ctrl_roam_held 每帧轮询），
                    // 松手沿的点击不产生额外动作
                    break;

                default:
                    // 视角九宫格：合成对应的 NANO_KEY_1~9 短按事件，
                    // 复用下方既有硬键处理逻辑（功能与十六键数字键 1~9 一一对应）。
                    // 8 枚方向/缩放按钮另有“按住反复触发”（见 ui_linglong_view_repeat），
                    // 短按仍走这里的一次触发。
                    if (action >= UI_LL_CTRL_VIEW_1 && action <= UI_LL_CTRL_VIEW_9) {
                        global_state->is_ctrl_enabled = 0;
                        synth_key = NANO_KEY_1 + (action - UI_LL_CTRL_VIEW_1);
                    }
                    break;
            }

            if (synth_key != NANO_KEY_IDLE) {
                key_event->key_code = (uint8_t)synth_key;
                key_event->key_edge = -1; // 合成短按（松手沿已过，直接视为一次完整点击）
                key_event->is_soft_key = 1;
            }
            else {
                // 时间控制已直接处理，吞掉本次事件，避免与同时按下的硬键重复响应
                key_event->key_code = NANO_KEY_IDLE;
                key_event->key_edge = 0;
            }
        }
        else if (repeat_action != UI_LL_CTRL_NONE && key_event->key_edge == 0) {
            // 按住反复触发：合成同键码的长按重复事件（key_edge=-2，与实体键长按同一语义），
            // 复用下方既有 NANO_KEY_1..9 处理链，保证“点按”与“按住”两条路径行为一致。
            // 只在“无沿事件”的帧注入（key_edge==0：空闲或实体键按住中）——此时不会丢掉
            // 任何上升/下降沿，而实体键按住（如漫游中按着 W）也不会阻断视角重复。
            global_state->is_ctrl_enabled = 0;
            key_event->key_code = (uint8_t)(NANO_KEY_1 + (repeat_action - UI_LL_CTRL_VIEW_1));
            key_event->key_edge = -2;
            key_event->is_soft_key = 1;
        }
    }

    // 设置界面显式触屏控件：5x4 按钮，替代旧 4x4 宫格软键。
    // 命中功能格后合成对应 NANO_KEY_* 短按事件，复用下方既有 Ctrl+键 处理链。
    if (linglong_state == LL_STATE_SETTING || linglong_state == LL_STATE_SETTING_CALLBACK) {
        int32_t cell = ui_linglong_ctrl_settings_handle_touch(
            key_event->touch_edge, key_event->touch_x, key_event->touch_y,
            key_event->touch_down_x, key_event->touch_down_y);
        if (cell >= 0 && cell <= 2) {
            // 云参数：直接循环档位（硬键没有空位可映射，与海洋参数按钮同为触屏专用）
            if (cell == 0) {
                ui_linglong_cloud_next_coverage(global_state);
            }
            else if (cell == 1) {
                ui_linglong_cloud_next_layer(global_state);
            }
            else {
                ui_linglong_cloud_next_precision(global_state);
            }
            linglong_state = LL_STATE_SETTING_CALLBACK;   // 重绘设置界面（现值已变）
        }
        else if (cell == 5) {
            // 镜头光晕开关（太阳离轴色散光斑；默认开）
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->enable_lens_flare = (oc->enable_lens_flare != 0) ? 0 : 1;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 6) {
            // 星芒开关（太阳十字星芒；默认关）
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->enable_star_burst = (oc->enable_star_burst != 0) ? 0 : 1;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell >= 11 && cell <= 13) {
            // 海洋参数：直接循环档位（硬键没有空位可映射，与云参数按钮同为触屏专用）
            Linglong_Config *oc = global_state->linglong_cfg;
            int32_t n = 0;
            if (cell == 11) {
                // 观察高度：档位表是“快捷跳转”（真实高度在漫游中是连续量）
                const Linglong_Ocean_Option *ho = linglong_ocean_height_options(&n);
                if (n > 0) {
                    oc->ocean_height_level = (oc->ocean_height_level + 1) % n;
                    if (ho != NULL) oc->obs_y = ho[oc->ocean_height_level].value;
                }
            }
            else if (cell == 12) {
                linglong_ocean_swell_hs_options(&n);
                oc->ocean_swell_hs_level = (oc->ocean_swell_hs_level + 1) % (n > 0 ? n : 1);
            }
            else {
                linglong_ocean_swell_len_options(&n);
                oc->ocean_swell_len_level = (oc->ocean_swell_len_level + 1) % (n > 0 ? n : 1);
            }
            linglong_state = LL_STATE_SETTING_CALLBACK;   // 重绘设置界面（现值已变）
        }
        else if (cell == 10) {
            // 岛屿地形开关（landscape_index==3 时生效）
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->terrain_enable = (oc->terrain_enable != 0) ? 0 : 1;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 14) {
            // 地形精度：省电 → 标准 → 精细 → 省电
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->terrain_quality = (oc->terrain_quality + 1) % 3;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 15) {
            // 换一座岛：种子 +1 并立即作废重建（渐进生成，界面不冻结）
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->terrain_seed = (oc->terrain_seed == 0) ? 1u : oc->terrain_seed + 1u;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 16) {
            // 高程图分辨率：512（50m 纹素）↔ 1024（25m 纹素）
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->terrain_res = (oc->terrain_res >= 1024) ? 512 : 1024;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 17) {
            // 漫游速度：10 / 100 / 1000 m/s
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->roam_speed_level = (oc->roam_speed_level + 1) % 3;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 19) {
            Linglong_Config *oc = global_state->linglong_cfg;
            oc->roam_enable = (oc->roam_enable != 0) ? 0 : 1;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
        else if (cell == 18) {
            // 观察位置：只读显示
        }
        else if (cell >= 0) {
            int32_t synth_key = ui_linglong_setting_cell_key(cell);
            if (synth_key != NANO_KEY_IDLE) {
                global_state->is_ctrl_enabled = 1;
                key_event->key_code = (uint8_t)synth_key;
                key_event->key_edge = -1;
                key_event->is_soft_key = 1;
            }
        }
    }

    // 漫游：方向键 / W A S D / Q E 按住连续移动（漫游开启时方向键不再兼作时光机）
    if (global_state->linglong_cfg->roam_enable &&
        (linglong_state == LL_STATE_SKY || linglong_state == LL_STATE_SETTING ||
         linglong_state == LL_STATE_SETTING_CALLBACK)) {
        switch (key_event->key_code) {
            case NANO_KEY_up:    case NANO_KEY_W: case NANO_KEY_w: ui_linglong_roam_key_hold(UI_LL_ROAM_FWD);   break;
            case NANO_KEY_down:  case NANO_KEY_S: case NANO_KEY_s: ui_linglong_roam_key_hold(UI_LL_ROAM_BACK);  break;
            case NANO_KEY_left:  case NANO_KEY_A: case NANO_KEY_a: ui_linglong_roam_key_hold(UI_LL_ROAM_LEFT);  break;
            case NANO_KEY_right: case NANO_KEY_D: case NANO_KEY_d: ui_linglong_roam_key_hold(UI_LL_ROAM_RIGHT); break;
            case NANO_KEY_E:     case NANO_KEY_e: case NANO_KEY_pgup: ui_linglong_roam_key_hold(UI_LL_ROAM_UP);   break;
            case NANO_KEY_Q:     case NANO_KEY_q: case NANO_KEY_pgdn: ui_linglong_roam_key_hold(UI_LL_ROAM_DOWN); break;
            default: break;
        }
    }

    int32_t is_setting_refresh = 0;

    // 按1键向左偏航（yaw--），或者Ctrl时切换投影算法
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_1) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_azi -= 5.0f;
            if (global_state->linglong_cfg->view_azi <= 0.0f) {
                global_state->linglong_cfg->view_azi = 360.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            if (global_state->linglong_cfg->projection == 0) {
                global_state->linglong_cfg->projection = 1;
            }
            else {
                global_state->linglong_cfg->projection = 0;
            }
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按2键推杆低头（pitch--），或者Ctrl时切换赤道坐标圈
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_2) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_alt -= 5.0f;
            if (global_state->linglong_cfg->view_alt <= -90.0f) {
                global_state->linglong_cfg->view_alt = -90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_equatorial_coord ++;
            global_state->linglong_cfg->enable_equatorial_coord = global_state->linglong_cfg->enable_equatorial_coord % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按3键向右偏航（yaw++），或者Ctrl时切换地平坐标
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_3) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_azi += 5.0f;
            if (global_state->linglong_cfg->view_azi >= 360.0f) {
                global_state->linglong_cfg->view_azi = 0.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_horizontal_coord++;
            global_state->linglong_cfg->enable_horizontal_coord = global_state->linglong_cfg->enable_horizontal_coord % 3;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按4键向左坡度（roll--），或者Ctrl时切换黄道
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_4) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_roll -= 5.0f;
            if (global_state->linglong_cfg->view_roll <= -90.0f) {
                global_state->linglong_cfg->view_roll = -90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_ecliptic_circle++;
            global_state->linglong_cfg->enable_ecliptic_circle = global_state->linglong_cfg->enable_ecliptic_circle % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按5键归中，或切换陀螺仪状态，或者Ctrl时切换天体名称
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_5) {
        if (global_state->is_ctrl_enabled == 0) {
            // 如果IMU已关闭，则开启
            if (global_state->linglong_cfg->enable_imu == 0) {
                global_state->linglong_cfg->enable_imu = 1;
            }
            // 如果IMU已开启，则关闭并归中
            else {
                global_state->linglong_cfg->enable_imu = 0;
                global_state->linglong_cfg->view_alt = 90.0f;
                global_state->linglong_cfg->view_azi = 180.0f;
                global_state->linglong_cfg->view_f = 1.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_star_name++;
            global_state->linglong_cfg->enable_star_name = global_state->linglong_cfg->enable_star_name % 4;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按6键向右坡度（roll++），或者Ctrl时切换姿态指示
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_6) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_roll += 5.0f;
            if (global_state->linglong_cfg->view_roll >= 90.0f) {
                global_state->linglong_cfg->view_roll = 90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_att_indicator++;
            global_state->linglong_cfg->enable_att_indicator = global_state->linglong_cfg->enable_att_indicator % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按7键拉远，或者Ctrl时切换大气散射模型
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_7) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->view_f -= 0.1f;
            if (global_state->linglong_cfg->view_f <= 0.1f) global_state->linglong_cfg->view_f = 0.1f;
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->sky_model =
                ui_linglong_next_sky_model(global_state->linglong_cfg->sky_model);
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按8键拉杆抬头（pitch++），或者Ctrl时切换地景
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_8) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_alt += 5.0f;
            if (global_state->linglong_cfg->view_alt >= 90.0f) {
                global_state->linglong_cfg->view_alt = 90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->landscape_index++;
            global_state->linglong_cfg->landscape_index = global_state->linglong_cfg->landscape_index % 4;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按9键推近，或者Ctrl时切换平滑滤波
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_9) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->view_f += 0.1f;
            if (global_state->linglong_cfg->view_f >= 5.0f) global_state->linglong_cfg->view_f = 5.0f;
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_opt_bilinear++;
            global_state->linglong_cfg->enable_opt_bilinear = global_state->linglong_cfg->enable_opt_bilinear % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按A键返回主菜单
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->is_ctrl_enabled = 0;
        global_state->STATE = STATE_MAIN_MENU;
    }
    // 按B键+Ctrl校准IMU
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_shift) {
        if (global_state->is_ctrl_enabled == 0) {
            // TODO
        }
        else {
            // global_state->is_ctrl_enabled = 0;
#ifdef IMU_ENABLED
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n    正在校准IMU...", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
            imu_calib();
            sleep_in_ms(500);
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n    校准完成", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
#endif
        }
    }
    // 按C键切换Ctrl
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_ctrl) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->is_ctrl_enabled = 1;
            linglong_state = LL_STATE_SETTING;
            ui_linglong_ctrl_settings_reset();
        }
        else {
            // 退出设置界面 → 回到控制面板（时间/视角/设置按钮同显），而不是空天空：
            // 设置界面自身没有顶部按钮栏，返回后落在“能继续操作”的地方才顺手。
            global_state->is_ctrl_enabled = 0;
            linglong_state = LL_STATE_SKY;
            ui_linglong_ctrl_show_controls();
        }
    }
    // 按*键时光机向前（过去）（反复按运行/暂停）；漫游模式下方向键用于移动
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_left &&
             global_state->linglong_cfg->roam_enable == 0) {
        ui_linglong_set_timemachine_speed(key_event, global_state, -120);
    }
    // 按0键回到实时（反复按运行/暂停），或者Ctrl时切换跟踪太阳
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_0) {
        if (global_state->is_ctrl_enabled == 0) {
            ui_linglong_set_realtime(key_event, global_state);
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_tracking_sun++;
            global_state->linglong_cfg->enable_tracking_sun = global_state->linglong_cfg->enable_tracking_sun % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按#键时光机向后（未来）（反复按运行/暂停）；漫游模式下方向键用于移动
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_right &&
             global_state->linglong_cfg->roam_enable == 0) {
        ui_linglong_set_timemachine_speed(key_event, global_state, 120);
    }

    if (linglong_state != LL_STATE_SKY) {
        // 设置界面不显示主界面控件；无论经控制面板还是硬 C 键进入，返回主界面时
        // 时间/视角面板都从隐藏状态开始，避免旧面板状态残留。
        ui_linglong_ctrl_reset();
    }
    if (linglong_state != LL_STATE_SETTING && linglong_state != LL_STATE_SETTING_CALLBACK) {
        ui_linglong_ctrl_settings_reset();
    }
}

// ===============================================================================
// 全局资源（main_init 调用一次）：配置对象 + 专用画布
//
// ESP32 上玲珑仪有自己的 RGB888 画布（渲染完转 RGB565 拷进主帧缓冲）；其它平台直接
// 复用主帧缓冲（llgfx = gfx），因此独立 gfx_init 的分支只在 ESP32 平台编译。
// 配置对象（Linglong_Config，含大气散射 LUT 参数与全部视图/天空档位）由 celestial.c
// 的 linglong_init 填默认值。
// ===============================================================================
void ui_linglong_global_init(Global_State *global_state) {
    global_state->linglong_cfg = (Linglong_Config *)platform_calloc(1, sizeof(Linglong_Config));
    linglong_init(global_state->linglong_cfg);

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    global_state->llgfx = (Nano_GFX*)platform_calloc(1, sizeof(Nano_GFX));
    global_state->llgfx->is_double_buffer = 0;
    gfx_init(global_state->llgfx, SCREEN_WIDTH, SCREEN_HEIGHT, GFX_COLOR_MODE_RGB888);
#else
    global_state->llgfx = global_state->gfx;
#endif
}
