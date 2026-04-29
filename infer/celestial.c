#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "platform.h"
#include "glyph.h"
#include "graphics.h"

#include "linglong_texture.h"
#include "celestial.h"
#include "ephemeris.h"

#ifndef M_PI
    #define M_PI (3.14159265358979323846)
#endif

#ifndef M_PI_2
    # define M_PI_2 (1.57079632679489661923)
#endif

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))



// 行星相关常量（用于渲染）
//                                        -     1Mer  2Ven  3Ear  4Mars 5Jup  6Sat  7Ura  8Nep
static const float   PLANET_RADIUS[9]   = {0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 3.0f, 2.0f, 1.0f, 1.0f};
static const uint8_t PLANET_COLOR_R[9]  = {0,    192,  255,  0,    255,  255,  192,  0,    192 };
static const uint8_t PLANET_COLOR_G[9]  = {0,    192,  255,  0,    64,   192,  255,  255,  128 };
static const uint8_t PLANET_COLOR_B[9]  = {0,    192,  64,   0,    0,    128,  128,  255,  255 };
static const wchar_t PLANET_NAME[9][10] = {L"", L"水星", L"金星", L"地球", L"火星", L"木星", L"土星", L"天王星", L"海王星"};

// 星表
#define STARS_NUM (89)
static const float STARS[STARS_NUM][7] = {
//(RA)h     m      s    (Dec)d       m      s        Mag
    {2.0f,  32.0f,  9.3f,     89.0f, 16.0f, 10.8f,   2.09f}, // 勾陈一（北极星）
    {11.0f,  3.0f, 45.2f,     61.0f, 44.0f, 47.5f,   1.95f}, // 天枢
    {11.0f,  1.0f, 52.5f,     56.0f, 22.0f, 43.0f,   2.34f}, // 天璇
    {11.0f, 53.0f, 51.4f,     53.0f, 41.0f, 24.5f,   2.42f}, // 天玑
    {12.0f, 15.0f, 27.0f,     57.0f,  1.0f, 39.6f,   3.33f}, // 天权
    {12.0f, 54.0f,  2.8f,     55.0f, 57.0f, 16.2f,   1.75f}, // 玉衡
    {13.0f, 23.0f, 56.3f,     54.0f, 55.0f, 11.3f,   2.25f}, // 开阳
    {13.0f, 47.0f, 32.2f,     49.0f, 18.0f, 28.7f,   1.80f}, // 摇光

    { 6.0f, 45.0f, 91.0f,    -16.0f, 43.0f, 35.0f,  -1.46f}, // 天狼
    {14.0f, 15.0f, 37.6f,     19.0f,  9.0f, 53.3f,  -0.05f}, // 大角
    {18.0f, 36.0f, 55.2f,     38.0f, 46.0f, 59.8f,   0.03f}, // 织女一
    {19.0f, 50.0f, 46.6f,      8.0f, 52.0f, 10.9f,   0.76f}, // 河鼓二
    { 0.0f, 42.0f, 43.6f,     41.0f, 16.0f, 15.7f,   3.40f}, // M31
    { 5.0f, 35.0f,  8.3f,      9.0f, 56.0f,  3.0f,   3.54f}, // 觜宿一
    { 5.0f, 55.0f, 11.4f,      7.0f, 24.0f, 22.2f,   0.42f}, // 参宿四
    { 5.0f, 25.0f,  8.8f,      6.0f, 20.0f, 55.1f,   1.64f}, // 参宿五
    { 5.0f, 40.0f, 46.6f,     -1.0f, 56.0f, 38.7f,   1.77f}, // 参宿一
    { 5.0f, 36.0f, 13.8f,     -1.0f, 12.0f, 12.3f,   1.69f}, // 参宿二
    { 5.0f, 32.0f,  1.4f,      0.0f,-18.0f,  1.9f,   2.41f}, // 参宿三
    { 5.0f, 47.0f, 46.5f,     -9.0f, 40.0f, 17.7f,   2.06f}, // 参宿六
    { 5.0f, 14.0f, 33.2f,     -8.0f, 12.0f, 13.0f,   0.13f}, // 参宿七
    { 5.0f, 35.0f, 27.0f,     -5.0f, 54.0f, 42.2f,   2.77f}, // 伐三
    { 3.0f, 47.0f, 29.4f,     24.0f,  6.0f, 19.6f,   2.87f}, // 昴宿星团

    // 以下是AI生成的，姑且保留，待确认
    { 6.0f, 23.0f, 57.1f,    -52.0f, 41.0f, 45.0f,  -0.72f}, // 老人星（船底座α）
    {14.0f, 39.0f, 35.9f,    -60.0f, 50.0f,  7.0f,  -0.01f}, // 南门二（半人马座α）
    { 5.0f, 16.0f, 41.4f,     45.0f, 59.0f, 53.0f,   0.08f}, // 五车二（御夫座α）
    { 7.0f, 39.0f, 18.1f,      5.0f, 13.0f, 30.0f,   0.38f}, // 南河三（小犬座α）
    { 1.0f, 37.0f, 42.9f,    -57.0f, 14.0f, 12.0f,   0.46f}, // 水委一（波江座α）
    {14.0f,  3.0f, 49.4f,    -60.0f, 22.0f, 23.0f,   0.61f}, // 马腹一（半人马座β）
    { 4.0f, 35.0f, 55.2f,     16.0f, 30.0f, 33.0f,   0.85f}, // 毕宿五（金牛座α）
    {16.0f, 29.0f, 24.4f,    -26.0f, 25.0f, 55.0f,   0.96f}, // 心宿二（天蝎座α，变星）
    {13.0f, 25.0f, 11.6f,    -11.0f,  9.0f, 41.0f,   0.98f}, // 角宿一（室女座α）
    { 7.0f, 45.0f, 18.9f,     28.0f,  1.0f, 34.0f,   1.14f}, // 北河三（双子座β）
    {22.0f, 57.0f, 39.1f,    -29.0f, 37.0f, 20.0f,   1.16f}, // 北落师门（南鱼座α）
    {12.0f, 47.0f, 43.2f,    -59.0f, 41.0f, 19.0f,   1.25f}, // 十字架三（南十字座β）
    {20.0f, 41.0f, 25.9f,     45.0f, 16.0f, 49.0f,   1.25f}, // 天津四（天鹅座α）
    {12.0f, 26.0f, 35.9f,    -63.0f,  5.0f, 57.0f,   1.33f}, // 十字架二（南十字座α）
    {10.0f,  8.0f, 22.3f,     11.0f, 58.0f,  2.0f,   1.35f}, // 轩辕十四（狮子座α）
    {18.0f, 55.0f, 15.9f,    -26.0f, 17.0f, 48.0f,   2.05f}, // 斗宿四/箕宿三（人马座σ）
    { 2.0f,  7.0f, 10.4f,     23.0f, 27.0f, 45.0f,   2.00f}, // 娄宿三（白羊座α）

    // 以下更不准确，仅用于视觉效果
    { 0.0f,  9.0f, 10.7f,      29.0f, 18.0f, 44.0f,   2.06f}, // 壁宿二（Alpheratz）
    { 0.0f, 13.0f, 14.2f,      15.0f, 11.0f,  1.0f,   3.41f}, // 雷电一（Lambda Pegasi）
    { 0.0f, 19.0f, 43.6f,      -8.0f, 49.0f, 26.0f,   3.97f}, // 离宫一（Lambda Piscium）
    { 0.0f, 43.0f, 35.4f,     -17.0f, 59.0f, 12.0f,   2.04f}, // 土司空（Diphda）
    { 1.0f,  8.0f, 44.9f,     -10.0f, 10.0f, 56.0f,   3.60f}, // 外屏七（Chi Piscium）
    { 1.0f,  9.0f, 43.9f,      35.0f, 37.0f, 14.0f,   2.05f}, // 奎宿九（Mirach）
    { 2.0f,  2.0f,  2.8f,      -2.0f, 45.0f, 49.0f,   3.82f}, // 右更二（Theta Ceti）
    { 2.0f,  3.0f, 53.9f,      42.0f, 19.0f, 47.0f,   2.27f}, // 王良一（Caph）
    { 3.0f, 47.0f, 29.1f,      24.0f,  6.0f, 18.0f,   1.65f}, // 昴宿六（Alcyone）
    { 4.0f,  5.0f, 16.9f,      15.0f, 23.0f, 54.0f,   3.77f}, // 卷舌二（Xi Persei）
    { 4.0f, 49.0f, 50.4f,       6.0f, 57.0f, 40.0f,   3.77f}, // 砺石四（Gamma Eridani）
    { 6.0f,  8.0f, 57.7f,     -14.0f, 42.0f,  9.0f,   2.89f}, // 军市一（Mirzam）
    { 6.0f, 22.0f, 41.9f,     -17.0f, 57.0f, 21.0f,   2.75f}, // 娄宿三（Hamal）
    { 6.0f, 37.0f, 42.8f,      16.0f, 23.0f, 57.0f,   1.93f}, // 井宿三（Alhena）
    { 6.0f, 58.0f, 37.6f,     -28.0f, 58.0f, 19.0f,   1.50f}, // 弧矢七（Adhara）
    { 7.0f, 24.0f,  5.7f,     -29.0f, 18.0f, 11.0f,   3.02f}, // 孙增一（Delta Canis Majoris）
    { 7.0f, 34.0f, 36.0f,      31.0f, 53.0f, 19.0f,   1.58f}, // 北河二（Castor）
    { 7.0f, 45.0f, 18.9f,      28.0f,  1.0f, 34.0f,   1.14f}, // 北河三（Pollux）
    { 9.0f, 27.0f, 35.2f,      -8.0f, 39.0f, 31.0f,   1.98f}, // 星宿一（Alphard）
    { 9.0f, 45.0f, 51.1f,     -14.0f, 40.0f, 35.0f,   3.11f}, // 柳宿六（Zeta Hydrae）
    {10.0f, 35.0f, 48.4f,      -9.0f, 48.0f, 36.0f,   3.90f}, // 西上相（Theta Leonis）
    {11.0f, 53.0f, 49.8f,      53.0f, 41.0f, 41.0f,   3.01f}, // 文昌四（Al Haud）
    {12.0f, 21.0f, 21.6f,     -57.0f,  6.0f, 45.0f,   1.63f}, // 十字架一（Gacrux）
    {13.0f, 39.0f, 53.1f,     -53.0f, 27.0f, 10.0f,   2.55f}, // 库楼三（Gamma Centauri）
    {13.0f, 47.0f, 32.4f,     -49.0f, 18.0f, 48.0f,   1.86f}, // 海石一（Avior）
    {13.0f, 54.0f, 41.1f,      18.0f, 23.0f, 51.0f,   2.68f}, // 常陈一（Muphrid）
    {14.0f, 16.0f,  6.9f,      51.0f, 22.0f, 21.0f,   3.72f}, // 紫微右垣三（Kappa Draconis）
    {14.0f, 50.0f, 52.5f,     -16.0f,  2.0f, 30.0f,   2.75f}, // 氐宿一（Zubeneschamali）
    {15.0f, 20.0f, 43.7f,      -9.0f, 46.0f, 12.0f,   2.89f}, // 周鼎一（Zubenelgenubi）
    {15.0f, 34.0f, 41.2f,       1.0f, 32.0f, 32.0f,   3.38f}, // 东次相（Delta Virginis）
    {16.0f,  0.0f, 20.0f,     -22.0f, 37.0f, 18.0f,   2.74f}, // 尾宿五（Sargas）
    {16.0f, 35.0f, 52.2f,     -28.0f, 12.0f, 57.0f,   2.60f}, // 尾宿八（Shaula）
    {16.0f, 48.0f, 39.9f,     -69.0f,  1.0f, 40.0f,   2.69f}, // 三角形三（Atria）
    {17.0f, 10.0f, 22.7f,      15.0f, 43.0f, 18.0f,   3.23f}, // 奚仲三（Theta Herculis）
    {17.0f, 14.0f, 38.8f,     -15.0f, 36.0f, 43.0f,   3.49f}, // 宋增一（Kappa Ophiuchi）
    {17.0f, 22.0f, 51.2f,     -24.0f, 59.0f, 58.0f,   2.43f}, // 韩增一（Gamma Ophiuchi）
    {17.0f, 33.0f, 36.5f,     -37.0f,  6.0f, 13.0f,   2.80f}, // 侯（Cebalrai）
    {17.0f, 37.0f, 19.1f,     -42.0f, 59.0f, 52.0f,   2.75f}, // 望远镜座一（Alpha Telescopii）
    {18.0f,  5.0f, 27.2f,     -30.0f, 25.0f,  3.0f,   2.72f}, // 傅说（Phi Sagittarii）
    {18.0f, 21.0f,  8.9f,      -2.0f, 53.0f, 49.0f,   3.39f}, // 箕宿三（Kaus Australis）
    {18.0f, 27.0f, 58.1f,     -25.0f, 15.0f, 18.0f,   2.82f}, // 斗宿四（Nunki）
    {19.0f,  2.0f, 36.7f,     -29.0f, 52.0f, 15.0f,   3.51f}, // 建三（Gamma Telescopii）
    {19.0f, 55.0f, 50.4f,     -60.0f, 21.0f, 34.0f,   2.86f}, // 鸟喙一（Alpha Tucanae）
    {20.0f, 18.0f,  3.4f,     -12.0f, 32.0f, 41.0f,   2.97f}, // 垒壁阵五（Delta Capricorni）
    {20.0f, 25.0f, 38.9f,     -56.0f, 44.0f,  6.0f,   2.75f}, // 孔雀十一（Peacock）
    {21.0f, 31.0f, 33.5f,      -5.0f, 34.0f, 16.0f,   2.90f}, // 虚宿一（Beta Aquarii）
    {21.0f, 44.0f,  8.2f,     -14.0f, 32.0f, 57.0f,   3.73f}, // 危宿一（Alpha Equulei）
    {22.0f,  2.0f, 51.7f,     -43.0f,  8.0f, 45.0f,   3.49f}, // 败臼一（Gamma Gruis）
    {22.0f,  8.0f, 14.0f,     -46.0f, 57.0f, 40.0f,   1.74f}, // 鹤一（Al Nair）
};

static const wchar_t STAR_NAME[STARS_NUM][10] = {
    L"北极星", L"天枢", L"天璇", L"天玑", L"天权", L"玉衡", L"开阳", L"摇光",
    L"天狼", L"大角", L"织女一", L"河鼓二", L"M31",
    L"觜宿一", L"参宿四", L"参宿五", L"参宿一", L"参宿二", L"参宿三", L"参宿六", L"参宿七", L"伐三", L"昴宿星团",
    L"老人", L"南门二", L"五车二", L"南河三", L"水委一", L"马腹一", L"毕宿五", L"心宿二", L"角宿一", L"北河三", L"北落师门", L"十字架三",
    L"天津四", L"十字架二", L"轩辕十四", L"斗宿四", L"娄宿三",
    L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", 
    L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", 
    L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", L"", 
    L"", L"", L"", L""
};


// 地景
static uint32_t landscape_texture_width = 600;
static uint32_t landscape_texture_height = 600;
static uint8_t *landscape_texture_rgb = NULL;

#if LINGLONG_ENABLE_DYNAMIC_LANDSCAPE
static uint8_t *landscape_buffer_rgb = NULL;
#endif


// ===============================================================================
// 幅度转换
// ===============================================================================

// 从星等转换为相对亮度（归一化到 [0,1]，mag=0 → 1.0，mag=6 → ~0.01）
// 增加一个衰减系数（平均蓝天亮度），以便白天能够掩盖亮星
float magnitude_to_relative_luminance(float mag) {
    return 0.7f * powf(10.0f, -0.4f * mag);
}

float get_luminance(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}


// ===============================================================================
// 坐标转换
// ===============================================================================

static inline float to_deg_float(float rad) {
    return rad * (180.0f / M_PI);
}

static inline float to_rad_float(float deg) {
    return deg * (M_PI / 180.0f);
}

static inline float normalize_angle_float(float angle) {
    angle = fmodf(angle, 360.0f);
    if (angle < 0) angle += 360.0f;
    return angle;
}

// 将hms格式的赤经转为度数
static inline float ra_hms_to_deg(float h, float m, float s) {
    float totalHours = h + m / 60.0f + s / 3600.0f;
    return normalize_angle_float(totalHours * 15.0f);
}

// 将度分秒格式的赤纬转为度数
static inline float dec_dms_to_decimal(float d, float m, float s) {
    float sign = (d < 0.0f) ? (-1.0f) : (1.0f);
    float absD = fabsf(d);
    float decimal = sign * (absD + m / 60.0f + s / 3600.0f);
    return decimal;
}

// 地平天球的球坐标系→地平天球的笛卡尔坐标系
void horizontal_to_xyz(float azimuth_deg, float altitude_deg, float R, float *x, float *y, float *z) {
    float altRad = to_rad_float(altitude_deg);
    float azRad = to_rad_float(azimuth_deg);
    *x = R * cosf(altRad) * sinf(azRad);
    *y = R * cosf(altRad) * cosf(azRad);
    *z = R * sinf(altRad);
}

// 地平天球的球坐标系→地平天球的笛卡尔坐标系→投影到地面投影坐标系
void horizontal_to_screen_xy(
    float azimuth_deg, float altitude_deg, float radius,
    float center_x, float center_y, float view_height, float *scr_x, float *scr_y
) {
    // 因为是躺在地上看天，所以方位角是从正北逆时针旋转
    float az_rad = (-azimuth_deg * M_PI) / 180.0f;
    float alt_rad = (altitude_deg * M_PI) / 180.0f;
    // 天顶距 θ = π/2 - altitude
    float theta = M_PI_2 - alt_rad;
    // 等距鱼眼投影：r = (2R / π) * θ
    float r = (view_height == 1.0f) ? ((theta / (M_PI / 2.0f)) * radius): (powf(theta / (M_PI / 2.0f), (1.0f / view_height)) * radius);
    // 视野系数，高度越高视野越完整（越接近1），高度越低视野越不完整（越接近0）
    float range = MAX(0.1f, 1.0f - expf(-10.0f * view_height));
    r *= range;
    // float r = (2.0f * radius / M_PI) * theta;
    // 投影到平面：X指向东（注意因为是躺在地上看天，所以东是屏幕坐标系的左侧/负半轴），Y指向北
    float X = r * sinf(az_rad);
    float Y = -r * cosf(az_rad);

    *scr_x = X + center_x;
    *scr_y = Y + center_y;
}


// NOTE 已废弃
// 屏幕坐标系→地面投影坐标系→反演回地平天球的直角坐标系
// 返回值为正：能够反演回球面；返回值为负：无法反演回球面
int32_t screen_xy_to_xyz(
    float scr_x, float scr_y,
    float radius, float center_x, float center_y,
    float *x, float *y, float *z
) {
    float dx = scr_x - center_x;
    float dy = scr_y - center_y;
    float r = sqrtf(dx*dx + dy*dy) / radius; // 归一化半径
    if (r <= 1) {
        float theta = r * M_PI_2; // 天顶角
        float phi = atan2f(dx, dy); // 方位角
        *x = -radius * sinf(theta) * sinf(phi);
        *y = -radius * sinf(theta) * cosf(phi);
        *z =  radius * cosf(theta);
        return 1;
    }
    else {
        *x = 0.0f;
        *y = 0.0f;
        *z = 0.0f;
        return -1;
    }
}


// 笛卡尔坐标系 → 地平天球球坐标系 (altitude, azimuth, radius)
void xyz_to_horizontal(float x, float y, float z, float *azimuth_deg, float *altitude_deg, float *radius) {
    // 1. 计算半径（处理原点情况）
    float R = sqrtf(x * x + y * y + z * z);
    *radius = R;

    if (R < 1e-15f) {
        *azimuth_deg = 0.0f;
        *altitude_deg = 0.0f;
        *radius = 0.0f;
        return; // 原点无定义，返回默认值
    }

    // 2. 计算高度角（钳制避免浮点误差导致 asin 越界）
    float sinAlt = z / R;
    sinAlt = MAX(-1.0f, MIN(1.0f, sinAlt)); // 严格限制在 [-1, 1]
    *altitude_deg = to_deg_float(asinf(sinAlt));

    // 3. 处理天顶/天底（|z| ≈ R 时方位角无定义）
    if (fabsf(z) >= R * (1.0f - 1e-10f)) {
        *azimuth_deg = 0.0f;
        *radius = 0.0f;
        return; // 方位角设为0（惯例）
    }

    // 4. 计算方位角（核心：atan2(x, y) 对应原函数中的 azRad）
    float azRad = atan2f(x, y); // 注意参数顺序：x 对应 sin(azRad), y 对应 cos(azRad)
    *azimuth_deg = normalize_angle_float(to_deg_float(azRad)); // 逆向抵消原函数的 +180°

    return;
}



void cross(float a0, float a1, float a2, float b0, float b1, float b2, float *out0, float *out1, float *out2) {
    *out0 = a1 * b2 - a2 * b1;
    *out1 = a2 * b0 - a0 * b2;
    *out2 = a0 * b1 - a1 * b0;
    return;
}

float dot(float a0, float a1, float a2, float b0, float b1, float b2) {
    return a0 * b0 + a1 * b1 + a2 * b2;
}

void norm(float a0, float a1, float a2, float *out0, float *out1, float *out2) {
    float len = sqrtf(a0 * a0 + a1 * a1 + a2 * a2);
    if (len < 1e-20f) {
        *out0 = 0.0f;
        *out1 = 0.0f;
        *out2 = 0.0f;
    }
    else {
        *out0 = a0 / len;
        *out1 = a1 / len;
        *out2 = a2 / len;
    }
    return;
}


// 地平天球的球坐标系→地平天球的笛卡尔坐标系→投影到地面投影坐标系
// 新增roll_deg参数：滚转角(±90°)，正值为顺时针旋转（从观察者视角，右侧地平线下倾）
void fisheye_project(
    float azimuth_deg, float altitude_deg, float radius,
    float center_x, float center_y, 
    float view_alt, float view_azi, float view_roll, float f,
    int32_t projection,
    float *scr_x, float *scr_y
) {
    // 快速路径：鱼眼，默认视角且无roll时使用简化投影
    if (projection == 0 && view_alt == 90.0f && view_azi == 180.0f && view_roll == 0.0f && f == 1.0f) {
        horizontal_to_screen_xy(azimuth_deg, altitude_deg, radius, center_x, center_y, 1.0f, scr_x, scr_y);
        return;
    }

    // 处理天顶视角的数值稳定性问题（避免除零/奇异点）
    if (fabsf(view_alt - 90.0f) < 1e-5f) {
        view_alt = 89.99f;
    }

    // ===== 1. 计算视线方向的单位向量 (局部坐标系Z轴) =====
    float v_view_x = 0.0f, v_view_y = 0.0f, v_view_z = 0.0f;
    horizontal_to_xyz(view_azi, view_alt, radius, &v_view_x, &v_view_y, &v_view_z);
    float v_view_norm_x = 0.0f, v_view_norm_y = 0.0f, v_view_norm_z = 0.0f;
    norm(v_view_x, v_view_y, v_view_z, &v_view_norm_x, &v_view_norm_y, &v_view_norm_z);

    // ===== 2. 计算目标点的单位向量 =====
    float v_x = 0.0f, v_y = 0.0f, v_z = 0.0f;
    horizontal_to_xyz(azimuth_deg, altitude_deg, radius, &v_x, &v_y, &v_z);
    float v_norm_x = 0.0f, v_norm_y = 0.0f, v_norm_z = 0.0f;
    norm(v_x, v_y, v_z, &v_norm_x, &v_norm_y, &v_norm_z);

    // ===== 3. 构建局部坐标系 (无roll时) =====
    // Z轴: 视线方向 v_view_norm
    // Y轴: 天顶方向[0,0,1]在垂直于视线平面上的投影 (指向"上")
    float zenith_x = 0.0f, zenith_y = 0.0f, zenith_z = 1.0f;
    float s = dot(zenith_x, zenith_y, zenith_z, 
                  v_view_norm_x, v_view_norm_y, v_view_norm_z);
    float proj_x = v_view_norm_x * s;
    float proj_y = v_view_norm_y * s;
    float proj_z = v_view_norm_z * s;
    
    float yv_x = 0.0f, yv_y = 0.0f, yv_z = 0.0f;
    norm(zenith_x - proj_x, zenith_y - proj_y, zenith_z - proj_z, 
         &yv_x, &yv_y, &yv_z);
    
    // X轴: Y轴 × Z轴 (右手坐标系，指向"右")
    float xv_x = 0.0f, xv_y = 0.0f, xv_z = 0.0f;
    cross(yv_x, yv_y, yv_z, 
          v_view_norm_x, v_view_norm_y, v_view_norm_z, 
          &xv_x, &xv_y, &xv_z);

    // ===== 4. 应用Roll旋转: 绕视线方向(Z轴)旋转局部坐标系 =====
    float roll_rad = view_roll * M_PI / 180.0f;
    float cos_roll = cosf(roll_rad);
    float sin_roll = sinf(roll_rad);
    
    // 旋转基向量 (绕Z轴的2D旋转矩阵)
    // 新X = 旧X*cos(roll) - 旧Y*sin(roll)
    // 新Y = 旧X*sin(roll) + 旧Y*cos(roll)
    float xv_rot_x = xv_x * cos_roll - yv_x * sin_roll;
    float xv_rot_y = xv_y * cos_roll - yv_y * sin_roll;
    float xv_rot_z = xv_z * cos_roll - yv_z * sin_roll;
    
    float yv_rot_x = xv_x * sin_roll + yv_x * cos_roll;
    float yv_rot_y = xv_y * sin_roll + yv_y * cos_roll;
    float yv_rot_z = xv_z * sin_roll + yv_z * cos_roll;

    // ===== 5. 将目标点投影到旋转后的局部坐标系 =====
    float vx = dot(v_norm_x, v_norm_y, v_norm_z, 
                   xv_rot_x, xv_rot_y, xv_rot_z);  // 局部X分量
    float vy = dot(v_norm_x, v_norm_y, v_norm_z, 
                   yv_rot_x, yv_rot_y, yv_rot_z);  // 局部Y分量
    float vz = dot(v_norm_x, v_norm_y, v_norm_z, 
                   v_view_norm_x, v_view_norm_y, v_view_norm_z);  // 局部Z分量

    // ===== 6. 投影计算 =====
    // 鱼眼透视
    if (projection == 0) {
        // 处理边界：点在视线后方时返回中心
        if (vz <= -1.0f) {
            *scr_x = center_x;
            *scr_y = center_y;
            return;
        }
        // 防止浮点误差导致acos域错误
        if (vz > 1.0f) vz = 1.0f;

        float theta = acosf(vz);                    // 与视线的夹角 (rad)
        float phi_v = -atan2f(vx, vy);              // 局部平面内的方位角 (rad)
        float r = (f * 2.0f * radius / M_PI) * theta; // 等距投影: r = f * θ
        
        // 转换为屏幕坐标 (注意y轴负号匹配原坐标系约定)
        float x =  r * sinf(phi_v);
        float y = -r * cosf(phi_v);

        *scr_x = x + center_x;
        *scr_y = y + center_y;
    }
    // 线性透视投影
    else if (projection == 1) {
        // *** 透视投影要求目标点严格在相机前方（vz > 0） ***
        // vz ≤ 0 意味着点在相机后方或侧面，返回大误差值驱离优化器
        if (vz <= 0.0f) {
            *scr_x = center_x + 1e6;
            *scr_y = center_y + 1e6;
            return;
        }

        // *** 透视正向公式：
        //   screen_x - center_x = -f·R·vx/vz
        //   screen_y - center_y = -f·R·vy/vz
        // 等价于 r_proj = f·R·tan(θ) = f·R·√(vx²+vy²)/vz ***
        *scr_x = center_x - f * radius * vx / vz;
        *scr_y = center_y - f * radius * vy / vz;
        return;
    }
    else {
        *scr_x = center_x;
        *scr_y = center_y;
    }
    return;
}


// 鱼眼反投影：屏幕平面(x,y) → 地平天球笛卡尔坐标系(x,y,z)
// 新增view_roll参数：滚转角(±90°)，正值为顺时针旋转（从观察者视角，右侧地平线下倾）
// 与fisheye_project的roll约定保持一致，确保互为逆运算
void fisheye_unproject(
    float scr_x, float scr_y,
    float radius, float center_x, float center_y, 
    float view_alt, float view_azi, float view_roll, float f, int32_t projection,
    float *x, float *y, float *z
) {

    // 处理天顶视角的数值稳定性问题
    if (fabsf(view_alt - 90.0f) < 1e-5f) {
        view_alt = 89.99f;
    }

    // ===== 1. 屏幕坐标转极坐标 =====
    float dx = center_x - scr_x;
    float dy = center_y - scr_y;
    float r = sqrtf(dx * dx + dy * dy);

    // 处理中心点退化情况
    float phi_prime = 0.0f;
    if (r < 1e-6f) {
        phi_prime = 0.0f;
    }
    else {
        // NOTE 注意：atan2f(dx, dy) 与 project 中的 -atan2f(vx, vy) 互为逆运算
        phi_prime = atan2f(dx, dy);
    }

    // ===== 2. 极坐标转局部坐标系单位向量 =====

    float theta = 0.0f;
    // 透视投影
    if (projection == 1) {
        theta = atan2f(r, f * radius);
    }
    // 鱼眼投影
    else {
        theta = (M_PI / 2.0f) * (r / (radius * f));
    }

    // 防止theta超出[0, π]导致数值问题
    if (theta > M_PI) theta = M_PI;
    if (theta < 0.0f) theta = 0.0f;
    
    float pxy = sinf(theta);
    float p_prime_x = pxy * sinf(phi_prime);   // 局部X分量
    float p_prime_y = pxy * cosf(phi_prime);   // 局部Y分量  
    float p_prime_z = cosf(theta);              // 局部Z分量 (沿视线)

    // ===== 3. 构建视线方向的局部坐标系 (无roll时) =====
    float V_alt_rad = to_rad_float(view_alt);
    float V_azi_rad = to_rad_float(view_azi);

    // Z轴: 视线方向单位向量 v
    float v_x = cosf(V_alt_rad) * sinf(V_azi_rad);
    float v_y = cosf(V_alt_rad) * cosf(V_azi_rad);
    float v_z = sinf(V_alt_rad);

    // Y轴: 天顶[0,0,1]在垂直于视线平面上的投影 (指向"上")
    float dd = v_z;  // dot(zenith, v)
    float y_prime_x = -dd * v_x;
    float y_prime_y = -dd * v_y;
    float y_prime_z = 1.0f - dd * v_z;
    norm(y_prime_x, y_prime_y, y_prime_z, &y_prime_x, &y_prime_y, &y_prime_z);

    // X轴: Y × Z (右手系，指向"右")
    float x_prime_x = 0.0f, x_prime_y = 0.0f, x_prime_z = 0.0f;
    cross(y_prime_x, y_prime_y, y_prime_z, v_x, v_y, v_z, 
          &x_prime_x, &x_prime_y, &x_prime_z);
    norm(x_prime_x, x_prime_y, x_prime_z, &x_prime_x, &x_prime_y, &x_prime_z);

    // ===== 4. 应用Roll旋转: 绕视线方向(Z轴)旋转局部XY平面 =====
    float roll_rad = view_roll * M_PI / 180.0f;
    float cos_roll = cosf(roll_rad);
    float sin_roll = sinf(roll_rad);
    
    // 旋转基向量 (绕Z轴的2D旋转矩阵)
    // 新X = 旧X*cos(roll) - 旧Y*sin(roll)
    // 新Y = 旧X*sin(roll) + 旧Y*cos(roll)
    float x_prime_rot_x = x_prime_x * cos_roll - y_prime_x * sin_roll;
    float x_prime_rot_y = x_prime_y * cos_roll - y_prime_y * sin_roll;
    float x_prime_rot_z = x_prime_z * cos_roll - y_prime_z * sin_roll;
    
    float y_prime_rot_x = x_prime_x * sin_roll + y_prime_x * cos_roll;
    float y_prime_rot_y = x_prime_y * sin_roll + y_prime_y * cos_roll;
    float y_prime_rot_z = x_prime_z * sin_roll + y_prime_z * cos_roll;

    // ===== 5. 局部坐标转地平坐标系 (使用旋转后的基向量) =====
    // P_world = X'*p_x + Y'*p_y + Z'*p_z
    float px = x_prime_rot_x * p_prime_x + y_prime_rot_x * p_prime_y + v_x * p_prime_z;
    float py = x_prime_rot_y * p_prime_x + y_prime_rot_y * p_prime_y + v_y * p_prime_z;
    float pz = x_prime_rot_z * p_prime_x + y_prime_rot_z * p_prime_y + v_z * p_prime_z;

    // ===== 6. 归一化并缩放到指定radius =====
    norm(px, py, pz, &px, &py, &pz);
    *x = px * radius;
    *y = py * radius;
    *z = pz * radius;

    return;
}





/**
 * 双线性插值采样
 * @param {Uint8ClampedArray} frame_buffer - 图像数据
 * @param {number} fb_width - 图像宽度
 * @param {number} fb_height - 图像高度
 * @param {number} x - 采样点 X 坐标 (浮点数)
 * @param {number} y - 采样点 Y 坐标 (浮点数)
 * @returns {number[]} [R, G, B]
 */
static void bilinear_sample(uint8_t *frame_buffer, uint32_t fb_width, uint32_t fb_height, float x, float y, float *r, float *g, float *b) {
    // 边界检查
    if (x < 0 || x >= fb_width || y < 0 || y >= fb_height) {
        *r = *g = *b = 0;
        return;
    }

    int32_t x0 = (int32_t)floorf(x);
    int32_t y0 = (int32_t)floorf(y);
    int32_t x1 = (x0 + 1) % fb_width;
    int32_t y1 = MIN(y0 + 1, fb_height - 1);

    float dx = x - (float)x0;
    float dy = y - (float)y0;

    int32_t idx00 = (y0 * fb_width + x0) * 3;
    int32_t idx10 = (y0 * fb_width + x1) * 3;
    int32_t idx01 = (y1 * fb_width + x0) * 3;
    int32_t idx11 = (y1 * fb_width + x1) * 3;

    uint8_t c00_r = frame_buffer[idx00 + 0];
    uint8_t c00_g = frame_buffer[idx00 + 1];
    uint8_t c00_b = frame_buffer[idx00 + 2];

    uint8_t c10_r = frame_buffer[idx10 + 0];
    uint8_t c10_g = frame_buffer[idx10 + 1];
    uint8_t c10_b = frame_buffer[idx10 + 2];

    uint8_t c01_r = frame_buffer[idx01 + 0];
    uint8_t c01_g = frame_buffer[idx01 + 1];
    uint8_t c01_b = frame_buffer[idx01 + 2];

    uint8_t c11_r = frame_buffer[idx11 + 0];
    uint8_t c11_g = frame_buffer[idx11 + 1];
    uint8_t c11_b = frame_buffer[idx11 + 2];

    float r0 = (float)c00_r + dx * (float)(c10_r - c00_r);
    float g0 = (float)c00_g + dx * (float)(c10_g - c00_g);
    float b0 = (float)c00_b + dx * (float)(c10_b - c00_b);

    float r1 = (float)c01_r + dx * (float)(c11_r - c01_r);
    float g1 = (float)c01_g + dx * (float)(c11_g - c01_g);
    float b1 = (float)c01_b + dx * (float)(c11_b - c01_b);

    *r = r0 + dy * (r1 - r0);
    *g = g0 + dy * (g1 - g0);
    *b = b0 + dy * (b1 - b0);
}


/**
 * 将正方形平面图像映射为圆形鱼眼图像
 * @param {Uint8ClampedArray} inputBuffer - 输入图像数据 (RGB888)
 * @param {Uint8ClampedArray} outputBuffer - 输出图像数据 (RGB888)
 * @param {number} input_width - 图像宽度 (假设宽高相等)
 * @param {number} input_height - 图像高度 (假设宽高相等)
 * @param {number} fovFactor - 鱼眼视场角因子 (0.5 ~ 1.56即pi/2), 越大畸变越强
 */
void to_fisheye(uint8_t *inputBuffer, uint8_t *outputBuffer, uint32_t input_width, uint32_t input_height, float fovFactor) {
    float centerX = input_width / 2.0f;
    float centerY = input_height / 2.0f;
    float maxRadius = input_width / 2.0f;
    
    // 输入图像的最大半径 (半对角线), 确保正方形角点映射到圆边缘
    float inputMaxRadius = maxRadius; // Math.sqrt(centerX * centerX + centerY * centerY);
    
    // 预计算 tan(fovFactor), 避免循环内重复计算
    float tanFov = tanf(fovFactor);
    
    // 遍历输出图像的每一个像素 (逆向映射)
    for (uint32_t y = 0; y < input_height; y++) {
        for (uint32_t x = 0; x < input_width; x++) {
            float dx = (float)(x - centerX);
            float dy = (float)(y - centerY);
            float rOut = sqrtf(dx * dx + dy * dy);
            
            // 如果在输出圆外，设为黑色
            if (rOut > maxRadius) {
                uint32_t idx = (y * input_width + x) * 3;
                outputBuffer[idx + 0] = 0;
                outputBuffer[idx + 1] = 64;
                outputBuffer[idx + 2] = 0;
                continue;
            }

            // 归一化半径 (0 ~ 1)
            float rNorm = rOut / maxRadius;

            // 计算角度
            float theta = atan2f(dy, dx);

            // 鱼眼映射核心公式 (正切投影)
            float rIn = 0.0f;
            if (fovFactor >= M_PI / 2.0f - 0.001f) {
                // 防止 tan 无穷大，退化处理
                rIn = rNorm * inputMaxRadius;
            }
            else {
                rIn = inputMaxRadius * tanf(rNorm * fovFactor) / tanFov;
            }

            // 计算输入图像的坐标
            uint32_t srcX = (uint32_t)(centerX + rIn * cosf(theta));
            uint32_t srcY = (uint32_t)(centerY + rIn * sinf(theta));

            // 双线性插值获取颜色
            float red = 0.0f;
            float green = 0.0f;
            float blue = 0.0f;
            bilinear_sample(inputBuffer, input_width, input_height, srcX, srcY, &red, &green, &blue);

            // 写入输出像素
            uint32_t outIdx = (y * input_width + x) * 3;
            outputBuffer[outIdx + 0] = MIN(255, roundf(red));
            outputBuffer[outIdx + 1] = MIN(255, roundf(green));
            outputBuffer[outIdx + 2] = MIN(255, roundf(blue));
        }
    }
}




// 欧拉角转换：避免pitch=90°时的环架锁问题
void transform_euler_angles(float pitch_in, float roll_in, float yaw_in, float *pitch_out, float *roll_out, float *yaw_out) {
    const float degtorad = M_PI / 180.0f;
    const float radtodeg = 180.0f / M_PI;

    float alpha = yaw_in;
    float beta = pitch_in;
    float gamma = roll_in;
    
    // 计算输入角度的三角函数值
    float cX = cosf(beta * degtorad);   // cos(beta)
    float cY = cosf(gamma * degtorad);    // cos(gamma)
    float cZ = cosf(alpha * degtorad);  // cos(alpha)
    float sX = sinf(beta * degtorad);   // sin(beta)
    float sY = sinf(gamma * degtorad);    // sin(gamma)
    float sZ = sinf(alpha * degtorad);  // sin(alpha)
    
    // 计算旋转矩阵元素
    float m11 = cZ * cY - sZ * sX * sY;
    float m12 = - cX * sZ;
    float m13 = cY * sZ * sX + cZ * sY;

    float m21 = cY * sZ + cZ * sX * sY;
    float m22 = cZ * cX;
    float m23 = sZ * sY - cZ * cY * sX;

    float m31 = - cX * sY;
    float m32 = sX;
    float m33 = cX * cY;

    (void)m11; (void)m12;

    // 计算 sy 判断是否奇异
    float sy = sqrtf(m13 * m13 + m23 * m23);
    
    float x, y, z;
    
    if (sy >= 1e-6f) {
        x = atan2f(m31, m32);
        y = atan2f(-m33, sy);
        z = atan2f(m23, m13);
    } else {
        x = atan2f(-m22, m21);
        y = atan2f(-m33, sy);
        z = 0.0f;
    }
    
    *pitch_out = y * radtodeg;
    *roll_out = x * radtodeg;
    *yaw_out = z * radtodeg;
}


/**
 * 将四元数转换为 ZXY 欧拉角（Tait-Bryan 角）
 * 
 * 坐标系约定:
 *   - yaw   (alpha): 绕 Z 轴方位角，范围 [0°, 360°)
 *   - pitch (beta):  绕 X 轴俯仰角，范围 [-90°, 90°]
 *   - roll  (gamma): 绕 Y 轴横滚角，范围 (-180°, 180°]
 *
 * 参数:
 *   q0,q1,q2,q3 - 输入四元数分量 (x, y, z, w)
 *   pitch, roll, yaw - 输出指针，结果以度为单位
 */
void quaternion_to_euler(float q0, float q1, float q2, float q3, float *pitch, float *roll, float *yaw) {
    // 映射到标准 x,y,z,w 以便阅读
    const float x = q1;
    const float y = q2;
    const float z = q3;
    const float w = q0;
    
    const float RAD2DEG = 180.0f / M_PI;
    const float epsilon = 1e-6f;
    
    // ── 第一步：四元数 → 3×3 旋转矩阵 (R = Rz(yaw)*Rx(pitch)*Ry(roll)) ──
    const float R00 = 1.0f - 2.0f * (y*y + z*z);
    const float R01 = 2.0f * (x*y - w*z);
    const float R10 = 2.0f * (x*y + w*z);
    const float R11 = 1.0f - 2.0f * (x*x + z*z);
    const float R20 = 2.0f * (x*z - w*y);
    const float R21 = 2.0f * (y*z + w*x);
    const float R22 = 1.0f - 2.0f * (x*x + y*y);
    
    // ── 第二步：从旋转矩阵提取 ZXY 欧拉角 ──────────────────────────────
    // R[2][1] = sin(pitch)
    float sin_pitch = R21;
    
    // 钳制到 [-1, 1]，防止浮点误差导致 asin 返回 NaN
    if (sin_pitch > 1.0f)  sin_pitch = 1.0f;
    if (sin_pitch < -1.0f) sin_pitch = -1.0f;
    
    float pitch_rad = asinf(sin_pitch);
    float cos_pitch = cosf(pitch_rad);
    
    float roll_rad, yaw_rad;
    
    if (fabsf(cos_pitch) > epsilon) {
        // 正常情况：无万向锁
        roll_rad = atan2f(-R20, R22);
        yaw_rad  = atan2f(-R01, R11);
    } else {
        // 万向锁：pitch ≈ ±90°，cos(pitch)≈0，yaw 与 roll 耦合
        // 约定 roll = 0，通过矩阵残余项恢复 yaw
        roll_rad = 0.0f;
        yaw_rad  = atan2f(R10, R00);
    }
    
    // ── 第三步：转换为角度并归一化 ────────────────────────────────────
    *pitch = pitch_rad * RAD2DEG;               // [-90, 90]
    *roll  = roll_rad  * RAD2DEG;               // (-180, 180]
    *yaw   = yaw_rad   * RAD2DEG;
}


// ===============================================================================
// 滤波器
// ===============================================================================

void star_burst_filter(Nano_GFX *gfx, float sun_screen_x, float sun_screen_y) {
    uint32_t width = gfx->width;
    uint32_t height = gfx->height;

    // 亮度阈值
    float threshold = 0.9;

    float cx = sun_screen_x;
    float cy = sun_screen_y;
    int32_t ix = (int32_t)(floorf(cx));
    int32_t iy = (int32_t)(floorf(cy));
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return;

    // 1. 读取原图中该像素的亮度
    uint8_t r8 = 0;
    uint8_t g8 = 0;
    uint8_t b8 = 0;
    gfx_get_pixel(gfx, ix, iy, &r8, &g8, &b8);
    float r = r8 / 255.f;
    float g = g8 / 255.f;
    float b = b8 / 255.f;
    float lum = get_luminance(r, g, b);

    if (lum <= threshold) return; // 不够亮，不绘制星芒

    // 2. 计算星芒颜色和强度
    float factor = MIN(1.0f, (lum - threshold) * 5.0f);
    float sr = r * factor * 255.0f;
    float sg = g * factor * 255.0f;
    float sb = b * factor * 255.0f;

    // 3. 绘制星芒（沿4个方向双向延伸）
    int32_t blurLength = LINGLONG_STAR_BURST_RADIUS;
    float decay = LINGLONG_STAR_BURST_DECAY;

    for (int32_t k = 0; k < 4; k++) {
        float dx = 0.0f;
        float dy = 0.0f;

        switch (k) {
            case 0: dx = 1.0f; dy = 0.0f; break;
            case 1: dx = 0.0f; dy = 1.0f; break;
            case 2: dx = 1.0f; dy = 1.0f; break;
            case 3: dx = 1.0f; dy = -1.0f; break;
            default: break;
        }

        float len = sqrtf(dx*dx + dy*dy);
        float ndx = dx / len;
        float ndy = dy / len;

        for (int32_t d = 1; d <= blurLength; d++) {
            float alpha = powf(decay, (float)d);
            if (alpha < 0.01f) break;

            // 正向
            int32_t px = (int32_t)floorf(cx + ndx * d);
            int32_t py = (int32_t)floorf(cy + ndy * d);
            if (px >= 0 && px < width && py >= 0 && py < height) {
                gfx_add_pixel(gfx, px, py, (uint8_t)(sr * alpha), (uint8_t)(sg * alpha), (uint8_t)(sb * alpha));
            }

            // 反向
            px = (int32_t)floorf(cx - ndx * d);
            py = (int32_t)floorf(cy - ndy * d);
            if (px >= 0 && px < width && py >= 0 && py < height) {
                gfx_add_pixel(gfx, px, py, (uint8_t)(sr * alpha), (uint8_t)(sg * alpha), (uint8_t)(sb * alpha));
            }
        }
    }
}



// ===============================================================================
// 绘制基本形状
// ===============================================================================


void draw_circle_outline(Nano_GFX *gfx, float cx, float cy, float radius, float line_weight, uint8_t red, uint8_t green, uint8_t blue) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    if (radius <= 0.0f || line_weight <= 0.0f) return;

    uint8_t r = MAX(0, MIN(255, red));
    uint8_t g = MAX(0, MIN(255, green));
    uint8_t b = MAX(0, MIN(255, blue));

    // 计算内外半径（考虑线宽居中）
    float halfWeight = line_weight / 2.0f;
    float outerR = radius + halfWeight;
    float innerR = MAX(0, radius - halfWeight);

    float outerRSq = outerR * outerR;
    float innerRSq = innerR * innerR;

    // 包围盒（含线宽）
    int32_t xMin = MAX(0, (int32_t)floorf(cx - outerR));
    int32_t xMax = MIN(fb_width - 1, (int32_t)ceilf(cx + outerR));
    int32_t yMin = MAX(0, (int32_t)floorf(cy - outerR));
    int32_t yMax = MIN(fb_height - 1, (int32_t)ceilf(cy + outerR));

    for (int32_t y = yMin; y <= yMax; y++) {
        for (int32_t x = xMin; x <= xMax; x++) {
            float dx = x - cx;
            float dy = y - cy;
            float distSq = dx * dx + dy * dy;

            // 判断是否在环形区域内（包含外边界，排除内边界）
            if (distSq < outerRSq && distSq >= innerRSq) {
                gfx_add_pixel(gfx, x, y, r, g, b);
            }
        }
    }
}




static inline uint8_t get_pixel_channel(const uint8_t *tex, int32_t width, int32_t height, int32_t x, int32_t y, int32_t c) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0;
    return tex[((y * width + x) * 3) + c];
}

// 绘制地景
void draw_horizon(
    Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection,
    float view_height, float sun_alt, int32_t landscape_index,
    int32_t enable_atmosphere_scattering, uint8_t atmo_r, uint8_t atmo_g, uint8_t atmo_b
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    float margin = LINGLONG_HORIZON_BLUR_MARGIN;
    // float k = MAX(0.2f, sinf(to_rad_float(sun_alt)) * 1.5f);
    float k = (1.0f / M_PI) * atanf((sun_alt - 6.0f) / 6.0f) + 0.5f + 0.2f;
    float max_scattering_depth = 0.8;
    float hx = 0.0f;
    float hy = 0.0f;
    float hz = 0.0f;
    for (int32_t y = 0; y < fb_height; y++) {
        for (int32_t x = 0; x < fb_width; x++) {
            fisheye_unproject(x, y, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &hx, &hy, &hz);
            if (hz < 0) {
                if (landscape_index == 0) {
                    gfx_set_pixel(gfx, x, y, 0, 0, 0);
                }
                else {
                    // 反解此处的xyz坐标，转为地平坐标
                    float alt = 0.0f;
                    float azi = 0.0f;
                    float rr = 0.0f;
                    xyz_to_horizontal(hx, hy, hz, &azi, &alt, &rr);

                    // 再转为地面贴图上的xy坐标
                    float tx = 0.0f;
                    float ty = 0.0f;
                    horizontal_to_screen_xy(-azi, -alt, landscape_texture_width/2, landscape_texture_width/2, landscape_texture_height/2, view_height, &tx, &ty); // R略小于地面贴图的半径，注意修改其中的center_x/y

                    uint8_t r = get_pixel_channel(landscape_texture_rgb, landscape_texture_width, landscape_texture_height, floorf(tx), floorf(ty), 0);
                    uint8_t g = get_pixel_channel(landscape_texture_rgb, landscape_texture_width, landscape_texture_height, floorf(tx), floorf(ty), 1);
                    uint8_t b = get_pixel_channel(landscape_texture_rgb, landscape_texture_width, landscape_texture_height, floorf(tx), floorf(ty), 2);

                    // 模拟近地大气散射
                    float scattered_r = 1.0f;
                    float scattered_g = 1.0f;
                    float scattered_b = 1.0f;
                    if (enable_atmosphere_scattering) {
                        float depth = sqrtf(hx * hx + hy * hy) / rr;
                        depth = MIN(1.0, powf(depth, 2.0f)) * max_scattering_depth;
                        scattered_r = (1 - depth) * (float)r + (float)atmo_r * depth;
                        scattered_g = (1 - depth) * (float)g + (float)atmo_g * depth;
                        scattered_b = (1 - depth) * (float)b + (float)atmo_b * depth;
                    }
                    else {
                        scattered_r = (float)r;
                        scattered_g = (float)g;
                        scattered_b = (float)b;
                    }

                    r = (uint8_t)MIN(255.0f, (scattered_r * k));
                    g = (uint8_t)MIN(255.0f, (scattered_g * k));
                    b = (uint8_t)MIN(255.0f, (scattered_b * k));

                    gfx_set_pixel(gfx, x, y, r, g, b);
                }
            }
            else if (enable_atmosphere_scattering && hz <= margin && hz >= 0.0f) {
                float cr = (float)atmo_r * k;
                float cg = (float)atmo_g * k;
                float cb = (float)atmo_b * k;
                float t = powf(hz / margin, 0.5f);

                uint8_t bg_r8 = 0;
                uint8_t bg_g8 = 0;
                uint8_t bg_b8 = 0;
                gfx_get_pixel(gfx, x, y, &bg_r8, &bg_g8, &bg_b8);
                gfx_set_pixel(gfx, x, y,
                    (uint8_t)MIN(255.0f,((1.0f - t) * cr + t * (float)bg_r8)),
                    (uint8_t)MIN(255.0f,((1.0f - t) * cg + t * (float)bg_g8)),
                    (uint8_t)MIN(255.0f,((1.0f - t) * cb + t * (float)bg_b8))
                );
            }
        }
    }
}



// 刷新地景
//   landscape_source - 地景纹理（可能是鱼眼贴图、平面贴图、平面的算法生成地景等）
//   is_flat - 0:贴图本身就是鱼眼，无需转鱼眼  1-贴图是平面的，需要使用fov参数转鱼眼
void update_landscape(uint8_t *landscape_source, uint32_t width, uint32_t height, int32_t is_flat, float fov) {
    if (is_flat) {
#if LINGLONG_ENABLE_DYNAMIC_LANDSCAPE
        landscape_texture_width = width;
        landscape_texture_height = height;
        to_fisheye(landscape_source, landscape_buffer_rgb, width, height, fov);
        landscape_texture_rgb = landscape_buffer_rgb;
#else
        landscape_texture_width = width;
        landscape_texture_height = height;
        landscape_texture_rgb = landscape_source;
#endif
    }
    else {
        landscape_texture_width = width;
        landscape_texture_height = height;
        landscape_texture_rgb = landscape_source;
    }
}



// ===============================================================================
// 绘制坐标系
// ===============================================================================

/**
 * 绘制赤道坐标系下的子午圈或纬度圈（投影到地平屏幕）
 * @param is_meridian - 1: 子午圈（固定RA）；0: 纬度圈（固定Dec）
 * @param ra_hours - 子午圈的赤经（小时），仅当 is_meridian=1 时有效
 * @param dec_deg - 纬度圈的赤纬（度），仅当 is_meridian=0 时有效
 * @param line_weight - 线宽（像素），建议 ≥1
 * @param colorR, colorG, colorB - RGB 颜色分量 [0-255]
 * @param year, month, day, hour, minute, second, timezone, longitude, latitude - 观测参数
 */
void draw_celestial_circle(
    Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection,
    int32_t is_meridian, float ra_hours, float dec_deg,
    int32_t line_weight, uint8_t colorR, uint8_t colorG, uint8_t colorB,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude
) {
    const int32_t POINTS = LINGLONG_CELESTIAL_CIRCLE_POINTS;
    const double eps = 0.5;

    float points_x[LINGLONG_CELESTIAL_CIRCLE_POINTS+1];
    float points_y[LINGLONG_CELESTIAL_CIRCLE_POINTS+1];
    int valid[LINGLONG_CELESTIAL_CIRCLE_POINTS+1];

    // 子午圈：固定 RA，遍历 Dec ∈ [-90°, +90°]
    if (is_meridian) {
        if (ra_hours < 0.0f || ra_hours > 24.0f) return;
        float ra_deg = fmodf(ra_hours, 24.0f) * 15.0f;
        double alt = 0.0;
        double azi = 0.0;
        for (int32_t i = 0; i <= POINTS; i++) {
            float dec = -90.0f + (180.0f * (float)i / (float)POINTS);
            equatorial_to_horizontal((double)ra_deg, (double)dec, year, month, day, hour, minute, second, timezone, longitude, latitude, &azi, &alt);
            if (alt > eps) {
                float x = 0.0f;
                float y = 0.0f;
                fisheye_project(azi, alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &x, &y);
                points_x[i] = x;
                points_y[i] = y;
                valid[i] = 1;
            } else {
                valid[i] = 0;
            }
        }

        // 连接可见段
        for (int32_t i = 0; i < POINTS; i++) {
            if (valid[i] && valid[i + 1]) {
                gfx_draw_line_anti_aliasing(gfx, points_x[i], points_y[i], points_x[i + 1], points_y[i + 1], line_weight, colorR, colorG, colorB, 3);
            }
        }
        // 首尾闭合
        if (valid[POINTS] && valid[0]) {
            gfx_draw_line_anti_aliasing(gfx, points_x[POINTS], points_y[POINTS], points_x[0], points_y[0], line_weight, colorR, colorG, colorB, 3);
        }
    }
    // 纬度圈：固定 Dec，遍历 RA ∈ [0h, 24h)
    else {
        if (dec_deg < -90.0f || dec_deg > 90.0f) return;
        double alt = 0.0;
        double azi = 0.0;
        for (int32_t i = 0; i <= POINTS; i++) {
            float ra_h = 24.0f * (float)i / (float)POINTS;
            float ra_deg = fmodf(ra_h, 24.0f) * 15.0f;
            equatorial_to_horizontal((double)ra_deg, (double)dec_deg, year, month, day, hour, minute, second, timezone, longitude, latitude, &azi, &alt);
            if (alt > eps) {
                float x = 0.0f;
                float y = 0.0f;
                fisheye_project(azi, alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &x, &y);
                points_x[i] = x;
                points_y[i] = y;
                valid[i] = 1;
            } else {
                valid[i] = 0;
            }
        }

        // 连接可见段
        for (int32_t i = 0; i < POINTS; i++) {
            if (valid[i] && valid[i + 1]) {
                gfx_draw_line_anti_aliasing(gfx, points_x[i], points_y[i], points_x[i + 1], points_y[i + 1], line_weight, colorR, colorG, colorB, 3);
            }
        }
        // 首尾闭合
        if (valid[POINTS] && valid[0]) {
            gfx_draw_line_anti_aliasing(gfx, points_x[POINTS], points_y[POINTS], points_x[0], points_y[0], line_weight, colorR, colorG, colorB, 3);
        }
    }
}


void draw_ecliptic_circle(
    Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection,
    int32_t line_weight, uint8_t colorR, uint8_t colorG, uint8_t colorB,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude
) {
    float x_prev = 0.0f;
    float y_prev = 0.0f;
    double alt_prev = 0.0f;
    float x_0 = 0.0f;
    float y_0 = 0.0f;
    double alt_0 = 0.0f;

    double eps = 0.5; // 防止两个端点处的点连成弓弦

    double RA = 0.0;
    double Dec = 0.0;
    double alt = 0.0;
    double azi = 0.0;
    for (int32_t i = 0; i <= LINGLONG_ECLIPTIC_CIRCLE_POINTS; i++) {
        // 计算黄道上各点的赤经赤纬
        float lambda = (360.0f * (float)i / (float)LINGLONG_ECLIPTIC_CIRCLE_POINTS);
        ecliptic_to_equatorial(lambda, 0, &RA, &Dec);
        // 将其转为地平坐标
        equatorial_to_horizontal(RA, Dec, year, month, day, hour, minute, second, timezone, longitude, latitude, &azi, &alt);

        if (alt > 0.0) {
            float x = 0.0f;
            float y = 0.0f;
            fisheye_project(azi, alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &x, &y);

            if (i == 0) {
                x_0 = x;
                y_0 = y;
                alt_0 = alt;
            }
            
            if (alt_prev > eps && alt > eps) {
                gfx_draw_line_anti_aliasing(gfx, x_prev, y_prev, x, y, line_weight, colorR, colorG, colorB, 3);
            }

            x_prev = x;
            y_prev = y;
            alt_prev = alt;
        }
    }
    if (alt_0 > eps && alt > eps) {
        gfx_draw_line_anti_aliasing(gfx, x_prev, y_prev, x_0, y_0, line_weight, colorR, colorG, colorB, 3);
    }

}

// 绘制地平等仰角圈（地平经度圈/高度圈）
void draw_horizontal_altitude_circle(
    Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection,
    float altitude_deg,
    int32_t line_weight, uint8_t colorR, uint8_t colorG, uint8_t colorB
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    const int32_t POINTS = LINGLONG_HORIZONTAL_CIRCLE_POINTS;
    const float R = sky_radius;

    float points_x[LINGLONG_HORIZONTAL_CIRCLE_POINTS+1];
    float points_y[LINGLONG_HORIZONTAL_CIRCLE_POINTS+1];
    int valid[LINGLONG_HORIZONTAL_CIRCLE_POINTS+1];

    // 固定仰角，遍历方位角 0°到360°
    for (int32_t i = 0; i <= POINTS; i++) {
        float azimuth_deg = (360.0f * (float)i / (float)POINTS);
        float x = 0.0f;
        float y = 0.0f;
        fisheye_project(azimuth_deg, altitude_deg, R, center_x, center_y,
                       view_alt, view_azi, view_roll, f, projection, &x, &y);
        // 检查点是否有效（在画面范围内）
        if (x >= 0 && x < fb_width && y >= 0 && y < fb_height) {
            points_x[i] = x;
            points_y[i] = y;
            valid[i] = 1;
        } else {
            valid[i] = 0;
        }
    }

    // 使用抗锯齿 draw_line 连接可见段
    for (int32_t i = 0; i < POINTS; i++) {
        if (valid[i] && valid[i + 1]) {
            gfx_draw_line_anti_aliasing(gfx, points_x[i], points_y[i], points_x[i + 1], points_y[i + 1],
                      line_weight, colorR, colorG, colorB, 3);
        }
    }
}

// 绘制地平等方位角圈（地平经线圈/方位圈）
void draw_horizontal_azimuth_circle(
    Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection,
    float azimuth_deg,
    int32_t line_weight, uint8_t colorR, uint8_t colorG, uint8_t colorB
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    const int32_t POINTS = LINGLONG_HORIZONTAL_CIRCLE_POINTS;
    const float R = sky_radius;

    // i 从 -POINTS 到 POINTS，共 2*POINTS+1 个点
    float points_x[2*LINGLONG_HORIZONTAL_CIRCLE_POINTS+1];
    float points_y[2*LINGLONG_HORIZONTAL_CIRCLE_POINTS+1];
    int valid[2*LINGLONG_HORIZONTAL_CIRCLE_POINTS+1];

    // 固定方位角，遍历仰角从地平线（0°）到天顶（90°）
    for (int32_t i = -POINTS; i <= POINTS; i++) {
        float altitude_deg = (90.0f * (float)i / (float)POINTS);
        float x = 0.0f;
        float y = 0.0f;
        fisheye_project(azimuth_deg, altitude_deg, R, center_x, center_y,
                       view_alt, view_azi, view_roll, f, projection, &x, &y);
        int32_t idx = i + POINTS;
        // 检查点是否有效
        if (x >= 0 && x < fb_width && y >= 0 && y < fb_height) {
            points_x[idx] = x;
            points_y[idx] = y;
            valid[idx] = 1;
        } else {
            valid[idx] = 0;
        }
    }

    // 使用抗锯齿 draw_line 连接可见段
    for (int32_t i = 0; i < 2 * POINTS; i++) {
        if (valid[i] && valid[i + 1]) {
            gfx_draw_line_anti_aliasing(gfx, points_x[i], points_y[i], points_x[i + 1], points_y[i + 1],
                      line_weight, colorR, colorG, colorB, 3);
        }
    }
}




// ===============================================================================
// 绘制天体
// ===============================================================================

void render_sun(Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float sun_proj_x, float sun_proj_y, float sun_altitude_deg
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    float sun_radius = sky_radius * 0.02;

    // 太阳半径（随高度变化）：地平线附近略大
    float radius = MAX(LINGLONG_SUN_RADIUS_MIN, sun_radius) * (1.0f + 0.2f * MAX(0.0f, 1.0f - sun_altitude_deg / 10.0f));

    // 光晕半径
    int32_t glow_radius = LINGLONG_SUN_GLOW_RADIUS;

    // 太阳颜色：随高度从红→黄→白
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    // 接近地平线：深红到橙
    if (sun_altitude_deg < 5.0f) {
        float t = MIN(1.0f, MAX(0.0f, sun_altitude_deg) / 5.0f);
        r = 255;
        g = (uint8_t)roundf(160.0f * t);
        b = (uint8_t)roundf(40.0f * t);
    }
    // 低空：橙黄
    else if (sun_altitude_deg < 15.0f) {
        float t = (sun_altitude_deg - 5.0f) / 10.0f;
        r = 255;
        g = (uint8_t)roundf(160.0f + 95 * t);
        b = (uint8_t)roundf(40.0f + 60 * t);
    }
    // 中天：白色
    else {
        r = 255;
        g = 255;
        b = 255;
    }

    // 抗锯齿边缘宽度
    float edgeSmoothWidth = 1.0f;

    // 绘制太阳本体（带软边缘）
    for (int32_t dy = -glow_radius; dy <= glow_radius; dy++) {
        for (int32_t dx = -glow_radius; dx <= glow_radius; dx++) {
            float dist = sqrtf(dx * dx + dy * dy);
            int32_t px = (int32_t)floorf(sun_proj_x + (float)dx);
            int32_t py = (int32_t)floorf(sun_proj_y + (float)dy);
            if (px < 0 || px >= fb_width || py < 0 || py >= fb_height) continue;

            // 太阳光晕（指数衰减）
            float glow = 0.0f;
            if (dist > radius && dist <= (float)glow_radius) {
                // 光晕强度随距离衰减
                glow = expf(-(dist / (float)glow_radius) * 3.0f);
            }

            // 太阳本体（带抗锯齿）
            float diskWeight = 0.0f;
            if (dist <= radius + edgeSmoothWidth) {
                if (dist <= radius - edgeSmoothWidth) {
                    diskWeight = 1.0f;
                }
                else if (dist < radius + edgeSmoothWidth) {
                    float t = (radius + edgeSmoothWidth - dist) / (2.0f * edgeSmoothWidth);
                    diskWeight = MAX(0.0f, MIN(1.0f, t * t * (3.0f - 2.0f * t)));
                }
            }

            // 合并：中心最亮（diskWeight=1），向外过渡到光晕
            float totalR = diskWeight * r + glow * r;
            float totalG = diskWeight * g + glow * g;
            float totalB = diskWeight * b + glow * b;

            gfx_add_pixel(gfx, px, py, totalR, totalG, totalB);
        }
    }
}


// 基于球面几何晨昏线的月相绘制（带物理合理的软边缘）
void render_moon(Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    float lum = 0.8f;

    // 计算月球位置、月相、月球方向角
    double moon_azi = 0.0;
    double moon_alt = 0.0;
    double moon_RA = 0.0;
    double moon_Dec = 0.0;
    double jd = julian_day(year, month, day, hour, minute, second, timezone);
    calculate_lunar_equatorial_coordinates(jd, &moon_RA, &moon_Dec);

    equatorial_to_horizontal(
        moon_RA, moon_Dec, year, month, day, hour, minute, second, timezone,
        longitude, latitude,
        &moon_azi, &moon_alt
    );

    // 计算月球的屏幕投影坐标
    float moon_scr_x = 0.0f;
    float moon_scr_y = 0.0f;
    fisheye_project(moon_azi, moon_alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &moon_scr_x, &moon_scr_y);

    // 以下是计算月面贴图的北极（月球地理北极）相对于屏幕y轴的旋转角
    // 这一步是旋转月球地理北极，使其指向其所在的天球子午圈北向切线方向，为后面计算月球亮区旋转角作准备
    // 天球赤道坐标系经圈（子午圈）上，月亮所在位置往北2度的位置，在屏幕上的投影坐标
    double northdelta_azi = 0.0;
    double northdelta_alt = 0.0;
    float northdelta_scr_x = 0.0f;
    float northdelta_scr_y = 0.0f;
    equatorial_to_horizontal(
        moon_RA, moon_Dec + 2, year, month, day, hour, minute, second, timezone,
        longitude, latitude,
        &northdelta_azi, &northdelta_alt
    );
    fisheye_project(northdelta_azi, northdelta_alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &northdelta_scr_x, &northdelta_scr_y);

    float xa = moon_scr_x;
    float ya = moon_scr_y;
    float xb = northdelta_scr_x;
    float yb = northdelta_scr_y;
    // 计算月面贴图的北极（月球地理北极）相对于屏幕y轴的旋转角
    float tilt_rad = atan2f(xb-xa, -(yb-ya)); // 屏幕坐标系y向下为正
    float tilt_deg = normalize_angle_float(to_deg_float(tilt_rad));

    // 月面视圆盘的亮区拱点，在天球赤道坐标系背景上，相对于天北极方向向东的偏转角度
    float bright_pos_deg = (float)moon_bright_limb_pos_angle(year, month, day, hour, minute, second, timezone);
    bright_pos_deg = normalize_angle_float(bright_pos_deg);
    // 假定月面亮区两端永远是月球的地理南北极
    bright_pos_deg = normalize_angle_float(tilt_deg - (bright_pos_deg-90)); // NOTE 仰视东西颠倒；亮区拱点和亮区的尖尖相差90度

    // 月相
    float moon_i = (float)moon_phase(year, month, day, hour, minute, second, timezone);
    // float moon_k = (1 + cosf(moon_i)) / 2;
    float phase_deg = normalize_angle_float(moon_i - 90);
    float cosPhase = cosf(to_rad_float(phase_deg));
    float sinPhase = sinf(to_rad_float(phase_deg));
    float sunDirX = cosPhase * cosf(to_rad_float(bright_pos_deg));
    float sunDirY = cosPhase * sinf(to_rad_float(bright_pos_deg));
    float sunDirZ = sinPhase;

    float sunLen = sqrtf(sunDirX * sunDirX + sunDirY * sunDirY + sunDirZ * sunDirZ);
    float sunUnitX = (sunLen > 0.0f) ? (sunDirX / sunLen) : 1.0f;
    float sunUnitY = (sunLen > 0.0f) ? (sunDirY / sunLen) : 0.0f;
    float sunUnitZ = (sunLen > 0.0f) ? (sunDirZ / sunLen) : 0.0f;

    // 颜色与环境光
    // const litColor = [220, 220, 200];
    // const darkColor = [10, 10, 20];
    float ambient = 0.01f;

    // 半影角宽度（弧度）
    float PENUMBRA_HALF_ANGLE = 0.2f;

    // 抗锯齿过渡带宽度（像素）
    float edgeSmoothWidth = 1.0f;

    // 显示直径
    float moonRadius = MAX(LINGLONG_MOON_RADIUS_MIN, sky_radius * 0.08);

    float r2 = moonRadius * moonRadius;
    for (int32_t dy = -(int32_t)moonRadius; dy <= (int32_t)moonRadius; dy++) {
        for (int32_t dx = -(int32_t)moonRadius; dx <= (int32_t)moonRadius; dx++) {
            float distance = sqrtf((float)(dx * dx + dy * dy));
            if (distance > moonRadius + edgeSmoothWidth) continue;

            int32_t px = (int32_t)floorf(moon_scr_x + (float)dx);
            int32_t py = (int32_t)floorf(moon_scr_y + (float)dy);
            if (px < 0 || px >= fb_width || py < 0 || py >= fb_height) continue;

            // 计算抗锯齿权重（基于圆盘覆盖比例）
            float moonWeight = 1.0f;
            if (distance <= moonRadius - edgeSmoothWidth) {
                moonWeight = 1.0; // 完全在内部
            }
            else if (distance >= moonRadius + edgeSmoothWidth) {
                moonWeight = 0.0; // 完全在外部（应跳过，但保留以防边界误差）
                continue;
            }
            else {
                // 平滑过渡 [moonRadius - w, moonRadius + w]
                float t = (moonRadius + edgeSmoothWidth - distance) / (2.0f * edgeSmoothWidth);
                moonWeight = MAX(0.0f, MIN(1.0f, t)); // 线性插值，也可用更平滑：moonWeight = t * t * (3 - 2 * t);
            }

            if (moonWeight <= 0) continue;


            // 实际上，(dx, dy, dz) 在半径为 moonRadius 的球面上，所以单位向量为：
            float nx = (float)dx / moonRadius;
            float ny = (float)dy / moonRadius;
            float nz = sqrtf(MAX(0.0f, r2 - dx * dx - dy * dy)) / moonRadius;

            // 关键：计算该点与晨昏线（太阳方向与月球中心构成的大圆）的球面角距离
            // 点积 dot = cos(θ)，其中 θ 是表面点与太阳方向的夹角
            float dot = nx * sunUnitX + ny * sunUnitY + nz * sunUnitZ;
            float angle_from_terminator = acosf(MAX(-1.0f, MIN(1.0f, dot))) - M_PI_2; // ∈ [-π/2, π/2]

            // 软边缘判断
            float intensity = 1.0f;
            if (angle_from_terminator >= PENUMBRA_HALF_ANGLE) {
                intensity = 1.0; // 完全照亮
            }
            else if (angle_from_terminator <= -PENUMBRA_HALF_ANGLE) {
                intensity = ambient; // 完全背光
            }
            else {
                // 映射到 [0,1]
                float t = (angle_from_terminator + PENUMBRA_HALF_ANGLE) / (2.0f * PENUMBRA_HALF_ANGLE);
                float smoothT = t * t * (3.0f - 2.0f * t); // smoothstep
                intensity = ambient + (1.0f - ambient) * smoothT;
            }

            // 调整全局亮度
            intensity *= lum;

            // 纹理映射：先绕 Z 轴旋转 -alpha，再映射到经纬度
            float u_norm = (float)dx / moonRadius;   // ∈ [-1, 1]
            float v_norm = (float)dy / moonRadius;   // ∈ [-1, 1]

            // 绕原点旋转 -alpha（使贴图随光照方向对齐）
            float cosA = cosf(-to_rad_float(tilt_deg));
            float sinA = sinf(-to_rad_float(tilt_deg));
            float u_rot = u_norm * cosA - v_norm * sinA;
            float v_rot = u_norm * sinA + v_norm * cosA;

            // 映射到 [0,1] 贴图空间
            float u = (u_rot + 1.0f) * 0.5f; // [-1,1] → [0,1]
            float v = (v_rot + 1.0f) * 0.5f;

            // 可选：限制在 [0,1] 内（避免边缘采样异常）
            if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
                // 可选择 clamp 或 skip；这里 clamp 更安全
                u = MAX(0.0f, MIN(1.0f, u));
                v = MAX(0.0f, MIN(1.0f, v));
            }

            // 双线性采样
            float tx = u * (float)moon_texture_width - 0.5f;
            float ty = v * (float)moon_texture_height - 0.5f;

            float texR = 0.0f;
            float texG = 0.0f;
            float texB = 0.0f;

            bilinear_sample((uint8_t*)moon_texture_rgb, moon_texture_width, moon_texture_height, tx, ty, &texR, &texG, &texB);

            // 应用光照强度
            float r = roundf(texR * intensity);
            float g = roundf(texG * intensity);
            float b = roundf(texB * intensity);

            gfx_add_pixel(gfx, px, py, moonWeight * r, moonWeight * g, moonWeight * b);
        }
    }
}


void draw_star(Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float sx, float sy, float magnitude, float radius, uint8_t red, uint8_t green, uint8_t blue
) {
    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    if (sx < 0 || sx >= fb_width || sy < 0 || sy >= fb_height) {
        return;
    }

    int32_t maxGlowRadius = LINGLONG_STAR_GLOW_RADIUS; // 光晕最大半径（像素）

    float starLumBase = magnitude_to_relative_luminance(magnitude);

    // 获取背景亮度用于对比度抑制
    uint8_t bg_r8 = 0;
    uint8_t bg_g8 = 0;
    uint8_t bg_b8 = 0;
    gfx_get_pixel(gfx, (int32_t)sx, (int32_t)sy, &bg_r8, &bg_g8, &bg_b8);

    float bgR = bg_r8 / 255.0f;
    float bgG = bg_g8 / 255.0f;
    float bgB = bg_b8 / 255.0f;
    float bgLum = get_luminance(bgR, bgG, bgB);

    // 遍历光晕区域（正方形包围圆）
    (void)maxGlowRadius;
    int32_t R = 0; // (int32_t)radius + maxGlowRadius;
    for (int32_t dy = -R; dy <= R; dy++) {
        for (int32_t dx = -R; dx <= R; dx++) {
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist > (float)R) continue;

            int32_t px = (int32_t)roundf(sx + (float)dx);
            int32_t py = (int32_t)roundf(sy + (float)dy);
            if (px < 0 || px >= fb_width || py < 0 || py >= fb_height) continue;

            float starLum = 0.0f;
            if (dist <= radius) {
                starLum = starLumBase;
            }
            // 光晕衰减
            else if (dist > radius) {
                starLum = starLumBase * expf(-dist * 1.0f);
            }

            // 对比度抑制：白天背景亮时，星星和光晕都应被压制
            float contrast = starLum / (bgLum + 0.001f);
            float visibility = 1.0f;
            if (contrast < 1.0f) {
                // visibility = Math.pow(contrast, 3);
                visibility = contrast * contrast * contrast;
            }
            starLum *= visibility;

            // 转为 0–255 范围
            uint8_t r = MIN(255, (uint8_t)floorf(starLum * (float)red));
            uint8_t g = MIN(255, (uint8_t)floorf(starLum * (float)green));
            uint8_t b = MIN(255, (uint8_t)floorf(starLum * (float)blue));

            // 叠加到 RGB（保持白色光晕，可改为彩色）
            gfx_add_pixel(gfx, px, py, r, g, b);
        }
    }
}



// ===============================================================================
// 散射
// ===============================================================================

// 计算大气散射强度
void scatter_model_1(
    float ray_x, float ray_y, float ray_z, float sun_x, float sun_y, float sun_z,
    float *red, float *green, float *blue,
    int32_t enable_opt_lut
) {

    // 视线向量归一化
    float ray_length = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    if (ray_length == 0.0f) {
        *red = 0;
        *green = 0;
        *blue = 0;
        return;
    }
    float ray_norm_x = ray_x / ray_length;
    float ray_norm_y = ray_y / ray_length;
    float ray_norm_z = ray_z / ray_length;


    // 太阳方向归一化
    float sun_vec_radius = sqrtf(sun_x * sun_x + sun_y * sun_y + sun_z * sun_z);
    if (sun_vec_radius == 0.0f) {
        *red = 0;
        *green = 255.0;
        *blue = 0;
        return;
    }
    float sun_norm_x = sun_x / sun_vec_radius;
    float sun_norm_y = sun_y / sun_vec_radius;
    float sun_norm_z = sun_z / sun_vec_radius;


    // 太阳光入射路径：计算仰角、天顶角、大气光学质量
    float sun_altitude = asinf(MAX(-1.0f, MIN(1.0f, sun_norm_z))); // [-π/2, π/2]
    float sun_altitude_deg = sun_altitude * 180.0f / M_PI;
    float sun_zenith = M_PI_2 - sun_altitude; // [0, π]
    float sun_zenith_deg = 90.0f - sun_altitude_deg;
    float sun_air_mass = 0.0f;
    // Kasten-Young, Ref. https://kexue.fm/archives/396
    if (sun_altitude_deg >= 0.0f) {
        sun_air_mass = 1.0f / (cosf(sun_zenith) + 0.50572f * powf(96.07995f - sun_zenith_deg, -1.6364f));
    }
    else {
        sun_air_mass = 40.0f;
    }


    // 观察者视线的仰角、天顶角、大气光学质量
    float view_zenith = acosf(MAX(0.0f, MIN(1.0f, ray_norm_z)));
    float view_zenith_deg = view_zenith * 180.0f / M_PI;
    float view_altitude_deg = 90.0f - view_zenith_deg;
    float view_air_mass = 0.0f;
    // Kasten-Young
    if (view_altitude_deg >= 0.0f) {
        view_air_mass = 1.0f / (cosf(view_zenith) + 0.50572f * powf(96.07995f - view_zenith_deg, -1.6364f));
    }
    else {
        view_air_mass = 40.0f;
    }


    // 观察方向与太阳方向夹角余弦
    float cos_theta = ray_norm_x * sun_norm_x + ray_norm_y * sun_norm_y + ray_norm_z * sun_norm_z;


    // 米氏散射相函数
    const float MIE_G = 0.9f;
    float mie_phase = powf(1.0f + MIE_G * MIE_G - 2.0f * MIE_G * cos_theta, -1.5f);


    // 瑞利散射相函数
    float rayleigh_phase = (1.0f + cos_theta * cos_theta) * 0.75f;


    // 计算视线方向的大气密度系数：直觉上来看，密度越大，对散射的贡献越大
    float rz = MAX(0.01, ray_norm_z); // 避免除零
    const float ATMOSPHERE_HEIGHT = 8.5f;
    float density_factor = expf(-rz / ATMOSPHERE_HEIGHT);


    // 瑞利散射系数：分波长（RGB通道）计算
    const float RAYLEIGH_BETA = 0.3f;
    const float RAYLEIGH_WAVELENGTH_FACTORS[3] = { 680, 550, 450 };
    // rayleigh_beta_at_wavelength[i] = RAYLEIGH_BETA * Math.pow(rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[i], 4);
    float rayleigh_ref_wavelength = RAYLEIGH_WAVELENGTH_FACTORS[1]; // G分量作为参考波长
    float wl_0 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[0]; // R
    float wl_1 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[1]; // G
    float wl_2 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[2]; // B

    float rayleigh_beta_0 = RAYLEIGH_BETA * wl_0 * wl_0 * wl_0 * wl_0; // R
    float rayleigh_beta_1 = RAYLEIGH_BETA * wl_1 * wl_1 * wl_1 * wl_1; // G
    float rayleigh_beta_2 = RAYLEIGH_BETA * wl_2 * wl_2 * wl_2 * wl_2; // B


    // 米氏散射系数：地平线附近，气溶胶浓度剧增（用viewAirMass模拟），米氏散射贡献显著上升
    const float MIE_BETA_BASE = 0.001f;
    const float MIE_BETA_MAX = 0.1f;
    float mie_beta = MIN(MIE_BETA_MAX, MIE_BETA_BASE * view_air_mass);


    // 计算每个通道的散射强度
    
    float mie_contrib = mie_beta * mie_phase;

    float rayleigh_contrib_0 = rayleigh_beta_0 * rayleigh_phase;
    float rayleigh_contrib_1 = rayleigh_beta_1 * rayleigh_phase;
    float rayleigh_contrib_2 = rayleigh_beta_2 * rayleigh_phase;

    float attn_scale = (sun_altitude_deg >= 0) ? (expf(-powf((sun_altitude_deg / 20.0f), 4.0f)) + 0.01f) : (1.0f);

    float attn_0 = expf(-(rayleigh_beta_0 + mie_beta) * view_air_mass * attn_scale);
    float attn_1 = expf(-(rayleigh_beta_1 + mie_beta) * view_air_mass * attn_scale);
    float attn_2 = expf(-(rayleigh_beta_2 + mie_beta) * view_air_mass * attn_scale);

    // 臭氧吸收
    const float OZONE_ABSORPTION[3] = { 0.005,  0.040,  0.025 };
    float ozone_transmittance_0 = expf(-OZONE_ABSORPTION[0] * sun_air_mass * 0.8);
    float ozone_transmittance_1 = expf(-OZONE_ABSORPTION[1] * sun_air_mass * 0.8);
    float ozone_transmittance_2 = expf(-OZONE_ABSORPTION[2] * sun_air_mass * 0.8);

    float scattered_0 = (rayleigh_contrib_0 + mie_contrib) * attn_0 * ozone_transmittance_0 * density_factor;
    float scattered_1 = (rayleigh_contrib_1 + mie_contrib) * attn_1 * ozone_transmittance_1 * density_factor;
    float scattered_2 = (rayleigh_contrib_2 + mie_contrib) * attn_2 * ozone_transmittance_2 * density_factor;

    // 太阳落到地平线以下时，进一步衰减散射光
    float night_attn = (sun_altitude_deg >= 0) ? (1.0f) : MAX(0.0f, expf(0.016f * sun_altitude_deg));
    scattered_0 *= night_attn;
    scattered_1 *= night_attn;
    scattered_2 *= night_attn;

    // 全局增益、限幅、输出
    const float global_gain = 2.0;
    *red   = MIN(1.0f, scattered_0 * global_gain);
    *green = MIN(1.0f, scattered_1 * global_gain);
    *blue  = MIN(1.0f, scattered_2 * global_gain);
}


// 计算大气散射强度（模型2：Nishita模型）
// Ref. https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/simulating-sky/simulating-colors-of-the-sky.html
void scatter_model_2(
    float ray_x, float ray_y, float ray_z, float sun_x, float sun_y, float sun_z,
    float *red, float *green, float *blue,
    int32_t enable_opt_lut /* dummy */
) {

    // 视线向量归一化
    float ray_length = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    if (ray_length == 0.0f) {
        *red = 0;
        *green = 0;
        *blue = 0;
        return;
    }
    float ray_norm_x = ray_x / ray_length;
    float ray_norm_y = ray_y / ray_length;
    float ray_norm_z = ray_z / ray_length;


    // 太阳方向归一化
    float sun_vec_radius = sqrtf(sun_x * sun_x + sun_y * sun_y + sun_z * sun_z);
    if (sun_vec_radius == 0.0f) {
        *red = 0;
        *green = 255.0;
        *blue = 0;
        return;
    }
    float sun_norm_x = sun_x / sun_vec_radius;
    float sun_norm_y = sun_y / sun_vec_radius;
    float sun_norm_z = sun_z / sun_vec_radius;

    // 太阳仰角
    float sun_altitude = asinf(MAX(-1.0f, MIN(1.0f, sun_norm_z))); // [-π/2, π/2]
    float sun_altitude_deg = sun_altitude * 180.0f / M_PI;

    // 观察者视线仰角
    float view_zenith = acosf(MAX(0.0f, MIN(1.0f, ray_norm_z)));
    float view_zenith_deg = view_zenith * 180.0f / M_PI;
    float view_altitude_deg = 90.0f - view_zenith_deg;

    // 观察方向与太阳方向夹角余弦
    float cos_theta = ray_norm_x * sun_norm_x + ray_norm_y * sun_norm_y + ray_norm_z * sun_norm_z;

    // 米氏散射相函数
    const float g = 0.76f;
    float mie_phase = 3.0f / (8.0f * M_PI) * ((1.0f - g*g) * (1.0f + cos_theta * cos_theta)) / ((2.0f + g*g) * powf(1.0f + g*g - 2.0f * g * cos_theta, 1.5f));

    // 瑞利散射相函数
    float rayleigh_phase = (1.0f + cos_theta * cos_theta) * 0.75f;

    // 瑞利散射系数：分波长（RGB通道）计算
    const float RAYLEIGH_WAVELENGTH_FACTORS[3] = { 680, 550, 450 };
    const float rayleigh_beta_base = 0.04f;
    // rayleigh_beta_at_wavelength[i] = rayleigh_beta_base * Math.pow(rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[i], 4);
    float rayleigh_ref_wavelength = RAYLEIGH_WAVELENGTH_FACTORS[1]; // G分量作为参考波长
    float wl_0 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[0]; // R
    float wl_1 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[1]; // G
    float wl_2 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[2]; // B

    float rayleigh_beta_0 = rayleigh_beta_base * wl_0 * wl_0 * wl_0 * wl_0; // R
    float rayleigh_beta_1 = rayleigh_beta_base * wl_1 * wl_1 * wl_1 * wl_1; // G
    float rayleigh_beta_2 = rayleigh_beta_base * wl_2 * wl_2 * wl_2 * wl_2; // B

    // 米氏散射系数
    float mie_beta = 0.05;

    const float H = 8.0f;   // 大气层厚度
    const float H_R = 8.0f;  // 瑞利散射高度
    const float H_M = 1.2f;  // 米氏散射高度
    // 地球+大气层半径，用于计算几何光程，控制维纳斯带强度
    float R = (sun_altitude_deg >= 0.0f) ? ((3000.0f-300.0f) * expf(-sun_altitude_deg * 0.1f) + 300.0f)
                                         : ((3000.0f-60.0f) * expf(sun_altitude_deg * 0.2f) + 60.0f);

    const int32_t S = 4; // 路径积分的取样点数

    // 已知三角形ABC，AB固定为(R-H)，AC固定为R，角B=α+90°，求BC关于α的表达式。

    float view_alt = (view_altitude_deg >= 90.0f) ? (180.0f - view_altitude_deg) : view_altitude_deg;
    float cos_view_alt = cosf(to_rad_float(view_alt));
    float sin_view_alt = sinf(to_rad_float(view_alt));
    float Lray = sqrtf(R*R - (R-H)*(R-H) * cos_view_alt * cos_view_alt) - (R-H) * sin_view_alt;

    float sun_alt = (sun_altitude_deg >= 90.0f) ? (180.0f - sun_altitude_deg) : sun_altitude_deg;
    float cos_sun_alt = cosf(to_rad_float(sun_alt));
    float sin_sun_alt = sinf(to_rad_float(sun_alt));
    float Lsun = sqrtf(R*R - (R-H)*(R-H) * cos_sun_alt * cos_sun_alt) - (R-H) * sin_sun_alt;

    float sum_R_r = 0.0f; float sum_R_g = 0.0f; float sum_R_b = 0.0f;
    float sum_M_r = 0.0f; float sum_M_g = 0.0f; float sum_M_b = 0.0f;
    float depth_R = 0.0f;
    float depth_M = 0.0f;

    for (int32_t i = 0; i < S; i++) {
        float t = (float)i / (float)S;

        float L_t = t * Lray;
        float dl = L_t / (float)S;
        float h_R_t = t * H_R;
        float h_M_t = t * H_M;
        float density_R_t = expf(-h_R_t / H_R);
        float density_M_t = expf(-h_M_t / H_M);

        depth_R += density_R_t * dl;
        depth_M += density_M_t * dl;

        float depth_sun_R = 0;
        float depth_sun_M = 0;

        float Lsun_t = (1-t) * Lsun;
        float dlsun = Lsun_t / (float)S;

        for (int32_t j = 0; j < S; j++) {
            float n = (float)j / (float)S;
            float hsun_R = h_R_t + n * (1-t) * H_R;
            float hsun_M = h_M_t + n * (1-t) * H_M;
            float density_sun_R = expf(-hsun_R / H_R);
            float density_sun_M = expf(-hsun_M / H_M);
            depth_sun_R += density_sun_R * dlsun;
            depth_sun_M += density_sun_M * dlsun;
        }

        float tau_r = rayleigh_beta_0 * (depth_sun_R + depth_R) + mie_beta * (depth_sun_M + depth_M) * 1.1;
        float tau_g = rayleigh_beta_1 * (depth_sun_R + depth_R) + mie_beta * (depth_sun_M + depth_M) * 1.1;
        float tau_b = rayleigh_beta_2 * (depth_sun_R + depth_R) + mie_beta * (depth_sun_M + depth_M) * 1.1;

        float attn_r = expf(-tau_r);
        float attn_g = expf(-tau_g);
        float attn_b = expf(-tau_b);

        sum_R_r += attn_r * depth_R; sum_R_g += attn_g * depth_R; sum_R_b += attn_b * depth_R;
        sum_M_r += attn_r * depth_M; sum_M_g += attn_g * depth_M; sum_M_b += attn_b * depth_M;
    }

    // 臭氧吸收
    const float OZONE_ABSORPTION[3] = { 0.005,  0.040,  0.025 };
    float oz_factor = 0.16f;
    float oz_r = expf(-OZONE_ABSORPTION[0] * Lsun * oz_factor);
    float oz_g = expf(-OZONE_ABSORPTION[1] * Lsun * oz_factor);
    float oz_b = expf(-OZONE_ABSORPTION[2] * Lsun * oz_factor);

    float scattered_0 = (sum_R_r * rayleigh_beta_0 * rayleigh_phase + sum_M_r * mie_beta * mie_phase) * oz_r;
    float scattered_1 = (sum_R_g * rayleigh_beta_1 * rayleigh_phase + sum_M_g * mie_beta * mie_phase) * oz_g;
    float scattered_2 = (sum_R_b * rayleigh_beta_2 * rayleigh_phase + sum_M_b * mie_beta * mie_phase) * oz_b;

    // 全局增益、限幅、输出
    // const float global_gain = (4.0f-1.8f) * expf(-(sun_altitude_deg / 3.0f) * (sun_altitude_deg / 3.0f)) + 1.8f;
    const float global_gain = 2.0f;
    *red   = MIN(1.0f, scattered_0 * global_gain);
    *green = MIN(1.0f, scattered_1 * global_gain);
    *blue  = MIN(1.0f, scattered_2 * global_gain);
}


/* ================================================================
 * 大气物理常数（国际单位制，长度单位：米）
 *
 * 模型基于 Nishita et al. 1993，扩展如下改进：
 *   · 球形几何（精确射线-球面求交）
 *   · 逐采样点地球遮蔽阴影检测
 *   · 二阶多次散射（对天球方向的数值积分）
 * ================================================================ */
static const float S3_Re  = 6360e3f;          /* 地球半径 (m) */
static const float S3_Ra  = 6420e3f;          /* 大气层外边界半径 (m) */
static const float S3_Hr  = 7994.0f;          /* Rayleigh 标高 (m) */
static const float S3_Hm  = 1200.0f;          /* Mie 标高 (m) */

/* Rayleigh 海平面散射系数 (m⁻¹)，对应 R(680nm) / G(550nm) / B(440nm) */
static const float S3_bR0 = 5.8e-6f;
static const float S3_bR1 = 13.5e-6f;
static const float S3_bR2 = 33.1e-6f;
static const float S3_bM  = 16e-6f;           /* Mie 散射系数 (m⁻¹) */
static const float S3_bMe = 16e-6f * 0.1f;    /* Mie 消光系数 ≈ 1.1 × 散射系数 NOTE 防止地平线消光过度，故将其调整到0.1 */
static const float S3_g   = 0.76f;            /* Mie 各向异性参数（前向散射优势） */
static const float S3_SUN = 20.0f;            /* 太阳强度归一化系数 */

/* 上半球离散采样方向（原始未归一化值，在函数入口处完成归一化）
 * 索引顺序：天顶、偏东上、偏西上、偏北上、偏南上、东北、西北、东南、西南
 * 重要：这些是固定的物理方向，不含任何太阳角度阈值逻辑。 */
static const float s3_amb_raw_x[9] = {  0.0f,  1.0f, -1.0f,  0.0f,  0.0f,  1.0f, -1.0f,  1.0f, -1.0f };
static const float s3_amb_raw_y[9] = {  0.0f,  0.0f,  0.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f };
static const float s3_amb_raw_z[9] = {  1.0f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f };

/* ================================================================
 * 散射相位函数
 *
 * 参数约定与 Nishita 参考代码一致：
 *   μ = dot(视线方向 d, 太阳方向 sunDir)
 *
 * 物理意义：μ 是视线方向与太阳方向的夹角余弦。
 * 对于 Rayleigh（μ² 对称），取值方向不影响结果；
 * 对于 Mie（g=0.76 前向散射），μ=1 时达到峰值，
 * 即观测者朝向太阳时散射最强——物理上正确。
 * ================================================================ */
static float s3_phaseR(float mu) {
    return 3.0f / (16.0f * (float)M_PI) * (1.0f + mu * mu);
}

static float s3_phaseM(float mu) {
    float base = 1.0f + S3_g * S3_g - 2.0f * S3_g * mu;
    if (base < 1e-10f) base = 1e-10f;
    return 3.0f / (8.0f * (float)M_PI)
           * ((1.0f - S3_g * S3_g) * (1.0f + mu * mu))
           / ((2.0f + S3_g * S3_g) * powf(base, 1.5f));
}

/* ================================================================
 * 射线–球面求交（球心在坐标原点，半径 r）
 *
 * |o + t·d|² = r²  =>  t² + 2(o·d)t + (|o|²−r²) = 0
 *
 * 注意：若射线起点在球内（如大气内），则 t₀ < 0 < t₁，
 * 调用处应取 t = max(0, t₀) 作为积分起点。
 *
 * 返回 1  → 有交点，结果写入 *t0, *t1；
 * 返回 0 → 判别式小于 0，无交点。
 * ================================================================ */
static int32_t s3_rsi(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float r,
    float *t0, float *t1) {
    float b    = 2.0f * (ox * dx + oy * dy + oz * dz);
    float c    = (ox * ox + oy * oy + oz * oz) - r * r;
    float disc = b * b - 4.0f * c;       /* a = 1（d 已归一化） */
    if (disc < 0.0f) return 0;
    float sq = sqrtf(disc);
    *t0 = (-b - sq) * 0.5f;
    *t1 = (-b + sq) * 0.5f;
    return 1;
}

/* ================================================================
 * 核心单次散射步进（Nishita 1993，公式 4 / 5）
 *
 * 沿射线 o + t·d（t ∈ [t0, t1]）数值积分。
 * 主射线取 nV 个采样点，每点向太阳方向追踪 nL 个次级采样，
 * 判断是否被地球遮挡，并计算两段路径的合并透射率。
 *
 * 输出（通过指针参数）：
 *   *sumR0/1/2 = Σᵢ exp(−τᶜᵢ) · ρR(hᵢ) · Δs   （Rayleigh 加权积分）
 *   *sumM0/1/2 = Σᵢ exp(−τᶜᵢ) · ρM(hᵢ) · Δs   （Mie 加权积分）
 *
 * 最终颜色 = (sumR[c]·βR[c]·PR + sumM[c]·βM·PM) · SUN
 * ================================================================ */
static void s3_marchRay(
    float ox,     float oy,     float oz,
    float dx,     float dy,     float dz,
    float t0,     float t1,     int nV,   int nL,
    float sun_dx, float sun_dy, float sun_dz,
    float *out_sumR0, float *out_sumR1, float *out_sumR2,
    float *out_sumM0, float *out_sumM1, float *out_sumM2
) {
    float segLen = (t1 - t0) / (float)nV;
    float sumR0 = 0.0f, sumR1 = 0.0f, sumR2 = 0.0f;
    float sumM0 = 0.0f, sumM1 = 0.0f, sumM2 = 0.0f;
    float odR = 0.0f, odM = 0.0f;    /* 沿主射线从起点累积的光学深度 */

    for (int i = 0; i < nV; i++) {
        float t     = t0 + ((float)i + 0.5f) * segLen;
        float pos_x = ox + dx * t;
        float pos_y = oy + dy * t;
        float pos_z = oz + dz * t;
        float h     = sqrtf(pos_x * pos_x + pos_y * pos_y + pos_z * pos_z) - S3_Re;
        /* if (h < 0.0f) continue; */   /* 采样点位于地表以下，跳过 */

        /* 当前采样点的大气密度（指数衰减）× 路径增量 */
        float dR = expf(-h / S3_Hr) * segLen;
        float dM = expf(-h / S3_Hm) * segLen;
        odR += dR;
        odM += dM;

        /* ── 向太阳方向追踪次级射线 ── */
        float sh_t0, sh_t1;
        if (!s3_rsi(pos_x, pos_y, pos_z, sun_dx, sun_dy, sun_dz, S3_Ra, &sh_t0, &sh_t1))
            continue;
        if (sh_t1 <= 0.0f) continue;

        float lSeg    = sh_t1 / (float)nL;
        float lR      = 0.0f, lM = 0.0f;
        int32_t blocked = 0;

        for (int j = 0; j < nL; j++) {
            float lt   = ((float)j + 0.5f) * lSeg;
            float lp_x = pos_x + sun_dx * lt;
            float lp_y = pos_y + sun_dy * lt;
            float lp_z = pos_z + sun_dz * lt;
            float lh   = sqrtf(lp_x * lp_x + lp_y * lp_y + lp_z * lp_z) - S3_Re;
            if (lh < 0.0f) { blocked = 1; break; }   /* ← 地球遮蔽：此采样点无直射阳光 */
            lR += expf(-lh / S3_Hr) * lSeg;
            lM += expf(-lh / S3_Hm) * lSeg;
        }
        if (blocked) continue;   /* 整段太阳射线被遮，跳过此视线采样点 */

        /* 合并透射率（公式5：e^a · e^b = e^(a+b)，避免两次指数调用） */
        float odR_lR = odR + lR;
        float odM_lM = odM + lM;
        float tau0   = S3_bR0 * odR_lR + S3_bMe * odM_lM;
        float tau1   = S3_bR1 * odR_lR + S3_bMe * odM_lM;
        float tau2   = S3_bR2 * odR_lR + S3_bMe * odM_lM;
        float atten0 = expf(-tau0);
        float atten1 = expf(-tau1);
        float atten2 = expf(-tau2);
        sumR0 += atten0 * dR;   sumM0 += atten0 * dM;
        sumR1 += atten1 * dR;   sumM1 += atten1 * dM;
        sumR2 += atten2 * dR;   sumM2 += atten2 * dM;
    }

    *out_sumR0 = sumR0;  *out_sumR1 = sumR1;  *out_sumR2 = sumR2;
    *out_sumM0 = sumM0;  *out_sumM1 = sumM1;  *out_sumM2 = sumM2;
}

/* HDR 色调映射（原 JS 局部内联函数，移至函数外部） */
static float s3_hdr_mapping(float x) {
    const float e = 0.2f;
    return (sqrtf(1.0f + e) / 2.0f)
           * (sqrtf((1.0f + x) * (1.0f + x) + e)
              - sqrtf((1.0f - x) * (1.0f - x) + e));
}

/* ================================================================
 * scatter_model_3：大气散射主函数
 * ================================================================ */
void scatter_model_3(
    float ray_x, float ray_y, float ray_z,
    float sun_x, float sun_y, float sun_z,
    float *red, float *green, float *blue,   /* Output */
    int32_t enable_opt_lut                   /* dummy */
) {
    (void)enable_opt_lut;

    /* ================================================================
     * 输入归一化与有效性检验
     * ================================================================ */
    float ray_proc_x = ray_x;
    float ray_proc_y = ray_y;
    float ray_proc_z = (ray_z < 0.0f) ? -ray_z : ray_z;

    /* dir = vNorm(ray_vec_processed) */
    float ray_proc_len = sqrtf(  ray_proc_x * ray_proc_x
                               + ray_proc_y * ray_proc_y
                               + ray_proc_z * ray_proc_z);
    float dir_x, dir_y, dir_z;
    if (ray_proc_len > 1e-12f) {
        dir_x = ray_proc_x / ray_proc_len;
        dir_y = ray_proc_y / ray_proc_len;
        dir_z = ray_proc_z / ray_proc_len;
    } else {
        dir_x = 0.0f;  dir_y = 0.0f;  dir_z = 0.0f;
    }

    /* sunDir = vNorm(sun_vec) */
    float sun_len = sqrtf(sun_x * sun_x + sun_y * sun_y + sun_z * sun_z);
    float sunDir_x, sunDir_y, sunDir_z;
    if (sun_len > 1e-12f) {
        sunDir_x = sun_x / sun_len;
        sunDir_y = sun_y / sun_len;
        sunDir_z = sun_z / sun_len;
    } else {
        sunDir_x = 0.0f;  sunDir_y = 0.0f;  sunDir_z = 0.0f;
    }

    /* if (vLen(dir) < 0.5 || vLen(sunDir) < 0.5) return [0, 0, 0] */
    float dir_vlen    = sqrtf(dir_x    * dir_x    + dir_y    * dir_y    + dir_z    * dir_z);
    float sunDir_vlen = sqrtf(sunDir_x * sunDir_x + sunDir_y * sunDir_y + sunDir_z * sunDir_z);
    if (dir_vlen < 0.5f || sunDir_vlen < 0.5f) {
        *red = 0.0f;  *green = 0.0f;  *blue = 0.0f;
        return;
    }

    /* 观测者位于海平面（坐标系：Z 轴朝上，与原代码一致） */
    const float orig_x = 0.0f;
    const float orig_y = 0.0f;
    const float orig_z = S3_Re + 1.0f;

    /* 太阳仰角 */
    float sdir_z_cl       = sunDir_z < -1.0f ? -1.0f : (sunDir_z > 1.0f ? 1.0f : sunDir_z);
    float sunElevation    = asinf(sdir_z_cl);              /* [-π/2, π/2] */
    float sunElevationDeg = sunElevation * 180.0f / (float)M_PI;

    /* 观察者视线的仰角、天顶角 */
    float dir_z_cl         = dir_z < 0.0f ? 0.0f : (dir_z > 1.0f ? 1.0f : dir_z);
    float viewZenith       = acosf(dir_z_cl);
    float viewZenithDeg    = viewZenith * 180.0f / (float)M_PI;
    float viewElevationDeg = 90.0f - viewZenithDeg;

    /* ================================================================
     * 确定主视线与大气球的有效积分范围
     * ================================================================ */
    float atmHit_t0, atmHit_t1;
    if (!s3_rsi(orig_x, orig_y, orig_z, dir_x, dir_y, dir_z, S3_Ra,
                &atmHit_t0, &atmHit_t1)
        || atmHit_t1 < 0.0f) {
        *red = 0.0f;  *green = 0.0f;  *blue = 0.0f;
        return;
    }
    float tMin = (atmHit_t0 > 0.0f) ? atmHit_t0 : 0.0f;
    float tMax = atmHit_t1;

    float earthHit_t0, earthHit_t1;
    if (s3_rsi(orig_x, orig_y, orig_z, dir_x, dir_y, dir_z, S3_Re,
               &earthHit_t0, &earthHit_t1)
        && earthHit_t0 > 1.0f) {
        /* 视线从外部射入地球（观测者在地面以上，ray 指向地面方向） */
        if (earthHit_t0 < tMax) tMax = earthHit_t0;
    }

    if (tMax <= tMin) {
        *red = 0.0f;  *green = 0.0f;  *blue = 0.0f;
        return;
    }

    /* ================================================================
     * 一阶单次散射（Nishita 核心路径积分）
     * ================================================================ */
    float mu1 = dir_x * sunDir_x + dir_y * sunDir_y + dir_z * sunDir_z;
    float pR1 = s3_phaseR(mu1);
    float pM1 = s3_phaseM(mu1);

    float sc1_sumR0, sc1_sumR1, sc1_sumR2;
    float sc1_sumM0, sc1_sumM1, sc1_sumM2;
    s3_marchRay(
        orig_x, orig_y, orig_z,
        dir_x,  dir_y,  dir_z,
        tMin, tMax, 36, 1,
        sunDir_x, sunDir_y, sunDir_z,
        &sc1_sumR0, &sc1_sumR1, &sc1_sumR2,
        &sc1_sumM0, &sc1_sumM1, &sc1_sumM2);

    /* color 数组直接累加后续贡献，此处先写入一阶结果 */
    float color0 = (sc1_sumR0 * S3_bR0 * pR1 + sc1_sumM0 * S3_bM * pM1) * S3_SUN;
    float color1 = (sc1_sumR1 * S3_bR1 * pR1 + sc1_sumM1 * S3_bM * pM1) * S3_SUN;
    float color2 = (sc1_sumR2 * S3_bR2 * pR1 + sc1_sumM2 * S3_bM * pM1) * S3_SUN;

    /* ================================================================
     * 二阶多次散射（第一性原理数值积分）
     *
     * 物理机制：视线上每个采样点 X 除接受直射阳光（一阶散射）外，
     * 还受到来自其他方向的已散射天光照射。
     *
     * 晨昏时段：下层大气处于地球阴影中，一阶散射贡献极少；
     * 但高层大气仍在阳光照射下，并向各方向发出一次散射天光，
     * 这些光子到达 X 后再次散射至观测者——这是真实黄昏/晨曦
     * 天空亮度的主要物理来源，无需任何阈值判断，完全由几何决定。
     *
     * 积分公式（对视线路径和天球方向各取一次数值积分）：
     *
     *   L₂(eye) = ∫_view T(eye→X) ·
     *       ∫_{4π} L₁(X,ω) · [βR·PR(ω,d) + βM·PM(ω,d)] · dω · dt
     *
     *   其中：
     *     · T(eye→X)  : 视线方向的透射率（已在主射线中累积）
     *     · L₁(X,ω)   : 从 X 沿方向 ω 的一次散射天光辐亮度
     *                    由轻量 marchRay 调用计算（4+4 采样）
     *     · βR·PR + βM·PM : X 处将入射天光向观测者方向散射的强度
     *     · 球面积分用 N_AMB 个均匀方向蒙特卡洛离散化（dΩ = 2π/N）
     *
     * 相位函数方向约定：
     *     入射光子行进方向 = −ω（天光从方向 ω 射向 X）
     *     出射光子行进方向 = −d（从 X 射向眼睛，d = orig→X 方向）
     *     μ_scatter = cos θ = dot(−ω, −d) = dot(ω, d)
     * ================================================================ */

    /* 上半球离散采样方向（9 方向，准均匀覆盖上半球 2π sr）
     * 重要：这些是固定的物理方向，不含任何太阳角度阈值逻辑 */
    float ambDirs_x[9], ambDirs_y[9], ambDirs_z[9];
    for (int k = 0; k < 9; k++) {
        float l = sqrtf(  s3_amb_raw_x[k] * s3_amb_raw_x[k]
                        + s3_amb_raw_y[k] * s3_amb_raw_y[k]
                        + s3_amb_raw_z[k] * s3_amb_raw_z[k]);
        if (l > 1e-12f) {
            ambDirs_x[k] = s3_amb_raw_x[k] / l;
            ambDirs_y[k] = s3_amb_raw_y[k] / l;
            ambDirs_z[k] = s3_amb_raw_z[k] / l;
        } else {
            ambDirs_x[k] = 0.0f;
            ambDirs_y[k] = 0.0f;
            ambDirs_z[k] = 0.0f;
        }
    }

    const int   N_AMB  = 9;
    const float dOmega = 2.0f * (float)M_PI / (float)N_AMB;  /* 每方向代表的立体角（上半球 2π / N_AMB） */

    const int N_V2  = 3;                          /* 二阶路径采样数（粗于一阶，节省算力） */
    float     segV2 = (tMax - tMin) / (float)N_V2;
    float     odR2  = 0.0f, odM2 = 0.0f;         /* 累积视线光学深度（eye → 当前采样点） */

    float global_gain = (16.0f - 2.0f)
                        * expf(-(sunElevationDeg / 10.0f) * (sunElevationDeg / 10.0f))
                        + 2.0f;

    for (int i = 0; i < N_V2; i++) {
        float t     = tMin + ((float)i + 0.5f) * segV2;
        float pos_x = orig_x + dir_x * t;
        float pos_y = orig_y + dir_y * t;
        float pos_z = orig_z + dir_z * t;
        float h     = sqrtf(pos_x * pos_x + pos_y * pos_y + pos_z * pos_z) - S3_Re;
        if (h < 0.0f) continue;

        float dR = expf(-h / S3_Hr) * segV2;
        float dM = expf(-h / S3_Hm) * segV2;
        odR2 += dR;
        odM2 += dM;

        /* T(eye → X)：视线方向累积透射率 */
        float Tv0 = expf(-(S3_bR0 * odR2 + S3_bMe * odM2));
        float Tv1 = expf(-(S3_bR1 * odR2 + S3_bMe * odM2));
        float Tv2 = expf(-(S3_bR2 * odR2 + S3_bMe * odM2));

        for (int k = 0; k < N_AMB; k++) {
            float ad_x = ambDirs_x[k];
            float ad_y = ambDirs_y[k];
            float ad_z = ambDirs_z[k];

            /* 从 X 出发沿方向 ad 找到大气出口（X 在大气内，t₀ < 0 < t₁） */
            float aHit_t0, aHit_t1;
            if (!s3_rsi(pos_x, pos_y, pos_z, ad_x, ad_y, ad_z,
                        S3_Ra, &aHit_t0, &aHit_t1))
                continue;
            if (aHit_t1 <= 0.0f) continue;

            /* 计算从 X 出发沿方向 ad 的一次散射天光 L₁(X, ad)
             * 使用轻量级步进（4 主采样 + 4 太阳采样），性能友好 */
            float muA = ad_x * sunDir_x + ad_y * sunDir_y + ad_z * sunDir_z;
            float sc_sumR0, sc_sumR1, sc_sumR2;
            float sc_sumM0, sc_sumM1, sc_sumM2;
            s3_marchRay(
                pos_x, pos_y, pos_z,
                ad_x,  ad_y,  ad_z,
                0.0f, aHit_t1, 2, 2,
                sunDir_x, sunDir_y, sunDir_z,
                &sc_sumR0, &sc_sumR1, &sc_sumR2,
                &sc_sumM0, &sc_sumM1, &sc_sumM2);

            float pRA = s3_phaseR(muA);
            float pMA = s3_phaseM(muA);
            float L1_0 = (sc_sumR0 * S3_bR0 * pRA + sc_sumM0 * S3_bM * pMA) * S3_SUN;
            float L1_1 = (sc_sumR1 * S3_bR1 * pRA + sc_sumM1 * S3_bM * pMA) * S3_SUN;
            float L1_2 = (sc_sumR2 * S3_bR2 * pRA + sc_sumM2 * S3_bM * pMA) * S3_SUN;

            /* X 处将天光 L₁ 散射向观测者方向
             * 相位函数参数：μ = dot(ad, dir)（见上方推导） */
            float muS = ad_x * dir_x + ad_y * dir_y + ad_z * dir_z;
            float pRS = s3_phaseR(muS);
            float pMS = s3_phaseM(muS);

            /* 二阶贡献 = L₁ · (βR·PR·ρR·Δs + βM·PM·ρM·Δs) · T(eye→X) · dΩ */
            float sigma0 = S3_bR0 * pRS * dR + S3_bM * pMS * dM;
            float sigma1 = S3_bR1 * pRS * dR + S3_bM * pMS * dM;
            float sigma2 = S3_bR2 * pRS * dR + S3_bM * pMS * dM;
            color0 += L1_0 * sigma0 * Tv0 * dOmega * global_gain;
            color1 += L1_1 * sigma1 * Tv1 * dOmega * global_gain;
            color2 += L1_2 * sigma2 * Tv2 * dOmega * global_gain;
        }
    }

    /* 臭氧吸收 */
    const float ozone_absorption0 = 0.020f;
    const float ozone_absorption1 = 0.040f;
    const float ozone_absorption2 = 0.000f;
    const float oz_scale = 90.0f;
    float oz_path = (tMax - tMin) / S3_Ra;
    color0 *= expf(-ozone_absorption0 * oz_path * oz_scale);
    color1 *= expf(-ozone_absorption1 * oz_path * oz_scale);
    color2 *= expf(-ozone_absorption2 * oz_path * oz_scale);

    /* 拍脑袋：夜天光 */
    float night_light_scale = 1.0f - viewElevationDeg / 90.0f;
    color0 += 0.05f * night_light_scale;
    color1 += 0.08f * night_light_scale;
    color2 += 0.12f * night_light_scale;

    /* HDR */
    color0 = s3_hdr_mapping(color0);
    color1 = s3_hdr_mapping(color1);
    color2 = s3_hdr_mapping(color2);

    // 限幅输出
    *red   = MIN(1.0f, color0);
    *green = MIN(1.0f, color1);
    *blue  = MIN(1.0f, color2);
}



void calculate_scattered_pixel(
    float ray_x, float ray_y, float ray_z, float sun_x, float sun_y, float sun_z,
    float *red, float *green, float *blue,
    int32_t enable_opt_lut,
    int32_t model_index
) {
    if (model_index == 1) {
        scatter_model_1(ray_x, ray_y, ray_z, sun_x, sun_y, sun_z, red, green, blue, enable_opt_lut);
    }
    else if (model_index == 2) {
        scatter_model_2(ray_x, ray_y, ray_z, sun_x, sun_y, sun_z, red, green, blue, enable_opt_lut);
    }
    else if (model_index == 3) {
        scatter_model_3(ray_x, ray_y, ray_z, sun_x, sun_y, sun_z, red, green, blue, enable_opt_lut);
    }
    else {
        *red   = 0.0f;
        *green = 0.0f;
        *blue  = 0.0f;
    }
}



// ===============================================================================
// 初始化天象仪语境
// ===============================================================================

void linglong_init(Linglong_Config *cfg) {

#if LINGLONG_ENABLE_DYNAMIC_LANDSCAPE
    landscape_buffer_rgb = (uint8_t *)platform_calloc(landscape_texture_width * landscape_texture_height * 3, sizeof(uint8_t));
#endif

    cfg->fb_width = 0;
    cfg->fb_height = 0;
    cfg->sky_radius = 0;
    cfg->center_x = 0;
    cfg->center_y = 0;
    cfg->view_alt = 90.0f;
    cfg->view_azi = 180.0f;
    cfg->view_roll = 0.0f;
    cfg->view_f = 1.0f;

    cfg->year = 2026;
    cfg->month = 2;
    cfg->day = 17;
    cfg->hour = 12;
    cfg->minute = 0;
    cfg->second = 0;
    cfg->timezone = 8.0;
    cfg->longitude = 119.0;
    cfg->latitude = 31.0;

    cfg->downsampling_factor = 0;     // 降采样因子（设为0为自动，建议设为2）
    cfg->enable_opt_sym = 0;          // 是否启用基于对称性的渲染优化（以画质为代价）
    cfg->enable_opt_lut = 0;          // 是否启用查找表计算加速（以画质为代价）
    cfg->enable_opt_bilinear = 1;     // 是否启用双线性插值以优化画质

    cfg->projection = 0;              // 投影算法（0-鱼眼；1-线性透视）
    cfg->sky_model = 3;               // 选择天空模型（0-不启用散射；1-简单散射；2-一次散射；3-二次散射）
    cfg->landscape_index = 2;         // 选择地景贴图（0-不启用，地景设为纯黑；其他-地景贴图序号）
    cfg->enable_equatorial_coord = 0; // 是否启用赤道坐标圈
    cfg->enable_horizontal_coord = 1; // 是否启用地平坐标圈（0-不启用；1-仅方位角文字；2-方位角+坐标圈）
    cfg->enable_star_burst = 1;       // 是否启用星芒效果
    cfg->enable_star_name = 0;        // 是否显示恒星名称（0-不显示；1-除行星；2-仅行星；3-全部）
    cfg->enable_planet = 1;           // 是否显示大行星
    cfg->enable_ecliptic_circle = 0;  // 是否显示黄道
    cfg->enable_att_indicator = 0;    // 是否显示姿态指示标记

    cfg->enable_imu = 1;              // 是否启用IMU（使视角随机器姿态旋转）
}


// ===============================================================================
// 渲染整个天空
// ===============================================================================

void render_sky(Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude,
    int32_t downsampling_factor,     // 降采样因子（设为0为自动，建议设为2）
    int32_t enable_opt_sym,          // 是否启用基于对称性的渲染优化（以画质为代价）
    int32_t enable_opt_lut,          // 是否启用查找表计算加速（以画质为代价）
    int32_t enable_opt_bilinear,     // 是否启用双线性插值以优化画质

    int32_t projection,              // 投影算法（0-鱼眼；1-线性透视）
    int32_t sky_model,               // 选择天空模型（0-不启用散射；1-简单散射；2-一次散射；3-二次散射）
    int32_t landscape_index,         // 选择地景贴图（0-不启用，地景设为纯黑；其他-地景贴图序号）
    int32_t enable_equatorial_coord, // 是否启用赤道坐标圈
    int32_t enable_horizontal_coord, // 是否启用地平坐标圈（0-不启用；1-仅方位角文字；2-方位角+坐标圈）
    int32_t enable_star_burst,       // 是否启用星芒效果
    int32_t enable_star_name,        // 是否显示恒星名称（0-不显示；1-除行星；2-仅行星；3-全部）
    int32_t enable_planet,           // 是否显示大行星
    int32_t enable_ecliptic_circle,  // 是否显示黄道
    int32_t enable_att_indicator     // 是否显示姿态指示标记
) {

    uint32_t fb_width = gfx->width;
    uint32_t fb_height = gfx->height;

    if (fb_width < (int32_t)sky_radius * 2 || fb_height < (int32_t)sky_radius * 2 ||
        center_x < (int32_t)sky_radius || (fb_width - center_x) < (int32_t)sky_radius ||
        center_y < (int32_t)sky_radius || (fb_height - center_y) < (int32_t)sky_radius
    ) {
        return;
    }

    // 计算太阳位置
    double sun_azi = 0.0;
    double sun_alt = 0.0;
    where_is_the_sun(year, month, day, hour, minute, second, timezone, longitude, latitude, &sun_azi, &sun_alt);

    // 夜晚默认启用查找表
    if (sun_alt < -18.0f) enable_opt_lut = 1;

    float sun_x = 0.0f;
    float sun_y = 0.0f;
    float sun_z = 0.0f;
    horizontal_to_xyz(sun_azi, sun_alt, sky_radius, &sun_x, &sun_y, &sun_z);

    float sun_proj_x = 0.0f;
    float sun_proj_y = 0.0f;
    fisheye_project(sun_azi, sun_alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &sun_proj_x, &sun_proj_y);

    // 天空绘制范围的屏幕坐标
    // int32_t y1 = (int32_t)floorf(center_y - sky_radius);
    // int32_t y2 = (int32_t)floorf(center_y + sky_radius);
    // int32_t x1 = (int32_t)floorf(center_x - sky_radius);
    // int32_t x2 = (int32_t)floorf(center_x + sky_radius);
    int32_t y1 = 0;
    int32_t y2 = fb_height;
    int32_t x1 = 0;
    int32_t x2 = fb_width;

    ////////////////////////////
    // 大气散射
    ////////////////////////////

    // 用于计算地景近地大气散射的天光平均色相
    float atmo_r = 0;
    float atmo_g = 0;
    float atmo_b = 0;

    if (sky_model > 0) {

        // 对天空半球进行采样，计算天光平均色相
        uint32_t atmo_count = 0;
        for (float ray_alt = 0.0f; ray_alt < 30.0f; ray_alt += 2.0f) {
            for (float ray_azi = 0.0f; ray_azi < 360.0f; ray_azi += 10.0f) {
                float _ray_x = 0.0f;
                float _ray_y = 0.0f;
                float _ray_z = 0.0f;
                horizontal_to_xyz(ray_azi, ray_alt, sky_radius, &_ray_x, &_ray_y, &_ray_z);
                float _ar = 0.0f;
                float _ag = 0.0f;
                float _ab = 0.0f;
                calculate_scattered_pixel(_ray_x, _ray_y, _ray_z, sun_x, sun_y, sun_z, &_ar, &_ag, &_ab, enable_opt_lut, sky_model);
                atmo_r += _ar * 255.0f;
                atmo_g += _ag * 255.0f;
                atmo_b += _ab * 255.0f;

                atmo_count++;
            }
        }
        atmo_r = (atmo_r / atmo_count) * 1.0f;
        atmo_g = (atmo_g / atmo_count) * 1.0f;
        atmo_b = (atmo_b / atmo_count) * 1.0f;

        // 调整饱和度、明度拉满
        float hv = MAX(MAX(atmo_r, atmo_g), atmo_b);
        float sat = 0.8;
        atmo_r = MAX(1.0f, sat * atmo_r + (1 - sat) * hv);
        atmo_g = MAX(1.0f, sat * atmo_g + (1 - sat) * hv);
        atmo_b = MAX(1.0f, sat * atmo_b + (1 - sat) * hv);
        float _k = 255.0f / hv;
        atmo_r = MIN(255.0f, atmo_r * _k);
        atmo_g = MIN(255.0f, atmo_g * _k);
        atmo_b = MIN(255.0f, atmo_b * _k);


        // 根据太阳高度自适应设置降采样因子
        int32_t _downsampling_factor = downsampling_factor;
        if (downsampling_factor == 0) {
            if (sun_alt < -18.0) {
                _downsampling_factor = (enable_opt_bilinear && !enable_opt_sym) ? 32 : 8;
            }
            else {
                _downsampling_factor = (enable_opt_bilinear && !enable_opt_sym) ? 4 : 2;
            }
        }

        if (enable_opt_sym) {
            for (int32_t y = y1; y < y2; y += 2) {
                for (int32_t x = x1; x < x2; x += 2) {
                    // 计算太阳与圆心的连线法向量，进而计算半圆范围和镜像点坐标
                    float dx = sun_proj_x - center_x;
                    float dy = sun_proj_y - center_y;
                    float t = (((float)x - center_x) * dx + ((float)y - center_y) * dy) / (dx * dx + dy * dy);
                    int32_t xx = (int32_t)floorf(center_x * 2 + 2 * t * dx - (float)x);
                    int32_t yy = (int32_t)floorf(center_y * 2 + 2 * t * dy - (float)y);

                    if ((xx - center_x) * (xx - center_x) + (yy - center_y) * (yy - center_y) > sky_radius * sky_radius ||
                        (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) > sky_radius * sky_radius
                    ) {
                        continue;
                    }

                    float c0 = (-center_x) * dy - (-center_y) * dx;
                    float nx = (c0 > 0) ? (dy) : (-dy);
                    float ny = (c0 > 0) ? (-dx) : (dx);

                    float dp = nx * (x - center_x) + ny * (y - center_y);

                    if (dp >= 0) {
                        // 观察者到该像素的方向向量（从屏幕坐标系转回地平天球的笛卡尔坐标系）
                        float ray_x = 0.0f;
                        float ray_y = 0.0f;
                        float ray_z = 0.0f;
                        fisheye_unproject((float)x, (float)y, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &ray_x, &ray_y, &ray_z);

                        float red = 0.0f;
                        float green = 0.0f;
                        float blue = 0.0f;
                        calculate_scattered_pixel(ray_x, ray_y, ray_z, sun_x, sun_y, sun_z, &red, &green, &blue, enable_opt_lut, sky_model);

                        uint8_t rr = (uint8_t)(red * 255.0f);
                        uint8_t gg = (uint8_t)(green * 255.0f);
                        uint8_t bb = (uint8_t)(blue  * 255.0f);

                        // 为了补偿上下错位而引入的启发式偏置。通过实验确定，原因不明，可能跟浮点数舍入有关。
                        float k = dy/dx;
                        int32_t offset_x = (k <  1.0f && k >= 0.0f) ? 1 : 0;
                        int32_t offset_y = (k >= 1.0f) ? 1 : 0;

                        for (int32_t i = 0; i < 3; i++) {
                            for (int32_t j = 0; j < 3; j++) {
                                gfx_set_pixel(gfx, (x+i), (y+j), rr, gg, bb);
                                gfx_set_pixel(gfx, (xx + i + offset_x), (yy + j + offset_y), rr, gg, bb);
                            }
                        }
                    }
                }
            }
        }
        else {
            #pragma omp parallel for schedule(dynamic)
            for (int32_t y = y1; y < y2; y += _downsampling_factor) {
                for (int32_t x = x1; x < x2; x += _downsampling_factor) {
                    // 观察者到该像素的方向向量（从屏幕坐标系转回地平天球的笛卡尔坐标系）
                    float ray_x = 0.0f;
                    float ray_y = 0.0f;
                    float ray_z = 0.0f;
                    fisheye_unproject((float)x, (float)y, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &ray_x, &ray_y, &ray_z);

                    float red = 0.0f;
                    float green = 0.0f;
                    float blue = 0.0f;
                    calculate_scattered_pixel(ray_x, ray_y, ray_z, sun_x, sun_y, sun_z, &red, &green, &blue, enable_opt_lut, sky_model);

                    if (enable_opt_bilinear) {
                        gfx_set_pixel(gfx, x, y, (uint8_t)(red * 255.0f), (uint8_t)(green * 255.0f), (uint8_t)(blue  * 255.0f));
                    }
                    else {
                        for (int32_t i = 0; i < _downsampling_factor; i++) {
                            for (int32_t j = 0; j < _downsampling_factor; j++) {
                                gfx_add_pixel(gfx,
                                    (x+i), (y+j), (uint8_t)(red * 255.0f), (uint8_t)(green * 255.0f), (uint8_t)(blue  * 255.0f));
                            }
                        }
                    }
                }
            }
        }

        // 双线性插值以缓解块效应
        if (enable_opt_bilinear) {
            int32_t C = _downsampling_factor;
            for (int32_t y = y1; y < y2; y += C) {
                for (int32_t x = x1; x < x2; x += C) {

                    // int32_t r2 = (x-center_x)*(x-center_x) + (y-center_y)*(y-center_y);
                    // if (r2 > (sky_radius+C)*(sky_radius+C)) continue; // NOTE 将滤波范围限定在视野圆内，然而对于任意视角，这是不合理的，故注释之

                    // int32_t idx_11 = ((y+0) * fb_width + (x+0)) * 3;
                    // int32_t idx_12 = (x+C < fb_width)  ? (((y+0) * fb_width + (x+C)) * 3) : idx_11;
                    // int32_t idx_21 = (y+C < fb_height) ? (((y+C) * fb_width + (x+0)) * 3) : idx_11;
                    // int32_t idx_22 = ((x+C < fb_width) && (y+C < fb_height)) ? (((y+C) * fb_width + (x+C)) * 3) : idx_11;

                    uint8_t r11 = 0; uint8_t r12 = 0; uint8_t r21 = 0; uint8_t r22 = 0;
                    uint8_t g11 = 0; uint8_t g12 = 0; uint8_t g21 = 0; uint8_t g22 = 0;
                    uint8_t b11 = 0; uint8_t b12 = 0; uint8_t b21 = 0; uint8_t b22 = 0;

                    // RGB_11
                    gfx_get_pixel(gfx, (x+0), (y+0), &r11, &g11, &b11);

                    // RGB_12
                    if (x+C < fb_width) {
                        gfx_get_pixel(gfx, (x+C), (y+0), &r12, &g12, &b12);
                    }
                    else {
                        gfx_get_pixel(gfx, (x+0), (y+0), &r12, &g12, &b12);
                    }

                    // RGB_21
                    if (y+C < fb_height) {
                        gfx_get_pixel(gfx, (x+0), (y+C), &r21, &g21, &b21);
                    }
                    else {
                        gfx_get_pixel(gfx, (x+0), (y+0), &r12, &g12, &b12);
                    }

                    // RGB_22
                    if ((x+C < fb_width) && (y+C < fb_height)) {
                        gfx_get_pixel(gfx, (x+C), (y+C), &r22, &g22, &b22);
                    }
                    else {
                        gfx_get_pixel(gfx, (x+0), (y+0), &r12, &g12, &b12);
                    }


                    // 检查并处理黑边
                    if (r11+b11+g11 == 0) {
                        if      (r22+b22+g22 != 0) { r11 = r22; g11 = g22; b11 = b22; }
                        else if (r12+b12+g12 != 0) { r11 = r12; g11 = g12; b11 = b12; }
                        else if (r21+b21+g21 != 0) { r11 = r21; g11 = g21; b11 = b21; }
                    }
                    if (r12+b12+g12 == 0) {
                        if      (r22+b22+g22 != 0) { r12 = r22; g12 = g22; b12 = b22; }
                        else if (r11+b11+g11 != 0) { r12 = r11; g12 = g11; b12 = b11; }
                        else if (r21+b21+g21 != 0) { r12 = r21; g12 = g21; b12 = b21; }
                    }
                    if (r21+b21+g21 == 0) {
                        if      (r22+b22+g22 != 0) { r21 = r22; g21 = g22; b21 = b22; }
                        else if (r11+b11+g11 != 0) { r21 = r11; g21 = g11; b21 = b11; }
                        else if (r12+b12+g12 != 0) { r21 = r12; g21 = g12; b21 = b12; }
                    }
                    if (r22+b22+g22 == 0) {
                        if      (r21+b21+g21 != 0) { r22 = r21; g22 = g21; b22 = b21; }
                        else if (r11+b11+g11 != 0) { r22 = r11; g22 = g11; b22 = b11; }
                        else if (r12+b12+g12 != 0) { r22 = r12; g22 = g12; b22 = b12; }
                    }

                    for (int32_t i = 0; i < C; i++) {
                        for (int32_t j = 0; j < C; j++) {
                            if ((x + j >= fb_width) || (y + i >= fb_height)) continue;
                            float v = (float)i / (float)C;
                            float u = (float)j / (float)C;
                            gfx_set_pixel(gfx, (x+j), (y+i),
                                (uint8_t)((1-u)*(1-v)*(float)r11 + u*(1-v)*(float)r12 + (1-u)*v*(float)r21 + u*v*(float)r22),
                                (uint8_t)((1-u)*(1-v)*(float)g11 + u*(1-v)*(float)g12 + (1-u)*v*(float)g21 + u*v*(float)g22),
                                (uint8_t)((1-u)*(1-v)*(float)b11 + u*(1-v)*(float)b12 + (1-u)*v*(float)b21 + u*v*(float)b22));
                        }
                    }
                }
            }
        }
    }

    // 绘制太阳
    render_sun(gfx,
        sky_radius, center_x, center_y,
        sun_proj_x, sun_proj_y, sun_alt);


    // 计算月球位置并绘制
    render_moon(gfx,
        sky_radius, center_x, center_y,
        view_alt, view_azi, view_roll, f, projection,
        year, month, day, hour, minute, second, timezone, longitude, latitude);

    // 绘制星芒
    if (enable_star_burst && sun_alt > 0) {
        star_burst_filter(gfx, sun_proj_x, sun_proj_y);
    }

    // 绘制恒星
    float mag_offset = -2.0f;
    for (int32_t i = 0; i < STARS_NUM; i++) {
        const float *star_item = STARS[i];
        double alt = 0.0;
        double azi = 0.0;
        equatorial_to_horizontal(
            ra_hms_to_deg(star_item[0], star_item[1], star_item[2]),
            dec_dms_to_decimal(star_item[3], star_item[4], star_item[5]),
            year, month, day, hour, minute, second, timezone, longitude, latitude, &azi, &alt);
        float mag = 0.5f + mag_offset + star_item[6];

        float sx = 0.0f;
        float sy = 0.0f;
        fisheye_project(azi, alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &sx, &sy);

        draw_star(gfx, sky_radius, center_x, center_y, sx, sy, mag, 1, 255, 255, 255);

        if (enable_star_name == 1 || enable_star_name == 3) {
            if (sx > 0.0f && sx < (float)fb_width && sy > 0.0f && sy < (float)fb_height) {
                gfx_draw_textline(gfx, (wchar_t*)STAR_NAME[i], sx+3, sy+3, 250, 250, 250, 1);
            }
        }
    }

    // 绘制黄道
    if (enable_ecliptic_circle) {
        draw_ecliptic_circle(gfx,
            sky_radius, center_x, center_y,
            view_alt, view_azi, view_roll, f, projection,
            6, 32, 32, 0,
            year, month, day, hour, minute, second, timezone, longitude, latitude
        );
    }

    // 绘制大行星
    if (enable_planet) {
        for (int32_t i = 8; i >= 1; i--) { // 之所以倒数，是为了让靠近太阳的行星后绘制，使其覆盖在远离太阳的行星上面
            if (i == 3) continue; // 跳过地球
            double planet_azi = 0.0;
            double planet_alt = 0.0;
            where_is_the_planet(year, month, day, hour, minute, second, timezone, longitude, latitude, i, &planet_azi, &planet_alt);

            float planet_proj_x = 0.0f;
            float planet_proj_y = 0.0f;
            fisheye_project(planet_azi, planet_alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &planet_proj_x, &planet_proj_y);

            draw_star(gfx, sky_radius, center_x, center_y, planet_proj_x, planet_proj_y,
                0.0f, PLANET_RADIUS[i], PLANET_COLOR_R[i], PLANET_COLOR_G[i], PLANET_COLOR_B[i]);

            if (enable_star_name == 2 || enable_star_name == 3) {
                if (planet_proj_x > 0.0f && planet_proj_x < (float)fb_width && planet_proj_y > 0.0f && planet_proj_y < (float)fb_height) {
                    gfx_draw_textline(gfx, (wchar_t*)PLANET_NAME[i], planet_proj_x+3, planet_proj_y+3, PLANET_COLOR_R[i], PLANET_COLOR_G[i], PLANET_COLOR_B[i], 1);
                }
            }
        }
    }

    // 绘制赤道坐标网格
    if (enable_equatorial_coord) {
        // 绘制赤道天球子午圈
        for (int32_t i = 0; i < 24; i += 2) {
            int32_t line_width = (i == 0 || i == 12) ? 4 : 1;
            draw_celestial_circle(gfx,
                sky_radius, center_x, center_y,
                view_alt, view_azi, view_roll, f, projection,
                1, (float)i, 0.0f,
                line_width, 8, 16, 32,
                year, month, day, hour, minute, second, timezone, longitude, latitude
            );
        }
        // 绘制赤道天球等纬度圈
        for (int32_t i = -90; i < 90; i += 10) {
            int32_t line_width = (i == 0) ? 4 : 1;
            draw_celestial_circle(gfx,
                sky_radius, center_x, center_y,
                view_alt, view_azi, view_roll, f, projection,
                0, 0.0f, (float)i,
                line_width, 8, 16, 32,
                year, month, day, hour, minute, second, timezone, longitude, latitude
            );
        }
    }


    // 绘制地景（天空投影圆盘之外的部分）
    float fov = 1.1f;
    float view_height = 1.0f;
    int32_t enable_atmosphere_scattering = 0; // 是否启用大气散射效果（只有在高空时启用）
    // 卫星图
    if (landscape_index == 1) {
        // TODO 几何关系待优化
        fov = (0.45f/90.0f) * fabsf(sun_alt) + 1.1f;
        update_landscape((uint8_t*)FLAT_TEXTURE_BUFFER, FLAT_TEXTURE_WIDTH, FLAT_TEXTURE_HEIGHT, 1, fov);
        view_height = 1.0f;
        // float view_height = 0.01f * fabsf(sun_alt) + 1.0f;
        enable_atmosphere_scattering = 1;
    }
    // 鱼眼照片
    else if (landscape_index == 2) {
        update_landscape((uint8_t*)FISHEYE_TEXTURE_BUFFER, FISHEYE_TEXTURE_WIDTH, FISHEYE_TEXTURE_HEIGHT, 0, fov);
        enable_atmosphere_scattering = 0;
    }

    draw_horizon(gfx,
        sky_radius, center_x, center_y,
        view_alt, view_azi, view_roll, f, projection,
        view_height, sun_alt, landscape_index, enable_atmosphere_scattering, (uint8_t)atmo_r, (uint8_t)atmo_g, (uint8_t)atmo_b);



    // 绘制地平坐标网格
    if (enable_horizontal_coord > 0) {
        if (enable_horizontal_coord == 2) {
            // 绘制等仰角圈（地平纬度圈）
            for (int32_t alt = -90; alt <= 90; alt += 10) {
                int32_t line_width = (alt == 0) ? 4 : 1;
                draw_horizontal_altitude_circle(gfx,
                    sky_radius, center_x, center_y,
                    view_alt, view_azi, view_roll, f, projection,
                    (float)alt,
                    line_width, 16, 32, 8
                );
            }
            // 绘制等方位角圈（地平经线圈）- 每隔30度绘制一条线，从0°（北）到330°
            for (int32_t azi = 0; azi < 360; azi += 30) {
                int32_t line_width = 1; // (azi === 0) ? 3 : 2;
                draw_horizontal_azimuth_circle(gfx,
                    sky_radius, center_x, center_y,
                    view_alt, view_azi, view_roll, f, projection,
                    (float)azi,
                    line_width, 16, 32, 8
                );
            }
        }

        float label_x = 0.0f;
        float label_y = 0.0f;
        fisheye_project(0, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &label_x, &label_y);
        gfx_draw_textline_centered(gfx, L"北", label_x, label_y, 255, 0, 0, 1);
        fisheye_project(90, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &label_x, &label_y);
        gfx_draw_textline_centered(gfx, L"东", label_x, label_y, 255, 0, 0, 1);
        fisheye_project(180, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &label_x, &label_y);
        gfx_draw_textline_centered(gfx, L"南", label_x, label_y, 255, 0, 0, 1);
        fisheye_project(270, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &label_x, &label_y);
        gfx_draw_textline_centered(gfx, L"西", label_x, label_y, 255, 0, 0, 1);

        fisheye_project(45, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &label_x, &label_y);
        gfx_draw_textline_centered(gfx, L"45", label_x, label_y, 255, 0, 0, 1);
        fisheye_project(135, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &label_x, &label_y);
        gfx_draw_textline_centered(gfx, L"135", label_x, label_y, 255, 0, 0, 1);
        fisheye_project(225, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &label_x, &label_y);
        gfx_draw_textline_centered(gfx, L"225", label_x, label_y, 255, 0, 0, 1);
        fisheye_project(315, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, projection, &label_x, &label_y);
        gfx_draw_textline_centered(gfx, L"315", label_x, label_y, 255, 0, 0, 1);

    }

    // 绘制姿态仪相关符号
    if (enable_att_indicator > 0) {
        gfx_draw_rectangle(gfx, (center_x - 60), center_y - 2, 50, 4, 255, 255, 0, 1);
        gfx_draw_rectangle(gfx, (center_x - 14), center_y - 2, 4, 10, 255, 255, 0, 1);
        gfx_draw_rectangle(gfx, (center_x + 10), center_y - 2, 50, 4, 255, 255, 0, 1);
        gfx_draw_rectangle(gfx, (center_x + 10), center_y - 2, 4, 10, 255, 255, 0, 1);
        gfx_draw_rectangle(gfx, (center_x - 1), center_y - 1, 2, 2, 255, 255, 0, 1);

        wchar_t euler_angle[59];
        swprintf(euler_angle, 59, L"Pitch:%d  Yaw:%d  Roll:%d  F:%.1f", (int32_t)roundf(view_alt), (int32_t)roundf(view_azi), (int32_t)roundf(view_roll), f);
        gfx_draw_textline_centered(gfx, euler_angle, fb_width/2, fb_height-26, 255, 255, 255, 1);
    }
}
