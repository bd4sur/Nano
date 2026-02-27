#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "utils.h"
#include "glyph.h"

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


// 大气光学参数
static float MIE_G = 0.9f;
static float RAYLEIGH_BETA = 0.3f;
static float MIE_BETA_BASE = 0.001f;
static float MIE_BETA_MAX = 0.1f;
static float ATMOSPHERE_HEIGHT = 8.5f;

// 波长相关的瑞利散射系数 (近似)
static float RAYLEIGH_WAVELENGTH_FACTORS[3] = { 680, 550, 450 };

// 臭氧吸收系数
static float OZONE_ABSORPTION[3] = { 0.005,  0.040,  0.025 };

// 大气光学质量查找表：index(deg)=[0,90]
// AIR_MASS_LUT[i] = 1.0 / (Math.cos(i/180*Math.PI) + 0.50572 * Math.pow(96.07995 - i, -1.6364));
static const float AIR_MASS_LUT[91] = {0.9997119918558381f, 0.9998592586926793f, 1.0003110890402365f, 1.0010681642273038f, 1.0021316319418698f, 1.0035031104733083f, 1.0051846947247018f, 1.0071789640369042f, 1.0094889918788565f, 1.0121183574721748f, 1.0150711594323112f, 1.0183520315238068f, 1.021966160643514f, 1.02591930716334f, 1.03021782778332f, 1.034868701066905f, 1.039879555853524f, 1.045258702769114f, 1.0510151690837524f, 1.0571587371972049f, 1.0636999870686308f, 1.0706503429463945f, 1.0780221247986177f, 1.085828604895485f, 1.0940840700512897f, 1.1028038900988062f, 1.1120045932419853f, 1.121703949016588f, 1.1319210596838862f, 1.1426764609918518f, 1.1539922333636758f, 1.1658921247176817f, 1.1784016862889535f, 1.1915484230151434f, 1.2053619602715009f, 1.2198742289986713f, 1.235119671567844f, 1.2511354710792817f, 1.2679618072017325f, 1.2856421421433408f, 1.304223540913535f, 1.3237570307072422f, 1.3442980050388187f, 1.3659066791992502f, 1.3886486047386126f, 1.4125952520262743f, 1.4378246715634218f, 1.4644222466782195f, 1.4924815526010962f, 1.5221053397946254f, 1.5534066629239196f, 1.5865101811584308f, 1.6215536607984282f, 1.6586897177821054f, 1.698087845793255f, 1.7399367858997083f, 1.7844473064938864f, 1.8318554785517882f, 1.8824265519054382f, 1.936459564717931f, 1.9942928525292494f, 2.0563106676544924f, 2.122951177872814f, 2.194716190118216f, 2.2721830471079465f, 2.3560192822092514f, 2.4470008042366937f, 2.54603463942889f, 2.6541876121514743f, 2.7727228429386073f, 2.9031466488030997f, 3.04726944828179f, 3.2072857614398935f, 3.385880605461222f, 3.5863729280725876f, 3.812911869220776f, 4.07074973811966f, 4.366628617623185f, 4.709338986746881f, 5.110545154023385f, 5.5860358798512f, 6.157673414942184f, 6.856529464259364f, 7.728117030932225f, 8.841485995032256f, 10.30579132793028f, 12.30208325139153f, 15.14773544253034f, 19.433245107572002f, 26.31055506838526f, 37.91960837783621f};

// 散射光随太阳高度变化的衰减因子：sun_altitude_deg=[-90,90]
// ATTN_SCALE_LUT[sun_altitude_deg+90] = (sun_altitude_deg >= 0) ? (expf(-powf((sun_altitude_deg / 20.0f), 4.0f)) + 0.01f) : (1.0f);
static const float ATTN_SCALE_LUT[181] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.01,1.009993750019531,1.0099000049998332,1.0094938781229095,1.0084012793176063,1.0061013694701175,1.0019327166055711,0.9951057826726806,0.9847249016017939,0.9698231310344834,0.9494130628134758,0.9225556126616087,0.8884467393499313,0.8465188286183754,0.796549202213455,0.7387633299194912,0.6739157633354735,0.6033289868052961,0.5288709904654524,0.45285793449343037,0.37787944117144234,0.3065598428646592,0.24128605528432856,0.18394671713945743,0.1357323295944279,0.0970383676562235,0.06749254452616452,0.04609841754220801,0.031459239080080414,0.02202814149649049,0.016329715427485746,0.013113504775028439,0.011424976438192463,0.010603957787932912,0.010235900606629577,0.010084487560285047,0.010027602616196642,0.0100081825538457,0.010002188925004884,0.010000525452390085,0.010000112535174719,0.010000021375803119,0.01000000357929494,0.01000000052506122,0.010000000067048827,0.0100000000074047,0.010000000000702522,0.01000000000005687,0.010000000000003902,0.010000000000000226,0.01000000000000001,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01};

// 夜间散射光的衰减因子：sun_altitude_deg=[-90,90]
// NIGHT_ATTN_LUT[sun_altitude_deg+90] = (sun_altitude_deg >= 0) ? (1.0f) : MAX(0.0f, expf(0.016f * sun_altitude_deg));
static const float NIGHT_ATTN_LUT[181] = {0.23692775868212176,0.24074909196587688,0.2446320583319975,0.24857765184107966,0.2525868825866102,0.2566607769535559,0.26080037788112365,0.2650067451297589,0.26928095555244996,0.27362410337040804,0.27803730045319414,0.2825216766033636,0.28707837984570167,0.2917085767211244,0.2964134525853191,0.3011942119122021,0.3060520786022707,0.3109882962959281,0.31600412869186245,0.32110085987056064,0.32627979462303947,0.3315422587848797,0.33688959957564707,0.34232318594378797,0.34784440891708746,0.35345468195878016,0.3591554413294046,0.3649481464544937,0.37083428029819565,0.37681534974292086,0.38289288597511206,0.38906844487723635,0.3953436074260998,0.401719980097586,0.4081991952779227,0.4147829116815814,0.4214728147759176,0.4282706172126597,0.43517805926635666,0.44219690927989863,0.44932896411722156,0.4565760496233147,0.46394002109164667,0.4714227637391309,0.47902619318875106,0.4867522559599717,0.494602929967057,0.5025802250254283,0.5106861833661879,0.5189228801589404,0.5272924240430485,0.535796957667456,0.5444386582392171,0.5532197380808739,0.5621424451968224,0.5712090638488149,0.5804219151407424,0.5897833576128504,0.5992957878455384,0.6089616410728969,0.6187833918061408,0.6287635544670984,0.6389046840319161,0.6492093766851474,0.659680270484389,0.6703200460356393,0.6811314271795471,0.6921171816887304,0.7032801219763409,0.7146231058160573,0.7261490370736909,0.7378608664505912,0.7497615922390413,0.7618542610898376,0.7741419687922484,0.7866278610665534,0.7993151343693651,0.812207036711939,0.8253068684916824,0.838617983337074,0.8521437889662113,0.865887748059205,0.8798533791446438,0.8940442575003572,0.9084640160687062,0.9231163463866358,0.9380049995307295,0.9531337870775047,0.9685065820791976,0.9841273200552851,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};


// 行星相关常量（用于渲染）
//                                        -     1Mer  2Ven  3Ear  4Mars 5Jup  6Sat  7Ura  8Nep
static const float   PLANET_RADIUS[9]   = {0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 3.0f, 2.0f, 1.0f, 1.0f};
static const uint8_t PLANET_COLOR_R[9]  = {0,    192,  255,  0,    255,  255,  192,  0,    192 };
static const uint8_t PLANET_COLOR_G[9]  = {0,    192,  255,  0,    64,   192,  255,  255,  128 };
static const uint8_t PLANET_COLOR_B[9]  = {0,    192,  64,   0,    0,    128,  128,  255,  255 };
static const wchar_t PLANET_NAME[9][10] = {L"", L"水星", L"金星", L"地球", L"火星", L"木星", L"土星", L"天王星", L"海王星"};

// 星表
#define STARS_NUM (22)
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
    { 5.0f, 55.0f, 11.4f,      7.0f, 24.0f, 22.2f,   0.42f}, // 参宿四
    { 5.0f, 25.0f,  8.8f,      6.0f, 20.0f, 55.1f,   1.64f}, // 参宿五
    { 5.0f, 40.0f, 46.6f,     -1.0f, 56.0f, 38.7f,   1.77f}, // 参宿一
    { 5.0f, 36.0f, 13.8f,     -1.0f, 12.0f, 12.3f,   1.69f}, // 参宿二
    { 5.0f, 32.0f,  1.4f,      0.0f,-18.0f,  1.9f,   2.41f}, // 参宿三
    { 5.0f, 47.0f, 46.5f,     -9.0f, 40.0f, 17.7f,   2.06f}, // 参宿六
    { 5.0f, 14.0f, 33.2f,     -8.0f, 12.0f, 13.0f,   0.13f}, // 参宿七
    { 5.0f, 35.0f, 27.0f,     -5.0f, 54.0f, 42.2f,   2.77f}, // 伐三
    { 3.0f, 47.0f, 29.4f,     24.0f,  6.0f, 19.6f,   2.87f}, // 昴宿星团
};

static const wchar_t STAR_NAME[STARS_NUM][10] = {
    L"北极星", L"天枢", L"", L"", L"", L"", L"开阳", L"",
    L"天狼", L"大角", L"织女一", L"河鼓二", L"M31",
    L"参宿四", L"", L"", L"", L"", L"", L"参宿七", L"", L"昴宿星团"
};


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
// 帧缓冲操作
// ===============================================================================

// 设置像素
static inline void set_pixel(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height, int32_t x, int32_t y, uint8_t r, uint8_t g, uint8_t b) {
    int32_t i = (y * fb_width + x) * 3;
    frame_buffer[ i ] = MIN(255, r);
    frame_buffer[i+1] = MIN(255, g);
    frame_buffer[i+2] = MIN(255, b);
}

// 叠加像素
static inline void add_pixel(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height, int32_t x, int32_t y, uint8_t r, uint8_t g, uint8_t b) {
    int32_t i = (y * fb_width + x) * 3;
    frame_buffer[ i ] = MIN(255, frame_buffer[ i ] + r);
    frame_buffer[i+1] = MIN(255, frame_buffer[i+1] + g);
    frame_buffer[i+2] = MIN(255, frame_buffer[i+2] + b);
}

// 数乘像素
static inline void scale_pixel(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height, int32_t x, int32_t y, float k) {
    int32_t i = (y * fb_width + x) * 3;
    frame_buffer[ i ] = MIN(255, (uint8_t)(k * (float)frame_buffer[ i ]));
    frame_buffer[i+1] = MIN(255, (uint8_t)(k * (float)frame_buffer[i+1]));
    frame_buffer[i+2] = MIN(255, (uint8_t)(k * (float)frame_buffer[i+2]));
}

static inline uint8_t get_pixel_channel(const uint8_t *tex, int32_t width, int32_t x, int32_t y, int32_t c) {
    return tex[((y * width + x) * 3) + c];
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
    float center_x, float center_y, float *scr_x, float *scr_y
) {
    // 因为是躺在地上看天，所以方位角是从正北逆时针旋转
    float az_rad = (-azimuth_deg * M_PI) / 180.0f;
    float alt_rad = (altitude_deg * M_PI) / 180.0f;
    // 天顶距 θ = π/2 - altitude
    float theta = M_PI_2 - alt_rad;
    // 等距鱼眼投影：r = (2R / π) * θ
    float r = (2.0f * radius / M_PI) * theta;
    // 投影到平面：X指向东（注意因为是躺在地上看天，所以东是屏幕坐标系的左侧/负半轴），Y指向北
    float X = r * sinf(az_rad);
    float Y = -r * cosf(az_rad);

    *scr_x = X + center_x;
    *scr_y = Y + center_y;
}


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
    *azimuth_deg = normalize_angle_float(to_deg_float(azRad) - 180.0f); // 逆向抵消原函数的 +180°

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
    float *scr_x, float *scr_y
) {
    // 快速路径：默认视角且无roll时使用简化投影
    if (view_alt == 90.0f && view_azi == 180.0f && view_roll == 0.0f && f == 1.0f) {
        horizontal_to_screen_xy(azimuth_deg, altitude_deg, radius, center_x, center_y, scr_x, scr_y);
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

    // ===== 6. 鱼眼投影计算 =====
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
    return;
}


// 鱼眼反投影：屏幕平面(x,y) → 地平天球笛卡尔坐标系(x,y,z)
// 新增view_roll参数：滚转角(±90°)，正值为顺时针旋转（从观察者视角，右侧地平线下倾）
// 与fisheye_project的roll约定保持一致，确保互为逆运算
void fisheye_unproject(
    float scr_x, float scr_y,
    float radius, float center_x, float center_y, 
    float view_alt, float view_azi, float view_roll, float f,
    float *x, float *y, float *z
) {
    // 快速路径：默认视角且无roll时使用简化反投影
    if (view_alt == 90.0f && view_azi == 180.0f && view_roll == 0.0f && f == 1.0f) {
        screen_xy_to_xyz(scr_x, scr_y, radius, center_x, center_y, x, y, z);
        return;
    }

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
        // 注意：atan2f(dx, dy) 与 project 中的 -atan2f(vx, vy) 互为逆运算
        phi_prime = atan2f(dx, dy);
    }

    // ===== 2. 极坐标转局部坐标系单位向量 =====
    // 等距投影反解: θ = (π/2) * r / (radius * f)
    float theta = (M_PI / 2.0f) * (r / (radius * f));
    
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




// ===============================================================================
// 滤波器
// ===============================================================================

void star_burst_filter(uint8_t *frame_buffer, int32_t width, int32_t height, float sun_screen_x, float sun_screen_y) {
    // 亮度阈值
    float threshold = 0.9;

    float cx = sun_screen_x;
    float cy = sun_screen_y;
    int32_t ix = (int32_t)(floorf(cx));
    int32_t iy = (int32_t)(floorf(cy));
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return;

    // 1. 读取原图中该像素的亮度
    int32_t idx = (iy * width + ix) * 3;
    float r = frame_buffer[idx] / 255.f;
    float g = frame_buffer[idx + 1] / 255.f;
    float b = frame_buffer[idx + 2] / 255.f;
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
                add_pixel(frame_buffer, width, height, px, py, (uint8_t)(sr * alpha), (uint8_t)(sg * alpha), (uint8_t)(sb * alpha));
            }

            // 反向
            px = (int32_t)floorf(cx - ndx * d);
            py = (int32_t)floorf(cy - ndy * d);
            if (px >= 0 && px < width && py >= 0 && py < height) {
                add_pixel(frame_buffer, width, height, px, py, (uint8_t)(sr * alpha), (uint8_t)(sg * alpha), (uint8_t)(sb * alpha));
            }
        }
    }
}

// 抖动：用于缓解低位深屏幕的色带现象
/**
 * Floyd-Steinberg误差扩散抖动
 * 在RGB888缓冲区上模拟RGB565量化过程，通过误差扩散缓解色带
 * 
 * @param frame_buffer RGB8888格式帧缓冲（3字节/像素，顺序R,G,B）
 * @param fb_width     帧缓冲宽度（像素）
 * @param fb_height    帧缓冲高度（像素）
 * 
 * 算法原理：
 * 1. 对每个像素，先叠加累积误差得到"增强值"
 * 2. 模拟RGB565量化：R8→R5(>>3), G8→G6(>>2), B8→B5(>>3)
 * 3. 量化后还原为8bit：R5→R8(<<3|>>2), G6→G8(<<2|>>4), B5→B8(<<3|>>2)
 * 4. 计算误差 = 增强值 - 量化还原值
 * 5. 按Floyd-Steinberg权重扩散误差到邻域：
 *        [ 0      0      0    ]
 *        [ 0      X    7/16   ]
 *        [ 3/16  5/16  1/16   ]
 */
void dithering_fs(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height) {
    if (!frame_buffer || fb_width <= 0 || fb_height <= 0) return;

    const int32_t stride = fb_width * 3;  // RGB888每行字节数

    // 为避免跨行误差污染，使用双行误差缓冲（当前行+下一行）
    // 误差范围：[-255, 255] → 用int16_t安全存储
    int16_t *err_r = (int16_t *)calloc(fb_width * 2, sizeof(int16_t));
    int16_t *err_g = (int16_t *)calloc(fb_width * 2, sizeof(int16_t));
    int16_t *err_b = (int16_t *)calloc(fb_width * 2, sizeof(int16_t));

    if (!err_r || !err_g || !err_b) {
        if (err_r) free(err_r);
        if (err_g) free(err_g);
        if (err_b) free(err_b);
        return; // 内存分配失败，静默退出
    }

    #define CLAMP(v, min, max) ((v) < (min) ? (min) : ((v) > (max) ? (max) : (v)))

    for (int32_t y = 0; y < fb_height; y++) {
        int16_t *curr_err_r = err_r + (y & 1) * fb_width;      // 双缓冲切换
        int16_t *curr_err_g = err_g + (y & 1) * fb_width;
        int16_t *curr_err_b = err_b + (y & 1) * fb_width;
        int16_t *next_err_r = err_r + ((y + 1) & 1) * fb_width;
        int16_t *next_err_g = err_g + ((y + 1) & 1) * fb_width;
        int16_t *next_err_b = err_b + ((y + 1) & 1) * fb_width;

        // 清空下一行误差缓冲（避免残留）
        if (y + 1 < fb_height) {
            memset(next_err_r, 0, fb_width * sizeof(int16_t));
            memset(next_err_g, 0, fb_width * sizeof(int16_t));
            memset(next_err_b, 0, fb_width * sizeof(int16_t));
        }

        uint8_t *row = frame_buffer + y * stride;

        for (int32_t x = 0; x < fb_width; x++) {
            // 1. 叠加累积误差（限制在合理范围防止溢出）
            int32_t r = CLAMP(row[x * 3 + 0] + curr_err_r[x], 0, 255);
            int32_t g = CLAMP(row[x * 3 + 1] + curr_err_g[x], 0, 255);
            int32_t b = CLAMP(row[x * 3 + 2] + curr_err_b[x], 0, 255);

            // 2. 模拟RGB565量化并还原到8bit（保留低位以维持亮度）
            uint8_t r5 = r >> 3;          // 8bit → 5bit
            uint8_t g6 = g >> 2;          // 8bit → 6bit
            uint8_t b5 = b >> 3;          // 8bit → 5bit

            uint8_t r_out = (r5 << 3) | (r5 >> 2);  // 5bit → 8bit (复制高位到低位)
            uint8_t g_out = (g6 << 2) | (g6 >> 4);  // 6bit → 8bit
            uint8_t b_out = (b5 << 3) | (b5 >> 2);  // 5bit → 8bit

            // 3. 写回量化+抖动后的像素值
            row[x * 3 + 0] = r_out;
            row[x * 3 + 1] = g_out;
            row[x * 3 + 2] = b_out;
            // Alpha保持原值

            // 4. 计算误差（原始增强值 - 量化还原值）
            int32_t err_r_val = r - r_out;
            int32_t err_g_val = g - g_out;
            int32_t err_b_val = b - b_out;

            // 5. Floyd-Steinberg误差扩散（仅传播到未处理像素）
            if (x + 1 < fb_width) {
                // 右侧像素 (7/16)
                curr_err_r[x + 1] += (err_r_val * 7) >> 4;
                curr_err_g[x + 1] += (err_g_val * 7) >> 4;
                curr_err_b[x + 1] += (err_b_val * 7) >> 4;
            }

            if (y + 1 < fb_height) {
                if (x > 0) {
                    // 左下 (3/16)
                    next_err_r[x - 1] += (err_r_val * 3) >> 4;
                    next_err_g[x - 1] += (err_g_val * 3) >> 4;
                    next_err_b[x - 1] += (err_b_val * 3) >> 4;
                }
                // 正下 (5/16)
                next_err_r[x] += (err_r_val * 5) >> 4;
                next_err_g[x] += (err_g_val * 5) >> 4;
                next_err_b[x] += (err_b_val * 5) >> 4;

                if (x + 1 < fb_width) {
                    // 右下 (1/16)
                    next_err_r[x + 1] += err_r_val >> 4;
                    next_err_g[x + 1] += err_g_val >> 4;
                    next_err_b[x + 1] += err_b_val >> 4;
                }
            }
        }
    }

    free(err_r);
    free(err_g);
    free(err_b);
}



// 8x8 Bayer矩阵（预缩放至0~255范围，避免运行时乘法）
static const uint8_t bayer8x8[64] = {
    0,  192, 48,  240, 12,  204, 60,  252,
    128,64,  176,112,140,76,  224,160,
    32, 224, 16,  208, 44, 236, 28,  220,
    160,96,  144,80,  172,108,252,188,
    8,  200, 56,  248, 4,  196, 52,  244,
    136,72,  184,120,132,68,  216,152,
    40, 232, 24,  216, 36, 228, 20,  212,
    168,104,152,88,  180,116,244,180
};

// 量化误差补偿表（RGB565量化后还原的亮度偏移补偿）
static const int8_t quant_bias[8] = {0, 1, 1, 2, 2, 3, 3, 4}; // 经验值

void dithering_fast(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height) {
    if (!frame_buffer || fb_width <= 0 || fb_height <= 0) return;

    const int32_t stride = fb_width * 3;

    for (int32_t y = 0; y < fb_height; y++) {
        uint8_t *row = frame_buffer + y * stride;
        for (int32_t x = 0; x < fb_width; x++) {
            uint8_t *px = row + x * 3;
            uint8_t r = px[0], g = px[1], b = px[2];

            // 1. 获取Bayer阈值（归一化到0~7范围，匹配5/6bit量化步长）
            uint8_t threshold = bayer8x8[(y & 7) * 8 + (x & 7)] >> 5; // 0~7

            // 2. 应用阈值偏移（模拟误差扩散的视觉效果）
            int32_t r_adj = r + quant_bias[threshold];
            int32_t g_adj = g + quant_bias[threshold];
            int32_t b_adj = b + quant_bias[threshold];

            // 3. 模拟RGB565量化并还原
            uint8_t r5 = (r_adj > 255) ? 31 : (r_adj >> 3);
            uint8_t g6 = (g_adj > 255) ? 63 : (g_adj >> 2);
            uint8_t b5 = (b_adj > 255) ? 31 : (b_adj >> 3);

            px[0] = (r5 << 3) | (r5 >> 2);
            px[1] = (g6 << 2) | (g6 >> 4);
            px[2] = (b5 << 3) | (b5 >> 2);
        }
    }
}


// ===============================================================================
// 绘制文字（临时实现）
// ===============================================================================

uint8_t *_get_glyph(uint32_t utf32, uint8_t *font_width, uint8_t *font_height) {
    if(utf32 < 127) {
        *font_width = 6;
        *font_height = 12;
        return ASCII_6_12[utf32 - 32];
    }
    else {
        int32_t index = binary_search(UTF32_LUT_SORTED, UTF32_LUT_INDEXS, GLYPH_CHAR_NUM, utf32);
        if (index >= 0 && index < 7445) {
            *font_width = 12;
            *font_height = 12;
            return GB2312_12_12[index];
        }
        else {
            return NULL;
        }
    }
}

// 显示汉字
void fb_draw_char(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    int32_t x, int32_t y, uint8_t *glyph, uint8_t font_width, uint8_t font_height,
    uint8_t red, uint8_t green, uint8_t blue
) {
    int32_t row_bytes = (font_height + 8 - 1) / 8;
    int32_t col_bytes = font_width;
    for (int32_t j = 0; j < row_bytes; j++) {
        int32_t bits = (j == (row_bytes-1)) ? (8 - ((8 * row_bytes) % font_height)) : 8;
        for (int32_t i = 0; i < col_bytes; i++) {
            uint8_t g = glyph[j * col_bytes + i];
            for (int32_t b = 0; b < bits; b++) {
                int32_t px = x + i;
                int32_t py = y + j*8 + b;
                if (px < 0 || px >= fb_width || py < 0 || py >= fb_height) continue;
                if ((g >> b) & 0x1) {
                    set_pixel(frame_buffer, fb_width, fb_height, px, py, red, green, blue);
                }
                else {
                    add_pixel(frame_buffer, fb_width, fb_height, px, py, 0, 0, 0);
                }
            }
        }
    }
}



// 绘制一行文本
void fb_draw_textline(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    wchar_t *line, int32_t x, int32_t y, uint8_t red, uint8_t green, uint8_t blue
) {
    uint32_t x_pos = x;
    uint32_t y_pos = y;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = _get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = _get_glyph(12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos + font_width >= fb_width) {
            break;
        }
        fb_draw_char(frame_buffer, fb_width, fb_height,
            x_pos, y_pos, glyph, font_width, font_height, red, green, blue);
        x_pos += font_width;
    }
}


// 绘制一行文本（居中）
void fb_draw_textline_centered(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    wchar_t *line, int32_t cx, int32_t cy, uint8_t red, uint8_t green, uint8_t blue
) {
    // 第一遍扫描：计算文本渲染长度
    int32_t total_width = 0;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = _get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = _get_glyph(12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        total_width += font_width;
    }

    // 第二遍扫描：渲染
    int32_t x_pos = cx - (total_width/2);
    int32_t y_pos = cy - 6;
    if (y_pos < 0 || y_pos + 6 > fb_height) return;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = _get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            glyph = _get_glyph(12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos < 0) {
            x_pos += font_width;
            continue;
        }
        else if (x_pos + font_width > fb_width) {
            break;
        }
        fb_draw_char(frame_buffer, fb_width, fb_height,
            x_pos, y_pos, glyph, font_width, font_height, red, green, blue);
        x_pos += font_width;
    }
}






// ===============================================================================
// 绘制基本形状
// ===============================================================================

void draw_line(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float x1, float y1, float x2, float y2, float line_width, uint8_t r, uint8_t g, uint8_t b
) {
    if (line_width <= 0.0f) return;

    // 确保颜色在 [0, 255] 范围内
    uint8_t cr = MAX(0, MIN(255, r));
    uint8_t cg = MAX(0, MIN(255, g));
    uint8_t cb = MAX(0, MIN(255, b));

    float dx = x2 - x1;
    float dy = y2 - y1;
    float len_sq = dx * dx + dy * dy;

    // 退化为点：绘制圆形
    if (len_sq == 0.0f) { // TODO 与0比较
        float radius = line_width / 2.0f;
        float r_sq = radius * radius;
        int32_t xMin = MAX(0, (int32_t)floorf(x1 - radius));
        int32_t xMax = MIN(fb_width - 1, (int32_t)ceilf(x1 + radius));
        int32_t yMin = MAX(0, (int32_t)floorf(y1 - radius));
        int32_t yMax = MIN(fb_height - 1, (int32_t)ceilf(y1 + radius));

        for (int32_t y = yMin; y <= yMax; y++) {
            for (int32_t x = xMin; x <= xMax; x++) {
                float dist_sq = (float)((x - x1) * (x - x1) + (y - y1) * (y - y1));
                if (dist_sq <= r_sq) {
                    add_pixel(frame_buffer, fb_width, fb_height, x, y, cr, cg, cb);
                }
            }
        }
        return;
    }

    float len = sqrtf(len_sq);
    float inv_len = 1.0f / len;
    float nx = -dy * inv_len; // 法向量（垂直于线段）
    float ny = dx * inv_len;

    float half_w = line_width / 2.0f;

    // 包围盒（含线宽）
    int32_t xMin = MAX(0, (int32_t)floorf(MIN(x1, x2) - half_w));
    int32_t xMax = MIN(fb_width - 1, (int32_t)ceilf(MAX(x1, x2) + half_w));
    int32_t yMin = MAX(0, (int32_t)floorf(MIN(y1, y2) - half_w));
    int32_t yMax = MIN(fb_height - 1, (int32_t)ceilf(MAX(y1, y2) + half_w));

    for (int32_t y = yMin; y <= yMax; y++) {
        for (int32_t x = xMin; x <= xMax; x++) {
            // 计算点 (x, y) 到线段的有符号距离
            int32_t px = x - x1;
            int32_t py = y - y1;

            // 投影长度（参数 t）
            float t = (float)(px * dx + py * dy) / len_sq;
            int32_t closest_x = x1;
            int32_t closest_y = y1;

            if (t < 0.0f) {
                closest_x = x1;
                closest_y = y1;
            }
            else if (t > 1.0f) {
                closest_x = x2;
                closest_y = y2;
            }
            else {
                closest_x = x1 + (int32_t)(t * dx);
                closest_y = y1 + (int32_t)(t * dy);
            }

            float dist = sqrtf((float)((x - closest_x) * (x - closest_x) + (y - closest_y) * (y - closest_y)));

            if (dist > half_w) continue;

            // 抗锯齿：边缘平滑过渡
            float alpha = 1.0f;
            float edge_fade = 1.0; // 像素，可调
            if (dist > half_w - edge_fade) {
                alpha = (half_w - dist) / edge_fade;
                alpha = MAX(0.0f, MIN(1.0f, alpha));
            }
            add_pixel(frame_buffer, fb_width, fb_height, x, y, cr * alpha, cg * alpha, cb * alpha);
        }
    }
}


void draw_circle_outline(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float cx, float cy, float radius, float line_weight, uint8_t red, uint8_t green, uint8_t blue
) {
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
                add_pixel(frame_buffer, fb_width, fb_height, x, y, r, g, b);
            }
        }
    }
}

// 画实心圆形
void draw_circle(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float cx, float cy, float radius, uint8_t red, uint8_t green, uint8_t blue
) {
    if (radius <= 0.0f) return;

    uint8_t r = MAX(0, MIN(255, red));
    uint8_t g = MAX(0, MIN(255, green));
    uint8_t b = MAX(0, MIN(255, blue));

    float outerRSq = radius * radius;

    // 包围盒（含线宽）
    int32_t xMin = MAX(0, (int32_t)floorf(cx - radius));
    int32_t xMax = MIN(fb_width - 1, (int32_t)ceilf(cx + radius));
    int32_t yMin = MAX(0, (int32_t)floorf(cy - radius));
    int32_t yMax = MIN(fb_height - 1, (int32_t)ceilf(cy + radius));

    for (int32_t y = yMin; y <= yMax; y++) {
        for (int32_t x = xMin; x <= xMax; x++) {
            float dx = x - cx;
            float dy = y - cy;
            float distSq = dx * dx + dy * dy;

            // 判断是否在区域内
            if (distSq < outerRSq) {
                add_pixel(frame_buffer, fb_width, fb_height, x, y, r, g, b);
            }
        }
    }
}

// 画实心矩形
void draw_rect(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float x0, float y0, float width, float height, uint8_t red, uint8_t green, uint8_t blue
) {
    uint8_t r = MAX(0, MIN(255, red));
    uint8_t g = MAX(0, MIN(255, green));
    uint8_t b = MAX(0, MIN(255, blue));
    for (int32_t y = y0; y < y0+height; y++) {
        for (int32_t x = x0; x < x0+width; x++) {
            set_pixel(frame_buffer, fb_width, fb_height, x, y, r, g, b);
        }
    }
}

// 地面填充纯黑色
void draw_horizon(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f,
    float sun_alt, int32_t landscape_index
) {
    float margin = LINGLONG_HORIZON_BLUR_MARGIN;
    float k = MAX(0.2f, sinf(to_rad_float(sun_alt)) * 1.5f);
    float hx = 0.0f;
    float hy = 0.0f;
    float hz = 0.0f;
    for (int32_t y = 0; y < fb_height; y++) {
        for (int32_t x = 0; x < fb_width; x++) {
            fisheye_unproject(x, y, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &hx, &hy, &hz);
            if (hz < 0) {
                if (landscape_index == 0) {
                    set_pixel(frame_buffer, fb_width, fb_height, x, y, 0, 0, 0);
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
                    horizontal_to_screen_xy(-azi, -alt, 300, 300, 300, &tx, &ty); // R略小于地面贴图的半径，注意修改其中的center_x/y

                    uint8_t r = get_pixel_channel(landscape_texture_rgb, landscape_texture_width, floorf(tx), floorf(ty), 0);
                    uint8_t g = get_pixel_channel(landscape_texture_rgb, landscape_texture_width, floorf(tx), floorf(ty), 1);
                    uint8_t b = get_pixel_channel(landscape_texture_rgb, landscape_texture_width, floorf(tx), floorf(ty), 2);

                    r = (uint8_t)MIN(255.0f, ((float)r * k));
                    g = (uint8_t)MIN(255.0f, ((float)g * k));
                    b = (uint8_t)MIN(255.0f, ((float)b * k));

                    set_pixel(frame_buffer, fb_width, fb_height, x, y, r, g, b);
                }
            }
            else if (landscape_index == 0 && hz <= margin && hz >= 0.0f) {
                float c = 0.0f;
                float t = hz / margin;
                scale_pixel(frame_buffer, fb_width, fb_height, x, y, t);
            }
        }
    }
}


// ===============================================================================
// 绘制坐标系
// ===============================================================================

/**
 * 绘制赤道坐标系下的子午圈或纬度圈（投影到地平屏幕）
 * @param frame_buffer - 帧缓冲区
 * @param fb_width - 缓冲区宽度
 * @param fb_height - 缓冲区高度
 * @param is_meridian - 1: 子午圈（固定RA）；0: 纬度圈（固定Dec）
 * @param ra_hours - 子午圈的赤经（小时），仅当 is_meridian=1 时有效
 * @param dec_deg - 纬度圈的赤纬（度），仅当 is_meridian=0 时有效
 * @param line_weight - 线宽（像素），建议 ≥1
 * @param colorR, colorG, colorB - RGB 颜色分量 [0-255]
 * @param year, month, day, hour, minute, second, timezone, longitude, latitude - 观测参数
 */
void draw_celestial_circle(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f,
    int32_t is_meridian, float ra_hours, float dec_deg,
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

    // 子午圈：固定 RA，遍历 Dec ∈ [-90°, +90°]
    if (is_meridian) {
        if (ra_hours < 0.0f || ra_hours > 24.0f) return;
        float ra_deg = fmodf(ra_hours, 24.0f) * 15.0f;
        double alt = 0.0;
        double azi = 0.0;
        for (int32_t i = 0; i <= LINGLONG_CELESTIAL_CIRCLE_POINTS; i++) {
            float dec = -90.0f + (180.0f * (float)i / (float)LINGLONG_CELESTIAL_CIRCLE_POINTS);
            equatorial_to_horizontal((double)ra_deg, (double)dec, year, month, day, hour, minute, second, timezone, longitude, latitude, &azi, &alt);

            if (alt > 0.0) {
                float x = 0.0f;
                float y = 0.0f;
                fisheye_project(azi, alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &x, &y);

                if (i == 0) {
                    x_0 = x;
                    y_0 = y;
                    alt_0 = alt;
                }
                
                if (alt_prev > 0.0 && alt > 0.0) {
                    draw_line(frame_buffer, fb_width, fb_height, x_prev, y_prev, x, y, line_weight, colorR, colorG, colorB);
                }

                x_prev = x;
                y_prev = y;
                alt_prev = alt;
            }
        }
        if (alt_0 > 0.0 && alt > 0.0) {
            draw_line(frame_buffer, fb_width, fb_height, x_prev, y_prev, x_0, y_0, line_weight, colorR, colorG, colorB);
        }
    }
    // 纬度圈：固定 Dec，遍历 RA ∈ [0h, 24h)
    else {
        if (dec_deg < -90.0f || dec_deg > 90.0f) return;
        double alt = 0.0;
        double azi = 0.0;
        for (int32_t i = 0; i <= LINGLONG_CELESTIAL_CIRCLE_POINTS; i++) {
            float ra_h = 24.0f * (float)i / (float)LINGLONG_CELESTIAL_CIRCLE_POINTS;
            float ra_deg = fmodf(ra_h, 24.0f) * 15.0f;

            equatorial_to_horizontal((double)ra_deg, (double)dec_deg, year, month, day, hour, minute, second, timezone, longitude, latitude, &azi, &alt);

            if (alt > 0.0) {
                float x = 0.0f;
                float y = 0.0f;
                fisheye_project(azi, alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &x, &y);

                if (i == 0) {
                    x_0 = x;
                    y_0 = y;
                    alt_0 = alt;
                }

                if (alt_prev > 0.0 && alt > 0.0) {
                    draw_line(frame_buffer, fb_width, fb_height, x_prev, y_prev, x, y, line_weight, colorR, colorG, colorB);
                }

                x_prev = x;
                y_prev = y;
                alt_prev = alt;
            }
        }
        if (alt_0 > 0.0 && alt > 0.0) {
            draw_line(frame_buffer, fb_width, fb_height, x_prev, y_prev, x_0, y_0, line_weight, colorR, colorG, colorB);
        }
    }
}


void draw_ecliptic_circle(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f,
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
            fisheye_project(azi, alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &x, &y);

            if (i == 0) {
                x_0 = x;
                y_0 = y;
                alt_0 = alt;
            }
            
            if (alt_prev > eps && alt > eps) {
                draw_line(frame_buffer, fb_width, fb_height, x_prev, y_prev, x, y, line_weight, colorR, colorG, colorB);
            }

            x_prev = x;
            y_prev = y;
            alt_prev = alt;
        }
    }
    if (alt_0 > eps && alt > eps) {
        draw_line(frame_buffer, fb_width, fb_height, x_prev, y_prev, x_0, y_0, line_weight, colorR, colorG, colorB);
    }

}


// ===============================================================================
// 绘制天体
// ===============================================================================

void render_sun(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    float sun_proj_x, float sun_proj_y, float sun_altitude_deg
) {

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

            add_pixel(frame_buffer, fb_width, fb_height, px, py, totalR, totalG, totalB);
        }
    }
}


// 基于球面几何晨昏线的月相绘制（带物理合理的软边缘）
void render_moon(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude
) {
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
    fisheye_project(moon_azi, moon_alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &moon_scr_x, &moon_scr_y);

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
    fisheye_project(northdelta_azi, northdelta_alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &northdelta_scr_x, &northdelta_scr_y);

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


            int32_t idx = (py * fb_width + px) * 3;

            // 构造单位球面上的表面点（归一化位置向量 = 法向量）
            float len = sqrtf(dx * dx + dy * dy + r2 - dx * dx - dy * dy); // = moonRadius
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

            int32_t x0 = (int32_t)floorf(tx);
            int32_t y0 = (int32_t)floorf(ty);
            int32_t x1 = (x0 + 1) % moon_texture_width;
            int32_t y1 = MIN(y0 + 1, moon_texture_height - 1);

            float wx = tx - (float)x0;
            float wy = ty - (float)y0;

            uint8_t c00_r = get_pixel_channel(moon_texture_rgb, moon_texture_width, x0, y0, 0);
            uint8_t c00_g = get_pixel_channel(moon_texture_rgb, moon_texture_width, x0, y0, 1);
            uint8_t c00_b = get_pixel_channel(moon_texture_rgb, moon_texture_width, x0, y0, 2);

            uint8_t c10_r = get_pixel_channel(moon_texture_rgb, moon_texture_width, x1, y0, 0);
            uint8_t c10_g = get_pixel_channel(moon_texture_rgb, moon_texture_width, x1, y0, 1);
            uint8_t c10_b = get_pixel_channel(moon_texture_rgb, moon_texture_width, x1, y0, 2);

            uint8_t c01_r = get_pixel_channel(moon_texture_rgb, moon_texture_width, x0, y1, 0);
            uint8_t c01_g = get_pixel_channel(moon_texture_rgb, moon_texture_width, x0, y1, 1);
            uint8_t c01_b = get_pixel_channel(moon_texture_rgb, moon_texture_width, x0, y1, 2);

            uint8_t c11_r = get_pixel_channel(moon_texture_rgb, moon_texture_width, x1, y1, 0);
            uint8_t c11_g = get_pixel_channel(moon_texture_rgb, moon_texture_width, x1, y1, 1);
            uint8_t c11_b = get_pixel_channel(moon_texture_rgb, moon_texture_width, x1, y1, 2);

            // bilinear interpolation
            float r0 = (float)c00_r + wx * (float)(c10_r - c00_r);
            float g0 = (float)c00_g + wx * (float)(c10_g - c00_g);
            float b0 = (float)c00_b + wx * (float)(c10_b - c00_b);

            float r1 = (float)c01_r + wx * (float)(c11_r - c01_r);
            float g1 = (float)c01_g + wx * (float)(c11_g - c01_g);
            float b1 = (float)c01_b + wx * (float)(c11_b - c01_b);

            float texR = r0 + wy * (r1 - r0);
            float texG = g0 + wy * (g1 - g0);
            float texB = b0 + wy * (b1 - b0);

            // 应用光照强度
            float r = roundf(texR * intensity);
            float g = roundf(texG * intensity);
            float b = roundf(texB * intensity);

            add_pixel(frame_buffer, fb_width, fb_height, px, py, moonWeight * r, moonWeight * g, moonWeight * b);
        }
    }
}


void draw_star(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    float sx, float sy, float magnitude, float radius, uint8_t red, uint8_t green, uint8_t blue
) {
    if (sx < 0 || sx >= fb_width || sy < 0 || sy >= fb_height) {
        return;
    }

    int32_t maxGlowRadius = LINGLONG_STAR_GLOW_RADIUS; // 光晕最大半径（像素）

    float starLumBase = magnitude_to_relative_luminance(magnitude);

    // 获取背景亮度用于对比度抑制
    int32_t bgIdx = ((int32_t)sy * fb_width + (int32_t)sx) * 3;
    float bgR = frame_buffer[bgIdx] / 255.0f;
    float bgG = frame_buffer[bgIdx + 1] / 255.0f;
    float bgB = frame_buffer[bgIdx + 2] / 255.0f;
    float bgLum = get_luminance(bgR, bgG, bgB);

    // 遍历光晕区域（正方形包围圆）
    int32_t R = (int32_t)radius + maxGlowRadius;
    for (int32_t dy = -R; dy <= R; dy++) {
        for (int32_t dx = -R; dx <= R; dx++) {
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist > (float)R) continue;

            int32_t px = (int32_t)roundf(sx + (float)dx);
            int32_t py = (int32_t)roundf(sy + (float)dy);
            if (px < 0 || px >= fb_width || py < 0 || py >= fb_height) continue;

            int32_t idx = (py * fb_width + px) * 3;

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
            frame_buffer[idx + 0] = (uint8_t)MIN(255, frame_buffer[idx + 0] + r);
            frame_buffer[idx + 1] = (uint8_t)MIN(255, frame_buffer[idx + 1] + g);
            frame_buffer[idx + 2] = (uint8_t)MIN(255, frame_buffer[idx + 2] + b);
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
        sun_air_mass = (enable_opt_lut) ?
                        AIR_MASS_LUT[(int32_t)floorf(sun_zenith_deg)] :
                        (1.0f / (cosf(sun_zenith) + 0.50572f * powf(96.07995f - sun_zenith_deg, -1.6364f)));
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
        view_air_mass = (enable_opt_lut) ? AIR_MASS_LUT[(int32_t)floorf(view_zenith_deg)] :
                        (1.0f / (cosf(view_zenith) + 0.50572f * powf(96.07995f - view_zenith_deg, -1.6364f)));
    }
    else {
        view_air_mass = 40.0f;
    }


    // 观察方向与太阳方向夹角余弦
    float cos_theta = ray_norm_x * sun_norm_x + ray_norm_y * sun_norm_y + ray_norm_z * sun_norm_z;


    // 米氏散射相函数
    float mie_phase = powf(1.0f + MIE_G * MIE_G - 2.0f * MIE_G * cos_theta, -1.5f);


    // 瑞利散射相函数
    float rayleigh_phase = (1.0f + cos_theta * cos_theta) * 0.75f;


    // 计算视线方向的大气密度系数：直觉上来看，密度越大，对散射的贡献越大
    float rz = MAX(0.01, ray_norm_z); // 避免除零
    float density_factor = expf(-rz / ATMOSPHERE_HEIGHT);


    // 瑞利散射系数：分波长（RGB通道）计算
    // rayleigh_beta_at_wavelength[i] = RAYLEIGH_BETA * Math.pow(rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[i], 4);
    float rayleigh_ref_wavelength = RAYLEIGH_WAVELENGTH_FACTORS[1]; // G分量作为参考波长
    float wl_0 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[0]; // R
    float wl_1 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[1]; // G
    float wl_2 = rayleigh_ref_wavelength / RAYLEIGH_WAVELENGTH_FACTORS[2]; // B

    float rayleigh_beta_0 = RAYLEIGH_BETA * wl_0 * wl_0 * wl_0 * wl_0; // R
    float rayleigh_beta_1 = RAYLEIGH_BETA * wl_1 * wl_1 * wl_1 * wl_1; // G
    float rayleigh_beta_2 = RAYLEIGH_BETA * wl_2 * wl_2 * wl_2 * wl_2; // B


    // 米氏散射系数：地平线附近，气溶胶浓度剧增（用viewAirMass模拟），米氏散射贡献显著上升
    float mie_beta = MIN(MIE_BETA_MAX, MIE_BETA_BASE * view_air_mass);


    // 计算每个通道的散射强度
    
    float mie_contrib = mie_beta * mie_phase;

    float rayleigh_contrib_0 = rayleigh_beta_0 * rayleigh_phase;
    float rayleigh_contrib_1 = rayleigh_beta_1 * rayleigh_phase;
    float rayleigh_contrib_2 = rayleigh_beta_2 * rayleigh_phase;

    float attn_scale = (enable_opt_lut) ?
                        ATTN_SCALE_LUT[(int32_t)floorf(sun_altitude_deg) + 90] :
                        ((sun_altitude_deg >= 0) ? (expf(-powf((sun_altitude_deg / 20.0f), 4.0f)) + 0.01f) : (1.0f));

    float attn_0 = expf(-(rayleigh_beta_0 + mie_beta) * view_air_mass * attn_scale);
    float attn_1 = expf(-(rayleigh_beta_1 + mie_beta) * view_air_mass * attn_scale);
    float attn_2 = expf(-(rayleigh_beta_2 + mie_beta) * view_air_mass * attn_scale);

    float ozone_transmittance_0 = expf(-OZONE_ABSORPTION[0] * sun_air_mass * 0.8);
    float ozone_transmittance_1 = expf(-OZONE_ABSORPTION[1] * sun_air_mass * 0.8);
    float ozone_transmittance_2 = expf(-OZONE_ABSORPTION[2] * sun_air_mass * 0.8);

    float scattered_0 = (rayleigh_contrib_0 + mie_contrib) * attn_0 * ozone_transmittance_0 * density_factor;
    float scattered_1 = (rayleigh_contrib_1 + mie_contrib) * attn_1 * ozone_transmittance_1 * density_factor;
    float scattered_2 = (rayleigh_contrib_2 + mie_contrib) * attn_2 * ozone_transmittance_2 * density_factor;

    // 太阳落到地平线以下时，进一步衰减散射光
    float night_attn = (enable_opt_lut) ?
                        NIGHT_ATTN_LUT[(int32_t)floorf(sun_altitude_deg) + 90] :
                        ((sun_altitude_deg >= 0) ? (1.0f) : MAX(0.0f, expf(0.016f * sun_altitude_deg)));
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
    float sun_azimuth = 90.0f + atan2f(sun_norm_y, sun_norm_x) * 180.0f / M_PI;

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
    float oz_factor = 0.16f;
    float oz_r = expf(-OZONE_ABSORPTION[0] * Lsun * oz_factor);
    float oz_g = expf(-OZONE_ABSORPTION[1] * Lsun * oz_factor);
    float oz_b = expf(-OZONE_ABSORPTION[2] * Lsun * oz_factor);

    float scattered_0 = (sum_R_r * rayleigh_beta_0 * rayleigh_phase + sum_M_r * mie_beta * mie_phase) * oz_r;
    float scattered_1 = (sum_R_g * rayleigh_beta_1 * rayleigh_phase + sum_M_g * mie_beta * mie_phase) * oz_g;
    float scattered_2 = (sum_R_b * rayleigh_beta_2 * rayleigh_phase + sum_M_b * mie_beta * mie_phase) * oz_b;

    // 全局增益、限幅、输出
    const float global_gain = 2.0;
    *red   = MIN(1.0f, scattered_0 * global_gain);
    *green = MIN(1.0f, scattered_1 * global_gain);
    *blue  = MIN(1.0f, scattered_2 * global_gain);
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
    else {
        *red   = 0.0f;
        *green = 0.0f;
        *blue  = 0.0f;
    }
}


// ===============================================================================
// 渲染整个天空
// ===============================================================================

void render_sky(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double timezone, double longitude, double latitude,
    int32_t downsampling_factor,     // 降采样因子（设为0为自动，建议设为2）
    int32_t enable_opt_sym,          // 是否启用基于对称性的渲染优化（以画质为代价）
    int32_t enable_opt_lut,          // 是否启用查找表计算加速（以画质为代价）
    int32_t enable_opt_bilinear,     // 是否启用双线性插值以优化画质

    int32_t sky_model,               // 选择天空模型（0-不启用散射；1-简单散射模型；2-西田算法）
    int32_t landscape_index,         // 选择地景贴图（0-不启用，地景设为纯黑；其他-地景贴图序号）
    int32_t enable_equatorial_coord, // 是否启用赤道坐标圈
    int32_t enable_horizontal_coord, // 是否启用地平坐标圈
    int32_t enable_star_burst,       // 是否启用星芒效果
    int32_t enable_star_name,        // 是否显示恒星名称
    int32_t enable_planet,           // 是否显示大行星
    int32_t enable_planet_name,      // 是否显示大行星名称
    int32_t enable_ecliptic_circle   // 是否显示黄道
) {

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
    fisheye_project(sun_azi, sun_alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &sun_proj_x, &sun_proj_y);

    // 天空绘制范围的屏幕坐标
    // int32_t y1 = (int32_t)floorf(center_y - sky_radius);
    // int32_t y2 = (int32_t)floorf(center_y + sky_radius);
    // int32_t x1 = (int32_t)floorf(center_x - sky_radius);
    // int32_t x2 = (int32_t)floorf(center_x + sky_radius);
    int32_t y1 = 0;
    int32_t y2 = fb_height;
    int32_t x1 = 0;
    int32_t x2 = fb_width;

    // 大气散射
    if (sky_model > 0) {
        // 首先根据太阳高度自适应设置降采样因子
        int32_t _downsampling_factor = downsampling_factor;
        if (downsampling_factor == 0) {
            if (sun_alt < -18.0) {
                _downsampling_factor = (enable_opt_bilinear && !enable_opt_sym) ? 32 : 8;
            }
            else {
                _downsampling_factor = (enable_opt_bilinear && !enable_opt_sym) ? 8 : 2;
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
                        fisheye_unproject((float)x, (float)y, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &ray_x, &ray_y, &ray_z);

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
                                set_pixel(frame_buffer, fb_width, fb_height, (x+i), (y+j), rr, gg, bb);
                                set_pixel(frame_buffer, fb_width, fb_height, (xx + i + offset_x), (yy + j + offset_y), rr, gg, bb);
                            }
                        }
                    }
                }
            }
        }
        else {
            for (int32_t y = y1; y < y2; y += _downsampling_factor) {
                for (int32_t x = x1; x < x2; x += _downsampling_factor) {
                    // 观察者到该像素的方向向量（从屏幕坐标系转回地平天球的笛卡尔坐标系）
                    float ray_x = 0.0f;
                    float ray_y = 0.0f;
                    float ray_z = 0.0f;
                    fisheye_unproject((float)x, (float)y, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &ray_x, &ray_y, &ray_z);

                    float red = 0.0f;
                    float green = 0.0f;
                    float blue = 0.0f;
                    calculate_scattered_pixel(ray_x, ray_y, ray_z, sun_x, sun_y, sun_z, &red, &green, &blue, enable_opt_lut, sky_model);

                    if (enable_opt_bilinear) {
                        set_pixel(frame_buffer, fb_width, fb_height, x, y, (uint8_t)(red * 255.0f), (uint8_t)(green * 255.0f), (uint8_t)(blue  * 255.0f));
                    }
                    else {
                        for (int32_t i = 0; i < _downsampling_factor; i++) {
                            for (int32_t j = 0; j < _downsampling_factor; j++) {
                                add_pixel(frame_buffer, fb_width, fb_height,
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

                    int32_t r2 = (x-center_x)*(x-center_x) + (y-center_y)*(y-center_y);
                    if (r2 > (sky_radius+C)*(sky_radius+C)) continue;

                    int32_t idx_11 = ((y+0) * fb_width + (x+0)) * 3;
                    int32_t idx_12 = (x+C < fb_width)  ? (((y+0) * fb_width + (x+C)) * 3) : idx_11;
                    int32_t idx_21 = (y+C < fb_height) ? (((y+C) * fb_width + (x+0)) * 3) : idx_11;
                    int32_t idx_22 = ((x+C < fb_width) || (y+C < fb_height)) ? (((y+C) * fb_width + (x+C)) * 3) : idx_11;

                    uint8_t r11 = frame_buffer[idx_11 + 0]; uint8_t r12 = frame_buffer[idx_12 + 0]; uint8_t r21 = frame_buffer[idx_21 + 0]; uint8_t r22 = frame_buffer[idx_22 + 0];
                    uint8_t g11 = frame_buffer[idx_11 + 1]; uint8_t g12 = frame_buffer[idx_12 + 1]; uint8_t g21 = frame_buffer[idx_21 + 1]; uint8_t g22 = frame_buffer[idx_22 + 1];
                    uint8_t b11 = frame_buffer[idx_11 + 2]; uint8_t b12 = frame_buffer[idx_12 + 2]; uint8_t b21 = frame_buffer[idx_21 + 2]; uint8_t b22 = frame_buffer[idx_22 + 2];

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
                            int32_t idx = ((y+i) * fb_width + (x+j)) * 3;
                            float v = (float)i / (float)C;
                            float u = (float)j / (float)C;
                            frame_buffer[idx + 0] = MIN(255, (uint8_t)((1-u)*(1-v)*(float)r11 + u*(1-v)*(float)r12 + (1-u)*v*(float)r21 + u*v*(float)r22));
                            frame_buffer[idx + 1] = MIN(255, (uint8_t)((1-u)*(1-v)*(float)g11 + u*(1-v)*(float)g12 + (1-u)*v*(float)g21 + u*v*(float)g22));
                            frame_buffer[idx + 2] = MIN(255, (uint8_t)((1-u)*(1-v)*(float)b11 + u*(1-v)*(float)b12 + (1-u)*v*(float)b21 + u*v*(float)b22));
                        }
                    }
                }
            }
        }
    }

    // 绘制太阳
    render_sun(
        frame_buffer, fb_width, fb_height,
        sky_radius, center_x, center_y,
        sun_proj_x, sun_proj_y, sun_alt);


    // 计算月球位置并绘制
    render_moon(
        frame_buffer, fb_width, fb_height,
        sky_radius, center_x, center_y,
        view_alt, view_azi, view_roll, f,
        year, month, day, hour, minute, second, timezone, longitude, latitude);

    // 绘制星芒
    if (enable_star_burst && sun_alt > 0) {
        star_burst_filter(frame_buffer, fb_width, fb_height, sun_proj_x, sun_proj_y);
    }

    // 绘制恒星
    float mag_offset = -2.0f;
    for (int32_t i = 0; i < STARS_NUM; i++) {
        float *star_item = STARS[i];
        double alt = 0.0;
        double azi = 0.0;
        equatorial_to_horizontal(
            ra_hms_to_deg(star_item[0], star_item[1], star_item[2]),
            dec_dms_to_decimal(star_item[3], star_item[4], star_item[5]),
            year, month, day, hour, minute, second, timezone, longitude, latitude, &azi, &alt);
        float mag = 0.5f + mag_offset + star_item[6];

        float sx = 0.0f;
        float sy = 0.0f;
        fisheye_project(azi, alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &sx, &sy);

        draw_star(frame_buffer, fb_width, fb_height, sky_radius, center_x, center_y, sx, sy, mag, 1, 255, 255, 255);

        if (enable_star_name) {
            fb_draw_textline(frame_buffer, fb_width, fb_height, STAR_NAME[i], sx+3, sy+3, 250, 250, 250);
        }
    }

    // 绘制黄道
    if (enable_ecliptic_circle) {
        draw_ecliptic_circle(
            frame_buffer, fb_width, fb_height,
            sky_radius, center_x, center_y,
            view_alt, view_azi, view_roll, f,
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
            fisheye_project(planet_azi, planet_alt, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &planet_proj_x, &planet_proj_y);

            draw_star(frame_buffer, fb_width, fb_height, sky_radius, center_x, center_y, planet_proj_x, planet_proj_y,
                0.0f, PLANET_RADIUS[i], PLANET_COLOR_R[i], PLANET_COLOR_G[i], PLANET_COLOR_B[i]);

            if (enable_planet_name) {
                fb_draw_textline(frame_buffer, fb_width, fb_height, PLANET_NAME[i], planet_proj_x+3, planet_proj_y+3, PLANET_COLOR_R[i], PLANET_COLOR_G[i], PLANET_COLOR_B[i]);
            }
        }
    }

    // 绘制赤道坐标网格
    if (enable_equatorial_coord) {
        // 绘制赤道天球子午圈
        for (int32_t i = 0; i < 24; i += 2) {
            int32_t line_width = (i == 0 || i == 12) ? 5 : 3;
            draw_celestial_circle(
                frame_buffer, fb_width, fb_height,
                sky_radius, center_x, center_y,
                view_alt, view_azi, view_roll, f,
                1, (float)i, 0.0f,
                line_width, 8, 16, 32,
                year, month, day, hour, minute, second, timezone, longitude, latitude
            );
        }
        // 绘制赤道天球等纬度圈
        for (int32_t i = -90; i < 90; i += 10) {
            int32_t line_width = (i == 0) ? 5 : 3;
            draw_celestial_circle(
                frame_buffer, fb_width, fb_height,
                sky_radius, center_x, center_y,
                view_alt, view_azi, view_roll, f,
                0, 0.0f, (float)i,
                line_width, 8, 16, 32,
                year, month, day, hour, minute, second, timezone, longitude, latitude
            );
        }
    }

    // 绘制地平坐标网格
    if (enable_horizontal_coord) {
        float label_x = 0.0f;
        float label_y = 0.0f;
        fisheye_project(0, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &label_x, &label_y);
        fb_draw_textline_centered(frame_buffer, fb_width, fb_height, L"北", label_x, label_y, 255, 0, 0);
        fisheye_project(90, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &label_x, &label_y);
        fb_draw_textline_centered(frame_buffer, fb_width, fb_height, L"东", label_x, label_y, 255, 0, 0);
        fisheye_project(180, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &label_x, &label_y);
        fb_draw_textline_centered(frame_buffer, fb_width, fb_height, L"南", label_x, label_y, 255, 0, 0);
        fisheye_project(270, 6, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, &label_x, &label_y);
        fb_draw_textline_centered(frame_buffer, fb_width, fb_height, L"西", label_x, label_y, 255, 0, 0);

    }

    // 绘制地景（天空投影圆盘之外的部分）
    draw_horizon(frame_buffer, fb_width, fb_height, sky_radius, center_x, center_y, view_alt, view_azi, view_roll, f, sun_alt, landscape_index);

}
