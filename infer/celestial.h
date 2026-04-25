#ifndef __NANO_CELESTIAL_H__
#define __NANO_CELESTIAL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "graphics.h"

// 若干视觉参数
#define LINGLONG_STAR_BURST_RADIUS        (60)
#define LINGLONG_STAR_BURST_DECAY         (0.94f)
#define LINGLONG_HORIZON_BLUR_MARGIN      (10.0f)
#define LINGLONG_CELESTIAL_CIRCLE_POINTS  (180)
#define LINGLONG_HORIZONTAL_CIRCLE_POINTS (180)
#define LINGLONG_ECLIPTIC_CIRCLE_POINTS   (360)
#define LINGLONG_SUN_RADIUS_MIN           (3.0f)
#define LINGLONG_SUN_GLOW_RADIUS          (4)
#define LINGLONG_MOON_RADIUS_MIN          (16.0f)
#define LINGLONG_STAR_GLOW_RADIUS         (0)
// 是否开启动态地景：动态地景需动态分配大量堆内存，若内存紧张，可关闭。同时关闭透视转鱼眼功能，详见代码。
#define LINGLONG_ENABLE_DYNAMIC_LANDSCAPE (1)

typedef struct Linglong_Config {
    int32_t fb_width;
    int32_t fb_height;
    float sky_radius;
    float center_x;
    float center_y;
    float view_alt;
    float view_azi;
    float view_roll;
    float view_f;

    int32_t year;
    int32_t month;
    int32_t day;
    int32_t hour;
    int32_t minute;
    int32_t second;
    double timezone;
    double longitude;
    double latitude;

    int32_t downsampling_factor;     // 降采样因子（设为0为自动，建议设为2）
    int32_t enable_opt_sym;          // 是否启用基于对称性的渲染优化（以画质为代价）
    int32_t enable_opt_lut;          // 是否启用查找表计算加速（以画质为代价）
    int32_t enable_opt_bilinear;     // 是否启用双线性插值以优化画质
 
    int32_t projection;              // 投影算法（0-鱼眼；1-线性透视）
    int32_t sky_model;               // 选择天空模型（0-不启用散射；1-简单散射；2-一次散射；3-二次散射）
    int32_t landscape_index;         // 选择地景贴图（0-不启用，地景设为纯黑；其他-地景贴图序号）
    int32_t enable_equatorial_coord; // 是否启用赤道坐标圈
    int32_t enable_horizontal_coord; // 是否启用地平坐标圈（0-不启用；1-仅方位角文字；2-方位角+坐标圈）
    int32_t enable_star_burst;       // 是否启用星芒效果
    int32_t enable_star_name;        // 是否显示恒星名称（0-不显示；1-除行星；2-仅行星；3-全部）
    int32_t enable_planet;           // 是否显示大行星
    int32_t enable_ecliptic_circle;  // 是否显示黄道
    int32_t enable_att_indicator;    // 是否显示姿态指示标记

    // 以下与天空渲染无关（非render_sky参数）
    int32_t enable_imu;              // 是否启用IMU（使视角随机器姿态旋转）
} Linglong_Config;


void transform_euler_angles(float pitch_in, float roll_in, float yaw_in, float *pitch_out, float *roll_out, float *yaw_out);
void quaternion_to_euler(float q0, float q1, float q2, float q3, float *pitch, float *roll, float *yaw);

void dithering_fast(Nano_GFX *gfx);


void linglong_init(Linglong_Config *cfg);

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
);

#ifdef __cplusplus
}
#endif

#endif
