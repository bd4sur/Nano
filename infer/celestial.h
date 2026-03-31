#ifndef __NANO_CELESTIAL_H__
#define __NANO_CELESTIAL_H__

#ifdef __cplusplus
extern "C" {
#endif

// 若干视觉参数
#define LINGLONG_STAR_BURST_RADIUS       (60)
#define LINGLONG_STAR_BURST_DECAY        (0.94f)
#define LINGLONG_HORIZON_BLUR_MARGIN     (10.0f)
#define LINGLONG_CELESTIAL_CIRCLE_POINTS (180)
#define LINGLONG_ECLIPTIC_CIRCLE_POINTS  (360)
#define LINGLONG_SUN_RADIUS_MIN          (5.0f)
#define LINGLONG_SUN_GLOW_RADIUS         (6)
#define LINGLONG_MOON_RADIUS_MIN         (16.0f)
#define LINGLONG_STAR_GLOW_RADIUS        (0)


typedef struct {
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

void dithering_fs(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height);
void dithering_fast(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height);

void draw_line(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float x1, float y1, float x2, float y2, float line_width, uint8_t r, uint8_t g, uint8_t b
);

// 绘制一行文本（居中）
void fb_draw_textline_centered(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    wchar_t *line, int32_t cx, int32_t cy, uint8_t red, uint8_t green, uint8_t blue
);

void draw_circle(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float cx, float cy, float radius, uint8_t red, uint8_t green, uint8_t blue
);

void draw_rect(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    float x0, float y0, float width, float height, uint8_t red, uint8_t green, uint8_t blue
);

void fb_draw_textline(
    uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
    wchar_t *line, int32_t x, int32_t y, uint8_t red, uint8_t green, uint8_t blue
);



void linglong_init();


void render_sky(uint8_t *frame_buffer, int32_t fb_width, int32_t fb_height,
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
