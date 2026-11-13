// ===============================================================================
// 实时海洋 + 海洋天空（web-ocean 移植）
//
// 本文件是 linglong.html 中 === [WebOcean] === 一节（JS 参考实现）到 C 的逐式移植，
// 保留原变量名（wo_ 前缀）与注释结构，便于与参考实现对照。参考实现的编号：
//   · 大气常数 / wo_medium / wo_raySphere …… 第 10686~10840 行
//   · 透射率 / 多重散射 / 天空视图 LUT …… 第 10861~11036 行
//   · AgX 色调映射 / scatter_model_6 …… 第 11073~11135 行
//   · 波列级数（JONSWAP 谱、分层抽样、烘焙）…… 第 11135~11480 行
//   · 海面求交 / 逐像素着色 / 地景绘制 …… 第 13064~13460 行
//
// 数值约定：
//   · LUT 构建与波场烘焙用 double（原实现是 JS Number，即 double）；
//   · 逐像素着色用 double 以保证与参考实现同量级的精度（软件渲染，性能暂不优化）；
//   · LUT / 波场以 float32 存储（原实现为 float32 + f16 上传量化，见文件头说明）。
// ===============================================================================

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "platform.h"
#include "graphics.h"
#include "celestial.h"
#include "celestial_ocean.h"
#include "celestial_night.h"

// ---- 设置界面档位表（单一真源：档位序号 → 显示名 + 物理量）----
// 观察高度档位：海洋天空/实时海洋与体积云（sky_model==4）共用同一张表。
// 云区 3.0~11.0 km，故 1000 m 仍在云下、4000 m 起进入低层云带、12000/20000 m 在云顶之上
// （用于“观察者漂浮在云上”的俯视视角；是否在云上由 ui_cloud_lowest_layer_base_km 判定）。
static const Linglong_Ocean_Option S_OCEAN_HEIGHT_OPTIONS[] = {
    { 0, L"5m",     5.0f },
    { 1, L"50m",   50.0f },
    { 2, L"200m",  200.0f },
    { 3, L"1000m", 1000.0f },
    { 4, L"4000m", 4000.0f },
    { 5, L"8000m", 8000.0f },
    { 6, L"12000m", 12000.0f },
    { 7, L"20000m", 20000.0f },
};
static const Linglong_Ocean_Option S_OCEAN_SWELL_HS_OPTIONS[] = {
    { 0, L"无",    0.0f },
    { 1, L"0.5m",  0.5f },
    { 2, L"1.0m",  1.0f },
    { 3, L"1.8m",  1.8f },
    { 4, L"2.5m",  2.5f },
    { 5, L"3.5m",  3.5f },
};
static const Linglong_Ocean_Option S_OCEAN_SWELL_LEN_OPTIONS[] = {
    { 0, L"30m",   30.0f },
    { 1, L"60m",   60.0f },
    { 2, L"90m",   90.0f },
    { 3, L"120m", 120.0f },
    { 4, L"160m", 160.0f },
    { 5, L"220m", 220.0f },
    { 6, L"300m", 300.0f },
};
#define S_OCEAN_OPT_NUM(t) ((int32_t)(sizeof(t) / sizeof((t)[0])))

const Linglong_Ocean_Option *linglong_ocean_height_options(int32_t *count) {
    if (count) *count = S_OCEAN_OPT_NUM(S_OCEAN_HEIGHT_OPTIONS);
    return S_OCEAN_HEIGHT_OPTIONS;
}
const Linglong_Ocean_Option *linglong_ocean_swell_hs_options(int32_t *count) {
    if (count) *count = S_OCEAN_OPT_NUM(S_OCEAN_SWELL_HS_OPTIONS);
    return S_OCEAN_SWELL_HS_OPTIONS;
}
const Linglong_Ocean_Option *linglong_ocean_swell_len_options(int32_t *count) {
    if (count) *count = S_OCEAN_OPT_NUM(S_OCEAN_SWELL_LEN_OPTIONS);
    return S_OCEAN_SWELL_LEN_OPTIONS;
}

#if LINGLONG_ENABLE_DYNAMIC_LANDSCAPE

// ===============================================================================
// 常量
// ===============================================================================

#define WO_PI            (3.14159265358979323846)

// ---- 大气常数（与 web-ocean common.js ATMO_CORE_WGSL 一致）----
#define WO_RG            (6360.0)      // 地球半径 km
#define WO_RT            (6460.0)      // 大气顶半径 km
static const double WO_RAY_S[3] = { 5.802e-3, 13.558e-3, 33.1e-3 };
#define WO_MIE_S         (3.996e-3)
#define WO_MIE_E         (4.40e-3)
static const double WO_OZO_A[3] = { 0.650e-3, 1.881e-3, 0.085e-3 };
#define WO_SUN_SCALE     (14.0)        // 太阳辐照定标（web-ocean sunDir.w）

// LUT 分辨率
#define WO_TRANS_W       (256)
#define WO_TRANS_H       (64)
#define WO_MS_SIZE       (32)
#define WO_SKY_W         (256)
#define WO_SKY_H         (144)

// 夜天光水体环境光提升：夜间「水体上涌 = 水反照率(约 2%) × 夜天光(≈0.0075)」≈ 1.5e-4，
// 任何色调映射下都是 8bit 全黑（真实月夜海面肉眼也近乎全黑，只剩掠射角的天空反射）。
// 「夜海不至漆黑一片」需要把水体/泡沫的环境光按暗视觉等效亮度抬升：
//   kAmb = 1 + WO_NIGHT_AMB_GAIN * night
// 只作用于环境光（hemi / 天光散射）项，不碰直射太阳与天空反射，所以日间/晨昏零影响，
// 夜间浪形仍由坡度与菲涅尔调制（不是一块均匀灰板）。
#define WO_NIGHT_AMB_GAIN (30.0)

// ---- 海况常量 ----
#define WO_SWELL_SPREAD  (48.0)        // cos^2s 方向分布指数
#define WO_HMAX          (2.2)         // 波场最大高程估计（射线步进起点）
#define WO_MARCH_FAR     (900.0)       // 超过该距离直接取平均海平面
#define WO_ISL_DEEP      (70.0)        // 无岛屿时的海底深度 m（原实现 isl_on=0 分支）

// web-ocean 着色常数（main.js writeFrame）
#define WO_UNRES         (0.0024)      // 未解析坡度方差基线
static const double WO_SSS_COL[3] = { 0.035, 0.120, 0.085 };
#define WO_SSS_K         (2.6)
static const double WO_SHALLOW_COL[3] = { 0.055, 0.190, 0.150 };

// 夜天光标定量（见 celestial_night.h）：天空视图 LUT 只含「散射太阳光」，太阳落到
// 地平线以下就逐像素归零（物理正确，但画面全黑）。真实夜空另有气辉 + 星光 + 黄道光：
// 天顶 ≈ 21.8 mag/arcsec² ≈ 1.5e-4 cd/m²，约为本模型白天晴天顶（≈0.15，模型单位）
// 的 3e-8。直接按真值叠加在任何色调映射下都是 8bit 全黑，故这里按「暗适应后等效
// 亮度」标定：0.15（白天晴天顶）× 3e-8（物理比）× 1.6e6（≈ +20.6 EV 的暗适应增益）
// ≈ 7e-3。该值经 AgX 后落在 8~14/255 的微光区间（天顶偏暗、地平略亮）。
#define WO_NIGHT_RAD    (7.0e-3)
#define WO_FOAM_J_THRESH (0.36 + 0.12 * 1.0)   // 雅可比白帽阈值
#define WO_FOAM_J_GAIN   (2.5)

#define WO_FOAM_N        (256)

// 表面重力波色散
#define WO_GRAV          (9.81)
#define WO_SURF          (7.4e-5)

// 波场级联（可平铺纹理）：L=patch 尺寸 m；n∈[nMin,nMax] 为整数波数模长（波长 L/n）
typedef struct {
    double L;
    int32_t N;
    int32_t nMin;
    int32_t nMax;
    int32_t waves;
    double flat;
    double gain;
    double cgroup;
} Wo_Tile;

static const Wo_Tile WO_TILES[2] = {
    { 512.0, 256, 4, 128, 384, 0.5, 1.0, 4.2 },  // 长波（波长 4~128m）
    {  16.0, 256, 4, 128, 384, 0.5, 0.7, 1.4 },  // 短波（0.125~4m）
};

// ===============================================================================
// 状态
// ===============================================================================

typedef struct {
    // LUT（RGB 交错；float32 存储）
    float  *trans_lut;              // WO_TRANS_W * WO_TRANS_H * 3
    float  *ms_lut;                 // WO_MS_SIZE * WO_MS_SIZE * 3
    float  *sky_lut;                // WO_SKY_W * WO_SKY_H * 3
    double  sun_dir[3];             // 海洋系太阳方向（y 上）
    double  sun_alt;                // 天空视图 LUT 缓存对应的太阳高度角
                                    // （该 LUT 与方位角无关，见 linglong_ocean_prepare）
    double  sky_cam_y;              // 天空视图 LUT 缓存对应的观察高度（m，漫游时连续变化）
    int32_t have_sun;

    double  last_now;               // 上次 prepare 的时刻 ms
    double  app_time;               // 海面动画时间 s
    double  pix_ang;                // 每像素角尺寸 rad

    // 每帧常量（阶段1：原本在逐像素路径里重复计算的天空环境量，现由 prepare 算一次）
    double  sky_up[3];              // 天顶辐亮度 wo_sky_atmo(0,1,0)
    double  hemi[3];                // 半球平均辐亮度（天顶 + 4 个斜方向）
    double  sun_e[3];               // 太阳辐照（透射率 LUT × WO_SUN_SCALE）
    double  night_vis;              // 夜天光可见度 0..1（太阳高度角单值，prepare 每帧算一次）

    int32_t waves_ready;
    float  *wave_data[2];           // 每级联 N*N*4：h, sx, sz, jtrace
    int32_t wave_list_n[2];         // 每级联波列数
    double  cov[3];                 // 坡度协方差 (cxx, czz, cxz)
    uint8_t *foam_tex;              // WO_FOAM_N^2 * 4 程序化泡沫纹理

    // 上一帧配置快照（用于检测参数变更）
    int32_t cfg_valid;
} Wo_State;

static Wo_State s_state;

static Linglong_Ocean_Config s_cfg;
static int32_t s_cfg_init = 0;

// 夜天光的仰角表（见 celestial_night.h）：太阳相关部分每帧算一次，逐像素只做插值。
static Linglong_Night_Sky s_night;

// 派生量（由 wo_update_derived_params 重算）
static double s_water_col[3];
static double s_fog_ext[3];
static double s_wind_vx = 0.0, s_wind_vz = 0.0;
static double s_swell_vx = 0.0, s_swell_vz = 0.0;
static double s_scroll1[2], s_scroll2[2];

// ===============================================================================
// 小工具
// ===============================================================================

static inline double wo_saturate(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }
static inline double wo_clamp(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }
static inline double wo_mix(double a, double b, double t) { return a + (b - a) * t; }
static inline double wo_smoothstep(double e0, double e1, double x) {
    double t = wo_saturate((x - e0) / (e1 - e0));
    return t * t * (3.0 - 2.0 * t);
}
static inline double wo_max(double a, double b) { return a > b ? a : b; }
static inline double wo_min(double a, double b) { return a < b ? a : b; }

// ===============================================================================
// 大气介质（web-ocean medium()）
// ===============================================================================

typedef struct {
    double ray_s[3];
    double mie_s;
    double ext[3];
} Wo_Medium;

static void wo_medium(double h, Wo_Medium *m) {
    double hh = wo_max(h, 0.0);
    double dR = exp(-hh / 8.0);
    // 气溶胶：常规 1.2km 层 + 浓密海洋边界层
    double dM = exp(-hh / 1.2);
    double dO = wo_max(0.0, 1.0 - fabs(hh - 25.0) / 15.0);
    double haze = (double)s_cfg.haze;
    for (int32_t c = 0; c < 3; c++) {
        m->ray_s[c] = WO_RAY_S[c] * dR;
        m->ext[c] = m->ray_s[c] + WO_MIE_E * dM * haze + WO_OZO_A[c] * dO;
    }
    m->mie_s = WO_MIE_S * dM * haze;
}

static void wo_ray_sphere(double ox, double oy, double oz,
                          double dx, double dy, double dz, double R,
                          double *t0, double *t1) {
    double l = sqrt(ox * ox + oy * oy + oz * oz);
    double b = ox * dx + oy * dy + oz * dz;
    double c = (l - R) * (l + R);
    double d = b * b - c;
    if (d < 0.0) {
        *t0 = -1.0;
        *t1 = -1.0;
        return;
    }
    double s = sqrt(d);
    *t0 = -b - s;
    *t1 = -b + s;
}

static inline double wo_rho_of(double r) {
    double v = (r - WO_RG) * (r + WO_RG);
    return sqrt(wo_max(v, 0.0));
}

static void wo_trans_rmu_to_uv(double r, double mu, double *u, double *v) {
    double H = sqrt(WO_RT * WO_RT - WO_RG * WO_RG);
    double rho = wo_rho_of(r);
    double disc = r * r * (mu * mu - 1.0) + WO_RT * WO_RT;
    double d = wo_max(0.0, -r * mu + sqrt(wo_max(disc, 0.0)));
    double dmin = WO_RT - r;
    double dmax = rho + H;
    *u = (d - dmin) / (dmax - dmin);
    *v = rho / H;
}

static void wo_trans_uv_to_rmu(double u, double v, double *r, double *mu) {
    double H = sqrt(WO_RT * WO_RT - WO_RG * WO_RG);
    double rho = H * v;
    double rr = sqrt(rho * rho + WO_RG * WO_RG);
    double dmin = WO_RT - rr;
    double dmax = rho + H;
    double d = dmin + u * (dmax - dmin);
    double m = 1.0;
    if (d > 0.0) m = (H * H - rho * rho - d * d) / (2.0 * rr * d);
    *r = rr;
    *mu = wo_clamp(m, -1.0, 1.0);
}

static inline double wo_from_unit_to_sub_uvs(double u, double res) {
    return (u + 0.5 / res) * (res / (res + 1.0));
}
static inline double wo_from_sub_uvs_to_unit(double u, double res) {
    return (u - 0.5 / res) * (res / (res - 1.0));
}

static inline double wo_phase_rayleigh(double c) {
    return 3.0 / (16.0 * WO_PI) * (1.0 + c * c);
}
static inline double wo_phase_mie(double c, double g) {
    double g2 = g * g;
    return 3.0 / (8.0 * WO_PI) * ((1.0 - g2) * (1.0 + c * c)) /
        ((2.0 + g2) * pow(wo_max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// 观测点 r 处太阳是否低于几何地平
static inline int32_t wo_sun_below_horizon(double r, double mu) {
    return mu < -wo_rho_of(r) / r;
}

// ---- 纹理中心双线性采样（与 GLSL LINEAR 采样约定一致）----
static void wo_bilinear3(const float *lut, int32_t w, int32_t h, double u, double v, double *out) {
    double x = u * w - 0.5, y = v * h - 0.5;
    double fx0 = floor(x), fy0 = floor(y);
    double fx = x - fx0, fy = y - fy0;
    int32_t x0 = (int32_t)fx0, y0 = (int32_t)fy0;
    int32_t x0c = (int32_t)wo_clamp(x0, 0, w - 1), x1c = (int32_t)wo_clamp(x0 + 1, 0, w - 1);
    int32_t y0c = (int32_t)wo_clamp(y0, 0, h - 1), y1c = (int32_t)wo_clamp(y0 + 1, 0, h - 1);
    int32_t i00 = (y0c * w + x0c) * 3, i10 = (y0c * w + x1c) * 3;
    int32_t i01 = (y1c * w + x0c) * 3, i11 = (y1c * w + x1c) * 3;
    double w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
    double w01 = (1 - fx) * fy, w11 = fx * fy;
    for (int32_t c = 0; c < 3; c++) {
        out[c] = (double)lut[i00 + c] * w00 + (double)lut[i10 + c] * w10
               + (double)lut[i01 + c] * w01 + (double)lut[i11 + c] * w11;
    }
}

static void wo_sample_trans_lut(double r, double mu, double *out) {
    if (s_state.trans_lut == NULL) {
        out[0] = 0.0; out[1] = 0.0; out[2] = 0.0;
        return;
    }
    double u = 0.0, v = 0.0;
    wo_trans_rmu_to_uv(r, mu, &u, &v);
    wo_bilinear3(s_state.trans_lut, WO_TRANS_W, WO_TRANS_H,
                 wo_clamp(u, 0.0, 1.0), wo_clamp(v, 0.0, 1.0), out);
}

// ===============================================================================
// 透射率 LUT（web-ocean TRANS_LUT）
// ===============================================================================

static int32_t wo_build_trans_lut(void) {
    const int32_t STEPS = 64;
    float *lut = (float *)platform_calloc((size_t)WO_TRANS_W * WO_TRANS_H * 3, sizeof(float));
    if (lut == NULL) return 0;
    if (s_state.trans_lut != NULL) free(s_state.trans_lut);
    s_state.trans_lut = lut;

    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < WO_TRANS_H; y++) {
        for (int32_t x = 0; x < WO_TRANS_W; x++) {
            double u = (x + 0.5) / (double)WO_TRANS_W;
            double v = (y + 0.5) / (double)WO_TRANS_H;
            double r = 0.0, mu = 0.0;
            wo_trans_uv_to_rmu(u, v, &r, &mu);
            double ro[3] = { 0.0, r, 0.0 };
            double rd[3] = { sqrt(wo_max(1.0 - mu * mu, 0.0)), mu, 0.0 };
            double t0 = 0.0, t1 = 0.0;
            wo_ray_sphere(ro[0], ro[1], ro[2], rd[0], rd[1], rd[2], WO_RT, &t0, &t1);
            double t_max = t1;
            double od[3] = { 0.0, 0.0, 0.0 };
            double dt = t_max / STEPS;
            for (int32_t i = 0; i < STEPS; i++) {
                double t = (i + 0.5) * dt;
                double px = ro[0] + rd[0] * t;
                double py = ro[1] + rd[1] * t;
                double pz = ro[2] + rd[2] * t;
                Wo_Medium m;
                wo_medium(sqrt(px * px + py * py + pz * pz) - WO_RG, &m);
                od[0] += m.ext[0] * dt;
                od[1] += m.ext[1] * dt;
                od[2] += m.ext[2] * dt;
            }
            int32_t idx = (y * WO_TRANS_W + x) * 3;
            lut[idx + 0] = (float)exp(-od[0]);
            lut[idx + 1] = (float)exp(-od[1]);
            lut[idx + 2] = (float)exp(-od[2]);
        }
    }
    return 1;
}

// ===============================================================================
// 多重散射 LUT（web-ocean MS_LUT）
// ===============================================================================

static int32_t wo_build_ms_lut(void) {
    const int32_t SQ = 8, STEPS = 20;
    float *lut = (float *)platform_calloc((size_t)WO_MS_SIZE * WO_MS_SIZE * 3, sizeof(float));
    if (lut == NULL) return 0;
    if (s_state.ms_lut != NULL) free(s_state.ms_lut);
    s_state.ms_lut = lut;

    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < WO_MS_SIZE; y++) {
        for (int32_t x = 0; x < WO_MS_SIZE; x++) {
            double uvx = wo_from_sub_uvs_to_unit((x + 0.5) / (double)WO_MS_SIZE, WO_MS_SIZE);
            double uvy = wo_from_sub_uvs_to_unit((y + 0.5) / (double)WO_MS_SIZE, WO_MS_SIZE);
            double cos_sun = uvx * 2.0 - 1.0;
            double h = wo_max(uvy * (WO_RT - WO_RG), 0.01);
            double r = WO_RG + h;
            double ro[3] = { 0.0, r, 0.0 };
            double sun[3] = { sqrt(wo_max(1.0 - cos_sun * cos_sun, 0.0)), cos_sun, 0.0 };
            double Lsum[3] = { 0.0, 0.0, 0.0 }, Fsum[3] = { 0.0, 0.0, 0.0 };

            for (int32_t a = 0; a < SQ; a++) {
                for (int32_t b = 0; b < SQ; b++) {
                    double th = acos(1.0 - 2.0 * (b + 0.5) / SQ);
                    double ph = 2.0 * WO_PI * (a + 0.5) / SQ;
                    double rd[3] = {
                        sin(th) * cos(ph), cos(th), sin(th) * sin(ph)
                    };
                    double t_top0 = 0.0, t_top1 = 0.0, t_gr0 = 0.0, t_gr1 = 0.0;
                    wo_ray_sphere(ro[0], ro[1], ro[2], rd[0], rd[1], rd[2], WO_RT, &t_top0, &t_top1);
                    wo_ray_sphere(ro[0], ro[1], ro[2], rd[0], rd[1], rd[2], WO_RG, &t_gr0, &t_gr1);
                    double t_max = t_top1;
                    int32_t hit_g = (t_gr0 > 0.0);
                    if (hit_g) t_max = t_gr0;
                    double dt = t_max / STEPS;
                    double T[3] = { 1.0, 1.0, 1.0 };
                    double L[3] = { 0.0, 0.0, 0.0 };
                    double Fms[3] = { 0.0, 0.0, 0.0 };
                    for (int32_t s = 0; s < STEPS; s++) {
                        double t = (s + 0.5) * dt;
                        double px = ro[0] + rd[0] * t;
                        double py = ro[1] + rd[1] * t;
                        double pz = ro[2] + rd[2] * t;
                        double pr = sqrt(px * px + py * py + pz * pz);
                        Wo_Medium m;
                        wo_medium(pr - WO_RG, &m);
                        double scat[3] = {
                            m.ray_s[0] + m.mie_s, m.ray_s[1] + m.mie_s, m.ray_s[2] + m.mie_s
                        };
                        double Ts[3] = {
                            exp(-m.ext[0] * dt), exp(-m.ext[1] * dt), exp(-m.ext[2] * dt)
                        };
                        double smu = (px * sun[0] + py * sun[1] + pz * sun[2]) / pr;
                        double sunT[3];
                        wo_sample_trans_lut(pr, smu, sunT);
                        if (wo_sun_below_horizon(pr, smu)) {
                            sunT[0] = 0.0; sunT[1] = 0.0; sunT[2] = 0.0;
                        }
                        for (int32_t c = 0; c < 3; c++) {
                            double ext = wo_max(m.ext[c], 1e-9);
                            double S = scat[c] * sunT[c] / (4.0 * WO_PI);
                            L[c] += T[c] * (S - S * Ts[c]) / ext;
                            Fms[c] += T[c] * (scat[c] - scat[c] * Ts[c]) / ext;
                            T[c] *= Ts[c];
                        }
                    }
                    if (hit_g) {
                        double pgx = ro[0] + rd[0] * t_max;
                        double pgy = ro[1] + rd[1] * t_max;
                        double pgz = ro[2] + rd[2] * t_max;
                        double ng = 1.0 / sqrt(pgx * pgx + pgy * pgy + pgz * pgz);
                        double smu = (pgx * sun[0] + pgy * sun[1] + pgz * sun[2]) * ng;
                        double transTmp[3];
                        wo_sample_trans_lut(WO_RG, smu, transTmp);
                        double g_wgt = wo_max(smu, 0.0) * 0.06 / WO_PI;
                        for (int32_t c = 0; c < 3; c++) L[c] += T[c] * transTmp[c] * g_wgt;
                    }
                    for (int32_t c = 0; c < 3; c++) {
                        Lsum[c] += L[c];
                        Fsum[c] += Fms[c];
                    }
                }
            }

            double inv_n = 1.0 / (SQ * SQ);
            int32_t idx = (y * WO_MS_SIZE + x) * 3;
            for (int32_t c = 0; c < 3; c++) {
                double L2 = Lsum[c] * inv_n;
                double fms = Fsum[c] * inv_n;
                lut[idx + c] = (float)(L2 / (1.0 - fms));
            }
        }
    }
    return 1;
}

// ===============================================================================
// 天空视图 LUT（web-ocean SKYVIEW_LUT，太阳变化时重算）
// ===============================================================================

static int32_t wo_build_sky_lut(void) {
    const int32_t STEPS = 48;
    if (s_state.trans_lut == NULL || s_state.ms_lut == NULL) return 0;

    float *lut = (float *)platform_calloc((size_t)WO_SKY_W * WO_SKY_H * 3, sizeof(float));
    if (lut == NULL) return 0;
    if (s_state.sky_lut != NULL) free(s_state.sky_lut);
    s_state.sky_lut = lut;

    const double smu = s_state.sun_dir[1];
    const double sun[3] = { sqrt(wo_max(1.0 - smu * smu, 0.0)), smu, 0.0 };
    const double r = WO_RG + (double)s_cfg.cam_y / 1000.0;
    const double v_horizon = wo_rho_of(r);
    const double cos_beta = v_horizon / r;
    const double beta = acos(wo_clamp(cos_beta, -1.0, 1.0));
    const double zha = WO_PI - beta;

    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < WO_SKY_H; y++) {
        for (int32_t x = 0; x < WO_SKY_W; x++) {
            double uvx = wo_from_sub_uvs_to_unit((x + 0.5) / (double)WO_SKY_W, WO_SKY_W);
            double uvy = wo_from_sub_uvs_to_unit((y + 0.5) / (double)WO_SKY_H, WO_SKY_H);
            double vza;
            if (uvy < 0.5) {
                double coord = 2.0 * uvy;
                coord = 1.0 - coord;
                coord *= coord;
                coord = 1.0 - coord;
                vza = zha * coord;
            }
            else {
                double coord = uvy * 2.0 - 1.0;
                coord *= coord;
                vza = zha + beta * coord;
            }
            double vzc = cos(vza), vzs = sin(vza);
            double cx = uvx * uvx;
            double lvc = -(cx * 2.0 - 1.0);
            double lvs = sqrt(wo_max(1.0 - lvc * lvc, 0.0));
            double rd[3] = { vzs * lvc, vzc, vzs * lvs };
            double ro[3] = { 0.0, r, 0.0 };
            double t_top0 = 0.0, t_top1 = 0.0, t_gr0 = 0.0, t_gr1 = 0.0;
            wo_ray_sphere(ro[0], ro[1], ro[2], rd[0], rd[1], rd[2], WO_RT, &t_top0, &t_top1);
            wo_ray_sphere(ro[0], ro[1], ro[2], rd[0], rd[1], rd[2], WO_RG, &t_gr0, &t_gr1);
            double t_max = t_top1;
            if (t_gr0 > 0.0) t_max = t_gr0;
            double cos_t = rd[0] * sun[0] + rd[1] * sun[1] + rd[2] * sun[2];
            double pR = wo_phase_rayleigh(cos_t);
            double pM = wo_phase_mie(cos_t, 0.84);
            double L[3] = { 0.0, 0.0, 0.0 };
            double T[3] = { 1.0, 1.0, 1.0 };
            double t_prev = 0.0;
            for (int32_t i = 0; i < STEPS; i++) {
                double fa = (i + 1.0) / STEPS;
                double tn = t_max * fa * fa;
                double dt = tn - t_prev;
                double t = t_prev + dt * 0.3;
                t_prev = tn;
                double px = ro[0] + rd[0] * t;
                double py = ro[1] + rd[1] * t;
                double pz = ro[2] + rd[2] * t;
                double pr = sqrt(px * px + py * py + pz * pz);
                double h = pr - WO_RG;
                Wo_Medium m;
                wo_medium(h, &m);
                double pmu = (px * sun[0] + py * sun[1] + pz * sun[2]) / pr;
                double sunT[3];
                wo_sample_trans_lut(pr, pmu, sunT);
                if (wo_sun_below_horizon(pr, pmu)) {
                    sunT[0] = 0.0; sunT[1] = 0.0; sunT[2] = 0.0;
                }
                double msu = wo_from_unit_to_sub_uvs(pmu * 0.5 + 0.5, WO_MS_SIZE);
                double msv = wo_from_unit_to_sub_uvs(wo_clamp(h / (WO_RT - WO_RG), 0.0, 1.0), WO_MS_SIZE);
                double ms[3];
                wo_bilinear3(s_state.ms_lut, WO_MS_SIZE, WO_MS_SIZE, msu, msv, ms);
                double Ts[3] = {
                    exp(-m.ext[0] * dt), exp(-m.ext[1] * dt), exp(-m.ext[2] * dt)
                };
                for (int32_t c = 0; c < 3; c++) {
                    double ext = wo_max(m.ext[c], 1e-9);
                    double S = sunT[c] * (m.ray_s[c] * pR + m.mie_s * pM)
                             + ms[c] * (m.ray_s[c] + m.mie_s);
                    L[c] += T[c] * (S - S * Ts[c]) / ext;
                    T[c] *= Ts[c];
                }
            }
            int32_t idx = (y * WO_SKY_W + x) * 3;
            lut[idx + 0] = (float)L[0];
            lut[idx + 1] = (float)L[1];
            lut[idx + 2] = (float)L[2];
        }
    }
    return 1;
}

// ---- 天空视图 LUT 查询（web-ocean skyViewUV / skyAtmo）----
// 阶段2：cos_beta / beta / zha 只依赖观察高度（r = RG + cam_y），原实现每次查询都要
// 重算一遍 acos + sqrt（每次查询 38ns 里占相当一部分，而每像素要查 2 次）；
// 现由 wo_sky_view_uv_params() 在 prepare 里算好，逐像素只读。
static double s_uv_cos_beta = 1.0, s_uv_beta = 0.0, s_uv_zha = WO_PI;

static void wo_sky_view_uv_params(void) {
    double r = WO_RG + (double)s_cfg.cam_y / 1000.0;
    s_uv_cos_beta = wo_rho_of(r) / r;
    s_uv_beta = acos(wo_clamp(s_uv_cos_beta, -1.0, 1.0));
    s_uv_zha = WO_PI - s_uv_beta;
}

// 快速 acos（性能专项 A5）：sqrt(1-x)·P(x)（P 为 [0,1] 上的 7 次 minimax 多项式，
// x<0 用 acos(x)=π−acos(−x) 折叠）。实测全区间最大绝对误差 2.2e-8 rad，已在 float
// 精度以内。wo_sky_view_uv 每次调用一次 acos（海面每像素 3 次、海洋天空每像素 1 次），
// 库 acos 是 double 的 fdlibm 路径（含范围归约与多分支），是这两条路径的固定大头。
static inline double wo_acos_poly(double x) {
    double s = sqrt(1.0 - x);
    double p = -0.0012624911;
    p = p * x + 0.0066700901;
    p = p * x - 0.0170881256;
    p = p * x + 0.0308918810;
    p = p * x - 0.0501743046;
    p = p * x + 0.0889789874;
    p = p * x - 0.2145988016;
    p = p * x + 1.5707963050;
    return s * p;
}
static inline double wo_fast_acos(double x) {
    if (x >= 1.0) return 0.0;
    if (x <= -1.0) return WO_PI;
    if (x < 0.0) return WO_PI - wo_acos_poly(-x);
    return wo_acos_poly(x);
}

static void wo_sky_view_uv(double dx, double dy, double dz, double *u, double *v) {
    const double cos_beta = s_uv_cos_beta;
    const double beta = s_uv_beta;
    const double zha = s_uv_zha;
    double vzc = wo_clamp(dy, -1.0, 1.0);
    double uv_y;
    if (vzc >= -cos_beta) {
        double coord = wo_fast_acos(vzc) / zha;
        coord = 1.0 - coord;
        coord = sqrt(wo_max(coord, 0.0));
        coord = 1.0 - coord;
        uv_y = coord * 0.5;
    }
    else {
        double coord = (wo_fast_acos(vzc) - zha) / beta;
        coord = sqrt(wo_max(coord, 0.0));
        uv_y = coord * 0.5 + 0.5;
    }
    double dl = sqrt(dx * dx + dz * dz);
    const double *s = s_state.sun_dir;
    double sl = sqrt(s[0] * s[0] + s[2] * s[2]);
    double lvc = 1.0;
    if (dl > 1e-6 && sl > 1e-6) lvc = wo_clamp((dx * s[0] + dz * s[2]) / (dl * sl), -1.0, 1.0);
    double uv_x = sqrt(wo_max(-lvc * 0.5 + 0.5, 0.0));
    *u = wo_from_unit_to_sub_uvs(uv_x, WO_SKY_W);
    *v = wo_from_unit_to_sub_uvs(uv_y, WO_SKY_H);
}

static void wo_sky_atmo(double dx, double dy, double dz, double *out) {
    if (s_state.sky_lut == NULL) {
        out[0] = 0.0; out[1] = 0.0; out[2] = 0.0;
        return;
    }
    double u = 0.0, v = 0.0;
    wo_sky_view_uv(dx, dy, dz, &u, &v);
    wo_bilinear3(s_state.sky_lut, WO_SKY_W, WO_SKY_H,
                 wo_clamp(u, 0.0, 1.0), wo_clamp(v, 0.0, 1.0), out);
    out[0] *= WO_SUN_SCALE;
    out[1] *= WO_SUN_SCALE;
    out[2] *= WO_SUN_SCALE;

    // 夜天光（无月：气辉 + 积分星光 + 黄道光，见 celestial_night.h）。在这里叠加的好处是
    // **一处生效、全局一致**：海洋天空像素、海面天空反射、水体/泡沫的环境光（sky_up/hemi）、
    // 大气透视的地平色都从同一份辐亮度取值，夜间不会出现「天空有微光但海面仍全黑」。
    double nw[3];
    linglong_night_sky_eval(&s_night, dx, dy, dz, nw);
    out[0] += nw[0] * WO_NIGHT_RAD;
    out[1] += nw[1] * WO_NIGHT_RAD;
    out[2] += nw[2] * WO_NIGHT_RAD;
}

// ===============================================================================
// AgX 色调映射（web-ocean COMPOSITE，去除 bloom/暗角/抖动）
// ===============================================================================

// ---- 快速超越函数（阶段2）----
// AgX 原本每像素调 3 次 log2 + 3 次 pow（实测 58ns/次调用，占整幅约 8%）。
// 这里用「拆阶码 + 多项式」的常用做法，精度足以覆盖 8bit 量化：
//   wo_fast_log2: 相对误差 < 1.5e-6（log2 单位，6 次多项式）
//   wo_fast_exp2: 相对误差 < 2.5e-7（5 次多项式 + 阶码拼接，指数部分精确）
// 两者均为确定性无表实现（不引入 LUT 缓存行开销），且无参数变大时的误差放大
// （exp2 的整数部分由位拼接给出，多项式只在 [-0.5,0.5] 上作用）。

static inline double wo_fast_log2(double x) {
    union { double d; uint64_t u; } v;
    v.d = x;
    int32_t e = (int32_t)((v.u >> 52) & 0x7ffu) - 1023;
    v.u = (v.u & 0x000fffffffffffffull) | 0x3ff0000000000000ull;   // m ∈ [1,2)
    double m = v.d;
    if (m > 1.4142135623730951) { m *= 0.5; e += 1; }              // m ∈ [0.7071,1.4142)
    double u = m - 1.0;                                           // |u| ≤ 0.4143
    double p = 0.17001604029869127;
    p = p * u - 0.26626508770316887;
    p = p * u + 0.29510991069280823;
    p = p * u - 0.35965643173979306;
    p = p * u + 0.48067613580330931;
    p = p * u - 0.72135813570393226;
    p = p * u + 1.4426960473682768;
    return (double)e + u * p;
}

static inline double wo_fast_exp2(double x) {
    double t = floor(x + 0.5);          // 最近整数（多项式区间取 [-0.5,0.5]）
    double f = x - t;
    double p = 0.00015465314751714857;
    p = p * f + 0.0013395282274223199;
    p = p * f + 0.0096180399388676365;
    p = p * f + 0.05550340675121699;
    p = p * f + 0.24022651101359899;
    p = p * f + 0.69314720006502339;
    p = p * f + 0.99999999997183853;    // 2^f，相对误差 < 7e-9
    int32_t n = (int32_t)t;
    if (n > 1023) return 1e300;         // 溢出不返回 inf，避免下游 inf-inf 产 NaN
    if (n < -1022) return 0.0;          // 2^-1022 以下对 8bit 输出等价 0
    union { double d; uint64_t u; } v;
    v.u = (uint64_t)(n + 1023) << 52;
    return p * v.d;
}

static inline double wo_fast_exp(double x) { return wo_fast_exp2(x * 1.4426950408889634); }

// x^1.06（AgX look 的幂项）；x ≤ 0 时返回 0（与原 pow(max(x,0),1.06) 一致）
static inline double wo_fast_pow106(double x) {
    if (x <= 0.0) return 0.0;
    if (x < 1e-300) return 0.0;         // 下溢（pow 的结果早已小于 double 最小正数）
    return wo_fast_exp2(1.06 * wo_fast_log2(x));
}

static inline double wo_agx_contrast(double x) {
    double x2 = x * x;
    double x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x
         + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

static void wo_agx_tonemap(double r, double g, double b, double *out) {
    // 曝光（web-ocean 默认 1.35）
    double c0 = r * 1.35, c1 = g * 1.35, c2 = b * 1.35;
    // 白平衡（compData[6] = -0.19）
    c0 *= 0.81;
    c2 *= 1.19;
    // 胶片型高光去饱和：极亮耀斑与太阳趋向白色
    double lum = 0.2126 * c0 + 0.7152 * c1 + 0.0722 * c2;
    double ds = wo_smoothstep(0.7, 6.0, lum) * 0.55;
    c0 = wo_mix(c0, lum, ds);
    c1 = wo_mix(c1, lum, ds);
    c2 = wo_mix(c2, lum, ds);
    // AgX 编码（注意：WGSL/GLSL mat 为列主序，此处按 m*v 的实际作用转写）
    const double min_ev = -12.47393, max_ev = 4.026069;
    double x0 = wo_max(c0, 1e-10), x1 = wo_max(c1, 1e-10), x2 = wo_max(c2, 1e-10);
    double y0 = 0.842479062253094 * x0 + 0.0784335999999992 * x1 + 0.0792237451477643 * x2;
    double y1 = 0.0423282422610123 * x0 + 0.878468636469772 * x1 + 0.0791661274605434 * x2;
    double y2 = 0.0423756549057051 * x0 + 0.0784336 * x1 + 0.879142973793104 * x2;
    y0 = wo_clamp(wo_fast_log2(y0), min_ev, max_ev);
    y1 = wo_clamp(wo_fast_log2(y1), min_ev, max_ev);
    y2 = wo_clamp(wo_fast_log2(y2), min_ev, max_ev);
    y0 = (y0 - min_ev) / (max_ev - min_ev);
    y1 = (y1 - min_ev) / (max_ev - min_ev);
    y2 = (y2 - min_ev) / (max_ev - min_ev);
    y0 = wo_agx_contrast(y0);
    y1 = wo_agx_contrast(y1);
    y2 = wo_agx_contrast(y2);
    // look：sat=1.05，power=1.06
    double l = 0.2126 * y0 + 0.7152 * y1 + 0.0722 * y2;
    double o0 = l + 1.05 * (wo_fast_pow106(y0) - l);
    double o1 = l + 1.05 * (wo_fast_pow106(y1) - l);
    double o2 = l + 1.05 * (wo_fast_pow106(y2) - l);
    // AgX 解码（Eotf，列主序转写）
    double e0 = 1.19687900512017 * o0 - 0.0980208811401368 * o1 - 0.0990297440797205 * o2;
    double e1 = -0.0528968517574562 * o0 + 1.15190312990417 * o1 - 0.0989611768448433 * o2;
    double e2 = -0.0529716355144438 * o0 - 0.0980434501171241 * o1 + 1.15107367264116 * o2;
    out[0] = wo_clamp(e0, 0.0, 1.0);
    out[1] = wo_clamp(e1, 0.0, 1.0);
    out[2] = wo_clamp(e2, 0.0, 1.0);
}

// ===============================================================================
// 海洋天空（散射模型 5，逐像素入口）
// ray_x/y/z 为页面地平坐标系（z 轴向上）向量，输出 0..1（已色调映射）
// ===============================================================================

void linglong_ocean_sky_pixel(float ray_x, float ray_y, float ray_z,
                               float *red, float *green, float *blue) {
    *red = 0.0f; *green = 0.0f; *blue = 0.0f;
    if (s_state.sky_lut == NULL) return;
    double rx = ray_x, ry = ray_y, rz = ray_z;
    double rl = sqrt(rx * rx + ry * ry + rz * rz);
    if (rl < 1e-10) return;
    // 换算到海洋系（x东, y上, z北）并按 skyRadiance 约束仰角
    double dx = rx / rl, dy = rz / rl, dz = ry / rl;
    if (dy < 0.0005) dy = 0.0005;
    double inl = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
    dx *= inl; dy *= inl; dz *= inl;
    double c[3];
    wo_sky_atmo(dx, dy, dz, c);
    double o[3];
    wo_agx_tonemap(c[0], c[1], c[2], o);
    *red = (float)o[0];
    *green = (float)o[1];
    *blue = (float)o[2];
}

// ===============================================================================
// 海浪：JONSWAP 风浪谱（Donelan-Banner 方向分布）+ 窄涌浪（web-ocean sim.js）
// ===============================================================================

typedef struct { uint32_t a; } Wo_Rng;

// mulberry32（与参考实现逐位一致；Math.imul 即 32 位无符号乘法的低 32 位）
static double wo_rng_next(Wo_Rng *r) {
    r->a = (uint32_t)(r->a + 0x6D2B79F5u);
    uint32_t t = (r->a ^ (r->a >> 15)) * (r->a | 1u);
    t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
    return (double)(t ^ (t >> 14)) / 4294967296.0;
}

static inline double wo_omega_of(double k) { return sqrt(WO_GRAV * k + WO_SURF * k * k * k); }
static inline double wo_domega_of(double k) {
    return (WO_GRAV + 3.0 * WO_SURF * k * k) / (2.0 * wo_omega_of(k));
}
static inline double wo_wrap_angle(double a) {
    return a - 2.0 * WO_PI * floor((a + WO_PI) / (2.0 * WO_PI));
}

static double wo_jonswap(double w, double wp, double alpha, double gamma) {
    double sigma = (w <= wp) ? 0.07 : 0.09;
    double d = (w - wp) / (sigma * wp);
    double rr = exp(-0.5 * d * d);
    double q = wp / w;
    return alpha * WO_GRAV * WO_GRAV / (w * w * w * w * w)
         * exp(-1.25 * q * q * q * q) * pow(gamma, rr);
}

// Donelan-Banner 方向分布（[-π,π] 上归一化）
static double wo_donelan_banner(double theta, double w, double wp) {
    double ratio = w / wp;
    double beta;
    if (ratio < 0.95) beta = 2.61 * pow(ratio, 1.3);
    else if (ratio < 1.6) beta = 2.28 * pow(ratio, -1.3);
    else {
        double eps = -0.4 + 0.8393 * exp(-0.567 * log(ratio * ratio));
        beta = pow(10.0, eps);
    }
    double sh = 1.0 / cosh(wo_clamp(beta * theta, -20.0, 20.0));
    return beta / (2.0 * tanh(beta * WO_PI)) * sh * sh;
}

// cos^2s(theta/2) 方向分布（归一化）
static double wo_cos_spread(double theta, double s) {
    double q = sqrt(s + 0.25) / (2.0 * sqrt(WO_PI));
    return q * pow(fabs(cos(theta * 0.5)), 2.0 * s);
}

// 二维波数谱 S(k)（web-ocean spectrum()：S(ω)·dω/dk/|k| × 方向分布 × 短波阻尼）
static double wo_spectrum(double kx, double kz) {
    double kl = sqrt(kx * kx + kz * kz);
    if (kl < 1e-5) return 0.0;
    double w = wo_omega_of(kl);
    double theta = atan2(kz, kx);
    double Sw = 0.0;
    double wind_u = s_cfg.wind_u;
    double fetch = s_cfg.fetch;
    if (wind_u > 0.1) {
        double alpha = 0.076 * pow(wind_u * wind_u / (fetch * WO_GRAV), 0.22);
        double wp = 22.0 * pow(WO_GRAV * WO_GRAV / (wind_u * fetch), 1.0 / 3.0);
        Sw = wo_jonswap(w, wp, alpha, 3.3)
           * wo_donelan_banner(wo_wrap_angle(theta - atan2(s_wind_vz, s_wind_vx)), w, wp);
    }
    if (s_cfg.swell_hs > 0.001) {
        double ks = 2.0 * WO_PI / s_cfg.swell_len;
        double ws = sqrt(WO_GRAV * ks);
        double gs = 6.0;
        double hs4 = s_cfg.swell_hs * 0.25;
        double alphaS = hs4 * hs4 * 5.0 * ws * ws * ws * ws / (WO_GRAV * WO_GRAV)
                      * (1.0 - 0.287 * log(gs));
        Sw += wo_jonswap(w, ws, alphaS, gs)
            * wo_cos_spread(wo_wrap_angle(theta - atan2(s_swell_vz, s_swell_vx)), WO_SWELL_SPREAD);
    }
    // 最短波阻尼（web-ocean short wave cut）
    Sw *= exp(-kl * kl * 0.0012 * 0.0012);
    return Sw * wo_domega_of(kl) / kl;
}

typedef struct {
    int32_t nx, nz;
    double mass;   // S·dk²（该格点的方差质量）
    double acc;    // 累计抽样权重
} Wo_Pt;

// 从整数波数格点（保证可平铺）抽取波列：
//   ① 枚举波段内全部格点，得到各自谱质量 S·dk² 与抽样权重 S^flat；
//   ② 按累计权重做分层系统抽样（每层随机位置）取 W 个格点；
//   ③ 振幅按重要性权重反算 a²/2 = S·dk²/(W·p)，总方差无偏。
// 返回 [kx, kz, a, a·kx, a·kz, a·|k|², φ] 交叉存储（*out_n 写入波列数）
static double *wo_build_wave_list(const Wo_Tile *tile, uint32_t seed, int32_t *out_n) {
    const double dk = 2.0 * WO_PI / tile->L;
    const int32_t nMin = tile->nMin, nMax = tile->nMax;
    Wo_Rng rng;
    rng.a = seed;

    const double flat = tile->flat;
    const int32_t max_pts = (2 * nMax + 1) * (2 * nMax + 1);
    Wo_Pt *pts = (Wo_Pt *)platform_calloc((size_t)max_pts, sizeof(Wo_Pt));
    if (pts == NULL) { *out_n = 0; return NULL; }

    int32_t n_pts = 0;
    double total_w = 0.0;
    const int32_t a2 = nMin * nMin, b2 = nMax * nMax;
    for (int32_t nz = -nMax; nz <= nMax; nz++) {
        for (int32_t nx = -nMax; nx <= nMax; nx++) {
            int32_t n2 = nx * nx + nz * nz;
            if (n2 < a2 || n2 > b2) continue;
            double S = wo_spectrum(nx * dk, nz * dk);
            if (S <= 0.0) continue;
            total_w += pow(S, flat);
            pts[n_pts].nx = nx;
            pts[n_pts].nz = nz;
            pts[n_pts].mass = S * dk * dk;
            pts[n_pts].acc = total_w;
            n_pts++;
        }
    }
    if (n_pts == 0 || total_w <= 0.0) {
        free(pts);
        *out_n = 0;
        return NULL;
    }

    const int32_t W = tile->waves;
    const double dWt = total_w / W;
    double *out = (double *)platform_calloc((size_t)W * 7, sizeof(double));
    if (out == NULL) { free(pts); *out_n = 0; return NULL; }

    int32_t idx = 0;
    int32_t n_out = 0;
    for (int32_t s = 0; s < W; s++) {
        double target = ((double)s + wo_rng_next(&rng)) * dWt;
        while (idx < n_pts - 1 && pts[idx].acc < target) idx++;
        int32_t nx = pts[idx].nx, nz = pts[idx].nz;
        double mass = pts[idx].mass;             // S·dk²
        double s_val = mass / (dk * dk);         // S(k)
        // 重要性加权：a²/2 = S·dk²/(W·p)，p = S^flat/totalW
        double a = sqrt(2.0 * mass * total_w / (W * pow(s_val, flat))) * tile->gain;
        if (!(a > 0.0) || !isfinite(a)) continue;
        double kx = nx * dk, kz = nz * dk;
        double kl = sqrt(kx * kx + kz * kz);
        double *o = out + n_out * 7;
        o[0] = kx;
        o[1] = kz;
        o[2] = a;
        o[3] = a * kx;
        o[4] = a * kz;
        // Gerstner 水平位移雅可比迹系数（choppy 同 web-ocean 默认 1.25）
        o[5] = -1.25 * a * kl;
        o[6] = wo_rng_next(&rng) * 2.0 * WO_PI;
        n_out++;
    }
    free(pts);
    *out_n = n_out;
    return out;
}

// 波场烘焙：逐纹理元求和波列（高度、两轴坡度、雅可比迹）
static float *wo_bake_tile(const Wo_Tile *tile, const double *list, int32_t nw) {
    const int32_t N = tile->N;
    const double L = tile->L;
    float *data = (float *)platform_calloc((size_t)N * N * 4, sizeof(float));
    if (data == NULL || nw == 0) return data;

    const double dx = L / N;
    double *cosD = (double *)platform_calloc((size_t)nw, sizeof(double));
    double *sinD = (double *)platform_calloc((size_t)nw, sizeof(double));
    double *cs = (double *)platform_calloc((size_t)nw * 2, sizeof(double));
    if (cosD == NULL || sinD == NULL || cs == NULL) {
        free(cosD); free(sinD); free(cs);
        free(data);
        return NULL;
    }

    // 每波沿 x 方向的相位步进（旋转递推用）：cos(kx·dx)、sin(kx·dx)
    for (int32_t w = 0; w < nw; w++) {
        double dphi = list[w * 7] * dx;
        cosD[w] = cos(dphi);
        sinD[w] = sin(dphi);
    }

    // 逐行递推：行首计算一次 sin/cos，行内用旋转递推 φ += kx·dx
    for (int32_t j = 0; j < N; j++) {
        double z = (j + 0.5) * dx;
        for (int32_t w = 0; w < nw; w++) {
            const double *o = list + w * 7;
            double ph = o[6] + o[1] * z + o[0] * (0.5 * dx);
            cs[w * 2] = cos(ph);
            cs[w * 2 + 1] = sin(ph);
        }
        int32_t p = j * N * 4;
        for (int32_t i = 0; i < N; i++) {
            double h = 0.0, sx = 0.0, sz = 0.0, jt = 0.0;
            for (int32_t w = 0; w < nw; w++) {
                double c = cs[w * 2], s = cs[w * 2 + 1];
                const double *o = list + w * 7;
                h  += o[2] * c;   // a·cos φ
                sx -= o[3] * s;   // -a·kx·sin φ
                sz -= o[4] * s;   // -a·kz·sin φ
                jt += o[5] * c;   // 雅可比迹扰动（含符号）
                // 旋转递推 (c,s) ← (c,s)·(cosΔ, sinΔ)
                double cd = cosD[w], sd = sinD[w];
                cs[w * 2] = c * cd - s * sd;
                cs[w * 2 + 1] = c * sd + s * cd;
            }
            data[p + 0] = (float)h;
            data[p + 1] = (float)sx;
            data[p + 2] = (float)sz;
            data[p + 3] = (float)jt;
            p += 4;
        }
    }
    free(cosD);
    free(sinD);
    free(cs);
    return data;
}

// 说明（性能专项 A4）：波场/泡沫纹理的边长 N 恒为 2 的幂（256），且采样坐标可能为负，
// 故用「与掩码」代替两次整数取模：二补码下 (i & (N-1)) 就是数学意义下的 mod N。
// 原写法每次采样 4 次 sdiv（WASM 的 i32.rem_s / ARM 的 sdiv 都是硬件除法，约 20 个周期），
// 而每个海面像素要查 40+ 次。
#define WO_WRAP(i, mask)  ((int32_t)(i) & (mask))

static void wo_tile_sample(const float *data, int32_t N, double L, double x, double z, double *out) {
    const int32_t mask = N - 1;
    double u = (x / L) * N - 0.5, v = (z / L) * N - 0.5;
    int32_t i0 = (int32_t)floor(u), j0 = (int32_t)floor(v);
    double fu = u - i0, fv = v - j0;
    int32_t i0c = WO_WRAP(i0, mask), i1c = WO_WRAP(i0 + 1, mask);
    int32_t j0c = WO_WRAP(j0, mask), j1c = WO_WRAP(j0 + 1, mask);
    int32_t i00 = (j0c * N + i0c) * 4, i10 = (j0c * N + i1c) * 4;
    int32_t i01 = (j1c * N + i0c) * 4, i11 = (j1c * N + i1c) * 4;
    double w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv);
    double w01 = (1 - fu) * fv, w11 = fu * fv;
    for (int32_t c = 0; c < 4; c++) {
        out[c] = (double)data[i00 + c] * w00 + (double)data[i10 + c] * w10
               + (double)data[i01 + c] * w01 + (double)data[i11 + c] * w11;
    }
}

// 波面高程采样（仅 h 通道；march 内层使用）
static double wo_tile_sample_h(const float *data, int32_t N, double L, double x, double z) {
    const int32_t mask = N - 1;
    double u = (x / L) * N - 0.5, v = (z / L) * N - 0.5;
    int32_t i0 = (int32_t)floor(u), j0 = (int32_t)floor(v);
    double fu = u - i0, fv = v - j0;
    int32_t i0c = WO_WRAP(i0, mask), i1c = WO_WRAP(i0 + 1, mask);
    int32_t j0c = WO_WRAP(j0, mask), j1c = WO_WRAP(j0 + 1, mask);
    int32_t i00 = (j0c * N + i0c) * 4, i10 = (j0c * N + i1c) * 4;
    int32_t i01 = (j1c * N + i0c) * 4, i11 = (j1c * N + i1c) * 4;
    double w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv);
    double w01 = (1 - fu) * fv, w11 = fu * fv;
    return (double)data[i00] * w00 + (double)data[i10] * w10
         + (double)data[i01] * w01 + (double)data[i11] * w11;
}

// 泡沫纹理采样（RGBA8 可平铺纹理；返回 0..255 通道值）
static void wo_tex_sample_u8(const uint8_t *data, int32_t N, double L, double x, double z, double *out) {
    const int32_t mask = N - 1;                  // 见 WO_WRAP 注释（N 恒为 2 的幂）
    double u = (x / L) * N - 0.5, v = (z / L) * N - 0.5;
    int32_t i0 = (int32_t)floor(u), j0 = (int32_t)floor(v);
    double fu = u - i0, fv = v - j0;
    int32_t i0c = WO_WRAP(i0, mask), i1c = WO_WRAP(i0 + 1, mask);
    int32_t j0c = WO_WRAP(j0, mask), j1c = WO_WRAP(j0 + 1, mask);
    int32_t i00 = (j0c * N + i0c) * 4, i10 = (j0c * N + i1c) * 4;
    int32_t i01 = (j1c * N + i0c) * 4, i11 = (j1c * N + i1c) * 4;
    double w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv);
    double w01 = (1 - fu) * fv, w11 = fu * fv;
    for (int32_t c = 0; c < 4; c++) {
        out[c] = (double)data[i00 + c] * w00 + (double)data[i10 + c] * w10
               + (double)data[i01 + c] * w01 + (double)data[i11 + c] * w11;
    }
}

// ---- 程序化泡沫纹理（256² RGBA 可平铺；r/g/b 不同尺度 fbm，a 供风向条纹各向异性采样）----
static double wo_hash2i(int32_t ix, int32_t iz, int32_t seed) {
    uint32_t h = (uint32_t)ix * 374761393u ^ (uint32_t)iz * 668265263u ^ (uint32_t)seed * 1274126177u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (double)h / 4294967296.0;
}

static double wo_value_noise_tile(double x, double z, int32_t period, int32_t seed) {
    int32_t ix = (int32_t)floor(x), iz = (int32_t)floor(z);
    double fx = x - ix, fz = z - iz;
    double ux = fx * fx * (3.0 - 2.0 * fx), uz = fz * fz * (3.0 - 2.0 * fz);
    int32_t x0 = ((ix % period) + period) % period;
    int32_t x1 = (((ix + 1) % period) + period) % period;
    int32_t z0 = ((iz % period) + period) % period;
    int32_t z1 = (((iz + 1) % period) + period) % period;
    double a = wo_hash2i(x0, z0, seed), b = wo_hash2i(x1, z0, seed);
    double c = wo_hash2i(x0, z1, seed), d = wo_hash2i(x1, z1, seed);
    return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}

static double wo_fbm_tile(double x, double z, int32_t cells, int32_t oct, int32_t seed) {
    double s = 0.0, amp = 0.5, f = 1.0;
    int32_t per = cells;
    for (int32_t o = 0; o < oct; o++) {
        s += amp * wo_value_noise_tile(x * f, z * f, per, seed + o * 101);
        f *= 2.0;
        per *= 2;
        amp *= 0.5;
    }
    return s;
}

static uint8_t *wo_bake_foam(void) {
    const int32_t N = WO_FOAM_N;
    uint8_t *data = (uint8_t *)platform_calloc((size_t)N * N * 4, sizeof(uint8_t));
    if (data == NULL) return NULL;
    int32_t p = 0;
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < N; i++) {
            double x = (double)i / N, z = (double)j / N;
            double r = wo_fbm_tile(x * 8.0, z * 8.0, 8, 4, 11);
            double g = wo_fbm_tile(x * 32.0, z * 32.0, 32, 4, 23);
            double b = wo_fbm_tile(x * 64.0, z * 64.0, 64, 3, 37);
            double a = wo_fbm_tile(x * 24.0, z * 24.0, 24, 4, 51);
            data[p + 0] = (uint8_t)(r * 255);
            data[p + 1] = (uint8_t)(g * 255);
            data[p + 2] = (uint8_t)(b * 255);
            data[p + 3] = (uint8_t)(a * 255);
            p += 4;
        }
    }
    return data;
}

// ===============================================================================
// === [Terrain] 随机岛屿地形（参考 linglong.html 的 === [Islands] ===）
//
// 世界：可平铺周期瓦片（world×world，默认 25.6 km），纹素 world/res（默认 50 m）。
//       中心为世界原点；相机水平位置按 world 回绕，故“飞多远都有地”。
// 数据：高程图 h（float32，陆正海负）+ 两级“格内最大高程”粗网格（64²/8²），
//       求交只依赖这两样，换成真实 DEM 时只需替换填 h 的那一段。
// 生成：分帧渐进（每帧预算 WO_ISL_GEN_BUDGET_MS），界面不冻结。
// 查询：两级 2D DDA（远 3200m 格 / 近 400m 格）+ 纹素级细步 + 二分细化；
//       格子按 DDA 精确进出、不跳格，掠射视线也不漏。
// 着色：高程梯度法线（差分步长按像素足迹放大）、太阳直射（复用透射率 LUT）、
//       天空半球环境光（每帧常量）、大气透视、AgX —— 与海面同一套 LUT/雾。
// ===============================================================================

#define WO_ISL_DEEP            (70.0)       // 深海海底高程 m
#define WO_ISL_K               (512)        // 海岸极径表分辨率（含细碎曲折）
#define WO_ISL_KS              (64)         // 海岸极径表分辨率（低阶谐波，平滑）
#define WO_ISL_COARSE          (64)         // 近层格数（world/64 格，25.6km → 400m）
#define WO_ISL_FAR             (8)          // 远层格数（world/8 格，25.6km → 3200m）
#define WO_ISL_MAX_ISLANDS     (6)
#define WO_ISL_WORLD           (25600.0)    // 世界边长 m（与参考实现一致）
#define WO_ISL_GEN_BUDGET_MS   (4.0)        // 渐进生成每帧时间预算
#define WO_ISL_DRAW_DIST       (40000.0)    // 地形绘制距离（更远的岛整座跳过）
#define WO_ISL_FAR_LOD_DIST    (12000.0)    // 远景 LOD 分界（更远的岛用粗网格）
#define WO_ISL_DEFAULT_SEED    (20260324u)

typedef struct {
    double  cx, cz;                     // 岛心（世界坐标，不随相机回绕）
    float   coast[WO_ISL_K];            // 海岸极径表 r(θ)（含细碎曲折）
    float   coast_s[WO_ISL_KS];         // 平滑极径表（离岸 500m 后过渡到它）
    double  r_land, r_out, r_out2;      // 最大陆缘半径 / 影响半径
    double  peak, dome_exp, hill_amp;   // 圆顶峰值 / 指数 / 丘陵幅度
    double  peak_add, peak_r2;          // 额外山峰（高斯）
    double  px, pz, shelf_len, h_cap;   // 山峰偏移 / 陆架长度 / 高程上限
    int32_t hill_scale, nseed;
    int32_t bx0, bx1, bz0, bz1;         // 生成用纹素包围盒
} Wo_Isl_Island;

typedef struct {
    int32_t  enable, quality, res;
    uint32_t seed;
    double   world, texel, half, coarse_cell, far_cell;
    float   *h;                         // res² 高程 m
    float   *coarse;                    // 64² 格内最大高程
    float   *far_grid;                  // 8² 格内最大高程
    double   max_h;
    Wo_Isl_Island islands[WO_ISL_MAX_ISLANDS];
    int32_t  island_n;
    int32_t  phase;                     // 0=idle 1=layout 2=rows 3=grids 4=ready
    int32_t  row;
    uint32_t gen_seed;
    int32_t  ready;
} Wo_Isl;

static Wo_Isl s_isl;

// 观察者世界位置（x 东 / y 上 / z 北，m）。由 linglong_ocean_set_observer 每帧更新。
static double s_obs[3] = { 0.0, 5.0, 0.0 };

// 地形层（1/div 分辨率求交 + 着色，合成时对轮廓块做全分辨率细化）
typedef struct {
    int32_t  div, lw, lh;
    uint8_t *color;     // lw*lh*3（已着色的地形颜色）
    uint8_t *hit;       // lw*lh（该层像素是否是陆地）
    size_t   cap_color, cap_hit;
} Wo_TerrLayer;

static Wo_TerrLayer s_layer;
static double s_terr_cover = 0.0;      // 上一帧地形覆盖率（命中块 / 层块）——诊断与自适应用
static double s_terr_cand_frac = 0.3;  // 上一帧“进了包围盒”的层像素比例
static double s_terr_hit_frac = 0.1;   // 上一帧命中的层像素比例

// ---- 海岸极径表：周期插值噪声 ----
static double wo_isl_tab_noise(const float *tab, double th, int32_t n) {
    double u = (th / (2.0 * WO_PI)) * (double)n;
    double i0d = floor(u);
    int32_t i0 = (int32_t)i0d;
    double f = u - i0d;
    f = f * f * (3.0 - 2.0 * f);
    int32_t a = ((i0 % n) + n) % n;
    int32_t b = (((i0 + 1) % n) + n) % n;
    return (double)tab[a] + ((double)tab[b] - (double)tab[a]) * f;
}

// ---- 周期表采样（线性插值 + 回绕）----
static double wo_isl_tab_sample(const float *tab, int32_t K, double th) {
    double u = th * ((double)K / (2.0 * WO_PI));
    u -= (double)K * floor(u / (double)K);
    int32_t i0 = (int32_t)u;
    double f = u - (double)i0;
    double a = (double)tab[i0];
    double b = (double)tab[(i0 + 1 < K) ? (i0 + 1) : 0];
    return a + (b - a) * f;
}

// ---- 海岸线极径表：低阶谐波（海湾/半岛）+ 两个尺度的角度噪声（小尺度曲折）----
static void wo_isl_coast_table(Wo_Rng *rnd, double R0, Wo_Isl_Island *is) {
    static const int32_t ks[5] = { 2, 3, 4, 5, 7 };
    double ka[5], kp[5];
    for (int32_t i = 0; i < 5; i++) {
        ka[i] = (0.030 + wo_rng_next(rnd) * 0.075) * ((i < 3) ? 1.0 : 0.7);
        kp[i] = wo_rng_next(rnd) * 2.0 * WO_PI;
    }
    // 角度噪声格数：约 460 m / 160 m 尺度（R≈4.5 km 时，与参考实现一致）
    const int32_t n1 = 61, n2 = 181;
    float t1[61], t2[181];
    for (int32_t i = 0; i < n1; i++) t1[i] = (float)(wo_rng_next(rnd) * 2.0 - 1.0);
    for (int32_t i = 0; i < n2; i++) t2[i] = (float)(wo_rng_next(rnd) * 2.0 - 1.0);
    for (int32_t i = 0; i < WO_ISL_KS; i++) {
        double th = ((double)i / (double)WO_ISL_KS) * 2.0 * WO_PI;
        double r = 1.0;
        for (int32_t k = 0; k < 5; k++) r += ka[k] * sin((double)ks[k] * th + kp[k]);
        if (r < 0.30) r = 0.30;
        is->coast_s[i] = (float)(r * R0);
    }
    for (int32_t i = 0; i < WO_ISL_K; i++) {
        double th = ((double)i / (double)WO_ISL_K) * 2.0 * WO_PI;
        double r = 1.0;
        for (int32_t k = 0; k < 5; k++) r += ka[k] * sin((double)ks[k] * th + kp[k]);
        r += 0.048 * wo_isl_tab_noise(t1, th, n1);
        r += 0.016 * wo_isl_tab_noise(t2, th, n2);
        if (r < 0.30) r = 0.30;
        is->coast[i] = (float)(r * R0);
    }
}

// ---- 单座岛的高程场（m，相对平均海平面）----
static double wo_isl_field_at(const Wo_Isl_Island *is, double x, double z) {
    const double dx = x - is->cx, dz = z - is->cz;
    const double d2 = dx * dx + dz * dz;
    if (d2 > is->r_out2) return -WO_ISL_DEEP;
    const double d = sqrt(d2);
    const double th = atan2(dz, dx);
    const double rc = wo_isl_tab_sample(is->coast, WO_ISL_K, th);
    const double rc_s = wo_isl_tab_sample(is->coast_s, WO_ISL_KS, th);
    if (d >= rc) {
        // 海床：从海线迅速跌到陆架深度，再渐近深海。近岸跟随曲折海线，
        // 离岸 500 m 后过渡到平滑海线（否则曲折细节的外偏移会在浅水里产生放射状条纹）。
        const double d_out = d - rc;
        const double w = wo_smoothstep(0.0, 500.0, d_out);
        const double d_eff = d - (rc + (rc_s - rc) * w);
        if (d_eff <= 0.0) return 0.0;
        return -WO_ISL_DEEP * (1.0 - exp(-d_eff / is->shelf_len));
    }
    const double inv = 1.0 - d / rc;                  // 0 = 海线，1 = 岛心
    double hh = is->peak * pow(inv, is->dome_exp);
    const double fade = wo_smoothstep(0.02, 0.45, inv);   // 丘陵在海线附近淡出
    if (fade > 0.0) {
        const double nx = (x + s_isl.half) * (1.0 / s_isl.world);
        const double nz = (z + s_isl.half) * (1.0 / s_isl.world);
        // 域扭曲：消除值噪声的方格点“绗缝”感
        const double wx = (wo_value_noise_tile(nx * 8.0, nz * 8.0, 8, is->nseed + 31) - 0.5) * 0.9;
        const double wz = (wo_value_noise_tile(nx * 8.0 + 3.0, nz * 8.0 + 5.0, 8, is->nseed + 57) - 0.5) * 0.9;
        const double hx = nx * (double)is->hill_scale + wx;
        const double hz = nz * (double)is->hill_scale + wz;
        const double f1 = wo_fbm_tile(hx, hz, is->hill_scale, 5, is->nseed) - 0.5;
        const double f2r = wo_fbm_tile(hx * 2.0, hz * 2.0, is->hill_scale * 2, 4, is->nseed + 7);
        const double f2 = (1.0 - fabs(2.0 * f2r - 1.0)) - 0.5;    // 山脊（ridged）
        hh += (f1 * 0.60 + f2 * 0.40) * 2.0 * is->hill_amp * fade;
        if (is->peak_add > 0.0) {
            const double ax = dx - is->px, az = dz - is->pz;
            hh += is->peak_add * fade * exp(-(ax * ax + az * az) / is->peak_r2);
        }
    }
    if (hh > is->h_cap) hh = is->h_cap;
    // 海线附近的细碎起伏：把水线打成锯齿状（小海湾/小岬角）
    if (hh > -14.0 && hh < 14.0) {
        const double nx2 = (x + s_isl.half) * (1.0 / s_isl.world);
        const double nz2 = (z + s_isl.half) * (1.0 / s_isl.world);
        const double w2 = 1.0 - fabs(hh) * (1.0 / 14.0);
        const double rp = wo_fbm_tile(nx2 * 220.0, nz2 * 220.0, 44, 3, is->nseed + 977) - 0.5;
        hh += rp * 3.6 * w2 * w2;
    }
    return hh;
}

// ---- 造一座岛（随机参数，全部来自同一 rnd 序列，保证与参考逐位一致）----
static void wo_isl_make(double cx, double cz, double R0, Wo_Rng *rnd, double hs, Wo_Isl_Island *is) {
    memset(is, 0, sizeof(*is));
    is->cx = cx;
    is->cz = cz;
    wo_isl_coast_table(rnd, R0, is);
    double r_land = 0.0;
    for (int32_t i = 0; i < WO_ISL_K; i++) if ((double)is->coast[i] > r_land) r_land = (double)is->coast[i];
    const double peak = R0 * (0.095 + wo_rng_next(rnd) * 0.115) * hs;
    const double az = wo_rng_next(rnd) * 2.0 * WO_PI;
    const double pd = r_land * (0.20 + wo_rng_next(rnd) * 0.25);
    const double r_out = r_land + 2600.0;
    is->r_land = r_land;
    is->r_out = r_out;
    is->r_out2 = r_out * r_out;
    is->peak = peak;
    is->dome_exp = 1.12 + wo_rng_next(rnd) * 0.22;
    is->hill_amp = peak * (0.18 + wo_rng_next(rnd) * 0.14);
    is->hill_scale = 18 + (int32_t)(wo_rng_next(rnd) * 16.0);
    is->peak_add = peak * (0.20 + wo_rng_next(rnd) * 0.35);
    const double pr = R0 * (0.26 + wo_rng_next(rnd) * 0.14);
    is->peak_r2 = pr * pr;
    is->px = cos(az) * pd;
    is->pz = sin(az) * pd;
    is->shelf_len = 950.0 + wo_rng_next(rnd) * 550.0;
    is->h_cap = 980.0;
    is->nseed = (int32_t)(wo_rng_next(rnd) * 2147483647.0);
    // 生成用纹素包围盒（世界中心在原点，故加 half）
    const double lim = s_isl.half - s_isl.texel;
    int32_t bx0 = (int32_t)floor((cx - r_out + s_isl.half) / s_isl.texel);
    int32_t bx1 = (int32_t)ceil((cx + r_out + s_isl.half) / s_isl.texel);
    int32_t bz0 = (int32_t)floor((cz - r_out + s_isl.half) / s_isl.texel);
    int32_t bz1 = (int32_t)ceil((cz + r_out + s_isl.half) / s_isl.texel);
    if (bx0 < 0) bx0 = 0;
    if (bz0 < 0) bz0 = 0;
    if (bx1 > s_isl.res - 1) bx1 = s_isl.res - 1;
    if (bz1 > s_isl.res - 1) bz1 = s_isl.res - 1;
    if (bx1 < bx0) bx1 = bx0;
    if (bz1 < bz0) bz1 = bz0;
    is->bx0 = bx0; is->bx1 = bx1; is->bz0 = bz0; is->bz1 = bz1;
    (void)lim;
}

// ---- 岛布局：一座主角岛靠近相机起点（海岸 2.6~5.2 km），外加 2~3 座小岛 ----
static int32_t wo_isl_layout(Wo_Rng *rnd) {
    const double W = s_isl.world, half = s_isl.half;
    const double lim_main = half - 700.0;          // 陆地不越过贴图边界
    const double lim_small = half - 800.0;
    const double pos_range = 0.4375 * W;           // 小岛随机分布范围（25.6km → 11200m）
    int32_t n = 0;

    for (int32_t att = 0; att < 60 && n < WO_ISL_MAX_ISLANDS; att++) {
        const double R0 = 3400.0 + wo_rng_next(rnd) * 1200.0;
        // 主角岛的方位角钉在默认视线附近（天象仪默认朝南 view_azi=180°，取 195°±15°）：
        // 否则开局有一半概率背对岛屿。仍取一次随机数，保持与参考实现相同的随机序列长度。
        const double az_rand = wo_rng_next(rnd);
        const double az = (195.0 + (az_rand - 0.5) * 30.0) * (WO_PI / 180.0);
        const double dist = R0 + 2600.0 + wo_rng_next(rnd) * 2400.0;
        // 方位角是罗盘方位（自北向东）：x=东=sin、z=北=cos
        const double cx = sin(az) * dist, cz = cos(az) * dist;
        wo_isl_make(cx, cz, R0, rnd, 1.0, &s_isl.islands[n]);
        if (fabs(cx) + s_isl.islands[n].r_land > lim_main) continue;
        if (fabs(cz) + s_isl.islands[n].r_land > lim_main) continue;
        n++;
        break;
    }
    if (n == 0) {
        wo_isl_make(0.0, -7000.0, 3200.0, rnd, 1.0, &s_isl.islands[0]);
        n = 1;
    }

    const int32_t want = 2 + (int32_t)(wo_rng_next(rnd) * 2.0);
    for (int32_t k = 0; k < want && n < WO_ISL_MAX_ISLANDS; k++) {
        for (int32_t att = 0; att < 240; att++) {
            const double R0 = 1400.0 + wo_rng_next(rnd) * 2800.0;
            const double cx = (wo_rng_next(rnd) * 2.0 - 1.0) * pos_range;
            const double cz = (wo_rng_next(rnd) * 2.0 - 1.0) * pos_range;
            if (fabs(cx) + R0 * 1.33 > lim_small) continue;
            if (fabs(cz) + R0 * 1.33 > lim_small) continue;
            int32_t ok = 1;
            for (int32_t i = 0; i < n; i++) {
                const Wo_Isl_Island *o = &s_isl.islands[i];
                const double ddx = cx - o->cx, ddz = cz - o->cz;
                if (sqrt(ddx * ddx + ddz * ddz) < o->r_land + R0 * 1.33 + 2900.0) { ok = 0; break; }
            }
            if (!ok) continue;
            wo_isl_make(cx, cz, R0, rnd, 0.8 + wo_rng_next(rnd) * 0.45, &s_isl.islands[n]);
            n++;
            break;
        }
    }
    return n;
}

// ---- 两级“格内最大高程”网格（程序化生成与真实 DEM 共用）----
static int32_t wo_isl_build_grids(void) {
    const int32_t N = s_isl.res, C = WO_ISL_COARSE, F = WO_ISL_FAR;
    const int32_t cs = N / C, fs = C / F;
    if (cs <= 0 || fs <= 0) return 0;
    s_isl.coarse = (float *)platform_calloc((size_t)C * C, sizeof(float));
    s_isl.far_grid = (float *)platform_calloc((size_t)F * F, sizeof(float));
    if (s_isl.coarse == NULL || s_isl.far_grid == NULL) return 0;
    for (int32_t j = 0; j < C; j++) {
        for (int32_t i = 0; i < C; i++) {
            float m = -1e9f;
            for (int32_t jj = 0; jj < cs; jj++) {
                const float *row = s_isl.h + (size_t)(j * cs + jj) * N + i * cs;
                for (int32_t ii = 0; ii < cs; ii++) if (row[ii] > m) m = row[ii];
            }
            s_isl.coarse[j * C + i] = m;
        }
    }
    for (int32_t j = 0; j < F; j++) {
        for (int32_t i = 0; i < F; i++) {
            float m = -1e9f;
            for (int32_t jj = 0; jj < fs; jj++) {
                const float *row = s_isl.coarse + (size_t)(j * fs + jj) * C + i * fs;
                for (int32_t ii = 0; ii < fs; ii++) if (row[ii] > m) m = row[ii];
            }
            s_isl.far_grid[j * F + i] = m;
        }
    }
    double mx = -1e9;
    for (size_t i = 0; i < (size_t)N * N; i++) if ((double)s_isl.h[i] > mx) mx = (double)s_isl.h[i];
    s_isl.max_h = mx;
    return 1;
}

// ---- 渐进生成：分配 + 布局 ----
static void wo_isl_free_data(void) {
    if (s_isl.h != NULL) { free(s_isl.h); s_isl.h = NULL; }
    if (s_isl.coarse != NULL) { free(s_isl.coarse); s_isl.coarse = NULL; }
    if (s_isl.far_grid != NULL) { free(s_isl.far_grid); s_isl.far_grid = NULL; }
    s_isl.max_h = 0.0;
    s_isl.island_n = 0;
    s_isl.phase = 0;
    s_isl.row = 0;
    s_isl.ready = 0;
}

static void wo_isl_begin(uint32_t seed) {
    wo_isl_free_data();
    s_isl.seed = seed;
    s_isl.gen_seed = seed;
    s_isl.world = WO_ISL_WORLD;
    s_isl.texel = s_isl.world / (double)s_isl.res;
    s_isl.half = s_isl.world * 0.5;
    s_isl.coarse_cell = s_isl.world / (double)WO_ISL_COARSE;
    s_isl.far_cell = s_isl.world / (double)WO_ISL_FAR;

    s_isl.h = (float *)platform_calloc((size_t)s_isl.res * s_isl.res, sizeof(float));
    if (s_isl.h == NULL) return;
    for (size_t i = 0; i < (size_t)s_isl.res * s_isl.res; i++) s_isl.h[i] = (float)(-WO_ISL_DEEP);

    Wo_Rng rnd;
    rnd.a = seed;
    s_isl.island_n = wo_isl_layout(&rnd);
    s_isl.phase = 2;
    s_isl.row = 0;
    s_isl.ready = 0;
}

static void wo_isl_gen_row(int32_t j) {
    const int32_t N = s_isl.res;
    const double wz = ((double)j + 0.5) * s_isl.texel - s_isl.half;
    float *row = s_isl.h + (size_t)j * N;
    for (int32_t k = 0; k < s_isl.island_n; k++) {
        const Wo_Isl_Island *is = &s_isl.islands[k];
        if (j < is->bz0 || j > is->bz1) continue;
        for (int32_t i = is->bx0; i <= is->bx1; i++) {
            const double wx = ((double)i + 0.5) * s_isl.texel - s_isl.half;
            const double v = wo_isl_field_at(is, wx, wz);
            if (v > (double)row[i]) row[i] = (float)v;
        }
    }
}

// 每帧推进一次（由 linglong_ocean_prepare 调用）
static void wo_isl_step(void) {
    if (!s_isl.enable) return;
    if (s_isl.phase == 0) {
        wo_isl_begin(s_isl.seed != 0 ? s_isl.seed : WO_ISL_DEFAULT_SEED);
        if (s_isl.phase == 0) return;
    }
    if (s_isl.phase == 2) {
        const double t0 = (double)get_timestamp_in_ms();
        int32_t guard = 0;
        while (s_isl.row < s_isl.res) {
            wo_isl_gen_row(s_isl.row);
            s_isl.row++;
            if ((++guard & 7) == 0) {
                if (((double)get_timestamp_in_ms() - t0) >= WO_ISL_GEN_BUDGET_MS) break;
            }
        }
        if (s_isl.row >= s_isl.res) s_isl.phase = 3;
    }
    if (s_isl.phase == 3) {
        if (wo_isl_build_grids()) {
            s_isl.phase = 4;
            s_isl.ready = 1;
        }
        else {
            wo_isl_free_data();   // 内存不足：退回纯深水
        }
    }
}

// ===============================================================================
// 地形高程/求交
// ===============================================================================

// 双线性 + 周期回绕（与 GLSL texture() 行为一致）
static double wo_isl_height(double x, double z) {
    const float *H = s_isl.h;
    if (H == NULL) return -WO_ISL_DEEP;
    const int32_t N = s_isl.res;
    const double u = (x + s_isl.half) / s_isl.texel - 0.5;
    const double v = (z + s_isl.half) / s_isl.texel - 0.5;
    const double i0d = floor(u), j0d = floor(v);
    const double fu = u - i0d, fv = v - j0d;
    const int32_t i0 = ((int32_t)i0d % N + N) % N;
    const int32_t j0 = ((int32_t)j0d % N + N) % N;
    const int32_t i1 = (i0 + 1 < N) ? (i0 + 1) : 0;
    const int32_t j1 = (j0 + 1 < N) ? (j0 + 1) : 0;
    const double a = (double)H[(size_t)j0 * N + i0];
    const double b = (double)H[(size_t)j0 * N + i1];
    const double c = (double)H[(size_t)j1 * N + i0];
    const double d = (double)H[(size_t)j1 * N + i1];
    return (a + (b - a) * fu) * (1.0 - fv) + (c + (d - c) * fu) * fv;
}

// 粗网格采样（最近邻）
static double wo_isl_cell_max(double x, double z, int32_t far) {
    const int32_t N = far ? WO_ISL_FAR : WO_ISL_COARSE;
    const float *arr = far ? s_isl.far_grid : s_isl.coarse;
    if (arr == NULL) return -1e9;
    const double cell = far ? s_isl.far_cell : s_isl.coarse_cell;
    int32_t i = (int32_t)floor((x + s_isl.half) / cell);
    int32_t j = (int32_t)floor((z + s_isl.half) / cell);
    i = ((i % N) + N) % N;
    j = ((j % N) + N) % N;
    return (double)arr[j * N + i];
}

// 射线穿越当前格子的出口距离（保证最小推进 = 0.2% 格边长，避免 float 舍入原地打转）
static double wo_isl_cell_exit(double px, double pz, double dx, double dz, double cell) {
    double tx = 1e9, tz = 1e9;
    if (dx > 1e-9 || dx < -1e-9) {
        const double i = floor((px + s_isl.half) / cell);
        tx = ((i + ((dx > 0.0) ? 1.0 : 0.0)) * cell - s_isl.half - px) / dx;
    }
    if (dz > 1e-9 || dz < -1e-9) {
        const double j = floor((pz + s_isl.half) / cell);
        tz = ((j + ((dz > 0.0) ? 1.0 : 0.0)) * cell - s_isl.half - pz) / dz;
    }
    if (tx < 0.0) tx = 0.0;
    if (tz < 0.0) tz = 0.0;
    double out = (tx < tz) ? tx : tz;
    const double min_step = 0.002 * cell;
    if (out < min_step) out = min_step;
    return out;
}

// 在 [tA,tB] 内按纹素步进取第一个交点（内部二分到毫米级）
static double wo_isl_fine_hit(double cx, double cy, double cz,
                              double dx, double dy, double dz,
                              double tA, double tB, double hl) {
    const double step = 0.9 * s_isl.texel / hl;
    double t_pos = tA;
    if (cy + dy * t_pos - wo_isl_height(cx + dx * t_pos, cz + dz * t_pos) <= 0.0) {
        return (t_pos > 1e-3) ? t_pos : 1e-3;
    }
    for (int32_t i = 0; i < 48; i++) {
        const double t_next = (t_pos + step < tB) ? (t_pos + step) : tB;
        if (cy + dy * t_next - wo_isl_height(cx + dx * t_next, cz + dz * t_next) <= 0.0) {
            // 二分细化：容差 1 m 或 t·1e-4（远处 1 m 的几何精度远低于一个像素足迹；
            // 参考实现的 20 次固定二分等于固定算到毫米级，实测占求交采样数的 40%）。
            double lo = t_pos, hi = t_next;
            const double tol = wo_max(1.0, hi * 1e-4);
            for (int32_t r = 0; r < 20 && (hi - lo) > tol; r++) {
                const double mid = 0.5 * (lo + hi);
                if (cy + dy * mid - wo_isl_height(cx + dx * mid, cz + dz * mid) > 0.0) lo = mid;
                else hi = mid;
            }
            return hi;
        }
        t_pos = t_next;
        if (t_pos >= tB) break;
    }
    return -1.0;
}

// 射线与地形求交（海洋系；返回最近交点距离 t，未命中返回 -1）
// 两级「格内最大高程」网格做粗筛，格子按 2D DDA 逐格穿越（不跳格）：
// 只要射线在该格出口处仍高于格内最大高程，整格跳过；否则降一级。
static double wo_isl_ray_hit(double cx, double cy, double cz,
                             double dx, double dy, double dz, double t_max) {
    if (s_isl.h == NULL) return -1.0;
    const double hl = sqrt(dx * dx + dz * dz);
    if (hl < 1e-7) return (cy <= wo_isl_height(cx, cz)) ? 1e-3 : -1.0;
    double t_end = wo_min(t_max, s_isl.world / hl);   // 一个世界周期内找不到就不必再找
    if (dy > 1e-9) {
        const double t_lim = (s_isl.max_h - cy) / dy;  // 上升射线越过全图最高点即不可能命中
        if (t_lim <= 0.0) return -1.0;
        if (t_lim < t_end) t_end = t_lim;
    }
    if (!(t_end > 0.0)) return -1.0;

    double t = 0.0;
    for (int32_t i = 0; i < 24; i++) {
        if (t >= t_end) return -1.0;
        const double px = cx + dx * t, pz = cz + dz * t;
        double t_exit = t + wo_isl_cell_exit(px, pz, dx, dz, s_isl.far_cell);
        if (t_exit > t_end) t_exit = t_end;
        if (t_exit <= t) return -1.0;
        if (cy + dy * t_exit > wo_isl_cell_max(px + dx * 0.5, pz + dz * 0.5, 1)) { t = t_exit; continue; }
        double tn = t;
        for (int32_t j = 0; j < 32; j++) {
            if (tn >= t_exit) break;
            const double pnx = cx + dx * tn, pnz = cz + dz * tn;
            double tn_exit = tn + wo_isl_cell_exit(pnx, pnz, dx, dz, s_isl.coarse_cell);
            if (tn_exit > t_exit) tn_exit = t_exit;
            if (tn_exit <= tn) break;
            if (cy + dy * tn_exit <= wo_isl_cell_max(pnx + dx * 0.5, pnz + dz * 0.5, 0)) {
                const double hit = wo_isl_fine_hit(cx, cy, cz, dx, dy, dz, tn, tn_exit, hl);
                if (hit > 0.0) return hit;
            }
            tn = tn_exit;
        }
        t = t_exit;
    }
    return -1.0;
}

// ---- 岛屿包围球（相机位置随移动变化，故每帧按“最近的那个世界副本”取心）----
// 返回射线离开该球体的距离（球心在 maxH/2 高度、半径覆盖整座岛与其海底坡度）：
// 岛体一定在球内，故可用作求交距离的上界；命中与否由 DDA 决定。
// 未与任何岛的球体相交返回 -1（整像素直接跳过求交）。
// 每帧一次：把各岛按“最近的世界副本”搬到相机附近，缓存球心/半径²
static double s_isl_img_x[WO_ISL_MAX_ISLANDS];
static double s_isl_img_z[WO_ISL_MAX_ISLANDS];
static double s_isl_img_r[WO_ISL_MAX_ISLANDS];
static void wo_isl_img_update(void) {
    const double W = s_isl.world;
    const double rr0 = s_isl.max_h * s_isl.max_h * 0.25;
    for (int32_t k = 0; k < s_isl.island_n; k++) {
        const Wo_Isl_Island *is = &s_isl.islands[k];
        s_isl_img_x[k] = is->cx + W * round((s_obs[0] - is->cx) / W);
        s_isl_img_z[k] = is->cz + W * round((s_obs[2] - is->cz) / W);
        // 用「陆缘半径」而不是影响半径：高于海平面的地形只可能出现在海线以内，
        // 海底坡度那 2.6 km 只影响海面浅水色，与地形轮廓无关。包围球收紧之后
        // 屏幕包围盒也随之收紧（海面上大片“可能在岛附近但看不到陆地”的像素直接跳过）。
        const double r_land = is->r_land + 400.0;
        s_isl_img_r[k] = r_land * r_land + rr0;
    }
}

static double wo_isl_sphere_exit(double dx, double dy, double dz) {
    double t_cap = 0.0;
    int32_t cand = 0;
    for (int32_t k = 0; k < s_isl.island_n; k++) {
        const double ox = s_obs[0] - s_isl_img_x[k];
        const double oy = s_obs[1] - s_isl.max_h * 0.5;
        const double oz = s_obs[2] - s_isl_img_z[k];
        const double rr = s_isl_img_r[k];
        const double b = ox * dx + oy * dy + oz * dz;         // o·d（o = 相机 − 球心）
        const double c = ox * ox + oy * oy + oz * oz - rr;
        const double disc = b * b - c;
        if (disc < 0.0) continue;
        const double t_out = -b + sqrt(disc);
        if (t_out <= 0.0) continue;
        if (!cand || t_out > t_cap) { t_cap = t_out; cand = 1; }
    }
    return cand ? t_cap : -1.0;
}

// 候选 + 求交；命中返回 1 并写 *t_out
static int32_t wo_terr_trace(double dx, double dy, double dz, double *t_out) {
    // 岛体只有在比海面更近时才算可见：海面交点即海平面 y=0 处，海床（海面之下的
    // 高程）永远不该盖到海面上。地平线以上的射线没有海面约束，照旧搜整张图。
    double t_max = WO_ISL_DRAW_DIST;
    if (dy < -1e-4) {
        const double t_plane = -s_obs[1] / dy;
        if (t_plane < t_max) t_max = t_plane;
    }
    const double t_cap = wo_isl_sphere_exit(dx, dy, dz);
    if (t_cap < 0.0) return 0;
    if (t_cap < t_max) t_max = t_cap;
    const double t = wo_isl_ray_hit(s_obs[0], s_obs[1], s_obs[2], dx, dy, dz, t_max);
    if (t <= 0.0) return 0;
    *t_out = t;
    return 1;
}

// ---- 地形着色（阳光/天空/雾全部取自与海面同一份 LUT/常量）----
static void wo_isl_shade(double dx, double dy, double dz, double t_hit, double *out) {
    const double px = s_obs[0] + dx * t_hit;
    const double pz = s_obs[2] + dz * t_hit;

    // 法线：差分步长随像素足迹放大（远处等效 mip，避免近处噪点、远处锯齿）
    double e = t_hit * s_state.pix_ang * 1.3;
    if (e < s_isl.texel) e = s_isl.texel;
    if (e > 200.0) e = 200.0;
    const double hx = (wo_isl_height(px + e, pz) - wo_isl_height(px - e, pz)) / (2.0 * e);
    const double hz = (wo_isl_height(px, pz + e) - wo_isl_height(px, pz - e)) / (2.0 * e);
    const double nl = 1.0 / sqrt(hx * hx + 1.0 + hz * hz);
    const double nx = -hx * nl, ny = nl, nz = -hz * nl;

    const double hgt = wo_isl_height(px, pz);
    const double slope = 1.0 - ny;

    // 反照率：草地绿（噪声调制）+ 陡坡偏岩 + 海线沙滩带；远处按足迹淡出到均值
    double tex[4];
    const double alb_w = 1.0 - wo_smoothstep(2500.0, 9000.0, t_hit);
    wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0, px * 0.00042, pz * 0.00042, tex);
    double n1 = wo_mix(0.5, tex[0] * (1.0 / 255.0), alb_w);
    wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0, px * 0.0019 + 0.31, pz * 0.0019, tex);
    double n2 = wo_mix(0.5, tex[1] * (1.0 / 255.0), alb_w);
    const double n3 = wo_mix(0.5, tex[2] * (1.0 / 255.0), alb_w);
    const double detail_w = 1.0 - wo_smoothstep(120.0, 900.0, t_hit);
    if (detail_w > 0.01) {
        wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0, px * 0.021 + 0.13, pz * 0.021 + 0.57, tex);
        n1 += (tex[2] * (1.0 / 255.0) - 0.5) * 0.20 * detail_w;
        n2 += (tex[0] * (1.0 / 255.0) - 0.5) * 0.16 * detail_w;
    }
    const double grass_mix = n1 * 0.65 + n2 * 0.35;
    double a0 = wo_mix(0.042, 0.105, grass_mix);
    double a1 = wo_mix(0.120 + 0.040 * n3, 0.245, grass_mix);
    double a2 = wo_mix(0.030, 0.062, grass_mix);
    const double rock = wo_smoothstep(0.30, 0.75, slope) * 0.65;
    a0 = wo_mix(a0, 0.100, rock);
    a1 = wo_mix(a1, 0.095, rock);
    a2 = wo_mix(a2, 0.078, rock);
    const double sand = wo_smoothstep(1.2, 0.2, hgt);
    a0 = wo_mix(a0, 0.430, sand);
    a1 = wo_mix(a1, 0.380, sand);
    a2 = wo_mix(a2, 0.270, sand);

    // 太阳辐照与天空环境（每帧常量，与海面共用）
    const double sun_e0 = s_state.sun_e[0], sun_e1 = s_state.sun_e[1], sun_e2 = s_state.sun_e[2];
    const double hemi0 = s_state.hemi[0], hemi1 = s_state.hemi[1], hemi2 = s_state.hemi[2];
    double sky_tmp[3];
    wo_sky_atmo(nx, ny, nz, sky_tmp);
    const double sky_n0 = sky_tmp[0], sky_n1 = sky_tmp[1], sky_n2 = sky_tmp[2];
    const double ldx = s_state.sun_dir[0], ldy = s_state.sun_dir[1], ldz = s_state.sun_dir[2];
    const double ndl = wo_max(nx * ldx + ny * ldy + nz * ldz, 0.0);
    // 天空环境光先向亮度去饱和再压低，否则蓝天的高蓝光会把草地染成青灰
    // 夜间环境光与海面同倍提升（WO_NIGHT_AMB_GAIN），否则夜里的岛屿会变成比海面更黑的剪影。
    const double kAmb = 1.0 + WO_NIGHT_AMB_GAIN * s_state.night_vis;
    const double amb0 = hemi0 + sky_n0 * 0.35, amb1 = hemi1 + sky_n1 * 0.35, amb2 = hemi2 + sky_n2 * 0.35;
    const double amb_l = 0.2126 * amb0 + 0.7152 * amb1 + 0.0722 * amb2;
    const double e0 = wo_mix(amb0, amb_l, 0.40) * 0.90 * kAmb;
    const double e1 = wo_mix(amb1, amb_l, 0.40) * 0.90 * kAmb;
    const double e2 = wo_mix(amb2, amb_l, 0.40) * 0.90 * kAmb;
    double col0 = a0 * (sun_e0 * ndl * 0.318 * 0.5 + e0);
    double col1 = a1 * (sun_e1 * ndl * 0.318 * 0.5 + e1);
    double col2 = a2 * (sun_e2 * ndl * 0.318 * 0.5 + e2);

    // 大气透视（与海面同一条雾）
    const double td = wo_fast_exp(-(double)s_cfg.fog_floor * t_hit);
    const double hzl = 1.0 / sqrt(dx * dx + 0.002 * 0.002 + dz * dz);
    wo_sky_atmo(dx * hzl, 0.002, dz * hzl, sky_tmp);
    col0 = col0 * td + sky_tmp[0] * (1.0 - td);
    col1 = col1 * td + sky_tmp[1] * (1.0 - td);
    col2 = col2 * td + sky_tmp[2] * (1.0 - td);

    wo_agx_tonemap(col0, col1, col2, out);
}

// ===============================================================================
// 地形层：1/div 分辨率求交 + 着色 → 合成（轮廓块全分辨率细化）
// ===============================================================================

static int32_t wo_terr_layer_ensure(int32_t div, int32_t lw, int32_t lh) {
    const size_t nc = (size_t)lw * lh * 3, nh = (size_t)lw * lh;
    if (s_layer.lw != lw || s_layer.lh != lh || s_layer.div != div) {
        s_layer.lw = lw; s_layer.lh = lh; s_layer.div = div;
    }
    if (nc > s_layer.cap_color) {
        free(s_layer.color);
        s_layer.color = (uint8_t *)malloc(nc);
        s_layer.cap_color = s_layer.color ? nc : 0;
    }
    if (nh > s_layer.cap_hit) {
        free(s_layer.hit);
        s_layer.hit = (uint8_t *)malloc(nh);
        s_layer.cap_hit = s_layer.hit ? nh : 0;
    }
    return (s_layer.color != NULL && s_layer.hit != NULL) ? 1 : 0;
}

// 单个屏幕像素：方向 → 求交 → 着色 → 写帧缓冲（用于轮廓块细化）
static void wo_terr_pixel(Nano_GFX *gfx, int32_t x, int32_t y,
                          float sky_radius, float center_x, float center_y,
                          float view_alt, float view_azi, float view_roll, float f,
                          int32_t projection, int32_t use_dir_map) {
    float rx = 0.0f, ry = 0.0f, rz = 0.0f;
    if (use_dir_map) {
        fisheye_dir_map_get(x, y, &rx, &ry, &rz);
    }
    else {
        fisheye_unproject((float)x, (float)y, sky_radius, center_x, center_y,
                          view_alt, view_azi, view_roll, f, projection, &rx, &ry, &rz);
    }
    double dx = (double)rx, dy = (double)rz, dz = (double)ry;
    const double dl = sqrt(dx * dx + dy * dy + dz * dz);
    if (dl < 1e-9) return;
    dx /= dl; dy /= dl; dz /= dl;
    double t = 0.0;
    if (!wo_terr_trace(dx, dy, dz, &t)) return;
    double rgb[3];
    wo_isl_shade(dx, dy, dz, t, rgb);
    gfx_set_pixel(gfx, (uint32_t)x, (uint32_t)y,
                  (uint8_t)wo_min(255.0, rgb[0] * 255.0),
                  (uint8_t)wo_min(255.0, rgb[1] * 255.0),
                  (uint8_t)wo_min(255.0, rgb[2] * 255.0));
}

// 地形绘制（在海洋之后调用：海洋先铺底，地形覆盖其上的陆地像素）
static void wo_terr_draw(Nano_GFX *gfx,
                         float sky_radius, float center_x, float center_y,
                         float view_alt, float view_azi, float view_roll, float f,
                         int32_t projection) {
    if (!s_isl.ready || !s_isl.enable) return;
    const int32_t W = (int32_t)gfx->width, H = (int32_t)gfx->height;
    if (W <= 0 || H <= 0) return;

    // 自适应分辨率（按上一帧的候选/命中比例做像素预算）：
    //   候选（进包围盒、要跑求交的层像素）≤ 12000，命中（要着色的层像素）≤ 1500。
    // 合成走双线性上采样 + 轮廓块全分辨率细化，降采样损失的是山体内部的高频，
    // 轮廓始终是全分辨率锐利的；因此“飞在岛上方、地形铺满屏幕”时自动降到 1/4~1/6，
    // 而远景小岛仍保持 1/2。
    int32_t div = (s_isl.quality == 0) ? 3 : 2;
    for (; div < 8; div++) {
        const double n = (double)((W + div - 1) / div) * (double)((H + div - 1) / div);
        if (s_terr_cand_frac * n <= 12000.0 && s_terr_hit_frac * n <= 1500.0) break;
    }
    const int32_t lw = (W + div - 1) / div, lh = (H + div - 1) / div;
    if (!wo_terr_layer_ensure(div, lw, lh)) return;
    const int32_t use_dir_map = fisheye_dir_map_ready(W, H);
    wo_isl_img_update();

    // 各岛的屏幕包围盒（每帧一次，≤6 次投影）：把逐像素的求交限制在“可能见到岛”的
    // 矩形里。视野里没有任何岛时整趟地形遍直接跳过（漫游到开阔海面时零开销）。
    int32_t box_x0[WO_ISL_MAX_ISLANDS], box_x1[WO_ISL_MAX_ISLANDS];
    int32_t box_y0[WO_ISL_MAX_ISLANDS], box_y1[WO_ISL_MAX_ISLANDS];
    int32_t box_n = 0;
    for (int32_t k = 0; k < s_isl.island_n; k++) {
        const double vx = s_isl_img_x[k] - s_obs[0];
        const double vy = s_isl.max_h * 0.5 - s_obs[1];
        const double vz = s_isl_img_z[k] - s_obs[2];
        const double dist = sqrt(vx * vx + vy * vy + vz * vz);
        const double r_bound = sqrt(s_isl_img_r[k]);
        if (dist <= r_bound + 1.0) {
            // 相机在包围球内：整屏都可能看到地形
            box_x0[box_n] = 0; box_y0[box_n] = 0;
            box_x1[box_n] = W - 1; box_y1[box_n] = H - 1;
            box_n++;
            continue;
        }
        const double ang = asin(wo_min(1.0, r_bound / dist));      // 包围球角半径（rad）
        const double ang_deg = ang * 180.0 / WO_PI;
        const double az_c = atan2(vx, vz) * 180.0 / WO_PI;          // 罗盘方位（自北向东）
        const double alt_c = asin(wo_clamp(vy / dist, -1.0, 1.0)) * 180.0 / WO_PI;
        double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
        const double pad_deg = ang_deg + 1.0;
        for (int32_t n = 0; n < 9; n++) {
            double alt2 = alt_c, az2 = az_c;
            if (n > 0) {
                const double th = (double)(n - 1) * (WO_PI / 4.0);
                alt2 = wo_clamp(alt_c + pad_deg * cos(th), -89.5, 89.5);
                az2 = az_c + pad_deg * sin(th) / wo_max(cos(alt_c * WO_PI / 180.0), 0.25);
                az2 = fmod(az2, 360.0);
                if (az2 < 0.0) az2 += 360.0;
            }
            float sx = 0.0f, sy = 0.0f;
            fisheye_project((float)az2, (float)alt2, sky_radius, center_x, center_y,
                            view_alt, view_azi, view_roll, f, projection, &sx, &sy);
            if (sx < x0) x0 = sx;
            if (sx > x1) x1 = sx;
            if (sy < y0) y0 = sy;
            if (sy > y1) y1 = sy;
        }
        if (x1 < -4.0 || x0 > (double)W + 4.0 || y1 < -4.0 || y0 > (double)H + 4.0) continue;
        if (x0 < 0.0) x0 = 0.0;
        if (y0 < 0.0) y0 = 0.0;
        if (x1 > (double)(W - 1)) x1 = (double)(W - 1);
        if (y1 > (double)(H - 1)) y1 = (double)(H - 1);
        box_x0[box_n] = (int32_t)x0; box_x1[box_n] = (int32_t)x1;
        box_y0[box_n] = (int32_t)y0; box_y1[box_n] = (int32_t)y1;
        box_n++;
    }

    // 1) 层构建（并行：不触碰帧缓冲）
    int32_t s_terr_cand_local = 0;
    #pragma omp parallel for schedule(static) reduction(+:s_terr_cand_local)
    for (int32_t ly = 0; ly < lh; ly++) {
        for (int32_t lx = 0; lx < lw; lx++) {
            const size_t i = (size_t)ly * lw + lx;
            s_layer.hit[i] = 0;
            int32_t px = lx * div + div / 2;
            int32_t py = ly * div + div / 2;
            if (px >= W) px = W - 1;
            if (py >= H) py = H - 1;
            int32_t in_box = 0;
            for (int32_t k = 0; k < box_n; k++) {
                if (px >= box_x0[k] && px <= box_x1[k] && py >= box_y0[k] && py <= box_y1[k]) { in_box = 1; break; }
            }
            if (!in_box) continue;
            s_terr_cand_local++;
            float rx = 0.0f, ry = 0.0f, rz = 0.0f;
            if (use_dir_map) {
                fisheye_dir_map_get(px, py, &rx, &ry, &rz);
            }
            else {
                fisheye_unproject((float)px, (float)py, sky_radius, center_x, center_y,
                                  view_alt, view_azi, view_roll, f, projection, &rx, &ry, &rz);
            }
            double dx = (double)rx, dy = (double)rz, dz = (double)ry;
            const double dl = sqrt(dx * dx + dy * dy + dz * dz);
            if (dl < 1e-9) continue;
            dx /= dl; dy /= dl; dz /= dl;
            double t = 0.0;
            if (!wo_terr_trace(dx, dy, dz, &t)) continue;
            double rgb[3];
            wo_isl_shade(dx, dy, dz, t, rgb);
            s_layer.hit[i] = 1;
            s_layer.color[i * 3 + 0] = (uint8_t)wo_min(255.0, rgb[0] * 255.0);
            s_layer.color[i * 3 + 1] = (uint8_t)wo_min(255.0, rgb[1] * 255.0);
            s_layer.color[i * 3 + 2] = (uint8_t)wo_min(255.0, rgb[2] * 255.0);
        }
    }

    // 2) 合成 + 轮廓细化（串行写帧缓冲，避免并行写脏区竞争）
    int32_t any_hit = 0;
    for (size_t i = 0; i < (size_t)lw * lh; i++) if (s_layer.hit[i]) { any_hit = 1; break; }
    if (!any_hit) {
        s_terr_cover = 0.0;
        s_terr_hit_frac = 0.0;
        s_terr_cand_frac = (double)s_terr_cand_local / (double)(lw * lh);
        return;
    }
    int32_t drawn = 0;
    for (int32_t ly = 0; ly < lh; ly++) {
        for (int32_t lx = 0; lx < lw; lx++) {
            const size_t i = (size_t)ly * lw + lx;
            if (!s_layer.hit[i]) continue;
            int32_t refine = 0;
            for (int32_t dy8 = -1; dy8 <= 1 && !refine; dy8++) {
                for (int32_t dx8 = -1; dx8 <= 1; dx8++) {
                    if (dx8 == 0 && dy8 == 0) continue;
                    const int32_t nxp = lx + dx8, nyp = ly + dy8;
                    if (nxp < 0 || nxp >= lw || nyp < 0 || nyp >= lh) continue;
                    const size_t ni = (size_t)nyp * lw + nxp;
                    // 只有“陆地/非陆地”的分界才算轮廓（海岸线与山脊剪影）。
                    // 早期版本还按层间深度断层判地形内部的遮挡边，实测在俯瞰大片山坡时
                    // 会把 30%+ 的块判成轮廓，细化遍（逐像素重求交+着色）吃掉整趟地形
                    // 3/4 的时间；地形内部的边在双线性上采样下只是柔 2~4 px，可接受。
                    if (!s_layer.hit[ni]) { refine = 1; break; }
                }
            }
            const int32_t x0 = lx * div, y0 = ly * div;
            if (!refine) {
                // 块内双线性上采样：地形是低频的（法线/反照率都随距离淡出），
                // 直接把块填成单色会留下可见的方块；四周 8 邻居全部命中时（refine 的
                // 判据），在块内按相邻层纹素做双线性插值，轮廓块仍走全分辨率细化。
                for (int32_t j = 0; j < div; j++) {
                    const int32_t y = y0 + j;
                    if (y >= H) break;
                    const double fy = ((double)j + 0.5) / (double)div - 0.5;
                    for (int32_t k = 0; k < div; k++) {
                        const int32_t x = x0 + k;
                        if (x >= W) break;
                        const double fx = ((double)k + 0.5) / (double)div - 0.5;
                        // 标准双线性：四点权重 (ox?|fx|:1-|fx|)·(oy?|fy|:1-|fy|)，总和恒为 1
                        double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;
                        for (int32_t n = 0; n < 4; n++) {
                            const int32_t ox = (n & 1) ? ((fx >= 0.0) ? 1 : -1) : 0;
                            const int32_t oy = (n & 2) ? ((fy >= 0.0) ? 1 : -1) : 0;
                            const double w = ((ox != 0) ? fabs(fx) : (1.0 - fabs(fx)))
                                           * ((oy != 0) ? fabs(fy) : (1.0 - fabs(fy)));
                            const size_t ni = (size_t)(ly + oy) * lw + (size_t)(lx + ox);
                            acc0 += w * (double)s_layer.color[ni * 3 + 0];
                            acc1 += w * (double)s_layer.color[ni * 3 + 1];
                            acc2 += w * (double)s_layer.color[ni * 3 + 2];
                        }
                        gfx_set_pixel(gfx, (uint32_t)x, (uint32_t)y,
                                      (uint8_t)wo_clamp(acc0, 0.0, 255.0),
                                      (uint8_t)wo_clamp(acc1, 0.0, 255.0),
                                      (uint8_t)wo_clamp(acc2, 0.0, 255.0));
                    }
                }
            }
            else {
                for (int32_t j = 0; j < div; j++) {
                    const int32_t y = y0 + j;
                    if (y >= H) break;
                    for (int32_t k = 0; k < div; k++) {
                        const int32_t x = x0 + k;
                        if (x >= W) break;
                        wo_terr_pixel(gfx, x, y, sky_radius, center_x, center_y,
                                      view_alt, view_azi, view_roll, f, projection, use_dir_map);
                    }
                }
            }
            drawn++;
        }
    }
    s_terr_cover = (double)drawn / (double)(lw * lh);
    s_terr_cand_frac = (double)s_terr_cand_local / (double)(lw * lh);
    s_terr_hit_frac = s_terr_cover;
    if (drawn > 0) gfx_mark_dirty_full(gfx);
}

// ===============================================================================
// 地形对外接口
// ===============================================================================

void linglong_ocean_terrain_default(Linglong_Terrain_Config *cfg) {
    if (cfg == NULL) return;
    cfg->enable = 1;
    cfg->quality = 1;
    cfg->res = 512;
    cfg->seed = WO_ISL_DEFAULT_SEED;
}

void linglong_ocean_terrain_configure(const Linglong_Terrain_Config *cfg) {
    if (cfg == NULL) return;
    const int32_t changed = (cfg->enable != s_isl.enable
                          || cfg->quality != s_isl.quality
                          || cfg->res != s_isl.res
                          || cfg->seed != s_isl.seed) ? 1 : 0;
    if (!changed && (s_isl.phase != 0 || !cfg->enable)) return;
    const int32_t enable = cfg->enable;
    const int32_t quality = (cfg->quality >= 0 && cfg->quality <= 2) ? cfg->quality : 1;
    int32_t res = (cfg->res == 1024) ? 1024 : 512;
    const uint32_t seed = (cfg->seed != 0) ? cfg->seed : WO_ISL_DEFAULT_SEED;
    s_isl.enable = enable;
    s_isl.quality = quality;
    s_isl.res = res;
    s_isl.seed = seed;
    if (!enable) {
        wo_isl_free_data();
        s_terr_cover = 0.0;
        return;
    }
    // 分辨率/种子/开关变化 → 作废并重新渐进生成（phase=0，下一帧 begin）
    if (s_isl.phase == 0 || changed) {
        if (changed) {
            wo_isl_free_data();
            s_isl.phase = 0;
        }
    }
}

int32_t linglong_ocean_terrain_progress(void) {
    if (!s_isl.enable) return 100;
    if (s_isl.ready) return 100;
    if (s_isl.phase == 0) return 0;
    if (s_isl.phase == 2) return (s_isl.res > 0) ? (s_isl.row * 90 / s_isl.res) : 0;
    return 95;
}

int32_t linglong_ocean_terrain_ready(void) {
    return (s_isl.enable && s_isl.ready) ? 1 : 0;
}

void linglong_ocean_terrain_regenerate(uint32_t seed) {
    s_isl.seed = (seed != 0) ? seed : WO_ISL_DEFAULT_SEED;
    wo_isl_free_data();
    s_isl.phase = 0;
    s_terr_cover = 0.0;
}

double linglong_ocean_terrain_height(double x, double z) {
    if (!s_isl.enable || !s_isl.ready) return -WO_ISL_DEEP;
    return wo_isl_height(x, z);
}

void linglong_ocean_set_observer(double x, double y, double z) {
    const double W = WO_ISL_WORLD;
    // 世界周期回绕：相机水平位置落在 [-W/2, W/2)，地形采样本身也按周期回绕，
    // 因此跨越边界时画面连续（不会看到瓦片接缝）。
    double wx = x - W * floor(x / W + 0.5);
    double wz = z - W * floor(z / W + 0.5);
    if (wx <= -W * 0.5) wx += W;
    if (wz <= -W * 0.5) wz += W;
    if (y < 1.5) y = 1.5;                 // 与参考 web-ocean 一致：不做水下视角
    // 不钻地：地形就绪时把观察者抬到地面之上 1.5 m（否则飞进山体后整屏是近裁剪的
    // 地面色，且与撞山判定不一致）。地形未就绪/未启用时按海平面处理。
    if (s_isl.enable && s_isl.ready) {
        const double ground = wo_isl_height(wx, wz);
        if (y < ground + 1.5) y = ground + 1.5;
    }
    s_obs[0] = wx;
    s_obs[1] = y;
    s_obs[2] = wz;
    s_cfg.cam_y = (float)y;               // 兼容：透射率/天空视图 LUT 取高度
}

void linglong_ocean_get_observer(double *x, double *y, double *z) {
    if (x) *x = s_obs[0];
    if (y) *y = s_obs[1];
    if (z) *z = s_obs[2];
}

// f32 → f16 → f32（就近舍入偶数），与原实现 CPU 路径的 RGBA16F 量化逐位一致
static double wo_to_half_round(double value) {
    union { float f; uint32_t i; } u;
    u.f = (float)value;
    uint32_t x = u.i;
    uint32_t s = (x >> 31) & 1u;
    uint32_t e32 = (x >> 23) & 0xffu;
    uint32_t m32 = x & 0x7fffffu;
    if (e32 == 0xffu) return value;                 // NaN/Inf 原样
    uint32_t h;
    if (e32 == 0) {
        h = s << 15;                                // f32 非规格化极小值 → f16 零
    }
    else {
        int32_t e = (int32_t)e32 - 127 + 15;
        if (e >= 31) {
            h = (s << 15) | 0x7c00u;                // 上溢饱和为 f16 Inf
        }
        else if (e >= 1) {
            uint32_t m = m32 >> 13;
            uint32_t rem = m32 & 0x1fffu;
            if (rem > 0x1000u || (rem == 0x1000u && (m & 1u) != 0u)) {
                m += 1;
                if (m == 0x800u) {
                    m = 0;
                    e += 1;
                    if (e >= 31) return (s != 0u) ? -INFINITY : INFINITY;
                }
            }
            h = (s << 15) | ((uint32_t)e << 10) | m;
        }
        else {
            uint32_t shift = (uint32_t)(1 - e);
            uint32_t full = m32 | 0x800000u;
            uint32_t m = full >> (13 + shift);
            uint32_t rem = full & ((1u << (13 + shift)) - 1u);
            uint32_t half = 1u << (12 + shift);
            if (rem > half || (rem == half && (m & 1u) != 0u)) m += 1;
            h = (s << 15) | m;
        }
    }
    uint32_t he = (h >> 10) & 0x1fu, hm = h & 0x3ffu, hs = (h >> 15) & 1u;
    if (he == 0) return (hs != 0u) ? -(double)hm * 5.960464477539063e-8 : (double)hm * 5.960464477539063e-8;
    if (he == 0x1fu) return (hs != 0u) ? -INFINITY : INFINITY;
    u.i = (hs << 31) | ((he - 15 + 127) << 23) | (hm << 13);
    return (double)u.f;
}

// 波场重烘焙（海况参数变更后调用）
static void wo_rebuild_waves(void) {
    double covx = 0.0, covz = 0.0, covxz = 0.0;
    for (int32_t c = 0; c < 2; c++) {
        int32_t nw = 0;
        double *list = wo_build_wave_list(&WO_TILES[c], (uint32_t)(1234 + c * 777), &nw);
        float *baked = wo_bake_tile(&WO_TILES[c], list, nw);
        if (baked != NULL) {
            size_t total = (size_t)WO_TILES[c].N * WO_TILES[c].N * 4;
            for (size_t q = 0; q < total; q++) baked[q] = (float)wo_to_half_round((double)baked[q]);
        }
        if (s_state.wave_data[c] != NULL) free(s_state.wave_data[c]);
        s_state.wave_data[c] = baked;
        s_state.wave_list_n[c] = nw;

        if (list != NULL) {
            for (int32_t w = 0; w < nw; w++) {
                const double *o = list + w * 7;
                double akx = o[3], akz = o[4];
                covx += 0.5 * akx * akx;
                covz += 0.5 * akz * akz;
                covxz += 0.5 * akx * akz;
            }
            free(list);
        }
    }
    s_state.cov[0] = covx;
    s_state.cov[1] = covz;
    s_state.cov[2] = covxz;

    if (s_state.foam_tex == NULL) s_state.foam_tex = wo_bake_foam();
    s_state.waves_ready = (s_state.wave_data[0] != NULL && s_state.wave_data[1] != NULL
                        && s_state.foam_tex != NULL) ? 1 : 0;
    s_state.last_now = -1.0;   // 避免重烘焙期间累计大步长
}

static void wo_ensure_waves(void) {
    if (s_state.waves_ready) return;
    wo_rebuild_waves();
}

// 波面高程（两级波场之和，各自按群速度顺风平流）
static double wo_height_at_fast(double px, double pz, double t) {
    return wo_tile_sample_h(s_state.wave_data[0], WO_TILES[0].N, WO_TILES[0].L,
                            px - s_scroll1[0] * t, pz - s_scroll1[1] * t)
         + wo_tile_sample_h(s_state.wave_data[1], WO_TILES[1].N, WO_TILES[1].L,
                            px - s_scroll2[0] * t, pz - s_scroll2[1] * t);
}

// ---- 由可调参数重算派生量（水体色/雾消光/风向向量/平流速度）----
static void wo_update_derived_params(void) {
    s_water_col[0] = 0.016 * s_cfg.turbidity;
    s_water_col[1] = 0.080 * s_cfg.turbidity;
    s_water_col[2] = 0.096 * s_cfg.turbidity;
    s_fog_ext[0] = (5.802e-3 + 4.40e-3 * s_cfg.haze) * 1e-3;
    s_fog_ext[1] = (13.558e-3 + 4.40e-3 * s_cfg.haze) * 1e-3;
    s_fog_ext[2] = (33.1e-3 + 4.40e-3 * s_cfg.haze) * 1e-3;
    s_wind_vx = sin(s_cfg.wind_dir * WO_PI / 180.0);
    s_wind_vz = cos(s_cfg.wind_dir * WO_PI / 180.0);
    s_swell_vx = sin(s_cfg.swell_dir * WO_PI / 180.0);
    s_swell_vz = cos(s_cfg.swell_dir * WO_PI / 180.0);
    s_scroll1[0] = s_wind_vx * WO_TILES[0].cgroup;
    s_scroll1[1] = s_wind_vz * WO_TILES[0].cgroup;
    s_scroll2[0] = s_wind_vx * WO_TILES[1].cgroup;
    s_scroll2[1] = s_wind_vz * WO_TILES[1].cgroup;
}

// ===============================================================================
// 参数
// ===============================================================================

void linglong_ocean_config_default(Linglong_Ocean_Config *cfg) {
    if (cfg == NULL) return;
    cfg->cam_y     = 5.0f;      // 观察高度默认 5m（设置界面共 5 档）
    cfg->wind_u    = 7.0f;
    cfg->wind_dir  = 118.0f;
    cfg->fetch     = 90000.0f;
    cfg->swell_hs  = 1.8f;
    cfg->swell_dir = 94.0f;
    cfg->swell_len = 120.0f;
    cfg->foam      = 0.0f;      // 泡沫强度最弱（无泡沫）
    cfg->turbidity = 0.2f;      // 水体泱浊度取区间最小值
    cfg->haze      = 0.0f;      // 霾最小（清透空气）
    cfg->fog_floor = 2.0e-5f;   // 能见度最大（≈150km）
}

void linglong_ocean_set_levels(int32_t swell_hs_level, int32_t swell_len_level) {
    if (!s_cfg_init) {
        Linglong_Ocean_Config c;
        linglong_ocean_config_default(&c);
        linglong_ocean_configure(&c);
    }
    if (swell_hs_level  < 0 || swell_hs_level  >= S_OCEAN_OPT_NUM(S_OCEAN_SWELL_HS_OPTIONS))  swell_hs_level = 3;
    if (swell_len_level < 0 || swell_len_level >= S_OCEAN_OPT_NUM(S_OCEAN_SWELL_LEN_OPTIONS)) swell_len_level = 3;

    Linglong_Ocean_Config c = s_cfg;   // 其余参数保持原值（模块默认）
    c.swell_hs  = S_OCEAN_SWELL_HS_OPTIONS[swell_hs_level].value;
    c.swell_len = S_OCEAN_SWELL_LEN_OPTIONS[swell_len_level].value;
    linglong_ocean_configure(&c);       // 内部只重建变动的部分
}

void linglong_ocean_configure(const Linglong_Ocean_Config *cfg) {
    if (cfg == NULL) return;

    // 选择性失效：只重建真正受影响的缓存，避免「调一个值就把三张 LUT + 波场全重烘」。
    //   观察高度  → 由 linglong_ocean_set_observer 每帧给出；天空视图 LUT 在 prepare 里
    //               按「相对高度阈值」重建（漫游时高度连续变化，不能靠参数比对触发）
    //   霾        → 介质 → 透射率 + 多重散射 + 天空视图 LUT 全部
    //   风/涌浪参数 → 波场（波谱 + 烘焙）/ 平流速度
    //   浑浊度/雾  → 只在派生量里（水体色 / 大气透视消光）
    const Linglong_Ocean_Config old = s_cfg;
    const int32_t had = s_cfg_init;

    s_cfg = *cfg;
    s_cfg_init = 1;
    wo_update_derived_params();
    wo_sky_view_uv_params();   // 天空视图 UV 的帧常量（只依赖 cam_y）

    if (!had) {
        // 首次配置：全部作废，由 prepare 统一构建
        if (s_state.sky_lut != NULL) { free(s_state.sky_lut); s_state.sky_lut = NULL; }
        if (s_state.ms_lut != NULL) { free(s_state.ms_lut); s_state.ms_lut = NULL; }
        if (s_state.trans_lut != NULL) { free(s_state.trans_lut); s_state.trans_lut = NULL; }
        s_state.waves_ready = 0;
        s_state.have_sun = 0;
        return;
    }

    const int32_t haze_changed = (old.haze != s_cfg.haze) ? 1 : 0;
    const int32_t wave_changed = (old.wind_u != s_cfg.wind_u
                               || old.wind_dir != s_cfg.wind_dir
                               || old.fetch != s_cfg.fetch
                               || old.swell_hs != s_cfg.swell_hs
                               || old.swell_dir != s_cfg.swell_dir
                               || old.swell_len != s_cfg.swell_len) ? 1 : 0;

    if (haze_changed) {
        // 霾进介质：三张 LUT 都要重算（prepare 里按 NULL 重建）
        if (s_state.sky_lut != NULL) { free(s_state.sky_lut); s_state.sky_lut = NULL; }
        if (s_state.ms_lut != NULL) { free(s_state.ms_lut); s_state.ms_lut = NULL; }
        if (s_state.trans_lut != NULL) { free(s_state.trans_lut); s_state.trans_lut = NULL; }
        s_state.have_sun = 0;
    }

    if (wave_changed) {
        s_state.waves_ready = 0;   // 下次 prepare 重烘焙波场
    }
}

void linglong_ocean_free(void) {
    if (s_state.trans_lut != NULL) { free(s_state.trans_lut); s_state.trans_lut = NULL; }
    if (s_state.ms_lut != NULL) { free(s_state.ms_lut); s_state.ms_lut = NULL; }
    if (s_state.sky_lut != NULL) { free(s_state.sky_lut); s_state.sky_lut = NULL; }
    for (int32_t c = 0; c < 2; c++) {
        if (s_state.wave_data[c] != NULL) { free(s_state.wave_data[c]); s_state.wave_data[c] = NULL; }
    }
    if (s_state.foam_tex != NULL) { free(s_state.foam_tex); s_state.foam_tex = NULL; }
    memset(&s_state, 0, sizeof(s_state));
    memset(&s_night, 0, sizeof(s_night));
    wo_isl_free_data();
    if (s_layer.color != NULL) { free(s_layer.color); s_layer.color = NULL; s_layer.cap_color = 0; }
    if (s_layer.hit != NULL) { free(s_layer.hit); s_layer.hit = NULL; s_layer.cap_hit = 0; }
    s_layer.lw = s_layer.lh = 0;
    s_layer.div = 0;
    s_terr_cover = 0.0;
}

// ===============================================================================
// 每帧预处理：时间推进 / 太阳方向 / LUT 构建 / 波场烘焙
// ===============================================================================

void linglong_ocean_prepare(float sun_alt, float sun_azi, float view_f, float sky_radius, int32_t projection) {
    if (!s_cfg_init) {
        Linglong_Ocean_Config c;
        linglong_ocean_config_default(&c);
        linglong_ocean_configure(&c);
    }

    double now = (double)get_timestamp_in_ms();
    if (s_state.last_now > 0.0) {
        double dt = (now - s_state.last_now) / 1000.0;
        if (dt < 0.0) dt = 0.0;
        if (dt > 0.1) dt = 0.1;
        s_state.app_time += dt;
    }
    s_state.last_now = now;

    // 每像素角尺寸（鱼眼：r = f·2R/π·θ；透视：θ ≈ r/(f·R)）
    double f = (view_f > 1e-3f) ? (double)view_f : 1e-3;
    double R = (sky_radius > 1e-3f) ? (double)sky_radius : 1e-3;
    s_state.pix_ang = (projection == 0) ? (WO_PI / (2.0 * f * R)) : (1.0 / (f * R));

    // 太阳方向（海洋系 y 上）
    double e = (double)sun_alt * WO_PI / 180.0;
    double a = (double)sun_azi * WO_PI / 180.0;
    s_state.sun_dir[0] = cos(e) * sin(a);
    s_state.sun_dir[1] = sin(e);
    s_state.sun_dir[2] = cos(e) * cos(a);

    // 夜天光：仰角表（只依赖太阳方向）+ 水体环境光提升用的可见度，每帧算一次
    linglong_night_sky_init(&s_night, s_state.sun_dir[0], s_state.sun_dir[1], s_state.sun_dir[2]);
    s_state.night_vis = linglong_night_visibility(s_state.sun_dir[1]);

    double d_alt = s_state.have_sun ? fabs((double)sun_alt - s_state.sun_alt) : 1e9;

    if (s_state.trans_lut == NULL) {
        wo_build_trans_lut();
        wo_build_ms_lut();
    }
    // 阈值 0.25°（与原实现一致）：太阳高度角变化超过该量才重建天空视图 LUT。
    // 注：天空视图 LUT 只依赖太阳高度角（构建时太阳被旋转到 x-y 平面，方位角信息
    // 全部落在 wo_sky_view_uv 的横向坐标里），**与太阳方位角无关**（实测：仅改方位角
    // 160°，LUT 逐元素差异 0/110592）。原实现还拿 dAzi>0.25 做触发条件，等于把重建
    // 频率提高到 ~2.5 倍（实测太阳速率中位 0.0024°/s 高度 + 0.0035°/s 方位，重建间隔
    // 42s → 104s），每次都是 100ms 级卡顿，故此处只看高度角。
    //
    // 观察高度（漫游）：LUT 依赖 r = RG + obs_y/1000，而漫游时高度是连续量，不能靠
    // “参数变了”触发。取相对阈值 Δy > max(25 m, 0.5%·y)：低空缓慢升降时几乎不重建，
    // 高空巡航时阈值自动放大；水平移动对天空零影响（无限平面大气无视差）。
    double d_cam = s_state.have_sun ? fabs(s_obs[1] - s_state.sky_cam_y) : 1e9;
    const double cam_eps = wo_max(25.0, 0.005 * s_obs[1]);
    if (s_state.sky_lut == NULL || d_alt > 0.25 || d_cam > cam_eps) {
        if (wo_build_sky_lut()) {
            s_state.sun_alt = (double)sun_alt;
            s_state.sky_cam_y = s_obs[1];
            s_state.have_sun = 1;
            wo_sky_view_uv_params();   // 天空视图 UV 的帧常量（只依赖观察高度）
        }
    }
    wo_ensure_waves();
    wo_isl_step();          // 地形：渐进生成（每帧最多 WO_ISL_GEN_BUDGET_MS）

    // 每帧常量：天顶辐亮度 / 半球平均辐亮度 / 太阳辐照。
    // 这些量只随太阳与观察高度变化，原本在 wo_ocean_pixel 里每像素重算一遍
    // （5 次天空视图 LUT 查询 + 1 次透射率查询，实测占整幅耗时 15%）。
    // 计算顺序与逐像素版本逐一对应，保证结果逐位相同。
    if (s_state.sky_lut != NULL && s_state.trans_lut != NULL) {
        double t[3];
        wo_sky_atmo(0.0, 1.0, 0.0, t);
        s_state.sky_up[0] = t[0];
        s_state.sky_up[1] = t[1];
        s_state.sky_up[2] = t[2];
        s_state.hemi[0] = t[0] * 0.4;
        s_state.hemi[1] = t[1] * 0.4;
        s_state.hemi[2] = t[2] * 0.4;
        static const double OB[4][3] = {
            { 0.94, 0.34, 0.0 }, { -0.94, 0.34, 0.0 }, { 0.0, 0.34, 0.94 }, { 0.0, 0.34, -0.94 }
        };
        for (int32_t i = 0; i < 4; i++) {
            wo_sky_atmo(OB[i][0], OB[i][1], OB[i][2], t);
            s_state.hemi[0] += t[0] * 0.15;
            s_state.hemi[1] += t[1] * 0.15;
            s_state.hemi[2] += t[2] * 0.15;
        }
        wo_sample_trans_lut(WO_RG + s_obs[1] / 1000.0, s_state.sun_dir[1], t);
        s_state.sun_e[0] = t[0] * WO_SUN_SCALE;
        s_state.sun_e[1] = t[1] * WO_SUN_SCALE;
        s_state.sun_e[2] = t[2] * WO_SUN_SCALE;
    }
}

// ===============================================================================
// 海面求交与逐像素着色
// ===============================================================================

// Smith 掩蔽项（web-ocean lambdaSmith；w 为方向，cov 为坡度协方差）
static double wo_lambda_smith(double wx, double wy, double wz,
                              double cxx, double czz, double cxz) {
    double xz2 = wo_max(wx * wx + wz * wz, 1e-8);
    double il = 1.0 / sqrt(xz2);
    double azX = wx * il, azZ = wz * il;
    double sig2 = azX * azX * cxx + azZ * azZ * czz + 2.0 * azX * azZ * cxz;
    double cotT = wy * il;
    double aa = wo_max(cotT, 1e-3) / sqrt(2.0 * wo_max(sig2, 1e-7));
    if (aa >= 1.6) return 0.0;
    return (1.0 - 1.259 * aa + 0.396 * aa * aa) / (3.535 * aa + 2.181 * aa * aa);
}

// Schlick Fresnel（n=1.333）
static inline double wo_fresnel_water(double c) {
    double x = 1.0 - wo_clamp(c, 0.0, 1.0);
    double x2 = x * x;
    return 0.02 + 0.98 * x2 * x2 * x;
}

// ---- 海面求交（射线步进；返回最近海面距离 t，未命中返回 -1）----
static double wo_ocean_march(double dx, double dy, double dz, double max_dist) {
    if (dy >= -1e-4) return -1.0;
    const double t = s_state.app_time;
    const double cam_y = s_obs[1];
    const double cam_x = s_obs[0];
    const double cam_z = s_obs[2];

    double t_plane = -cam_y / dy;
    if (t_plane > max_dist) return -1.0;
    double t_hit = t_plane;
    if (t_plane <= WO_MARCH_FAR) {
        double t_end = t_plane * 0.999;
        double t0 = wo_max(0.0, (WO_HMAX - cam_y) / dy);
        t0 = wo_min(t0, t_end);
        // 步数：阶段2 实测减半（24/16/10 → 12/8/5）后，二分仍能把交点细化到容差内，
        // 逐像素比对只有 0.02% 像素差 >2/255（波峰轮廓边缘微移），整幅耗时 -11%。
        int32_t steps = (t_plane < 30.0) ? 12 : ((t_plane < 120.0) ? 8 : 5);
        double t_prev = t0;
        int32_t hit = 0;
        for (int32_t i = 1; i <= steps; i++) {
            double ti = wo_mix(t0, t_end, (double)i / steps);
            double fi = cam_y + dy * ti - wo_height_at_fast(cam_x + dx * ti, cam_z + dz * ti, t);
            if (fi <= 0.0) {
                // 二分细化：容差随距离放大（1mm 的几何精度在 100m 外远低于一个像素的
                // 足迹，固定 1mm 只是白算）。实测 0.64% 像素差 >2/255，整幅 -5%。
                double ta = t_prev, tb = ti;
                double tol = wo_max(0.001, tb * 1e-4);
                for (int32_t r = 0; r < 24 && tb - ta > tol; r++) {
                    double tm = 0.5 * (ta + tb);
                    double fm = cam_y + dy * tm - wo_height_at_fast(cam_x + dx * tm, cam_z + dz * tm, t);
                    if (fm > 0.0) ta = tm;
                    else tb = tm;
                }
                t_hit = tb;
                hit = 1;
                break;
            }
            t_prev = ti;
        }
        if (!hit) t_hit = t_plane;
    }
    if (t_hit > max_dist) return -1.0;
    return t_hit;
}

// ---- 海洋逐像素着色（web-ocean ocean.js fs 移植）；t_hit 由 wo_ocean_march 给出 ----
static void wo_ocean_pixel(double dx, double dy, double dz, double t_hit, double *out) {
    const double t = s_state.app_time;

    const double dist = t_hit;
    // 世界坐标（相机可移动：波场/泡沫/海底高程均锚定在世界空间，故浪会从身边流过）
    const double px = s_obs[0] + dx * t_hit, pz = s_obs[2] + dz * t_hit;
    const double Vx = -dx, Vy = -dy, Vz = -dz;

    // ---- 坡度与协方差（LEAN）----
    // 远处波高低于像素足迹：坡度按像素足迹衰减（等效 web-ocean 的 mip 滤波）
    const double wf = 1.0 / (1.0 + (dist / 400.0) * (dist / 400.0));
    double f1[4], f2[4];
    wo_tile_sample(s_state.wave_data[0], WO_TILES[0].N, WO_TILES[0].L,
                   px - s_scroll1[0] * t, pz - s_scroll1[1] * t, f1);
    wo_tile_sample(s_state.wave_data[1], WO_TILES[1].N, WO_TILES[1].L,
                   px - s_scroll2[0] * t, pz - s_scroll2[1] * t, f2);
    double sx = (f1[1] + f2[1]) * wf;
    double sz = (f1[2] + f2[2]) * wf;
    double jt = (f1[3] + f2[3]) * wf * wf;
    double hit_h = f1[0] + f2[0];

    double cxx = s_state.cov[0], czz = s_state.cov[1], cxz = s_state.cov[2];
    // 未解析毛细波粗糙度；随距离增长以在远处耀光带达到 Cox-Munk 量级
    const double unres = WO_UNRES + 0.0035 * wo_smoothstep(30.0, 500.0, dist);
    cxx += unres;
    czz += unres;
    // 镜面抗锯齿：像素足迹内坡度变化展宽瓣宽（以足迹差分代替屏幕空间导数）
    {
        const double pf = wo_max(dist * s_state.pix_ang / wo_max(-Vy, 0.05), 1e-4);
        double a1[4], a2[4], b1[4], b2[4];
        wo_tile_sample(s_state.wave_data[0], WO_TILES[0].N, WO_TILES[0].L,
                       px + pf - s_scroll1[0] * t, pz - s_scroll1[1] * t, a1);
        wo_tile_sample(s_state.wave_data[1], WO_TILES[1].N, WO_TILES[1].L,
                       px + pf - s_scroll2[0] * t, pz - s_scroll2[1] * t, a2);
        const double dAxxx = (a1[1] + a2[1]) * wf - sx;
        const double dAxsx = (a1[2] + a2[2]) * wf - sz;
        wo_tile_sample(s_state.wave_data[0], WO_TILES[0].N, WO_TILES[0].L,
                       px - s_scroll1[0] * t, pz + pf - s_scroll1[1] * t, b1);
        wo_tile_sample(s_state.wave_data[1], WO_TILES[1].N, WO_TILES[1].L,
                       px - s_scroll2[0] * t, pz + pf - s_scroll2[1] * t, b2);
        const double dBxxx = (b1[1] + b2[1]) * wf - sx;
        const double dBxsx = (b1[2] + b2[2]) * wf - sz;
        const double aax = dAxxx * dAxxx + dBxxx * dBxxx;
        const double aay = dAxsx * dAxsx + dBxsx * dBxsx;
        const double aaz = dAxxx * dAxsx + dBxxx * dBxsx;
        const double aa_max = wo_max(wo_max(aax, aay), 1e-8);
        const double kAA = wo_min(1.0, 0.03 / aa_max);
        cxx += aax * 0.5 * kAA;
        czz += aay * 0.5 * kAA;
        cxz += aaz * 0.5 * kAA;
    }

    const double slx = sx, slz = sz;
    const double inl = 1.0 / sqrt(slx * slx + 1.0 + slz * slz);
    const double nrmX = -slx * inl, nrmY = inl, nrmZ = -slz * inl;
    // 夜间环境光提升系数（见 WO_NIGHT_AMB_GAIN 注释）：太阳落下后才生效，白天恒为 1
    const double night = s_state.night_vis;
    const double kAmb = 1.0 + WO_NIGHT_AMB_GAIN * night;

    // ---- 太阳辐照与天空环境（均由 LUT 得出；每帧常量已由 prepare 算好）----
    const double sunE0 = s_state.sun_e[0];
    const double sunE1 = s_state.sun_e[1];
    const double sunE2 = s_state.sun_e[2];
    const double skyUp0 = s_state.sky_up[0], skyUp1 = s_state.sky_up[1], skyUp2 = s_state.sky_up[2];
    const double hemi0 = s_state.hemi[0], hemi1 = s_state.hemi[1], hemi2 = s_state.hemi[2];

    double sky_tmp[3];
    // skyN（沿波面法向的天光）只服务泡沫项（白帽气泡/受光）——默认配置泡沫为 0、
    // 无岛屿浅滩，这一路的求值是死算。改为在泡沫段开头按需求值（见下文），
    // 逐像素省一次天空 LUT 查询（夜间还省一次夜天光取值），白天输出逐位不变。
    const double Ldx = s_state.sun_dir[0], Ldy = s_state.sun_dir[1], Ldz = s_state.sun_dir[2];

    // ---- 太阳耀光：各向异性高斯坡度分布 ----
    double Hvx = Vx + Ldx, Hvy = Vy + Ldy, Hvz = Vz + Ldz;
    double Hvl = 1.0 / sqrt(Hvx * Hvx + Hvy * Hvy + Hvz * Hvz);
    Hvx *= Hvl; Hvy *= Hvl; Hvz *= Hvl;
    const double hsX = -Hvx / wo_max(Hvy, 1e-3);
    const double hsZ = -Hvz / wo_max(Hvy, 1e-3);
    const double dslX = hsX - slx, dslZ = hsZ - slz;
    const double det = wo_max(cxx * czz - cxz * cxz, 1e-10);
    const double eG = (dslX * dslX * czz + dslZ * dslZ * cxx - 2.0 * dslX * dslZ * cxz) / det;
    const double pdf = wo_fast_exp(-0.5 * eG) / (2.0 * WO_PI * sqrt(det));
    const double hy2 = Hvy * Hvy;
    const double Dm = pdf / wo_max(hy2 * hy2, 1e-6);
    // Smith 掩蔽项 + Schlick Fresnel（n=1.333）
    const double Gsm = 1.0 / (1.0 + wo_lambda_smith(Vx, Vy, Vz, cxx, czz, cxz)
                                 + wo_lambda_smith(Ldx, Ldy, Ldz, cxx, czz, cxz));
    const double Fh = wo_fresnel_water(Vx * Hvx + Vy * Hvy + Vz * Hvz);
    double spec0 = 0.0, spec1 = 0.0, spec2 = 0.0;
    if (Ldy >= 0.0) {
        const double sK = Fh * Dm * Gsm / (4.0 * wo_max(Vy, 0.02));
        spec0 = sunE0 * sK;
        spec1 = sunE1 * sK;
        spec2 = sunE2 * sK;
    }

    // ---- 天空反射 ----
    const double sig = sqrt(wo_max(cxx + czz, 1e-6) * 0.5);
    const double vhl = 1.0 / sqrt(Vx * Vx + Vz * Vz + 1e-10);
    const double vhX = Vx * vhl, vhZ = Vz * vhl;
    const double sigV = sqrt(wo_max(vhX * vhX * cxx + vhZ * vhZ * czz + 2.0 * vhX * vhZ * cxz, 1e-6));
    const double aV = wo_max(Vy, 1e-3) / wo_max(sqrt(wo_max(1.0 - Vy * Vy, 0.0)), 1e-3) / (1.4142 * sigV);
    const double tilt = 0.32 * sigV * (1.0 - wo_smoothstep(0.0, 2.5, aV));
    double nVx = nrmX + vhX * tilt, nVy = nrmY, nVz = nrmZ + vhZ * tilt;
    const double nVl = 1.0 / sqrt(nVx * nVx + nVy * nVy + nVz * nVz);
    nVx *= nVl; nVy *= nVl; nVz *= nVl;
    // 反射方向（镜像反射：R = 2(nV·V)nV − V）
    const double dVN = 2.0 * (nVx * Vx + nVy * Vy + nVz * Vz);
    double Rx = nVx * dVN - Vx, Ry = nVy * dVN - Vy, Rz = nVz * dVN - Vz;
    Ry = fabs(Ry) * 0.4 + wo_max(Ry, 0.0) * 0.6;
    const double Rl = 1.0 / sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
    Rx *= Rl; Ry *= Rl; Rz *= Rl;
    const double envLod = wo_clamp(log2(wo_max(2.0 * sig * (256.0 / 1.5708), 1.0)), 0.0, 8.0);
    wo_sky_atmo(Rx, Ry, Rz, sky_tmp);
    const double rough = envLod / 8.0 * 0.65;
    const double refl0 = wo_mix(sky_tmp[0], hemi0, rough);
    const double refl1 = wo_mix(sky_tmp[1], hemi1, rough);
    const double refl2 = wo_mix(sky_tmp[2], hemi2, rough);
    // Fresnel：部分平滑法向（长波面元 + 掠射细波统计）
    const double nFx = wo_mix(nrmX, 0.0, 0.45), nFy = wo_mix(nrmY, 1.0, 0.45), nFz = wo_mix(nrmZ, 0.0, 0.45);
    const double nFl = 1.0 / sqrt(nFx * nFx + nFy * nFy + nFz * nFz);
    const double Fr = wo_fresnel_water(wo_max((nFx * Vx + nFy * Vy + nFz * Vz) * nFl, 0.0));

    // ---- 水体：上涌散射 + 背光波峰透射辉光 ----
    double body0 = s_water_col[0] * (hemi0 * 1.6 * kAmb + sunE0 * wo_max(Ldy, 0.0) * 0.8);
    double body1 = s_water_col[1] * (hemi1 * 1.6 * kAmb + sunE1 * wo_max(Ldy, 0.0) * 0.8);
    double body2 = s_water_col[2] * (hemi2 * 1.6 * kAmb + sunE2 * wo_max(Ldy, 0.0) * 0.8);
    {
        const double crest = wo_clamp((hit_h - 0.05) / 0.9, 0.0, 1.0);
        const double toSun = wo_max(Ldx * Vx + Ldy * Vy + Ldz * Vz, 0.0);
        const double thin = wo_clamp(0.5 - 0.5 * (Ldx * nrmX + Ldy * nrmY + Ldz * nrmZ), 0.0, 1.0);
        const double thin3 = thin * thin * thin;   // pow(thin,3) → 乘法（阶数固定）
        const double k1 = WO_SSS_K * crest * crest * toSun * toSun * toSun * toSun * thin3
                        * wo_fast_exp(-dist / 600.0);
        body0 += WO_SSS_COL[0] * sunE0 * k1;
        body1 += WO_SSS_COL[1] * sunE1 * k1;
        body2 += WO_SSS_COL[2] * sunE2 * k1;
    }

    // ---- 浅水：海底高程（岛屿高程场的负值部分）决定水色与拍岸泡沫 ----
    // 近岸 10 m 以内水体明显偏青绿，越过陆架（约 40 m）回到深水色。
    // 距离很远时海面交点已超出岛屿高程场的有效范围（周期贴图副本会随机落在陆地上），
    // 且雾早已把那里压成天色，索性淡出浅水项，避免出现颗粒状伪色。
    const double near_isl = 1.0 - wo_smoothstep(0.45 * WO_ISL_WORLD, 0.75 * WO_ISL_WORLD, dist);
    double bathy = -WO_ISL_DEEP;
    if (s_isl.ready && s_isl.enable && near_isl > 0.0) {
        // 早退：先用 400 m 粗网格的“格内最大高程”判一下。整格比 40 m 还深时，
        // 浅水项 exp(-depth·0.115) < 0.01、也不可能触发拍岸白带，直接按深水算，
        // 省掉逐像素的双线性高程取样（深水占海面绝大多数像素）。
        if (wo_isl_cell_max(px, pz, 0) > -40.0) bathy = wo_isl_height(px, pz);
    }
    const double depth = (bathy < 0.0) ? -bathy : 0.0;
    const double shal = wo_fast_exp(-depth * 0.115) * near_isl;
    if (shal > 0.004) {
        const double sdY = wo_max(Ldy, 0.0);
        const double k0 = hemi0 * 1.6 * kAmb + sunE0 * sdY * 0.8;
        const double k1 = hemi1 * 1.6 * kAmb + sunE1 * sdY * 0.8;
        const double k2 = hemi2 * 1.6 * kAmb + sunE2 * sdY * 0.8;
        body0 = wo_mix(body0, WO_SHALLOW_COL[0] * k0, shal * 0.9);
        body1 = wo_mix(body1, WO_SHALLOW_COL[1] * k1, shal * 0.9);
        body2 = wo_mix(body2, WO_SHALLOW_COL[2] * k2, shal * 0.9);
    }

    // ---- 夜间水体去饱和 ----
    // 深海吸收比约 1:5:6（红光被水吸收），白天被太阳耀光与天空反射淹掉；夜间只剩环境光，
    // 整片海面会被染成高饱和青——而人眼在暗视觉下靠视杆细胞（无辨色能力），真实夜海
    // 观感接近中性灰蓝。故按 night 权重把水体色向自身的亮度靠拢（保持亮度不变）。
    if (night > 0.0) {
        const double bl = 0.2126 * body0 + 0.7152 * body1 + 0.0722 * body2;
        const double kd = 0.6 * night;
        body0 = wo_mix(body0, bl, kd);
        body1 = wo_mix(body1, bl, kd);
        body2 = wo_mix(body2, bl, kd);
    }

    // ---- 泡沫 ----
    // 泡沫强度为 0（默认）时 cover 恒为 0，下面整段（风向条纹 + 两个尺度的图案 +
    // 气泡亮度，共 4 次泡沫纹理双线性采样及其全部光照）都是死算，且结果会被
    // foamMask=0 全部丢掉；直接早退，输出逐位不变（实测 max|Δ| = 0）。
    const double J = 1.0 + jt;   // 雅可比行列式（仅存迹通道，J ≈ 1 + trace）
    const double fresh = wo_clamp((WO_FOAM_J_THRESH - J) * WO_FOAM_J_GAIN, 0.0, 1.0);
    double cover = wo_clamp(fresh * (double)s_cfg.foam, 0.0, 1.5);
    double foam[4] = { 0.0, 0.0, 0.0, 0.0 };
    double foamMask = 0.0, covC = 0.0, foamLit0 = 0.0, foamLit1 = 0.0, foamLit2 = 0.0;
    double foam_bub = 0.5;

    // 沿波面法向的天光：只在上两条早退分支（白帽 / 拍岸白带）里用到
    double skyN0 = 0.0, skyN1 = 0.0, skyN2 = 0.0;
    double amb0 = 0.0, amb1 = 0.0, amb2 = 0.0;
    if ((cover > 1e-6) || (depth < 3.4 && near_isl > 0.0)) {
        wo_sky_atmo(nrmX, nrmY, nrmZ, sky_tmp);
        skyN0 = sky_tmp[0]; skyN1 = sky_tmp[1]; skyN2 = sky_tmp[2];
        amb0 = skyUp0 * 1.3 * kAmb + skyN0 * 0.5 * kAmb;
        amb1 = skyUp1 * 1.3 * kAmb + skyN1 * 0.5 * kAmb;
        amb2 = skyUp2 * 1.3 * kAmb + skyN2 * 0.5 * kAmb;
    }
    if (cover > 1e-6) {
        // 风向条纹：老泡沫破碎成风纹，新白帽保持致密
        const double wsU = px * s_wind_vx + pz * s_wind_vz;
        const double wsV = px * (-s_wind_vz) + pz * s_wind_vx;
        wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0, wsU * 0.035, wsV * 0.11, foam);
        const double streak = foam[3] / 255.0;
        cover *= wo_mix(wo_smoothstep(0.25, 0.75, streak) * 1.5, 1.0, fresh);
        // 泡沫图案（两个尺度混合）
        wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0, px * 0.23 + 0.013 * t, pz * 0.23, foam);
        const double patA = foam[0] / 255.0;
        wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0, px * 0.71 + 0.37, pz * 0.71 + 0.71, foam);
        const double pat = patA * 0.78 + foam[1] / 255.0 * 0.22;
        const double fw = 0.05;   // 图案屏幕变化率近似（fwidth；两路一致取常数）
        const double soft = 0.04 + 0.14 * (1.0 - wo_clamp(cover, 0.0, 1.0));
        const double fm = wo_smoothstep(1.0 - cover - fw - soft, 1.0 - cover + fw + soft, pat);
        foamMask = wo_clamp(fm * wo_smoothstep(0.0, 0.5, cover), 0.0, 1.0)
                  * (0.68 + 0.32 * wo_smoothstep(0.3, 0.9, cover));
        // 泡沫下方气泡增亮并给水体染色
        covC = wo_clamp(cover, 0.0, 1.0);
        body0 += WO_SSS_COL[0] * (sunE0 * 0.05 + amb0 * 0.35) * covC;
        body1 += WO_SSS_COL[1] * (sunE1 * 0.05 + amb1 * 0.35) * covC;
        body2 += WO_SSS_COL[2] * (sunE2 * 0.05 + amb2 * 0.35) * covC;
        // 泡沫内部气泡亮度变化
        wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0, px * 1.9 + 0.21, pz * 1.9 + 0.53, foam);
        foam_bub = foam[1] / 255.0;
    }   // end if (cover > 1e-6)

    // 泡沫受光：白帽（foamMask>0）与拍岸白带共用。泡沫强度为 0（无白帽）但处于
    // 浅水拍岸带时也必须算：否则混到 foamLit=0 会在岸边得到一条黑带。
    if ((cover > 1e-6) || (depth < 3.4 && near_isl > 0.0)) {
        if (cover <= 1e-6) {
            wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0, px * 1.9 + 0.21, pz * 1.9 + 0.53, foam);
            foam_bub = foam[1] / 255.0;
        }
        const double foamAlb0 = 0.82 * (0.78 + 0.3 * foam_bub);
        const double foamAlb1 = 0.87 * (0.78 + 0.3 * foam_bub);
        const double foamAlb2 = 0.92 * (0.78 + 0.3 * foam_bub);
        const double fAmb0 = hemi0 * 1.0 + skyN0 * 0.3;
        const double fAmb1 = hemi1 * 1.0 + skyN1 * 0.3;
        const double fAmb2 = hemi2 * 1.0 + skyN2 * 0.3;
        const double fLum = 0.2126 * fAmb0 + 0.7152 * fAmb1 + 0.0722 * fAmb2;
        const double fAmbN0 = wo_mix(fAmb0, fLum, 0.85);
        const double fAmbN1 = wo_mix(fAmb1, fLum, 0.85);
        const double fAmbN2 = wo_mix(fAmb2, fLum, 0.85);
        const double NdL = wo_max(nrmX * Ldx + nrmY * Ldy + nrmZ * Ldz, 0.0);
        foamLit0 = foamAlb0 * (sunE0 * NdL * 0.318 * 0.5 + fAmbN0 * 1.55);
        foamLit1 = foamAlb1 * (sunE1 * NdL * 0.318 * 0.5 + fAmbN1 * 1.55);
        foamLit2 = foamAlb2 * (sunE2 * NdL * 0.318 * 0.5 + fAmbN2 * 1.55);
    }

    double col0 = body0 * (1.0 - Fr) + refl0 * Fr + spec0;
    double col1 = body1 * (1.0 - Fr) + refl1 * Fr + spec1;
    double col2 = body2 * (1.0 - Fr) + refl2 * Fr + spec2;
    col0 = wo_mix(col0, foamLit0, foamMask * 0.95);
    col1 = wo_mix(col1, foamLit1, foamMask * 0.95);
    col2 = wo_mix(col2, foamLit2, foamMask * 0.95);

    // ---- 拍岸泡沫：水深 3 m 以内的窄带里一条断续、缓慢推移的白带 ----
    if (depth < 3.4 && near_isl > 0.0) {
        const double band = wo_smoothstep(3.2, 0.20, depth);
        wo_tex_sample_u8(s_state.foam_tex, WO_FOAM_N, 1.0,
                         px * 0.055, pz * 0.055 + 0.02 * t, foam);
        const double sn = foam[2] / 255.0 * 0.65 + foam[1] / 255.0 * 0.35;
        const double surf = band * wo_smoothstep(0.34, 0.60, sn + 0.06 * sin(t * 0.8)) * 0.9;
        col0 = wo_mix(col0, foamLit0, surf);
        col1 = wo_mix(col1, foamLit1, surf);
        col2 = wo_mix(col2, foamLit2, surf);
    }

    // ---- 大气透视 ----
    const double Td = wo_fast_exp(-(double)s_cfg.fog_floor * dist);
    const double hzl = 1.0 / sqrt(Vx * Vx + 0.002 * 0.002 + Vz * Vz);
    wo_sky_atmo(-Vx * hzl, 0.002, -Vz * hzl, sky_tmp);
    col0 = col0 * Td + sky_tmp[0] * (1.0 - Td);
    col1 = col1 * Td + sky_tmp[1] * (1.0 - Td);
    col2 = col2 * Td + sky_tmp[2] * (1.0 - Td);

    wo_agx_tonemap(col0, col1, col2, out);
}

// ===============================================================================
// 实时海洋地景绘制（地平线以下像素）
// ===============================================================================

void linglong_ocean_draw_landscape(
    Nano_GFX *gfx,
    float sky_radius, float center_x, float center_y,
    float view_alt, float view_azi, float view_roll, float f, int32_t projection
) {
    if (gfx == NULL) return;
    if (!s_state.waves_ready || s_state.sky_lut == NULL || s_state.trans_lut == NULL) return;

    const int32_t fb_width = (int32_t)gfx->width;
    const int32_t fb_height = (int32_t)gfx->height;

    // 方向图复用：视角不变时直接查表（天象仪在 render_sky 里已经按当前视角刷过这张图），
    // 省掉逐像素的 fisheye_unproject（实测 56ns/px）。表里的值就是同一函数算出来的，
    // 因此结果逐位不变。尺寸不符或未就绪时自动退回逐像素反投影。
    const int32_t use_dir_map = fisheye_dir_map_ready(fb_width, fb_height);

    // 性能专项 A7：RGB888 帧缓冲下直接按行指针写，省掉逐像素 gfx_set_pixel 的
    // dirty_expand（4 次比较 + 4 次写）与色彩模式分支；整趟结束后统一 mark_dirty_full。
    // 其它色彩模式（RGB565 等）仍走 gfx_set_pixel，行为不变。
    const int fast888 = (gfx->color_mode == GFX_COLOR_MODE_RGB888 && gfx->frame_buffer_rgb888 != NULL);
    const size_t stride888 = (size_t)fb_width * 3;

    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < fb_height; y++) {
        uint8_t *dst = fast888 ? (gfx->frame_buffer_rgb888 + (size_t)y * stride888) : NULL;
        for (int32_t x = 0; x < fb_width; x++) {
            float ray_x = 0.0f, ray_y = 0.0f, ray_z = 0.0f;
            if (use_dir_map) {
                fisheye_dir_map_get(x, y, &ray_x, &ray_y, &ray_z);
            }
            else {
                fisheye_unproject((float)x, (float)y, sky_radius, center_x, center_y,
                                  view_alt, view_azi, view_roll, f, projection,
                                  &ray_x, &ray_y, &ray_z);
            }
            // 换算到海洋系（x东, y上, z北）
            double dx = (double)ray_x, dy = (double)ray_z, dz = (double)ray_y;
            double dl = sqrt(dx * dx + dy * dy + dz * dz);
            if (dl < 1e-9) continue;
            dx /= dl; dy /= dl; dz /= dl;
            if (ray_z >= 0.0f) continue;    // 地平线以上：天空，不覆盖
            double t_ocean = wo_ocean_march(dx, dy, dz, 1e9);
            if (t_ocean <= 0.0) continue;
            double out[3];
            wo_ocean_pixel(dx, dy, dz, t_ocean, out);
            uint8_t r = (uint8_t)wo_min(255.0, out[0] * 255.0);
            uint8_t g = (uint8_t)wo_min(255.0, out[1] * 255.0);
            uint8_t b = (uint8_t)wo_min(255.0, out[2] * 255.0);
            if (fast888) {
                dst[(size_t)x * 3 + 0] = r;
                dst[(size_t)x * 3 + 1] = g;
                dst[(size_t)x * 3 + 2] = b;
            }
            else {
                gfx_set_pixel(gfx, (uint32_t)x, (uint32_t)y, r, g, b);
            }
        }
    }
    if (fast888) gfx_mark_dirty_full(gfx);

    // 地形：在海洋之后绘制（海洋铺底，陆地覆盖其上的像素）。地形可出现在地平线
    // 以上（山脊遮挡日月星），也可出现在海面之前（岛优先于海——参考实现同顺序）。
    wo_terr_draw(gfx, sky_radius, center_x, center_y,
                 view_alt, view_azi, view_roll, f, projection);
}

#else  // !LINGLONG_ENABLE_DYNAMIC_LANDSCAPE

// 内存紧缺平台（M5Stack 等）：动态地景关闭，海洋模块退化为空实现
void linglong_ocean_config_default(Linglong_Ocean_Config *cfg) { (void)cfg; }
void linglong_ocean_configure(const Linglong_Ocean_Config *cfg) { (void)cfg; }
void linglong_ocean_set_levels(int32_t swell_hs_level, int32_t swell_len_level) {
    (void)swell_hs_level; (void)swell_len_level;
}
void linglong_ocean_free(void) {}
void linglong_ocean_set_observer(double x, double y, double z) { (void)x; (void)y; (void)z; }
void linglong_ocean_get_observer(double *x, double *y, double *z) {
    if (x) *x = 0.0;
    if (y) *y = 5.0;
    if (z) *z = 0.0;
}
double linglong_ocean_terrain_height(double x, double z) { (void)x; (void)z; return -70.0; }
void linglong_ocean_terrain_default(Linglong_Terrain_Config *cfg) {
    if (cfg == NULL) return;
    cfg->enable = 0; cfg->quality = 1; cfg->res = 512; cfg->seed = 0;
}
void linglong_ocean_terrain_configure(const Linglong_Terrain_Config *cfg) { (void)cfg; }
int32_t linglong_ocean_terrain_progress(void) { return 100; }
int32_t linglong_ocean_terrain_ready(void) { return 0; }
void linglong_ocean_terrain_regenerate(uint32_t seed) { (void)seed; }

#endif  // LINGLONG_ENABLE_DYNAMIC_LANDSCAPE
