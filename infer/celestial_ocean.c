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
    int32_t have_sun;

    double  last_now;               // 上次 prepare 的时刻 ms
    double  app_time;               // 海面动画时间 s
    double  pix_ang;                // 每像素角尺寸 rad

    // 每帧常量（阶段1：原本在逐像素路径里重复计算的天空环境量，现由 prepare 算一次）
    double  sky_up[3];              // 天顶辐亮度 wo_sky_atmo(0,1,0)
    double  hemi[3];                // 半球平均辐亮度（天顶 + 4 个斜方向）
    double  sun_e[3];               // 太阳辐照（透射率 LUT × WO_SUN_SCALE）

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

static void wo_sky_view_uv(double dx, double dy, double dz, double *u, double *v) {
    const double cos_beta = s_uv_cos_beta;
    const double beta = s_uv_beta;
    const double zha = s_uv_zha;
    double vzc = wo_clamp(dy, -1.0, 1.0);
    double uv_y;
    if (vzc >= -cos_beta) {
        double coord = acos(vzc) / zha;
        coord = 1.0 - coord;
        coord = sqrt(wo_max(coord, 0.0));
        coord = 1.0 - coord;
        uv_y = coord * 0.5;
    }
    else {
        double coord = (acos(vzc) - zha) / beta;
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

static void wo_tile_sample(const float *data, int32_t N, double L, double x, double z, double *out) {
    double u = (x / L) * N - 0.5, v = (z / L) * N - 0.5;
    int32_t i0 = (int32_t)floor(u), j0 = (int32_t)floor(v);
    double fu = u - i0, fv = v - j0;
    int32_t i0c = ((i0 % N) + N) % N, i1c = (((i0 + 1) % N) + N) % N;
    int32_t j0c = ((j0 % N) + N) % N, j1c = (((j0 + 1) % N) + N) % N;
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
    double u = (x / L) * N - 0.5, v = (z / L) * N - 0.5;
    int32_t i0 = (int32_t)floor(u), j0 = (int32_t)floor(v);
    double fu = u - i0, fv = v - j0;
    int32_t i0c = ((i0 % N) + N) % N, i1c = (((i0 + 1) % N) + N) % N;
    int32_t j0c = ((j0 % N) + N) % N, j1c = (((j0 + 1) % N) + N) % N;
    int32_t i00 = (j0c * N + i0c) * 4, i10 = (j0c * N + i1c) * 4;
    int32_t i01 = (j1c * N + i0c) * 4, i11 = (j1c * N + i1c) * 4;
    double w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv);
    double w01 = (1 - fu) * fv, w11 = fu * fv;
    return (double)data[i00] * w00 + (double)data[i10] * w10
         + (double)data[i01] * w01 + (double)data[i11] * w11;
}

// 泡沫纹理采样（RGBA8 可平铺纹理；返回 0..255 通道值）
static void wo_tex_sample_u8(const uint8_t *data, int32_t N, double L, double x, double z, double *out) {
    double u = (x / L) * N - 0.5, v = (z / L) * N - 0.5;
    int32_t i0 = (int32_t)floor(u), j0 = (int32_t)floor(v);
    double fu = u - i0, fv = v - j0;
    int32_t i0c = ((i0 % N) + N) % N, i1c = (((i0 + 1) % N) + N) % N;
    int32_t j0c = ((j0 % N) + N) % N, j1c = (((j0 + 1) % N) + N) % N;
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

void linglong_ocean_set_levels(int32_t height_level, int32_t swell_hs_level, int32_t swell_len_level) {
    if (!s_cfg_init) {
        Linglong_Ocean_Config c;
        linglong_ocean_config_default(&c);
        linglong_ocean_configure(&c);
    }
    if (height_level    < 0 || height_level    >= S_OCEAN_OPT_NUM(S_OCEAN_HEIGHT_OPTIONS))    height_level = 0;
    if (swell_hs_level  < 0 || swell_hs_level  >= S_OCEAN_OPT_NUM(S_OCEAN_SWELL_HS_OPTIONS))  swell_hs_level = 3;
    if (swell_len_level < 0 || swell_len_level >= S_OCEAN_OPT_NUM(S_OCEAN_SWELL_LEN_OPTIONS)) swell_len_level = 3;

    Linglong_Ocean_Config c = s_cfg;   // 其余参数保持原值（模块默认）
    c.cam_y     = S_OCEAN_HEIGHT_OPTIONS[height_level].value;
    c.swell_hs  = S_OCEAN_SWELL_HS_OPTIONS[swell_hs_level].value;
    c.swell_len = S_OCEAN_SWELL_LEN_OPTIONS[swell_len_level].value;
    linglong_ocean_configure(&c);       // 内部只重建变动的部分
}

void linglong_ocean_configure(const Linglong_Ocean_Config *cfg) {
    if (cfg == NULL) return;

    // 选择性失效：只重建真正受影响的缓存，避免「调一个值就把三张 LUT + 波场全重烘」。
    //   观察高度  → 天空视图 LUT（依赖 r = RG + cam_y）、天空视图 UV 参数
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
    const int32_t cam_changed = (old.cam_y != s_cfg.cam_y) ? 1 : 0;
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
    else if (cam_changed) {
        // 只有天空视图 LUT 依赖观察高度（多重散射/透射率只与介质有关）
        if (s_state.sky_lut != NULL) { free(s_state.sky_lut); s_state.sky_lut = NULL; }
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
    if (s_state.sky_lut == NULL || d_alt > 0.25) {
        if (wo_build_sky_lut()) {
            s_state.sun_alt = (double)sun_alt;
            s_state.have_sun = 1;
        }
    }
    wo_ensure_waves();

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
        wo_sample_trans_lut(WO_RG + (double)s_cfg.cam_y / 1000.0, s_state.sun_dir[1], t);
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
    const double cam_y = (double)s_cfg.cam_y;

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
            double fi = cam_y + dy * ti - wo_height_at_fast(dx * ti, dz * ti, t);
            if (fi <= 0.0) {
                // 二分细化：容差随距离放大（1mm 的几何精度在 100m 外远低于一个像素的
                // 足迹，固定 1mm 只是白算）。实测 0.64% 像素差 >2/255，整幅 -5%。
                double ta = t_prev, tb = ti;
                double tol = wo_max(0.001, tb * 1e-4);
                for (int32_t r = 0; r < 24 && tb - ta > tol; r++) {
                    double tm = 0.5 * (ta + tb);
                    double fm = cam_y + dy * tm - wo_height_at_fast(dx * tm, dz * tm, t);
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
    const double px = dx * t_hit, pz = dz * t_hit;
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

    // ---- 太阳辐照与天空环境（均由 LUT 得出；每帧常量已由 prepare 算好）----
    const double sunE0 = s_state.sun_e[0];
    const double sunE1 = s_state.sun_e[1];
    const double sunE2 = s_state.sun_e[2];
    const double skyUp0 = s_state.sky_up[0], skyUp1 = s_state.sky_up[1], skyUp2 = s_state.sky_up[2];
    const double hemi0 = s_state.hemi[0], hemi1 = s_state.hemi[1], hemi2 = s_state.hemi[2];

    double sky_tmp[3];
    wo_sky_atmo(nrmX, nrmY, nrmZ, sky_tmp);
    const double skyN0 = sky_tmp[0], skyN1 = sky_tmp[1], skyN2 = sky_tmp[2];

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
    const double amb0 = skyUp0 * 1.3 + skyN0 * 0.5;
    const double amb1 = skyUp1 * 1.3 + skyN1 * 0.5;
    const double amb2 = skyUp2 * 1.3 + skyN2 * 0.5;
    double body0 = s_water_col[0] * (hemi0 * 1.6 + sunE0 * wo_max(Ldy, 0.0) * 0.8);
    double body1 = s_water_col[1] * (hemi1 * 1.6 + sunE1 * wo_max(Ldy, 0.0) * 0.8);
    double body2 = s_water_col[2] * (hemi2 * 1.6 + sunE2 * wo_max(Ldy, 0.0) * 0.8);
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

    // ---- 浅水：无岛屿（海底恒为深水），该分支不产生可见效果 ----
    // depth 与 exp(-depth*0.115) 都是常量（无岛屿时 bathy 恒为 -WO_ISL_DEEP），提出逐像素计算。
    const double near_isl = 1.0 - wo_smoothstep(12000.0, 20000.0, dist);
    const double depth = WO_ISL_DEEP;
    const double shal = 3.19101922481203e-4 * near_isl;   // = exp(-WO_ISL_DEEP * 0.115)
    if (shal > 0.004) {
        const double sdY = wo_max(Ldy, 0.0);
        const double k0 = hemi0 * 1.6 + sunE0 * sdY * 0.8;
        const double k1 = hemi1 * 1.6 + sunE1 * sdY * 0.8;
        const double k2 = hemi2 * 1.6 + sunE2 * sdY * 0.8;
        body0 = wo_mix(body0, WO_SHALLOW_COL[0] * k0, shal * 0.9);
        body1 = wo_mix(body1, WO_SHALLOW_COL[1] * k1, shal * 0.9);
        body2 = wo_mix(body2, WO_SHALLOW_COL[2] * k2, shal * 0.9);
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
        const double bub = foam[1] / 255.0;
        const double foamAlb0 = 0.82 * (0.78 + 0.3 * bub);
        const double foamAlb1 = 0.87 * (0.78 + 0.3 * bub);
        const double foamAlb2 = 0.92 * (0.78 + 0.3 * bub);
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
    }   // end if (cover > 1e-6)

    double col0 = body0 * (1.0 - Fr) + refl0 * Fr + spec0;
    double col1 = body1 * (1.0 - Fr) + refl1 * Fr + spec1;
    double col2 = body2 * (1.0 - Fr) + refl2 * Fr + spec2;
    col0 = wo_mix(col0, foamLit0, foamMask * 0.95);
    col1 = wo_mix(col1, foamLit1, foamMask * 0.95);
    col2 = wo_mix(col2, foamLit2, foamMask * 0.95);

    // ---- 拍岸泡沫：水深 3 m 以内的窄带（无岛屿，不出现）----
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

    #pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < fb_height; y++) {
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
            gfx_set_pixel(gfx, (uint32_t)x, (uint32_t)y, r, g, b);
        }
    }
}

#else  // !LINGLONG_ENABLE_DYNAMIC_LANDSCAPE

// 内存紧缺平台（M5Stack 等）：动态地景关闭，海洋模块退化为空实现
void linglong_ocean_config_default(Linglong_Ocean_Config *cfg) { (void)cfg; }
void linglong_ocean_configure(const Linglong_Ocean_Config *cfg) { (void)cfg; }
void linglong_ocean_set_levels(int32_t height_level, int32_t swell_hs_level, int32_t swell_len_level) {
    (void)height_level; (void)swell_hs_level; (void)swell_len_level;
}
void linglong_ocean_free(void) {}

#endif  // LINGLONG_ENABLE_DYNAMIC_LANDSCAPE
