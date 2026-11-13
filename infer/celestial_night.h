// ===============================================================================
// 夜天光（night-sky airglow）：散射模型 4 / 5 与实时海洋共用的夜间底色
//
// 太阳落下地平线之后，散射模型只剩「散射太阳光」这一项，于是整幅天空、海面反射、
// 环境光、大气透视全部归零 —— 物理上太阳确实不再照亮大气，画面确实该变黑，但真实
// 夜空并不黑：还有一层与太阳无关的微弱自发光。本文件给出这层的方向分布与颜色，
// 供 celestial_ocean.c（模型 5 / 实时海洋）与 ui_cloud.c（模型 4 的体积云天空）共用，
// 保证两个模型的夜天光在物理成分与方向上一致。
//
// ---- 物理依据（无月、无云、远离城市灯光）----
//   夜空总面亮度 ≈ 21.8 mag/arcsec² ≈ 1.5e-4 cd/m²；白天晴天顶 ≈ 4 mag/arcsec²
//   ≈ 5e3 cd/m²。两者比值 ≈ 3e-8（约 17 个星等）。成分与 V 波段占比：
//     · 气辉 airglow   ≈ 60%  90~100 km 的 OI 557.7 nm 绿线 + Na 589 nm 双线 +
//                             OH Meinel 带（红）；随天顶角按 van Rhijn 效应增亮
//     · 积分星光        ≈ 25%  蓝白，随大气质量消光（近地平变暗、偏红）
//     · 黄道光          ≈ 15%  暖白，沿黄道集中在太阳方位、贴地平最亮
//
// ---- 为什么不能直接用物理真值 ----
//   3e-8 的相对亮度落在任何色调映射的可分辨下限之下（8bit 输出恒为 0）。人眼/相机
//   在夜间靠暗适应（视杆细胞，约 +20 EV）把它抬进可见范围。故本函数输出的是
//   「暗适应后的等效辐亮度」：按成分比例归一化到 **天顶 G 通道 = 1**；各模块再乘
//   自己的标定量（见 celestial_ocean.c 的 WO_NIGHT_RAD、ui_cloud.c 的 CLOUD_NIGHT_RAD），
//   标定量按各自色调映射（AgX / ACES）标定到「肉眼可辨的微光」而非常规亮度。
//
// ---- 性能 ----
//   celestial_ocean.c 的逐像素着色要查 3 次天空辐亮度（法向/反射/地平雾），逐次调用
//   完整公式（2 次 sqrt + 3 次 exp）会吃掉约 10% 帧时间。故把「太阳相关」的部分每帧
//   算一次（linglong_night_sky_init 建仰角表），逐像素只剩「归一化 + 线性插值 + 一个
//   方位余弦」：约 25 ns/次，且白天 vis=0 时整表全零，行为与优化前逐位一致。
//
// 说明：本模型不含月光（天象仪的月亮未参与介质照明），也不含城市光污染。
// ===============================================================================
#ifndef __NANO_CELESTIAL_NIGHT_H__
#define __NANO_CELESTIAL_NIGHT_H__

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LL_NIGHT_LUT_N (25)     // 仰角表节点数（sin alt 0..1 均匀，线性插值）

typedef struct {
    int32_t ready;                          // init 成功（否则 eval 返回 0）
    double  sun_hx, sun_hz;                 // 太阳方向的水平单位向量（黄道光方位）
    double  rad[LL_NIGHT_LUT_N * 3];        // 气辉 + 星光（已含可见度淡入）R/G/B
    double  zod[LL_NIGHT_LUT_N];            // 黄道光的仰角包络（已含可见度淡入）
} Linglong_Night_Sky;

static inline double ll_night_clamp(double x, double lo, double hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

// 反向边界也成立（e0 > e1 时单调递减），便于直接写「越暗越亮」的淡入曲线
static inline double ll_night_smoothstep(double e0, double e1, double x) {
    double t = ll_night_clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// 夜天光可见度：太阳高度角正弦 → [0,1]。散射天光消失之后夜天光才成为天空的主项
// （真值里气辉常亮，只是白天被散射日光淹没）：-1.5° 起淡入、-12° 后满值。
static inline double linglong_night_visibility(double sun_sin_alt) {
    double sun_alt_deg = asinf((float)ll_night_clamp(sun_sin_alt, -1.0, 1.0)) * 57.29577951308232;
    return ll_night_smoothstep(-1.5, -12.0, sun_alt_deg);
}

// 单个仰角节点的完整夜天光计算（init 内部用；uy 为 sin(仰角)，已钳到 [0,1]）
static inline void ll_night_single(double uy, double vis, double out[3], double *zod_env) {
    // ---- 气辉：95 km 发光层的 van Rhijn 增亮 ----
    //   V(z) = 1 / sqrt(1 - (R/(R+H))² sin²z)，R = 6360 km、H = 95 km。
    //   理论值在地平线趋于 5.8；实际被大气消光与有限层厚压住，这里上限取 3.2。
    double vair = 1.0 / sqrt(1.0 - 0.97078 * (1.0 - uy * uy));
    if (vair > 3.2) vair = 3.2;

    // ---- 积分星光：大气质量消光（平面平行 + 曲率近似，天顶 = 1）----
    //   近地平路径变长 → 整体变暗且偏红（蓝光消光系数更大）。
    double am = 1.0 / sqrt(uy * uy + 0.0025);      // 天顶 1.0，仰角 5° ≈ 8.9
    if (am > 25.0) am = 25.0;
    double xm = am - 1.0;
    double stR = exp(-0.14 * xm);
    double stG = exp(-0.19 * xm);
    double stB = exp(-0.26 * xm);

    // 三成分合成（相对权重；天顶 ≈ (0.56, 0.83, 0.77) → 微偏青蓝的夜空）
    const double W_AIR = 0.55, W_STAR = 0.30;
    out[0] = vis * (W_AIR * 0.55 * vair + W_STAR * 0.85 * stR);
    out[1] = vis * (W_AIR * 1.00 * vair + W_STAR * 0.92 * stG);
    out[2] = vis * (W_AIR * 0.85 * vair + W_STAR * 1.00 * stB);

    // 黄道光的仰角包络：贴地平最亮（高度 ≲20° 的锥），天顶方向为 0
    *zod_env = vis * exp(-(uy * uy) / 0.09);
}

// 每帧初始化：太阳方向 (sx,sy,sz)（任意长度，y 轴向上）。太阳在地平线以上时整表为 0。
static inline void linglong_night_sky_init(Linglong_Night_Sky *n,
                                          double sx, double sy, double sz) {
    if (n == NULL) return;
    n->ready = 0;
    n->sun_hx = 1.0; n->sun_hz = 0.0;
    n->zod[0] = 0.0;
    for (int32_t i = 0; i < LL_NIGHT_LUT_N * 3; i++) n->rad[i] = 0.0;

    double sl = sqrt(sx * sx + sy * sy + sz * sz);
    if (sl < 1e-9) return;
    double vx = sx / sl, vy = sy / sl, vz = sz / sl;
    double hl = sqrt(vx * vx + vz * vz);
    if (hl > 1e-6) { n->sun_hx = vx / hl; n->sun_hz = vz / hl; }

    double vis = linglong_night_visibility(vy);
    n->ready = 1;
    if (vis <= 0.0) return;     // 白天/晨昏：整表为 0（eval 早退，逐位无影响）

    for (int32_t i = 0; i < LL_NIGHT_LUT_N; i++) {
        double uy = (double)i / (double)(LL_NIGHT_LUT_N - 1);
        ll_night_single(uy, vis, &n->rad[i * 3], &n->zod[i]);
    }
}

// 夜天光方向分布（相对权重，天顶 G ≈ 1）。
// **视线方向必须是单位向量、y 轴向上**（各调用方传的都是已归一化的视线方向：
// celestial_ocean.c 的 wo_sky_atmo（法向 / 反射方向 / 地平雾方向）与 ui_cloud.c
// 的 rayDir/upDir）；本函数不再做归一化——逐像素路径上省一次 sqrt（实测占夜天光
// 开销的 40%）。
// 黄道光按「水平方向与太阳方位夹角的余弦四次方」近似锥状分布（不做方位归一化：
// 天顶附近其仰角包络本身已衰减到 0）。
static inline void linglong_night_sky_eval(const Linglong_Night_Sky *n,
                                           double dx, double dy, double dz,
                                           double out[3]) {
    out[0] = 0.0; out[1] = 0.0; out[2] = 0.0;
    if (n == NULL || !n->ready) return;

    double ux = dx, uy = dy, uz = dz;   // 已约定为单位向量（见上）

    double f = (uy > 0.0) ? (uy * (double)(LL_NIGHT_LUT_N - 1)) : 0.0;
    int32_t i = (int32_t)f;
    if (i >= LL_NIGHT_LUT_N - 1) { i = LL_NIGHT_LUT_N - 2; f = (double)i; }
    double t = f - (double)i;

    const double *a = &n->rad[i * 3], *b = &n->rad[(i + 1) * 3];
    double r0 = a[0] + (b[0] - a[0]) * t;
    double r1 = a[1] + (b[1] - a[1]) * t;
    double r2 = a[2] + (b[2] - a[2]) * t;
    double ze = n->zod[i] + (n->zod[i + 1] - n->zod[i]) * t;

    // 黄道光：仅当太阳在地平线以下（ze > 0）时才需要算方位
    if (ze > 0.0) {
        double c = ux * n->sun_hx + uz * n->sun_hz;
        if (c > 0.0) {
            double c2 = c * c;
            double w = c2 * c2 * ze;
            r0 += w * 0.15 * 1.00;
            r1 += w * 0.15 * 0.97;
            r2 += w * 0.15 * 0.90;
        }
    }
    out[0] = r0; out[1] = r1; out[2] = r2;
}

#ifdef __cplusplus
}
#endif

#endif // __NANO_CELESTIAL_NIGHT_H__
