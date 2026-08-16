// ===============================================================================
// 体积云与天空仿真（CPU 实时 ray-march）
//
// 算法核心完全参照 flower（/mnt/d/Desktop/GitRepos/flower）Vulkan 引擎实现：
//   - 云形状：cloud_noise_common.glsl 的 tileable Perlin-Worley / Worley FBM 噪声，
//     cloud_render_common.glsl 的 cloudMap0/1/2 三层高度分层（0.5-3km / 3-7km / 7-11km）
//   - 云着色：cloud_render_common.glsl cloudColorCompute —— 球壳相交、双叶 HG 相位、
//     二次采样步长的光照线积分 volumetricShadow、多重散射（2 阶）近似、powder 项
//   - 天空：sky_render.glsl —— Bruneton 密度剖面（Rayleigh/Mie/臭氧）、单次散射积分 +
//     多重散射 LUT 近似 + 地面回弹；透射率 LUT 网格
//   - 后处理：aces.glsl 的 filmic ACES 色调映射
//
// 针对 CPU 的适配（不改算法核心，仅调整工程实现）：
//   * 3D/2D 噪声图样由 C 端按 flower 的噪声函数即时生成（本工程无 shader 编译管线与贴图资产），
//     基础噪声 64^3、细节噪声 32^3，采样为三线性 + wrap（对应 linearRepeatSampler）
//   * 透射率 LUT（64x16，40 步）与多重散射 LUT（48x24，16 方向球采样）逐帧预计算
//   * 云步进 128 步 / 光步进 12 步（flower 默认 128/12，保持默认）
//   * 渲染分辨率 = 屏幕一半（tty 为 320x240 全屏 → 内部 160x120），双线性放大到整屏
// ===============================================================================

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include <omp.h>

#include "ui_cloud.h"
#include "input_device.h"
#include "platform.h"

// ---------------------------------------------------------------------------
// 向量与标量工具
// ---------------------------------------------------------------------------
typedef struct { float x, y, z; } Cv3;
typedef struct { float x, y; } Cv2;

static inline Cv2 cv2(float x, float y) { Cv2 r = {x, y}; return r; }
static inline Cv3 v3(float x, float y, float z) { Cv3 r = {x, y, z}; return r; }
static inline Cv3 v3add(Cv3 a, Cv3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Cv3 v3sub(Cv3 a, Cv3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Cv3 v3mul(Cv3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline Cv3 v3mul3(Cv3 a, Cv3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline Cv3 v3div(Cv3 a, float s) { return v3(a.x / s, a.y / s, a.z / s); }
static inline Cv3 v3div3(Cv3 a, Cv3 b) { return v3(a.x / b.x, a.y / b.y, a.z / b.z); }
static inline float v3dot(Cv3 a, Cv3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Cv3 v3cross(Cv3 a, Cv3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float v3len(Cv3 a) { return sqrtf(v3dot(a, a)); }
static inline Cv3 v3norm(Cv3 a) { float l = v3len(a); return (l > 1e-12f) ? v3div(a, l) : v3(0, 1, 0); }
static inline Cv3 v3exp(Cv3 a) { return v3(expf(a.x), expf(a.y), expf(a.z)); }
static inline Cv3 v3clamp(Cv3 a, float lo, float hi) {
    return v3(fminf(hi, fmaxf(lo, a.x)), fminf(hi, fmaxf(lo, a.y)), fminf(hi, fmaxf(lo, a.z)));
}
static inline Cv3 v3lerp(Cv3 a, Cv3 b, float t) {
    return v3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}
static inline int v3anynan(Cv3 a) { return isnan(a.x) || isnan(a.y) || isnan(a.z); }
static inline int v3anyinf(Cv3 a) { return isinf(a.x) || isinf(a.y) || isinf(a.z); }

static inline float clampf(float x, float lo, float hi) { return fminf(hi, fmaxf(lo, x)); }
static inline float saturatef(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
static inline Cv3 v3maxs(Cv3 a, float s) { return v3(fmaxf(a.x, s), fmaxf(a.y, s), fmaxf(a.z, s)); }
static inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
// flower common_shader.glsl remap()：内部先 saturate（1205 行）
static inline float remapf(float value, float oMin, float oMax, float nMin, float nMax) {
    return nMin + (saturatef((value - oMin) / (oMax - oMin)) * (nMax - nMin));
}

#define kPI 3.14159265358979323846f

// ---------------------------------------------------------------------------
// 噪声：完全来自 flower cloud_noise_common.glsl
// ---------------------------------------------------------------------------
static inline float gfmod(float x, float m) { return x - m * floorf(x / m); }

static Cv3 hash33(Cv3 p) {
    uint32_t x = ((uint32_t)(int32_t)p.x) * 1597334673u;
    uint32_t y = ((uint32_t)(int32_t)p.y) * 3812015801u;
    uint32_t z = ((uint32_t)(int32_t)p.z) * 2798796415u;
    uint32_t t = x ^ y ^ z;
    const float uif = 1.0f / 4294967295.0f;
    Cv3 r;
    r.x = -1.0f + 2.0f * (float)(t * 1597334673u) * uif;
    r.y = -1.0f + 2.0f * (float)(t * 3812015801u) * uif;
    r.z = -1.0f + 2.0f * (float)(t * 2798796415u) * uif;
    return r;
}

// 梯度噪声（iq，tileable）
static float gradientNoise3(Cv3 x, float freq) {
    float fx = floorf(x.x), fy = floorf(x.y), fz = floorf(x.z);
    Cv3 p = v3(fx, fy, fz);
    Cv3 w = v3(x.x - fx, x.y - fy, x.z - fz);
    float u = w.x * w.x * w.x * (w.x * (w.x * 6.0f - 15.0f) + 10.0f);
    float v = w.y * w.y * w.y * (w.y * (w.y * 6.0f - 15.0f) + 10.0f);
    float q = w.z * w.z * w.z * (w.z * (w.z * 6.0f - 15.0f) + 10.0f);

    Cv3 ga = hash33(v3(gfmod(p.x, freq), gfmod(p.y, freq), gfmod(p.z, freq)));
    Cv3 gb = hash33(v3(gfmod(p.x + 1.0f, freq), gfmod(p.y, freq), gfmod(p.z, freq)));
    Cv3 gc = hash33(v3(gfmod(p.x, freq), gfmod(p.y + 1.0f, freq), gfmod(p.z, freq)));
    Cv3 gd = hash33(v3(gfmod(p.x + 1.0f, freq), gfmod(p.y + 1.0f, freq), gfmod(p.z, freq)));
    Cv3 ge = hash33(v3(gfmod(p.x, freq), gfmod(p.y, freq), gfmod(p.z + 1.0f, freq)));
    Cv3 gf = hash33(v3(gfmod(p.x + 1.0f, freq), gfmod(p.y, freq), gfmod(p.z + 1.0f, freq)));
    Cv3 gg = hash33(v3(gfmod(p.x, freq), gfmod(p.y + 1.0f, freq), gfmod(p.z + 1.0f, freq)));
    Cv3 gh = hash33(v3(gfmod(p.x + 1.0f, freq), gfmod(p.y + 1.0f, freq), gfmod(p.z + 1.0f, freq)));

    float va = v3dot(ga, w);
    float vb = v3dot(gb, v3(w.x - 1.0f, w.y, w.z));
    float vc = v3dot(gc, v3(w.x, w.y - 1.0f, w.z));
    float vd = v3dot(gd, v3(w.x - 1.0f, w.y - 1.0f, w.z));
    float ve = v3dot(ge, v3(w.x, w.y, w.z - 1.0f));
    float vf = v3dot(gf, v3(w.x - 1.0f, w.y, w.z - 1.0f));
    float vg = v3dot(gg, v3(w.x, w.y - 1.0f, w.z - 1.0f));
    float vh = v3dot(gh, v3(w.x - 1.0f, w.y - 1.0f, w.z - 1.0f));

    return va +
           u * (vb - va) +
           v * (vc - va) +
           q * (ve - va) +
           u * v * (va - vb - vc + vd) +
           v * q * (va - vc - ve + vg) +
           q * u * (va - vb - ve + vf) +
           u * v * q * (-va + vb + vc - vd + ve - vf - vg + vh);
}

// Perlin FBM（flower perlinfbm）
static float perlinfbm(Cv3 p, float freq, int octaves) {
    float G = exp2f(-0.85f);
    float amp = 1.0f;
    float noise = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        noise += amp * gradientNoise3(v3mul(p, freq), freq);
        freq *= 2.0f;
        amp *= G;
    }
    return noise;
}

// tileable worley（倒置，返回 1 - minDist）
static float worleyNoise(Cv3 uv, float freq) {
    Cv3 id = v3(floorf(uv.x), floorf(uv.y), floorf(uv.z));
    Cv3 p = v3(uv.x - id.x, uv.y - id.y, uv.z - id.z);
    float minDist = 10000.0f;
    for (float ox = -1.0f; ox <= 1.0f; ox += 1.0f) {
        for (float oy = -1.0f; oy <= 1.0f; oy += 1.0f) {
            for (float oz = -1.0f; oz <= 1.0f; oz += 1.0f) {
                Cv3 offset = v3(ox, oy, oz);
                Cv3 h = v3add(v3mul(hash33(v3(gfmod(id.x + offset.x, freq),
                                              gfmod(id.y + offset.y, freq),
                                              gfmod(id.z + offset.z, freq))), 0.5f), v3(0.5f, 0.5f, 0.5f));
                h = v3add(h, offset);
                Cv3 d = v3sub(p, h);
                float dd = v3dot(d, d);
                if (dd < minDist) minDist = dd;
            }
        }
    }
    return 1.0f - minDist;
}

static float worleyFbm(Cv3 p, float freq) {
    return worleyNoise(v3mul(p, freq), freq) * 0.625f +
           worleyNoise(v3mul(p, freq * 2.0f), freq * 2.0f) * 0.25f +
           worleyNoise(v3mul(p, freq * 4.0f), freq * 4.0f) * 0.125f;
}

// ===========================================================================
// 噪声纹理（CPU 生成，采样为三线性 + wrap）
// ===========================================================================
#define BASIC_DIM   64
#define DETAIL_DIM  32
#define CURL_DIM    64

static float *s_basicNoise  = NULL;
static float *s_detailNoise = NULL;
static float *s_curlNoise   = NULL; // CURL_DIM^2 * 3
static float *s_localNoise  = NULL; // CURL_DIM^2（云局部覆盖率）
static float *s_weather     = NULL; // CURL_DIM^2（天气覆盖率图）

// 三线性 + wrap 采样（对应 linearRepeatSampler），uvw ∈ [0,1]
// NOTE 大 |uvw| 时 cx/cy/cz 的取模存在浮点消去误差，floor 后索引可能==dim 或<0，
//      这里统一钳制到 [0,dim-1]，避免越界读（曾导致偶发段错误）。
static inline float sampleWrap3D(const float *tex, int dim, Cv3 uvw) {
    float cx = uvw.x * dim - 0.5f;
    float cy = uvw.y * dim - 0.5f;
    float cz = uvw.z * dim - 0.5f;
    float gx = cx - floorf(cx / dim) * dim;
    float gy = cy - floorf(cy / dim) * dim;
    float gz = cz - floorf(cz / dim) * dim;
    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    if (ix < 0) ix = 0; else if (ix >= dim) ix = dim - 1;
    if (iy < 0) iy = 0; else if (iy >= dim) iy = dim - 1;
    if (iz < 0) iz = 0; else if (iz >= dim) iz = dim - 1;
    float fx = gx - (float)ix, fy = gy - (float)iy, fz = gz - (float)iz;
    int ix1 = (ix + 1) % dim, iy1 = (iy + 1) % dim, iz1 = (iz + 1) % dim;
    if (ix1 < 0) ix1 += dim;
    if (iy1 < 0) iy1 += dim;
    if (iz1 < 0) iz1 += dim;
    float c000 = tex[(iz * dim + iy) * dim + ix];
    float c100 = tex[(iz * dim + iy) * dim + ix1];
    float c010 = tex[(iz * dim + iy1) * dim + ix];
    float c110 = tex[(iz * dim + iy1) * dim + ix1];
    float c001 = tex[(iz1 * dim + iy) * dim + ix];
    float c101 = tex[(iz1 * dim + iy) * dim + ix1];
    float c011 = tex[(iz1 * dim + iy1) * dim + ix];
    float c111 = tex[(iz1 * dim + iy1) * dim + ix1];
    float x00 = mixf(c000, c100, fx);
    float x10 = mixf(c010, c110, fx);
    float x01 = mixf(c001, c101, fx);
    float x11 = mixf(c011, c111, fx);
    return mixf(mixf(x00, x10, fy), mixf(x01, x11, fy), fz);
}

static inline float sampleWrap2D(const float *tex, int dim, Cv2 uv) {
    float cx = uv.x * dim - 0.5f;
    float cy = uv.y * dim - 0.5f;
    float gx = cx - floorf(cx / dim) * dim;
    float gy = cy - floorf(cy / dim) * dim;
    int ix = (int)floorf(gx), iy = (int)floorf(gy);
    if (ix < 0) ix = 0; else if (ix >= dim) ix = dim - 1;
    if (iy < 0) iy = 0; else if (iy >= dim) iy = dim - 1;
    float fx = gx - (float)ix, fy = gy - (float)iy;
    int ix1 = (ix + 1) % dim, iy1 = (iy + 1) % dim;
    if (ix1 < 0) ix1 += dim;
    if (iy1 < 0) iy1 += dim;
    float c00 = tex[iy * dim + ix];
    float c10 = tex[iy * dim + ix1];
    float c01 = tex[iy1 * dim + ix];
    float c11 = tex[iy1 * dim + ix1];
    return mixf(mixf(c00, c10, fx), mixf(c01, c11, fx), fy);
}

static inline Cv3 sampleWrap2D3(const float *tex, int dim, Cv2 uv) {
    float cx = uv.x * dim - 0.5f;
    float cy = uv.y * dim - 0.5f;
    float gx = cx - floorf(cx / dim) * dim;
    float gy = cy - floorf(cy / dim) * dim;
    int ix = (int)floorf(gx), iy = (int)floorf(gy);
    if (ix < 0) ix = 0; else if (ix >= dim) ix = dim - 1;
    if (iy < 0) iy = 0; else if (iy >= dim) iy = dim - 1;
    float fx = gx - (float)ix, fy = gy - (float)iy;
    int ix1 = (ix + 1) % dim, iy1 = (iy + 1) % dim;
    if (ix1 < 0) ix1 += dim;
    if (iy1 < 0) iy1 += dim;
    const float *r = tex;
    const float *g = tex + dim * dim;
    const float *b = tex + 2 * dim * dim;
    Cv3 c00 = v3(r[iy * dim + ix], g[iy * dim + ix], b[iy * dim + ix]);
    Cv3 c10 = v3(r[iy * dim + ix1], g[iy * dim + ix1], b[iy * dim + ix1]);
    Cv3 c01 = v3(r[iy1 * dim + ix], g[iy1 * dim + ix], b[iy1 * dim + ix]);
    Cv3 c11 = v3(r[iy1 * dim + ix1], g[iy1 * dim + ix1], b[iy1 * dim + ix1]);
    return v3lerp(v3lerp(c00, c10, fx), v3lerp(c01, c11, fx), fy);
}

static void cloud_generate_noise_textures(void) {
    if (s_basicNoise) return;
    s_basicNoise  = (float *)malloc(sizeof(float) * BASIC_DIM * BASIC_DIM * BASIC_DIM);
    s_detailNoise = (float *)malloc(sizeof(float) * DETAIL_DIM * DETAIL_DIM * DETAIL_DIM);
    s_curlNoise   = (float *)malloc(sizeof(float) * CURL_DIM * CURL_DIM * 3);
    s_localNoise  = (float *)malloc(sizeof(float) * CURL_DIM * CURL_DIM);
    s_weather     = (float *)malloc(sizeof(float) * CURL_DIM * CURL_DIM);

    // ---- 基础噪声：Perlin-Worley ----
    const float kBasicFrequency = 4.0f;
    const float kBasicNoiseMixFactor = 0.5f;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int z = 0; z < BASIC_DIM; z++) {
        for (int y = 0; y < BASIC_DIM; y++) {
            int x;
            #pragma omp simd
            for (x = 0; x < BASIC_DIM; x++) {
                Cv3 uvw = v3((x + 0.5f) / BASIC_DIM, (y + 0.5f) / BASIC_DIM, (z + 0.5f) / BASIC_DIM);
                float pfbm = mixf(1.0f, perlinfbm(uvw, kBasicFrequency, 7), kBasicNoiseMixFactor);
                pfbm = fabsf(pfbm * 2.0f - 1.0f); // billowy
                float wfbm_g = worleyFbm(uvw, kBasicFrequency * 1.0f);
                float wfbm_b = worleyFbm(uvw, kBasicFrequency * 2.0f);
                float wfbm_a = worleyFbm(uvw, kBasicFrequency * 4.0f);
                float pw = remapf(pfbm, 0.0f, 1.0f, wfbm_g, 1.0f);
                float wfbm = wfbm_g * 0.625f + wfbm_b * 0.25f + wfbm_a * 0.125f;
                s_basicNoise[(z * BASIC_DIM + y) * BASIC_DIM + x] = remapf(pw, wfbm - 1.0f, 1.0f, 0.0f, 1.0f);
            }
        }
    }

    // ---- 细节噪声：特定频段 worley FBM ----
    const float kDetailFrequency = 8.0f;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int z = 0; z < DETAIL_DIM; z++) {
        for (int y = 0; y < DETAIL_DIM; y++) {
            int x;
            #pragma omp simd
            for (x = 0; x < DETAIL_DIM; x++) {
                Cv3 uvw = v3((x + 0.5f) / DETAIL_DIM, (y + 0.5f) / DETAIL_DIM, (z + 0.5f) / DETAIL_DIM);
                float v = worleyFbm(uvw, kDetailFrequency * 1.0f) * 0.625f +
                          worleyFbm(uvw, kDetailFrequency * 2.0f) * 0.250f +
                          worleyFbm(uvw, kDetailFrequency * 4.0f) * 0.125f;
                s_detailNoise[(z * DETAIL_DIM + y) * DETAIL_DIM + x] = clampf(v, 0.0f, 1.0f);
            }
        }
    }

    // ---- 卷曲噪声（3 通道）与局部覆盖率/天气图 ----
    // 2D 旋转梯度的无散度 curl 场（对应 flower 的 curl 贴图），配合第 3 标量场构成 3 通道。
    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 0; y < CURL_DIM; y++) {
        for (int x = 0; x < CURL_DIM; x++) {
            float u = (float)x / CURL_DIM;
            float vv = (float)y / CURL_DIM;
            float eps = 1.0f / CURL_DIM;
            // 三个独立的势标量场
                        // 中心差分梯度
            float dp0dx = (perlinfbm(v3mul(v3(u + eps, vv, 0.0f), 2.0f), 2.0f, 3) -
                           perlinfbm(v3mul(v3(u - eps, vv, 0.0f), 2.0f), 2.0f, 3)) / (2.0f * eps);
            float dp0dy = (perlinfbm(v3mul(v3(u, vv + eps, 0.0f), 2.0f), 2.0f, 3) -
                           perlinfbm(v3mul(v3(u, vv - eps, 0.0f), 2.0f), 2.0f, 3)) / (2.0f * eps);
            float dp1dy = (perlinfbm(v3mul(v3(u, vv + eps, 1.0f), 2.0f), 2.0f, 3) -
                           perlinfbm(v3mul(v3(u, vv - eps, 1.0f), 2.0f), 2.0f, 3)) / (2.0f * eps);
            // 无散度旋转梯度场（-dP/dy, dP/dx），第三通道用另一标量场
            Cv3 g = v3(-dp0dy, dp0dx, -dp1dy);
            float gl = v3len(g);
            if (gl > 1e-5f) g = v3div(g, gl); else g = v3(0, 0, 1);
            s_curlNoise[(y * CURL_DIM + x) * 3 + 0] = g.x * 0.5f + 0.5f;
            s_curlNoise[(y * CURL_DIM + x) * 3 + 1] = g.y * 0.5f + 0.5f;
            s_curlNoise[(y * CURL_DIM + x) * 3 + 2] = g.z * 0.5f + 0.5f;
            // 局部覆盖率噪声
            float lc = perlinfbm(v3mul(v3(u, vv, 3.0f), 2.0f), 2.0f, 3) * 0.5f + 0.5f;
            s_localNoise[y * CURL_DIM + x] = clampf(lc, 0.0f, 1.0f);
            // 天气覆盖率图：低频 blob
            float w = worleyFbm(v3(u, vv, 0.25f), 0.5f) * 0.65f +
                      (perlinfbm(v3(u * 3.0f, vv * 3.0f, 0.5f), 3.0f, 2) * 0.5f + 0.5f) * 0.35f;
            s_weather[y * CURL_DIM + x] = clampf(w, 0.0f, 1.0f);
        }
    }
}

// ===========================================================================
// 大气与云参数（flower sky_component.cpp 直译）
// ===========================================================================
#define kEarthBottomRadius 6360.0f
#define kEarthTopRadius    6420.0f
#define kAtmospherePlanetRadiusOffset (21.0f * 0.001f)
#define kAtmosphereCameraOffset      kAtmospherePlanetRadiusOffset

typedef struct {
    // ---- atmosphere ----
    Cv3 rayleighScattering;
    Cv3 mieScattering;
    Cv3 mieExtinction;
    Cv3 absorptionExtinction;
    Cv3 groundAlbedo;
    float rayleighDensityExpScale;
    float mieDensityExpScale;
    float miePhaseG;
    float multipleScatteringFactor;
    float bottomRadius;
    float topRadius;
    float absorptionDensity0LayerWidth;
    float absorptionDensity0LinearTerm, absorptionDensity0ConstantTerm;
    float absorptionDensity1LinearTerm, absorptionDensity1ConstantTerm;
    // ---- cloud ----
    float cloudAreaStartHeight;
    float cloudAreaThickness;
    Cv2 cloudWeatherUVScale;
    float cloudCoverage;
    float cloudDensity;
    float cloudShadingSunLightScale;
    float cloudFogFade;
    float cloudMaxTraceingDistance;
    float cloudTracingStartMaxDistance;
    Cv3 cloudDirection;
    float cloudSpeed;
    float cloudMultiScatterExtinction;
    float cloudMultiScatterScatter;
    float cloudBasicNoiseScale;
    float cloudDetailNoiseScale;
    Cv3 cloudAlbedo;
    float cloudPhaseForward, cloudPhaseBackward, cloudPhaseMixFactor;
    float cloudPowderScale, cloudPowderPow;
    float cloudLightBasicStep;
    int   cloudLightStepNum;
    int   cloudMarchingStepNum;
    float cloudAmbientScale;
    // ---- light ----
    Cv3 sunDirection; // 指向太阳
    Cv3 sunColor;
    float sunIntensity;
} CloudScene;

static CloudScene s_scene;

static void cloud_default_scene(CloudScene *s) {
    s->bottomRadius = kEarthBottomRadius;
    s->topRadius    = kEarthTopRadius;
    s->groundAlbedo = v3(0.3f, 0.3f, 0.3f);
    s->rayleighDensityExpScale = -1.0f / 8.0f;
    s->mieDensityExpScale      = -1.0f / 1.2f;
    s->rayleighScattering = v3(0.005802f, 0.013558f, 0.033100f);
    s->mieScattering      = v3(0.003996f, 0.003996f, 0.003996f);
    s->mieExtinction      = v3(0.004440f, 0.004440f, 0.004440f);
    s->absorptionExtinction = v3(0.000650f, 0.001881f, 0.000085f);
    s->miePhaseG = 0.8f;
    s->multipleScatteringFactor = 1.0f;
    s->absorptionDensity0LayerWidth = 25.0f;
    s->absorptionDensity0LinearTerm = 1.0f / 15.0f;
    s->absorptionDensity0ConstantTerm = -2.0f / 3.0f;
    s->absorptionDensity1LinearTerm = -1.0f / 15.0f;
    s->absorptionDensity1ConstantTerm = 8.0f / 3.0f;

    // cloud（flower defaultCloudParameters）
    s->cloudAreaStartHeight = s->bottomRadius + 3.0f;
    s->cloudAreaThickness   = 8.0f;
    s->cloudWeatherUVScale  = cv2(0.02f, 0.02f);
    s->cloudCoverage        = 0.5f;
    s->cloudDensity         = 1.0f;
    s->cloudShadingSunLightScale = 1.3f; // 太阳光对云的直射增强（默认 1.0，观感更亮）
    s->cloudFogFade         = 1.0f;
    s->cloudMaxTraceingDistance     = 100.0f;
    s->cloudTracingStartMaxDistance = 350.0f;
    s->cloudDirection = v3norm(v3(0.8f, 0.0f, 0.4f));
    s->cloudSpeed = 0.05f;
    s->cloudMultiScatterExtinction = 0.175f;
    s->cloudMultiScatterScatter    = 1.0f;
    s->cloudBasicNoiseScale  = 0.3f;
    s->cloudDetailNoiseScale = 0.6f;
    s->cloudAlbedo = v3(1.0f, 1.0f, 1.0f);
    s->cloudPhaseForward  = 0.5f;
    s->cloudPhaseBackward = -0.5f;
    s->cloudPhaseMixFactor = 0.5f;
    // powder 参数放宽：让逆光边缘（金边）增益充足
    s->cloudPowderScale = 1.5f;
    s->cloudPowderPow   = 1.0f;
    s->cloudLightBasicStep = 15.0f;
    s->cloudLightStepNum = 12;
    s->cloudMarchingStepNum = 128;
    s->cloudAmbientScale = 1.0f;

    // 太阳（正午）
    s->sunDirection = v3norm(v3(0.15f, 0.96f, 0.0f));
    s->sunColor = v3(1.0f, 0.92f, 0.80f);
    s->sunIntensity = 48.0f;
}

// ===========================================================================
// 大气介质
// ===========================================================================
typedef struct {
    Cv3 scattering;
    Cv3 extinction;
    Cv3 scatteringMie;
    Cv3 extinctionMie;
    Cv3 scatteringRay;
} CloudMedium;

static inline float cloudAltitude(Cv3 worldPos, const CloudScene *s) {
    return v3len(worldPos) - s->bottomRadius;
}

static CloudMedium sampleMediumRGB(Cv3 worldPos, const CloudScene *s) {
    const float viewHeight = cloudAltitude(worldPos, s);
    const float densityMie = expf(s->mieDensityExpScale * viewHeight);
    const float densityRay = expf(s->rayleighDensityExpScale * viewHeight);
    const float densityOzo = saturatef(viewHeight < s->absorptionDensity0LayerWidth ?
        s->absorptionDensity0LinearTerm * viewHeight + s->absorptionDensity0ConstantTerm :
        s->absorptionDensity1LinearTerm * viewHeight + s->absorptionDensity1ConstantTerm);

    CloudMedium m;
    m.scatteringMie = v3mul(s->mieScattering, densityMie);
    m.extinctionMie = v3mul(s->mieExtinction, densityMie);
    m.scatteringRay = v3mul(s->rayleighScattering, densityRay);
    m.scattering = v3add(m.scatteringMie, m.scatteringRay); // 臭氧只吸收不散射
    m.extinction = v3add(m.extinctionMie, v3add(m.scatteringRay,
                        v3mul(s->absorptionExtinction, densityOzo)));
    return m;
}

// ===========================================================================
// 球相交（flower common_shader.glsl 直译）
// ===========================================================================
static inline float raySphereIntersectNearest(Cv3 r0, Cv3 rd, Cv3 s0, float sR) {
    float a = v3dot(rd, rd);
    Cv3 s02r0 = v3sub(r0, s0);
    float b = 2.0f * v3dot(rd, s02r0);
    float c = v3dot(s02r0, s02r0) - sR * sR;
    float delta = b * b - 4.0f * a * c;
    if (delta < 0.0f || a == 0.0f) return -1.0f;
    float sol0 = (-b - sqrtf(delta)) / (2.0f * a);
    float sol1 = (-b + sqrtf(delta)) / (2.0f * a);
    if (sol1 < 0.0f) return -1.0f;
    if (sol0 < 0.0f) return fmaxf(0.0f, sol1);
    return fmaxf(0.0f, fminf(sol0, sol1));
}

static inline float raySphereIntersectInside(Cv3 r0, Cv3 rd, Cv3 s0, float sR) {
    float a = v3dot(rd, rd);
    Cv3 s02r0 = v3sub(r0, s0);
    float b = 2.0f * v3dot(rd, s02r0);
    float c = v3dot(s02r0, s02r0) - sR * sR;
    float delta = fmaxf(0.0f, b * b - 4.0f * a * c);
    return (-b + sqrtf(delta)) / (2.0f * a);
}

static inline int raySphereIntersectOutSide(Cv3 r0, Cv3 rd, Cv3 s0, float sR, Cv2 *t0t1) {
    float a = v3dot(rd, rd);
    Cv3 s02r0 = v3sub(r0, s0);
    float b = 2.0f * v3dot(rd, s02r0);
    float c = v3dot(s02r0, s02r0) - sR * sR;
    float delta = b * b - 4.0f * a * c;
    if (delta < 0.0f || a == 0.0f) return 0;
    float sol0 = (-b - sqrtf(delta)) / (2.0f * a);
    float sol1 = (-b + sqrtf(delta)) / (2.0f * a);
    if (sol1 <= 0.0f || sol0 <= 0.0f) return 0;
    t0t1->x = sol0;
    t0t1->y = sol1;
    return 1;
}

// ===========================================================================
// 相位函数（flower common_shader.glsl）
// ===========================================================================
static inline float getUniformPhase() { return 1.0f / (4.0f * kPI); }
static inline float rayleighPhase(float cosTheta) {
    return (3.0f / (16.0f * kPI)) * (1.0f + cosTheta * cosTheta);
}
static inline float hgPhase(float g, float cosTheta) {
    float numer = 1.0f - g * g;
    float denom = 1.0f + g * g + 2.0f * g * cosTheta;
    return numer / (4.0f * kPI * denom * sqrtf(denom));
}
static inline float dualLobPhase(float g0, float g1, float w, float cosTheta) {
    return mixf(hgPhase(g0, cosTheta), hgPhase(g1, cosTheta), w);
}

// ===========================================================================
// 透射率 LUT 与多重散射 LUT（flower sky_render.glsl 直译，逐帧计算）
// ===========================================================================
#define TRANS_LUT_W 64
#define TRANS_LUT_H 16
#define MS_LUT_W    48
#define MS_LUT_H    24
#define MS_DIR_N    16   // 4x4 球面方向采样
#define MS_SAMPLE_STEPS 12

// 透射率/多重散射查找表：体积云渲染所用（约 26KB），按功能生命周期放 PSRAM——
// 原为静态数组，占内部 DRAM（与 DMA 帧缓冲同池）共 26KB，直接把启动 DMA 堆顶破，
// 导致 "Failed to alloc frame buffers!"。进入体积云时一次性申请（PSRAM）、不复释放
//（与应用生命周期一致，等价于原静态语义）。
static Cv3 *s_transLut = NULL;
static Cv3 *s_msLut = NULL;

static Cv2 lutTransmittanceParamsToUv(const CloudScene *s, float viewHeight, float viewZenithCosAngle) {
    float H = sqrtf(fmaxf(0.0f, s->topRadius * s->topRadius - s->bottomRadius * s->bottomRadius));
    float rho = sqrtf(fmaxf(0.0f, viewHeight * viewHeight - s->bottomRadius * s->bottomRadius));
    float y = rho / H;
    float discriminant = viewHeight * viewHeight * (viewZenithCosAngle * viewZenithCosAngle - 1.0f) +
                         s->topRadius * s->topRadius;
    float d = fmaxf(0.0f, -viewHeight * viewZenithCosAngle + sqrtf(fmaxf(0.0f, discriminant)));
    float dMin = s->topRadius - viewHeight;
    float dMax = rho + H;
    float x = (d - dMin) / (dMax - dMin);
    return cv2(x, y);
}

static void uvToLutTransmittanceParams(const CloudScene *s, float *viewHeight, float *viewZenithCosAngle, Cv2 uv) {
    float H = sqrtf(s->topRadius * s->topRadius - s->bottomRadius * s->bottomRadius);
    float rho = H * uv.y;
    float vh = sqrtf(rho * rho + s->bottomRadius * s->bottomRadius);
    float dMin = s->topRadius - vh;
    float dMax = rho + H;
    float d = dMin + uv.x * (dMax - dMin);
    float cosz = (d == 0.0f) ? 1.0f : (H * H - rho * rho - d * d) / (2.0f * vh * d);
    *viewHeight = vh;
    *viewZenithCosAngle = clampf(cosz, -1.0f, 1.0f);
}

// LUT 网格采样（linearClampEdge 等效：双线性 + 边界钳制）
static inline Cv3 sampleLut2D(const Cv3 *lut, int w, int h, float u, float v) {
    float cx = u * (float)w - 0.5f;
    float cy = v * (float)h - 0.5f;
    int ix = (int)floorf(cx), iy = (int)floorf(cy);
    float fx = cx - (float)ix, fy = cy - (float)iy;
    ix = (ix < 0) ? 0 : (ix > w - 2 ? w - 2 : ix);
    iy = (iy < 0) ? 0 : (iy > h - 2 ? h - 2 : iy);
    Cv3 c00 = lut[iy * w + ix];
    Cv3 c10 = lut[iy * w + ix + 1];
    Cv3 c01 = lut[(iy + 1) * w + ix];
    Cv3 c11 = lut[(iy + 1) * w + ix + 1];
    return v3lerp(v3lerp(c00, c10, fx), v3lerp(c01, c11, fx), fy);
}

typedef struct {
    Cv3 scatteredLight;
    Cv3 opticalDepth;
    Cv3 transmittance;
    Cv3 multiScatAs1;
} SingleScatteringResult;

// flower integrateScatteredLuminance 直译
static SingleScatteringResult integrateScatteredLuminance(
    Cv3 worldPos, Cv3 worldDir, Cv3 sunDir, const CloudScene *s,
    int bGround, int bMieRayPhase, int sampleCountFixed, int bVariableSampleCount,
    float tMaxMax, int sampleCountIni)
{
    SingleScatteringResult r;
    r.scatteredLight = v3(0, 0, 0);
    r.opticalDepth = v3(0, 0, 0);
    r.transmittance = v3(0, 0, 0);
    r.multiScatAs1 = v3(0, 0, 0);

    const Cv3 kEarthOrigin = v3(0, 0, 0);
    float tBottom = raySphereIntersectNearest(worldPos, worldDir, kEarthOrigin, s->bottomRadius);
    float tTop = raySphereIntersectNearest(worldPos, worldDir, kEarthOrigin, s->topRadius);

    float tMax = 0.0f;
    if (tBottom < 0.0f) {
        if (tTop < 0.0f) return r;
        tMax = tTop;
    } else {
        if (tTop > 0.0f) tMax = fminf(tTop, tBottom);
    }
    tMax = fminf(tMax, tMaxMax);
    if (tMax <= 0.0f) return r;

    float sampleCount, sampleCountFloor, tMaxFloor;
    if (bVariableSampleCount) {
        float lo = (float)(sampleCountIni > 0 ? sampleCountIni : 14);
        sampleCount = mixf(lo, 31.0f, saturatef(tMax * 0.01f));
        sampleCountFloor = floorf(sampleCount);
        tMaxFloor = tMax * sampleCountFloor / sampleCount;
    } else {
        sampleCount = (float)sampleCountFixed;
        sampleCountFloor = sampleCount;
        tMaxFloor = tMax;
    }
    if (sampleCount <= 0.0f) return r;

    const float uniformPhase = getUniformPhase();
    float cosTheta = v3dot(sunDir, worldDir);
    float miePhaseValue = hgPhase(s->miePhaseG, -cosTheta);
    float rayleighPhaseValue = rayleighPhase(cosTheta);
    Cv3 globalL = v3mul(s->sunColor, s->sunIntensity);

    Cv3 L = v3(0, 0, 0);
    Cv3 throughput = v3(1, 1, 1);
    Cv3 opticalDepth = v3(0, 0, 0);
    float t = 0.0f;
    const float sampleSegmentT = 0.3f;

    for (int sIdx = 0; sIdx < (int)sampleCount; sIdx++) {
        float dt;
        if (bVariableSampleCount) {
            float t0 = (float)sIdx / sampleCountFloor;
            float t1 = (float)(sIdx + 1) / sampleCountFloor;
            t0 = t0 * t0;
            t1 = t1 * t1;
            t0 = tMaxFloor * t0;
            t1 = (t1 > 1.0f) ? tMax : tMaxFloor * t1;
            t = t0 + (t1 - t0) * sampleSegmentT;
            dt = t1 - t0;
        } else {
            float newT = tMax * ((float)sIdx + sampleSegmentT) / sampleCount;
            dt = newT - t;
            t = newT;
        }

        Cv3 P = v3add(worldPos, v3mul(worldDir, t));
        CloudMedium medium = sampleMediumRGB(P, s);
        Cv3 sampleOpticalDepth = v3mul(medium.extinction, dt);
        Cv3 sampleTransmittance = v3exp(v3mul(sampleOpticalDepth, -1.0f));
        opticalDepth = v3add(opticalDepth, sampleOpticalDepth);

        float pHeight = v3len(P);
        Cv3 upVector = v3div(P, pHeight);
        float sunZenithCosAngle = v3dot(sunDir, upVector);
        Cv2 uu = lutTransmittanceParamsToUv(s, pHeight, sunZenithCosAngle);
        Cv3 transmittanceToSun = sampleLut2D(s_transLut, TRANS_LUT_W, TRANS_LUT_H, uu.x, uu.y);

        Cv3 phaseTimesScattering;
        if (bMieRayPhase) {
            phaseTimesScattering = v3add(v3mul(medium.scatteringMie, miePhaseValue),
                                         v3mul(medium.scatteringRay, rayleighPhaseValue));
        } else {
            phaseTimesScattering = v3mul(medium.scattering, uniformPhase);
        }

        // 地球阴影
        float tEarth = raySphereIntersectNearest(P, sunDir, v3add(kEarthOrigin,
            v3mul(upVector, kAtmospherePlanetRadiusOffset)), s->bottomRadius);
        float earthShadow = (tEarth >= 0.0f) ? 0.0f : 1.0f;

        // 多重散射近似（LUT）
        float mh = pHeight - s->bottomRadius;
        Cv2 muv = cv2(saturatef(sunZenithCosAngle * 0.5f + 0.5f),
                      clampf(mh / (s->topRadius - s->bottomRadius), 0.0f, 1.0f));
        Cv3 multiScatteredLuminance = sampleLut2D(s_msLut, MS_LUT_W, MS_LUT_H, muv.x, muv.y);

        Cv3 earthShadowScale = v3(earthShadow, earthShadow, earthShadow);
        Cv3 directTerm = v3mul3(v3mul3(earthShadowScale, transmittanceToSun), phaseTimesScattering);
        Cv3 S = v3add(v3mul3(globalL, directTerm), v3mul3(multiScatteredLuminance, medium.scattering));

        // multiScatAs1 累积
        Cv3 msint = v3div3(v3sub(medium.scattering, v3mul3(medium.scattering, sampleTransmittance)),
                          v3maxs(medium.extinction, 1e-4f));
        r.multiScatAs1 = v3add(r.multiScatAs1, v3mul3(throughput, msint));

        // 沿步长积分（frostbite 公式）
        Cv3 sint = v3div3(v3sub(S, v3mul3(S, sampleTransmittance)),
                         v3maxs(medium.extinction, 1e-4f));
        L = v3add(L, v3mul3(throughput, sint));
        throughput = v3mul3(throughput, sampleTransmittance);
    }

    if (bGround && tMax == tBottom && tBottom > 0.0f) {
        // 地面回弹
        Cv3 P = v3add(worldPos, v3mul(worldDir, tBottom));
        float pHeight = v3len(P);
        Cv3 upVector = v3div(P, pHeight);
        float sunZenithCosAngle = v3dot(sunDir, upVector);
        Cv2 uu = lutTransmittanceParamsToUv(s, pHeight, sunZenithCosAngle);
        Cv3 transmittanceToSun = sampleLut2D(s_transLut, TRANS_LUT_W, TRANS_LUT_H, uu.x, uu.y);
        float NdotL = saturatef(v3dot(v3norm(upVector), v3norm(sunDir)));
        Cv3 groundTerm = v3(NdotL * s->groundAlbedo.x / kPI,
                               NdotL * s->groundAlbedo.y / kPI,
                               NdotL * s->groundAlbedo.z / kPI);
        Cv3 bounce = v3mul3(v3mul3(v3mul3(globalL, transmittanceToSun),
                                    v3mul3(throughput, groundTerm)), v3(1, 1, 1));
        L = v3add(L, bounce);
    }

    r.scatteredLight = L;
    r.opticalDepth = opticalDepth;
    r.transmittance = throughput;
    return r;
}

// 预计算透射率 LUT（flower TRANSMITTANCE_LUT_PASS）
static void cloud_compute_transmittance_lut(CloudScene *s) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 0; y < TRANS_LUT_H; y++) {
        for (int x = 0; x < TRANS_LUT_W; x++) {
            Cv2 uv = cv2((x + 0.5f) / TRANS_LUT_W, (y + 0.5f) / TRANS_LUT_H);
            float viewHeight, viewZenithCosAngle;
            uvToLutTransmittanceParams(s, &viewHeight, &viewZenithCosAngle, uv);
            Cv3 worldPos = v3(0.0f, viewHeight, 0.0f);
            Cv3 worldDir = v3(0.0f, viewZenithCosAngle,
                              sqrtf(fmaxf(0.0f, 1.0f - viewZenithCosAngle * viewZenithCosAngle)));
            Cv3 sunDir = s->sunDirection; // flower 中 = -normalize(direction)；本端口 direction 已取“指向太阳”，故直接用
            SingleScatteringResult rr = integrateScatteredLuminance(
                worldPos, worldDir, sunDir, s, 0, 0, 40, 0, 9e6f, 40);
            s_transLut[y * TRANS_LUT_W + x] = v3exp(v3mul(rr.opticalDepth, -1.0f));
        }
    }
}

// 预计算多重散射 LUT（flower MULTI_SCATTER_PASS：球面 4x4=16 方向采样）
static void cloud_compute_multiscatter_lut(CloudScene *s) {
    const float sphereSolidAngle = 4.0f * kPI;
    const float isotropicPhase = 1.0f / sphereSolidAngle;
    const int kSqrtSampleCount = 4;
    const float invSC = 1.0f / (float)(kSqrtSampleCount * kSqrtSampleCount);

    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 0; y < MS_LUT_H; y++) {
        for (int x = 0; x < MS_LUT_W; x++) {
            Cv2 uv = cv2((x + 0.5f) / MS_LUT_W, (y + 0.5f) / MS_LUT_H);
            float cosSunZenithAngle = uv.x * 2.0f - 1.0f;
            Cv3 sunDir = v3(0.0f, cosSunZenithAngle,
                            sqrtf(fmaxf(0.0f, 1.0f - cosSunZenithAngle * cosSunZenithAngle)));
            float viewHeight = s->bottomRadius +
                saturatef(uv.y + kAtmospherePlanetRadiusOffset) *
                (s->topRadius - s->bottomRadius - kAtmospherePlanetRadiusOffset);
            Cv3 worldPos = v3(0.0f, viewHeight, 0.0f);

            Cv3 multiScatAs1 = v3(0, 0, 0);
            Cv3 scatteredLight = v3(0, 0, 0);
            for (int f = 0; f < kSqrtSampleCount * kSqrtSampleCount; f++) {
                float i = 0.5f + (float)(f / kSqrtSampleCount);
                float j = 0.5f + (float)(f % kSqrtSampleCount);
                float randA = i / (float)kSqrtSampleCount;
                float randB = j / (float)kSqrtSampleCount;
                float theta = 2.0f * kPI * randA;
                float phi = acosf(1.0f - 2.0f * randB);
                float cosPhi = cosf(phi), sinPhi = sinf(phi);
                float cosTheta = cosf(theta), sinTheta = sinf(theta);
                Cv3 worldDir = v3(cosTheta * sinPhi, cosPhi, sinTheta * sinPhi);

                SingleScatteringResult result = integrateScatteredLuminance(
                    worldPos, worldDir, sunDir, s, 1, 0, 0, 0, 9e6f, MS_SAMPLE_STEPS);
                multiScatAs1 = v3add(multiScatAs1, v3mul(result.multiScatAs1, sphereSolidAngle * invSC));
                scatteredLight = v3add(scatteredLight, v3mul(result.scatteredLight, sphereSolidAngle * invSC));
            }

            Cv3 multiScatFactor = v3(
                1.0f / fmaxf(1e-4f, 1.0f - multiScatAs1.x * isotropicPhase),
                1.0f / fmaxf(1e-4f, 1.0f - multiScatAs1.y * isotropicPhase),
                1.0f / fmaxf(1e-4f, 1.0f - multiScatAs1.z * isotropicPhase));
            Cv3 L = v3mul(v3mul3(v3mul(scatteredLight, isotropicPhase), multiScatFactor),
                          s->multipleScatteringFactor);
            s_msLut[y * MS_LUT_W + x] = L;
        }
    }
}

// ===========================================================================
// 云形状（flower cloud_render_common.glsl cloudMap0/1/2 直译）
// 云量档位：直接改变 cloudCoverage → 各层 coverage=kCoverage×(local+weather)
// 与 remap(basic, 1-coverage, 1, 0, 1) 阈值，即改变“有云区域”占空比。
// 该响应的有效窗口实测约 [0.33,0.46]，故六档按实测等间隔标定在此窗口内
// （见 CLOUD_COVERAGE_LEVELS），保证档间云量递进均匀、最高档不进入变暗区。
// ===========================================================================
static struct {
    float yaw;
    float pitch;
    float roll;            // 滚转（弧度，绕视线轴；天象仪集成时由 view_roll 提供）
    int   sun_preset;
    int   sun_auto;          // 1=太阳自动运动（东→天顶→西循环）；0=暂停/手动
    float sun_alt;           // 自动运动当前高度角 [ -18°, +90° ]（-18° 以下为完整黑夜，观感恒定）
    int   sun_dir;           // +1 上升 / -1 下降
    int   coverage_level;    // 云量档位
    float brightness;        // 云亮度
    int   sun_dirty;         // 太阳/大气参数变化 → 需重算 LUT
    int   frame_index;
    float app_time;
    uint64_t last_ts;        // 上一帧时间戳（毫秒）；必须用 64 位存 epoch 毫秒，
                             // 用 float 会丢失 ~0.1s 精度导致 dt=0、太阳/风冻结
    int   first_frame;
    // 云层显示掩码（按 1 键循环：所有 → 仅低层 → 仅中层 → 低+中 → 无云）
    int   layer_mask;
    int   layer_idx;
    // 投影与视场角：透视/鱼眼两套投影各带独立 FOV 档位
    int   proj;             // 0=透视  1=鱼眼
    int   fov_idx[2];       // 各投影的档位索引
    int   fov_deg[2];       // 各投影的档位值（度）
    int   fov_cur;          // 当前投影的档位值
    // 太阳镜头光晕（每帧由相机决定一次）
    int   sun_visible;
    float sun_u;             // 太阳屏幕 NDC 坐标
    float sun_v;
    float sun_vis;           // 太阳方向云透射率（= 太阳被云遮挡程度）
} s_ui;

static float cloudMap0(Cv3 posMeter, float normalizeHeight, float appTime, const CloudScene *s) {
    const float kCoverage = 0.5f;   // 与 flower 一致（底层云固有覆盖）
    const float kDensity  = s->cloudDensity * 2.0f;

    const Cv3 windDirection = s->cloudDirection;
    const float cloudSpeed = s->cloudSpeed;

    posMeter = v3add(posMeter, v3mul(windDirection, normalizeHeight * 500.0f));
    Cv3 posKm = v3mul(posMeter, 0.001f);

    Cv2 curlUv = cv2((appTime * cloudSpeed * 50.0f + posMeter.x) * 0.0000008f + 0.7f,
                     (appTime * cloudSpeed * 50.0f + posMeter.z) * 0.0000008f + 0.7f);
    Cv3 curl = v3sub(v3mul(sampleWrap2D3(s_curlNoise, CURL_DIM, curlUv), 2.0f), v3(1, 1, 1));
    posKm = v3add(posKm, v3mul(curl, 2.0f));

    Cv3 windOffset = v3mul(v3add(windDirection, v3(0.0f, 0.1f, 0.0f)), appTime * cloudSpeed);
    Cv2 sampleUv = cv2(posKm.x * s->cloudWeatherUVScale.x, posKm.z * s->cloudWeatherUVScale.y);
    float weatherValue = sampleWrap2D(s_weather, CURL_DIM, sampleUv);

    float localCoverage = sampleWrap2D(s_localNoise, CURL_DIM,
        cv2((appTime * cloudSpeed * 50.0f + posMeter.x) * 0.000001f + 0.5f,
            (appTime * cloudSpeed * 50.0f + posMeter.z) * 0.000001f + 0.5f));
    localCoverage = saturatef(localCoverage * 3.0f - 0.75f) * 0.2f;

    float coverage = saturatef(kCoverage * (localCoverage + weatherValue));
    float gradienShape = remapf(normalizeHeight, 0.10f, 0.80f, s->cloudCoverage * 1.9f, 0.2f) *
                         remapf(normalizeHeight, 0.00f, 0.1f, 0.5f, 1.0f);

    float basicNoise = sampleWrap3D(s_basicNoise, BASIC_DIM,
        v3add(posKm, windOffset)); // scale=1 已并入下面调用
    basicNoise = sampleWrap3D(s_basicNoise, BASIC_DIM,
        v3mul(v3add(posKm, windOffset), s->cloudBasicNoiseScale));

    float basicCloudNoise = gradienShape * basicNoise;
    float basicCloudWithCoverage = coverage * remapf(basicCloudNoise, 1.0f - coverage, 1.0f, 0.0f, 1.0f);

    Cv3 sampleDetailNoise = v3sub(posKm, v3mul(windOffset, 0.15f));
    float detailNoiseComposite = sampleWrap3D(s_detailNoise, DETAIL_DIM,
        v3mul(sampleDetailNoise, s->cloudDetailNoiseScale));
    float detailNoiseMixByHeight = 0.2f * mixf(detailNoiseComposite, 1.0f - detailNoiseComposite,
                                               saturatef(normalizeHeight * 10.0f));

    float densityShape = saturatef(0.01f + (1.0f - normalizeHeight) * 0.5f) * 0.25f *
        remapf(normalizeHeight, 0.0f, 0.3f, 0.0f, 1.0f) *
        remapf(normalizeHeight, 0.7f, 1.0f, 1.0f, 0.0f);

    float cloudDensity = densityShape * remapf(basicCloudWithCoverage, detailNoiseMixByHeight, 1.0f, 0.0f, 1.0f);
    cloudDensity = powf(cloudDensity, saturatef(1.0f - normalizeHeight) * 0.4f + 0.1f) * kDensity * 0.1f;
    return saturatef(cloudDensity);
}

static float cloudMap1(Cv3 posMeter, float normalizeHeight, float appTime, const CloudScene *s) {
    const float kCoverage = saturatef(s->cloudCoverage);
    const float kDensity  = s->cloudDensity * 0.35f;
    const Cv3 windDirection = s->cloudDirection;
    const float cloudSpeed = s->cloudSpeed;

    posMeter = v3add(posMeter, v3mul(windDirection, normalizeHeight * 500.0f));
    Cv3 posKm = v3mul(posMeter, 0.001f);

    Cv2 curlUv = cv2((appTime * cloudSpeed * 50.0f + posMeter.x) * 0.000001f - 0.3f,
                     (appTime * cloudSpeed * 50.0f + posMeter.z) * 0.000001f - 0.3f);
    Cv3 curl = v3sub(v3mul(sampleWrap2D3(s_curlNoise, CURL_DIM, curlUv), 2.0f), v3(1, 1, 1));
    posKm = v3add(posKm, v3mul(curl, 5.0f));

    Cv3 windOffset = v3mul(v3add(windDirection, v3(0.0f, 0.1f, 0.0f)), appTime * cloudSpeed);
    Cv2 sampleUv = cv2(posKm.x * s->cloudWeatherUVScale.x * 0.5f + 0.39f,
                       posKm.z * s->cloudWeatherUVScale.y * 0.5f + 0.39f);
    sampleUv.y *= 2.0f;
    float weatherValue = sampleWrap2D(s_weather, CURL_DIM, sampleUv);

    float localCoverage = sampleWrap2D(s_localNoise, CURL_DIM,
        cv2((appTime * cloudSpeed * 50.0f + posMeter.x) * 0.000001f - 0.11f,
            (appTime * cloudSpeed * 50.0f + posMeter.z) * 0.000001f - 0.11f));
    localCoverage = saturatef(localCoverage * 4.0f - 2.0f) * 0.5f;

    // 中层覆盖增益：程序化 basic 噪声分布集中于中低段，flower 原门槛 1-coverage
    // 下有效云面积趋零（实测中层仅 0.1%~0.4% 有云）。对 weather+local 掩膜放大
    // 两倍以上，把形状门槛压到噪声主体区间，恢复中层云的合理占比。
    float coverage = saturatef(kCoverage * (localCoverage + weatherValue) * 1.5f);
    float gradienShape = remapf(normalizeHeight, 0.00f, 0.01f, 0.1f, 1.0f) *
                         remapf(normalizeHeight, 0.10f, 0.80f, 0.7f, 0.2f);

    float basicNoise = sampleWrap3D(s_basicNoise, BASIC_DIM,
        v3mul(v3add(posKm, windOffset), s->cloudBasicNoiseScale * 2.0f));

    float basicCloudNoise = gradienShape * basicNoise;
    float basicCloudWithCoverage = coverage * remapf(basicCloudNoise, 1.0f - coverage, 1.0f, 0.0f, 1.0f);

    Cv3 sampleDetailNoise = v3sub(posKm, v3mul(windOffset, 0.15f));
    float detailNoiseComposite = sampleWrap3D(s_detailNoise, DETAIL_DIM,
        v3mul(sampleDetailNoise, s->cloudDetailNoiseScale * 2.0f));
    float detailNoiseMixByHeight = 0.2f * mixf(detailNoiseComposite, 1.0f - detailNoiseComposite,
                                               saturatef(normalizeHeight * 10.0f));

    float densityShape = saturatef(0.01f + (1.0f - normalizeHeight) * 0.5f) * 0.1f *
        remapf(normalizeHeight, 0.0f, 0.3f, 0.0f, 1.0f) *
        remapf(normalizeHeight, 0.7f, 1.0f, 1.0f, 0.0f);

    float cloudDensity = densityShape * remapf(basicCloudWithCoverage, detailNoiseMixByHeight, 1.0f, 0.0f, 1.0f);
    cloudDensity = powf(cloudDensity, saturatef(1.0f - normalizeHeight) * 0.4f + 0.1f) * kDensity * 0.1f;
    return saturatef(cloudDensity);
}

static float cloudMap2(Cv3 posMeter, float normalizeHeight, float appTime, const CloudScene *s) {
    const float kCoverage = s->cloudCoverage * 0.75f;
    const float kDensity  = s->cloudDensity * 0.20f;
    const Cv3 windDirection = s->cloudDirection;
    const float cloudSpeed = s->cloudSpeed;

    posMeter = v3add(posMeter, v3mul(windDirection, normalizeHeight * 500.0f));
    Cv3 posKm = v3mul(posMeter, 0.001f);

    Cv2 curlUv = cv2((appTime * cloudSpeed * 50.0f + posMeter.x) * 0.00000125f + 0.7f,
                     (appTime * cloudSpeed * 50.0f + posMeter.z) * 0.00000125f + 0.7f);
    Cv3 curl = v3sub(v3mul(sampleWrap2D3(s_curlNoise, CURL_DIM, curlUv), 2.0f), v3(1, 1, 1));
    posKm = v3add(posKm, v3mul(curl, 10.0f));

    Cv3 windOffset = v3mul(v3add(windDirection, v3(0.0f, 0.1f, 0.0f)), appTime * cloudSpeed);
    Cv2 sampleUv = cv2(posKm.x * s->cloudWeatherUVScale.x * 0.6f + 0.739f,
                       posKm.z * s->cloudWeatherUVScale.y * 0.6f + 0.739f);
    sampleUv.y *= 6.0f;
    float weatherValue = sampleWrap2D(s_weather, CURL_DIM, sampleUv);

    float localCoverage = sampleWrap2D(s_localNoise, CURL_DIM,
        cv2((appTime * cloudSpeed * 50.0f + posMeter.x) * 0.000001f - 0.39f,
            (appTime * cloudSpeed * 50.0f + posMeter.z) * 0.000001f - 0.39f));
    localCoverage = saturatef(1.0f - powf(localCoverage, 8.0f));

    float coverage = saturatef(kCoverage * (localCoverage + weatherValue));
    float gradienShape = remapf(normalizeHeight, 0.00f, 0.01f, 0.1f, 1.0f) *
                         remapf(normalizeHeight, 0.10f, 0.20f, 0.8f, 0.5f);

    float basicNoise = sampleWrap3D(s_basicNoise, BASIC_DIM,
        v3mul(v3add(v3add(posKm, windOffset), v3(0.39f, 0.39f, 0.39f)), s->cloudBasicNoiseScale * 3.0f));

    float basicCloudNoise = gradienShape * basicNoise;
    float basicCloudWithCoverage = coverage * remapf(basicCloudNoise, 1.0f - coverage, 1.0f, 0.0f, 1.0f);

    float densityShape = saturatef(0.01f + (1.0f - normalizeHeight) * 0.5f) * 0.1f *
        remapf(normalizeHeight, 0.0f, 0.3f, 0.0f, 1.0f) *
        remapf(normalizeHeight, 0.7f, 1.0f, 1.0f, 0.0f);

    float cloudDensity = densityShape * basicCloudWithCoverage;
    cloudDensity = cloudDensity * kDensity;
    return saturatef(cloudDensity);
}

// 云层显示掩码（按 1 键循环：所有 → 仅低层 → 仅中层 → 低+中 → 无云）
#define CLOUD_LAYER0 (1)
#define CLOUD_LAYER1 (2)
#define CLOUD_LAYER2 (4)

// 投影算法（7 键循环）：0=透视投影（默认），1=等距鱼眼投影
#define CLOUD_PROJ_PERSP    0
#define CLOUD_PROJ_FISHEYE  1

// 透视投影 FOV 档位（纵向半角°）
static const float FOV_DEGREES[] = { 12.0f, 20.0f, 28.0f, 38.0f, 50.0f };
#define FOV_DEGREES_NUM ((int)(sizeof(FOV_DEGREES) / sizeof(FOV_DEGREES[0])))
#define FOV_DEFAULT_IDX (2)   // 28° = 与历史默认一致

// 鱼眼投影 FOV 档位（最大视角半角°，对角线方向）
static const float FISHEYE_FOV_DEGREES[] = { 60.0f, 80.0f, 100.0f, 130.0f, 165.0f };
#define FISHEYE_FOV_DEGREES_NUM ((int)(sizeof(FISHEYE_FOV_DEGREES) / sizeof(FISHEYE_FOV_DEGREES[0])))
#define FISHEYE_FOV_DEFAULT_IDX (2)   // 100°

static float cloudMap(Cv3 posMeter, float normalizeHeight, float appTime, const CloudScene *s,
    float *actualH01, int shadowDepth)
{
    // 按当前云层掩码决定该样本是否参与密度场（光线阴影与视线采样一致）
    if (normalizeHeight < 0.4f) {
        if (!(s_ui.layer_mask & CLOUD_LAYER0)) { *actualH01 = 0.0f; return 0.0f; }
        *actualH01 = normalizeHeight / 0.4f;
        return cloudMap0(posMeter, *actualH01, appTime, s);
    }
    if (!shadowDepth) {
        if (normalizeHeight < 0.8f) {
            if (!(s_ui.layer_mask & CLOUD_LAYER1)) { *actualH01 = 0.0f; return 0.0f; }
            *actualH01 = (normalizeHeight - 0.4f) / 0.4f;
            return cloudMap1(posMeter, *actualH01, appTime, s);
        }
        if (!(s_ui.layer_mask & CLOUD_LAYER2)) { *actualH01 = 0.0f; return 0.0f; }
        *actualH01 = (normalizeHeight - 0.8f) / 0.2f;
        return cloudMap2(posMeter, *actualH01, appTime, s);
    }
    *actualH01 = 0.0f;
    return 0.0f;
}

typedef struct {
    float transmittanceToLight[2];
} CloudShadow;

// flower volumetricShadow 直译
static CloudShadow volumetricShadow(Cv3 posKm, Cv3 sunDirection, float appTime, const CloudScene *s,
    float msExtinctionFactor)
{
    CloudShadow pm;

    float kTotalLen = s->cloudLightBasicStep;
    float shadowStepCount = (float)s->cloudLightStepNum;
    float invShadowStepCount = 1.0f / shadowStepCount;

    float extinctionAcc0 = 0.0f;
    float extinctionAcc1 = 0.0f;
    float prevT = 0.0f;
    for (float shadowT = invShadowStepCount; shadowT <= 1.00001f; shadowT += invShadowStepCount) {
        float curT = shadowT * shadowT;
        float deltaT = curT - prevT;
        float extinctionFactor = deltaT * kTotalLen;
        float shadowSampleDis = kTotalLen * (prevT + deltaT * 0.5f);
        prevT = curT;

        Cv3 samplePosKm = v3add(posKm, v3mul(sunDirection, shadowSampleDis));
        float sampleHeightKm = v3len(samplePosKm);
        float sampleDt = sampleHeightKm - s->cloudAreaStartHeight;
        float normalizeHeight = sampleDt / s->cloudAreaThickness;
        Cv3 samplePosMeter = v3mul(samplePosKm, 1000.0f);
        float actualH01;
        float density = cloudMap(samplePosMeter, normalizeHeight, appTime, s, &actualH01, 0);
        extinctionAcc0 += density * extinctionFactor;
        extinctionAcc1 += (density * msExtinctionFactor) * extinctionFactor;
    }

    pm.transmittanceToLight[0] = expf(-extinctionAcc0 * 1000.0f);
    pm.transmittanceToLight[1] = expf(-extinctionAcc1 * 1000.0f);
    return pm;
}

static float powderEffectNew(float depth, float height, float VoL) {
    // 博客原文的"银边/金边"原版：方向性 —— 视线朝向太阳（VoL>0）时边缘受光增强；
    // flower shader 用的是卡通变体 -abs(VoL)，会抹掉方向性，本工程采用原版。
    float r = VoL * 0.5f + 0.5f;
    r = r * r;
    height = height * (1.0f - r) + r;
    return depth * height;
}

// 地面上涌光（见 render_frame 每帧由天顶天空亮度更新）。
static Cv3 s_groundUpwelling;

// flower cloudColorCompute 直译（剔除对场景深度、Froxel、远辉光网格等装饰贴图依赖）
typedef struct { Cv3 color; float transmittance; } CloudPixel;

static CloudPixel cloudPixelCompute(Cv3 worldPos, Cv3 worldDir, float appTime, float jitter,
    const CloudScene *s)
{
    CloudPixel out;
    out.color = v3(0, 0, 0);
    out.transmittance = 1.0f;

    const Cv3 kOrigin = v3(0, 0, 0);
    float radiusCloudStart = s->cloudAreaStartHeight;
    float radiusCloudEnd = radiusCloudStart + s->cloudAreaThickness;
    float viewHeight = v3len(worldPos);

    float tMin = 0.0f, tMax = 0.0f;
    int bEarlyOut = 0;
    if (viewHeight < radiusCloudStart) {
        float tEarth = raySphereIntersectNearest(worldPos, worldDir, kOrigin, s->bottomRadius);
        if (tEarth > 0.0f) bEarlyOut = 1;
        tMin = raySphereIntersectInside(worldPos, worldDir, kOrigin, radiusCloudStart);
        tMax = raySphereIntersectInside(worldPos, worldDir, kOrigin, radiusCloudEnd);
    } else if (viewHeight > radiusCloudEnd) {
        Cv2 t0t1 = {0, 0};
        if (!raySphereIntersectOutSide(worldPos, worldDir, kOrigin, radiusCloudEnd, &t0t1)) bEarlyOut = 1;
        Cv2 t2t3 = {0, 0};
        if (raySphereIntersectOutSide(worldPos, worldDir, kOrigin, radiusCloudStart, &t2t3)) {
            tMin = t0t1.x; tMax = t2t3.x;
        } else {
            tMin = t0t1.x; tMax = t0t1.y;
        }
    } else {
        float tStart = raySphereIntersectNearest(worldPos, worldDir, kOrigin, radiusCloudStart);
        tMax = (tStart > 0.0f) ? tStart : raySphereIntersectInside(worldPos, worldDir, kOrigin, radiusCloudEnd);
        tMin = 0.0f;
    }

    tMin = fmaxf(tMin, 0.0f);
    tMax = fmaxf(tMax, 0.0f);
    if (tMax <= tMin || tMin > s->cloudTracingStartMaxDistance) bEarlyOut = 1;
    float marchingDistance = fminf(s->cloudMaxTraceingDistance, tMax - tMin);
    if (marchingDistance <= 0.0f) bEarlyOut = 1;
    tMax = tMin + marchingDistance;
    if (bEarlyOut) return out;

    const int stepCountUnit = s->cloudMarchingStepNum;
    const float stepCount = (float)stepCountUnit;
    const float stepT = (tMax - tMin) / stepCount;

    float sampleT = tMin + 0.001f * stepT + stepT * saturatef(jitter);

    Cv3 sunColor = v3mul(s->sunColor, s->sunIntensity);
    Cv3 sunDirection = s->sunDirection;
    float VoL = v3dot(worldDir, sunDirection);

    float transmittance = 1.0f;
    Cv3 scatteredLight = v3(0, 0, 0);

    float phase = dualLobPhase(s->cloudPhaseForward, s->cloudPhaseBackward, s->cloudPhaseMixFactor, -VoL);
    // 多重散射相位项（flower getParticipatingMediaPhase(phase, 0.5)）
    float phase1 = mixf(getUniformPhase(), phase, 0.5f);

    // 两端大气透射率（flower：inTransmittanceLut 采样）
    Cv3 atmosphereTransmittance0, atmosphereTransmittance1;
    {
        Cv3 sp = v3add(worldPos, v3mul(worldDir, sampleT));
        float sh = v3len(sp);
        Cv3 upv = v3div(sp, sh);
        float cosz = v3dot(sunDirection, upv);
        Cv2 uu = lutTransmittanceParamsToUv(s, viewHeight, cosz);
        atmosphereTransmittance0 = sampleLut2D(s_transLut, TRANS_LUT_W, TRANS_LUT_H, uu.x, uu.y);
    }
    {
        Cv3 sp = v3add(worldPos, v3mul(worldDir, tMax));
        float sh = v3len(sp);
        Cv3 upv = v3div(sp, sh);
        float cosz = v3dot(sunDirection, upv);
        Cv2 uu = lutTransmittanceParamsToUv(s, viewHeight, cosz);
        atmosphereTransmittance1 = sampleLut2D(s_transLut, TRANS_LUT_W, TRANS_LUT_H, uu.x, uu.y);
    }

    // 日落时多重散射增强
    float sunSetScale = 1.0f + saturatef(1.0f - sunDirection.y * 2.0f);

    for (int i = 0; i < stepCountUnit; i++) {
        Cv3 samplePos = v3add(worldPos, v3mul(worldDir, sampleT));
        float sampleHeight = v3len(samplePos);
        Cv3 atmosphereTransmittance = v3lerp(atmosphereTransmittance0, atmosphereTransmittance1,
                                             saturatef(sampleT / marchingDistance));

        float normalizeHeight = (sampleHeight - s->cloudAreaStartHeight) / s->cloudAreaThickness;
        Cv3 samplePosMeter = v3mul(samplePos, 1000.0f);
        float actualH01 = 0.0f;
        float stepCloudDensity = cloudMap(samplePosMeter, normalizeHeight, appTime, s, &actualH01, 0);

        if (stepCloudDensity > 0.0f) {
            float opticalDepth = stepCloudDensity * stepT * 1000.0f;
            float stepTransmittance = fmaxf(expf(-opticalDepth), expf(-opticalDepth * 0.25f) * 0.7f);

            CloudShadow shadow = volumetricShadow(samplePos, sunDirection, appTime, s,
                                                  s->cloudMultiScatterExtinction);
            CloudShadow shadowAmbient;
            int bGroundContrib = (s->cloudFogFade > 0.0f);
            if (bGroundContrib) {
                shadowAmbient = volumetricShadow(samplePos, v3(0, 1, 0), appTime, s,
                                                 s->cloudMultiScatterExtinction);
            }

            // powder（奶油感边缘，方向性银边——逆光金边核心）
            float depthProbability = powf(clampf(stepCloudDensity * 10.0f * s->cloudPowderPow,
                                                 0.0f, s->cloudPowderScale),
                                          remapf(actualH01, 0.3f, 0.85f, 0.5f, 2.0f));
            depthProbability += 0.05f;
            float verticalProbability = powf(remapf(actualH01, 0.07f, 0.22f, 0.1f, 1.0f), 0.8f);
            float powderEffect = powderEffectNew(depthProbability, verticalProbability, VoL);

            Cv3 sunlightTerm = v3mul3(atmosphereTransmittance,
                v3mul(sunColor, s->cloudShadingSunLightScale));

            Cv3 ambientLit = v3mul3(v3mul3(s_groundUpwelling, v3(powderEffect, powderEffect, powderEffect)),
                v3mul3(v3(s->cloudAmbientScale, s->cloudAmbientScale, s->cloudAmbientScale),
                       v3mul3(v3(1.0f - sunDirection.y * sunDirection.y,
                                 1.0f - sunDirection.y * sunDirection.y,
                                 1.0f - sunDirection.y * sunDirection.y),
                              v3lerp(atmosphereTransmittance, v3(1, 1, 1), saturatef(1.0f - transmittance)))));

            float sigmaS = stepCloudDensity;
            float sigmaE = fmaxf(sigmaS, 1e-8f);
            Cv3 scatteringCoeff0 = v3(sigmaS, sigmaS, sigmaS);
            scatteringCoeff0 = v3mul3(scatteringCoeff0, s->cloudAlbedo);
            float extinctionCoeff0 = sigmaE;

            // 多散射阶系数（flower：系数逐阶乘 MsScatter / MsExtinction）
            float MsExtinctionFactor = s->cloudMultiScatterExtinction / sunSetScale;
            float MsScatterFactor    = s->cloudMultiScatterScatter;
            Cv3 scatteringCoeff1 = v3mul3(scatteringCoeff0,
                                          v3(MsScatterFactor, MsScatterFactor, MsScatterFactor));
            float extinctionCoeff1 = extinctionCoeff0 * MsExtinctionFactor;

            // 高阶散射（本端口 kMsCount=2）
            // ms=1（多重散射阶）：相位向均匀相位混合；分数阶透射率衰减系数
            {
                float sunVisibility0 = shadow.transmittanceToLight[0];
                float sunVisibility1 = shadow.transmittanceToLight[1];

                // 一阶（ms=0）：HG 双叶相位
                Cv3 sunSky0 = v3mul3(v3mul3(v3mul3(v3(sunVisibility0, sunVisibility0, sunVisibility0),
                                                   sunlightTerm), v3(phase, phase, phase)),
                                     v3(powderEffect, powderEffect, powderEffect));
                if (bGroundContrib) {
                    Cv3 amb = v3mul3(v3(shadowAmbient.transmittanceToLight[0],
                                         shadowAmbient.transmittanceToLight[0],
                                         shadowAmbient.transmittanceToLight[0]), ambientLit);
                    sunSky0 = v3add(sunSky0, amb);
                }

                Cv3 ext0 = v3(fmaxf(1e-4f, extinctionCoeff0), fmaxf(1e-4f, extinctionCoeff0), fmaxf(1e-4f, extinctionCoeff0));
                Cv3 stepScatter0 = v3mul3(v3(transmittance, transmittance, transmittance),
                    v3div3(v3sub(v3mul3(sunSky0, scatteringCoeff0),
                                 v3mul(v3mul3(sunSky0, scatteringCoeff0), stepTransmittance)), ext0));
                scatteredLight = v3add(scatteredLight, stepScatter0);
                transmittance *= stepTransmittance;

                // 二阶（ms=1）：多重散射
                Cv3 sunSky1 = v3mul3(v3mul3(v3mul3(v3(sunVisibility1, sunVisibility1, sunVisibility1),
                                                   sunlightTerm), v3(phase1, phase1, phase1)),
                                     v3(powderEffect, powderEffect, powderEffect));
                if (bGroundContrib) {
                    Cv3 amb = v3mul3(v3(shadowAmbient.transmittanceToLight[1],
                                         shadowAmbient.transmittanceToLight[1],
                                         shadowAmbient.transmittanceToLight[1]), ambientLit);
                    sunSky1 = v3add(sunSky1, amb);
                }

                Cv3 ext1 = v3(fmaxf(1e-4f, extinctionCoeff1), fmaxf(1e-4f, extinctionCoeff1), fmaxf(1e-4f, extinctionCoeff1));
                Cv3 stepScatter1 = v3mul3(v3(transmittance, transmittance, transmittance),
                    v3div3(v3sub(v3mul3(sunSky1, scatteringCoeff1),
                                 v3mul(v3mul3(sunSky1, scatteringCoeff1), stepTransmittance)), ext1));
                scatteredLight = v3add(scatteredLight, stepScatter1);
            }
        }

        if (transmittance <= 0.001f) break;
        sampleT += stepT;
    }

    out.color = scatteredLight;
    out.transmittance = transmittance;
    if (v3anynan(out.color) || v3anyinf(out.color)) { out.color = v3(0, 0, 0); out.transmittance = 1.0f; }
    return out;
}

// ===========================================================================
// 色调映射（aces.glsl filmic 近似）
// ===========================================================================
// ===========================================================================
// 太阳强点光源/镜头光晕
// 参照 flower install/shader/post_tonemapper.glsl + lens.glsl：
//   - 太阳在画面前方时，把 sunDirection 投影为屏幕坐标 sunUv；
//   - 可见性 = 沿太阳方向云透射率（等价 lens_visible.glsl 的 cloudColor.a × 大气透射）；
//   - 在 HDR 合成后、色调映射前叠加 lensFlare 的 glare/flare/ring/orb/星芒。
//   其中各向异性十字星芒 anflares2 在 flower 中默认关闭(#if 0)，此处按用户要求开启。
// ===========================================================================
static inline Cv2 cv2sub2(Cv2 a, Cv2 b) { return cv2(a.x - b.x, a.y - b.y); }
static inline float cv2len2(Cv2 a) { return sqrtf(a.x * a.x + a.y * a.y); }

// lens.glsl flare()：带色散(色差)的小光斑，RGB 分量分别偏移
static Cv3 cloud_flare_c(Cv2 v, Cv2 sun, float dist, float chroma, float size, float gain) {
    Cv2 d0 = cv2sub2(v, cv2(sun.x * (dist - chroma), sun.y * (dist - chroma)));
    Cv2 d1 = cv2sub2(v, cv2(sun.x * dist, sun.y * dist));
    Cv2 d2 = cv2sub2(v, cv2(sun.x * (dist + chroma), sun.y * (dist + chroma)));
    float l0 = cv2len2(d0), l1 = cv2len2(d1), l2 = cv2len2(d2);
    float r = fmaxf(0.0f, 0.01f - powf(l0, 2.4f) * (1.0f / (size * 2.0f))) * 0.85f;
    float g = fmaxf(0.0f, 0.01f - powf(l1, 2.4f) * (1.0f / (size * 2.0f))) * 1.0f;
    float b = fmaxf(0.0f, 0.01f - powf(l2, 2.4f) * (1.0f / (size * 2.0f))) * 1.5f;
    return v3mul(v3(r, g, b), gain);
}

// lens.glsl ring()：衍射环
static Cv3 cloud_ring_c(Cv2 v, Cv2 sun, float dist, float chroma, float blur) {
    float l0 = cv2len2(cv2sub2(v, cv2(sun.x * (dist - chroma), sun.y * (dist - chroma))));
    float l1 = cv2len2(cv2sub2(v, cv2(sun.x * dist, sun.y * dist)));
    float l2 = cv2len2(cv2sub2(v, cv2(sun.x * (dist + chroma), sun.y * (dist + chroma))));
    float r = fmaxf(0.0f, 1.0f / (1.0f + 250.0f * powf(l0, blur))) * 0.8f;
    float g = fmaxf(0.0f, 1.0f / (1.0f + 250.0f * powf(l1, blur))) * 1.0f;
    float b = fmaxf(0.0f, 1.0f / (1.0f + 250.0f * powf(l2, blur))) * 1.5f;
    return v3(r, g, b);
}

// 星芒（lens.glsl anflares2() 的各向异性十字思路，flower 默认 #if 0，此处按需求开启）：
// 竖直+水平两条亮臂，外加两条 45° 弱臂，形成星光。
static Cv3 cloud_starburst_c(Cv2 v, Cv2 sun, float armLen, float armW, float brightness) {
    Cv2 d = cv2sub2(v, sun);
    float ax = fabsf(d.x), ay = fabsf(d.y);
    // 主十字（竖臂 + 横臂）
    float rv = fmaxf(0.0f, 1.0f - ax / armW) * fmaxf(0.0f, 1.0f - ay / armLen);
    float rh = fmaxf(0.0f, 1.0f - ax / armLen) * fmaxf(0.0f, 1.0f - ay / armW);
    // 45° 次臂（弱）
    float r45 = fmaxf(0.0f, 1.0f - (ax + ay) / (armLen * 1.6f)) *
                fmaxf(0.0f, 1.0f - fabsf(ax - ay) / armW);
    Cv3 s = v3add(v3mul(v3(rv, rv, rv), 1.0f), v3mul(v3(rh, rh, rh), 1.0f));
    Cv3 s45 = v3mul(v3(r45, r45, r45), 0.45f);
    Cv3 star = v3mul(v3add(s, s45), brightness);
    return v3mul3(star, star); // 尖峰化
}

// HDR 太阳光晕：返回待累加到场景 HDR 的增量
// (ux,uy)/(sux,suy) 为 NDC 坐标（±1）
static Cv3 cloud_lens_hdr(float ux, float uy, float sux, float suy, float sunVis,
                          const CloudScene *s)
{
    if (sunVis <= 0.004f) return v3(0, 0, 0);
    Cv2 v = cv2(ux, uy), sunp = cv2(sux, suy);

    // 太阳核心盘（角半径≈0.25°，NDC 半高≈tan(14°)≈0.25；放大至内部分辨率可见）
    Cv3 core = v3(0, 0, 0);
    {
        float ds = cv2len2(cv2sub2(v, sunp));
        float disc = fmaxf(0.0f, 1.0f - ds / 0.030f);
        disc = disc * disc;
        core = v3mul3(v3mul3(s->sunColor, v3(s->sunIntensity, s->sunIntensity, s->sunIntensity)),
                      v3(disc, disc, disc));
    }

    const float fovFactor = 2.0f; // 1/tan(fovy/2) 下限，与 flower 一致
    const float size = 0.5f * fovFactor * 0.9f;

    // orb 序列（lens.glsl orb() 精简为 4 阶）
    Cv3 l = v3(0, 0, 0);
    for (int i = 0; i < 4; i++) {
        float j = (float)i + 1.0f;
        float offset = j / (j + 0.1f);
        float ss = size / (j + 1.0f);
        l = v3add(l, v3mul(cloud_flare_c(v, sunp, offset, (j / 6.0f) * 0.5f, ss, ss), 0.15f * j));
    }

    // 衍射环 + 数个色散光斑（flower lens.glsl 中相同参数族）
    l = v3add(l, v3mul(cloud_ring_c(v, sunp, 1.0f, 0.02f, 1.4f), 0.02f));
    l = v3add(l, v3mul(cloud_flare_c(v, sunp, -2.00f, 0.05f, size * 0.05f, 1.0f), 0.5f));
    l = v3add(l, v3mul(cloud_flare_c(v, sunp, -0.90f, 0.02f, size * 0.03f, 1.0f), 0.25f));
    l = v3add(l, v3mul(cloud_flare_c(v, sunp, -0.35f, 0.02f, size * 0.04f, 1.0f), 1.0f));
    l = v3add(l, v3mul(cloud_flare_c(v, sunp, -0.25f, 0.01f, size * 0.15f, 1.0f), 0.6f));
    l = v3add(l, v3mul(cloud_flare_c(v, sunp, 0.30f, 0.02f, size * 0.20f, 1.0f), 0.8f));
    l = v3add(l, v3mul(cloud_flare_c(v, sunp, 1.20f, 0.03f, size * 0.10f, 1.0f), 0.5f));

    // 太阳眩光辐射（lens.glsl glare 公式）
    float gd = cv2len2(cv2sub2(v, sunp)) * 25.0f / fovFactor;
    float gd2 = gd * gd;
    float phase = atan2f(uy - suy, ux - sux) + 0.131f;
    float gl = 2.0f - fminf(1.0f, gd) + sinf(phase * 12.0f) * fminf(1.0f, fmaxf(0.0f, gd * 2.5f - 0.2f));
    gl = gl * gl;
    gl *= 3e-4f / (gd2 * gd2 + 1e-6f);

    Cv3 ghost = v3mul3(s->sunColor, v3(gl * 0.1f, gl * 0.1f, gl * 0.1f));

    // 星芒（十字 + 45° 次臂；臂长/臂宽以 NDC 计，内部分辨率下可辨）
    Cv3 star = cloud_starburst_c(v, sunp, 0.14f, 0.045f, 1.0f);
    star = v3add(star, v3mul3(star, v3(2.0f, 2.0f, 2.0f)));

    Cv3 out = v3add(core, v3add(v3add(l, ghost), star));
    out = v3mul(out, sunVis * 3.0f); // 可见性（云透射）× 曝光增益
    return out;
}

static Cv3 cloudAcesFilm(Cv3 x) {
    // Narkowicz 2015 ACES filmic 近似（aces.glsl 家族的紧凑实现）
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    Cv3 r = v3div3(v3mul3(x, v3add(v3mul(x, a), v3(b, b, b))),
                   v3add(v3mul3(x, v3add(v3mul(x, c), v3(d, d, d))), v3(e, e, e)));
    return v3clamp(r, 0.0f, 1.0f);
}


// ===========================================================================
// 应用状态与渲染
// ===========================================================================
// 内部渲染分辨率 = 屏幕的一半（tty 已修正为 320x240，则内部 160x120），
// 双线性放大到整屏。步进默认（128/12）下单像素代价约 1.6us，320x240 全屏约 8fps，
// 半分辨率计算约 30fps，兼顾清晰度与终端吞吐。
#define CLOUD_MAX_W (256)
#define CLOUD_MAX_H (192)

// 太阳位置预设：仰角（度）/方位角（度）/颜色/强度
typedef struct {
    int   elev_deg;
    int   azim_deg;
    Cv3   color;
    float intensity;
    const wchar_t *name;
} SunPreset;

static const SunPreset SUN_PRESETS[] = {
    { 82,  -10, {1.00f, 0.93f, 0.82f}, 48.0f, L"正午" },
    { 45,   30, {1.00f, 0.93f, 0.80f}, 44.0f, L"午后" },
    { 18,   70, {1.00f, 0.70f, 0.40f}, 40.0f, L"傍晚" },
    {  5,   80, {1.00f, 0.45f, 0.20f}, 36.0f, L"日落" },
    { -8,  100, {0.50f, 0.55f, 0.80f}, 10.0f, L"月夜" },
    { -18, 90,  {0.20f, 0.26f, 0.45f},  3.0f, L"深夜" },
};
#define SUN_PRESET_NUM ((int)(sizeof(SUN_PRESETS) / sizeof(SUN_PRESETS[0])))

// ===========================================================================
// 云量档位（用户可调）：0=晴空 … 5=满天
// 注意 cloudMap0 层自带 0.5 覆盖率，故 0 档不是完全无云，而是低云稀薄、蓝天太阳为主
// ===========================================================================
// 档位按新增益后的实测响应标定（云占比 ≈ 25/45/65/85/97%），
// 晴空→疏云→半云→多云→阴天→满云均匀递进；最高档止于 0.395（暗区仅 ~0.1%）。
static const float CLOUD_COVERAGE_LEVELS[] = {
    0.000f, 0.325f, 0.340f, 0.353f, 0.368f, 0.395f
};
#define CLOUD_COVERAGE_LEVEL_NUM ((int)(sizeof(CLOUD_COVERAGE_LEVELS) / sizeof(CLOUD_COVERAGE_LEVELS[0])))
// 默认档位（默认偏少，避免挡住蓝天与太阳）
#define CLOUD_COVERAGE_DEFAULT_LEVEL (2)
// 云亮度（观感）＝介质光学参数的物理联动，而非事后乘颜色：
//   反照率 albedo          ↑：散射/吸收比上升，反射更亮、不改变遮挡范围
//   直射辐照 sunScale      ↑：入射照度上升（辐射度线性叠加）
//   多重散射出射/损耗      ↑/↓：暗部来自多次散射的回光增强
//   环境反射 ambientScale  ↑：天空环境下行光对云的补光增强
// 亮度 b∈[0.5,2.0]，内部归一化为 t∈[0,1]
#define CLOUD_BRIGHTNESS_DEFAULT (1.4f)



// 太阳自动运动：从高度角 -18°、方位角正东起步，每渲染帧高度角 +1°，
// 到天顶 90° 后反向下降，到 -18° 方位角正西后折返，如此往返循环。
// （帧驱动：不依赖墙钟，保证"每帧 1° 高度角"）
// 由太阳高度角（度）推导太阳色温/辐照（供独立应用与天象仪集成共用）。
// 任意高度角均有定义：正午类 → 低空橙 → 日落地平 → 暮光（民用/航海/天文）→ 完整黑夜。
// -10° 处的取值与历史版本一致（避免已有观感变化）。
void ui_cloud_sun_color(float elev_deg, float *or_, float *og, float *ob, float *ointen) {
    Cv3 col;
    float inten;
    if (elev_deg >= 15.0f) {
        col = v3(1.0f, 0.93f, 0.82f);
        inten = 48.0f;
    } else if (elev_deg >= 0.0f) {
        // 低空橙 → 正午白（0°..15°）
        float t = elev_deg / 15.0f;
        col = v3lerp(v3(1.0f, 0.45f, 0.20f), v3(1.0f, 0.93f, 0.82f), t);
        inten = mixf(36.0f, 48.0f, t);
    } else if (elev_deg >= -18.0f) {
        // 暮光段（-18°..0°）：民用(-6°)/航海(-12°)/天文(-18°) 暮光连续过渡
        // 颜色从日落红过渡到深蓝夜色，辐照继续衰减。
        //   -10° 锚点：色 (0.50,0.55,0.80)、强度 10（历史一致）
        float t;   // 0=段底（更夜） 1=段顶（更日）
        if (elev_deg >= -10.0f) {
            t = (elev_deg + 10.0f) / 10.0f;                       // -10°..0°
            col = v3lerp(v3(0.50f, 0.55f, 0.80f), v3(1.0f, 0.45f, 0.20f), t);
            inten = mixf(10.0f, 36.0f, t);
        } else {
            t = (elev_deg + 18.0f) / 8.0f;                       // -18°..-10°
            col = v3lerp(v3(0.20f, 0.26f, 0.45f), v3(0.50f, 0.55f, 0.80f), t);
            inten = mixf(3.0f, 10.0f, t);
        }
    } else {
        // 完整黑夜（天文夜之后，如 -35° 午夜的江宁）：保持微弱月夜底色
        col = v3(0.20f, 0.26f, 0.45f);
        inten = 3.0f;
    }
    *or_ = col.x; *og = col.y; *ob = col.z; *ointen = inten;
}

static void cloud_sun_auto_tick(void) {
    if (!s_ui.sun_auto) return;
    float alt = s_ui.sun_alt + (float)s_ui.sun_dir;   // 每帧 ±1° 高度角
    if (s_ui.sun_dir > 0) {
        if (alt >= 90.0f) { alt = 90.0f; s_ui.sun_dir = -1; }
    } else {
        if (alt <= -18.0f) { alt = -18.0f; s_ui.sun_dir = 1; }
    }
    s_ui.sun_alt = alt;

    float q = (alt + 18.0f) / 108.0f;                 // 行程进度 0..1（东→西）
    float elevDeg = alt;
    float azimDeg = 90.0f - 180.0f * q;               // 正东(+90°) → 正西(-90°)

    float cr, cg, cb, ci;
    ui_cloud_sun_color(elevDeg, &cr, &cg, &cb, &ci);
    float e = elevDeg * kPI / 180.0f;
    float a = azimDeg * kPI / 180.0f;
    s_scene.sunDirection = v3norm(v3(cosf(e) * sinf(a), sinf(e), cosf(e) * cosf(a)));
    s_scene.sunColor = v3(cr, cg, cb);
    s_scene.sunIntensity = ci;
    s_ui.sun_dirty = 1;
}

static void cloud_apply_sun_preset(int idx) {
    const SunPreset *p = &SUN_PRESETS[idx];
    float elev = (float)p->elev_deg * kPI / 180.0f;
    float azim = (float)p->azim_deg * kPI / 180.0f;
    s_scene.sunDirection = v3norm(v3(cosf(elev) * sinf(azim), sinf(elev), cosf(elev) * cosf(azim)));
    s_scene.sunColor = p->color;
    s_scene.sunIntensity = p->intensity;
    s_ui.sun_dirty = 1;
}

static int cloud_apply_coverage_level(int level) {
    if (level < 0) level = CLOUD_COVERAGE_LEVEL_NUM - 1;
    if (level >= CLOUD_COVERAGE_LEVEL_NUM) level = 0;
    s_ui.coverage_level = level;
    s_scene.cloudCoverage = CLOUD_COVERAGE_LEVELS[level];
    return level;
}

int ui_cloud_coverage_level_num(void) {
    return CLOUD_COVERAGE_LEVEL_NUM;
}

float ui_cloud_coverage_for_level(int level) {
    if (level < 0) level = 0;
    if (level >= CLOUD_COVERAGE_LEVEL_NUM) level = CLOUD_COVERAGE_LEVEL_NUM - 1;
    return CLOUD_COVERAGE_LEVELS[level];
}

static void cloud_apply_brightness(float b) {
    b = clampf(b, 0.5f, 2.0f);
    s_ui.brightness = b;
    // t∈[0,1]：亮度 b=0.5→t=0（最暗），b=2.0→t=1（最亮）
    float t = saturatef((b - 0.5f) / 1.5f);

    // 反照率：仅影响散射/消光之比（≤1 保持能量守恒），不改变云的遮挡
    float albedo = 0.72f + 0.26f * t;
    s_scene.cloudAlbedo = v3(albedo, albedo, albedo);

    // 太阳直射辐照（入射光强线性叠加）
    s_scene.cloudShadingSunLightScale = 0.85f + 0.75f * t;

    // 多重散射：出射（scatter）增强 / 每次交互的光学损耗（extinction）下降
    s_scene.cloudMultiScatterScatter    = 0.65f + 0.85f * t;
    s_scene.cloudMultiScatterExtinction = 0.26f - 0.13f * t;

    // 环境反射补光（天空下行光）
    s_scene.cloudAmbientScale = 0.5f + 1.0f * t;
}

// 每帧一次：太阳或大气变化时重算 LUT（与相机无关，故按需缓存）
static void cloud_ensure_luts(void) {
    if (!s_ui.sun_dirty) return;
    s_ui.sun_dirty = 0;
    // 注意：透射率 LUT 计算需要 s_msLut 已就绪？flower 中透射率 LUT 与多重散射 LUT 相互独立——
    // 透射率只依赖介质消光（与多重散射无关），多重散射 LUT 内部会采样透射率 LUT。
    // 因此先算透射率，再算多重散射 LUT。
    cloud_compute_transmittance_lut(&s_scene);
    cloud_compute_multiscatter_lut(&s_scene);
}

// 色调映射 → 8bit 行缓冲
static void cloud_write_pixel(int32_t x, Cv3 hdr, uint8_t *rgb_line) {
    Cv3 t = cloudAcesFilm(hdr);
    t = v3(powf(t.x, 1.0f/2.2f), powf(t.y, 1.0f/2.2f), powf(t.z, 1.0f/2.2f)); // sRGB 伽马
    rgb_line[x * 3 + 0] = (uint8_t)(clampf(t.x, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgb_line[x * 3 + 1] = (uint8_t)(clampf(t.y, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgb_line[x * 3 + 2] = (uint8_t)(clampf(t.z, 0.0f, 1.0f) * 255.0f + 0.5f);
}

int32_t ui_cloud_init(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    // LUT 缓冲一次性申请（PSRAM，见 s_transLut/s_msLut 处注释）；失败则回主菜单
    if (!s_transLut || !s_msLut) {
        s_transLut = (Cv3 *)platform_malloc(sizeof(Cv3) * TRANS_LUT_W * TRANS_LUT_H);
        s_msLut    = (Cv3 *)platform_malloc(sizeof(Cv3) * MS_LUT_W * MS_LUT_H);
        if (!s_transLut || !s_msLut) {
            if (s_transLut) { free(s_transLut); s_transLut = NULL; }
            if (s_msLut)    { free(s_msLut);    s_msLut = NULL; }
            global_state->STATE = STATE_MAIN_MENU;
            return -1;
        }
    }
    cloud_generate_noise_textures();
    if (!s_ui.first_frame) {
        cloud_default_scene(&s_scene);
        s_ui.yaw = 0.0f;
        s_ui.pitch = 25.0f * kPI / 180.0f;
        s_ui.sun_preset = 0;
        s_ui.sun_dirty = 1;
        s_ui.app_time = 0.0f;
        s_ui.last_ts = global_state->timestamp;
        s_ui.frame_index = 0;
        s_ui.first_frame = 0;
        cloud_apply_coverage_level(CLOUD_COVERAGE_DEFAULT_LEVEL);
        cloud_apply_brightness(CLOUD_BRIGHTNESS_DEFAULT);
    }
    s_ui.sun_alt = -18.0f;  // 自动运动起点：高度角 -18°、方位角正东
    s_ui.sun_dir = 1;
    s_ui.layer_mask = CLOUD_LAYER0 | CLOUD_LAYER1 | CLOUD_LAYER2; // 默认所有云
    s_ui.layer_idx = 0;
    // 投影：默认透视；两套档位各自独立
    s_ui.proj = CLOUD_PROJ_PERSP;
    s_ui.fov_idx[CLOUD_PROJ_PERSP] = FOV_DEFAULT_IDX;
    s_ui.fov_idx[CLOUD_PROJ_FISHEYE] = FISHEYE_FOV_DEFAULT_IDX;
    s_ui.fov_deg[CLOUD_PROJ_PERSP] = (int)FOV_DEGREES[FOV_DEFAULT_IDX];
    s_ui.fov_deg[CLOUD_PROJ_FISHEYE] = (int)FISHEYE_FOV_DEGREES[FISHEYE_FOV_DEFAULT_IDX];
    s_ui.fov_cur = s_ui.fov_deg[s_ui.proj];
    s_ui.last_ts = global_state->timestamp;
    cloud_apply_sun_preset(s_ui.sun_preset);
    s_ui.sun_dirty = 1;
    cloud_ensure_luts();
    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);
    return 0;
}

int32_t ui_cloud_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 返回
    if (((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc)) {
        global_state->STATE = STATE_MAIN_MENU;
        return 0;
    }
    // 视角
    if (key_event->key_code == NANO_KEY_left || key_event->key_code == NANO_KEY_right ||
        key_event->key_code == NANO_KEY_up || key_event->key_code == NANO_KEY_down) {
        float speed = (key_event->key_edge == -2) ? 6.0f : 1.2f; // 长按重复加速
        if (key_event->key_code == NANO_KEY_left)  s_ui.yaw   -= 0.06f * speed;
        if (key_event->key_code == NANO_KEY_right) s_ui.yaw   += 0.06f * speed;
        if (key_event->key_code == NANO_KEY_up)    s_ui.pitch += 0.04f * speed;
        if (key_event->key_code == NANO_KEY_down)  s_ui.pitch -= 0.04f * speed;
        s_ui.pitch = clampf(s_ui.pitch, -85.0f * kPI / 180.0f, 85.0f * kPI / 180.0f);
    }
    // 太阳自动运动 开始/暂停（回车切换）
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_enter) {
        s_ui.sun_auto = !s_ui.sun_auto;
    }
    // 手动太阳预设切换（逻辑 NANO_KEY_9，tty 物理键 '3'）；切换即暂停自动运动
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_9) {
        s_ui.sun_auto = 0;
        s_ui.sun_preset = (s_ui.sun_preset + 1) % SUN_PRESET_NUM;
        cloud_apply_sun_preset(s_ui.sun_preset);
    }
    // 投影算法循环（7 键）：透视 ↔ 等距鱼眼（各自保留 FOV 档位）
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_7) {
        s_ui.proj = (s_ui.proj == CLOUD_PROJ_PERSP) ? CLOUD_PROJ_FISHEYE : CLOUD_PROJ_PERSP;
        s_ui.fov_cur = s_ui.fov_deg[s_ui.proj];
    }
    // 视场角循环（3 键）：作用于当前投影的档位表
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_3) {
        int n = (s_ui.proj == CLOUD_PROJ_PERSP) ? FOV_DEGREES_NUM : FISHEYE_FOV_DEGREES_NUM;
        s_ui.fov_idx[s_ui.proj] = (s_ui.fov_idx[s_ui.proj] + 1) % n;
        s_ui.fov_deg[s_ui.proj] = (s_ui.proj == CLOUD_PROJ_PERSP)
            ? (int)FOV_DEGREES[s_ui.fov_idx[s_ui.proj]]
            : (int)FISHEYE_FOV_DEGREES[s_ui.fov_idx[s_ui.proj]];
        s_ui.fov_cur = s_ui.fov_deg[s_ui.proj];
    }
    // 云层种类循环（1 键）：所有 → 仅低层 → 仅中层 → 低+中 → 无云
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        s_ui.layer_idx = (s_ui.layer_idx + 1) % 5;
        switch (s_ui.layer_idx) {
            case 0: s_ui.layer_mask = CLOUD_LAYER0 | CLOUD_LAYER1 | CLOUD_LAYER2; break; // 所有
            case 1: s_ui.layer_mask = CLOUD_LAYER0; break;                               // 仅低层
            case 2: s_ui.layer_mask = CLOUD_LAYER1; break;                               // 仅中层
            case 3: s_ui.layer_mask = CLOUD_LAYER0 | CLOUD_LAYER1; break;                // 低+中
            default: s_ui.layer_mask = 0; break;                                         // 无云
        }
    }
    // 云量（档位）与云亮度（消光观感）
    if ((key_event->key_edge == -1 || key_event->key_edge == -2)) {
        switch (key_event->key_code) {
            case NANO_KEY_4: cloud_apply_coverage_level(s_ui.coverage_level - 1); break; // 云量减档
            case NANO_KEY_6: cloud_apply_coverage_level(s_ui.coverage_level + 1); break; // 云量加档
            case NANO_KEY_2: cloud_apply_brightness(s_ui.brightness + 0.15f); break;     // 云更亮
            case NANO_KEY_8: cloud_apply_brightness(s_ui.brightness - 0.15f); break;     // 云更暗
            case NANO_KEY_5: cloud_apply_coverage_level(CLOUD_COVERAGE_DEFAULT_LEVEL);   // 复位
                             cloud_apply_brightness(CLOUD_BRIGHTNESS_DEFAULT);
                             s_ui.sun_auto = 0;
                             cloud_apply_sun_preset(s_ui.sun_preset);
                             s_ui.layer_mask = CLOUD_LAYER0 | CLOUD_LAYER1 | CLOUD_LAYER2;
                             s_ui.layer_idx = 0;
                             s_ui.proj = CLOUD_PROJ_PERSP;
                             s_ui.fov_idx[CLOUD_PROJ_PERSP] = FOV_DEFAULT_IDX;
                             s_ui.fov_deg[CLOUD_PROJ_PERSP] = (int)FOV_DEGREES[FOV_DEFAULT_IDX];
                             s_ui.fov_idx[CLOUD_PROJ_FISHEYE] = FISHEYE_FOV_DEFAULT_IDX;
                             s_ui.fov_deg[CLOUD_PROJ_FISHEYE] = (int)FISHEYE_FOV_DEGREES[FISHEYE_FOV_DEFAULT_IDX];
                             s_ui.fov_cur = s_ui.fov_deg[s_ui.proj];
                             break;
            default: break;
        }
    }
    return 0;
}

// 内部渲染缓冲（半分辨率，动态分配）+ 全屏输出缓冲
static uint8_t *s_scene_img = NULL;
static size_t  s_scene_img_bytes = 0;
static uint8_t *s_fb = NULL;      // 全屏（整机帧缓冲尺寸）
static size_t  s_fb_bytes = 0;
static int s_out_fw = 0, s_out_fh = 0; // 最近一帧的全屏尺寸（供 flush 使用）

static void cloud_ensure_buffers(int iw, int ih, int fw, int fh) {
    size_t scene_need = (size_t)iw * ih * 3;
    if (scene_need > s_scene_img_bytes) {
        if (s_scene_img) free(s_scene_img);
        s_scene_img = (uint8_t *)malloc(scene_need);
        s_scene_img_bytes = scene_need;
    }
    size_t fb_need = (size_t)fw * fh * 3;
    if (fb_need > s_fb_bytes) {
        if (s_fb) free(s_fb);
        s_fb = (uint8_t *)malloc(fb_need);
        s_fb_bytes = fb_need;
    }
}

// 向 Wimg x Himg 的 RGB888 缓冲 stamp 一行文本（二值点阵字模，位图布局同 gfx_draw_char）
static void cloud_stamp_text(Nano_GFX *gfx, uint8_t *img, int Wimg, int Himg,
                             const wchar_t *str,
                             int32_t x, int32_t y, uint8_t r, uint8_t g, uint8_t b) {
    while (*str) {
        uint8_t fw = 12, fh = 12;
        const uint8_t *glyph = gfx_get_glyph(gfx, (uint32_t)*str, &fw, &fh);
        if (!glyph) glyph = gfx_get_glyph(gfx, 12307, &fw, &fh);
        if (!glyph) { str++; continue; }
        int32_t row_bytes = (fh + 7) / 8;
        for (int32_t c = 0; c < (int32_t)fw; c++) {
            for (int32_t j = 0; j < row_bytes; j++) {
                uint8_t byte = glyph[j * fw + c];
                int32_t bits = (j == row_bytes - 1) ? (8 - ((8 * row_bytes) % fh)) : 8;
                for (int32_t bb = 0; bb < bits; bb++) {
                    if ((byte >> bb) & 1) {
                        int32_t px = x + c, py = y + j * 8 + bb;
                        if (px >= 0 && px < Wimg && py >= 0 && py < Himg) {
                            int32_t i = (py * Wimg + px) * 3;
                            img[i] = r; img[i + 1] = g; img[i + 2] = b;
                        }
                    }
                }
            }
        }
        x += fw;
        str++;
    }
}

// 双线性放大 s_scene_img(iw,ih) → s_fb(fw,fh)
static void cloud_upscale(int iw, int ih, int fw, int fh) {
    const uint8_t *s = s_scene_img;
    uint8_t *d = s_fb;
    for (int Y = 0; Y < fh; Y++) {
        float sy = ((float)Y + 0.5f) * (float)ih / (float)fh - 0.5f;
        int y0 = (int)floorf(sy);
        y0 = (y0 < 0) ? 0 : (y0 > ih - 2 ? ih - 2 : y0);
        float fy = sy - (float)y0;
        if (fy < 0.0f) fy = 0.0f;
        const uint8_t *row0 = s + ((size_t)y0 * iw) * 3;
        const uint8_t *row1 = row0 + (size_t)iw * 3;
        uint8_t *out = d + ((size_t)Y * fw) * 3;
        for (int X = 0; X < fw; X++) {
            float sx = ((float)X + 0.5f) * (float)iw / (float)fw - 0.5f;
            int x0 = (int)floorf(sx);
            x0 = (x0 < 0) ? 0 : (x0 > iw - 2 ? iw - 2 : x0);
            float fx = sx - (float)x0;
            if (fx < 0.0f) fx = 0.0f;
            int a = x0 * 3, b2 = (x0 + 1) * 3;
            for (int c = 0; c < 3; c++) {
                float p0 = mixf(row0[a + c], row0[b2 + c], fx);
                float p1 = mixf(row1[a + c], row1[b2 + c], fx);
                out[c] = (uint8_t)(mixf(p0, p1, fy) + 0.5f);
            }
            out += 3;
        }
    }
}

static void cloud_setup_camera(float roll_rad, Cv3 *camPos, Cv3 *forward, Cv3 *right, Cv3 *camUp) {
    *camPos = v3(0, s_scene.bottomRadius + 2.0f, 0); // 地面 2km
    *forward = v3norm(v3(cosf(s_ui.pitch) * sinf(s_ui.yaw), sinf(s_ui.pitch),
                         cosf(s_ui.pitch) * cosf(s_ui.yaw)));
    Cv3 worldUp = v3(0, 1, 0);
    Cv3 r = v3cross(*forward, worldUp);
    if (v3len(r) < 1e-4f) r = v3(1, 0, 0);
    *right = v3norm(r);
    *camUp = v3cross(*right, *forward);
    *camUp = v3norm(*camUp);
    // 绕视线轴滚转（与天象仪 fisheye_project 的 roll 约定一致：正值为顺时针）
    if (roll_rad != 0.0f) {
        float cr = cosf(roll_rad), sr = sinf(roll_rad);
        Cv3 rr = v3add(v3mul(*right, cr), v3mul(*camUp, -sr));
        Cv3 uu = v3add(v3mul(*right, sr), v3mul(*camUp, cr));
        *right = rr;
        *camUp = uu;
    }
}

// ---------------------------------------------------------------------------
// 核心渲染（内部缓冲 → 半分辨率 ray-march → 双线性放大到 s_fb）
// 所有场景/相机/云参数均来自外部 params（供独立应用与天象仪集成共用）。
// 不负责 HUD 叠加与显存输出，由调用方先 flush 再在其上叠加绘制。
// ---------------------------------------------------------------------------
static void cloud_core_render(Nano_GFX *gfx, const UiCloud_Render_Params *p) {
    const int FW = (int)gfx->width, FH = (int)gfx->height;

    // 内部渲染 = 屏幕一半（上限 CLOUD_MAX_*）
    int W = FW / 2, H = FH / 2;
    if (W < 8) W = 8;
    if (W > CLOUD_MAX_W) W = CLOUD_MAX_W;
    if (H < 8) H = 8;
    if (H > CLOUD_MAX_H) H = CLOUD_MAX_H;
    cloud_ensure_buffers(W, H, FW, FH);
    s_out_fw = FW;
    s_out_fh = FH;

    // 防御：LUT 未就绪（PSRAM 申请失败）时跳过本帧渲染（调用方负责退出状态）
    if (!s_transLut || !s_msLut) return;

    cloud_ensure_luts();

    Cv3 camPos, forward, right, camUp;
    cloud_setup_camera(p->roll_rad, &camPos, &forward, &right, &camUp);

    // 地面上涌光：天顶方向的天空亮度（除地面回弹的纯大气散射），加权漫反射等效
    {
        Cv3 upDir = v3(0, 1, 0);
        SingleScatteringResult zen = integrateScatteredLuminance(camPos, upDir, s_scene.sunDirection,
            &s_scene, 0, 1, 0, 1, 9e6f, 14);
        // 地面上涌光：仅用天顶亮度×反照率 会骗低云底亮度（云底可视环境光≈地面漫反射+大气回光）。
        // 抬高尺度：上行辐照≈天顶亮度×(1+反照率) 的半球等效，再随云遮蔽淡出。
        s_groundUpwelling = v3mul(v3mul3(zen.scatteredLight,
            v3add(v3(1.0f, 1.0f, 1.0f), s_scene.groundAlbedo)), 4.0f);
    }

    // 透视投影：fov_deg 为纵向半角；tanV/tanH 用于屏幕射线与太阳反投影
    float tanV = tanf((float)p->fov_deg * kPI / 180.0f);
    float tanH = tanV * (float)W / (float)H;
    // 鱼眼投影：屏幕对角方向映射到 fov_deg（最大半角）
    const float fishRmax = sqrtf((float)(W * W + H * H) / (float)H / (float)H); // sqrt((W/H)^2+1)
    const float fishFovRad = (float)p->fov_deg * kPI / 180.0f;

    // 太阳镜头光晕（每帧一次）：太阳在相机前方时投影到屏幕 NDC，并求其被云遮挡的透射率
    s_ui.sun_visible = 0;
    if (p->enable_sun_lens) {
        Cv3 sc = v3(v3dot(s_scene.sunDirection, right),
                    v3dot(s_scene.sunDirection, camUp),
                    v3dot(s_scene.sunDirection, forward));
        if (sc.z > 0.02f) {
            if (p->proj == CLOUD_PROJ_FISHEYE) {
                // 等距鱼眼反投影：与正向构造互为逆映射
                float th = acosf(clampf(sc.z, -1.0f, 1.0f));
                float rn = th / fishFovRad;
                float r = rn * fishRmax;
                float ph = atan2f(sc.y, sc.x);
                s_ui.sun_u = r * cosf(ph) / (float)W * (float)H; // u NDC(带 aspect 还原)
                s_ui.sun_v = r * sinf(ph);
            } else {
                s_ui.sun_u = sc.x / (sc.z * tanH);
                s_ui.sun_v = sc.y / (sc.z * tanV);
            }
            // 沿太阳方向的云透射率（等价 lens_visible.glsl 的 cloudColor.a）
            CloudPixel sunc = cloudPixelCompute(camPos, s_scene.sunDirection, p->app_time_sec, 0.5f, &s_scene);
            s_ui.sun_vis = sunc.transmittance;
            s_ui.sun_visible = 1;
        }
        else {
            s_ui.sun_vis = 0.0f;
        }
    }

    // 半分辨率逐像素 ray-march 渲染
    #pragma omp parallel for schedule(static) if (H >= 8)
    for (int y = 0; y < H; y++) {
        uint8_t rowbuf[CLOUD_MAX_W * 3];
        float vv = 1.0f - ((float)y + 0.5f) / (float)H * 2.0f; // 上正
        const int32_t fj = s_ui.frame_index % 9973;
        for (int x = 0; x < W; x++) {
            float uu = ((float)x + 0.5f) / (float)W * 2.0f - 1.0f; // 左负右正

            // 屏幕射线：透视（pinhole）或等距鱼眼投影
            Cv3 rayDir;
            if (p->proj == CLOUD_PROJ_FISHEYE) {
                float fx = uu * (float)W / (float)H;     // NDC 带 aspect
                float fy = vv;
                float r = sqrtf(fx * fx + fy * fy);
                float rn = r / fishRmax;                  // 对角=1
                float th = rn * fishFovRad;               // 等距角
                float ph = atan2f(fy, fx);
                rayDir = v3add(v3mul(forward, cosf(th)),
                               v3mul(v3add(v3mul(right, cosf(ph)), v3mul(camUp, sinf(ph))), sinf(th)));
            } else {
                rayDir = v3add(v3add(forward, v3mul(right, uu * tanH)), v3mul(camUp, vv * tanV));
            }
            rayDir = v3norm(rayDir);

            // per-pixel 蓝噪抖动（对应 flower 网格化 blue noise 种子旋转）
            float njx = (float)x + (float)fj * 0.695f * 47.0f;
            float njy = (float)y + (float)fj * 0.695f * 17.0f;
            float jitter = fmodf(52.9829189f * fmodf(njx * 0.06711056f + njy * 0.00583715f, 1.0f), 1.0f);

            // 天空
            SingleScatteringResult sky = integrateScatteredLuminance(camPos, rayDir, s_scene.sunDirection,
                &s_scene, 1, 1, 0, 1, 9e6f, 14);

            // 云
            CloudPixel cloud = cloudPixelCompute(camPos, rayDir, p->app_time_sec, jitter, &s_scene);

            // 合成：天空透过后方云层 + 云本身散射
            // 亮度已由介质光学参数（albedo/辐照/多重散射/环境）在着色阶段体现，不再后乘
            Cv3 final = v3add(v3mul(sky.scatteredLight, cloud.transmittance), cloud.color);

            // 太阳镜头光晕（太阳/星芒/光斑，HDR 叠加，色调映射前）
            if (s_ui.sun_visible) {
                Cv3 lensAdd = cloud_lens_hdr(uu, vv, s_ui.sun_u, s_ui.sun_v, s_ui.sun_vis, &s_scene);
                final = v3add(final, lensAdd);
            }

            cloud_write_pixel(x, final, rowbuf);
        }
        memcpy(&s_scene_img[((size_t)y * W) * 3], rowbuf, W * 3);
    }

    // 放大到整屏
    cloud_upscale(W, H, FW, FH);
}

// 将外部参数应用到内部场景（光源/相机/云参数），并执行一次核心渲染。
// 独立应用与天象仪集成共用此入口，保证两端的体积云/大气渲染行为一致。
void ui_cloud_render_core(Nano_GFX *gfx, const UiCloud_Render_Params *p) {
    cloud_generate_noise_textures();
    static int s_scene_ready = 0;
    if (!s_scene_ready) {
        cloud_default_scene(&s_scene);
        s_scene_ready = 1;
    }

    // 光源由调用方给出（天象仪 → where_is_the_sun；独立应用 → 预设/自动运动）
    float dl = sqrtf(p->sun_dx * p->sun_dx + p->sun_dy * p->sun_dy + p->sun_dz * p->sun_dz);
    if (dl < 1e-6f) {
        s_scene.sunDirection = v3(0, 1, 0);
    } else {
        s_scene.sunDirection = v3(p->sun_dx / dl, p->sun_dy / dl, p->sun_dz / dl);
    }
    s_scene.sunColor = v3(p->sun_r, p->sun_g, p->sun_b);
    s_scene.sunIntensity = p->sun_intensity;
    s_ui.sun_dirty = 1;

    // 云参数（云量/云层种类/亮度沿用独立应用已有的控制曲线）
    s_scene.cloudCoverage = clampf(p->coverage, 0.0f, 1.0f);
    s_ui.layer_mask = p->layer_mask;
    cloud_apply_brightness(p->brightness);

    // 相机与投影（天象仪视角 → 云渲染 yaw/pitch/roll；独立应用直接传入自身状态）
    s_ui.yaw = p->yaw_rad;
    s_ui.pitch = clampf(p->pitch_rad, -85.0f * kPI / 180.0f, 85.0f * kPI / 180.0f);
    s_ui.roll = p->roll_rad;
    s_ui.proj = (p->proj == CLOUD_PROJ_FISHEYE) ? CLOUD_PROJ_FISHEYE : CLOUD_PROJ_PERSP;
    s_ui.fov_cur = p->fov_deg;
    if (s_ui.fov_cur < 1) s_ui.fov_cur = 1;
    if (s_ui.fov_cur > 170) s_ui.fov_cur = 170;
    s_ui.app_time = p->app_time_sec;
    s_ui.frame_index++;

    cloud_core_render(gfx, p);
}

// 将最近一帧的内部渲染结果输出到 gfx 显存（整屏，按 gfx->width stride）。
// 天象仪集成：在调用 render_sky 的其余叠加绘制之前先 flush，把体积云/大气
// 作为背景落入帧缓冲，再在其上绘制太阳/月亮/恒星/坐标圈/地景等天象仪要素。
void ui_cloud_flush(Nano_GFX *gfx) {
    const int FW = s_out_fw, FH = s_out_fh;
    if (s_fb == NULL || FW <= 0 || FH <= 0) return;
    if ((int)gfx->width != FW) { // 尺寸不匹配则按当前 gfx 最接近的尺寸（保守起见直接整屏逐像素）
        if (gfx->color_mode == GFX_COLOR_MODE_RGB888 && gfx->frame_buffer_rgb888 != NULL) {
            uint8_t *fb = gfx->frame_buffer_rgb888;
            for (int32_t y = 0; y < (int32_t)gfx->height && y < FH; y++) {
                for (int32_t x = 0; x < (int32_t)gfx->width && x < FW; x++) {
                    int32_t i = (y * FW + x) * 3;
                    int32_t o = ((size_t)y * gfx->width + x) * 3;
                    fb[o] = s_fb[i]; fb[o + 1] = s_fb[i + 1]; fb[o + 2] = s_fb[i + 2];
                }
            }
        } else {
            for (int32_t y = 0; y < FH; y++) {
                for (int32_t x = 0; x < FW; x++) {
                    int32_t i = (y * FW + x) * 3;
                    gfx_set_pixel(gfx, (uint32_t)x, (uint32_t)y, s_fb[i], s_fb[i + 1], s_fb[i + 2]);
                }
            }
        }
        return;
    }
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888 && gfx->frame_buffer_rgb888 != NULL) {
        uint8_t *fb = gfx->frame_buffer_rgb888;
        memcpy(fb, s_fb, (size_t)FW * FH * 3);
    } else {
        for (int32_t y = 0; y < FH; y++) {
            for (int32_t x = 0; x < FW; x++) {
                int32_t i = (y * FW + x) * 3;
                gfx_set_pixel(gfx, (uint32_t)x, (uint32_t)y, s_fb[i], s_fb[i + 1], s_fb[i + 2]);
            }
        }
    }
}

int32_t ui_cloud_render_frame(Key_Event *key_event, Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    const int FW = (int)gfx->width, FH = (int)gfx->height;

    // 时间推进（秒）+ 帧计数
    // 注意：两帧时间戳差值须先用 64 位整数相减，再转 float；
    // 直接 (float)timestamp 会丢失 epoch 毫秒的低位精度（约 0.1s），导致 dt=0。
    float dt = (float)((int64_t)global_state->timestamp - (int64_t)s_ui.last_ts) / 1000.0f;
    if (dt < 0.0f) dt = 0.0f;
    if (dt > 0.06f) dt = 0.06f;
    s_ui.last_ts = global_state->timestamp;
    s_ui.app_time += dt;

    // 太阳自动运动推进（每帧 +1° 高度角；使 LUT/光晕/场景随太阳方向一致更新）
    cloud_sun_auto_tick();

    // 组装渲染参数（从内部状态捕获场景/相机/云设置）
    UiCloud_Render_Params p;
    memset(&p, 0, sizeof(p));
    p.yaw_rad = s_ui.yaw;
    p.pitch_rad = s_ui.pitch;
    p.roll_rad = s_ui.roll;
    p.proj = s_ui.proj;
    p.fov_deg = s_ui.fov_cur;
    p.app_time_sec = s_ui.app_time;
    p.sun_dx = s_scene.sunDirection.x;
    p.sun_dy = s_scene.sunDirection.y;
    p.sun_dz = s_scene.sunDirection.z;
    p.sun_r = s_scene.sunColor.x;
    p.sun_g = s_scene.sunColor.y;
    p.sun_b = s_scene.sunColor.z;
    p.sun_intensity = s_scene.sunIntensity;
    p.coverage = s_scene.cloudCoverage;
    p.layer_mask = s_ui.layer_mask;
    p.brightness = s_ui.brightness;
    p.enable_sun_lens = 1;

    ui_cloud_render_core(gfx, &p);

    // HUD（stamp 进整屏缓冲，缩放的文字不再模糊）
    const SunPreset *p2 = &SUN_PRESETS[s_ui.sun_preset];
    static const wchar_t *LAYER_NAMES[5] = { L"所有云", L"仅低层", L"仅中层", L"低+中", L"无云" };
    float elevNow = asinf(saturatef(s_scene.sunDirection.y)) * 180.0f / (float)kPI;
    wchar_t line[64];
    if (s_ui.sun_auto) {
        swprintf(line, 64, L"云量%d/%d 亮%.2f %ls 自动%+d",
                 s_ui.coverage_level + 1, CLOUD_COVERAGE_LEVEL_NUM,
                 s_ui.brightness, LAYER_NAMES[s_ui.layer_idx], (int)lrintf(elevNow));
    } else {
        swprintf(line, 64, L"云量%d/%d 亮%.2f %ls %ls%+d",
                 s_ui.coverage_level + 1, CLOUD_COVERAGE_LEVEL_NUM,
                 s_ui.brightness, LAYER_NAMES[s_ui.layer_idx], p2->name, (int)lrintf(elevNow));
    }
    cloud_stamp_text(gfx, s_fb, FW, FH, line, 2, 2, 235, 235, 240);
    {
        wchar_t h2[48];
        swprintf(h2, 48, L"%ls%d 1云型 3视场 9太阳",
                 (s_ui.proj == CLOUD_PROJ_FISHEYE) ? L"鱼眼" : L"透视", s_ui.fov_cur);
        cloud_stamp_text(gfx, s_fb, FW, FH, h2, 2, 15, 175, 180, 195);
    }

    // 输出到显存（整屏，按 gfx->width stride）
    ui_cloud_flush(gfx);
    gfx_refresh(gfx);
    return 0;
}
