// ===============================================================================
// 体积云与天空仿真（CPU 实时 ray-march）
//
// 算法核心完全参照 flower（/mnt/d/Desktop/GitRepos/flower）Vulkan 引擎实现：
//   - 云形状：cloud_noise_common.glsl 的 tileable Perlin-Worley / Worley FBM 噪声，
//     cloud_render_common.glsl 的 cloudMap0/1/2 三层高度分层，外加浓积云层 cloudMapCu
//     （移植自 linglong.html：云底 ~1.0 km 的平底积云，塔顶由疙瘩噪声决定，0.8-4.1 km）
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
//   * 渲染分辨率 = 屏幕全分辨率（320x240 内渲）；历史上曾用半分辨率（160x120）双线性放大
//     以降低算力，PocketTerm35（RPI5）算力充足改回全分辨率（半分辨率再经显示 2 倍放大
//     后颗粒/方块感明显）
// ===============================================================================

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include <omp.h>

#include "ui_cloud.h"
#include "hal_key.h"
#include "platform.h"
#include "celestial_night.h"

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
static inline Cv3 v3exp(Cv3 a);              // 快速 exp3（定义见下方 A5 段）
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

// ---- 云壳与层带（物理高度，km）----
// 浓积云 0.8–4.1 km、高积云 3.0–6.2 km、层积云 6.2–9.4 km、卷层云 9.4–11.0 km
// （与 linglong.html 的 CLOUD_SKY_* 常量逐值一致）
#define CLOUD_SHELL_BOTTOM_KM   (0.80f)
#define CLOUD_SHELL_TOP_KM      (11.00f)
#define CLOUD_SHELL_THICK_KM    (CLOUD_SHELL_TOP_KM - CLOUD_SHELL_BOTTOM_KM)

static inline float smoothstepf(float e0, float e1, float x) {
    float t = saturatef((x - e0) / (e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}

// ---------------------------------------------------------------------------
// 快速超越函数（性能专项 A5）
//
// 本内核每像素要调上百次 expf（大气积分每样本 5 次）与 3 次 powf（sRGB 伽马）。
// 库函数 expf 走 fdlibm 的通用路径（含范围归约与异常分支），在 WASM/ARM 上是
// 逐像素路径里最大的单类开销。这里换成「拆阶码 + 6 次多项式」：
//   · 整数部分由位拼接精确给出（无查表、无缓存行开销）；
//   · 多项式只在 |f| ≤ 0.5 上作用，相对误差 < 1e-7（远小于 8bit 量化步长）；
//   · 越界先钳，避免 (int) 转换溢出与阶码回绕（celestial_ocean.c 的
//     wo_fast_exp2 曾因 n+127<0 时按无符号回绕产生 -1e300，这里同样处理）。
// 所有调用点的实参都是非正数（光学厚度取负），下溢一律返回 0。
// ---------------------------------------------------------------------------
static inline float cloud_fast_exp2(float x) {
    if (x < -160.0f) return 0.0f;          // 2^-160 对 8bit 输出恒为 0
    if (x > 160.0f) return 1e30f;          // 上游有 fmaxf 保护，不返回 inf 以免 inf-inf 产 NaN
    float t = floorf(x + 0.5f);            // 最近整数（多项式区间取 [-0.5, 0.5]）
    float f = x - t;
    float p = 0.00015465314751714857f;
    p = p * f + 0.0013395282274223199f;
    p = p * f + 0.0096180399388676365f;
    p = p * f + 0.05550340675121699f;
    p = p * f + 0.24022651101359899f;
    p = p * f + 0.69314720006502339f;
    p = p * f + 0.99999999997183853f;      // 2^f
    int n = (int)t;
    if (n < -126) return 0.0f;             // 低于 float 正规数下界：等价 0
    if (n > 127) return 1e30f;
    union { float f; uint32_t u; } v;
    v.u = (uint32_t)(n + 127) << 23;
    return p * v.f;
}
static inline float cloud_fast_exp(float x) {
    return cloud_fast_exp2(x * 1.4426950408889634f);
}

// 三通道 exp（原 v3exp）：三通道各自一次快速 exp，与库版逐通道等价
static inline Cv3 v3exp(Cv3 a) {
    return v3(cloud_fast_exp(a.x), cloud_fast_exp(a.y), cloud_fast_exp(a.z));
}

// ---------------------------------------------------------------------------
// sRGB 伽马查表（性能专项 A5）
// cloud_tonemap 每像素调 3 次 powf(x, 1/2.2)（天空遍 + 云遍共 2 次 → 每屏幕像素
// 6 次 powf）。powf 是 fdlibm 里最贵的函数之一；改为 8192 项表 + 线性插值：
//   表值 = x^(1/2.2)，区间宽 1/8192，插值误差 < 2e-5（< 0.005/255），
//   远小于 8bit 量化步长，与 powf 的观感无可辨差异。表只建一次（32KB）。
// ---------------------------------------------------------------------------
#define CLOUD_GAMMA_LUT_N (8192)
static float s_gammaLut[CLOUD_GAMMA_LUT_N + 1];
static int s_gammaLutReady = 0;

static void cloud_gamma_lut_ensure(void) {
    if (s_gammaLutReady) return;
    for (int i = 0; i <= CLOUD_GAMMA_LUT_N; i++) {
        s_gammaLut[i] = powf((float)i / (float)CLOUD_GAMMA_LUT_N, 1.0f / 2.2f);
    }
    s_gammaLutReady = 1;
}

static inline float cloud_gamma(float x) {
    if (!(x > 0.0f)) return 0.0f;
    float f = x * (float)CLOUD_GAMMA_LUT_N;
    int i = (int)f;
    if (i >= CLOUD_GAMMA_LUT_N) return 1.0f;
    float t = f - (float)i;
    return s_gammaLut[i] + (s_gammaLut[i + 1] - s_gammaLut[i]) * t;
}

// ---------------------------------------------------------------------------
// 鱼眼屏幕射线用 sin/cos 查表（性能专项 A5）
// cloud_core_render 的鱼眼分支原本逐像素调 atan2f + 2×cosf + 2×sinf（每次 5 个
// 超越函数，天空遍与云遍各一遍）。ph 的 cos/sin 可直接由 (fx, fy)/r 得出（恒等式，
// 零误差）；th = (r/fishRmax)*fishFovRad 的 cos/sin 则查表——th∈[0, π] 上 2048 项
// 线性插值误差 ~3e-7，远小于 1/255。
// ---------------------------------------------------------------------------
#define CLOUD_SINCOS_N (2048)
#define CLOUD_SINCOS_MAX (3.2f)          // ≥ 170° 鱼眼半视场角上限
static float s_sinTab[CLOUD_SINCOS_N + 2];
static float s_cosTab[CLOUD_SINCOS_N + 2];
static int s_sinTabReady = 0;

static void cloud_sincos_tab_ensure(void) {
    if (s_sinTabReady) return;
    for (int i = 0; i <= CLOUD_SINCOS_N + 1; i++) {
        float a = (float)i * (CLOUD_SINCOS_MAX / (float)CLOUD_SINCOS_N);
        s_sinTab[i] = sinf(a);
        s_cosTab[i] = cosf(a);
    }
    s_sinTabReady = 1;
}

static inline void cloud_sincos(float a, float *sn, float *cs) {
    float f = a * ((float)CLOUD_SINCOS_N / CLOUD_SINCOS_MAX);
    if (!(f >= 0.0f)) f = 0.0f;
    int i = (int)f;
    if (i > CLOUD_SINCOS_N) i = CLOUD_SINCOS_N;
    float t = f - (float)i;
    *sn = s_sinTab[i] + (s_sinTab[i + 1] - s_sinTab[i]) * t;
    *cs = s_cosTab[i] + (s_cosTab[i + 1] - s_cosTab[i]) * t;
}

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

// ---------------------------------------------------------------------------
// 非平铺哈希值噪声（linglong.html 模型5改进回移）
// 用途一：cloudMap1/2 的域扭曲破平铺——basic/detail 为 tileable 纹理，大尺度观看
//   呈周期重复；用两个去相关值噪声通道把采样域逐格错开 ±0.5 纹理周期，相邻瓦片
//   错相。值噪声无 wrap 周期，不会像 local 纹理那样每 50km 复现同一扭曲场。
// 用途二：逐像素帧间去相关抖动（见 cloud_core_render）。
// ---------------------------------------------------------------------------
static inline uint32_t cloudHashT(uint32_t px, uint32_t py, uint32_t pz) {
    return (px * 1597334673u) ^ (py * 3812015801u) ^ (pz * 2798796415u);
}
// 末尾 avalanche 混合（MurmurHash3 finalizer）：仅乘常数时相邻格点高位几乎相同，
// 必须充分扩散
static inline float cloudHashFloat(uint32_t ix, uint32_t iz, uint32_t seed) {
    uint32_t t = cloudHashT(ix, iz, seed);
    t ^= t >> 16;
    t *= 2246822507u;
    t ^= t >> 13;
    t *= 3266489909u;
    t ^= t >> 16;
    return (float)(t >> 8) * (1.0f / 16777215.0f);
}
static float valueNoise2(float x, float z, uint32_t seed) {
    float fx = floorf(x), fz = floorf(z);
    float u = x - fx, v = z - fz;
    u = u * u * (3.0f - 2.0f * u);
    v = v * v * (3.0f - 2.0f * v);
    int ix = (int)fx, iz = (int)fz;
    float n00 = cloudHashFloat((uint32_t)ix, (uint32_t)iz, seed);
    float n10 = cloudHashFloat((uint32_t)(ix + 1), (uint32_t)iz, seed);
    float n01 = cloudHashFloat((uint32_t)ix, (uint32_t)(iz + 1), seed);
    float n11 = cloudHashFloat((uint32_t)(ix + 1), (uint32_t)(iz + 1), seed);
    return mixf(mixf(n00, n10, u), mixf(n01, n11, u), v);
}
// 双频值噪声 FBM（粗 + 细两层，与 linglong cloud_sky_valueNoise2Fbm 一致）
static float valueNoise2Fbm(float x, float z, uint32_t seed) {
    return valueNoise2(x, z, seed) * 0.65f + valueNoise2(x * 2.0f, z * 2.0f, seed + 17) * 0.35f;
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


// ---------------------------------------------------------------------------
// 3D 噪声 mip 链（仅浓积云使用）
// 浓积云在 1~4 km 高度，远景像素足迹（数十米~数公里）远大于噪声纹素（78 m / 24 m），
// 单点采样会把花椰菜细节采成逐像素翻动的噪点。按足迹选 mip 级把噪声低通到可分辨
// 尺度即可（移植自 linglong.html 的 cloud_sky_build_mip_chain3D / sampleMip3D）。
// 其它三层保持原样（它们的噪声尺度大、且观感已标定），故 mip 只服务 cloudMapCu。
// 内存：基础 64³ 的 mip 合计约 1.14 MB、细节 32³ 约 38 KB（WASM/Linux 可接受）。
// ---------------------------------------------------------------------------
#define CLOUD_MIP_MAX_LEVELS (7)
typedef struct {
    float *data[CLOUD_MIP_MAX_LEVELS];
    int    dim[CLOUD_MIP_MAX_LEVELS];
    int    n;
} CloudMip3D;

static CloudMip3D s_basicMip;
static CloudMip3D s_detailMip;

static void cloud_mip3d_build(CloudMip3D *m, const float *src, int dim) {
    m->n = 0;
    m->data[0] = (float *)src;          // 0 级直接引用原纹理（不复制、不释放）
    m->dim[0] = dim;
    m->n = 1;
    int cur = dim;
    const float *curData = src;
    while (cur > 1 && m->n < CLOUD_MIP_MAX_LEVELS) {
        const int nd = cur >> 1;
        float *dst = (float *)malloc(sizeof(float) * (size_t)nd * nd * nd);
        if (dst == NULL) break;
        for (int z = 0; z < nd; z++) {
            for (int y = 0; y < nd; y++) {
                for (int x = 0; x < nd; x++) {
                    // 8 邻域盒式平均（与参考实现的 downsample3D 一致）
                    float acc = 0.0f;
                    for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
                        const int sx = x * 2 + dx, sy = y * 2 + dy, sz = z * 2 + dz;
                        acc += curData[((size_t)sz * cur + sy) * cur + sx];
                    }
                    dst[((size_t)z * nd + y) * nd + x] = acc * 0.125f;
                }
            }
        }
        m->data[m->n] = dst;
        m->dim[m->n] = nd;
        m->n++;
        curData = dst;
        cur = nd;
    }
}

static inline float sampleMip3D(const CloudMip3D *m, Cv3 uvw, float lod) {
    if (m->n <= 1 || !(lod > 0.0f)) return sampleWrap3D(m->data[0], m->dim[0], uvw);
    const int top = m->n - 1;
    int l0 = (int)floorf(lod);
    if (l0 < 0) l0 = 0;
    else if (l0 > top) l0 = top;
    const float a = sampleWrap3D(m->data[l0], m->dim[l0], uvw);
    if (l0 >= top) return a;
    const float b = sampleWrap3D(m->data[l0 + 1], m->dim[l0 + 1], uvw);
    return a + (b - a) * (lod - (float)l0);
}

// 足迹（km）→ mip 级别；texelMeters 为 0 级纹素边长（米）
static inline float cloud_noise_lod(float footprintKm, float texelMeters) {
    return log2f(fmaxf(footprintKm * 1000.0f / texelMeters, 1.0f));
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

static void cloud_mip3d_build_all(void);   // 前置声明

static void cloud_generate_noise_textures(void) {
    // 快速查表（sRGB 伽马 / 鱼眼 sin-cos）先建（幂等，只真正执行一次）
    cloud_gamma_lut_ensure();
    cloud_sincos_tab_ensure();
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
    cloud_mip3d_build_all();
}

// 3D 噪声 mip 链（浓积云专用，见上面的 CloudMip3D 注释）
static void cloud_mip3d_build_all(void) {
    if (s_basicNoise != NULL && s_basicMip.n == 0) {
        cloud_mip3d_build(&s_basicMip, s_basicNoise, BASIC_DIM);
    }
    if (s_detailNoise != NULL && s_detailMip.n == 0) {
        cloud_mip3d_build(&s_detailMip, s_detailNoise, DETAIL_DIM);
    }
}

// ===========================================================================
// 大气与云参数（flower sky_component.cpp 直译）
// ===========================================================================
#define kEarthBottomRadius 6360.0f
#define kEarthTopRadius    6420.0f
#define kAtmospherePlanetRadiusOffset (21.0f * 0.001f)
#define kAtmosphereCameraOffset      kAtmospherePlanetRadiusOffset

// ===========================================================================
// 性能档位（可用 -D 覆盖做基准测试；默认值即发布值，实测数据见 doc/AGENTS.md）
//   CLOUD_OPT_RENDER_SCALE   内部渲染降采样倍数（1=全分辨率，2=半分辨率→像素数÷4）
//                            实测 2.8 → 10.4fps（仅低层云、320x240）
//   CLOUD_OPT_AMBIENT_SHADOW 1=每个受光样本再算一次竖直(环境)阴影，0=只算直射阴影
//                            实测（仅高积云）两者相差 <2%：竖直阴影射线离开 3.0~6.2km
//                            云带后样本全为空（cloudMap 早退），故默认保持 1（不丢暗部
//                            遮蔽）；多开浓积云/其它层时这一项才是有意义的旋钮
//   CLOUD_OPT_BAND_TRIM      1=把 ray-march 区间裁到当前云层掩码的高度带，并按路径长度
//                            收缩步数；0=固定走满整层云区（3~11km）
//   CLOUD_OPT_STEP_NOMINAL_KM 裁剪后的目标采样步距（km）。0.25 为实测折中：
//                            1/128 厚度（0.0625，flower 原步距）在裁剪后路径变短时会
//                            把掠射光线的采样加密到反而更慢，0.25 约等于 160x120 内部
//                            分辨率下的像素足迹，画面无可辨差异而快 ~1.6×
// ===========================================================================
#ifndef CLOUD_OPT_RENDER_SCALE
#define CLOUD_OPT_RENDER_SCALE (2)
#endif
#ifndef CLOUD_OPT_AMBIENT_SHADOW
#define CLOUD_OPT_AMBIENT_SHADOW (1)
#endif
#ifndef CLOUD_OPT_BAND_TRIM
#define CLOUD_OPT_BAND_TRIM (1)
#endif
#ifndef CLOUD_OPT_STEP_NOMINAL_KM
#define CLOUD_OPT_STEP_NOMINAL_KM (0.25f)
#endif
#define CLOUD_MARCHING_STEP_MIN (24)

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
    // ---- 性能档位（见文件头 CLOUD_OPT_* 宏；默认值在 cloud_default_scene 里落地） ----
    int   cloudRenderScale;     // 内部渲染降采样倍数（1=全分辨率，2=半分辨率）
    int   cloudAmbientShadow;   // 1=额外算竖直(环境)阴影，0=只算直射阴影
    int   cloudBandTrim;        // 1=ray-march 区间/步数按云层掩码裁剪
    float cloudStepNominalKm;   // 裁剪后的目标采样步距（km）
    // ---- light ----
    Cv3 sunDirection; // 指向太阳
    Cv3 sunColor;
    float sunIntensity;
    // 天空散射光源（linglong.html 模型5改进回移）：与太阳圆盘观感曲线解耦——
    // 曲线是“地面观测太阳圆盘”观感，直接用作天空光源会在暮光段“曲线红 × 透射率红”
    // 双重叠红；物理上太阳光谱恒定，红化由逐采样透射率承担。skySunColor 在低空段
    // 混入近白并对 ≤0° 暮光施加蓝紫基调（见 cloud_update_sky_light）。
    Cv3 skySunColor;
    float skySunIntensity;
} CloudScene;

static CloudScene s_scene;

// ===========================================================================
// 夜天光（散射模型 4）：天空积分只含「散射太阳光」，太阳落到地平线以下后直射项
// 被地球阴影归零、多重散射 LUT 也只剩暮光余晖，于是整片天空与云都变纯黑。真实夜空
// 另有气辉 + 积分星光 + 黄道光（见 celestial_night.h 的物理成分与方向分布）。
//
// 标定量：本模块 HDR 的白天晴天顶 ≈ 0.6，物理夜天光约为其 3e-8（≈ 1.8e-8）；直接叠加
// 在 ACES 下仍是 8bit 全黑，故同样按「暗适应后等效亮度」标定：1.8e-8 × 6.7e5（≈ +19.4 EV）
// ≈ 1.2e-2。该值经 ACES + sRGB 后落在 15~25/255 的微光区间，与模型 5 的夜空亮度
// （天顶 10~20/255）同量级。
// ===========================================================================
#define CLOUD_NIGHT_RAD (1.2e-2f)

// 夜天光的仰角表（见 celestial_night.h 的「性能」一节）：太阳相关部分每帧算一次，
// 逐像素只做线性插值 + 一个方位余弦。
static Linglong_Night_Sky s_night;

// 夜天光辐亮度（云世界坐标：x 东、y 上、z 南；太阳方向为 s_scene.sunDirection）
static Cv3 cloud_night_sky(Cv3 dir) {
    double w[3];
    linglong_night_sky_eval(&s_night, (double)dir.x, (double)dir.y, (double)dir.z, w);
    float k = CLOUD_NIGHT_RAD;
    return v3((float)w[0] * k, (float)w[1] * k, (float)w[2] * k);
}

// 夜天光可见度（0 = 白天/晨昏，1 = 天文夜）。由 cloud_update_sky_light 逐帧随太阳更新，
// 供逐像素路径做早退（白天不付任何代价）。
static float s_night_vis = 0.0f;

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
    // 云壳从浓积云云底之下 0.8 km 起（linglong.html 的 CLOUD_SKY_SHELL_BOTTOM_KM），
    // 顶仍是 11 km（卷层云顶）。四层云带都在这个壳里按物理高度切分。
    s->cloudAreaStartHeight = s->bottomRadius + CLOUD_SHELL_BOTTOM_KM;
    s->cloudAreaThickness   = CLOUD_SHELL_THICK_KM;
    s->cloudWeatherUVScale  = cv2(0.02f, 0.02f);
    s->cloudCoverage        = 0.5f;
    s->cloudDensity         = 1.0f;
    s->cloudShadingSunLightScale = 1.3f; // 太阳光对云的直射增强（默认 1.0，观感更亮）
    s->cloudFogFade         = 1.0f;
    // 远场裁剪：与 linglong.html 的 CLOUD_SKY_MAX_TRACE_KM=150 对齐（原 100 km）。
    // 云上俯视时云层延伸到地平线附近，100 km 会在约 5° 俯角处切出一条硬边。
    s->cloudMaxTraceingDistance     = 150.0f;
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

    // 性能档位（默认值见文件头 CLOUD_OPT_* 宏的说明）
    s->cloudRenderScale   = CLOUD_OPT_RENDER_SCALE;
    s->cloudAmbientShadow = CLOUD_OPT_AMBIENT_SHADOW;
    s->cloudBandTrim      = CLOUD_OPT_BAND_TRIM;
    s->cloudStepNominalKm = CLOUD_OPT_STEP_NOMINAL_KM;

    // 太阳（正午；纯白，同 ui_cloud_sun_color 白天段）
    s->sunDirection = v3norm(v3(0.15f, 0.96f, 0.0f));
    s->sunColor = v3(1.0f, 1.0f, 1.0f);
    s->sunIntensity = 48.0f;
    s->skySunColor = v3(1.0f, 1.0f, 1.0f);
    s->skySunIntensity = 48.0f;
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
    const float densityMie = cloud_fast_exp(s->mieDensityExpScale * viewHeight);
    const float densityRay = cloud_fast_exp(s->rayleighDensityExpScale * viewHeight);
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
// MS-LUT 每条方向积分的初始采样数：变步长二次分布（近端加密，与逐像素天空积分一致）。
// 旧固定步数 12 对曙暮光几何欠采样（阴影界上方数公里厚的“照亮带”在近水平方向
// 一个采样都落不进去）；且旧代码误传 sampleCountFixed=0 实际积分 0 步、LUT 恒为 0。
#define CLOUD_SKY_MS_VIEW_STEPS_INI 14

// 透射率/多重散射查找表：体积云渲染所用（约 26KB），按功能生命周期放 PSRAM——
// 原为静态数组，占内部 DRAM（与 DMA 帧缓冲同池）共 26KB，直接把启动 DMA 堆顶破，
// 导致 "Failed to alloc frame buffers!"。进入体积云时一次性申请（PSRAM）、不复释放
//（与应用生命周期一致，等价于原静态语义）。
static Cv3 *s_transLut = NULL;
static Cv3 *s_msLut = NULL;

// 确保 LUT 缓冲已分配（只有此函数分配；ui_cloud_init 与天象仪集成共用）。
// 返回 0 成功，-1 分配失败（已回滚释放）。
static int cloud_ensure_luts_buffers(void) {
    if (s_transLut && s_msLut) return 0;
    s_transLut = (Cv3 *)platform_malloc(sizeof(Cv3) * TRANS_LUT_W * TRANS_LUT_H);
    s_msLut    = (Cv3 *)platform_malloc(sizeof(Cv3) * MS_LUT_W * MS_LUT_H);
    if (!s_transLut || !s_msLut) {
        if (s_transLut) { free(s_transLut); s_transLut = NULL; }
        if (s_msLut)    { free(s_msLut);    s_msLut = NULL; }
        return -1;
    }
    return 0;
}

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
    float tMaxMax, int sampleCountIni, int bZeroMsAmbient)
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
    Cv3 globalL = v3mul(s->skySunColor, s->skySunIntensity); // 天空散射光源（暮光段近白，见 cloud_update_sky_light）

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
        float tEarth = raySphereIntersectNearest(v3add(P,
            v3mul(upVector, kAtmospherePlanetRadiusOffset)), sunDir, kEarthOrigin, s->bottomRadius);
        float earthShadow = (tEarth >= 0.0f) ? 0.0f : 1.0f;
        float mh = pHeight - s->bottomRadius;
        Cv2 muv = cv2(saturatef(sunZenithCosAngle * 0.5f + 0.5f),
                      clampf(mh / (s->topRadius - s->bottomRadius), 0.0f, 1.0f));
        // bZeroMsAmbient：MS-LUT 构建期以零环境光求一阶散射场（linglong 单遍构建语义，
        // 等价其零填充 scratch 读取；避免构建中读写同一 LUT 的数据竞争）
        Cv3 multiScatteredLuminance = bZeroMsAmbient ? v3(0, 0, 0) :
            sampleLut2D(s_msLut, MS_LUT_W, MS_LUT_H, muv.x, muv.y);

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
        // 地面回弹的地球阴影判定（linglong.html 模型5改进回移）：曙暮光时地面位于
        // 地影内，回弹应≈0；否则太阳低于地平时透射率 LUT 查询越界被钳制到地平线
        // 掠射列（最深红），把纯红环境光注入地基环境光/MS-LUT 阴影区（暮光偏红主因）
        float tEarthG = raySphereIntersectNearest(v3add(P,
            v3mul(upVector, kAtmospherePlanetRadiusOffset)), sunDir, kEarthOrigin, s->bottomRadius);
        float bounceShadow = (tEarthG >= 0.0f) ? 0.0f : 1.0f;
        Cv3 groundTerm = v3(NdotL * s->groundAlbedo.x / kPI,
                               NdotL * s->groundAlbedo.y / kPI,
                               NdotL * s->groundAlbedo.z / kPI);
        Cv3 bounce = v3mul3(v3mul3(globalL, transmittanceToSun),
                            v3mul3(throughput, groundTerm));
        L = v3add(L, v3mul(bounce, bounceShadow));
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
                worldPos, worldDir, sunDir, s, 0, 0, 40, 0, 9e6f, 40, 0);
            s_transLut[y * TRANS_LUT_W + x] = v3exp(v3mul(rr.opticalDepth, -1.0f));
        }
    }
}

// 预计算多重散射 LUT（flower MULTI_SCATTER_PASS：球面 4x4=16 方向采样；
// linglong.html 模型5改进回移：变步长二次采样单遍构建——变步长近端加密使曙暮光
// “照亮带”能被采样到；单遍以零环境光求一阶散射场（bZeroMsAmbient=1））
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
                    worldPos, worldDir, sunDir, s, 1, 0, 0, 1, 9e6f, CLOUD_SKY_MS_VIEW_STEPS_INI, 1);
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

    // 域扭曲破平铺（linglong.html 模型5改进回移）：本层 basic/detail 平铺周期约
    // 1.7km/0.8km，用非平铺值噪声两个去相关通道（~0.8km 特征）把采样域在 x/z 方向
    // 逐格错开 ±0.5 纹理周期，相邻瓦片错相，消除大尺度周期重复感
    float wA1 = appTime * cloudSpeed * 50.0f;
    float warpU1 = valueNoise2Fbm((wA1 + posMeter.x) / 800.0f + 0.17f,
                                  (wA1 + posMeter.z) / 800.0f + 0.17f, 101) - 0.5f;
    float warpV1 = valueNoise2Fbm((wA1 + posMeter.x) / 800.0f + 0.53f,
                                  (wA1 + posMeter.z) / 800.0f + 0.53f, 211) - 0.5f;
    Cv3 bnUv1 = v3mul(v3add(posKm, windOffset), s->cloudBasicNoiseScale * 2.0f);
    float basicNoise = sampleWrap3D(s_basicNoise, BASIC_DIM,
        v3(bnUv1.x + warpU1, bnUv1.y, bnUv1.z + warpV1));

    float basicCloudNoise = gradienShape * basicNoise;
    float basicCloudWithCoverage = coverage * remapf(basicCloudNoise, 1.0f - coverage, 1.0f, 0.0f, 1.0f);

    Cv3 sampleDetailNoise = v3sub(posKm, v3mul(windOffset, 0.15f));
    Cv3 dnUv1 = v3mul(sampleDetailNoise, s->cloudDetailNoiseScale * 2.0f);
    float detailNoiseComposite = sampleWrap3D(s_detailNoise, DETAIL_DIM,
        v3(dnUv1.x + warpU1, dnUv1.y, dnUv1.z + warpV1));
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

    // 域扭曲破平铺（同 cloudMap1；本层 basic 平铺周期约 1.1km）
    float wA2 = appTime * cloudSpeed * 50.0f;
    float warpU2 = valueNoise2Fbm((wA2 + posMeter.x) / 800.0f + 0.29f,
                                  (wA2 + posMeter.z) / 800.0f + 0.29f, 121) - 0.5f;
    float warpV2 = valueNoise2Fbm((wA2 + posMeter.x) / 800.0f + 0.61f,
                                  (wA2 + posMeter.z) / 800.0f + 0.61f, 223) - 0.5f;
    Cv3 bnUv2 = v3mul(v3add(v3add(posKm, windOffset), v3(0.39f, 0.39f, 0.39f)),
                      s->cloudBasicNoiseScale * 3.0f);
    float basicNoise = sampleWrap3D(s_basicNoise, BASIC_DIM,
        v3(bnUv2.x + warpU2, bnUv2.y, bnUv2.z + warpV2));

    float basicCloudNoise = gradienShape * basicNoise;
    float basicCloudWithCoverage = coverage * remapf(basicCloudNoise, 1.0f - coverage, 1.0f, 0.0f, 1.0f);

    float densityShape = saturatef(0.01f + (1.0f - normalizeHeight) * 0.5f) * 0.1f *
        remapf(normalizeHeight, 0.0f, 0.3f, 0.0f, 1.0f) *
        remapf(normalizeHeight, 0.7f, 1.0f, 1.0f, 0.0f);

    float cloudDensity = densityShape * basicCloudWithCoverage;
    cloudDensity = cloudDensity * kDensity;
    return saturatef(cloudDensity);
}

// 云层显示掩码（按 1 键循环：所有 → 浓积云 → 高积云 → 层积云 → 卷层云 → 无云）
#define CLOUD_LAYER_CU (1)   // 浓积云：云底 ~1.0 km，塔顶随参数动态（默认 ~4.1 km）
#define CLOUD_LAYER_AC (2)   // 高积云：3.0–6.2 km（原“低层”）
#define CLOUD_LAYER_SC (4)   // 层积云：6.2–9.4 km（原“中层”）
#define CLOUD_LAYER_CS (8)   // 卷层云：9.4–11.0 km（原“高层”）

// 云带边界（物理高度 km → 云壳归一化高度）：cloudMap 与 ray-march 云区裁剪共用同一套边界。
// 云壳 0.8~11.0 km；浓积云层带 [0.8, cuBandTop]，高积云 [3.0, 6.2]，层积云 [6.2, 9.4]，
// 卷层云 [9.4, 11.0]。浓积云与高积云在 3.0~cuBandTop 段重叠（同列可共存，取密度大者）。
#define CLOUD_BAND_CU_BOTTOM_KM  (0.80f)
#define CLOUD_BAND_AC0_KM        (3.00f)
#define CLOUD_BAND_AC1_KM        (6.20f)
#define CLOUD_BAND_SC1_KM        (9.40f)
#define CLOUD_NORM_HEIGHT(hkm)   (((hkm) - CLOUD_SHELL_BOTTOM_KM) / CLOUD_SHELL_THICK_KM)
#define CLOUD_N_CU_BOTTOM        CLOUD_NORM_HEIGHT(CLOUD_BAND_CU_BOTTOM_KM)
#define CLOUD_N_AC0              CLOUD_NORM_HEIGHT(CLOUD_BAND_AC0_KM)
#define CLOUD_N_AC1              CLOUD_NORM_HEIGHT(CLOUD_BAND_AC1_KM)
#define CLOUD_N_SC1              CLOUD_NORM_HEIGHT(CLOUD_BAND_SC1_KM)

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


// ===========================================================================
// 浓积云（Cu congestus）：云底 ~1.0 km 的平底积云，塔顶由疙瘩噪声决定
// 移植自 linglong.html 的 cloud_sky_cloudMapCu（其参数表 CLOUD_SKY_CU_PARAMS）
//
// 与其它三层的本质差别：那三层是“薄层”，用高度包络慢慢淡出即可；积云是**塔**——
//   · 底面固定（同一天空下凝结高度几乎相同）：云底只给 9 km 尺度的缓慢起伏与 100 m
//     抗锯齿过渡，不让底面跟着 3D 噪声起伏；
//   · 顶面不是包络，而是一张由疙瘩噪声决定的**截面高度场**：列顶 ± 疙瘩噪声，越过它
//     就在 90 m 内切到 0，于是顶面是高低错落、边缘仍然锐利的一团团花椰菜；
//   · 侧壁用覆盖率门限 + 3D 噪声平移做出鼓包，再用细节噪声啃出海绵状细边。
// ===========================================================================

// 几何/形态参数（linglong.html 的 CLOUD_SKY_CU_PARAMS；本项目不开放滑块，取默认值）
typedef struct {
    float baseKm;    // 云底高度（km）
    float depthKm;   // 塔身平均厚度（km，云底 → 塔顶）
    float spread;    // 塔顶参差（0=齐平，1=高低差近三倍）
    float cauli;     // 花椰菜强度：顶部破碎 + 侧壁鼓包 + 边界侵蚀的总增益
    float sharp;     // 边界锐度（1/过渡宽度）
    float cellKm;    // 云胞尺度（km）
    float density;   // 云芯密度
} CloudCuParams;

static const CloudCuParams S_CU_PARAMS = {
    1.00f,   // baseKm：云底 1.0 km（与 linglong.html 默认一致）
    0.60f,   // depthKm
    1.00f,   // spread
    2.00f,   // cauli
    8.0f,    // sharp
    3.00f,   // cellKm
    0.0050f, // density
};

// 内部常量（不对外暴露）
#define CLOUD_CU_BASE_AMP_KM       (0.10f)  // 云底缓慢起伏峰峰值
#define CLOUD_CU_BASE_RAMP_KM      (0.10f)  // 云底 0→1 过渡厚度（≈100 m：平底且清晰）
#define CLOUD_CU_NOISE_SCALE       (0.20f)  // 基础噪声尺度：鼓包 ≈1.2 km
#define CLOUD_CU_DETAIL_SCALE      (0.62f)  // 细节噪声尺度：细边 ≈200 m
#define CLOUD_CU_LOBE_SCALE        (0.46f)  // 疙瘩噪声尺度：花椰菜疙瘩 ≈270 m
#define CLOUD_CU_TOP_RAMP_KM       (0.09f)  // 塔顶切边厚度（越小顶面越锐）
#define CLOUD_CU_TOP_RELIEF_KM     (0.45f)  // 塔顶疙瘩高度场幅度
#define CLOUD_CU_DEPTH_GAIN        (1.72f)  // 塔高标定：列顶高度 → 实测厚度
#define CLOUD_CU_TAPER             (0.13f)  // 门限随高度抬升 → 塔身锥形收窄
#define CLOUD_CU_GATE_NOISE        (0.26f)  // 3D 噪声对门限的平移量（× cauli）
#define CLOUD_CU_ERO_BASE          (0.22f)  // 云底处的侵蚀幅度
#define CLOUD_CU_ERO_TOP           (0.34f)  // 塔顶额外侵蚀幅度（× cauli）
#define CLOUD_CU_BASIC_MEAN        (0.83f)  // Perlin-Worley 基础噪声实测均值
#define CLOUD_CU_DETAIL_MEAN       (0.70f)  // Worley FBM 细节噪声实测均值
#define CLOUD_CU_SHADOW_LITE_GAIN  (1.10f)  // 低配密度的平均光学厚度补偿
#define CLOUD_CU_BAND_MARGIN_KM    (0.25f)  // 层带顶在最高塔顶之上留的余量
#define CLOUD_CU_BAND_MAX_KM       (4.60f)  // 层带顶硬上限
// powder 项的“层内归一化高度”固定按 4.6 km 标定（与层带几何解耦，观感不随层带顶变）
#define CLOUD_CU_POWDER_N1         CLOUD_NORM_HEIGHT(CLOUD_CU_BAND_MAX_KM)
// 云量档位的两端（= CLOUD_COVERAGE_LEVELS[1] 与末项；该表在文件后段定义，
// 这里是常量表达式，改动档位表时需同步）
#define CLOUD_SHADOW_FOOTPRINT_KM  (0.25f)  // 阴影采样的固定足迹（linglong.html 同值）
#define CLOUD_CU_LEVEL_MIN         (0.325f)
#define CLOUD_CU_LEVEL_MAX         (0.395f)

// 浓积云层带顶（km）：随形状参数动态生成——塔顶最高的情形 = 云底 + 列顶上限
// + 疙瘩噪声抬升上限，再留 0.25 km 余量；上限压在云壳顶之下 0.2 km。
// 默认参数下 ≈4.11 km（与 linglong.html 一致）。参数在本项目里是编译期常量，
// 故首次调用算一次后缓存（该函数在密度分发里每样本都要用）。
static float s_cu_band_top_km = -1.0f;
static inline float cloud_cu_band_top_km(void) {
    if (s_cu_band_top_km < 0.0f) {
        const CloudCuParams *P = &S_CU_PARAMS;
        const float tMax = P->depthKm * CLOUD_CU_DEPTH_GAIN * 1.10f * (1.0f + P->spread)
            + CLOUD_CU_TOP_RELIEF_KM * P->cauli * 0.6f
            + CLOUD_CU_BASE_AMP_KM * 0.5f;
        s_cu_band_top_km = fminf(fminf(P->baseKm + tMax + CLOUD_CU_BAND_MARGIN_KM,
                                       CLOUD_CU_BAND_MAX_KM),
                                 CLOUD_SHELL_TOP_KM - 0.20f);
    }
    return s_cu_band_top_km;
}

// 浓积云密度（linglong.html cloud_sky_cloudMapCu 直译；hKm 为样本点海拔）
//   posMeter：样本点世界坐标（m）  hKm：样本点海拔（km）
//   bandTopKm：层带顶（km，塔顶不会被层带边界平切）  lite：阴影光线用低配密度
static float cloudMapCu(Cv3 posMeter, float hKm, float appTime, const CloudScene *s,
                        float bandTopKm, int lite, float footprintKm) {
    const CloudCuParams *P = &S_CU_PARAMS;
    const float cauli = P->cauli;

    // ---- 最便宜的两个否定判据（挡掉约 1/5 的求值）----
    if (s->cloudCoverage <= 0.0f) return 0.0f;
    if (hKm < P->baseKm - CLOUD_CU_BASE_AMP_KM * 0.5f) return 0.0f;

    const float wdx = s->cloudDirection.x, wdz = s->cloudDirection.z;
    const float wA = appTime * s->cloudSpeed * 50.0f;   // 平流位移（米，与其它层同速）
    const float woT = appTime * s->cloudSpeed;

    // 塔身顺风轻微倾斜（其它层的 shear 是给薄层做风切变的，积云塔被斜切会散架）
    const float shear = fmaxf(0.0f, hKm - P->baseKm) * 34.0f;
    const float mx = posMeter.x + wdx * shear;
    const float mz = posMeter.z + wdz * shear;
    const float kx = mx * 0.001f, ky = posMeter.y * 0.001f, kz = mz * 0.001f;

    // ---- 云量档位 → 门限（u∈[0,1]：1 号档疏云 → 5 号档满云）----
    const float u = saturatef((s->cloudCoverage - CLOUD_CU_LEVEL_MIN) /
                              (CLOUD_CU_LEVEL_MAX - CLOUD_CU_LEVEL_MIN));
    const float gateC = 0.625f - 0.325f * u;

    // ---- 横向细胞场（2D 值噪声）----
    // 云胞（cellKm，默认 3 km）决定单朵云的粗细与位置；云团 5.2 km 决定哪片天空对流活跃。
    // 两个场都只需低倍频就能做门限预判，越界整段跳过。
    const float invCell = 1.0f / (P->cellKm * 1000.0f);
    const float cellX = (wA + mx) * invCell + 1.1f, cellZ = (wA + mz) * invCell + 2.3f;
    const float regX = (wA + mx) / 5200.0f + 0.31f, regZ = (wA + mz) / 5200.0f + 0.31f;
    const float cellA = valueNoise2(cellX, cellZ, 829);
    const float regA  = valueNoise2(regX, regZ, 821);
    // 覆盖场的保守上界（第二倍频按 1.0 计）
    const float coverUpper = (cellA * 0.65f + 0.35f) * 0.72f + (regA * 0.65f + 0.35f) * 0.28f;
    // 3D 噪声最多把门限拉低 GATE_NOISE×cauli×0.17（basicNoise 上限 1.0）
    const float gateMargin = CLOUD_CU_GATE_NOISE * cauli * 0.17f;
    if (coverUpper < gateC - gateMargin) return 0.0f;

    // 通过预判后才补第二倍频，拼出与 valueNoise2Fbm 一致的覆盖场
    const float clusCell = cellA * 0.65f + valueNoise2(cellX * 2.0f, cellZ * 2.0f, 846) * 0.35f;
    const float clusReg  = regA  * 0.65f + valueNoise2(regX * 2.0f, regZ * 2.0f, 838) * 0.35f;
    const float coverField = clusCell * 0.72f + clusReg * 0.28f;

    // ---- 云底（放在门限预判之后：被否掉的样本不必再算这个值噪声）----
    const float baseN = valueNoise2Fbm((wA + mx) / 9000.0f + 0.13f, (wA + mz) / 9000.0f + 0.13f, 811);
    const float hBase = P->baseKm + (baseN - 0.5f) * CLOUD_CU_BASE_AMP_KM;
    const float yb = hKm - hBase;                       // 距云底高度（km）
    if (yb <= 0.0f) return 0.0f;

    // ---- 列顶高度（一朵云一个顶，空间相干；云量档位越高整体越旺盛）----
    const float towerField = saturatef((clusReg - 0.20f) * 1.9f);
    const float tMean = P->depthKm * CLOUD_CU_DEPTH_GAIN * (0.90f + 0.20f * u);
    const float bandTop = fminf(bandTopKm, CLOUD_CU_BAND_MAX_KM);
    const float tCap = fmaxf(bandTop - 0.20f - hBase, 0.35f);
    const float hTopRel = fminf(tMean * (1.0f + P->spread * (2.0f * towerField - 1.0f)), tCap);
    const float upDown = saturatef(yb / fmaxf(hTopRel, 0.30f));   // 0=云底 1=本列塔顶
    const float upPow = powf(upDown, 0.7f);

    // ---- 阴影光线的低配密度：3D 噪声按实测均值代入（省掉 1 张 curl + 3 次 3D 采样）----
    if (lite) {
        if ((coverField - gateC - CLOUD_CU_TAPER * upPow + gateMargin) * P->sharp <= 0.0f) return 0.0f;
        const float cellL = saturatef((coverField - gateC - CLOUD_CU_TAPER * upPow) * P->sharp);
        const float vertTopL = 1.0f - smoothstepf(hTopRel - CLOUD_CU_TOP_RAMP_KM, hTopRel, yb);
        const float shapeL = saturatef(cellL * saturatef(yb / CLOUD_CU_BASE_RAMP_KM) * vertTopL);
        if (shapeL <= 0.02f) return 0.0f;
        const float eroAmpL = (CLOUD_CU_ERO_BASE + CLOUD_CU_ERO_TOP * cauli * upDown) *
            (1.20f - 0.45f * u) * (0.60f + 0.85f * CLOUD_CU_BASIC_MEAN);
        const float eroL = eroAmpL * (1.0f - CLOUD_CU_DETAIL_MEAN);
        float dL = saturatef((shapeL - eroL) / fmaxf(1.0f - eroL, 1e-3f));
        dL *= 0.75f + 0.25f * saturatef((CLOUD_CU_BASIC_MEAN - 0.6f) * 2.2f);
        dL *= mixf(0.60f, 1.0f, saturatef(yb / 0.30f));
        return saturatef(dL * s->cloudDensity * P->density * CLOUD_CU_SHADOW_LITE_GAIN);
    }

    // 保守界（数学上等价）：门限里“高度抬升”这一项与噪声无关，先把它的上界判一次
    if ((coverField - gateC - CLOUD_CU_TAPER * upPow + gateMargin) * P->sharp <= 0.0f) return 0.0f;

    // ---- 弱点域扭曲：打散 2D 值噪声的格点感（积云边界要清晰，幅度很小）----
    Cv3 curl = sampleWrap2D3(s_curlNoise, CURL_DIM,
                             cv2((wA + mx) * 0.0000013f + 0.17f, (wA + mz) * 0.0000013f + 0.61f));
    const float warpX = (curl.x * 2.0f - 1.0f) * 0.16f;   // ±160 m
    const float warpY = (curl.y * 2.0f - 1.0f) * 0.05f;   // ±50 m
    const float warpZ = (curl.z * 2.0f - 1.0f) * 0.16f;

    // ---- 3D 噪声：基础（鼓包）、细节（海绵边）、疙瘩（塔顶起伏）----
    // 按本样本足迹选 mip 级（纹素边长 = 1000/(scale·dim) 米：基础 78 m、细节 24 m）
    const float lodBasic  = cloud_noise_lod(footprintKm, 1000.0f / (CLOUD_CU_NOISE_SCALE * (float)BASIC_DIM));
    const float lodDetail = cloud_noise_lod(footprintKm, 1000.0f / (CLOUD_CU_DETAIL_SCALE * (float)DETAIL_DIM));
    const float lodLobe   = cloud_noise_lod(footprintKm, 1000.0f / (CLOUD_CU_LOBE_SCALE * (float)DETAIL_DIM));
    const float wx = wdx * woT, wy = (s->cloudDirection.y + 0.1f) * woT, wz = wdz * woT;
    const float basicNoise = sampleMip3D(&s_basicMip,
        v3((kx + warpX + wx) * CLOUD_CU_NOISE_SCALE,
           (ky + warpY + wy) * CLOUD_CU_NOISE_SCALE,
           (kz + warpZ + wz) * CLOUD_CU_NOISE_SCALE), lodBasic);
    const float detailNoise = sampleMip3D(&s_detailMip,
        v3((kx - wx * 0.15f) * CLOUD_CU_DETAIL_SCALE,
           (ky - wy * 0.15f) * CLOUD_CU_DETAIL_SCALE,
           (kz - wz * 0.15f) * CLOUD_CU_DETAIL_SCALE), lodDetail);
    const float lobeNoise = sampleMip3D(&s_detailMip,
        v3((kx + warpX * 0.5f + wx) * CLOUD_CU_LOBE_SCALE + 11.3f,
           (ky + warpY + wy) * CLOUD_CU_LOBE_SCALE + 5.7f,
           (kz + warpZ * 0.5f + wz) * CLOUD_CU_LOBE_SCALE + 19.1f), lodLobe);

    // ---- 横向轮廓：门限 = 云量门限 + 高度抬升（塔身收窄）− 3D 噪声平移（花椰菜鼓包）----
    const float gateH = gateC + CLOUD_CU_TAPER * upPow
                      - CLOUD_CU_GATE_NOISE * cauli * (basicNoise - 0.83f);
    const float cell = saturatef((coverField - gateH) * P->sharp);
    if (cell <= 0.0f) return 0.0f;

    // ---- 顶盖：高度由疙瘩噪声决定 → 锐利而破碎的花椰菜顶 ----
    const float capKm = hTopRel + CLOUD_CU_TOP_RELIEF_KM * cauli * (lobeNoise - 0.70f) * 2.0f;
    const float vertTop = 1.0f - smoothstepf(capKm - CLOUD_CU_TOP_RAMP_KM, capKm, yb);
    const float vertBase = saturatef(yb / CLOUD_CU_BASE_RAMP_KM);   // 云底 100 m 内 0→1：平底
    const float shape = saturatef(cell * vertBase * vertTop);
    if (shape <= 0.02f) return 0.0f;

    // ---- 边界侵蚀（随高度增大，并逐点被基础噪声调制 → 侧壁鼓包）----
    const float eroAmp = (CLOUD_CU_ERO_BASE + CLOUD_CU_ERO_TOP * cauli * upDown) *
        (1.20f - 0.45f * u) * (0.60f + 0.85f * basicNoise);
    const float ero = eroAmp * (1.0f - detailNoise);
    float density = saturatef((shape - ero) / fmaxf(1.0f - ero, 1e-3f));

    // 云体内密度随基础噪声起伏（云芯更实、边缘更薄）；云底附近略薄（透出下方暗色）
    density *= 0.75f + 0.25f * saturatef((basicNoise - 0.6f) * 2.2f);
    density *= mixf(0.60f, 1.0f, saturatef(yb / 0.30f));
    return saturatef(density * s->cloudDensity * P->density);
}

// 云密度场分发（四层按物理高度分层带；浓积云层带与高积云层带在 3.0~cuBandTop 重叠，
// 两层的层内归一化高度各自独立，故分别求值后取密度大者，并把胜出层的 h01 回传给 powder 项）。
// cuLite：阴影光线用浓积云的“低配密度”（不做 3D 采样，见 cloudMapCu）。
static float cloudMap(Cv3 posMeter, float normalizeHeight, float appTime, const CloudScene *s,
    float *actualH01, int cuLite, float footprintKm)
{
    const int mask = s_ui.layer_mask;
    static float s_nCu1 = -1.0f;
    if (s_nCu1 < 0.0f) s_nCu1 = CLOUD_NORM_HEIGHT(cloud_cu_band_top_km());
    const float nCu1 = s_nCu1;
    const float hKm = CLOUD_SHELL_BOTTOM_KM + normalizeHeight * CLOUD_SHELL_THICK_KM;
    float best = 0.0f, bestH01 = 0.0f;

    if (normalizeHeight < nCu1) {
        if (mask & CLOUD_LAYER_CU) {
            // h01 用固定标定（CLOUD_CU_POWDER_N1），不用动态层带顶：powder 的观感不能随层带变
            const float h01 = saturatef(normalizeHeight / CLOUD_CU_POWDER_N1);
            const float d = cloudMapCu(posMeter, hKm, appTime, s, cloud_cu_band_top_km(), cuLite, footprintKm);
            if (d > best) { best = d; bestH01 = h01; }
        }
        // 与高积云层带重叠的那一段：两层都求值取大者（真实的“积云在下、高积云在上”）
        if (normalizeHeight >= CLOUD_N_AC0 && (mask & CLOUD_LAYER_AC)) {
            const float h01 = (normalizeHeight - CLOUD_N_AC0) / (CLOUD_N_AC1 - CLOUD_N_AC0);
            const float d = cloudMap0(posMeter, h01, appTime, s);
            if (d > best) { best = d; bestH01 = h01; }
        }
    }
    else if (normalizeHeight < CLOUD_N_AC1) {
        if (normalizeHeight >= CLOUD_N_AC0 && (mask & CLOUD_LAYER_AC)) {
            bestH01 = (normalizeHeight - CLOUD_N_AC0) / (CLOUD_N_AC1 - CLOUD_N_AC0);
            best = cloudMap0(posMeter, bestH01, appTime, s);
        }
    }
    else if (normalizeHeight < CLOUD_N_SC1) {
        if (mask & CLOUD_LAYER_SC) {
            bestH01 = (normalizeHeight - CLOUD_N_AC1) / (CLOUD_N_SC1 - CLOUD_N_AC1);
            best = cloudMap1(posMeter, bestH01, appTime, s);
        }
    }
    else if (mask & CLOUD_LAYER_CS) {
        bestH01 = (normalizeHeight - CLOUD_N_SC1) / (1.0f - CLOUD_N_SC1);
        best = cloudMap2(posMeter, bestH01, appTime, s);
    }
    *actualH01 = (best > 0.0f) ? bestH01 : 0.0f;
    return best;
}

// 当前云层掩码覆盖的归一化高度区间 [h0, h1]（三层是连续的，故并集即 [最小起点, 最大终点]）。
// 返回 0 表示当前无任何云层启用（可用作 early-out，省掉整条 ray-march）。
static int cloud_layer_band(float *h0, float *h1) {
    const int m = s_ui.layer_mask & (CLOUD_LAYER_CU | CLOUD_LAYER_AC | CLOUD_LAYER_SC | CLOUD_LAYER_CS);
    if (m == 0) return 0;
    float lo = 2.0f, hi = -1.0f;
    if (m & CLOUD_LAYER_CU) { lo = fminf(lo, CLOUD_N_CU_BOTTOM); hi = fmaxf(hi, CLOUD_NORM_HEIGHT(cloud_cu_band_top_km())); }
    if (m & CLOUD_LAYER_AC) { lo = fminf(lo, CLOUD_N_AC0);       hi = fmaxf(hi, CLOUD_N_AC1); }
    if (m & CLOUD_LAYER_SC) { lo = fminf(lo, CLOUD_N_AC1);       hi = fmaxf(hi, CLOUD_N_SC1); }
    if (m & CLOUD_LAYER_CS) { lo = fminf(lo, CLOUD_N_SC1);       hi = fmaxf(hi, 1.0f); }
    *h0 = lo;
    *h1 = hi;
    return 1;
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
        // 饱和早退（性能专项 A3）：两条累积都饱和后 T0/T1 逐位为 0（乘法累积），
        // 继续 march 纯属白算（云芯内部常见）。acc1 = acc0×msExt < acc0，
        // 故以 acc1 为判据：acc1×1000 > 13.8 ⇒ T1 = exp(-13.8) < 1e-6（远低于 1/255）。
        if (extinctionAcc1 * 1000.0f > 13.8f) break;
        float curT = shadowT * shadowT;
        float deltaT = curT - prevT;
        float extinctionFactor = deltaT * kTotalLen;
        float shadowSampleDis = kTotalLen * (prevT + deltaT * 0.5f);
        prevT = curT;

        Cv3 samplePosKm = v3add(posKm, v3mul(sunDirection, shadowSampleDis));
        float sampleHeightKm = v3len(samplePosKm);
        float sampleDt = sampleHeightKm - s->cloudAreaStartHeight;
        float normalizeHeight = sampleDt / s->cloudAreaThickness;
        // 层带外直接跳过（性能专项 A3）：云壳 0~1 归一高度以外的密度恒为 0
        // （三层的高度包络 remap 已饱和到 0，浓积云在云底以下亦为 0），
        // 而 cloudMap 对 n>1 仍会走 cloudMap2 的 3D 采样，白付代价。
        if (normalizeHeight < 0.0f || normalizeHeight > 1.0f) continue;
        Cv3 samplePosMeter = v3mul(samplePosKm, 1000.0f);
        float actualH01;
        // 阴影只需要“沿光线的光学厚度”：浓积云在这里走低配密度（不做 3D 采样）。
        // 足迹取固定 0.25 km（阴影采样间隔远大于像素足迹，点采样只会得到逐像素不同的阴影）
        float density = cloudMap(samplePosMeter, normalizeHeight, appTime, s, &actualH01,
                                 1, CLOUD_SHADOW_FOOTPRINT_KM);
        extinctionAcc0 += density * extinctionFactor;
        extinctionAcc1 += (density * msExtinctionFactor) * extinctionFactor;
    }

    pm.transmittanceToLight[0] = cloud_fast_exp(-extinctionAcc0 * 1000.0f);
    pm.transmittanceToLight[1] = cloud_fast_exp(-extinctionAcc1 * 1000.0f);
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

// 云的直射光随“样本点处太阳低于其几何地平”衰减（linglong.html 模型5改进回移，
// 根因修复）：太阳低于样本点几何地平（cosz ≤ -sqrt(1-(R/r)²)）后直射光不存在；
// 0 → cosz_hor 之间用 smoothstep 平滑过渡，保证曙暮光“红→暗”临界连续、物理合理
//（高层云地平凹陷角更小、可看夕阳更久——符合实际）
static float cloudDirectFade(float sunZenithCos, float radiusKm) {
    float ratio = kEarthBottomRadius / radiusKm;
    float coszHor = -sqrtf(fmaxf(0.0f, 1.0f - ratio * ratio));
    if (sunZenithCos >= 0.0f) return 1.0f;
    if (sunZenithCos <= coszHor) return 0.0f;
    float t = (sunZenithCos - coszHor) / (0.0f - coszHor);
    return t * t * (3.0f - 2.0f * t);
}

// 地面上涌光（见 render_frame 每帧由天顶天空亮度更新）。
static Cv3 s_groundUpwelling;

// flower cloudColorCompute 直译（剔除对场景深度、Froxel、远辉光网格等装饰贴图依赖）
typedef struct { Cv3 color; float transmittance; } CloudPixel;

static CloudPixel cloudPixelCompute(Cv3 worldPos, Cv3 worldDir, float appTime, float jitter,
    float pixAng, const CloudScene *s)
{
    CloudPixel out;
    out.color = v3(0, 0, 0);
    out.transmittance = 1.0f;

    const Cv3 kOrigin = v3(0, 0, 0);
    float radiusCloudStart = s->cloudAreaStartHeight;
    float radiusCloudEnd = radiusCloudStart + s->cloudAreaThickness;
    float viewHeight = v3len(worldPos);

    // 云区裁剪：ray-march 区间限制在当前云层掩码对应的归一化高度带内
    // （默认只渲染低层时，整层云区 3.0~11.0km 中的 6.2~11.0km 根本不用走）。
    float bandH0 = 0.0f, bandH1 = 1.0f;
    if (s->cloudBandTrim) {
        if (!cloud_layer_band(&bandH0, &bandH1)) return out; // 无云层：整条光线提前返回
        radiusCloudStart = s->cloudAreaStartHeight + bandH0 * s->cloudAreaThickness;
        radiusCloudEnd   = s->cloudAreaStartHeight + bandH1 * s->cloudAreaThickness;
    }

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

    const int stepCountUnitMax = s->cloudMarchingStepNum;
    float stepCount = (float)stepCountUnitMax;
    // 按目标步距定步数（只减不增）：裁剪后路径变短，若仍固定 128 步，掠射光线的采样会被
    // 加密（更精细但更慢）；反之放宽步距则采样变稀。上限仍是 cloudMarchingStepNum。
    if (s->cloudBandTrim && s->cloudStepNominalKm > 0.0f) {
        float need = ceilf(marchingDistance / s->cloudStepNominalKm);
        if (need < (float)CLOUD_MARCHING_STEP_MIN) need = (float)CLOUD_MARCHING_STEP_MIN;
        if (need < stepCount) stepCount = need;
    }
    const float stepT = (tMax - tMin) / stepCount;

    float sampleT = tMin + 0.001f * stepT + stepT * saturatef(jitter);

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
        // 用样本点半径 sh（原为相机半径 viewHeight）：高空观察时相机在云层上方数公里，
        // 用相机半径会显著高估太阳透射率，云海发白。透射率只取决于样本点到太阳的路径。
        Cv2 uu = lutTransmittanceParamsToUv(s, sh, cosz);
        atmosphereTransmittance0 = sampleLut2D(s_transLut, TRANS_LUT_W, TRANS_LUT_H, uu.x, uu.y);
    }
    {
        Cv3 sp = v3add(worldPos, v3mul(worldDir, tMax));
        float sh = v3len(sp);
        Cv3 upv = v3div(sp, sh);
        float cosz = v3dot(sunDirection, upv);
        Cv2 uu = lutTransmittanceParamsToUv(s, sh, cosz);   // 同上：样本点半径
        atmosphereTransmittance1 = sampleLut2D(s_transLut, TRANS_LUT_W, TRANS_LUT_H, uu.x, uu.y);
    }

    // 日落时多重散射增强
    float sunSetScale = 1.0f + saturatef(1.0f - sunDirection.y * 2.0f);

    for (int i = 0; i < (int)stepCount; i++) {
        Cv3 samplePos = v3add(worldPos, v3mul(worldDir, sampleT));
        float sampleHeight = v3len(samplePos);
        Cv3 atmosphereTransmittance = v3lerp(atmosphereTransmittance0, atmosphereTransmittance1,
                                             saturatef(sampleT / marchingDistance));

        float normalizeHeight = (sampleHeight - s->cloudAreaStartHeight) / s->cloudAreaThickness;
        Cv3 samplePosMeter = v3mul(samplePos, 1000.0f);
        float actualH01 = 0.0f;
        // 本样本的足迹（km）：像素足迹与该处步长取大者。浓积云据此选 3D 噪声的 mip 级，
        // 否则远景（足迹 >> 噪声纹素）会把花椰菜细节采成逐像素翻动的噪点。
        const float sampleFootprintKm = fmaxf(fmaxf(sampleT * pixAng, stepT), 0.01f);
        float stepCloudDensity = cloudMap(samplePosMeter, normalizeHeight, appTime, s,
                                          &actualH01, 0, sampleFootprintKm);

        if (stepCloudDensity > 0.0f) {
            float opticalDepth = stepCloudDensity * stepT * 1000.0f;
            float stepTransmittance = fmaxf(cloud_fast_exp(-opticalDepth),
                                            cloud_fast_exp(-opticalDepth * 0.25f) * 0.7f);

            // 样本点-太阳几何（提前于阴影 march 求值，见下）：低于样本点几何地平后
            // 直射光平滑衰减到 0（linglong 根因修复回移；夜间仅余中性环境补光）
            float sunZenithCosAtSample = v3dot(sunDirection, samplePos) / sampleHeight;
            float directFade = cloudDirectFade(sunZenithCosAtSample, sampleHeight);

            // 直射阴影只在 directFade > 0 时才有意义（性能专项 A3）：
            // sunlightTerm 逐通道含因子 directFade，其值为 0 时整个直射项（含两阶
            // 多重散射的直射部分）恒为 0，而它唯一的作用就是乘 shadow.transmittanceToLight[]。
            // 故 directFade==0 时跳过这 12 步 march，结果逐位不变；夜间/天文暮光整屏生效。
            CloudShadow shadow;
            shadow.transmittanceToLight[0] = 0.0f;
            shadow.transmittanceToLight[1] = 0.0f;
            if (directFade > 0.0f) {
                shadow = volumetricShadow(samplePos, sunDirection, appTime, s,
                                          s->cloudMultiScatterExtinction);
            }
            CloudShadow shadowAmbient;
            // 环境（竖直）阴影：再跑一遭 12 步 ray-march。关掉后环境光退化为均匀天空光
            // （云底暗部略变平）；仅低层云时竖直阴影射线大多落在空云层（cloudMap 早退），
            // 实测收益 <2%，故默认保持开启（见 CLOUD_OPT_AMBIENT_SHADOW）。
            int bGroundContrib = (s->cloudFogFade > 0.0f) && (s->cloudAmbientShadow != 0);
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

            // 云直射光用解耦后的天空光（暮光段与天空散射一致，避免“曲线红 × 透射率红”双重叠红）
            Cv3 sunlightTerm = v3mul3(atmosphereTransmittance,
                v3mul(v3mul(s->skySunColor, s->skySunIntensity),
                      s->cloudShadingSunLightScale * directFade));

            // 环境补光用中性化透射率（取三通道最大，linglong.html 改进回移）：
            // 环境光是天穹下行光，颜色应由上涌光决定，不继承“朝太阳透射率 LUT”
            // 在 cosz<0 时的深红钳制，否则夜间环境残余被染红
            float atmosphereNeutral = fmaxf(atmosphereTransmittance.x,
                fmaxf(atmosphereTransmittance.y, atmosphereTransmittance.z));
            Cv3 ambientLit = v3mul3(v3mul3(s_groundUpwelling, v3(powderEffect, powderEffect, powderEffect)),
                v3mul3(v3(s->cloudAmbientScale, s->cloudAmbientScale, s->cloudAmbientScale),
                       v3mul3(v3(1.0f - sunDirection.y * sunDirection.y,
                                 1.0f - sunDirection.y * sunDirection.y,
                                 1.0f - sunDirection.y * sunDirection.y),
                              v3lerp(v3(atmosphereNeutral, atmosphereNeutral, atmosphereNeutral),
                                     v3(1, 1, 1), saturatef(1.0f - transmittance)))));

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
    // 夜间环境散射（见 cloud_night_sky）：云挡住了后方的气辉/星光层，自身又散射上/下行的
    // 夜天光。白云反照率 ≈ 0.8，故云在夜间的亮度应与夜空同量级（真实无月夜的云几乎看不出
    // 轮廓）；不补这一项，云的直射/多重散射在太阳落下后归零，云会变成比夜空更黑的剪影带
    // （天空是微光、云层是纯黑的“暗带”）。s_night_vis 在白天为 0，此分支逐位不生效。
    if (s_night_vis > 0.0f) {
        Cv3 night = cloud_night_sky(worldDir);
        float cov = 1.0f - transmittance;   // 累积覆盖率（color 为预乘色，同量纲）
        out.color = v3add(out.color, v3mul(night, 0.8f * cov));
    }
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
// 内部渲染分辨率 = 屏幕全分辨率（上限 CLOUD_MAX_* = 320x240）。
// 历史：曾为屏幕的一半（tty 320x240 → 内部 160x120）双线性放大，半分辨率计算约
// 30fps（全屏约 8fps，单核估计）；PocketTerm35 改全分辨率（颗粒感整改，2026-10）。
#define CLOUD_MAX_W (320)
#define CLOUD_MAX_H (240)

// 太阳位置预设：仰角（度）/方位角（度）/颜色/强度
typedef struct {
    int   elev_deg;
    int   azim_deg;
    Cv3   color;
    float intensity;
    const wchar_t *name;
} SunPreset;

static const SunPreset SUN_PRESETS[] = {
    { 82,  -10, {1.00f, 1.00f, 1.00f}, 48.0f, L"正午" },  // 白天纯白（同 ui_cloud_sun_color）
    { 45,   30, {1.00f, 1.00f, 1.00f}, 44.0f, L"午后" },
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
// 档位名称（玲珑仪“云量”按钮直接显示；档位语义的唯一真源在此处）
static const wchar_t *CLOUD_COVERAGE_NAMES[] = {
    L"晴空", L"疏云", L"半云", L"多云", L"阴天", L"满云",
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
#define CLOUD_BRIGHTNESS_DEFAULT (0.5f)



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
        col = v3(1.0f, 1.0f, 1.0f);  // 白天太阳纯白（Bruneton 模型的太阳辐照为白色；
                                       // 原 (1.0,0.93,0.82) 暖色是天空偏黄的主因）
        inten = 48.0f;
    } else if (elev_deg >= 0.0f) {
        // 低空橙 → 正午白（0°..15°）
        float t = elev_deg / 15.0f;
        col = v3lerp(v3(1.0f, 0.45f, 0.20f), v3(1.0f, 1.0f, 1.0f), t);
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

// 天空散射光源逐帧更新（linglong.html 模型5改进回移）：
// skySunColor 在太阳低空段混入近白（<4° 全白 → 4°..15° 渐回曲线色），并对 ≤0°
// 暮光施加蓝紫基调（0°→-8°：R-22%、G-6%、B+45%；工程近似高空多次散射与臭氧
// Chappuis 吸收的“蓝小时”观感，单遍 MS-LUT 无法把蓝紫输运到阴影区）；强度沿用
// 圆盘曲线（无亮度跳变）。云直射光与天空散射共用此光源；太阳圆盘/日冕/镜头
// 光晕仍用 sunColor 曲线（日落云色不变）。
static void cloud_update_sky_light(void) {
    float elevDeg = asinf(saturatef(s_scene.sunDirection.y)) * 180.0f / kPI;
    float tMix = saturatef(elevDeg / 4.0f);
    Cv3 sl = v3lerp(v3(1.0f, 1.0f, 1.0f), s_scene.sunColor, tMix);
    float kTwl = saturatef(-elevDeg / 8.0f);
    s_scene.skySunColor = v3(sl.x * (1.0f - 0.22f * kTwl),
                             sl.y * (1.0f - 0.06f * kTwl),
                             sl.z * (1.0f + 0.45f * kTwl));
    s_scene.skySunIntensity = s_scene.sunIntensity;
    // 夜天光：仰角表每帧随太阳更新（白天 vis=0，整表为 0，逐像素早退）
    linglong_night_sky_init(&s_night, (double)s_scene.sunDirection.x,
                            (double)s_scene.sunDirection.y,
                            (double)s_scene.sunDirection.z);
    s_night_vis = (float)linglong_night_visibility((double)s_scene.sunDirection.y);
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
    cloud_update_sky_light();
}

static void cloud_apply_sun_preset(int idx) {
    const SunPreset *p = &SUN_PRESETS[idx];
    float elev = (float)p->elev_deg * kPI / 180.0f;
    float azim = (float)p->azim_deg * kPI / 180.0f;
    s_scene.sunDirection = v3norm(v3(cosf(elev) * sinf(azim), sinf(elev), cosf(elev) * cosf(azim)));
    s_scene.sunColor = p->color;
    s_scene.sunIntensity = p->intensity;
    s_ui.sun_dirty = 1;
    cloud_update_sky_light();
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

const wchar_t *ui_cloud_coverage_level_name(int level) {
    if (level < 0) level = 0;
    if (level >= CLOUD_COVERAGE_LEVEL_NUM) level = CLOUD_COVERAGE_LEVEL_NUM - 1;
    return CLOUD_COVERAGE_NAMES[level];
}

// 云亮度随太阳高度的经验调制（与 linglong.html 的 cloud_sky_brightness_from_sun_alt 同式）：
//   0.75·tanh(0.2·h − 0.6) + 1.25
// tanh 的值域把结果天然夹在 [0.5, 2.0]，正好是 cloud_apply_brightness 的亮度区间：
//   h=+90° → 2.00   h=25° → 2.00   h=10° → 1.91   h=3° → 1.25
//   h= 0°  → 0.85   h=-6° → 0.54   h=-18° → 0.50
// 太阳越高云越白亮（上限 2.0），日落后转暗，天文暮光之后压到下限 0.5。
float ui_cloud_brightness_for_sun_alt(float elev_deg) {
    return 0.75f * tanhf(0.2f * elev_deg - 0.6f) + 1.25f;
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

// 饱和度提升系数：补偿逐通道 ACES 色调映射的去饱和（HDR 深蓝被洗淡是天空
// “不够蓝”的主因之一；白色云天饱和度为 0，不受影响）。经宿主机仿真标定。
#define CLOUD_SATURATION_BOOST (1.25f)

// 色调映射 → 显示空间 RGB（0..1）：ACES + 饱和度提升 + sRGB 伽马
static Cv3 cloud_tonemap(Cv3 hdr) {
    Cv3 t = cloudAcesFilm(hdr);
    // 饱和度提升（亮度保持，仅拉伸色度）
    float lum = 0.2126f * t.x + 0.7152f * t.y + 0.0722f * t.z;
    t = v3clamp(v3add(v3(lum, lum, lum), v3mul(v3sub(t, v3(lum, lum, lum)), CLOUD_SATURATION_BOOST)), 0.0f, 1.0f);
    t = v3(cloud_gamma(t.x), cloud_gamma(t.y), cloud_gamma(t.z)); // sRGB 伽马（查表）
    return v3clamp(t, 0.0f, 1.0f);
}

// 不透明像素写出（PASS_SKY / PASS_BOTH）
static void cloud_write_pixel(int32_t x, Cv3 hdr, uint8_t *rgb_line) {
    Cv3 t = cloud_tonemap(hdr);
    rgb_line[x * 3 + 0] = (uint8_t)(t.x * 255.0f + 0.5f);
    rgb_line[x * 3 + 1] = (uint8_t)(t.y * 255.0f + 0.5f);
    rgb_line[x * 3 + 2] = (uint8_t)(t.z * 255.0f + 0.5f);
}

// 云层像素写出（PASS_CLOUDS）：照搬 linglong.html render_decoupled_volumetric_clouds——
//   ① 先把累积内散射还原成“完全覆盖时的云色”（除以覆盖率），单独做一次 ACES；
//   ② 再乘回覆盖率，得到预乘前景色 rgb；alpha 存云透射率 T；
//   ③ 合成时 dst = dst·T + rgb，即标准 alpha 合成，两层只在 0..1 显示空间内混合，
//      不会因“各自色调映射后相加”而过曝，云边缘也不会出现暗边。
static void cloud_write_pixel_premul(int32_t x, Cv3 cloudHDR, float transmittance,
                                     uint8_t *rgb_line, uint8_t *a_line) {
    float T = clampf(transmittance, 0.0f, 1.0f);
    float cov = 1.0f - T;
    Cv3 d = v3(0, 0, 0);
    if (cov > 1e-3f) d = cloud_tonemap(v3mul(cloudHDR, 1.0f / cov));
    rgb_line[x * 3 + 0] = (uint8_t)(clampf(d.x * cov, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgb_line[x * 3 + 1] = (uint8_t)(clampf(d.y * cov, 0.0f, 1.0f) * 255.0f + 0.5f);
    rgb_line[x * 3 + 2] = (uint8_t)(clampf(d.z * cov, 0.0f, 1.0f) * 255.0f + 0.5f);
    a_line[x] = (uint8_t)(T * 255.0f + 0.5f);
}

// ===========================================================================
// 天空遍「地平线以下」方位角环缓存（性能专项 A10）
//
// PASS_SKY 已把射线仰角钳到 +0.0005（见 cloud_core_render 内注释），因此**整个
// 地平线以下的天空积分结果只取决于水平方向**（与射线原仰角无关）。按方位角分桶
// 后同一桶只算一次：半屏像素从「每个跑一道 31 步大气积分」变成「1 次 atan2f +
// 3 字节查表」。分桶步长 2π/2048 ≈ 0.18°，远小于一个像素
// 的角尺寸（160x120 鱼眼 150° 下≈0.9°/px）。
//
// 注：共享缓存与 OpenMP 多行并行写不安全（Linux 开发构建带 -fopenmp），故在
// _OPENMP 下整段编译成透传（逐像素照旧）；WASM / luckfox 均为单线程，走缓存。
// ===========================================================================
#define CLOUD_SKYRING_BINS (2048)
#if defined(_OPENMP)
#define CLOUD_SKYRING_ENABLE (0)
#else
#define CLOUD_SKYRING_ENABLE (1)
#endif
#if CLOUD_SKYRING_ENABLE
static uint8_t  s_skyring_rgb[CLOUD_SKYRING_BINS * 3];
static uint16_t s_skyring_epoch[CLOUD_SKYRING_BINS];
static uint16_t s_skyring_cur = 0;

static void cloud_skyring_begin(void) {
    s_skyring_cur++;
    if (s_skyring_cur == 0) {          // 回绕：清空后从 1 开始（epoch 0 即“未填”）
        memset(s_skyring_epoch, 0, sizeof(s_skyring_epoch));
        s_skyring_cur = 1;
    }
}

static inline int cloud_skyring_bin(Cv3 dir) {
    float a = atan2f(dir.z, dir.x) + kPI;                  // [0, 2π)
    int b = (int)(a * ((float)CLOUD_SKYRING_BINS / (2.0f * kPI)));
    if (b < 0) b = 0;
    else if (b >= CLOUD_SKYRING_BINS) b = CLOUD_SKYRING_BINS - 1;
    return b;
}

// 命中返回 1 并写入该像素行缓冲；未命中返回 0（调用方去算完整积分）
static int cloud_skyring_lookup(Cv3 dir, uint8_t *rgb_line, int x) {
    int b = cloud_skyring_bin(dir);
    if (s_skyring_epoch[b] != s_skyring_cur) return 0;
    const uint8_t *s = &s_skyring_rgb[b * 3];
    rgb_line[x * 3 + 0] = s[0];
    rgb_line[x * 3 + 1] = s[1];
    rgb_line[x * 3 + 2] = s[2];
    return 1;
}

static void cloud_skyring_store(Cv3 dir, Cv3 hdr) {
    int b = cloud_skyring_bin(dir);
    Cv3 t = cloud_tonemap(hdr);
    s_skyring_rgb[b * 3 + 0] = (uint8_t)(t.x * 255.0f + 0.5f);
    s_skyring_rgb[b * 3 + 1] = (uint8_t)(t.y * 255.0f + 0.5f);
    s_skyring_rgb[b * 3 + 2] = (uint8_t)(t.z * 255.0f + 0.5f);
    s_skyring_epoch[b] = s_skyring_cur;
}
#else
static void cloud_skyring_begin(void) {}
static int cloud_skyring_lookup(Cv3 dir, uint8_t *rgb_line, int x) {
    (void)dir; (void)rgb_line; (void)x; return 0;
}
static void cloud_skyring_store(Cv3 dir, Cv3 hdr) { (void)dir; (void)hdr; }
#endif

int32_t ui_cloud_init(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    // LUT 缓冲一次性申请（PSRAM，见 s_transLut/s_msLut 处注释）；失败则回主菜单
    if (cloud_ensure_luts_buffers() != 0) {
        global_state->STATE = STATE_MAIN_MENU;
        return -1;
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
    s_ui.layer_mask = CLOUD_LAYER_CU | CLOUD_LAYER_AC | CLOUD_LAYER_SC | CLOUD_LAYER_CS; // 默认所有云
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
    // 云层种类循环（1 键）：所有 → 浓积云 → 高积云 → 层积云 → 卷层云 → 无云
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        s_ui.layer_idx = (s_ui.layer_idx + 1) % 6;
        switch (s_ui.layer_idx) {
            case 0: s_ui.layer_mask = CLOUD_LAYER_CU | CLOUD_LAYER_AC | CLOUD_LAYER_SC | CLOUD_LAYER_CS; break; // 所有
            case 1: s_ui.layer_mask = CLOUD_LAYER_CU; break;   // 仅浓积云
            case 2: s_ui.layer_mask = CLOUD_LAYER_AC; break;   // 仅高积云
            case 3: s_ui.layer_mask = CLOUD_LAYER_SC; break;   // 仅层积云
            case 4: s_ui.layer_mask = CLOUD_LAYER_CS; break;   // 仅卷层云
            default: s_ui.layer_mask = 0; break;               // 无云
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
                             s_ui.layer_mask = CLOUD_LAYER_CU | CLOUD_LAYER_AC | CLOUD_LAYER_SC | CLOUD_LAYER_CS;
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

// 内部渲染缓冲（可降采样，动态分配）+ 全屏输出缓冲
// 另有一份“云透射率”平面（每像素 1 字节），仅 PASS_CLOUDS 需要：与预乘云色一起
// 双线性放大，供 ui_cloud_flush_over 做 dst = dst·T + premul_rgb 的 alpha 合成。
// 独立“体积云”演示 App 走 PASS_BOTH，不会申请这块内存。
static uint8_t *s_scene_img = NULL;
static size_t  s_scene_img_bytes = 0;
static uint8_t *s_fb = NULL;      // 全屏（整机帧缓冲尺寸）
static size_t  s_fb_bytes = 0;
static uint8_t *s_scene_a = NULL; // 低分辨率云透射率（iw*ih）
static size_t  s_scene_a_bytes = 0;
static uint8_t *s_fb_a = NULL;    // 全屏云透射率（fw*fh）
static size_t  s_fb_a_bytes = 0;
static int s_out_fw = 0, s_out_fh = 0; // 最近一帧的全屏尺寸（供 flush 使用）

static void cloud_ensure_buffers(int iw, int ih, int fw, int fh, int need_alpha) {
    size_t scene_need = (size_t)iw * ih * 3;
    if (scene_need > s_scene_img_bytes) {
        if (s_scene_img) free(s_scene_img);
        s_scene_img = (uint8_t *)malloc(scene_need);
        s_scene_img_bytes = s_scene_img ? scene_need : 0;
    }
    size_t fb_need = (size_t)fw * fh * 3;
    if (fb_need > s_fb_bytes) {
        if (s_fb) free(s_fb);
        s_fb = (uint8_t *)malloc(fb_need);
        s_fb_bytes = s_fb ? fb_need : 0;
    }
    if (need_alpha) {
        size_t sa_need = (size_t)iw * ih;
        if (sa_need > s_scene_a_bytes) {
            if (s_scene_a) free(s_scene_a);
            s_scene_a = (uint8_t *)malloc(sa_need);
            s_scene_a_bytes = s_scene_a ? sa_need : 0;
        }
        size_t fa_need = (size_t)fw * fh;
        if (fa_need > s_fb_a_bytes) {
            if (s_fb_a) free(s_fb_a);
            s_fb_a = (uint8_t *)malloc(fa_need);
            s_fb_a_bytes = s_fb_a ? fa_need : 0;
        }
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

// 双线性放大 s_scene_img/s_scene_a(iw,ih) → s_fb/s_fb_a(fw,fh)
// RGB 与透射率用同一套中心对齐权重（预乘色对覆盖率是线性的，可一起插值）。
static void cloud_upscale(int iw, int ih, int fw, int fh, int with_alpha) {
    const uint8_t *s = s_scene_img;
    uint8_t *d = s_fb;
    for (int Y = 0; Y < fh; Y++) {
        float sy = ((float)Y + 0.5f) * (float)ih / (float)fh - 0.5f;
        int y0 = (int)floorf(sy);
        y0 = (y0 < 0) ? 0 : (y0 > ih - 2 ? ih - 2 : y0);
        float fy = sy - (float)y0;
        // 边缘钳制（与 linglong.html 的 v=0/1 处理一致）：y0 被钳到 [0, ih-2] 后，
        // 首/末行的 sy-y0 会超出 [0,1]（如末行 1.25、首行 -0.25），直接当权重会
        // 外推采样值 → 画面最外一行偏黑/偏亮，云遍带抖动的预乘色还会逐帧闪。
        if (fy < 0.0f) fy = 0.0f;
        else if (fy > 1.0f) fy = 1.0f;
        const uint8_t *row0 = s + ((size_t)y0 * iw) * 3;
        const uint8_t *row1 = row0 + (size_t)iw * 3;
        uint8_t *out = d + ((size_t)Y * fw) * 3;
        const uint8_t *a0 = with_alpha ? (s_scene_a + (size_t)y0 * iw) : NULL;
        const uint8_t *a1 = with_alpha ? (a0 + iw) : NULL;
        uint8_t *aout = with_alpha ? (s_fb_a + (size_t)Y * fw) : NULL;
        for (int X = 0; X < fw; X++) {
            float sx = ((float)X + 0.5f) * (float)iw / (float)fw - 0.5f;
            int x0 = (int)floorf(sx);
            x0 = (x0 < 0) ? 0 : (x0 > iw - 2 ? iw - 2 : x0);
            float fx = sx - (float)x0;
            if (fx < 0.0f) fx = 0.0f;
            else if (fx > 1.0f) fx = 1.0f;   // 同上：首/末列不外推
            int a = x0 * 3, b2 = (x0 + 1) * 3;
            for (int c = 0; c < 3; c++) {
                float p0 = mixf(row0[a + c], row0[b2 + c], fx);
                float p1 = mixf(row1[a + c], row1[b2 + c], fx);
                out[c] = (uint8_t)(mixf(p0, p1, fy) + 0.5f);
            }
            if (with_alpha) {
                float q0 = mixf(a0[x0], a0[x0 + 1], fx);
                float q1 = mixf(a1[x0], a1[x0 + 1], fx);
                aout[X] = (uint8_t)(mixf(q0, q1, fy) + 0.5f);
            }
            out += 3;
        }
    }
}

static void cloud_setup_camera(float roll_rad, float cam_alt_km, Cv3 *camPos, Cv3 *forward, Cv3 *right, Cv3 *camUp) {
    // 观察高度（km，地面以上）：与 linglong.html 的 cloud_sky_sync_camera 同源。
    // 未传（<=0）时保持历史默认 2 km（独立“体积云”演示 App 的取景高度）。
    if (!(cam_alt_km > 0.0f)) cam_alt_km = 2.0f;
    if (cam_alt_km > 100.0f) cam_alt_km = 100.0f; // 大气顶 60 km，留余量
    *camPos = v3(0, s_scene.bottomRadius + cam_alt_km, 0);
    *forward = v3norm(v3(cosf(s_ui.pitch) * sinf(s_ui.yaw), sinf(s_ui.pitch),
                         cosf(s_ui.pitch) * cosf(s_ui.yaw)));
    Cv3 worldUp = v3(0, 1, 0);
    Cv3 r = v3cross(*forward, worldUp);
    if (v3len(r) < 1e-4f) r = v3(1, 0, 0);
    *right = v3norm(r);
    *camUp = v3cross(*right, *forward);
    *camUp = v3norm(*camUp);
    // 绕视线轴滚转（与天象仪 fisheye_project 的 roll 约定一致：正值为顺时针）
    // 注意：天象仪 fisheye_project 的屏幕 x 与 right 轴反号（screen_x ∝ -vx），
    // 因此本侧 roll 需与此处相反的 sin 符号才能产生相同的屏幕方向。
    if (roll_rad != 0.0f) {
        float cr = cosf(roll_rad), sr = sinf(roll_rad);
        Cv3 rr = v3add(v3mul(*right, cr), v3mul(*camUp, sr));
        Cv3 uu = v3add(v3mul(*right, -sr), v3mul(*camUp, cr));
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
    int pass = p->pass;
    if (pass < UI_CLOUD_PASS_BOTH || pass > UI_CLOUD_PASS_CLOUDS) pass = UI_CLOUD_PASS_BOTH;

    // 内部渲染分辨率 = 屏幕 / cloudRenderScale（上限 CLOUD_MAX_*）。
    // 倍数 2（默认）→ 160x120：ray-march 的逐像素成本是线性量，像素数÷4 即约 4× 提速；
    // 结果由 cloud_upscale 双线性放大（天空/云都是低频渐变，放大后观感损失远小于
    // 体积云步进噪声的损失）。倍数 1 时 cloud_upscale 退化为逐像素拷贝。
    int scale = s_scene.cloudRenderScale;
    if (scale < 1) scale = 1;
    int W = FW / scale, H = FH / scale;
    if (W < 8) W = 8;
    if (W > CLOUD_MAX_W) W = CLOUD_MAX_W;
    if (H < 8) H = 8;
    if (H > CLOUD_MAX_H) H = CLOUD_MAX_H;
    cloud_ensure_buffers(W, H, FW, FH, pass == UI_CLOUD_PASS_CLOUDS);
    s_out_fw = FW;
    s_out_fh = FH;
    // 缓冲不可用（内存不足）：跳过本帧（PASS_CLOUDS 时调用方只会得到“没有云”）
    if (s_scene_img == NULL || s_fb == NULL) return;
    if (pass == UI_CLOUD_PASS_CLOUDS && (s_scene_a == NULL || s_fb_a == NULL)) return;

    // 防御：LUT 未就绪（PSRAM 申请失败）时跳过本帧渲染（调用方负责退出状态）
    if (!s_transLut || !s_msLut) return;

    cloud_ensure_luts();

    Cv3 camPos, forward, right, camUp;
    cloud_setup_camera(p->roll_rad, p->cam_alt_km, &camPos, &forward, &right, &camUp);

    // 地面上涌光：天顶方向的天空亮度（除地面回弹的纯大气散射），加权漫反射等效
    {
        Cv3 upDir = v3(0, 1, 0);
        SingleScatteringResult zen = integrateScatteredLuminance(camPos, upDir, s_scene.sunDirection,
            &s_scene, 0, 1, 0, 1, 9e6f, 14, 0);
        // 地面上涌光：仅用天顶亮度×反照率 会骗低云底亮度（云底可视环境光≈地面漫反射+大气回光）。
        // 抬高尺度：上行辐照≈天顶亮度×(1+反照率) 的半球等效，再随云遮蔽淡出。
        // 夜间加夜天光天顶项：云的环境补光（ambientLit）由上涌光驱动，不补的话夜间云会
        // 变成纯黑剪影（只剩天空底）。
        Cv3 upL = v3add(zen.scatteredLight, cloud_night_sky(upDir));
        s_groundUpwelling = v3mul(v3mul3(upL,
            v3add(v3(1.0f, 1.0f, 1.0f), s_scene.groundAlbedo)), 4.0f);
    }

    // 透视投影：fov_deg 为纵向半角；tanV/tanH 用于屏幕射线与太阳反投影
    float tanV = tanf((float)p->fov_deg * kPI / 180.0f);
    float tanH = tanV * (float)W / (float)H;
    // 鱼眼投影：屏幕对角方向映射到 fov_deg（最大半角）
    const float fishRmax = sqrtf((float)(W * W + H * H) / (float)H / (float)H); // sqrt((W/H)^2+1)
    const float fishFovRad = (float)p->fov_deg * kPI / 180.0f;
    // 每像素角尺寸（rad/px）：浓积云按它把 3D 噪声低通到可分辨尺度（mip）
    const float pixAng = (p->proj == CLOUD_PROJ_FISHEYE)
        ? (fishFovRad / ((float)H * 0.5f * fishRmax))
        : (2.0f * tanV / (float)H);

    // 太阳镜头光晕（每帧一次）：太阳在相机前方时投影到屏幕 NDC，并求其被云遮挡的透射率
    s_ui.sun_visible = 0;
    // 太阳低于地平线时无条件关闭光晕（linglong.html 同步）：地平线下无直射光，
    // 避免太阳盘/光晕绘制在地平线以下的“黑色地面”上
    float sunElevDeg = asinf(saturatef(s_scene.sunDirection.y)) * 180.0f / kPI;
    if (p->enable_sun_lens && sunElevDeg >= 0.0f) {
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
            CloudPixel sunc = cloudPixelCompute(camPos, s_scene.sunDirection, p->app_time_sec, 0.5f, pixAng, &s_scene);
            s_ui.sun_vis = sunc.transmittance;
            s_ui.sun_visible = 1;
        }
        else {
            s_ui.sun_vis = 0.0f;
        }
    }

    // 半分辨率逐像素 ray-march 渲染
    // PASS_SKY：只积分大气；PASS_CLOUDS：只 ray-march 云（输出预乘色 + 透射率）；
    // PASS_BOTH：两者一次算完（历史行为）。
    // 天空遍的地平线以下按方位角分桶重用（性能专项 A10）：见循环内注释。
    if (pass == UI_CLOUD_PASS_SKY) cloud_skyring_begin();
    #pragma omp parallel for schedule(static) if (H >= 8)
    for (int y = 0; y < H; y++) {
        uint8_t rowbuf[CLOUD_MAX_W * 3];
        uint8_t rowa[CLOUD_MAX_W];
        float vv = 1.0f - ((float)y + 0.5f) / (float)H * 2.0f; // 上正
        for (int x = 0; x < W; x++) {
            float uu = ((float)x + 0.5f) / (float)W * 2.0f - 1.0f; // 左负右正

            // 屏幕射线：透视（pinhole）或等距鱼眼投影
            Cv3 rayDir;
            if (p->proj == CLOUD_PROJ_FISHEYE) {
                // 性能专项 A5：原实现逐像素 atan2f + 2×cosf + 2×sinf（天空遍/云遍各
                // 一遍，共 5 个超越函数/像素）。
                //   · ph 的 cos/sin 由恒等式直接给出：cos(atan2(fy,fx)) = fx/r、
                //     sin(...) = fy/r（零误差）；
                //   · th = (r/fishRmax)*fishFovRad 的 cos/sin 查 2048 项表 + 线性插值。
                float fx = uu * (float)W / (float)H;     // NDC 带 aspect
                float fy = vv;
                float r = sqrtf(fx * fx + fy * fy);
                float rn = r / fishRmax;                  // 对角=1
                float th = rn * fishFovRad;               // 等距角
                float cph, sph;
                if (r > 1e-12f) { cph = fx / r; sph = fy / r; }
                else            { cph = 1.0f;   sph = 0.0f; }
                float cth, sth;
                cloud_sincos(th, &sth, &cth);
                rayDir = v3add(v3mul(forward, cth),
                               v3mul(v3add(v3mul(right, cph), v3mul(camUp, sph)), sth));
            } else {
                rayDir = v3add(v3add(forward, v3mul(right, uu * tanH)), v3mul(camUp, vv * tanV));
            }
            rayDir = v3norm(rayDir);

            if (pass == UI_CLOUD_PASS_SKY) {
                // 只算天空（供解耦云层的底层；天体/地景随后叠加）。
                // 视线仰角钳到地平线以上 ε（同 linglong.html 的 wo_sky_atmo：dy<0.0005 → 0.0005）：
                // 内部 160x120 的“地平线行”在 ±0.43°，天顶/地平视角时上采样会按 50/50 把
                // 地平线下一行（地面回弹项，低空时暗红）混进地平线像素，形成一条绕地平线的
                // 暗带。钳到 +ε 后该行取“地平线天空”，与其它散射模型一致；真正的地平线以下
                // 区域仍由地景（海洋/黑色）覆盖。
                //
                // 性能专项 A10：钳位后**整个地平线以下的天空积分结果只取决于水平方向**
                // （与射线仰角无关），故按方位角分桶缓存，同一桶只算一次。半屏像素因此
                // 从“每个都跑 31 步大气积分”变成“查表 + 一个 atan2f”。分桶步长
                // 2π/2048 ≈ 0.18°，远小于一个像素的角尺寸；桶内首个像素的完整积分值
                // 就是该桶后续像素原本会算出的值（差异仅来自水平方向的桶内量化）。
                Cv3 skyDir = rayDir;
                if (skyDir.y < 0.0005f) {
                    skyDir.y = 0.0005f;
                    skyDir = v3norm(skyDir);
                    if (cloud_skyring_lookup(rayDir, rowbuf, x)) continue;
                }
                SingleScatteringResult sky = integrateScatteredLuminance(camPos, skyDir, s_scene.sunDirection,
                    &s_scene, 1, 1, 0, 1, 9e6f, 14, 0);
                // 叠夜天光（太阳落下后的气辉 + 星光 + 黄道光，见 cloud_night_sky）
                Cv3 skyL = v3add(sky.scatteredLight, cloud_night_sky(skyDir));
                if (rayDir.y < 0.0005f) cloud_skyring_store(rayDir, skyL);
                cloud_write_pixel(x, skyL, rowbuf);
                continue;
            }

            // per-pixel 时空去相关抖动（linglong.html 模型5改进回移，替代 IGN+线性帧偏移）：
            // 原实现每帧对整个 IGN 图案施加全局平移，平移量超过一个空间周期时发生时间
            // 混叠（“水波纹”式方向性滚动干扰）。改为对 (整数像素坐标, 帧号) 做
            // full-avalanche 哈希：像素与帧号均为独立哈希输入，帧间/像素间抖动互不相关，
            // 只呈现无结构的沸腾噪声，不再有可辨识的滚动图案。
            float jitter = cloudHashFloat((uint32_t)x, (uint32_t)y,
                                          (uint32_t)(s_ui.frame_index + 1013));

            // 云
            CloudPixel cloud = cloudPixelCompute(camPos, rayDir, p->app_time_sec, jitter, pixAng, &s_scene);

            if (pass == UI_CLOUD_PASS_CLOUDS) {
                // 解耦云层：输出预乘覆盖率的显示空间云色 + 透射率 alpha
                // （镜头光晕不在此处：天象仪自己画太阳/星芒；独立 App 走 PASS_BOTH）
                cloud_write_pixel_premul(x, cloud.color, cloud.transmittance, rowbuf, rowa);
                continue;
            }

            // PASS_BOTH：天空透过后方云层 + 云本身散射，再统一色调映射（历史行为）
            SingleScatteringResult sky = integrateScatteredLuminance(camPos, rayDir, s_scene.sunDirection,
                &s_scene, 1, 1, 0, 1, 9e6f, 14, 0);
            // 亮度已由介质光学参数（albedo/辐照/多重散射/环境）在着色阶段体现，不再后乘
            // 夜天光同样参与云的透射（云后面的微光夜空）
            Cv3 final = v3add(v3mul(v3add(sky.scatteredLight, cloud_night_sky(rayDir)), cloud.transmittance),
                              cloud.color);

            // 太阳镜头光晕（太阳/星芒/光斑，HDR 叠加，色调映射前）
            if (s_ui.sun_visible) {
                Cv3 lensAdd = cloud_lens_hdr(uu, vv, s_ui.sun_u, s_ui.sun_v, s_ui.sun_vis, &s_scene);
                final = v3add(final, lensAdd);
            }

            cloud_write_pixel(x, final, rowbuf);
        }
        memcpy(&s_scene_img[((size_t)y * W) * 3], rowbuf, W * 3);
        if (pass == UI_CLOUD_PASS_CLOUDS) memcpy(&s_scene_a[(size_t)y * W], rowa, W);
    }

    // 放大到整屏
    cloud_upscale(W, H, FW, FH, pass == UI_CLOUD_PASS_CLOUDS);
}

// 将外部参数应用到内部场景（光源/相机/云参数），并执行一次核心渲染。
// 独立应用与天象仪集成共用此入口，保证两端的体积云/大气渲染行为一致。
void ui_cloud_render_core(Nano_GFX *gfx, const UiCloud_Render_Params *p) {
    // LUT 缓冲首次分配：独立云应用由 ui_cloud_init 分配；天象仪等直接调用方
    // 也在此兜底分配，避免只渲染出黑屏（分配失败则本帧跳过，下帧重试）。
    if (cloud_ensure_luts_buffers() != 0) return;
    cloud_generate_noise_textures();
    static int s_scene_ready = 0;
    if (!s_scene_ready) {
        cloud_default_scene(&s_scene);
        s_scene_ready = 1;
    }

    // 光源由调用方给出（天象仪 → where_is_the_sun；独立应用 → 预设/自动运动）
    float dl = sqrtf(p->sun_dx * p->sun_dx + p->sun_dy * p->sun_dy + p->sun_dz * p->sun_dz);
    Cv3 newSunDir = (dl < 1e-6f) ? v3(0, 1, 0)
                                 : v3(p->sun_dx / dl, p->sun_dy / dl, p->sun_dz / dl);
    Cv3 newSunCol = v3(p->sun_r, p->sun_g, p->sun_b);
    // 太阳方向/颜色/强度变化超过阈值才置脏重算 LUT（linglong.html 同步的按需缓存）：
    // 独立应用每帧回填相同值时不再逐帧重建 LUT；天象仪太阳连续移动时仍逐帧更新
    if (fabsf(newSunDir.x - s_scene.sunDirection.x) +
        fabsf(newSunDir.y - s_scene.sunDirection.y) +
        fabsf(newSunDir.z - s_scene.sunDirection.z) > 1e-5f ||
        fabsf(newSunCol.x - s_scene.sunColor.x) +
        fabsf(newSunCol.y - s_scene.sunColor.y) +
        fabsf(newSunCol.z - s_scene.sunColor.z) > 1e-5f ||
        fabsf(p->sun_intensity - s_scene.sunIntensity) > 1e-5f) {
        s_ui.sun_dirty = 1;
    }
    s_scene.sunDirection = newSunDir;
    s_scene.sunColor = newSunCol;
    s_scene.sunIntensity = p->sun_intensity;
    cloud_update_sky_light();

    // 云参数（云量/云层种类/亮度沿用独立应用已有的控制曲线）
    s_scene.cloudCoverage = clampf(p->coverage, 0.0f, 1.0f);
    s_ui.layer_mask = p->layer_mask;
    cloud_apply_brightness(p->brightness);

    // 性能档位覆盖（0 = 用 CLOUD_OPT_* 的内部默认值；玲珑仪“云精度”按钮走这条路）
    if (p->render_scale > 0) s_scene.cloudRenderScale = p->render_scale;
    if (p->step_nominal_km > 0.0f) s_scene.cloudStepNominalKm = p->step_nominal_km;

    // 相机与投影（天象仪视角 → 云渲染 yaw/pitch/roll；独立应用直接传入自身状态）
    s_ui.yaw = p->yaw_rad;
    // 俯仰上限必须与天象仪 fisheye 的“天顶数值稳定角”（89.99°）对齐：原 ±85° 会让
    // 云/大气图像整体低 5°，天顶视角时画面底部 5° 露出云渲染器的地平线下地面项——
    // 即“画面下方固定宽度的暗带”。89.9° 时 cross(forward,worldUp) 仍非退化（≠0），
    // 屏幕基向量与 fisheye_unproject 一致；0.09° 残差 ≈ 0.1 像素。
    s_ui.pitch = clampf(p->pitch_rad, -89.9f * kPI / 180.0f, 89.9f * kPI / 180.0f);
    s_ui.roll = p->roll_rad;
    s_ui.proj = (p->proj == CLOUD_PROJ_FISHEYE) ? CLOUD_PROJ_FISHEYE : CLOUD_PROJ_PERSP;
    s_ui.fov_cur = p->fov_deg;
    if (s_ui.fov_cur < 1) s_ui.fov_cur = 1;
    if (s_ui.fov_cur > 170) s_ui.fov_cur = 170;
    s_ui.app_time = p->app_time_sec;
    // 解耦后一帧会调用两次（PASS_SKY + PASS_CLOUDS）：帧号只在“算云”的那一遍推进，
    // 保证抖动/动画序列每帧只前进一步。
    if (p->pass != UI_CLOUD_PASS_SKY) s_ui.frame_index++;

    cloud_core_render(gfx, p);
}

// 当前启用云层里最低的云底（km，地面以上）。云壳 0.8~11.0 km，按层带裁剪后
// 浓积云底 = 0.8 km（云底本身约 1.0 km，起伏 ±50 m）、高积云底 = 3.0 km、
// 层积云底 = 6.2 km、卷层云底 = 9.4 km（见 cloud_layer_band）。
// 无云层或场景尚未初始化时返回 -1。
float ui_cloud_lowest_layer_base_km(void) {
    if (s_scene.bottomRadius < 1.0f) return -1.0f;
    float h0 = 0.0f, h1 = 1.0f;
    if (!cloud_layer_band(&h0, &h1)) return -1.0f;
    return s_scene.cloudAreaStartHeight + h0 * s_scene.cloudAreaThickness - s_scene.bottomRadius;
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

// 预乘合成单通道：dst·T + premul（T=a/255，premul 已含 (1−T)，不要再乘）。
//
// 【必须钳到 255】premul 与 T 是两张平面**各自独立**双线性放大的：放大后同一像素的
// premul 与 T 不再严格满足 premul ≤ 1−T（原图里成立，插值后不成立），再加上整数
// 四舍五入，dst·T + premul 可以略微超过 1.0（实测 255.7）。直接 (uint8_t) 截断会把
// 256 回绕成 0：天空接近饱和白、云边缘又很亮时（太阳附近最典型），R 通道偶发回绕成
// 0 而 G/B 仍为 255 → 太阳附近随机跳动的青色/蓝色（或全通道回绕的黑色）单像素噪点。
// 因此这里先钳到 255 再转换（数学上 dst·T+premul ∈ [0,1]，钳位即还原其定义域）。
static inline uint8_t cloud_composite_ch(uint32_t dst, uint32_t a, uint32_t premul) {
    uint32_t v = (dst * a + premul * 255u + 127u) / 255u;
    return (uint8_t)((v > 255u) ? 255u : v);
}

// 将最近一帧 PASS_CLOUDS 的结果按标准 alpha 合成叠加到当前帧缓冲：
//   dst = dst·T + premul_rgb
// T 为云透射率（0=完全不透明云，255=无云），premul_rgb 为云色×(1−T)。
// 照搬 linglong.html render_decoupled_volumetric_clouds 的显示空间合成：
// 两层各自完成色调映射后只在 0..1 内混合，不会过曝；云隙处 T≈255 保留底层（地景/天空）。
void ui_cloud_flush_over(Nano_GFX *gfx) {
    const int FW = s_out_fw, FH = s_out_fh;
    if (s_fb == NULL || s_fb_a == NULL || FW <= 0 || FH <= 0) return;
    const int W = (int)gfx->width, H = (int)gfx->height;
    if (gfx->color_mode == GFX_COLOR_MODE_RGB888 && gfx->frame_buffer_rgb888 != NULL) {
        uint8_t *fb = gfx->frame_buffer_rgb888;
        for (int32_t y = 0; y < FH && y < H; y++) {
            for (int32_t x = 0; x < FW && x < W; x++) {
                size_t i = (size_t)y * FW + x;
                uint32_t a = s_fb_a[i];
                if (a == 255) continue;                       // 无云：完全保留底层
                uint8_t *d = fb + ((size_t)y * W + x) * 3;
                const uint8_t *s = s_fb + i * 3;
                if (a == 0) { d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; continue; }
                // 预乘合成：dst = dst·T + premul（s 已含 (1−T)，不要再乘；结果钳到 255，见上）
                d[0] = cloud_composite_ch(d[0], a, s[0]);
                d[1] = cloud_composite_ch(d[1], a, s[1]);
                d[2] = cloud_composite_ch(d[2], a, s[2]);
            }
        }
        return;
    }
    // 非 RGB888（RGB565 等）或尺寸不一致：逐像素读写
    for (int32_t y = 0; y < FH && y < H; y++) {
        for (int32_t x = 0; x < FW && x < W; x++) {
            size_t i = (size_t)y * FW + x;
            uint32_t a = s_fb_a[i];
            if (a == 255) continue;
            const uint8_t *s = s_fb + i * 3;
            if (a == 0) { gfx_set_pixel(gfx, (uint32_t)x, (uint32_t)y, s[0], s[1], s[2]); continue; }
            uint8_t dr = 0, dg = 0, db = 0;
            gfx_get_pixel(gfx, (uint32_t)x, (uint32_t)y, &dr, &dg, &db);
            // 预乘合成：dst = dst·T + premul（结果钳到 255，见上）
            gfx_set_pixel(gfx, (uint32_t)x, (uint32_t)y,
                cloud_composite_ch(dr, a, s[0]),
                cloud_composite_ch(dg, a, s[1]),
                cloud_composite_ch(db, a, s[2]));
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
    static const wchar_t *LAYER_NAMES[6] = { L"所有云", L"浓积云", L"高积云", L"层积云", L"卷层云", L"无云" };
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
