////////////////////////////////////////////////////////////////////////////
//  Native Library : Raster —— 水池 3D 软光栅**后端**（薄 C / 重 Scheme）
//
//  这一版与 (native Water)（am_native_Water.c，"整机封装"版）的分工不同：
//  本库只提供**逐像素 / 逐格点 / 逐三角形**的内层循环，扮演"固定功能 GPU"；
//  渲染器前端（几何、相机、绘制顺序、光照、物理、交互、HUD）全部留在 Scheme 里。
//
//  为什么边界在这一层（而不是更细或更粗）——实测归因（宿主 x86-64，steady state，
//  用 A/B/C 三个变体把「逐像素着色」「光栅化插值」「逐顶点+逐三角形 setup」拆开）：
//
//     pass      全量      只光栅化    只 setup     着色
//     池壁      3060 ms   383        0.5          2677
//     水面      5005 ms   980        816          4025
//     圆球       163 ms   111        86            52
//     焦散      2000 ms   —          大头（3872 三角形 × 3 顶点的法线/折射/投影） —
//     波场       307 ms   —          —            （16384 格点 × 4 个内核）
//     blit       135 ms   —          —            （76800 像素上屏）
//     相机/HUD     1 ms   —          —            ← 可以留在解释器
//
//  · **再细一档做不到**：让 Scheme 驱动"逐三角形 draw call"，光是水面的逐顶点
//    阶段就要 816 ms/帧（+ 圆球 86 + 焦散 ~900），是 10fps 预算（100ms）的十几倍。
//  · **再粗一档就是 Water 版**：连相机、网格、输入状态机、物理、HUD 都搬进 C，
//    Scheme 只剩 3 行骨架 —— 那是"整机封装"，不是"GPU 后端"。
//
//  所以：本库只管像素/格点/三角形内层循环 + 它必需的数据（帧缓冲、z-buffer、
//  波场双页、焦散纹理、已上传的网格）。**它不知道**相机角度、圆球物理、暂停/重力/
//  光源状态、控件栏、标题文字；每次 draw 的变换矩阵/相机位置/光源/圆球都是实参。
//
//  Scheme 侧（app/waterpool_raster.scm）负责：
//    · 建网格（水面平面 / 池壁（含 (1-c)·7/12−1 的池沿映射）/ UV 球）并上传；
//    · 相机：MV/P/MVP 组合、MVP 求逆解 eye、四角射线、逐像素射线（圆球拾取）；
//    · 绘制顺序与剔除模式（池壁 → 水面×2 → 圆球）、暂停时是否重画；
//    · 圆球物理（重力/浮力/黏滞/池底反弹）、光照方向、触屏交互、控件栏与标题、推屏。
//
//  栈纪律（遵循上游"编译期单值栈纪律"契约 doc/todo/stack_bal.md §11）：
//    值型函数（init）压 1 个值；语句型（mesh/begin/draw/sim_*/caustics/blit/dispose）
//    压 1 个 #undefined。任何 native 函数都不得压 0 个或 2 个以上值。
//
//  Scheme 接口一览（详细语义见 am_native_Raster.h）：
//      (Raster.init)                       按 Gfx 尺寸分配缓冲（幂等）
//      (Raster.mesh id verts tris)         上传网格（verts 每 5 个数一顶点：x,y,z,u,v）
//      (Raster.begin)                      清帧缓冲(黑) + 深度(远)
//      (Raster.draw mesh kind mvp params)  变换 + 剔除 + 光栅化 + 着色
//      (Raster.sim_move ox oy oz cx cy cz r) / (Raster.sim_step) / (Raster.sim_normals)
//      (Raster.caustics mesh ldx ldy ldz cx cy cz r)
//      (Raster.blit)                       RGB565 → 屏幕
//      (Raster.dispose)                    释放全部缓冲
//
//  渲染内核（波场 / 焦散 / 着色 / 光栅化）是 infer/ui_water.c 的**逐行副本**
//  （ui_water.c 一字未改），所以本 App 与设备侧 C 版水池的画面一致：同一份
//  float32 公式、同一网格细分、同一剔除/深度/混合次序。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

#include "am_native_Raster.h"

// ===============================================================================
// 原版常量（与 ui_water.c 一致：只保留后端用得上的）
// ===============================================================================
#define WT_IOR_AIR              (1.0f)
#define WT_IOR_WATER            (1.333f)
#define WT_POOL_HEIGHT          (1.0f)
#define WT_COLOR_ABOVEWATER_R   (0.25f)
#define WT_COLOR_ABOVEWATER_G   (1.00f)
#define WT_COLOR_ABOVEWATER_B   (1.25f)
#define WT_COLOR_UNDERWATER_R   (0.40f)
#define WT_COLOR_UNDERWATER_G   (0.90f)
#define WT_COLOR_UNDERWATER_B   (1.00f)
#define WT_COLOR_SKY2_R         (0.80f)
#define WT_COLOR_SKY2_G         (1.00f)
#define WT_COLOR_SKY2_B         (1.10f)

#define WT_GRID_N           (128)     // 水面高度场分辨率
#define WT_CAUSTIC_N        (256)     // 焦散纹理分辨率
#define WT_CELLS            (WT_GRID_N * WT_GRID_N)

// 网格上传上限（水面 2025 顶点 / 3872 三角形；圆球 740 / 1368；池壁 20 / 10）
#define RS_MAX_MESH         (4)
#define RS_MAX_VERTS        (4096)
#define RS_MAX_TRIS         (8192)

// 片元着色器种类（= 原版的三个 pass + 水面上下两遍）
#define RS_KIND_WALL        (0)   // 池壁：瓷砖 + AO + 焦散 + 水下透染
#define RS_KIND_WATER_ABOVE (1)   // 水面（水面以上视角：反射/折射 × 菲涅尔）
#define RS_KIND_WATER_BELOW (2)   // 水面（水面以下视角）
#define RS_KIND_SPHERE      (3)   // 圆球

// 剔除模式（原版三个 pass 的 CULL_FACE 策略，改由 Scheme 逐次传入）
#define RS_CULL_NONE        (0)
#define RS_CULL_NEG         (1)   // 只留屏幕带符号面积 < 0（池壁、水面 pass1）
#define RS_CULL_POS         (2)   // 只留屏幕带符号面积 >= 0（水面 pass0，cull FRONT）

// ===============================================================================
// 向量 / 矩阵（与 lightgl.js 一致：行主序、v'=M·v）
// ===============================================================================
typedef struct { float x, y, z; } V3;
typedef struct { float m[16]; } M4;

static V3 v3f(float x, float y, float z) { V3 v = {x, y, z}; return v; }
static V3 v3a(V3 a, V3 b) { return v3f(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 v3s(V3 a, V3 b) { return v3f(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 v3k(V3 a, float k) { return v3f(a.x * k, a.y * k, a.z * k); }
static V3 v3n(V3 a) { return v3f(-a.x, -a.y, -a.z); }
static float v3d(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 v3c(V3 a, V3 b) {
    return v3f(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static float v3l(V3 a) { return sqrtf(v3d(a, a)); }
static V3 v3u(V3 a) { float l = v3l(a); return v3f(a.x / l, a.y / l, a.z / l); }
static float v3clmp(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }

static V3 wt_reflect(V3 I, V3 N) { return v3s(I, v3k(N, 2.0f * v3d(N, I))); }
static V3 wt_refract(V3 I, V3 N, float eta, int *ok) {
    float dd = v3d(N, I);
    float k = 1.0f - eta * eta * (1.0f - dd * dd);
    if (k < 0.0f) { if (ok) *ok = 0; return v3f(0, 0, 0); }
    if (ok) *ok = 1;
    return v3s(v3k(I, eta), v3k(N, eta * dd + sqrtf(k)));
}

static void wt_mvp4(const M4 *M, float x, float y, float z, float *cx, float *cy, float *cz, float *cw) {
    const float *m = M->m;
    *cx = m[0] * x + m[1] * y + m[2] * z + m[3];
    *cy = m[4] * x + m[5] * y + m[6] * z + m[7];
    *cz = m[8] * x + m[9] * y + m[10] * z + m[11];
    *cw = m[12] * x + m[13] * y + m[14] * z + m[15];
}

// ===============================================================================
// 后端状态：帧缓冲 / 深度 / 波场双页 / 焦散纹理 + **本帧 uniform**（由 Scheme 传入）
//
// 字段名与 ui_water.c 的 WT_State 保持一致，使渲染内核可以逐行复制。
// ===============================================================================
typedef struct { float vx, vy, vz; float u, v; } WT_GeoV;    // 网格顶点（5 floats）

typedef struct {
    // ---- 每帧 uniform（Scheme 每次 draw / caustics 传入）----
    M4 MVP;
    V3 eye;              // 相机世界位置（水面着色需要）
    V3 lightDir;         // 定向光（单位向量；Scheme 算好）
    V3 center;           // 圆球中心
    float radius;        // 圆球半径
    // ---- 后端自持的数据 ----
    float *water[2];     // 每页：RGBA float = (h, v, n.x, n.z)
    int cur;
    float *caustR, *caustG;
    uint16_t *frame;     // RGB565 输出
    uint16_t *zbuf;
    int WW, HH;
    int ready;
} RS_State;

static RS_State s_wt;

// 已上传的网格
typedef struct {
    int used;
    int nv, nt;
    float *v;            // 5 floats / 顶点
    uint16_t *tri;       // 3 索引 / 三角形
} RS_Mesh;

static RS_Mesh s_mesh[RS_MAX_MESH];

// ===============================================================================
// 渲染内核（逐行复制自 infer/ui_water.c；ui_water.c 本身未改动）
// ===============================================================================

// ---- 水面高度场采样 / 焦散采样 / 波场 / 程序化天空与瓷砖 / 求交 / 着色 / 光栅化 ----
//        （以下逐行复制自 infer/ui_water.c）
static float wt_wat_at(const float *page, int x, int y, int ch) {
    if (x < 0) x = 0; else if (x > WT_GRID_N - 1) x = WT_GRID_N - 1;
    if (y < 0) y = 0; else if (y > WT_GRID_N - 1) y = WT_GRID_N - 1;
    return page[((y * WT_GRID_N) + x) * 4 + ch];
}

static float wt_wat_sample(const float *page, float u, float v, int ch) {
    float x = u * (float)WT_GRID_N - 0.5f;
    float y = v * (float)WT_GRID_N - 0.5f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = wt_wat_at(page, x0, y0, ch), b = wt_wat_at(page, x0 + 1, y0, ch);
    float c = wt_wat_at(page, x0, y0 + 1, ch), d = wt_wat_at(page, x0 + 1, y0 + 1, ch);
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
}

static void wt_caustic_sample(float u, float v, float *r, float *g) {
    u = v3clmp(u, 0, 1); v = v3clmp(v, 0, 1);
    float x = u * (float)WT_CAUSTIC_N - 0.5f;
    float y = v * (float)WT_CAUSTIC_N - 0.5f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = x0 < 0 ? 0 : (x0 > WT_CAUSTIC_N - 1 ? WT_CAUSTIC_N - 1 : x0);
    y0 = y0 < 0 ? 0 : (y0 > WT_CAUSTIC_N - 1 ? WT_CAUSTIC_N - 1 : y0);
    x1 = x1 < 0 ? 0 : (x1 > WT_CAUSTIC_N - 1 ? WT_CAUSTIC_N - 1 : x1);
    y1 = y1 < 0 ? 0 : (y1 > WT_CAUSTIC_N - 1 ? WT_CAUSTIC_N - 1 : y1);
    float fx = x - (float)floorf(x), fy = y - (float)floorf(y);
    float w0 = (1 - fx) * (1 - fy), w1 = fx * (1 - fy), w2 = (1 - fx) * fy, w3 = fx * fy;
    *r = s_wt.caustR[y0 * WT_CAUSTIC_N + x0] * w0 + s_wt.caustR[y0 * WT_CAUSTIC_N + x1] * w1
       + s_wt.caustR[y1 * WT_CAUSTIC_N + x0] * w2 + s_wt.caustR[y1 * WT_CAUSTIC_N + x1] * w3;
    *g = s_wt.caustG[y0 * WT_CAUSTIC_N + x0] * w0 + s_wt.caustG[y0 * WT_CAUSTIC_N + x1] * w1
       + s_wt.caustG[y1 * WT_CAUSTIC_N + x0] * w2 + s_wt.caustG[y1 * WT_CAUSTIC_N + x1] * w3;
}

static void wt_caustic_uv(V3 point, V3 refr, float *cu, float *cv) {
    *cu = 0.75f * (point.x - point.y * refr.x / refr.y) * 0.5f + 0.5f;
    *cv = 0.75f * (point.z - point.y * refr.z / refr.y) * 0.5f + 0.5f;
}

static void wt_step_sim(void) {
    const float *src = s_wt.water[s_wt.cur];
    float *dst = s_wt.water[1 - s_wt.cur];
    int nx = WT_GRID_N;
    for (int y = 0; y < WT_GRID_N; y++) {
        for (int x = 0; x < nx; x++) {
            int i = (y * nx + x) * 4;
            // 邻域（原版浮点纹理默认 CLAMP_TO_EDGE：边界以自身为邻）
            int lt = (x > 0) ? i - 4 : i;
            int rt = (x < nx - 1) ? i + 4 : i;
            int up = (y > 0) ? i - nx * 4 : i;
            int dn = (y < nx - 1) ? i + nx * 4 : i;
            float h = src[i], vv = src[i + 1];
            float avg = (src[lt] + src[rt] + src[up] + src[dn]) * 0.25f;
            vv += (avg - h) * 2.0f;                        // updateShader：info.g += (avg-h)*2
            vv *= 0.995f;                                  // updateShader：info.g *= 0.995
            h += vv;                                       // updateShader：info.r += info.g
            dst[i] = h; dst[i + 1] = vv;
            dst[i + 2] = src[i + 2]; dst[i + 3] = src[i + 3];
        }
    }
    s_wt.cur = 1 - s_wt.cur;
}

static void wt_update_normals(void) {
    float *pg = s_wt.water[s_wt.cur];
    int nx = WT_GRID_N;
    for (int y = 0; y < WT_GRID_N; y++) {
        for (int x = 0; x < nx; x++) {
            int i = (y * nx + x) * 4;
            int rt = (x < nx - 1) ? i + 4 : i;
            int dn = (y < nx - 1) ? i + nx * 4 : i;
            float h = pg[i];
            float dhdx = pg[rt] - h;                       // texture2D(P+dx).r - info.r
            float dhdy = pg[dn] - h;
            float delta = 1.0f / (float)WT_GRID_N;
            // dx=(delta,dhdx,0) dy=(0,dhdy,delta) → cross(dy,dx) ∝ (-dhdx,·,-dhdy)
            V3 nrm = v3f(-delta * dhdx, delta * delta, -delta * dhdy);
            V3 n = v3u(nrm);
            pg[i + 2] = n.x;                               // info.b
            pg[i + 3] = n.z;                               // info.a
        }
    }
}

static float wt_volume_in_sphere(V3 center, float u, float v) {
    V3 toCenter = v3f(u * 2.0f - 1.0f, 0.0f, v * 2.0f - 1.0f);
    toCenter = v3s(toCenter, center);
    float t = v3l(toCenter) / s_wt.radius;
    float dy = expf(-powf(t * 1.5f, 6.0f));
    float ymin = fminf(0.0f, center.y - dy);
    float ymax = fminf(fmaxf(0.0f, center.y + dy), ymin + 2.0f * dy);
    return (ymax - ymin) * 0.1f;
}

static void wt_move_sphere(V3 oldCenter, V3 newCenter) {
    if (v3d(v3s(oldCenter, newCenter), v3s(oldCenter, newCenter)) < 1e-12f) return;
    const float *src = s_wt.water[s_wt.cur];
    float *dst = s_wt.water[1 - s_wt.cur];
    for (int i = 0; i < WT_CELLS; i++) {
        int uidx = i % WT_GRID_N;
        int vidx = i / WT_GRID_N;
        float u = (float)uidx / (float)WT_GRID_N;
        float v = (float)vidx / (float)WT_GRID_N;
        dst[i * 4 + 0] = src[i * 4 + 0] + wt_volume_in_sphere(oldCenter, u, v)
                        - wt_volume_in_sphere(newCenter, u, v);   // info.r += old - new
        dst[i * 4 + 1] = src[i * 4 + 1];
        dst[i * 4 + 2] = src[i * 4 + 2];
        dst[i * 4 + 3] = src[i * 4 + 3];
    }
    s_wt.cur = 1 - s_wt.cur;
}

static void wt_sky_sample(V3 ray, float *r, float *g, float *b) {
    float up = fmaxf(0.0f, ray.y);
    float t = powf(fminf(1.0f, up * 1.7f), 0.85f);
    float rock = 0.04f * sinf(ray.x * 6.0f) * powf(up * 0.6f, 3.0f);
    float rr = (0.04f + 0.55f * t) + rock;
    float gg = (0.06f + 0.75f * t) + rock;
    float bb = (0.11f + 0.98f * t) + rock;
    *r = rr > 1 ? 1 : rr; if (rr < 0) *r = 0;
    *g = gg > 1 ? 1 : gg; if (gg < 0) *g = 0;
    *b = bb > 1 ? 1 : bb; if (bb < 0) *b = 0;
}

static void wt_tiles_sample(float tu, float tv, float *r, float *g, float *b) {
    const float T = 1.0f / 5.0f;
    float fu = floorf(tu / T), fv = floorf(tv / T);
    float xu = tu / T - fu, xv = tv / T - fv;
    int cu = (int)fmodf(fu, 2.0f), cv = (int)fmodf(fv, 2.0f);
    float shade = 1.0f - 0.06f * (float)((cu ^ cv) & 1);
    float grout = (xu < 0.05f || xv < 0.05f) ? 0.25f : 1.0f;
    *r = (0.42f + 0.20f * shade) * grout;
    *g = (0.50f + 0.20f * shade) * grout;
    *b = (0.56f + 0.20f * shade) * grout;
    if (*r > 1) *r = 1;
    if (*g > 1) *g = 1;
    if (*b > 1) *b = 1;
}

static void wt_intersect_cube(V3 o, V3 rd, V3 mn, V3 mx, float *tnear, float *tfar) {
    float tmnx  = (mn.x - o.x) / rd.x,  tmny = (mn.y - o.y) / rd.y, tmnz = (mn.z - o.z) / rd.z;
    float tmx   = (mx.x - o.x) / rd.x,  tmy  = (mx.y - o.y) / rd.y, tmxz = (mx.z - o.z) / rd.z;
    float t1x = fminf(tmnx, tmx), t1y = fminf(tmny, tmy), t1z = fminf(tmnz, tmxz);
    float t2x = fmaxf(tmnx, tmx), t2y = fmaxf(tmny, tmy), t2z = fmaxf(tmnz, tmxz);
    *tnear = fmaxf(fmaxf(t1x, t1y), t1z);
    *tfar  = fminf(fminf(t2x, t2y), t2z);
}

static float wt_intersect_sphere(V3 o, V3 rd, V3 center, float radius) {
    V3 toSphere = v3s(o, center);
    float a = v3d(rd, rd), b = 2.0f * v3d(toSphere, rd), c = v3d(toSphere, toSphere) - radius * radius;
    float disc = b * b - 4.0f * a * c;
    if (disc > 0.0f) {
        float t = (-b - sqrtf(disc)) / (2.0f * a);
        if (t > 0.0f) return t;
    }
    return 1.0e6f;
}

static void wt_get_sphere_color(V3 point, float *or, float *og, float *ob) {
    float r = s_wt.radius;
    float color = 0.5f;
    color *= 1.0f - 0.9f / powf((1.0f + r - fabsf(point.x)) / r, 3.0f);
    color *= 1.0f - 0.9f / powf((1.0f + r - fabsf(point.z)) / r, 3.0f);
    color *= 1.0f - 0.9f / powf((point.y + 1.0f + r) / r, 3.0f);

    V3 sphereNormal = v3k(v3s(point, s_wt.center), 1.0f / r);
    V3 refractedLight = wt_refract(v3n(s_wt.lightDir), v3f(0, 1, 0),
                                   WT_IOR_AIR / WT_IOR_WATER, NULL);
    float diffuse = fmaxf(0.0f, v3d(v3n(refractedLight), sphereNormal)) * 0.5f;
    float h = wt_wat_sample(s_wt.water[s_wt.cur], point.x * 0.5f + 0.5f,
                            point.z * 0.5f + 0.5f, 0);
    if (point.y < h) {
        float cu, cv, cr, cg;
        wt_caustic_uv(point, refractedLight, &cu, &cv);
        wt_caustic_sample(cu, cv, &cr, &cg);
        diffuse *= cr * 4.0f;
    }
    color += diffuse;
    *or = color; *og = color; *ob = color;
}

static void wt_get_wall_color(V3 point, float *or, float *og, float *ob) {
    float scale = 0.5f;
    float wallR, wallG, wallB, nX, nY, nZ;
    if (fabsf(point.x) > 0.999f) {
        wt_tiles_sample(point.y * 0.5f + 1.0f, point.z * 0.5f + 0.5f, &wallR, &wallG, &wallB);
        nX = -point.x; nY = 0; nZ = 0;
    } else if (fabsf(point.z) > 0.999f) {
        wt_tiles_sample(point.y * 0.5f + 1.0f, point.x * 0.5f + 0.5f, &wallR, &wallG, &wallB);
        nX = 0; nY = 0; nZ = -point.z;
    } else {
        wt_tiles_sample(point.x * 0.5f + 0.5f, point.z * 0.5f + 0.5f, &wallR, &wallG, &wallB);
        nX = 0; nY = 1; nZ = 0;
    }
    V3 normal = v3f(nX, nY, nZ);

    scale /= v3l(point);
    scale *= 1.0f - 0.9f / powf(v3l(v3s(point, s_wt.center)) / s_wt.radius, 4.0f);

    V3 refractedLight = v3n(wt_refract(v3n(s_wt.lightDir), v3f(0, 1, 0),
                                       WT_IOR_AIR / WT_IOR_WATER, NULL));
    float diffuse = fmaxf(0.0f, v3d(refractedLight, normal));
    float h = wt_wat_sample(s_wt.water[s_wt.cur], point.x * 0.5f + 0.5f,
                            point.z * 0.5f + 0.5f, 0);
    if (point.y < h) {
        float cu, cv, cr, cg;
        wt_caustic_uv(point, refractedLight, &cu, &cv);
        wt_caustic_sample(cu, cv, &cr, &cg);
        scale += diffuse * cr * 2.0f * cg;
    } else {
        V3 mn = v3f(-1, -WT_POOL_HEIGHT, -1), mx = v3f(1, 2, 1);
        float t0, t1;
        wt_intersect_cube(point, refractedLight, mn, mx, &t0, &t1);
        float rim = 1.0f / (1.0f + expf(-200.0f / (1.0f + 10.0f * (t1 - t0))
                          * (point.y + refractedLight.y * t1 - 2.0f / 12.0f)));
        diffuse *= rim;
        scale += diffuse * 0.5f;
    }
    *or = wallR * scale; *og = wallG * scale; *ob = wallB * scale;
}

static void wt_surface_ray_color(V3 origin, V3 ray, float waterR, float waterG, float waterB,
                                 float *or, float *og, float *ob) {
    V3 mn = v3f(-1, -WT_POOL_HEIGHT, -1), mx = v3f(1, 2, 1);
    float q = wt_intersect_sphere(origin, ray, s_wt.center, s_wt.radius);
    float cr, cg, cb;
    if (q < 1.0e6f) {
        wt_get_sphere_color(v3a(origin, v3k(ray, q)), &cr, &cg, &cb);
    } else if (ray.y < 0.0f) {
        float t0, t1;
        wt_intersect_cube(origin, ray, mn, mx, &t0, &t1);
        wt_get_wall_color(v3a(origin, v3k(ray, t1)), &cr, &cg, &cb);
    } else {
        float t0, t1;
        wt_intersect_cube(origin, ray, mn, mx, &t0, &t1);
        V3 hit = v3a(origin, v3k(ray, t1));
        if (hit.y < 2.0f / 12.0f) {
            wt_get_wall_color(hit, &cr, &cg, &cb);
        } else {
            wt_sky_sample(ray, &cr, &cg, &cb);
            float sun = powf(fmaxf(0.0f, v3d(s_wt.lightDir, ray)), 5000.0f);
            cr += sun * 10.0f; cg += sun * 8.0f; cb += sun * 6.0f;
        }
    }
    if (ray.y < 0.0f) { cr *= waterR; cg *= waterG; cb *= waterB; }
    *or = cr; *og = cg; *ob = cb;
}

typedef struct { float sx, sy, ndcZ, w; V3 world; float u, v; } WT_Vtx;
typedef void (*WT_Frag)(float b0, float b1, float b2,
                        const WT_Vtx *v0, const WT_Vtx *v1, const WT_Vtx *v2,
                        V3 world, float U, float V, float *r, float *g, float *b);


static void wt_ndc_to_screen(const M4 *MVP, V3 world, float u, float v, WT_Vtx *out) {
    float cx, cy, cz, cw;
    wt_mvp4(MVP, world.x, world.y, world.z, &cx, &cy, &cz, &cw);
    out->world = world; out->u = u; out->v = v;
    if (cw <= 0.0f) { out->w = 1.0f; out->ndcZ = 2.0f; out->sx = 0; out->sy = 0; return; }
    out->w = cw;
    float iw = 1.0f / cw;
    out->ndcZ = cz * iw;
    out->sx = (cx * iw * 0.5f + 0.5f) * (float)s_wt.WW;
    out->sy = (1.0f - (cy * iw * 0.5f + 0.5f)) * (float)s_wt.HH;
}

static void wt_fill(const WT_Vtx *v0, const WT_Vtx *v1, const WT_Vtx *v2, int use_depth, WT_Frag frag) {
    const float acc = 0.5f;   // 像素中心
    if (v0->w <= 0 || v1->w <= 0 || v2->w <= 0) return;
    float Ax = v0->sx, Ay = v0->sy, Bx = v1->sx, By = v1->sy, Cx = v2->sx, Cy = v2->sy;
    float e0 = (By - Cy) * (Ax - Cx) + (Cx - Bx) * (Ay - Cy);   // 2×面积（带符号）
    if (fabsf(e0) < 1e-6f) return;
    float rc = 1.0f / e0;
    float w0x = (By - Cy), w0y = (Cx - Bx), w1x = (Cy - Ay), w1y = (Ax - Cx);
    float minx = fminf(Ax, fminf(Bx, Cx)), maxx = fmaxf(Ax, fmaxf(Bx, Cx));
    float miny = fminf(Ay, fminf(By, Cy)), maxy = fmaxf(Ay, fmaxf(By, Cy));
    if (maxx < 0 || minx >= (float)s_wt.WW || maxy < 0 || miny >= (float)s_wt.HH) return;
    int x0 = (int)floorf(minx); if (x0 < 0) x0 = 0;
    int x1 = (int)ceilf(maxx);  if (x1 > s_wt.WW - 1) x1 = s_wt.WW - 1;
    int y0 = (int)floorf(miny); if (y0 < 0) y0 = 0;
    int y1 = (int)ceilf(maxy);  if (y1 > s_wt.HH - 1) y1 = s_wt.HH - 1;

    float iwv[3] = { 1.0f / v0->w, 1.0f / v1->w, 1.0f / v2->w };
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            float px = (float)x + acc, py = (float)y + acc;
            float b0 = (w0x * (px - Cx) + w0y * (py - Cy)) * rc;
            float b1 = (w1x * (px - Cx) + w1y * (py - Cy)) * rc;
            float b2 = 1.0f - b0 - b1;
            if (b0 < 0 || b1 < 0 || b2 < 0) continue;
            float denom = b0 * iwv[0] + b1 * iwv[1] + b2 * iwv[2];
            float ndcZ = (b0 * iwv[0] * v0->ndcZ + b1 * iwv[1] * v1->ndcZ + b2 * iwv[2] * v2->ndcZ) / denom;
            int idx = y * s_wt.WW + x;
            uint16_t d16 = (uint16_t)(ndcZ * 0.5f * 65535.0f + 32767.5f);   // [-1,1]→[0,65535]
            if (use_depth && d16 >= s_wt.zbuf[idx]) continue;
            V3 world;
            world.x = (b0 * iwv[0] * v0->world.x + b1 * iwv[1] * v1->world.x + b2 * iwv[2] * v2->world.x) / denom;
            world.y = (b0 * iwv[0] * v0->world.y + b1 * iwv[1] * v1->world.y + b2 * iwv[2] * v2->world.y) / denom;
            world.z = (b0 * iwv[0] * v0->world.z + b1 * iwv[1] * v1->world.z + b2 * iwv[2] * v2->world.z) / denom;
            float U = (b0 * iwv[0] * v0->u + b1 * iwv[1] * v1->u + b2 * iwv[2] * v2->u) / denom;
            float V = (b0 * iwv[0] * v0->v + b1 * iwv[1] * v1->v + b2 * iwv[2] * v2->v) / denom;
            float rr, gg, bb;
            frag(b0, b1, b2, v0, v1, v2, world, U, V, &rr, &gg, &bb);
            if (rr < 0) rr = 0; else if (rr > 1) rr = 1;
            if (gg < 0) gg = 0; else if (gg > 1) gg = 1;
            if (bb < 0) bb = 0; else if (bb > 1) bb = 1;
            s_wt.frame[idx] = (uint16_t)((((int)(rr * 31.0f + 0.5f) & 31) << 11)
                            | (((int)(gg * 63.0f + 0.5f) & 63) << 5)
                            | ((int)(bb * 31.0f + 0.5f) & 31));
            if (use_depth) s_wt.zbuf[idx] = d16;
        }
    }
}

static void wt_clear_frame(void) {
    memset(s_wt.frame, 0, (size_t)s_wt.WW * s_wt.HH * 2);
    memset(s_wt.zbuf, 0xFF, (size_t)s_wt.WW * s_wt.HH * 2);   // 深度置远
}

static void wt_wat_info_slot(const float *page, float u, float v, float *h, float *b, float *a) {
    *h = wt_wat_sample(page, u, v, 0);
    *b = wt_wat_sample(page, u, v, 2);
    *a = wt_wat_sample(page, u, v, 3);
}

static void wt_cube_frag(float b0, float b1, float b2, const WT_Vtx *v0, const WT_Vtx *v1,
                         const WT_Vtx *v2, V3 world, float U, float V,
                         float *or, float *og, float *ob) {
    (void)b0; (void)b1; (void)b2; (void)v0; (void)v1; (void)v2; (void)U; (void)V;
    float cr, cg, cb;
    wt_get_wall_color(world, &cr, &cg, &cb);
    float h = wt_wat_sample(s_wt.water[s_wt.cur], world.x * 0.5f + 0.5f, world.z * 0.5f + 0.5f, 0);
    if (world.y < h) {
        cr *= WT_COLOR_UNDERWATER_R * 1.2f;
        cg *= WT_COLOR_UNDERWATER_G * 1.2f;
        cb *= WT_COLOR_UNDERWATER_B * 1.2f;
    }
    *or = cr; *og = cg; *ob = cb;
}

static void wt_water_frag(int pass, float b0, float b1, float b2, const WT_Vtx *v0,
                          const WT_Vtx *v1, const WT_Vtx *v2, V3 world,
                          float U, float V, float *or, float *og, float *ob) {
    (void)b0; (void)b1; (void)b2; (void)v0; (void)v1; (void)v2;
    const float *page = s_wt.water[s_wt.cur];
    float coordU = U, coordV = V;
    float infoB, infoA;
    float h_;
    for (int i = 0; i < 5; i++) {          // “peaked”：5 次沿切线方向偏移重采样
        wt_wat_info_slot(page, coordU, coordV, &h_, &infoB, &infoA);
        coordU += infoB * 0.005f;
        coordV += infoA * 0.005f;
    }
    V3 normal = v3f(infoB, sqrtf(fmaxf(0.0f, 1.0f - (infoB * infoB + infoA * infoA))), infoA);
    V3 incomingRay = v3u(v3s(world, s_wt.eye));

    if (pass == 0) {
        V3 refl = wt_reflect(incomingRay, normal);
        V3 refr = wt_refract(incomingRay, normal, WT_IOR_AIR / WT_IOR_WATER, NULL);
        float fresnel = 0.25f + 0.75f * powf(1.0f - v3d(normal, v3n(incomingRay)), 3.0f);
        float rr, rg, rb, er, eg, eb;
        wt_surface_ray_color(world, refl, WT_COLOR_ABOVEWATER_R, WT_COLOR_ABOVEWATER_G, WT_COLOR_ABOVEWATER_B, &rr, &rg, &rb);
        wt_surface_ray_color(world, refr, WT_COLOR_ABOVEWATER_R, WT_COLOR_ABOVEWATER_G, WT_COLOR_ABOVEWATER_B, &er, &eg, &eb);
        *or = er + (rr - er) * fresnel;
        *og = eg + (rg - eg) * fresnel;
        *ob = eb + (rb - eb) * fresnel;
    } else {
        normal = v3n(normal);
        V3 refl = wt_reflect(incomingRay, normal);
        V3 refr = wt_refract(incomingRay, normal, WT_IOR_WATER / WT_IOR_AIR, NULL);
        float fresnel = 0.5f + 0.5f * powf(1.0f - v3d(normal, v3n(incomingRay)), 3.0f);
        float rr, rg, rb, er, eg, eb;
        wt_surface_ray_color(world, refl, WT_COLOR_UNDERWATER_R, WT_COLOR_UNDERWATER_G, WT_COLOR_UNDERWATER_B, &rr, &rg, &rb);
        wt_surface_ray_color(world, refr, 1, 1, 1, &er, &eg, &eb);
        er *= WT_COLOR_SKY2_R; eg *= WT_COLOR_SKY2_G; eb *= WT_COLOR_SKY2_B;
        float t = (1.0f - fresnel) * v3l(refr);
        *or = rr + (er - rr) * t;
        *og = rg + (eg - rg) * t;
        *ob = rb + (eb - rb) * t;
    }
}

static void wt_water_frag0(float b0, float b1, float b2, const WT_Vtx *v0, const WT_Vtx *v1,
                           const WT_Vtx *v2, V3 w, float U, float V, float *r, float *g, float *b) {
    wt_water_frag(0, b0, b1, b2, v0, v1, v2, w, U, V, r, g, b);
}

static void wt_water_frag1(float b0, float b1, float b2, const WT_Vtx *v0, const WT_Vtx *v1,
                           const WT_Vtx *v2, V3 w, float U, float V, float *r, float *g, float *b) {
    wt_water_frag(1, b0, b1, b2, v0, v1, v2, w, U, V, r, g, b);
}

static void wt_sphere_frag(float b0, float b1, float b2, const WT_Vtx *v0, const WT_Vtx *v1,
                           const WT_Vtx *v2, V3 world, float U, float V,
                           float *or, float *og, float *ob) {
    (void)b0; (void)b1; (void)b2; (void)v0; (void)v1; (void)v2; (void)U; (void)V;
    float cr, cg, cb;
    wt_get_sphere_color(world, &cr, &cg, &cb);
    float h = wt_wat_sample(s_wt.water[s_wt.cur], world.x * 0.5f + 0.5f, world.z * 0.5f + 0.5f, 0);
    if (world.y < h) {
        cr *= WT_COLOR_UNDERWATER_R * 1.2f;
        cg *= WT_COLOR_UNDERWATER_G * 1.2f;
        cb *= WT_COLOR_UNDERWATER_B * 1.2f;
    }
    *or = cr; *og = cg; *ob = cb;
}

static float wt_screen_area(const WT_Vtx *a, const WT_Vtx *b, const WT_Vtx *c) {
    return (b->sx - a->sx) * (c->sy - a->sy) - (b->sy - a->sy) * (c->sx - a->sx);
}

static void wt_caustic_project(V3 origin, V3 ray, V3 refractedLight, V3 *out) {
    V3 mn = v3f(-1, -WT_POOL_HEIGHT, -1), mx = v3f(1, 2, 1);
    float t0, t1;
    wt_intersect_cube(origin, ray, mn, mx, &t0, &t1);
    origin = v3a(origin, v3k(ray, t1));
    float tplane = (-origin.y - 1.0f) / refractedLight.y;
    *out = v3a(origin, v3k(refractedLight, tplane));
}

static void wt_update_caustics(const WT_GeoV *wm, const uint16_t *wtris, int ntris) {
    const float *page = s_wt.water[s_wt.cur];
    const float C = (float)WT_CAUSTIC_N;
    memset(s_wt.caustR, 0, (size_t)WT_CAUSTIC_N * WT_CAUSTIC_N * 4);
    memset(s_wt.caustG, 0, (size_t)WT_CAUSTIC_N * WT_CAUSTIC_N * 4);

    V3 refractedLight = wt_refract(v3n(s_wt.lightDir), v3f(0, 1, 0),
                                   WT_IOR_AIR / WT_IOR_WATER, NULL);

    for (int t = 0; t < ntris; t++) {
        const uint16_t *ix = &wtris[t * 3];
        // 每顶点：info → normal → ray；oldPos/newPos（world）与 NDC 屏幕位置
        struct { float sx, sy; V3 oldP, newP; float ratio; } vrt[3];
        V3 oldV[3], newV[3];
        for (int k = 0; k < 3; k++) {
            const WT_GeoV *g = &wm[ix[k]];
            float hk, b, a;
            wt_wat_info_slot(page, g->u, g->v, &hk, &b, &a);
            b *= 0.5f; a *= 0.5f;                       // info.ba *= 0.5
            V3 normal = v3f(b, sqrtf(fmaxf(0.0f, 1.0f - (b * b + a * a))), a);
            V3 ray = wt_refract(v3n(s_wt.lightDir), normal, WT_IOR_AIR / WT_IOR_WATER, NULL);
            V3 xz = v3f(g->vx, 0, g->vy);               // gl_Vertex.xzy
            V3 rl = refractedLight;
            wt_caustic_project(xz, rl, rl, &oldV[k]);
            wt_caustic_project(v3f(g->vx, hk, g->vy), ray, rl, &newV[k]);
            // gl_Position = (0.75*(newPos.xz + refr.xz/refr.y), 0, 1)
            float ndx = 0.75f * (newV[k].x + rl.x / rl.y);
            float ndy = 0.75f * (newV[k].z + rl.z / rl.y);
            vrt[k].sx = (ndx * 0.5f + 0.5f) * C;
            vrt[k].sy = (1.0f - (ndy * 0.5f + 0.5f)) * C;
            vrt[k].oldP = oldV[k]; vrt[k].newP = newV[k];
        }
        // 该三角形 old/new 面积比（光强聚焦度，等价 dFdx/dFdy 的比值）
        V3 o1 = v3s(vrt[1].oldP, vrt[0].oldP), o2 = v3s(vrt[2].oldP, vrt[0].oldP);
        V3 n1 = v3s(vrt[1].newP, vrt[0].newP), n2 = v3s(vrt[2].newP, vrt[0].newP);
        float oldArea = v3l(v3c(o1, o2));
        float newArea = v3l(v3c(n1, n2));
        vrt[0].ratio = vrt[1].ratio = vrt[2].ratio = (newArea < 1e-9f) ? 0.0f : oldArea / newArea;

        // 包围盒光栅化（无深度、无剔除）
        float minx = fminf(vrt[0].sx, fminf(vrt[1].sx, vrt[2].sx));
        float maxx = fmaxf(vrt[0].sx, fmaxf(vrt[1].sx, vrt[2].sx));
        float miny = fminf(vrt[0].sy, fminf(vrt[1].sy, vrt[2].sy));
        float maxy = fmaxf(vrt[0].sy, fmaxf(vrt[1].sy, vrt[2].sy));
        if (maxx < 0 || minx >= C || maxy < 0 || miny >= C) continue;
        int x0 = (int)floorf(minx); if (x0 < 0) x0 = 0;
        int x1 = (int)ceilf(maxx);  if (x1 > WT_CAUSTIC_N - 1) x1 = WT_CAUSTIC_N - 1;
        int y0 = (int)floorf(miny); if (y0 < 0) y0 = 0;
        int y1 = (int)ceilf(maxy);  if (y1 > WT_CAUSTIC_N - 1) y1 = WT_CAUSTIC_N - 1;
        float Ax = vrt[0].sx, Ay = vrt[0].sy, Bx = vrt[1].sx, By = vrt[1].sy, Cx = vrt[2].sx, Cy = vrt[2].sy;
        float e0 = (By - Cy) * (Ax - Cx) + (Cx - Bx) * (Ay - Cy);
        if (fabsf(e0) < 1e-6f) continue;
        float rc = 1.0f / e0;
        float w0x = (By - Cy), w0y = (Cx - Bx), w1x = (Cy - Ay), w1y = (Ax - Cx);
        V3 rl = refractedLight;
        V3 sphereC = s_wt.center; float r = s_wt.radius;
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                float px = (float)x + 0.5f, py = (float)y + 0.5f;
                float b0 = (w0x * (px - Cx) + w0y * (py - Cy)) * rc;
                float b1 = (w1x * (px - Cx) + w1y * (py - Cy)) * rc;
                float b2 = 1 - b0 - b1;
                if (b0 < 0 || b1 < 0 || b2 < 0) continue;
                V3 newPos;
                newPos.x = b0 * vrt[0].newP.x + b1 * vrt[1].newP.x + b2 * vrt[2].newP.x;
                newPos.y = b0 * vrt[0].newP.y + b1 * vrt[1].newP.y + b2 * vrt[2].newP.y;
                newPos.z = b0 * vrt[0].newP.z + b1 * vrt[1].newP.z + b2 * vrt[2].newP.z;
                float ratio = b0 * vrt[0].ratio + b1 * vrt[1].ratio + b2 * vrt[2].ratio;
                // 圆球挡光（blob shadow）
                V3 dir = v3k(v3s(sphereC, newPos), 1.0f / r);
                V3 area = v3c(dir, rl);
                float sh = v3d(area, area);
                float dist = v3d(dir, v3n(rl));
                sh = 1.0f + (sh - 1.0f) / (0.05f + dist * 0.025f);
                sh = v3clmp(1.0f / (1.0f + expf(-sh)), 0.0f, 1.0f);
                sh = 1.0f + (sh - 1.0f) * v3clmp(dist * 2.0f, 0.0f, 1.0f);   // mix(1, shadow, clamp(dist*2,0,1))
                // 池沿阴影（rim）
                V3 mn = v3f(-1, -WT_POOL_HEIGHT, -1), mx = v3f(1, 2, 1);
                float t0, t1;
                wt_intersect_cube(newPos, v3n(rl), mn, mx, &t0, &t1);
                float rim = 1.0f / (1.0f + expf(-200.0f / (1.0f + 10.0f * (t1 - t0))
                                  * (newPos.y - rl.y * t1 - 2.0f / 12.0f)));
                int idx = y * WT_CAUSTIC_N + x;
                s_wt.caustR[idx] = ratio * 0.2f * rim;
                s_wt.caustG[idx] = sh;
            }
        }
    }
}

// ===============================================================================
// 顶点处理 + 逐三角形绘制（原版 wt_render_cube / wt_render_water / wt_render_sphere
// 三个 pass 的公共骨架，改成"按 kind 选顶点处理与片元着色器、剔除模式由调用方给"；
// 三角形次序、深度测试、着色调用与三个原函数逐一对应）
// ===============================================================================
static void rs_vertex(int kind, const float *v, V3 *w, float *ou, float *ov) {
    if (kind == RS_KIND_WATER_ABOVE || kind == RS_KIND_WATER_BELOW) {
        // 水面：网格存平面坐标 (x, ·, z) 与纹理坐标 (u,v)；高度取自波场（同原版
        // wt_water_vert_world：world.y = info.r）
        float u = v[3], vv = v[4];
        *ou = u; *ov = vv;
        w->x = v[0];
        w->z = v[1];          // 注意：水面网格沿用原版 WT_GeoV 布局——(vx, vy) 是平面 (x, z)
        w->y = wt_wat_sample(s_wt.water[s_wt.cur], u, vv, 0);
    } else if (kind == RS_KIND_SPHERE) {
        // 圆球：网格是单位球，按 center/radius 缩放（同原版 wt_render_sphere）
        *ou = 0.0f; *ov = 0.0f;
        w->x = s_wt.center.x + v[0] * s_wt.radius;
        w->y = s_wt.center.y + v[1] * s_wt.radius;
        w->z = s_wt.center.z + v[2] * s_wt.radius;
    } else {
        // 池壁：网格顶点已是世界坐标（池沿的 (1−c)·7/12−1 映射由 Scheme 建网格时算好）
        *ou = 0.0f; *ov = 0.0f;
        w->x = v[0];
        w->y = v[1];
        w->z = v[2];
    }
}

static void rs_draw(int mesh_id, int kind, int cull) {
    RS_Mesh *m;
    WT_Frag frag;

    if (mesh_id < 0 || mesh_id >= RS_MAX_MESH) return;
    m = &s_mesh[mesh_id];
    if (!m->used) return;

    frag = (kind == RS_KIND_WALL)        ? wt_cube_frag
         : (kind == RS_KIND_WATER_ABOVE) ? wt_water_frag0
         : (kind == RS_KIND_WATER_BELOW) ? wt_water_frag1
         :                                 wt_sphere_frag;

    for (int t = 0; t < m->nt; t++) {
        const uint16_t *ix = &m->tri[t * 3];
        V3 w[3];
        float uu[3], vv[3];
        WT_Vtx vt[3];
        float area;
        int bad = 0;

        for (int k = 0; k < 3; k++) {
            if (ix[k] >= m->nv) { bad = 1; break; }
            rs_vertex(kind, &m->v[(size_t)ix[k] * 5], &w[k], &uu[k], &vv[k]);
            wt_ndc_to_screen(&s_wt.MVP, w[k], uu[k], vv[k], &vt[k]);
            if (vt[k].w <= 0.0f) bad = 1;
        }
        if (bad) continue;

        // 屏幕空间带符号面积剔除（策略由 Scheme 逐次指定）
        area = wt_screen_area(&vt[0], &vt[1], &vt[2]);
        if (cull == RS_CULL_NEG && area >= 0.0f) continue;
        if (cull == RS_CULL_POS && area < 0.0f) continue;

        wt_fill(&vt[0], &vt[1], &vt[2], 1, frag);
    }
}

// ===============================================================================
// 本地库外壳：绘图目标绑定、操作数栈辅助
// ===============================================================================
static Nano_GFX *s_gfx = NULL;

void am_native_raster_bind(Nano_GFX *gfx) {
    s_gfx = gfx;
}

static Nano_GFX *native_gfx(am_runtime_t *rt) {
    if (s_gfx == NULL && rt != NULL) {
        am_runtime_error(rt, L"[Raster] 未绑定绘图目标（宿主未调用 am_native_raster_bind）\n");
    }
    return s_gfx;
}

static bool native_pop_number(am_process_t *proc, am_float_t *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (am_value_is_float(v)) { *out = am_value_to_float(v); return true; }
    if (am_value_is_int(v))   { *out = (am_float_t)am_value_to_int(v); return true; }
    if (am_value_is_uint(v))  { *out = (am_float_t)am_value_to_uint(v); return true; }
    return false;
}

static bool native_pop_int(am_process_t *proc, int32_t *out) {
    am_float_t f;
    if (!native_pop_number(proc, &f)) return false;
    if (isnan(f)) return false;
    if (f > 2147483647.0)  f = 2147483647.0;
    if (f < -2147483648.0) f = -2147483648.0;
    *out = (int32_t)f;
    return true;
}

// 弹出一个列表，把它前面的 n 个数值拷进 out（不足则补 0）。成功返回 true。
static bool native_pop_float_list(am_process_t *proc, float *out, int n) {
    am_value_t v = am_process_pop_operand(proc);
    am_handle_t hd;
    am_value_t obj_val;
    am_object_t *obj;
    am_list_t *lst;

    for (int i = 0; i < n; i++) out[i] = 0.0f;
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;
    hd = am_value_to_handle(v);
    obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;
    obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return false;
    lst = (am_list_t *)obj;
    for (int i = 0; i < n && (size_t)i < lst->length; i++) {
        am_value_t e = lst->children[i];
        if (am_value_is_float(e))      out[i] = (float)am_value_to_float(e);
        else if (am_value_is_int(e))   out[i] = (float)am_value_to_int(e);
        else if (am_value_is_uint(e))  out[i] = (float)am_value_to_uint(e);
        else return false;
    }
    return true;
}

static int32_t native_statement_done(am_process_t *proc) {
    if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t native_push_number(am_process_t *proc, am_float_t result) {
    if (am_process_push_operand(proc, am_make_value_of_float(result)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// ===============================================================================
// 缓冲分配 / 释放（= ui_water.c 的 wt_alloc 中属于后端的那一部分）
// ===============================================================================
static void rs_free(void) {
    if (s_wt.water[0] != NULL) { free(s_wt.water[0]); s_wt.water[0] = NULL; s_wt.water[1] = NULL; }
    if (s_wt.caustR != NULL)   { free(s_wt.caustR);   s_wt.caustR = NULL; }
    if (s_wt.caustG != NULL)   { free(s_wt.caustG);   s_wt.caustG = NULL; }
    if (s_wt.frame != NULL)    { free(s_wt.frame);    s_wt.frame = NULL; }
    if (s_wt.zbuf != NULL)     { free(s_wt.zbuf);     s_wt.zbuf = NULL; }
    memset(&s_wt, 0, sizeof(s_wt));
    for (int i = 0; i < RS_MAX_MESH; i++) {
        if (s_mesh[i].v != NULL)   { free(s_mesh[i].v);   s_mesh[i].v = NULL; }
        if (s_mesh[i].tri != NULL) { free(s_mesh[i].tri); s_mesh[i].tri = NULL; }
        s_mesh[i].used = 0;
        s_mesh[i].nv = s_mesh[i].nt = 0;
    }
}

static int rs_alloc(int WW, int HH) {
    rs_free();
    s_wt.WW = WW;
    s_wt.HH = HH;
    if (s_wt.WW <= 0 || s_wt.HH <= 0) { s_wt.WW = 320; s_wt.HH = 240; }

    s_wt.water[0] = (float *)platform_calloc((size_t)(2 * WT_CELLS * 4), sizeof(float));
    s_wt.water[1] = s_wt.water[0] + (size_t)WT_CELLS * 4;   // 两页连排
    s_wt.caustR = (float *)platform_calloc((size_t)WT_CAUSTIC_N * WT_CAUSTIC_N, sizeof(float));
    s_wt.caustG = (float *)platform_calloc((size_t)WT_CAUSTIC_N * WT_CAUSTIC_N, sizeof(float));
    s_wt.frame  = (uint16_t *)platform_calloc((size_t)s_wt.WW * s_wt.HH, sizeof(uint16_t));
    s_wt.zbuf   = (uint16_t *)platform_calloc((size_t)s_wt.WW * s_wt.HH, sizeof(uint16_t));
    if (s_wt.water[0] == NULL || s_wt.caustR == NULL || s_wt.caustG == NULL ||
        s_wt.frame == NULL || s_wt.zbuf == NULL) {
        rs_free();
        return 0;
    }
    s_wt.cur = 0;
    s_wt.ready = 1;
    return 1;
}

// ===============================================================================
// Scheme 可见过程
// ===============================================================================

// (Raster.init) : Number —— 按 Gfx 尺寸分配缓冲（幂等：已有则先整体释放）；成功 1 / 失败 0
int32_t am_native_Raster_init(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    return native_push_number(proc, (am_float_t)rs_alloc((int)g->width, (int)g->height));
}

// (Raster.mesh id verts tris) : #undefined —— 上传网格
//   verts：每 5 个数一个顶点 (x, y, z, u, v)；tris：每 3 个数一个三角形（顶点下标）
int32_t am_native_Raster_mesh(am_runtime_t *rt, am_process_t *proc) {
    am_value_t v_tris, v_verts, v_id;
    int32_t id;
    am_list_t *lv, *lt;
    int nv, nt, i;

    (void)rt;
    // 实参从右往左弹
    v_tris = am_process_pop_operand(proc);
    v_verts = am_process_pop_operand(proc);
    v_id = am_process_pop_operand(proc);
    if (!am_value_is_int(v_id) && !am_value_is_uint(v_id) && !am_value_is_float(v_id)) return -1;
    id = am_value_is_float(v_id) ? (int32_t)am_value_to_float(v_id)
       : am_value_is_int(v_id)   ? (int32_t)am_value_to_int(v_id)
       :                           (int32_t)am_value_to_uint(v_id);
    if (id < 0 || id >= RS_MAX_MESH) return -1;
    if (!am_value_is_handle(v_verts) || !am_value_is_handle(v_tris)) return -1;

    lv = (am_list_t *)am_value_to_ptr(am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap,
                                                  am_value_to_handle(v_verts)));
    lt = (am_list_t *)am_value_to_ptr(am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap,
                                                  am_value_to_handle(v_tris)));
    if (lv == NULL || lv->base.type != AM_OBJECT_TYPE_LIST) return -1;
    if (lt == NULL || lt->base.type != AM_OBJECT_TYPE_LIST) return -1;

    nv = (int)(lv->length / 5);
    nt = (int)(lt->length / 3);
    if (nv <= 0 || nt <= 0) return native_statement_done(proc);
    if (nv > RS_MAX_VERTS || nt > RS_MAX_TRIS) {
        am_runtime_error(rt, L"[Raster] 网格超出后端上限（顶点 4096 / 三角形 8192）\n");
        return -1;
    }

    if (s_mesh[id].v != NULL)   { free(s_mesh[id].v);   s_mesh[id].v = NULL; }
    if (s_mesh[id].tri != NULL) { free(s_mesh[id].tri); s_mesh[id].tri = NULL; }
    s_mesh[id].used = 0;
    s_mesh[id].v = (float *)platform_malloc((size_t)nv * 5 * sizeof(float));
    s_mesh[id].tri = (uint16_t *)platform_malloc((size_t)nt * 3 * sizeof(uint16_t));
    if (s_mesh[id].v == NULL || s_mesh[id].tri == NULL) {
        if (s_mesh[id].v) { free(s_mesh[id].v); s_mesh[id].v = NULL; }
        if (s_mesh[id].tri) { free(s_mesh[id].tri); s_mesh[id].tri = NULL; }
        am_runtime_error(rt, L"[Raster] 网格内存不足\n");
        return -1;
    }
    for (i = 0; i < nv * 5; i++) {
        am_value_t e = lv->children[i];
        if (am_value_is_float(e))      s_mesh[id].v[i] = (float)am_value_to_float(e);
        else if (am_value_is_int(e))   s_mesh[id].v[i] = (float)am_value_to_int(e);
        else if (am_value_is_uint(e))  s_mesh[id].v[i] = (float)am_value_to_uint(e);
        else s_mesh[id].v[i] = 0.0f;
    }
    for (i = 0; i < nt * 3; i++) {
        am_value_t e = lt->children[i];
        int32_t k = 0;
        if (am_value_is_float(e))      k = (int32_t)am_value_to_float(e);
        else if (am_value_is_int(e))   k = (int32_t)am_value_to_int(e);
        else if (am_value_is_uint(e))  k = (int32_t)am_value_to_uint(e);
        s_mesh[id].tri[i] = (uint16_t)(k < 0 ? 0 : (k > 65535 ? 65535 : k));
    }
    s_mesh[id].nv = nv;
    s_mesh[id].nt = nt;
    s_mesh[id].used = 1;
    return native_statement_done(proc);
}

// (Raster.begin) : #undefined —— 清帧缓冲（黑）+ 深度（远）
int32_t am_native_Raster_begin(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (s_wt.ready) wt_clear_frame();
    return native_statement_done(proc);
}

// (Raster.draw mesh kind mvp params) : #undefined
//   mvp    ：16 元列表（行主序 P·MV）。
//   params ：11 元列表
//            (eye.x eye.y eye.z  light.x light.y light.z  ball.x ball.y ball.z radius cull)
//   kind   ：0 池壁 / 1 水面（水上）/ 2 水面（水下）/ 3 圆球
//   cull   ：0 不剔除 / 1 只留面积<0 / 2 只留面积>=0
int32_t am_native_Raster_draw(am_runtime_t *rt, am_process_t *proc) {
    float prm[11], mvp[16];
    int32_t kind, mesh_id;
    int i;

    (void)rt;
    // 实参按书写顺序压栈，故从右往左弹出
    if (!native_pop_float_list(proc, prm, 11)) return -1;
    if (!native_pop_float_list(proc, mvp, 16)) return -1;
    if (!native_pop_int(proc, &kind))          return -1;
    if (!native_pop_int(proc, &mesh_id))       return -1;
    if (!s_wt.ready) return native_statement_done(proc);   // 未 init：静默跳过（栈纪律不变）

    for (i = 0; i < 16; i++) s_wt.MVP.m[i] = mvp[i];
    s_wt.eye      = v3f(prm[0], prm[1], prm[2]);
    // 光源由 Scheme 给出**未归一化**方向，后端归一化（与 ui_water.c 的 v3u(2,2,-1) 逐位一致）
    s_wt.lightDir = v3u(v3f(prm[3], prm[4], prm[5]));
    s_wt.center   = v3f(prm[6], prm[7], prm[8]);
    s_wt.radius   = prm[9];
    rs_draw(mesh_id, kind, (int)prm[10]);
    return native_statement_done(proc);
}

// (Raster.sim_move ox oy oz cx cy cz radius) : #undefined
//   —— 圆球对水体积的排开/回填（原版 wt_move_sphere，逐格点）
int32_t am_native_Raster_sim_move(am_runtime_t *rt, am_process_t *proc) {
    am_float_t r, cz, cy, cx, oz, oy, ox;
    (void)rt;
    if (!native_pop_number(proc, &r))  return -1;
    if (!native_pop_number(proc, &cz)) return -1;
    if (!native_pop_number(proc, &cy)) return -1;
    if (!native_pop_number(proc, &cx)) return -1;
    if (!native_pop_number(proc, &oz)) return -1;
    if (!native_pop_number(proc, &oy)) return -1;
    if (!native_pop_number(proc, &ox)) return -1;
    if (s_wt.ready) {
        s_wt.radius = (float)r;
        wt_move_sphere(v3f((float)ox, (float)oy, (float)oz),
                       v3f((float)cx, (float)cy, (float)cz));
    }
    return native_statement_done(proc);
}

// (Raster.sim_step) : #undefined —— 波场推进一拍（原版 wt_step_sim，逐格点）
int32_t am_native_Raster_sim_step(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (s_wt.ready) wt_step_sim();
    return native_statement_done(proc);
}

// (Raster.sim_normals) : #undefined —— 水面法线（原版 wt_update_normals，逐格点）
int32_t am_native_Raster_sim_normals(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (s_wt.ready) wt_update_normals();
    return native_statement_done(proc);
}

// (Raster.caustics mesh light.x light.y light.z ball.x ball.y ball.z radius) : #undefined
//   —— 焦散 pass（原版 wt_update_caustics：水面网格沿折射光投影到池底）
int32_t am_native_Raster_caustics(am_runtime_t *rt, am_process_t *proc) {
    am_float_t r, cz, cy, cx, lz, ly, lx;
    int32_t mesh_id;
    RS_Mesh *m;

    (void)rt;
    if (!native_pop_number(proc, &r))   return -1;
    if (!native_pop_number(proc, &cz))  return -1;
    if (!native_pop_number(proc, &cy))  return -1;
    if (!native_pop_number(proc, &cx))  return -1;
    if (!native_pop_number(proc, &lz))  return -1;
    if (!native_pop_number(proc, &ly))  return -1;
    if (!native_pop_number(proc, &lx))  return -1;
    if (!native_pop_int(proc, &mesh_id)) return -1;
    if (s_wt.ready && mesh_id >= 0 && mesh_id < RS_MAX_MESH) {
        m = &s_mesh[mesh_id];
        if (m->used) {
            s_wt.lightDir = v3u(v3f((float)lx, (float)ly, (float)lz));
            s_wt.center   = v3f((float)cx, (float)cy, (float)cz);
            s_wt.radius   = (float)r;
            wt_update_caustics((const WT_GeoV *)m->v, m->tri, m->nt);
        }
    }
    return native_statement_done(proc);
}

// (Raster.blit) : #undefined —— 后端帧缓冲（RGB565）→ 屏幕帧缓冲
int32_t am_native_Raster_blit(am_runtime_t *rt, am_process_t *proc) {
    Nano_GFX *g = native_gfx(rt);
    if (g == NULL) return -1;
    if (s_wt.ready && s_wt.frame != NULL) gfx_blit_rgb565(g, s_wt.frame);
    return native_statement_done(proc);
}

// (Raster.dispose) : #undefined —— 释放缓冲与网格（幂等）
int32_t am_native_Raster_dispose(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    rs_free();
    return native_statement_done(proc);
}

static const am_native_func_entry_t am_native_Raster_funcs[] = {
    { L"init",      am_native_Raster_init },
    { L"mesh",      am_native_Raster_mesh },
    { L"begin",     am_native_Raster_begin },
    { L"draw",      am_native_Raster_draw },
    { L"sim_move",  am_native_Raster_sim_move },
    { L"sim_step",  am_native_Raster_sim_step },
    { L"sim_normals", am_native_Raster_sim_normals },
    { L"caustics",  am_native_Raster_caustics },
    { L"blit",      am_native_Raster_blit },
    { L"dispose",   am_native_Raster_dispose },
};

const am_native_lib_entry_t am_native_Raster_lib = {
    L"Raster",
    am_native_Raster_funcs,
    sizeof(am_native_Raster_funcs) / sizeof(am_native_Raster_funcs[0])
};
