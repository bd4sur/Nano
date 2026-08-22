/* ===============================================================================
 * 水池（WebGL Water）—— 忠实移植 madebyevan.com/webgl-water（water/ 目录，MIT）
 *
 * 把原作的 GLSL 渲染管线逐行直译成 C 浮点软件渲染/光线追踪，力求算法一致
 * （常量、公式、次序、命名均保留原样），包括：
 *
 *  - Water 仿真（water.js）
 *      updateShader / normalShader / sphereShader（moveSphere＝圆球排开/回填水体）
 *    高度场双页乒乓、浮点纹理 256x256，语义同原版。
 *  - 渲染（renderer.js）
 *      水面片段：逐像素 反射/折射 光线追迹（peaked 五次法线偏移、菲涅尔、全内反射、
 *               介质水色相乘）
 *      池壁片段：瓷砖（直接绘制）+ 环境光遮蔽 + 焦散 / 池沿阴影 + 水下透染
 *      圆球片段：环境光遮蔽 + 焦散 + 水下透染
 *      焦散 pass ：水面网格沿折射光投影到池底，生成聚焦亮度 + 圆球挡光 + 池沿阴影
 *  - 相机（main.js）：perspective(45,…) + modelview（translate/rotate 组合），
 *                     拖动旋转视角；Raytracer 复刻（eye + 四角射线 + 逐像素射线）
 *  - 圆球物理（main.js）：重力、水中黏滞浮力、池底反弹；G 切换重力、空格暂停、L 定向光
 *
 * 天空/瓷砖按要求保留为“直接绘制”（程序化生成），不加载贴图。
 * 软渲染、float 全精度、无性能优先级——帧率低是刻意为之。
 * =============================================================================== */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "platform.h"

#ifndef WATER_HOST_TEST
    #include "ui_water.h"
    #include "hal_key.h"
    #include "hal_touch.h"
#endif

// ===============================================================================
// 原版常量（helperFunctions / main.js 直译）
// ===============================================================================
#define WT_IOR_AIR              (1.0f)
#define WT_IOR_WATER            (1.333f)
#define WT_POOL_HEIGHT          (1.0f)
#define WT_COLOR_ABOVEWATER_R   (0.25f)   // abovewaterColor
#define WT_COLOR_ABOVEWATER_G   (1.00f)
#define WT_COLOR_ABOVEWATER_B   (1.25f)
#define WT_COLOR_UNDERWATER_R   (0.40f)   // underwaterColor
#define WT_COLOR_UNDERWATER_G   (0.90f)
#define WT_COLOR_UNDERWATER_B   (1.00f)
#define WT_COLOR_SKY2_R         (0.80f)   // 水下折射色 vec3(0.8,1.0,1.1)
#define WT_COLOR_SKY2_G         (1.00f)
#define WT_COLOR_SKY2_B         (1.10f)

#define WT_GRID_N           (128)       // 水面高度场分辨率（原版 256；为内存压缩取 128，算法不变）
#define WT_CAUSTIC_N        (256)       // 焦散纹理分辨率（原版 1024；为内存压缩取 256，算法不变）
#define WT_WATER_DETAIL     (44)        // 水面/焦散网格细分（原版 200；原 1600px 画布上
                                        //   每三角形约 8px，本屏 320px 宽取 44 保持同观感）
#define WT_WATER_VERTS     ((WT_WATER_DETAIL + 1) * (WT_WATER_DETAIL + 1))
#define WT_WATER_TRIS      (WT_WATER_DETAIL * WT_WATER_DETAIL * 2)
#define WT_CELLS           (WT_GRID_N * WT_GRID_N)
#define WT_WATER_DETAIL_P1 (WT_WATER_DETAIL + 1)

// 立方体池壁（除底面外的 5 个面，同 cubeMesh：cubeData 去掉 -y）
#define WT_CUBE_FACES      (5)
#define WT_CUBE_VERTS      (WT_CUBE_FACES * 4)
#define WT_CUBE_TRIS       (WT_CUBE_FACES * 2)

// 圆球网格（UV 球细分：原 geodesic detail≈6，规模相当）
#define WT_SPH_RING        (20)
#define WT_SPH_SEG         (36)
#define WT_SPH_VN          (WT_SPH_RING * (WT_SPH_SEG + 1))
#define WT_SPH_TN          ((WT_SPH_RING - 1) * WT_SPH_SEG * 2)

// 圆球（main.js）
#define WT_RADIUS          (0.25f)
#define WT_GRAVITY         (-4.0f)

// ===============================================================================
// 向量/矩阵（与 lightgl.js 一致：行主序、v'=M·v、transformPoint 含除法）
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
static V3 v3x(V3 a, V3 b, float t) { return v3a(a, v3k(v3s(b, a), t)); }
static float v3clmp(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }

static V3 wt_reflect(V3 I, V3 N) { return v3s(I, v3k(N, 2.0f * v3d(N, I))); }
static V3 wt_refract(V3 I, V3 N, float eta, int *ok) {
    float dd = v3d(N, I);
    float k = 1.0f - eta * eta * (1.0f - dd * dd);
    if (k < 0.0f) { if (ok) *ok = 0; return v3f(0, 0, 0); }
    if (ok) *ok = 1;
    return v3s(v3k(I, eta), v3k(N, eta * dd + sqrtf(k)));
}

static void m4id(M4 *o) { memset(o->m, 0, sizeof(o->m)); o->m[0] = o->m[5] = o->m[10] = o->m[15] = 1; }
static void m4mul(const M4 *A, const M4 *B, M4 *o) {   // o = A·B（行主序）
    const float *a = A->m, *b = B->m;
    M4 r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            r.m[i * 4 + j] = a[i * 4 + 0] * b[j] + a[i * 4 + 1] * b[4 + j]
                           + a[i * 4 + 2] * b[8 + j] + a[i * 4 + 3] * b[12 + j];
    *o = r;
}
static void m4persp(float fov, float aspect, float n, float f, M4 *o) {
    float y = tanf(fov * (float)M_PI / 360.0f) * n;
    float x = y * aspect;
    memset(o->m, 0, sizeof(o->m));
    o->m[0] = 2 * n / (2 * x);
    o->m[5] = 2 * n / (2 * y);
    o->m[10] = -(f + n) / (f - n);
    o->m[11] = -2 * f * n / (f - n);
    o->m[14] = -1;
}
static void m4tr(float x, float y, float z, M4 *o) { m4id(o); o->m[3] = x; o->m[7] = y; o->m[11] = z; }
static void m4rot(float deg, float ax, float ay, float az, M4 *o) {
    float d = sqrtf(ax * ax + ay * ay + az * az);
    float a = deg * (float)M_PI / 180.0f; ax /= d; ay /= d; az /= d;
    float c = cosf(a), s = sinf(a), t = 1 - c;
    M4 r;
    r.m[0]  = ax * ax * t + c;      r.m[1]  = ax * ay * t - az * s; r.m[2]  = ax * az * t + ay * s; r.m[3]  = 0;
    r.m[4]  = ay * ax * t + az * s; r.m[5]  = ay * ay * t + c;      r.m[6]  = ay * az * t - ax * s; r.m[7]  = 0;
    r.m[8]  = az * ax * t - ay * s; r.m[9]  = az * ay * t + ax * s; r.m[10] = az * az * t + c;      r.m[11] = 0;
    r.m[12] = 0; r.m[13] = 0; r.m[14] = 0; r.m[15] = 1;
    *o = r;
}
static void wt_mvp4(const M4 *M, float x, float y, float z, float *cx, float *cy, float *cz, float *cw) {
    const float *m = M->m;
    *cx = m[0] * x + m[1] * y + m[2] * z + m[3];
    *cy = m[4] * x + m[5] * y + m[6] * z + m[7];
    *cz = m[8] * x + m[9] * y + m[10] * z + m[11];
    *cw = m[12] * x + m[13] * y + m[14] * z + m[15];
}
static void m4inv(const M4 *s, M4 *o) {   // Mesa 伴随式求逆（与 lightgl Matrix.inverse 一致）
    const float *m = s->m;
    float r[16];
    r[0]  = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
          + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    r[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
          - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    r[2]  = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
          + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    r[3]  = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
          - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    r[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
          - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    r[5]  = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
          + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    r[6]  = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
          - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    r[7]  = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
          + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    r[8]  = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
          + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    r[9]  = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
          - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    r[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
          + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    r[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
          - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    r[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
          - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    r[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
          + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    r[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
          - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    r[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
          + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * r[0] + m[1] * r[4] + m[2] * r[8] + m[3] * r[12];
    if (det == 0.0f) { m4id(o); return; }
    for (int i = 0; i < 16; i++) o->m[i] = r[i] / det;
}

// ===============================================================================
// 几何与渲染状态
// ===============================================================================
typedef struct { float vx, vy, vz; float u, v; } WT_GeoV;    // 水面/焦散网格顶点

// 网格内存一律走 PSRAM（platform_malloc），进入分配、退出交还；
// 不留静态数组占 .bss（内部 DRAM 与 DMA 帧缓冲同池，曾导致启动分配帧缓冲失败）。
static WT_GeoV  *s_wm;        // 水面网格
static uint16_t *s_wtris;     // 水面三角形
static float    (*s_cv)[3];   // 池壁网格
static uint16_t (*s_ctri)[3]; // 池壁三角形
static float    (*s_sph)[3];  // 圆球网格
static uint16_t (*s_stri)[3]; // 圆球三角形

typedef struct {
    float angleX, angleY;            // 相机（main.js：angleX=-25, angleY=-200.5）
    V3 eye;
    V3 ray00, ray10, ray01, ray11;   // 四角像素射线（Raytracer）
    M4 MV, P, MVP;
    V3 lightDir;
    V3 center, oldCenter, velocity, gravity;
    float radius;
    int useSpherePhysics, paused;
    int mode;                        // -1 无；0 旋转相机；1 拖动圆球
    V3 prevHit, planeNormal;
    int mouseDown, prevX, prevY;
    float *water[2];                 // 每页：RGBA float = (h, v, n.x, n.z)
    int cur;
    float *caustR, *caustG;          // 焦散 C*C
    uint16_t *frame;                 // RGB565 输出
    uint16_t *zbuf;                  // 深度（16 位，0..65535）
    int WW, HH;
    uint64_t last_t;
    int ready;
} WT_State;

static WT_State s_wt;

// ===============================================================================
// 水面纹理采样（GL_LINEAR 双线性、边界 CLAMP）
// ===============================================================================
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

// 读取水面某点 (h, n.x, n.z)（= info.rba）之槽位
static void wt_wat_info_slot(const float *page, float u, float v, float *h, float *b, float *a);

// ===============================================================================
// 焦散纹理采样（双线性）
// ===============================================================================
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

// ===============================================================================
// 水面仿真（water.js：双页乒乓，读 cur → 写 1-cur → 交换）
// ===============================================================================
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

// normalShader：normal = normalize(cross(dy,dx)).xz → 存入 (info.b, info.a)
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

// sphereShader：圆球对水体积的排开/回填（moveSphere）
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

// ===============================================================================
// 程序化天空（原 textureCube(sky, ray)——按要求直接绘制）
// ===============================================================================
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

// ===============================================================================
// 程序化瓷砖（原 texture2D(tiles, uv)——REPEAT 环绕，按要求直接绘制）
// ===============================================================================
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

// ===============================================================================
// 光线与盒/球交点（helperFunctions 直译）
// ===============================================================================
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

// ===============================================================================
// helper 着色：getSphereColor / getWallColor / getSurfaceRayColor（直译）
// ===============================================================================
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

// ===============================================================================
// 光栅化（透视校正插值 + 深度测试）
// ===============================================================================
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

// ===============================================================================
// 网格构建
// ===============================================================================
static void wt_build_meshes(void) {
    // 水面平面（同 Mesh.plane(detail)）
    int D = WT_WATER_DETAIL, stride = WT_WATER_DETAIL_P1;
    for (int y = 0; y <= D; y++) {
        float t = (float)y / (float)D;
        for (int x = 0; x <= D; x++) {
            float s = (float)x / (float)D;
            WT_GeoV *g = &s_wm[y * stride + x];
            g->vx = 2 * s - 1; g->vy = 2 * t - 1; g->vz = 0;
            g->u = s; g->v = t;
        }
    }
    int ti = 0;
    for (int y = 0; y < D; y++) {
        for (int x = 0; x < D; x++) {
            uint16_t i = (uint16_t)(y * stride + x);
            s_wtris[ti * 3 + 0] = i; s_wtris[ti * 3 + 1] = i + 1; s_wtris[ti * 3 + 2] = i + stride; ti++;
            s_wtris[ti * 3 + 0] = i + stride; s_wtris[ti * 3 + 1] = i + 1; s_wtris[ti * 3 + 2] = i + stride + 1; ti++;
        }
    }

    // 池壁（cubeData 同序、去掉 -y 顶沿面；+y 一组为底面 [2,6,3,7]，重映射到 y=-1）
    {
        static const int faces[WT_CUBE_FACES][4] = {
            { 0, 4, 2, 6 },   // -x
            { 1, 3, 5, 7 },   // +x
            { 2, 6, 3, 7 },   // +y（底面，y=+1 顶点 → 世界 y=-1）
            { 0, 2, 1, 3 },   // -z
            { 4, 5, 6, 7 },   // +z
        };
        int v = 0;
        for (int fi = 0; fi < WT_CUBE_FACES; fi++) {
            for (int j = 0; j < 4; j++) {
                int d = faces[fi][j];
                s_cv[v][0] = (float)((d & 1) * 2 - 1);
                s_cv[v][1] = (float)((d & 2) - 1);
                s_cv[v][2] = (float)((d & 4) / 2 - 1);
                v++;
            }
            uint16_t b = (uint16_t)(fi * 4);
            s_ctri[fi * 2 + 0][0] = b; s_ctri[fi * 2 + 0][1] = b + 1; s_ctri[fi * 2 + 0][2] = b + 2;
            s_ctri[fi * 2 + 1][0] = b + 2; s_ctri[fi * 2 + 1][1] = b + 1; s_ctri[fi * 2 + 1][2] = b + 3;
        }
    }

    // 圆球（UV 球）
    {
        int vi = 0;
        for (int r = 0; r < WT_SPH_RING; r++) {
            float phi = (float)M_PI * (float)r / (float)(WT_SPH_RING - 1);   // 0..π
            float y = cosf(phi), rad = sinf(phi);
            for (int sx = 0; sx <= WT_SPH_SEG; sx++) {
                float th = 2.0f * (float)M_PI * (float)sx / (float)WT_SPH_SEG;
                s_sph[vi][0] = rad * cosf(th);
                s_sph[vi][1] = y;
                s_sph[vi][2] = rad * sinf(th);
                vi++;
            }
        }
        int t = 0;
        for (int r = 0; r < WT_SPH_RING - 1; r++) {
            for (int sx = 0; sx < WT_SPH_SEG; sx++) {
                uint16_t a = (uint16_t)(r * (WT_SPH_SEG + 1) + sx);
                uint16_t b = (uint16_t)(a + 1);
                uint16_t c = (uint16_t)(a + WT_SPH_SEG + 1);
                uint16_t d = (uint16_t)(c + 1);
                s_stri[t][0] = a; s_stri[t][1] = c; s_stri[t][2] = b; t++;
                s_stri[t][0] = b; s_stri[t][1] = c; s_stri[t][2] = d; t++;
            }
        }
    }
}

// ===============================================================================
// 相机（main.js draw()：translate/rotate 组合）+ Raytracer
// ===============================================================================
static void wt_compose_camera(void) {
    M4 mv, t;
    m4id(&mv);
    m4tr(0, 0, -4, &t);   m4mul(&mv, &t, &mv);
    m4rot(-s_wt.angleX, 1, 0, 0, &t); m4mul(&mv, &t, &mv);
    m4rot(-s_wt.angleY, 0, 1, 0, &t); m4mul(&mv, &t, &mv);
    m4tr(0, 0.5f, 0, &t); m4mul(&mv, &t, &mv);
    s_wt.MV = mv;
    m4persp(45.0f, (float)s_wt.WW / (float)s_wt.HH, 0.01f, 100.0f, &s_wt.P);
    m4mul(&s_wt.P, &mv, &s_wt.MVP);

    // Raytracer：由 modelview 反解世界系相机位置 eye
    {
        const float *m = mv.m;
        V3 axisX = v3f(m[0], m[4], m[8]);
        V3 axisY = v3f(m[1], m[5], m[9]);
        V3 axisZ = v3f(m[2], m[6], m[10]);
        V3 off   = v3f(m[3], m[7], m[11]);
        s_wt.eye = v3f(-v3d(off, axisX), -v3d(off, axisY), -v3d(off, axisZ));
    }
    // 四角射线 = unProject(角,1) - eye
    {
        M4 invMVP;
        m4inv(&s_wt.MVP, &invMVP);
        int corners[4][2] = {{0,0},{s_wt.WW,0},{0,s_wt.HH},{s_wt.WW,s_wt.HH}};
        static V3 *dst[4] = { &s_wt.ray00, &s_wt.ray10, &s_wt.ray01, &s_wt.ray11 };
        for (int c = 0; c < 4; c++) {
            float nx = 2.0f * (float)corners[c][0] / (float)s_wt.WW - 1.0f;
            float ny = 2.0f * (float)corners[c][1] / (float)s_wt.HH - 1.0f;
            float wx, wy, wz, ww;
            wt_mvp4(&invMVP, nx, ny, 1.0f, &wx, &wy, &wz, &ww);
            if (ww <= 0) { *dst[c] = v3f(0, 0, 0); continue; }
            float iw = 1.0f / ww;
            V3 world = v3f(wx * iw, wy * iw, wz * iw);
            *dst[c] = v3u(v3s(world, s_wt.eye));
        }
    }
}

// getRayForPixel(x, y)（Raytracer.prototype 直译）
static V3 wt_get_ray_for_pixel(int x, int y) {
    float fx = (float)x / (float)s_wt.WW;
    float fy = (float)y / (float)s_wt.HH;
    fy = 1.0f - fy;
    V3 r0 = v3x(s_wt.ray00, s_wt.ray10, fx);
    V3 r1 = v3x(s_wt.ray01, s_wt.ray11, fx);
    return v3u(v3x(r0, r1, fy));
}

// ===============================================================================
// 渲染几何对象
// ===============================================================================

// ---- 池壁（cube fragment：getWallColor + 水下透染） ----
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

// 屏幕空间带符号面积：front = 面积>0（与 GL 按 NDC 绕序判 front 等价，y 翻转互抵）
static float wt_screen_area(const WT_Vtx *a, const WT_Vtx *b, const WT_Vtx *c) {
    return (b->sx - a->sx) * (c->sy - a->sy) - (b->sy - a->sy) * (c->sx - a->sx);
}

static void wt_render_cube(void) {
    for (int t = 0; t < WT_CUBE_TRIS; t++) {
        const uint16_t *ix = s_ctri[t];
        WT_Vtx va, vb, vc;
        V3 wa = v3f(s_cv[ix[0]][0], ((1 - s_cv[ix[0]][1]) * (7.0f / 12.0f) - 1.0f) * WT_POOL_HEIGHT, s_cv[ix[0]][2]);
        V3 wb = v3f(s_cv[ix[1]][0], ((1 - s_cv[ix[1]][1]) * (7.0f / 12.0f) - 1.0f) * WT_POOL_HEIGHT, s_cv[ix[1]][2]);
        V3 wc = v3f(s_cv[ix[2]][0], ((1 - s_cv[ix[2]][1]) * (7.0f / 12.0f) - 1.0f) * WT_POOL_HEIGHT, s_cv[ix[2]][2]);
        wt_ndc_to_screen(&s_wt.MVP, wa, 0, 0, &va);
        wt_ndc_to_screen(&s_wt.MVP, wb, 0, 0, &vb);
        wt_ndc_to_screen(&s_wt.MVP, wc, 0, 0, &vc);
        if (vc.w <= 0 || vb.w <= 0 || va.w <= 0) continue;
        // 背面剔除（对标原版 renderCube：CULL_FACE+cull BACK 保留正面）。
        // 网格绕序为内向 + 本屏 y 向下 ⇒ GL 正面（NDC 逆时针）等价于本屏面积 <0；
        // 由此：远壁(-x,+z)与底面(+y)可见，遮挡视线的近壁(+x,-z)被剔除（与原版一致）。
        if (wt_screen_area(&va, &vb, &vc) >= 0) continue;
        wt_fill(&va, &vb, &vc, 1, wt_cube_frag);
    }
}

// ---- 水面（waterShader ×2：0=水面以上、1=水面以下） ----
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

/* ===========================================================================
 * （以下由追加部分完成：水面/圆球渲染、焦散、相机更新、交互与设备接线）
 * =========================================================================== */

// 读取水面某点的 (n.x, n.z)（= info.ba）
static void wt_wat_info_slot(const float *page, float u, float v, float *h, float *b, float *a) {
    *h = wt_wat_sample(page, u, v, 0);
    *b = wt_wat_sample(page, u, v, 2);
    *a = wt_wat_sample(page, u, v, 3);
}

// ---- 水面网格 → 该帧的世界位置（顶点：xzy + 高度） ----
static void wt_water_vert_world(int vi, V3 *world, float *u, float *v) {
    const WT_GeoV *g = &s_wm[vi];
    *u = g->u; *v = g->v;
    float h = wt_wat_sample(s_wt.water[s_wt.cur], g->u, g->v, 0);
    world->x = g->vx;                 // 2s-1
    world->z = g->vy;                 // 2t-1
    world->y = h;                     // position.y += info.r
}

static void wt_render_water(void) {
    const float *page = s_wt.water[s_wt.cur];
    int pass;
    for (pass = 0; pass < 2; pass++) {
        WT_Frag frag = pass ? wt_water_frag1 : wt_water_frag0;
        for (int t = 0; t < WT_WATER_TRIS; t++) {
            const uint16_t *ix = &s_wtris[t * 3];
            V3 wa, wb, wc; float u, v;
            WT_Vtx va, vb, vc;
            wt_water_vert_world(ix[0], &wa, &u, &v); wt_ndc_to_screen(&s_wt.MVP, wa, u, v, &va);
            wt_water_vert_world(ix[1], &wb, &u, &v); wt_ndc_to_screen(&s_wt.MVP, wb, u, v, &vb);
            wt_water_vert_world(ix[2], &wc, &u, &v); wt_ndc_to_screen(&s_wt.MVP, wc, u, v, &vc);
            if (va.w <= 0 || vb.w <= 0 || vc.w <= 0) continue;
            // 背面剔除（对标原版：pass0 cull FRONT、pass1 cull BACK；正面=本屏面积<0）。
            // 俯视下多数水面三角形属“正面”→pass1 绘制；pass0 仅补背面。
            float area = wt_screen_area(&va, &vb, &vc);
            if (pass == 0) { if (area < 0) continue; }   // cull FRONT → 只画背面
            else           { if (area >= 0) continue; }  // cull BACK  → 只画正面
            wt_fill(&va, &vb, &vc, 1, frag);
        }
    }
    (void)page;
}

// ---- 圆球（sphere fragment：getSphereColor + 水下透染） ----
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

static void wt_render_sphere(void) {
    for (int t = 0; t < WT_SPH_TN; t++) {
        const uint16_t *ix = s_stri[t];
        V3 wa = v3a(s_wt.center, v3k(v3f(s_sph[ix[0]][0], s_sph[ix[0]][1], s_sph[ix[0]][2]), s_wt.radius));
        V3 wb = v3a(s_wt.center, v3k(v3f(s_sph[ix[1]][0], s_sph[ix[1]][1], s_sph[ix[1]][2]), s_wt.radius));
        V3 wc = v3a(s_wt.center, v3k(v3f(s_sph[ix[2]][0], s_sph[ix[2]][1], s_sph[ix[2]][2]), s_wt.radius));
        WT_Vtx va, vb, vc;
        wt_ndc_to_screen(&s_wt.MVP, wa, 0, 0, &va);
        wt_ndc_to_screen(&s_wt.MVP, wb, 0, 0, &vb);
        wt_ndc_to_screen(&s_wt.MVP, wc, 0, 0, &vc);
        if (va.w <= 0 || vb.w <= 0 || vc.w <= 0) continue;
        wt_fill(&va, &vb, &vc, 1, wt_sphere_frag);
    }
}

// ===============================================================================
// 焦散 pass（causticsShader 直译；渲到 C×C 的 caustR/caustG）
//   project(origin, ray, refractedLight)：沿 ray 穿出池盒后，再沿折射光投影到 y=-1 平面
// ===============================================================================
static void wt_caustic_project(V3 origin, V3 ray, V3 refractedLight, V3 *out) {
    V3 mn = v3f(-1, -WT_POOL_HEIGHT, -1), mx = v3f(1, 2, 1);
    float t0, t1;
    wt_intersect_cube(origin, ray, mn, mx, &t0, &t1);
    origin = v3a(origin, v3k(ray, t1));
    float tplane = (-origin.y - 1.0f) / refractedLight.y;
    *out = v3a(origin, v3k(refractedLight, tplane));
}

static void wt_update_caustics(void) {
    const float *page = s_wt.water[s_wt.cur];
    const float C = (float)WT_CAUSTIC_N;
    memset(s_wt.caustR, 0, (size_t)WT_CAUSTIC_N * WT_CAUSTIC_N * 4);
    memset(s_wt.caustG, 0, (size_t)WT_CAUSTIC_N * WT_CAUSTIC_N * 4);

    V3 refractedLight = wt_refract(v3n(s_wt.lightDir), v3f(0, 1, 0),
                                   WT_IOR_AIR / WT_IOR_WATER, NULL);

    for (int t = 0; t < WT_WATER_TRIS; t++) {
        const uint16_t *ix = &s_wtris[t * 3];
        // 每顶点：info → normal → ray；oldPos/newPos（world）与 NDC 屏幕位置
        struct { float sx, sy; V3 oldP, newP; float ratio; } vrt[3];
        V3 oldV[3], newV[3];
        for (int k = 0; k < 3; k++) {
            const WT_GeoV *g = &s_wm[ix[k]];
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
// 圆球物理 & 场景更新（main.js update() 直译）
// ===============================================================================
static void wt_update(float seconds) {
    if (seconds > 1.0f) seconds = 1.0f;
    if (s_wt.mode == 1) {
        s_wt.velocity = v3f(0, 0, 0);                    // 拖动圆球：从静止重新计
    } else if (s_wt.useSpherePhysics) {
        float pUW = v3clmp((s_wt.radius - s_wt.center.y) / (2.0f * s_wt.radius), 0.0f, 1.0f);
        s_wt.velocity = v3a(s_wt.velocity,
            v3k(s_wt.gravity, seconds - 1.1f * seconds * pUW));       // 水中浮力（黏滞）
        float vl = v3l(s_wt.velocity);
        if (vl > 1e-6f)
            s_wt.velocity = v3s(s_wt.velocity,
                v3k(v3k(s_wt.velocity, 1.0f / vl),
                    pUW * seconds * v3d(s_wt.velocity, s_wt.velocity)));
        s_wt.center = v3a(s_wt.center, v3k(s_wt.velocity, seconds));
        if (s_wt.center.y < s_wt.radius - 1.0f) {        // 池底反弹
            s_wt.center.y = s_wt.radius - 1.0f;
            s_wt.velocity.y = fabsf(s_wt.velocity.y) * 0.7f;
        }
    }
    // 圆球对水体积的排开水/回填 + 两次波动推进 + 法线 + 焦散（main.js update 顺序）
    wt_move_sphere(s_wt.oldCenter, s_wt.center);
    s_wt.oldCenter = s_wt.center;
    wt_step_sim();
    wt_step_sim();
    wt_update_normals();
    wt_update_caustics();
}

// 拖动旋转视角开关：1=启用（按住除圆球外的区域拖动旋转相机），0=关闭（默认，
// 视角仅由 2/4/6/8 键旋转）。圆球拖动移动不受本宏影响。
#ifndef WT_ENABLE_DRAG_ROTATE
    #define WT_ENABLE_DRAG_ROTATE (0)
#endif

// 按键步进（2/4/6/8 一次旋转的角度，度）
#define WT_ROTATE_STEP  (4.0f)

// ===============================================================================
// 交互（main.js startDrag/duringDrag/stopDrag 直译；原“向水面落水”分支改为旋转相机）
// ===============================================================================
static void wt_start_drag(int x, int y) {
    s_wt.prevX = x; s_wt.prevY = y;
    V3 ray = wt_get_ray_for_pixel(x, y);
    // 圆球命中检测（hitTestSphere）
    V3 toSphere = v3s(s_wt.eye, s_wt.center);
    float a = v3d(ray, ray);
    float b = 2.0f * v3d(toSphere, ray);
    float c = v3d(toSphere, toSphere) - s_wt.radius * s_wt.radius;
    float disc = b * b - 4.0f * a * c;
    if (disc > 0) {
        float t = (-b - sqrtf(disc)) / (2.0f * a);
        if (t > 0) {
            s_wt.mode = 1;
            s_wt.prevHit = v3a(s_wt.eye, v3k(ray, t));
            s_wt.planeNormal = v3n(wt_get_ray_for_pixel(s_wt.WW / 2, s_wt.HH / 2));
            s_wt.mouseDown = 1;
            return;
        }
    }
#if WT_ENABLE_DRAG_ROTATE
    s_wt.mode = 0;                              // 其余：拖动旋转相机
#else
    s_wt.mode = -1;                             // 未启用：非球区域拖动无操作
#endif
    s_wt.mouseDown = 1;
}

static void wt_during_drag(int x, int y) {
#if WT_ENABLE_DRAG_ROTATE
    if (s_wt.mode == 0) {
        s_wt.angleY -= (float)(x - s_wt.prevX);
        s_wt.angleX -= (float)(y - s_wt.prevY);
        if (s_wt.angleX > 89.999f) s_wt.angleX = 89.999f;
        else if (s_wt.angleX < -89.999f) s_wt.angleX = -89.999f;
    } else if (s_wt.mode == 1) {
#else
    if (s_wt.mode == 1) {
#endif
        V3 ray = wt_get_ray_for_pixel(x, y);
        // 圆球在平行于视线的平面内移动：t = -(N·(eye-prevHit))/(N·ray)
        float denom = v3d(s_wt.planeNormal, ray);
        if (fabsf(denom) > 1e-9f) {
            float t = -v3d(s_wt.planeNormal, v3s(s_wt.eye, s_wt.prevHit)) / denom;
            V3 nextHit = v3a(s_wt.eye, v3k(ray, t));
            s_wt.center = v3a(s_wt.center, v3s(nextHit, s_wt.prevHit));
            s_wt.center.x = v3clmp(s_wt.center.x, s_wt.radius - 1, 1 - s_wt.radius);
            s_wt.center.y = v3clmp(s_wt.center.y, s_wt.radius - 1, 10);
            s_wt.center.z = v3clmp(s_wt.center.z, s_wt.radius - 1, 1 - s_wt.radius);
            s_wt.prevHit = nextHit;
        }
    }
    s_wt.prevX = x; s_wt.prevY = y;
}

// ===============================================================================
// 一帧绘制（main.js draw() 直译 + 应用层胶水）
// ===============================================================================
static void wt_scene_render(void) {
    wt_clear_frame();
    wt_render_cube();
    wt_render_water();
    wt_render_sphere();
}

// 分配内存 + 网格 + 重置默认场景（相机/圆球/光源）；设备 init 与宿主机自测共用
static int wt_alloc(int WW, int HH) {
    memset(&s_wt, 0, sizeof(s_wt));
    s_wt.WW = WW; s_wt.HH = HH;
    if (s_wt.WW <= 0 || s_wt.HH <= 0) { s_wt.WW = 320; s_wt.HH = 240; }

    s_wt.water[0] = (float *)platform_calloc((size_t)(2 * WT_CELLS * 4), sizeof(float));
    s_wt.water[1] = s_wt.water[0] + (size_t)WT_CELLS * 4;   // 两页连排
    s_wt.caustR = (float *)platform_calloc((size_t)WT_CAUSTIC_N * WT_CAUSTIC_N, sizeof(float));
    s_wt.caustG = (float *)platform_calloc((size_t)WT_CAUSTIC_N * WT_CAUSTIC_N, sizeof(float));
    s_wt.frame  = (uint16_t *)platform_calloc((size_t)s_wt.WW * s_wt.HH, sizeof(uint16_t));
    s_wt.zbuf   = (uint16_t *)platform_calloc((size_t)s_wt.WW * s_wt.HH, sizeof(uint16_t));
    s_wm   = (WT_GeoV *)platform_calloc(WT_WATER_VERTS, sizeof(WT_GeoV));
    s_wtris = (uint16_t *)platform_calloc((size_t)WT_WATER_TRIS * 3, sizeof(uint16_t));
    s_cv   = (float (*)[3])platform_calloc((size_t)WT_CUBE_VERTS, 3 * sizeof(float));
    s_ctri = (uint16_t (*)[3])platform_calloc((size_t)WT_CUBE_TRIS, 3 * sizeof(uint16_t));
    s_sph  = (float (*)[3])platform_calloc((size_t)WT_SPH_VN, 3 * sizeof(float));
    s_stri = (uint16_t (*)[3])platform_calloc((size_t)WT_SPH_TN, 3 * sizeof(uint16_t));
    if (s_wt.water[0] == NULL || s_wt.caustR == NULL || s_wt.caustG == NULL ||
        s_wt.frame == NULL || s_wt.zbuf == NULL || s_wm == NULL || s_wtris == NULL ||
        s_cv == NULL || s_ctri == NULL || s_sph == NULL || s_stri == NULL) {
        free(s_wt.water[0]); free(s_wt.caustR); free(s_wt.caustG);
        free(s_wt.frame); free(s_wt.zbuf); free(s_wm); free(s_wtris);
        free(s_cv); free(s_ctri); free(s_sph); free(s_stri);
        s_cv = NULL; s_ctri = NULL; s_sph = NULL; s_stri = NULL;
        memset(&s_wt, 0, sizeof(s_wt));
        return 0;
    }
    wt_build_meshes();

    // 原版默认状态（useSpherePhysics 置 1，进入即演示“移动的圆球+水体物理”）
    s_wt.angleX = -25.0f;
    s_wt.angleY = -200.5f;
    s_wt.lightDir = v3u(v3f(2, 2, -1));
    s_wt.center = s_wt.oldCenter = v3f(-0.4f, -0.75f, 0.2f);
    s_wt.velocity = v3f(0, 0, 0);
    s_wt.gravity  = v3f(0, WT_GRAVITY, 0);
    s_wt.radius   = WT_RADIUS;
    s_wt.useSpherePhysics = 1;
    s_wt.paused = 0;
    s_wt.mode = -1;
    s_wt.mouseDown = 0;
    s_wt.cur = 0;
    return 1;
}

#ifndef WATER_HOST_TEST

// 设备侧：内存分配与生命周期
int32_t ui_water_init(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    if (!wt_alloc((int)global_state->gfx->width, (int)global_state->gfx->height)) {
        s_wt.ready = 0;
        return -1;
    }
    s_wt.last_t = global_state->timestamp;
    s_wt.ready = 1;

    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);
    return 0;
}

int32_t ui_water_event_handler(Key_Event *key_event, Global_State *global_state) {
    (void)global_state;
    // A键(十六宫格右上角)返回
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_GAME_MENU;
        return 0;
    }
    // D键(十六宫格右下角)：开始/暂停（短按）
    if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
        s_wt.paused = !s_wt.paused;
        return 0;
    }
    // *键(十六宫格最左下角)：重力开关（短按）
    if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_left) {
        s_wt.useSpherePhysics = !s_wt.useSpherePhysics;
        return 0;
    }
    // 0键(*键右边的相邻键)：定向光源（短按）
    if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_0) {
        s_wt.lightDir = v3f(cosf((90 - s_wt.angleY) * (float)M_PI / 180.0f) * cosf(-s_wt.angleX * (float)M_PI / 180.0f),
                            sinf(-s_wt.angleX * (float)M_PI / 180.0f),
                            sinf((90 - s_wt.angleY) * (float)M_PI / 180.0f) * cosf(-s_wt.angleX * (float)M_PI / 180.0f));
        return 0;
    }
    // 2/4/6/8 键旋转视角（按住连转）
    if (key_event->key_edge == -1 || key_event->key_edge == -2) {
        if (key_event->key_code == NANO_KEY_8)       s_wt.angleX += WT_ROTATE_STEP;       // 上
        else if (key_event->key_code == NANO_KEY_2)  s_wt.angleX -= WT_ROTATE_STEP;       // 下
        else if (key_event->key_code == NANO_KEY_4)  s_wt.angleY += WT_ROTATE_STEP;       // 左
        else if (key_event->key_code == NANO_KEY_6)  s_wt.angleY -= WT_ROTATE_STEP;       // 右
        if (key_event->key_code == NANO_KEY_8 || key_event->key_code == NANO_KEY_2 ||
            key_event->key_code == NANO_KEY_4 || key_event->key_code == NANO_KEY_6) {
            if (s_wt.angleX > 89.999f) s_wt.angleX = 89.999f;
            else if (s_wt.angleX < -89.999f) s_wt.angleX = -89.999f;
            s_wt.prevX = s_wt.WW; s_wt.prevY = s_wt.HH;   // 丢弃可能残留的拖动起点
            return 0;
        }
    }
    return 0;
}

int32_t ui_water_render_frame(Key_Event *key_event, Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    (void)key_event;
    if (!s_wt.ready) {
        gfx_soft_clear(gfx);
        gfx_font_draw_text(gfx, GFX_FONT_ALPHA_16, L"水池", 6, 2, 255, 255, 255, 1);
        gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"内存不足", 6, 100, 255, 80, 80, 1);
        gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"A返回", 6, 220, 180, 180, 180, 1);
        gfx_refresh(gfx);
        return -1;
    }

    // ---- 触屏（按住拖动：圆球移动 / 视角旋转；PC 端对应 mousedown/mousemove/mouseup） ----
    int32_t t_x = 0, t_y = 0, t_p = 0;
    if (touch_read(&t_x, &t_y, &t_p) == 0) {
        if (t_p) {
            if (!s_wt.mouseDown) {
                wt_start_drag((int)t_x, (int)t_y);
                wt_during_drag((int)t_x, (int)t_y);
            } else {
                wt_during_drag((int)t_x, (int)t_y);
            }
        } else {
            if (s_wt.mouseDown) {
                s_wt.mouseDown = 0;                      // stopDrag：回到静止
                s_wt.mode = -1;
            }
        }
    }

    // ---- 帧时间步进（main.js：秒数、>1 忽略） ----
    uint64_t now = global_state->timestamp;
    float dt = (float)((now - s_wt.last_t) / 1000.0);
    s_wt.last_t = now;
    if (s_wt.paused) return 0;                            // 暂停：画面冻结

    // ---- 物理 + 仿真 + 渲染 ----
    wt_update(dt);
    wt_compose_camera();
    wt_scene_render();

    // ---- 拷贝到帧缓冲（仿真帧为 RGB565；单/双缓冲布局与色彩模式转换由图形层封装） ----
    gfx_blit_rgb565(gfx, s_wt.frame);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"水池", 4, 2, 208, 232, 255, 1);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"A返回 D暂停 *重力 0光 2/4/6/8视角", 120, 2, 130, 130, 130, 1);
    gfx_refresh(gfx);
    return 0;
}

void ui_water_on_exit(void) {
    if (s_wt.water[0] != NULL)  { free(s_wt.water[0]);  s_wt.water[0] = NULL; s_wt.water[1] = NULL; }
    if (s_wt.caustR != NULL)    { free(s_wt.caustR);    s_wt.caustR = NULL; }
    if (s_wt.caustG != NULL)    { free(s_wt.caustG);    s_wt.caustG = NULL; }
    if (s_wt.frame != NULL)     { free(s_wt.frame);     s_wt.frame = NULL; }
    if (s_wt.zbuf != NULL)      { free(s_wt.zbuf);      s_wt.zbuf = NULL; }
    if (s_wm != NULL)           { free(s_wm);           s_wm = NULL; }
    if (s_wtris != NULL)        { free(s_wtris);        s_wtris = NULL; }
    if (s_cv != NULL)           { free(s_cv);           s_cv = NULL; }
    if (s_ctri != NULL)         { free(s_ctri);         s_ctri = NULL; }
    if (s_sph != NULL)          { free(s_sph);          s_sph = NULL; }
    if (s_stri != NULL)         { free(s_stri);         s_stri = NULL; }
    memset(&s_wt, 0, sizeof(s_wt));
}

#else   // WATER_HOST_TEST：宿主自测通过 include 本文件直接访问静态函数
#endif  // !WATER_HOST_TEST
