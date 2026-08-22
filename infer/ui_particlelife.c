#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#include "ui_particlelife.h"
#include "hal_key.h"
#include "platform.h"

// ===============================================================================
// 粒子生命（Particle Life）
//
// 本文件包含两套等价实现，由 PL_USE_FIXED_POINT 宏切换：
//   - 定点实现（1，默认）：ESP32 无硬件 FPU，软浮点过慢（个位数 FPS）；
//     全链路定点数（位置/速度 Q20、参数 Q30、距离平方 Q40）+ 整数 rsqrt 查表
//     + 无序对对称化（半程邻域壳）+ 格宽恰为 RMAX 的均匀网格，预计约 25 FPS。
//   - 浮点实现（0）：直接对照 particle.html 的朴素写法，人类可读性最好，
//     适合作算法参考与宿主机对照实验；真机上较慢。
//
// 物理常数一律以下方浮点形式给出（调参入口）；定点实现在运行时经
// pl_fixed_const_init() 换算为定点常量（见 PL_Fixed_Const）。
// ===============================================================================

// ========== 实现选择：1=定点（快，默认），0=浮点（可读性参考） ==========
#define PL_USE_FIXED_POINT (1)

// ========== 物理常数（浮点形式，与 particle.html 一致，两种实现共用） ==========
#define PL_NUM_KIND     (6)       // 粒子种类数
#define PL_RTH1         (0.1f)    // 近距排斥阈值
#define PL_RTH2         (0.2f)    // 中距阈值
#define PL_RMAX         (0.4f)    // 相互作用半径（世界宽度的一半为 1.0）
#define PL_JITTER       (0.002f)  // 粒子随机震颤幅度
#define PL_DAMPING      (0.5f)    // 每帧速度衰减系数

// ========== 世界与种群 ==========
// 世界比例与机器屏幕一致（320x240 = 4:3）：x∈[-1,1]，y∈[-0.75,0.75] 的环面
#define PL_WORLD_HW     (1.0f)    // 世界半宽
#define PL_WORLD_HH     (0.75f)   // 世界半高
#define PL_WORLD_W      (2.0f * PL_WORLD_HW)  // 回绕周期（宽）
#define PL_WORLD_H      (2.0f * PL_WORLD_HH)  // 回绕周期（高）
#define PL_SCATTER_N    (18)      // 初始撒布网格边长（原作 N=30）
#define PL_NUM_PARTICLES (PL_SCATTER_N * PL_SCATTER_N)  // 粒子数

// ========== 视口 ==========
#define PL_HUD_H        (16)      // 顶部信息栏高度
#define PL_SCALE        (149.0f)  // 像素/世界单位（世界 2.0x1.5 -> 约 298x224 像素）
#define PL_CENTER_X     (160)     // 世界原点的屏幕坐标
#define PL_CENTER_Y     (PL_HUD_H + 112)

// 种类调色板（与原作 palette 一致：#f80 #ff0 #4f0 #0cf #04f #f08）
static const uint8_t S_PL_PALETTE[PL_NUM_KIND][3] = {
    {255, 136,   0},
    {255, 255,   0},
    { 68, 255,   0},
    {  0, 204, 255},
    {  0,  68, 255},
    {255,   0, 136},
};

// [0,1) 均匀随机数（仅初始化与浮点实现使用，对应 JS Math.random()）
static float pl_frand(void) {
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

#if PL_USE_FIXED_POINT

// ===============================================================================
// 定点实现
// ===============================================================================

// ========== 定点数格式 ==========
// 位置/速度：Q20（1.0 -> 1048576）；参数 a/b：Q30；距离平方：Q40
#define PL_Q20(f)       ((int32_t)((f) * 1048576.0f))
#define PL_PARAM_Q30(f) ((int32_t)((f) * 1073741824.0f))

// 速度钳制（浮点形式，仅定点实现需要的防爆兜底；正常速度 ~0.002 远低于此）
#define PL_VEL_CLAMP    (0.25f)

// rsqrt 结果钳制 2^25（对应 d≈3e-5，亚像素级重叠；定点实现细节，非物理常数）
#define PL_S_CLAMP      (33554432LL)

// 均匀网格（实现细节）：格宽恰为 RMAX，格数 = ceil(世界尺寸/RMAX)，
// 任一距离 ≤ RMAX 的粒子对必落在相邻格内；两维格数均 ≥3，半程壳环绕取模不重复计数
#define PL_GRID_DIM_X   ((int)(PL_WORLD_W / PL_RMAX + 0.5f))
#define PL_GRID_DIM_Y   ((int)(PL_WORLD_H / PL_RMAX + 0.5f))
#define PL_GRID_CELLS   (PL_GRID_DIM_X * PL_GRID_DIM_Y)

// 运行时换算的定点常量（由上方浮点物理常数推导，pl_fixed_const_init() 填充）
typedef struct {
    int32_t world_hw_q, world_hh_q;   // 世界半宽/半高（Q20）
    int32_t world_w_q, world_h_q;     // 回绕周期（Q20）
    int32_t cell_q;                   // 网格宽度 = RMAX（Q20）
    int32_t rth1_q20, rmax_q20;       // 分段阈值（Q20）
    int32_t inv_rth1_q20;             // 1/RTH1（Q20）
    int64_t rth1_2_q40, rth2_2_q40, rmax2_q40; // 阈值平方（Q40）
    int32_t jitter_half_q;            // 震颤半幅 JITTER/2（Q20）
    int32_t damping_q;                // 衰减系数（Q20）
    int32_t vel_clamp_q;              // 速度钳制（Q20）
} PL_Fixed_Const;

static PL_Fixed_Const s_fc;

static void pl_fixed_const_init(void) {
    s_fc.world_hw_q    = PL_Q20(PL_WORLD_HW);
    s_fc.world_hh_q    = PL_Q20(PL_WORLD_HH);
    s_fc.world_w_q     = PL_Q20(PL_WORLD_W);
    s_fc.world_h_q     = PL_Q20(PL_WORLD_H);
    s_fc.cell_q        = PL_Q20(PL_RMAX);
    s_fc.rth1_q20      = PL_Q20(PL_RTH1);
    s_fc.rmax_q20      = PL_Q20(PL_RMAX);
    s_fc.inv_rth1_q20  = PL_Q20(1.0f / PL_RTH1);
    s_fc.rth1_2_q40    = (int64_t)(PL_RTH1 * PL_RTH1 * 1099511627776.0f);
    s_fc.rth2_2_q40    = (int64_t)(PL_RTH2 * PL_RTH2 * 1099511627776.0f);
    s_fc.rmax2_q40     = (int64_t)(PL_RMAX * PL_RMAX * 1099511627776.0f);
    s_fc.jitter_half_q = PL_Q20(PL_JITTER * 0.5f);
    s_fc.damping_q     = PL_Q20(PL_DAMPING);
    s_fc.vel_clamp_q   = PL_Q20(PL_VEL_CLAMP);
}

typedef struct {
    int32_t x, y;     // 位置（Q20，环面）
    int32_t vx, vy;   // 速度（Q20）
    int32_t kind;     // 种类
    int32_t next;     // 均匀网格链表
} PL_Particle;

typedef struct {
    PL_Particle *parts;                        // PSRAM 分配，退出时释放
    int32_t param_a[PL_NUM_KIND][PL_NUM_KIND]; // 近距（<RTH1）作用强度（Q30）
    int32_t param_b[PL_NUM_KIND][PL_NUM_KIND]; // 中远距作用强度（Q30）
    int32_t head[PL_GRID_CELLS];               // 均匀网格桶链表头
    uint32_t lcg;                              // 震颤随机数状态（廉价 LCG）
    uint32_t frame;                            // 帧计数
} PL_State;

static PL_State s_pl;

// rsqrt 查找表：u ∈ [1,4) 按 1/256 间隔中点采样，Q14 定点（16384/sqrt(u)），768 项
static const uint16_t S_PL_RSQRT_LUT[768] = {
    16368, 16336, 16305, 16273, 16242, 16211, 16180, 16149, 16119, 16088, 16058, 16028,
    15998, 15968, 15939, 15909, 15880, 15851, 15822, 15794, 15765, 15737, 15708, 15680,
    15652, 15624, 15597, 15569, 15542, 15514, 15487, 15460, 15434, 15407, 15380, 15354,
    15328, 15302, 15276, 15250, 15224, 15198, 15173, 15148, 15122, 15097, 15072, 15047,
    15023, 14998, 14974, 14949, 14925, 14901, 14877, 14853, 14829, 14805, 14782, 14758,
    14735, 14712, 14689, 14666, 14643, 14620, 14597, 14575, 14552, 14530, 14508, 14486,
    14463, 14441, 14420, 14398, 14376, 14355, 14333, 14312, 14290, 14269, 14248, 14227,
    14206, 14185, 14165, 14144, 14124, 14103, 14083, 14062, 14042, 14022, 14002, 13982,
    13962, 13943, 13923, 13903, 13884, 13864, 13845, 13826, 13807, 13788, 13768, 13750,
    13731, 13712, 13693, 13674, 13656, 13637, 13619, 13601, 13582, 13564, 13546, 13528,
    13510, 13492, 13474, 13457, 13439, 13421, 13404, 13386, 13369, 13351, 13334, 13317,
    13300, 13283, 13266, 13249, 13232, 13215, 13198, 13182, 13165, 13148, 13132, 13115,
    13099, 13083, 13066, 13050, 13034, 13018, 13002, 12986, 12970, 12954, 12938, 12923,
    12907, 12891, 12876, 12860, 12845, 12830, 12814, 12799, 12784, 12769, 12753, 12738,
    12723, 12708, 12693, 12679, 12664, 12649, 12634, 12620, 12605, 12591, 12576, 12562,
    12547, 12533, 12519, 12504, 12490, 12476, 12462, 12448, 12434, 12420, 12406, 12392,
    12378, 12364, 12351, 12337, 12323, 12310, 12296, 12283, 12269, 12256, 12243, 12229,
    12216, 12203, 12189, 12176, 12163, 12150, 12137, 12124, 12111, 12098, 12085, 12073,
    12060, 12047, 12034, 12022, 12009, 11996, 11984, 11971, 11959, 11947, 11934, 11922,
    11909, 11897, 11885, 11873, 11861, 11849, 11836, 11824, 11812, 11800, 11788, 11777,
    11765, 11753, 11741, 11729, 11718, 11706, 11694, 11683, 11671, 11659, 11648, 11636,
    11625, 11614, 11602, 11591, 11580, 11568, 11557, 11546, 11535, 11524, 11512, 11501,
    11490, 11479, 11468, 11457, 11446, 11435, 11425, 11414, 11403, 11392, 11381, 11371,
    11360, 11349, 11339, 11328, 11318, 11307, 11297, 11286, 11276, 11265, 11255, 11244,
    11234, 11224, 11214, 11203, 11193, 11183, 11173, 11163, 11153, 11142, 11132, 11122,
    11112, 11102, 11092, 11083, 11073, 11063, 11053, 11043, 11033, 11024, 11014, 11004,
    10994, 10985, 10975, 10966, 10956, 10946, 10937, 10927, 10918, 10908, 10899, 10890,
    10880, 10871, 10862, 10852, 10843, 10834, 10824, 10815, 10806, 10797, 10788, 10779,
    10770, 10760, 10751, 10742, 10733, 10724, 10715, 10706, 10698, 10689, 10680, 10671,
    10662, 10653, 10644, 10636, 10627, 10618, 10610, 10601, 10592, 10584, 10575, 10566,
    10558, 10549, 10541, 10532, 10524, 10515, 10507, 10498, 10490, 10482, 10473, 10465,
    10457, 10448, 10440, 10432, 10423, 10415, 10407, 10399, 10391, 10382, 10374, 10366,
    10358, 10350, 10342, 10334, 10326, 10318, 10310, 10302, 10294, 10286, 10278, 10270,
    10262, 10255, 10247, 10239, 10231, 10223, 10216, 10208, 10200, 10192, 10185, 10177,
    10169, 10162, 10154, 10146, 10139, 10131, 10124, 10116, 10109, 10101, 10094, 10086,
    10079, 10071, 10064, 10056, 10049, 10042, 10034, 10027, 10020, 10012, 10005, 9998,
    9991, 9983, 9976, 9969, 9962, 9954, 9947, 9940, 9933, 9926, 9919, 9912,
    9905, 9898, 9890, 9883, 9876, 9869, 9862, 9855, 9848, 9842, 9835, 9828,
    9821, 9814, 9807, 9800, 9793, 9787, 9780, 9773, 9766, 9759, 9753, 9746,
    9739, 9732, 9726, 9719, 9712, 9706, 9699, 9692, 9686, 9679, 9673, 9666,
    9659, 9653, 9646, 9640, 9633, 9627, 9620, 9614, 9607, 9601, 9595, 9588,
    9582, 9575, 9569, 9563, 9556, 9550, 9544, 9537, 9531, 9525, 9518, 9512,
    9506, 9500, 9493, 9487, 9481, 9475, 9469, 9462, 9456, 9450, 9444, 9438,
    9432, 9426, 9420, 9413, 9407, 9401, 9395, 9389, 9383, 9377, 9371, 9365,
    9359, 9353, 9347, 9341, 9336, 9330, 9324, 9318, 9312, 9306, 9300, 9294,
    9289, 9283, 9277, 9271, 9265, 9260, 9254, 9248, 9242, 9236, 9231, 9225,
    9219, 9214, 9208, 9202, 9197, 9191, 9185, 9180, 9174, 9168, 9163, 9157,
    9152, 9146, 9141, 9135, 9129, 9124, 9118, 9113, 9107, 9102, 9096, 9091,
    9085, 9080, 9075, 9069, 9064, 9058, 9053, 9048, 9042, 9037, 9031, 9026,
    9021, 9015, 9010, 9005, 8999, 8994, 8989, 8984, 8978, 8973, 8968, 8963,
    8957, 8952, 8947, 8942, 8936, 8931, 8926, 8921, 8916, 8911, 8905, 8900,
    8895, 8890, 8885, 8880, 8875, 8870, 8865, 8860, 8854, 8849, 8844, 8839,
    8834, 8829, 8824, 8819, 8814, 8809, 8804, 8799, 8795, 8790, 8785, 8780,
    8775, 8770, 8765, 8760, 8755, 8750, 8745, 8741, 8736, 8731, 8726, 8721,
    8716, 8712, 8707, 8702, 8697, 8692, 8688, 8683, 8678, 8673, 8669, 8664,
    8659, 8654, 8650, 8645, 8640, 8636, 8631, 8626, 8622, 8617, 8612, 8608,
    8603, 8598, 8594, 8589, 8585, 8580, 8575, 8571, 8566, 8562, 8557, 8552,
    8548, 8543, 8539, 8534, 8530, 8525, 8521, 8516, 8512, 8507, 8503, 8498,
    8494, 8489, 8485, 8481, 8476, 8472, 8467, 8463, 8458, 8454, 8450, 8445,
    8441, 8437, 8432, 8428, 8423, 8419, 8415, 8410, 8406, 8402, 8397, 8393,
    8389, 8385, 8380, 8376, 8372, 8367, 8363, 8359, 8355, 8350, 8346, 8342,
    8338, 8334, 8329, 8325, 8321, 8317, 8313, 8308, 8304, 8300, 8296, 8292,
    8288, 8284, 8279, 8275, 8271, 8267, 8263, 8259, 8255, 8251, 8247, 8242,
    8238, 8234, 8230, 8226, 8222, 8218, 8214, 8210, 8206, 8202, 8198, 8194,
};

// ===============================================================================
// 整数反平方根：dd（Q40，>0）-> 1/sqrt(dd)（Q20，钳制到 PL_S_CLAMP）
// 归一化 dd = m' × 2^T（m'∈[1,4)，T 偶数），查表得 rsqrt(m') 后按指数移位还原
// ===============================================================================
static inline int64_t pl_rsqrt_q20(int64_t dd) {
    int32_t t = 63 - __builtin_clzll((unsigned long long)dd);
    int32_t T = (t & 1) ? (t - 1) : t;
    int32_t idx;
    if (T >= 8) idx = (int32_t)(dd >> (T - 8));
    else        idx = (int32_t)(dd << (8 - T));
    // idx ∈ [256,1024)
    int64_t s = (int64_t)S_PL_RSQRT_LUT[idx - 256] << (26 - T / 2);
    if (s > PL_S_CLAMP) s = PL_S_CLAMP;
    return s;
}

// ===============================================================================
// 世界生成（等价于原网页刷新）：随机相互作用参数 + 均匀撒布粒子
// ===============================================================================

static void pl_generate_world(PL_State *s) {
    // 异种间相互作用参数
    for (int32_t i = 0; i < PL_NUM_KIND; i++) {
        for (int32_t j = 0; j < PL_NUM_KIND; j++) {
            if (i == j) continue;
            s->param_a[i][j] = PL_PARAM_Q30(-(0.0005f + 0.0005f * pl_frand()));
            s->param_b[i][j] = PL_PARAM_Q30(0.001f * (pl_frand() - 0.5f));
        }
    }
    // 同种间相互作用参数
    for (int32_t i = 0; i < PL_NUM_KIND; i++) {
        s->param_a[i][i] = PL_PARAM_Q30(-(0.001f + 0.0005f * pl_frand()));
        s->param_b[i][i] = PL_PARAM_Q30(0.001f * (pl_frand() - 0.5f));
    }

    // 粒子撒布：NxN 网格均匀铺开并加少量抖动，种类随机
    int32_t n = 0;
    for (int32_t iy = 0; iy < PL_SCATTER_N; iy++) {
        for (int32_t ix = 0; ix < PL_SCATTER_N; ix++) {
            PL_Particle *p = &s->parts[n++];
            float xf = (2.0f * (float)ix / (float)(PL_SCATTER_N - 1) - 1.0f) * PL_WORLD_HW + 0.2f * (pl_frand() - 0.5f);
            float yf = (2.0f * (float)iy / (float)(PL_SCATTER_N - 1) - 1.0f) * PL_WORLD_HH + 0.2f * (pl_frand() - 0.5f);
            p->kind = rand() % PL_NUM_KIND;
            p->x = PL_Q20(xf);
            p->y = PL_Q20(yf);
            p->vx = 0;
            p->vy = 0;
        }
    }
    s->frame = 0;
}

// ===============================================================================
// 物理步进
// ===============================================================================

// 重建均匀网格（每帧一次，整数运算）
static void pl_grid_build(PL_State *s) {
    for (int32_t c = 0; c < PL_GRID_CELLS; c++) s->head[c] = -1;
    for (int32_t i = 0; i < PL_NUM_PARTICLES; i++) {
        PL_Particle *p = &s->parts[i];
        int32_t cx = (p->x + s_fc.world_hw_q) / s_fc.cell_q;
        int32_t cy = (p->y + s_fc.world_hh_q) / s_fc.cell_q;
        if (cx < 0) cx = 0; else if (cx >= PL_GRID_DIM_X) cx = PL_GRID_DIM_X - 1;
        if (cy < 0) cy = 0; else if (cy >= PL_GRID_DIM_Y) cy = PL_GRID_DIM_Y - 1;
        int32_t c = cy * PL_GRID_DIM_X + cx;
        p->next = s->head[c];
        s->head[c] = i;
    }
}

// 单对粒子的相互作用（对称化：距离只算一次，双向力同时施加）
static inline void pl_interact_pair(PL_State *s, int32_t i, int32_t j) {
    PL_Particle *pi = &s->parts[i];
    PL_Particle *pj = &s->parts[j];

    // 周期边界下的最近镜像位移（Q20）
    int32_t dx = pj->x - pi->x;
    int32_t dy = pj->y - pi->y;
    if (dx >  s_fc.world_hw_q) dx -= s_fc.world_w_q; else if (dx < -s_fc.world_hw_q) dx += s_fc.world_w_q;
    if (dy >  s_fc.world_hh_q) dy -= s_fc.world_h_q; else if (dy < -s_fc.world_hh_q) dy += s_fc.world_h_q;
    int64_t dd = (int64_t)dx * dx + (int64_t)dy * dy;   // Q40
    if (dd == 0 || dd > s_fc.rmax2_q40) return;

    int32_t ki = pi->kind, kj = pj->kind;
    int64_t rs = pl_rsqrt_q20(dd);      // 1/d（Q20）
    int32_t gij, gji;                   // 两方向的加速度系数（Q20）
    if (dd < s_fc.rth1_2_q40) {
        // 0...RTH1：F = a·(1/d − 1/RTH1)
        int64_t t = rs - s_fc.inv_rth1_q20;
        gij = (int32_t)(((int64_t)s->param_a[ki][kj] * t) >> 30);
        gji = (int32_t)(((int64_t)s->param_a[kj][ki] * t) >> 30);
    }
    else if (dd < s_fc.rth2_2_q40) {
        // RTH1...RTH2：F = b·(1 − RTH1/d)/(RTH2−RTH1)
        int64_t t = PL_Q20(1.0f) - ((rs * s_fc.rth1_q20) >> 20);
        int32_t inv_w = PL_Q20(1.0f / (PL_RTH2 - PL_RTH1));
        // b(Q30)×t(Q20)>>20 -> Q30；×inv_w(Q20)>>30 -> Q20
        gij = (int32_t)(((((int64_t)s->param_b[ki][kj] * t) >> 20) * inv_w) >> 30);
        gji = (int32_t)(((((int64_t)s->param_b[kj][ki] * t) >> 20) * inv_w) >> 30);
    }
    else {
        // RTH2...RMAX：F = b·(RMAX/d − 1)/(RMAX−RTH2)
        int64_t t = ((rs * s_fc.rmax_q20) >> 20) - PL_Q20(1.0f);
        int32_t inv_w = PL_Q20(1.0f / (PL_RMAX - PL_RTH2));
        gij = (int32_t)(((((int64_t)s->param_b[ki][kj] * t) >> 20) * inv_w) >> 30);
        gji = (int32_t)(((((int64_t)s->param_b[kj][ki] * t) >> 20) * inv_w) >> 30);
    }

    // 力 = 系数 × 单位方向（dx/d 已并入系数中的 1/d）
    int32_t fx_ij = (int32_t)(((int64_t)gij * dx) >> 20);
    int32_t fy_ij = (int32_t)(((int64_t)gij * dy) >> 20);
    int32_t fx_ji = (int32_t)(((int64_t)gji * dx) >> 20);
    int32_t fy_ji = (int32_t)(((int64_t)gji * dy) >> 20);
    pi->vx += fx_ij; pi->vy += fy_ij;
    pj->vx -= fx_ji; pj->vy -= fy_ji;
}

// 相互作用：均匀网格 + 半程邻域壳（同格无序对 + 4 个前向邻格，每对只算一次）
static void pl_interact(PL_State *s) {
    // 前向壳偏移（环绕取模；两维格数均 ≥3，无重复计数）
    static const int8_t SHELL[4][2] = {{1, 0}, {-1, 1}, {0, 1}, {1, 1}};
    for (int32_t gy = 0; gy < PL_GRID_DIM_Y; gy++) {
        for (int32_t gx = 0; gx < PL_GRID_DIM_X; gx++) {
            int32_t c = gy * PL_GRID_DIM_X + gx;
            // 同格内的无序对
            for (int32_t i = s->head[c]; i >= 0; i = s->parts[i].next) {
                for (int32_t j = s->parts[i].next; j >= 0; j = s->parts[j].next) {
                    pl_interact_pair(s, i, j);
                }
            }
            // 前向邻格对
            for (int32_t k = 0; k < 4; k++) {
                int32_t nx = (gx + SHELL[k][0] + PL_GRID_DIM_X) % PL_GRID_DIM_X;
                int32_t ny = (gy + SHELL[k][1]) % PL_GRID_DIM_Y;
                int32_t nc = ny * PL_GRID_DIM_X + nx;
                for (int32_t i = s->head[c]; i >= 0; i = s->parts[i].next) {
                    for (int32_t j = s->head[nc]; j >= 0; j = s->parts[j].next) {
                        pl_interact_pair(s, i, j);
                    }
                }
            }
        }
    }
}

// 移动：震颤 + 钳制 + 积分 + 衰减 + 环面回绕（对应原作 move + moveOut/moveIn）
static void pl_move(PL_State *s) {
    uint32_t jitter_span = (uint32_t)(2 * s_fc.jitter_half_q + 1);
    for (int32_t i = 0; i < PL_NUM_PARTICLES; i++) {
        PL_Particle *p = &s->parts[i];
        // 廉价 LCG 产生 ±JITTER/2 的震颤（Q20）
        s->lcg = s->lcg * 1664525u + 1013904223u;
        p->vx += (int32_t)(s->lcg % jitter_span) - s_fc.jitter_half_q;
        s->lcg = s->lcg * 1664525u + 1013904223u;
        p->vy += (int32_t)(s->lcg % jitter_span) - s_fc.jitter_half_q;
        // 速度钳制（防重叠爆发溢出；正常速度远低于钳制值）
        if (p->vx >  s_fc.vel_clamp_q) p->vx =  s_fc.vel_clamp_q; else if (p->vx < -s_fc.vel_clamp_q) p->vx = -s_fc.vel_clamp_q;
        if (p->vy >  s_fc.vel_clamp_q) p->vy =  s_fc.vel_clamp_q; else if (p->vy < -s_fc.vel_clamp_q) p->vy = -s_fc.vel_clamp_q;

        p->x += p->vx;
        p->y += p->vy;
        p->vx = (int32_t)(((int64_t)p->vx * s_fc.damping_q) >> 20);  // 衰减
        p->vy = (int32_t)(((int64_t)p->vy * s_fc.damping_q) >> 20);
        while (p->x < -s_fc.world_hw_q) p->x += s_fc.world_w_q;
        while (p->x >  s_fc.world_hw_q) p->x -= s_fc.world_w_q;
        while (p->y < -s_fc.world_hh_q) p->y += s_fc.world_h_q;
        while (p->y >  s_fc.world_hh_q) p->y -= s_fc.world_h_q;
    }
}

#else // !PL_USE_FIXED_POINT

// ===============================================================================
// 浮点实现（直接对照 particle.html，人类可读性参考）
// ===============================================================================

// 均匀网格加速：格宽不小于 RMAX（宽向 0.4、高向 0.5），5x3 格覆盖整个世界，
// 任一颗粒的近邻必落在环绕 3x3 格内
#define PL_GRID_DIM_X   (5)                             // PL_WORLD_W / 0.4
#define PL_GRID_DIM_Y   (3)                             // PL_WORLD_H / 0.5
#define PL_GRID_CELL_W  (PL_WORLD_W / PL_GRID_DIM_X)
#define PL_GRID_CELL_H  (PL_WORLD_H / PL_GRID_DIM_Y)
#define PL_GRID_CELLS   (PL_GRID_DIM_X * PL_GRID_DIM_Y)

typedef struct {
    float x, y;     // 位置（[-1,1) 环面）
    float vx, vy;   // 速度
    int32_t kind;   // 种类
    int32_t next;   // 均匀网格链表
} PL_Particle;

typedef struct {
    PL_Particle *parts;                 // PSRAM 分配，退出时释放
    float param_a[PL_NUM_KIND][PL_NUM_KIND]; // 近距（<RTH1）作用强度
    float param_b[PL_NUM_KIND][PL_NUM_KIND]; // 中远距作用强度
    int32_t head[PL_GRID_CELLS];        // 均匀网格桶链表头（按行优先索引）
    uint32_t frame;                     // 帧计数（HUD 显示）
} PL_State;

static PL_State s_pl;

// ===============================================================================
// 世界生成（等价于原网页刷新）：随机相互作用参数 + 均匀撒布粒子
// ===============================================================================

static void pl_generate_world(PL_State *s) {
    // 异种间相互作用参数
    for (int32_t i = 0; i < PL_NUM_KIND; i++) {
        for (int32_t j = 0; j < PL_NUM_KIND; j++) {
            if (i == j) continue;
            s->param_a[i][j] = -(0.0005f + 0.0005f * pl_frand());
            s->param_b[i][j] = 0.001f * (pl_frand() - 0.5f);
        }
    }
    // 同种间相互作用参数
    for (int32_t i = 0; i < PL_NUM_KIND; i++) {
        s->param_a[i][i] = -(0.001f + 0.0005f * pl_frand());
        s->param_b[i][i] = 0.001f * (pl_frand() - 0.5f);
    }

    // 粒子撒布：NxN 网格均匀铺开并加少量抖动，种类随机
    int32_t n = 0;
    for (int32_t iy = 0; iy < PL_SCATTER_N; iy++) {
        for (int32_t ix = 0; ix < PL_SCATTER_N; ix++) {
            PL_Particle *p = &s->parts[n++];
            p->kind = rand() % PL_NUM_KIND;
            p->x = (2.0f * (float)ix / (float)(PL_SCATTER_N - 1) - 1.0f) * PL_WORLD_HW + 0.2f * (pl_frand() - 0.5f);
            p->y = (2.0f * (float)iy / (float)(PL_SCATTER_N - 1) - 1.0f) * PL_WORLD_HH + 0.2f * (pl_frand() - 0.5f);
            p->vx = 0.0f;
            p->vy = 0.0f;
        }
    }
    s->frame = 0;
}

// ===============================================================================
// 物理步进
// ===============================================================================

// 重建均匀网格（每帧一次）
static void pl_grid_build(PL_State *s) {
    for (int32_t c = 0; c < PL_GRID_CELLS; c++) s->head[c] = -1;
    for (int32_t i = 0; i < PL_NUM_PARTICLES; i++) {
        PL_Particle *p = &s->parts[i];
        int32_t cx = (int32_t)((p->x + PL_WORLD_HW) / PL_GRID_CELL_W);
        int32_t cy = (int32_t)((p->y + PL_WORLD_HH) / PL_GRID_CELL_H);
        if (cx < 0) cx = 0; else if (cx >= PL_GRID_DIM_X) cx = PL_GRID_DIM_X - 1;
        if (cy < 0) cy = 0; else if (cy >= PL_GRID_DIM_Y) cy = PL_GRID_DIM_Y - 1;
        int32_t c = cy * PL_GRID_DIM_X + cx;
        p->next = s->head[c];
        s->head[c] = i;
    }
}

// 相互作用：对每颗粒子累计 3x3 环绕格内邻居施加的加速（周期边界，等价于原作区块邻居克隆）
static void pl_interact(PL_State *s) {
    const float rmax2 = PL_RMAX * PL_RMAX;
    for (int32_t i = 0; i < PL_NUM_PARTICLES; i++) {
        PL_Particle *pi = &s->parts[i];
        int32_t cx = (int32_t)((pi->x + PL_WORLD_HW) / PL_GRID_CELL_W);
        int32_t cy = (int32_t)((pi->y + PL_WORLD_HH) / PL_GRID_CELL_H);
        if (cx < 0) cx = 0; else if (cx >= PL_GRID_DIM_X) cx = PL_GRID_DIM_X - 1;
        if (cy < 0) cy = 0; else if (cy >= PL_GRID_DIM_Y) cy = PL_GRID_DIM_Y - 1;
        for (int32_t dy = -1; dy <= 1; dy++) {
            int32_t gy = (cy + dy + PL_GRID_DIM_Y) % PL_GRID_DIM_Y;
            for (int32_t dx = -1; dx <= 1; dx++) {
                int32_t gx = (cx + dx + PL_GRID_DIM_X) % PL_GRID_DIM_X;
                for (int32_t j = s->head[gy * PL_GRID_DIM_X + gx]; j >= 0; j = s->parts[j].next) {
                    if (j == i) continue;
                    PL_Particle *pj = &s->parts[j];
                    // 周期边界下的最近镜像距离
                    float ddx = pj->x - pi->x;
                    float ddy = pj->y - pi->y;
                    if (ddx > PL_WORLD_HW) ddx -= PL_WORLD_W; else if (ddx < -PL_WORLD_HW) ddx += PL_WORLD_W;
                    if (ddy > PL_WORLD_HH) ddy -= PL_WORLD_H; else if (ddy < -PL_WORLD_HH) ddy += PL_WORLD_H;
                    float dd = ddx * ddx + ddy * ddy;
                    if (dd == 0.0f || dd > rmax2) continue;
                    float d = sqrtf(dd);

                    // 距离分段的作用强度（与原作一致）
                    float accel = 0.0f;
                    if (d < PL_RTH1) {
                        accel = s->param_a[pi->kind][pj->kind] * (PL_RTH1 - d) / PL_RTH1;
                    }
                    else if (d < PL_RTH2) {
                        accel = s->param_b[pi->kind][pj->kind] * (d - PL_RTH1) / (PL_RTH2 - PL_RTH1);
                    }
                    else {
                        accel = s->param_b[pi->kind][pj->kind] * (PL_RMAX - d) / (PL_RMAX - PL_RTH2);
                    }
                    pi->vx += accel * ddx / d;
                    pi->vy += accel * ddy / d;
                }
            }
        }
    }
}

// 移动：震颤 + 积分 + 衰减 + 环面回绕（对应原作 move + moveOut/moveIn）
static void pl_move(PL_State *s) {
    for (int32_t i = 0; i < PL_NUM_PARTICLES; i++) {
        PL_Particle *p = &s->parts[i];
        p->vx += PL_JITTER * (pl_frand() - 0.5f);
        p->vy += PL_JITTER * (pl_frand() - 0.5f);
        p->x += p->vx;
        p->y += p->vy;
        p->vx *= PL_DAMPING;
        p->vy *= PL_DAMPING;
        while (p->x < -PL_WORLD_HW) p->x += PL_WORLD_W;
        while (p->x >  PL_WORLD_HW) p->x -= PL_WORLD_W;
        while (p->y < -PL_WORLD_HH) p->y += PL_WORLD_H;
        while (p->y >  PL_WORLD_HH) p->y -= PL_WORLD_H;
    }
}

#endif // PL_USE_FIXED_POINT

// ===============================================================================
// 游戏接口（两种实现共用）
// ===============================================================================

int32_t ui_particlelife_init(Key_Event *key_event, Global_State *global_state) {
    // 粒子数组按需从 PSRAM 申请（进入游戏分配，退出释放）
    if (s_pl.parts == NULL) {
        s_pl.parts = (PL_Particle *)platform_malloc(sizeof(PL_Particle) * PL_NUM_PARTICLES);
        if (s_pl.parts == NULL) {
            global_state->STATE = STATE_GAME_MENU;
            return -1;
        }
    }
    srand((uint32_t)(global_state->timestamp ^ 0x3C3C));
#if PL_USE_FIXED_POINT
    pl_fixed_const_init();
    s_pl.lcg = (uint32_t)global_state->timestamp | 1u;
#endif
    pl_generate_world(&s_pl);

    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);
    return 0;
}

int32_t ui_particlelife_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 按A键(ESC)返回小游戏菜单
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_GAME_MENU;
        return 0;
    }
    // 按D键(回车)或2键：重新随机生成参数与世界（等价于原网页刷新）
    if ((key_event->key_edge == -1 || key_event->key_edge == -2)
        && (key_event->key_code == NANO_KEY_enter || key_event->key_code == NANO_KEY_2)
        && s_pl.parts != NULL) {
        pl_generate_world(&s_pl);
    }
    return 0;
}

int32_t ui_particlelife_render_frame(Key_Event *key_event, Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    if (s_pl.parts == NULL) return -1;

    // ---------------- 逻辑更新 ----------------
    pl_grid_build(&s_pl);
    pl_interact(&s_pl);
    pl_move(&s_pl);
    s_pl.frame++;

    // ---------------- 渲染 ----------------
    gfx_soft_clear(gfx);

    // 顶栏信息
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"粒子生命", 6, 2, 255, 255, 255, 1);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, L"A返回 D重置", 236, 2, 180, 180, 180, 1);

    // 粒子（2x2 像素点；周期边界在视口边缘自然截断）
    for (int32_t i = 0; i < PL_NUM_PARTICLES; i++) {
        PL_Particle *p = &s_pl.parts[i];
#if PL_USE_FIXED_POINT
        int32_t sx = PL_CENTER_X + ((p->x * (int32_t)PL_SCALE) >> 20);
        int32_t sy = PL_CENTER_Y + ((p->y * (int32_t)PL_SCALE) >> 20);
#else
        int32_t sx = PL_CENTER_X + (int32_t)(p->x * PL_SCALE);
        int32_t sy = PL_CENTER_Y + (int32_t)(p->y * PL_SCALE);
#endif
        const uint8_t *col = S_PL_PALETTE[p->kind];
        for (int32_t oy = 0; oy < 2; oy++) {
            int32_t py = sy + oy;
            if (py < PL_HUD_H || py >= (int32_t)gfx->height) continue;
            for (int32_t ox = 0; ox < 2; ox++) {
                int32_t px = sx + ox;
                if (px < 0 || px >= (int32_t)gfx->width) continue;
                gfx_set_pixel(gfx, (uint32_t)px, (uint32_t)py, col[0], col[1], col[2]);
            }
        }
    }

    gfx_refresh(gfx);
    return 0;
}

void ui_particlelife_on_exit(void) {
    if (s_pl.parts != NULL) {
        free(s_pl.parts);
        s_pl.parts = NULL;
    }
}
