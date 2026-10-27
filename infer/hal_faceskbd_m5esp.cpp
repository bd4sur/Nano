#include <stdio.h>
#include <string.h>

#include "hal_faceskbd.h"
#include "hal_os.h"

// ===============================================================================
// Faces Keyboard3 硬件全键盘 —— M5CoreS3 实现（Direct 模式，协议与键码映射见
// hal_faceskbd.h 头注释；矩阵位映射/KEYMAP/校验和算法经 M5Faces 库源码核实）
// ===============================================================================

#if NANO_HAS_FACES_KBD

#include <M5Unified.h>

#define FACESKBD_I2C_ADDR        (0x08)  // I2C 7位从机地址
#define FACESKBD_REG_KEY         (0x00)  // 键值/矩阵数据寄存器（写地址后读）
#define FACESKBD_REG_MODEL_ID    (0xD0)  // 设备类型 ID（Keyboard3 = 0x02）
#define FACESKBD_REG_MODE        (0xF0)  // 操作模式（0=Normal / 1=Direct）
#define FACESKBD_REG_FW_VERSION  (0xFE)  // 固件版本号
#define FACESKBD_MODEL_ID        (0x02)  // Faces Keyboard3

// LED 手动控制（寄存器 0xF1，Bit7=1 手动位控模式；协议 §8.2）：
// Direct 模式固件不再驱动修饰键灯效，由主机接管——FN 层生效时右灯常亮
//（对应 Normal 模式预设 4“FN 单击”），失效即灭
#define FACESKBD_REG_LED         (0xF1)
#define FACESKBD_LED_OFF         (0x80)  // 左右都灭
#define FACESKBD_LED_RIGHT_ON    (0xA0)  // 右灯亮（FN 生效指示）

#define FACESKBD_I2C_FREQ        (400000) // Fast 模式（与 ES7210 初始化同档）
#define FACESKBD_POLL_INTERVAL_MS (5)     // I2C 轮询节流（官方教程同量级）
#define FACESKBD_MAX_FRAMES_PER_POLL (4)  // 单次轮询最多排水的矩阵帧数（防积压丢释放帧）
#define FACESKBD_QUEUE_CAP       (16)     // 事件队列容量（单帧至多产数个事件）

// bring-up 调试：串口打印每个校验通过的矩阵帧。默认必须置 0——本打印位于 Core1
// 热路径：陈旧重读会放大打印量（实测每 5ms 达 4 行 ≈ 36KB/s，远超 115200 的
// 11.5KB/s），TX 缓冲写满后 printf 阻塞 Core1 → 全机 UI 卡顿数秒
//（项目既有教训：-2 重复流时期“阻塞式串口告警拖垮 Core1”），需要时短暂置 1
#define FACESKBD_DEBUG_RAW       (0)

// 逻辑键位 ID：0-34 = KeyMap 行索引（矩阵键），35-39 = 修饰键
#define FACESKBD_MOD_AA          (35)
#define FACESKBD_MOD_ALT         (36)
#define FACESKBD_MOD_ENT         (37)
#define FACESKBD_MOD_SYM         (38)
#define FACESKBD_MOD_FN          (39)
#define FACESKBD_BIT(id)         (1ULL << (id))

// 矩阵（行,位）→ KeyMap 行索引（M5Faces keyboard3_key_map 核实；位序为物理键反序）：
//   row0: b0=P(9) .. b9=Q(0)；row1: b0=DEL(19) .. b9=A(10)；
//   row2: b0=SPC(32) b1=$(28) b2=M(27) b3=N(26) b4=B(25) b5=V(24) b6=C(23) b7=X(22) b8=Z(21) b9=0(31)
static const uint8_t S_MATRIX_KEY_IDX[3][10] = {
    { 9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
    {19, 18, 17, 16, 15, 14, 13, 12, 11, 10},
    {32, 28, 27, 26, 25, 24, 23, 22, 21, 31}
};

// 按键映射表（协议 §10，35 键 × 5 列：默认/aA/SYM/FN/ALT；255=无效；与 M5Faces KEYMAP 一致）
// 注：ALT 列（144-176）不使用——Alt 在本驱动中作为粘滞 Ctrl，不按固件 ALT 层映射
static const uint8_t S_KEYMAP[35][5] = {
    /*  0 */ {'q', 'Q', '#', '~',  144},
    /*  1 */ {'w', 'W', '1', '^',  145},
    /*  2 */ {'e', 'E', '2', '&',  146},
    /*  3 */ {'r', 'R', '3', '`',  147},
    /*  4 */ {'t', 'T', '(', '<',  148},
    /*  5 */ {'y', 'Y', ')', '>',  149},
    /*  6 */ {'u', 'U', '_', '{',  150},
    /*  7 */ {'i', 'I', '-', '}',  151},
    /*  8 */ {'o', 'O', '+', '[',  152},
    /*  9 */ {'p', 'P', '@', ']',  153},
    /* 10 */ {'a', 'A', '*', '|',  154},
    /* 11 */ {'s', 'S', '4', '=',  155},
    /* 12 */ {'d', 'D', '5', '\\', 156},
    /* 13 */ {'f', 'F', '6', '%',  157},
    /* 14 */ {'g', 'G', '/', 180,  158},
    /* 15 */ {'h', 'H', ':', 181,  159},
    /* 16 */ {'j', 'J', ';', 182,  160},
    /* 17 */ {'k', 'K', '\'', 183, 161},
    /* 18 */ {'l', 'L', '"', 184,  162},
    /* 19 */ {8,   8,   127, 8,    163}, // del：BS(8)，SYM=DEL(127)
    /* 20 */ {255, 255, 255, 255,  255}, // aA 修饰键（矩阵无对应位）
    /* 21 */ {'z', 'Z', '7', 186,  165},
    /* 22 */ {'x', 'X', '8', 187,  166},
    /* 23 */ {'c', 'C', '9', 188,  167},
    /* 24 */ {'v', 'V', '?', 189,  168},
    /* 25 */ {'b', 'B', '!', 190,  169},
    /* 26 */ {'n', 'N', ',', 191,  170},
    /* 27 */ {'m', 'M', '.', 192,  171},
    /* 28 */ {'$', '$', 255, 193,  172},
    /* 29 */ {13,  13,  13,  13,   13 }, // Enter（修饰键行 bit2）
    /* 30 */ {255, 255, 255, 255,  255},
    /* 31 */ {'0', '0', 27,  '0',  175}, // SYM+0 = ESC(0x1B)
    /* 32 */ {' ', ' ', ' ', ' ',  176},
    /* 33 */ {255, 255, 255, 255,  255},
    /* 34 */ {255, 255, 255, 255,  255}
};

// FN 层扩展码（180-193，索引 = 码 - 180）→ 导航键：0 = 忽略（预留/Insert）
static const uint8_t S_FN_NAV[14] = {
    /*180*/ 0,             /*181*/ 0,             /*182*/ 0,
    /*183*/ NANO_KEY_up,   /*184*/ 0,             /*185*/ 0,
    /*186*/ NANO_KEY_tab,  /*187*/ NANO_KEY_home, /*188*/ NANO_KEY_end,
    /*189*/ NANO_KEY_pgup, /*190*/ NANO_KEY_pgdn, /*191*/ NANO_KEY_left,
    /*192*/ NANO_KEY_down, /*193*/ NANO_KEY_right
};

static uint8_t  s_available = 0;        // 初始化探测结果
static uint64_t s_last_poll_ts = 0;     // 上次 I2C 轮询时刻（节流）
static uint64_t s_prev_state = 0;       // 上一帧 40 位按下状态（边沿检测）
static uint64_t s_cur_state = 0;        // 本帧处理期间的当前按下状态（层判定）
static uint8_t  s_shift_armed = 0;      // aA 单击粘滞（下次按键大写）
static uint8_t  s_sym_armed = 0;        // Sym 单击粘滞（下次按键符号层）
static uint8_t  s_fn_armed = 0;         // Fn 点按粘滞（下次按键 FN 层）
static uint8_t  s_fn_pending = 0;       // Fn 已按下且尚未修饰任何键（松开=点按，翻转粘滞）
static uint8_t  s_led_fn_on = 0;        // FN 灯效当前输出状态（变化时才写寄存器）
static uint8_t  s_last_frame[10] = {0}; // 上一有效帧原始字节（陈旧重读判停）
// 事件队列：单帧可产多个事件（和弦/Ctrl 组合对），逐次吐出
static Faces_Kbd_Event s_queue[FACESKBD_QUEUE_CAP];
static uint8_t s_queue_num = 0;
static uint8_t s_queue_idx = 0;

static int32_t faceskbd_read_reg(uint8_t reg, uint8_t *buf, uint32_t len) {
    return M5.In_I2C.readRegister(FACESKBD_I2C_ADDR, reg, buf, len, FACESKBD_I2C_FREQ) ? 1 : 0;
}

static int32_t faceskbd_write_reg(uint8_t reg, uint8_t value) {
    return M5.In_I2C.writeRegister(FACESKBD_I2C_ADDR, reg, &value, 1, FACESKBD_I2C_FREQ) ? 1 : 0;
}

int32_t faceskbd_init(void) {
    s_available = 0;
    if (!M5.In_I2C.isEnabled()) {
        // 官方教程范式：内部 I2C 未使能时按板型引脚启动（M5.begin 正常已使能，此为兜底）
        M5.In_I2C.begin(I2C_NUM_1, M5.getPin(m5::pin_name_t::in_i2c_sda),
            M5.getPin(m5::pin_name_t::in_i2c_scl));
    }
    uint8_t model_id = 0;
    if (!faceskbd_read_reg(FACESKBD_REG_MODEL_ID, &model_id, 1) || model_id != FACESKBD_MODEL_ID) {
        Serial.println("Faces Keyboard3: not detected");
        return -1;
    }
    // 切换 Direct 模式：Normal 模式固件对 Alt 单击不产生任何输出（瞬时键，真机实测），
    // 只有 Direct 模式的原始矩阵含 ALT 电平；写模式会清固件矩阵历史（每次开机幂等执行）
    if (!faceskbd_write_reg(FACESKBD_REG_MODE, 0x01)) {
        Serial.println("Faces Keyboard3: switch to direct mode failed");
        return -1;
    }
    uint8_t fw_ver = 0;
    faceskbd_read_reg(FACESKBD_REG_FW_VERSION, &fw_ver, 1);
    s_fn_armed = 0;
    s_fn_pending = 0;
    s_led_fn_on = 0;
    faceskbd_write_reg(FACESKBD_REG_LED, FACESKBD_LED_OFF); // 灯效初始化为已知态（键盘跨复位常供电）
    s_available = 1;
    s_prev_state = 0;
    s_cur_state = 0;
    s_shift_armed = 0;
    s_sym_armed = 0;
    s_queue_num = 0;
    s_queue_idx = 0;
    Serial.printf("Faces Keyboard3: ready (fw v%u, direct mode)\n", (unsigned int)fw_ver);
    return 0;
}

// 键值 → NANO_KEY_*（可打印 ASCII 键码即 ASCII，同软键盘）；无映射返回 NANO_KEY_IDLE
static uint8_t faceskbd_map_base(uint8_t value) {
    if (value >= 0x20 && value <= 0x7E) return value;
    switch (value) {
        case 8:    return NANO_KEY_backspace;
        case 9:    return NANO_KEY_tab;
        case 13:   return NANO_KEY_enter;
        case 0x1B: return NANO_KEY_esc;
        case 0x7F: return NANO_KEY_esc; // Sym+退格 → 软 Esc（维护者裁决）
        default:   return NANO_KEY_IDLE;
    }
}

static void faceskbd_queue_event(uint8_t key_code, uint8_t ctrl_reset) {
    if (s_queue_num < FACESKBD_QUEUE_CAP) {
        s_queue[s_queue_num].key_code = key_code;
        s_queue[s_queue_num].ctrl_reset = ctrl_reset;
        s_queue_num++;
    }
}

// KEYMAP 某列键值 → 事件（255 无效；180-193 为 FN 导航扩展码）
static void faceskbd_emit_column(uint8_t value) {
    if (value == 255) return;
    if (value >= 180 && value <= 193) {
        uint8_t nav = S_FN_NAV[value - 180];
        if (nav != 0) faceskbd_queue_event(nav, 1); // FN 导航：注入前复位软件 Ctrl
        return;
    }
    uint8_t key_code = faceskbd_map_base(value);
    if (key_code != NANO_KEY_IDLE) faceskbd_queue_event(key_code, 0);
}

// 普通键/Enter 按下沿：按当前层状态（Fn > Sym > aA > 默认）取 KEYMAP 列产出事件
static void faceskbd_emit_key(uint8_t key_idx) {
    uint64_t cur = s_cur_state;
    int32_t fn_held  = (cur & FACESKBD_BIT(FACESKBD_MOD_FN))  != 0;
    int32_t aa_held  = (cur & FACESKBD_BIT(FACESKBD_MOD_AA))  != 0;
    int32_t sym_held = (cur & FACESKBD_BIT(FACESKBD_MOD_SYM)) != 0;
    // Fn 为纯 FN 层（Normal 模式行为：Fn+字母→~^& 等、Fn+K/N/M/$ 等→方向/翻页），
    // 不承担任何 Ctrl 职责——Ctrl 全部由 Alt 粘滞键承担。
    // 按住=瞬时层（并取消“点按”认定）；点按粘滞=下次按键 FN 层，用后一次性失效
    if (fn_held || s_fn_armed) {
        if (fn_held) s_fn_pending = 0;
        else         s_fn_armed = 0;
        faceskbd_emit_column(S_KEYMAP[key_idx][3]);
        return;
    }
    if (sym_held || s_sym_armed) {
        faceskbd_emit_column(S_KEYMAP[key_idx][2]);
        if (!sym_held) s_sym_armed = 0; // 单击粘滞一次性生效
        return;
    }
    if (aa_held || s_shift_armed) {
        faceskbd_emit_column(S_KEYMAP[key_idx][1]);
        if (!aa_held) s_shift_armed = 0; // 单击粘滞一次性生效
        return;
    }
    faceskbd_emit_column(S_KEYMAP[key_idx][0]);
}

// 处理一帧 10 字节矩阵数据（调用前已校验 byte0 与校验和）：
// 40 位按下状态做边沿检测——重复读到同一帧产生零边沿零事件，
// 陈旧重读/事件洪泛在机制上不存在（Normal 模式盲轮询痼疾的根治）
static void faceskbd_process_packet(const uint8_t *pkt) {
    uint64_t cur = 0;
    for (int32_t r = 0; r < 3; r++) {
        uint16_t raw = (uint16_t)(((uint16_t)(pkt[1 + r * 2] & 0x03) << 8) | pkt[2 + r * 2]);
        uint16_t pressed = (uint16_t)(~raw) & 0x3FF; // 电平规则：0=按下
        for (int32_t b = 0; b < 10; b++) {
            if (pressed & (1U << b)) cur |= FACESKBD_BIT(S_MATRIX_KEY_IDX[r][b]);
        }
    }
    uint16_t mraw = (uint16_t)(((uint16_t)(pkt[7] & 0x03) << 8) | pkt[8]);
    uint16_t mpressed = (uint16_t)(~mraw) & 0x1F; // bit0=aA bit1=ALT bit2=Enter bit3=SYM bit4=FN
    for (int32_t m = 0; m < 5; m++) {
        if (mpressed & (1U << m)) cur |= FACESKBD_BIT(FACESKBD_MOD_AA + m);
    }
    s_cur_state = cur;

    uint64_t press = cur & ~s_prev_state;
    uint64_t release = ~cur & s_prev_state;

    // 按下沿：修饰键先行（同包和弦的字母才能取到正确层）。
    // Fn 不承担任何 Ctrl 职责——Ctrl 全部由 Alt 粘滞键承担
    if (press & FACESKBD_BIT(FACESKBD_MOD_FN)) {
        s_fn_pending = 1; // 暂记“点按”候选：松开前未修饰任何键 → 松开时翻转点按粘滞
    }
    if (press & FACESKBD_BIT(FACESKBD_MOD_ALT)) {
        faceskbd_queue_event(NANO_KEY_ctrl, 0); // Alt = 粘滞 Ctrl（按一次激活/再按取消）
    }
    if (press & FACESKBD_BIT(FACESKBD_MOD_AA)) {
        faceskbd_queue_event(NANO_KEY_shift, 0); // 与软键盘 SFT 对齐
        s_shift_armed = 1 - s_shift_armed;       // 单击粘滞：下次按键大写；再按取消
    }
    if (press & FACESKBD_BIT(FACESKBD_MOD_SYM)) {
        s_sym_armed = 1 - s_sym_armed; // 单击粘滞：下次按键符号层；再按取消（不产键码，同软键盘 Sym）
    }
    if (press & FACESKBD_BIT(FACESKBD_MOD_ENT)) {
        faceskbd_emit_key(29); // Enter（修饰键行 bit2）
    }
    for (int32_t idx = 0; idx <= 32; idx++) {
        if (press & FACESKBD_BIT(idx)) faceskbd_emit_key((uint8_t)idx);
    }

    // 释放沿：Fn 点按认定——按下后未修饰任何键即松开 = 点按，翻转点按粘滞
    //（FN 层用于下次按键）；按住期间修饰过键的（s_fn_pending 已被清除）不翻转
    if (release & FACESKBD_BIT(FACESKBD_MOD_FN)) {
        if (s_fn_pending) {
            s_fn_pending = 0;
            s_fn_armed = 1 - s_fn_armed;
        }
    }

    // FN 灯效同步（Direct 模式固件不驱动修饰键灯效，主机接管）：
    // FN 层生效（点按粘滞武装或按住）→ 右灯常亮（对应 Normal 预设 4“FN 单击”），
    // 否则灭；仅状态变化时写寄存器（0xF1 手动控制模式，Bit7=1）
    {
        int32_t fn_on = (s_fn_armed || (cur & FACESKBD_BIT(FACESKBD_MOD_FN))) ? 1 : 0;
        if (fn_on != s_led_fn_on) {
            s_led_fn_on = (uint8_t)fn_on;
            faceskbd_write_reg(FACESKBD_REG_LED, fn_on ? FACESKBD_LED_RIGHT_ON : FACESKBD_LED_OFF);
        }
    }

    s_prev_state = cur;
}

int32_t faceskbd_poll(Faces_Kbd_Event *out) {
    if (!s_available) return 0;

    // 先吐出队列中缓存的事件（单帧可产多个事件）
    if (s_queue_idx < s_queue_num) {
        *out = s_queue[s_queue_idx++];
        if (s_queue_idx >= s_queue_num) { s_queue_num = 0; s_queue_idx = 0; }
        return 1;
    }

    uint64_t now = get_timestamp_in_ms();

    // I2C 节流
    if ((now - s_last_poll_ts) < FACESKBD_POLL_INTERVAL_MS) return 0;
    s_last_poll_ts = now;

    // 帧排水：固件对每次键状态变化产生一帧并排队，单次轮询最多读
    // FACESKBD_MAX_FRAMES_PER_POLL 帧，防快打/忙期积压丢失释放帧（卡键）
    for (int32_t f = 0; f < FACESKBD_MAX_FRAMES_PER_POLL; f++) {
        uint8_t pkt[10];
        if (!faceskbd_read_reg(FACESKBD_REG_KEY, pkt, 10)) break; // 总线忙/失败
        if (pkt[0] != 0x0A) break; // 全零=无待读帧；其余=损坏，放弃本轮
        // 陈旧重读判停：与上一有效帧逐字节相同 = 固件清除窗口内读到的旧帧
        //（连续两个真实帧必然不同——每帧对应一次键状态变化），此时 FIFO 已空，
        // 立即停止排水。杜绝陈旧帧在窗口期内被反复处理/打印（真机实测：配合热路径
        // 打印曾形成串口洪泛阻塞 Core1、UI 卡顿数秒）
        if (memcmp(pkt, s_last_frame, 10) == 0) break;
        memcpy(s_last_frame, pkt, 10);
        uint8_t sum = 0;
        for (int32_t i = 0; i < 10; i++) sum += pkt[i];
        if (sum != 0) continue; // 校验失败（10 字节累加和低 8 位应为 0）：丢弃该帧
#if FACESKBD_DEBUG_RAW
        Serial.printf("FacesKbd pkt: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\n",
            pkt[0], pkt[1], pkt[2], pkt[3], pkt[4], pkt[5], pkt[6], pkt[7], pkt[8], pkt[9]);
#endif
        faceskbd_process_packet(pkt);
    }

    // 取出本帧（组）产出的事件
    if (s_queue_idx < s_queue_num) {
        *out = s_queue[s_queue_idx++];
        if (s_queue_idx >= s_queue_num) { s_queue_num = 0; s_queue_idx = 0; }
        return 1;
    }
    return 0;
}

#else // !NANO_HAS_FACES_KBD：无此硬件的平台，空实现保底

int32_t faceskbd_init(void) {
    return -1;
}

int32_t faceskbd_poll(Faces_Kbd_Event *out) {
    (void)out;
    return 0;
}

#endif
