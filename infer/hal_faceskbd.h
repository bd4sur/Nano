#ifndef __NANO_FACESKBD_H__
#define __NANO_FACESKBD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"
#include "hal_key.h"

// ===============================================================================
// Faces Keyboard3 硬件全键盘（I2C 7位地址 0x08，经 Faces Bottom3 的 M5-Bus 挂在
// 内部 I2C 总线，与触屏/PMIC/IMU 共总线；仅 NANO_HAS_FACES_KBD 平台编译实现）
//
// 工作方式（固件 v03，Direct 模式）：
//   - 为什么用 Direct 模式：Normal 模式固件对 Alt 单击不产生任何输出（Alt 是瞬时键，
//     真机实测按 Alt 无任何数据），只有 Direct 模式的原始矩阵含 ALT 电平——
//     Alt=粘滞 Ctrl 只能在 Direct 模式下实现；
//   - 任意键状态变化（按下/松开，含修饰键）固件产生一帧 10 字节原始矩阵并排队，
//     主机写寄存器地址 0x00 后连读 10 字节取走；空闲读出全零=无数据；
//   - 帧格式：byte0=0x0A 长度；bytes1-6 三组矩阵行（高字节 bit7=变化标志、
//     bit6-4=行索引、bit1-0=PA9-8，低字节=PA7-0；电平 0=按下）；bytes7-8 修饰键行
//     （bit0=aA bit1=ALT bit2=Enter bit3=SYM bit4=FN，0=按下）；byte9 校验和
//     （10 字节累加和低 8 位为 0）。矩阵位→键映射与 35×5 KEYMAP 经 M5Faces 库核实；
//   - 主机对 40 位按下状态做边沿检测：重复读到同一帧产生零边沿零事件——
//     Normal 模式盲轮询的陈旧重读/事件洪泛痼疾（真机实测曾致 UI 卡顿）在机制上消失；
//   - 单次轮询最多排水 FACESKBD_MAX_FRAMES_PER_POLL 帧，防快打/忙期积压丢释放帧。
//
// 键码语义（与触屏软键盘全键盘键码对齐，维护者 2026-09 裁决）：
//   字母/数字/符号键（按下沿）→ 按层取 KEYMAP 列（层优先级 Fn > Sym > aA > 默认）：
//     默认层=小写/数字/符号（键码即 ASCII，同软键盘）；
//   Alt → 粘滞 Ctrl：按下沿产 NANO_KEY_ctrl（与软键盘 CTRL 完全等价，按一次激活、
//     再按一次取消）；Alt 不作为固件 ALT 层（144-176 不使用）——Alt 激活 Ctrl 后
//     任意键即软件 Ctrl 组合（Alt+空格=切换输入模式、Alt+0=呼出软键盘、
//     Alt+Enter=换行等，全部经既有软件 Ctrl 层生效）；
//   aA → 产 NANO_KEY_shift（同软键盘 SFT）+ 单击粘滞下次按键大写（再按取消）；
//     按住 aA = 瞬时大写层；
//   Sym → 不产键码（同软键盘 Sym）；单击粘滞下次按键符号层（再按取消）；
//     按住 Sym = 瞬时符号层（Sym+退格 → 软 Esc，Sym+0 → Esc）；
//   Fn → 纯 FN 层（Normal 模式行为，不承担任何 Ctrl 职责——Ctrl 全由 Alt 承担）：
//     点按（按下后未修饰任何键即松开）= 粘滞 FN 层，下次按键生效、再点按取消；
//     按住 = 瞬时 FN 层；FN 层：字母→~^& 等符号、K/N/M/$/X/C/V/B/Z→方向/Home/End/
//     翻页/Tab（导航键注入前复位软件 Ctrl）；
//     FN 灯效：FN 层生效（粘滞武装或按住）→ 右灯常亮、否则灭（主机经 0xF1 手动
//     控制，对应 Normal 预设 4“FN 单击”；Direct 模式固件不驱动修饰键灯效）；
//   Enter（修饰键行 bit2）→ NANO_KEY_enter。
//
// 已知取舍：固件 Normal 模式的 aA/Sym/Fn 双击锁定不支持（以单击粘滞替代）；
// 无按住自动重复（与固件 Normal 模式一致）；修饰键冲突时层优先级 Fn>Sym>aA
//（与固件的 ALT>SYM>FN>aA 不同，固件互斥规则在自主解码下不适用）。
// ===============================================================================

// Faces 键盘事件（每次从 faceskbd_poll 取一个）
typedef struct Faces_Kbd_Event {
    uint8_t key_code;    // NANO_KEY_* 键码
    uint8_t ctrl_reset;  // 1=注入本事件前需先复位全局软件 Ctrl（FN 导航组合键）
} Faces_Kbd_Event;

// 探测并初始化（切换 Direct 模式）：0=检测到键盘；-1=未检测到/切换失败（轮询恒无事件）
int32_t faceskbd_init(void);

// 轮询取键（Core1 主循环每次调用；内部 5ms 节流 I2C + 多帧排水 + 矩阵边沿检测）：
// 返回 1=取到事件（out 有效），0=无事件。单帧可产多个事件（和弦/Ctrl 组合对），
// 经内部队列逐次吐出，调用方应循环取到 0 为止。
int32_t faceskbd_poll(Faces_Kbd_Event *out);

#ifdef __cplusplus
}
#endif

#endif
