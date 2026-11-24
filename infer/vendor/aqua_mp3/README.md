# vendor/aqua_mp3 —— Aqua MP3 编码器（**编码端**，原样拷贝）

来源：`/home/bd4sur/Aqua` 的 `aqua-c/src/`（C99 命令行编解码器，2026-08 由 JS 版移植）。
本目录只收了 **MP3 编码**所需的文件，供 `infer/am_native_Rec.c` 的录音机
（`(native Rec)`）做 320kbps CBR 压缩：

```
aqua_encoder.c/.h   帧组装、比特储备（reservoir）、窗切换状态机（编码器主体）
aqua_qloop.c/.h     内层循环（量化/码率）+ 外层循环（噪声整形）
aqua_huff.c/.h      哈夫曼编解码（编码端只用写侧）
aqua_mdct.c/.h      MDCT / 混叠消除
aqua_subband.c/.h   分析子带滤波器
aqua_bits.c/.h      比特流写入器（MSB 优先，动态字节缓存）
aqua_common.h       常量与共用宏（采样率/码率/块尺寸枚举）
aqua_tables.h       各种数表（由 Aqua 仓库 gen/gen_tables.py 生成）
```

未收录：`aqua_decoder.*`、`aqua_wav.*`、`aqua_main.c`（解码端与 CLI，本项目用不到；
播放走 `infer/ui_musicbox_mp3.c` 的 minimp3）。

## 改动策略

**零改动**。上面 15 个文件与 Aqua 仓库逐字节一致（算法、数表、默认参数全不动），
升级方式就是整目录重新拷贝。理由与 `vendor/Animac` 一致：算法实现的可信度依赖
"与上游一致"这一事实，本地微调会让"这份码流是不是 Aqua 编的"变得无法回答。

对上游行为的唯一影响来自**调用方式**（在 `am_native_Rec.c` 里）：

- 单声道、44100Hz、320kbps（`aqua_encoder_init(&enc, 1, SAMPLE_RATE_44100, BIT_RATE_320K)`）；
- 逐帧喂 1152 采样（`aqua_encoder_encode_frame` 每次拿到"上一帧 + 当前帧"共 2304 个
  采样，offset 指向当前帧开头）——与 Aqua CLI 的 `doEncode` 循环逐行同构，因此产出的
  码流在 PCM 相同的前提下与 CLI **逐字节一致**（实测同一段 1kHz 正弦：帧结构、
  带内能量占比、RMS 完全相同）；
- 每个 `AquaBitsWriter` 用完即 `aqua_bw_free`（一帧一个，1045 字节量级）。

## 已知特性

- 上游的 `AQUA_DEBUG` / `AQUA_LONG_ONLY` 环境变量仍可用（`getenv`；嵌入式宿主拿不到值，
  等同未设置），本项目不设置它们。
- 窗切换（长块/短块）使用上游的**确定性**攻击检测，不含随机数，因此录音结果可复现。
- 编码器有少量进程级全局数表（`aqua_qloop_init` / `aqua_mdct_init` 构建一次），
  单实例使用即可（录音机是单例）。
