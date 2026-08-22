# 2026-07-13

根据以下汉字字频+拼音的CSV数据，写一个输入拼音、输出候选字列表的C语言程序，可以有必要的辅助函数（例如从文件系统中读取CSV文件并据此建立全局查找表）和用于测试的主函数。详细需求如下：

1、提供必要的对外接口，主要的对外的函数签名为 size_t pinyin_to_hanzi(char *pinyin, uint32_t *hanzi_candidates);

2、函数的输入是拼音字符串（例如“di2”），含尾部的数字；输出是hanzi_candidates，也就是所有备选汉字的Unicode码点（UTF-32编码值）的列表，按照CSV给出的字频顺序降序排列，使得字频高的排在前面；返回值是备选汉字数。

3、输入的pinyin支持通过前缀查找，也就是输入“de”，能够返回所有的拼音以“de”为前缀的汉字，且按照以下策略排序：完全匹配的汉字排在前面，且完全匹配的汉字按照字频降序排列；不完全匹配的汉字排在后面，也按照字频顺序降序排列。

4、由于要用到嵌入式环境，所以内存占用要最小化。从文件系统中读取CSV文件并据此建立全局查找表的过程，是离线的。一旦建立了全局查找表，你的程序应当能够读取硬编码在程序中的全局查找表，这样以后在嵌入式环境上运行的时候就无需读文件反复解析。

CSV的格式如下（按照字频倒序排列，每一行的各列从左到右分别是：序号、汉字、字频、字频百分位数、汉语拼音，后面的列不用管），完整文件大概有12000多行。

```
1	的	8302698	3.2074998098725	de/di2/di4	...
3	不	3083707	5.8391534593647	bu4/bu2	...
5	了	2588626	8.004814183044	le/liao3/liao4	...
36	得	889089	23.874921469799	de2/de/dei3	...
103	部	469765	39.459344594984	bu4	...
......大概有12000条......
```

你最终应该输出两个产物：

- 一是读取并解析CSV文件、生成C语言硬编码查找表的Python脚本，用于离线建立查找表。
- 二是实现拼音到汉字算法的C语言代码文件，里面应该有一个硬编码的查找表，这个查找表就是上面那个Python脚本生成的。你提供的代码可以先用上面的例子生成一个样例，实际使用时我会手动替换。

------------------------------------

# 2026-08-02

本项目 infer 目录实现了一个自带UI的手持设备，其中大量硬件接口代码是ESP32的。现在，请你参照项目已有代码，针对普通的Linux平台（包括PC、树莓派等使用Linux系统的平台），完善以下文件的实现： @infer/audio_out_linux.c   @infer/mic_linux.c   @infer/platform_linux.c 。这些文件中有大量的桩，你需要尽可能实现这些桩，并指出需要哪些依赖。你在实现这些桩的时候，一方面需要理解它在被调用处是如何调用的，另一方面需要参照ESP32（代码文件名后缀可能是esp32或者core2）上是如何实现的。本需求实现完成后，将实现摘要追加到 @doc/AGENTS.md 中。


## 实现摘要（Linux 平台 HAL 桩实现）

已参照 ESP32 实现（`audio_out_m5core2.cpp`、`mic_m5core2.cpp`、`platform_esp32.cpp`）及各调用点（`ui_ofdm.c`、`ui_musicbox.c`、`ui_ebook.c`、`ui_spectrogram.c`、`ui_animac.h`、`ui_app.c`）的语义约定，完成三个 Linux 平台文件的桩实现，并更新 `infer/Makefile`。

### 依赖

- **alsa-lib**（`audio_out_linux.c`、`mic_linux.c`）：Debian/Ubuntu 安装 `libasound2-dev`，链接 `-lasound`（已加入 Makefile 的 `pod_lite`/`pod`/`tty` 目标）。
- **pthread**（`platform_linux.c` 任务抽象）：编译/链接加 `-pthread`（已加入 Makefile 全局 `CCFLAGS`/`LDFLAGS`）。
- 其余均为 glibc/POSIX 标准接口，无第三方依赖。

### infer/audio_out_linux.c（ALSA 播放）

- 语义对应：ALSA 内部环形缓冲（约 1s，须容纳 OFDM 一帧 31680 采样 ≈0.66s）扮演 ESP32 M5Unified 双槽队列的角色；enqueue 将数据拷贝入缓冲（拷贝语义是"引用+乒乓双缓冲"契约的安全超集）。
- `audio_out_init`：以非阻塞模式打开 `default` 播放设备，单声道 S16_LE，开启 ALSA 软重采样以兼容音乐盒任意文件采样率；支持重复 init（切歌采样率变化）；保存原音量供 close 恢复。
- `audio_out_queue_free`：先冲刷待写缓存（仍有积压则无空槽），再以 `snd_pcm_avail_update()` ≥ 最近一个块长判定空槽（等价于 `isPlaying(0) < AUDIO_OUT_QUEUE_DEPTH`）；自动处理 underrun（`-EPIPE` → `snd_pcm_prepare`）。
- `audio_out_enqueue`：非阻塞，整块必被接受（除不可恢复错误）：尽量写入 ALSA，写不下的剩余采样转入**待写缓存**（pending buffer），由 queue_free 后续冲刷。该设计与 ALSA 插件协商出的缓冲尺寸无关，严格保证 ESP32 的“整块接受/拒绝”双槽契约——修复了 WSLg PulseAudio 插件下缓冲（48000 帧）装不下两个 OFDM 帧（2×31680）导致 `ofdm-tx: enqueue failed` 的问题；音量用软件增益（0~255 线性）实现，不依赖具体声卡 Mixer 元素；`-EPIPE`/`-ESTRPIPE` 经 `snd_pcm_recover` 恢复后自动重发手中数据。
- 设备名可用环境变量 `NANO_ALSA_DEVICE` 覆盖（默认 `default`；WSL2 需配置 `/etc/asound.conf` 将 default 路由到 pulse 插件，或用 `NANO_ALSA_DEVICE=pulse`）。
- `audio_out_stop`：`snd_pcm_drop` + `snd_pcm_prepare`；`audio_out_close`：关设备并恢复进入前音量。

### infer/mic_linux.c（ALSA 采集）

- `mic_init`：非阻塞打开 `default` 采集设备，单声道 S16_LE，允许软重采样，缓冲约 100ms（对应 ESP32 的 DMA 8×512 帧 ≈85ms）；幂等。
- `mic_read`：`snd_pcm_wait`（超时上限 100ms，对齐 ESP32 `i2s_channel_read` 的 100ms）+ `snd_pcm_readi` 循环，阻塞至读满或超时，允许返回部分采样；overrun（`-EPIPE`）自动 `prepare` 恢复。可被独立任务线程安全调用（ui_ofdm 采集任务场景）。
- `mic_close`：幂等；Linux 上采集/播放为独立 PCM 设备（dmix/dsnoop），无需像 ESP32 那样切换 I2S 外设、恢复扬声器。

### infer/platform_linux.c（平台抽象）

- `set_sys_time`：`timegm`（UTC，对齐调用点语义）+ `clock_settime`；需 root/CAP_SYS_TIME，非特权时静默失败。
- `fs_init`：原生文件系统，直接返回 0。
- `list_files`：`opendir`/`readdir`，完整遵守 ESP32 契约——`filenames==NULL` 仅计数；非 NULL 时逐个 `platform_malloc` 填充（调用方 free）；分配失败回滚并返回 -2；跳过 `.`/`..`。
- `platform_write_buffer_to_file`：`fopen("wb")` 创建/截断写入；`platform_is_directory`：`stat` + `S_ISDIR`。
- 随机访问文件 API（`platform_file_open/size/seek/read/close`）：`static FILE*` 全局单句柄（与 ESP32 SD File 语义一致），open 时先关旧句柄，size 查询保持原读写位置，read 返回实际字节数。
- 堆查询（`platform_get_free_heap_size`/`platform_get_largest_free_block` 及 `_internal` 变体）：以 `sysinfo` 可用物理内存近似（Linux 无堆碎片化问题，虚拟内存下最大连续块≈可用内存）；失败时兜底 512MB。**返回值不可为 0**，否则 `ui_animac.h` 会拒绝创建编辑器（门槛 512K+64K）。
- 任务抽象（对应 FreeRTOS）：基于 pthread——trampoline 包装 `void(*)(void*)` 入口；`stack_bytes` 经 `pthread_attr_setstacksize` 设置（低于 `PTHREAD_STACK_MIN` 用默认）；任务以 DETACHED 创建，入口返回即回收；`core>=0` 时 `pthread_setaffinity_np` 绑核（失败忽略）；`priority` 在无特权 Linux 下无法映射为实时优先级，忽略；`platform_task_delete_self` = `pthread_exit`（不返回）；`platform_task_delete` = `pthread_cancel`（清理兜底）；`platform_task_delay_ms` = `usleep`。
- `set_vibration`：无振动马达，空操作；主音量 get/set：进程内静态变量，初值 16（与 `ui_app.c` `global_state->volume` 初值一致）。
- **路径语义差异**：路径按原样传递宿主文件系统，"/" 指真实根目录（非 ESP32 的 SD 卡根）；电子书/音乐盒/相册等以 "/" 枚举的应用将枚举真实根目录，模型等绝对路径（`MODEL_ROOT_DIR`、`WALLPAPER_PATH`）行为不变。

### 验证

本机（Windows）无 Linux 音频开发环境，已在 WSL 中以最小 ALSA 头桩对三文件做 `gcc -fsyntax-only -Wall -Wextra` 检查全部通过；`platform_linux.c` 在 `NANO_TTY`/`NANO_POD_LITE_RPI5`/`NANO_POD_RPI5`/`NANO_CLI`/`NANO_SORT`/`NANO_WSS` 各产品宏下均无错误。真实 ALSA 行为（出声、采集）需在装有 `libasound2-dev` 的 Linux 机器上 `make tty`（或 `pod`/`pod_lite`）后实测。

### WSL2 下的音频配置与测试方法

WSL2 中**没有真实声卡**（`/dev/snd` 只有 `timer`，属正常现象），音频是通过 WSLg 在 Windows 侧运行的 PulseAudio 服务器转发出去的（声音最终在 Windows 上播放/采集）。WSLg 会自动挂载套接字 `/mnt/wslg/PulseServer` 并设置环境变量 `PULSE_SERVER=unix:/mnt/wslg/PulseServer`，因此 PulseAudio 通路（`paplay`/`pactl`）开箱即用；但 ALSA 的 `default` 设备默认按真实声卡解析配置，找不到时会报：

```
ALSA lib conf.c:5208:(_snd_config_evaluate) function snd_func_concat returned error: No such file or directory
```

解决办法：安装 ALSA 的 PulseAudio 插件（`libasound2-plugins`），并把 ALSA `default` 路由到 PulseAudio。完整配置与测试步骤（WSL2 Ubuntu 内执行，已实测）：

```bash
# 1. 安装 pulse 插件、ALSA 工具和 PulseAudio 工具
sudo apt update
sudo apt install -y libasound2-plugins alsa-utils pulseaudio-utils

# 2. 测试 PulseAudio 通路（应直接在 Windows 侧听到声音）
pactl info
paplay /usr/share/sounds/alsa/Front_Center.wav

# 3. 把 ALSA default 路由到 PulseAudio（全局配置，一次即可）
sudo tee /etc/asound.conf <<'EOF'
pcm.!default {
    type pulse
}
ctl.!default {
    type pulse
}
EOF

# 4. 测试 ALSA 通路（audio_out_linux.c / mic_linux.c 使用的就是这条路径）
speaker-test -c1 -t wav -l 1
aplay /usr/share/sounds/alsa/Front_Center.wav

# 5. 测试麦克风（WSLg 支持采集转发；若无声，检查 Windows 设置→隐私→麦克风 中对 WSL 的授权）
arecord -d 3 -f S16_LE -r 48000 -c 1 /tmp/t.wav && aplay /tmp/t.wav
```

说明：

- 若不想改系统配置，可跳过第 3 步，运行程序时用环境变量指定设备（代码已支持，无需改代码）：`NANO_ALSA_DEVICE=pulse ./bin/nano_tty`。
- 排查命令：`ls /mnt/wslg/`（确认 WSLg 存在，应有 `PulseServer`）、`echo $PULSE_SERVER`、`aplay -l` / `arecord -l`（WSL2 下显示无实体声卡是正常的，不代表音频不通）。
- 排障判据：第 2 步通而第 4 步不通 → ALSA 未正确路由到 PulseAudio（检查第 1/3 步）；第 2 步就不通 → WSLg/PulseAudio 通路问题（检查 `PULSE_SERVER`、`/mnt/wslg/pulseaudio.log`）。

------------------------------------

# 2026-08-22

HAL 模块体系重构：硬件抽象接口按模块拆分为 `hal_<模块>.h` 头文件（ram/fs/os/misc/audio_in/audio_out/display/imu/key/touch/power），各平台实现命名为 `hal_<模块>_<平台>.c/.cpp`（ESP32 平台后缀为 `m5esp`，Linux 平台后缀为 `linux` 或具体外设名）。上文（2026-08-02 条目）所述三个 Linux 文件的对应关系：

- `audio_out_linux.c` → `infer/hal_audio_out_alsa_linux.c`（并新增 `audio_out_set/get_master_volume` 主音量接口的 Linux 实现：全局缓存 + 软件增益立即生效，对齐 `hal_audio_out_m5esp.cpp` 语义）
- `mic_linux.c` → `infer/hal_audio_in_alsa_linux.c`（`mic_init` 签名更新为 `(uint32_t sample_rate, uint8_t restore_volume)`；Linux 采集/播放为独立 PCM 设备，restore_volume 忽略）
- `platform_linux.c` → 按模块拆分为 `infer/hal_ram_linux.c`（内存分配与堆查询）、`infer/hal_fs_linux.c`（文件系统/对话日志/wchar 转换）、`infer/hal_os_linux.c`（延时/时间戳/关机/RTC/pthread 任务抽象）、`infer/hal_misc_linux.c`（指示灯/振动/蜂鸣，Linux 无对应外设，空操作）；`platform_linux.c` 仅保留 `platform_set/get_master_volume` 全局主音量状态（对齐 `platform_esp32.cpp` 的保留内容）

`infer/Makefile` 与 `infer/mp135.mk` 各目标源文件清单已同步更新为 hal_* 命名，并补充遗漏的 `ui_llm.c`；tty/cli/sort/wss/pod 目标已在 WSL2 中编译链接通过。`pod_lite` 目标存在重构前遗留的源文件清单缺口（ui_app.c 新增的 calendar/dict/ebook/musicbox/ofdm/animac 等模块及 hal_audio_out_alsa_linux.c、IMU 桩未列入），与本次拆分无关，暂未处理。
