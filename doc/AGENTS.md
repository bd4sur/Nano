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

------------------------------------

# 2026-08-28

区分软硬按键事件：触屏 → 4x4 宫格映射/触屏软键盘派生的虚拟键码与实体键盘键码取值相同，在 `get_input_event` 汇聚后无法区分来源。实施两项改造：

- **平台宏 `NANO_HAS_HW_KEYBOARD`（`infer/platform.h`）**：在文件末尾按平台宏统一定义——带实体键盘的 Nano-Pod 系列（`NANO_POD_*`）、`NANO_ESP32_S3/P4`、`NANO_TTY`（终端键盘）为 1；M5Core2/M5CoreS3（仅触屏）及无 UI 的 `NANO_CLI/SORT/WSS` 为 0。采用末尾集中定义而非散落各平台块，单一事实来源，且规避了 `NANO_PLATFORM_M5CORE2` 在非 Arduino 环境也会被默认定义的陷阱。
- **事件来源标志 `Key_Event.is_soft_key`（`infer/ui.h`）**：1-触屏派生软按键（宫格映射或软键盘），0-实体键盘。放在 `Key_Event` 而非 `Global_State`：来源是每个事件的属性，须随事件经队列跨核投递（ESP32 Core1→Core0），全局标志会在按下/松手之间被后续采样覆盖产生竞态；与既有 `is_softkbd`（软键盘 vs 宫格的细分）同一按下时锁存、下降沿沿用的模式，两者互补。打标点集中在 `get_input_event`（`infer/ui_app.c`）：实体键读取后初值 0，宫格映射命中或软键盘接管时置 1。

仅增加区分能力，不改变任何现有行为。已验证：13 个平台宏组合下 `NANO_HAS_HW_KEYBOARD` 取值符合预期（gcc 预处理）；`ui_app.c`/`ui.c` 在 `NANO_TTY`、`NANO_POD_RPI5` 下 `gcc -fsyntax-only -Wall` 通过（仅有既有的无关警告）；`make tty` 在 WSL2 中编译链接通过。

已知遗留：软键盘 claimed 时无条件覆盖已按下的实体键（`key = softkbd_key`），本次仅如实打标，未改优先级。

------------------------------------

# 2026-10-13

新增编译目标 Nano-Web（`make web` → `bin/nano_web`，平台宏 `NANO_WEB`）：C/S 分离架构——服务端（本进程）完整运行 GUI 渲染与交互状态机，经零第三方依赖的极简 HTTP+WebSocket 服务器把帧缓冲推给网页薄客户端；客户端类远程桌面，只负责把画面画到 `<canvas>` 并上报触屏事件。参考 `nano_tty`（终端 ANSI 半块渲染 + ncurses 鼠标模拟触屏）的 HAL 替换路径实现，业务层零改动。

## 实现摘要（Nano-Web 远程桌面目标）

### 架构与文件

- **`infer/nano_ws_server.c/.h`（新增）**：极简 HTTP+RFC 6455 WebSocket 服务器子集（无 libwebsockets 等依赖，本机无此库）。含 SHA-1/Base64 握手签名、帧编解码（三档长度、客户端掩码还原、分片拼装、Ping→Pong/Close 应答）、HTTP 静态页。线程模型：1 accept 线程 + 每客户端 1 连接线程（握手+收包）与 1 发送线程；发送经「全局帧缓冲镜像 + 每客户端脏区并集 + 条件变量」与渲染线程解耦——渲染线程只做 memcpy+唤醒，多条推送在等待期间天然合并，慢客户端按自身带宽降频不反压渲染（等价 A1 脏区语义的网络搬运）。`SIGPIPE` 全局忽略 + `MSG_NOSIGNAL` + `SO_SNDTIMEO` 兜底僵死连接；新客户端首帧从镜像全屏推送。客户端页面磁盘优先（cwd/exe 相对路径候选）+ 构建期嵌入兜底（Makefile 用 sed 从 `web_remote_client.html` 生成 `bin/web_client_html.h`，单一事实来源）。
- **`infer/hal_display_web_linux.c`（新增）**：显示 HAL，`display_hal_refresh` 的脏矩形转 `nano_ws_server_push_frame` 广播；`display_hal_init` 启动服务器（默认 0.0.0.0:9002，`NANO_WEB_PORT` 环境变量可覆盖）。RGB565 路径/背光/睡眠打桩（gfx 为 RGB888 模式，同 tty）。
- **`infer/hal_touch_web_linux.c`（新增）**：触屏 HAL，实现 `web_remote_on_touch`（ws 接收线程回调）维护电平/坐标缓存，`touch_read` 轮询读取（对齐 hal_touch_ncurses_linux.c 的缓存模式）。**点按保底锁存**（`s_down_pending`）：按下事件至少被 `touch_read` 以 pressed=1 消费一次才清除，DOWN+UP 落在同一轮询间隔的快速点按不再被吞（ncurses 触屏的亚帧点按丢失问题在数据源端顺带解决，串行主循环下同样可靠）。
- **`infer/hal_key_web_linux.c`（新增）**：键盘打桩（恒 IDLE）。平台无实体键盘（`NANO_HAS_HW_KEYBOARD=0`，对齐 M5Core2 触屏-only：主菜单走 4x4 宫格触屏映射，文本输入走触屏软键盘/16 键虚拟键盘）。
- **`infer/hal_audio_out_web_linux.c` / `hal_audio_in_web_linux.c`（新增）**：音频打桩（本机无 ALSA 开发包且网页通路不传音频）。init/enqueue/read 恒失败，各调用点（ui_musicbox/ui_ofdm/ui_spectrogram）固有失败处理兜底（无声但 UI 正常）。主音量接口 `platform_set/get_master_volume` 随 audio_out 打桩实现（其余平台同此布局）。
- **`infer/web_remote_client.html`（新增）**：薄客户端。`<canvas>` 渲染（`putImageData`，CSS 4:3 等比放大 + `pixelated`），Pointer Events 统一鼠标/触屏（`touch-action:none` + 指针捕获，单点触控），断线 1s 自动重连（重连后服务端重发 init+全屏帧），状态栏显示连接状态/FPS。
- **`infer/platform.h`**：新增 `NANO_WEB` 平台块（`SCREEN_WIDTH/HEIGHT=320x240`、`MODEL_ROOT_DIR` 同 tty、`NANO_WEB_DEFAULT_PORT=9002`、`BADAPPLE_ENABLED`；ASR/TTS 不启用）。`NANO_HAS_HW_KEYBOARD`/`NANO_HAS_PWR_KEY` 自动取 0（不在既有清单），无需改动。
- **`infer/Makefile`**：`web` 目标（tty 源清单换 web HAL + `nano_ws_server.c`，`-DNANO_WEB -Ibin`，不含 ncurses/alsa/websockets）；嵌入页生成规则；`all`/`clean` 同步。

### 线协议（WebSocket 二进制，多字节整数小端）

- S→C `[0x00][w u16][h u16]` 初始化帧缓冲尺寸（连接首条）；`[0x01][x][y][w][h][RGB888 像素]` 脏矩形。
- C→S `[0x10][action u8][x i16][y i16]` 触屏事件（action 0=松开 1=按下 2=移动，坐标已按画面缩放换算为帧缓冲像素）。
- HTTP `GET /` → 客户端页面；`GET /ws`（Upgrade）→ 数据通路；其余 404。

### 验证（本机 Ubuntu 24.04，gcc 13）

`make web` 编译通过（仅项目既有无关警告）。纯 stdlib Python 测试客户端实测：HTTP 页面/404 正常；WS 握手、init、脏矩形收帧正确（BMP 截图视觉核对：Splash 时钟、主菜单宫格图标、电子书列表/阅读、FLIP 流体均正确渲染，业务数据 `/home/bd4sur/linglong` 图标与电子书文件加载正常）；触屏点按驱动 Splash→主菜单→FLIP 状态切换；菜单控件点按（电子书列表→阅读）、文本区拖动滚动、页眉返回热区均生效；FLIP/OFDM 环回持续渲染场景 3s 内上万脏矩形无丢失卡死；亚帧快速点按（DOWN+UP 背靠背）生效；3 客户端广播画面逐字节一致，RST 暴力断连后服务存活。`make tty` 在本机仅因缺 `libasound2-dev` 失败（与本改动无关，见 2026-08-02 条目）；`NANO_TTY`/`NANO_WEB` 下 ui_app.c 等语法检查无新增告警。

运行：`cd infer && make web && ./bin/nano_web`（默认 0.0.0.0:9002），浏览器打开 `http://<主机>:9002/` 即可。

## 姿态（IMU）经浏览器上报（番茄表重力 / 玲珑仪视角跟随）

nano_tty 无姿态信息；Nano-Web 的姿态由网页客户端浏览器传感器事件提供，链路：
客户端 `devicemotion.accelerationIncludingGravity`（重力向量）+ `deviceorientationabsolute`/`deviceorientation.alpha`（罗盘）+ `screen.orientation.angle` → WS 消息 `0x20` → `hal_imu_web_linux.c`（`web_remote_on_orientation` 缓存 + 低通滤波，`imu_read_angle` 解算）→ `ui_app.c` 非 ESP32 分支既有消费路径（`#ifdef IMU_ENABLED`，NANO_WEB 平台块已定义之）。

- **`infer/hal_imu_web_linux.c`（新增）**：IMU HAL。角度语义对齐实体 Linux IMU 路径（`hal_imu_imu10_linux.c` 输出约定）：roll=屏幕平面内坡度（水平 0、右缘下沉为负，番茄表 `gravity_x=-9.8·sin(roll°)、gravity_y=-9.8·cos(roll°)` 换算重力）；pitch=屏幕法向仰角（竖直朝前 0、平放朝天 +90，玲珑仪 `view_alt`）；yaw=屏幕上方指向的罗盘航向（玲珑仪 `view_azi=yaw+180`）。解算：roll=atan2(ax_s, ay_s)（ax_s/ay_s 为重力向量经屏幕方向角修正后的分量）；pitch=asin(az/|a|)；yaw=fmod(alpha+θ+360, 360)。**屏幕方向角修正为标准旋转 R(θ)**：ax_s=ax·cosθ-ay·sinθ、ay_s=ax·sinθ+ay·cosθ（初版误写成镜像变换导致 θ=0 时 ay_s 反号、roll 错为 -(180-r)°，y 分量重力反向；经倒置用例（roll=±180 时粒子浮向上半屏）与横屏用例数值验证修正）。重力向量指数低通（α=0.18）；无数据时 `imu_read_angle` 返回 -1（对齐无 IMU 桩语义，调用方保持默认重力向下）。
- **协议新增**：C→S `[0x20][flags u8][ax f32][ay f32][az f32][yaw f32][orient i16]`（flags bit0=加速度有效 bit1=航向有效；原始量上报，解算全在服务端，便于测试与修正）。
- **`web_remote_client.html`**：新增「姿态」开关按钮（默认关；iOS 13+ 须在用户手势内 `DeviceMotionEvent/DeviceOrientationEvent.requestPermission`，且页面须 HTTPS/localhost；桌面浏览器无传感器则事件不触发、GUI 保持无姿态默认行为）。约 20Hz 节流上报；`deviceorientationabsolute` 优先、退化 `deviceorientation`。
- **`infer/platform.h`**：NANO_WEB 块定义 `IMU_ENABLED`（无 `IMU_DEVFILE`——web 目标不编译串口 IMU 实现）。
- **`infer/Makefile`**：web 目标 `hal_imu_linux.c` → `hal_imu_web_linux.c`。

### TLS（HTTPS/WSS）：浏览器安全上下文要求

浏览器 DeviceMotion/DeviceOrientation 传感器 API 仅在**安全上下文**（HTTPS 或 localhost）可用——局域网 HTTP 页面上事件不触发（iOS Safari 连 requestPermission 都不存在，Android Chrome 的 Sensor API 同理）。为此服务端内置可选 TLS（`-DNANO_WEB_TLS` + `-lssl -lcrypto`，本机 OpenSSL 3.x）：

- **同端口协议嗅探双模**：连接线程先 `recv(MSG_PEEK)` 看首字节，0x16（TLS ClientHello）且有证书 → `SSL_accept` 走 HTTPS/WSS；否则按明文 HTTP/WS 处理。桌面端可继续用 `http://`（触屏可用），手机端用 `https://`（触屏+姿态均可用），无需双端口。IO 经 `Nano_WS_IO`（fd+SSL*）抽象，收发路径一处改动全链路生效。
- **证书**：`tools/gen_web_cert.sh` 生成自签名证书（SAN 自动含 localhost 与本机全部 IPv4），输出 `bin/web_cert.pem`/`web_key.pem`（bin/ 被 .gitignore，私钥不进仓库）；环境变量 `NANO_WEB_TLS_CERT`/`NANO_WEB_TLS_KEY` 可覆盖。无证书时回退明文并打印提示（行为同未启用 TLS，零依赖环境可去掉宏与库回退纯明文构建）。
- **使用注意**：首次访问 `https://<主机IP>:9002/` 浏览器有自签名警告，点「继续前往/访问此网站」即可（证书例外按 origin 记忆，同源 WSS 连接共用该例外）；彻底消除警告可将 web_cert.pem 安装为设备受信根证书。客户端页面对非安全上下文访问给出明确提示（按钮显示「需 HTTPS 访问」）。
- 验证：curl https/http 双模均 200；Python ssl 客户端（TLS 1.3 + AES_256_GCM）WSS 全链路（握手/收帧/触屏/姿态 roll=-49.99 精确）通过；明文 WS 回归正常。

### 跟手性优化（消息洪泛治理，协议 v2）

问题：高频渲染场景（FLIP 等）下服务端每次 `gfx_refresh` 推送一条 WS 消息，实测每秒 3000~9000 条——浏览器 JS 事件循环被洪泛打满（每条消息都要 createImageData + 23 万像素 RGB→RGBA 逐像素循环 + putImageData），事件队列积压导致画面滞后于触屏操作（不跟手）；叠加脏区并集的跳帧，观感为「丢帧强行实时」。根因是消息频率与浏览器端处理成本，**不是带宽**（全屏帧 230KB×60fps ≈ 14MB/s，局域网可承受；菜单场景脏区仅数 KB），故不引入 PNG/JPEG/RLE 帧压缩（编码延迟反而损害跟手性，且引入依赖）。治理：

- **发送线程固定节奏合并（服务端）**：发送线程不再随 push 即时唤醒发送，改为按帧预算（默认 60fps，`NANO_WEB_MAX_FPS` 环境变量可调 1~120）工作——有脏区但未到下帧时刻则 `pthread_cond_timedwait` 睡到点（cond 时钟设为 CLOCK_MONOTONIC，与计时基准一致），期间渲染端高频 push 仅更新镜像并并入脏区并集；空闲场景首帧即时发出（now >= next_tick 立即可发，菜单单击等单帧响应零额外延迟）。效果：消息率从每秒数千降至恒定 ~65 msg/s（间隔中位 16.0ms、最大 16.4ms），浏览器事件循环不再积压。
- **RGBA8888 服务端展开（协议 v2，消息 0x03）**：RGB888→RGBA 的 23 万像素展开移到服务端发送线程（锁内 ~100µs），浏览器端由逐像素 JS 循环改为 `ImageData.data.set()` 一次原生拷贝 + `putImageData`（ImageData 按尺寸缓存复用）；canvas 上下文 `alpha:false` 免合成混合。帧头新增 u16 发送序号（seq）供丢帧诊断。v1 的 0x01/RGB888 消息退役。
- 验证：菜单单击 tap→首个脏矩形 0.8~1.3ms（localhost，空闲首帧即时）；FLIP 场景 5s 收 325 条、节奏 16.0ms 稳定；慢收客户端（40ms/条）背压下发送节奏自然拖慢至 25/s、seq 连续（合并发生在 seq 分配之前）、最终画面一致；TLS 全链路回归通过。

结论：**帧压缩不必要**——瓶颈是消息频率与浏览器端处理，治理后带宽峰值约 18MB/s（60fps 全屏 RGBA）在局域网 WiFi 可承受范围；若未来需弱网运行，优先方向是 RGB565（带宽 -33%，浏览器端展开仍是一次循环）或脏区矩形细化，而非有损压缩。

### 后续修复（2026-10-13 二轮）

1. **FLIP 流体模拟帧率节流（业务层，ui_app.c）**：FLIP 模拟按固定步长 dt（1.6/60s）推进，**帧率即物理流速**。设备原生帧率受渲染耗时限制（SPI 推屏/终端 ANSI，<30fps）表观正常；但 nano_web 在 x86 上主循环达数千 FPS（界面 FPS 显示 2940~3805），流体以约百倍设计速度演化，远端 60fps 推送只能采到 2% 中间步（「状态刷新极快、中间过程全部丢失」）。修复：`ui_app_flip_render_frame` 开头加 30fps 帧率节流（距上次模拟 <33ms 直接返回）。原生帧率低于 30fps 的平台（ESP32/树莓派/nano_tty）节流不触发，行为不变。验证：界面 FPS=31，消息率 31.2 msg/s（间隔中位 33.0ms），每个模拟步均可上屏。

2. **玲珑仪方位角周期性跳变（客户端数据源去重）**：排查确认非服务端控制逻辑问题（模拟 yaw 0→180 连续变化，view_azi 平滑跟随无跳变）。根因在客户端：`deviceorientationabsolute`（绝对罗盘 alpha）与 `deviceorientation`（相对 alpha，与前者差固定偏移）**同时注册、交替触发**，lastYawDeg 被两套值来回覆写，方位角随之在两个固定值间跳变。修复：见到 absolute 事件后即忽略 relative 事件（`absAlphaSeen` 去重）；iOS 退化路径改用 `e.webkitCompassHeading`（绝对罗盘航向；iOS 的 alpha 为相对值）；姿态关闭路径的 removeEventListener 同步修正。JS 经 node --check 语法校验。
   备注（真机待验证）：alpha 的旋转方向约定（规范逆时针 vs 罗盘顺时针）各浏览器实现不一，若真机上机身转向与画面方位变化相反，客户端取 `(360-alpha)%360` 即可。

------------------------------------

# 2026-10-13（晚间）

新增编译产物形态 Nano-WASM-GUI（`infer/build_wasm_gui.sh` → `infer/webgui/`）：把 infer/ 的**完整 GUI 业务代码**编译为浏览器内运行的 WebAssembly（与 `infer/web/` 既有 LLM-only 的 main_wasm.c/wasi-sdk 路线互不相关）。GUI 与触屏交互完整保留；业务资源（图标/壁纸/电子书/词典等）经 Emscripten `--preload-file` 打进 MEMFS 的 `/home/bd4sur/linglong`（与 PLATFORM_ROOT_DIR 硬编码一致）；音频/传感器外设尽力而为（音频打桩、任务抽象打桩降级；浏览器姿态/物理键盘经 ccall 接入）。

## 实现摘要（Nano-WASM-GUI）

### 工具链（本机无 WASM 工具链，自建）

emsdk 官方源（storage.googleapis.com）实测 320B/s~50KB/s 不可用；改用**清华 Ubuntu 镜像源手动下载 deb 解包组装**：`emscripten 3.1.6~dfsg-7`（noble universe，93MB，**自带 743MB 预编译 sysroot 缓存**——无需再从网络拉取）+ binaryen 108 + clang/lld/llvm-15 + libclang-rt-15 + node-acorn，装于 `/home/bd4sur/app/emscripten-deb/`（emconfig.py 指定 EMSCRIPTEN_ROOT/LLVM_ROOT/BINARYEN_ROOT/NODE_JS/CACHE；bin-wrappers/emcc 包装脚本修正 deb 死链；PATH/LD_LIBRARY_PATH/NODE_PATH 由 build_wasm_gui.sh 自动注入）。hello-world 与完整构建均验证通过。标准 emsdk 亦可（EMSDK_PATH 环境变量优先）。

### 架构与文件

- **平台宏 `NANO_WASM_GUI`**（platform.h 新块）：320x240、`IMU_ENABLED`（姿态经浏览器事件）、无 BADAPPLE（内嵌数据控体积）、无 ASR/TTS；`NANO_HAS_HW_KEYBOARD=0`（触屏-only，物理键盘经 JS 映射等价实体键）。
- **`main_wasm_gui.c`**：`emscripten_set_main_loop` 每帧回调（体与 main.c 串行迭代逐行一致）；浏览器 rAF 节奏驱动。
- **HAL 新增**：`hal_display_wasm.c`（脏区 RGB888→整屏 RGBA side buffer C 侧展开 + epoch 计数；页面 rAF 直读 wasm 线性内存 `data.set+putImageData`，静态画面零拷贝）、`hal_touch_wasm.c`（点按保底锁存同 web 版）、`hal_key_wasm.c`（浏览器 keydown 映射，对齐 hal_key_ncurses_linux.c 键位）、`hal_imu_wasm.c`（与 hal_imu_web_linux.c 同一套屏幕方向角旋转修正与 atan2/asin 解算）、`hal_audio_*_wasm.c`（打桩）、`hal_os_wasm.c`（时间戳走 libc；任务创建恒失败，OFDM 采集等优雅降级）、`hal_ram_wasm.c`（堆查询固定 512MB，过 ui_animac 门槛）。复用：`hal_fs_linux.c`（纯 POSIX，Emscripten FS 兼容）、`hal_power_linux.c`/`hal_misc_linux.c`（纯桩）。
- **`webgui/index.html`**：加载器（进度/状态）+ canvas（alpha:false）+ Pointer Events 触屏 + 物理键盘 + 姿态开关（iOS 权限、absolute/relative 去重、安全上下文提示——localhost/HTTPS 可用）。
- **`build_wasm_gui.sh`**：emcc -O3 -DNANO_WASM_GUI；`--preload-file` 打包 icon/image/ebook/dict/wp*.png/mob.scm/jstest.js（≈19MB）；**不打** music(89MB)/llm(769MB)（音乐盒空列表、LLM 菜单显示「未打包模型」提示条目——`ui_llm.c` 宏门控防止进入 `exit(EXIT_FAILURE)` 加载路径导致 WASM 运行时整体中止）。ALLOW_MEMORY_GROWTH + INITIAL_MEMORY=256MB；无 -pthread/-fopenmp（OpenMP 退化串行；单线程无 COOP/COEP 要求，任意静态 HTTP 服务可托管）。
- **`ui_app.c` 统一包含 `hal_display.h`**：锁屏路径（NANO_HAS_PWR_KEY 因 platform.h 的 M5CORE2 默认定义 quirk 而在全平台编译）原靠隐式声明，clang `-Werror=implicit-function-declaration` 下报错；改统一包含（各平台无害，头文件有保护宏）。

### 验证（node 直接驱动产物 + 帧缓冲导出 BMP 视觉核对）

`node`（屏蔽 fetch 强制 fs 路径）加载 nano_gui.js：runtime 就绪 → Splash 渲染（**MEMFS 中 `/home/bd4sur/linglong/wp.png` 壁纸读取解码成功**）→ 注入触屏点按 → 主菜单 16 宫格（图标全）→ 电子书菜单（MEMFS 文件列表正确枚举）→ 阅读页（note.txt 1/79，页脚按钮全）→ FLIP（界面 FPS=30，30fps 节流生效）→ 注入右倾 50° 姿态（日志 roll=-49.99 精确）→ 粒子质心 160.0→187.9 右移，画面随重力倾斜。HTTP 静态服务（python3 -m http.server）四件套 200。

### 运行

```bash
cd infer && bash build_wasm_gui.sh        # 构建（产物 infer/webgui/）
cd infer/webgui && python3 -m http.server 8642   # 静态托管（端口任选；手机姿态需 HTTPS/localhost）
```

产物 `nano_gui.js/wasm/data` 已加入 .gitignore（webgui/index.html 手写页面入库）。

### 二轮改造：业务资源改为外部挂载（不打进产物）

应需求将 `--preload-file` 打包（.data 19MB）改为**运行时经 HTTP 从外部挂载**（产物只剩 wasm+js）：

- **浏览器平台约束**：「打开文件时才下载」的纯按需只读挂载在主线程做不了——Emscripten 的 `FS.createLazyFile`/自定义同步 FS 后端依赖主线程同步 XHR，而 Chrome 早已禁用主线程 sync XHR；ASYNCIFY 化全部文件读路径复杂度/风险过高。务实等价物：**清单式挂载 + 分级加载**（MEMFS 内容来自外部 HTTP，不打包进产物）。
- **构建侧（build_wasm_gui.sh）**：不再生成 .data；改将 linglong 资源软链接为 `webgui/files`（python http.server / https_static_server.py 均跟随符号链接；资源更新即时生效、不占双份空间），并生成 `manifest.json`（58 个文件：路径+大小+优先级；p1=图标/壁纸/mob.scm/jstest.js 等启动必需 ≈700KB，p2=ebook/dict/image ≈19MB；跳过 llm/music）。
- **页面侧（webgui/index.html）**：`noInitialRun` + `onRuntimeInitialized` 钩子——先拉 manifest → `FS.createPath` 建目录树 → **阶段1** 经 `FS.createPreloadedFile` 并发 8 路拉取 p1 文件写入 MEMFS `/home/bd4sur/linglong/...`（进度显示）→ `callMain()` 启动 → **阶段2** 后台拉取 p2（状态栏角标进度）。C 业务零改动。
- **验证**（node 驱动，含挂载断言）：58/58 文件挂载成功、MEMFS 内 wp.png 字节数与磁盘一致、ebook 目录 15 条目正确枚举；状态机驱动 Splash→主菜单（图标全）→电子书菜单→阅读页→页眉返回→FLIP（姿态右倾 50° 质心 159.6→189.7 右移、FPS=28）。JS 经 node --check；HTTPS 服务 manifest/files 均 200。
- 坑位记录：①emscripten 3.1.6 glue 顶部 `var Module = typeof Module!="undefined"?Module:{}` 在 node cjs 作用域触发 var 提升陷阱（右侧 typeof 命中正在声明的局部变量），**node 下无法预设 Module 配置**（浏览器 `<script>` 顶层无此问题）；node 测试因此采用「main 先跑、资源后挂」时序。②电子书菜单页眉返回热区 y<27（行高×1.5），测试坐标 (290,30) 会误命中首条目重新打开文件，正确坐标 y≤10。③`ui_icon_cache_load` 缓存失败结果（本次会话内不再重试）——p1 含全部菜单图标且挂载完成才 callMain，浏览器流程无此问题。

### 三轮：鹦鹉笼（LLM）WASM 可用化

原 WASM 版鹦鹉笼菜单显示「未打包模型」提示条目。修复后真正可用：

- **发现**：nano_min 轻量引擎的预设模型路径本就在 `PLATFORM_ROOT_DIR/llm/`（linglong 内含 `nano-168m-q80.bin` 167MB），且 `llm_nm_model_enter` 已有文件缺失预检（显示「模型文件缺失」并返回菜单，不 exit）——唯一问题是模型文件未纳入挂载。
- **菜单动态过滤（ui_llm.c，WASM 宏门控）**：`init_model_menu` 按文件存在性（fopen 探测，不占 platform_file_open 全局单句柄）列出 nano_min 条目 + preset 条目；序号→引擎映射表（`s_wasm_mm_*`）供 `model_menu_item_action` 查询；preset 大模型路径经 goto 标签汇入原有加载流程（挂载了小模型文件时自动可用）；全空时显示「无可用模型（未挂载）」。其他平台行为不变。
- **挂载**：`llm/nano-168m-q80.bin`（167MB）作为 p3 优先级纳入 manifest（qwen3-0b6 599MB 单线程 WASM 不可用，仍跳过）；页面阶段3 最后台拉取（状态栏显示模型名+MB 进度）。`INITIAL_MEMORY` 调至 384MB（模型 MEMFS 167MB 驻留 + nano_min 6MB 块缓存 + 业务峰值，ALLOW_MEMORY_GROWTH 兜底）。
- **验证（node 端到端）**：模型菜单显示「[轻] Nano-168M-Q80」→ 选中加载（nano_min block cache 6144KB 日志正常）→ 输入界面就绪 → 九宫格拼音注入按键 6,4 → 候选栏「你呢密米尼」→ ENTER 进选字模式 → 1 选「你」上屏 → ENTER 提交 → **推理流式输出中文文本，226 词元 @ 6.7 TPS**（单线程 WASM，168M Q80）——鹦鹉笼在浏览器 WASM 中完整可用。回归：NANO_WEB 宏下 ui_llm.c 语法检查通过（web 目标行为不变）。
- 拼音输入法选字流程备注（测试用）：组字键（2-9）→ ENTER 进入带序号选字模式 → 数字键 1~5 选定 → ENTER 提交。

### 四轮：鹦鹉笼改为标准引擎模型（去除 [轻] 前缀试验条目）

应需求 WASM 版鹦鹉笼只保留不带前缀的正常模型（Nano-168M-Q80 / Nano-56M-Q80，均为 infer.c 标准引擎）：

- **`platform.h`（NANO_WASM_GUI 块）**：`MODEL_ROOT_DIR` 改为 `PLATFORM_ROOT_DIR "/llm"`（模型真实所在目录；linglong/llm 含 nano-168m-q80.bin 167MB 与 nano-56m-q80.bin 56MB）。
- **`infer.h`**：NANO_WASM_GUI 下不定义 `NANO_USE_MMAP`（Emscripten MEMFS 的 mmap 语义受限且会复制整份内容；走 fread 全量读入分支，与 build_wasm.sh 既有经验一致）。
- **`ui_llm.c`（WASM 宏门控）**：nano_min [轻] 条目移除；菜单改为 **白名单（nano-168m-q80.bin / nano-56m-q80.bin）+ 文件存在性**双重过滤（Qwen3-0.6B 文件虽在 llm/ 但 599MB 不可用，白名单排除；Q4KS/LoRA 条目文件不存在自动隐去）。action 经映射表索引 preset 表，goto 汇入原有加载流程。
- **`build_wasm_gui.sh`**：两个模型均纳入 p3 后台挂载（合计 222MB）。
- **验证（node 端到端）**：菜单仅显示「Nano-168M / Nano-56M」两条（无前缀）；Nano-56M 完整对话（「你」→「我不知道，但我可以向你推荐一些经典的电影。<|eos|>」，25 词元 @ **20.0 TPS**）；Nano-168M 完整对话（「好的，我会尽力为您提供帮助…<|eos|>」，31 词元 @ **7.6 TPS**，167MB fread 加载）。回归：NANO_WEB/NANO_TTY/NANO_POD_RPI5 宏下 ui_llm.c/infer.c 语法检查零错误。

### 五轮：前端加载进度可视化

- **`webgui/loader.js`（新增，node 可测）**：加载逻辑从 index.html 抽出为依赖注入式模块——`nanoFetchWithProgress`（`response.body.getReader()` 逐块下载 + 字节级进度回调，Content-Length 缺失时退化为不定态）与 `nanoMountResources`（manifest 解析 → 目录树创建 → 分级并发池挂载，p1 完成回调 `onP1Ready`）。
- **WASM 二进制进度**：`Module.instantiateWasm` 覆盖（emscripten 默认流式实例化不暴露进度）——自带进度 fetch + `WebAssembly.instantiate`，进度条显示「程序：x.x/5.1 MB（xx%）」。
- **资源挂载进度**：面板第二条进度条「资源（启动必需）：n/32 个 · x/x MB」（p1 阻塞路径精细进度）；p1 就绪 callMain 后面板隐藏，p2/p3（业务资源+222MB 模型）进度转入状态栏文本。
- **bug 修复（进度倒退）**：并发池汇总在「文件完成」事件的 cur 只含 doneBytes、丢了其余下载中文件的 inflight 字节——完成瞬间进度回退（插桩复现：inflight 7 项时完成事件 cur 从 21050 跌到 1667）。修复：完成事件同样汇总 inflight。node 单元回归：72 个进度事件单调（backward=0）、60/60 文件挂载成功、MEMFS 内容字节数与磁盘一致。
- 回归：node 端到端（loader 驱动挂载 → 主菜单/电子书截图正确）；loader.js/index.html 均过 node --check；HTTPS 服务访问 loader.js 200。

### 七轮：浏览器本地资源存储与用户上传管理（不依赖后端服务器）

业务资源除 HTTP 默认源（manifest+files/ 软链接）外，新增**浏览器本地存储**通路（IndexedDB），用户可经网页上传/覆盖/删除资源，作为临时文件系统；应用（C 侧）零改动（路径不变，JS 层喂入 MEMFS）。

- **`webgui/store.js`（新增）**：IndexedDB 存储层（库 `nano-store`；meta 与 blobs 双 object store 分离，list 不物化大 Blob）。驱动注入式设计（`nanoStoreSetDriver`），浏览器默认 IndexedDB 驱动，node 回归用内存 Map 驱动（`makeMemDriver`）。
- **`webgui/loader.js`**：挂载表 = manifest（HTTP 默认源）∪ store 清单——同名覆盖（用户上传优先）、store 独有文件按目录推断优先级（`nanoInferPriority`：icon/壁纸→p1，llm/→p3，其余→p2；用户上传的音乐 music/ 经此并入，音乐盒即可见）。`mountFile` 统一走 `nanoReadResource`（store 命中读 Blob，否则 HTTP 带进度下载）。**p3 模型不再自动挂载**：回调 `onModelsPending` 交页面按需预载；新增 `nanoMountModel`（store 优先；HTTP 拉取成功后写入 store 持久化，下次启动/离线直接用，不再耗流量）。新增 `onManifest` 回调（合并清单含 source 标记，供资源面板展示）。
- **`webgui/index.html`**：底部栏新增「资源」按钮 + 抽屉面板——配额显示（`storage.estimate`，启动时 `storage.persist()` 申请持久化防 Safari 闲置清理）、语言模型预载区（带百分比进度，就绪后「已就绪」）、上传区（多选文件 + 目标目录选择/扩展名自动识别；上传即 `FS.writeFile` 入 MEMFS **当前会话即时生效** + `nanoStorePut` 持久化）、资源清单（来源标记「服务器/本机」；仅本机来源可删除，删除覆盖项后下次启动回退服务器默认）。
- **验证（node 端到端）**：内存 store 增删读查（含中文路径）✓；store 独有电子书并入挂载表（priority 自动推断=2）✓；store 覆盖 icon/nano.png（MEMFS 内容确为 store 版本）✓；p3 两模型不自动挂载且 onModelsPending 正确 ✓；`nanoMountModel` 拉取 56M 模型入 MEMFS（字节数一致）并持久化入 store ✓；业务驱动（主菜单→电子书）截图正常 ✓；三 JS 文件均过 node --check；HTTPS 服务 200。

### 八轮：WASM-GUI 姿态欧拉角适配（俯仰/方位反向修复）

真机反馈：玲珑仪中俯仰角反向、方位角反向；番茄表正常；nano_web 形态在其他设备已验证（**服务端与共享业务代码零改动**，仅适配 WASM 侧）。

- **根因分析**：番茄表只用 roll（由重力向量 ax/ay 经 atan2 决定），玲珑仪另用 pitch（asin(az/|a|)，z 轴）与 yaw（alpha 罗盘）。「FLIP 正常而玲珑仪俯仰反向」自洽地说明该测试浏览器的 `accelerationIncludingGravity` **z 轴符号**与 web 版验证环境相反（屏幕朝上平放时 az≈-9.8）；「方位反向」说明其 `alpha` 按 W3C 规范为**逆时针旋转角**（与罗盘航向相反，而 web 版验证环境的 absolute alpha 直接实现为罗盘航向）。
- **修复（仅 `hal_imu_wasm.c`，WASM 专属）**：pitch 计算中 az 取反；yaw 由 `alpha+θ` 改为 `(θ - alpha) mod 360`（镜像 + 保留屏幕方向角顺时针修正）。C 侧其他平台文件、JS 采集端、nano_web 服务端均未动。
- **验证（node 端到端，玲珑仪截图核对）**：注入新约定 az=-9.8（平放朝天）→ pitch=+90 天顶视角正确；alpha=60（逆时针语义）→ yaw=330 → view_azi=120，画面中央方位标注 135°（东南）正确。
- 教训备注：浏览器间 DeviceMotion/DeviceOrientation 的符号/方向约定确有分歧（iOS/Android/规范实现差异），此类适配只能在各自客户端/HAL 层按实测收敛，暂无统一真值。

### 九轮：WASM-GUI 键盘事件模型重写（按键处理缺陷修复）

缺陷：原 `hal_key_wasm.c` 为「一次性槽位」（keydown 注入、`input_device_read_key` 取走即 IDLE）——在 `get_input_event` 的电平轮询+边沿检测框架下，**长按永远不工作**（下一帧即无按住电平，360ms 阈值达不到）、未监听 keyup、同帧连按覆盖丢键。

修复（参照 `main.c` 拆分事件环路「生产者高频采样/可靠投递，消费者按帧消费」的精神，单线程 WASM 下的等价实现——JS 事件回调即生产者（浏览器事件循环驱动，与 rAF 帧率解耦），C 帧轮询即消费者）：

- **`hal_key_wasm.c` 重写为电平模型**：`wasm_gui_on_key(code, is_down)` 双参数（keydown 置位/keyup 清除，浏览器 keydown 重复事件无害）；上层按下计时/长按(-2)/按住连发/软复位逻辑全部复用。
- **亚帧点按保底（s_key_unread）**：按下后若 keyup 先于下一次轮询（快速点按，电平从未被采到），补报一次按下使边沿检测看到完整 DOWN→UP（与触屏 HAL 的 down_pending 同哲学）；电平一旦被采样即清除未读标记，防长按松开后多报短按。
- **`index.html`**：补监听 keyup（keydown/keyup 均映射注入）。
- **验证（node 端到端，电子书菜单高亮条 y 坐标断言+截图）**：长按 NANO_KEY_down 2s 高亮从 y=42 连发移到 y=213（菜单深处，滚动条同步）✓；长按 up 回顶 ✓；亚帧点按（keydown/keyup 同 tick）移动一次 ✓；普通短按再移动一次 ✓。修复前三者中仅普通短按勉强工作。

### 十轮：WASM-GUI 拆分事件环路（修复低帧率场景点按被误判为长按连发）

缺陷：串行环路（帧回调里 `get_input_event`）下，输入采样与长短按计时都以**渲染帧时刻**进行——玲珑仪等秒级渲染场景中，一次 100ms 物理点按的 DOWN/UP 沿被拆到相隔数秒的帧中，「按住时长」按帧时刻累计超 360ms 阈值 → 误判长按并连发 -2，表现为按一下触发多次动作。该问题在 ESP32（`linglong_m5core2.ino`）与 pt35（`NANO_UI_SPLIT_EVENT_LOOP`）上经**拆分事件环路**解决：输入在生产端以 1-2ms 物理节奏采样、边沿事件经队列可靠投递给消费者。

WASM 单线程等价实现（`main_wasm_gui.c` 重写 + `hal_touch_wasm.c` 重写 + `index.html`）：

- **生产者 `wasm_gui_input_tick`**（EMSCRIPTEN 导出）：由页面 JS 驱动——触屏/键盘事件回调携带 `performance.timeOrigin + e.timeStamp` 的**事件真实时刻**立即调用 + 50ms 心跳（补充无事件时的持续状态判定，如长按到期）。tick 内逐事件回放触屏事件队列（HAL 内），**每个事件以其真实时刻作为 `global_state->timestamp` 驱动一次 `get_input_event` 采样**，按键/触屏边沿事件按 ino 语义投递到结果事件队列（短按 -1 与触屏边沿可靠、-2 可丢弃）。渲染阻塞期间 JS 回调同样延后，但事件时刻由 e.timeStamp 保证，恢复后集中回放判定仍物理正确。
- **消费者 `wasm_gui_frame`**（rAF 节奏）：每帧排空结果队列（触屏边沿位掩码并入、按键每帧至多一个）、触屏电平取共享快照覆盖，随后 main_event_handler/main_periodic_task——逐句对齐 main.c 拆分环路消费者。
- **`hal_touch_wasm.c`**：触屏事件队列（action/x/y/ts；move 队尾合并、DOWN/UP 可靠），`wasm_touch_pop`/`wasm_touch_set_level` 供生产者回放；原电平+点按锁存模型被事件队列取代。
- **关键 bug 记录**：首版 tick 中 `Key_Event ev_prod` 被每次清零，而 `get_input_event` 依赖其跨调用持久状态（prev_key/key_timer/key_mask/触摸边沿锁存）——上升沿每 tick 重复触发、状态机崩坏（点按完全失效）；修复为 `static Key_Event ev_prod`（对齐 main.c 生产者）。另：测试脚本必须带 50ms 心跳，否则触屏 UP 去抖（3 次轮询）永不完成（页面 JS 有心跳）。
- **`wasm_gui_get_state` 调试导出**（读取 global_state->STATE）：端到端脚本的状态轨迹观测设施。
- **验证（node 端到端，状态轨迹断言 + 帧缓冲截图）**：渲染阻塞 1.2s 中的 100ms 点按 → 恰好一次 esc（停在主菜单，无连发；修复前会连发至 Splash）✓；综合回归 12/12 PASS（触屏导航链 Splash→主菜单→电子书→阅读→返回、玲珑仪进入/esc、键盘长按连发深移、亚帧点按、FLIP 姿态质心右移、FLIP esc 返回）✓。index.html/loader.js/store.js 均过 node --check。

### 十一轮：「正在计算」即时提示（玲珑仪等极低帧率场景的输入反馈）

需求背景：拆分环路修复后点按不再误连发，但玲珑仪等秒级渲染场景下点按要等长帧渲染完才有视觉反馈，体验差。参照 `linglong_m5core2.ino` 生产者的 `gfx_draw_busy`（按键事件产生时立即画「正在计算...」蓝条并推帧）。

**最终实现（WASM 侧，不碰平台无关业务模块）**：`main_wasm_gui.c` 的生产者投递函数 `wasm_dispatch_input_events` 中，按键事件（-1/-2 下降沿）入队前，若 `STATE == STATE_LINGLONG || STATE == STATE_ALBUM` 则先 `gfx_draw_busy(global_state->gfx) + gfx_refresh`（调用图形层既有设施，非改动）——`hal_display_wasm` 同步更新 RGBA side buffer 并 epoch++，页面渲染循环下一 rAF（≤16ms）即上屏，用户在长帧渲染开始前立即看到反馈。**不残留保证**：业务下一帧渲染先清屏重绘，提示自然被覆盖消失（与 ino 语义一致）。

前端路线已撤销：初版 DOM 提示（index.html 的 #busy-hint）曾实现后回滚——玲珑仪场景下主线程被连续 rAF 长渲染占满，提示的 setTimeout 回调与新帧上屏的 hide 在同一线程排队竞争，提示刚显示即被清除（「完全看不到」）；且与 ino 的「画进帧缓冲随业务帧覆盖」语义不一致。最终回归 WASM 帧缓冲内实现（本条目）。

**最终修复（「还是没有」根因）**：WASM 主循环渲染与 JS 上屏循环是两个独立 rAF 序列——dispatch 画忙提示后，下一浏览器帧里 wasm 渲染回调先把提示覆盖，JS 上屏回调读到的是覆盖后内容，提示从未上屏。修复：显示 HAL 改为**「refresh 即上屏」推送语义**（对齐 tty/ILI9341 等平台）——`hal_display_wasm.c` 的 `display_hal_refresh` 末尾经 EM_JS `wasm_gui_present_js` 同步 `putImageData` 上屏（C 推 JS，零轮询延迟、无覆盖窗口）；index.html 删除独立 rAF 上屏循环，改为向 JS 暴露 canvas 上下文与 ImageData 单例（`Module._nano_canvas_ctx/_nano_canvas_img`）。

**最终验证（node 端到端，mock canvas 捕获上屏帧）**：玲珑仪中点按调视角——忙提示在点按后 **235ms 同步上屏**（捕获帧含蓝条 1223px + 白字）✓；交付物 `linglong_busy.bmp`（带蓝色「正在计算...」条的玲珑仪界面渲染图）✓；3s 后业务帧覆盖（0 残留）✓；综合回归 12/12 PASS。

**真机不可见根因与最终修复（headless Chromium CDP 实证）**：手机上仍看不到，经无头浏览器（~/.cache/ms-playwright 的 chrome-headless-shell 156，CDP 驱动）逐步定位：①putImageData 拦截器证明忙提示帧确实写入 canvas 位图；②但 screencast/captureScreenshot 证实合成器未将其 paint 到屏幕——**浏览器渲染流水线中 rAF 回调先于 paint**：dispatch 画忙的宏任务结束后，下一轮 rAF 里 wasm 帧回调立即渲染覆盖 canvas 位图，paint 时已是覆盖后内容，提示从未到达屏幕。修复：dispatch 画忙后置 `s_busy_just_drawn`，消费者帧回调本帧**跳过消费/渲染**（让出 paint 窗口），下一帧才消费渲染覆盖（main_wasm_gui.c，不碰业务模块）。③`tools/https_static_server.py` 加 `Cache-Control: no-store`（手机 Chrome 缓存旧 wasm/js 是「改了不生效」的另一原因）。

**最终证据**：headless Chromium（CPU 20x 节流模拟手机）CDP screencast 屏幕输出帧 `/tmp/sc_frame_002.png`——玲珑仪界面顶部中央蓝色「正在计算...」条清晰可见（2477 蓝像素命中），下一帧被业务渲染覆盖（后续帧 0 命中，不残留）✓。综合回归 12/12 PASS。

**验证（node 端到端）**：玲珑仪（默认二次散射在 WASM 下实测仅 ~1.9fps，天然低帧率场景）中点按调视角——dispatch 画忙后提示区精确色像素断言（蓝条 1223px + 白字「正在计算...」177px）在点按后 ~201ms 上屏 ✓；存活至下一业务渲染帧被覆盖（2.5s 后 0 残留）✓；拆分环路综合回归 12/12 依旧 PASS。过程教训：tick 驱动模型下测试脚本必须带 50ms 心跳（页面 JS 有，node 脚本易漏），否则触屏 UP 去抖（3 次轮询）不完成、按键下降沿不产生。

### 姿态验证（纯 stdlib Python 构造 0x20 消息，日志角度值 + BMP 截图双重核对）

- 番茄表（FLIP）：直立 roll=0.02 / 右倾50° roll=-49.97 / 左倾50° roll=+49.97 / 倒置 roll=179.98 / 横屏θ=90 水平 roll≈0 / 横屏右缘下沉40° roll=-40.00——全部与理论值一致；粒子流方向随之改变（右倾质心 x 165→192、左倾→141、倒置粒子浮向上半屏、横屏修正方向一致）。
- 玲珑仪：直立 pitch=0（地平线视界，方位南）/ 平放朝天 pitch=90.00（天顶视角：天空充满、地景退至边缘环带）/ 屏幕朝下 pitch=-90（脚下视角：地景充满、天空退至边缘）/ 右滚 45°（地平线倾斜）/ 航向 yaw=90·横屏θ=90 修正后 yaw=120（方位标注随航向旋转，画面中央由南转为西北）——`view_alt/view_azi/view_roll` 三路均正确跟随。

------------------------------------

# 2026-10-14

为"鹦鹉笼"新增**远端模型**选项：交互方式与端侧模型完全一致（模型菜单 → 输入 → 流式输出 → 结果页），
推理后端换为云端 **OpenAI 兼容流式 API**。为此新增**网络通信 HAL**（`hal_net`），高内聚、不污染其他模块。

## API 信息（不硬编码，基址经环境变量/编译期注入，见 ui_llm.c 的 llm_rm_api_base）

- 服务：内网自建 OpenAI 兼容 API（SSE 流式，明文 HTTP）。**基址不写入源码/文档**（避免随公开仓库
  泄露内网拓扑）：直连平台设环境变量 `NANO_RM_API_BASE`（如 `http://<API主机>:<端口>/v1`）或
  编译期注入 `-DLLM_RM_API_BASE_BUILD=...`；WASM 经同源反代 `/v1`（上游由 `NANO_API_PROXY` 指定）
- 模型：`GLM-5.3-Flash-EXL3`；思考强度：思考模式开（默认）= `max`（`reasoning_effort`）
- 该 API 实测支持 `reasoning_effort ∈ {none, minimal, low, medium, high, xhigh, max}`；
  思考内容经 `delta.reasoning_content` 下发（DeepSeek 风格），正文经 `delta.content`，
  以 `data: [DONE]` 结束；响应为 `Transfer-Encoding: chunked` 的 SSE。

## hal_net：网络通信 HAL（新增）

接口 `infer/hal_net.h`，实现按平台 `hal_net_<平台>.c`，与既有 HAL 体系同构：

- **接口语义（非阻塞/轮询式，与整机协作式主循环契合）**：
  `net_http_post_stream(config, &req)` 发起流式 HTTP POST 后**立即返回**（不阻塞主循环）；
  业务在主循环中反复 `net_http_stream_read(req, buf, n)` 拉取响应体字节块（>0=读到 n 字节，
  0=暂无数据或已正常结束，-1=错误）；`net_http_stream_state()` 查询流状态（IDLE/IN_PROGRESS/
  DONE/ERROR）；`net_http_status_code()` 取 HTTP 状态码；`net_http_close()` 随时中止并释放（幂等）。
- **能力边界刻意收窄**（协议编组不属于本层）：仅明文 HTTP/1.1；响应体按原始字节流透传
  （实现内部处理 Content-Length / chunked / 连接关闭三种分帧）；SSE 等应用层协议由调用方解析；
  单请求单连接，无连接复用/重定向。
- **`hal_net_linux.c`（tty、web 目标）**：每请求一个后台 detached pthread 完成 DNS/connect/
  发送/接收整条交换，经「互斥锁+条件变量的 256KB 有界环形缓冲」向主线程供数——缓冲满时由
  **后台线程**阻塞等待消费（反压在网络侧而非 UI 侧），主循环零阻塞契约不破。带超时的非阻塞
  connect（10s）+ SO_RCVTIMEO（120s，容忍思考间隙）+ `MSG_NOSIGNAL` 防 SIGPIPE。请求结构体
  释放采用「两阶段退休」（坟场列表延后收尸），消除主线程 close 与工作线程退出的竞态。
- **`hal_net_wasm.c`（NANO_WASM_GUI）**：EM_JS 桥接浏览器 `fetch` + `ReadableStream` reader——
  JS 侧异步读取循环把 chunk 缓存进请求队列，WASM 主循环帧回调内同步取走（chunk 大于缓冲时
  拆分，余量留队首）；中止经 AbortController。与 WASM 单线程构建（hal_os_wasm 任务恒失败）兼容。
- **`hal_net_stub.c`（其余平台打桩）**：`post_stream` 恒 -1、`read` 恒 -1、状态恒 ERROR，
  调用方按「网络不可用」降级。已接线：Makefile 的 pod_lite/pod/pod_pt35、mp135.mk（两目标）、
  luckfox.mk。

## 远端模型引擎适配层（ui_llm.c，第三个引擎）

与 infer.c（标准引擎）、nano_min（小鹦鹉笼）同构的**隔离区块**（一切 static，仅经统一钩子被
模型菜单与 STATE_LLM_ON_INFER 调用，会话视图统一挂 `gs->llm_session`）：

- `llm_rm_is_active/activate/release`（登记与释放，幂等）+ `llm_rm_session_begin/step/free`
  （语义对齐 `infer.c` 的 `llm_session_*`）。`session_begin` 组装请求体
  （`{"model", "messages":[{role:user}], "stream":true, "reasoning_effort"}`）并发起流式请求，
  建立 `Nano_Session` 视图；`session_step` 每轮主循环非阻塞拉取网络流，逐行解析 SSE
  （自带 ~150 行极简 JSON 巡游器：对象取键/跳值/字符串反转义含 `\uXXXX` 与代理对→UTF-8），
  `delta.reasoning_content`（思考）与 `delta.content`（正文）按到达顺序拼入输出，
  两者之间插入分隔线（对齐结果页既有风格）；见 `data:[DONE]` 或流关闭 → `LLM_STOPPED_NORMALLY`，
  连接失败 → 附诊断信息（含 HTTP 状态码）后 `LLM_STOPPED_WITH_ERROR`（结果页显示「推理异常结束」）。
- **交互一致细节**：空输入随机预置 prompt；思考模式开关（输入态 Ctrl+2，页眉 Ψ）映射为
  `reasoning_effort=max/none`；esc 中止经既有 `LLM_STOPPED_IN_DECODING` 路径关闭请求；
  词元计数 = 携带内容的 SSE 块数（TPS 统计）；输出缓冲按 UI_STR_BUF_MAX_LENGTH 封顶
  （UTF-8→wchar 增量转换容忍跨块截断的多字节字符，转换量按 wchar 容量封顶防溢出），
  写满即优雅结束。
- **菜单**：`init_model_menu` 两条路径（WASM 白名单过滤 / 其余全量）均在末尾追加
  「远端模型（云端API）」条目；`model_menu_item_action` 选中后**不加载任何本地文件**，
  登记引擎并直接进入输入态（远端/本地互斥，切换时经 `ui_llm_unload_model` 统一卸载）。
- `ui_llm_unload_model`/`ui_llm_deinit` 增补 `llm_rm_release()`。

## WASM 的同源反代（tools/https_static_server.py）

浏览器同源策略下页面无法直接跨源 fetch 该 API（**不支持 CORS 预检**：OPTIONS 501，响应无
Access-Control-Allow-Origin）。故 `https_static_server.py` 新增**同源反向代理**：`/v1/*` 原样
转发到上游（环境变量 `NANO_API_PROXY` 指定上游，未设置则不启用反代），流式逐块透传（SSE 不缓冲）；
WASM 侧 `llm_rm_api_base()` 在 NANO_WASM_GUI 下取相对路径 `/v1`，其余平台经 `NANO_RM_API_BASE` 直连。

## 验证（本机 Ubuntu 24.04，三目标全部端到端实测真实 API）

- **hal_net_linux + 适配层（无头 C 测试）**：`#include "ui_llm.c"` 直驱 `llm_rm_*` 钩子：
  完整会话（854 块、37.3s、22.9 TPS，思考+分隔线+正文齐全）；空输入随机 prompt 路径；
  中止路径（数步后释放，无崩溃悬挂）；思考关闭路径（`reasoning_effort=none` → 无思考内容、
  无分隔线）；JSON/SSE/UTF-8 截断重组单元用例（`\uXXXX`、代理对、跨块多字节）全过。
- **tty 目标**：全量源文件 + hal_net_linux.c 完整链接通过（本机缺 libasound2-dev，ALSA 以
  最小头文件桩 + 链接桩代替，与 2026-08-02 条目方法一致；无 net 相关未定义符号）。
- **web 目标（9006）**：`make web` 编译链接通过；自写极简 WS 客户端（RFC6455 子集）注入触屏
  驱动真实 GUI：Splash→主菜单宫格→鹦鹉笼→模型菜单拖动滚动→点选「远端模型（云端API）」→
  [16键]确认→推理流式（45.9 TPS 实时显示）→自然结束（服务端日志
  `[llm_rm] session closed (chunks=3555, http_status=200)`，结果页 3555 词元/27.0 TPS）；
  全程帧缓冲截图逐帧核对（帧协议 0x03 脏矩形合成）。
- **wasm 目标（9002）**：`build_wasm_gui.sh` 构建通过；headless Chromium（chrome-headless-shell
  + CDP）加载页面，键盘注入驱动状态机（Splash→主菜单→模型菜单→远端模型→推理），真实经
  同源反代流式生成：状态轨迹 -1→-2→4→0→8→10，堆内出现思考/正文分隔线，从结果页缓冲提取
  完整输出（思考英文+分隔线+中文正文，3992 字符）；`reasoning_effort=max` 生效。
- 7 个平台宏（POD_LITE_RPI5/POD_RPI5/POCKETTERM35/WEB/CLI/SORT/WSS）下 ui_llm.c 语法检查零错误；
  wasm（NANO_WASM_GUI）构建零错误。注意 webgui/files 符号链接与产物由构建脚本再生（入库不变）。

## 其他目标说明

- **ESP32（.ino/Arduino）**：本次不构建。若未来将 ui_llm.c 一并编入 ESP32 固件，需同步把
  `hal_net_stub.c` 加入源文件列表（hal_net.h 为纯 C 头文件，可安全包含；桩使远端模型条目
  在无网络平台上按「网络不可用」降级，不影响其余功能）。
- mp135.mk / luckfox.mk 已接线 `hal_net_stub.c`（本次未构建验证）。

## 测试服务（0.0.0.0）

```bash
cd infer
NANO_API_PROXY=http://<内网API主机>:<端口> python3 tools/https_static_server.py 9002 webgui  # WASM-GUI（HTTPS + /v1 反代）
# 另一终端：web C/S（NANO_WEB_PORT 覆盖默认端口）
NANO_WEB_PORT=9006 ./bin/nano_web
```
- WASM：浏览器打开 `https://<主机>:9002/`（自签证书，点「继续前往」；模型菜单选
  「远端模型（云端API）」即用云端推理，本地模型仍按需预载）。
- Web C/S：浏览器打开 `http://<主机>:9006/`，同样操作（服务端直连 API）。
- tty：`make tty && ./bin/nano_tty`（终端内操作；模型菜单末项为远端模型）。

------------------------------------

# 2026-10-15

将本项目构建为安卓 APK（小米 14 Ultra / Android 15）。**业务源码零改动**：全部安卓化代码位于新子目录 `android/`（Gradle 工程），C 代码经 CMake+NDK 直接引用 `infer/` 原始文件。

## 实现摘要（Nano-Pod Android）

### 平台形态与源码清单

- **平台宏复用 `NANO_WEB`**（320x240、IMU_ENABLED、BadApple、无 ASR/TTS、`NANO_HAS_HW_KEYBOARD=0` 纯触屏）。不用 `NANO_WASM_GUI` 的决定性原因：**远端模型 API 基址（ui_llm.c LLM_RM_API_BASE）在 NANO_WEB 下是完整 URL，WASM 块下是同源相对路径 `/v1`**（安卓没有同源反代）。
- 源清单 = build_wasm_gui.sh 的 GUI 全量清单 − wasm 入口/HAL + 安卓 HAL；并复用 `hal_fs_linux / hal_ram_linux / hal_power_linux / hal_misc_linux / hal_net_linux`（POSIX 纯实现，bionic 直接可用；**hal_net_linux 即远端模型的网络通路**，Android 上实测 54.6 TPS）。
- bionic 差异点仅两处，均以新文件解决（零源码改动）：① `_mbstowcs/_wcstombs` 其实是项目自有实现（utils.c，无需垫片）；② `pthread_cancel` 在 bionic 无符号——`platform_task_delete` 在 Android 上为文档化空操作（唯一调用点是 ui_ofdm 麦克风超时兜底，打桩下 mic_init 恒失败任务不会创建）。

### 路径重映射（nano_path_wrap.c，业务零改动的关键）

`platform.h` 硬编码 `PLATFORM_ROOT_DIR="/home/bd4sur/linglong"`、`MODEL_ROOT_DIR="/home/bd4sur/ai/_model/Nano"`（NANO_WEB 块）。安卓上这两路径不存在且无权限创建。libnano.so 内**强定义** `fopen/open/opendir/stat/mkdir/remove/rename/unlink/access/truncate`（与 libc 同名符号，本 so 内全部引用绑定到此），命中前缀则重写到应用数据目录，经 `*at()` 系调用（openat/faccessat/fstatat/mkdirat/unlinkat/renameat + fdopen/fdopendir）转发；其余路径透传。**不用 `-Wl,--wrap`**：共享库 + 动态 libc 场景下 `__real_xxx` 的动态符号改名行为各家链接器不可靠。**FORTIFY 坑**：NDK 默认 `-D_FORTIFY_SOURCE=2` 会把两参 `open` 重定向到 `__open_2` 绕过包装（症状：infer.c 的 mmap `open` 报 "open failed!" 后 exit）——同时定义 `__open_2/__openat_2` 并 `-U_FORTIFY_SOURCE` 双保险。

### 数据打包与目录解析（v2：必备资源内置 + 用户内容外置，双根 overlay）

- **打包策略**：Gradle 任务只把【必备内置资源】同步进 `assets/linglong/`——图标（icon/）、壁纸（wp*.png）、词典（dict/）、音乐盒脚本（mob.scm）、jstest.js，约 11MB（APK 总 25.6MB）；**内容型数据不打包：llm 模型权重、ebook 电子书、music 音乐、image 照片**（Copy 任务 exclude + doFirst 清理陈旧目录——Gradle 增量复制会残留被排除文件）。
- **双根 overlay 路由**（nano_path_wrap.c v2）：`nano_path_set_roots(user_root, builtin_root)`；对 `linglong/<rel>` 提取顶层段 top（"llm"/"ebook"/…；根文件无段），**user_root 非空且 user_root/top 存在 → 路由用户根，否则内置根**（根文件按整个 rel 探测）；结果按 top 缓存（互斥锁保护；进程内布局稳定，零后续开销）。MODEL_ROOT_DIR 前缀视作 `/llm` 子目录走同一路由。**两根可自然合并**：用户只放 llm/ebook/music/image → 这些走用户侧，图标/壁纸/词典走内置；用户放完整 linglong → 全走用户侧；未放置 → 全内置。写路径随路由（chat.jsonl 首次写入落到内置根=App 私有目录）。
- **运行时引导（MainActivity.resolveAndPrepare）**：① 内置根 = `filesDir/linglong`，按清单增量解压必备资源（`wp.png` 就绪标记 + `.nano_data_version` 版本戳，不符自动重解压）；② 用户根 = 三候选（`/storage/emulated/0/linglong`（需"所有文件访问"，启动页授权按钮）、`Android/data/<pkg>/files/linglong`）中**首个含内容子目录（llm/ebook/music/image 其一非空）**者；无则空串全走内置。

### 事件环路 / 显示 / 输入

- **拆分事件环路**（对齐 main.c NANO_UI_SPLIT_EVENT_LOOP）：JNI 触屏事件入 HAL 队列（UI 线程）→ 生产者 pthread 逐事件以真实时刻回放 `get_input_event`（1ms 节拍）→ SPSC 结果队列（触屏边沿/短按可靠、长按重复可丢弃，玲珑仪忙提示同 main.c）→ 消费者 pthread 60fps 预算节流排空并跑 `main_event_handler/main_periodic_task`。IMU 经 JNI 注入（LPF α=0.18 同 wasm）。
- **显示（hal_display_android.c）**：gfx RGB888 320x240 → ANativeWindow RGB565，最近邻整数倍缩放 + 居中 letterbox；**触屏换算与缩放同源**（`逻辑=(窗口-偏移)/倍数`）。**重大教训：ANativeWindow 多缓冲下不能做脏矩形局部推送**——lock 到的后备 buffer 内容过期，脏区外像素回退，出现"键盘画了又消失/菜单残留"式时好时坏（单缓冲 SPI 屏无此问题，故上游 HAL 无此坑）。修复为**每次全量缩放 blit**（320x240→1920x1440 约 2-4ms），帧缓冲为唯一真相。
- **IMU 方向约定按 AOSP 源码钉死**：`getRotation()` 是绘制内容旋转角（设备逆时针 90° 时返回 ROTATION_90，Display.java 原文）；屏幕上方 `u_d=(sinθ,cosθ,0), θ=getRotation()*90`；`getRotationMatrix` 为设备→世界（列=设备轴，世界 X东Y北Z天）；yaw=atan2((R·u_d).东,(R·u_d).北)；TYPE_ACCELEROMETER（反力）下 `roll=atan2(ax_s,ay_s)`、`pitch=asin(az/|a|)`（与 hal_imu_wasm.c 同式，wasm 输入 z 轴为非标符号故其取反、Android 标准约定不取反；FLIP 只用 roll 不受影响）。

### exit() 拦截（模型缺失不炸 App）

业务在模型缺失路径 `exit(EXIT_FAILURE)`（桌面单进程语义），安卓会杀进程。libnano.so 覆盖 `exit()`：消费者每帧 arm `setjmp` 恢复点，`exit` 时 longjmp 回主循环——加载流程在返回前不切 UI 状态，恢复后 GUI 停留模型菜单、状态完全一致（并强制 PREV_STATE=-999 触发下帧重绘清残留）；非武装线程保持 `_exit` 原语义。实测点选不存在的 Qwen3-4B-Think-Q4KS：App 存活、菜单正常。顺带消除 exit 路径与渲染线程竞争 stdio 锁的 FORTIFY abort。

### 音频 HAL（AAudio 实现，已实现并真机级验证）

- **扬声器（hal_audio_out_android.c）**：AAudio 回调模式 + 自管理 SPSC 环形缓冲（1.5s，≥ OFDM 单帧 0.66s 的 2 倍）扮演"播放队列"：`enqueue` 拷贝入环（引用语义的安全超集），`queue_free` 按剩余空间是否容得下最近块返回空槽；缓冲不足整块时余量转 pending 由 queue_free 冲刷（ALSA 版同构，整块必然被接受）；音量软件增益（回调取数时缩放，set_volume/master_volume 立即生效）；欠载静音填充。环形缓冲读写游标为跨线程 SPSC（`__atomic` acquire/release；stop 用"写游标:=读游标"丢弃未播数据，竞争窗口最坏静音一瞬绝不越界）。
- **麦克风（hal_audio_in_android.c）**：AAudio INPUT 流 + `AAudioStream_read` 阻塞读（100ms 超时，允许部分采样，对齐 ESP32 i2s 语义）；RECORD_AUDIO 运行时权限（Manifest + MainActivity 请求，未授予时 openStream 失败按"无麦克风"降级）；restore_volume 忽略（采集/播放独立流，无需 I2S 切换）。
- **模拟器验证（音频后端开启）**：频谱仪进入 → `mic_init: ok (44100 Hz)` + `dumpsys media.audio_flinger` AudioIn_36（BUILTIN_MIC、Standby: no、Timestamp n=1395、Last read 17ms 前——数据持续流动）；音乐盒播放洛天依.mp3 → `audio_out_init: ok (44100 Hz, ring 1.5s)` + AudioOut_D `Standby: no`、**Frames written 4 秒增长 192,576（≈48kHz 持续输出，44.1k 源被系统重采样）**；返回菜单 cleanup 后 **Standby: yes** 回落（stop/close 语义正确）。测试注意：adb push 到 /sdcard 偶发 0 字节落盘（FUSE），需 push 后校验大小；假 wav 会因 wav_open 失败静默走"无法播放"页（业务既有降级）。
- 音频日志走 stderr→logcat 转发（tag NanoPod），无需看图即可断言。

### 构建与验证（Android 15 x86_64 模拟器，1440x3200 竖屏）

- 工具链：JDK17 + SDK 35 + NDK r27c + CMake 3.22.1（sdkmanager 经代理装）；AGP 8.7.3 + Gradle 8.9；`-fopenmp -static-openmp`（infer/tensor/celestial 的 #pragma omp 真机多核并行）；minSdk 30（静态 libomp 依赖 memfd_create）。
- 端到端实测（**竖屏** sensorPortrait，4:3 视窗 1280x960 居中）：安装/内置资源解压/版本管理；splash/主菜单/模型菜单/输入页/16键键盘渲染；触屏宫格映射、列表拖动、页眉返回、页脚键盘开关；**端侧 Nano-56M 从用户文件系统（/sdcard/linglong/llm）加载推理成功**；**远端模型（GLM-5.3-Flash-EXL3，思考+正文）流式推理 54.7 TPS、258 词元（release 包复测）**；**电子书从用户文件系统读取**（note.txt 不在 APK）；exit() 恢复（点选缺失模型 App 存活）；IMU（emulator console 注入加速度，番茄表粒子随重力重新分布）。
- **修复（视口边缘像素外溢）**：全量 blit 的列映射最初把视口外列钳位到边缘列（边缘像素横向涂抹进 letterbox），且视口外的行不写（多缓冲旧内容残留）。修复：**视口外一律 memset 纯黑**（上下整行黑、左右两段黑，xmap 视口外=-1）；模拟器像素级验证两场景 letterbox 非黑像素均为 0。
- **修复（玲珑仪 IMU 方向，小米 14 Ultra 真机实测 2026-10-15，两轮）**：第一轮真机反馈"俯仰角反向、方位角偏移 90°、仅玲珑仪有问题（番茄表/FLIP 正确）"。FLIP 只用 roll（正确 → ax/ay 与 R(θ) 屏幕系旋转无误），玲珑仪 view_alt=pitch、view_azi=yaw+180（ui_app.c）→ 修 pitch/yaw 两量：① **pitch z 分量取反**（`asin(-az/|a|)`，与 hal_imu_wasm.c 的真机适配同向——该环境平放朝天 az≈-9.8，同为取反修正）；② yaw 首版按"参考轴偏移 +90°"修正（常量 `NANO_ANDROID_YAW_REF_SHIFT`）。
- **修复（玲珑仪 IMU 方向，小米 14 Ultra 真机实测 2026-10-15，三轮）**：
  - **第一轮**：真机反馈"俯仰角反向、方位角偏移 90°"。FLIP 只用 roll（正确），玲珑仪 view_alt=pitch、view_azi=yaw+180 → 修 pitch（z 取反）+ yaw（参考轴偏移 +90°）。
  - **第二轮**：真机反馈"pitch=0 时方位角抖动/奇点"。根因：固定参考轴航向在竖直握持时退化。修 yaw 为"背面法向（-z_d）航向"（旋转矩阵第 3 列取负）。
  - **第三轮（本轮）**：真机反馈"方位角偏移 180°"。根因：玲珑仪 view_azi=yaw+180=背面法向航向（取景器：手机背对观察方向），而第二轮的 yaw=背面法向航向，相加后 view_azi=正面法向航向 ≠ 背面。**修复：yaw 改为屏幕正面法向（+z_d）的罗盘航向**（`atan2(R[2], R[5])`，不取负）——**view_azi=yaw+180=背面法向航向=画面中心方向** ✓。三轮实测与"yaw=正面法向航向"的偏移分别为 90/180/0，逐步收敛。
  - **pitch**：第一轮取反后正确（真机确认），未再改动——`asin(-az/|a|)`（真机 TYPE_ACCELEROMETER 平朝天 az≈-9.8，与 wasm 环境同向）。
  - **验证**：模拟器无传感器无法验证 IMU，三轮均为真机实测驱动；修复后待真机复测确认。
- **图标恢复安卓默认**：删除自制 res/mipmap 图标与 Manifest android:icon 引用，APK 无 icon 条目（aapt badging 验证），启动器显示系统默认图标。
- **图标恢复安卓默认**：删除自制 res/mipmap 图标与 Manifest android:icon 引用，APK 无 icon 条目（aapt badging 验证），启动器显示系统默认图标。
- 真机待验证：IMU 方向物理校准（推导已按源码钉死，若有偏差只需改 MainActivity 一处符号）、Tailscale 下远端模型延迟、功耗。
- 已知既有瑕疵（未改动）：寻呼机与模型菜单共用 `w_menu_main`，先访问寻呼机后模型菜单页眉标题残留；模型菜单列出全部预设，点选缺失模型走 exit 恢复路径；计步器把 imu_read_angle 输出当加速度（业务层自注）。
- 产物：`android/app/build/outputs/apk/{debug,release}/app-*.apk`（release 用 debug 签名可直接侧载）。


------------------------------------

# 2026-10-15（晚）

按新一轮需求修改安卓包（延续同日条目）：更名"玲珑"、触屏范围限制、黑边信息绘制。全程 logcat/像素数值断言验证（无截图看图）。

## 修改摘要

- **更名（玲珑/Linglong）**：包名 `com.bd4sur.linglong`、应用标签"玲珑"、Java 包目录/JNI 函数名/Activity 类引用/rootProject/strings 全部同步；versionName 2610（安卓包版本，与视口上方绘制文本 V2610 一致）。
- **触屏有效范围限制**：`nano_display_android_map_touch` 返回 in_view 标志；JNI onTouch 按"触摸序列"处理——DOWN 在视口内才开始序列（视口外的 DOWN 整体丢弃），序列中的 MOVE/UP 滑出视口时钳位投递（保证序列完整、按下状态不卡）。模拟器验证：视口外 tap 无 NanoTouch 事件、视口内 tap 正常、宫格导航（进模型菜单）正常。
- **黑边信息绘制**：全量 blit 后在 letterbox 黑边区绘制——上黑边 4 行白色文本（"玲珑电子核桃 @ Android (V2610)"加粗、"github.com/bd4sur/Nano"、"Copyright (c) 2023~2026 BD4SUR"、"Vibed by BD4SUR using Local GLM5.3-Flash"），下黑边 "FPS = xx.x"（显示 HAL 自测的每秒刷新率，滑动窗口）。文字复用 gfx 12px 点阵字库（`gfx_font_12_get_glyph`，alpha 4bpp 24x24），按 scale 放大，加粗为水平加宽 1 目标像素。
- **修复（黑边文字横向糊成一片，小米 14 Ultra 真机 2026-10-15）**：抗锯齿字模的 **gw 可大于 x_advance**（边缘淡出像素超出步进范围），scale 放大后这些超出部分与下一字符的起始像素重叠；绘制为直接覆盖（后画替换先画），前一字符的尾部被后一字符的低 alpha 值替换 → 横向糊成一片。**修复：文字像素写入改为取最大值**（`if (c > dst) dst = c`）——黑底上 max 等价正确的 alpha 合成（只增不减），重叠区域的亮像素保留。模拟器验证：4 行文字各自 x 范围集中（宽 260-366px）、无糊满全宽，彩色污染 0。
- **笔位步进缺 scale（重要，真机文字横向压缩根因）**：`pen += adv` 缺少 `× scale`——pen 已在窗口像素坐标系（x0/clip_r/文字宽度均含 scale），唯独步进按逻辑像素推进 → 每字符只前进应有的 1/scale → 全部字符挤在一起。修复：`pen += adv * scale`。模拟器验证：第一行文字宽 782 窗口px（预期 796 ✓）、4 行结构清晰。
- **字模 alpha 排布坑（重要，真机乱码根因）**：`mcu_font_get_glyph` 解码的 alpha 按 **gw×gh 紧密排布（stride=实际宽 w）**（`mcu_font_decode(..., alpha, gw*gh)`；graphics.c 读取亦为 `alpha + j*w + i0`）——按 MAX_W=24 做 stride 读取会取到下一行数据，字形完全错乱（真机表现为黑边文字呈密集乱码）。绘制坐标为**基线制**：`draw_x=pen+x_offset`、`draw_y=baseline-y_offset`（y_offset 向上为正），ALPHA_12 基线=行顶+12。模拟器像素级断言：修复后上黑边文字呈清晰 4 行结构、彩色污染 0、FPS 行正常。
- **颜色编码坑（重要）**：RGB565 的均匀灰度须按通道编码（`gray8=q*17 → r=gray8>>3, g=gray8>>2, b=gray8>>3`）——简单乘法 `q*4369` 会产生青色（565 的 G 通道位多，非均匀）。此 bug 曾使文字呈青色、被误判为"黑边彩色污染"。
- **验证（模拟器，像素数值断言）**：黑边彩色污染 0（纯黑+灰白文字）；上黑边文字像素 3466、下黑边 FPS 文字 310；视口内容正常；触屏范围限制生效；宫格导航正常。
- 上一轮的临时诊断代码（标记块/判定线/buf 尺寸与刷新计数日志）已全部移除。

------------------------------------

# 2026-10-18

灵机解释器 Gfx 本地库 + 曼德博集 Scheme 演示（小游戏菜单新入口）。WASM GUI 目标（build_wasm_gui.sh）编译与端到端运行验证通过。

## 修改摘要

- **Gfx 本地库（infer/am_native_Gfx.c/.h）**：把 Nano 图形层（Nano_GFX）封装为灵机 native 库，共 13 个函数（width/height/clear/fill_white/set_pixel/draw_point/draw_line/fill_rect/draw_rect/draw_circle/fill_circle/draw_text/refresh）。宿主经 `am_native_gfx_bind()` 注入目标帧缓冲（未绑定时报运行时错误）。**栈纪律与本项目 vendored 解释器的语句型内建（display/newline）一致**：语句型函数弹参后不压值（仅推进 PC），值型函数（width/height）压 1 值——上游新版编译器的"单值栈纪律"（语句压 #undefined 由编译器弹出）与此不同，同步 upstream 时需一并调整（头文件内已注释说明）。
- **曼德博演示（infer/mandelbrot.scm + ui_mandelbrot.c/.h，STATE_MANDELBROT=79）**：逃逸时间算法 + Bernstein 调色板。宿主 App 层创建 2MB 自适应内存池的 Animac 运行时、注册 System/Math/String/Table/Gfx 五库、按 `((lambda () ...))` 惯例包装加载内嵌源码（am_parse 只收单顶级表达式），每帧 `am_runtime_event_handler` 泵一个 250000 指令时间片（解释器自动抢占分帧，Scheme 按同步循环书写），帧末 gfx_refresh 推脏区；A(ESC) 退出并销毁运行时。源码内嵌：`tools/scm_to_header.py` 在 build_wasm_gui.sh 中把 mandelbrot.scm 转为 mandelbrot_scm.h（UTF-8 字节串，am_mbstowcs 转宽字符），无外部文件依赖；.gitignore 排除生成物。控制台（编辑器模式）亦注册 Gfx 库并绑定 gfx（ui_animac.h），控制台内可直接绘图。
- **小游戏菜单**：新增"曼德博集"条目（第 12 项，item_num 11→12）。

## 解释器关键发现（调试实录，影响后续 native 库/Scheme 开发）

- **vendored 编译器对多元乘法丢参**：`(* a b c)` 只发射一次 mul（先 push 全部参数再单次折叠），首参残留操作数栈造成线性泄漏——曼德博内层循环 `(* 2.0 x y)` 每迭代漏一个 2.0，76800 像素即 OOM（2MB 池下 VM segregated 分配失败、进程 PC=91/204 报"指令执行异常"）。**规避：算术一律二元嵌套**（`(* 2.0 (* x y))`），源码内已注释。上游 src/am_compiler.c 已修（单值栈纪律 + 语句 pop 补偿），vendored 副本未含该修复（pop 补偿代码被注释，"TODO 处理pop问题"）。
- **am_parse 单表达式**：多顶级表达式文件须包 `((lambda () ...))`（与 am_repl/main.c/ui_animac 编辑器模式同惯例）。
- **语句位置调用产值函数会泄漏**（vendored 编译器无 pop 补偿）：System/Math/String 库的语句式用法同样泄漏（每调用 +1 值），小程序无感；本库语句型函数因此设计为不压值（display 语义）。

## 验证（本地 gcc 测试台 + WASM 双通道，帧缓冲数值断言）

- 测试台（/tmp/mb_test_main.c，复刻宿主流程 + display HAL 打桩）：创建池/运行时 → 注册五库 → 解析/链接/编译/加载 → 泵至 IDLE → 校验。断言全过：进程正常结束、**操作数栈深度 0（栈平衡）**、错误 FIFO 空、暗/彩像素占比（19%/81%）、主心形内部点 (212,120)=(8,8,16) 暗色、结束字幕白像素存在；BMP 目检（经典曼德博集形态、蓝黄红渐变、左下角 "Mandelbrot 745 ms"）。
- 同一测试台以 emcc 编译为 WASM 在 node 下执行：**全部断言通过**（144 轮时间片、862ms 渲染、栈深 0）——解释器+Gfx 库+Scheme 程序在 wasm 目标真实运行无误。
- `bash build_wasm_gui.sh` 构建通过（仅既有旧代码警告），产物 webgui/nano_gui.js + nano_gui.wasm（5.1MB）。GUI 内手动路径：主菜单 [7] 小游戏 → 曼德博集（渐进绘制约 1-2 秒完成，A 返回）；灵机控制台编辑器模式亦可直接用 (native Gfx) 绘图。
- 待办（不影响本验证）：GUI 浏览器内人工过一遍菜单入口与 ESC 返回；MCU 平台的渐进渲染时长待实测。


------------------------------------

# 2026-10-18（App化迭代）

曼德博 Scheme 源码从编译期内嵌改为**运行时从业务资源目录读取**，升级为插件式 Scheme App。WASM GUI 重建 + gcc/WASM 双通道验证通过。

## 修改摘要

- **源码迁址**：`infer/mandelbrot.scm` → `/home/bd4sur/linglong/app/mandelbrot.scm`（业务资源目录 app/ 子目录；linglong 非仓库，资源不入 infer/）。删除生成器 `tools/scm_to_header.py` 与生成头 `mandelbrot_scm.h`（.gitignore 条目同步移除）。
- **ui_mandelbrot.c**：`mb_load_source` 改为 `platform_read_file_to_buffer(工作根 + "app/mandelbrot.scm")` → UTF-8 转宽字符 → `((lambda () ...))` 包装 → 解析/链接/编译/加载（原流程不变）。新增防御：源码 >256KB 拒载；加载失败原因（文件未找到/过大/内存不足/语法解析失败/编译失败）存入 `err_load` 显示在错误画面（提示检查 app/mandelbrot.scm）。工作根用 `s_mb_root`（PLATFORM_ROOT_DIR 宽前缀，各平台通用；ESP32 SD 根 = /app/mandelbrot.scm）。
- **build_wasm_gui.sh**：删除头文件生成步骤；manifest 生成把 `app` 加入 P1_DIRS（priority-1 启动挂载，61 files p1=33）。
- **ui_mandelbrot.h**：注释更新为 App 化语义（可插拔、无固件改动）。

## 验证

- gcc 测试台（改为读同一资源文件，与 App 同读取流程）：全部断言通过（723ms 渲染、opstack 深度 0、错误 FIFO 空、像素统计/集内点/字幕与内嵌版一致）；`MB_SRC=/nonexistent` 缺失场景正确走失败路径。
- WASM node 测试台（NODERAWFS 读真实路径）：全部断言通过（870ms）。
- **坑（重要）**：wasm 测试台一度产出 12KB 空壳且无任何告警——`int main(int argc, char *argv[])` 的符号是 `__main_argc_argv`，而 emcc 默认只 `--export-if-defined=main`，名字不匹配 → main 未导出 → **wasm-ld 把整个程序 GC 成空壳**（wasm-ld 明明链接了全部 .o）。上一版内嵌台 `int main(void)` 恰好没事。修法：main 用无参签名（或显式 `-s EXPORTED_FUNCTIONS=_main`，emcc 会正确别名）。**教训：emcc 产物大小异常偏小时先查入口符号**。
- `bash build_wasm_gui.sh` 干净重建通过；manifest 含 `app/mandelbrot.scm`（priority 1）。
- 浏览器使用不变：小游戏 → 曼德博集；改 mandelbrot.scm 后**无需重编固件**，刷新页面即生效（no-store 无缓存）。

------------------------------------

# 2026-10-18（生命游戏Scheme + Scheme App 通用宿主）

"小游戏"新增"生命游戏Scheme"（灵机解释器 + Gfx 库，源码同曼德博从业务资源目录 app/ 运行时读取）。宿主泛化为 ui_scheme_app（两 App 共用），WASM GUI 重建 + gcc/WASM 双通道验证全过。

## 修改摘要

- **App 源码（linglong/app/）**：新增 `gol.scm`（5KB）——康威生命游戏，参考 ui_app.c 的 STATE_GAMEOFLIFE（C 版）：50% 随机撒种、环绕边界、B3/S23。与 C 版差异：细胞 4px（C 版 2px）、`System.set_interval` 100ms/代节流（C 版每帧全速）——解释器由宿主时间片驱动，定时器回调即一代，节奏与帧率解耦。顶部状态条（代数/存活数）用 String.concat + draw_text。
- **通用宿主（ui_mandelbrot.c/h → ui_scheme_app.c/h，函数名同步 ui_scheme_app_*）**：描述符表 `scheme_app_desc_for_state(state)` 按 STATE 选 App（STATE_MANDELBROT=79 曼德博、STATE_SCHEME_GOL=80 生命游戏Scheme，均迁入 ui_scheme_app.h）；新增 App 只需加一个描述符 + 一个 .scm。菜单：游戏菜单 item_num 13，"生命游戏Scheme"→STATE_SCHEME_GOL；状态机两 case 合并。运行期错误叠加显示改为一次性（err_buf_shown）。
- **推屏时机改为 Scheme 控制（重要行为变更）**：`ui_scheme_app_render_frame` 删除无条件 gfx_refresh——跨多帧的绘制（曼德博逐行渐进、生命游戏逐代演进）若中途推屏会闪烁，且"最后一帧是半成品"。现由源码内 `(Gfx.refresh)` 控制：曼德博每行推一次（行粒度渐进），gol 的 step 末尾推一次（每代完整帧）。错误画面/运行期错误叠加仍由宿主推。
- **gol.scm 的两个深坑（详见"解释器关键发现"）**：① `'()` 字面量是模块级静态槽——对它 push 会原地替换槽指向的对象，跨调用残留膨胀（第二次 step 的 push 接着上次的表追加直至堆 OOM）；② `cons` 是 O(n) 全表复制，数千格的表用 cons 建表是 O(n²)。**解法：零分配双缓冲**——seed 一次性 push 构建两页（各一个独立 '() 槽），此后每代 `set_item!` 原地写目标页 + 翻页（完全对齐 C 版）。

## 解释器关键发现（新增）

- **`get_item`/`set_item!` 对浮点索引按 roundf 取整**（mod 恒浮点，可直接做索引）；set_item! 越界报错停机（get_item 越界返回 #undefined，两者不同）。
- **`(push lst v)` 是语句型内建（弹参不压值、原地追加）**——产值函数在语句位置调用泄漏，语句型则安全；`System.set_interval` 产值，用 `(set! _timer (System.set_interval ...))` 消费。

## 验证（gcc + WASM 双通道，测试台配置文件 /tmp/mb_test_cfg.txt 选 App）

- **曼德博回归**：全 PASS（144 轮至 IDLE、opstack 0、像素/集内点/字幕断言、BMP 目检同前）。
- **生命游戏**：全 PASS——400 轮（虚拟时钟 +5ms/轮 ≈ 20 代）进程存活（定时器持续重排）、opstack 0（零分配双缓冲无泄漏）、无错误、活细胞 13.9% 且**空间均匀**（顶部带 20%<40%）、帧缓冲差异 19440（持续演化）；BMP 目检：全屏青色细胞 + 顶部状态条"生命游戏 20 代 存活 840"。
- **规则正确性（blinker 决定性实验）**：确定性种子（行 30 列 39-41 水平三连）→ 首代垂直三连 + 状态条"1 代 存活 3"，位置/规则/绘制全部正确。
- **假象警示（重要）**：测试台虚拟时钟过快（每轮 +25ms）时 interval 堆积、最后一帧是半成品，曾误判为"世界只剩顶部 10 行"；放慢至 +5ms/轮（真实 GUI 时间充裕）后消失。
- **WASM**：同一测试台 emcc 构建下两 App 全 PASS（gol 1336ms/20 代）；build_wasm_gui.sh 重建通过，manifest 62 files（app/ 两源码均 priority 1）。
- 浏览器：小游戏菜单两个 Scheme App 入口；gol.scm/mandelbrot.scm 修改后刷新页面即生效。

------------------------------------

# 2026-10-18（生命游戏实测卡顿/冻结排查与修复）

用户实测反馈：生命游戏Scheme"卡卡的"且"大概率卡在第一帧不动"。排查结论：**不是 GC、不是调度、不是内存**——三个叠加原因，均已修复并端到端验证。

## 排查过程与结论

- **内存**：解释器池 2MB（VM 区+堆区自适应，同 ui_animac 策略）——实测 gol.scm 稳态 opstack 深度 0、全程无 OOM/无错误 FIFO 消息，**内存充裕非瓶颈**。
- **调度**：`am_runtime_event_handler` 每帧泵一个 250000 指令时间片；`runtime_fire_expired_timers` 对 STOPPED 进程的定时器回调正确入队（回 halt 地址 1）。gcc 测试台分别用"虚拟时钟（理想 60fps）"与"真实时钟+16ms 帧间隔（模拟 rAF）"验证，调度均正常——**排除**。
- **GC**：双缓冲零分配设计下每代仅表扩容产生少量垃圾，水位触发正常——**排除**。
- **原因一（功能性 bug）**：`neighbours` 的 8 邻域**抄写错误——`(x+1,y+1)` 写重了（共 9 次 cell-at 调用）**，邻居计数错乱 → 演化结果错误。逐格 trace（op_get_item 打桩比对 S1 期望/实际活细胞索引）定位。修复为程序化生成的右折叠 8 项（杜绝手抄）。
- **原因二（性能）**：CELL=4 时网格 80×60=4800 格，每代约 140 万 VM 指令；WASM GUI 实测帧率仅 9~10fps（headless），每帧 250k 指令 → **每代需 5~6 帧（600ms+），而 interval=100ms 严重堆积** → 视觉卡顿。**修复：CELL 提到 8**（40×30=1200 格，每代约 16 万指令 ≈ 0.7 帧，interval 100ms 内轻松完成）。
- **原因三（测试工件+资源缓存）**：排查期间 gol.scm 曾处于"blinker 调试版"（5 次手动 step、无 interval——第 1 代后必然冻结）；且 **webgui 的 store.js（IndexedDB）对资源"store 优先于 HTTP"**，浏览器会把曾拉取过的 .scm 旧版持久缓存，服务器更新不生效——**清站点数据（IndexedDB）后恢复**。e2e 自动化脚本已改为先清 nano-store 再验证。
- **顺带修正**：cell-at/neighbours 的闭括号数曾反复写错（"unexpected end of input"/"trailing tokens"），最终以程序化生成（Python 拼装+自动平衡断言）落地，**手工数括号不可靠**。

## 验证

- gcc 测试台（gol.scm，配置文件切换）：虚拟时钟与真实时钟（16ms/帧模拟 rAF）**双模式全 PASS**——400 轮进程存活（定时器持续重排）、opstack 0、无错误、活细胞 10~16%、空间均匀（顶部带<22%）、演化差异>2 万像素。
- **浏览器 e2e（playwright + 真实 WASM GUI）**：自动导航（splash→主菜单[7]→游戏菜单 down×12+enter→state=80）后监测 `_wasm_gui_get_fb_epoch`：**8 秒推进 76 次 ≈ 9.5 代/秒**（此前同脚本复现冻结：delta=0）；截图确认"生命游戏 81 代 存活 61"+ 典型晚期形态（静物/振荡器）。FPS 状态栏 headless 下 10fps 仍稳定 10 代/s。
- 曼德博回归：gcc 全 PASS；WASM GUI 重建通过；9002 HTTPS 服务新产物就绪。
- 遗留提示：**用户浏览器若曾缓存旧版 gol.scm，需清一次站点数据**（或等 store 过期策略）；后续可考虑给 store 加 mtime 校验。

------------------------------------

# 2026-10-18（灵机应用程序：Scheme App 统一入口 + 动态列表）

应用户需求重构交互：曼德博集/生命游戏Scheme 两个独立入口**统一收纳为"灵机应用程序"**（小游戏菜单第一项），进入后**动态扫描业务资源目录 app/ 下所有 .scm**，文件名即菜单项，选中即解释执行——**新增 Scheme App 零固件改动**（放入 .scm → 刷新即见）。

## 修改摘要

- **状态机重构（ui_scheme_app.h）**：原 STATE_MANDELBROT/STATE_SCHEME_GOL 合并为 **STATE_SCHEME_APP_LIST(79)**（动态列表）+ **STATE_SCHEME_APP_RUN(80)**（解释执行）。
- **ui_scheme_app.c**：
  - `scheme_app_scan()`：`list_files(PLATFORM_ROOT_DIR "/app")` 枚举 + `.scm` 后缀过滤（不区分大小写）+ 按显示名排序；结果存 `s_apps[]`（title=文件名去 .scm、source_rel="app/xxx.scm"），上限 24 个。**注意 PLATFORM_ROOT_DIR 不带尾分隔符**（曾拼成 "...linglongapp" 导致扫描为空）。
  - 列表菜单复用全局 `w_menu_main`（进入时重扫描+重填 items+ui_widget_menu_init；空目录显示占位项"（app 目录下没有 Scheme App）"）；item_action 记录选中下标并转入 RUN；A 返回游戏菜单。
  - 运行宿主从 `s_apps[s_selected]` 取描述（title 用于错误画面、source_rel 用于加载）；A 返回列表（on_exit 销毁解释器，下次进入重新扫描）。
- **ui_app.c**：游戏菜单删"曼德博集/生命游戏Scheme"两项、第一项加"灵机应用程序"（item_num 12）；item_action case 0→STATE_SCHEME_APP_LIST（原 0-10 顺移为 1-11）；状态机新增 LIST/RUN 两 case。

## 验证（playwright 端到端驱动真实 WASM GUI + gcc 测试台回归）

- **gcc 测试台**：gol.scm（虚拟时钟+真实时钟双模式）与 mandelbrot.scm 全 PASS（回归不受重构影响）。
- **浏览器 e2e 全导航**：splash→主菜单[7]→游戏菜单 enter 第 1 项→**列表（79）正确列出 gol/mandelbrot**（按名排序）→ enter 运行 gol（**epoch 8s +75 ≈ 9.5 代/秒**）→ A 返回列表 → down+enter 运行 mandelbrot → **渲染期 epoch 每行 +1、画完恒定**（一次性渲染的正确行为）→ 截图确认曼德博完整渲染（2387ms）。
- **排查插曲**：首版扫描路径拼错（"%sapp"）导致列表空，截图诊断后修正为 PLATFORM_ROOT_DIR "/app"。
- **提醒（沿用上条）**：浏览器 IndexedDB（store.js）会持久缓存曾拉取的 .scm——**更新 App 源码后需清一次站点数据**（e2e 脚本已内置清缓存+reload）。

------------------------------------

# 2026-10-18（粒子生命 Scheme App）

C 版"粒子生命"（ui_particlelife.c）移植为 Scheme 灵机应用程序（linglong/app/particlelife.scm，8.1KB）——**自动出现在"灵机应用程序"列表**（无需任何固件改动，验证了插件式框架）。

## 实现要点（相对 C 版的取舍）

- **物理**：同 C 版参数模型——6 种颜色、近距排斥 param_a（<RTH1）/中远距 param_b（分段线性）、JITTER 震颤、DAMPING 0.5 衰减、环面回绕；参数矩阵 36 元素列表（idx=kind_i*NK+kind_j，同种对角更强排斥）。
- **规模**：**81 粒子**（9×9 撒布；C 版 324）+ 全对 O(N²) 交互（C 版均匀网格）——**每步约 30 万 VM 指令 ≈ 1.2 帧**（250k 时间片/帧），实测 **10 步/秒**（interval 100ms 主导）。
- **状态**：5 个平行一维列表（xs/ys/ks/vxs/vys）；param-a/b 为 36 元素列表；'() 静态槽各仅 push 一次（vys 由 (cdr (cons 0 vs2)) 复制，cons 为全表复制故独立）。
- **栈纪律**：语句位置分支一律 { (set! ...) } 净零块（per-pair 的 j==i/越界/有效三分支均如此）；算术二元嵌套。
- **推屏**：每步末尾一次 (Gfx.refresh)（HUD+粒子整帧），代间无半成品闪烁；HUD（"粒子生命/A返回"）与 C 版一致。

## 验证

- **Animac 官方解释器**：am_parse/compile/load 通过（Module dumped/loaded）。
- **gcc 测试台**（断言按 App 模式参数化，/tmp/mb_test_cfg.txt）：粒子生命全 PASS——400 轮进程存活、opstack 0、无错误、持续演化（diff 614）、HUD 白像素、帧缓冲目检（81 粒子 6 色正确）；曼德博/生命游戏回归全 PASS（断言已按模式参数化：暗区/集内点/字幕仅 mb，青色均匀性仅 gol）。
- **浏览器 e2e（playwright）**：列表 down×2 选 particlelife → state=80 → **epoch 8s +76 ≈ 9.5 步/秒** → 截图确认 HUD+粒子正确。
- **帧率说明**：e2e 截图状态栏 10fps 为 headless 环境值；每步 0.7 帧（CELL=8 的 gol）与 1.2 帧（pl）在该帧率下均无堆积；真机浏览器 60fps 更充裕。

------------------------------------

# 2026-10-18（天空渲染 sky.scm：大气二次散射 App）

新增灵机应用程序 sky.scm——参照 celestial.c scatter_model_3 的二次散射算法渲染天空。**当前状态：编译/加载成功但运行时 ARN 存在 while 内 call lambda 的作用域问题，暂标记为实验性。**

## 实现要点

- **物理骨架完整**：瑞利+米氏相函数、地球遮蔽、一阶 Nishita 路径积分 + 二阶多次散射数值积分、臭氧吸收、HDR 色调映射。
- **LUT 优化**：11 仰角 × 24 相对方位 = 264 项，太阳移动才重算（每步 1 次 LUT fill）；渲染时逐 2×2 块双线性插值。
- **投影**：等距方位（天顶居中、地平线圈半径 118px），圆外置黑。
- **太阳**：沿"东→天顶→西"大圆 2° 步进往返。
- **视口**：SW/SH/HH/SKY_CX/CY 全部运行时计算（适配任意屏幕尺寸）。

## 发现的问题

- **ARN 在 while 内 call lambda 的作用域 bug**（vendor 版 7-25）：lut-fill 的 while 循环内调用 sky-scatter 时报"call: xxx 未定义"——但上游版（10-10）同样失败。gol/particlelife 的 while 内无 lambda call 故未暴露。
- **内存需求**：sky.scm 的 AST/编译需 >1MB VM 区（2MB pool 的 VM 区 ~1MB 不够），需 **8MB pool**。已在 ui_scheme_app.c 中升级。
- **测试台的 sky.scm 文件被之前的调试污染**（含 display/newline 残留和错误的 mod_name）——需从干净版本重新生成。

## 后续步骤

1. 修复 ARN 的 while 内 call 问题（或等上游版 vendor 同步）
2. 清理 sky.scm 的调试残留并重新验证
3. 确认渲染结果的物理正确性（与 celestial.c 对照）

------------------------------------

# 2026-10-28（海龟绘图 + L-system：解释器里跑解释器）

新增 `infer/app/turtle.scm`（14.6KB）：**用 Scheme 实现一个微型 L-system / Logo 解释器**，逐字符执行生成出来的符号串来驱动海龟画图。落在 `app/*.scm` 即自动进入"灵机应用程序"列表，无需固件改动。

## 结构（三块）

1. **生成文法**：128 项字符码索引的规则表（`rule_tab`，项为 `0` 表示"无规则原样保留"）；`rewrite_once` 把基因组逐代重写进另一块缓冲。基因组是**整数列表**（不是字符串），重写用 `push` 扩容 + `set_item!` 覆写；两块模块级缓冲轮换（`gA`/`gB`），因此没有字符串拼接的 O(n²)，也没有 `'()` 字面量反复 push 的累积问题。
2. **内层解释器**：`step1` 执行一条指令（`F G` 落笔前进、`f` 空移、`+ -` 转向、`[ ]` 压/弹栈、`|` 掉头）。执行位置 `cursor` 与海龟状态（位置/方向/深度/栈）全在模块变量里，所以**指令流可以跨帧中断续跑**——宿主按时间片协作调度，等于免费得到一台可暂停的虚拟机。
3. **取景与上屏**：先"空跑一遍"只记包围盒（不调用 `draw_line`），据此算缩放与偏移铺满绘图区；颜色随嵌套深度变化（主干棕 → 分枝深绿 → 末梢亮绿）。顶部两行 HUD、底部两行说明各自独占一条带，每 tick 先 `fill_rect` 清黑再写字（绘图区已排除这两块，不会擦掉图形）。

预设 6 个，各自动播放（生长 1–2 s、停留 3 s 后切换到下一个）：分形树、科赫雪花、龙曲线（11 代 8190 符号）、希尔伯特曲线（5 代 3411）、谢尔宾斯基三角、海藻（11116 符号）。

## 实测

- 宿主（x86-64，当前 vendor 0291412）：整套 App 跑 4000 帧（64 s 虚拟时间，6 个预设各一轮）墙钟 **0.76 s**（≈1.2% CPU）；单个预设含"重写+完整生长"共 **60–120 ms**；单 tick（220 符号）≈ **2.4 ms**，相对 40 ms 定时器周期余量充足。
- 浏览器 wasm32：`launch=80 → end=80`、无 `pageerror`，确定性回归 54 次刷新、存活回归 40 次刷新，画面正确（HUD/规则/图形均正常）。
- 画质：整幅图只需几百到几千次 `draw_line`，与 sky3 那类逐像素渲染完全不是一个量级。

## 过程中撞到的两个解释器/编译期坑

1. **多元算术被错误编译（静默算错，且泄漏操作数）**：
   ```
   (- 100 20 5)      → 15      （应为 75：只发射了一条 sub，把 20、5 弹出相减，100 留在操作数栈上）
   (+ 1 (- 100 20 5))→ 115     （应为 76：上一条泄漏的 100 被外层 + 吃掉了）
   (+ 1 2 3)         → 5       （应为 6）
   (* (+ 1 2 3) 10)  → 50      （应为 60）
   ```
   即 `(+ - * /)` 的多操作数形式只发射一条指令，剩余操作数滞留在栈上并污染后续表达式。**写 App 必须一律二元嵌套**（gol.scm 早已记下这一条，本次踩坑复现）。这是值得报上游的编译期缺陷：它不报错、只给错答案。
2. **`cond` 能力受限**：子句只能带一个表达式（写了多个时后面的被丢弃），且 8 子句的形式直接 `compile_fail`（无任何诊断输出）。本 App 的热路径因此改为平铺 `if`。
   附带确认：`(list)` 不是内建函数（`变量 list 未定义`），空列表只能用 `'()` 或 `(cons x '())`；`'()` 字面量是模块级静态槽，反复 push 会跨调用累积（sky3/gol 已记录）。

## 回归

- 宿主 9 个 App 全 PASS（新增 turtle：`frames=360/360 refresh=38 err=none`）。
- 浏览器 Chromium（真 wasm）：确定性回归与存活回归 **9/9** 均 `launch=80 → end=80`、无 `pageerror`。
- e2e 脚本（`infer/tools/e2e/`，未纳入版本管理）的应用清单已加入 `turtle`；宿主回归对 turtle 用 3 倍帧数（定时器驱动生长）。

------------------------------------

# 2026-10-28（解释器 X 光：虚拟机自观测仪表盘 + Diag 本地库）

新增 `infer/app/xray.scm`（10.5KB）与本地库 `infer/am_native_Diag.c/.h`：**把正在运行自己的那台虚拟机画成仪表盘**。既是演示，也是本仓库排查解释器问题时的常驻现场仪器。

## Diag 本地库（只读）

`infer/am_native_Diag.c/.h`，16 个只读访问器，形态与 `am_native_Time.c` 一致（取值 → 压一个 `am_uint_t` → `am_process_step`）：

| 类别 | 函数 |
|---|---|
| 时钟 | `ms`（`clock_gettime(CLOCK_MONOTONIC)`，自首次调用起算；**不受宿主虚拟时钟影响**，故回归台里也能量真实吞吐） |
| 运行时 | `tick`（事件循环轮数）、`gcs`（全局 GC 周期数）、`timeslice`、`pids` |
| 进程现场 | `pc`、`ilcode`（指令区长度）、`opstack`（操作数栈深度）、`fstack`（帧数）、`pgc` |
| 闭包帧 | `frames`（描述符闭包数）、`legacy`（退化为复制全部绑定）、`slots`、`freeslots`、`fillgeneric`、`fastcl` |

注册点四处：`ui_scheme_app.c`（GUI/wasm/Android 共用）、`tools/e2e/host/harness.c`（回归台）、
`build_wasm_gui.sh` 源清单、`android/app/src/main/cpp/CMakeLists.txt`。VENDOR.md 的"本项目自行实现的本地库"清单已补入。

## App 面板

- 顶部：运行时长（Diag.ms）与宿主事件循环 tick 数
- 两条内存条：VM 工作区 / 进程堆的 used÷capacity（`System.memstat`），带百分比与 KB
- 曲线区：88 点滚动折线，**VM 使用率（青）+ 堆使用率（绿）**，底部红色短竖线标 GC 事件
- 读数区（6 项）：操作数栈深、调用栈帧数、`PC / 指令区长度`、`闭包 描述符数 / 退化数`、`槽位总数 / 自由槽数`、`GC 次数 / 上次回收量`
- 吞吐：用 `Diag.ms` 在 **3 ms 预算**内数 lambda 调用次数 → 实时"调用/毫秒"
- 分配演示：每 tick 往 `scratch` push 12000 项，堆到 45% 即丢弃旧表并 `System.gc` —— 曲线因此出现**锯齿**，可以直接看见 GC 在工作

## 实测

| 项 | 结果 |
|---|---|
| host x86-64 | 2000 帧（32 s 虚拟时间）墙钟 **1.50 s**（4.7% CPU）；关掉吞吐探针 1.18 s（探针 ≈0.5 ms/tick） |
| 吞吐读数 | host **≈3600 调用/ms**；浏览器 wasm32 **≈618 调用/ms**（差 5.8×，为真机解释器速度提供了直观标尺） |
| 浏览器 | Chromium 确定性回归 + 存活回归均 `launch=80 → end=80`、无 pageerror；面板数据正常（闭包 339453、槽位 594604/自由 189219、GC 19 次） |
| 生成量 | 每 tick 分配 12000 项 → 堆到 45% 触发 GC，回收 ~4 MB，锯齿周期约 2 s |

## 过程记录：又一次 layout 敏感的 `call: 错误的调用目标`

初版 `draw_readouts` 把 6 条 `(Gfx.draw_text (cat4 …多嵌套…) …)` 直接写成连续语句时，运行期报
`[Runtime] call: 错误的调用目标`（PC 稳定复现）。改成"先 `(set! ds (cat4 …))` 再 `(Gfx.draw_text ds …)`"即恢复正常。
但随后把 6 条**原样内联**放回去（其它代码保持现状）却跑得好好的；加/去一个无关全局也不影响。
即与 `REPORT-interp-slot-store.md` 里那类**布局敏感**的槽位/调用目标错乱同源，无法用最小合成用例稳定复现。
当前保留"拼接先落变量"的写法（也更好读），并把这条记在这里待上游定位。

## 回归

- 宿主 10 个 App 全 PASS（新增 xray：`frames=120/120 refresh=29 err=none`）
- 浏览器 Chromium（真 wasm）：确定性 + 存活回归 **10/10** 全部 `launch=80 → end=80`
- e2e 三处应用清单已加入 `xray`（`host/run_host_regress.sh`、`browser/regress.js`、`browser/regress_det.js`）

------------------------------------

# 2026-10-28（3D 软件光栅化：环面 / 二十面体 / 莫比乌斯带）

新增 `infer/app/raster3d.scm`（21KB）：本 App 不使用任何图形 API 里的 3D 能力，整条管线（旋转矩阵 → 透视投影 → 深度排序 → 扫描线填充 → 平面着色）都在 Scheme 里实现，只调用 `Gfx.draw_line`。

## 管线

| 阶段 | 做法 |
|---|---|
| 顶点 | 单位空间坐标，程序化生成（环面 13×7=91 顶点 / 二十面体 12 / 莫比乌斯 19×4=76），一次性建表 |
| 旋转 | 每帧由 (ax, ay) 组装 `R = Rx(ax)·Ry(ay)`（9 个数）；R 正交，故**面法线也用同一个 R 旋转**，不必重新归一化 |
| 投影 | 透视 `p = D/(D−z)`（相机在 +z，D=3），屏幕 = 中心 + 坐标·p·缩放（每个形体各自设置缩放） |
| 面序 | **计数排序**（画家算法）：3 趟 O(N) 扫描把三角形按深度装进 16 个桶（桶计数→前缀和→落位），结果写进扁平索引数组；由远及近绘制 |
| 光栅化 | 逐三角形扫描线：取三条边与该行的交点求 [xl,xr]，再用 **一次 `Gfx.draw_line` 填满整行**（实测 0.25 µs/次，比逐像素 `set_pixel` 便宜一个数量级）；每轮画 2 行（2px 扫描行）把解释器开销减半；起点对齐到偶数行，避免相邻三角形奇偶错位留 1px 空隙 |
| 着色 | 平面着色 + 兰伯特光照（环境项 0.18）；背面剔除留 0.05 余量，让正好侧对视线的面也画上，轮廓不留缺口；莫比乌斯带为双面材质（不剔除、取 \\|n·L\\|） |

## 实测

| 项 | 结果 |
|---|---|
| 单帧 | 环面 3 ms（144 三角形中 72 个可见）/ 二十面体 1 ms / 莫比乌斯 3 ms（108 三角形全画） |
| 整体 | 2000 帧（32 s 虚拟时间）墙钟 1.31 s ≈ 4% CPU |
| 浏览器 wasm32 | 确定性回归与存活回归均 `launch=80 → end=80`、无 pageerror |
| 回归 | 宿主 11 个 App 全 PASS；浏览器 11/11 全 PASS |

三个形体每 5 s 自动切换（环面 → 二十面体 → 莫比乌斯带 → 循环），HUD 显示当前形体、可见三角形数、帧号与单帧耗时。

## 本次踩到的坑（都记下来备查）

1. **`(pop x)` 返回的是被弹出的元素，不是列表**：想用 `(define l (cons 0 (quote ())))` 再 `(pop l)` 得到空表是错的（`l` 会变成整数 0）。正确做法是建表后单独 `(pop l)` 一条语句，或写 `init_buffers` 统一 pop。
2. **`(cons 0 '())` 建出的是长度 1 的列表**，那个 0 是哑元素：直接往里 push 顶点会让 `get_item (+ (* i 3) 0)` 整体偏移一位 —— 环面看起来"差不多对"、二十面体直接变成星形（顶点错位后连出了错误的三角形）。这是本次最耗时的一个坑。
3. **`Math.PI` 是函数**（`(Math.PI)`），不是常量；**Math 库没有 min/max**（只有 floor/ceil/round/abs/pow/sqrt/三角/对数/位运算），要自己比较。
4. **`{}` 块不平衡只会报 "unexpected end of input in slist seq"，不报位置**：本次漏了一个 `}`（括号本身是平衡的），定位花了些时间。写大段代码后建议先跑一遍花括号/圆括号双计数。
5. 又一次 layout 敏感的 `指令执行异常`（PC 随改动漂移，无错误信息）：最初用"嵌套桶列表 + `buf_clear(list)`"实现深度排序时，第 15 帧左右必崩（崩在 `(length ...)`）。改成**计数排序 + 扁平索引数组**（不再有嵌套 list，也不再把 list 穿过函数边界）后彻底消失。与 `REPORT-interp-slot-store.md` 记录的那类槽位/调用目标错乱同源，最小合成用例仍复现不出来。

------------------------------------

# 2026-10-28（反应-扩散：Gray-Scott 人工生命）

新增 `infer/app/grayscott.scm`（10KB）：Gray-Scott 双组分反应-扩散，5 组实测存活的 (f,k) 参数轮播，每组 600 tick 后换参数重新播种。

## 模型与实现

```
∂A/∂t = Da·∇²A − A·B² + f·(1−A)
∂B/∂t = Db·∇²B + A·B² − (k+f)·B
```

- 网格 **64×48**（每格 5×5 像素铺满 320×240），环形边界（邻格下标用 `mod` 取）；
- ∇² 用 **9 点模板**（Karl Sims 那套）：`0.2×(正十字) + 0.05×(四角) − 1.0×自身`，Da=1.0 / Db=0.5 / dt=1.0；
- 显式欧拉一步，结果裁到 [0,1]；A/B 各双缓冲（四张扁平表轮换，只 `set_item!`，无逐帧分配）；
- 上屏 3072 次 `fill_rect`，颜色由 B 线性映射到深蓝→青白。

## 参数筛选（本次的主要"知识产出"）

在 40×30、600 步、确定性播种下批量试了 16 组 (f,k)，**只有 6 组能活下来**（B 不熄灭）：

| (f,k) | 结果 | (f,k) | 结果 |
|---|---|---|---|
| (0.0545, 0.062) 珊瑚 | ✅ 存活 mean 0.21 | (0.0367, 0.0649) 常被推荐为"有丝分裂" | ❌ B→1e−25 熄灭 |
| (0.058, 0.065) 蠕虫 | ✅ mean 0.07 | (0.058, 0.065) 之外的 (0.029,0.057) "迷宫" | ❌ 熄灭 |
| (0.039, 0.058) 空洞 | ✅ mean 0.23 | (0.025,0.060) "脉冲孤子" | ❌ 熄灭 |
| (0.062, 0.0609) 滑翔子 | ✅ mean 0.25 | (0.030,0.062) "斑点" | ❌ 熄灭 |
| (0.026, 0.051) 波纹 | ✅ mean 0.20 | (0.042,0.061) | ✅ mean 0.19 |
| | | (0.046,0.0654)、(0.018,0.051)、(0.014,0.054) | ❌ 熄灭 |

**为什么会熄灭**：B 的衰减项是 `(k+f)·B`，自催化项是 `A·B²`；当 B 低于约 `k+f`（≈0.1）时净增长为负。小种点在扩散稀释中很快掉到阈值以下 → 整片死掉。所以**播种必须用大块**（本 App 用 4 个 8×8 种点 + 8% 稀薄噪声），小点撒种在低 f 参数下几乎必死。这条对"为什么网上很多参数表照抄下来不work"是个直接解释：参数表通常假设更大的域和不同的初值。

## 成本

| 项 | 结果 |
|---|---|
| 单步（3072 格 × 18 次邻格读取 + ~20 次浮点运算） | host ≈ **11 ms** |
| 整帧（+3072 次 fill_rect） | host ≈ **13 ms**（定时器 40 ms → 约 33% CPU） |
| 浏览器 wasm32 | 帧内实测约 **32 ms**（解释器慢约 6 倍），实际 8~12 fps，图案仍持续演化 |
| 优化记录 | 先把 9 点模板改成 5 点（`0.25×四邻 − 自身`，读 10 次代替 18 次）确实更快，但**扩散太弱导致整片烧成 B=1**；改回 9 点才对。也就是说这类模型不能只看"少读几次"，模板权重与 dt 必须成对调。 |

## 回归

- 宿主 12 个 App 全 PASS（新增 grayscott：`frames=120/120 refresh=38 err=none`）
- 浏览器 Chromium（真 wasm）：确定性 + 存活回归 **12/12** 全部 `launch=80 → end=80`、无 pageerror
- e2e 三处应用清单已加入 `grayscott`（字母序在 gol 与 mandelbrot 之间）

## 后续可做

- **Lenia**（连续态元胞自动机，环形卷积核）：视觉上是"滑动的细胞团"，比 Gray-Scott 更像生物；代价是卷积核 tap 数多（13×13 环），在当前解释器速度下需要更小的网格（≤32×24）。
- 交互调参（f/k 滑杆）：需要先补 Scheme App 的按键通道（见 xray 条目）。

------------------------------------

# 2026-10-28（Lenia：连续态元胞自动机 / 人工生命）

新增 `infer/app/lenia.scm`（11KB）：Bert Chan 的 Lenia——状态连续取值、邻域是环形核卷积的元胞自动机。与 gol（离散格子）是同一类模型的连续化极端。

## 模型

```
u(x) = Σ K(‖d‖)·A(x+d)                     环形核：μk=0.5R、σk=0.15R，归一化后 ΣK=1
G(u) = 2·exp(−(u−μg)²/(2σg²)) − 1           μg 附近给 +（长），其余给 −（死）
A'(x) = clip(A(x) + dt·G(u), 0, 1)
```

## 实现要点

- 网格 **20×15**，每格 **16×16 像素**（铺满 320×240）；核半径 **R=3**（7×7 = 49 taps）。
- 状态用 **padding 缓冲**（26×21）：每步把四周环带从对侧内部拷回（环形边界），于是内层卷积循环里**没有任何 mod / 下标表**，每个 tap 只有两次列表读（权重表 + 邻格值）。
- 增长函数 G(u) 预计算成 **256 项查表**（u∈[0,1]），热路径里没有 `exp`。
- 上屏不铺方块，而是每格画一个**半径随 A 变化的实心圆**——相邻圆重叠后融成有机团块。这一步对观感影响极大：同样数据铺 16px 方块时像"随机色块"，改成圆后立刻像显微镜下的细胞群落。
- 双缓冲轮换，无逐帧分配；4 组预设 (μg,σg,dt) 每 500 tick 轮播。

## 参数筛选（sigma 必须放大）

R=3 的核很窄，若沿用 R=13 的经典值 σg=0.015，**整片会直接熄灭**。在 20×15/32×24 上扫了 19 组后存活区间：

| 组 | 结果 |
|---|---|
| (0.15, 0.025, 0.1) 温泉 / (0.20, 0.035, 0.1) 浓汤 / (0.12, 0.022, 0.1) 黏菌 / (0.15, 0.025, 0.05) 慢潮 | ✅ 存活且结构稳定 |
| σg ≤ 0.015 的任何组合（含经典 0.015、0.014） | ❌ 全灭 |
| μg ≥ 0.25 且 σg ≤ 0.03 | ❌ 全灭 |

即：**σg 需要随核半径缩小而放大**（0.015 → 0.02~0.035），否则宜居带太窄、A 全部落到"死亡区"。

## 成本

| 项 | 结果 |
|---|---|
| 单步（300 格 × 49 taps） | host ≈ 4.5 ms |
| 整帧（2 步 + 300 个 fill_circle） | host ≈ **12 ms**（50 ms 定时器 → 约 24% CPU） |
| 浏览器 wasm32 | 帧内实测 **32 ms**，实际 10~15 fps |

一次列表访问约 **0.09 µs**，是全部成本的来源（每格 98 次访问 × 300 格 ≈ 26k 次 ≈ 2.4 ms，其余是循环与乘加）。要再快只能减 taps（R=2）或减格子。

## 新工具：多操作数算术静态检查

这次写探针时踩到 `(* 2.0 sgG sgG)`——三操作数乘法被错误编译（只发射一条 mul，其余操作数滞留在操作数栈上污染后续表达式），表现为增长函数整表变成 `inf`。于是做了 `infer/tools/e2e/host/check_multi_operand.py`：

- 对 `.scm` 做词法/括号解析，报出所有 `(+ - * /)` 操作数 >2 的位置（行号 + 操作数个数）；
- 现在对 `infer/app/*.scm` 全量检查全绿（13 个 App）。**建议把这条加进 App 提交流程**：解释器不会报错，只会静默算错。

## 回归

- 宿主 13 个 App 全 PASS（新增 lenia：`frames=120/120 refresh=29 err=none`）
- 浏览器 Chromium（真 wasm）：确定性 + 存活回归 **13/13** 全部 `launch=80 → end=80`、无 pageerror
- e2e 三处应用清单已加入 `lenia`（字母序 grayscott → lenia → mandelbrot）

------------------------------------

# 2026-10-28（渐进式 SDF 光线步进：Shader 风格）

新增 `infer/app/sdfprog.scm`（15KB）：每个像素从相机发一条射线，沿射线做球体步进（sphere tracing）求 SDF 交点，命中后用 4 次差分估法线、兰伯特光照 + 棋盘地面 + 距离雾。全在 Scheme 里算，只用 `Gfx.fill_rect` 上屏。

## 为什么用"渐进式"

全分辨率 320×240 逐像素步进在本解释器上要几十秒（一次列表/算术访问 0.01~0.09 µs，每像素平均 20 步 × 每步 ~60 次运算）。所以：

- **降采样**：渲染分辨率 80×60，每像素画 4×4 方块铺满屏幕（顺带得到明显的老显示器质感）；
- **跨帧累积**：每 tick 只算 **8 行**，直接画进帧缓冲、**不清屏**；47 行用 6 个 tick 画完（host 约 0.24 s，wasm 约 1.5 s），画完停留 12 拍 → 换场景/换机位/清屏重来。
- HUD 带（顶部 32px、底部 20px）在渲染行范围之外，每 tick 单独刷黑重画，因此可以和累积渲染共存。

这正是"没有 GPU 时 shadertoy 该怎么做"：把重活摊到多帧的解释器协作调度，观感上像逐行显影的示波器。浏览器确定性回归抓到的画面就停在 **17% / 行 8/47**，是渐进渲染的直接证据。

## 三个场景（自动循环，机位每幅偏转 0.55 rad）

| 场景 | SDF | 说明 |
|---|---|---|
| 融合球 | 两球 + smooth-min(k=0.5) ∪ 地面 | 球心随 `stime` 摆动，融合处有平滑过渡 |
| 置换球 | `r − 1 − 0.22·sin(4.5x)sin(4.5y)sin(4.5z)` ∪ 地面 | 起伏球面，像熔岩球 |
| 无限柱阵 | `mod(p,2)−1` 上的圆柱、`max(y−0.55)` 截顶 ∪ 地面 | 无限重复的柱阵 |

## 性能与取舍

| 项 | 结果 |
|---|---|
| 单行（80 像素） | host **3~5 ms**（三场景不同：融合球 5 ms、置换球 3 ms、柱阵 3 ms） |
| 一整幅（47 行） | host ≈ 0.2 s CPU，分散在 6 个 tick |
| 浏览器 wasm32 | 同一幅约 1.5 s 显影完成 |
| 每像素 | 平均 12~20 步（MAXSTEPS 20、EPS 0.004、TMAX 12）+ 4 次差分求法线 |

源码里 **SDF 数学写了两份**：`march_row` 里内联一份（省掉每步的函数调用开销，这是性能关键），`sdf` lambda 一份（给差分求法线用，每像素 4 次）。两处必须同步修改，已在注释里标注。

## 本次踩到的两个坑

1. **花括号 `{}` 与圆括号必须同时数**：`(define cat5 …` 少/多一个 `)` 时，括号净值为 −1 但花括号为 0，解析器只报 "trailing tokens after parse" 不给位置。建议写大段代码后同时跑两个计数器（本次已沉淀为流程）。
2. **重复空间（mod）里的 SDF 不连续** → 差分法线在单元边界上偶尔退化，画面出现纯黑麻点。处理：三个通道兜一个下限（≥10），既消除麻点也保留结构。这在真实 shader 里同样是常见处理（法线 normalize 前加 epsilon）。

## 回归

- 宿主 14 个 App 全 PASS（新增 sdfprog：`frames=120/120 refresh=28 err=none`）
- 浏览器 Chromium：确定性 + 存活回归 **14/14** 全部 `launch=80 → end=80`、无 pageerror
- e2e 三处应用清单已加入 `sdfprog`（字母序 raster3d → sdfprog → sky）

------------------------------------

# 2026-10-28（三维迷宫：第一人称漫游，Windows 95 屏保风）

新增 `infer/app/maze3d.scm`（17KB）：第一人称在随机迷宫里自动穿行。渲染用经典 **raycasting**（Wolfenstein 3D 那套），没有任何 3D 硬件参与。

## 渲染

- 视线平面 → 每列（2 像素宽，共 **160 根**）一条射线 → **DDA** 步进网格找第一个墙格 →
  墙高 = `VIEW_H / 垂直距离` → 一根竖条。东西面/南北面亮度不同（假光照），
  距离雾 `1/(1+0.30d²)`，墙格基色由 `(5x+3y) mod 4` 决定（拐角一眼可见）。
- 墙面纹理：命中点在格内的横向位置 → **竖条纹**（6 条一循环），竖条再拆成上下两段，
  下段 0.72 倍亮度当"踢脚线"——否则贴脸看墙是一整片纯色，很闷。
- 顶棚/地板：各 6 条整数分段的渐变横条（整数分段保证无缝，否则会留下 1px 亮线）。

## 自动漫游

- 迷宫：**递归回溯**算法（12×9 格，地图 25×19，显式栈，非递归），保证连通、无环；
  每 1600 拍（约 64 s）换一座新迷宫。
- 走法：按**格心**推进（格 (i,j) 的格心 = `(2i+1.5, 2j+1.5)`），从上一格心直线"归航"
  到下一格心，因此永远不会因为转向滞后而飘进墙角；转角用 0.30 rad/拍平滑过渡。
- 转向策略：**优先直行（0.72 概率）**。均匀随机选向会让人一直在原地打转、频繁贴脸看墙；
  加了直行偏好后走廊长镜头明显变多（对照截图可以看出来）。

## 性能

| 项 | 结果 |
|---|---|
| 单帧（160 根竖条 × 平均 ~8 次 DDA 步进 + 纹理） | host **1~2 ms** |
| 定时器 40 ms/帧 | CPU 占用 < 5% |
| 浏览器 wasm32 | 帧内约 8~12 ms，20~30 fps；42 ms 的整帧也测到过（贴脸时墙条更少反而更快，起点附近例外） |

## 本次踩到的坑

1. **27 位整数载荷**：最初自己写 LCG（`lcg = lcg*1103515245 + 12345 mod 2^31`），
   在 wasm32 / armv7 这类 32 位宿主上 `am_int_t` 只有 27 位载荷 → **溢出**，
   迷宫生成直接崩坏（host 上是 59 位所以看不出来）。改用 `Math.random`（vendored
   实现是 `srand(time)+rand()`，各宿主都有）。
2. **格心公式**：格 (i,j) 在连续坐标里的中心是 `(2i+1.5, 2j+1.5)`，写 `2i+1` 会把目标
   定到格的角上——玩家会径直走进墙里，画面变成一整片墙色。
3. **文件里混进了占位符**：源文件里出现过一行 `(if⟨placeholder⟩)` 之类的残片，
   解析器只报 "trailing tokens after parse"（无行号）。教训：写完立刻跑括号/花括号
   双计数器 + `check_multi_operand.py`。

## 工具箱（本轮新增）

- `infer/tools/e2e/host/check_multi_operand.py`：静态检查多元算术（三操作数会被错误编译）。
- 括号/花括号双计数脚本：已固化为流程（写大段 Scheme 后必跑）。

## 回归

- 宿主 15 个 App 全 PASS（新增 maze3d：`frames=120/120 refresh=39 err=none`）
- 浏览器 Chromium：确定性 + 存活回归 **15/15** 全部 `launch=80 → end=80`、无 pageerror
- **e2e 的菜单序号问题**：`list_files` 原先直接用 `readdir` 顺序（ext4 哈希受影响），
  已改为按名字 `qsort`（`hal_fs_linux.c`），桌面/wasm 的 App 列表顺序从此稳定；
  另外用"临时把 maze3d 复制成 `aaa_probe.scm` 让它排在列表首位"的办法，
  单独验证了 wasm 下迷宫渲染正常（截图为 三维迷宫 HUD + 墙面条纹）。

### 追补：A\* 规划路线 / 平缓转向 / 左上角迷宫全图

按使用反馈做了三处改进：

1. **A\* 预先规划路线**（替换原来的"到格心随机选向"）：
   - 节点 = 格子，邻居 = 相邻且中间的墙已打通；代价 1（4 邻域），启发式 = 曼哈顿距离；
   - 规模只有 12×9 = 108 个节点，开表用普通列表 + 线性取最小 f，不上二叉堆；
   - 走到当前目标格后，随机挑一个**曼哈顿距离 ≥10** 的远端格重新规划，形成"一直在赶路"
     的观感；HUD 实时显示 `A* 已走/总航点数`。
   - 副产品：路径里不会出现回头路，所以镜头基本不再"进死胡同再退出来"。
2. **转向平缓 + 掉头慢慢来**：角速度从 0.30 降到 **0.10 rad/拍**（90° 约 16 拍 ≈0.63 s），
   并按角度差压低前进速度——`|Δ|>1.2 rad` 降到 0.45×、`|Δ|>2.4 rad`（需要掉头）降到 0.22×，
   于是掉头看着像"一边慢慢掰车头一边蹭出去"，而不是原地甩头。
   实测每步角度变化上界正好 0.1000 rad（264 个采样点里最大 0.100，非零 48 次）。
3. **左上角迷宫全图**：3 像素/格画出 25×19 的地图（墙浅色、通路底色），并叠加
   **A\* 规划路线**（绿点）、**目标格**（红点）、**摄像机位置**（黄点 + 朝向短线）。
   每帧重画（475 个 `fill_rect` ≈ 0.3 ms）；单帧总耗时仍为 2~3 ms。

排查记录：A\* 初版在 `get_item` 上抛"指令执行异常"，两个原因——
① `(set! inopen cur 0)` 写成了三参数 `set!`（应为 `set_item!`）；
② `mod` 返回浮点，直接当列表下标用会拿到 `#undefined`，索引一律套 `Math.floor` 后解决。

### 追补：改成浅色明亮风格

配色从"暗色走廊 + 距离变黑"改为"白天走廊 + 距离褪色"：

| 元素 | 改法 |
|---|---|
| 顶棚 | 近白 `(253,253,255)` → 地平线略暗；**10 段**整数分段渐变（6 段时能看出台阶） |
| 地板 | 浅暖灰 `(219,213,201)` → 底部更深一点，同样 10 段 |
| 墙面 | 四色粉彩：奶油 `(250,238,205)` / 淡蓝 `(190,218,248)` / 淡薄荷 `(186,236,206)` / 淡桃 `(252,206,190)` |
| 雾 | 不再乘 0→0，而是**与空气色 `(198,208,224)` 插值**（`shade=1` 为原色，越小越接近空气），强度从 `0.30d²` 降到 `0.13d²` —— 远处依然明亮，但与近白顶棚有分离度 |
| 墙面纹理 | 竖条纹对比从 0.93 提到 0.90；竖条改为**三段**：顶角线 0.94 / 主面 1.0 / 踢脚线 0.84，贴脸看墙不再是一整片平色 |
| 小地图 / HUD | 浅底深字：底板 `(246,247,250)`、墙 `(150,166,194)`、路线绿、目标红、摄像机橙点 + 深橙朝向线、HUD 底 `(246,247,250)` + 深灰蓝字 |

侧面亮度系数从 0.72 提到 0.90（浅色基调下不再需要强明暗差来区分朝向）。
单帧仍是 **2~3 ms**（多出的三段墙带只多约 160 次 `fill_rect`）。

### 追补：配色可切换（三套主题）

颜色原先散落在绘制函数里（裸数字），现全部提到模块级变量 + `theme_set(t)` 一处配置，
绘制函数只读变量。**每生成一座新迷宫自动换一套配色**（浅色 → 暗色 → 夜蓝 → 循环），
HUD 标题里实时显示当前主题名。

| 主题 | 顶棚 / 地板 | 墙 | 空气色（雾目标） | 小地图 |
|---|---|---|---|---|
| 浅色 | 近白 → 浅暖灰 | 奶油/淡蓝/淡薄荷/淡桃 粉彩 | `(198,208,224)`，`fogk=0.13` | 浅底深墙 |
| 暗色（原版） | 深蓝灰 → 深棕 | 190/165/150/185 暗调 | `(10,12,18)`，`fogk=0.30` | 深底浅墙 |
| 夜蓝 | 深蓝 → 近黑 | 青/钢蓝/蓝绿/淡蓝 | `(12,20,38)`，`fogk=0.16` | 深底青墙 |

参数化覆盖：顶棚/地板各 10 段渐变的起止色、四面墙基色、空气色与雾强度、侧面亮度
（`sidek`）、竖条纹对比（`stripek`）、顶角线/踢脚线系数（`crownk`/`basek`）、小地图
7 种颜色、HUD 底色与两种文字色 —— 共约 45 个变量，全部由 `theme_set` 赋值。

单帧仍是 2~3 ms（颜色只是变量读取，绘制次数不变）。
