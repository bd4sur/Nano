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

### 修复：天花板/地面每隔几段漏一行（黑线/亮线）

现象：天花板与地面每隔约 56 像素出现一条 1 像素的黑线（夹杂亮线）。

定位过程：先排除了墙条（把所有墙条压成单段仍复现）、也排除了 `fill_rect` 本身
（用探针测 h=11/12/13/24 都精确画满）；最后**关掉墙列与小地图、只画背景**时黑线
依然存在，于是问题锁定在 `draw_backdrop` 的分段循环。

根因：**`/` 在本解释器里恒为浮点**。分段边界写成
`(/ (* i half) 10)`（half=112）得到 0, 11.2, 22.4, …，直接交给 `fill_rect` 时
native 对 `y` 和 `h=(y1-y0)` **各自截断**：`y0=44.8→44`、`h=11.2→11`，只覆盖
44..54，于是 **55 这一行没人画**（保持上一帧/黑底）。每 56 行出现一次，正好对上
观察到的那几条线。

修复：边界先 `Math.floor` 再相减：`y0=floor(44.8)=44`、`y1=floor(56)=56`、
`h=y1-y0=12` → 首尾相接，无漏行。修复后纯背景测试中**黑行数为 0**，
只剩渐变台阶（Δ≈5）与地平线过渡（Δ≈21）。

顺手检查了其余 6 个新 App：没有任何 `fill_rect` 参数含除法，因此无同类隐患。

### 修复：通关后不开下一局、停在原地、小地图目标点乱跳

**现象**：走到终点后玩家原地不动，既不换迷宫也不换配色；小地图上的目标点每帧乱跳。

**根因**（读手册后确认，与手册多条约定相关）：A\* 的工作数组 `gs/fs/came/inopen`
是模块级复用的，但 `plan_path` 只初始化了起点，**没有重置其余节点**。第二次及以后的
搜索于是：① 上一轮留在开表里的节点 `inopen` 仍为 1 → 不会再入表；② `gs` 里残留旧值
→ 新路径不再更新。开表很快耗尽 → `found=0` → **返回空路径**。

空路径的后果正好对上两个现象：
- `take_waypoint` 取不到航点 → `tgtx/tgty` 保持不变 → 玩家停在原地（"通关后没开始下一局"）；
- 由于路径一直空，每个 tick 都重新 `plan_leg` → 每 tick 重新随机一个目标格 →
  **小地图目标点每帧乱跳**。

**修复**：
1. 新增 `astar_reset`，在 `plan_path` 开头把 `gs/fs/came/inopen` 全部重置
   （规模只有 108 个节点，代价可忽略）；
2. **通关即开下一局**：航点走完且当前格 == 目标格时调用 `next_round`
   （`rounds+1` → `theme_set(seed mod 3)` 换配色 → `gen_maze` 生成新迷宫 → `reset_player`）；
3. 目标选择改为"先列候选（曼哈顿距离 ≥10）再随机取"，保证每局都有像样的路程
   （原来的循环重试 30 次可能落到近处，出现 29 拍就结束的短局）；
4. 兜底：连续 3 次规划失败、或单局超过 `MAZE_TICKS` 拍，都强制 `next_round`，绝不卡死；
5. HUD 增加局数显示（`三维迷宫 · 夜蓝  第2局  A* 2/35  3ms`）。

**验证**：连跑 12000 帧（约 192 s 虚拟时间），稳定跑完 5 局，每局结束时
`cell == goal`（真正走到终点），配色按 夜蓝 → 浅色 → 暗色 循环，单局耗时 512~1368 拍。

### 对照手册自查（顺手做的合规性审计）

| 手册条目 | 本项目现状 |
|---|---|
| 算术运算**固定二参数**（§6.2.4） | 全部 App 通过 `check_multi_operand.py` 静态检查 |
| `/` 恒为浮点、`mod` 恒返回浮点且符号随被除数（§6.2.4） | 所有列表下标都用 `Math.floor` 显式取整；`fill_rect` 的 y/h 先取整（见上一条黑线修复） |
| `let/case/do/for/return` 保留但**未实现**（§2.6） | 全项目未使用 |
| 无 `list`/`append`/`map`/`for-each`（§6.4）；`cons` 第二参数须为列表、`cdr` 返回拷贝 | 一律用 `'(...)` 字面量 + `push/pop/get_item/set_item!` |
| `define` 无过程简写，须写 `(define f (lambda ...))`（§5.2） | 全项目一致 |
| `cond` 每个子句只接受**单个表达式**（§4.2） | 复杂分支一律用花括号包成 `begin` |
| `while` 循环体**不是独立词法区域**（§4.2/附录 C） | 热路径局部量统一在函数级 `define`；循环体内不依赖"每次迭代独立绑定" |
| `and`/`or` 结果恒为布尔（§4.2） | 只用于条件位置 |
| 单值栈纪律：语句形式的值是 `#undefined`（附录 C） | 语句位置的 `set_item!`/`push`/`while` 不参与取值 |
| `get_item` 越界返回 `#undefined`、`set_item!` 越界报错（§6.4） | 索引均由同一套 `CW/CH` 常量推出，并做 `mod` 收口 |

------------------------------------

# 2026-10-28（Verlet 物理玩具：布 / 绳 / 双摆）

新增 `infer/app/verlet.scm`（18KB）：三个场景轮播（每场景约 28 s），全部是经典 Verlet 玩具。动手前先重读了 `~/Animac/doc/manual.md`，按手册约束编写。

## 三个场景

| 场景 | 模型 | 规模 | 单帧（host） |
|---|---|---|---|
| 双摆 | 拉格朗日方程 + 半隐式欧拉（每拍 8 子步，dt=0.005）；笔尖轨迹用 420 点环状缓冲渐隐 | 2 个自由度 | < 1 ms |
| 绳 | 两条各 28 质点的绳子，重力 + 摆动风 + **锚点横向摆动**驱动 | 56 质点 / 54 约束 | ≈ 2 ms |
| 布 | 20×13 网格，顶边每 5 个点固定；重力 + 风 + 锚点微摆 | 260 质点 / 487 约束 | ≈ 5 ms |

**Verlet 位置积分**：先由 `当前位置 − 上一位置` 得到速度（乘阻尼），再加外力更新位置；
然后对每根约束做若干轮松弛——两端沿连线各推回偏差的一半，固定点权重 0。速度隐含在
位置差里，不需要显式速度，也不会数值爆掉。

## 调参过程中学到的两点

1. **风是加速度，别盖过重力**：第一版给绳 0.55、布 0.85 的风力（而重力只有 0.30），
   结果绳子被吹成横线、布被吹出屏幕外拉成长条。降到 0.16 / 0.26 后正常。
2. **光靠初始扰动，几秒后就静止了**：改成让**锚点缓慢摆动**（`sway_pins`，绳 22 px、
   布 7 px）持续注入能量，画面才一直"活"；同时加了"质点限制在画面内"的收束，
   防止个别质点飞出后拉出长线。

## 手册合规清单（写代码前逐条对照，见手册 §2–§8 与附录 C）

- 算术固定二参数；取负写 `(- 0 x)`（`check_multi_operand.py` 全绿）；
- `/` 恒为浮点、`mod` 恒返回浮点 → 列表下标一律 `Math.floor`；
- 未使用 `let/case/do/for/return`，也未用 `list/append/map/for-each`；
- `define` 一律 `(define f (lambda …))`，无过程简写、无重名（重名会报
  `failed to add defines to scope`）；
- `cond` 未用（多分支一律平铺 `if`）；`while` 体不依赖"每次迭代独立绑定"；
- `and`/`or` 只用于条件位置（其结果恒为布尔）；`Math.PI` 需调用 `(Math.PI)`；
- `(cons 0 '())` 建表后先 `pop` 掉哑元素。

## 回归

- 宿主 16 个 App 全 PASS（新增 verlet：`frames=120/120 refresh=39 err=none`）
- 浏览器 Chromium：确定性 + 存活回归 **16/16** `launch=80 → end=80`、无 pageerror
- 另用"把 verlet 复制成 `aaa_probe.scm` 排到列表首位"的办法单独确认了 wasm 下三个
  场景都能正常渲染（截图为双摆 + 轨迹，帧内 2 ms）

## 已知的测试工具局限

浏览器回归脚本按"字母序假设"用方向键选 App，而实际菜单顺序偶发与假设不一致
（截图串位：标着 verlet 的截图里是生命游戏）。这不影响 App 本身（16/16 存活、无
pageerror 均正常），但**逐 App 的截图/CRC 归属需要打折扣看**；单 App 验证建议用
上面的 `aaa_probe` 办法（让它排到首位）。

------------------------------------

# 2026-10-28（FLIP 流体 · Scheme 版）

新增 `infer/app/flip.scm`（26KB）：照 `infer/flip.c`（Ten Minute Physics 的 FLIP）移植的小规模版本，整个画布就是容器（没有沙漏/瓶颈节流），重力方向缓慢随机游走以引入扰动。

## 管线（与 flip.c 同构）

`积分（重力+碰壁） → P2G（粒子速度按双线性权重散布到 MAC 网格 + 权重和）→ 归一化 →
压力求解（Gauss-Seidel 原地松弛，压到近似无散）→ G2P（网格速度插值回粒子，FLIP/PIC 混合）`

- 网格 32×22（每格 10 px），粒子 **595**（交错排列，约 2 粒/格），压力迭代 20 轮；
- 物理每 2 拍推进一次（`PHYS_EVERY=2`），渲染每拍都做；
- 与 C 版的差异：不做 `push_particles_apart`（代价太高）与 drift 补偿（要再跑一遍密度 P2G），
  改用两个便宜稳定器——粒子速度阻尼 `VEL_DAMP`、按粒子密度梯度推开过密粒子 `SEP_K`；
  不存压力数组（只用于上色，这里按速度上色）。

## 实测（host x86-64）

| 项 | 结果 |
|---|---|
| 平均每拍 | **17 ms**（物理拍约 33 ms，渲染拍约 1 ms） |
| 浏览器 wasm32 | 物理拍约 100 ms、平均约 50 ms → 约 20 fps |
| 稳定性 | 连跑 5000 帧（约 83 s 虚拟时间）流体保持成团、随重力角倾斜流动，无发散 |

## 这轮踩的坑（对以后写任何网格流体都有用）

1. **y 轴方向**：`flip.c` 是 y 向上，本项目屏幕 y 向下。贴壁面速度归零时照抄它的
   `j-1` 会漏掉"贴地"那一面 —— 粒子在墙上堆积出来的大速度被当成流入，压力永远堵不住，
   流体会被顶到天花板上。改成 `j+1` 才对。
2. **双线性权重必须先把采样点夹进 `[1.5, GW-1.5]`**：否则贴壁粒子算出负权重，
   等于每步往网格里注射能量（表现为流体"沸腾"）。
3. **每步重力不能大**：`G_RATE > 0.008`（格/步²）时压力求解跟不上，一步就把流体踢飞；
   0.004~0.006 稳定。压力迭代也不能太少：本项目条件下 20 轮稳定、12 轮开始发散。
4. 粒子数/网格分辨率要配套：2 粒/格是这条管线能稳住的下限附近；把 `RAD` 调小加粒子
   会同时降低稳定性与帧率（`RAD` 是"加粒子"的主旋钮）。
5. 重力角做了 ±50° 限制：不限制时随机游走会让重力绕到"朝上"，流体被拍到天花板
   （虽然物理正确，但不好看）。

## 回归

- 宿主 17 个 App 全 PASS（新增 flip：`frames=120/120 refresh=24 err=none`）
- 浏览器 Chromium：确定性 + 存活回归 **17/17** `launch=80 → end=80`、无 pageerror
- 另用 `aaa_probe.scm`（排到列表首位）单独确认 wasm 下渲染正常（截图为 595 粒、18° 重力角）

### 调优：粒子降到 1/3（换帧率）+ 重力改成"船在浪里"的摇摆

使用反馈：在 PC 的 Chrome 上只有个位数 fps；重力摆动幅度太小、看不出摇晃。

**① 粒子 595 → 204（约 1/3）**
- 网格从 32×22（10 px 格）改成 **26×18（12 px 格）**，粒子半径 `RAD` 0.34 → 0.45；
- 压力迭代 20 → **11 轮**；粒子速度阻尼 0.90 → 0.86；密度梯度分离 0.02 → 0.05；
- 每帧耗时：host **17.5 → 11.6 ms**，浏览器 wasm32 **~100 → ~35 ms**（约 28 fps）；
- 粒子块画成 8×8（格子 12 px），稀疏时也能连成流体。

**② 重力改为正弦摇摆（周期 + 幅度都随机游走）**
```
gang = gr_amp · sin(2π·t/gr_period + gr_phase)
每 32 拍微调：gr_amp ±0.11（限 0.25~0.85 rad ≈ 14°~49°）
              gr_period ±20 拍（限 120~280 拍 ≈ 4.8~11 s）
              gr_phase ±0.25 rad
gx = G·sin(gang)，gy = G·cos(gang)   （gang=0 即竖直向下）
```
于是摇晃既有明显的左右倾（最大近 49°），又不会机械地严格周期 —— 观感像船在浪里。
角度大时流体会被甩上侧壁（画面上的"浪花"），属正常。

**取舍记录**：粒子数/网格/迭代三者互相牵制 —— 粒子少于约 1.2 粒/格时网格压力难以维持
连贯流体（会出现零散颗粒溅到天花板），所以 204 粒是"1/3 粒子 + 仍像流体"的落点；
再想更快只能继续减粒子或减网格。

### 修：重力"几乎不动" —— 摆动是靠加大重力与减小阻尼，不是调角度公式

反馈：改了正弦摇摆后仍然"几乎不动"。查下来问题不在角度公式，而在**量级**：

- 原配置 `G_RATE=0.005`、`VEL_DAMP=0.86` —— 阻尼每步吃掉 14% 速度，终端速度只有
  `g·sinθ/(1−0.86) ≈ 0.018 格/步`，液面左右高度差实测只摆动 **3.7 格**（≈40 px），
  视觉上就是"不动"。
- 把重力加到 `0.010`、阻尼放宽到 `0.94`、迭代升到 14 轮后，同一套摇摆公式的
  液面高度差摆动达到 **6~11 格（≈70~130 px）**，周期 3.6~6.8 s —— 明显"船在浪里"。

新增了一个可量化的判据：**液面左右端平均高度之差随时间的摆幅**（按格计）。
用它扫参比看截图快得多：

| 配置 | 液面摆幅 | 观感 |
|---|---|---|
| g=0.005, damp=0.86, 204 粒 | 3.7 格 | 几乎不动（用户反馈的那版） |
| g=0.010, damp=0.94, 290 粒, it=14 | **6~11 格** | 明显摇晃、水拍侧壁 ✓ |
| g=0.012, damp=0.96, 204 粒, it=11 | 10.9 格 | 数字大但**炸成水花**（太稀） |

**粒子数与晃动幅度的耦合**：204 粒（1.2 粒/格）在"大幅摇摆"下撑不住连贯流体，
会被甩成水花；要到 290 粒（1.7 粒/格）才既晃得明显、又保持成团。所以最终取
**290 粒（RAD 0.40）/ 14 轮迭代 / g=0.010 / damp=0.94 / 摇摆 26°~54°、周期 3.6~6.8 s**，
代价是 host 11.6 → 16 ms/拍、浏览器 35 → 41 ms（约 24 fps）。

摇摆参数也一并对齐"像船"：幅度随机游走区间 0.45~0.95 rad、周期 90~170 拍、每 32 拍微调。

### 再调：真正"下狠手"的摇摆 + 补上粒子分离

反馈仍不够浪。这轮做了两件事：

**① 摇摆彻底放大（幅度是常驻的，不再被随机游走压回小值）**

上一版的幅度是"随机游走 + 钳位"，实测经常贴在 0.45 rad 下界（打印 gr_amp 发现一直是 0.45），
所以永远晃不起来。改成**慢正弦调制**，幅度基准就在 60°：

```
amp    = 1.05 + 0.16·sin(2πt/430)      → 0.89~1.21 rad（51°~69°）
period = 78 + 22·sin(2πt/620)          → 56~100 拍（2.2~4 s）
phase  每 40 拍随机游走一次（±0.18）
gang = amp·sin(2πt/period + phase)，gx=G·sin(gang)，gy=G·cos(gang)
```
重力矢量在 ±60° 附近来回扫（扫过约 120° 的弧）——浏览器截图里能看到 −67°，
水整堆到右壁、顺着壁爬上去了。

**② 补上 `push_particles_apart`（这是"敢大幅摇晃"的前提）**

没有粒子分离时，248~290 粒的流体在大幅摇晃下几十秒就会扩散成雾（截图可证）。
照 flip.c 的做法实现了桶排序版本：每步先按网格给粒子分桶（计数→前缀和→落位），
再对每个粒子扫它所在格 + 3×3 邻格、距离 < 2R 的成对推开一半。代价约 +5 ms/拍，
换来流体长时间保持成团（2600 帧后仍是带波浪的一池水）。

**最终参数**：网格 26×18 / 粒子 248（RAD 0.42）/ 压力 12 轮 / 分离 1 遍 /
G_RATE 0.010 / VEL_DAMP 0.94 / 摇摆 51°~69°、周期 2.2~4 s。
host **18 ms/拍**、浏览器 **约 53 ms（≈19 fps）**。

**教训**：这类"幅度不够"的反馈，先量一下**生效值**（打印实际用的 amplitude/period），
很可能是调制/钳位把它压回了小值；再其次才是量级（重力 vs 阻尼）。

### 再调：重力变化速度 ×2

把摇摆周期减半：`gr_period` 78 → **39 拍**（一个来回约 2 s，此前约 4 s），
周期调制幅度 22 → 11（28~50 拍），两个慢调制周期也各减半（620 → 310、430 → 215），
相位抖动间隔 40 → 20 拍。幅度不变（51°~69°）。

效果：同样的 60° 摆幅下节奏快一倍 —— host 端 2600 帧（≈90 s）后流体仍是带波浪的
一池水，没有因为摇得快而散开；浏览器里 248 粒、约 52 ms/拍（≈19 fps）。

### 改成"洗衣机滚筒"：重力方向连续旋转 360° + 洗涤/脱水循环

需求从"来回摆"升级成"滚筒"：重力方向**连续旋转**，转速按洗衣机程序变化，并加入离心项。

```
drum_ang += ω                                  重力方向连续转，不再来回摆
ω：洗涤 0.030 rad/步（约 6 s 一圈） ↔ 脱水 0.170 rad/步（约 1.1 s 一圈）
循环 900 拍（≈45 s@20fps）：0~500 洗涤 → 500~600 加速 → 600~800 脱水 → 800~900 减速
gm = G_RATE·(1 + 1.6·s)                        s∈[0,1] 为程序进度，脱水时表观重力 2.6×
gx = gm·sin(drum_ang)，gy = gm·cos(drum_ang)
离心项：integrate 里额外加 cent_k·(p − 筒心)，cent_k = CENT_K·s²，脱水时把水甩向筒壁
```

效果（截图可证）：
- **洗涤段**：重力缓慢扫过各个方向，水在筒里被带着绕圈、贴着壁流动；
- **脱水段**：离心项起作用，水被压到四壁、**中间空出一大块** —— 就是甩干时"水贴筒壁转"的样子；
- HUD 改为显示 `洗涤/脱水 + 当前筒角`。

实现要点：`drum_ang` 用取模保持在 [0, 2π) 防止浮点漂移；离心项只在 `cent_k>0` 时计算（省一次内层判断）。
成本不变（host 18 ms/拍、浏览器约 53 ms）。

### 加入小黄鸭：浮在滚筒里的浮动障碍物（带稳性）

**素材**：`openclipart.org/detail/325903/rubber-duck`，**CC0 / 公有领域**、透明底 PNG（762×800 RGBA）。
离线用 PIL 重采样成 **30×31** 的字符位图（7 色调色板 + 透明通道），源码里以 31 行字符串内嵌，
加载时展开成调色板索引表（930 项）。Gfx 没有位图 API，所以逐像素 `set_pixel` 绘制，
按当前倾角旋转贴图、透明像素跳过；**画在粒子之前**，没入水中的部分自然被水盖住。

**物理（刚体 + 浮力 + 稳性）**：
- 浮力：把鸭子中心弹簧式地拉向"当地液面 + 吃水"，竖向阻尼 —— 它因此一直浮着不沉；
- **稳性/回正力矩**：目标倾角 = 当地液面坡度的 **40%**（限幅 ±26°），用角弹簧 + 角阻尼逼近。
  水面一平就自己扶正，所以"尽量朝上"；`dka` 硬夹在 ±0.60 rad，绝不翻跟头；
- 横向：跟随所在格的流体速度（水推着鸭子跑）；
- 与水的双向作用：鸭身椭圆内的粒子被推到椭圆边缘，同时给鸭子一点反作用冲量；
- 液面高度来自 `ct`（流体格掩码）：每列最上面那一行流体格 → `surf[col]`。

**踩坑（都是手册里写明的，但我还是先踩了）**：
1. `(- x)` **一元取负不存在**！旋转公式里写成 `(- (* sa dx))`，编译期不报错，运行期
   `Math.floor` 收到 `#undefined` → "指令执行异常"。必须写 `(- 0.0 (* sa dx))`。
2. `Math.min` 不存在（Math 库只有 floor/ceil/round/abs/sqrt/三角函数/对数/位运算）→ 自己比较。
3. 残留的一行 `(set! px_i ...)`（草稿变量，从未定义）同样只会给一句
   "指令执行异常"，定位靠 PC + 逐阶段打点。

**成本**：鸭子绘制 + 注入 + 推开约 +1~2 ms；host 约 20 ms/拍、浏览器约 55 ms（≈18 fps）。

------------------------------------

# 2026-10-29（FLIP 流体：小黄鸭浮力模型重写）

针对 `infer/app/flip.scm` 里的小黄鸭，按三条反馈重写浮力/稳性模型：① 稳性不该回到**世界**
向上，而应相对于**重力与水的向上**；② 鸭子对水平方向的重力变化毫无反应；③ 惯性太小，
经常独自剧烈晃动、不随水漂。**只改 WASM 编译目标所用的那份 `app/flip.scm`**（`infer/app/`
为仓库副本，`/home/bd4sur/linglong/app/` 为运行时资源副本，两者保持一致）。

## 原实现的问题（定位）

- 浮力 = 把中心弹簧拉向「`surf_at(dkx)` 所在列的液面 + 吃水」，倾角 = 当地液面坡度的 40%
  （限幅 ±26°）。`surf` 是**世界竖直方向**每列最上面的流体格，等于假定水永远沉在桶底、
  水面永远垂直于世界坐标——滚筒把水甩到侧壁/顶部时这个假定就崩了。
- 横向只有 `dkvx = 0.92·dkvx + 0.022·u`（网格 x 速度），竖向完全由弹簧决定，**不用**
  网格 v 速度；重力只有竖直分量进入模型 → 水平重力几乎无响应。
- 反作用冲量 `dkvx -= 0.02·lx` 未除以质量，等于鸭子质量 ≈ 单颗粒子。实测鸭子速度是
  流体平均速度的 **4.2 倍**、单拍速度上限 2.0（超过 VMAX），即"自己抖得比水快"。

## 新模型（牛顿第二定律 + 阿基米德浮力，全在鸭身局部系里算）

1. **水面测量（`duck_sweep`，与"挤开水"合并成同一次粒子遍历）**
   - 当地向上 `u = −有效重力`（滚筒重力 + 离心项），`ê = u` 转 90°；
   - 在鸭身左右各两条采样带（横向 1.2~4.2 格、沿 u 只取鸭心上方 3 格内的粒子）里记
     "沿 u 最高的两颗粒子的均值"（抗飞溅），四点**最小二乘**拟合 → 鸭心处水面高度 `dk_lv`
     与水线斜率 `dk_sk`（一阶低通 0.35）；
   - **只在鸭身之外测**：鸭身所在处的水已被 `duck_sweep` 挤空，那里的"水面"是坑底。
2. **排水面积与浮心**：鸭身按宽度切 5 条竖条，各条水线 = `dk_lv` 处、法向为水面法向的直线，
   两端点吃水深线性插值得吃水长度，累加得面积 A 与形心（浮心）。静水吃水处面积
   `duck_aeq` 在 `duck_setup` 里**解析**算出，故 `A = duck_aeq` 时浮力严格抵消重力
   （`gn = 1 + BUOY·(A/aeq − 1)` 只调软硬，不动平衡点）。
3. **浮力方向 = 水面法向**，作用在浮心：水堆到哪边鸭子就往反方向立（= 重力与水的向上）；
   吃水左右不对称时浮心偏移自然产生回正力矩。另加一个弱角弹簧把鸭背拉向当地向上，
   保证整只被水淹没（水线不跨鸭身）时姿态仍有确定收敛方向。
4. **惯性与水阻**：线性按 `M = 60`（粒子质量单位）、角度按 `I = 36` 积分；速度每拍按
   `DRAG = 0.22` 拉向鸭身周围 5 个流体格的网格速度、角速度按 `0.15` 衰减；
   粒子挤开时的反作用按 `1/M` 折减。倾角不再夹在 ±0.6 rad，改为回绕到 ±π（随水翻）。
5. `simulate` 顺序改为 `… p2g → separate → duck_sweep → duck_step → save_prev → solve_pressure → g2p`
   （先量水面再算受力，当拍即用最新水面），并**删掉**原来的 `update_surf`/`surf` 每列扫描。

## 量化对比（headless 宿主台，400 拍，同一初始条件）

| 指标 | 原实现 | 新实现 |
|---|---|---|
| 单拍速度变化均值 `|Δv|`（抖动） | 0.142 | **0.0073**（平滑 19×） |
| 单拍角速度变化均值 | 0.0112 | **0.0009** |
| 鸭子平均速度 / 流体平均速度 | 4.21（最高 2.0/拍） | **0.94**（最高 0.22/拍） |
| 姿态与"当地向上"夹角中位数 | 82.4° | **17.2°**（余量是水自身的滞后） |
| 物理耗时（host，400 拍） | 22.8 ms/拍 | **20.1 ms/拍**（-12%） |
| 浏览器 wasm32（每 20 拍） | 477→625 ms（随时间劣化） | 466→477 ms（稳定） |

## 定点验证

- **水平重力**（把 `turn_gravity` 换成常量 `gx = G_RATE, gy = 0`）：鸭子 160 拍内从 0° 转到
  **90°** 并沿 +x 随水漂，吃水收敛到 −0.32~−0.48（目标 −0.55）→ ①②③ 均满足。
- **整圈滚筒**（默认参数）：倾角跟随 `atan2(gx,gy)`，稳态滞后约 15~20°（角弹簧的
  ramp 跟随误差 `ω_drum·KAD/KA`，与水自身的滞后同量级）。
- **长跑**：3000 拍（2 min）无 NaN/无报错；WASM GUI 端 40 s 连续抓帧鸭子始终贴在水面附近；
  `tools/e2e/browser/regress_det.js` 全 17 个 App 均 `state 80`、刷新持续、无 pageerror。

## 踩坑记录（灵机语言）

- **算术恒二参数**：`(* a b c)`、`(+ a b c)` 不报错但结果是垃圾（多余参数留在操作数栈上），
  直接表现成"浮力方向随机反转"。用 `tools/e2e/host/check_multi_operand.py` 扫全文才发现。
  本次新增代码一度有 7 处三元运算，全部改成嵌套二参数。
- 浮力"向上"的正负号极易搞反：`accel = g_eff + u·|g|·gn`（u 是向上单位向量）；
  写成减号会让鸭子被"浮力"按进水里、永远浮不上来。

## 追加：重力指示器 + 小黄鸭旋转混叠修复（同日）

### 1）左上角重力指示器

`draw_gravity`（`infer/app/flip.scm`，由 `render` 末尾调用，画在流体之上）：

- 底板 68×30 @ (3,3)（深色 + 描边），内部一个半径 11 px 的**满量程参考圈**；
- 箭头从圈心 (18,17) 出发，方向 = 当前重力 `(gx, gy)`（屏幕 y 向下，与数值同号，无需翻号），
  长度 4.5~11 px 线性映射 `|g| / (2.6·G_RATE)`——恰好顶到参考圈时就是脱水满重力；
  箭杆 `Gfx.draw_line` + 两根倒刺（倒刺长/半宽随长度缩放，避免短箭头糊成一团）；
- 圈心一点 + 右侧一句 `String.atom_to_string(Math.to_fixed |g|/G_RATE 1) + "G"`，洗涤期 "1G"、
  脱水期 "2.6G"，与 HUD 的"洗涤/脱水 + 角度"合起来就是滚筒的完整状态。
- 开销：~12 次本地库调用/帧（基本都是 C 侧一次 fill_rect/draw_circle），可忽略。

### 2）小黄鸭旋转后的黑色点阵（混叠）

**根因**：原来的贴图是**正映射**——遍历源图 930 个像素，各自旋转后 `Math.floor` 落到目的像素。
旋转角非 0/90° 时源像素间距在目的网格上出现空档（45° 时相邻源像素落到对角相邻格，中间
沿反对角线的格子永远没人写），于是鸭身内部随角度浮出黑色点阵。

**修法**：改为**目的像素驱动**（反映射）——遍历鸭心周围 44×44 的包围盒，逐像素反向旋转回
源图连续坐标，越界跳过，命中不透明像素才落笔。反映射对任意角度都是满射，因此绝不会漏点。
为控制解释器开销做两点优化：

1. 内层循环**不调用本地库**：源索引 `(jj, ii)` 与其展平下标 `idx` 用增量维护——
   目的像素每前进一格，源坐标各加 `(ca, sa)`，`|ca|,|sa| < 1` 保证索引每步至多进/退 1，
   越界判据比较 `lx` 与 `ii` 的差即可（每行只在行首调 2 次 `Math.floor`）；
2. 先判 `ii/jj` 是否落在源图范围内，再取像素/调色板，跳过包围盒四角的空转。

实测（host x86-64，250 拍含渲染）：20.5 → 22.3 ms/拍（+9%，约 +1.7 ms/拍）；
浏览器 wasm32 仍是 21~25 ms/拍，40 ms 定时器下余量充足。逐角度（0°~360°，9° 一档）
渲染核对：无空洞、无色偏、朝向与 `dka` 一致。

**踩坑**：① 不要试图把"行的整数部分"提到行外——源 `ly` 在行内随 `sa` 漂移（45° 时一行跨 7 个
源行），那样算出的下标会跑到负数（`get_item` 越界返回 `#undefined` → "指令执行异常"）；
② 顶层 `define` 重名会以 `[Parser Error] failed to add defines to scope` 报出（本次重构
误留了一份重复的 `DUCK_HWD/DUCK_HHD/DUCK_BOX`）。

## 追加：粒子数 −1/3（性能）与两处热点清理

反馈"跑起来卡卡的"。WASM GUI 目标里实测（`Diag.ms` 打点 + 逐帧把 `frame_ms` 编码进
一个像素再从 RGBA 帧缓冲读回，比看 fps 更可靠）：**248 粒时每拍 68 ms**，而定时器是 40 ms
→ 实际只有 14.7 fps，确实卡。

### 改了什么

1. **粒子数 248 → 158（−36%）**：`RAD 0.42 → 0.51`（间距 2R 由 0.84 格变 1.02 格）。
   `init_fluid` 的填充体不变，所以水量/水形不变，只是"颗粒更粗"；同时粒子方块
   `PSZ 8 → 11`（≈间距 2R·CELL=12.2 px）保持水看起来仍是连续一片。
2. **渲染内层循环少调本地库**：粒子颜色由"速度→连续 RGB"改成 **12 档查表**（`CR/CG/CB`，
   `build_colors` 在启动时建好），每颗粒子的本地库调用 7 次 → 3 次（`sqrt`、一次 `floor`、
   一次 `fill_rect`）；坐标也不再 `Math.floor`——`fill_rect` 的整型参数在 C 侧按
   float→int 截断（正值即 floor）。
3. **小黄鸭贴图改成行内游程**：同一调色板索引连续命中时合并成一次 `fill_rect`，
   730 次 `set_pixel` → ~150 次 `fill_rect`。逐像素核对过：与逐像素版本**逐字节相同**。
4. **压力求解去掉"每格每轮算行号"**：原来是 `(j = floor(mod c GH))` + 一层 `if`，
   12 轮 × 流体格数 = 每拍 ~1500 次本地库调用。粒子的位置始终夹在
   `[1+RAD, GW−1−RAD]×[1+RAD, GH−1−RAD]`，**进不了边界格**，所以改为在 p2g 建 `fl` 表时
   就把边界格过滤掉（顺带兜住万一被挤出去的粒子），求解内层直接用 `c±GH / c±1`。

### 结果（WASM GUI，headless chromium，帧内 `frame_ms` 中位数）

| | 248 粒 | 158 粒（本次） |
|---|---|---|
| `frame_ms` | **68 ms** | **50 ms** |
| 相对 40 ms 定时器 | 14.7 fps | ~20 fps |
| host（含渲染） | 22.3 ms/拍 | 16.3 ms/拍 |

剩余耗时分布（host/拍，300 拍平均）：压力求解 4.98、p2g 3.22、分离 3.14、渲染 3.31、
g2p 1.32、其余 <1。**50 ms 仍高于 40 ms 预算**，若要真正稳住 25 fps 还需再省 ~10 ms
（wasm），可选：① 粒子再减到 ~130；② `PRESS_ITERS 12 → 8`（求解降三分之一，但压力场
收敛变差、水会更"软"）；③ 关掉密度的 drift 补偿，省掉 p2g 的第三次遍历；
④ 粒子分离改用半邻域（9 格 → 5 格，省 ~1.2 ms/拍，结果等价）。

验证：1200 拍长跑无报错；`tools/e2e/browser/regress_det.js` 各 App 仍 `state 80`、刷新持续。
（注意：该回归的 App 列表是扫 `app/` 目录得到的，本次运行期间工作区里新增了
`app/touchgravity.scm`，因此后续 App 的序号/CRC 会整体位移，与本次改动无关。）

------------------------------------

# 2026-10-29（触屏 / 重力传感器本地库 + 触屏重力球 App）

为"灵机应用程序"（Scheme App）补上**输入通道**：新增两个宿主本地库 `Touch` 与 `Imu`，
把 HAL 触屏（`hal_touch.h`）与 HAL 姿态（`hal_imu.h`）以只读方式暴露给 Scheme。
此前 App 只能画（Gfx）、计时（Time）、自观测（Diag），无法接收任何用户输入，交互式 App
无从写起；本次之后「手指在哪 + 机器怎么倾斜」都能在 Scheme 里直接读到。
**编译目标 WASM 浏览器，已用三个无头浏览器（Chromium / chrome-headless-shell / Firefox）
端到端验证。**

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `infer/am_native_Touch.c/.h` | 本地库 `Touch`：触屏电平/坐标/边沿锁存（6 个过程） |
| `infer/am_native_Imu.c/.h` | 本地库 `Imu`：欧拉角 + 屏幕坐标系重力分量（6 个过程） |
| `infer/app/touchgravity.scm` | 示例兼验收 App「触屏重力球」：手指抓/甩小球 + 重力物理 + 罗盘/仰角指示 |

接线改动：`infer/ui_scheme_app.c`（注册两个库）、`infer/build_wasm_gui.sh`、
`android/app/src/main/cpp/CMakeLists.txt`（源清单）。

## 2. API 与语义

栈纪律严格按上游契约（stack_bal.md §11 的 N1/N6）：两库全部过程**零参数、值型**——
压 1 个值后 `am_process_step` 恰好一次，单出口 `*_push()`；每次 `return -1` 前不压值。

`(native Touch)`：

| 过程 | 返回 | 说明 |
|---|---|---|
| `Touch.is_available` | Boolean | HAL 探测成功（`touch_read` ≥ 0） |
| `Touch.is_pressed` | Boolean | 当前电平 |
| `Touch.x` / `Touch.y` | Number | 触点像素坐标；**未按下时返回最后一次有效值**（HAL 契约：松开时坐标不保证有效，故库内缓存） |
| `Touch.take_press` / `Touch.take_release` | Boolean | 自上次调用以来是否出现过按下/松开沿，**读后清零**（一次沿只报一次） |

`(native Imu)`：

| 过程 | 返回 | 说明 |
|---|---|---|
| `Imu.is_available` | Boolean | `imu_read_angle` 成功 |
| `Imu.roll` / `Imu.pitch` / `Imu.yaw` | Number 或 `#null` | 欧拉角（度）；无数据时 `#null`（`number?` 可判） |
| `Imu.gravity_x` / `Imu.gravity_y` | Number | 屏幕像素坐标系下的重力分量（m/s²），无数据时缺省 `(0, +9.8)` |

**重力符号（易错，已写死在头文件注释里）**：`gravity_x = -9.8·sin(roll)`、
`gravity_y = +9.8·cos(roll)`，即 **x 向右、y 向下为正**（Gfx 的像素坐标系，可直接累加进
Scheme 里的速度）。C 层 FLIP（`ui_app.c` → `flip.c`）用的是 `gravity_y = -9.8·cos(roll)`
——那是 y 向上为正的物理坐标系，二者物理含义相同（竖直立起时物体朝屏幕下方掉），**只是
y 轴朝向不同**；`app/flip.scm` 等 Scheme 侧一律按"y 向下"处理，与本库一致。

软失败约定（manual §1.1.1）：无触屏/无 IMU（Linux 打桩 HAL、浏览器未授权或非安全上下文）
时不报错——`is_available` 为 `#f`、角度为 `#null`、重力和坐标取缺省值，App 可据此降级
（`touchgravity.scm` 在无姿态设备上依然按"下"落球）。

**已知限制**：HAL 只给电平不给事件队列。本库的 `take_press/take_release` 在**被调用时**
采样一次电平并与上次比较，因此**同一帧内的按下+松开（亚帧点按）可能漏检**；人手点按远长
于一帧（~16~33ms），实际使用无影响。需要严格事件语义时应扩展 HAL（Web C/S 形态的
`hal_touch_web_linux.c` 已有 down_pending 锁存可参照）。另：`Imu.*` 每次调用各读一次
`imu_read_angle`，一个典型 App 每帧 2~5 次调用（wasm/desktop 开销可忽略；ESP32 若接入
Scheme App 需按 HAL 的 I2C 成本再议，可在 App 侧按帧缓存）。

## 3. 与 GUI 输入管线的关系

本库**只读旁路**：不消费、不改变 HAL 状态，`ui_app.c` 的 `get_input_event` 照旧读同一份
电平（两边各读各的）。因此：

- Scheme App 运行时仍受 GUI 的**宫格软按键**机制影响——触屏右上角那一格（x≥240 且 y<60）
  映射为 `NANO_KEY_esc`，会被 `ui_scheme_app_event_handler` 当作"返回列表"。故 App 的
  操作区避开该格即可（`touchgravity.scm` 的其余区域任意点按安全）；若将来要让某个 App
  独占触屏，按 `ui_app.c` 注释里的原则把该状态加入 `ui_app_state_is_menu` 抑制表。
- 退出 App 依旧是 A（实体键盘 Esc / 触屏右上角宫格）。

## 4. 示例 App：`app/touchgravity.scm`

- 手指按住 → 小球被抓到指尖（松开时带手指速度甩出），绿色十字准星画在触点上；
- 重力取自 `Imu.gravity_x/gravity_y`：直立向下落，右缘下沉向右滚，左倾向左滚；
- 右上角**罗盘**（黄点指向 `yaw`）与**仰角条**（洋红点表示 `pitch`），HUD 显示触点坐标、
  roll/pitch、重力分量、yaw；
- 33ms `set_interval`（≈30fps）；解释器约束逐条遵守（算术二元、无一元取负、无 list/let、
  块内不用 `define` 只用模块级 `set!`、`/`/`mod` 结果一律 `Math.round/floor`）。
- 小坑记录：VM 浮点载荷位宽有限（见 `am_native_Time.h` 注释），`Math.to_fixed` 只是把
  double 舍入，`%g` 仍会打出 `9.79999` 之类；HUD 的定点格式化改为"×10 取整 → 整数段 +
  小数段拼接"（负数段用 `Math.ceil` 截断）。

## 5. 测试（全部通过）

**A. 宿主无头回归台**（`python3 infer/tools/e2e/native/test_touch_imu.py`，36 项断言全 PASS）

- `tools/e2e/host/harness.c` 增强：注册 Touch/Imu 库，并新增触屏/姿态 HAL 桩，读数按
  **宿主帧号分段常值**由环境变量给出——`HAL_TOUCH_AT="0:1:100:150,40:0:0:0"`
  （帧:按下:x:y）、`HAL_IMU_AT="0:-10:25:120"`（帧:pitch:roll:yaw）、
  `HAL_TOUCH_AVAIL=0` / `HAL_IMU_AVAIL=0` 模拟无硬件。
- `api_probe.scm`：一次采样全量打印 → 断言触屏电平/坐标、`take_press` 首次为真、
  `take_release` 未发生、三个角度精确、`gravity = (-9.8 sin roll, +9.8 cos roll)`
  （roll=25° 时 `-4.14166 / 8.88182`，与公式逐位一致）；无硬件场景断言 `#f`/`#null`/缺省重力。
- `edge_probe.scm`：定时器逐拍采样 → 断言 `take_press`、`take_release` **各恰好一次**、
  `pressed` 由 T 单调转 F、松开后 `x/y` 全程保持最后一次有效值 200/90。
- 端到端渲染（`app/touchgravity.scm` + PPM 帧缓冲质心判据，无 OCR）：直立落底部中央 /
  右倾右下角 / 左倾左下角 / 按住时准星与小球都在触点 / `yaw=120` 黄点方位 / `pitch=-10`
  仰角点位置 / 无姿态时按缺省重力下落且不画罗盘。

**B. 浏览器端到端**（`node infer/tools/e2e/browser/regress_touch_imu.js <URL> <engine> <outdir>`，
Chromium / chrome-headless-shell / Firefox **各 12 项断言全 PASS**）

真实链路：页面 Pointer Events → `wasm_gui_on_touch` → `hal_touch_wasm` 事件队列 → 生产者 tick
回放 → `touch_read` → `am_native_Touch` → App 绘制；DeviceMotion/DeviceOrientation →
`wasm_gui_on_orientation` → `hal_imu_wasm`（低通 + 欧拉角解算）→ `imu_read_angle` →
`am_native_Imu` → App 物理。判据同为帧缓冲质心（红球/绿准星/黄点/洋红点）。

- 导航按 manifest 里 `app/*.scm` 的排序推出 App 在"灵机应用程序"列表中的下标（避免硬编码）。
- 无头构建的 `DeviceMotionEvent.requestPermission` 会默认拒绝（手机上该授权发生在点击
  「姿态」按钮的手势里），测试用 initScript 放行；若某引擎实现了 `DeviceMotionEvent`
  却不填充 `accelerationIncludingGravity`，脚本自动退化为直接调 `Module._wasm_gui_on_orientation`
  注入（本地库链路仍全覆盖），并在日志/结果 JSON 的 `sensorPath` 字段记录实际通路。
- 副作用修正：`app/` 新增文件改变了"灵机应用程序"列表的排序下标，`regress.js` /
  `regress_det.js` 的 APPS 清单与 `run_host_regress.sh` 的 APPS 已同步加入 `touchgravity`；
  全量确定性回归 18/18 App 均 `launch=80 → end=80`。

## 6. 待办 / 备注

- Animac **控制台**（`ui_animac.h`，编辑器/REPL 模式）尚未注册这两个库：它的宿主席位
  （tty/pod 的 Makefile 源清单）目前既缺 `ui_scheme_app.c` 也缺这两个 HAL 组合，故未接线；
  将来接线只需在 `ui_animac.h` 的注册块加两行，并把 `am_native_Touch.c`/`am_native_Imu.c`
  加入对应目标的源清单（ESP32 的 `pod_lite` 目标没有 IMU HAL，只能接 Touch）。
- 触屏多指/手势（滑动、长按、双击）目前留待 Scheme 侧自行实现（库只给原始电平与坐标）。

## 追加：FLIP 可选重力感应（Touch / Imu 本地库接入）

给 `app/flip.scm` 增加重力源切换，用上新增的 `Touch` / `Imu` 本地库
（`am_native_Touch.c` / `am_native_Imu.c`，见两者的 .h 头注释）。

### 交互

- **画布左下角小按钮**（(3,202) 64×18，在 HUD 条上方、画在流体之上）：
  自动（蓝灰色 "自动"）/ 重力感应（绿色 "重力感应"）互切；`(Touch.take_press)` 取按下沿
  + `(Touch.x)/(Touch.y)` 命中判定，一次按下只切一次。若重力感应模式下
  `(Imu.is_available)` 为假（无传感器 / 浏览器未授权姿态），按钮文字转橙色的"无传感器"。
- 该位置在触屏 4x4 宫格里对应 `NANO_KEY_left`，灵机 App 运行态只响应 ESC，
  所以点按钮不会误触别的功能（左上角宫格是 esc，故按钮特意放左下角）。

### 重力

- `turn_gravity` 分两支：自动支路是原有滚筒逻辑（不变）；重力感应支路读
  `(Imu.gravity_x)/(Imu.gravity_y)`——头文件约定是**屏幕像素坐标系**（x 右、y 下）
  的 1G=9.8 分量，与本工程 `gx/gy` 同向，故 `gx = G_RATE·imu_gx/9.8`，直接可用；
  同时 `cent_k = 0`（滚筒离心项无意义）、`drum_s = 0`（HUD 不再显示洗涤/脱水）。
  `|g| = 9.8` 恒成立 → 重力感应下箭头长度固定、只有方向随设备倾斜变化。
- 左上角重力矢量箭头本来就是按 `(gx,gy)` 画的，因此**零改动**自动反映感应重力；
  仅把面板描边/参考圈在感应模式改成绿色系以便一眼分辨重力源。
- HUD：标题缩短为 "FLIP 流体"（给左下按钮让位），右侧状态文字在感应模式显示
  "重力感应 <角度>°"（角度 = atan2(gx,gy)，与自动模式的滚筒角同一约定）。

### 验证

- **WASM GUI（headless chromium，真注入）**：点按钮 → 按钮/面板像素由蓝变绿（(26,36,56)/(62,84,116)
  → (22,56,38)/(52,132,78)）；注入 `wasm_gui_on_orientation`（姿态 HAL 有 0.18 低通，
  需连续注入 40 次才收敛）后，从帧缓冲里量出的箭头质心方向 −33°/+23°/−1° 与注入的
  ±30°/0° 相符；水面像素质心随倾角左右移动（+30°→129.3，−30°→148）。再点一次回到自动 ✓，
  无 pageError。
- **宿主台**（给 `/tmp` 的 harness 副本补注册 Touch/Imu 库，用 `HAL_TOUCH_AT` / `HAL_IMU_AT`
  脚本化输入）：按钮内点按 → mode 0→1（按钮外点按保持 0）；IMU roll=−30° →
  g=(0.005, 0.00866)、水面质心右移、鸭子倾角收敛到 +25~30°（= atan2(gx,gy)）；
  `HAL_IMU_AVAIL=0` → mode=1 且 imu_ok=0、重力退化为 (0, G_RATE)（屏幕向下）；
  感应模式连跑 1200 拍无报错。
- `regress_det.js` 全 App 仍 `state 80`、刷新持续。

### 备注

- 本次改动只碰 `app/flip.scm`；`am_native_Imu.c` / `am_native_Touch.c` 已由他人加入
  `build_wasm_gui.sh`，故 WASM 目标需**重新构建**一次才带这两个库（已重建）。
  `infer/Makefile` / `mp135.mk` 的源文件清单尚未包含这两个 .c，而 `ui_scheme_app.c`
  已引用其库符号——非 WASM 目标（tty/pod 等）在补齐前会链接失败，属他人并行工作范围。

------------------------------------

# 2026-10-29（俄罗斯方块 Scheme 版：触屏本地库的第二个落地 App + 逻辑/界面分层）

把 C 版 `infer/ui_tetris.c`（STATE_TETRIS）移植为灵机 Scheme App：规则、布局、配色、
虚拟按键交互逐条对齐，输入走上一轮实现的 `Touch` 本地库（Scheme App 没有键盘通道）。
**首次把一个 App 拆成「纯逻辑模块 + 界面/输入层」两文件**，逻辑因此可以脱离图形单独回归。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `infer/app/lib/tetris_core.scm` | 纯游戏逻辑：场地/7 种方块/旋转/碰撞/生成/锁定/消行/计分/升级/重力。只依赖 `Math`（随机数与取整），不碰 Gfx/Touch |
| `infer/app/tetris.scm` | 布局、绘制、触屏虚拟按键、退出确认框（`(import T "app/lib/tetris_core.scm")`） |
| `infer/tools/e2e/native/tetris_logic.scm` | 规则单元测试（71 条断言，直接 import 核心模块） |
| `infer/tools/e2e/native/test_tetris.py` | 一键测试：规则单测 + 界面/触屏像素断言（21 条） |
| `infer/tools/e2e/browser/regress_tetris.js` | 浏览器端回归（真 WASM + 真触屏链路，18 条断言） |

## 2. 与 C 版的规则一致性

10x20 场地；7 种方块（4x4 包围盒、顺时针 `(x,y)->(3-y,x)`）；踢墙序列 `0/-1/+1/-2/+2`；
生成位置 `px=3, py=-1`、无法入场即结束；消行自下而上、满行下移后**原地重查同一行**
（C 版 for 循环里的 `y++` 的等价写法）；计分 `100/300/500/800 × 关卡`；行数每满 10 升一关；
落地间隔 `max(800-70*(level-1),100)` ms；左面板（标题/得分/行数/关卡/下一个预览）与
配色表（`S_TRS_COLORS`）、虚拟按键同款。

**与 C 版的三处平台差异**（均因"Scheme App 没有键盘、且运行在宿主 GUI 里"）：

1. 输入只有触屏：C 版的 ←/→/1/2/D 硬按键动作全部由虚拟按键提供（左移/旋转/右移/速降）。
2. **「退出」按钮从右上角移到场地右侧空白列**：触屏右上角（`x≥SW*3/4` 且 `y<SH/4`）是宿主
   4x4 宫格映射的 esc 格，点它会直接被宿主当"返回列表"退出 App（C 版把 STATE_TETRIS 列入
   `ui_app_state_is_menu` 抑制表才敢用那个角，Scheme App 无法抑制）。「速降」留在顶部左侧
   —— 场地横向居中（320 宽时从 x=110 起），顶部四分之一落在场地左侧，不遮挡入场中的方块。
   （初版把「退出」放在顶部第二格，正好压住场地 x 110..160，导致方块入场时被按钮盖住，
   测试用"场地彩色格包围盒"一眼看出来，才改成现在的位置。）
3. 确认框的「确认」无法真正切回小游戏菜单（Scheme App 不能改宿主 `STATE`），故**冻结游戏**
   并提示"按 A 返回"；`A` / 触屏右上角仍是宿主提供的唯一返回路径。

## 3. 交互与节奏

- 虚拟按键语义同 C 版：左移/右移/速降 **按下沿武装、松手沿结算**（短按单次），按住满
  500ms 后**每拍反复触发**；旋转/退出只在松手沿单次触发；确认框打开时游戏暂停。
- 物理步长改用 `System.timestamp` 的实测间隔并钳到 50ms（`DT_MAX`，同 C 版 `TRS_DT_MAX`）。
  **踩坑**：宿主对过期的定时器按"当前时刻 + 周期"重排，`set_interval 33` 实测约 48~50ms
  才触发一次；若按固定 33ms/拍累计重力，游戏速度只有正常的 2/3。改成实测间隔后
  960ms/1920ms/4800ms 三个采样点的下落格数与理论一致（1/2/5 格，首拍 dt≈0 少算一格属正常）。
- 节拍内渲染整屏（清屏 → 场地边框 → 已锁定格 → 当前方块 → 面板 → 虚拟按键 → 遮罩）。
  宿主上实测约 20~30 拍/秒（每拍含 200 格扫描 + 十余个 fill_rect + 5 行文字），
  wasm32 浏览器里约 30 拍/秒。

## 4. 解释器踩坑（本次新增，均已记入代码注释）

1. **`(T.px)` 是"以变量为运算符的调用"**，不是取值——把模块变量当函数调用会得到
   `[Runtime] call: 错误的调用目标`。在测试里写 `(eq "x" (T.px) 3)` 曾让整个测试
   停在第 5 节（且报错无位置信息），排查耗时；正确写法是 `T.px`。（这不是解释器 bug，
   是"点号标识符既是模块变量也可能是库过程"的语法陷阱。）
2. **跨函数共用模块级临时量会串味**：`rotate_piece` 用 `t_t` 存踢墙偏移，而 `collide` →
   `fill_cells` 用 `t_t` 做旋转交换，于是 `(set! px (+ px t_kick))` 里的 `t_t` 被覆盖成
   方块格子的 x（I 块为 3），一次旋转把方块右推 3 格。修法：踢墙偏移改用独立变量
   `t_kick`、`fall_ms` 中间量改 `t_fm`、`collide` 的命中标志改 `t_hit`（原与
   `rotate_piece` 的"已成功"标志 `t_ok` 撞名，会让踢墙只试第一档）。**规则：模块级
   临时量按"谁写谁读"分配，跨函数调用链上的变量绝不共用。**
3. 跨模块 `set!`（`(set! T.fall_acc 0)`）不可依赖 → 核心模块补了 `T.reset_fall`
   这类"小动作函数"给界面层调用。
4. 界面层 `(if ...)` 里不要写 `define`（沿用既有结论），所有局部量仍走模块级 `set!`。

## 5. 验证

- **规则单测**（`python3 infer/tools/e2e/native/test_tetris.py`）：**71/71 断言 PASS** ——
  开局状态、四种旋转（含 O 旋转后集合不变）、旋转四次归位、碰撞（左右越界/底边/已锁定格
  上下沿）、硬降锁定（4 格、颜色索引 = piece+1）、消一行（+100 分、竖 I 上段随之下移后
  余 3 格——这正是正确的俄罗斯方块行为）、连消四行三轮（800/1600/2400 分，12 行 → 2 关，
  2 关间隔 730ms）、`fall_ms` 五档、重力累加（700ms 不动 / 800ms 落一格）、撞墙后 `px`
  不再变、游戏结束置位且 `move`/`rotate`/`hard_drop` 全被屏蔽。
- **界面/触屏**（同一脚本）：**21/21 PASS** —— 开局 4 格入场于列 3..6 行 0、960ms/1920ms
  分别落到第 1/2 行、短按左移/右移各一格、旋转后竖 I 在第 5 列（顶部一格在场地上方故可见
  3 格）、速降落一格、长按左移连发到墙边（列 0）、退出框打开（确认按钮强调色像素）、
  留下后关闭且不冻结、确认后关闭并显示冻结提示、**框开 1.4s 后关框方块仍在第 0 行**
  （证明暂停期间重力不累计）。
- **浏览器端到端**（`node infer/tools/e2e/browser/regress_tetris.js <URL> <engine> <outdir>`）：
  Chromium / chrome-headless-shell / Firefox **各 18/18 PASS**。真链路
  Pointer Events → `wasm_gui_on_touch` → `hal_touch_wasm` → 生产者回放 → `touch_read` →
  `am_native_Touch` → Scheme 虚拟按键 → 核心逻辑。断言同样基于帧缓冲（方块格、确认按钮
  强调色、冻结提示文字像素），不依赖方块类型（浏览器里 `Math.random` 以真实时间播种，
  首块随机）；旋转用例对 O 块（旋转不变）自动放宽。
- 回归：宿主 19 个 App 全 `err=none`；浏览器确定性回归 **19/19** `launch=80 → end=80`。
  `app/` 新增文件把"灵机应用程序"列表下标整体后移，`regress.js`/`regress_det.js`/
  `run_host_regress.sh` 的 APPS 清单已同步加入 `tetris`。
- 另做了 40 次速降的浏览器压测（无 pageerror、状态保持 80、方块正确堆叠）与
  "顶部四行填满 → 游戏结束遮罩"的渲染核对（黑底 + 红字"游戏结束" + 得分，位置同 C 版）。

## 6. 后续可做

- 键盘通道：若将来给 Scheme App 补按键本地库（`(native Key)`），可把 C 版的方向键/
  旋转键/硬降（Enter）/加速（2）接上，虚拟按键退化为触屏备选。
- 加分项（C 版也没有）：下一个方块队列（7-bag 随机）、暂存（hold）、消行动画、最高分持久化。
- `tetris_core.scm` 的接口已可直接给"AI 自动游玩"或录像回放复用（`spawn_piece` /
  `advance` / `move` / `rotate_piece` / `hard_drop` + 只读状态）。

------------------------------------

# 2026-10-30（交互式支持向量机 SVM 演示 App：SMO 训练 + 分类区域/边界/支持向量）

按参考页面 `https://bd4sur.github.io/html/svm.html`（Karpathy svmjs：简化 SMO + canvas
绘制）实现灵机 Scheme App「支持向量机」。编译目标 **Nano-WASM-GUI**（`build_wasm_gui.sh`
产物），三个无头浏览器（Chromium / chrome-headless-shell / Firefox）端到端验证通过。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `infer/app/svm.scm` | App 本体（880 行）：触屏加点、三按钮、SMO 训练、分类区域/边界/间隔线/支持向量绘制、参数面板 |
| `infer/tools/e2e/browser/regress_svm.js` | 浏览器端回归（真 WASM + 真触屏链路，18 项帧缓冲像素断言；`tools/e2e/` 不入库） |

接线：`build_wasm_gui.sh` 的 `cp app/*.scm $LINGLONG/app/` + manifest 生成已自动收录
（`app/svm.scm` priority 1）；"灵机应用程序"列表按文件名自动扫描，无需改 C。

## 2. 交互与布局

- **点按画面空白处**：按当前类别即时加一个样本（正橙 `#ff9933` / 负蓝 `#0086bb`）；
- **左下角三按钮**（y 212..234，宽 64）：`正样本/负样本`（切换待添加类别，橙/蓝底）、
  `线性核/RBF核`（切换核函数，绿/紫底）、`重置`（样本/模型/类别/核全部回初始态）；
  每个按钮**只显示当前状态**，与参考页面一致；
- **参数面板（左上角）**：线性核 `wx / wy / b`（对参考页面的 `wx/wy/bias`）；RBF 核
  `σ / C / b / SV / n`；第 5 行是与类别按钮同色的提示行；
- **右下角状态**：`n=样本数/48  SV=支持向量数`（样本满 48 后不再增加，一眼可见）；
- **右上角**：宿主 4x4 宫格的 ESC 格（点它返回列表），画了「返回」标条，避免误当成绘图区。

## 3. 算法（逐条对齐参考实现，改动处见第 5 节）

- 简化 SMO（随机取 j、盒约束 [L,H]、eta<0 判据、b1/b2 择优），内核矩阵按 `n×n` 缓存于
  预分配列表（行主序 `i*MAXN+j`）；线性核训练完成后折算 `w = Σ αᵢyᵢxᵢ`，预测退化为一次点积；
- 判别函数 `f = b + Σ αᵢyᵢK(x,xᵢ)`；RBF 只保留 α > 1%·max(α) 的支持向量参与求值；
- 绘制顺序：分类区域 → 线性核的间隔线(w·x+b=±1，灰)+分类边界(w·x+b=0，红)+支持向量
  垂线 → 支持向量圈注 → 样本点 → 参数面板/按钮；
- 直线一律按 `p₀ = (k-b)/|w|²·w`、方向 `(w₁,-w₀)/|w|` 参数化（对竖直方向成立，避免
  "w₁=0 除零"——解释器除零会直接终止进程）；线段画前用 Liang–Barsky 裁到屏幕内。

## 4. 性能：三处必做优化（帧数 = 312500 条 VM 指令/帧的推进量）

1. **分类区域不能逐格 `fill_rect`**：320×240/2px 共 19200 次 native 调用（每次约 130 条
   指令）⇒ 单次重绘 13 帧（≈200ms，期间点按会被整段吞掉）。改为：
   - **线性核**：`f` 是 x 的一次函数，每行颜色分界点解析求出（`x* = -(b+w₁y)/w₀`），
     240 行每行至多 2 次 `fill_rect` ⇒ 重绘 < 1 帧；
   - **RBF 核**：行内**行程合并**（同色相邻采样格合成一次填充）+ 相邻采样点间按 `f`
     **线性插值求亚像素分界** ⇒ 4px 采样也能得到 1px 精度的边缘；`d² > 7200` 时跳过
     指数调用（核值 < 1.2e-4，不影响判决）。
2. **采样步长按支持向量数自适应**（每格成本 ∝ nsv）：nsv ≤ 10 → 4px、≤ 24 → 6px、
   否则 8px。实测：4 SV 3 帧、12 SV 5 帧、20 SV 7 帧（≈110ms）。
3. **训练量必须封顶**：参考实现是 `numpasses=200 / maxiter=10000`（JS 毫秒级跑完，解释器
   下 20 点就是几十秒）。改为 `MAXITER=300` + **工作量预算** `iters·n² ≤ TRAIN_WORK(40000)`
   + 相对收敛判据（见下）。实测：线性训练 0~1 帧（可分数样本）、20 点不可分 10 帧上限。

## 5. 参考实现的两处"照抄即坏"（关键发现）

1. **α 变化的绝对阈值 1e-4 在屏幕坐标量级下失效**。参考页面 `if(|aj-newaj| < 1e-4) continue`
   隐含"α 的量级 ≥ 1e-2"，但线性核 + 坐标量级 10² 时最优 α 只有 **1e-4 量级**（实测某个
   6 点样本集：α = 1.0363e-4，`|w| = 2/margin`），于是每一步都被当成噪声跳过 ——
   **训练后 α 全 0、分类区域一片蓝、既无边界也无支持向量**（本机宿主/浏览器均复现）。
   改为相对门槛 `EPSA×C`（`EPSA=1e-6`）后同一数据得到正确最大间隔解（w=(−0.0124,0.0073)、
   b=1.0415，间隔线恰过两个支持向量）。同理 `|L−H| < 1e-4` 与"收敛判据"也一并相对化。
2. **随机 j 的简化 SMO 在不可分数据上不收敛**（用 node 直接跑参考 svmjs 实测：9 点交错
   标签、`numpasses=200` 时跑满 10000 轮，结果为 w≈0、全部 margin 为负）。故不追求
   "跑满参考轮数"，而是用工作量预算 + `|Δα|max < EPSA×C` 的安静轮判据提前收敛；
   可分数样本（典型使用方式）几轮即收敛，不可分样本本就在最优点附近游走、多跑无益。

## 6. 其它踩坑

- **列表当数组**：`get_item/set_item!` O(1) 随机访问，预分配 `MAXN` 与 `MAXN×MAXN` 两个
  列表即够；空表用 `(cdr '(0))` 现场取**新**对象（顶层 `'()` 是模块级静态槽，push 会
  原地替换槽指向的对象）。
- **`Gfx.draw_line` 坐标是 uint32**：负值会被当成 4e9 量级的天文坐标（不是被裁掉），
  故所有可能越界的线段先裁剪再画。
- **字模步进实测**（`glyph_12.h`）：ASCII 7px、CJK 12px、σ（U+03C3）7px 且字模存在；
  按钮文字居中、状态行右对齐都按此度量（`tw()` 按码点 < 0x2E80 判 7/12）。
- **触屏语义**：加点用**按下沿**（即时反馈），按钮用**按下沿武装 + 松开仍在按钮内才结算**
  （对齐项目触屏范式，可滑出取消）；`take_press/take_release` 是电平差分锁存，跨帧
  （≥20ms 一拍）采样，人手点按无碍；但**重算阻塞期间（>100ms）的整段点按仍可能被吞**
  —— 这正是第 4 节要把重算压到 100ms 量级的原因。
- **浮点定点格式化**：wasm32 下 `am_float_t` 是 float 且尾数又被截 5 位，`%g` 会打出
  `0.123000003` 之类；参数面板的 3 位小数按"×1000 取整 → 整数段 + 补零小数段"自拼
  （首版 `pad3` 与 `fmt3` 共用模块级临时量导致负号丢失，已按"谁写谁读"分名修复）。

## 7. 验证

- **宿主无头回归台**（`tools/e2e/host/run_host_regress.sh APP=svm`）：`err=none`；
  另用临时 `HAL_TOUCH_AT` 脚本化的多组样本（可分/交错/XOR/单类/重置后重来）逐帧截图核对
  （PPM→PNG 目检 + 像素统计），最大间隔解、间隔线过支持向量、RBF 团块与重置均正确。
- **浏览器端到端**（`node tools/e2e/browser/regress_svm.js <URL> <engine> <outdir>`）：
  Chromium / chrome-headless-shell / Firefox **各 18/18 PASS** —— 初始白屏、加点后区域/
  边界/圈注出现、左下黄右上蓝、RBF 后区域变化且红色边界消失、切回线性边界恢复、
  重置回白屏、重置后可继续加点、Esc 返回列表、无 pageerror。判据全部是帧缓冲像素
  （区域/边界/圈注/纯白四类计数 + 取样点颜色），不做 OCR。
- **交互延迟实测**（headless chromium，`_wasm_gui_get_fb_epoch` 计数）：点按后
  ~90ms 完成"训练 + 整帧重绘"（线性核 1 帧量级）、RBF 核 3~7 帧（≈90~120ms）；
  首个"计算中"提示在点按后立即上屏（重算前的 `gfx_refresh`）。
- 回归：`regress.js` / `regress_det.js` / `run_host_regress.sh` 的 APPS 清单已加入 `svm`
  （位于 `sky3` 与 `tetris` 之间，避免"灵机应用程序"列表下标错位）。

## 续（2026-10-30）：线性核弱不可分病态 —— 训练特征归一化

### 现象

"大部分数据线性可分、只有极少数点不可分"时，线性核每次（随机 `j` 决定的）运行得到
**完全不同的边界**：有时是直觉上完美的那条，有时则完全没有边界（分类区域一片单色）。

### 定位：不是多解，而是"目标函数对几何盲 + SMO 尾部极慢"

对偶目标 `D(α) = Σα − ½‖w‖²`，**只有 `½‖w‖²` 带几何信息**。定义
`ρ = ½‖w*‖² / Σα*`（几何信息在对偶目标中的占比），用双精度复刻本 App 的算法
（200 次随机复跑，数据 = 左下 6 正 + 右上 6 负，逐步渗入"内鬼"点）：

| 数据 | 内鬼 | Σα* | ½‖w*‖² | ρ | 收敛所需轮数（实测） |
|---|---|---|---|---|---|
| 完全可分 | 0 | 1.25e-4 | 6.2e-5 | **0.50** | **50** |
| +1 内鬼 | 1 | 2.21 | 5.9e-5 | **2.7e-5** | **12800** |
| +2 内鬼 | 2 | 4.29 | 5.9e-5 | **1.4e-5** | **12800** |
| +3 内鬼 | 3 | 5.91 | 5.8e-5 | **9.8e-6** | **25600** |

- 一个内鬼必然被顶到盒约束上界（`α=C`），而等式约束 `Σ yᵢαᵢ = 0` 要求别处配平
  ⇒ `Σα*` 从 `1e-4` 跳到 `O(1)`（4~5 个数量级）；而 `½‖w*‖² = 2/margin²` 仍是
  像素场景的 `~6e-5`。几何信息占比 ρ 因此塌到 `1e-5`：**目标函数对边界的法向与
  幅度几乎无感**（把 `|w|` 从 0.011 改成 0.045 只花 1e-3 的目标值）。
- SMO 每步 `Δα ≈ 违反量/η`，`η = ‖xᵢ−xⱼ‖² ≈ 1e4~1e5`（像素单位）；尾部违反量被
  `TOL=1e-3` 压住 ⇒ `Δα ≈ 1e-8`，而 α 空间里的距离从 `1e-4` 变成 `O(1)`
  ⇒ 需要 1e4 量级的轮数（实测 12800~25600）。App 预算是 `TRAIN_WORK/n² ≈ 2e2` 轮，
  **差 46~125 倍**，于是停在随机游走的半路点。实测同预算 6 个种子：
  `Σα` 稳定在 3.09~3.21（±2%），而 `‖w‖` 0.0029~0.141（50 倍）、角度 43°~120°、
  边界距原点 25~561 px（出屏即"没有边界"）、误分 2~12。
- 判据公式：`ρ ≈ 2/(C·margin²)` ⇒ 良定条件 `C·margin² ~ O(1)`。像素特征
  （margin ~ 200 px）配 `C=1` 比"平衡点"（`C≈2.5e-5`）深了 4e4 倍——这也是参考页面
  自身会退化的原因（用 node 直跑页面里的 svmjs，11 点像素数据得 `w=0`）。
- 次级原因：2-D 线性核的 Gram 秩 ≤ 2 ⇒ **最优 α 不唯一**（只有 w 唯一），
  支持向量集合与 `b`（参考实现的 `b1/b2` 折中）即使在最优点也会抖。
- RBF 核没有此病：`K` 对角为 1、取值 ∈(0,1]，核自带尺度归一（σ 已按像素尺度选），
  ρ 从不到病态区（实测 11~22 轮收敛、margin 恰为 ±1、20/20 稳定）。

### 修正：线性核在归一化特征上训练

- 训练特征 `x' = (x − 屏幕中心)/160`（绘图区 ≈ `[-1,1]×[-0.75,0.75]`），常量
  `TCX/TCY/TSC/TISC`；样本坐标另存 `pxn/pyn`（`add_sample` 内同步写入）。
- `kern_build`：线性核用 `pxn/pyn` 求 `x'·xⱼ'`；**RBF 分支仍用像素距离**——因为
  `K = exp(−‖Δx‖²/2σ²)` 只依赖 `‖Δx‖/σ` 的比值，`σ' = σ/160` 时核值逐位相同
  （RBF 解与归一化前完全一致，且省掉栅格内每格一次换算）。
- **偏置拆两个变量**：`bn` = 训练空间偏置（SMO 内部与 `margin()` 用）；`b` =
  判别/绘制用像素偏置。训练结束后对线性核回算：
  `w_px = w'/160`、`b_px = b' − w_px·中心`（`f(x) = w'·x' + b'` 与像素形式是同一个数，
  `w·x+b=±1` 的间隔线因此完全一致）；RBF 分支 `b = bn`。
- `C_LIN = 1` 现在作用在**归一化特征**上（等价像素单位 `C_px ≈ 3.9e-5`）；`C_RBF`、
  `σ` 数值不变（核值不变）。`TRAIN_WORK` 保持 40000（≈10 帧/170 ms）。

### 验证

- **wasm32 目标台**（emcc 编译宿主回归台，真解释器 + 真 native，脚本化数据集并对样本
  顺序轮换来换随机轨迹），每组 10 次复跑：

  | 数据 | ‖w‖ 范围 | 角度范围 | 误分 | 收敛 |
  |---|---|---|---|---|
  | 可分 12 点 | 0.010668~0.010685 | 144.7~146.0° | 0 | 10/10 |
  | 弱不可分 13 点（1 内鬼） | 0.009594~0.009738 | 143.9~146.2° | 恒为 1（内鬼） | 10/10 |
  | 弱不可分 14 点（2 内鬼） | 0.009524~0.009591 | 143.1~143.8° | 恒为 2 | 10/10 |

  （对照归一化前：`‖w‖` 跨 50 倍、角度跨 90°、误分 2~12、0/20 收敛。）
- n=30 时预算内 2/3 次达到收敛判据（34~44 轮），未收敛的一次结果也只差 ~1%；
  n=48 时预算只够 17 轮，`‖w‖` 差 ~10%、角度差 2~3°（视觉一致）。
- 宿主无头回归：全部 20 个 App `err=none`；宿主台脚本化触屏的多组数据逐帧截图核对。
- 浏览器三引擎（Chromium / chrome-headless-shell / Firefox）端到端 **各 18/18 PASS**；
  交互延迟不变（线性核点按后 ~90ms 完成"训练 + 整帧重绘"，RBF 3~7 帧）。

### 遗留 / 可选优化

线性核训练里 `margin(i)` 仍是 `O(n)`（走内核矩阵），故单轮 `O(n²)`、预算随点数被摊薄。
若把线性核改为**增量维护 `w'`**（`margin(i) = bn + w'·xᵢ'` 为 O(1)，`η`/`K(i,j)` 现算），
单轮成本降到 `O(n)`（n=48 时约 50 倍），轮数预算可放大到数百轮，n 很大时也能完全收敛。

------------------------------------

# 2026-10-30（键盘 / 文件两个本地库 + 便签 App）

给「灵机应用程序」补上两条此前完全缺失的通道：**键盘输入**与**文件读写**。
此前 App 只能触屏+姿态（Touch/Imu）且是只读世界（数据只能硬编码或写成数据模块），
于是便签、存档/高分榜、可打字工具一个都写不出来（app/tetris.scm 只能用触屏虚拟按键即
此限制的直接后果）。本次 **编译目标 Nano-WASM-GUI**，宿主回归台 + 三个无头浏览器端到端验证。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `infer/nano_key_events.c/.h` | 按键边沿**旁路队列**（平台无关）：HAL 投递边沿，GUI 读取路径零改动 |
| `infer/am_native_Key.c/.h` | 本地库 `Key`：`is_available/is_down(code)/take_press/take_release/dropped` |
| `infer/am_native_Fs.c/.h` | 本地库 `Fs`：`exists/is_dir/size/list/mkdir/read/write/append` |
| `infer/hal_fs_persist_wasm.c` | WASM 专属：写入成功后把文件同步进浏览器 IndexedDB（弱钩子实现） |
| `infer/app/notepad.scm` | 便签 App —— 两个库的用法示范兼端到端验收程序（自动保存、载入回显） |
| `tools/e2e/native/{key_probe,fs_probe}.scm`、`test_key_fs.py` | 宿主台 API 语义 + 端到端测试（44 项断言） |
| `tools/e2e/browser/regress_key_fs.js` | 真浏览器回归（17 项断言，Chromium/chrome-headless-shell/Firefox 均全过） |

## 2. 关键设计：Key 为什么不能直接读 `input_device_read_key()`

HAL 的键盘接口是「电平 + 单槽位」：wasm 实现读到电平后会清掉未读锁存，ncurses 实现更是
直接从 stdin 取字节。Scheme 侧若照 Touch 库的写法去读，就会与 GUI 输入层**抢事件**（谁先读
谁拿到）。因此改为**旁路**：HAL 在注入 GUI 电平槽的同时另投一份边沿事件进
`nano_key_events`（按每键电平表去重，浏览器 auto-repeat 天然无害；队列满丢最旧并计数，
投递方永不阻塞）。GUI 的 `get_input_event` / 长按连发 / 宫格映射 / 软复位全部照旧。

- 投递点：`hal_key_wasm.c` 新增 `wasm_gui_on_key_event()`（页面每个按键都调，**忠实键码**：
  可打印字符即 ASCII，命名键用 `NANO_KEY_*`）；`hal_key_ncurses_linux.c` 在映射表旁另投
  「按下+松开」成对事件（终端无 keyup/电平，故 `Key.is_down` 在终端恒 #f）。
- 页面 `webgui/index.html`：新增 `RAW_KEY_MAP` + `rawKeyCode()`；`reportKey` 先投旁路、再走
  原有 `KEY_MAP`（十六键重排）→ **GUI 行为逐字节不变**，字母等未映射键只进旁路。
- `am_native_Key` 把两种边沿分栈存放，`take_press`/`take_release` **互不吞噬**（可先取若干
  按下沿再取松开沿）；`wasm_gui_on_key_event` 上限 255，`NANO_KEY_POWER(128)` 等不冲突。
- `ui_scheme_app.c` 在创建运行时后调用 `am_native_Key_reset()`：App 未运行期间（菜单里）敲的
  键不会涌给新启动的 App（e2e 专门验证了这条）。
- 平台能力声明：HAL 的 `input_device_init()` 调 `nano_key_events_mark_source()`，未接入的
  平台 `Key.is_available` 返回 #f，App 据此降级为纯触屏（M5 触屏机 / web / android 现状）。

## 3. Fs 的语义取舍

- **路径**：以 `/` 开头 = 宿主绝对路径原样使用；其余 = 相对业务资源根（`PLATFORM_ROOT_DIR`），
  因此 `"data/note.txt"` 在桌面/WASM/掌机三处通用。含 `..` 与空路径一律拒绝（不穿越资源根）。
- **只做文本整读整写**（不做随机访问/二进制/删除改名——hal_fs 也没有）；读入上限 128 KB，
  超限返回 `#null` 而**不截断**（截断会静默损坏数据，App 可先 `Fs.size` 判断）。
- `Fs.write` 自动逐级创建父目录；`Fs.append` 走「读旧 + 拼接 + 整写」（小文件足够）。
- 失败一律软失败（`#null`/`#f`），与 manual §1.1.1 一致。
- `exists`/`size`/`read` 会短暂借用 hal_fs 的**全局单句柄**（ebook/dict 用）；单前台 App 模型
  下安全，若将来并发读需给 hal_fs 换多句柄（已写进头文件注释）。

## 4. WASM 持久化（写入落 IndexedDB）

MEMFS 是内存文件系统，刷新即丢。`am_native_Fs.c` 在写入成功后调用**可选弱钩子**
`am_native_fs_on_write()`；`hal_fs_persist_wasm.c` 用 EM_JS 把「资源根相对路径 + 内容」
`nanoStorePut` 进 `nano-store`（与用户上传资源同一通路），下次启动由 `loader.js` 以「本机来源」
挂回 MEMFS。未链接该文件的平台零开销（弱符号为空）。页面未加载 store.js（如 node）时静默跳过。

## 5. 同时修掉的三处周边问题

1. `build_wasm_gui.sh` 的 manifest 增加 `SKIP_DIRS={"data"}`：`data/` 是 App **运行时数据**，
   不该随产物发布（此前一次宿主测试落下的 `data/note.txt` 被当作 p2 资源发布，e2e 因此
   非确定——现由测试脚本从零构造）。
2. `tools/e2e/browser/{regress.js,regress_det.js}` 的 `APPS_ALL` 补入 `notepad`（清单须与
   `app/*.scm` 实际排序一致，否则其后的 App 下标整体错位）。
3. `notepad.scm` 的光标 y 曾按 `(if (> cw cp) ...)` 分支画，未滚动时落到原点——截图看出来后
   改为统一按可见行序号算。**教训：这类纯视觉 bug 只能靠截图核对，像素断言抓不到。**

## 6. 验证

- **宿主无头回归台**（`python3 tools/e2e/native/test_key_fs.py`，44 项全 PASS）：边沿序列
  （按下/松开各恰好一次、多键并存、去重、无事件源软失败）；Fs 全 API（UTF-8 字节数 9/16、
  append 后码点 12、覆盖写、目录枚举按名升序、`#null`/`#f` 软失败、`..` 拒绝、128KB 上限）；
  便签端到端（注入 h,i,e,退格,回车,`!` → 落盘 `"hi\n!"` 与键入一致、二次启动读回不清空）。
- **宿主全 App 回归**（`run_host_regress.sh`，21 个 App）：全部 `err=none`，其余 App 帧序列不变。
- **浏览器 e2e**（`regress_key_fs.js`，Chromium / chrome-headless-shell / Firefox 各 **17/17 PASS**）：
  键入上屏（正文区亮像素 0→100）；自动保存进 MEMFS（"abc123"）；Delete 退格（"abc12"）；
  写入落 IndexedDB；**重开页面后 MEMFS 仍有该文件**（持久化闭环）；再进 App 读回并绘制；
  Esc 仍返回列表（GUI 通路未受影响）；列表页敲的键不渗入便签（reset 生效）；无 pageerror。
- **A/B 无回归**：用 `git worktree` 在 HEAD 另建基线构建，同一 `regress_det.js` 跑全部 App ——
  20 个既有 App 的**帧 CRC 全部逐字节相同**（`svm` 的 `refreshΔ=0` 在基线同样存在，属既有现象）。
  `notepad` 为新 App，`80→80`、48 次刷新、CRC `83d146cf`。
- 早前一次全量 det 里 sky3 的刷新数差异经单独重跑证实为机器负载噪声（两次运行 CRC 一致）。

## 7. 已知取舍

- **浏览器里 Backspace = 返回键**：`webgui/index.html` 的 `KEY_MAP.Backspace = 27`（对齐
  ncurses 的"返回"惯例）是既有行为，本轮**未改动** GUI 语义。因此浏览器里删字请用 **Delete**
  （ASCII 127，notepad 两者都接受）；终端/掌机上 ASCII 8 直接生效。
- 多指/组合键：本库只有单键边沿与电平，Shift+字母等组合键需 App 自行看 `is_down(shift)`。
- ESP32 的键盘 HAL（`hal_key_m5esp.cpp` / `hal_key_m5kb.cpp` / `hal_key_mp135.c`）尚未接
  `nano_key_events_push`；接上只需在按键采样处加一行投递并在 init 里声明来源（本轮未验证
  其构建，故未动）。同样，`infer/Makefile` 的 tty/web 目标已加源文件，但本机缺 `libasound2-dev`
  未做 tty 链接验证。

------------------------------------

# 2026-10-30

sky3.scm（大气散射天空 III）性能剖析与优化：WASM-GUI 目标下一次完整天空渲染（48 个采样行）从
**7.2 s 降到 2.3~2.9 s**（约 2.1~3.1×，随宿主帧节奏而异），画面**逐像素相同**。

## 量测方法与结论（探针不入库）

- 无头 Chromium（playwright `chromium-1248`）+ 真 wasm 产物；页面内 rAF 钩子逐帧记录
  `(performance.now, Gfx.refresh 的 epoch, VM 指令数)`，一帧天空 = 48 次采样行推屏；
- **解释器吞吐（wasm）≈ 18.7 M 条 VM 指令/秒，且与指令种类无关**（紧循环微基准
  `(set! s (+ s (* a b)))` 与 sky3 同值）。故一帧墙钟 ≈ 该帧总指令数 / 18.7 M 秒，
  优化只剩「少跑指令」与「把每帧时间片用满」两条路；
- 单帧成本归因（`tools/e2e/perf` 插桩 + 删块消融，1810 个圆内采样点）：散射 63.2 M 条指令，
  其中二阶多次散射 41.7 M（66%，内层 9 方向 × V2 的 `march` 独占 30.5 M）、一阶 `march` 16.8 M
  （32%），双线性重建/字幕 ≈ 6 M。`Math.exp/sqrt/pow` 共 150 万次调用只对应 2.4% 指令
  ——**热点是解释器的每指令开销，不是 libm**。

## 改动（`infer/app/sky3.scm`，+166/−55）

1. **帧预算用满（最大项）**：旧版每 `step` 只算 5 个采样点（≈13 万指令），而每帧时间片是
   312500 条（`MB_TIMESLICE_TICKS`），且 16 ms interval 与 16.7 ms vsync 拍频使**约 57% 的帧
   一条指令都没跑**。改为 `SAMPLES_PER_STEP=4096`（一步吃满一个时间片，切片责任交回解释器抢占）；
2. **指令削减（−16%，逐像素相同）**：`rsi` 射线求交全部内联（复用 `march` 已算的 |w|²、
   省掉 b/2 与 b²−4c 的 4 次乘；原每采样点 92 次闭包调用）；`Math.pow(b,1.5)`→`b·√b`；
   `dom_g` 与 9 个环境方向的相位项/SUNF·β 系数上提为帧级常量（`frame_setup`，原每采样点算 18 次）；
   相位项公共子式只算一次。

验证：宿主回归台（冻结时钟 `NANO_VNOW=1792461600000`）A/B 逐像素比对 76800 像素全等、
最终帧 CRC 一致（`426cbd48`）；`tools/e2e/host/run_host_regress.sh` 全 21 个 App 无回归；
wasm 侧近实时渲染、`advance()` 重绘路径（临时把 `RE_RENDER_DEG` 置 0 强制每 100 ms 重绘）
连续 10 帧无错误。

## 未采纳 / 后续

- **镜像对称剪枝被否**：天空关于「天顶-太阳」竖直平面镜像对称，但该平面一般不与 5×5 采样格
  对齐，只能旋转采样格再做旋转双线性重建，收益 2× 却要改重建与圆周抗锯齿——复杂度不划算；
- **画质旋钮（默认全关，已实测）**：`V2 2→1` 散射 35.7 M 指令、圆内色差 ≤9/255；一阶
  `nV 36→24` 47.4 M、≤33/255；`SS 5→8` 采样点 1810→708、21.0 M 指令、≤49/255（地平线带更明显）；
- **解释器侧下一个杠杆（需改 `vendor/Animac`）**：指令分布 `load_slot` 33%、`mul` 13%、
  `push` 11%、`add` 9%、`set_slot` 8%、`store_slot` 6%；`(set! x (* a b))` 现需 4 条指令。
  在已有 frame-rewrite 后置遍历里加 peephole（`[load_slot A][load_slot B][mul]` → 一条
  `mul_slots(A,B)`，PC+3 且保证无跳转目标落在被跳过位置）可再省 30%~40% 指令，
  对所有灵机 App 普适；验证路径 `vendor/Animac/testall.sh` + `tools/e2e/host` 呈现序列 A/B。

## 量测注意事项

- 绝对墙钟受宿主帧节奏影响很大：同源不同构建（期间 `ui_scheme_app.c` 等文件被并发修改）实测
  「每帧只服务一次 interval」时为 7.2 s → 2.3 s，而「每帧服务多次」时两者都 ≈0.8 s（vsync 下限，
  48 行刷新 × 16.7 ms）。指令数、逐像素比对等确定性指标不受此影响，跨构建比较请以它们为准；
- `page.clock.setFixedTime` 会冻结 `Date.now()`，而解释器定时器走 `clock_gettime` → sky3 的
  interval 永不触发，故固定太阳位置须用 `page.clock.install` + `runFor` 步进（`regress_det.js` 同法）。

## 2026-10-30（二）sky3 增加「时刻快退/快进」触屏按钮

需求：屏幕左右两端各加一个触屏按钮（−/+10 分钟），点按后即不再保持实时刷新，用于查看其他时刻的天象。

实现（`infer/app/sky3.scm`）：

- **时间偏移只在「取时」一处生效**：新增 `now_shifted`，把 `Time.utc_days/utc_sod` 加 `t_shift_min*60`
  秒并做日/秒进位，写入 `cur_days/cur_sod`；`sun_from_ephem`、`sun_moved_enough`、`now_string`
  统一改用它。物理模型、渲染管线、采样网格一行未动。
- **手动/实时双模式**：`manual` 标志。点按按钮 → `t_shift_min ∓10`、`manual=1` → `advance()`
  重取时刻、重装帧级常量、清屏、重画按钮、从头渐进渲染；手动模式下**跳过**太阳移动判据
  （不再自动重绘），右下时钟也冻结（`clock_tick` 直接返回）；长按任一按钮 0.7 s → 偏移归零、
  回到实时跟随。
- **按钮布置**：`BTN_W=24 / BTN_H=52`，贴左右边缘、纵向居中（y=94~146）。这一带在天空圆
  （内切圆 r=120）之外恒黑，且逐行双线性重建只写圆内像素，故按钮画一次不会被覆盖；同时避开
  左上 80×60 的触屏宫格（esc 格），不会误触系统手势。按钮为深底直角框 + 三角箭头 + "-10"/"+10"
  文本；手动模式下边框转金色。无触屏宿主（`Touch.is_available` 为假）不画按钮。
- **交互反馈不靠帧尾**：手动状态提示（"手动±N分" / "长按回实时"）画在圆外左上黑区
  （x<96 且 y<45 永不参与重建），起帧瞬间即可见；点按在**每个 step 开头**与**每个采样行完成时**
  各采一次触屏（`Touch.is_pressed/take_press/x/y`），长帧渲染途中点按也能在一行内（~50 ms）响应，
  响应即重起帧。

验证（playwright + 真 wasm，临时把 `RE_RENDER_DEG` 置 0 以便观察）：

- 点右键：HUD 由「太阳高度-21.3° 方位93.1° / 04:41:24」变为「-19.0° / 94.5° / 04:51:26」，
  左列出现「手动+10分」，按钮边框转金；再点一次 → 手动+20分（05:01:30）；点左键 → 手动（偏移归零
  但仍是手动）；长按 900 ms → 提示消失、回到实时（04:42:32、-21.1°）。
- **实时 vs 手动**：`RE_RENDER_DEG=0` 时实时模式 6 s 内刷新 112 次，点一次右键进入手动后
  同样 6 s 内刷新 **0** 次——"点击后即不再保持实时刷新"落实。
- 渲染途中（进 App 后 0.4 s）点右键：epoch 在 400 ms 内 +13 并重起帧，最终画面为手动+10分。
- 时间偏移进位：+10/+1300/−100/−40000 分钟的日/秒进位与回绕均正确（差量精确为 600/78000/−6000/−2400000 秒）。
- 宿主回归台：有触屏时与改造前相比**仅**两个按钮矩形（各 1248 像素）不同、天空 0 差；无触屏
  （`HAL_TOUCH_AVAIL=0`）时与改造前**逐像素全等**；`run_host_regress.sh` 全 22 个 App 无回归。

备注：本次只改 Scheme 业务文件，wasm 产物无需重建（资源经 HTTP 挂载，App 源码在 MEMFS 里按需读取）。

------------------------------------

# 2026-10-30（宿主模态文本输入/输出本地库 `(native Ui)`）

给「灵机应用程序」补上**文本输入输出**：App 里写一行 `(Ui.input "请输入名字")`，就会弹出
宿主（鹦鹉笼同款）的文本输入控件——全键盘拼音 + 16 键九宫格输入法 + 触屏软键盘 + 候选字
翻页 + 光标/滚动——用户提交后把文本作为返回值交给 Scheme。概念上等价于 J2ME 的 TextBox /
浏览器 `prompt()`，但「阻塞」是**挂起 Scheme 进程**而非死等界面（宿主事件循环照常跑）。
编译目标 **Nano-WASM-GUI**，宿主回归台 + 三个无头浏览器端到端验证。

## 1. 新增文件与 API

| 文件 | 作用 |
|---|---|
| `infer/am_native_Ui.c/.h` | 本地库 `Ui` + 宿主服务接口（发起/查询/提交） |
| `infer/app/uinput.scm` | 示范兼验收 App：两问 + 一个消息框 + 落盘汇报，可循环重来 |
| `tools/e2e/native/ui_probe.scm`、`test_ui_prompt.py` | 宿主台「阻塞/唤醒契约」测试（11 项断言） |
| `tools/e2e/browser/regress_uiprompt.js` | 真浏览器端到端（20 项断言，Chromium/chrome-headless-shell/Firefox 全过） |

Scheme 侧（`(native Ui)`）：

```
(Ui.input 标题)              : String | #null   —— 阻塞，提交返回文本；取消返回 #null
(Ui.input_with 标题 初值)    : String | #null   —— 同上，输入框预置初值
(Ui.message 正文)            : #undefined       —— 阻塞到用户关闭（只读文本，可拖动滚动）
(Ui.message_titled 标题 正文) : #undefined
```

没有可选参数（本地库固定元数），故「带初值」是独立过程而不是重载。
取消 = **长按 A/Esc** 或点按页眉右侧「返回」热区（短按 Esc 是控件自己的退格键，平台惯例）。

## 2. 阻塞机制（复用上游的 N7 契约，不加新机制）

- native 函数里把**进程**置 `AM_PROCESS_STATE_BLOCKED` 后直接 `return 0`——不压值、不 step
  （上游本地库契约 N7 的「阻塞型函数」形态，与 `System.read` 队列等待同源）。虚拟机随即把该
  进程移出调度队列；宿主的事件循环、渲染、其它进程照常运作。
- 宿主提交时 `am_native_ui_resolve(rt, text)` 照抄 `runtime_queue_wake_process` 的四步：
  `push_operand` → `am_process_step`（越过 callnative 指令）→ 置 READY → 重新入队 pid。
  进程于是在**同一条指令**处继续，Scheme 侧看到的就是一个返回文本的函数。
- 附带语义（有意）：BLOCKED 期间该进程的 `set_interval` 定时器暂停（上游既定行为），
  App 的动画在模态框期间自然停住，不需要 App 自己判断。
- 同时只允许一个请求：等待期间第二个 `(Ui.input ...)` 立即返回 `#null`（软失败）。

## 3. 宿主侧（ui_scheme_app.c 的模态框服务）

- 控件**直接借用**宿主现成的：`global_state->w_input_main`（`ui_widget_input_*`，
  与鹦鹉笼的输入框同一个控件）与 `global_state->w_textarea_main`（消息框）。
  于是输入法、软键盘、候选字、滚动、粘滞修饰键全部免费，Scheme 侧零输入法代码。
- 提交/取消的区分：`ui_widget_input_event_handler(... prev, cur, next)` 的上一/下一焦点态
  传两个哨兵值（`-1001` 取消 / `-1002` 提交），一套控件代码即可表达两种退出方向。
- `ui_app.c` 的触屏 4x4 宫格映射在模态框期间被抑制（新增
  `ui_scheme_app_hosts_input_widget()` 查询），触屏归控件所有；否则点按/拖动会不断产生
  软按键、右上角还会变成 Esc。
- 关框时不改 App 状态：进程唤醒后由 App 自己下一拍重绘接管画面。

### 三个必须处理的边沿陷阱（都是实测踩出来的）

1. **打开模态框那次按键的松开沿**：按键边沿由生产端逐事件投递、消费者每帧只取一个，
   且**只投递下降沿**（-1/-2）。App 通常在被取走按下沿的那一帧里发起 `(Ui.input ...)`，
   于是这次按键迟到的 -1 落进刚打开的模态框 → 控件当成「短按 Enter 提交」，表现为
   **输入框一出现就自己交了空串**（消息框则是"一闪而过"）。
   解法：打开时向 Key 库问「最近一次被取走的按下沿是哪个键、什么时刻」
   （新增 `am_native_Key_last_press()`，用 VM 时钟 `am_runtime_now_ms`，与宿主时间线一致）；
   若就在刚才（≤200ms），则剔除**该键**接下来的第一条 -1/-2，且只在一个 300ms 窗口内、
   一旦出现别的按键或触屏动作立即解除。
2. **长按连发**：事件层的长按（-2）在按住期间会持续重复（按键连发机制）。模态框用长按 Esc
   取消，若不拦截，同一次长按会连续取消多个模态框、最后把 App 也退出。
   解法：模态框消费一个 -2 后置位抑制标志，其后相同长按的重复事件丢弃，直到出现一次
   **-1**（说明这次长按早已结束、用户又按了一下）才解除。**注意：不能用"上升沿(1)"解除**
   ——生产端从不投递上升沿。
3. **按键渗漏**：模态框期间的按键属于控件，不能进 `(native Key)` 的旁路队列，否则用户敲的
   每次 Enter/退格都会在 App 恢复后被它当成新按键读走（症状：关掉消息框后 App 又"自己按了
   一次 Enter"）。解法：模态框打开时与该期间的每帧都 `am_native_Key_reset()`。

## 4. 验证

- **宿主台阻塞契约**（`python3 tools/e2e/native/test_ui_prompt.py`，**11/11 PASS**）：
  提交路径（打点前后都执行到、`input_with` 参数顺序、message 关闭后继续）、取消路径
  （`#null`）、无人应答（全程 `blocked=57` 帧、不崩不自行恢复）。
  harness 新增脚本化应答 `HAL_UI_AT="帧:动作:文本"` 与 `RESULT ... blocked=N` 计数。
- **浏览器端到端**（`regress_uiprompt.js`，Chromium / chrome-headless-shell / Firefox
  **各 20/20 PASS**）：说明画面自绘 → 按 Enter 进输入框后 **epoch 停止增长**（证明进程真的
  挂起、App 不再自绘）→ 十六键九宫格组字 `ni` + 选字 + 提交 → `Ui.input_with` 直接回车接受
  初值 → 日志 `data/ui_demo.txt` = `name=你\nnote=你`（中文经输入法进 Scheme 字符串再落盘）
  → 消息框同样挂起、任意键关闭 → 长按 Esc 取消得到 `#null` → 点按页眉返回热区取消（且不会
  退出 App）→ 非模态态长按 Esc 仍返回列表。
- **回归**：宿主全 App（22 个，含 `uinput`）`err=none`；`regress_key_fs`（便签）17/17 仍过；
  浏览器确定性回归 22 个 App 全部 `launch=80 → end=80`，未改动 App 的帧 CRC 与基线逐字节
  相同。`svm` 的 `refreshΔ=0` 属既有现象；`verlet`/`xray` 的 CRC 变化经复跑确认是**App 列表
  下标后移导致启动时刻偏移**（确定性回归按 16ms/帧步进虚拟时钟，时间相关画面因此不同），
  非行为回归。sky3 当时另有并行改动，未纳入比对。

## 5. 已知取舍 / 后续

- 消息框的滚动复用 `ui_widget_textarea_touch_handler`；键盘翻行未接（点按/拖动可滚）。
- 一次只服务一个模态请求，且模态框只服务**当前 App**；App 退出时 `am_native_ui_abort()`
  丢弃挂起请求（进程随后被销毁，不留悬空 PID）。
- 其他平台（tty/web/android）的宿主侧接线未做：它们同样只差在 `ui_scheme_app.c` 里加同样
  的模态框服务（本模块与平台无关，控件是共用的 UI 组件），但本机未逐个构建验证。

------------------------------------

# 2026-10-30（三）二维码 App（QR Code 编码器 + 字符画导出）

给「灵机应用程序」加一个**二维码生成器**：用宿主模态输入框（`(native Ui)`）取一段文本，
在 Scheme 里完成 ISO/IEC 18004 编码并画到屏幕上；点「保存」把二维码导成**字符画纯文本**
（黑点 `●` = U+25CF，白点 = 全角空格 U+3000，含 4 模块静区），可直接贴进聊天窗口/终端。
编译目标 **Nano-WASM-GUI**，宿主无头回归台 + 三个无头浏览器端到端验证。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `infer/app/lib/qr_core.scm` | 纯逻辑编码器（999 行）：版本 1..20、字节模式、L/M/Q/H、8 掩码择优、格式/版本信息 BCH。只依赖 Math/String，可在命令行解释器里单独回归 |
| `infer/app/qrcode.scm` | App 本体（440 行）：布局/绘制、Ui 输入、触屏按钮、字符画导出（Fs.write） |
| `tools/e2e/native/test_qrcode.py` + 生成式 `qr_probe.scm` | 宿主台一键测试（37 项断言）：编码器契约（容量边界/结构/BCH）+ App 端到端（输入→生成→保存） |
| `tools/e2e/browser/regress_qrcode.js` | 浏览器端回归（21 项断言，三引擎全过；含"提交→上屏"计时） |

接线零改动（`app/*.scm` 自动进入「灵机应用程序」列表、`build_wasm_gui.sh` 自动收录 app 与
app/lib 到 manifest 的 p1）。**另动了三处周边**：`ui_scheme_app.c` 的 `SCHEME_APP_MAX`
24→32（已有 23 个 App，只剩 1 个空位）；`tools/e2e/host/harness.c` 的脚本化输入缓冲
160B→640B、stdout 缓冲 16KB→128KB（大文本/大矩阵探针用）；`test_touch_imu.py` 的构建
清单补齐 `hal_fs_linux.c` + `nano_key_events.c` + Key/Fs/Ui 三个 native（harness.c 早已
注册它们，缺源文件导致 test_touch_imu/test_tetris 链接失败——本次顺带修好）。

## 2. 编码器（qr_core.scm）

- **范围**：字节模式（UTF-8），版本 1..20（21×21..97×97），L/M/Q/H；字节容量
  L 17..858 / M 14..666 / Q 11..482 / H 7..382（自动选最小版本，超容量返回 1）。
- **流程**：模式 0100 + 字符计数（v<10 为 8 位、v≥10 为 16 位）+ 数据 + 终止符 + 字节对齐
  + `0xEC/0x11` 填充 → GF(256)（本原多项式 0x11D）上算 RS 纠错 → 分块 + 交错 → 功能图形
  （定位/分隔/校正/时序/暗模块/预留格式与版本区）→ 蛇形放置数据 → 8 掩码罚分择优 →
  写格式信息（15 位 BCH，含 0x5412 掩码）与版本信息（18 位 BCH，v≥7）。
- **标准表**：RS 分块表 80 项（20 版本 × 4 等级）、校正图形坐标 20 项、格式信息 32 项、
  版本信息 14 项，全部由 Python `qrcode` 库的常量生成后硬编码（`RS_TABLE` 的每项是
  `(每块纠错码字数 块数1 数据码字数1 [块数2 数据码字数2])`——**注意原始表是
  (count,total,data) 三元组，纠错码字数要算 `total-data`，第一版抄错导致 RS 全错**）。
- **掩码罚分**：实现标准四条规则里的 **1（行列连续同色 ≥5）、2（2×2 同色块）、4（深色占比
  偏离 50%）**，略去**规则 3（1:1:3:1:1 伪定位图形）**：它几乎只被定位/校正图形触发
  （8 种掩码下逐格相同），对选择影响小、代价最高（每个位置看 11 格 ≈ 再扫一遍全矩阵）。
  与参考实现（Python qrcode）对比：96 个用例中 38 个选了不同掩码，但**同一掩码下矩阵
  逐模块相同**，且对完整四规则评分平均只差 +25 分（约 2.5%），全部用例 zxing 解码通过。

### 性能：解释器里每次 native 调用都是钱

`get_item`/`push` 每次约百条 VM 指令，掩码择优要做 8×(n² 放置 + 3 条规则)，因此：

1. **模块级临时量与递增下标**：内层循环用 `idx`/`row_base` 计数器代替 `(mi r c)`
   （`mi` 是闭包调用，比一次乘加贵得多）；
2. **掩码平铺表**：8 种掩码周期都不超过 12（掩码 4 为 行:4/列:6），预生成 8×12×12 平铺图，
   逐格查表代替 `mod/floor` 运算；列方向用递增计数器循环取模；
3. **功能图形与数据合并在同一张表**：`q_mods` 里功能图形存 0/1、数据区存 `2+原始位`，
   `apply_mask` 一次 `get_item` 就能区分两者（省掉原来每格多读一次 `q_func`）；
4. **规则 2 与规则 1（行）合成一遍**：本格取一次、上邻取一次，2×2 判据用"本格=左邻=上邻=
   左上邻"的递推（`upprev` 携带），每格 2 次取指；规则 1（列）单独一遍 1 次取指。

优化前后（宿主台 -O2，同一用例）：v1 48→25ms、v4 122→72ms、v6 190→105ms、
v10 376→215ms、v20 1153→620ms（约 2×）。wasm 实测纯编码：**v1 43ms、v3 67ms、v6 128ms、
v11 306ms、v17 573ms**（与宿主 -O2 同量级）；提交→二维码上屏约 **180~200ms**（v1，含
模态框收尾与两次推帧）。

## 3. App（qrcode.scm）

- **布局**（320×240）：顶栏 16px（标题 + `v{版本} · {纠错}`）；左上 232×208 为二维码区
  （白底 + **4 模块静区**，按行把连续黑格合并成一次 `fill_rect`）；右侧 88px 面板
  （字节/上限/掩码 + 输入/保存/纠错 三个按钮 + 两行快捷键提示）；底栏 16px 状态条
  （保存路径与字节数、错误提示）。
- **缩放**：`scale = max(1, floor(min(232, 208) / (n+8)))`——v1 7px、v6 4px、v10 3px、
  v20 2px；白底按可用区域居中。
- **字符画**：每行 = 4 格静区 + n 个模块 + 4 格静区，共 n+8 行；`●` 与全角空格等宽，
  文件本身是一张带静区的合法二维码（zxing 可直接扫码文件内容）。
- **交互**：`输入`/Enter → `(Ui.input ...)`；`保存` → 生成字符画并 `(Fs.write "data/qrcode.txt")`
  （相对业务资源根：桌面 `/home/bd4sur/linglong/data/`、浏览器 MEMFS + IndexedDB、
  掌机 SD 卡根）；`纠错` 循环 L→M→Q→H→L 并立即重新生成；无码时点画面任意处等同「输入」。
- **一致性**：新纠错等级放不下当前文本时（rc=1）**回退到原等级**并在状态条说明
  （"文本放不下（纠错 Q），已保持原等级"），否则面板显示的等级与屏幕上那张码会脱节。

## 4. 踩坑清单（都是实测出来的）

1. **`set!` 的目标必须先在模块级 `define`**：只写 `(set! q_z_i 0)` 会得到
   `[Runtime] load: 变量 q_z_i 未定义`。本模块在文件中部集中 `(define q_xxx 0)` 声明了
   114 个临时量；同一 scope 里 `define` 重名会直接
   `[Parser Error] failed to add defines to scope`（探针脚本里"循环体里 define 一遍同名
   变量"就撞上了这个）。
2. **三参数 `-` 会毁掉操作数栈**：`(- q_n 1 q_rf_k)`（应为 `(- (- q_n 1) q_rf_k)`）多余
   参数残留在栈上，症状是几层调用之后莫名 `[Runtime] 指令执行异常`（无消息、无位置）。
   本次写了个"二元/一元算符元数检查"脚本对所有 .scm 过了一遍才揪出来。
3. **位运算只接受 uint 表示**：`Math.floor`/`/`/`mod` 返回浮点，喂给 `Math.bit_xor` 会
   报参数类型不符。解法是 `(u8 v)`——一张 256 个 uint 字面量的表，用 `get_item` 把浮点
   字节值"洗"成 uint；GF 表初始化则全程只用 `(* x 2)`+`bit_xor(…,285)` 保持 uint。
4. **数据放置跳过第 6 列时把奇偶搞坏了**：参考实现是 `for col in range(n-1,0,-2)` 里
   `if col<=6: col-=1`（每次迭代重新取 range 值，所以只有一次偏移）；照抄成 `while` 后
   变成"每轮都减 1"，最后一组成了 (2,1)，**第 0 列整列漏放**（表现为矩阵里残留 `-1`，
   掩码后变成 2）。判据：8 个 `-1`，正好落在 (9..12, 0) 与 (9..12, 3)。
5. **校正图形坐标表下标**：`(get_item ALIGN_TABLE (- 0 (- ver 1)))` 是"取负"，索引变成
   -1 → `#undefined` → 随后报错；应为 `(- ver 1)`。
6. **浏览器里的软键盘会盖掉实体键**：模态框打开后若再呼出触屏软键盘，实体 Enter 不再
   提交（既有行为：软键盘 claimed 时实体键被其电平覆盖，见 2026-08-28 条目）。因此
   e2e 用软键盘右上角的 **ENT 键**提交；软键盘未打开时实体 Enter 正常。
7. **长编码期间的点按会被丢**：(native Touch) 是电平差分锁存，进程忙（v17 编码要跨几十
   帧）时没人采样，按下沿就看不见——测试里点「保存」必须把按住时间拉长到 80 帧以上。
   App 侧的对策只是"先上屏 `正在生成二维码…`"再算，让用户知道在忙。
8. **宿主回归台脚本化输入的上限**：`HAL_UI_AT` 的文本缓冲原来 160B（长文本被静默截断，
   表现为"200 字节的文本只解出 159 个字符"）、`g_out` 只有 16KB（大矩阵探针输出被截）。
   已分别扩到 640B / 128KB。

## 5. 验证

- **编码器契约**（`python3 tools/e2e/native/test_qrcode.py`，37 项全 PASS）：
  容量边界（17/18、14/15、7/8、213/214、718/719、858/859）版本选择逐一正确；每个矩阵
  `n = 4v+17`；三个定位图形/时序图形/固定暗模块/格式信息 15 位 BCH/版本信息 18 位 BCH
  的**位置与取值**用 Python 侧独立实现逐位核对；zxing-cpp 解码还原原文（未装则 SKIP）。
- **App 端到端**（同一脚本）：脚本化触屏点「输入」→ 宿主模态框提交文本 → 编码上屏（整帧
  PPM 用 zxing 解码回原文）→ 点「保存」→ 字符画落盘，校验正方形/仅两种字符/四周静区/
  黑点数，并用 zxing 解码字符画回原文；三种文本覆盖 11B(ASCII)、18B(中文)、200B(v10)；
  另验：纠错升档失败时保留原等级的码（400B 用例，解码结果的 EC 等级确为 Q）、取消输入
  （`#null`）不产生文件。
- **浏览器 e2e**（`node tools/e2e/browser/regress_qrcode.js <URL> <engine> <outdir>`）：
  Chromium / chrome-headless-shell / Firefox **各 21/21 PASS**。链路是真机链路：Pointer
  Events → 宿主模态框 → 软键盘「键盘/CTRL/SP（切英文模式）/逐键/ENT」→ `(Ui.input)`
  → `qr_core` → `Gfx`；判据是帧缓冲（白底方形尺寸、四角静区纯白、定位图形中心为黑、
  字节/上限像素、保存前后画面变化），并计时"提交→上屏"约 **181~201ms**（v1）。
- **回归**：宿主全 App（23 个）`err=none`；`regress_det.js` 全 App `launch=80 → end=80`
  （`qrcode`/`svm` 的 `refreshΔ=0` 是静态 App 的既有现象）；`regress_uiprompt`（20/20）、
  `test_key_fs`（44/44）、`test_ui_prompt`（11/11）、`test_touch_imu`（36/36，本次修好
  链接）、`test_tetris`（21/21）全过。
- **额外扫测**（临时脚本，不入库）：96 个用例与 Python `qrcode` 库**逐模块比对**（同掩码
  下 0 差异）；>300 个用例（ASCII/中文/随机/满载/边界）经 zxing-cpp 全部解码还原。

## 6. 后续可做

- **规则 3 罚分**：想更贴近参考实现可在行方向加一遍（11 格窗口，用 11 个移位变量而非取指），
  App 内实测掩码选择与 Python 的一致率可从 ~60% 提到 ~100%，代价约 +30% 掩码搜索时间。
- **大文本体验**：v11 以上编码 300~600ms，期间 App 不响应输入；若要更跟手，可把"8 掩码
  择优"改成两阶段（先用便宜规则筛 2 个候选，再全规则评分），或让 App 分帧编码（把编码
  切成若干步，每步经 `set_interval` 推进并在状态条显示进度）。
- 微调：字符画可选不带静区、按版本命名文件（`data/qr-v6.txt`）、长按保存连出 PNG（渲染层
  没有编码器，需新本地库）。

------------------------------------

# 2026-10-30（四）触屏计算器 App（复用上游中缀表达式解析器 + 计算历史）

给「灵机应用程序」加一个**触屏计算器**：6×5 矩形虚拟键盘输入中缀表达式，显示区上方为
最近 5 次计算的**历史记录**（先进先出、仅内存、不落盘、不持久化），最下一整行是**当前
计算**（高亮条 + 亮字），输入过程中实时预览、按 `=` 定稿；支持 `+ - * / % ^`、括号、一元
负号、自动补右括号，函数 `sin cos tan log exp sqrt`，`π`/`Ans` 常量键，`AC`/`DEL`
（`DEL` 可长按连删）。编译目标 **Nano-WASM-GUI**，验证走宿主无头回归台 + 三个无头浏览器。

**解析器不新写**：`app/lib/infix.scm` 的 `lexer`/`parser`/`type_of_token`/`show_*` 与上游
`Animac/test/calculator.scm`（2017, 2025-07-11）**逐字相同**（脚本逐函数比对确认），本 App
不实现任何表达式语法；因此优先级/结合性/词法规则与上游用例完全一致（上游那条长表达式仍得 320）。

## 1. 新增文件

| 文件 | 作用 |
|---|---|
| `infer/app/lib/infix.scm` | 中缀解析器（上游逐字 + 下述三处增补/加固）：`lexer` `parser` 原样，新增 `validate`（词元级前置校验）、`calc`/`calc_reason`（一站式求值 + 中文原因）、`FUNC_NAMES`/`OP_NAMES` 白名单 |
| `infer/app/lib/list.scm` | 上游 `test/list.scm` 逐字副本（解析器只用其中的 `List.append`；import 路径改为 `app/lib/list.scm`） |
| `infer/app/calculator.scm` | App 本体（约 560 行）：布局/键位、历史记录、触屏状态机、动作与显示逻辑 |
| `tools/e2e/native/infix_probe.scm` | 解析器语义探针（47 项断言：上游长表达式 320、优先级/一元负号/函数、残缺输入与值域失败） |
| `tools/e2e/browser/regress_calculator.js` | 浏览器端回归（24 项断言；判据为帧缓冲**逐行墨迹统计**，不做 OCR） |

接线零改动（`app/*.scm` 自动进入列表、`build_wasm_gui.sh` 自动把 app 与 app/lib 收进
manifest 的 p1）；仅把 `ui_scheme_app.c` 里 `SCHEME_APP_MAX` 的注释改为「当前 24 个；余量留 8 个」，
并放宽 `tools/e2e/host/harness.c` 的触屏脚本上限（`HAL_MAX_STEPS` 32→512、解析缓冲 512B→8KB：
计算器一次交互连打二十几个键，32 段触屏事件根本不够，症状是「打到第 16 个键之后完全没响应」）。

## 2. 界面形态（320x240）

```
   0 ..  98   显示区（6 行 x 14px）：
                行 0..4  计算历史（灰字，最旧在上、最新紧贴当前行）
                行 5     当前计算（高亮条 30,45,64 + 上下 1px 描边）
   100 .. 240  键区（6 列 x 5 行，矩形键 53x28，无圆角）
```

- **矩形键**：`Gfx.fill_rect` + 1px 间隙，按下时底色统一提亮 +52（无圆角、无斜边）。
- **文本而非数码管**：进展与结果一律 12px 字模。当前行分两段——表达式（近白）+ 尾段
  （` = 值` 绿 / 错误原因红 / 初始提示灰），按**像素宽度**右对齐；表达式过长时从左侧裁掉
  保留尾部，确保右侧的值/原因始终可见。
- **字体宽度实测**（`tools/e2e` 字体探针）：该 12px 字模 ASCII/希腊/数学符号步进 7px，
  **CJK 与全角步进 12px**，故对齐/裁剪必须按像素宽度算（`char_w`：码点 ≥ U+2E80 记 12），
  早期按「7px × 字符数」算导致中文提示被截断、右对齐偏移。
- **历史（FIFO，仅内存）**：5 个槽位 `h0..h4`（`hist_push` 整体上移，最旧被挤掉），条目文本为
  「表达式 = 结果」（`*`→`×`、`/`→`÷` 只为显示好看，表达式本体保持 ASCII 供解析器使用）。
  「当前计算」与历史不重复：`=` 时结果**留在高亮行**，直到下一次按键把这一条 `close_pending`
  推进历史；`AC` 也会把当前条推入历史（历史不因 AC 清空，退出 App 重进则从空开始——
  不落盘、不持久化）。失败的算式（`5÷0`）只在高亮行显示红字原因，**不入历史**。
- **键位**：`AC DEL ( ) % ÷` / `7 8 9 Ans π ×` / `4 5 6 sin cos -` / `1 2 3 tan log +` /
  `0 . √ exp ^ =`。动作由键面文字派生：`÷→"/"`、`×→"*"`、
  `√→"sqrt("`、`sin/cos/tan/log/exp→同名 + "("`、`π→"3.14159265358979"`。
- **键区整体位于 y>=100**，避开宿主 4x4 宫格的 esc 格（x≥240 且 y<60，见 `ui_app.c` 的
  `ui_app_map_touch_to_grid16_key`）——该格内的点按会被宿主当成返回键直接退出 App。
  显示区（y<98）里的右上角仍落在该 esc 格内：那里只是只读的历史文本，点按等于「返回列表」，
  不会误触算键（与 tetris 等 App 同一约定）。
- **触屏状态机**：`Touch.is_pressed` 电平差分自建边沿（按下记录命中的键，松手且仍在同一键才
  触发；`am_native_Touch` 松手后保留最后坐标，故松开时仍能算出键位）；`DEL` 长按 450ms 后每
  90ms 连发退格，松手不再补一次；重绘只在状态变化时发生，节拍 `set_interval 33ms`。

## 3. 为什么必须加固 eval / 加 validate（本 App 与上游用例的关键差异）

上游是**批处理用例**（输入固定且合法、`run` 甚至没被调用）；计算器则是**任意按键序列**。
灵机没有异常机制，一旦虚拟机报错：**出错进程立即终止**，且宿主 `ui_scheme_app.c` 会把
错误文字以红字叠加到画面（`err_buf` 一次性显示、无法从 Scheme 侧清除）。实测两类必崩输入：

- **解析越界**：`parse_primary`/`parse_literal` 里的 `is_number`/`is_identifier` 没有
  `index < length` 保护（`match` 有），于是 `"1+"`、`"sqrt"`、`"("` 这类「期待操作数却已到
  末尾」的输入会对 `#undefined` 再取一次下标 → 运行时错误。
- **求值失败**：`(/ x 0)` 直接报「除零」；`Math.log(-1)` 返回 `#null`（NaN 约定）后参与算术
  报「参数类型不符」。而 `mod`/`pow` 恒返回浮点（NaN 也是数字），所以 `5%0`、`sqrt(-1)` 本身不崩。

处置：

1. **`validate(tokens)`**：按文法做一遍等价结构检查（状态机 `want_operand` + 括号深度 +
   函数名白名单 + 「函数名后必须紧跟 `(` 且实参非空」），只在通过后才调用 `parser`；失败
   返回 `#f` 并把中文原因写入 `v_reason`（`表达式不完整`／`括号不匹配`／`两个数字之间缺少
   运算符`／`缺少左操作数`／`函数调用缺少参数`／`不支持的函数：foo`）。**parser 的崩溃面被
   完全消除**，而未改动 parser 一行。
2. **`eval` 加固**（唯一被改写的上游函数，运算表逐条保留，含一元/二元 `-` 的区分）：
   先 `(list? ast)` 保护 `type_of_token`；递归求出 `lv`/`rv` 后先判 `number?`，任一非数字即
   返回符号 `'err`（天然向上传播）；`/` 除数为 0 → `'err`；`Math.*` 结果统一过
   `num_or_err`（`#null` → `'err`）。
3. **`calc(expr)`**：`lexer → validate → parser → eval`，成功返回数值，失败返回 `'syntax`
   或 `'value`，原因经 `calc_reason` 取。界面只做文案，不做任何语法/求值判断。

于是任意按键序列（含 `1+`、`sqrt`、`()`、`5/0`、`log(-1)`、超长串）都只得到一句提示，
App 进程不会被打死——这是本 App 能持续运行的前提。

## 4. 验证

- **解析器探针**（`/tmp/nano_host_regress/run_new tools/e2e/native/infix_probe.scm 0`）：
  **47/47 PASS**。含上游原长表达式 `= 320`；`2^3^2=512`（右结合）、`-2^2=4`（unaried 先结合）、
  `2*-3=-6`、`log(exp(0))`、`+5=5`（词法层并入数字）；残缺输入全部 `'syntax` 且原因正确；
  `5/0`、`1/(2-2)`、`log(-1)` → `'value`；`sqrt(-1)` → NaN（界面层拦截为「= 非实数」）。
- **宿主回归台**（`tools/e2e/host/run_host_regress.sh`，已把 `calculator` 加入默认 App 列表）：
  **24 个 App 全部 `err=none`**，`calculator` 静态页仅 1 次 refresh（预期）。
  逐像素核对（PPM→PNG）：多轮计算后历史 5 条按 FIFO 上移、当前行高亮 `6×7 = 42`、
  `5÷0` 红字原因且不入历史、`123456789×987654321 = 1.21933e+17`、超长表达式从左侧截断。
- **浏览器端回归**（`tools/e2e/browser/regress_calculator.js`，判据是帧缓冲逐行墨迹：
  历史行墨迹数、当前行墨迹数、红字像素数、高亮条底色）：**Chromium / chrome-headless-shell /
  Firefox 各 24/24 PASS**。覆盖：进入列表→启动 App（state 80）、初始历史为空且高亮条底色正确、
  触屏按键后当前行内容变化、`=` 后当前行留在原处（历史不增）、第 2/5/6/7 次计算的历史条数
  （1→4→5→5，验证 FIFO 不增长）、`5÷0=` 红字且不入历史、`AC` 清空当前行但保留历史、
  `123456` 后长按 DEL 1.4s 连删回到提示态、`A(esc)` 返回列表、**二次进入 App 历史为空**
  （不持久化）、全程无 pageerror。

## 5. 踩坑与注意

- **三元算术是未定义行为**：`(+ y (* 2 t) half)` 多传一个参数会把操作数栈搞乱，症状极难归因。
  所有算术一律二参数嵌套（旧版 7 段数码管就是被这个坑害得起初只画出 a/d 段）。
- **`(if x ...)` 里 x 为 `0` 仍是真**（只有 `#f` 是假）：`(if alert ...)` 写成 `alert=0`
  会让提示在每次按键时都刷新，必须写 `(== alert 1)`。
- **块内不 `define`**：热路径局部量一律放模块级（`kr/kg/kb`、`cc`、`cw`、`now_ms`），与
  `touchgravity.scm`/`tetris.scm` 的约定一致。
- **CJK 与 ASCII 步进不同**（12 vs 7）：按字符数算宽度会把中文提示挤出屏幕。
- **wasm32 的浮点只有约 6 位有效数字**：`am_float_t` 在 32 位宿主是 `float`，且
  `am_make_value_of_float` 还要截掉低 5 位尾数（腾出 TPV 类型标签位），故 24 位尾数只剩
  19 位 ≈ 相对误差 1e-6。实测 `π×2` 在 WASM 里是 `6.28317`、64 位桌面宿主是 `6.28319`。
- **上游 lexer 的 `read_number` 不认指数形式**：`%g` 输出的 `1.21933e+17` 无法再被词法分析，
  所以「按运算符接着上次结果续算」与 `Ans` 键都改用「把上次表达式括号化」的方式
  （`(987654321*123456789)+`），既无损又能解析；只有结果为普通十进制时才直接插入结果串。
- **上游 `lexer` 对独立 `"."` 是软失败**（`Unexpected character` 后 `break`，静默截断整串），
  故 `.` 键在末尾非数字时自动补 `0`，且当前数字段已有 `.` 时拒绝重复输入。

## 6. 后续可做

- **历史可滚动/更多条**：现在固定 5 条（显示区高度决定）；若要更多可加"上/下翻页"键，
  但仍受显示区高度限制。
- **角度/弧度模式**：`sin/cos/tan` 目前按弧度（与上游 eval 一致），可加 `DEG` 开关在 App 侧
  把参数改写为 `(x*3.14159265358979/180)`——不必动解析器。
- **`Ui.input` 直接输入表达式**：可加一个"键盘"键走宿主模态输入框（`(native Ui)`），给愿意
  打字的宿主（Nano-Pod/桌面）多一条入口。

------------------------------------

# 2026-10-30（四）灵机 App 退出接口：`App` 本地库 + 全局退出确认框（P1）

把"退出灵机应用程序"从宿主的**统一交互**（无条件拦截 A/ESC → 销毁实例）改成
「封装良好的、各 App 按需调用的退出接口」，并把"离开"这件事拆成可扩展的词汇，
为将来的多任务/后台运行留位。编译目标 **Nano-WASM-GUI**，宿主回归台 + 三个无头浏览器
端到端验证。

## 0. 决策（与需求方确认的六条）

1. **不放 `Ui`**：新建平台库 **`App`**（应用外壳/生命周期）。`Ui` 保持"借用宿主模态控件"
   的定位；`(Ui.confirm ...)` 作为通用确认框补进去。
   库名刻意不用设备名（Linglong）——那是杂物袋，设备能力（振动/电量/网络）将来另开 `Device`。
2. **(App.exit) 阻塞到用户答复**（复用 Ui 的 BLOCKED/唤醒机制，契约 N7）。
3. **触屏不再派生返回键**：`STATE_SCHEME_APP_RUN` 整体退出 4×4 宫格映射（含右上角 esc 格）；
   **硬 A/ESC 保留**，但宿主只置"返回请求"，由 App 消费 → 弹全局确认框（J2ME 挂断键语义）；
   网页工具条新增 **「ESC」按钮**，把硬 Esc 同时送进旁路 `(native Key)` 与 GUI 电平槽。
4. **零快照**：确认框不做帧缓冲快照/恢复，关闭后由 App 自己重绘（`install_with` 登记重绘函数）。
5. **同步补 `(Ui.confirm 标题 正文)`**（阻塞、返回 `#t/#f`），与退出确认框共用同一套绘制/命中。
6. **`(App.hide)` 本期打桩**（宿主按退出处理），接口先立住。

## 1. 新增文件与 Schema 侧 API

| 文件 | 作用 |
|---|---|
| `infer/am_native_App.c/.h` | 本地库 `App` + 宿主服务接口（请求位/退出意愿/名字） |
| `infer/app/lib/exit.scm` | App 侧辅助模块：`(EX.install)` / `(EX.install_with 重绘函数)` |
| `infer/tools/e2e/native/app_probe.scm`、`test_app_exit.py` | 宿主台「请求/阻塞/唤醒/退出意愿」契约测试（19 项断言） |
| `infer/tools/e2e/browser/regress_exit.js` | 浏览器端端到端（19 项断言，Chromium/headless-shell/Firefox 全过） |

```
(native App)
(App.exit)              : Boolean  —— 弹全局确认框（**阻塞**）；#t=确认（宿主本帧末销毁实例），#f=留下
(App.exit_with 标题)     : Boolean  —— 同上，自定义标题
(App.exit_now)          : Boolean  —— 不问，直接退出（App 自己已确认过）
(App.hide)              : Boolean  —— 请求切后台（P1 打桩 = 退出）
(App.back_requested)    : Boolean  —— 硬 A/ESC 待处理（**取走即清零**）
(App.modal_closed)      : Boolean  —— 宿主模态框刚关闭（取走即清零）
(App.is_active)         : Boolean  —— 是否前台活动（退出流程启动后为 #f）
(App.name)/（App.platform）: String —— 当前 App 名 / 平台标识
(native Ui) (Ui.confirm 标题 正文) : Boolean —— 通用确认框（阻塞，#t=确认）
```

每个 App 只需两行就得到"按挂断键→全局提示是否退出"：

```scheme
(import EX "app/lib/exit.scm")      ; 头部
(EX.install)                        ; 启动时（或 (EX.install_with render)）
```

## 2. 时序与关键语义

```
硬 A/ESC ─┐                                            App 自己的按钮/菜单项
          ▼                                                      │
  宿主置"返回请求"(ui_scheme_app.c)                                │
          │  ←── App 的 EX.poll（80ms 定时器）消费 ──┐              │
          └──────────────────────────────────────► (App.exit) ─────┘
                                                         │ 进程 BLOCKED
                                    宿主每帧 am_native_ui_pending() → 画确认框
                                                         │
                             ┌───────────────────────────┴──────────────┐
                         「留下」#f                                   「确认」#t
                   App 继续（可选调重绘函数擦残影）        置退出意愿 → 本帧**泵完之后**销毁实例
                                                          （App 因此还有一个时间片做收尾，如落盘）
```

- **不能在 native 函数里销毁运行时**（栈上还压着该进程的上下文）：App 侧只能"发请求"，
  teardown 必须回到宿主帧末——`ui_app.c` 的 `STATE_SCHEME_APP_RUN` 在 `render_frame` **之后**
  调 `ui_scheme_app_should_exit()`，成立才 `on_exit()` + 回列表。
- **阻塞会暂停该进程的全部定时器**（上游既定语义）：确认框存续期间 App 自然定住（J2ME 式暂停），
  不需要额外代码；也正因如此，"轮询返回请求"必须写在**阻塞之前**（`app_probe.scm` 记了这一条）。
- **兜底**：返回请求 700ms 内无人消费（App 未迁移/已停机报错）就按旧行为退出，不会把人困在 App 里；
  模态框期间的 esc 归控件，不参与兜底。
- **零快照的代价**：确认框关闭后画面上的对话框像素要靠 App 重绘覆盖。绝大多数 App 每拍整屏重绘，
  `(EX.install)` 足够；事件驱动型（计算器/二维码等）用 `(EX.install_with 重绘函数)`。

## 3. 三个必须处理的边沿（都是实测踩出来的）

1. **退出后"在途触摸序列"会落到列表上变成一次点击**：App 侧 `(native Touch)` 由电平派生边沿，
   往往先于宿主输入层认定松手（宿主有 UP 去抖），于是 App 已销毁而宿主仍认为"按住"；列表初始化时
   的电平兜底会把它当成**新的按下**，紧接着的 UP 就成了点击——实测"俄罗斯方块确认退出"会直接
   又启动 gol。处置：`ui_scheme_app_on_exit()` 置 `s_list_touch_guard`，列表在**新的一次按下**
   出现前剥掉触摸字段（600ms 窗口兜底解除）。
2. **长按 A 的 -2 连发**：会把刚弹开的确认框连坐关掉。复用 Ui 模态框的
   `s_modal_ignore_longpress_repeat`（宿主在收到 -2 返回请求时置位，见到下一次 -1 才解除）。
3. **列表事件入口走包装函数**：`ui_app.c` 的 LIST 分支原先直接调 `ui_widget_menu_event_handler`，
   绕过 `ui_scheme_app_list_event_handler`（上面那条抑制就在其中，曾经"改了没生效"）。现已改为
   调用模块自己的包装。

## 4. 迁移

- **24 个 App 全部接上**（`(import EX ...)` + 末尾 `(EX.install...)`，逐个按其绘制方式选
  `install` 或 `install_with`），App 列表零改动（仍是放 `.scm` 即出现）。
- `tetris.scm`：自己的「退出」按钮 → 确认框「确认」改成 `(App.exit_now)`（**真退出**，
  不再"冻结游戏等 A 返回"），删掉 `stopped/draw_stopped/stopped_drawn` 死代码。
- 过时注释/布局说明同步更新（calculator/svm/notepad/qrcode/particlelife/uinput/touchgravity）；
  `am_native_Key.h` 里"ESC 依然是宿主的返回键"改成新的归属说明。
- 构建清单同步：`build_wasm_gui.sh`、`Makefile`、`mp135.mk`、`android/.../CMakeLists.txt`、
  `tools/e2e/host/run_host_regress.sh` 各加 `am_native_App.c`；宿主回归台注册 `App` 库并
  `am_native_app_reset(L"probe")`。

## 5. 验证

- **宿主台契约**（`python3 tools/e2e/native/test_app_exit.py`，**19/19 PASS**）：
  确认→`#t`+退出意愿置位、留下→`#f`+意愿 0、`(App.back_requested)` 取走即清零、
  `(Ui.confirm)` 共用机制、无人应答时全程 BLOCKED 且不崩。
- **浏览器端到端**（`regress_exit.js`，Chromium / chrome-headless-shell / Firefox **各 19/19 PASS**；
  事件驱动型另跑 `E2E_APP=calculator E2E_EPOCH=0` **16/16**）：触屏右上角不再退出、硬 Esc 弹确认框、
  确认框期间 App 冻结（epoch 不涨）、点「留下」框消失且 App 恢复推进、网页「ESC」按钮等价、
  框内非回车键=留下（安全默认）、点「确认」回列表（state 79）、可再次启动。
- **回归**：宿主回归台 24 个 App `err=none`；浏览器存活回归 23 个 App 全部 `80 → 80`；
  `regress_tetris` 20/20、`regress_calculator` 24/24、`regress_key_fs` 17/17、
  `regress_qrcode` 21/21、`regress_uiprompt` 21/21；`test_ui_prompt` 11/11、
  `test_touch_imu` 36/36、`test_tetris` 21/21、`test_qrcode` 29/29、`test_key_fs` 44/44。
- 既有测试脚本同步改造：凡"按 Esc 退出 App"的用例都补一步回车（= 确认），
  `regress_uiprompt` 的第 8 项从"长按 Esc 直接回列表"改为"长按 Esc 弹确认框 → 确认回列表"。
- 环境备注：`regress_tetris` 的「速降」判据改为"下落 1~2 格"——探针窗口会跨过重力节拍（800ms/格），
  原先"恰好 +1"的断言本身就不稳（改动前后各复跑均能复现）。

## 6. 遗留（P2~P4）

- **P2 两段式收尾**：`(App.exit)` 确认后给 App 一段显式收尾窗口（现在是"本帧一个时间片"），
  以及"退出前询问是否保存"的范式（便签/游戏高分）。
- **P3 各 App 的退出入口**：把右上角让出来的空间用起来，统一 HUD 文案（现在多写"A退出"）；
  tetris 的自绘确认框可换成 `Ui.confirm`（视觉与宿主框完全一致）。
- **P4 `hide`/多任务**：实例表（单前台 + N 后台槽）、列表"运行中"角标、统一 `suspend/resume`
  原语（锁屏现在仍是"伪 esc 硬退外设态"，应改为 suspend + `App.is_active=#f`）、解释器池预算
  （WASM 12MB / ESP32 2MB）。
- **触屏宫格整体退场**：本次只让 Scheme App 运行态退出该映射；存量 C 状态仍是过渡态
  （见 ui_app.c 输入原则第 5 条）。

------------------------------------

# 2026-10-30（五）App 退出接口 P2：两段式收尾 + 退出前落盘范式

P1 的 `(App.exit)` 确认后是"本帧末就销毁"——App 只有**一个时间片**做收尾，靠巧合而不是
契约。P2 把这段窗口写进协议：**确认 ≠ 可以销毁**，中间加一段显式的收尾窗口。

## 1. 协议（两段式）

```
用户点「确认」
   → 本库置「退出已确认」（App.exit_grace 计时开始，is_active 变 #f）
   → (App.exit) 返回 #t
App 收尾：落盘 / 停外设 / 画最后一帧 …（可跨多拍，默认上限 2000ms）
   → (App.exit_now)            ← 第二段：宣告收尾完毕
宿主：本帧末销毁实例（回应用列表）
```

- 收尾窗口 `AM_NATIVE_APP_EXIT_GRACE_MS = 2000`；超时由宿主强制销毁并打印
  `[scheme-app] 退出收尾超时（N ms 仍未 (App.exit_now)），强制销毁实例`。
- **模态框存续期间自动续期**（`am_native_app_extend_grace`）：App 收尾时再弹
  `(Ui.confirm ...)` 问用户，不会被收尾窗口从对话中掐断。
- 判定收敛到库里：`am_native_app_exit_wanted(rt)` 同时表达"`exit_now` 已调"与"收尾超时"，
  宿主只做执行者（`ui_scheme_app_should_exit`）。
- 退出流程中再按硬 A/ESC 被忽略（不再重复弹确认框）。

## 2. App 侧：收尾登记（`app/lib/exit.scm`）

```scheme
(import EX "app/lib/exit.scm")
...
(EX.on_exit do_save)        ; 用户确认退出后、实例销毁前调用一次
(EX.install_with draw_all)  ; 硬 A/ESC 轮询 + “留下”后重绘
```

`EX.poll` 的确认分支现在是 `(收尾函数) → (App.exit_now)`。原理说明写在模块头：
"自动保存要等静止 0.5s，刚打完字立刻退出就可能还没落盘——收尾函数补上这一手"。

## 3. 验收（新增/加强）

- **宿主台 `test_app_exit.py` 扩到 30 项**（新增两组，靠 harness 的 `APP_PEEK=帧,...`
  按帧打印 `wanted/confirmed/grace_ms`）：
  - 收尾可跨多拍：确认后第 60 帧 `wanted=0 confirmed=1 grace_ms=320`，App 忙 12 拍后
    `exit_now`；第 200 帧 `wanted=1`；
  - 收尾超时：确认后一直不调 `exit_now` → 窗口内 `wanted=0`，第 230 帧（virtual 3.0s）
    `wanted=1 grace_ms=3040`。
- **浏览器 `regress_key_fs.js` 加第 8 步"退出前落盘"**（23/23）：便签打字后**立刻**退出，
  断言文件已含该字符，且"输入→确认框出现"仅 93ms ≪ 自动保存阈值 480ms（否则自动保存
  可能抢先，判据失效）。
  - 顺带修好该用例的一个**假通过**：退出回列表后直接按回车落在列表第 0 项
    （每次进列表高亮都会重置），第 7 步其实一直在测 calculator 而不是 notepad；
    现改为重新下移 idx 项再进入（`enterAppFromList`）。
- 回归全绿：`regress_exit` 19/19（Chromium；上一轮三引擎均过）、`regress_tetris` 20/20、
  `regress_calculator` 24/24、`regress_uiprompt` 21/21、`regress_qrcode` 21/21、
  `regress.js` 全 App 存活（Firefox）、宿主台 24 App `err=none`、
  `test_ui_prompt` 11/11、`test_tetris` 21/21、`test_key_fs` 44/44。

## 4. 有意留下的口子

- 收尾函数里若发生运行时错误（进程停机），宿主按"进程已停"处理仍会销毁实例，但那时
  已经没有机会落盘了——所以收尾函数应尽量简单（同步写、不弹框）。要在收尾阶段问用户，
  用模态框是安全的（会续期），但要意识到"用户可能看到两连问"。
- 收尾窗口固定 2000ms，不做按 App 配置；真需要更长的收尾（大文件导出）应改为后台任务 +
  `(App.exit_now)` 延后，而不是拉长窗口。
- `(App.hide)` 仍是打桩（= 退出）。多任务/后台（P4）落地时会复用同一套"确认→收尾→执行"
  三段式，只是最后一段从"销毁"变成"冻结 + 登记实例"。

------------------------------------

# 2026-10-30（六）App 退出接口 P3：统一的触屏退出按钮 + 各 App 接入

P1/P2 把"退出"变成了 App 可调用的接口，但**触屏用户在本机却几乎没有出路**：宫格软按键
（含右上角 esc 格）已经整体退场，屏幕上又没有可见入口，只能靠实体键盘的 A 或网页上的
ESC 按钮。P3 补上统一入口，并顺手把俄罗斯方块的一处重复实现并掉。

## 1. 统一的「退出」按钮（app/lib/exit.scm）

```
(EX.install) / (EX.install_with 重绘函数)   ; 启动时一次
(EX.draw_button)                            ; 画在**右上角**（36x16；无触屏时自动不画）
(EX.check)                                  ; 每拍一次（画完按钮、推屏之前）
(EX.draw_button_at x y)                     ; 少数把整屏用满的 App 换位置（计算器用左上角）
```

- 点按钮 = 与硬 A/ESC 完全相同的流程（全局确认框 → 收尾 → 退出）。落在按钮矩形内的那一次
  **按下/松开沿由 (EX.check) 吞掉**（`Touch.take_press/take_release`），所以 App 自己的点按
  逻辑不会也响应这一下；落在别处的沿原样留给 App。
- 面板统一「深底 + 细描边 + 12px 字」，与各 App 的 HUD 同色系；24 个 App 逐个确认右上角
  40x18 不被内容占住（gol 状态文案压短、touchgravity 的 g/yaw 读数左移、particlelife 与 svm
  原先的"A退出/返回"标记直接删掉——按钮取代了它们）。
- 计算器是唯一换位置的：它的顶部整块是"历史 + 当前行"显示区（表达式右对齐），
  所以按钮放**左上角**（`EX.draw_button_at 2 2`）；对应 e2e 用 `E2E_BTN="20,10"` 覆盖坐标。

## 2. 关键教训：不要在 App 里另装定时器

第一版 `EX.install` 用 `System.set_interval 80 poll` 轮询退出请求，结果**所有 App 变慢**：
`tools/e2e/native/test_tetris.py` 的重力断言立刻炸了（960ms 应该落一格，实际没落）。
用探针（`tools/e2e/native/dt_probe.scm`，同进程装 1 vs 2 个 interval）定位到：

> 上游运行时**每轮事件循环对同一进程只触发一个定时器回调**（回调不可重入，其余顺延到下一轮）。
> 于是"App 的 tick + 本模块的 poll"会互相错位：实测 tick 间隔从 33ms 拉到约 63ms；
> 而把 dt 钳到 50ms 防跳变的 App（俄罗斯方块）每拍丢掉十几毫秒——重力/动画整体变慢。

改法：**轮询点回到 App 自己的每拍**——`(EX.check)` 一行，不用任何额外定时器；探针显示
`dt_sum` 与墙钟一致，各 App 的 refresh 计数也回到基线（tetris 36→40、touchgravity 36→40）。
副产物：App 的 TIMER 数量回到"每个 App 只有一个"，对多任务（P4）也更友好。

另一处更隐蔽的坑：`(EX.check)` **必须放在每拍无条件执行的位置**。第一版按"顺手机械插入"
把它放在 `(Gfx.refresh)` 前面，对"有脏标记才重绘"的 App（计算器）就永远轮询不到——
表现为该 App 里按 A 没反应（`regress_tetris` 复进入后按 Esc 不弹框即此兆）。

## 3. 俄罗斯方块：并掉重复实现

- 删掉自绘确认框（`draw_modal/modal_input/modal` 状态与几何常量），退出入口只保留统一按钮；
  原"场地右侧空白列"的「退出」虚拟键与 `vkey_hit` 的 5 号分支一并删除（`vkey_hit` 只剩
  速降 / 左移 / 旋转 / 右移）。
- 顺带删掉 `(native Ui)`：确认框由 `(App.exit)` 内部走宿主的 `Ui.confirm` 那套，不必自己声明。
- 测试同步：`regress_tetris.js` 改为点右上角按钮、并在宿主确认框的按钮中心（确认 112,141 /
  留下 208,141）应答；`test_tetris.py` 的确认框场景改用 `HAL_APP_AT`（帧:1 置返回请求 /
  2 确认 / 3 留下）驱动，断言从"框的画面"改为"进程是否 BLOCKED + 退出意愿"（画面归浏览器用例）。

## 4. HUD 文案统一（不再教用户"按 A"）

notepad「键盘输入 · 触屏=保存 · 右上角／Esc 退出」、qrcode「右上角=退出」、
uinput「再按 Enter 重来 · 右上角「退出」或 A」、gol 状态文案压短、touchgravity 读数避让、
particlelife/svm 删旧标记。

## 5. 验收

- 宿主台：24 App `err=none`（含新增按钮与 (EX.check)）；`test_tetris` 21/21（含"确认框期间
  不下落"——现在由 `Ui.confirm` 的 BLOCKED 天然保证）、`test_app_exit` 30/30、
  `test_ui_prompt` 11/11、`test_key_fs` 44/44、`test_qrcode` 29/29、`test_touch_imu` 36/36。
- 浏览器（WASM，Chromium）：`regress_exit` **21/21**（新增"点右下角旧宫格位置不退出、
  点统一「退出」按钮弹确认框"）、`regress_tetris` 20/20、`regress_key_fs` 23/23、
  `regress_calculator` 24/24（探针排除按钮矩形）、`regress_uiprompt` 21/21、
  `regress_qrcode` 21/21、`regress.js` 全 App 存活；计算器变体
  `E2E_APP=calculator E2E_EPOCH=0 E2E_BTN="20,10"` 18/18。
- 视觉核对：把 24 个 App 的右上角 64x26 逐帧裁出来拼成对照图（宿主台 `DUMP_PPM`）逐个过了一遍，
  确认按钮在这些 App 里都不压内容。

## 6. 遗留

- 按钮是"角标"式的小控件；若将来要更规范的"系统条"，可以考虑宿主级 overlay（需要帧缓冲
  快照/分层合成，与"零快照"的取舍要重新评估）。
- 各 App 的 `EX.draw_button` 都画在自己 `(Gfx.refresh)` 之前；progressive 类 App（sky*、
  mandelbrot、sdfprog）因此每推一帧都会重画按钮——幂等、成本可忽略，但将来若要更省可以
  只在整帧完成时画。

------------------------------------

# 2026-10-30（七）App 退出接口 P4：后台运行（实例表 / 切后台 / LRU 驱逐）

前三期把"离开"做成了一条显式的通道（`(App.exit)` 三段式：确认 → 收尾 → 销毁）。
P4 给"离开"加上第二条出路——**切后台**：不销毁实例，冻结后回列表，选中即恢复。

## 1. 实例表（infer/ui_scheme_app.c）

原先是"单前台单实例"的 `static UI_Scheme_App_State s_app;`，现在改成槽位表：

```c
#define SCHEME_APP_INST_MAX  (NANO_IS_M5ESP ? 1 : 3)   // 前台 1 + 后台 2（ESP32：无后台）
static UI_Scheme_App_State s_inst[SCHEME_APP_INST_MAX];
static UI_Scheme_App_State *s_app = NULL;   // 指向**前台**实例
```

- 实例身份用**源码相对路径**（`source_rel`）而不是 `s_apps[]` 指针——列表每次进入都会重扫描并
  复用同一块缓冲，指针会失效（这是本期最容易踩的一脚）。
- `app_instance_find / alloc / foreground / suspend / destroy` 五个原语：
  - **alloc**：优先空槽；没有空槽就驱逐 `last_active_ms` 最小的**后台**实例（前台永不驱逐，日志
    `[scheme-app] 实例槽位已满（3），驱逐后台实例：…`）；
  - **foreground**：清其它槽的 foreground 位、`am_native_app_set_active(1)`、
    `am_native_app_notify_resumed()`、`am_native_Key_reset()`（后台期间积压的按键不带给 App）、
    `am_native_ui_abort()`；
  - **suspend**：`active=0` + 丢弃挂起的模态请求；**运行时与内存池原地保留**。
- `ui_scheme_app_init` 变成"有同源实例就恢复、否则冷启动"，因此列表里选中带角标的 App 就是
  **恢复同一运行时**（App 内的状态、定时器、闭包帧全都还在），不是重启。

## 2. 用户可见：三按钮退出确认框 + 列表角标

- 退出确认框对**退出型请求**多画一枚「后台」：`确认 / 后台 / 留下`（三枚 54px 窄按钮、12px 字；
  控制台的 C 场景仍是两枚 84px）。`ui_exit_confirm_draw_msg_ex/hit_msg_ex` 带 `with_bg` 参数，
  `am_native_ui_resolve_confirm(rt, answer)` 的 answer 取 0=留下 / 1=确认 / 2=后台。
- 「后台」→ `am_native_app_request_hide()` → 宿主帧末 `ui_scheme_app_should_hide()` 成立 →
  `ui_scheme_app_suspend_to_list()`（不动运行时）→ 回列表。
- 列表项对"有存活实例"的 App 追加 **`● 运行中`** 角标（标题截断到 40 字给角标留位）。
- `(App.hide)` 是同一条通道的 App 侧入口（无确认框，直接请求）；调用前该落的盘先落完
  ——冻结后不再有执行机会。

## 3. suspend / resume 成为统一原语

- `ui_scheme_app_suspend_current()` / `ui_scheme_app_resume_current()`：只翻转 active 状态，
  不切状态机。**锁屏**改用它（`lock_screen_enter/exit` 里对 `STATE_SCHEME_APP_RUN` 调用），
  不再依赖"没进状态机所以没泵"这种隐含语义；(App.is_active) 在锁屏期间为 #f。
- `(App.resumed)`：前台化时置位、取走即清零；`app/lib/exit.scm` 的 `(EX.check)` 见到它就调一次
  登记的重绘函数——挂起期间画面归别人，恢复后 App 需要补一帧（对"有脏才画"的 App 尤其重要）。
- `(App.is_active)` 的语义统一为：`active && !exit_wanted && !exit_confirmed`。

## 4. 内存预算

每实例一个解释器池（WASM/桌面 12MB、ESP32 2MB），所以槽位上限直接就是内存预算：
WASM 3 实例最多 36MB（浏览器堆 384MB，充裕）；**ESP32 取 1**——没有后台能力，
`(App.hide)` 退化为退出（`ui_scheme_app_should_hide()` 恒 0 并打一行日志）。

## 5. 排查记录（本期唯一的真 bug）

重构时把 `am_native_gfx_bind(global_state->gfx)` 从冷启动路径挪丢了（只留在"恢复"分支），
症状极具误导性：App 能加载、宿主每帧都在泵（`alive=1`、`rt` 非空）、但**屏幕全黑、tick 完全
不跑**——因为 App 第一次 `(Gfx.*)` 就报"未绑定绘图目标"，进程当场停机；而运行期错误只进
error FIFO 由宿主画成红色叠加文字（不打印到控制台），所以浏览器 console 里一片安静。
定位手法：在 `render_frame` 里打一次"pump #N s_app=… alive=… rt=… pid=…"确认宿主侧正常，
再在 `(App.back_requested)` 里打调用计数——**计数为 0 即证明 Scheme 侧根本没跑到那一行**。

## 6. 验收

- 浏览器 `regress_exit.js` **38/38 × Chromium / headless-shell / Firefox**，新增：
  - `[8]` 三按钮确认框 → 点「后台」→ 回列表且**不产生新的冷启动**（用控制台里
    "解释器内存池" 的行数判定）→ 再选中 → 走的是"恢复后台实例"（日志判定）；
  - `[10]` 实例槽位：连续冷启动并切后台 3 个 App（占满槽位）→ 第 4 个触发
    `驱逐后台实例` 日志、且新 App 正常运行。
  - 列表高亮是**有状态**的（`w_menu_main` 单例），故导航一律"先顶到第一项再下移 idx 次"。
- 宿主台 `test_app_exit.py` 扩到 **36/36**（新增 `(App.hide)`/`(App.resumed)` 契约：
  hide 只置切后台请求、不等于退出意愿；`RESULT` 新增 `app_hide=` 字段）。
- 回归：宿主台 24 App `err=none`；`regress_tetris` 20/20（确认框改三按钮几何：确认 x=98 /
  后台 x=160 / 留下 x=222，y=141）、`regress_key_fs` 23/23、`regress_calculator` 24/24、
  `regress_uiprompt` 21/21、`regress_qrcode` 21/21、`regress.js` 全 App 存活；
  `test_ui_prompt/test_key_fs/test_qrcode/test_tetris/test_touch_imu` 全过。
- 视觉核对：三按钮确认框与"gol ● 运行中"角标截图确认。

## 7. 遗留

- 后台实例**完全不跑**（冻结=不泵）。真后台执行（音乐继续播、定时器照跑）需要给定时器加
  优先级/预算机制，属于下一步的"后台执行策略"，不要靠"继续泵"糊过去（会与前台抢时间片）。
- 从列表**直接关闭**某个后台实例的入口还没有（现在只能靠 LRU 驱逐或进去再退出）。
- 列表角标目前是文本后缀；若将来列表要显示图标/多行，需要扩展 menu 控件的条目渲染。

------------------------------------

# 2026-10-30（八）修复：星历（ephemeris）里「退出」按钮无响应

## 症状与根因

点 App 右上角「退出」按钮完全没反应（硬 A 也不稳）。根因是**点按判定跟着 App 的拍子走**：
`app/lib/exit.scm` 的 `(EX.check)` 只在 App 自己的 tick 里跑，按钮靠"每拍采样触屏电平"
（按下→松开两次采样）判定命中。而 `ephemeris.scm` 的 tick 是 **1000ms**，人手一次点按只有
~100~150ms，几乎永远落在两次采样之间——采样不到，就等于按钮不存在。
（顺带：硬 A/ESC 置的"返回请求"也可能被宿主 700ms 兜底窗口抢先退掉，因为 1 秒一拍的 App
还没来得及消费。）

## 修复（三段，前两段是通用的）

1. **按钮命中判定交给宿主（事件级）**：新增 `(App.set_exit_button x y w h)`，`EX.draw_button`
   在画按钮时把矩形登记给宿主；`ui_scheme_app_event_handler` 在**触摸 UP 边沿**上问
   `am_native_app_hit_exit_button()`，命中即与硬 A/ESC 走同一条通道（置"返回请求"）——
   与 App 的轮询快慢无关。App 侧那份电平式判定作为"没有该宿主服务"时的兜底保留。
   · 无触屏硬件时登记 (0,0,0,0)（= 没有按钮）；实例销毁/前台化时宿主清空。
2. **兜底窗口 700ms → 1500ms**：该值必须大于**最慢 App 的轮询周期**（星历 1s，其余 ≤100ms）。
3. **星历自己**：轮询周期 1000ms → **100ms**，但重算（星历级数 + 整屏重绘）只在"秒数变了"
   时做一次——CPU 与原来相同（实测刷新仍是 1 Hz），退出请求的响应则从"下一次 1 秒拍"变成
   "下一次 100ms 拍"。

## 验证

- 星历实测（Chromium）：点「退出」按钮 → 确认框 **181ms**；硬 Esc → **70ms**；点「留下」框
  消失、点「确认」回列表（state 79）；刷新频率不变（3s 内 epoch +3）。
- `regress_exit.js` **38/38 × Chromium / headless-shell / Firefox**；新增一次慢拍 App 的回归
  跑法：`E2E_APP=ephemeris E2E_EPOCH=0`（**35/35**）。
- 其余回归全绿：宿主台 24 App `err=none`、`regress_tetris` 20/20、`regress_key_fs` 23/23、
  `regress_calculator` 24/24、`regress_uiprompt` 21/21、`regress_qrcode` 21/21、
  `regress.js` 全 App 存活；原生 `test_app_exit` 36/36、其余四套全过。
- 测试脚本顺带修：应用列表的高亮**每次进入都重置到第 0 项**，所以"选中第 idx 项"只需
  `ArrowDown × idx`；原来那种"先连按 ArrowUp 二十次顶到顶部"的写法会撞上按键连发机制，
  把高亮带到错误条目（曾让 `E2E_APP=ephemeris` 实际测的是 notepad/lenia）。

## 经验

- 凡是"由 App 自己的 tick 驱动的交互"，其可靠性上限就是 tick 周期。系统级入口（退出、返回）
  不要依赖 App 的节拍：宿主已经有可靠的触摸边沿，让宿主判定 + 通过"请求位"交给 App 处理。
- 兜底超时是"用户体验的最后一道闸"，取值必须与最慢的合法消费者对齐，否则会误杀正常流程。

------------------------------------

# 2026-10-30（九）多进程 / 队列 IPC 演示 App（fork × 4 + 环形列交换 + 进程表 HUD）

新增 `infer/app/forkqueue.scm`（落在 `app/` 下即自动进「灵机应用程序」列表，零固件改动）：
把操作系统概念画出来——1 个父进程 + `System.fork` 出来的 4 个子进程，每个子进程独占屏幕
1/4（4 条竖列之一）管自己那批粒子；粒子越过列边界时经 `System.write/read` 队列**移交给
邻列的进程**；顶部 HUD 就是这张进程表（每个 PID 的收发消息量、入队水位、堆活对象数、
VM 栈深），第二行是池级 `System.memstat`。也顺带是 fork/队列的压力测试（持续跨界 =
持续出队入队，实测约 60 消息/秒、20 秒长跑丢包 0）。

## 设计要点（可复用的模式）

- **拓扑**：4 列成环，列 c 右越界 → 列 (c+1)%4、左越界 → 列 (c+3)%4；屏幕最左/最右是
  环面绕回点（±SW 绕回**只**在这里做）。共 16 条队列，全部在 fork **之前**创建（队列编号
  是虚拟机全局资源，子进程天然共享）：8 条粒子有向边队列（每条边一条 = 单生产者单消费者）
  + 4 条统计上报（子→父）+ 4 条节拍令牌（父→子）。
- **只传立即值**：把柄（handle）是进程私有堆内索引（manual §3.4 的 fork 深拷贝），把列表
  塞进队列后接收方拿到的是**它自己堆里同号的对象**——纯属巧合。故一颗粒子编码成两个数：
  位置（偶数）= `2*(xq + 160*yq)`，速度（奇数）= `2*((vxq+16) + 32*(vyq+16)) + 1`；
  **奇偶即记录类型**，所以"位置写进去、速度那条写失败"这种半条记录只会丢掉一颗，
  不会错配成畸形粒子（自同步协议）。所有数值压在 2^19 以内——32 位宿主（wasm32）浮点
  只有约 19 位有效位，再大就做不了 mod/floor 解码（同 `am_native_Time.h` 的结论）。
- **父进程是唯一刷新者，也是节拍源**：每 40ms 一次 interval 回调 = 汇总统计 → 画 HUD →
  `Gfx.refresh` → 给 4 个子进程各发一个**节拍令牌**。子进程平时阻塞在控制队列上等令牌
  （阻塞态不占时间片），醒来先非阻塞排空两条入边队列收粒子，再走一步物理并只重画自己
  那一列（擦旧画新，不整列清屏 → 无撕裂）。
  由此白得两个好处：① 5 个进程节拍天然对齐；② `(App.exit)` 阻塞期间父进程不发令牌，
  子进程自动停笔，**不需要额外设计暂停/恢复协议**（只多一个"重绘令牌"用于确认框关闭
  或后台恢复后的整列重绘）。
- **HUD 的数**：`发/收` 是子进程自己的累计条数；`队峰` 是入队队列的**实测峰值水位**
  （队列没有"查询长度"接口，但子进程排空时数走几条 = 它的瞬时长度，而该长度恰好在
  排空前最大 → 取历次最大值即峰值；父进程按 `发/收` 差额推算瞬时值也可行，但采样时刻
  永远在排空之后，读出来恒为 0，没有信息量——踩过的坑）；`堆/栈` 用 `Diag` 的新增与
  既有接口（见下）；池级 `VM/堆` 用 `System.memstat`。
- **画面即证据**：粒子颜色 = 归属进程（跨界后被接管方换色），距列边框 < 4px 的点画白色
  = 正要交出去/刚被移交进来（纯几何判定，零额外状态）。速度场刻意只用"右向漂移 +
  随 y 变化的 sin 剪切"——涡旋/向心类场会把粒子挤到某几列，密度不均就看不出"每列一个
  进程"。

## 附带的固件侧改动（两处，都是必要的）

1. **`(Diag.heap)`（`am_native_Diag.c/.h` 新增）**：返回**当前进程**堆里的存活对象数
   （把柄表有效条目数，O(1) 只读）。原因：`System.memstat` 是**池级**口径（解释器池由
   全部进程共享，`vm_used/heap_used` 是整池数字），fork 之后各进程读到的几乎一样；
   要显示"每个进程自己占了多少堆"只能数它自己那把柄表。per-PID 的 VM 占用则用既有的
   `Diag.opstack + Diag.fstack`。
2. **`(App.set_exit_button)` 数值判定 bug 修复（`am_native_App.c`）**：原实现只接受
   `am_value_is_int`，而灵机里**非负整数字面量是无符号整数**（只有 `-` 的结果是有符号
   整数），于是 App 里写 `(App.set_exit_button 280 2 36 16)` 会被静默降级成 `(280 0 0 0)`,
   宿主侧热区判定直接失效——这正是「退出按钮热区」长期只靠 `app/lib/exit.scm` 的电平
   兜底才可用（慢拍 App 就失灵）的根因。改为统一 `app_pop_pixel()`：int/uint/float 都收
   （按像素四舍五入）。`regress_exit.js` 38/38 仍全绿。

## 一个有价值的实测结论（影响所有多进程 App）

解释器每帧只推进**一个**进程的时间片（`am_runtime_event_handler` → 一次
`am_runtime_tick`，`AM_COMPUTATION_PHASE_LENGTH=1`），因此 **N 个可运行进程的显示节拍
≈ 宿主帧率 / N**：本 App 5 个进程在浏览器 WASM 下约 **12 帧/秒**（单进程 App 仍是满
帧/自定节拍，例如 gol 12/s、flip 30/s）。若将来要多进程高帧率，方向是提高
`AM_COMPUTATION_PHASE_LENGTH`（会让所有 WASM App 每帧多跑几轮时间片，需评估其它 App
的帧时预算）而不是在 App 侧绕。本 App 因此把物理步长按 12~25Hz 调参。

## 验证

- 宿主回归台：`APP=forkqueue` `err=none`；1200 帧长跑粒子总数守恒（Σ列内 = 160 − 在途）、
  丢包 0；全 24 App 的宿主回归与改动前一致（`err=none`）。
- 浏览器（Chromium headless，真 wasm + 真资源挂载）：导航至 App（列表第 4 项）→
  state 80、刷新 12/s、粒子均匀铺满 4 列、`丢 0`、per-PID 堆/栈数字各自不同且长跑稳定
  （≈1000 对象不增长）；点右上角「退出」按钮/硬 Esc → 全局确认框（**子进程确实停笔**：
  确认框存续 1.2s 内刷新增量 0）→「留下」→ 整列重绘把确认框残影擦净、恢复 12/s；
  「后台」→ 列表 → 重选同一项恢复（同一实例、计数继续、刷新恢复 12/s）；「确认」→
  回列表 state 79。
- `regress_exit.js`（含后台/恢复/LRU 驱逐，且顺带把 forkqueue 当第 4 个实例）38/38；
  `regress.js` 全 App 存活无 pageerror；`tools/e2e/{host/run_host_regress.sh, browser/regress.js}`
  的 App 清单已加入 `forkqueue`（本目录不入库）。

------------------------------------

# 2026-10-30（十）渐进路径追踪 App：40×30 累积缓冲 + 色调映射显影

新增 `infer/app/pathtracer.scm`（约 15KB）：**渐进式蒙特卡洛路径追踪**——40×30 渲染
分辨率，每 tick 为每像素追加 1 个采样（1200 采样/帧），累积缓冲取平均后经
「曝光 → Reinhard → gamma 2.2」色调映射，以 8×8 方块铺满 320×240 上屏。画面从满屏
噪声开始逐渐收敛（"照片显影"观感；与 sdfprog 的逐行显影同哲学——重活摊到多帧的
解释器协作调度——但视觉不同：整幅画面同步从噪声里长出来，颗粒即胶片颗粒）。
编译目标 **Nano-WASM-GUI**，宿主回归台 + 三个无头浏览器端到端验证通过。

## 场景与算法（Cornell box 简版）

- 单位盒子 [0,1]³（前口 z=0 敞开，相机在盒外向内看；开口对盒内即纯黑背景/吸收）；
  左墙黄 (0.62,0.50,0.12)、右墙蓝 (0.10,0.20,0.58)、地板/天花板/后墙白 (0.80)；
  天花板上的 0.24×0.24 矩形**面光源**（暖色 Le=(24,22,16.5)，与天花板共面——
  同一次平面求交兼顾"天花板"与"光源"两种身份）；盒内三个落在地板上的球
  （两大一小，其一淡钢蓝）。
- **NEE（Next-Event Estimation）直接光采样**是低 spp 下可辨识的关键：每次命中非光源
  表面时，在光源矩形上均匀采点、解析算直接光 `Le·ρ·ndotl·cosl·A/(π·d²)`；可见性的
  阴影射线**只测 3 个球**——盒子是凸的，墙不可能遮挡盒内点到光源的连线（省一半阴影
  开销）。随后按**余弦加权半球采样**继续弹射（Lambertian 的 BRDF/pdf 相消，吞吐量只乘
  反照率）；固定深度 3（无俄罗斯轮盘）；弹射射线命中光源不再加发射项（该连接已由上一
  命中点的 NEE 采样过，防双计）；相机射线直接看到光源时正常计 Le（画面里能看到灯）。
- 像素内抖动采样（抗锯齿 + 颗粒感来源）；最后一弹跳过半球采样（方向不再使用，省 ~5%）。
- 累积缓冲：单张扁平列表 3600 项（rgb 交错），`set_item!` 原地加；spp 上限 4096
  （wasm32 浮点有效位与显影时长双保险，到顶后 HUD 显示"显影完成"、画面静止）。

## 实测

| 项 | 结果 |
|---|---|
| 单 tick（1200 采样 + 1200 次色调映射/上屏） | 宿主回归台 **~50ms**（≈2.5M VM 指令/tick ≈ 2100 指令/采样） |
| 浏览器 wasm32（chrome-headless-shell / chromium / firefox 一致） | **~106ms/tick ≈ 7.5 spp/秒**（spp 21 → 65 → 170 三阶段截图确认显影过程） |
| 观感 | spp 2 已能辨认房间轮廓（NEE 的功劳）；spp ~10 结构清晰；spp ~60 柔和成形；spp 170+ 接近干净，球体边缘可见黄/蓝墙的颜色渗色 |

## 解释器踩坑/合规（本次逐条对照）

- 算术二元嵌套（`check_multi_operand.py` 全绿）；无一元取负（`(- 0.0 x)`）；`/`、`mod`
  恒浮点（列表下标保持纯整数表达式：`(* (+ (* py 40) px) 3)`）；`Math.PI` 是函数 →
  π/τ 常数直接硬编码；cond 多表达式子句受限 → 全部平铺 `if` + `{}` 块；
  `'() 静态槽` → 累积表用 `(cdr '(0.0))` 取新对象后 push 3600 项。
- `Math.pow(0, γ)` 安全（0 的正次幂为 0），但负值会 NaN→`#null`→类型错误——辐射度量
  恒非负，无此路径；NEE 的 `ndotl>0` 守卫同时消除了"光源与天花板共面"导致的 0/0 奇异。
- 退出接口：`(EX.install_with redraw_all)`——确认框"留下"/后台恢复后按累积缓冲重绘
  同一显影进度；`(EX.check)` 在每 tick 开头（tick 约跨 5~8 个宿主帧，退出响应延迟
  ≈ 1 个 tick，与 sdfprog 同级）。

## 验证

- 静态：`check_multi_operand.py` ✓；括号/花括号双计数 0/0 ✓。
- 宿主回归台：spp 2/10/50/187 各阶段 DUMP_PPM 目检（噪声→显影过程正确，墙体红绿、
  面光源、三球与软阴影、颜色渗色均符合预期）；`err=none`。
- **浏览器 e2e**（新增 `tools/e2e/browser/regress_pathtracer.js`，按 manifest 推导 App
  下标=12）：Chromium / chrome-headless-shell / Firefox **三引擎均 PASS**——
  启动 state 80、epoch 7.5/秒持续推进、无 pageerror、Esc→确认框（state 保持 80）→
  Enter 回列表（79）。
- 回归：`run_host_regress.sh` 全 26 App `err=none`；`regress.js`（shell）全 26 App
  `launch=80 → end=80`（calculator 的 refreshΔ=0 为静态 App 既有现象）；
  `E2E_APPS=pathtracer` 的 `regress_det.js`（冻结时钟）正常（23 次刷新、913 色）。
- 清单同步：`run_host_regress.sh` / `regress.js` / `regress_det.js` 的 APPS 已加入
  `pathtracer`（字母序在 particlelife 与 qrcode 之间）；**顺带修正**了后两者缺失
  `calculator`（及 regress_det 缺 `forkqueue`）导致的菜单下标整体错位（按下标导航的
  脚本此前全都错位一位，forkqueue/xray 实际未被覆盖）。

------------------------------------

# 2026-10-31

新增灵机应用程序 **五子棋（AI 与人对弈）**：15×15 触屏落子，AI 用"活三/冲四"棋形打分 +
α-β（深度 2~4，迭代加深），并把**候选点分数热图**画在棋盘上。WASM GUI 目标编译与端到端
验证通过（Chromium / chrome-headless-shell 两引擎 21/21 断言）。**未改动任何 C/HAL/宿主代码**
——App 由 `ui_scheme_app.c` 扫描业务资源目录 `app/*.scm` 自动发现，零固件改动。

## 文件

- **`infer/app/lib/gomoku_core.scm`（新增，~740 行）**：纯逻辑，可脱离界面回归。
  - 棋盘：15×15 放进 23×23 的**带 4 圈边的填充盘**（边界格恒为 2）。所有棋形读取（沿 4 个
    方向各看 4 格）变成纯下标运算，无边界判断；"盘外"与"对方棋子"在棋形语义上等价，同码处理。
  - 棋形评分（两条互补视角）：①连续子串 + 两端开闭 → 显式区分 五连/活四/冲四/活三/眠三/
    活二/眠二（分值表 S_FIVE 5e4、S_OPEN4 1e4、S_FOUR 2e3、S_OPEN3 1.5e3 …）；②五元滑窗
    → 补"跳形"（`X_XX`、`XX_X` 这类连续子串看不到的棋形）。
  - **增量局面评估**：全盘 88 条线（横15+纵15+斜29+反斜29）各存一个带符号分值（黑正白负），
    落子只重算过该点的 4 条线并做差分 `gtotal += new - old`；叶子评估 O(1)，**不重扫全盘**。
    回退更快：`mk` 把 4 条线旧分值与总分**压栈**，`unmk` 弹栈恢复（完全不重算）。
    五连判定也 O(1)：只有刚落子的一方可能成五，看 4 条更新后的线分值即可。
  - 候选点维护：24 邻域计数 `nr` + 原地候选表 `cand`（`push` 增、`pop` 截断回 savedlen），
    落子/回退都是 O(24)。
  - 单点棋形分 `psc(x,y,pl)`：只看过该点的 4 条线、每条取 ±4 的 9 格窗口（前缀和数 5 个
    含中心的五元窗口）→ 双活三自然叠加成 2×S_OPEN3 > 一个冲四，符合"双三极强"的棋理。
  - 搜索：α-β + 迭代加深（2→4），根候选表一次生成（`我方棋形分 + 0.8×对方棋形分` + 局部手/
    靠天元微调，取前 20），全深度复用；深层（ply≥2）只展开前 10 个（NARROW）压节点数。
    时间预算用 **`(Diag.ms)`（单调实时毫秒）**——不受宿主虚拟时钟影响，是用户真实等待时长；
    超预算立刻回退到上一深度的结果。另有"下一深度约需 5 倍耗时"的预估，剩余时间不够就不
    白跑一次深搜（实测让首手响应从 677ms 降到 412ms）。
- **`infer/app/gomoku.scm`（新增，~470 行）**：布局/绘制/触屏/节拍。棋盘 13px 格距
  （交叉点 `(8+13c, 22+13r)`），右侧面板：状态/用时/节点/评估 + 四枚按钮（新局、悔棋、
  热图开关、强度 2/3/4）。热图按**棋形等级**分档上色（活二青、活三黄、冲四橙、活四橙红、
  五连红），死点不画；棋子用"亮环 + 深色子身"以便在暗色棋盘上辨认；最后一手加小方块标记。
  退出走 `app/lib/exit.scm`（右上角按钮 / 硬 A/ESC → 全局确认框 → 收尾 → 销毁）。
  "AI 思考中 / α-β 深度 n"画在**右侧面板**而不是盖在棋盘上（初版盖在棋盘中央，把刚落的子
  藏住了，端到端断言也因此取不到人手的落子）。
  AI 落子在同一次 `tick` 内完成（阻塞至搜索结束）：先 `render` 推一帧"思考中"，再搜索；
  搜索前后各排空一次触屏边沿（`Touch.take_press/take_release`），避免思考期间的点按在
  结束后被当成新落子。

## 实测（本机 Ubuntu 24.04 + 无头 Chromium 1248）

- **棋力**：深度 4 对"贪心（深度 1）"19 手取胜；自对弈（双方深度 2）52 手正常成五终局，
  全程增量总分与从零重算一致（无漂移）。
- **速度**（WASM，Chrome headless，`Diag.ms` 口径）：开局首应手 412ms、中局 ~130ms；
  节点数 185~500。宿主回归台（native gcc）上同一搜索 ~200ms，即 wasm 约 1.5~2 倍。
  深度 4 在 700ms 预算内基本都能跑完；跑不完时回退深度 3 的结果，绝不超 1s。
- **宿主回归台**（`tools/e2e/host/harness.c`）：`APP=gomoku` `err=none`；另有 35 条逻辑断言
  （增量一致性、撤子恢复、成五判定、活三/眠三/活四/跳三/双三识别、AI 立即取胜/挡冲四/
  挡活三、深搜后棋盘与增量分不漂移、热图非空）。
- **浏览器 e2e**（新增 `tools/e2e/browser/regress_gomoku.js`，按 manifest 推导 App 下标=5）：
  21 条断言全 PASS——进入列表(79)→启动(80)→空盘→点天元落子(31ms)→AI 应手(412ms <1s)→
  热图出现→多手对局每步都有 AI 应手→无 pageerror→热图开关→悔棋退两手→强度切换→
  新局清空→右上「退出」弹确认框→「确认」回到列表(79)。Chromium 与 chrome-headless-shell
  两引擎均 21/21。
- 回归清单已同步加入 `gomoku`（`host/run_host_regress.sh`、`browser/regress.js`、
  `browser/regress_det.js`，字母序在 `gol` 与 `grayscott` 之间）——否则按下标导航的回归脚本
  整体错位一位。`regress_det.js` 下标 gomoku 判为 ABNORMAL（refreshΔ=0）是**事件驱动静态界面**
  的既有口径，与 `calculator` 同类（无输入就不重绘）。

## 解释器踩坑（本次实测，均已在代码内注释）

1. **多参数算术会崩**：`(+ a b 20)` 编译通过但运行时 `[Runtime] 指令执行异常`（只溢出一次，
   栈上多余参数残留）。所有算术一律二元嵌套（与 2026-10-18 曼德博条目同源，但那次是乘法，
   这次是加法的**三参**形式：画面上表现为「渲染到某一行就整机报错」）。
2. **`(G.var)` 带括号访问模块变量 = 调用它** → `[Runtime] call: 错误的调用目标`。
   模块导出变量（`G.gtotal`、`G.bx`…）必须写成裸名。
3. **`(break)` 带括号是编译错误**：`break`/`continue` 是裸符号；`(while c { ... (if p break) ... })`
   可用，但 `(if p (break))` 直接 compile_fail（无任何错误信息，只能二分定位）。
4. **`'()` 字面量是静态槽**：`(set! l  '())` 不会清空——同一处字面量上次 push 进去的元素还在。
   本 App 的固定表（填充盘/邻域计数/88 条线）改为"装载时建一次 + 每次用 `set_item!` 归零"，
   动态表（候选表/热图表/搜索结果表）改为"`pop` 到空"（`pop` 是原地 O(1)，表对象不变）。
5. **lambda 体内的 `define` 是每次调用的独立局部量**（实测递归验证）——递归函数（`negamax`）
   的循环下标/本层最优值/落点必须用它；模块级 `set!` 临时量会被下一层递归改写，
   症状是"回退到错误的落点"（unmk 用错坐标）导致增量总分漂移。
6. **32 位宿主浮点只有约 19 位有效位**（`am_native_Time.h` 记录）：所有棋盘分值压在 50 万以内
   （S_FIVE=5e4、SC_WIN=2e5），避免整数精度损失。

------------------------------------

# 2026-10-31（二）CNN 手写数字识别与可视化 App（卷积核/特征图/感受野/通道贡献）

按参考页面 `https://adamharley.com/nn_vis/cnn/2d.html`（Adam Harley 的 2D CNN 可视化：
画布输入 → 卷积核与特征图 → 逐节点查看计算 → 输出概率）实现灵机 Scheme App「CNN 手写数字
识别」。编译目标 **Nano-WASM-GUI**（`build_wasm_gui.sh` 产物），三个无头浏览器
（Chromium / chrome-headless-shell / Firefox）端到端各 22/22 断言通过。**未改动任何
C/HAL/宿主代码**——`ui_scheme_app.c` 扫描业务资源目录 `app/*.scm` 自动发现。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `infer/app/cnn.scm` | App 本体（~1270 行）：绘图板、brush、前向推理、两个图层视图、检视面板、概率柱 |
| `infer/app/lib/cnn_weights.scm` | 网络权重（**自动生成，勿手改**，19KB，以 quote 列表导出） |
| `infer/tools/train_cnn_mnist.py` | numpy 训练脚本（本机无 torch）：MNIST → 权重 Scheme 模块 |
| `infer/tools/e2e/browser/regress_cnn.js` | 浏览器端回归（22 项帧缓冲像素/哈希断言；`tools/e2e/` 不入库） |

接线：`build_wasm_gui.sh` 的 `cp app/*.scm` + `cp app/lib/*.scm` 与 manifest 生成已自动收录
（`app/cnn.scm` 与 `app/lib/cnn_weights.scm` 均 priority 1）。回归清单已同步加入 `cnn`
（`browser/regress.js`、`browser/regress_det.js`、`host/run_host_regress.sh`）——`cnn` 按
字母序紧跟 `calculator`，不更新会让按下标导航的既有回归整体错位一位。
附一条既有回归基建的观察（与本次改动无关，实测复核过）：**退出 App 回到列表时，列表选中项
复位到第 0 项**（退出 calculator / cnn 后列表画面逐像素相同，都高亮首项），因此
`regress.js` 那种"每轮只按一次 ArrowDown"的导航在第二次迭代起会反复启动第 1 项；
`regress_svm.js`/`regress_cnn.js` 那种"每轮从列表首项按下 idx 次"的写法才是对的。

## 2. 网络与训练（为什么这么小）

解释器每帧只给 31.25 万条 VM 指令，实测**一次乘加约 38~40 条指令（≈1 微秒墙钟）**，
所以网络规模由"一次推理 ≤ 4 帧"倒推：

```
输入 1x28x28（像素/255）
conv1  6 个 5x5，步长 2，无填充 → 6x12x12，ReLU     21,600 MAC
pool1  2x2 最大池化             → 6x6x6
conv2  8 个 3x3x6，步长 1        → 8x4x4，ReLU        6,912 MAC
fc     128 → 10 → softmax                            1,280 MAC
                                                合计 29,792 MAC ≈ 3.6 帧
```

- 训练脚本只用 numpy（`im2col` + 手写反向），**12 epoch / 27 秒**（本机 CPU），
  MNIST 测试集 **98.11%**；Adam 原页面是 6×(5x5,stride1)+16×(5x5)+fc，MAC 数约 24 万，
  解释器跑一次要 ~9 秒，不可用——改用 stride 2 与 3x3 的第二层把开销压到 1/8。
- 权重按 `%.6g` 导出（wasm32 上灵机浮点是单精度**且尾数被截 5 位**，相对精度 ~4e-6，
  写更多位没有意义）；`(define W1 '(…))` 用 **quote 列表**——灵机没有内建 `list` 过程。
- 训练/推理一致性已数值校验：同一批 10 张 MNIST 测试图（量化到 0..9 电平）分别喂给 numpy
  与 Scheme 侧实现，**logits 最大差 5.1e-05、概率最大差 6.2e-07、10/10 预测同号**
  （差值即 float 尾数截断的量级）。

## 3. 交互与布局（320x240）

```
┌ CNN 手写数字识别   [清空] [层2]  推理 62ms        [退出] ┐  y 0..18
│ ┌────────────┐  卷积1：6×5×5 步长2                      │
│ │  112x112   │  [核0][核1][核2]        ← 核正上方对齐自己的特征图
│ │  绘图板    │  [图0][图1][图2]        （12x12，每格 4px，热色）
│ │ 28x28网格  │  [核3][核4][核5]
│ │  每格4px   │  [图3][图4][图5]
│ │            │  输入块 × 卷积核 → 和
│ └────────────┘  [入5x5][核5x5][积5x5] 核1 / 和 -1.95 / 激活 0.00
│ 预测 3  99.7%                                            │
│ ▁▂▃█▁▁▁▁▁▁  ← 10 类 softmax 概率柱                          │
│ 0123456789   MNIST 98.1%                                 │
└──────────────────────────────────────────────────────────┘
```

- **绘图板**：112x112 = 28x28 网格、每格 4px——画上去的就是网络看到的（不做居中/缩放，
  与参考页面一致）。笔刷半径 1.3 格（≈MNIST 笔画宽度）带软边，采样点之间按 0.5 格插值
  盖章，快速拖动不断笔；手指滑出板外不落笔但同步"上一采样点"，回板内不会连出跨屏直线。
- **推理时机**：只在**松手**后跑一次；拖动笔画期间零推理开销，那一拍只重画绘图板
  （`render_pad_only`，约 800 次 `fill_rect`）。推理跨 3~4 帧，期间预测行显示「计算中…」。
- **右栏两个视图**（顶部 `层2/层1` 按钮切换，对应参考页面的图层选择器）：
  - ‹层1› 6 个卷积核 + 6 张 12x12 特征图；点按任一格 → 三块 5x5 网格
    「输入块 / 卷积核 / 逐元素积」+ 加权和 + ReLU 后激活，并在绘图板上框出该神经元的
    **5x5 感受野**（参考页面的 hover 检视改为触屏点按）。
  - ‹层2› 卷积2 的 8 张 4x4 特征图；点按任一格 → 6 根**左右向**条形图（右正左负）
    显示池化后 6 个通道各自的贡献 Σ(3x3 块×核) + 和 + 激活，绘图板上框出 **15x15 感受野**
    （3x3 池化块 × 每池化格 2x2 卷积1 格 × 每格 5x5 输入）。这是多通道卷积最直观的一屏。
  - 两个视图的默认选中格都是**全图最强**的那个神经元（推理后自动更新），一进来就有东西可看。
- **配色**：激活值 = 白→橙→深红热色（每图按自身最大值归一化，只为看清图案；检视面板给出
  的是未归一化的真值）；权重 = 蓝负/红正（全部 150 个权重共用的 ±max|W1|）；墨迹 = 灰阶。
- **清空**：清墨迹**并清掉各层激活与概率**（否则右栏会留着上一次的特征图，看着像"识别过"）。
- 退出走 `app/lib/exit.scm`（右上角按钮 / 硬 A/ESC → 全局确认框）。

## 4. 未做的（有意）

- 全连接层 128→10 的权重不做空间可视化（128 个数画不下）：它的输出就是左栏概率柱，
  识别结果本身就是它的可视化。
- 不做自动居中/尺寸归一化：本网络对"写得太偏/太大"确实会掉准确率，这是它的真实行为，
  不做掩饰（参考页面同样不做）。
- 不做参考页面的 3D 立体视图、逐层连线动画——320x240 放不下，且信息密度反而下降。

## 5. 实测（本机 Ubuntu 24.04 + 无头 Chromium/Firefox 1248/1555）

排查手法（本次用，未入库）：把 App 源码用一个 ~200 行的一次性桌面宿主编译运行（与
`ui_scheme_app.c` 同构的 `((lambda () …))` 包装 → `am_parse` → `am_link` → `am_compile`
→ `am_runtime_load_module`，运行期错误打到 stdout，`System.set_interval` 走每步 +20ms 的
虚拟时钟，帧缓冲存 PPM 看图）——浏览器里语法/运行期错误实测只表现为"黑屏 + 无输出"，
没有这条通路只能靠猜。

- **推理耗时**（WASM，`Diag.ms` 口径，显示在标题栏）：**60~65ms**（≈3.6 帧），与"38 条
  指令/乘加"的估算一致——耗时由 VM 指令数决定，与浏览器/宿主实现无关。
- **识别**（用 10 条手绘折线在 App 里逐个画，读回蓝柱所在列）：0/2/3/4/5/8/9 正确（7/10）；
  失败的「1」「6」「7」是折线画法本身不像（带长脚的"1"被判 7、画成闭合环的"6"被判 0）——
  与"不做归一化"的取舍一致。真实 MNIST 数据上 98.11%（量化到 0..9 电平后 9/10，
  其中一张 3 被判 8）。
- **浏览器 e2e**（`regress_cnn.js`，三引擎各 **22/22**）：进入列表(79) → 启动(80) →
  初始（绘图板纯白、无热色、卷积核蓝/红已着色、只有空轨道） → 画「3」（墨迹、命中类蓝柱
  落在第 3 列、特征图出热色） → 点按特征图格（选中高亮 + 算式更新，整帧哈希变化） →
  切「层2」（画面变化、仍有热色图） → 第二层点按（通道贡献更新） → 切回层1（哈希回到
  第一层那张） → 清空（墨迹/蓝柱/热色全部归零） → Esc 确认退出 → 列表(79)，无 pageerror。
- **宿主回归台**（`tools/e2e/host/run_host_regress.sh`，`APP=cnn FRAMES=200`）：
  `refresh=1(err=none)`——事件驱动静态界面（无输入就不重绘），与 `calculator` 同类口径。

## 6. 解释器踩坑（本次实测）

1. **没有 `list` 过程**：权重模块最初写成 `(define W1 (list 0.1 0.2 …))`，报
   `[Runtime] call: 变量 list 未定义`。若干字面量必须用 **quote 列表** `'(0.1 0.2 …)`。
2. **除零会终止进程**：第一次进入时特征图着色归一化分母 `mx1` 还是 0.0（要推理后才有值），
   `(/ dm_v mx1)` 直接把进程打死，画面上只留一帧空白。故启动时把归一化分母先置 1.0。
3. **`draw_rect` 边界会被随后画的格子盖住**：先画外框再填满格子，右/下两条边就消失了
   （表现为"图框只有左上两条线"）。改为**先格子、后图框**，且图框外扩 2px。
4. **`define` 只放在 lambda 体顶层**：`pick_map` 初版在 `{…}` 块里 `(define k …)` 取局部量，
   虽然按上游语义"lambda 体内的 define 是每次调用的独立局部量"，但项目既有约定是
   "块内只用模块级 `set!`"，故改为模块级临时量（`pm_*`/`pq_*` 前缀各不相同，遵守
   "跨调用链临时量绝不共用"）。
5. **`(if p a b #f)` 多写一个分支**是"未规定行为"（多余参数残留在操作数栈上），务必数清分支数
   ——本次初版 tick 里有两处，纯属手误，靠桌面宿主跑一遍拿到运行期错误才发现。

------------------------------------

# 2026-10-31（三）续体可视化 App（yin-yang puzzle / call-cc 生成器 / 协程，双色流时间线）

新增 `infer/app/callcc.scm`（21.7KB，落到 `linglong/app/` 后自动出现在「灵机应用程序」列表，
零固件改动）。三个例子都跑**真正的 call/cc**，共用一套"两条时间线"的呈现方式：

- 横向 = 时间（1 格 = 一次控制流推进，6px；窗口 44 格，满了整体左移 = 滚动时间线）；
- 两条轨道 = 两个控制上下文（两种颜色），最新一次换轨的竖线高亮；
- 段长标签（≥4 格才标，取泳道色的暗版）直接读出"这一段跑了多少步"；
- 顶部「双色流」把最近 20 格按轨道颜色逐字写出来（yin-yang 就是 `@ * @ ** @ *** …`）。

三个例子：① yin-yang（阴 @ = 第一个 call/cc 抓下的 k1，阳 * = 第二个 call/cc 抓下的 k2，
阳走完调用 k1 跳回阴；阴每轮 1 步、阳每轮多 1 步）；② 生成器（体被 call/cc 切成可续跑片段，
每产出一个素数就经续体把控制交还调用者，绿段 = 试除步数、橙格 = 调用者取值，绿段随
素数增大而变长）；③ 协程（A/B 各存一个续体，靠调用对方的续体来回切换，每回合步数递增）。
点按屏幕换下一个例子，每例跑满 240 格（约 15 s）也自动换；右上角「退出」按钮 + 硬 A/ESC
走 app/lib/exit.scm 的全局确认框。

## 1. 节奏：阻塞睡眠，而不是逐拍 call/cc（关键设计决定）

初版把"每拍推进一格"实现成"tick 里捕获逃逸续体 → 恢复上次挂起的 emit 续体"（教科书写法，
语义完全正确，宿主回归台也跑得通）。但**本解释器不回收续体**：每次 `call/cc` 捕获的栈快照
永久占内存，`System.gc` 无效（实测：只有捕获、从不恢复的探针也线性增长；丢弃引用也不减）。
逐拍捕获 ≈ 0.7 KB/拍，12 MB 池下 **30000 帧（约 8 分钟虚拟时间）后必死**
（`[Runtime] 指令执行异常`，堆涨到 8 MB 上限）。

改为 **`(System.read 睡眠队列 40)` 阻塞睡眠**做节拍（队列无人写入 → 超时返回，进程进
BLOCKED，零 CPU 开销；宿主 harness 里 30000 帧墙钟从 5.5 分钟降到 2.4 秒）。于是三个例子都是
**普通同步代码**：yin-yang 就是那句经典表达式原样跑下去，生成器是"取值—打印"循环，
协程是 20 行的 if 调度器。call/cc 只出现在语义真正需要的地方（每轮 2 次 / 每次产出 1 次 /
每回合 1~2 次），捕获次数随时间**次线性**增长（yin-yang 每轮比上一轮多走一步），
实测 60000 帧（≈16000 格、66 个例子回合）堆只在 170~380 KB 之间锯齿、VM 区稳定，
无错误。跨例子切换用**每轮一次**的逃逸续体（`esc_k`）从深层代码跳回调度器。

## 2. 解释器踩坑（两条都会持续坑后续 App）

1. **续体永不回收**（上面已述）。顺带验证：`System.kill` 杀掉子进程能真正释放它堆里的续体
   （fork + kill 探针：79 KB → 19 KB），所以"续体爆炸"唯一可行的兜底是**换进程**，代价是状态
   全部重来，对演示类 App 不划算。推论：**演示 call/cc 的 App，续体捕获次数必须随时间是次线性的**，
   绝不能拿来当逐帧暂停用。
2. **运行期造短字符串 = 永久泄漏**：解释器把 ≤32 字符的运行期字符串**驻留**（intern），
   驻留表是 GC 根、永不释放。逐帧 `String.concat`/`String.atom_to_string` 的写法
   （turtle/xray 等老 App 都这么写）在长寿命 App 里会稳定涨内存。本 App 的文案全部是
   **字面量常量**（编译期驻留一次），动态数字用 `draw_num` / `draw_numr` **逐位贴数字字模**
   （10 个 "0".."9" 常量 + 定宽 7px 步进 + 右对齐变体），运行期零字符串分配。

## 3. 慢节奏 App 的触屏点按

本 App 每 40 ms 才轮询一次触屏（不是每帧）。`Touch.take_press` 的边沿是**由电平采样派生**
的，只看边沿会整段错过一次快速点按（浏览器鼠标点击 ~5 ms 实测完全抓不到）。改为
"`Touch.is_pressed` 电平 + 一次按下只认一次（`tap_armed`）"，并**按坐标排除右上角「退出」
按钮热区**（那个按钮由宿主在触摸边沿判定，见 exit.scm 的说明），否则点退出会顺带换题。
浏览器回归里用 **160 ms 长点按**模拟真实手指（<60 ms 的合成点击不可靠）。

## 4. 验证

- **宿主回归台**（`tools/e2e/host/harness.c`）：`callcc` 已加入 `run_host_regress.sh` 的
  默认 App 清单。60000 帧无错误、堆锯齿、`fstack` 稳定 5~7、三个例子按序自动轮换；
  点按注入（`HAL_TOUCH_AT`）连按 3 次回到例 1，证明点按换题与 `set_demo` 复位正确；
  三个例子的画面（帧缓冲 → PNG）逐张目检：双色流、两轨格子、换轨竖线、段长标签、
  三套配色/文案全部正确。
- **浏览器**（真 wasm，`tools/e2e/browser/regress_callcc.js`，chromium 1248 headless）：
  `launch=80 → end=80`、`listExit=79`（ESC → 全局确认 → 回车，回列表）、无 `pageerror`；
  三例各 20 s，刷新 250 次/20 s（`TICK_MS=50` 时≈12.5 格/秒）；改 `TICK_MS=40` 后
  实测 155 次/10 s（≈15.5 格/秒，节拍约 64 ms —— 宿主事件循环粒度 16 ms 会再吃掉一点）。
- **内存池下限**：`link_fail` 于 4 MB（VM 工作区 ~2.1 MB 不够）；6 MB 及 12 MB（GUI 默认）正常。
  即 ESP32 的 2 MB 池装不下（与 xray / turtle 同类），属既有约束。

# 2026-10-31（四）多层感知机训练可视化 App（决策面 / 权重热图 / 损失曲线 / 取点概率）

按参考页面 `https://bd4sur.com/html/mlp.html`（2 维输入、5 类样本、全连接网络 + ReLU +
softmax + 交叉熵、小批量 SGD；页面画「损失曲线」与「决策区域 + 样本点」两幅图）实现灵机
Scheme App。编译目标 **Nano-WASM-GUI**，三个无头浏览器（Chromium 1248 /
chrome-headless-shell / Firefox 1555）端到端各 **21/21** 断言通过。**未改动任何
C/HAL/宿主代码**——`ui_scheme_app.c` 扫描业务资源目录 `app/*.scm` 自动发现。

相对参考页面：新增**权重热图**与**点按/拖拽取点（该点 5 类概率）**，损失曲线放大成整幅
（316x53）并给出 loss / acc / epoch / lr 文字行。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `infer/app/mlp.scm` | App 本体（~1140 行）：训练、决策面、权重热图、损失曲线、取点 |
| `infer/app/lib/mlp_data.scm` | 256 个样本（坐标平铺 + 类别下标），**脚本生成，勿手改** |
| `infer/tools/gen_mlp_data.py` | 从参考页面的 `x_array` / `y_array` JS 字面量生成上面的模块 |
| `infer/tools/e2e/browser/regress_mlp.js` | 浏览器端回归（21 项帧缓冲像素断言，含拖拽取点） |

回归清单已同步加入 `mlp`（`tools/e2e/browser/regress.js`、`regress_det.js`、
`tools/e2e/host/run_host_regress.sh`）——`mlp` 按字母序排在 `maze3d` 与 `nongli` 之间，
不更新会让按下标导航的既有回归整体错位一位。

## 2. 界面（320x240）

```
┌ MLP 2-6-6-5  ep 43  [暂停][重置]                    [退出] ┐ 0..18
│ ┌────────────────────┐  权重热图      ▁▂▃蓝→红▃▂▁          │
│ │                    │  W1 b1  W2  b2   W3   b3            │ 18..86
│ │  决策面 160x160     │  (5px 色块矩阵 + 各自的偏置列)      │
│ │  32x32 格 × 5px     ├────────────────────────────────────┤
│ │  + 256 个白圈彩点    │ 该点概率                            │
│ │  + 深色四角框取点标记 │ x -1.54  y -1.37                    │ 88..184
│ │                    │ ■██████████ 99%   （5 类概率条）     │
│ └────────────────────┘                                    │
│ ┌ 损失曲线  loss 0.285  acc 89%  ep 43  lr 0.20 ─────────┐ │ 184..237
│ │  ╲___  （横轴一轮 epoch，纵轴 [0, 全程最大值]）           │ │
└──────────────────────────────────────────────────────────┘
```

- **决策面**：类色块按 `argmax(logits)`（softmax 单调，argmax 不变，省 5 次 exp）；
  样本点用「白底 5x5 + 类色芯 3x3」（与参考页面的白圈彩点同构）。
- **权重热图**：W1(2x6) / W2(6x6) / W3(6x5) 三块并排，每块右侧隔 3px 是它自己的偏置列；
  全体共用 ±max|W| 归一化（蓝负 / 暗零 / 红正），标题右侧是同一色标的 16 段色条。
- **损失曲线**：纵轴 `[0, 全程最大值]`（只增不减 → 轴不抖，收敛段自然贴近底部）。
- **取点**：按下即取点，按住拖动跟着手走（像素移动 ≥2px 才重算）；只重画被标记覆盖的
  那几行（旧位置 ∪ 新位置各外扩 8px），不做全屏重绘——拖拽是逐帧交互，全屏重绘要多花
  ~7 万条指令。

## 3. 算力预算（为什么是这个规模）

实测口径（宿主回归台 + `Diag.tick` 差分，timeslice=2000）：**前向约 29 条 VM 指令/乘加**
（78 次乘加的两隐层前向 ≈ 2250 条），**一步训练（前向+反向+更新）约 7400 条/样本**。

| 项 | 规模 | 每拍代价 |
|---|---|---|
| 网络 | 2→6→6→5（参考页面是 2→6→6→6→5；少一层省 1/3 前向） | — |
| 训练 | 每拍 1 个批量 16 样本（一次 SGD 更新），16 拍 = 1 轮 epoch | ~118k |
| 决策面 | 32x32=1024 格，**每拍只算 1 行**（整图一次算完要 200 万条 = 6 帧） | ~72k |
| 全量评估 | 每拍 16 个样本（给出可读的 loss/acc 曲线） | ~41k |
| 重绘 | 决策行 RLE + 样本点 + 权重热图 + 概率条 + 数字字模 | ~30k |
| **合计** | timeslice 二分实测 **265~275k**（280k 通过 / 260k 变两片） | ≈ 0.85 片 |

单帧预算 31.25 万条 ⇒ **一拍正好一片**：宿主回归台 `refresh=119/120 帧`（1 拍/帧），
浏览器实测 ≈4 轮 epoch/秒（16 拍/轮），**约 5 秒到 loss≈0.44 / 准确率≈88%**。
`lr=0.2`、批 16 是对着「20 轮内成形、之后不震荡」挑的（numpy 复算扫描过 0.03/0.05/0.1/0.15/0.2/0.3
与多个种子）。决策面 32 行要 32 拍才扫完 = **两轮 epoch 刷一遍**：训练在同时推进，所以上下
相隔几行可能差半轮权重——视觉上像「慢慢长出来」（与 mandelbrot / sky3 的渐进渲染同一哲学）。

## 4. 解释器踩坑（本次实测，均已写进 app/mlp.scm 的注释）

1. **wasm32 上 `Math.to_fixed` + `String.atom_to_string` 会多出小数位**（本次新发现，
   影响所有「数字→文字」的排版）：`(Math.to_fixed 0.3429398 3)` 在 wasm 上返回的浮点
   最接近 0.343 的可表示值 ≈0.34299946，而 `atom_to_string` 走 `%g`（6 位有效数字）
   得到 **"0.342999"**（8 个字符），比预期多 3 位——排版按 5 字符算就会撞上后面的文字。
   根因是「单精度 + 尾数截 5 位」的精度（~2e-6）刚好落在 `%g` 的第 6 位有效数字上。
   **对策：不要用 `to_fixed`+`%g` 排版数字**，本 App 改成自己拼定点小数（现为逐位贴
   数字字模），位数恒定、布局可算。
   实测证据：临时把 `atom_to_string(to_fixed(x,3))` 画进帧缓冲，浏览器截图读到
   `tf3=0.342999 / raw=0.342939`。
2. **运行期造短字符串 = 永久泄漏**（照 2026-10-31（三）条目的告诫）：最初把 loss/百分比/
   坐标都用 `String.concat` + `atom_to_string` 拼，逐帧/逐轮都在造新串；而 ≤32 字符的
   运行期字符串会被**驻留**且驻留表是 GC 根、永不释放。改为 **10 个编译期字面量字模
   逐位贴数字**（`DGS` / `draw_int` / `draw_fixed`，与 callcc.scm 的 `draw_num` 同一手法），
   现在**运行期零字符串分配**（也就不再需要 `(native String)`）。验证：12MB 内存池连跑
   20000 帧（1250 轮 epoch）`err=none`。
3. **`'()` 是模块级静态槽**（复测确认）：同一个 `'()` 字面量在**不同** `define` 处是两个
   不同对象（`(push a ...)` 不影响 `b`），但**同一个 lambda 体里**的 `'()` 在多次调用间
   共享——`(define mk (lambda (n) { (define l '()) ... push ... }))` 调 `mk 3` 再 `mk 4`
   得到长度 7。故本 App 的三个大表（决策格缓存 1024 / 损失环 120 / 洗牌表 256）只在
   加载时**一次性** push 构建，此后只 `set_item!`（「重置」也是原地重写，不新建表）。
4. **`Gfx.draw_line` 的坐标是 uint32**：负数会变成天文数字（画不出来或画满屏），
   所有坐标（尤其 `cx-8`、`cy-5` 这类相对量）先用 `fill_rect` 的整型语义或显式判界钳住。
5. **字体度量要实测**：12px 抗锯齿字模的 ASCII 步进约 **6.9px**（不是 6），CJK 12px。
   按 6px 排中文+数字混排会全体错位 10%——本次先用一个探针 App 把 `"0000000000"` /
   `"MMMMMMMMMM"` / 中文串画到帧缓冲量出宽度，再定所有文字的 x 坐标。
6. **算术恒二参数 / 没有一元取负 / `mod` 返回浮点**：与前几个 App 相同；另外
   `(define x 0)` 之类只写在 lambda 体顶层，循环变量全部模块级 `set!`（项目既有约定）。
7. **右上角「退出」按钮会被自己的 HUD 盖掉**（本次实测的界面 bug，容易重犯）：
   `EX.draw_button` 把按钮画在 **y 2..17**（36x16），而顶栏是 App 自己画的整条矩形。
   最初把 `(EX.draw_button)` 只放在「整屏重绘」末尾，于是**每轮 epoch 结束**与
   **暂停/继续**时的那次「只重画顶栏」把按钮涂掉——画面上只剩 y=16 那一行细边，
   按钮等于消失（触屏热区还在，但用户看不见入口）。修法两条并用：
   · 顶栏高度取 **18px** 且按钮画在 y 2..17，让按钮完整落在顶栏内（不再跨到面板上）；
   · 把 `(EX.draw_button)` 放进 `draw_header` 的**末尾**——任何一条重画顶栏的路径
     都会顺带补画按钮并重新登记宿主命中矩形；`draw_all` 里不再单独调用。
   凡是「只重画局部」的 App 都要查一遍：**哪些绘制路径会覆盖 EX 按钮所在的矩形**。

## 5. 随机性与种子

权重初始化与样本洗牌都用 App 内的 **LCG**（`x ← (x·33+7) mod 2048`，状态与乘积累加都
< 2¹⁹，wasm32 单精度下逐位精确），**种子取宿主当前时间戳的低位**（`reseed`）：每次启动、
每次「重置」都从不同的初始权重开始——这正是这个 App 想让人看到的（换一次初始化，决策面
长成什么样）。

- **为什么不用 `(mod ts 2048)`**：`mod` 会把参数转成浮点，而 wasm32 尾数只有 19 位；
  `System.timestamp` 在 32 位宿主上本身已被值编码截到 27 位，再转浮点低位全被抹平，
  `(mod ts 2048)` 只剩少数几个可能值（种子退化，等于「每次同一套权重」）。改用
  `Math.bit_and / bit_lshr / bit_xor` 把时间戳的低 22 位折叠成 `[0,2048)`——位运算全程
  是整数、逐位精确；随后空转 8 步 LCG，让首个输出与种子低位去相关。
- **回归怎么做**：不再逐像素比对。`regress_mlp.js` 的断言全是与种子无关的性质（决策面
  出现 ≥4 类色块、损失曲线出线、暂停后两次采样哈希一致、重置后曲线像素骤减、拖拽标记
  落点与残影）。连续 7 次运行（Chromium×4 / shell / Firefox×2，各自不同的时间戳种子）
  21/21 全通过，且每次 `init` 帧的哈希互不相同（证明确实换了初始化）。
- **为什么这样也稳**：numpy 侧按同一 LCG 播种方式扫了 21 个种子（1 步 97 取整），第 12 轮
  loss ≤0.5（最差 1.02）、第 24 轮 loss ≤0.40（最差 0.68）、决策面 4~5 类——6 秒内必然
  是「已经像样」的画面。
- **宿主回归台仍然确定**：harness 把 `now_ms` 注入成虚拟时钟（`NANO_VNOW` + 16ms/帧），
  那里种子是固定的；也顺手用它验证了「不同时间戳 ⇒ 不同初始化」——`NANO_VNOW` 差 1ms
  就把整帧 CRC 变掉。

## 6. 验证

- **宿主回归台**（`tools/e2e/host`，真图形层 + 虚拟时钟 + timeslice 312500）：
  `mlp` 已加入默认清单，120 帧 `refresh=119`（1 拍/帧）、`err=none`；5000 帧长跑无错误；
  触屏注入（`HAL_TOUCH_AT="...:1:30:150,..."`）验证按下取点与拖拽换点，帧缓冲导 PNG 目检
  （取点标记、概率条、坐标文字、热图、曲线全部正确）；12MB 池 20000 帧无泄漏
  （4MB `parse_fail`——单次分配 1.8MB；8MB `link_fail`——VM 工作区 4MB 不够；12MB 正常）。
- **浏览器**（真 wasm，`tools/e2e/browser/regress_mlp.js`，三引擎各 **21/21**）：
  列表(79) → 启动(80) → 初始（决策面已铺满、样本点 2500 像素、热图蓝红两色）→ 训练 6s
  （决策面出现 ≥4 类色块、损失曲线出线、画面持续变化）→ 暂停（两次采样哈希一致）/ 继续
  （哈希恢复变化）→ 点按取点（深色标记出现、概率面板更新）→ 拖拽取点（标记落到
  (140,140)±10px，且**只留一个标记**——局部重绘没有残影）→ 重置（曲线像素骤减、
  样本点仍在、画面变化）→ 「退出」按钮 + Enter 确认退出到列表(79)，无 pageerror。
- 运行：`cd infer && bash build_wasm_gui.sh`，静态托管 `infer/webgui/` 后打开页面
  （`mlp` 在「灵机应用程序」列表第 13 项）。

## 7. 已知取舍

- 决策面是**渐进刷新**（每拍 1 行，32 拍扫完），不是每拍全图——全图刷新需要 6 帧/次。
- 网络比参考页面少一层隐层（预算决定），故决策边界比参考页面略「方正」。
- 类别 0 是弥漫的大簇（std≈1.0，其余四类 std≈0.4），所以准确率上限约 89~91%：
  这是**数据本身**的性质（该页面同一份数据），不是训练不足。

------------------------------------

# 2026-11-01（沙盒颗粒物理 Falling Sand：106×80 网格 / 8 材质 / 活跃格脏表 / 8 方位重力）

新增灵机应用程序「沙盒颗粒物理」（`infer/app/fallingsand.scm` + 物理内核
`infer/app/lib/sand_core.scm`）：106×80 格的颗粒沙盒，沙 / 水 / 油 / 火 / 植物 / 酸 /
蒸汽 / 石 八种材质，触屏撒料与擦除，重力方向实时取自 `(Imu.gravity_x, Imu.gravity_y)`
—— 把机器倒过来沙就往上流，侧过来就往侧壁流。**未改动任何 C/HAL/宿主代码**：
`ui_scheme_app.c` 扫描业务资源目录 `app/*.scm` 自动发现（当前第 5 项，列表共 31 个 App）。

## 1. 文件与交互

| 文件 | 作用 |
|---|---|
| `infer/app/lib/sand_core.scm` | 物理内核（37KB）：网格 / 材质 / 两张规则表 / 活跃队列 / 单步推进 / 撒料 / 初始场景。只依赖 `(native Math)`，**可脱离 Gfx/Touch/Imu 单独回归** |
| `infer/app/fallingsand.scm` | 界面层：读触屏与姿态、把脏格画出来、顶部 HUD、底部材质面板、退出按钮 |

- **画面**：每格 3px，106×3=318 宽 + 左右各 1px 边框，80×3=240 高，正好铺满 320×240。
- **顶部 HUD（小控件覆在沙面上，不整条遮挡）**：左=当前材质色块（点按展开材质面板）+
  重力方位箭头 + 「活跃格数 / 单步毫秒」读数条；右上=统一的「退出」按钮
  （`app/lib/exit.scm`，点按 → 宿主全局确认框 → 确认/后台/留下）。
- **撒料**：按住拖动即撒（半径 2 的圆盘、隔格撒一半，~6 格/拍）；下落路径上按上一拍
  坐标插值补一格，快速拖动不会画成虚线。选「擦」（材质 0）时是整盘覆盖擦除。
- **材质面板**：底部 44px 面板，10 个 32px 槽（8 材质 + 擦除 + 清屏），点槽即选中并收起；
  面板展开期间世界照常模拟，点面板外只收起不撒料。
- 触屏只在 `y ≥ 20` 才撒料（顶部 HUD 带留白），与 `EX.check` 的按钮热区互不干扰。

## 2. 网格与活跃格脏表（本题的核心）

- **带 1 圈墙环的扁平网格**：`PWW=108 / PHH=82 / NP=8856`，邻居偏移量全是常数
  （`OFF = (108 107 -1 -109 -108 -107 1 109)`），**热路径里没有任何边界判断**；
  边框环上的格子材质恒为 WALL，任何规则都不作用于它，`set_item!` 永远落在合法下标上。
- **活跃队列**（active / dirty table）：只为“上一步变动过的格子”取样，静止时队列为空，
  单步成本≈0（实测 0.01ms）。
  - 队列是**集合语义**：`mk[k]=1` 表示“已在队列里”，出队时清零 → 队列中绝无重复项，
    长度天然 ≤ NP。早先用“代戳 + 追加到下一张表”的写法在预算不够时会无限膨胀。
  - **FIFO + 游标**：`st_qpos/st_qlen` 只是队列头尾下标，预算用完就停在原地（游标保留），
    下一步接着处理 —— 单步耗时有上界，且不会让结转项饿死在队尾。游标走过一个 NP 就
    整体前移压缩（摊还成本 ≈ 每格 2 次操作）。
  - **空格不入队**（`addf!` 里判材质）：一次移动要摸 9 个邻格，若把空格也排进队，
    队列里绝大多数是无事可做的格子——实测 1060 颗沙的堆会把队列膨胀到 3000+，
    每颗沙七八步才轮到一次（沙落得像慢动作）。过滤后队列 = 1060，两拍轮一遍。
- **重排邻域 = 源格的 3×3**（不是目的格）：颗粒一走，只有“源格空了”改变别人的可行性
  （正上方能接着掉、斜上方能斜着滑、左右邻液体能摊进来），这三类全在源格的 3×3 里
  （目的格本身也在其中）；目的格被占住后它周围只会更动弹不得。**只激活目的格邻域时
  “源格正上方”恰好落在 3 行之外，实测会留下悬空的沙**（浏览器/宿主都能复现）。
- **“本拍已更新”戳** `mu[k]=gen`：每格每拍至多动一次（`mu` 命中的目标格拒绝进入，
  防“一步走两格”）；若拒绝原因是“目标格刚被别人占过”而材质本身可进入，则把自己
  重新入队，否则会永久冻结。被 `mu` 跳过（刚被占用/本拍处理过）的格子进**延迟队列**
  `st_def`，下一步开头并入队尾 —— 少了这一步，刚落位的颗粒会被丢掉（同样表现为悬空沙）。

## 3. 重力：8 方位量化 + 换向分片全扫

- `set_gravity(gx, gy)` 用 `tan(22.5°)=0.41421356` 三条带判别量化出 8 个方位之一，
  再查 `OFF` 得到「正下 / 左下 / 右下 / 侧向 / 正上 / 斜上」8 个常数偏移量。
  粉体的三格检查、液体的侧向滑移、气体的逆重力上升全部复用同一套偏移量，
  所以**倒置/侧放不需要另写一套逻辑**。
- 方位一变就置 `st_sweep=1`：每步分片扫描 1100 个格子，把所有非空格子重新入队
  （约 8 步扫完一遍），于是“把机器翻过来”是一次可见的整体重排（像一道波）。
- 液体侧向滑移带**落差判据**（目标格再往下一格也可进入才横move）：否则平地上的一层水
  会永远横着漂，活跃表永不空、稳态成本失控。气体侧向扩散同理看“再往上一格”。

## 4. 两张 10×10 扁平规则表

- `MV`（可移动表）：`MV[我][对方]=1` 表示“我可以与该格交换（沉/浮穿过它）”——
  沙沉水、水沉油、汽浮水…… 移动判据只查这一张表。
- `RT`（交互表）：`RT[我][对方]` 是标志位：1 点燃、2 溶解、4 化汽、8 生长、16 稀释、
  32 窒熄、64 烧开。火→植物/油=点燃；水→火=烧开（火化汽，水隔次也化汽）；
  酸→石/沙/植物=溶解（酸带 ACID_LIFE 寿命，溶完自己也消失）；植物→水=生长；
  水→酸=稀释；汽→火=窒熄。
- 规则只在**反应性材质**（水/火/酸/植物/蒸汽）处理时查表，且每步只抽查 2 个邻格
  （4 组方向轮换、4 步覆盖 8 邻域）—— 斜向接触也能触发，单步成本不变。火是例外：
  寿命短、数量有限，八邻全查（接触即燃，否则火会先飘走）。

## 5. 单步预算（实测）

| 项 | 数值 |
|---|---|
| 步频 | `TICK_MS=33` → **30 步/秒**（与 60fps 渲染解耦；每拍恰好一步） |
| 单步预算 | `MAXPROC=400` 个活跃格（其余结转） |
| 满预算单步（wasm，Chromium，世界内 7000 颗沙） | **6.5 ms** |
| 满预算单步（宿主 gcc -O2，同场景） | 4~5 ms |
| 稳态（队列清空）单步 | **≈0.01 ms**（宿主实测 100 步 1ms） |
| 浏览器刷新节拍（空闲 / 施虐 / 重力换向） | 30.0 / 29.6 / 29.0 Hz（三引擎一致） |

单步耗时是这么量的：把 `TICK_MS` 临时改成 1（刷新节拍 = 单步墙钟），以及用一个临时
量测 App 把 40 步的平均耗时画成红条（1px=0.1ms）再从帧缓冲读回 —— 后者得到 6.5ms，
与“预算打满仍是 30 步/秒”互为印证。

## 6. 解释器踩坑（本次新增，都记进代码注释）

1. **热路径必须宏内联，但宏不能套太深**：一次 Scheme 闭包调用 ≈ 5 次内建操作
   （实测 0.28ms/1000 次 vs 0.06ms/1000 次）。原来的写法每格每步要三十来次调用
   （`addf`×9、`trymv`×3、`mv`、`tch`…），单步 3.7ms；把 `addf!`/`addp!` 改成
   `define-syntax` 宏后降到 3.0ms。**但把 `mv!`/`try!`/`powder!`/`liquid!`/`gas!`
   全做成嵌套宏（宏展开里再套宏）会把编译期内存打爆**：模块连 16MB 池都
   `link_fail`（VM 区 > 8MB）。最终方案：只把**叶子级**的 `addf!`/`addp!` 做成宏，
   策略函数保持为普通 lambda。
2. **`'()` 是模块级静态槽**：7 张 8856 元素的状态表 + 2×8856 的活跃队列全部在加载时
   `(cons 0 '())` + `push` 建一次，此后只 `set_item!`。
3. **`mod`/`/` 恒返回浮点**：像素/网格坐标一律 `Math.floor`/`Math.round` 收口；
   `fill_rect` 的整型参数在 C 侧按 float→int 截断（正值即 floor），所以绘制循环里
   不必再套一层 floor。
4. **运行期不造字符串**（≤32 字符会被驻留且永不释放）：HUD 的数字用 10 个字模
   `"0".."9"` 逐位贴、高位画空格去掉前导零；标签一律是编译期字面量。
5. **模块变量按“谁写谁读”分名**：`p_`（处理一格）、`s_`（分片扫描）、`q_`（撒料）、
   `b_`（建表）、`d_`（方位）、`st_`（步进状态）——跨调用链共用临时量会串味。
6. 算术恒二参数、没有一元取负、`and`/`or` 只用于条件位置、块内只用 `set!`（沿用前几轮）。

## 7. 验证

- **内核契约**（`python3 infer/tools/e2e/native/test_fallingsand.py`，**35/35 PASS**）：
  沙落到底 / 砂量守恒 / 落定后零活跃格；倒置 → 沙流到顶；侧放 → 沙流到侧缘；
  单颗水落地、水堆摊平、水池静止后队列清空；沙沉水（水被顶上来）、油浮水；
  火＋植物（正邻与斜邻都点燃、植物烧尽）、水＋火 → 生蒸汽且火灭；酸溶石、酸寿命；
  蒸汽上升、到期凝结成水；撒料/擦除；单步处理数 ≤ MAXPROC。
- **界面/触屏**（同一脚本，**11/11 PASS**）：初始场景五种材质像素齐备；按住拖动撒料后
  沙落到画面下部；`HAL_IMU_AT` 注入 roll=180 → 沙流到画面上部；全程 `err=none`。
- **浏览器端到端**（`node infer/tools/e2e/browser/regress_fallingsand.js <URL> <engine> <out>`，
  Chromium / chrome-headless-shell / Firefox **各 29/29 PASS**）：真链路
  Pointer Events → `wasm_gui_on_touch` → 生产者回放 → `touch_read` → 撒料；
  DeviceMotion → `wasm_gui_on_orientation` → `hal_imu_wasm`（低通+欧拉角）→
  `Imu.gravity_*` → 物理。断言全用帧缓冲像素（各材质像素数、区域计数、质心）：
  初始场景、撒料增量与落点、材质面板展开/选中并收起、材质按钮变色、清屏、
  植物被烧、酸溶石（石像素 437→218）、侧倾后沙堆到右缘（右缘 936px、底部清空）、
  擦除减水、右上「退出」→ 确认框 → 回列表(79)、无 pageerror。
- **回归**：`run_host_regress.sh` 全 31 个 App `err=none`；浏览器 `regress.js` /
  `regress_det.js` 的 App 清单已加入 `fallingsand`（字母序在 `ephemeris` 与 `flip` 之间）。

## 8. 已知取舍

- 一步一格：粒子速度上限固定为 1 格/步（30 格/秒），没有速度场，所以“沙柱”下落是靠
  逐格推进而不是加速 —— 与经典 falling-sand 一致，也保证了守恒与稳定。
- 预算不足时物理整体降速（而非丢格）：满屏 8000 颗沙翻重力一轮要约 20 步（0.7s）。
  实测正常玩法（几百颗沙在流）始终是 1 格/步。
- 液体只做“侧向滑移 + 落差判据”，没有压力/连通器模型；水池会摊平、会从缺口流下，
  但不会自动找平到同一个高度。
- 顶部 HUD 的三个小面板会遮住最上面 6 行浮灰（很小，且在最边上）；材质面板展开时
  遮住最下 15 行（收起后即时重绘）。
- **内存**：9 张 NP 长度的状态表 + 2×NP 的活跃队列 ≈ 9.7 万元素（约 0.8MB 堆），
  WASM/桌面的 12MB 池余量充足（加载后 VM 1.8MB / 堆 4.4MB）；**ESP32 的 2MB 池装不下**
  （与 xray / turtle 同量级），若要在 MCU 上跑需把网格降到 ~40×30 或换成分块网格。

------------------------------------

# 2026-10-31（五）2048：4×4 滑动合并 + 撤销 + 自动存档（纯逻辑 / 零物理 / 事件驱动）

新增灵机应用程序「2048」（`infer/app/2048.scm` + 逻辑内核 `infer/app/lib/g2048_core.scm`）：
4×4 棋盘、触屏滑动手势（自己从触点位移判方向）、撤销栈、自动存档（浏览器侧真正落
IndexedDB）。**未改动任何 C/HAL/宿主代码**——`ui_scheme_app.c` 扫描业务资源目录
`app/*.scm` 自动发现（现列表第 1 项，共 32 个 App）。

这个 App 是「单帧极少 native 调用」的样板：**纯逻辑、零物理**。静止时每拍只采样一次
触屏电平（约 4 次本地库调用）、只有按下/松手/按方向键的那一下才真正算一步并重绘一屏；
没有定时动画、没有粒子、没有浮点累积。棋盘状态是 16 个整数，撤销是 17 个整数的快照，
存档是 19 个整数的一行文本。

## 1. 文件

| 文件 | 作用 |
|---|---|
| `infer/app/lib/g2048_core.scm` | 逻辑内核（~600 行）：棋盘（**指数**表示）、四方向合并、随机落子、终局判定、撤销栈、存档文本序列化/反序列化。只依赖 `(native Math/String/System)`，可脱离界面单独回归 |
| `infer/app/2048.scm` | 界面层（~570 行）：布局、绘制、滑动手势、节拍、文件读写、浮层 |
| `infer/tools/e2e/native/g2048_logic.scm` | 内核单测 **107 条断言**（宿主回归台直接跑） |
| `infer/tools/e2e/native/test_g2048.py` | 一键：构建宿主回归台 → 跑内核单测 → 界面/触屏像素断言（**56 条**） |
| `infer/tools/e2e/browser/regress_2048.js` | 浏览器端到端（**24 条**断言；Chromium / chrome-headless-shell / Firefox 三引擎各 24/24） |

回归清单已同步加入 `2048`（`host/run_host_regress.sh`、`browser/regress.js`、
`browser/regress_det.js`）——**注意它按 `wcscmp` 排在列表第 1 项**（`'2' < 'c'`），
按下标导航的既有脚本必须把 `2048` 插到最前面，否则整体错位一位。

## 2. 表示与算法（内核）

- **指数而不是数值**：`bd[i] = 0` 空格，`k ≥ 1` 表示数值 `2^k`。合并 = 相邻两个 k 变
  `k+1`（一次加法）；"是否相等"就是整数比较（wasm32 单精度下 2^18 以内逐位精确，永远
  不会出现 2048 与 2047.999 的浮点相等性问题）；分数用一张 2 的幂常量表取值，全程整数。
- **四方向共用一套合并算法**：`LINE_IDX` 是 4 方向 × 4 条线 × 4 格 = 64 项常量表，
  每行/列按"**靠拢端在前**"给出扁平下标（下标本就是 `r*4+c`，全在编译期算好，
  运行期零下标除法）。逐线做"压缩 → 相邻相同者合并（只并一次）→ 补零"。
- **动效来源表 `an_src`**：每格方块记"从哪一格滑过来"（-1 = 新生成的）。合并时记
  **滑过来那一枚**的来源（原位那一枚看上去是被它吃掉了），界面层据此做滑动动画。
  内核只管如实记录，动画完全在界面层。
- **撤销**：每步先记一份 17 个整数（16 格 + 分数）的快照；走不动就把快照丢掉
  （`unsnapshot`，不算一次撤销）。深度上限 10 份，超了从头部 `cdr` 掉最旧一份。
- **存档**：`"<16 个指数> <score> <best> <won>\n"`。反序列化逐码点扫一遍，遇分隔符
  把累积值压进词表——**不造任何中间字符串**；取值为 0 的格子也照常入表，故用
  "是否在数字中"标志而不是"值非零"来切词。指数越界 / 词数不足 19 一律判为损坏
  （App 转开新局，不截断使用）。

## 3. 手势与交互（界面）

- 每拍一次 `Touch.is_pressed` + `Touch.x/y`：按下沿记起点 `(gsx,gsy)`，按住期间刷新
  最近触点 `(glx,gly)`，松手沿算位移 `(dx,dy)`：
  `max(|dx|,|dy|) ≥ 16px` → 主轴方向走一步；否则算点按 → 只认面板/浮层按钮。
- **为什么不用 `take_press/take_release` 边沿**：手指滑动的物理时长 100~300ms，跨 3~9 个
  tick（33ms/拍），电平采样必能采到"按住中"的位置；电平 + 坐标这套对慢节奏 App 更稳
  （边沿锁存只报一次，一旦某拍漏采就整段丢失）。快速点按（<1 帧）确实采不到，但本 App
  的按钮都是人手点按（≥80ms），实测 110ms 合成点按稳定命中。
- **右上角「退出」热区（x ≥ 268 且 y ≤ 20）里的按下不参与手势**：否则点退出会被判成
  一次滑动（`EX.check` 会吞掉边沿，但电平式手势照样看得见）。同理，浮层打开时手势全部
  让位给浮层按钮。
- 方向键（浏览器物理键盘 / 掌机实体键，`(native Key)` 的 1/2/3/4）作为等价输入，
  端到端测试与桌面操作都用它替代滑动，省去手势抖动。
- **滑动动画**：一步用 4 拍（≈130ms）把方块从来源格插值到目标格（`p = (AN_N-an_t)/AN_N`），
  期间每拍 `dirty=1`；到 0 那一拍再画最后一帧（正好落在格子里）。合并后的方块从被吃掉
  的那一枚滑进来——观感与经典 2048 一致。
- **浮层**：达成 2048（一次性，确认后写进存档的 `won`，重启不再弹）与无路可走（启动时
  若读回的存档已是终局也会弹）。各两枚按钮：左「新局」、右「继续」/「撤销」。
- **自动存档**：每次有效移动置脏，静止 20 拍（≈0.7s）后落一次盘（不是每步都写），
  退出确认后由 `(EX.on_exit do_save)` 再补一次——刚走完立刻退出也不丢。启动时读回，
  文件不存在/损坏就开新局。浏览器目标写入经 `hal_fs_persist_wasm.c` 同步进 IndexedDB，
  **刷新页面仍在**（e2e 里就是这么验的：走一步 → 等存档 → reload → 进 App → 局面一致）。

## 4. 实测

| 项 | 结果 |
|---|---|
| 单拍成本（静止：EX.check + 触屏采样 + 键盘排空） | 4~6 次本地库调用，无重绘（宿主回归台 3000 帧 `refresh=1`） |
| 一屏重绘 | 1 次整屏 + 16 次空格底 + 每枚方块 2 次（底色 + 数字字模），典型 60~80 次调用 |
| 滑动一步（含 4 拍动画） | 5 次推屏；宿主回归台 30 帧内完成 |
| 端到端（三引擎） | 启动/读档/左滑合并/四方向/位移阈值/方向键/撤销/新局/自动存档/刷新后仍在/胜利浮层/退出确认 **各 24/24** |
| 内存 | 12MB 池：加载后 VM 约 1.4MB；存档文本反复序列化/反序列化 900 轮后堆稳定在 2.5MB（无线性增长） |

## 5. 解释器踩坑（本次新增，都记进代码注释）

1. **模块导出名不带模块前缀**：内核里定义 `move`/`cell`/`over`，界面侧写 `G.move`、
   `G.cell`。本次初版把内核函数起名 `g_move`，界面写 `(G.cell …)` → `am_link` 直接返回
   NULL，**没有任何错误信息**（`[Runtime]`、`[Parser Error]` 都没有），只有
   `RESULT link_fail`。定位手法：按"顶层形式边界"把源码截断成前缀逐段试，
   第一个失败的那一段就是问题所在（本次定位到 `draw_tiles` 里的 `G.an_src`）。
   → 教训：**看到 link_fail 先查 `别名.名字` 是否与定义名完全一致**（前缀、大小写）。
2. **`(native …)` 是逐模块的**：内核用了 `(System.timestamp)` 却只在界面层声明过
   `(native System)`，编译链接都通过，运行到 `reseed` 时才报
   `[Runtime] call: 变量 System.timestamp 未定义` + `指令执行异常`。
   → 每个模块自己写全 `(native …)`。
3. **深层 `{…}` 块的收尾最容易数错**：lambda 体的最后一个表达式本身是块时，
   结尾是 `}})))`（2 个花括号 + 3 个右圆括号）。初版写成 `})))` —— 圆括号平衡、花括号
   少一个，编译期才报错。可靠写法：**每个块的 `}` 单独占一行，再在外层收 `))`**：
   ```
   (define f
     (lambda ()
       {
         (if c
             {
               …
             }
             #f)
       }))
   ```
4. **查表别忘"空格子上限"**：`VALS[0]` 是 1（2 的 0 次幂），所以"数值和"这类统计
   必须先判空格再查表（初版把 16 个空格当成 16 个 1，守恒断言直接 48→58 露馅）。
5. **像素判据要按 framebuffer 的色深做最近邻**：宿主回归台是 RGB565 缓冲，
   RGB888 → RGB565 → RGB888 往返后每个通道会偏 0~7，`dist2 < 12` 这种严格阈值全部失效。
   改用"与调色板做最近邻 + 容差 60"，并注意**别让探针区域扫到别的文字**——
   本次浮层边框探针初版扫了 x 50..270，把右侧面板标签色 (120,130,150) 当成边框
   （与边框色 (122,132,152) 只差 14），"浮层已消失"的断言因此假失败。
6. **`regress.js`（存活回归）的导航是坏的，本次顺手修了**：旧写法每轮只按一次
   `ArrowDown`，而退出 App 回列表时选中项复位到第 0 项 → 第 2 轮起永远重开列表第 1 项，
   于是除首个 App 外全部 `refreshΔ=0`（假 ABNORMAL）。改为"每轮从首项按下 i 次"
   （与 `regress_det.js` 同一写法）后，32 个 App 的 `refreshΔ` 立刻恢复真实值，
   ABNORMAL 只剩本来就"无输入不重绘"的 calculator/cnn/gomoku/qrcode/svm。

## 6. 验证命令

```bash
cd infer
python3 tools/e2e/native/test_g2048.py          # 107 条内核断言 + 56 条界面断言
APP=2048 FRAMES=120 ./tools/e2e/host/run_host_regress.sh   # 宿主回归台（err=none / refresh=1）
bash build_wasm_gui.sh                          # 出 wasm 产物（App 源码拷进 linglong/app + 重生成 manifest）
cd webgui && python3 ../tools/https_static_server.py 9002 .
cd ../tools/e2e/browser
node regress_2048.js https://127.0.0.1:9102/index.html chromium /tmp/e2e-2048
ENGINES="chromium shell firefox" ./run_browser_regress.sh      # 全量存活/确定性回归
```

------------------------------------

# 2026-11-01（二）弹幕陨石 App（程序化 sprite：飞船 / 多边形陨石 / 爆炸粒子，触屏拖动 + 自动开火 + 波次难度）

新增灵机应用程序「弹幕陨石」（`infer/app/danmaku.scm`，~1800 行 / 42KB）。
**零位图资源**：飞船、陨石（多边形 + 旋转）、爆炸粒子、子弹、星空全部用
`fill_rect` / `fill_circle` / `draw_line` 现拼。编译目标 **Nano-WASM-GUI**，
三个无头浏览器（Chromium 1248 / chrome-headless-shell / Firefox 1555）端到端各 **25/25**
断言通过；过程中查出并修掉两个**真 bug**（`graphics.c` 的反向线段死循环、`SCHEME_APP_MAX`
装不下第 33 个 App），App 源码放在 `app/` 即自动进列表。

## 1. 文件与玩法

| 文件 | 作用 |
|---|---|
| `infer/app/danmaku.scm` | App 本体：程序化 sprite、触屏拖动、自动开火、波次/难度、粒子、HUD、结局浮层 |
| `infer/tools/e2e/browser/regress_danmaku.js` | 浏览器端到端回归（32 项帧缓冲像素断言；`tools/e2e/` 不入库） |
| `infer/graphics.c` | **修 bug**：慢路径反向水平/垂直线死循环（见 §3.1） |
| `infer/ui_scheme_app.c` | **修 bug**：`SCHEME_APP_MAX` 32 → 48（见 §3.2） |

- **画面**（320x240）：顶部 HUD 19px（波次 / 得分 / 生命图标 / 右上角统一「退出」按钮），
  下方是游玩区（深空底色 + 46 颗三档亮度星点做视差）。
- **飞船**：触屏拖动跟随（飞船跟在**指尖上方 26px**，不被手指盖住；平滑系数 0.42/拍，
  x 夹在 [12, 308]、y 夹在 [128, 212]）；自动开火（间隔 `max(78, 122-5×波)` ms，
  第 4 波起三连发）。
- **陨石**：两种——「下落型」从顶部落下、撞左右壁反弹；「横掠型」（第 2 波起，概率随波次
  上升）从左右边缘在飞船高度带横穿，**垂直弹幕打不到它**，这是"不能一直蹲着"的关键机制。
  大陨石（r ≥ 12）被击碎分裂成两颗小的（横向散开）。
- **难度**：下落速度 `52 + 7×波 + rand×38` px/s、刷怪间隔 `max(300, 780-70×波)` ms、
  每波数量 `4 + 2×波`、射速随波次提高；3 条命，撞毁后 1.8s 无敌（闪烁）+ 红色边框闪烁；
  命尽 → 「游戏结束」浮层（得分 / 最佳 / 点按屏幕重开）。
- **弹幕强度按钮（HUD 140..192，y 2..17；默认「常规弹幕」）**：点按在「常规弹幕」/「无限弹幕」
  之间切换，底色即状态（石板灰 = 常规 / 青 = 无限）。切模式顺手把 `fire_t` 清零，切换后不会
  立刻喷一发。两枚按钮的命中判定合并成一个通用 `hit_rect`，`ui_toggle` 记 1=生命 / 2=弹幕；
  **落在任一按钮上的按下都不参与操船、也不触发结局态的重开**。
  常规弹幕 = 原来的行为：1 条弹道（第 4 波起 3 条、间隔 9px）、射速间隔 `max(78, 122-5×波)` ms。
- **无限弹幕：宽度/射速随波次增长（无上限，天花板是屏宽与同屏弹数）**：一发「齐射」= `st` 条弹道，
  以飞船为中心按 15px 向两侧展开（超出游玩区就夹边）：
  `st = 1 + floor((波-1)×1.5)`，上限 `BSTREAM_MAX = 1 + floor((SW-12)/15) = 21`（= 屏宽放得下
  的条数）；射速间隔 `max(20, 118-7×波)` ms，**允许低于一拍**（`while` 一拍内最多打 3 轮齐射，
  且 `fire_t` 封顶 3×间隔，避免卡顿后连喷）；弹速 `min(620, 390+15×波)`。
  子弹表因此从 96 项单独放大到 **384**（实测同屏峰值 ~294）；其余定长表仍是 96。
  实测（宿主回归台，飞船停在中列不动，同样 6000 帧 ≈ 第 11 波）：**常规 3 条弹道 / 386 像素，
  无限 18 条 / 1932 像素**；无限弹幕第 3 波 4 条、第 48 波 21 条（封顶）/ 同屏 294 发；饱和时
  单拍 VM 指令 ~60k（70k 时间片仍 1 拍/帧，40k 时略掉帧），相对 312500 余量充足。
  注：HUD 因此变成「波/分 | 生命图标(仅三条命) | 常规弹幕 | 无限命 | 退出」，
  生命图标挪到 94..136，四段都不重叠；得分 5 位数时 47..82，6 位时到 89（仍不压按钮）。
- **生命模式按钮（HUD 中部 146..190，y 2..17）**：点按在「无限命」/「三条命」之间切换，
  **默认无限命**（撞毁只放烟花 + 红框闪烁 + 1.8s 无敌，不扣命、不结束）；切到「三条命」时
  命数复位成 3，命尽才弹「游戏结束」浮层。按钮底色即状态（青 = 无限 / 琥珀 = 三条命），
  生命图标只在三条命模式下画（无限命时扣不了命，图标只会一直亮着）。
  实现要点：①「按下沿」用 `Touch.is_pressed` 电平 + `prev_down` 自算（`ui_tap`），命中按钮
  矩形则置 `ui_toggle`，**这一下不参与操船、也不触发结局态的重开**；②`y ≤ HUD_H` 的触摸
  一律只当按钮用（否则点按钮会把飞船拽到上界）；③切换不碰 `over`/`playing`，所以结局态
  下也能先切模式再点屏幕重开。
- **退出**：右上角按钮 / 硬 A/ESC → 宿主全局确认框（`app/lib/exit.scm`）。

## 2. 图形技法（本题的核心）

**陨石 = 旋转的 8 边不规则多边形 + 扫描线实心填充**：

1. 单位顶点表 `UX/UY` 在**加载期**算好（3 种形状 × 8 顶点；顶点方向角 45° 步进 + ±6° 抖动
   `JA`，半径系数 `UF` 0.76~1.12），运行时**每颗陨石只算一次** `(ca,sa)`（2 次三角函数），
   之后是纯乘加：`x = cx + r·(ux·ca − uy·sa)`、`y = cy + r·(ux·sa + uy·ca)`，
   结果写进模块级 `vx8/vy8`（逐颗复用，不分配）。
2. 填充：逐行（步长 1）收集扫描线与 8 条边的交点（半开区间判据，顶点不会被算两次）→
   插入排序 → **奇偶规则成对 `fill_rect`**。用奇偶而不是"最小/最大跨度"，凹进去的缺口
   才不会被填平（半径抖动 ±20% 时肉眼可见差别）。
3. 每行按"距中心的高度"取三档颜色（中心上方最亮 → 下方最暗）= 顶光，几乎零成本；
   8 条边用 `draw_line` 描亮边，2 个 `fill_circle` 陨坑（位置随多边形一起转）。

**飞船**：15 行 `fill_rect` 拼三角机身（半宽查表 `HW`），行色沿 i 线性渐变
（机头亮青 (150,240,255) → 机尾深蓝 (52,114,185)）；座舱 `fill_circle`；尾焰两枚
`fill_rect`，长度每帧用 `(rnd)` 抖动 = 火焰闪烁。

**爆炸**：两种粒子——火星（`pkd=0`，颜色按剩余寿命比走 白黄 → 橙 → 暗红，尺寸 2px → 1px）
与**中心闪光**（`pkd=1`，半径随寿命收缩，三层同心 `fill_circle` 橙/亮黄/白）。
定长表 + swap-remove，上限 96 个。

**数字**：10 个编译期字模逐位贴（`DGS` + `draw_int`），运行期零字符串分配
（沿用 mlp/callcc 的手法）。

## 3. 解释器 / 宿主踩坑（两条是真 bug，已修）

### 3.1 `graphics.c` 慢路径的**反向**水平/垂直线会死循环（真 bug）

`gfx_draw_line` 的慢路径（非 RGB565 帧缓冲 / `mode>=4`）原来是：

```c
int32_t delta = y2 - y1;
for (int32_t y = y1; ((delta >= 0) ? (y <= y2) : (y >= y2)); ...)   // y 是 int32，y2 是 uint32
```

C 的常用算术转换会把 `y` 提升成 `uint32`：反向竖线（`y1 > y2`）递减到 `-1` 后变成
`4294967295`，`y >= y2` 恒真 → **循环永不结束**（水平线同理）。修法：两端先转 `int32`
再比较（`graphics.c` 两处，附注释）。

为什么此前从没暴露：**RGB565 有行直写快路径，直接 return，走不到慢路径**。
本机 gcc 回归台是 RGB565，所以永远复现不出来；而 **WASM GUI 的帧缓冲是 RGB888**，
于是浏览器里"整页卡死"（30~60 秒内必现，随机）。表现是 Playwright 报
`Target page, context or browser has been closed` / `page.evaluate` 超时，且卡死后
25 秒内不恢复。

**定位手法（值得复用）**：

1. 用 `-O2 -g` 重编 GUI（wasm 带 name 段），CDP `Debugger.enable` + `Debugger.pause`
   在卡死时暂停 → `Debugger.paused` 给出 wasm 调用栈**函数名**：
   `gfx_draw_line ← am_native_Gfx_draw_line ← op_callnative ← am_runtime_tick …`。
2. 在 `am_native_Gfx_draw_line` 入口打印每次调用的四个坐标（临时桩，跑完还原），
   卡死前最后一条就是罪魁：`(275,9)-(275,0)`、`(181,8)-(181,0)` —— 正是陨石多边形
   那条 `y1 > y2` 的竖直边。
3. App 侧另加 `line_xy` 归一化（垂直/水平线段一律按坐标**升序**交给 `Gfx.draw_line`）
   做双保险，不依赖宿主某条路径的实现细节。

顺带排除了几条错假设（记录下来省后人的时间）：`MATH` 返回 NaN、`ymax` 巨大、
分裂/粒子表越界——都不是；**wasm32 版宿主回归台**（下面 §4 的 node 版）连跑 40 个
种子 × 5000 拍也复现不出来，因为它的帧缓冲也是 RGB565。

### 3.2 `SCHEME_APP_MAX` 装不下第 33 个 App（真 bug）

`app/*.scm` 已达 33 个，而 `ui_scheme_app.c` 的上限是 32：扫描按名排序后**静默丢掉最后
一项**（当时是 `xray.scm`，列表里凭空少一项、无任何提示，浏览器端 App 下标也跟着错位）。
32 → 48，并在注释里写清"数量变化时务必核对这里与 `browser/regress` 的 App 清单"。

### 3.3 `regress_det.js` 的 `APPS_ALL` 漏了 `callcc`（既有缺陷，已修）

该脚本按"在 `APPS_ALL` 里的下标"导航（`ArrowDown` 按 index 次）。漏一项 → `cnn` 之后的
所有行都测成了**别的 App**（实测 `danmaku` 行拍到的画面其实是 `cnn` 的）。补上 `callcc`
后 danmaku 行 refresh=83、CRC 三次运行完全一致。**加 App 时这两个清单都要同步。**

### 3.4 多元算术仍然是"静默算错"（复现）

`(+ SW r 6.0)` 得到 **14**（应为 334）—— 横掠型陨石于是永远在屏幕**外**生成、下一拍就被
回收（表现为"横掠陨石从没出现过"）。写完后务必扫一遍：`(+ - * /)` 是否都只有两个参数。
本次用一段 ~20 行的 Python 括号/参数计数脚本（不入库）扫出 3 处。

### 3.5 子弹在屏幕上方就命中（本次初版实测）

子弹回收条件写成 `y < 0` 时，子弹会在 **HUD 之上**打中刚生成的陨石：命中特效与加分全部
发生在看不见的地方（前 20 秒只加了 10 分，看起来像"打了不中"）。改为游玩区上沿
（`y < HUD_H` 即回收）后立刻正常。

### 3.6 爆炸闪光的下标

`addpart` 内部已把 `npart` 加一，闪光本身是 `npart-1`；先写成 `npart` 等于把**下一颗**
火星标成闪光（半径取到火星速度 → 偶尔画出一个大圆盘）。

## 4. 算力与实测

- **单拍 VM 指令数**：开局 ≈**2.5~3 万**（`NANO_TIMESLICE` 二分：30k 时 refresh 已到顶，
  40k~312.5k 无差别）；波次饱和（21 条弹道 / 同屏 ~294 发）时 ≈**60k**（24000 帧长跑：
  50k 略掉帧、70k 起完全无差别）。相对 312500 的时间片余量仍有 4~10 倍。
- **节拍**：`TICK_MS=33`；宿主回归台 `refresh=40/120 帧`（与 tetris / touchgravity 同口径）；
  浏览器实测 **30 Hz**（2.5 s 内 epoch +75）。
- **宿主界面/难度回归**（`tools/e2e/native/test_danmaku.py`，**20/20**，确定性）：初始画面五类
  判据、两枚按钮各三态、常规弹幕高波次仍封顶 3 条、无限弹幕 3 → 18 条 / 386 → 1932 像素、
  无限命 24000 帧不出现结局浮层、三条命同长跑会出现（浮层底色 16096 px）、退出两段式收尾。
- **浏览器端到端**（`regress_danmaku.js`，三引擎各 **32/32**）：列表(79) → 启动(80) →
  HUD 条 / 「第N波」横幅 / 星空 / 飞船在下半屏 / 自动开火有子弹 / 陨石在屏 →
  弹幕强度默认常规（石板灰）→ 切无限弹幕（青底）→ 切回 → 生命模式默认无限命（青底、无图标）→
  切三条命（琥珀底 + 三枚图标）→ 切回 →
  拖动飞船左（x≈109）右（x≈209）、夹边（x 夹到 12 / y 夹到 128）→ 子弹跟着飞船那一列（远侧无残弹）→
  画面持续演化、陨石从顶部进入并穿过游玩区 → 得分区像素变化（说明打掉了东西）→
  右上角「退出」→ 确认框 → Enter 回列表(79)，无 pageerror。
- **宿主回归台**（`run_host_regress.sh`）：**33 个 App 全部 err=none**；danmaku 追加
  20000 帧长跑无错误。
- **确定性**：`regress_det.js` 冻结时钟下三次运行 CRC 完全一致（`3b8c176b`，95 次刷新）。
- **内存池**：**≥10MB**（8MB `load_fail`；10MB/12MB/16MB 均正常）。GUI 默认 12MB 可用；
  ESP32 的 2MB 池装不下（与 xray / turtle / fallingsand 同类）。
- **顺手搭的 wasm32 版宿主回归台**（不入库，纯排查用）：用 `emcc` 把
  `tools/e2e/host/harness.c` 编成 node 可跑的 wasm（`-Dmain=harness_main` + `-sINVOKE_RUN=0`
  + `-sMODULARIZE=1`，再用 `ccall` 传 argv），得到**与浏览器同语义**（float32 / 32 位 uint）
  的可 printf 调试环境：3000 帧 ≈ 1 秒墙钟，一次扫 40 个 `NANO_VNOW` 种子 × 5000 拍。
  注意它仍是 RGB565，复现不了 §3.1。

## 5. 难度与手感（有意为之）

- **蹲坑不再安全**：只靠垂直弹幕时，玩家停在一个列里几乎无敌（自己把会撞到自己的陨石全
  打掉了）。横掠型陨石从侧面横穿飞船高度带，垂直弹幕打不到 → 必须挪位躲开，或者把弹幕柱
  移到它的航路上。实测：飞船停在 x=160 不动，第 5 波左右三条命耗尽；一路扫射移动能撑到
  7 波以后。
- **得分只在"真的打中"时产生**：击碎 10（小）/20（大）分，清场奖励 5×波次。
- 波次横幅开局停 2.4s（首屏能读到「第 1 波」），后续波次 1.3s。

## 6. 已知取舍

- 陨石是"扫描线 + 每行 `fill_rect`"，没有三角形填充/纹理贴图；陨石上限 20 颗，观感够用。
- 未做：飞船侧倾（需要旋转 sprite）、音效、连击倍率、最高分落盘（`best` 只在本次会话内）。
- 整屏重绘（一次背景 `fill_rect` + sprite），未做脏区局部刷新（同 tetris / fallingsand 口径）。
- 一个反直觉的观察：`draw_ship` 里的尾焰抖动用了全局 LCG，于是**绘制会推进随机序列**；
  本 App 不依赖"绘制不推进 RNG"，但以后若要做可重放的确定性演示，得把视觉抖动改用
  时间派生（`gt`）而不是 `(rnd)`。
- 回归脚本里别用"屏幕左上角"的触点验证拖拽：实测 x<20 且 y<100 那一小片区域里，
  宿主那边的点按分支会把事件吃掉（`Touch` 电平根本到不了 App，飞船停在原地不动），
  Chromium/Firefox 都能复现。夹边验证改用 (10,150) 验 x 夹、(160,60) 验 y 夹。

## 7. 验证命令

```bash
cd infer
APP=danmaku FRAMES=120 ./tools/e2e/host/run_host_regress.sh     # 宿主回归台（err=none / refresh=40）
bash build_wasm_gui.sh                                          # 出 wasm 产物（含 App 源码同步 + manifest）
cd webgui && python3 ../tools/https_static_server.py 9002 .
cd ../tools/e2e/browser
node regress_danmaku.js https://127.0.0.1:9102/index.html chromium /tmp/e2e-danmaku
node regress_danmaku.js <url> shell|firefox /tmp/e2e-danmaku   # 三引擎各 32/32
```

------------------------------------

# 2026-11-01（三）手写汉字识别 App（纯 Scheme 识别器 + 1000 类线性分类器）

新增灵机 App `infer/app/hwr.scm`：手指在绘图板上写一个汉字，**抬笔静置 420ms** 即识别，
给出 16 个候选（两页各 8 个），点按候选上屏到文本行（可退格/全清）。
识别算法、特征、分类器、权重读取全部在 Scheme 侧，**零 C/HAL/宿主改动**
（唯一动到构建的是 `build_wasm_gui.sh` 增加一行数据文件拷贝）。
WASM GUI 目标编译与浏览器端回归通过（Chromium / chrome-headless-shell / Firefox 各 28/28）。

## 1. 为什么是这个模型（规模倒推）

灵机解释执行，wasm 上一次乘加约 40 条 VM 指令。实测（原生宿主，`/tmp` 微基准）：

- 模块级变量做热循环变量：**0.49 µs/乘加**（`op_load` 走闭包环境链查找）
- 改成 **lambda 形参 + 函数体顶部 define**（帧槽位）：**0.16 µs/乘加 —— 快 3 倍**

据此定规模：1000 类 × 256 维 = 25.6 万次乘加 ≈ 0.3 秒（WASM）/ 67ms（原生宿主）。
再往上（类数或特征维数翻倍）就要排队等 0.6 秒以上，体验不可接受，故：

```
输入  墨迹网格 64×64 → 墨迹包围盒 → 双线性重采样到 26×26 → 居中放进 32×32
特征  4 方向梯度（上下/左右/两对角中心差分取绝对值）在 8×8 网格上分块求和
      → 256 维，L2 归一化（对笔画粗细、墨迹浓淡不敏感）
分类  256 → 1000（最常用汉字，字频降序）线性 softmax
      权重 int8 逐类量化（每类一个尺度，精度损失 < 0.5%）
```

试过、被否掉的：单隐层 MLP（256→96/128→1000）在同样训练量下严重欠拟合（train 11%），
而且参数 12~16 万并不比线性少多少；coarse-to-fine 剪枝（簇均值当粗模型）真类簇召回
只有 60%（k=2），会吃掉 10 个点以上准确率；8 方向/更大网格的特征维度翻倍、参数与耗时同步翻倍。

## 2. 权重为什么是「数据文件」而不是 Scheme 源码

灵机解析器在 `am_parse` 里按 **源码字符数** 预分配 token 数组：
`max_tokens = wcslen(code) + 1`，`am_token_t` 在 wasm32 上 40 字节（原生 48）——
25.6 万字符的字符串字面量要吃掉 **10MB**，而 App 的解释器内存池总共 12MB（实测直接
`link_fail`：`VM segregated 分配失败: 请求 14622328 bytes`）。

故权重导出成 `app/lib/hwr/*.txt` 数据文件，App 启动时 `(Fs.read)` 读入：

- 每个权重字节写成**码点 256+q 的字符**（UTF-8 两字节），读进来就是一个字符串，
  热循环直接 `String.charCodeAt` 随机访问，**不需要解码成列表**（省 1MB 堆 + 0.3s 启动）
- 每个文件 200 类 × 256 维 = 51200 字符 = 102400 字节（`AM_NATIVE_FS_READ_MAX` 是 128KB，
  类不跨文件，故内层循环按文件切换权重串、下标从 0 重新起算）
- `params.txt` 装 1000 个尺度（每类 2 字符 16 位定点）+ 1000 个偏置 + 1000 个汉字；
  `meta.scm` 是元信息（类数/维数/量化尺度），只有 6 个 define
- 量化偏移技巧：权重串里存的是 q+256，于是 `Σ(q+256)f = Σqf + 256Σf`，
  合并后 `score = scal*(Σ charCode*f - 384*Σf) + bias` —— 内层循环里**没有减法**，
  偏移项每类只算一次

踩坑记录（值得记下来）：**三元 `(+ a b c)` 会静默算错**，不是崩溃。`draw_viz` 里
`(+ FMAP_X (* d FMAP_P) (* j FMAP_S))` 让第一张特征图跑到了 x=0（本该 44），
画面看着"就是不对劲"却没有任何报错。写了个 s-表达式扫描器（把操作数是 3 个的
`+ - * / < > <= >= ==` 全列出来）才发现仅此一处。AGENTS.md 早先记的"多参数算术会崩"
在这次的宿主上表现为**算错**而不是崩溃，两者都要防。

## 3. 训练数据：字体渲染 + 强增广，不依赖真人笔迹数据集

`infer/tools/train_hwr_hanzi.py`（~330 行，numpy + PIL，无第三方 ML 库）：
12 种字体风格（Noto Sans/Serif CJK 的多个字重、LXGW 文楷、AR PL UKai、马善政/志莽行、
站酷快乐/庆科黄油、Droid Sans Fallback）× 1000 字 × 16 次增广 ≈ 19 万样本。

增广（关键在**分布要贴手指书写**）：随机长宽比、笔画粗细（膨胀/腐蚀）、弹性形变、
旋转/错切/缩放、墨迹浓淡、随机污点（撑大包围盒）、随机摆位与大小；
另有 **45% 的"手指模式"样本**：细笔画 + 手抖 + 倾斜 + 缩到 32×32 后按 0/1 格半径
"笔刷封样"（模拟绘图板一笔一格采出来的粗笔迹）。

这一项对准确率影响最大：同样的分类器，训练集里**没有**手指模式时，"手指仿真"测试集
top-1 只有 52%，加上之后 68%（早期数据）→ 最终 60%（更难的测试集）。

评估（脚本自动打印，并写 `app/lib/hwr/train_report.json`）：

```
未见字体（uming / NotoSans-Black / droid，全不参与训练）: top1 68.1%  top5 80.6%  top8 83.0%
手指仿真（上面两字体 × 笔画宽度随机）                : top1 59.5%  top5 76.3%  top8 79.5%
训练 100 epoch（线性 softmax + Adam，lr 0.03，batch 256）：train_acc 77.7%，约 250 秒
```

**Adam + 权重衰减(L2) 会互相打架**：Adam 的步长与梯度大小无关，L2 那点恒定收缩力会被
归一化放大，权重反而被钉在很小的值上（实测 |W| 停在 0.13、train_acc 卡在 37% 不动）。
本项目全部训练都取 `l2=0`。

## 4. 交互设计要点

- **多笔画字不能一抬笔就识别**（汉字平均 8 笔）：抬笔后静置 `PEND_MS=420ms` 才认为
  "一个字写完"；期间再落笔 = 同一个字的下一笔（不清屏），超时后落笔 = 下一个字（清屏）。
  计时用 `System.timestamp`（宿主时钟）而不是 `Diag.ms`：宿主回归台的虚拟时钟按帧推进，
  用真实墙钟量"静置 420ms"会让脚本不可复现。
- 拖动笔画期间**零推理开销**，只按脏矩形重画墨迹格（`draw_pad_dirty`）。
- 推理前先画一帧「识别中…」再推屏，避免 0.3 秒的空白等待。
- 底部常驻两块"教学"可视化：归一化后的 32×32 输入（1px/格）+ 4 张方向梯度特征图
  （8×8、6px/格、按各自最大值着色）。

## 5. 验证

```bash
cd infer
python3 tools/train_hwr_hanzi.py            # 生成数据→训练→评估→导出（约 6 分钟，16 核）
bash build_wasm_gui.sh                      # 产物 webgui/（会自动同步 app/ 与 app/lib/hwr/）
cd webgui && python3 -m http.server 9124 --bind 127.0.0.1 &
cd ../tools/e2e/browser
node regress_hwr.js http://127.0.0.1:9124/index.html chromium /tmp/e2e-hwr   # 28/28
node regress_hwr.js <url> shell|firefox /tmp/e2e-hwr                        # 三引擎各 28/28
APP=hwr FRAMES=200 ./tools/e2e/host/run_host_regress.sh                     # refresh=1(err=none)
```

- **浏览器端回归（`regress_hwr.js`，28 条）**：全程只用帧缓冲像素判据 + 触屏链路——
  进入列表→启动 App→绘图板画「一」→8 个候选格都有字形 / 归一化视图有墨 / 特征图有色 /
  top-1 字形是"一条横（≤4 行、≥8 列有墨）"→点候选上屏（**上屏的字形位图必须与刚才
  top-1 的位图逐位相同**）→画「十」（两笔；跨笔画墨迹必须合并，top-1 变成含竖笔的字形）
  →换成「工」（候选随之变化）→候选翻页→退格→清屏→Esc 退出。
- **识别正确性**另在**原生宿主回归台**上用临时 `(display ...)` 打印 top-8 逐字核对
  （20 个手绘折线字：一/二/三/十/人/大/口/日/中/上/下/工/王/木/小/山/田/七/女/本，
  top-1 命中 11/20，其中 口→图、日→目/曰、中→申、上→业 都是"同类字族"的合理混淆，
  top-3 内命中 15/20）。这些折线是"LLM 想象的手写"，比真人笔迹更不像，属于悲观下界。
- 识别耗时（WASM，Chromium，界面显示）：**约 300ms**；原生宿主 67ms。
- 解释器内存池：App 源码 32KB + 5 个 100KB 权重文件（各读成一个 51200 字符的字符串）
  + `params.txt`，12MB 池充裕；`app/lib/hwr/` 全部是 manifest 的 priority 1。

## 6. 已知限制 / 有意取舍

- 1000 类（最常用 1000 字，覆盖现代汉语约 91% 字频）；一级字库 3755 字做不到
  （参数 4 倍、耗时 4 倍）。
- 训练数据全是**字体渲染**，没有真人笔迹数据集（CASIA-HWDB 之类要注册/几十 GB）。
  因此对"笔画顺序无关但字形潦草""连笔书写"的鲁棒性有限：实测 handwriting 草书字体
  （Long Cang）几乎全错，属于预期。
- 归一化用"包围盒拉伸填满 26×26"：写得歪/大/小都能救回来，但**偏旁比例**会被拉伸，
  上下结构写得太散时容易误判成别的字。
- 候选区的"置信度"是**候选集合内**的 softmax（top-16 归一化），不是全 1000 类的概率；
  1000 类的 exp 归一化对用户没有额外信息，徒增 1000 次 exp。
- 没有做"整句语言模型纠错"（手写输入法常见的下一步）：平台放不下 n-gram 表，
  但 top-8 里通常有正确字，点一下即可。

------------------------------------

# 2026-11-01（Markdown 阅读器：解析 / 排版 / 渲染全用 Scheme）

新增 `infer/app/markdown.scm` + 两个库模块，**完全用灵机 Scheme** 实现 Markdown 的解析、
排版与渲染（WASM-GUI 目标验证；顺带修了 GUI 宿主一处输出丢弃问题，见第 4 节）。
动手前重读了 `~/Animac/doc/manual.md`，并逐条对照了本文件里历轮记录的 App 编写约束。

## 1. 分层与文件

| 文件 | 职责 | 依赖 |
|---|---|---|
| `app/lib/md_parse.scm`（~600 行） | 文本 → **块树**：块级规则表 + 行内规则表 | 仅 String/Math |
| `app/lib/md_layout.scm`（~470 行） | 块树 → **显示行**（行 = 高度/背景/段列表；段自带绝对 x 与宽度） | 仅 String/Math |
| `app/lib/md_demo.scm` | 内置示例文档（**行数组**，不是大字符串——灵机字符串字面量不能跨行） | 无 |
| `app/markdown.scm`（~430 行） | 绘制 + 交互（滚动/翻页/点链接/退出） | Gfx/Key/Touch/Fs/Ui/App/EX |

支持的子集：`#` 1~3 级标题、`**加粗**`/`__加粗__`、`` `行内代码` ``、无嵌套无序列表
（`-`/`*`/`+`）、`[链接](url)`、围栏代码块（``` / ~~~，剥起始缩进）、分隔线（含 `- - -`）、
段落（CJK 软换行、避头点）。另外支持 `\*` 转义、收尾 `#`、`1.` 之类有序列表留给后续。

## 2. 三个扩展点（"为完整 Markdown 留口"落到具体代码位置）

1. **块级规则表** `block_rules`：每行 `(种类 谓词 处理函数)`，谓词只看单行，
   处理函数 `(lines 下标) → (块 下一行下标)`，段落是兜底规则。新增块类型 = 一行
   `(rule_add ...)`，**解析主循环不用改**；顺序即优先级。
2. **行内规则表** `inline_rules`：每行 `(触发字符 匹配函数)`，匹配函数
   `(s 下标) → #f 或 (节点 下一下标)`。未命中任何规则的字符进文本缓冲，
   因此任何字符都能安全作触发符（斜体/图片/删除线各加一行即可）。
3. **块排版表** `block_layouts`：每行 `(种类 排版函数)`。排版函数把块变成若干
   `(高 背景 段列表)` 的显示行。

**表格**（本轮不实现，但两端已备好）：解析端在 `md_parse.scm` 文件尾给了
「表格接入指南」（谓词只看当前行的约定下，前瞻放在处理函数里），排版端
`lay_table` 已实现并**单测通过**——它把 `(table 表头行 数据行)` 排成等宽分栏的显示行
（表头加粗 + 表头下横线 + 单元格超宽 … 截断）。因为段自带绝对 x，绘制层一行都不用改。

## 3. 关键设计：排版期就把坐标算死

- **行 + 段**（而不是"块 + 位置"）：绘制端只按 y 累加往下画，完全不懂 Markdown；
  滚动、翻页、链接命中测试全部退化成算术。
- **字宽纯算术算**：实测 12px alpha 字模是等宽的（半角 7px、全角 12px，见 `glyph_12.h`），
  因此排版宽度与绘制结果**逐像素一致**，不必向宿主要度量（Gfx 也没有度量接口）。
- **折行**：先把行内节点摊平成"原子流"（半角成词、全角逐字、空格单独成原子），
  再贪心填行；相邻同样式且首尾相接的段会并成一段（减少 draw_text 次数）。
  避头点（`。，、；：？！）」』】`）宁可悬挂 12px 也不换行，否则会出现整行只有一个「。」。
- **层级表达**：字模只有一套 12px、没有粗体字形，所以「加粗」= 同一行 1px 偏移重复描。
  描 3 次对汉字会糊成一团（试过，行首 H1 变模糊），故最多描 2 次；标题层级改用
  颜色亮度 + 段前留白 + 分隔线（H1 2px 通栏、H2 1px、H3 无线且更暗）表达。这是这套
  字模下的取舍，写在 `markdown.scm` 的配色注释里。

## 4. 顺带修的宿主缺陷：GUI 目标的 `(display ...)` 被丢弃

灵机程序的 `(display ...)` 不直接写 stdout，而是压进 `rt->output_fifo`，**由宿主取出落地**
（`animac_core.h` 的 `am_runtime_output`）。`ui_scheme_app.c` 此前只排空了 `error_fifo`，
于是 Scheme App 的调试输出在 GUI 目标（掌机/浏览器）上人间蒸发（`tools/e2e/browser/regress_hwr.js`
里那个 `recog` 变量一直是空的，就是这么回事）。本轮新增 `app_drain_output_fifo()`
（每帧在 `app_drain_error_fifo()` 之后调用，按 UTF-8 打印），端到端测试因此可以断言
App 打印的诊断行；其余 App 几乎不用 `display`（仅 callcc/infix 错误路径），代价可忽略。

## 5. 解释器池与文档大小上限（实测）

Markdown 阅读器是"整篇解析 + 整篇排版"，显示行约占 **1KB/行**（可见+不可见全驻留）。
`md_parse + md_layout + md_demo + exit` 四个模块一起编译，编译期工作区就吃掉约 13MB，
于是 12MB 池下连 8KB 的文档都装不下。处理：

- `ui_scheme_app.c` 的非 ESP32 池 **12MB → 24MB**（浏览器 INITIAL_MEMORY 384MB、桌面充裕；
  ESP32 仍是 2MB）。实测 24MB 下 31.8KB 文档（678 块 / 2260 显示行）可排版完成。
- App 侧 `DOC_MAX = 30KB`：超过就先给可读提示（"文档过大（N 字节）：本阅读器一次排版整篇…"），
  而不是让用户看到 `[Runtime] 指令执行异常`。`Fs.read` 自身还有 128KB 上限，也一并兜住。

## 6. 验证

- **解析器单测** `tools/e2e/markdown/test_parse.scm`：**57/57 PASS**
  （标题 1~6 级与 `#不是标题`/7 个 `#` 反例、CJK 软换行不插空格 vs 拉丁插空格、`**`/`__`/`` ` ``/
  链接/嵌套加粗/未闭合标记当普通字符/转义、围栏（含未闭合、缩进剥离、`~~~`）、
  分隔线三种写法、列表（续行、`*`/`+`）、split/join 往返与 CRLF）。
- **排版器单测** `test_layout.scm`：**42/42 PASS**（字宽与 clip、折行不超宽、首行贴满、
  H1/H2 分隔线高度、加粗/行内代码/链接样式与 url、标题内加粗不降级、代码块底与折行、
  项目符号 + 14px 悬挂缩进、表格排版钩子（手工构造块验证）、total_height/line_top、
  连排两次结果一致＝无模块级状态泄漏）。
- **App 端到端（宿主回归台，真图形层）** `test_app.py`：**28/28 PASS**
  （文档列表高亮 → 打开 .md 后 MDSTAT 规模 → 标题亮像素/药丸底/链接下划线/项目符号/
  代码块底/分隔线铺满 → 键盘下翻（画面+百分比变化）→ 触屏拖动 → 点按链接弹窗（MDLINK）
  → 内置示例直接打开）。
- **浏览器端到端（真 WASM + 真触屏）** `tools/e2e/browser/regress_markdown.js`：**29/29 PASS**
  （MEMFS 写 `doc/*.md` → 真 `Fs.read` 链路 → 上述像素判据 + 退出确认后回列表 + 模态框
  开关后画面复原 + 无 pageerror）。截图见输出目录（`read.png` 等）。
- **回归**：宿主 35 个 App 全 PASS（新增 markdown 一行）；浏览器存活回归 35/35
  `launch=80 → end=80`、无 pageerror。注：`regress.js` 的方向键导航在长列表上仍会串位
  （本文件早前记录的既有问题），逐 App 的截图归属以专用脚本为准。
- 浏览器实测：31KB 文档（525 块 / 1750 显示行 / 高 16450px）在 24MB 池下正常滚动渲染。

## 7. 本轮踩到的坑（对以后写任何 Scheme App 都有用）

1. **`(cons 'bold inlines)` 会把节点摊平**：`(bold . 节点列表)` 在"元素 1 是节点列表"
   的约定下是错的，下游 `get_item nd 1` 拿到的其实是**第一个节点**，随后 `get_item`
   越界，报出来的却是含糊的 `[Runtime] 指令执行异常`（栈上操作数已错位）。
   正确写法 `(mk2 'bold (parse_inlines content))`。为定位它花了很久——
   教训：AST 形状一旦约定，构造处一律用 `mk<n>` 显式建表，不要用 `cons` 拼"头 + 表"。
2. 同一条 `display` 出的 `(bold (text c))` 是**歧义**的打印结果（节点列表恰好只有一个节点时），
   看输出无法区分"摊平"与"未摊平"，得看构造代码。
3. **`(set! i 末值)` 当 break 会毁掉结果变量**：`lead_ws` 原本用 `(set! i n)` 跳出循环，
   于是"无前导空白"也返回整行长度（首字符不是空白 → 直接把 i 设成 n）。一律用 go/stop 标志。
4. **`cond` 是关键字，不能作形参名**（`(lambda (name cond) ...)` 编译期报
   `lambda parameter must be variable`）。
5. **`(list ...)` 不是内建**：测试里构造表格块用引号字面量 `'(table (("a" "b")) (...))`。
6. 宿主回归台的 `STDOUT<<...>>` 是**排空 output_fifo** 后的结果，所以纯逻辑单测在
   桌面解释器与 harness 上都能跑；GUI 目标要等第 4 节的排水补丁才看得到。

------------------------------------

# 2026-11-01（四）手写汉字识别 v2：在线笔迹特征 + 3755 类 + 真实笔迹训练

新增 `infer/app/hwr2.scm`（与 v1 `hwr.scm` **并存**，v1 未改动）：改用**在线（轨迹）特征**
与 **CASIA-OLHWDB1.1 真迹**训练，类数从 1000 扩到 GB2312 一级字 **3755**。
训练脚本 `infer/tools/train_hwr_online.py`，模型数据 `infer/app/lib/hwr2/`（1.1 MB）。
调研与选型见 `doc/手写识别技术路线调研.md`。

## 1. 识别流程（每步都能在灵机里跑）

```
采轨迹（每 20ms 一点，绘图板 128px = 64 格）
 → 细线墨迹图 64×64（按段双线性铺点，笔宽不参与特征）
 → 伪二维归一化：行/列线密度投影等化映射（alpha=0.7 与恒等混合）后重采样
 → 8 方向梯度（中心差分 + 相邻方向软分配）→ 8×8 分块 → 512 维
 → 投影到 128 维（PCA＋类内散布对角白化，白化折进投影矩阵）
 → 最近类均值（NCM）判 3755 类 → 前 16 名候选（两页各 8）
```

**NCM 是关键简化**：训练只需类均值（分钟级、无梯度下降），推理只有 K×D 乘加，
且能 int8 逐类量化。在官方 512 维在线特征上的实测（`doc/手写识别技术路线调研.md` §5）：

| 分类器（3755 类，官方在线特征） | top-1 | top-5 |
|---|---|---|
| **最近类均值 NCM** | **81.5%** | **93.6%** |
| 线性 softmax（15 万样本） | 74.3% | 91.0% |
| 对角协方差高斯 | 76.8% | 91.8% |
| 文献 MQDF（PR 2013） | 93.2% | — |

## 2. 实测精度（自研特征，**未达到**官方特征的量级）

| 配置（3755 类，CASIA 测试书写人 60 人） | top-1 | top-5 |
|---|---|---|
| 自研 512 维 + NCM（App 输入仿真） | 57.4% | 77.8% |
| PCA-256 + 对角白化 + NCM | 60.6% | 80.4% |
| **PCA-128 + 对角白化 + NCM（本 App 采用）** | **58.5%** | **79.1%** |
| PCA-64 + 对角白化 + NCM | 47.8% | 70.6% |

- "App 输入仿真"= 把高精度笔迹降级为 64 格量化 + 2.5 格抽稀（真机手感），训练与评估都这么做。
- int8 量化后不变（58.41%/79.05%，按导出文件重建模型实测）。
- **与官方 512 维在线特征（NCM 81.5%）差 23 个点**——那 512 维是 CASIA 的
  "伪二维双矩归一化 + 8 方向"专有实现，我用等化映射近似，达不到同样效果。这是本路线
  的已知上限差距，后续可换用更精确的归一化或小型 CNN（见调研报告 §7 路线 C）。

**输入分布的坑（重要）**：本版模型对"理想折线"（每笔一条直线，例如程序生成的
`一`=一条横线）不鲁棒——20 个手绘折线字只对 2 个；但**回放真人书写轨迹**时
20 个字对 10 个、14 个进 top-8（与离线 58% 的评测一致）。曾试图用"折线简化"增广
（straight_p=0.45）弥补，结果 CASIA 精度从 58.5% 掉到 35.5%（类均值被糊掉），已回退。
因此 e2e 测试改为**回放真实书写轨迹**（`tools/e2e/browser/regress_hwr2.js` 内嵌了
6 个 CASIA 样本的轨迹常量）。

## 3. 性能（WASM，Chromium 实测，界面显示）

**约 700 ms**（原生宿主 220 ms：特征 85ms + 投影 20ms + 判类 115ms）。判类 3755×128 = 48 万乘加，
是恒定瓶颈；想再快可上两阶段剪枝（估计 ~150ms）或减到 1000 类（~250ms）。
纯 Scheme 侧的关键性能点依旧是「热循环用 lambda 形参/局部量」（比模块级变量快约 3 倍）。

## 4. 解释器踩坑（本次新增，均已在代码内注释）

1. **`(mkn 2 NG 0.0)` 这类多传参数**：`mkn` 只接受 2 参，多传的第 3 个参数被忽略、
   第 2 个变成初值，于是 `ftab` 只有 2 个元素 → 后续 `set_item!` 越界 → 表现为
   `[Runtime] 指令执行异常`（无具体原因），极难定位。教训：自己写的定参函数也要数清参数。
2. **`(list a b)` 不存在**（manual 已记"没有 list"）：两点坐标改用
   `(cons x (cons y (cdr '(0))))`。
3. **局部 define 用于嵌套循环的计数器容易踩坑**：`(define s 0)` + `while { while {…} (set! s …) }`
   在最小复现里会**挂死**（首个调用即挂）；同样的写法在另一些形状下又正常。
   本次把特征提取一侧全部改为模块级变量（安全），只在 classify/project 这两个死循环里
   保留局部量（与 v1 同款写法，实测无问题）——这也是性能所需（见 §3）。
4. **角度→方向编号必须与训练侧一致**：`mod(atan2, 2π)/(2π)*8`，用 `(ang+π)/(2π)*8` 会整体
   旋转 4 个方向、并在 bf≥8 处被 clamp 到同一格，表现为"特征里大量格为 0、识别全错"。

## 5. 验证

```bash
cd infer
python3 tools/train_hwr_online.py            # 载入 /tmp/potr|/tmp/pot（见脚本头注释下载地址）
                                             # → 特征 → NCM 评测 → 导出 app/lib/hwr2/
bash build_wasm_gui.sh                       # 产物 webgui/
cd webgui && python3 -m http.server 9131 --bind 127.0.0.1 &
cd ../tools/e2e/browser
node regress_hwr2.js http://127.0.0.1:9131/index.html chromium /tmp/e2e-hwr2   # 25/25
node regress_hwr2.js <url> shell|firefox /tmp/e2e-hwr2                        # 三引擎各 25/25
```

e2e 断言（25 条，纯帧缓冲像素）：进入列表→启动 App→回放「一」→候选 8 格/归一化视图/
方向响应图都有内容、top-1 字形是"一条横"→点候选上屏（**上屏字形位图与 top-1 位图逐位相同**）
→回放「十」（两笔）→回放「口」（三笔）→翻页→退格→清屏→Esc 退出。

## 6. 数据结构（`app/lib/hwr2/`，1.1 MB，manifest priority 1）

- `p0/p1.txt`：PCA 投影矩阵 512×128，**行对齐**（每文件 400 行 = 51200 字符 = 102400 字节 <
  `AM_NATIVE_FS_READ_MAX` 128KB），int8 存为码点 256+q。
- `m0..m9.txt`：类均值 3755×128，同样每文件 400 类、行对齐。
- `params.txt`：每类尺度（2 字符 16 位定点）+ 每类 |m|²（2 字符）+ 3755 个汉字。
- `meta.scm`：类数/维数/分段大小/量化尺度（6 个 define）。
- 为什么是数据文件而不是源码：见 `hwr.scm` 头注释（灵机按源码字符数预分配 token 数组）。

## 7. 数据许可（务必注意）

CASIA-OLHWDB/HWDB 为**学术研究免费**（2020 起无需申请表单）但**禁止商用**，
商用需向 CASIA/NLPR（Cheng-Lin Liu）申请授权。本 App 的模型权重由该数据训练而来，
随仓库分发属"研究用途"灰区；若需商用必须重新训练或取得授权。引用：
Liu et al., CASIA Online and Offline Chinese Handwriting Databases, ICDAR 2011；
Liu et al., Pattern Recognition 46(1):155-162, 2013。

------------------------------------

# 2026-11-02（Markdown 阅读器：补上表格解析与渲染）

上一轮把表格的**排版端**（`lay_table`）留成"已就绪、待接入"，这轮把**解析端**补齐并把两端
一起做实：GFM 表格语法、三种列对齐、单元格内行内样式、按内容自适应的列宽、单元格内折行。

## 1. 解析端（`app/lib/md_parse.scm`）

块形状定为 `(table aligns header cells)`：
- `aligns` —— 每列对齐码（0 左 / 1 中 / 2 右，来自分隔行的冒号）；
- `header` —— 表头的单元格列表（一行）；
- `cells`  —— 数据行列表，行 = 单元格列表，**单元格 = 行内节点列表**
  （所以格子里可以再写 `**加粗**` / `` `代码` `` / `[链接](url)`）。

判定规则（全部集中在 `table_start` / `h_table` / `split_cells` / `delim_aligns`）：
1. 表头行含 `|`，**下一行是合法分隔行**（每格只由 `-` `:` 空格组成且至少一个 `-`）；
2. **表头与分隔行的格数必须一致**——这条是关键：否则「一行带竖线的普通文字 + 下面一行
   `---`」会被误判成单列表格，而它在 Markdown 里应当是「段落 + 分隔线」（GFM 同样要求列数匹配）；
3. 行首/行尾的 `|` 造成的空单元丢弃；`\|` 是字面竖线（反斜杠原样保留，交给行内转义规则）；
4. 数据行在空行、无 `|` 的行、或围栏代码块处结束；
5. 列数取表头与所有数据行的最大值，缺格由排版端补空（不报错）。

**谓词签名升级**：`(rule_add 种类 谓词 处理函数)` 的谓词从"只吃单行"改成 `(lines 下标)`。
原因就是表格：它必须前瞻一行才能判定；只给单行就只能"先猜再回退"，会把普通段落里
偶然出现的一个 `|` 切断（旧写法下实测会把一段切成两段，多出一个段间距）。
处理函数仍然必须**总能推进下标**，这条约定不变。

## 2. 排版端（`app/lib/md_layout.scm`）

- 折行逻辑抽成 `pack_segs`（返回"段列表的列表"，不落地成显示行），`pack_atoms` 变成它的
  落地封装。于是**单元格复用与正文完全相同的折行**（含避头点、空格丢弃、风格合并）。
- 列宽"先量后分"：每列取所有格子的自然宽，夹在 `[TABLE_MINW=26, TABLE_MAXW=120]`；
  自然宽总和放得下就用自然宽（表格只占需要的宽度，不硬拉通栏），放不下就按比例压到正文宽，
  每列不低于下限，再对余数逐 1px 削减（`shrink_to_fit`）。
- 对齐：`align_shift`（右 = 吃掉全部余量、中 = 余量一半、左 = 0），单元格左右各留
  `TABLE_PAD=2` 内缩，文字不贴竖线。
- 行高 = 该行各列子行数的最大值（长单元格自然把整行撑高，后面的列只出现在第一子行）。
- 新增样式 `ST_TABLE_SEP`（表格列间竖线）：段的文本为空、宽 1，绘制端按行高画 1px 竖条，
  于是竖线跨子行天然连成一条；表头用 `BG_TABLE` 淡底 + 表头下方 1px `BG_RULE` 横线。
  绘制端**依然不知道"表格"是什么**——它只认"段"，这正是上一轮设计的目标。

## 3. 顺带的两处调整

- 绘制端 `draw_seg` 增加一个"行高"参数（表格竖线要画满整行高度）。
- 内置示例文档（`md_demo.scm`）加了两张表：一张演示三种对齐 + 单元格内行内样式，
  一张演示长单元格折行；块数 16 → 21、显示行 74 → 105、总高 766 → 1119。

## 4. 验证

- **解析单测 87/87 PASS**（新增 30 条）：基本表格、可省首尾竖线、三种对齐码、非法分隔行→段落、
  段落里的竖线不切段、`---` 仍是分隔线、格数不匹配→段落+分隔线、空行/无竖线行结束、
  单元格内加粗/代码/链接、`\|` 不切格且还原、格数不齐仍是一张表。
- **排版单测 54/54 PASS**（新增 12 条）：行数构成、表头底色与横线、竖线段样式、左/右对齐
  的列边缘、单元格行内样式透传、列宽和 ≤ 正文宽、长单元格撑高行高、8 列窄屏能排版。
- **App 端到端 31/31 PASS**、**浏览器端到端（真 WASM）31/31 PASS**：新增表格表头底色与
  列间竖线的像素判据（表头淡底 4066px、竖线 90px），截图见 `/tmp/e2e-tbl*/read.png`。
- **回归**：宿主 35 个 App 全 `err=none`；浏览器端 App 存活回归与前两轮一致。

## 5. 这轮的坑

1. **格数一致性不能省**：初版只判"下一行是分隔行"，于是 `文字 | 更多` + `---` 被吃掉成
   一张单列表格，本该出现的分隔线消失。加一格数比较即解决，并补了回归用例。
2. 测试里的下标又错了一级（`(thd tbl 0)` 已经是"单元格列表"，取第 0 格才是行内列表）——
   AST 层级一多就容易差一层，这轮的表格用例统一加了 `hcell_text` / `bcell_text` 两个
   取值辅助函数，断言里不再手写四层 `get_item`。
3. `get_item` 越界返回 `#undefined`（不报错），但随后 `String.equals` 等原生调用吃到
   `#undefined` 才是"指令执行异常"的真凶——测试里这类失败要先怀疑取值链多/少了一层。

------------------------------------

# 2026-11-02（Markdown 阅读器：示例落盘成 .md、按文件打开；扩展到资源根）

需求：「阅读器要能打开文件系统里的 Markdown，并把现有示例写成 md 文件再打开」。
这轮把示例从"只存在于内存里的字面量"改成**首次启动落盘成真文件、之后一律按文件打开**，
顺带把可发现的文件范围从 `doc/` 扩到资源根。

## 1. 数据流（首次启动 vs 之后）

```
boot → seed_demo_file()      // doc/Markdown 示例.md 不存在才写（Fs.exists → Fs.write）
     → scan_docs()           // doc/ 下的 .md，再扫资源根 "."；一个都没有才用内存里的内置示例
     → 1 篇直接 open_doc(0)，多篇进列表
open_doc → Fs.size 上限检查 → Fs.read → MD.parse → MDL.layout → 绘制
```

- **落盘内容只有一处真源**：`md_demo.scm`（在 git 里，行数组格式）。用
  `MD.join_lines` 拼成文本写出去，不额外维护一份 `.md` 副本，升级时不会两边漂移。
- **不覆盖已存在的文件**：用户可以直接把 `doc/Markdown 示例.md` 当便签编辑，
  重启后看到的是自己改过的内容（实测：改成 4 行小文档 → 重启仍是 2 块，没有 MDSEED 行）。
- **写失败（只读文件系统）**：`Fs.write` 返回 `#f` 时保留内存里的内置示例兜底，
  列表里显示「内置」标记（`(mk2 "" SEED_TITLE)`，path 为空串）。
- **资源根也扫**：`doc/` 之外，`<资源根>/*.md` 同样能发现（`(Fs.list ".")`；注意
  `Fs` 的路径规则拒绝空串，所以用 `"."` 而不是 `""`）。这解决了"临时想打开一篇 md
  却不知道要放哪个目录"的问题。
- 文档列表超过 9 项会自动滚动（`list_first()`：窗口首项 = max(0, sel-8)），
  页眉右侧显示「N 篇」，避免文档一多就画到屏外。

## 2. 顺带修的一个坑

`Fs.list ""` 会**失败**（`am_native_Fs.c` 的 `fs_resolve` 明确拒绝空路径，返回 `#null`），
初次实现扫描资源根时踩到——表现为"放在根目录的 md 看不见"。改用 `"."`（解析成 `<root>/.`，
`opendir` 正常）。已写进测试 H 用例。

## 3. 验证

- 宿主端到端 **40/40 PASS**（新增 9 条）：
  - F：删掉 `doc/` 后启动 → 打印 `MDSEED ... written` → 磁盘上出现 `doc/Markdown 示例.md`
    且内容就是内置示例（2229 字节，首行 `# Markdown 阅读器`）→ 直接打开、`title=Markdown 示例`、
    块数 21；
  - G：把该文件改成 4 行小文档后重启 → **不再落盘**（无 MDSEED 行）、按文件内容渲染（2 块）、
    文件内容仍是用户写的那份；
  - H：资源根放 `root-note.md` → 列表里出现 2 篇，选中后打开 `title=root-note`。
- 浏览器端到端（真 WASM）**32/32 PASS**：新增断言「App 启动后 MEMFS 里出现
  `doc/Markdown 示例.md`（1037 字符，首行 `# Markdown 阅读器`）」，其余用例改为
  按"文档列表 = doc/ 下 .md 名字升序（含 seed）+ 资源根"计算下标。
- 解析单测 87/87、排版单测 54/54 不受影响；宿主 35 个 App 回归全 `err=none`。

## 4. 备注（给以后改这块的人）

- 种子的文件名即标题（`strip_ext`），改 `DEMO_PATH` 会同时改界面标题；
- 想换示例内容只改 `md_demo.scm`；已落盘的旧副本不会自动更新（有意为之：不覆盖用户文件）。
  真要强制刷新，删掉资源目录里的那个 .md 再启动即可。

------------------------------------

# 2026-11-02（二）文件管理器 + 可复用文件选择器（灵机应用程序层）

需求：做一个**完全权限**的文件管理器（系统资源根以下所有文件：列表/删除/重命名/新建文件
（默认 .txt + 时间戳名）/新建目录，为复制与移动留扩展空间），并在其之上实现**可复用的
文件选择器 UI 组件**（Scheme 应用层），供 Markdown 阅读器等"打开文件"用。
本轮只针对 WASM 目标（`build_wasm_gui.sh`，不是 `infer/web/` 那条 LLM-only 路线）。

## 1. 分层与新增文件

| 文件 | 作用 |
|---|---|
| `infer/hal_fs.h`、`hal_fs_linux.c`、`hal_fs_m5esp.cpp` | HAL 新增 `platform_remove_path` / `platform_rename_path`（Linux：POSIX `remove`/`rename`；ESP32：`SD.rmdir`/`SD.remove`/`SD.rename`，**未编译验证**） |
| `infer/am_native_Fs.c/.h` | 库过程 `(Fs.remove path)`、`(Fs.rename old new)`；WASM 落盘钩子 `am_native_fs_on_remove`；`Fs.write ""`（空文件）修复 |
| `infer/hal_fs_persist_wasm.c` | 删除/改名同步进浏览器 IndexedDB（否则刷新后"已删文件"会从 store 复活） |
| `infer/app/lib/fsx.scm`（新） | 文件系统小工具库：`join/leaf/parent`、**按名字升序排序**（`sort_names`/`sort_pairs`）、12px 文本宽度与裁剪、尺寸格式化、扩展名过滤 |
| `infer/app/filemgr.scm`（新） | 文件管理器 App（操作表驱动：绘制/命中/键盘全按表走） |
| `infer/app/lib/filepicker.scm`（新） | 文件选择器组件：`FP.start/start_at`、`FP.active`、`FP.step`、`FP.result`、`FP.reset` |
| `infer/app/markdown.scm` | 底栏右侧新增「打开」按钮 → 唤起选择器（`open_doc` 抽出 `open_path`，两条入口共用一条打开链路） |
| `infer/vendor/Animac/VENDOR.md` | 更新本地库清单里的 Fs API 一览 |

## 2. 语义与设计取舍

- **"完全权限"的范围 = 业务资源根**（`PLATFORM_ROOT_DIR`）以下的**全部**文件与目录，包括
  `icon/ app/ doc/ data/` 这些系统自己在用的目录；边界即根本身——根不可删/不可改名
  （`am_native_Fs.c` 的 `fs_is_root_path` 在原生层兜一道，`.` 与绝对路径两种写法都认），
  也没有 `..` 越界（沿用 `(native Fs)` 的路径规则）。三个平台同一套路径语义。
- **排序在 Scheme 层做**（`FSX.sort_names`，按 Unicode 码点）：`list_files` 的顺序各平台
  不保证（Linux 为回归稳定按 strcmp 排过，ESP32 是 SD 卷顺序），需求要求"字母升序"，
  放在 App 层才三平台一致。目录在名字后加 `/` 并用蓝字区分（文件白字 + 右侧尺寸）。
- **递归删除**：`Fs.remove` 只删文件/空目录（POSIX `remove` 与 `SD.rmdir` 的原语边界），
  目录树由 `filemgr.scm` 的 `remove_tree` 先递归清空再删自身；删除前弹宿主两按钮确认框
  并写明"及其全部内容"。
- **移动 = 重命名**：同一资源根内 `(Fs.rename ...)` 即是移动——重命名框里输入
  `目标目录/新名字` 即可跨目录搬。改名**拒绝覆盖**（目标已存在返回 `#f`，不像 POSIX
  `rename` 那样静默覆盖），这是"文件管理器"该有的安全语义。
- **复制/移动的扩展空间（两处，落地时不用改别的代码）**：
  ① 操作表 `build_ops` 是唯一按钮清单（每项 = 名称/处理函数/可用标志），绘制、命中、
  键盘都按表驱动；「复制」「移动」两枚按钮**已占位并灰显**（`enabled=0`），落地 = 标志改 1 +
  实现 `op_copy`/`op_move`；② 数据通路：小文件用 `(Fs.read)+(Fs.write)` 组合，大文件/目录树
  建议在 `am_native_Fs.c` 加 `Fs.copy`（`hal_fs` 的 read/write 原语已够用）。
- **选择器不引入宿主新能力**：`(Ui.input ...)` 那种阻塞模态框靠宿主控件；文件选择器完全用
  Scheme + Gfx 绘制，形态是"每拍 `(FP.step)` 的状态机"（0 进行中 / 1 已选中 / 2 已取消），
  与 App 的定时器节拍天然对齐。**活动期间调用方不要调 `(EX.check)`**——返回请求
  （硬 A/ESC）归选择器消费 = 取消选择；它右上角自己画的「取消」按钮与各 App 的「退出」
  按钮同一位置，点那儿也会变成"取消"。
- **时间戳文件名**：用 `Time.utc_days/utc_sod` 而不是 `System.timestamp`（wasm32 上毫秒值被
  截断成 mod 2^27，见 `am_native_Time.h`），按 Howard Hinnant 的 days→y/m/d 算法算成
  `20261020-020000.txt`（**UTC**，宿主不提供时区）。

## 3. 踩到的坑（都已修）

1. **`(Fs.write path "")` 永远失败**：`fs_wstring_to_utf8` 的容量检查写成 `cap < 2`，而空串的
   合法容量恰好是 1（只留一个 NUL）→ 改成 `cap < 1`。"新建空 .txt"必须靠它。
2. **`(null? x)` 不能用来判断"目录读不出来"**：`null?` 对 `#null` **和** `'()` 都为真，空目录
   会被误判成错误（表现为进空目录就 `FMERR`）。判空表用 `(list? x)`（`#null` 不是列表）。
3. **`#t` 不等于 `1`**：`(== (Fs.is_dir p) 1)` 恒为 `#f`（`#t` 是布尔、`1` 是数字）。凡是把
   布尔结果存进表里再比较的地方，一律先归一化成 `1`/`0`。
4. **`begin` 是语法关键字**，`(define begin ...)` 会**静默编译失败**（症状只是
   `RESULT compile_fail`，没有报错行）。选择器入口因此叫 `FP.start` / `FP.start_at`。
5. **插入排序方向写反**：`while` 条件里要比"已有序项 < 新项"才升序（写反就是降序，症状是
   列表倒着排）。
6. **手工数嵌套括号不可靠**：`op_new_file/op_new_dir/op_rename/op_delete` 初版是层层嵌套
   `if`，四处括号错位；改成**平铺书写**（先归一化参数，再顺序判断），并用脚本逐文件
   对括号做平衡检查（`depth` 必须归零）后一次通过。
7. **池子变紧**：markdown 现在连 picker + fsx 一起编译，编译期工作区从 ~13MB 涨到 ~20MB
   （16MB 池的 8MB VM 区装不下，`RESULT link_fail`）。生产宿主池 `MB_POOL_SIZE=24MB` 仍够
   （实测 29KB/1600 行的文档照样排版得下，`DOC_MAX` 未动）；dev 回归台
   （`run_host_regress.sh` / `tools/e2e/markdown/test_app.py`）已对 markdown 单独给 24MB。
8. **"再点一次已选中的行 = 进入/详情"**（小屏没有双击）：e2e 脚本里点行做选择时要注意
   "第一下就点中已选行"会直接进入——本轮的浏览器用例就因此误进目录并弹出详情框。

## 4. 验证

- **宿主无头回归台**（`python3 tools/e2e/native/test_filemgr.py`，**43 项全 PASS**）：
  A 组 15 项 `Fs.remove/rename` 语义（改名成功/目标已存在拒绝/目录改名/源缺失/`..` 拒绝/
  删文件/非空目录拒绝/空目录/资源根保护）；B 组 21 项文件管理器端到端（进目录、**升序**、
  新建文件默认时间戳名 `.txt`、无扩展名自动补 `.txt`、重命名、删文件、新建目录、进子目录建文件、
  上级、**递归删除目录树**、刷新，全部对磁盘实测）；C 组 7 项选择器（唤起、逐级进入、过滤、
  选中返回相对路径、阅读器按真文件打开取标题）。
- **全 App 回归**：`run_host_regress.sh` 全部 App `err=none`；markdown 既有用例
  `tools/e2e/markdown/test_app.py` **40/40 PASS**（重构 `open_path` 无回归）。
- **浏览器端（真 WASM + 真触屏）** `tools/e2e/browser/regress_filemgr.js`：
  **Chromium / chrome-headless-shell / Firefox 各 49/49 PASS** —— 布局像素判据、目录导航、
  新建文件（回车接受默认时间戳名）→ **MEMFS 真有该文件** → **IndexedDB（nano-store）真有该键**、
  重命名（初值=当前名 → "名字未变"分支）、删除文件（确认框回车）→ MEMFS + IndexedDB 都消失、
  新建目录/进入/删除、刷新、退出回列表；再进 Markdown 点「打开」→ 选择器（只列目录 +
  扩展名过滤）→ 选 `data/fmtest/pick_me.md` → `MDSTAT title=pick_me`；**Escape 取消选择器**
  （`FP cancel`，不弹 App 退出框）。
- **Markdown 浏览器回归** `regress_markdown.js` **32/32 PASS**（三种引擎）。
- `git worktree` 之外的既有 App 帧序列未变（新增的是独立 App + 两处本地库能力，
  不触碰渲染路径）。
- 另外把 dev 脚本的 `APPS_ALL/APPS`（`regress_det.js`、`regress.js`）同步为 `app/*.scm`
  的真实升序清单（此前缺 `hwr2`/`markdown`，其后所有 App 的下标整体错位）。

## 5. 已知取舍

- ESP32 的 `hal_fs_m5esp.cpp` 新函数**未编译验证**（本机无 Arduino 工具链）。
- 选择器目前只能选**文件**（"选目录"留作扩展点，注释在文件头）；单选、无路径手输。
- 文件管理器把资源根视作"系统根"，不提供越出根的浏览（各平台语义一致，也避免误删系统文件）；
  目录条目不在顶部聚合，严格按字母升序（需求如此）。
- 浏览器上目录**改名**会逐文件搬运 IndexedDB 键（深度 ≤8、文件数 ≤512，超出静默跳过），
  大目录树改名会慢一点；桌面/掌机零开销（没有落盘钩子）。

------------------------------------

# 2026-11-02（三）文件管理器/选择器三处调整（排序、高亮对齐、".." 条目）

用户反馈三项，均已实现并回归：

1. **排序改为"目录永远在前"**：`app/lib/fsx.scm` 的排序入口由 `sort_names`/`sort_pairs`
   统一为 **`sort_entries`**，比较函数 `cmp_entry` 先比"是否目录"（目录 < 文件），
   同类再按名字码点升序。条目约定：元素 0 = 名字、元素 1 = 是否目录(1/0)、元素 2 可选 = 字节数。
   **不依赖 `Fs.list` 的顺序**（Linux 排过、ESP32 是卷顺序），三平台一致。
2. **高亮条与文字垂直居中对齐**：原来高亮带画在 `y-3`（16px），比文字行框中心高 2~3px。
   现在统一画在 **`y+1` 起、高 `ROW_H-2`=16px**（`filemgr.scm` 与 `filepicker.scm` 各自的
   普通行与 ".." 行共 4 处）。像素实测（宿主回归台导出 PPM）：选中行文字墨迹 57..64
   （中心 60.5）与高亮带 53..68（中心 60.5）**完全同轴**。注意别按"行框"对齐——12px alpha
   字模的墨迹整体比行框中心低约 2px（基线偏下），按墨迹对齐才是眼睛看到的居中。
3. **列表顶部新增 ".."（返回上级目录）条目**：列表改成"显示行"模型——
   **显示行 0 固定是 ".."，显示行 d（d≥1）= rows[d-1]**（`n_disp`/`disp_count`）。
   进入目录后默认选中第一条**真实条目**（不是 ".."，避免顺手回车退回上级）；选中 ".." 再
   点按/回车 = 返回上级；根目录下 ".." 变灰、点按只提示"已在资源根"。原来的「上级」按钮
   保留（两条路都通）。`FM items`/`FP items` 诊断行也随之以 `../` 打头，e2e 直接拿 token
   下标当显示行号用。

**踩坑（重要）**：空目录进入即崩（`[Runtime] 指令执行异常`）——根因是新增的提示行写了
`(+ LIST_Y ROW_H 2)`：**Animac 的算术是二元**，三参数 `+` 会让操作数栈失衡。
已改为 `(+ (+ LIST_Y ROW_H) 2)`，并加了脚本扫描 3 参数算术/比较（顺带核对全文）。

验证：宿主 `tools/e2e/native/test_filemgr.py` **49/49 PASS**（新增：首项为 ".."、目录在前 +
文件在后各自升序、根目录顺序与 Python 独立算出的期望表逐项比对、".." 条目返回上级）；
既有 markdown 用例 40/40；`run_host_regress.sh` 全部 App `err=none`；浏览器
`regress_filemgr.js` **Chromium / chrome-headless-shell / Firefox 各 51/51 PASS**（含选择器里
".." 首项与"只看目录/扩展名过滤"仍成立）；`regress_markdown.js` 32/32。

------------------------------------

# 2026-11-02（Markdown：引用块 + 嵌套列表；H2/H3 装饰；插图调研）

需求三条：① 补引用块与嵌套列表；② 二级标题左侧加粗竖线（#66ccff）、三级标题文字加下划线；
③ 插图（图片）渲染只出调研报告、不实施（报告见 `doc/Markdown插图渲染可行性调研.md`）。

## 1. 引用块（`> ...`）

- 块形状 `(quote blocks)`：内层是**任意块**（标题/列表/表格/代码块/嵌套引用）。
- 判定：行首（允许 ≤3 空格）是 `>`，其后可有 1 个空格；连续 `>` 行构成一块；
  **不做** CommonMark 的"懒惰续行"（非 `>` 行一律结束，写明在注释里）。
- 排版：内层块按"右移后的宽度"排版，排完整段显示行整体右移 `QUOTE_IND=10`，
  并在**每一行**（含段间留白行）最前面插一条 2px 竖线段 `ST_QUOTE_BAR` ——
  竖线因此是连续的；嵌套引用每深一层多一条竖线（x = 0, 10, 20 …）。

## 2. 嵌套列表

- 块形状变了：`(list items)`，item = **(item inlines children)**（原来是裸的行内列表）。
- 层级用"祖先栈"算（栈里存每层的内容起点缩进）：新行先把比它浅的祖先弹出，栈深即层号；
  建树时 `path[k]` = 第 k 层当前 item，新 item 挂到 `path[层-1]` 的 children。
- 排版递归 `lay_items`：每层缩进 `LIST_IND=14`，项目符号悬挂，折行右边界随层数收窄。

## 3. 标题装饰（字模只有一套 12px，靠装饰而非字号表达层级）

| 级别 | 装饰 |
|---|---|
| H1 | 粗体 + 段前留白最多 + 2px 通栏横线（不变） |
| H2 | 粗体 + **左侧 3px 装饰竖线 `ST_H2BAR`，颜色 #66ccff = (102,204,255)**；原来的 1px 横线取消 |
| H3 | 粗体 + 更暗配色 + **文字下划线**（绘制端在 `draw_seg` 里按 `ST_H3` 画 1px） |

装饰全部以"段"的形式表达（竖线段文本为空、宽 3），绘制端依旧不知道这是标题。

## 4. 这轮踩到并绕开的一个解释器坑（**重要，写 App 必备**）

现象：引用块在文档**中间**出现且内部还有嵌套引用时，一个引用块被解析成三块（内容重复）。

定位过程（可复现的最小场景已留在会话记录里）：给 `h_quote` 加日志后发现，
`parse_lines → h_quote → parse_lines` 这种**两级回调**里，外层 `h_quote` 的局部变量 `i`
（记录"下一行下标"）在递归返回后变成了**内层**的值 → 返回给主循环的 next-index 偏小 →
主循环从头再解析一遍引用内容。对照实验：

- 直接自递归（`dbg_rec`）在多级深度下**局部变量互不影响**（每层独立帧）；
- 通过块规则表 `(h lines i)` 间接重入**同一个闭包**时，帧是**共用**的。

结论与对策（两条，都已在代码里落实）：

1. **不要让同一个闭包被重入**。引用块因此拆成两步：`h_quote` 只剥一层前缀产出中间块
   `(quote-raw 行数组)`；`parse_lines` 返回后由 `fixup_quotes` **逐层展开**——
   每次 `parse_lines_once` 调用都等上一次返回，天然不嵌套；递归只发生在 `fixup_quotes` 自己身上。
2. **排版同理**：`lay_quote` 不递归调用自己，而是先把引用树**摊平**成 `(层号 块)` 序列
   （`flatten_quote`，普通自递归，安全），再逐块排版 + 右移 + 插竖线。

顺带记录两个同源的坑（本轮都踩了）：
- AST 取值链每多一层就多一次 `get_item`，越界返回 `#undefined` 而**不报错**，
  真正的报错出现在后续 `String.equals` 等原生调用上（"指令执行异常"）——排查时先怀疑"多/少了一层"。
- 测试脚本里 `define` 重名会报 `failed to add defines to scope`（本轮 `di` 撞名两次）。

## 5. 验证

- 解析单测 **111/111 PASS**（新增 24 条）：嵌套列表（2/3 层、回到上层、子项续行）、
  引用块（单块、内层段落合并、引用里的标题+列表、嵌套引用、**引用在文档中间不被重复解析**、
  空引用、非 `>` 行结束）。
- 排版单测 **69/69 PASS**（新增 27 条）：三层列表的项目符号 x = 0/14/28/0、引用竖线段与右移、
  嵌套引用竖线数、H2 装饰竖线（宽 3、文字右移 3、且不再画横线）、H3 样式存在。
- App 端到端 **45/45 PASS**（新增 6 条）：H2 竖线颜色 #66ccff、H3 下划线、引用竖线的**像素**断言；
  文档变高后代码块/表格移到"滚动到末尾"的子用例断言。
- 浏览器端到端（真 WASM）**36/36 PASS**：同上判据（RGB888 颜色精确）+ 滚动后代码块/表格底色。
- 宿主 35 个 App 回归全 `err=none`；内置示例加了两节（嵌套列表、引用块），块数 21 → 23。

## 6. 插图调研（只出报告）

`doc/Markdown插图渲染可行性调研.md`。核心发现：**宿主图形层里已经有 `gfx_draw_image`**
（文件 → `stb_image` 解码 → `stbir` 缩放到目标尺寸 → LRU 缓存 → 落屏），
解码器是 vendored 的 `stb_image/stb_image_resize2`，ESP32 也已有 32KB 栈任务分支；
所以只需给 `Gfx` 加两个薄封装（`image_size` 用 `stbi_info` 只读文件头、`draw_image` 调 `gfx_draw_image`），
约 80~120 行 C 代码 + 排版期注入一个"度量回调"。推荐方案 A（约 3~3.5 人日），
兜底方案 D（虚线框 + alt）。报告含风险表（缓存只有 2 个槽位、alpha、首帧延迟）与分期建议。

------------------------------------

# 2026-11-02（Markdown：插图渲染实施 —— 方案 A + 方案 D 兜底）

按 `doc/Markdown插图渲染可行性调研.md` 落地了插图的**一期**。核心结论没变：
宿主早就有一条"读文件→解码→缩放→LRU 缓存→落屏"的链路（`gfx_draw_image`），
本轮只是把它**接出来**给 Scheme 用，外加缺图时的虚线框兜底（方案 D）。

## 1. 接口（新增 2 个 Gfx 本地库函数）

```scheme
(Gfx.image_size path)                ; → (宽 高) | #null   只读文件头（stbi_info），不解码
(Gfx.draw_image path x y w h)        ; → #t / #f           按给定尺寸解码+缩放+落屏（宿主 LRU 缓存）
```

- C 侧新增/改动：`graphics.c` 的 `gfx_image_info()`（`stbi_info`）、`gfx_draw_image()` 返回值
  `void → int32_t`；`am_native_Gfx.c` 两个 native 包装；`am_native_Fs.c` 把 `fs_resolve`
  提升为公共 `am_native_fs_resolve()`，Gfx 与 Fs 共用同一套路径规则（相对业务资源根、拒绝 `..`）。
- **等比缩放放在排版端**（不是 C 侧）：排版本来就要算行高，顺手算出缩放宽高，
  绘制端只负责"按给定尺寸画"，契约更简单。

## 2. Scheme 侧

- `md_parse.scm`：行内规则 `![alt](path)` → `(image path alt)`（**必须注册在 `[` 之前**）；
  整段只有一个图片节点时提升为块 `(image path alt)`（块级图片独占一行、居中）。
- `md_layout.scm`：`layout` 增加第三个参数——**图片度量回调**（`(measure path) → (w h) | #f`），
  真机传 `Gfx.image_size` 包装、单测传桩（`md_parse/md_layout` 仍是纯逻辑、可脱离 GUI 单测）；
  新样式 `ST_IMAGE / ST_IMAGE_MISSING / ST_IMAGE_TEXT`；`IMG_MAX_H=160`（等比缩放进
  「正文宽 × 160」）、`IMG_MAX_COUNT=24`（超出退化为占位框，防一次排版吃光池子）。
- `markdown.scm`：`image_path` 按**文档所在目录**解析相对路径（Markdown 习惯）；
  `draw_image_missing` + `draw_dashed_rect` 画虚线框 + alt（方案 D）；`MDSTAT` 增加 `images=N`。
- `md_demo.scm`：加「插图」一节说明语法与兜底（不引用具体文件，跨平台一致）。

## 3. 坑：`#null` 是**真值**

`(Gfx.image_size ...)` 失败返回 `#null`，而本解释器里除 `#f` 外一切都是真——
排版端原本写 `(if dims ...)`，于是把 `#null` 当尺寸去 `get_item`，直接
`[Runtime] 指令执行异常`（表现为"打开含缺图的文档就崩"）。两处同时修：
App 侧把 `#null` 归一成 `#f`，排版端再用 `(list? dims)` 类型判定（双保险）。

顺带记一个同类事故：`Gfx.draw_image` 的段样式如果继续走"通用画文字"分支，
**alt 文本会叠在图片上**（第一版就是这样），现在图片/占位框样式跳过通用文字绘制。

## 4. 一期范围与遗留

- 只做**块级图片**；行内 `![..](..)`（段落中间）退化为暗色占位文本「[图片] alt」——
  行内图片会让行高与图片高度耦合，留到二期。
- alpha 通道未做混合（置色语义）；`GFX_IMAGE_CACHE_SIZE` 仍是默认 2（多图文档可考虑调大）。

## 5. 验证

- 解析单测 **124/124**（新增 13 条：块级提升、行内节点、无 alt、路径去空白、残缺写法退化）；
- 排版单测 **89/89**（新增 19 条：行高=图片高、水平居中、路径/alt 存放位置、
  缺图占位框、超大图等比缩到 160 高、张数上限、行内降级为占位文本）；
- 宿主端到端 **48/48**（真 PNG 画出来：品红像素 720；缺图虚线框像素 172；`images=1`）；
- 浏览器端到端（真 WASM）**40/40**（base64→`FS.writeFile` 写 `doc/img.png`，
  品红 1200 像素、虚线框 338 像素、无 pageerror）；
- 宿主 35 个 App 回归全 `err=none`。

------------------------------------

# 2026-11-02（Markdown：扩充示例文档 + 图片路径的资源根回退）

## 1. 示例文档大幅扩充（`app/lib/md_demo.scm`）

从 23 块 / 114 行扩到 **29 块 / 140 显示行 / 高 1637px**（源码 96 行 / 5.3KB），
目标是"把当前支持的每种语法都摆出来，并刻意做复杂组合"，而不是罗列语法清单：

| 小节 | 覆盖点 |
|---|---|
| 行内样式 | 加粗 / 行内代码 / 带文字链接 / 链接里套加粗 / 反斜杠转义（`\*`、`` \` ``、`\|`）/ 中英混排折行 |
| 标题 | H1（通栏线）、H2 里放 `代码` 与链接（装饰竖线 + 行内样式共存）、H3、H4 按 H3 处理 |
| 列表 | 三层嵌套且每层样式不同；超长条目演示悬挂缩进与续行对齐 |
| 代码块 | 中文注释、**制表符缩进**、超长行折行、空行保留 |
| 表格 | 四种对齐 + 单元格内的加粗/代码/链接 + 长单元格折行撑高整行 |
| 引用块 | 引用里放三级标题、段落、列表、**嵌套引用（两道竖线）**、围栏代码块 |
| 插图 | 两张 `icon/` 真图（36×36 与 80×52，居中）+ 行内图片降级占位 + 缺图兜底说明 |
| 其它 | `---` 与 `***` 两种分隔线、末尾"支持范围与已知取舍" |

规模刻意压在 2MB 池（ESP32）能承受的范围：显示行约 1KB/行，140 行 ≈ 140KB。

## 2. 图片路径新增「资源根回退」（否则示例根本引不到 `icon/`）

图片路径此前只按**文档所在目录**解析，而示例文档落在 `doc/` 里，于是
`![x](icon/nano.png)` 会解析成 `doc/icon/nano.png`（不存在）；又不能写 `../icon/...`
——`Fs`/`Gfx` 的路径规则明确拒绝 `..`。现在 `image_path` 的查找顺序是：

1. 以 `/` 开头 → 宿主绝对路径，原样交给 Gfx；
2. 相对**文档所在目录**（Markdown 习惯）；
3. 找不到则回退到**资源根**（`icon/xxx.png` 这类公共资源放在根下）。

代价是每次解析多一次 `Fs.exists`（stat），每帧每图 ≤2 次，可忽略。

## 3. 验证

- 宿主端到端 **49/49**：内置示例 29 块、`images=2`（两张 icon 都测量成功并被排版进去），
  落盘文件内容校验加上了各小节标记（`## 表格` / `## 引用块` / `icon/gm_miner.png`）。
- 浏览器端到端（真 WASM）**41/41**：同样的 29 块 / `images=2`，说明资源根回退在
  MEMFS 里也成立（`icon/` 在 manifest 的 p1）。
- 解析 124/124、排版 89/89 不变；宿主 35 个 App 回归全 `err=none`。
- 逐屏截图肉眼核对：H2 装饰竖线与行内代码共存、三层列表缩进、代码块内的中文与制表符、
  表格四种对齐与长格折行、引用里的标题/列表/嵌套引用/代码块、两张图标居中、
  行内图片显示 `[图片] 内联图` 占位。

------------------------------------

# 2026-11-02（示例文档：从"运行时落盘"改为"构建期导出"）

## 问题

用户反馈「示例文档在哪？找不到」——一查确实如此：

- 仓库里只有**真源** `infer/app/lib/md_demo.scm`（行数组，灵机字符串不能跨行）；
- 成品 `Markdown 示例.md` **只在阅读器第一次运行时**由 `seed_demo_file()` 写到
  `<资源根>/doc/` 下，所以没运行过 App（或只跑过别的 App）时资源目录里什么都没有；
- 更糟的是 e2e 测试的 F/G/H 用例会**删掉 doc/ 做"首次启动落盘 / 用户编辑不被覆盖"**验证，
  收尾清理后文件就没了——开发者跑完测试再去资源目录找，必然什么都找不到。

## 处理

1. **新增导出工具 `infer/tools/gen_md_demo.py`**：解析 `md_demo.scm` 里的字符串字面量
   （按 manual §2.5 的转义集合 `\" \\ \n \t \r` 反解，未知转义原样保留），拼成 markdown，
   写到 `<资源根>/doc/Markdown 示例.md`。**真源仍只有 md_demo.scm 一处**，导出物不手改。
   用法：`python3 tools/gen_md_demo.py [资源根]`、`--stdout`、`--src`（自测用）。
2. **`make md-demo`** 目标（不并入 `all`，不动默认构建；`LINGLONG ?= /home/bd4sur/linglong`）。
3. **`build_wasm_gui.sh` 自动调用**：WASM 构建后会先把示例 .md 摊到资源目录，
   于是 manifest 里出现 `doc/Markdown 示例.md`（p2，3.4KB），浏览器端开箱就能在文档列表里看到。
4. **测试收尾恢复**：`test_app.py` 跑完后调用生成器把示例重新导出，
   避免"跑完测试资源目录里没有示例文档"再次误导人。
5. 仍保留运行时落盘（`seed_demo_file`）作为兜底：只读文件系统、或直接把 App 拷到设备
   没带 `doc/` 时，首次启动自己写一份；文件存在则不覆盖（可当便签编辑）。

## 验证

- `make md-demo` / `gen_md_demo.py` 生成 96 行 / 3482 字节，制表符与中文/引号转义均正确
  （`grep -n "3628800" | cat -A` 可见 `^I` 制表符）；
- 预置文件存在时，阅读器**直接按文件打开**（无 `MDSEED` 行，`blocks=29 images=2`）；
- 删掉文件后启动，仍会自己落盘并打开（`MDSEED ... written`）；
- 宿主端到端 49/49、浏览器端到端 41/41、解析 124/124、排版 89/89；
- WASM manifest 112 个文件，其中 `doc/Markdown 示例.md` 为 p2。

------------------------------------

# 2026-11-02（Markdown：表格加上下框线，框线宽度收到表格实际宽度）

需求：表格上下要有框线。

## 改法

新增样式 `ST_HR`（水平线段，宽 1px、宽度由**段宽**决定），表格的三条横线都改用它：

```
留白 7 → 上框线 → 间隔 2 → 表头行（BG_TABLE 淡底 + 加粗）→ 表头下框线 → 数据行… → 下框线 → 留白 9
```

- 为什么不用原来那种 `BG_RULE` 行：`BG_RULE` 在绘制端是**画满正文宽**的，
  而表格列宽是"按内容自适应"的（窄表可能只有 100 多像素）——原来的表头横线会拖到尾、
  看着像一条被截断的分隔线。改用"段"来表达横线后，宽度 = 表格实际宽度
  （各列宽之和 + 列间距），三条框线与列间竖线的左右端正好对齐。
- 绘制端新增 `ST_HR` 分支：`fill_rect(x, y, w, 1)`，颜色与分隔线一致（78,98,132）；
  表格仍然是"段"的纯消费者，绘制端不知道这是表格。

## 验证

- 排版单测 **91/91**（新增：三条框线都在、框线宽度 = 表格实际宽度、且不再画满正文宽；
  表格行数从 6 行变 9 行 —— 上框线 / 2px 间隔 / 表头 / 表头框线 / 2 数据行 / 下框线 / 留白）；
- 宿主端到端 **50/50**、浏览器端到端 **42/42**：新增"表格框线像素"判据
  （宿主 940px、WASM 602px 的横线像素），并肉眼核对窄表框线不再拖尾；
- 解析 124/124；宿主 35 个 App 回归全 `err=none`；
- 示例文档因多出 3 行框线，显示行 140 → 143、总高 1637 → 1641（块数不变，仍是 29）。

------------------------------------

# 2026-11-02（Markdown：表格行间也加框线 → 全网格）

需求：表格行与行之间也要有框线。

## 改法

`lay_table` 在**每两条相邻数据行之间**再画一条 `ST_HR`（除最后一行外），于是表格变成全网格：

```
留白 7 → 上框线 → 间隔 2 → 表头（淡底+加粗）→ 表头下框线
       → 行1 → 行间框线 → 行2 → 行间框线 → … → 行N → 下框线 → 留白 9
```

- 框线总数 = 数据行数 + 2（上、表头下、N−1 条行间、下），全部用 `ST_HR` 段表达，
  宽度都是表格实际宽度，与列间竖线左右端对齐；
- 行间框线画在**逻辑行**之后——单元格折行把某行撑成多条显示行时，框线仍然只出现一次，
  落在该逻辑行的下沿（不是每条显示行下面都画）。

## 验证

- 排版单测 **91/91**：行数断言从 9 行变 10 行（留白/上框线/间隔/表头/表头框线/行1/行间框线/
  行2/下框线/留白），框线总数断言改为"数据行数 + 2"；数据行下标随之更新（3/5/7）；
- 宿主端到端 **50/50**（表格框线像素 940 → 1038）、浏览器端到端 **42/42**（602 → 700）；
- 肉眼核对：两张表（宽度不同）都有完整网格，折行的那一行框线只出现一次、位置正确；
- 解析 124/124；宿主 35 个 App 回归全 `err=none`；
- 示例文档显示行 143 → 146、总高 1641 → 1644（两张表共多 3 条行间框线）。

------------------------------------

# 2026-11-02（Markdown：目录（TOC）功能）

需求：按标题生成目录（最多到 H3），底栏「目录」按钮弹出目录结构，点标题跳到对应位置。

## 1. 数据来源：排版阶段顺带收集（不额外扫一遍 AST）

`md_layout.scm` 里加了三样模块级状态：

- `cur_y`：已排显示行的累计高度，由 `emit_line` 维护（每次 `emit_line` 累加行高）；
- `toc_entries`：本次 layout 收集到的标题，元素 `(层级 y偏移 文本)`；
- `quote_depth`：`lay_quote` 排引用内容期间 > 0。

`lay_heading` 在排标题文本**之前**把 `(显示层级 cur_y 纯文本)` 推进目录——
`cur_y` 就是标题行的顶端偏移，所以"跳到某个标题"= 把这个值当 `scroll` 用，**一次减法**，
不用反查行号（这正是"排版期把坐标定死"的又一次收益）。

- 层级取**显示层级**（4~6 级按 3 级归档），所以目录最多三档，与渲染一致；
- 文本用新加的 `inlines_text()` 摊平（加粗/链接取内容、行内代码取其串、图片取 alt）；
- **引用块里的标题不进目录**（`quote_depth > 0` 时跳过）——那是"引文"，不是本文结构；
- 每次 `layout` 重置这三样（否则第二次打开文档会累积）。

对外多一个 `(MDL.toc)`（App 在 `layout` 之后立刻取走）。

## 2. UI（app/markdown.scm）

- **底栏「目录」按钮**：摆在「打开」左侧（40×14），仅在**阅读页且文档有标题**时显示；
- **面板**（`draw_toc_panel`）：全屏覆盖，页眉「目录 N 条」，条目按层级缩进
  （每级 10px）+ 配色（一级最亮、二三级渐暗），选中条左侧是 #66ccff 强调竖条；
  一屏放不下时窗口跟随选中项滚动（`toc_first()`）；
- **跳转**：`toc_jump(i)` → `scroll = y - 6` → `clamp_scroll()`，关面板；
  打开面板时**默认选中"当前阅读位置之前最近的标题"**（`toc_open_panel`），
  这样"打开目录 → 回车"就是回到本节开头；
- **输入归属**：面板活动期间 `on_tick` 走 `tick_toc`，**不调 `(EX.check)`**，
  并把 `(App.back_requested)` 消费为"关闭面板"——与文件选择器同一套思路；
- 点按：页眉区 = 关闭，条目区 = 跳转（`toc_tap`）；键盘：↑↓ 选择、Enter 跳转、Esc 关闭。

## 3. 顺带：解释器池 24MB → 32MB

App 又并进了 `filepicker.scm` / `fsx.scm` 两个模块、且本轮继续长大，实测 24MB 时
**编译期 VM 区（池的一半 = 12MB）就装不下**（`[allocator] VM segregated 分配失败`，
表现为 link_fail）。按既有惯例上调 `MB_POOL_SIZE` 到 32MB（ESP32 仍是 2MB），
宿主回归脚本与 `test_app.py` 的池参数同步改成 32768KB。

## 4. 验证

- 排版单测 **98/98**（新增 8 条：条数、层级序列 1/2/3/3/2、文本摊平、y 递增、
  首条 y = H1 行顶端、按 y 找到的是标题行、重新排版后重置、引用里的标题不计）；
- 宿主端到端 **61/61**（新增 12 条）：点「目录」按钮 → `MDTOC n=11` + 面板底色/选中条像素；
  点第 8 条 → `MDJUMP i=7 y=780` + 跳转后 H2 装饰竖线出现在屏幕顶部；
  键盘 ↑↓+Enter 也能跳；**返回键只关面板、不退出 App**；
- 浏览器端到端（真 WASM）**46/46**：同样的面板/跳转判据（`MDTOC`、`MDJUMP i=7 y=780`、
  面板像素 bg=19468 sel=45），截图 `/tmp/e2e-toc2/toc.png`、`toc-jump.png`；
- 解析 124/124；宿主 35 个 App 回归全 `err=none`；
- 示例文档加一行说明"目录"功能（97 行 / 3625 字节，块数仍 29）。

------------------------------------

# 2026-11-02（Markdown：设置模态框 + 第一个设置项「切换字体」）

需求：增加设置模态框，第一项设置 = 在系统支持的三套字模之间切换（两套抗锯齿 + 一套点阵）。

## 1. 三套字模的真实度量（实测，不是猜的）

| id | 字模 | 半角步进 | 全角步进 | 行高 | 基线 |
|---|---|---|---|---|---|
| 0 | `GFX_FONT_BITMAP_12`（12px 二值点阵） | 6 | 12 | 13 | 12 |
| 1 | `GFX_FONT_ALPHA_12`（12px 抗锯齿，默认） | 7 | 12 | 15 | 12 |
| 2 | `GFX_FONT_ALPHA_16`（16px 抗锯齿） | 9 | 16 | 18 | 14 |

**关键**：字模决定**行高与字符步进**，所以换字体不是"换个画法"——折行位置、行高、
表格列宽、目录 y 偏移全都会变，必须**重新排版**。

## 2. 宿主侧（`am_native_Gfx`）

```scheme
(Gfx.set_font id)     ; 0/1/2；非法 id 返回 #f；影响后续所有 Gfx.draw_text
(Gfx.font_info id)    ; → (半角 全角 行高 基线) | #null
```

`draw_text` 里原先硬编码的 `GFX_FONT_ALPHA_12` 改成文件级 `s_font_id`（默认 1，
保持历史行为）。`font_info` 用 `gfx_font_char_advance('i')` / `gfx_font_char_advance(0x4E2D)`
/ `gfx_font_line_height` / `gfx_font_baseline` 现查，**不写死数字**（换字模库时不会漂）。

## 3. 排版侧（`md_layout`）

原先的常量 `W_HALF/W_FULL/LINE_H` 改成可变槽位，新增
`(MDL.set_font_metrics 半角 全角 行高)`；其余代码不用动（它们本来就只读这三个槽）。

## 4. App 侧（`markdown.scm`）

- **设置面板**（`draw_settings` / `tick_settings`）：全屏覆盖、页眉「设置」，
  小节「字体」下三行单选（点阵 12px / 抗锯齿 12px / 抗锯齿 16px），当前项前面画实心方块，
  底部显示「当前：xxx  行高 N」——换字体后这行数字会立刻变，肉眼即可确认真的换了度量；
- **底栏第三个按钮**「设置」（置于「目录」左侧；两种模式都显示），footer 提示文案相应缩短；
  底栏三按钮：设置(182..222) / 目录(226..266) / 打开(270..314)；
- **`apply_font(id)`** 一次做四件事：`Gfx.set_font` → 查 `font_info` 更新 App 的画字几何
  （`text_pad`/`font_base`/`pill_h`，以及列表行高 `ROW_H`、目录行高 `TOC_ROW`、设置行高）
  → `MDL.set_font_metrics` → **重新排版当前文档**（`open_path`/`open_builtin`，保留 scroll）；
- **持久化**：选择写入 `data/md-font.txt`，启动时 `load_font` 在**打开文档之前**恢复
  （否则首个文档会用错度量排版）；浏览器目标由 Fs 的落盘钩子同步进 IndexedDB；
- 画字几何从常量改成"随字模算出来的变量"：链接下划线 = `text_pad + 基线 + 3`、
  三级标题下划线 = `+2`、行内代码药丸高 = `行高 - 2`、项目符号垂直居中于行高。

## 5. 验证

- 排版单测 **104/104**（新增 6 条：16px 度量下行高变大/行数变多/不超宽、点阵 12px 更紧凑、
  还原默认后与最初一致）；
- 宿主端到端 **71/71**（新增 10 条）：启动即 `MDFONT id=1 line_h=15`；点「设置」出面板
  （选中条像素）；点第 3 行 → `MDFONT id=2 line_h=18` **且出现第二条 MDSTAT**（说明重新排版了，
  行数 150→161、总高 1692→2121）；`data/md-font.txt` 内容为 2；**重启进程后仍是 id=2**
  （持久化生效）；再切点阵（id=0 line_h=13，总高 1538）与切回默认；
- 浏览器端到端（真 WASM）**50/50**：同样判据（截图 `/tmp/e2e-set/settings-16px.png`）；
- 解析 124/124；宿主 35 个 App 回归全 `err=none`；
- 示例文档加一行「设置」说明（99 行 / 3779 字节，块数仍 29）。

---

# 2026-11-06（黄金矿工：灵机 App 版 + 图形层透明精灵接口）

需求：把 C 版黄金矿工（`infer/ui_goldminer.c`）移植为「灵机应用程序」（Scheme App），只做
**WASM-GUI**（`build_wasm_gui.sh` 产物，不是 `infer/web/` 那条 LLM-only 路线）。贴图路径必须
与 C 版一致（`icon/gm_*.png`，透明背景 PNG），读图/绘图参照 Markdown 阅读器的做法。

## 1. 新 App（`infer/app/goldminer.scm`，**单文件**）

逻辑与界面合在一个模块里，按 C 版的组织顺序排列（规则段 → 绘制/输入/主循环）。
最初曾拆为 `app/goldminer_core.scm` + `app/goldminer.scm`，用两个注入接口
（`(set_screen w h)`、`(set_sprite_size id w h)`）换取"无 UI 依赖的纯逻辑单测"；
但规则量（约 120 行）摊不平这处接缝，且物品槽位表示照样泄漏给绘制端，故**合并为单文件**，
回归改为「HAL 驱动真实 App + 控制台诊断行」（见下 §3）。

- **规则与状态段**：逐条对照 C 版
  `gm_generate_level` / `ui_goldminer_render_frame` 的逻辑段——摆动
  （`hook_angle = 1.30·sin(phase)`，速度 `2.2 + 0.1·(level-1)` rad/s）、出钩 300px/s、
  边界（`x≤3 / x≥W-3 / y≥H-4`）或抓取即回收、回收 `240/weight` px/s、命中判定
  （钩尖进入"以精灵中心为圆心、直径 = 精灵宽高较小者"的内切圆）、结算加分与过关、
  关卡物品构成（大/中金块各 1，小金块 `3+min(level-1,3)`，岩石 `3+level/2`，
  钻石 `level≥2`，福袋 1，宝箱 `level≥3`）、随机放置 + 拒绝重叠（外扩 8px 椭圆）。
  单拍步长钳到 50ms（= C 版 `GM_DT_MAX`）。精灵尺寸在 `boot` 里用
  `(Gfx.sprite_info ...)` 一次性填充（C 版是 `ui_icon_get_size`，同一语义）。
- **绘制/输入段**：天空 `(25,45,75)`、地面 `(110,75,40)`、
  地表线 `(200,160,110)`、顶栏「得分 N  第M关  剩余 K」、物品 → 矿工（`PIVOT_X+20,
  PIVOT_Y-20`）→ 钩绳 → 钩爪的绘制顺序，全部与 C 版同源常量；场景常量直接从核心库
  取（`GM.GROUND_Y` 等，单一事实来源）。节奏 `System.set_interval(33ms)` ≈30fps，
  物理步长用 `System.timestamp` 实测差（同 C 版按帧时间戳推进）。
  输入：触屏**松手沿**发射（= C 版 `touch_edge UP`），键盘回车/`'2'` 按下沿发射
  （= C 版 D 键/数字 2）；退出统一走 `app/lib/exit.scm`（右上角「退出」按钮 + 硬
  A/ESC → 全局确认框 → 销毁实例），替代 C 版自绘的「返回」按钮（Scheme App 没有
  「小游戏菜单」这个状态，退出即回应用列表——与其它 App 同款约定）。
  诊断行（只在状态变化时打印）：`GMSTAT`（精灵数 / 物品数 / **七类构成 counts** /
  **物品包围盒 rect** / 分值 / 关卡）、`GMFIRE`、`GMGRAB`、`GMSHOT`（类型/分值/重量/
  绳长/**出钩与回收耗时**）、`GMSCORE`、`GMLEVEL`（过关后的新构成与包围盒）。

## 2. 图形层新增"透明精灵"接口（`graphics.c/h` + `am_native_Gfx`）

**为什么不能照抄 Markdown 的 `(Gfx.draw_image ...)`**：那条通道是"缩放到目标矩形 +
`stbi_load(...,3)` RGB 直写"，**丢弃 alpha**（适合照片/插图）。拿它画透明背景精灵会
得到不透明方块（实测精灵底色为黑）。故在图形层新增一组精灵接口：

```
gfx_sprite_info(path, &w, &h)                     // 解码一次并常驻 RGBA 缓存
gfx_draw_sprite(gfx, path, cx, cy)                // 原点尺寸、中心对齐、alpha 混合
gfx_draw_sprite_rotated(gfx, path, px, py, angle) // 枢轴 = 贴图上缘横向中点
```

- 实现：`graphics.c` 末尾新增 RGBA 精灵缓存（键=文件路径，LRU 12 槽，**失败也占槽**
  避免反复读盘；像素用 `stbi_load(...,4)` 解码），绘制走 `gfx_blend_pixel`
  （RGB565/RGB888 双路径、自动标脏区）；旋转版与 `ui_icon.c` 的
  `ui_icon_draw_rotated` **逐行同算法**（逆映射最近邻 + 四角包围盒裁剪），因此与 C 版
  黄金矿工的钩爪姿态一致。缓存/实现独立于 UI 层，链接 `graphics.c` 的宿主即可用
  （无需 `ui_icon.c`）。
- 灵机本地库（`am_native_Gfx`）新增三条，路径规则与 `(native Fs)` 完全一致
  （相对业务资源根、拒 `..`）：`(Gfx.sprite_info p) : (宽 高) | #null`、
  `(Gfx.draw_sprite p cx cy) : Boolean`、`(Gfx.draw_sprite_rotated p px py angle) : Boolean`。
  失败返回 `#f`（软失败），App 可据此降级——但注意贴图缺失时 App 不会画方块：
  核心库保留了 C 版"无贴图时的原型包围盒"，游戏逻辑仍可玩。
- **抗锯齿钩绳**（`(Gfx.draw_line_aa x1 y1 x2 y2 width r g b)`，`am_native_Gfx` 第四条新增）：
  直接转 `gfx_draw_line_anti_aliasing`（1px 走吴小林算法），端点/线宽按**浮点**取、
  不取整——摆动的绳角是连续的，只有亚像素端点才换来平滑斜线。语义与 `(Gfx.draw_line)`
  不同：底层 `gfx_add_pixel` 是**加法混合**（底色 + 覆盖率×颜色），不是置色。
  钩绳绘制即 `(Gfx.draw_line_aa PIVOT_X PIVOT_Y GM.hook_x GM.hook_y 1 230 230 230)`，
  与 C 版 `gfx_draw_line_anti_aliasing(gfx, 160, 66, hook_x, hook_y, 1.0f, 230,230,230, 1)`
  逐参数等价（含 C 版同款"绳在浅底上偏亮"的加法效果）。

## 3. 验证

- `tools/e2e/native/test_goldminer.py`：**28/28**，三层——
  ① 抗锯齿专项（`aa_line_probe.scm`：同一纯色底上并排画 `(Gfx.draw_line)`（整数+置色）
     与 `(Gfx.draw_line_aa)`（吴小林+加法混合），用"覆盖率反解模型"
     `c - bg = min(255-bg, 230a)`（以最晚钳位的蓝通道解 a）证明**抗锯齿侧存在中间覆盖率
     像素 168 个、整数侧 0 个**）——这是"钩绳必须是抗锯齿线"的可执行判据；
  ② 规则与关卡（HAL 按键脚本驱动真实 App，`NANO_VSTEP=50` 压缩虚拟时间，故 6000 帧
     墙钟约 1s）：第 1 关构成 `1/1/3/3/0/1/0`、物品包围盒落在屏内且在地表以下、
     `GMSHOT` 的 value/weight 与原型表一致、**出钩耗时 ≈(绳长-16)/300**（实测最大误差
     50ms）、**带物回收耗时 ≈(绳长-16)/(240/weight)**（最大误差 44ms）、分值累计 =
     各次结算之和、收完整关后 `GMLEVEL` 显示第 2 关 `items=12 counts=1/1/4/4/1/1/0`
     （出现钻石）且落点仍合法；
  ③ 界面与输入（RGB565 帧缓冲像素判据）：天空/地面分区、HUD 文字、金块/岩石贴图、
     **矿工盒内有天空色像素**（透明背景证据）、钩绳/钩爪随摆动移动（两帧差异像素）、
     键盘/触屏发射、抓取计分、硬 A/ESC 挂起；
  （回归台为此新增 `NANO_VSTEP` 每帧虚拟毫秒数、并把 `KEY_MAX_STEPS` 由 64 提到 512。）
- `tools/e2e/browser/regress_goldminer.js`（真 WASM + 真无头浏览器）：chromium
  **22/22**、firefox **22/22**（导航进入应用列表→启动→贴图/透明/摆动（差异像素质心 +
  中间覆盖率像素）→触屏发射与计分→键盘回车→退出确认框「留下/确认」→硬 Esc 全局确认，
  无 pageerror）；
- 宿主回归台 38 个 App 全 `err=none`（含新 App）；Markdown 专项浏览器回归 50/50
  不受图形层改动影响；`regress.js` 存活回归中 goldminer `refreshΔ=110/4s`（≈27fps）。

## 4. 运行

```bash
cd infer && bash build_wasm_gui.sh            # 产物 infer/webgui/
cd infer/webgui && python3 ../tools/https_static_server.py 9002 .
# 浏览器打开 https://<主机>:9002/ → 任意键 → 主菜单「小游戏」→「灵机应用程序」→「黄金矿工」
```

------------------------------------

# 2026-11-06（水波：灵机 App 版 + 图形层位移贴图接口）

参照 `infer/ui_ripple.c`（C 版 `STATE_RIPPLE`）做一份**灵机（Animac）应用程序**版水波，
只针对 **WASM-GUI** 目标（`build_wasm_gui.sh` 产物，不是 `infer/web/` 那条 LLM-only 路线）。

## 1. 分工：物理在 Scheme，像素在 C

| | C 版（`ui_ripple.c`） | 灵机版（`app/ripple.scm`） |
|---|---|---|
| 波场分辨率 | 160×120（半分辨率，2×2 px/格） | **80×60（4×4 px/格）** |
| 波场推进 | C，定点 int16 | **Scheme，浮点列表 + set_item!** |
| 逐像素折射采样 | C，2×2 块最近邻直写帧缓冲 | **C（新接口 `Gfx.draw_image_warp`）**，位移栅格双线性放大 + 纹理双线性过滤 |
| 输入 | `key_event` 触屏电平 | `(native Touch)` 电平采样 |
| 退出 | 硬 A/ESC 直接切 `STATE_GAME_MENU` | `app/lib/exit.scm`（右上角按钮 + 全局确认框） |
| 底图 | `PLATFORM_ROOT_DIR "/wp.png"` | `"wp.png"`（相对业务资源根 = 同一文件） |

为什么必须拆：解释器每帧最多只有宿主给的 312500 条 IL 指令，**逐像素读 320×240 个纹素
在 Scheme 里不可能**（系数差 3 个数量级），而波场推进每格只有十几次读写、量级合适。
实测（浏览器 wasm32，本机 Chromium）单步耗时 80×60=**18.5 ms**、120×90=40 ms、
160×120=72 ms（30fps 预算 33 ms）→ 取 80×60，整拍实测稳定 **29~30 fps**。

## 2. 图形层新增：`gfx_draw_image_warp` + `(Gfx.draw_image_warp ...)`

```
gfx_draw_image_warp(gfx, path, x, y, w, h, gw, gh, const float *disp, float neutral)
→ (Gfx.draw_image_warp path x y w h gw gh disp neutral) : Boolean
```

- **语义**：图像先缩放到 `(w,h)`，再对每个目标像素按 `gw×gh` 位移栅格做**以矩形中心为
  原点的径向缩放**采样：`scale = disp[j*gw+i] / neutral`，`src = (dst − 中心)·scale + 中心`
  （越界钳位）——即 `ui_ripple.c` 的 `wr_render` 里那两行 dx/dy 位移公式的通用化。
  `neutral` 是"无位移"的基准值（水波用 1024），故 App 可以直接把 `1024 − amp` 当作位移表。
- **两处平滑**：栅格不必与目标尺寸成整数倍关系——按**格中心双线性插值**放大到像素
  （所以 4×4 的粗格子不会露出块状边界）；纹理采样用**双线性过滤**（最近邻会把连续的
  位移量化为整数像素，折射出现马赛克；实测该问题在 4×4 格上非常明显）。
- **复用既有设施**：纹理走 `graphics.c` 的图像 LRU 缓存（键 = 路径 + 目标尺寸，RGB888，
  像素**不进解释器堆**）。为此把 `gfx_draw_image` 里的"查缓存 / 解码缩放 / 回填缓存"
  抽成 `gfx_image_texture_get()`，两条绘制路径共用（`gfx_draw_image` 行为不变）。
- **本地库**：`am_native_Gfx` 新增一条；`disp` 为行优先的 `gw*gh` 个数值，逐元素转 float
  进一份**跨调用复用**的暂存缓冲（`platform_malloc`，不每帧申请）；列表长度不足/参数非法/
  读图失败一律软失败返回 `#f`。RGB565 帧缓冲走行指针直写，RGB888 走 `gfx_write_pixel_rgb565`。

## 3. App（`infer/app/ripple.scm`）

- 波场与 C 版同一套离散格式：`amp' = (上+下+左+右)×0.5 − amp_上一时刻`，再衰减 1/32
  （C 版是算术右移 5 位）；双缓冲两页、写完翻页；`disp[i] = 1024 − amp[i]` 每拍原地刷新。
- **页布局带一圈零填充**（上下左右各 1 格，行跨度 `GW+2`）：邻格读取不必做边界判定，
  比 C 版"只填上下两行 + 左右分支"少两条 `if`（每格省下的指令在 4800 格 × 30fps 上是实打实的）。
- 触点激发：触点所在格及其邻域**累加**震源振幅（C 版 `wr_disturb` 同款）。落点半径取
  2 格（20×20 px，与 C 版 7×7 个半分辨率格 = 14×14 px 同量级），振幅取 1024
  （C 版 512 的 2 倍：粗格子的位移场要被双线性放大，单次点按的视觉强度会打折）。
- 与 C 版的观感差异（已在源码头部注明）：格子大 2 倍 ⇒ 波前**在屏幕上**约快 2 倍；
  折射是平滑的而不是 2×2 方块感。
- 底图读不到时画与 C 版同文案/同坐标的错误画面（16px「水波」+ 12px 三行提示），并切到
  200ms 慢拍，只为让硬 A/ESC 仍能弹退出确认框。

## 4. 验证

- 宿主回归台（真图形层 + 真解释器，RGB565 + 虚拟时钟）：`APP=ripple` 120 帧刷新 40 次
  （33ms/拍）`err=none`；宿主回归台全部 App（36 个，含新 App）`err=none` 无回归（含 `gfx_draw_image`
  重构涉及的 Markdown/相册读图路径）。
- 新增 `tools/e2e/browser/regress_ripple.js`（真 WASM + 真无头 Chromium）：**12/12**——
  静水期两帧逐字节相同、底图确已绘制、按住 (100,150) 出现波纹（与静水帧差异 >1.5 万字节、
  影响面 >1.5 万像素）、按住期间波场持续演化、拖动生新波源、松开后继续传播、
  刷新率 29.4 fps、硬 Esc → 确认框 → Enter 回列表、无 pageerror。
- `regress.js` 存活回归：ripple `refreshΔ=113/4s`，与其它 38 个 App 一同 80 态稳定。

## 5. 运行

```bash
cd infer && bash build_wasm_gui.sh            # 产物 infer/webgui/
cd infer/webgui && python3 ../tools/https_static_server.py 9002 .
# 浏览器打开 https://<主机>:9002/ → 任意键 → 主菜单「小游戏」→「灵机应用程序」→「ripple」
#   （列表项 = 源码文件名。按住/拖动画面即起波纹；右上角「退出」或硬 Esc 退出）
```

### 二轮修复（水波不是圆形对称：折射位移锚错了地方 + 一处三参数算术踩坑）

**症状**（用户报告）：水波看起来不是圆形对称的，波纹中心不在触摸点。

**根因一（显示侧，设计错误）**：上一轮照搬了 `ui_ripple.c` 的位移公式——
`disp = 1024 − amp`，然后**以**画面中心**为原点**做径向缩放（`src = (dst − 中心)·disp/1024 + 中心`）。
位移量与"离画面中心的距离"成正比，于是：波场再圆，折射形变也永远**对称于屏幕中心**，
偏心触点的波纹被拉成"离屏心越远越夸张"的斜形变。C 版没暴露该问题只是因为它把半屏
2×2 块直写、且测试点大多在屏心附近。

**修复一**：折射位移改按**水面法向**算，即**高度场梯度**：
`src = dst − strength·∇h`（∇h 用逐格中心差分，双线性插值到像素）。
圆形外扩的波纹，梯度天然沿半径方向 ⇒ 折射纹样**圆形对称于波源（= 触点）**，
与波源在屏幕何处无关。图形层接口随之替换：

```
gfx_draw_image_warp(gfx, path, x, y, w, h, gw, gh, const float *disp, float neutral)
  → gfx_draw_image_refract(gfx, path, x, y, w, h, gw, gh, const float *hgt, float strength)
(Gfx.draw_image_warp ... disp neutral)  → (Gfx.draw_image_refract ... height strength)
```

`disp`（1024 = 无位移）改成 `height`（0 = 平静水面）；`neutral` 变成 `strength`
（单位坡度的像素位移量，水波取 0.02）。梯度行有跨调用的暂存缓冲（`platform_malloc`，
不每帧申请），位移另加 ±1/4 画面的安全钳位。App 侧 `disp` 列表随之改成 `water`（水位表，
`set_item! water ci na`）——**注意不能叫 amp**：`step` 里的局部量 `amp` 是单格振幅，
同名会遮蔽模块级列表。

**根因二（物理侧，解释器陷阱）**：`app/ripple.scm` 里写了 `(+ pcur P 1)` /
`(+ pnxt P 1)` ——**三参数算术**。本解释器的算术/比较运算符固定二元，多传的参数
不报错、结果**未定义**（残留操作数栈上）。实测该式在 `pcur = 0` 时凑巧给出正确值、
`pcur = PAGE`（第二个时间片起）时给出错误基址 ⇒ 读写指向错页/错位 ⇒ 波场被就地更新
（读到刚写出的邻居），波前只朝 +i/+j 方向扫，**本身就是非圆形对称的**。
修复：抽出常量 `BASE = (+ P 1)`，所有页面基址写成 `(+ pcur BASE)` / `(+ pnxt BASE)`
（二元嵌套）。审计：`grep` 全 `app/` 只有这一处三参数算术。

**验证（两级判据）**
1. **旋转对照（严格的几何判据，新增 `tools/e2e/host/check_ripple_symmetry.sh`）**：
   把底图旋转 180°（`wp_rot.png`，跑完即删），触点同时换成它的 180° 对极点
   （`(x,y) → (319−x, 239−y)`），跑同一时长的两帧。若位移只由梯度决定，B 旋转 180°
   后应与 A **逐字节相同**（双线性过滤与格中心插值在点反射下都是精确对称的）。
   实测触点 (60,60) 与 (240,180)：**差异 0/75040 像素、最大差 0 → PASS**；
   修复前同口径 **17.8% 像素差异、最大差 181**。
2. **浏览器回归（`regress_ripple.js` 新增两条）**：偏心触点 (60,60) 单点按住 700ms，
   与静水帧对比 ⇒ **变化像素质心 (61.5, 66.5) ≈ 触点**（< 8px），90% 分位半径 65px
   （扰动是触点周围的圆盘）。旧实现质心会被拉向屏心（实测 (140,101)、r90=226px）。
   chromium **14/14**、chrome-headless-shell **14/14**。
3. 宿主回归台全部 36 个 App `err=none`（含新 App，`refreshΔ=40/120 帧` = 33ms/拍）。

------------------------------------

# 2026-11-06（弹幕陨石：自动驾驶 AI —— 避弹优先、兼顾瞄准的代价场；四种模式组合都跑通）

给弹幕陨石 App 加了第三枚按钮「手动 / 自动」（HUD `134..164`，**默认手动**），打开后 AI 接管
操船：目标是"**先保证不死，再尽量多打**"。在游玩区按下手指 = 立刻交还手动（HUD 按钮不算）。
四种模式组合（常规/无限弹幕 × 无限命/三条命）下都能用；三条命模式实测连续 26 分钟游戏时间内
**一条命都没掉**。

## 1. 决策模型（`ai_cost` / `ai_think`）

每个候选位置 (x, y) 算一个代价，取代价最小者：

| 项 | 内容 |
|---|---|
| 威胁项 | 对每颗陨石解"未来碰撞时间"：飞船视作钉在候选点的静止圆，陨石走直线，求 \|dx+vx·t\|<R 与 \|dy+vy·t\|<R 同时成立的最早 t（R=半径+9）。t<1s 按 `200·(1−t)²` 罚，t<0.35s 再加 300。横掠型（vy≈0）走同一条公式（y 轴区间退化）。 |
| 近距罚 | 0.9s 内"最近接近距离"小于安全圈 `R₂=半径+16` 就扣 `80u² + 120u/(0.4+t)`（u=1−d/R₂）。**光判"会不会撞"不够**：擦着过去（余量 2px）也算安全，下一拍任何移动都可能撞上 —— 这一项给 AI 一个排斥场。 |
| 瞄准项 | 对"还来得及打"的陨石（1.0s < 到达候选高度的时间 < 4s）：候选位置对准它届时所在列（±11px）就减 `22/(1+0.3t)`。弹幕柱对准了才打得掉。 |
| 平滑项 | `0.10·(|Δx|+|Δy|)`，避免每拍横跳。 |

候选 = **21 个 x**（14.6px 步长，覆盖游玩区）；常规高度 `y=206`（贴下方 = 预警时间最长）。
**必须两维**：横掠型陨石会横扫整个 x 宽度，横着躲是躲不掉的，只能上下错开 —— 故当"常规高度
最小代价 > 25"时追加 `y=172`、`y=138` 两档再比一遍（只在真被逼时才付这份计算）。
另外候选代价取 `C(候选) + 0.8·C(下一拍位置)`：AI 每拍只走 30%，先算"这一步要迈到哪儿"安不安全
—— 少了这一项，AI 会**一步跨进**陨石的碰撞圈（实测第一版就是这样被撞的）。

## 2. 迭代过程（失败方案也记下来，省后人时间）

1. **1D（只有 x）+ 线性威胁罚（30/s）**：三条命模式下 2 分钟就被横掠型连撞三次。两个原因：
   "待在原地"的平滑代价比远处威胁的罚分更便宜 → 它拖到最后一刻才躲；而且横掠型横扫整个屏宽，
   纯横向无解。
2. **加两档垂直躲避 + 平方威胁罚（180·(1.6−t)²）**：仍然第 6 波被撞 —— 陨石还在屏幕外时就该
   开始躲，且"擦边过（余量 2px）"被判成安全。
3. **最终（近距排斥场 + 下一拍位置 + 瞄准权重调平衡 + 每拍 30% 步长）**：稳。三处权重是试出来的：
   近距罚太强（R₂ 给到 +20、罚分 300）会把 AI 逼成"一直在逃、不敢对准"，反而 4 波就死。

## 3. 实测（宿主回归台，虚拟时钟 → 完全确定性）

| 组合 | 24000 帧（≈6.4 分钟） | 80000 帧（≈26 分钟） |
|---|---|---|
| 常规弹幕 + 三条命 | 第 27 波 / 6125 分 / **3 命全在** | 第 55 波 / 21955 分 / **3 命全在** |
| 无限弹幕 + 三条命 | 第 28 波 / 10665 分 / **3 命全在** | 第 56 波 / 37835 分 / **3 命全在** |

- 对照（同 24000 帧、三条命、不碰屏幕）：**手动必死**（结局浮层 16106 px）。
- **开销**：AI 开 + 无限弹幕饱和时单拍 ≈**150k 条 VM 指令**（时间片二分：150k 时 7995/8000 帧、
  200k 起完全无差别），仍在一帧 312500 之内 → 浏览器仍是 30 Hz；手操时零开销。

## 4. HUD（第三枚按钮后的重排）

`波/分` 回到左侧（4..82）→ 生命图标 92..129（仅三条命模式）→ **手动/自动** 134..164 →
**常规弹幕/无限弹幕** 168..220 → **无限命/三条命** 224..268 → 退出 280..316。
三枚模式按钮一律放在 **x ≥ 134**：宿主在屏幕左上角（实测 `x<20 且 y<100`）那一小片区域的点按会走
别的分支，`Touch` 电平到不了 App —— 按钮放那儿会点不动（浏览器三引擎都能复现）。

## 5. 验证

- `tools/e2e/native/test_danmaku.py`：**26/26**（新增 6 项：默认手动 / 切自动变绿 / AI 不碰屏幕也会
  自己挪窝（5 次采样摆幅 > 10px）/ AI + 三条命 24000 帧不死 / 手动不动必死 / 长跑得分数字区变化）。
- `regress_danmaku.js`：三引擎（Chromium / chrome-headless-shell / Firefox）**各 35/35**
  （新增 3 项按钮链路；AI 的行为质量交给上面的确定性用例 —— 真实时钟下 AI 完全可能"按最优解原地
  不动"，用像素位移当判据会假失败）。
- 宿主回归台：39 个 App 全 `err=none`。顺带把 `SCHEME_APP_MAX` 的注释从"当前 33 个"更新到 39 个
  （上限 48 仍有余量；这个数变了要一起改）。

------------------------------------

# 2026-11-07（鹦鹉笼之外：纯 Scheme 的语言模型推理测试台 —— 不依赖 LLM 本地库）

需求：参照上游 `/home/bd4sur/Animac/test/llm.scm` 这个语言模型推理测试用例，在本项目的
**WASM 目标**（`build_wasm_gui.sh` → Emscripten WebAssembly GUI，不是 `infer/web/` 那条
LLM-only 路线）上写一个**完全纯 Scheme** 的等价 App：不得使用仓库里现成的 LLM 本地库
（`vendor/Animac/am_native_LLM.c`），而要参照它用 Scheme 重新实现一个**完全功能等价**的
库（权重读取与解析含 base64 解码、词元编解码 tokenizer、矩阵乘法），整个推理链路跑在
灵机解释器里。随后又按需求把交互重做成**两个互斥场景**（见 §2）。

## 1. 产物

| 文件 | 作用 |
|---|---|
| `infer/app/lib/llm_pure.scm` | **纯 Scheme 的 LLM 库**：与 `(native LLM)` 同名同语义的 `init/get_config/get_param/encode/decode/matmul`，外加流式加载三段式 `init_begin/init_feed/init_end` |
| `infer/app/llm.scm` | 测试台 App（GUI）：顶部两个**互斥**场景标签 + 各自的交互/渲染；每拍只推进一步 |
| `infer/app/lib/llm/{sort6,psycho}/{meta.scm,mNN.b64}` | 模型数据：base64 分块（sort6 = 1 块 64KB；psycho 230k = 14 块 × 96KB），`meta.scm` 由生成脚本产出 |
| `infer/tools/gen_llm_model_b64.py` | 把 Nano F32 模型 `.bin` 切成 base64 分块 + 生成 meta（含往返自检） |
| `infer/tools/llm_equiv/{nref.c,native_shim.scm,run.py,README.md}` | **等价性证明工具**：同一份脚本分别链接本地库与纯 Scheme 库，逐行 diff stdout |
| `infer/tools/check_scm_arity.py` | 灵机 Scheme 源码的**调用元数 / cond 子句**静态检查（灵机编译器不检查参数个数，见 §5 的坑） |
| `infer/tools/e2e/browser/regress_llm.js` | 浏览器端回归（真 WASM）——本目录按惯例不入 Git |

`infer/build_wasm_gui.sh` 同步改动两处：①构建期把 `app/lib/llm/{sort6,psycho}` 拷进业务
资源目录；②manifest 里 `app/lib/llm/**` **降为 p2 后台挂载**（它是 1.4MB 数据，不属于
"启动必需"，App 侧缺文件会自动等待重试）。

## 2. 两个互斥场景（2026-11-07 二轮：交互重做）

顶部两个标签页 `[场景1 · LLM 排序] [场景2 · 可视化生成]`，点按即切换，**任一时刻只有一个
场景在跑**；切场景时该场景的输入/进度归零。模型是全局单例（与上游本地库一致），故切到
需要另一个模型的场景时会重新载入（载入面板显示 `已读块/总块` 进度条），载入完成后自动跑
自检。

- **场景 1「LLM 排序」**：6 个数字槽位（像银行密码/短信验证码）+ 数字软键盘
  （`1 2 3 / 4 5 6 / 7 8 9 / 退格 0 输入`）。按「输入」后把 6 个数字**直接拼成** 6 字符串
  （如 `351420`）编码成 6 个 token 送进 sort6 模型，用全局自注意力逐位输出排序结果；
  该小模型在含重复数字的序列上有误差，故**每轮温度采样（T=0.8）、跑 5 轮、逐位投票取多数**
  （平票取较小值）作为最终结果。
  版式（2026-11-07 重排，**结果一目了然**）：输入槽位常驻顶部 → 中间是「输入/参考 + 5 轮明细
  + 一致 k/5 次/差 n 位」→ **底部六个方框（与输入槽位同尺寸 40x30、同排布 x0=20/step=48，16px 字）显示排序结果**；运行中方框
  显示"目前为止的逐位多数"（每轮结束 `vote_collect` 更新一次），标题行写「实时投票 / 第 k/5 轮」。
  「重新输入」按钮移到结果区右上（`208,165,92,17`），自检摘要移到标题栏右侧。
  推理是 6 次预填充 + 6 次生成前向；**k/v 只由 (token,pos) 决定**，所以只有第 1 轮需要
  预填充，后 4 轮直接生成（省 5×6 次前向，功能完全等价）。
- **场景 2「可视化生成」**：**没有任何输入入口**。提示词固定为「**人类的本质**」，
  种子每次取**当前时间戳**（`Time.utc_sod` + 轮次×7919，保证同一秒内连点「重来」也
  拿到不同随机流）初始化 App 内 LCG，逐 token 温度采样（T=0.75）、**逐字流式**输出并
  实时统计 TPS；生成长度 `N_GEN = 64` 步（含 5 步提示词预填充 → 约 59 个词元，
  wasm 上 ≈24 秒）。底部两个按钮：生成中显示「停止生成」（保留已输出部分并显示
  `已停止 · TPS …`，`[gen] … stopped=1`），常驻「重来」只换种子、提示词不变。
  **左列是模型框图 + 推理过程可视化**（见下），标题栏有「框图 开/关」开关：
  关掉后文字占满整行。

### 场景 2 的推理过程可视化（参照 `ui_llm.c`）

上游 `ui_app_llm_model_diagram_draw` 在推理中由 `llm_observation` 钩子反复重画：
左侧是 Nano 变换块的结构框图（残差 ×2、Attn Norm、Q/K/V、RoPE、Q*K、Mask→softmax→A*V、
O、FFN Norm、W1/W3/SiLU、⊗、W2），中间一列层条（当前层青色），右侧 top-6 候选词元；
相位（`NANO_LLM_PHASE_*`）命中哪个部件就把哪个部件点亮成绿色。

本 App 用 Scheme 复刻了同一套（`vis_set` / `vis_draw` / `dg_*`）：

- **左列（146x190）**：同一拓扑的紧凑版，纵向按 4/5 压缩（数据自下而上，与上游一致）；
  层条 `n_layer+2`（顶层=分类器、底层=embedding）用一条**竖脊线把灰/青小方块串起来**
  （方块画在线上，缝隙处露出直线，一眼看出"逐层往下推进"）。左列**不再画**黄色候选词元
  列表——候选信息全部交给右侧概率图（那里既有词元文本也有真实概率与采样指针）。
- **相位驱动 + 一帧一相位（不跳帧）**：前向按**相位**拆成可逐步推进的状态机
  （`fw_begin` → `fw_phase_step` ×N：每层 8 个相位 + final norm / classify / sample）。
  **开可视化时一拍只推进一个相位**、画一次框图并立刻推屏——浏览器只在 JS 任务之间
  合成上屏，一拍里画多少次都只会看到最后一张，所以"每步都画出来"必须靠"一帧一相位"。
  关可视化时同一拍内跑完全部相位（全速）。
  实测（宿主回归台，呈现帧序列）：开框图时 **1880 帧呈现、相邻重复 0 帧、去重 1880**，
  即每一步都真的画出来了；浏览器端 e2e 另按 (帧序号, 帧内容哈希) 采样断言
  「帧序号持续增长」且「新帧内容必变」（陈旧=0）。
  速度（**宁可慢**，过程比速度重要）：关框图 59 词元 20.4s（TPS 2.89，一拍一 token）；
  开框图一拍一相位，节拍 20ms（50 帧/秒，刻意低于 60Hz 合成率，确保每一相位都被 paint）
  → 约 1.4 秒/词元。开/关切换时用 `rearm_tick` 重挂定时器（16→20ms / 40ms）。
- **逐相位重绘只画"会变的部分"**：左列框图（`vis_draw` 自清自画）+ 右列状态行
  （`draw_s2_status`），token 完成时才整屏重画以更新流式文本。
- **右侧候选词元概率图（`draw_prob_chart`）**：6 根竖柱，柱高 ∝ 候选词元的**采样概率**，
  柱下是词元文本与整数百分比；**本次采样选中的那根用橙色 + 上方小三角指针**标出。
  概率口径与采样器一致：`p_i = exp((v_i − max)/T) / Σ_j exp((v_j − max)/T)`，**全词表归一**，
  T 取 `T_GEN`（贪心时按 1.0 显示）——因此柱高就是"真的会被抽中的概率"，不是相对值。
  因为是从全词表温度采样，选中的词元**约四成不在 top-6 内**；此时图会**追加第 7 列**
  把选中的词元画出来（列宽按列数自适应），确保指针永远指着真正选中的那一个。
  代价：每 token 多一遍 4096 次 `Math.exp`（≈20 万条 VM 指令 ≈ 10ms，只在开可视化时）。
- **top-6**：单遍扫描 logits 维护长度 6 的降序表（插入排序，O(vocab)），每 token 一次。
- **开关**：标题栏「框图 开/关」（`170,2,96,14`）翻转 `vis_on`；关闭时跳过所有
  `vis_set`/`vis_draw`（零开销），文字宽度从 27 个半角单位回到 51 个。
- **前向拆成「层」但同一拍跑完**：`fw_begin`/`fw_step`（等价性工具与排序任务走
  `llm_forward` 的 while 包装，数值完全一致）。**不要**把一拍拆成多层推进：
  事件循环轮数暴涨后，上游运行时每 32 轮一次的兜底 GC 也跟着暴涨，实测 59 个词元
  从 20.4s 掉到 36.7s（TPS 2.89 → 1.61）。

### 顺带修的 WASM 触屏缺陷：亚帧点按被吞（`hal_touch_wasm.c`）

场景 2 一拍要几百毫秒，期间浏览器事件全排队；恢复后一次点按的 DOWN/UP 会**背靠背**
回放在同一个生产者批次里，而 Scheme App 的 `(native Touch)` 是**按电平采样**（一拍一次），
于是「停止生成」按钮偶尔点不动。修法（只在 Scheme App 运行态启用，不影响菜单等 GUI
状态的点按语义）：

- `hal_touch_wasm.c` 新增 `wasm_touch_set_sticky()` / `wasm_touch_frame_boundary()` 与
  一个「亚帧点按锁存」：同批次内 DOWN 后又 UP 时记下坐标并锁存 **4 帧**（Scheme 的
  tick 约 2.5 帧一次，必须覆盖到它被采样一次），`touch_read` 在未按压但存在锁存时仍
  报"按下"，锁存过期后自然得到松开沿。
- `main_wasm_gui.c` 每帧开头调 `wasm_touch_frame_boundary()`，并按
  `global_state->STATE == STATE_SCHEME_APP_RUN` 设置 sticky。
- 不能用"读一次就清"：生产者的 `get_input_event` 也会读，会抢先把锁存吃掉（踩过）。
- 这也解释了 **e2e 脚本的坑**：生成进行中调用 `page.screenshot()` 会卡几十秒
  （主线程被推理占满 + 每帧推 300KB 帧缓冲），把后续点按全部推迟——实测「停止生成」
  的点按因此晚到 35 秒。回归脚本改成生成期间只转存 RGBA（判据用得到），收尾时才截图。

## 3. 为什么这样分层（接口照抄、数据外置、流式加载）

- **接口照抄**：库里的公开名字与本地库逐一对应，于是 App 里写
  `(import LLM "app/lib/llm_pure.scm")` 之后，调用点 `LLM.matmul`、`LLM.encode` … 与上游
  `(native LLM)` 的写法完全一样——上游测试用例几乎原样搬过来。
- **数据外置 + 分块流式解码**：灵机解析器按源码字符数预分配 token 数组（见 app/hwr.scm
  注释），1.3MB base64 塞进 `.scm` 字面量会吃掉解释器内存池；而 `(native Fs)` 单文件读入
  上限 128KB。故模型切成 96KB 分块，运行时逐块 `Fs.read` → `init_feed` 流式解码，峰值只有
  一个分块字符串。分块边界**不必对齐 4 字符组**（组状态跨 feed 保留），末组按 `=` 个数出
  1~3 字节。
- **两阶段解码器**：0=头(256B) 1=词表长度(4B) 2=词表数据 3=权重。权重区（占 96% 的字节）
  走 3 字节一起拼浮点的**内联快路径**，其余走逐字节慢路径。
- **浮点还原不用 32 位整数拼装**：wasm32 宿主的无符号整数有效宽度只有 27 位，F32 的 4 字节
  按 `符号/阶码/尾数` 分解，再由 2 的幂表还原——数学上精确，与 C 侧 `float→double` 逐位一致。

## 4. 与本地库保持一致（照抄不修正）的怪癖

`token_classifier` 恒与 `token_embedding` **内容相同但把柄不同**（上游 `get_param` 用同一份
数据另建一个列表，`eq?` 为假、`equal?` 为真）；贪心最长匹配分词（单字符一律取 charset 表
id、表中没有则 0，多字符最长尝试 17）；base64 跳过所有非 base64 字符；词表同名 token 与
单字符 charset 映射均**先到先得**；matmul 在 `d<=0 || n<=0` 时不写入、越界位置跳过写入。

## 5. 踩坑记录（都在 WASM 侧或交互重做时先炸，桌面/单测正常）

1. **阶段切换落在 4 字符组中间**：词表字段长度通常不是 3 的倍数（psycho 是 49732 字节，
   1516 = 3×505+1），切换点所在的那一组里 1 个字节属于词表、2 个属于权重。首版按"组开始时
   的阶段"整组走快路径 → 2 个权重字节被塞进词表缓冲 → 权重整体错位、`init_end` 报失败。
   修法：加 `gdirty` 标记，本组内发生过阶段切换就整组走逐字节慢路径。
2. **`-100000000` 这样的整数字面量在 wasm32 上溢出**：`am_int_t = int32_t` 且逻辑宽度只有
   27 位（约 ±6.7e7）。`sample_argmax` 的 `maxv` 初值写成 `-100000000` 后绕回成正数，
   于是**永远不更新** `maxv/maxi`，所有位置都取 0 号词元——表现为"排序输出全 -1、
   生成文本全是 `<|padding|>`"。修法：下界改用浮点字面量 `-1e30`。**灵机里跨 32/64 位移植
   时，凡"哨兵大数"一律写成浮点。**
3. **wasm32 的浮点是单精度**（`am_float_t = float`，64 位宿主是 double）：逐元素值与双精度
   差 1e-6 量级，连加十万项的和甚至漂到百分级。因此 App 内自检不能拿"求和"与桌面记录硬比：
   改为**长度 + 第 0/中/末元素 + 相对容差 1e-3**。上游本地库在 wasm32 上同样走单精度。
4. **上游 `list_to_str` 没有分隔符**：排序任务的输入是 6 个数字**直接拼成**的 `"351420"`
   （不是 `"3,5,1,4,2,0"`）。带逗号时输入变成 11 个 token、超过 `block_size=6`，模型输出
   退化成逗号/字母。
5. **灵机 `cond` 的每个子句只接受单个表达式**（manual.md §4.2）：写成
   `((== s2_stage 1) (if tap_new …) (if (gen_step) …))` 不会报错，但**第二个表达式不会
   被执行**——表现为"场景 2 一直停在 `推理中 0/28`、画面不再刷新"。
6. **进度条"不动然后突然完成"**：载入是每拍喂**一整块** 96KB base64（≈0.4s、十几个
   时间片），期间只画一次 → 看起来卡住。改成每拍只喂 `LOAD_SUB = 12288` 个字符
   （4 的倍数，解码组不跨段），进度按「块 + 当前块已喂比例」折算 → 条子逐格推进
   （实测 20%→34%→50%@三帧），退出键/标签页始终可点。总耗时不变，观感完全不同。
7. **调用参数个数不符不会报错，而是把多余值留在操作数栈上**（manual.md §4.3）：本 App 曾
   写出 `(log (String.concat "a" x) (String.concat "b" y))`（`log` 只收 1 个参数）与括号错位
   的 `(Gfx.draw_text (String.concat …) 10 20 …)`，症状分别是"某分支再也不执行"和"摘要行
   画不出来"。长 concat 链一律用中间变量拆成几条语句；并加了静态检查工具
   `infer/tools/check_scm_arity.py`（白名单元数 + cond 子句形状，零误报跑过全部 App）。

## 6. 验证

- **等价性（最强证据）**：`python3 infer/tools/llm_equiv/run.py` →
  `PASS：本地库与纯 Scheme 库的 76 行输出逐字节相同`。比对内容包括两个模型的配置、14 个权重
  张量的长度/求和/首中末元素、classifier 的独立性语义、多组文本的 encode id 与 decode 往返、
  词表前 40 项、逐位置前向的 argmax 与 logits 抽样、**5 轮温度采样的排序输出与逐位投票**、
  以及 28 步温度采样生成的完整文本。测试脚本里的前向/采样代码是从 `app/llm.scm` 按顶层
  define 名字**抽取**的（不是手抄副本），因此它证明的就是 App 真正在跑的那份代码；随机源
  是 App 内的 LCG（两条实现共用同一份 App 代码），所以连采样结果都逐位相同。
- **WASM GUI（真浏览器）**：`node tools/e2e/browser/regress_llm.js https://127.0.0.1:9002/index.html chromium`
  → **33/33 PASS**（含可视化：左列框图非背景像素 > 3000、生成中"高亮绿"质心出现
  ≥2 个不同位置（相位高亮在推进）、「框图 关」后左列清空、「框图 开」后恢复）：软键盘输入（含一次退格）后按「输入」→ `[sort] input=351420` → 5 条
  `[sort] round` → `[sort] voted=… agree=k/5 diff=n`；切到场景 2 → `[gen] seed=… prompt=…`
  → 生成中帧缓冲持续刷新 → `[gen] text=…`（非空、无 `<|padding|>`）与
  `[gen] tokens=27 ms=… tps≈3.0`；切回场景 1 重新载入 sort6（第三次 `[chk] OK=`）；
  全程无 pageerror、自检无 `[NG]`。
- **宿主回归台**（gcc，虚拟时钟）：同一 App 用 `HAL_TOUCH_AT` 注入 6 位数字 + 「输入」，
  5 轮投票后 `voted=0 1 2 3 4 5 ref=0 1 2 3 4 5 agree=4/5 diff=0`；场景 2 生成
  `血，因为诸体系列的意识的关于它的规律，不是自在存在的形态`（27 词元 / TPS 3.16）。
- 其它 App 无回归（mandelbrot / mlp / gol 宿主回归台 `err=none`）。

## 7. 运行

```bash
cd infer && bash build_wasm_gui.sh                 # 产物 webgui/（模型数据走 p2 后台挂载）
cd webgui && python3 ../tools/https_static_server.py 9002 .
# 浏览器打开 https://<主机>:9002/ → 主菜单 [7] 小游戏 → 灵机应用程序 → llm
```

模型数据重新生成（换模型时）：

```bash
python3 infer/tools/gen_llm_model_b64.py <model.bin> infer/app/lib/llm/<name> \
        --name <name> --label "<中文名>"
```

两个 F32 nano 模型来自上游 Animac 发行版的 `test/llm_model.scm`（psycho 230k =
huggingface `bd4sur/Nano-Psycho`；sort6 = 6 元升序序列生成模型）。**本项目自带的
`linglong/llm/*.bin` 是 Q80 量化权重，本地库与纯 Scheme 库都只支持 F32，不能用于本 App。**

# 2026-11-07（研究：灵机解释器时间片/GC 参数在 WASM 目标下的调优）

需求：研究 `infer/vendor/Animac` 里与时间片长度、GC 周期与策略相关的可调参数，
在 **WASM-GUI 目标**（`build_wasm_gui.sh`，不是 `infer/web/`）下、针对 `app/` 现有 39 个
App，在**内存波动平稳**的前提下尽量提升性能。**本轮只做研究、不改产品代码**，
报告见 `doc/灵机解释器参数调优研究-WASM.md`，结论如下（供决定是否落地）。

## 1. 结论（推荐改两个宏，其余保持）

```diff
-"$EMCC" -O3 -ffast-math -DNANO_WASM_GUI \
+"$EMCC" -O3 -ffast-math -DNANO_WASM_GUI \
+    -DAM_GC_HEAP_HIGH_WATER_RATIO=0.65 \
+    -DAM_GC_HEAP_CRITICAL_RATIO=0.80 \
+    -DAM_GC_PERIODIC_INTERVAL=16 \
```

保持默认：`AM_HEAP_COMPACT_INTERVAL=1`、`AM_GC_WATERMARK_CHECK_STRIDE=256`、
`AM_GC_HEAP_FRAG_*`、`MB_TIMESLICE_TICKS=312500`、池 32MB。三者都有实测理由，
不是"没试"（详见报告 §4）。

## 2. 三条关键事实（解释了为什么是这样调）

1. **GC 成本 ∝ 把柄表长度（两趟 O(N) 扫描 + 逐把柄哈希查找），不 ∝ 活跃字节**：
   `mandelbrot` 稳态活跃堆仅 30KB 却每轮扫 16.7k 个把柄（1.4ms/宿主、≈5ms/浏览器）；
   `2048` 只有 1.5k 把柄（0.14ms）。所以"回收得更勤"只多付 ~30% 扫描量
   （垃圾没长大就被收），单次停顿却能减半 —— 这是本次最有效的旋钮。
2. **"堆水位比例"其实是唯一能设"堆绝对上限"的旋钮**：池 50/50 切开，运行期
   `am_allocator_pool_adjust_boundary()` 要求"堆扩张时 VM 区必须为空"（永不成立），
   所以堆 16MB 是硬上限，`0.75` 就是"12MB 回收"。但水位**有硬下限**：低于 App 的
   **瞬时活跃集**（实测最坏 `xray` 8.58MB / `hwr2` 常驻 5.01MB）就会 GC 风暴
   （0.50→xray 746 次 GC/600 帧；0.25→41,969 次，占运行时间 66%）。故取 0.65（10.4MB）。
3. **压缩频率不能降**：直觉"少压缩省时间"是错的 —— 不压缩后堆碎片化，标记/清除的
   局部性变差，总 GC 时间反而上升（7 App 实测 200.6ms → 235.0ms），峰值也更高；
   1200 帧长跑下 `COMPACT_INTERVAL=4/0` 让 `xray` 直接分配失败（堆无法扩界）。

## 3. 实测收益（宿主确定性台 + 真 WASM）

- 39 App × 600 帧（两配置交替跑、取 min-of-2 CPU 时钟，两者总 CPU 25115ms vs 25102ms）：
  GC 次数 762→1476，GC CPU 422.9→580.9ms（占 CPU **1.68%→2.31%**，+0.63pp），
  堆峰值均值 4796K→**3687K（−23%）**、最坏 12272K→**10594K（−14%）**，
  每 App 平均停顿普遍 −20%~−50%（mlp 1.89→0.91ms、xray 1.48→0.99ms、gol 2.08→1.48ms）。
- 长跑 2000 帧（8 个最重 App，`oom=0 / level2=0 / err=none`，`refresh` 完全一致）：
  峰值 hwr2 11.4→7.6MB、pathtracer 12.3→6.9MB、mlp 9.2→4.6MB、gol 12.2→10.3MB；
  最大停顿全部下降（mlp 2.92→1.52ms、fallingsand 3.88→2.84ms、mandelbrot 3.17→2.45ms）。
- 真 WASM（Chromium 无头，每 App 5s，顺序交替）：fps 全部不变；
  `mandelbrot` 堆峰值 12.0→8.8MB、>16.7ms 的帧 7→1；GC CPU 增量 5~20ms/5s。
- **语义不变**：用产品构建脚本编两份产物跑 `regress_det.js`，37 App 的
  state/refreshΔ/CRC 逐项相等（`forkqueue` 例外——它显示 `Diag.heap` 活对象数，
  同一产物连跑两次 CRC 都不同，本身不能当判据；`2048` 首轮的差异是 refresh 抖动，
  单独重跑一致）。

## 4. 时间片：现在这个 312500 是对的

WASM 实测 **22–23ns/指令**（与宿主 x86 的 20.7ns 几乎一致）：312500 → 最重 App
每帧 7.3ms（60Hz 预算 44%）；450000 → 10.5ms（63%，推进 +44%）；625000 → 13.7ms（82%，
pathtracer/flip/sky3/gol 开始掉帧 6~11 帧/4s）。时间片只决定"每帧推进量上限"，
不影响自定拍 App、也不影响峰值内存，动它等于拿掉慢设备的余量 → **不改**。

## 5. 复现

计测脚手架本轮放在 `/tmp/gcbench`（宿主 bench + 带 `AM_BENCH_GC` 的 vendor 副本 +
变体矩阵脚本）与 `/tmp/wsbench`（wasm 测量构建 + 浏览器 bench.js），未进仓库；
报告 §8 给了完整复现命令与判据。**注意两件事**：WASM 下 `CLOCK_THREAD_CPUTIME_ID`
不可用（返回 0，须用 `CLOCK_MONOTONIC`/`emscripten_get_now()`）；本机负载会让单跑
时间差到 ±30%，多配置比较必须**交替重复**并看 CPU 时钟。

------------------------------------

# 2026-11-06（弹幕陨石：AI 进攻性调优 —— 瞄准提前量算错导致"不出击"，修正后击杀 +48%）

反馈：AI 躲弹很灵，但**远处有陨石时只待原地不主动出击**。查下来是一个真 bug + 权重太软。

## 1. 根因（真 bug）：瞄准提前量用错了时间

老代码的瞄准项算的是"**陨石落到飞船高度**的时间" `tc = (ship_y − ay)/vy`，然后用
`xa = ax + vx·tc` 当弹幕柱的目标。但子弹是**从飞船往上飞**的，需要的是"**子弹飞到陨石那个
高度**的时间" `tb = (ship_y − 12 − ay)/弹速`。陨石下落 60px/s、弹速 390px/s 时两者差 **6~8 倍**，
预测列偏出几十像素 —— AI 于是在对着一片空气开火：**看起来就是"远处有石头也不出击、只待原地"**
（它躲弹的判据是对的，所以躲得很灵）。

修正后（`ai_cost` 里的瞄准项）：

```
tb = (candy − ay) / ai_bv            ; ai_bv = 本拍弹速（常规 390 / 无限 min(620, 390+15·波)）
xa = ax + vx · tb                    ; 子弹到达时陨石所在的列
du = |xa − cand|
if du < 13: cost −= 120 · (1 − du/13) / (0.6 + ta)     ; ta = (candy − ay)/max(vy,20) 到达紧迫度
```

即：**距离梯度**（正对着最值钱）+ **紧迫度加权**（越接近头顶越该先打掉，一颗弹就拆掉威胁）。
权重从老版的 `16/(1+0.4tc)` 提到 `120/(0.6+ta)`（约 7.5 倍），AI 才愿意跑过去打远处的石头。

## 2. 效果

- **行为**：AI 现在整场在跑（198 次采样覆盖 x = 14…306，**95% 的拍子在移动**），
  而不是"站在原地等石头靠近"。
- **A/B（24000 帧 ≈ 6.4 分钟，三条命，AI 开）**：

  | 弹幕强度 | 旧 AI | 新 AI |
  |---|---|---|
  | 常规弹幕 | 393 杀 / 第 27 波 | **582 杀 / 第 27 波（+48%）** |
  | 无限弹幕 | 818 杀 / 第 28 波 | **903 杀 / 第 28 波（+10%）** |

- **长跑（80000 帧 ≈ 26 分钟，三条命）**：常规 **1742 杀 / 第 55 波**、无限 **2948 杀 / 第 56 波**
  ——两种模式下 **3 条命一条没掉、无结局浮层**（"尽量不碰陨石"仍然成立）。
- 无限弹幕提升小是结构性的：它的 21 条弹道本来就覆盖全屏，击杀上限由弹量与屏宽决定，与瞄准关系不大。

## 3. 权重是试出来的（记录一下，免得以后重复扫）

| 变体 | 常规弹幕 24000 帧击杀 |
|---|---|
| 窗口 ±13、近距罚半径 +16、粘性 0.05（**采用**） | 582 |
| 窗口 ±16 | 563 |
| 窗口 ±16、近距罚半径 +13 | 582 |
| 窗口 ±18、近距罚半径 +13 | 602（+3%，没采用：离"擦边"更近） |
| 粘性 0.12（太黏） | 524（粘住旧目标、错过新石头） |

## 4. 开销

AI 开 + 无限弹幕饱和时单拍 ≈ **170k** 条 VM 指令（时间片二分：150k 时 7939/8000 帧、200k 起满帧），
仍在一帧 312500 之内 → 浏览器仍 30 Hz；手操零开销。

## 5. 验证

`tools/e2e/native/test_danmaku.py` **26/26**（含 AI 六项：默认手动 / 切自动 / 不碰屏幕也会挪窝 /
三条命 24000 帧不死 / 手动不动必死 / 长跑持续得分）；`regress_danmaku.js` 三引擎**各 35/35**；
宿主回归台清单里的 **36 个 App 全 err=none**（`app/` 目录下目前 39 个 .scm，清单以
`tools/e2e/host/run_host_regress.sh` 的 APPS 为准）。
