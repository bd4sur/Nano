# vendor/Animac —— 灵机（Animac）解释器副本（单文件 amalgam 集成形态）

本目录采用上游 `amalgamate.sh` 注释描述的**单文件分发形态**：

```
解释器核心 = animac_core.c / animac_core.h   （amalgamation 合并产物，自带全部核心实现）
宿主外围   = am_host.c / am_native_*.c / am_repl.c  （按需单独编译链接）
```

原始头文件（`include/am_*.h` 24 个）与原始实现（`src/*.c`）**不随附**——它们已被合并进
`animac_core.*`；这也是上游对该形态的设计意图（见上游 `amalgamate.sh` 文件头注释）。

## 上游与版本

| 项 | 值 |
|---|---|
| 上游仓库 | `/home/bd4sur/Animac`（BD4SUR） |
| 同步基线 | commit `a60b599`（2026-11-25 11:51） |
| 合并产物生成 | `git archive a60b599` 导出干净源码树 → 在该副本内执行上游 `amalgamate.sh`（**不在上游仓库内生成任何文件**） |
| 校验 | 重新生成物与「上一基线（`771b7c7`）重新生成物」逐行 `diff`，差异**仅**为本次上游改动（内建过程元数表的编译期检查）与"生成时间"行；全部宿主侧文件（`am_host.*`、`am_repl.*`、`am_native_*.{c,h}`）自上一基线起未变（`cmp` 全等） |

## 文件清单（18 个）

| 文件 | 来源 | 性质 |
|---|---|---|
| `animac_core.c`、`animac_core.h` | 上游 `amalgamate.sh` 产物 | 上游原样（不要手改） |
| `am_host.c`、`am_host.h` | 上游 `src/`、`include/` | 上游原样 |
| `am_native_{Math,String,System,Table,LLM}.c/.h` | 上游 `src/`、`include/` | 上游原样 |
| `am_repl.c`、`am_repl.h` | 上游 `src/`、`include/` | 上游原样（线性包含 LLM 库注册，故 LLM 库一并收录） |
| `am_host_esp32.cpp` | 上游 `src/` | 上游原样；ESP32/Arduino 宿主端口（由本仓库 ESP32 PlatformIO 工程使用，构建脚本 glob `*.c` 时不参与 wasm/Linux 目标） |
| `animac.h` | **本项目自有** | amalgam 适配层：把上游伞形头文件的收录语义转发为 `#include "animac_core.h"` |
| `VENDOR.md` | **本项目自有** | 本说明 |

`animac.h` 是本目录内唯一非上游文件。上游的 `include/animac.h` 逐个 `#include` 23 个核心
头文件；在单文件形态下这些头文件不存在，故用本适配层等价转发（靠同名 include guard
与 `animac_core.h` 共存，任何包含顺序都不会重复定义）。**同步上游时不要覆盖它。**

## 未收录的上游文件（有意为之）

| 文件 | 原因 |
|---|---|
| `include/am_*.h`（24 个核心头，含 `am_frame.h`） | 已合并进 `animac_core.h`，单文件形态不再随附 |
| `src/am_*.c`（24 个核心实现，含 `am_frame.c`） | 已合并进 `animac_core.c` |
| `src/am_highlight.c` + `include/am_highlight.h` | 仅上游桌面 `main.c`（终端语法高亮）使用，本项目不使用 |
| `Makefile`、`amalgamate.sh`、`main.c`、`test/`、`doc/`、`typescript/` | 上游开发/测试/文档设施，非集成产物 |

## 同步上游的操作步骤

```bash
# 1) 在临时副本里生成合并产物（不污染上游仓库）
rm -rf /tmp/amalg_gen && mkdir -p /tmp/amalg_gen
( cd /path/to/Animac && git archive <commit> ) | tar -x -C /tmp/amalg_gen
( cd /tmp/amalg_gen && bash amalgamate.sh )

# 2) 覆盖本目录（保留 animac.h / VENDOR.md）
cp /tmp/amalg_gen/animac_core.c /tmp/amalg_gen/animac_core.h .
cp /path/to/Animac/src/{am_host,am_repl}.c /path/to/Animac/src/am_native_*.c .
cp /path/to/Animac/include/{am_host,am_repl}.h /path/to/Animac/include/am_native_*.h .

# 3) 同步后必须复核本项目自实现的本地库契约（见下）
```

## 变更历史

| 同步日期 | 上游基线 | 要点 |
|---|---|---|
| 2026-11-25 | `a60b599` | **内建过程参数个数不符改为编译期错误（工单 #35）**：原先内建过程（`+ - * / mod pow`、比较符、谓词、列表操作、IO、`call/cc`、`dynamic-wind`）在参数个数不符时只发射一条 VM 指令，而指令只是按下标从操作数栈弹固定个数的操作数，多余参数原样残留——**静默的栈污染与错误结果**（如 `(display (+ 1 2 3))` 显示 5）——现在编译期直接拒绝。实现：新增 `AM_BUILTIN_ARITY_MAP`（37 项，与 `AM_GLOBAL_BUILTIN_VAR`/`AM_BUILTIN_OPCODE_MAP` 同下标并列维护）与 `am_ast_find_builtin_var_index`，`compile_application` 在编译实参与发射指令之前检查（含 `call/cc`、`dynamic-wind` 两条特殊形式路径，`System.eval` 动态编译同样受检），诊断形如 `[Compiler] 编译错误: 内建过程 + 只接受 2 个参数，实际传入 3 个`；检查按拼写匹配 varid，被 ARN 换名的同名形参不误判。已知限制：内建过程经变量间接调用（`(define f +)` 后 `(f 1 2 3)`）时元数不检查（内建并非一等值）。<br>**对本项目的影响**：①**发现并修复两处真实缺陷**——`app/llm.scm` 的 TPS 计算 `(/ (* 1.0 gen_n 1000.0) el)`（3 个参数，旧核心下 `*` 只取两个操作数、`1000.0` 残留栈上，显示值错 1000 倍）与 `app/vcodec.scm` 的块网格线 `(* i BS sc)`（旧核心下坐标算错），已改为嵌套二元形式；②全量核对：55 个 App 源码用新核心做宿主编译扫描，**0 个编译期错误**（`app/*.scm` + `app/lib/*.scm` 均无元数违规）；③`ui_scheme_app.c` 的加载失败文案同步改为「Scheme 编译失败（参数个数不符或内存不足）」——原文案只说"可能内存不足"，会把新增的元数拒绝误导向内存问题（精确原因由编译器打到 stderr）。 |
| 2026-11-08 | `771b7c7` | **短字符串驻留改为弱驻留（工单 #39，修内存泄漏）**：`proc->strindex` 不再作为 GC 根，降级为"活串去重缓存"——`gc_sweep` 给每个存活对象（含 static/keepalive）打本轮 GC 轮次戳（复用 `gcmark` 低 31 位，新增 `am_object_get/set_gc_epoch`），sweep 之后 `am_strindex_prune` 只保留指向本轮存活 WSTRING 的表项；存活率低于 50% 时重建压缩驻留表，避免驻留表占的 VM 区只增不减；短串 hash/查询与候选 handle 改用栈上缓冲（不再每串向 VM 区申请临时缓冲）；模块加载时清空 gcmark 低位轮次戳。语义不变量：同内容短串只要**同时存活**仍在创建期去重，`eq?` 与 Table 字符串键保证不变。<br>配套可观测性：`System.memstat` 由四元组扩为**六元组** `(vm_capacity vm_used heap_capacity heap_used interned_count interned_bytes)`；OOM 从笼统的"指令执行异常"中拆出为 `[Runtime] 内存耗尽(OOM)` 并附内存池用量。<br>**对本项目的影响**：①memstat 是**末尾追加**，`app/xray.scm` 与 `app/forkqueue.scm` 只读 0–3 号元素（vm_cap/vm_used/heap_cap/heap_used），语义不变、无需改动；②这两个 App 显示的 VM/堆百分比与 `Diag.heap` 会**合法地变小**（不再有泄漏的驻留串），回归基线与上一版不同属预期；③本项目自实现本地库均不在 C 侧跨调用持有 Scheme 句柄（只存 wchar 文本缓冲），因此驻留表失去"永久根"属性不会造成悬空把柄；④修复对本项目**实际有效**：App 的 HUD/状态条普遍每帧 `String.concat` 拼文本，旧核心下这些短串永不回收。 |
| 2026-11-07 | `5f82267` | **GC 误扫静态容器引用的存活对象**：`alive` 位是「本轮已标记」语义，但 `gc_sweep` 对 static 对象直接 `continue` 从不复位，而 `gc_mark` 以 `alive` 作环检测短路——静态对象在第二轮 GC 时短路返回，其运行期子对象再也不会被标记而被 sweep 全部释放（句柄悬空）。触发面：`am_list_resize` 会原样复制对象头，故被 `push` 扩容的列表字面量（如 `'(0)`）即「静态容器 + 运行期子对象」。修复：`gc_sweep` 的 static 分支先 `am_object_set_alive(obj, -1)` 再 `continue`，恢复「每轮 sweep 复位表内全部对象 alive」的不变式。**此缺陷对本项目实际可达**：灵机 `app/` 普遍使用 `(define x '())` 后 `push` 建表的模式（如 `app/forkqueue.scm` 的 `pids`/`qr`/`ql`/`qs`/`qc`、`app/callcc.scm` 的 `u_lane`/`u_ch`），已用最小用例在本仓验证：旧基线编译的二进制 `(define l '())`+5 次 `push`+两轮 `System.gc` 后取元素即报 `[Runtime] 指令执行异常: PID=0 PC=37`，新基线正常输出。本地库契约 N1–N10 未变，`am_native_Gfx.c`/`am_native_Time.c` 等本项目自实现库无需改动。 |
| 2026-10-21（第二次） | `0291412` | ①**闭包帧分析父链修复**：词法父子关系改用 AST 父链接（parser 宏展开后释放 scope 对象，原实现导致 parent 恒空、自由变量 src 退化为运行期动态查找）；②**定时器回调不可重入**：进程已在异步回调中时顺延定时器（工单 #38）；`am_parser.c` 清理 scope 后同步清空 `ast->scopes`。本项目的 `am_native_Gfx.c`/`am_native_Time.c` 无需改动 |
| 2026-10-21 | `5c244e9` | 闭包变量访问改为「帧描述符 + 静态槽位寻址」（P0–P2）：新增 `am_frame.{h,c}`、5 条槽位指令（`load/store/set/call/tailcall_slot` 63–67）、模块磁盘格式新增帧描述符表；`am_repl.c` 无改动 |
| 2026-10-20 | `0f5e9d4` | 首次按 amalgam 形态同步（单值栈纪律、`,@`、异步帧 #36、Math 扩展） |

> 宿主侧 ABI 只增不改：本地库（`infer/am_native_Gfx.c`、`infer/am_native_Time.c` 等）在上游各次升级中均**无需改动**仍可编译链接（契约 N1–N10 未变）。

## 本项目自行实现的本地库（不在本目录）

- `infer/am_native_Gfx.c/.h` —— Nano 图形层（Nano_GFX）FFI
- `infer/am_native_Time.c/.h` —— 整数时间（utc_days / utc_sod）
- `infer/am_native_Diag.c/.h` —— 虚拟机自观测（只读：操作数栈/调用栈深度、PC、指令区长度、
  闭包帧描述符与退化闭包计数、槽位统计、GC 计数、单调实时毫秒、**本进程堆活对象数**
  `heap`——`System.memstat` 是池级口径，多进程下要区分单个进程的堆占用只能数它自己的
  把柄表）；供 `app/xray.scm` 的"解释器 X 光"仪表盘与 `app/forkqueue.scm` 的多进程 HUD
  使用，也是排查解释器问题时的现场仪器
- `infer/am_native_Key.c/.h` + `infer/nano_key_events.c/.h` —— 键盘输入（按键边沿事件旁路队列 +
  Scheme 侧 `Key.is_available/is_down/take_press/take_release/dropped`）；HAL 在注入 GUI 电平槽的
  同时另投一份边沿事件，GUI 读取路径零改动
- `infer/am_native_Fs.c/.h` —— 文本文件读写（`exists/is_dir/size/list/mkdir/read/write/append`
  以及 `remove`/`rename`/`abspath`），相对业务资源根、UTF-8、128KB 读入上限；`abspath`
  把资源根相对路径换成**宿主绝对路径**（App 间传递路径用，见下面 App 库的 `launch_with`）；
  WASM 目标另有 `hal_fs_persist_wasm.c` 把写入/删除/改名同步进浏览器 IndexedDB（文件管理器
  app/filemgr.scm 依赖 `remove`/`rename`；递归删除目录树由 Scheme 侧组合原语完成）
- `infer/am_native_Audio.c/.h` —— 音频**输入**与频谱分析（`is_available/open/close/info/read/
  capture/bin/sample/rms/level/set_level`）：麦克风经 `hal_audio_in.h`（WASM = getUserMedia + Web Audio，
  其它平台 = ALSA/ESP-IDF I2S），Hann 加窗 + `nano_fft.c` 的基 2 FFT + 对数幅度映射全在 C 侧
  （1024 点 FFT 在解释器里要 ~3×10⁵ VM 指令/列、每 23ms 一列，Scheme 侧跟不上），跨 FFI 只传
  标量；供 `app/spectrogram.scm`（声谱图）使用。`read` 顺带做 C 侧 FFT（"硬 FFT"），`capture`
  只交付原始采样（供 `app/lib/fft.scm` 的"软 FFT"在 Scheme 里变换）。**本库是第 17 个本地库**，
  见下面「本地库数量上限」
- `infer/am_native_Ui.c/.h` —— 宿主模态文本输入/输出（`input`/`input_with`/`message`/
  `message_titled`/`confirm`）：按上游契约 N7 把**进程**置 BLOCKED，宿主用现成的文本控件当
  模态框（`confirm` 借图形层的两按钮确认框），提交后压值 + step + 置 READY + 重新入队
  （机制与 `System.read` 队列等待同源）
- `infer/am_native_App.c/.h` —— 应用外壳/生命周期（`exit`/`exit_with`/`exit_now`/`hide`/
  `back_requested`/`modal_closed`/`is_active`/`name`/`platform`）：退出确认框走 Ui 的
  「退出型确认框」（N7 阻塞），确认后置退出意愿，由宿主 `ui_scheme_app.c` 在帧末销毁实例。
  **App 间拉起**：`launch`/`launch_with`（向宿主排一条“拉起另一个 App + 1 个字符串启动参数”
  的请求，帧末执行；调用者被挂起保留，单实例平台退化为销毁）与 `launch_arg`（目标 App
  取走启动参数，一次性）；名字存在性校验通过宿主注册的钩子
  `am_native_app_bind_lookup()` 完成（`ui_scheme_app.c` 用 `s_apps[]` 实现）

它们必须遵守上游本地库契约（上游 `doc/todo/stack_bal.md` §11 N1–N10）：

| 条目 | 要求 |
|---|---|
| N1 | 每条 `return 0` 路径**恰好压 1 个**结果值（净栈效应 −N+1） |
| N2 | 语义上不产值的语句型函数，**必须压 1 个 `#undefined`** |
| N4 | 失败路径也要压 1 个约定值（或按硬错误 `return -1`） |
| N5 | 每次 `am_process_pop_operand` 必须检查 `UINTPTR_MAX` 哨兵 |
| N6 | 同步成功路径**恰好**调用 1 次 `am_process_step`（压值辅助函数内含） |
| N7 | 阻塞型函数阻塞时压 0 不 step；唤醒值由运行时补压 |

## 本地库数量上限（AM_NATIVE_MAX_LIBS）

上游 `animac_core.h` 的 `AM_NATIVE_MAX_LIBS` **默认为 16**，本项目注册的本地库已达 17 个
（System/Math/String/Table/Gfx/Time/Diag/Touch/Imu/Key/Fs/Ui/App/Chat/Raster/Music/Audio）。
超出上限时 `am_runtime_register_native_lib` 会**静默失败**（返回 -1、无日志），表现为运行时
`callnative: 未找到native函数 Audio.open` —— 极难定位。

处理方式：该宏在上游是 `#ifndef` 保护（"宿主可覆盖"的构建选项），因此**不改本目录的源码**，
各构建入口统一加 `-DAM_NATIVE_MAX_LIBS=32`：

| 构建入口 | 位置 |
|---|---|
| `infer/Makefile` | `ANIMAC_DEFS`（pod_lite/pod/pod_pt35/tty/web） |
| `infer/mp135.mk`、`infer/luckfox.mk` | `CCFLAGS` |
| `infer/build_wasm_gui.sh` | `emcc` 命令行 |
| `infer/platformio.ini`（ESP32） | `build_flags` |
| `android/app/src/main/cpp/CMakeLists.txt` | `target_compile_options` |
| `infer/tools/e2e/{host,perf,browser}` 回归台 | 各自的 `gcc` 命令行 |

取 **32**（而不是刚好 17）：给后续新增本地库留余量，免得每加一个库都要动 8 个构建入口。
`infer/am_native_Audio.c` 顶部有编译期断言（`#error`）：忘记加该宏的构建会**编译失败**，
而不是在运行时静默失效。

## 构建接入

- `infer/build_wasm_gui.sh`：`ANIMAC_DIR=vendor/Animac; ANIMAC_SRCS=$(ls $ANIMAC_DIR/*.c)` 全量编译
- `android/app/src/main/cpp/CMakeLists.txt`：`file(GLOB ANIMAC_SRCS "${ANIMAC_DIR}/*.c")`

两者都按"目录内全部 `.c`"收录，因此本目录的增删文件无需改构建脚本；仅需保证
`infer/am_native_Gfx.c`、`infer/am_native_Time.c`、`infer/ui_scheme_app.c` 在业务源清单中。