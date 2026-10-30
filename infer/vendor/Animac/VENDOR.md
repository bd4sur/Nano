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
| 同步基线 | commit `0291412`（2026-10-21 02:47） |
| 合并产物生成 | `git archive 0f5e9d4` 导出干净源码树 → 在该副本内执行上游 `amalgamate.sh`（**不在上游仓库内生成任何文件**） |
| 校验 | `animac_core.c/h` 与上游仓库内自带产物、以及本次重新生成物三方内容一致（仅"生成时间"行不同）；其余文件与上游 `src/` `include/` 对应文件 `cmp` 全等 |

## 文件清单（17 个）

| 文件 | 来源 | 性质 |
|---|---|---|
| `animac_core.c`、`animac_core.h` | 上游 `amalgamate.sh` 产物 | 上游原样（不要手改） |
| `am_host.c`、`am_host.h` | 上游 `src/`、`include/` | 上游原样 |
| `am_native_{Math,String,System,Table,LLM}.c/.h` | 上游 `src/`、`include/` | 上游原样 |
| `am_repl.c`、`am_repl.h` | 上游 `src/`、`include/` | 上游原样（线性包含 LLM 库注册，故 LLM 库一并收录） |
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
| `src/am_host_esp32.cpp` | ESP32/Arduino 宿主端口；本仓库现有构建脚本只 glob `*.c`，不参与编译 |
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
| 2026-10-21（第二次） | `0291412` | ①**闭包帧分析父链修复**：词法父子关系改用 AST 父链接（parser 宏展开后释放 scope 对象，原实现导致 parent 恒空、自由变量 src 退化为运行期动态查找）；②**定时器回调不可重入**：进程已在异步回调中时顺延定时器（工单 #38）；`am_parser.c` 清理 scope 后同步清空 `ast->scopes`。本项目的 `am_native_Gfx.c`/`am_native_Time.c` 无需改动 |
| 2026-10-21 | `5c244e9` | 闭包变量访问改为「帧描述符 + 静态槽位寻址」（P0–P2）：新增 `am_frame.{h,c}`、5 条槽位指令（`load/store/set/call/tailcall_slot` 63–67）、模块磁盘格式新增帧描述符表；`am_repl.c` 无改动 |
| 2026-10-20 | `0f5e9d4` | 首次按 amalgam 形态同步（单值栈纪律、`,@`、异步帧 #36、Math 扩展） |

> 宿主侧 ABI 只增不改：本地库（`infer/am_native_Gfx.c`、`infer/am_native_Time.c`）在上游这两次升级中均**无需改动**仍可编译链接（契约 N1–N10 未变）。

## 本项目自行实现的本地库（不在本目录）

- `infer/am_native_Gfx.c/.h` —— Nano 图形层（Nano_GFX）FFI
- `infer/am_native_Time.c/.h` —— 整数时间（utc_days / utc_sod）
- `infer/am_native_Diag.c/.h` —— 虚拟机自观测（只读：操作数栈/调用栈深度、PC、指令区长度、
  闭包帧描述符与退化闭包计数、槽位统计、GC 计数、单调实时毫秒）；供 `app/xray.scm` 的
  "解释器 X 光"仪表盘使用，也是排查解释器问题时的现场仪器
- `infer/am_native_Key.c/.h` + `infer/nano_key_events.c/.h` —— 键盘输入（按键边沿事件旁路队列 +
  Scheme 侧 `Key.is_available/is_down/take_press/take_release/dropped`）；HAL 在注入 GUI 电平槽的
  同时另投一份边沿事件，GUI 读取路径零改动
- `infer/am_native_Fs.c/.h` —— 文本文件读写（`exists/is_dir/size/list/mkdir/read/write/append`，
  相对业务资源根、UTF-8、128KB 读入上限；WASM 目标另有 `hal_fs_persist_wasm.c` 把写入同步进
  浏览器 IndexedDB）
- `infer/am_native_Ui.c/.h` —— 宿主模态文本输入/输出（`input`/`input_with`/`message`/
  `message_titled`）：按上游契约 N7 把**进程**置 BLOCKED，宿主用现成的文本控件当模态框，
  提交后压值 + step + 置 READY + 重新入队（机制与 `System.read` 队列等待同源）

它们必须遵守上游本地库契约（上游 `doc/todo/stack_bal.md` §11 N1–N10）：

| 条目 | 要求 |
|---|---|
| N1 | 每条 `return 0` 路径**恰好压 1 个**结果值（净栈效应 −N+1） |
| N2 | 语义上不产值的语句型函数，**必须压 1 个 `#undefined`** |
| N4 | 失败路径也要压 1 个约定值（或按硬错误 `return -1`） |
| N5 | 每次 `am_process_pop_operand` 必须检查 `UINTPTR_MAX` 哨兵 |
| N6 | 同步成功路径**恰好**调用 1 次 `am_process_step`（压值辅助函数内含） |
| N7 | 阻塞型函数阻塞时压 0 不 step；唤醒值由运行时补压 |

## 构建接入

- `infer/build_wasm_gui.sh`：`ANIMAC_DIR=vendor/Animac; ANIMAC_SRCS=$(ls $ANIMAC_DIR/*.c)` 全量编译
- `android/app/src/main/cpp/CMakeLists.txt`：`file(GLOB ANIMAC_SRCS "${ANIMAC_DIR}/*.c")`

两者都按"目录内全部 `.c`"收录，因此本目录的增删文件无需改构建脚本；仅需保证
`infer/am_native_Gfx.c`、`infer/am_native_Time.c`、`infer/ui_scheme_app.c` 在业务源清单中。