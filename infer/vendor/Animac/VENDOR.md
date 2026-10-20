# vendor/Animac —— 灵机（Animac）解释器副本（单文件 amalgam 集成形态）

本目录采用上游 `amalgamate.sh` 注释描述的**单文件分发形态**：

```
解释器核心 = animac_core.c / animac_core.h   （amalgamation 合并产物，自带全部核心实现）
宿主外围   = am_host.c / am_native_*.c / am_repl.c  （按需单独编译链接）
```

原始头文件（`include/am_*.h` 23 个）与原始实现（`src/*.c`）**不随附**——它们已被合并进
`animac_core.*`；这也是上游对该形态的设计意图（见上游 `amalgamate.sh` 文件头注释）。

## 上游与版本

| 项 | 值 |
|---|---|
| 上游仓库 | `/home/bd4sur/Animac`（BD4SUR） |
| 同步基线 | commit `0f5e9d4`（2026-10-11） |
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
| `include/am_*.h`（23 个核心头） | 已合并进 `animac_core.h`，单文件形态不再随附 |
| `src/am_*.c`（23 个核心实现） | 已合并进 `animac_core.c` |
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

## 本项目自行实现的本地库（不在本目录）

- `infer/am_native_Gfx.c/.h` —— Nano 图形层（Nano_GFX）FFI
- `infer/am_native_Time.c/.h` —— 整数时间（utc_days / utc_sod）

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