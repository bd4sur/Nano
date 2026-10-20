# vendor/Animac —— 灵机（Animac）解释器副本出处说明

本目录下的**全部** `.c` / `.h` / `.cpp` 文件均逐字节取自上游仓库，**不含任何本地改动**。
升级时把上游对应文件整份覆盖即可，不要在此目录内手改任何文件。

## 上游与版本

| 项 | 值 |
|---|---|
| 上游仓库 | `/home/bd4sur/Animac`（BD4SUR） |
| 同步基线 | commit `0f5e9d4`【测试】增加大量复杂测试用例并修复缺陷（2026-10-11） |
| 同步日期 | 2026-10-20 |
| 校验 | 本目录每个文件与上游工作树对应文件 `cmp` 全等 |

## 文件映射（上游 → 本目录，扁平化）

| 本目录文件 | 上游来源 |
|---|---|
| `animac_core.c` / `animac_core.h` | 仓库根目录（`amalgamate.sh` 生成物，上游原样分发） |
| `animac.h`、`am_*.h`（含 `am_host.h`、`am_native_*.h`、`am_repl.h`、`am_highlight.h`、`am_base.h`） | `include/*.h` |
| `am_host.c`、`am_native_{Math,String,System,Table,LLM}.c`、`am_repl.c`、`am_highlight.c` | `src/*.c`（解释器核心实现已并入 `animac_core.c`，故不单独收录） |
| `am_host_esp32.cpp` | `src/am_host_esp32.cpp`（ESP32/Arduino 宿主端口，本仓库现有构建脚本不含 `.cpp`，不参与编译） |

## 构建约定（与上游 `amalgamate.sh` 注释一致）

- 解释器核心 = `animac_core.c`（单文件，自带全部核心实现）。
- 宿主侧文件（`am_host.c`、`am_native_*.c`、`am_repl.c`）按需单独编译链接；本项目的
  `infer/build_wasm_gui.sh` 与 `android/app/src/main/cpp/CMakeLists.txt` 均以
  `vendor/Animac/*.c` 通配收录。
- `include/` 里的核心头文件同时被 `animac_core.h`（合并副本）和 `animac.h`（伞形头文件）
  引用，两者靠同名 include guard 互斥，任选其一包含都可以。

## 自定义本地库的位置

本项目自行实现的 native 库**不在此目录**，而是放在 `infer/` 根下：

- `infer/am_native_Gfx.c/.h` —— Nano 图形层（Nano_GFX）FFI
- `infer/am_native_Time.c/.h` —— 整数时间（utc_days / utc_sod）

它们必须遵守上游本地库契约（`doc/todo/stack_bal.md` §11 N1–N10，摘要）：

| 条目 | 要求 |
|---|---|
| N1 | 每条 `return 0` 路径**恰好压 1 个**结果值（净栈效应 −N+1） |
| N2 | 语义上不产值的语句型函数，**必须压 1 个 `#undefined`** |
| N4 | 失败路径也要压 1 个约定值（或按硬错误 `return -1`） |
| N5 | 每次 `am_process_pop_operand` 必须检查 `UINTPTR_MAX` 哨兵 |
| N6 | 同步成功路径**恰好**调用 1 次 `am_process_step`（压值辅助函数内含） |
| N7 | 阻塞型函数阻塞时压 0 不 step；唤醒值由运行时补压 |