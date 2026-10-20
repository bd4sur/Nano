#ifndef __AM_BASE_H__
#define __AM_BASE_H__

#ifdef __cplusplus
extern "C" {
#endif




/*


阅读C语言代码，回答下面的问题，不要修改代码。

本项目大量使用fprintf输出错误信息，这非常糟糕。因为不是所有的平台都有fprintf。请设计一种平台无关的方案，如何从解释器核心将错误信息取出来，不依赖标准库函数。进一步地讲，如何实现更好的可观测性设计。






## 一、现状盘点：错误信息现在是怎么"漏"出去的

项目实际上已经有一半做对了，另一半还依赖 `fprintf`：

**已经平台无关的部分（好的一面）**

- VM 运行时错误：`am_runtime_error()`（`src/am_runtime.c:3204`）把错误逐 wchar 压入 `rt->error_fifo`，并触发 `am_runtime_vtable_t::on_error` 回调。宿主在 `on_tick`/`on_error` 中把 FIFO 排空，自行决定写到 stderr、串口还是屏幕。
- 用户输出（display/newline）同理走 `output_fifo`。
- 内存、时间、文件读取已经通过 `am_allocator_host_vtable_t` 和 `am_runtime_vtable_t` 依赖注入（`include/am_runtime.h:75`），ESP32 平台用 `am_host_esp32.cpp` 整体替换宿主适配层。

**仍然依赖 stdio 的部分（问题所在）**，约 136 处 `fprintf/printf`，其中错误类约 107 处：

| 位置 | 问题 |
|---|---|
| `am_parser.c:2019-2108`、`am_macro.c:2031+` | 语法/宏错误：消息其实已存在 `ctx.error_msg[256]` 里，但返回 NULL 的同时**顺手 fprintf 到 stderr**，调用者拿不到 |
| `am_module.c`（17 处） | 模块 dump/load 校验失败，错误只有 stderr 这一个出口，函数只返回失败码 |
| `am_allocator.c`（44 处） | OOM、内存池统计报告，直接打 stderr |
| `am_gc.c` | GC 一致性错误 |
| `am_debug.h` / `am_debug.c` | 强制 `#include <stdio.h>`，API 里直接出现 `FILE *` 参数——这是最深的耦合 |
| `am_native_System.c`、`am_js2scm.c` | native 库和翻译器内部的 fprintf |

另外三个结构性缺陷：

1. **没有错误码**：错误只有自由文本，宿主无法编程地区分"OOM"和"语法错误"，无法做本地化，也无法按类别过滤。
2. **没有级别概念**：致命错误（GC 损坏）、可恢复错误（语法错误）、诊断报告（内存池统计）混在同一个 stderr 里。
3. **上下文丢失**：模块加载失败时只说"bad magic"，不带模块路径、偏移量；解析错误带了行列号但被 fprintf 消费掉了，没传给调用者。

## 二、设计方案：三层解耦

核心原则一句话：**核心只产生"结构化诊断事件"，永远不产生"输出"；输出是宿主的私事。**

### 第 1 层：诊断事件与 Sink 回调（替换 fprintf 的最小方案）

定义一个诊断事件结构和注入式 sink，与现有 `am_runtime_vtable_t` 的风格保持一致：

```c
typedef enum {
    AM_DIAG_DEBUG, AM_DIAG_INFO, AM_DIAG_WARNING, AM_DIAG_ERROR, AM_DIAG_FATAL
} am_diag_level_t;

typedef enum {
    AM_ERR_NONE = 0,
    // 按模块分段编号，宿主可 switch 处理
    AM_ERR_ALLOC_OOM = 0x0100, ...
    AM_ERR_PARSE_SYNTAX = 0x0200, ...
    AM_ERR_MODULE_BAD_MAGIC = 0x0300, ...
    AM_ERR_GC_CORRUPT = 0x0400, ...
} am_error_code_t;

typedef struct {
    am_diag_level_t level;
    am_error_code_t code;
    const char    *module;      // "parser" / "gc" / ...（字面量，零开销）
    const wchar_t *message;     // 已格式化的人类可读文本，可为 NULL
    // 结构化上下文（可选，视模块填充）
    int32_t pid;                // 进程号，无则 -1
    size_t  line, column;       // 源码位置，无则 0
} am_diag_event_t;

// 宿主注入的接收器：不返回、不分配、可重入
typedef void (*am_diag_sink_fn)(void *ctx, const am_diag_event_t *ev);
```

- Sink 挂在 `am_runtime_t`（运行时错误沿用并增强现有 `error_fifo` + `on_error` 机制）和**创建期对象**上（parser/module/allocator 的错误目前没有 runtime 可依附，所以 sink 需要作为参数传入 `am_parse`、`am_module_load` 等，或挂在一个全局可替换的 `am_diag_set_sink()` 上——全局 sink 实现最简单，且这些前端错误本来就是"编译期"行为，与运行时实例无关）。
- 桌面宿主的 sink 实现就是一行 `fwprintf(stderr, ...)`；ESP32 的 sink 写到 UART；浏览器/TS 版的对应物就是已有的 `ANIMAC_STDERR_CALLBACK`。**核心代码里从此不出现 `stdio.h`**。

### 第 2 层：解决"格式化"这个硬骨头

`fprintf(stderr, "... %d ...", x)` 的便利来自变参格式化，而 `vsnprintf` 恰恰也是不可移植的。两个务实选项，建议组合使用：

1. **静态消息 + 结构化字段**：绝大多数错误点的格式串是固定的（"bad magic"、"版本不符"），把变量部分放进 `am_diag_event_t` 的字段（code、line、pid）或让消息本身不含变量。这覆盖了约八成调用点，且天然适合本地化。
2. **自带微型格式化器**：核心内实现一个只支持 `%s %d %u %x %ls` 的 `am_fmt()`（约 100 行，不依赖 libc 的浮点/locale 代码），写入调用者提供的固定大小缓冲区。用于确实需要拼变量的场景。项目已有先例：`am_host.c` 就自实现了 UTF-8↔UTF-32 转换而不依赖 locale。

配套一个宏，把级别过滤和可裁剪性一次解决：

```c
#define AM_DIAG(rt, level, code, ...) \\
    do { if (level >= am_diag_min_level) am_diag_emit(rt, level, code, __VA_ARGS__); } while(0)
```

`AM_CONFIG_DIAG_LEVEL` 在编译期裁剪：release 版可把 DEBUG/INFO 直接编译为空语句，零成本；这与 amalgamation 把 `am_highlight`、`am_debug` 排除出核心的思路一脉相承。`am_debug.c` 的 AST/IL dump 应整体移出核心（同 `am_highlight` 的现状），或改造为写入 sink 而非 `FILE *`。

### 第 3 层：前端 API 返回结构化错误（修"拿不到"的问题）

parser/macro/module 的 fprintf 之所以存在，是因为 API 只返回 NULL，错误无处可去。改造为输出参数：

```c
am_ast_t *am_parse(const wchar_t *src, am_allocator_t *alloc, am_diag_t *err_out);
// err_out: 调用者提供的小结构体 { code, line, column, wchar_t msg[256] }，可为 NULL
```

`ctx.error_msg` 已经存在，只是被 fprintf 半路消费了——把它改为填入 `err_out`，fprintf 自然消失。这样**不注册 sink 也能拿到错误**（REPL 可以直接把结构化错误渲染成高亮提示），sink 只负责"没人接的错误往哪倒"。

## 三、进一步：更好的可观测性设计

只把 fprintf 换成回调，解决的是"可移植"；真正的可观测性还需要四样东西：

**1. 指标（Metrics）——在 `am_runtime_t` 里维护计数器，提供只读 getter**

数据其实已经散落在结构体里（`tick_counter`、`gc_count`、`process_poll_counter`），只需补齐并暴露：

```c
typedef struct {
    size_t tick_count, gc_cycle_count, gc_objects_reclaimed;
    size_t heap_bytes_used, heap_bytes_peak;     // 分配器已在统计（pool_print_stats 的数据源）
    size_t process_alive, process_total_created;
    size_t error_count_by_level[AM_DIAG_FATAL+1];
} am_runtime_stats_t;
void am_runtime_get_stats(const am_runtime_t *rt, am_runtime_stats_t *out);
```

现在 `am_allocator_pool_print_stats()` 把统计**格式化打印**到 stderr；应改为 `am_allocator_get_stats()` 返回结构体，打印留给宿主或 REPL 命令。这是"数据 vs 呈现分离"的典型修复。

**2. 追踪（Tracing）——编译期可选的探针回调**

```c
typedef struct {
    void (*on_gc_begin)(void *ctx, size_t heap_used);
    void (*on_gc_end)(void *ctx, size_t reclaimed, uint64_t duration_us);
    void (*on_process_spawn)(void *ctx, int32_t pid);
    void (*on_process_exit)(void *ctx, int32_t pid, am_error_code_t reason);
    void (*on_instruction)(void *ctx, int32_t pid, am_iaddr_t pc); // 仅调试构建
} am_trace_vtable_t;   // 全部可为 NULL；AM_CONFIG_ENABLE_TRACE=0 时编译为空
```

关键点：**全部回调允许为 NULL， hot path 上只有一次空指针判断**，禁用时零开销。宿主可以用它实现 ESP32 上的 GC 耗时统计、桌面端的性能火焰图。注意 `duration` 需要微秒级时钟——现有 vtable 只有 `now_ms`，可给 sink/trace 场景增加可选的 `now_us`（为 NULL 则退化为 0）。

**3. 事件分级路由，而非分级打印**

`am_diag_event_t` 里的 `level + code + module` 三要素让宿主可以做策略：FATAL 立即落盘/重启、ERROR 进环形缓冲、DEBUG 丢弃。核心内建议保留一个**固定大小的诊断环形缓冲区**（无锁、无分配，比如 16 条事件）：即使宿主还没来得及 drain FIFO 就发生崩溃，也能在事后（调试器、看门狗复位后）从已知地址读出最后几条错误——这是嵌入式可观测性的关键手法，FIFO 链表做不到这一点（它在 OOM 时自己也会分配失败）。

**4. 语言层可观测性（收尾项）**

目前项目没有 Scheme 级的 `raise/guard` 条件系统（调查确认过）。诊断设施就位后，可以低成本地：

- 把 `error_fifo` 的内容以 `(current-error-string)` 之类的 native 函数暴露给 Scheme 层；
- 让运行时错误除文本外携带 error code，未来实现 `guard` 时按 code 分派——C 层的 `am_error_code_t` 就是它的地基。

## 四、迁移路径（按风险从低到高）

1. **先做 sink 基础设施**（事件结构 + 全局 sink + `AM_DIAG` 宏），此时所有 fprintf 原样保留，行为不变；
2. **机械替换** parser/macro/module/allocator/gc 的 107 处 fprintf 为 `AM_DIAG`，桌面宿主的默认 sink 打印到 stderr——行为完全等价，可随时回滚；
3. **修 API**：`am_parse`/`am_module_load` 增加 `err_out`；`print_stats` 类函数拆分为 `get_stats` + 宿主侧格式化；
4. **裁剪 stdio**：把 `am_debug.*` 移出核心（或改造），核心头文件不再出现 `FILE *`；amalgamation 验证核心可在无 stdio 环境下编译通过（加一个 `AM_CONFIG_NO_STDIO` 的 CI 检查）；
5. **加 trace/metrics**：按需渐进，不阻塞前面任何一步。

总结：本项目其实已经具备了正确的架构基因（vtable 依赖注入、FIFO 解耦、amalgamation 划清核心边界），问题只是前端和底层组件没有用上这套机制。方案的本质不是发明新东西，而是**把 `error_fifo`/`vtable` 这套已被验证的模式推广到 parser、module、allocator、gc，并补上错误码和级别这两个缺失的维度**，观测性再以"数据与呈现分离"为原则逐层叠加。
















从头文件-模块依赖的角度看，诊断数据结构放在哪里合适？我感觉这是个“一捅到底”的东西，横切关注点。当前项目各头文件依赖层级的最基层是 @include/am_allocator.h 和 @include/am_object.h 。有必要在它们下面再增加所有核心模块都依赖的头文件吗？



`am_wchar_t` 确认是 `uint32_t`（`am_object.h:68`），这一点对答案很重要。下面给出完整分析。

## 结论：有必要，而且它就应该是新的最底层——一个零依赖的 `am_diag.h`

你的直觉是对的：诊断是横切关注点，调用方遍布 L0（allocator 的 OOM）、L4（parser 的语法错误）、L5（module 的校验失败）、L7（gc 的一致性错误）。这种"被所有层调用"的东西，**唯一能待的位置就是依赖图的最底部**。问题只剩两个：放哪个文件、怎么控制它不变烂。

### 一、为什么不能放在现有的任何一个头文件里

对照依赖图逐一排除：

| 候选位置 | 问题 |
|---|---|
| `am_runtime.h`（L7） | 方向颠倒。allocator、parser、gc 都在它下面，让 L0 的分配器去 include L7 的运行时，等于把分层倒过来 |
| `am_object.h`（L0） | 语义污染：object.h 是值系统（TPV、打包/解包），塞入诊断后它就不再纯粹。更关键的是 **allocator.h 当前不依赖 object.h**，若诊断并入 object.h，allocator 为了报错就要新增一条指向 object 的边，凭空耦合两个本来正交的 L0 模块 |
| `am_allocator.h`（L0） | 语义完全不符，且 parser/macro 等纯前端模块并不需要分配器接口，会被迫传递依赖 |
| 并入 `animac.h` 伞形头 | 伞形头是 L8，所有核心模块都不能依赖它（那是反方向） |

所以"在 L0 双基下面/旁边再加一个所有核心模块都依赖的头文件"不仅是必要的，而且是唯一不破坏 DAG 的放法。准确说它不是"在 L0 下面"，而是**与 L0 平级甚至更低的第三个零依赖头文件**——成为事实上的 L(-1)。

### 二、控制代价的关键：让它"小而稳"，而不是"大而全"

对"一捅到底"的顾虑，本质担心的是底层文件变成人人依赖、频繁变动的瓶颈。化解手段是**严格控制这个头文件的内容边界**——只放三样东西：

```c
// am_diag.h —— 只依赖 <stdint.h> <stddef.h>，不依赖任何项目头文件
// 1. 级别枚举（5 个值，几乎永远不会变）
// 2. 错误码枚举（集中注册表，见下）
// 3. 事件结构体 + sink 函数指针类型 + 唯一入口声明
typedef void (*am_diag_sink_fn)(void *ctx, const am_diag_event_t *ev);
void am_diag_emit(const am_diag_event_t *ev);   // 实现和全局默认 sink 在 am_diag.c
```

两个具体决策：

**1. 保持零依赖，连 `am_object.h` 都不要引。** 事件里的消息字段直接声明为 `const uint32_t *`——由于 `am_wchar_t` 就是 `uint32_t` 的 typedef，类型透明兼容，上层传 `am_wchar_t*` 无需任何转换。这样 `am_diag.h` 的依赖只有 `stdint.h/stddef.h`，是真正意义上的"捅到底"，不会把 object/allocator 拖进任何人的依赖闭包。这是本答案里最重要的一个细节：如果 diag.h 贪图类型一致性去 include object.h，它就立刻失去"最底层"资格，并且会让 allocator→diag→object 这条新边污染原本干净的 L0 双基结构。

**2. 错误码集中在 diag.h，不要分散到各模块头文件。** 集中枚举是单一注册表，天然避免编号碰撞，宿主 `switch(code)` 时只需 include 一个文件。代价是"新增错误码会导致全项目重编译"——但对 30 个头文件规模的项目这完全可接受，且错误码是**追加式**变化，不改已有声明，ABI 稳定。若将来模块多到受不了，再退化为"diag.h 定义模块 ID 段 + 各模块头文件定义段内编号"的折中方案也不迟。

### 三、sink 的"状态"不放底层——底层只有类型，实例各有归属

头文件只承载类型和入口，sink 指针本身按生命周期分层存放，避免全局可变状态被滥用：

- **全局默认 sink**：`am_diag.c` 内的一个静态变量 + `am_diag_set_sink()`，服务于没有 runtime 依附的"编译期"错误（parser、module load、js2scm）。这是唯一合法的全局态，因为前端错误本来就与运行时实例无关；
- **runtime 实例 sink**：挂在 `am_runtime_t` 上（`am_runtime.h` include `am_diag.h`，是一条向下的新边，合法），运行时错误优先走实例 sink，未设置时回落全局 sink——与现有 `error_fifo` + `on_error` 机制并存，相当于给它接上结构化入口；
- **allocator**：当前它是纯 vtable 接口，不感知诊断。如果希望 OOM 时发事件，让它 include `am_diag.h` 并走全局 sink 即可——这条 `allocator.h → diag.h` 的边是**可选的、后加的**，不引也完全成立。这一点说明新底层头不要求任何既有文件立刻改依赖。

### 四、对现有层级图的影响验证

新增文件后唯一的依赖变化是：若干上层头文件各自多一条指向 `am_diag.h` 的边，而 `am_diag.h` 出度为零——出度为零的节点**不可能参与任何环**，DAG 性质保持。`animac.h` 在基础设施段的最前面（`am_allocator.h` 之前）收录它即可。`amalgamate.sh` 只需把 `am_diag.c/h` 加进核心文件清单。

### 五、一个必须警惕的反模式

给项目开"最底层公共头"这个口子后，最大的风险是它退化成 `common.h` 式的杂物间——今天放诊断，明天放工具宏，后天放字符串 helper，最后变成每个文件都隐式依赖一切的泥潭。纪律只有一条：**`am_diag.h` 只收纳"诊断事件"这一件事的三件套（级别、错误码、事件+sink），实现、格式化器、环形缓冲全部放 `am_diag.c`**。判断标准很直接：如果一个符号不是"跨模块共享的诊断类型"，它就没有资格进入这个文件。

总结：你的判断成立，诊断就是典型的横切关注点，正确的做法不是回避"一捅到底"，而是**捅得足够薄**——一个零项目依赖、只含类型定义和唯一入口的 `am_diag.h`，占据 L0 之下（或者说与 L0 平级）的新基座位置，上层各模块按需引入，错误码集中注册，sink 状态按生命周期分层持有。


















除了可观测性、诊断这样的一捅到底的横切关注点，对于一个语言解释器、语言引擎这样的系统来说，还有哪些必须做的东西，有资格进入这个负一层的地狱头文件，成为所有东西的全局依赖？




这个问题值得先立入场资格的标准，再点名——否则"地狱头文件"真会变成地狱。

## 一、入场资格：四条硬标准

一个东西要有资格住进 -1 层，必须**同时**满足：

1. **普遍性**：依赖图里每一层都真的需要它，不是"大部分"而是"全部"；
2. **不变性（invariant）**：它表达的是**跨平台、跨宿主不变的事实**——类型、常量、纯函数、宏。这一条是分水岭：凡是平台间会变的东西（时间、I/O、线程、文件系统），再普遍也没资格进来，它们必须走 vtable 注入；
3. **零出度**：它自己不依赖任何项目头文件，最好只依赖 `stdint.h/stddef.h` 这一级；
4. **稳定性**：变化是追加式的（加枚举值、加宏），几乎不改已有声明。

反向测试一句话：**"它能不能被宿主注入替代？"能，就不该进 -1 层。** 这正是本项目已有的智慧——时间（`now_ms`）、睡眠、文件读取都没有进底层头，而是进了 `am_runtime_vtable_t`；内存分配进了 `am_allocator_host_vtable_t`。-1 层只收留"连注入都无从注入"的东西。

## 二、有资格者的完整清单

对一个语言引擎而言，够格的一共只有六类，本项目已经占了两席：

**1. 值表示与基本类型 —— 已占位：`am_object.h`**
解释器里唯一真正"一切都是它"的东西：tagged pointer、基本类型、`am_value_t` 的标签代数。它定义了"这个世界的物理单位"，理所当然在最底层。

**2. 内存分配接口 —— 已占位：`am_allocator.h`**
所有模块都要分配内存，但"怎么分配"是宿主可注入的策略，所以进来的只有**接口类型**（vtable 结构、`am_allocator_t` 的不透明声明），不含任何策略。这是 -1 层和注入机制配合的范本：类型沉底，行为上浮。

**3. 诊断/错误 —— 前两轮讨论的 `am_diag.h`**
级别、错误码、事件结构、sink 类型。已论证，不重复。

**4. 编译期配置头（`am_config.h`）—— 本项目目前缺失**
功能开关（`AM_CONFIG_ENABLE_TRACE`、`AM_CONFIG_DIAG_LEVEL`）、值表示的静态断言（`static_assert(sizeof(void*)==8)` 之类）、调试/发布模式宏。每个模块都可能要按配置裁剪代码，而配置本身只能是纯宏——它是 -1 层里**最底的那个**，连 diag 的级别过滤宏都要依赖它。

**5. 平台探测与可移植性原语（`am_platform.h`）—— 本项目目前缺失**
编译器/架构探测及衍生的纯宏：`AM_LIKELY/AM_UNLIKELY`、`AM_INLINE`、`AM_RESTRICT`、对齐说明符、字节序判定与 byteswap（模块序列化——`am_module.c` 的 dump/load——天然需要它）、`AM_EXPORT`（如果核心要编成动态库）。它们是全项目都会用到的"编译器方言翻译层"，自身永远是纯宏，零依赖零变化。注意边界：**只放编译期能确定的事**；运行期才能确定的事（比如实际页大小）不属于这里。

**6. 断言与不可达（`AM_ASSERT` / `AM_UNREACHABLE`）—— 可并入 diag**
每个模块都有内部不变式要检查（GC 的对象头检查、分配器的边界检查），断言是诊断的特例——"只在调试构建发声的诊断"。可以单列一个头，但我建议**并入 `am_diag.h`**：它们共享同一个 sink、同一套裁剪开关，拆开反而制造"断言走 stderr、诊断走 sink"的二轨制，重蹈现状覆辙。

就没了。解释器这种系统里，真正够格的常驻者就是这六类，其中第 1、2 类是"领域地基"，3、6 是"横切观测"，4、5 是"构建事实"。

## 三、反面清单：看起来该进、其实没资格的"伪横切"

这部分和正面清单同样重要，因为 -1 层的死因从来不是缺少住户，而是收留了不该收留的：

| 伪横切 | 为什么没资格 | 正确归宿 |
|---|---|---|
| 时间/随机数 | 平台可变，可注入 | vtable（本项目已做对） |
| 字符串/宽串操作 | 只有部分模块需要，且依赖 object/allocator | L1 的 `am_wstring.h`（现状正确） |
| 容器（list/map） | 同上 | L1（现状正确） |
| 数学辅助（min/max、对齐取整、2 的幂） | 诱惑最大的灰色地带 | 少量纯宏可进 platform.h；一旦开始积累，立刻收紧 |
| 日志格式化器 | 是实现不是类型 | `am_diag.c` 内部 |
| errno 式全局错误变量 | 全局可变状态 + 线程/多 runtime 不友好 | 错误码走返回值/`err_out` 参数 |
| setjmp/longjmp 异常基座 | 这是错误处理**策略**，不是普遍事实；且不可移植 | 维持现状的返回值约定 |
| "实用工具"（`am_util.h`） | 名字本身就是警报 | 禁止创建 |

判据还是那一句反向测试。特别点名数学辅助宏：`min/max/align_up` 这类东西每个项目都忍不住往底层塞，一两个无伤大雅，但它是"杂物间化"的头号入口——建议本项目明令：platform.h 只收**平台探测**，数学宏哪怕只被三个模块用，也放 L1 或就近复制。

## 四、-1 层内部的次序与落地形态

即使同为 -1 层，内部依然有依赖方向：

```
am_config.h      （纯构建开关，出度 0 中的 0）
   ↓
am_platform.h    （平台探测宏，可读 config）
   ↓
am_diag.h        （类型+入口，可用 platform 宏、受 config 裁剪）
   ↓
am_object.h  /  am_allocator.h   （现有 L0，按需引 diag）
```

对这个项目的落地建议很收敛：**只新增三个小头文件**——`am_config.h`（几十个宏）、`am_platform.h`（纯探测宏）、`am_diag.h`（三件套）。三个加起来应该不超过三百行，出度全部为零，而且每一个都能回答"为什么你不能被注入、不能被放在上一层"。`animac.h` 按上面的次序把它们收在基础设施段最前。

最后回到你"地狱头文件"的说法：它之所以是地狱，不是因为住进去的东西多，而是因为**住进去的东西没有退租机制**——每一行都会被全项目永久继承。所以守门规则应该比任何一层都严：新增一个符号进 -1 层，需要能同时过普遍性、不变性、零出度、稳定性四关；过不了，它属于某个具体的层，只是"碰巧用的人多了点"而已。


















为了保持简洁，我希望负一层只有一个头文件。那么这个头文件叫什么比较好？




一个头文件承担全部 -1 层职责的话，我推荐 **`am_base.h`**。理由和排除过程如下。

## 命名约束

先从项目现实出发划掉几个候选：

| 候选 | 排除原因 |
|---|---|
| `am_common.h` | 上一轮说过，这个名字本身就是杂物间的邀请函——"common"没有语义边界，任何符号都能自称 common |
| `am_core.h` | **与 amalgamation 产物 `animac_core.c/h` 撞车**。一个是手写的最底层头，一个是生成的合并文件，名字相近会永久制造混淆 |
| `am_types.h` | 过窄。这个文件要装的不只是类型，还有配置宏、平台探测宏、sink 入口声明——叫 types 名不副实 |
| `am_util.h` / `am_utils.h` | 上一轮点名的警报词，直接枪毙 |
| `am_platform.h` | 过窄，它只描述了六类住户中的一类 |
| `am_prelude.h` | 意境好（Rust 传统），但在 C 项目里不合惯例，团队看到要愣一下 |

## 为什么是 `am_base.h`

1. **语义准确**："base"表达的是"地基/基座"，恰好对应我们定义的 -1 层——不是"公共的"，而是"万物立于其上的"。这个名字自带方向感：它暗示依赖只能从上往下指向它，而不像 "common" 那样暗示"谁都可以往里放"。
2. **有成熟的先例语感**：C/C++ 生态里 `base` 一直承担这个角色（Chromium 的 `base/`、Abseil 的 `absl/base/`），看到名字的人第一反应是"这里放的是最底层的原语和契约"，而不是"工具箱"。
3. **与现有命名和谐**：`am_base.h` 与 `am_object.h`、`am_allocator.h` 并列时不突兀，且在 `animac.h` 的基础设施段排在最前时，阅读顺序自然——base → allocator → object。
4. **长度合适**：`am_foundation.h` 语义也对，但 14 个字符的头文件名在本项目清一色短名字的 include/ 目录里显得笨重。

## 配套建议

既然六类住户挤进一个文件，**内部要用清晰的分节注释维持秩序**，让"一个文件"不等于"一锅粥"：

```c
// am_base.h —— Animac 核心基座（Layer -1）
// 入场规则：只收跨平台不变、零出度、全层普遍需要的定义。
//          可变行为一律走 vtable 注入，禁止入内。
//
// §1 编译期配置开关   (AM_CONFIG_*)
// §2 平台探测与可移植宏 (AM_LIKELY / AM_INLINE / 字节序 ...)
// §3 断言与不可达     (AM_ASSERT / AM_UNREACHABLE)
// §4 诊断             (级别 / 错误码 / am_diag_event_t / sink / am_diag_emit)
```

次序就是上一轮说的内部依赖方向：config → platform → assert → diag。每个分节顶部一句话写清该节的入场标准，后来人想往里加东西时，先看到的就是门槛而不是空地。

一句话：**`am_base.h`**——名字即门槛，"base" 回答"它是什么"（地基），也就同时回答了"什么能进"（只有地基材料能进）。


*/




#ifdef __cplusplus
}
#endif

#endif
