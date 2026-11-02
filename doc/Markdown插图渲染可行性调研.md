# 报告：Markdown 阅读器渲染插图（图片）的可行性与方案

> 日期：2026-11-02　　状态：**一期已实施**（方案 A + 方案 D 兜底；见文末「实施记录」）
> 范围：`infer/app/markdown.scm`（灵机 Scheme 应用）+ 宿主侧薄封装（`am_native_Gfx.c`）
> 结论先行：**可行，而且比预期便宜**——宿主的图形层里已经有"读文件 → 解码 → 缩放 → 带缓存绘制"
> 的完整实现（`gfx_draw_image`），Scheme 侧不需要实现任何解码器，也不需要持有像素。

## 1. 需求与现状

Markdown 的插图语法是 `![alt](path "title")`。要把它做进当前阅读器，需要三段能力：

| 环节 | 现状 | 缺口 |
|---|---|---|
| 解析 | `md_parse.scm` 有块级/行内两张规则表 | 没有 `![...](...)` 规则；图片既可能独占一段（块级）也可能在段落里（行内） |
| 排版 | `md_layout.scm` 产出"行 = 高度 + 背景 + 段列表"，段自带绝对 x/宽，**行高可变** | 需要在**排版期**拿到图片尺寸才能算行高与缩放宽高 |
| 渲染 | `Gfx` 本地库只有 `fill_rect/draw_line/draw_text/...` | 需要暴露一个"画图片"的原语（宿主其实早就有了，见 §2） |

现状约束（都来自本仓库代码，不是假设）：

- **解释器堆是固定的池**（`ui_scheme_app.c` 的 `MB_POOL_SIZE`：桌面/WASM 24MB、ESP32 2MB），
  一篇 30KB 文档的显示行就占 ~1KB/行；把像素搬进 Scheme 字符串（`Fs.read` 上限 128KB、
  base64 再 ×1.33）会立刻吃光池子——320×240 RGB888 一张就是 230KB。→ **Scheme 侧不能持有像素**。
- 绘制是"语句型"即时落屏，没有图层；页眉/页脚靠"后画覆盖"实现裁剪（见 `draw_content` 的注释）。
- 段坐标在排版期定死，所以图片位置/尺寸必须在排版阶段确定。

## 2. 宿主已有的图片能力（这是本报告的核心发现）

| 设施 | 位置 | 说明 |
|---|---|---|
| 解码器 | `infer/vendor/stb_image.h`（单头文件，已入库） | `STB_IMAGE_IMPLEMENTATION` 在 `graphics.c` 里被展开；支持 PNG / JPG / BMP / GIF / TGA / PSD / HDR / PIC / PNM |
| 缩放器 | `infer/vendor/stb_image_resize2.h` | `stbir_resize_uint8_linear`，解码后一次缩放（RGB888） |
| **`gfx_draw_image(gfx, path, x, y, w, h, force)`** | `graphics.c` | **完整的一条链路**：读文件 → `stbi_load` → resize 到 (w,h) → **LRU 缓存**（键 = 路径+目标尺寸）→ 画到帧缓冲 |
| `gfx_draw_image_buffer(gfx, buf, len, x, y, w, h)` | `graphics.c` | 内存数据版本（资源已在内存时用） |
| `gfx_decode_image_rgba(buf, len, &w, &h)` | `graphics.c` | 解码成 RGBA（不缩放、不绘制），图标缓存 `ui_icon_cache_*` 用的就是它 |
| 尺寸查询 | stb_image 的 `stbi_info(path, &w, &h, &ch)` | **只读文件头**，不解码像素——正好给排版用 |
| ESP32 适配 | `graphics.c` 的 `#if defined(ESP32)` 分支 | 解码放到 **32KB 栈**的 FreeRTOS 任务里跑（<|place_holder_mm_span_0442|> stb_image 会吃栈），完成后回填 |

两个需要留意的细节（都已核实）：

1. `GFX_IMAGE_CACHE_SIZE` 默认 **2**（`graphics.h`），即"只缓存最近两张"。文档里图片多时会来回抖动，
   需要注意（见 §5 的对策）。
2. `gfx_draw_image` 目前**没有任何调用者**（壁纸/图标走的是另一条 `ui_icon_cache` 路径）——
   属于"已经写好但还没接出来的设施"，正好给这次用。

## 3. 候选方案

### 方案 A（推荐）：`Gfx` 暴露两个薄封装，复用 `gfx_draw_image`

```scheme
(Gfx.image_size path)                 ; → (宽 高) 或 #null   ；只读文件头（stbi_info）
(Gfx.draw_image path x y maxw maxh)   ; → 实际占用的 (宽 高) 或 #null ；内部 gfx_draw_image
```

- C 侧工作量很小：两个 native 函数 ≈ **80~120 行**（参数弹栈 + 路径解析 + 调 `gfx_draw_image`/`stbi_info`
  + 按本项目"单值栈纪律"压 1 值），缓存、缩放、格式识别全部复用现成实现。
- Scheme 侧只多三处：`md_parse` 出一条图片规则；`md_layout` 在排版期用尺寸算行高；
  `markdown.scm` 的 `draw_seg` 加一个 `ST_IMAGE` 分支。
- 优点：像素不进 Scheme 堆；跨平台一致（stb_image 是纯 C，已在 ESP32 分支里跑通）；
  帧耗时可控（缓存命中时只是 RGB888→帧缓冲的搬运）。
- 缺点：`md_layout` 需要"图片尺寸"这个外部信息 → 破坏"排版纯逻辑、可与宿主无关"的现状。
  **对策**：把度量做成**注入的回调**（`layout` 多接一个 `measure_image` 参数），
  单测传桩函数、真机传 `Gfx.image_size` 的包装。解析阶段完全不碰宿主。

### 方案 B：构建期转成"原始 RGB565 资源"，运行时只 blit

构建期（Python）把图片转成 `<name>.rgb565 + <name>.meta`，运行时直接 memcpy/抽样。

- 优点：端上零解码，ESP32 友好，帧耗时最低；尺寸在 meta 里，排版不用问宿主。
- 缺点：体积暴涨（320×240 RGB565 = 150KB/张，20 张 3MB，WASM 的 p2 挂载变慢）；
  换图要重跑构建；alpha 需要额外 mask 或丢弃。
- 适用：只有"某些目标没有 stb_image"时才需要考虑——**当前不是这种情况**（stb_image 已全平台编译）。

### 方案 C：纯 Scheme 解码（BMP / stored-PNG）

- **明确不推荐**：① 像素必须放 Scheme 堆（池子只有 24MB/2MB）；② 逐像素 `Gfx.set_pixel` 调用，
  320×240 = 76800 次 native 调用，按本项目 native 调用开销估算单张 **> 1 秒**；
  ③ 真 PNG 需要 zlib inflate，在 Scheme 里既慢又容易撞上解释器的性能墙。

### 方案 D：字符画/色块占位（兜底）

用 `▀`/`█` 等字形或纯色块近似图片（每格取平均色）。

- 优点：零宿主改动，纯 Scheme；内存占用等于文本量。
- 缺点：分辨率极低（每格 12×12），只适合"一眼看出有张图"。
- **用途**：作为**兜底**——`Gfx.draw_image` 不可用 / 文件缺失 / 格式不支持时，
  画一个虚线框 + alt 文本（必要时再叠一层色块），保证文档结构与阅读体验不崩。

## 4. 推荐方案（A，配 D 兜底）的落地方案

### 4.1 宿主侧（`am_native_Gfx.c` + `am_native_Gfx.h`）

```c
// (Gfx.image_size path:String) : List | #null —— (w h)，只读文件头，不解码
int32_t am_native_Gfx_image_size(am_runtime_t *rt, am_process_t *proc);
// (Gfx.draw_image path:String x:Number y:Number w:Number h:Number) : List | #null
//   —— 等比缩放到 (w,h) 框内并绘制，返回实际占用的 (w h)
int32_t am_native_Gfx_draw_image(am_runtime_t *rt, am_process_t *proc);
```

实现要点：

1. 路径解析沿用 `am_native_Fs.c` 的规则（相对业务资源根、拒绝 `..`），统一成一个
   `native_resolve_path()` 小工具（现在 `Fs` 本地库里是 static 的，可以抽出来共用）；
2. `image_size` 用 `stbi_info(path, &w, &h, &ch)`——只读头，**不分配像素**；
3. `draw_image` 调 `gfx_draw_image(g, path, x, y, w, h, 0)`（`force=0` 走缓存），
   失败（文件缺失/格式不支持）压 `#null`（**软失败**，与 `Fs.read` 的约定一致，不报错）；
4. 值型函数压 1 值 + `am_process_step`（契约 N1）；越界裁剪交给 `gfx_draw_image`/帧缓冲。

### 4.2 解析（`md_parse.scm`）

- 行内规则表加一条触发字符 `!` 的规则：`![alt](path)` → `(image path alt)` 行内节点
  （注册在 `[`→link 之前，避免 `[` 先吃掉）；
- 块级：整行只有一个图片节点时，`classify` 判成 `'image` 块（独占段 + 居中），
  否则留在段落里做行内图片；
- 解析阶段**不读文件**，只产出路径 + alt 文本。

### 4.3 排版（`md_layout.scm`）

- `layout(blocks, width, measure)`：`measure` 是 `(measure path) → (w h) | #f` 的回调；
  App 传 `(lambda (p) (Gfx.image_size p))` 的包装，单测传桩（返回固定尺寸）。
- `lay_image`：`maxw = 正文宽`、`maxh = 160`（约 2/3 屏），等比缩放到框内（整数取整），
  产出一行 `(高 = ret_h + 上下间距, 背景 NONE, 段 = [(ST_IMAGE path x w h alt)])`。
- 行内图片：按缩放后尺寸作为"原子"参与折行（与行内代码同理），行高取该行最大值——
  现有结构天然支持（表格多行单元格就在用可变行高）。
- 尺寸取不到：产出 `ST_IMAGE_MISSING` 段（绘制端画虚线框 + alt）。

### 4.4 渲染（`markdown.scm`）

```scheme
(if (== st MDL.ST_IMAGE)
    { (set! dr (Gfx.draw_image (get_item sg 0) x y w h))
      (if (null? dr) (draw_missing_box x y w h (get_item sg 4)) #f) }
    #f)
```

- 位置/尺寸都来自段；不需要裁剪（页眉页脚后画覆盖，与正文一致）；
- 失败兜底：虚线框 + alt 文本（方案 D 的极简版）。

### 4.5 资源与路径

- 文档里的相对路径**以文档所在目录为基准**（Markdown 习惯）：App 用 `Fs` 的路径规则
  拼成资源根相对路径后再传给 `Gfx.draw_image`（宿主因此不需要知道 base）；
- WASM 侧：`manifest.json` 已按目录分级（`image/` 是 p2），构建脚本无需改动；
- 单篇文档的图片总量建议加个软上限（例如 32 张），超过就只画前 N 张 + 提示，
  防止一次解析/排版把池子吃满。

## 5. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| `GFX_IMAGE_CACHE_SIZE = 2` | 图片多的文档反复解码，滚动卡顿 | 提到 8~16（或按平台配）；再配合"每帧最多解码 1 张，其余画占位框"的渐进策略 |
| 大图解码峰值内存 | ESP32 PSRAM 紧张 | `draw_image` 内部限制目标边长（≤ 320×240），`stbir` 直接缩到目标尺寸，不保留原图 |
| 首帧延迟 | 打开含多图的文档会卡一下 | 排版期只 `stbi_info`；绘制期按"每帧一张"渐进（未解码的先画框） |
| 纯逻辑单测被破坏 | `md_layout` 要外部尺寸 | 度量回调注入（§4.3），单测传桩；`md_parse` 完全不碰宿主 |
| alpha 通道 | 现有绘制是置色语义，无混合 | 一期忽略 alpha（不透明处理）；需要透明时可加 `draw_image_blend`（用 `gfx_blend_pixel` 的既有能力） |
| 路径穿越/越权 | 读到资源根之外 | 沿用 `Fs` 的解析规则（拒绝 `..`、相对资源根） |
| ESP32 栈溢出 | stb_image 递归/大缓冲 | 已有 32KB 栈任务分支（`graphics.c`），沿用即可 |

## 6. 工作量估计

| 工作项 | 估计 |
|---|---|
| `Gfx.image_size` + `Gfx.draw_image`（含路径解析抽取、软失败） | **0.5 天** |
| `md_parse` 图片规则（行内 + 块级）+ 单测 | 0.5 天 |
| `md_layout` 度量注入 + `lay_image` + 行内图片 + 单测 | 1 天 |
| `markdown.scm` 绘制分支 + 缺失兜底 + 端到端像素断言 | 0.5 天 |
| 缓存调优（`GFX_IMAGE_CACHE_SIZE`）+ 三平台验证（桌面/WASM/掌机） | 0.5~1 天 |
| **合计** | **约 3~3.5 人日** |

## 7. 结论与建议分期

- **可行且成本可控**：宿主的 `gfx_draw_image` 已经提供了"读文件→解码→缩放→LRU 缓存→落屏"的全链路，
  缺的只是两个 native 包装；Scheme 侧只在排版期要尺寸、绘制期发一次调用。
- **与现有架构兼容**：显示行"高度可变"、段"自带绝对坐标"的设计天然容纳图片块与行内图片；
  唯一的架构代价是排版需要一个外部度量，用回调注入即可，且 `md_parse` 仍可纯逻辑单测。
- **分期建议**：
  1. 一期：`Gfx.image_size` + `Gfx.draw_image`、**块级图片**、最大尺寸限制、缺失兜底；
  2. 二期：行内图片、缓存扩容与"每帧一张"渐进解码、alpha 混合；
  3. 兜底：任何平台拿不到图片时用虚线框 + alt（方案 D 的极简形式），保证文档结构不崩。

---

# 附：实施记录（2026-11-02，一期）

按本报告 §4 落地了「方案 A（宿主解码）+ 方案 D（虚线框兜底）」，一期范围是**块级图片**。

## 实际实现与当初设计的差异（都以"更简单"为准）

| 设计稿 | 实施结果 | 原因 |
|---|---|---|
| `(Gfx.draw_image path x y maxw maxh)` 内部等比缩放 | **`(Gfx.draw_image path x y w h)` 按给定尺寸绘制**；等比缩放放在排版端 | 排版端本来就要算行高，顺带算出缩放宽高，C 侧少一份逻辑；绘制端只做"画"这一件事 |
| `(Gfx.image_size path)` 自己解析文件头 | 复用 stb_image 的 `stbi_info`（新增 `gfx_image_info()`，`graphics.c`） | stb 已经编进来了，不必手写 PNG/JPEG 头解析 |
| 需要新的缓存层 | 复用现有的 `gfx_draw_image`（**LRU 缓存 + 缩放 + 落屏**，本来就有、此前无人调用） | 见 §2；只把返回值从 `void` 改成 `int32_t`（成功/失败），并新增 `gfx_image_info` |
| 行内图片一期也做 | 一期**只做块级**；行内 `![](...)` 退化为暗色占位文本「[图片] alt」 | 行内图片会让行高与图片高度耦合（一行里有文字又有图），留到二期 |
| 图片数量无上限 | 单篇最多渲染 `IMG_MAX_COUNT=24` 张，超出退化为占位框 | 防止一次排版把解释器池吃满（一屏都放不下的文档没必要解码） |

## 改动清单

**C 侧**
- `graphics.c/.h`：新增 `gfx_image_info(path, &w, &h)`（`stbi_info`，只读头）；`gfx_draw_image` 返回值 `void → int32_t`（0 成功/−1 失败），失败点改为 `return -1`。
- `am_native_Gfx.c/.h`：新增 `(Gfx.image_size path)`（→ `(w h)` | `#null`）与 `(Gfx.draw_image path x y w h)`（→ `#t`/`#f`）；路径解析复用 Fs 的规则。
- `am_native_Fs.c/.h`：把内部的 `fs_resolve` 提升为公共的 `am_native_fs_resolve()`，供 Gfx 复用同一套路径规则（相对业务资源根、拒绝 `..`）。

**Scheme 侧**
- `md_parse.scm`：行内规则 `![`→ `(image path alt)`（注册在 `[` 之前）；整段只有一个图片节点时提升为块 `(image path alt)`。
- `md_layout.scm`：`layout` 新增第三个参数（图片度量回调）；新增 `ST_IMAGE` / `ST_IMAGE_MISSING` / `ST_IMAGE_TEXT` 三个样式、`fit_image`（等比缩放进 `(正文宽, IMG_MAX_H=160)`）、`lay_image`（居中、行高=图片高、缺图走占位框、张数上限）、`image_count()`；行内图片在 `atomize` 里降级为占位文本。
- `markdown.scm`：`doc_image_size`（`Gfx.image_size` 包装，`#null` → `#f`）、`image_path`（**相对文档所在目录**解析）、`draw_image_missing` + `draw_dashed_rect`（方案 D）；`draw_seg` 增加 `ST_IMAGE`/`ST_IMAGE_MISSING` 分支，并跳过图片段的通用文字绘制（否则 alt 会叠在图上）；`MDSTAT` 增加 `images=N` 诊断字段。
- `md_demo.scm`：加一节「插图」说明语法与兜底行为（不引用具体文件，保持各平台一致）。

## 踩到的坑（值得记住）

`(Gfx.image_size ...)` 失败返回 `#null`，而**在本解释器里 `#null` 是真值**（只有 `#f` 为假），
排版端如果只写 `(if dims ...)` 就会把 `#null` 当成尺寸去 `get_item` → 直接
`[Runtime] 指令执行异常`。两处同时修：App 侧把 `#null` 归一成 `#f`，
排版端再用 `(list? dims)` 做类型判定（双保险）。

## 验证

- 宿主端到端（真文件系统 + 真图形层）：真 PNG（40×30 品红）画出来（品红像素 720）；
  缺图路径画出虚线框（`92,104,128` 像素）；`MDSTAT` 里 `images=1`。
- 浏览器端到端（真 WASM + MEMFS）：base64 → `FS.writeFile` 写入 `doc/img.png` 后，
  品红像素 1200、虚线框像素 338，无 pageerror。
- 单测：解析 124/124（新增 13 条图片用例）、排版 89/89（新增 19 条：缩放/居中/行高/
  占位框/张数上限/行内降级）。
