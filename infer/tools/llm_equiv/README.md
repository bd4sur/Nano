# llm_equiv —— 「纯 Scheme LLM 库」与「上游本地库」的等价性验证

`app/lib/llm_pure.scm` 是上游本地库 `vendor/Animac/am_native_LLM.c`（Nano 架构 /
F32 权重）的**纯 Scheme 重实现**：base64 解码、模型头/词表/权重张量解析、分词、
矩阵乘法全部在灵机解释器里完成。本目录证明两者**完全功能等价**。

## 怎么证明

同一份测试脚本跑两遍，链接不同的实现，然后逐行 diff stdout：

| 变体 | 链接的实现 |
|---|---|
| `native` | 上游本地库 `am_native_LLM.c`（经 `native_shim.scm` 封装成同名接口） |
| `pure` | `app/lib/llm_pure.scm`（不注册任何 LLM 本地库） |

测试脚本**不是手写的副本**：`run.py` 从 `app/llm.scm` 里按顶层 define 名字抽出前向
传播与采样代码原样拼进脚本，因此比对的就是 App 真正在跑的那份代码。

比对内容（68 行）：模型配置 9 元组、14 个权重张量的长度/求和/首中末元素、
`token_classifier` 与 `token_embedding` 的「内容相同但把柄不同」语义、若干文本的
encode id 序列与 decode 往返、词表前 40 项的展开、逐位置前向的 argmax 与 logits
抽样、6 元素排序任务的输出、以及 24 步贪心解码的完整生成文本。

## 用法

```bash
cd infer/tools/llm_equiv
python3 run.py          # 构建（gcc，链接 vendor 解释器 + 本地库）→ 对跑 → 比对
```

输出 `PASS：本地库与纯 Scheme 库的 68 行输出逐字节相同` 即通过；
失败时打印首个差异行，中间产物在 `/tmp/llm_equiv/{native,pure}.{scm,out}`。

## 文件

| 文件 | 作用 |
|---|---|
| `nref.c` | 最小无头参考台：真实 vendor 解释器 + System/Math/String/Table/Fs(/LLM)，display 打 stdout |
| `native_shim.scm` | 把 `(native LLM)` 包成与 `llm_pure.scm` 同名的接口（含流式三分段的整串拼接） |
| `run.py` | 抽取 App 的前向代码 → 生成两个变体 → 构建 → 对跑 → diff |

## 注意

- 本工具在 x86-64 上跑，灵机浮点是双精度；wasm32 宿主上是**单精度**
  （`am_float_t = float`），数值会有末位差异，但**在同一宿主上两条实现仍然逐字节相同**
  ——App 内的自检因此用「长度 + 首/中/末元素 + 相对容差」而不是求和比对。
- 模型分块 base64（`app/lib/llm/<name>/mNN.b64`）由 `infer/tools/gen_llm_model_b64.py`
  生成；本工具用绝对路径读取仓库内的这份数据，不依赖业务资源目录。
