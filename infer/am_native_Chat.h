#ifndef __AM_NATIVE_CHAT_H__
#define __AM_NATIVE_CHAT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Chat —— 鹦鹉笼（端侧 LLM）推理引擎的灵机本地库封装
//
//  为什么需要本库：
//    鹦鹉笼（infer/infer.c + tensor.c + tokenizer.c）是本机的**端侧语言模型推理**
//    实现：模型文件解析、Q80 量化反量化、KV-Cache、自注意力/FFN 前向、采样、
//    prompt 模板、分词器全在 C 侧。它此前只服务于 C 写的 GUI 状态机（ui_llm.c），
//    灵机（Scheme）程序完全用不上——app/llm.scm 只好用**纯 Scheme 重写**一遍
//    前向推理（几千行、单 token 数秒）。本库把那套 C 引擎按「模型 → 分词器 →
//    推理内核 → 观测（可视化）」四层封装出来，Scheme 侧只写界面与交互。
//
//  不得改动既有实现：本库**只调用** infer.h/tokenizer.h 的公开函数，以及给
//    Nano_Context 的两个观测钩子字段赋值（infer.c 专为此设计，见 ENABLE_NANO_OBSERVATION）。
//
//  交互模型（关键设计——「一拍一步」，不阻塞 GUI）：
//    引擎的一次前向（一个词元）在浏览器 WASM 上要几十~上百毫秒。若像 C 的
//    generate_sync 那样同步跑到结束，界面会整段冻死（灵机 App 的宿主每帧只给
//    解释器一个时间片，Scheme 里的长循环=掉帧）。因此本库把自回归生成**拆成可
//    单步调用的状态机**，与 ui_llm.c 把 generate_sync 的 while 打开成状态机的做法
//    同构：
//
//        (Chat.begin prompt)  → 建会话（组装 prompt 模板 + 分词 + 思考模式处理），
//                              返回提示词元数
//        (Chat.step)          → 推进一步（预填充：读一个提示词元；解码：生成一个词元）
//        (Chat.end)           → 释放会话
//
//    Scheme 侧每拍调一次 (Chat.step)，期间照常画界面、显示流式文本、进度与 TPS。
//
//  统计（进度/TPS 都由本库在 C 侧算好，不依赖 Scheme 的数值精度）：
//    · prefill_ms：从 begin 到最后一个提示词元前向完成的耗时；
//    · decode_ms ：从预填充完成到当前时刻的耗时；
//    · tps       ：已生成词元数 / decode_ms。
//    32 位（wasm32）宿主的 am_uint_t 只有 27 位有效宽度、am_float_t 只有约 19 位
//    尾数，因此 TPS 在 C 侧用整数毫秒算好再返回，避免 Scheme 侧精度坑。
//
//  状态观测（**只提供数据，不做任何绘制**——图形在 Scheme 侧画，见
//  app/nanochat.scm 的可视化一节，布局与样式忠实沿用 app/llm.scm）：
//    · (Chat.obs)          观测钩子给出的**当前层/当前相位**（Nano_Observation 语义：
//                          layer=-1 词嵌入、0..n_layer-1 各层、n_layer 收尾）；
//    · (Chat.top_ids/top_texts/top_probs)
//        —— 当前前向输出 logits 的 top-K（文本 + 采样概率）。概率语义与引擎
//           一致：解码步取**采样分布本身**（复读惩罚 + 温度 + softmax 之后，
//           与 top-p 核采样看到的是同一个分布）；预填充步取温度 softmax；
//    · (Chat.pick_prob)    本次采样词元在同一分布下的概率（核采样可能抽到 top-K 之外）。
//
//  栈纪律（上游本地库契约 N1/N2/N6）：除 (Chat.begin) 的极少数失败路径外，
//  每个过程弹完参数后恰好压 1 个值并 step 一次；类型不符按软失败处理
//  （返回 #null / #f / 0），不抛运行时错误——推理失败不应杀死 App。
//
//  Scheme 侧用法：
//      (native Chat)
//      (Chat.load (quote ("llm/nano-56m-q80.bin" "" 512 1.05 1.0 0.5 0)))  ; 载入（阻塞）
//      (Chat.begin "你好")                      ; 返回提示词元数
//      (while #t { ... (Chat.step) ... })        ; App 自己按拍驱动
//
//  错误码（(Chat.load)）：
//      0  成功
//     -1  配置列表非法 / 模型或 LoRA 文件不存在（未被挂载 / 未拷贝）
//     -2  引擎初始化失败（内存不足）
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Chat_lib;

// top-K 候选词元缓存深度（可视化柱状图；也是 (Chat.top_*) 列表长度上限）
#define AM_NATIVE_CHAT_TOP_K (6)

// ---------------------------------------------------------------------------
// 模型与引擎
// ---------------------------------------------------------------------------

// (Chat.load cfg:List) : Number —— 按 App 给出的配置载入模型（**阻塞**：解析模型头 +
//   反量化 + 分配 KV-Cache；56MB 约 1~2s，600MB 量级要数十秒）。重复调用先卸载旧模型。
//
//   cfg 是 7~8 元素列表，顺序固定：
//     (模型文件 LoRA文件 上下文 复读惩罚 温度 top_p top_k [思考模式])
//       · 第 8 项（可选）= 该模型是否有思考模式（Qwen3/Qwen3.5 这类可开/关思考的模型
//         为 1；普通模型或省略为 0）。它是**模型属性**，与 (Chat.set_thinking) 的
//         用户开关配合决定 (Chat.begin) 的行为，详见 set_thinking 的说明。
//       · 模型文件 / LoRA文件：与 (native Fs) **同一套路径规则**（相对业务资源根
//         PLATFORM_ROOT_DIR，或以 '/' 开头的宿主绝对路径）；LoRA 传空串 "" = 不挂；
//       · 上下文：词元数。KV-Cache 按上下文线性增长（Qwen3-0.6B 约 224KB/词元），
//         不要照抄模型的 block_size，按可用内存与需要的生成长度取（如 2048）；
//       · top_k = 0 表示不做 top-k 截断（只由 top-p 核采样决定）。
//
//   **模型预设表不在本库里**：它是 App 的数据（app/nanochat.scm 的 model_defs），
//   逐项对齐 C 版 ui_llm.c 的 preset_model_configs。本库只按参数载入，于是加模型 /
//   调采样参数是纯 Scheme 改动。
//
//   返回 0 成功 / -1 配置非法或文件缺失 / -2 初始化失败。
int32_t am_native_Chat_load(am_runtime_t *rt, am_process_t *proc);

// (Chat.unload) : Boolean —— 卸载当前模型（幂等；有会话时先释放会话）。恒 #t
int32_t am_native_Chat_unload(am_runtime_t *rt, am_process_t *proc);

// (Chat.loaded) : Boolean —— 是否已载入模型
int32_t am_native_Chat_loaded(am_runtime_t *rt, am_process_t *proc);

// (Chat.config) : List | #null —— 当前模型超参数
//   (arch block_size vocab_size n_layer n_embd n_head n_kv_head n_hidden head_dim seq_len)
//   arch：0=Nano 2=Qwen2 3=Qwen3 4=Qwen3.5（与 infer.h 的 LLM_ARCH_* 一致）
int32_t am_native_Chat_config(am_runtime_t *rt, am_process_t *proc);

// (Chat.set_seed seed:Number) : Boolean —— 重设采样随机种子（#f=未载入模型）
int32_t am_native_Chat_set_seed(am_runtime_t *rt, am_process_t *proc);

// (Chat.set_thinking flag:Number) : Boolean —— 开关思考模式（#f=未载入模型）
//   语义与 C 版 ui_llm.c 的 global_state->is_thinking_enabled 一致，**下一次
//   (Chat.begin) 生效**（切换逻辑在 C 侧，逐条对齐 ui_llm.c 的 STATE_LLM_ON_INFER 初始化）：
//     · 模型有思考模式且开关为 0：用户文本末尾追加 " /no_think"、采样参数切到
//       无思考一套（温度 0.7 / top-p 0.8）、会话模板插入**空思考块**（enable_thinking=0）
//       → 模型直接作答，不吐思考内容；
//     · 模型有思考模式且开关为 1：采样参数切到思考一套（温度 0.6 / top-p 0.95）；
//     · 模型没有思考模式（含 Nano）：开关被忽略，模板恒按 enable_thinking=1 建会话
//       （避免次元编码器输出多余的空思考占位词元）。
//   载入模型时开关复位为 1（与 C 版 is_thinking_enabled 初值一致）。
int32_t am_native_Chat_set_thinking(am_runtime_t *rt, am_process_t *proc);

// (Chat.thinking) : Number —— 思考模式状态：-1=当前模型没有思考模式（或未载入），
//   0=关闭，1=开启
int32_t am_native_Chat_thinking(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 分词器
// ---------------------------------------------------------------------------

// (Chat.encode text:String) : List —— 原始分词（**不含** prompt 模板），返回词元编号列表
int32_t am_native_Chat_encode(am_runtime_t *rt, am_process_t *proc);

// (Chat.token_text id:Number) : String —— 单个词元的文本（含特殊词元如 <|eos|>）
int32_t am_native_Chat_token_text(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 推理内核（一拍一步）
// ---------------------------------------------------------------------------

// (Chat.begin prompt:String) : Number —— 建立生成会话（Nano 架构自动套
//   「<|instruct_mark|>…<|response_mark|>」模板；Qwen 走 chat 模板）。
//   返回提示词元数（>0）；0=未载入模型或分词结果为空；-1=会话创建失败。
int32_t am_native_Chat_begin(am_runtime_t *rt, am_process_t *proc);

// (Chat.step) : Number —— 推进一步（一次前向）。返回值与引擎状态码一致：
//   LLM_RUNNING_IN_PREFILLING(11) / LLM_RUNNING_IN_DECODING(12) 继续；
//   LLM_STOPPED_*(≤-10) 结束（-10 正常结束/遇 EOS、-11/-12 中止、-20 错误）；
//   0 = 没有会话。
int32_t am_native_Chat_step(am_runtime_t *rt, am_process_t *proc);

// (Chat.end) : Boolean —— 释放会话（幂等）。恒 #t
int32_t am_native_Chat_end(am_runtime_t *rt, am_process_t *proc);

// (Chat.output) : String —— 已生成的文本（预填充阶段为空串；末尾的特殊结束词元
//   如 <|eos|> 已被剔除，供直接上屏）
int32_t am_native_Chat_output(am_runtime_t *rt, am_process_t *proc);

// (Chat.info) : List | #null —— 会话状态
//   (pos prompt_tokens gen_tokens tps prefill_ms decode_ms state pick)
//     pos           已推进的词元位置（含提示词元）
//     prompt_tokens (Chat.begin) 返回的提示词元数
//     gen_tokens    已生成词元数
//     tps           生成速率（词元/秒，整数毫秒口径）
//     prefill_ms    预填充耗时（毫秒）
//     decode_ms     解码耗时（毫秒）
//     state         最近一次 (Chat.step) 的返回值（0=尚未推进）
//     pick          最近采样的词元编号（预填充阶段为 -1）
int32_t am_native_Chat_info(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 可视化
// ---------------------------------------------------------------------------

// (Chat.obs) : List —— **最近一次**观测点 (layer phase)
//   layer：-1=词嵌入，0..n_layer-1=第 i 层，n_layer=收尾（final norm/classify/sample）
//   phase：1..12，见 infer.h 的 NANO_LLM_PHASE_*（Scheme 侧自带相位名表）
int32_t am_native_Chat_obs(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 一步前向内部的相位轨迹（供 Scheme 侧"逐相位回放"可视化）
//
//   引擎的一次前向是原子的（不能中途交还控制权），但 infer.c 会在**每个子步骤入口**
//   回调观测钩子（层 × 相位，顺序即执行顺序）。本库把这条序列按步记录下来：
//     (Chat.trace_len)      —— 最近一次 (Chat.step) 记录的相位个数
//     (Chat.trace_at i)     —— 第 i 项的 (layer phase)；越界返回 #null
//   Scheme 侧据此把"这一步算了哪些部件"逐帧回放（app/nanochat.scm 的可视化一节），
//   观感与 app/llm.scm 的「一相位一帧」一致，而本库**不做任何绘制**。
// ---------------------------------------------------------------------------

// (Chat.trace_len) : Number —— 最近一次 (Chat.step) 的相位轨迹长度
int32_t am_native_Chat_trace_len(am_runtime_t *rt, am_process_t *proc);

// (Chat.trace_at i:Number) : (layer phase) | #null
int32_t am_native_Chat_trace_at(am_runtime_t *rt, am_process_t *proc);

// (Chat.top_ids) / (Chat.top_texts) / (Chat.top_probs) : List
//   当前前向输出分布的 top-K（默认 K=6，按概率降序）：词元编号 / 文本 / 采样概率。
int32_t am_native_Chat_top_ids(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Chat_top_texts(am_runtime_t *rt, am_process_t *proc);
int32_t am_native_Chat_top_probs(am_runtime_t *rt, am_process_t *proc);

// (Chat.pick_prob) : Number —— 本次采样到的词元在本步输出分布下的概率（预填充阶段 0）。
//   用途：核采样从截断后的全词表抽取，被选中的词元**可能不在 top-K 里**，可视化需要
//   单独把它画出来（与 (Chat.top_ids)/(Chat.top_probs) 同一分布，可直接比较）。
int32_t am_native_Chat_pick_prob(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 宿主服务（供 ui_scheme_app.c 在退出 App / 切换实例时清理全局引擎，非 Scheme 可见）
// ---------------------------------------------------------------------------

// 释放会话与模型（幂等）。宿主在销毁解释器实例前调用：引擎是**虚拟机全局单例**
// （与 C 版鹦鹉笼一致），不随 Scheme 进程销毁而释放。
void am_native_chat_release(void);

#ifdef __cplusplus
}
#endif

#endif
