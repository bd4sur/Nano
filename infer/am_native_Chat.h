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
//        (Chat.begin prompt)  → 建会话（组装 prompt 模板 + 分词），返回提示词元数
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
//      (define models (Chat.models))            ; ((名称 文件) ...)
//      (Chat.load 0)                            ; 按 (Chat.models) 的下标载入（阻塞，约 0.2~2s）
//      (Chat.begin "你好")                      ; 返回提示词元数
//      (while #t { ... (Chat.step) ... })        ; App 自己按拍驱动
//
//  错误码（(Chat.load)）：
//      0  成功
//     -1  模型文件不存在（未被挂载 / 未拷贝）
//     -2  引擎初始化失败（内存不足）
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Chat_lib;

// 模型候选表上限（与 ui_llm.c 的 preset 表同一取舍：白名单 + 文件存在性过滤）
#define AM_NATIVE_CHAT_MODEL_MAX (8)

// top-K 候选词元缓存深度（可视化柱状图；也是 (Chat.top_*) 列表长度上限）
#define AM_NATIVE_CHAT_TOP_K (6)

// ---------------------------------------------------------------------------
// 模型与引擎
// ---------------------------------------------------------------------------

// (Chat.models) : List —— 可用的本地模型，每项为 (显示名 相对文件名) 两元素列表；
//   仅列出白名单内且**文件确实存在**的条目（浏览器目标需先在「资源」面板预载
//   llm/*.bin，否则为空表）。空表时 Scheme 侧应给出「未挂载模型」提示。
int32_t am_native_Chat_models(am_runtime_t *rt, am_process_t *proc);

// (Chat.load index:Number) : Number —— 按 (Chat.models) 的**列表下标**载入模型
//   （阻塞调用：解析模型头 + 反量化 + 分配 KV-Cache，WASM 上约 0.2~2s）。
//   返回 0 成功 / -1 文件不存在 / -2 初始化失败。重复调用会先卸载旧模型。
int32_t am_native_Chat_load(am_runtime_t *rt, am_process_t *proc);

// (Chat.unload) : Boolean —— 卸载当前模型（幂等；有会话时先释放会话）。恒 #t
int32_t am_native_Chat_unload(am_runtime_t *rt, am_process_t *proc);

// (Chat.loaded) : Boolean —— 是否已载入模型
int32_t am_native_Chat_loaded(am_runtime_t *rt, am_process_t *proc);

// (Chat.config) : List | #null —— 当前模型超参数
//   (arch block_size vocab_size n_layer n_embd n_head n_kv_head n_hidden head_dim seq_len)
//   arch：0=Nano 2=Qwen2 3=Qwen3（与 infer.h 的 LLM_ARCH_* 一致）
int32_t am_native_Chat_config(am_runtime_t *rt, am_process_t *proc);

// (Chat.set_seed seed:Number) : Boolean —— 重设采样随机种子（#f=未载入模型）
int32_t am_native_Chat_set_seed(am_runtime_t *rt, am_process_t *proc);

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
