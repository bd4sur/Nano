#ifndef __AM_NATIVE_LLM_H__
#define __AM_NATIVE_LLM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"


////////////////////////////////////////////////////////////////////////////
//  Native Library : LLM
////////////////////////////////////////////////////////////////////////////

// 由 src/native_LLM.c 定义
extern const am_native_lib_entry_t am_native_LLM_lib;


// (LLM.init modelFileBase64:String) : void
// 从 base64 编码的模型文件加载 Nano 架构语言模型。
int32_t am_native_LLM_init(am_runtime_t *rt, am_process_t *proc);

// (LLM.get_config) : List
// 返回模型结构参数列表：
// '(block_size vocab_size n_layer n_embd n_head n_kv_head n_hidden is_shared_classifier head_dim)
int32_t am_native_LLM_get_config(am_runtime_t *rt, am_process_t *proc);

// (LLM.get_param) : List
// 返回模型权重参数的嵌套列表：
// '(rms_norm_attn rms_norm_ffn rms_norm_final token_embedding wq wk wv wo w1 w2 w3 freq_cis_real freq_cis_imag token_classifier)
int32_t am_native_LLM_get_param(am_runtime_t *rt, am_process_t *proc);

// (LLM.encode text:String) : List<Number>
// 使用 Nano 分词器将字符串编码为词元 ID 列表。
int32_t am_native_LLM_encode(am_runtime_t *rt, am_process_t *proc);

// (LLM.decode token_id:Number) : String
// 将单个词元 ID 解码为字符串。
int32_t am_native_LLM_decode(am_runtime_t *rt, am_process_t *proc);

// (LLM.matmul xout x w xout_offset w_offset n d) : void
// 计算矩阵乘：xout[xout_offset..xout_offset+d) = W[w_offset..] @ x[0..n)
// 其中 W 的形状为 (d, n)，按行优先存储。
int32_t am_native_LLM_matmul(am_runtime_t *rt, am_process_t *proc);


#ifdef __cplusplus
}
#endif

#endif
