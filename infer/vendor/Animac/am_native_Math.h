#ifndef __AM_NATIVE_MATH_H__
#define __AM_NATIVE_MATH_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"


////////////////////////////////////////////////////////////////////////////
//  Native Library : Math
////////////////////////////////////////////////////////////////////////////

// 由 src/native_Math.c 定义
extern const am_native_lib_entry_t am_native_Math_lib;


// (Math.PI) : Number
int32_t am_native_Math_PI(am_runtime_t *rt, am_process_t *proc);

// (Math.pow base:Number exponent:Number) : Number
int32_t am_native_Math_pow(am_runtime_t *rt, am_process_t *proc);

// (Math.sqrt x:Number) : Number
int32_t am_native_Math_sqrt(am_runtime_t *rt, am_process_t *proc);

// (Math.exp x:Number) : Number
int32_t am_native_Math_exp(am_runtime_t *rt, am_process_t *proc);

// (Math.log x:Number) : Number
int32_t am_native_Math_log(am_runtime_t *rt, am_process_t *proc);

// (Math.log10 x:Number) : Number
int32_t am_native_Math_log10(am_runtime_t *rt, am_process_t *proc);

// (Math.log2 x:Number) : Number
int32_t am_native_Math_log2(am_runtime_t *rt, am_process_t *proc);

// (Math.sin x:Number) : Number
int32_t am_native_Math_sin(am_runtime_t *rt, am_process_t *proc);

// (Math.cos x:Number) : Number
int32_t am_native_Math_cos(am_runtime_t *rt, am_process_t *proc);

// (Math.tan x:Number) : Number
int32_t am_native_Math_tan(am_runtime_t *rt, am_process_t *proc);

// (Math.atan x:Number) : Number
int32_t am_native_Math_atan(am_runtime_t *rt, am_process_t *proc);

// (Math.atan2 y:Number x:Number) : Number
int32_t am_native_Math_atan2(am_runtime_t *rt, am_process_t *proc);

// (Math.floor x:Number) : Number
int32_t am_native_Math_floor(am_runtime_t *rt, am_process_t *proc);

// (Math.bit_and a:UInt b:UInt) : UInt
int32_t am_native_Math_bit_and(am_runtime_t *rt, am_process_t *proc);

// (Math.bit_or a:UInt b:UInt) : UInt
int32_t am_native_Math_bit_or(am_runtime_t *rt, am_process_t *proc);

// (Math.bit_xor a:UInt b:UInt) : UInt
int32_t am_native_Math_bit_xor(am_runtime_t *rt, am_process_t *proc);

// (Math.bit_not a:UInt) : UInt
int32_t am_native_Math_bit_not(am_runtime_t *rt, am_process_t *proc);

// (Math.bit_shl a:UInt n:UInt) : UInt    左移，n >= 位宽时结果为 0
int32_t am_native_Math_bit_shl(am_runtime_t *rt, am_process_t *proc);

// (Math.bit_ashr a:UInt n:UInt) : UInt   算术右移（按位宽解释符号位）
int32_t am_native_Math_bit_ashr(am_runtime_t *rt, am_process_t *proc);

// (Math.bit_lshr a:UInt n:UInt) : UInt   逻辑右移，n >= 位宽时结果为 0
int32_t am_native_Math_bit_lshr(am_runtime_t *rt, am_process_t *proc);

// (Math.ceil x:Number) : Number
int32_t am_native_Math_ceil(am_runtime_t *rt, am_process_t *proc);

// (Math.round x:Number) : Number
int32_t am_native_Math_round(am_runtime_t *rt, am_process_t *proc);

// (Math.to_fixed x:Number n:Number) : Number
int32_t am_native_Math_to_fixed(am_runtime_t *rt, am_process_t *proc);

// (Math.abs x:Number) : Number
int32_t am_native_Math_abs(am_runtime_t *rt, am_process_t *proc);

// (Math.random) : Number
int32_t am_native_Math_random(am_runtime_t *rt, am_process_t *proc);



#ifdef __cplusplus
}
#endif

#endif
