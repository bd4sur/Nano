#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#include "am_native_Math.h"


#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif


// 从操作数栈中弹出一个数值，统一转换为 float。
// 成功返回 true，失败返回 false。
static bool native_pop_number(am_process_t *proc, am_float_t *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;

    if (am_value_is_float(v)) {
        *out = am_value_to_float(v);
        return true;
    }
    if (am_value_is_int(v)) {
        *out = (am_float_t)am_value_to_int(v);
        return true;
    }
    if (am_value_is_uint(v)) {
        *out = (am_float_t)am_value_to_uint(v);
        return true;
    }
    return false;
}

// 将 float 结果压回操作数栈；若结果为 NaN，则压入 null。
static int32_t native_push_result(am_process_t *proc, am_float_t result) {
    if (isnan(result)) {
        if (am_process_push_operand(proc, AM_VALUE_NULL) != 0) return -1;
    }
    else {
        if (am_process_push_operand(proc, am_make_value_of_float(result)) != 0) return -1;
    }
    am_process_step(proc);
    return 0;
}

// 从操作数栈中弹出一个无符号整数。
// 成功返回 true，失败（非 uint 值）返回 false。
static bool native_pop_uint(am_process_t *proc, am_uint_t *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_uint(v)) return false;
    *out = am_value_to_uint(v);
    return true;
}

// 将 uint 结果压回操作数栈。
static int32_t native_push_uint_result(am_process_t *proc, am_uint_t result) {
    if (am_process_push_operand(proc, am_make_value_of_uint(result)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


int32_t am_native_Math_PI(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return native_push_result(proc, (am_float_t)M_PI);
}


int32_t am_native_Math_pow(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t exponent, base;
    if (!native_pop_number(proc, &exponent)) return -1;
    if (!native_pop_number(proc, &base)) return -1;
    return native_push_result(proc, pow(base, exponent));
}


int32_t am_native_Math_sqrt(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, sqrt(x));
}


int32_t am_native_Math_exp(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, exp(x));
}


int32_t am_native_Math_log(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, log(x));
}


int32_t am_native_Math_log10(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, log10(x));
}


int32_t am_native_Math_log2(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, log2(x));
}


int32_t am_native_Math_sin(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, sin(x));
}


int32_t am_native_Math_cos(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, cos(x));
}


int32_t am_native_Math_tan(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, tan(x));
}


int32_t am_native_Math_atan(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, atan(x));
}


int32_t am_native_Math_atan2(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x, y;
    if (!native_pop_number(proc, &x)) return -1;
    if (!native_pop_number(proc, &y)) return -1;
    return native_push_result(proc, atan2(y, x));
}


int32_t am_native_Math_floor(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, floor(x));
}


int32_t am_native_Math_ceil(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, ceil(x));
}


int32_t am_native_Math_round(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, round(x));
}


int32_t am_native_Math_to_fixed(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t n, x;
    if (!native_pop_number(proc, &n)) return -1;
    if (!native_pop_number(proc, &x)) return -1;

    int32_t digits = (int32_t)n;
    if (digits < 0) digits = 0;
    if (digits > 15) digits = 15;

    am_float_t factor = pow((am_float_t)10.0, (am_float_t)digits);
    am_float_t result = round(x * factor) / factor;
    return native_push_result(proc, result);
}


int32_t am_native_Math_abs(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t x;
    if (!native_pop_number(proc, &x)) return -1;
    return native_push_result(proc, fabs(x));
}


// uint 值的逻辑位宽：TPV 用低 5 位作类型标签，uint 有效负载为 (指针位宽 - 5) 位，
// 即 64 位宿主为 59 位、32 位宿主为 27 位。位运算（含符号位解释、移位计数规整）
// 一律以该逻辑位宽为准，保证结果仍可表示为 uint 值。
#define AM_UINT_BIT_WIDTH ((am_uint_t)(sizeof(am_value_t) * 8 - 5))

// 移位计数规整：超出位宽时返回位宽（供调用方特判），否则返回原值。
static am_uint_t native_clamp_shift(am_uint_t n) {
    return (n >= AM_UINT_BIT_WIDTH) ? AM_UINT_BIT_WIDTH : n;
}

int32_t am_native_Math_bit_and(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_uint_t a, b;
    if (!native_pop_uint(proc, &b)) return -1;
    if (!native_pop_uint(proc, &a)) return -1;
    return native_push_uint_result(proc, a & b);
}


int32_t am_native_Math_bit_or(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_uint_t a, b;
    if (!native_pop_uint(proc, &b)) return -1;
    if (!native_pop_uint(proc, &a)) return -1;
    return native_push_uint_result(proc, a | b);
}


int32_t am_native_Math_bit_xor(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_uint_t a, b;
    if (!native_pop_uint(proc, &b)) return -1;
    if (!native_pop_uint(proc, &a)) return -1;
    return native_push_uint_result(proc, a ^ b);
}


int32_t am_native_Math_bit_not(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_uint_t a;
    if (!native_pop_uint(proc, &a)) return -1;
    return native_push_uint_result(proc, ~a);
}


int32_t am_native_Math_bit_shl(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_uint_t a, n;
    if (!native_pop_uint(proc, &n)) return -1;
    if (!native_pop_uint(proc, &a)) return -1;
    n = native_clamp_shift(n);
    return native_push_uint_result(proc, (n >= AM_UINT_BIT_WIDTH) ? 0 : (a << n));
}


int32_t am_native_Math_bit_ashr(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_uint_t a, n;
    if (!native_pop_uint(proc, &n)) return -1;
    if (!native_pop_uint(proc, &a)) return -1;
    n = native_clamp_shift(n);
    if (n >= AM_UINT_BIT_WIDTH) n = AM_UINT_BIT_WIDTH - 1;
    am_uint_t sign_bit = (am_uint_t)1 << (AM_UINT_BIT_WIDTH - 1);
    am_uint_t result;
    if (n == 0) {
        result = a;
    }
    else if (a & sign_bit) {
        // 负数：算术右移，高位补 1（可移植实现，不依赖有符号右移的具体行为）
        result = (a >> n) | (~(am_uint_t)0 << (AM_UINT_BIT_WIDTH - n));
    }
    else {
        result = a >> n;
    }
    return native_push_uint_result(proc, result);
}


int32_t am_native_Math_bit_lshr(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_uint_t a, n;
    if (!native_pop_uint(proc, &n)) return -1;
    if (!native_pop_uint(proc, &a)) return -1;
    n = native_clamp_shift(n);
    return native_push_uint_result(proc, (n >= AM_UINT_BIT_WIDTH) ? 0 : (a >> n));
}


int32_t am_native_Math_random(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = true;
    }
    am_float_t result = (am_float_t)rand() / ((am_float_t)RAND_MAX + (am_float_t)1.0);
    return native_push_result(proc, result);
}


static const am_native_func_entry_t am_native_Math_funcs[] = {
    { L"PI",       am_native_Math_PI },
    { L"pow",      am_native_Math_pow },
    { L"sqrt",     am_native_Math_sqrt },
    { L"exp",      am_native_Math_exp },
    { L"log",      am_native_Math_log },
    { L"log10",    am_native_Math_log10 },
    { L"log2",     am_native_Math_log2 },
    { L"sin",      am_native_Math_sin },
    { L"cos",      am_native_Math_cos },
    { L"tan",      am_native_Math_tan },
    { L"atan",     am_native_Math_atan },
    { L"atan2",    am_native_Math_atan2 },
    { L"bit_and",  am_native_Math_bit_and },
    { L"bit_or",   am_native_Math_bit_or },
    { L"bit_xor",  am_native_Math_bit_xor },
    { L"bit_not",  am_native_Math_bit_not },
    { L"bit_shl",  am_native_Math_bit_shl },
    { L"bit_ashr", am_native_Math_bit_ashr },
    { L"bit_lshr", am_native_Math_bit_lshr },
    { L"floor",    am_native_Math_floor },
    { L"ceil",     am_native_Math_ceil },
    { L"round",    am_native_Math_round },
    { L"to_fixed", am_native_Math_to_fixed },
    { L"abs",      am_native_Math_abs },
    { L"random",   am_native_Math_random },
};

const am_native_lib_entry_t am_native_Math_lib = {
    L"Math",
    am_native_Math_funcs,
    sizeof(am_native_Math_funcs) / sizeof(am_native_Math_funcs[0])
};
