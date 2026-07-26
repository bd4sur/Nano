/* =============================================================================
 * Animac（灵机）解释器 —— Amalgamation 单文件头文件
 *
 * 本文件由 amalgamate.sh 自动生成，请勿手工编辑。
 * 内容来源：include/animac.h 伞形头文件登记的解释器核心头文件，
 *           按依赖顺序拓扑排序合并；局部 #include 已剔除。
 *           不含 am_host.h / am_native_*.h / am_highlight.h / am_repl.h 等宿主相关头文件。
 * 生成时间：2026-07-25 18:08:59 +0800
 * ============================================================================ */

#ifndef __ANIMAC_CORE_H__
#define __ANIMAC_CORE_H__

#ifdef __cplusplus
extern "C" {
#endif


/* ===== begin: include/am_allocator.h ===== */
#ifndef __AM_ALLOCATOR_H__
#define __AM_ALLOCATOR_H__

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////
// 抽象内存分配器
// NOTE 注意allocator返回的指针必须是2字节对齐的！以确保其am_value_t的最低位恒0。
///////////////////////////////////////////

// 定义抽象内存管理的虚接口（虚函数表）
typedef struct am_allocator_vtable_t {
    void* (*malloc)(void *state, size_t size);
    void* (*calloc)(void *state, size_t size);
    void* (*realloc)(void *state, void *ptr, size_t size);
    void  (*free)(void *state, void *ptr);
    void  (*destroy)(void *state); // 销毁整个分配器
} am_allocator_vtable_t;

// 抽象内存管理器：其实现待定
typedef struct am_allocator_t {
    const am_allocator_vtable_t *vtable; // 指向具体的实现
    void *state; // TODO 具体策略的上下文（如FreeList的头指针，Arena的指针等）
} am_allocator_t;

// 抽象内存管理接口
static inline void* am_malloc(am_allocator_t *alloc, size_t size) {
    return alloc->vtable->malloc(alloc->state, size);
}
static inline void* am_calloc(am_allocator_t *alloc, size_t size) {
    return alloc->vtable->calloc(alloc->state, size);
}
static inline void* am_realloc(am_allocator_t *alloc, void *ptr, size_t size) {
    return alloc->vtable->realloc(alloc->state, ptr, size);
}
static inline void am_free(am_allocator_t *alloc, void *ptr) {
    return alloc->vtable->free(alloc->state, ptr);
}

// 示例：虚函数的具体实现
// void* am_malloc_impl(void *state, size_t size) { return malloc(size); }
// void am_free_impl(void *state, void *ptr) { free(ptr); }
// const am_allocator_vtable_t malloc_vtable = { am_malloc_impl, am_free_impl, NULL };





#ifndef AM_ALLOCATOR_PRINT_COMPACT_REPORT
#define AM_ALLOCATOR_PRINT_COMPACT_REPORT (0)
#endif

///////////////////////////////////////////
// 宿主内存分配虚函数表（依赖倒置）
// 说明：allocator 不直接依赖宿主系统的 malloc/calloc/realloc/free，
// 而是由宿主在调用 am_allocator_pool_create 时，通过本虚函数表注入具体实现。
// 四个成员均为必需能力，任一为 NULL 时 am_allocator_pool_create 失败。
///////////////////////////////////////////

typedef struct am_allocator_host_vtable_t {
    void *(*host_malloc)(size_t nbytes);
    void *(*host_calloc)(size_t n, size_t sizeoftype);
    void *(*host_realloc)(void *ptr, size_t n);
    void  (*host_free)(void *ptr);
} am_allocator_host_vtable_t;

///////////////////////////////////////////
// 共享内存池与双分配器管理
///////////////////////////////////////////

// 动态边界调整相关阈值与限制。
// 边界以占总池比例表示；heap 区最小/最大比例受以下宏约束。
#ifndef AM_POOL_MIN_HEAP_RATIO
#define AM_POOL_MIN_HEAP_RATIO (0.1)
#endif
#ifndef AM_POOL_MIN_VM_RATIO
#define AM_POOL_MIN_VM_RATIO (0.1)
#endif

#ifndef AM_POOL_VM_EXPAND_THRESHOLD
#define AM_POOL_VM_EXPAND_THRESHOLD (0.75)
#endif
#ifndef AM_POOL_HEAP_EXPAND_THRESHOLD
#define AM_POOL_HEAP_EXPAND_THRESHOLD (0.75)
#endif
#ifndef AM_POOL_VM_SLACK_THRESHOLD
#define AM_POOL_VM_SLACK_THRESHOLD (0.30)
#endif
#ifndef AM_POOL_HEAP_SLACK_THRESHOLD
#define AM_POOL_HEAP_SLACK_THRESHOLD (0.30)
#endif
#ifndef AM_POOL_BOUNDARY_ADJ_STEP
#define AM_POOL_BOUNDARY_ADJ_STEP (0.05)
#endif

// 不透明内存池类型
typedef struct am_allocator_pool_t am_allocator_pool_t;

// 创建/销毁统一内存池。成功返回池指针，失败返回 NULL。
// host_vtable 为宿主内存分配虚函数表，不允许为 NULL，且四个成员均不允许为 NULL；
// 池仅保存指针，不拷贝，宿主须保证 vtable 的生命周期不短于池。
am_allocator_pool_t *am_allocator_pool_create(size_t total_size, const am_allocator_host_vtable_t *host_vtable);
void am_allocator_pool_destroy(am_allocator_pool_t *pool);

// 获取池中 VM 工作区与堆区分配器。
am_allocator_t *am_allocator_pool_get_vm(am_allocator_pool_t *pool);
am_allocator_t *am_allocator_pool_get_heap(am_allocator_pool_t *pool);

// 重置 VM 工作区/堆区。重置会丢弃当前已分配内容，回到初始状态。
void am_allocator_pool_reset_vm(am_allocator_pool_t *pool);
void am_allocator_pool_reset_heap(am_allocator_pool_t *pool);

// 查询池大小与已使用字节数。
size_t am_allocator_pool_total_size(const am_allocator_pool_t *pool);
size_t am_allocator_pool_vm_used(const am_allocator_pool_t *pool);
size_t am_allocator_pool_heap_used(const am_allocator_pool_t *pool);
size_t am_allocator_pool_heap_capacity(const am_allocator_pool_t *pool);

// 经池的宿主内存分配虚函数表分配/释放临时内存（供 GC 等上层做暂存）。
// 仅支持内存池的堆区分配器（am_allocator_pool_get_heap 的返回值），其余返回 NULL。
void *am_allocator_host_malloc(am_allocator_t *alloc, size_t size);
void *am_allocator_host_realloc(am_allocator_t *alloc, void *ptr, size_t size);
void  am_allocator_host_free(am_allocator_t *alloc, void *ptr);

// 查询堆区分配器的使用量、最大空闲块与近期最大分配请求（供 GC 水位与碎片判断；后两个参数可传 NULL 跳过）。
// 仅支持内存池的堆区分配器（am_allocator_pool_get_heap 的返回值），其余返回 -1。
int32_t am_allocator_heap_usage(const am_allocator_t *alloc, size_t *used_bytes, size_t *capacity,
                                size_t *largest_free_block, size_t *largest_request);

// 读取并清除堆区分配失败标志：此前曾发生彻底分配失败（边界让渡重试后仍失败）
// 返回 1 并清除标志，否则返回 0；alloc 非堆区分配器返回 -1。
int32_t am_allocator_heap_take_oom_flag(am_allocator_t *alloc);

// 重定位回调：存活对象被搬移到 new_payload 后由压缩引擎回调，按地址升序逐次触发。
typedef void (*am_allocator_relocate_fn)(void *ctx, void *old_payload, void *new_payload);

// 对堆区执行标记-压缩引擎（纯物理操作，不依赖逻辑堆）：
// 遍历堆区物理块，将 payload 出现在 live_payloads 中的已用块搬移到堆区前端，
// 每搬移一个对象经 on_relocate 回调报告一次重定位（old/new payload 均按地址升序），
// 最后在尾部重建一个空闲块。live_payloads 必须是按指针升序且无重复的数组。
// 必须在 GC 安全点调用。成功返回 0，失败返回 -1。
int32_t am_allocator_heap_compact(am_allocator_t *heap_alloc,
                                  void *const *live_payloads, size_t live_count,
                                  am_allocator_relocate_fn on_relocate, void *ctx);

// 按占总池比例调整 VM/heap 边界。
// - ratio 为 heap 区所占比例；内部会被裁剪到 [AM_POOL_MIN_HEAP_RATIO, 1 - AM_POOL_MIN_VM_RATIO]。
// - 若新边界大于当前边界（heap 扩张），仅当 VM 工作区为空时才允许。
// - 若新边界小于当前边界（VM 扩张），要求当前已用 heap 对象能够放入新的 heap 容量中。
int32_t am_allocator_pool_adjust_boundary(am_allocator_pool_t *pool, double ratio);

// 根据 VM/heap 使用压力自动调整边界。通常在每个 GC 安全点之后调用。
int32_t am_allocator_pool_auto_adjust(am_allocator_pool_t *pool);

// 返回当前活动的内存池（单池场景下使用）。
am_allocator_pool_t *am_allocator_pool_current(void);

#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_allocator.h ===== */

/* ===== begin: include/am_object.h ===== */
#ifndef __AM_OBJECT_H__
#define __AM_OBJECT_H__

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct am_object_t;
typedef struct am_object_t am_object_t;

struct am_wstring_t;
typedef struct am_wstring_t am_wstring_t;

struct am_list_t;
typedef struct am_list_t am_list_t;

struct am_map_t;
typedef struct am_map_t am_map_t;

struct am_array_t;
typedef struct am_array_t am_array_t;

struct am_heap_t;
typedef struct am_heap_t am_heap_t;



///////////////////////////////////////////
// 对象语言数据值：TPV (Tagged Pointer Value)
///////////////////////////////////////////

// 与架构相关的基本类型
#if UINTPTR_MAX == 0xFFFFFFFF
    // 32 位系统
    typedef int32_t  am_int_t;
    typedef uint32_t am_uint_t;
    typedef float    am_float_t;
    typedef uint32_t am_float_bits_t;
    typedef size_t   am_symbol_t;
    typedef size_t   am_iaddr_t;
    typedef size_t   am_handle_t;
    typedef size_t   am_varid_t;
    typedef size_t   am_label_t;
#elif UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFu
    // 64 位系统
    typedef int64_t  am_int_t;
    typedef uint64_t am_uint_t;
    typedef double   am_float_t;
    typedef uint64_t am_float_bits_t;
    typedef size_t   am_symbol_t;
    typedef size_t   am_iaddr_t;
    typedef size_t   am_handle_t;
    typedef size_t   am_varid_t;
    typedef size_t   am_label_t;
#else
    #error "Only 32-bit and 64-bit architectures are supported."
#endif


// 与架构无关的基本类型
typedef bool am_boolean_t;
typedef uint32_t am_wchar_t;
typedef uint8_t am_undefined_t;
typedef uint8_t am_null_t;


// TPV(Tagged Pointer Value)作为唯一的值类型
typedef uintptr_t am_value_t;



// TPV的类型枚举
#define AM_VALUE_TYPE_PTR (0x00)
// 以下均为IMME
#define AM_VALUE_TYPE_HANDLE    (0x01) // uint_like
#define AM_VALUE_TYPE_IADDR     (0x02) // uint_like
#define AM_VALUE_TYPE_VARID     (0x03) // uint_like
#define AM_VALUE_TYPE_LABEL     (0x04) // uint_like
#define AM_VALUE_TYPE_BOOLEAN   (0x05) // uint_like
#define AM_VALUE_TYPE_NULL      (0x06) // uint_like, 单例
#define AM_VALUE_TYPE_UNDEFINED (0x07) // uint_like, 单例
#define AM_VALUE_TYPE_SYMBOL    (0x08) // uint_like, keyword也是一种特殊的symbol，在编译时就应该放进symbol映射表中
#define AM_VALUE_TYPE_WCHAR     (0x09) // wchar_t, 仅用于组成字符串
#define AM_VALUE_TYPE_UINT      (0x0A) // number
#define AM_VALUE_TYPE_INT       (0x0B) // number
#define AM_VALUE_TYPE_FLOAT     (0x0C) // number

// TPV的类型标记，占用TPV低5位：最低位为0则为PTR；最低位为1则为立即数，其余4位对应AM_VALUE_TYPE_*
#define AM_VALUE_TAG_PTR       ((am_value_t)0x00ULL) // 指向堆上对象的指针
#define AM_VALUE_TAG_HANDLE    ((am_value_t)0x03ULL)
#define AM_VALUE_TAG_IADDR     ((am_value_t)0x05ULL)
#define AM_VALUE_TAG_VARID     ((am_value_t)0x07ULL)
#define AM_VALUE_TAG_LABEL     ((am_value_t)0x09ULL)
#define AM_VALUE_TAG_BOOLEAN   ((am_value_t)0x0BULL)
#define AM_VALUE_TAG_NULL      ((am_value_t)0x0DULL)
#define AM_VALUE_TAG_UNDEFINED ((am_value_t)0x0FULL)
#define AM_VALUE_TAG_SYMBOL    ((am_value_t)0x11ULL)
#define AM_VALUE_TAG_WCHAR     ((am_value_t)0x13ULL)  // 最少27bits，能装得下unicode全部码点
#define AM_VALUE_TAG_UINT      ((am_value_t)0x15ULL)
#define AM_VALUE_TAG_INT       ((am_value_t)0x17ULL)
#define AM_VALUE_TAG_FLOAT     ((am_value_t)0x19ULL)

#define AM_VALUE_TAG_MASK      ((am_value_t)0x1FULL)
#define AM_VALUE_TAG_LSB_MASK  ((am_value_t)0x1ULL)



// 方便构建uint_like的value
#define AM_MAKE_VALUE_OF_UINT_LIKE(x, imme_type_tag) ((am_value_t)(((am_value_t)(x) << 5) | (imme_type_tag)))


///////////////////////////////////////////
// 特殊（单例）TPV
///////////////////////////////////////////

#define AM_VALUE_NULL      AM_MAKE_VALUE_OF_UINT_LIKE(0x0, AM_VALUE_TAG_NULL)
#define AM_VALUE_UNDEFINED AM_MAKE_VALUE_OF_UINT_LIKE(0x0, AM_VALUE_TAG_UNDEFINED)
#define AM_VALUE_TRUE      AM_MAKE_VALUE_OF_UINT_LIKE(0x1, AM_VALUE_TAG_BOOLEAN)
#define AM_VALUE_FALSE     AM_MAKE_VALUE_OF_UINT_LIKE(0x0, AM_VALUE_TAG_BOOLEAN)

// 首把柄和空把柄（值为(UINTPTR_MAX>>5)的把柄）
#define AM_HANDLE_BASE ((am_handle_t)0x0)
#define AM_HANDLE_NULL ((am_handle_t)(UINTPTR_MAX>>5))
#define AM_VALUE_HANDLE_BASE  AM_MAKE_VALUE_OF_UINT_LIKE(0x0, AM_VALUE_TAG_HANDLE)
#define AM_VALUE_HANDLE_NULL  AM_MAKE_VALUE_OF_UINT_LIKE(UINTPTR_MAX, AM_VALUE_TAG_HANDLE)

// 关键字（保留的symbol）
// NOTE 关键字在词法上属于identifier，与variable接近；但是在语义上属于symbol，全局保留的symbol，不带前导单引号的特殊symbol
// lambda define set! let begin return ... _
// if and or cond else for while break continue case do
// quote quasiquote unquote
// import native
// define-syntax let-syntax letrec-syntax syntax-rules
#define AM_VALUE_KW_lambda     AM_MAKE_VALUE_OF_UINT_LIKE(0x00, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_define     AM_MAKE_VALUE_OF_UINT_LIKE(0x01, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_set        AM_MAKE_VALUE_OF_UINT_LIKE(0x02, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_let        AM_MAKE_VALUE_OF_UINT_LIKE(0x03, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_begin      AM_MAKE_VALUE_OF_UINT_LIKE(0x04, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_return     AM_MAKE_VALUE_OF_UINT_LIKE(0x05, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_dot3       AM_MAKE_VALUE_OF_UINT_LIKE(0x06, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_underscore AM_MAKE_VALUE_OF_UINT_LIKE(0x07, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_if         AM_MAKE_VALUE_OF_UINT_LIKE(0x08, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_and        AM_MAKE_VALUE_OF_UINT_LIKE(0x09, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_or         AM_MAKE_VALUE_OF_UINT_LIKE(0x0A, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_cond       AM_MAKE_VALUE_OF_UINT_LIKE(0x0B, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_else       AM_MAKE_VALUE_OF_UINT_LIKE(0x0C, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_for        AM_MAKE_VALUE_OF_UINT_LIKE(0x0D, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_while      AM_MAKE_VALUE_OF_UINT_LIKE(0x0E, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_break      AM_MAKE_VALUE_OF_UINT_LIKE(0x0F, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_continue   AM_MAKE_VALUE_OF_UINT_LIKE(0x10, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_case       AM_MAKE_VALUE_OF_UINT_LIKE(0x11, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_do         AM_MAKE_VALUE_OF_UINT_LIKE(0x12, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_quote      AM_MAKE_VALUE_OF_UINT_LIKE(0x13, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_quasiquote AM_MAKE_VALUE_OF_UINT_LIKE(0x14, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_unquote    AM_MAKE_VALUE_OF_UINT_LIKE(0x15, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_import     AM_MAKE_VALUE_OF_UINT_LIKE(0x16, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_native     AM_MAKE_VALUE_OF_UINT_LIKE(0x17, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_define_syntax AM_MAKE_VALUE_OF_UINT_LIKE(0x18, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_let_syntax    AM_MAKE_VALUE_OF_UINT_LIKE(0x19, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_letrec_syntax AM_MAKE_VALUE_OF_UINT_LIKE(0x1A, AM_VALUE_TAG_SYMBOL)
#define AM_VALUE_KW_syntax_rules  AM_MAKE_VALUE_OF_UINT_LIKE(0x1B, AM_VALUE_TAG_SYMBOL)



// TPV基本操作

// 获取类型（AM_VALUE_TYPE_*）
static inline int32_t am_value_type(am_value_t v) {
    if ((v & AM_VALUE_TAG_LSB_MASK) == AM_VALUE_TAG_PTR) {
        return AM_VALUE_TYPE_PTR;
    }
    else {
        return ((v & AM_VALUE_TAG_MASK) >> 1);
    }
}

// 类型谓词
static inline bool am_value_is_ptr(am_value_t v)       { return (v & AM_VALUE_TAG_LSB_MASK) == AM_VALUE_TAG_PTR; }
static inline bool am_value_is_imme(am_value_t v)      { return (v & AM_VALUE_TAG_LSB_MASK) == 0x1; }
static inline bool am_value_is_handle(am_value_t v)    { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_HANDLE; }
static inline bool am_value_is_iaddr(am_value_t v)     { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_IADDR; }
static inline bool am_value_is_varid(am_value_t v)     { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_VARID; }
static inline bool am_value_is_label(am_value_t v)     { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_LABEL; }
static inline bool am_value_is_boolean(am_value_t v)   { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_BOOLEAN; }
static inline bool am_value_is_null(am_value_t v)      { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_NULL; }
static inline bool am_value_is_undefined(am_value_t v) { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_UNDEFINED; }
static inline bool am_value_is_symbol(am_value_t v)    { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_SYMBOL; }
static inline bool am_value_is_wchar(am_value_t v)     { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_WCHAR; }
static inline bool am_value_is_uint(am_value_t v)      { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_UINT; }
static inline bool am_value_is_int(am_value_t v)       { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_INT; }
static inline bool am_value_is_float(am_value_t v)     { return (v & AM_VALUE_TAG_MASK) == AM_VALUE_TAG_FLOAT; }
static inline bool am_value_is_number(am_value_t v)    { return am_value_is_float(v) || am_value_is_int(v) || am_value_is_uint(v); }



// 解包（不做类型检查，直接解包）
static inline am_object_t*   am_value_to_ptr(am_value_t v)       { return (am_object_t*)(v & ~AM_VALUE_TAG_LSB_MASK); }
static inline am_handle_t    am_value_to_handle(am_value_t v)    { return (am_handle_t)(v >> 5); }
static inline am_iaddr_t     am_value_to_iaddr(am_value_t v)     { return (am_iaddr_t)(v >> 5); }
static inline am_varid_t     am_value_to_varid(am_value_t v)     { return (am_varid_t)(v >> 5); }
static inline am_label_t     am_value_to_label(am_value_t v)     { return (am_label_t)(v >> 5); }
static inline am_boolean_t   am_value_to_boolean(am_value_t v)   { return (am_boolean_t)(v >> 5); } // 整数部分非0即为#t，除此之外全部为#f
static inline am_null_t      am_value_to_null(am_value_t v)      { (void)v; return (am_null_t)(1); } // 单例：常函数，且具体值不重要
static inline am_undefined_t am_value_to_undefined(am_value_t v) { (void)v; return (am_undefined_t)(1); } // 单例：常函数，具体值不重要
static inline am_symbol_t    am_value_to_symbol(am_value_t v)    { return (am_symbol_t)(v >> 5); }
static inline am_wchar_t     am_value_to_wchar(am_value_t v)     { return (am_wchar_t)(v >> 5); }
static inline am_uint_t      am_value_to_uint(am_value_t v)      { return (am_uint_t)(v >> 5); }
static inline am_int_t am_value_to_int(am_value_t v) {
    am_value_t data = v >> 5; // 剥离类型标签
    am_int_t shifted = (am_int_t)(data << 5); // 跨平台符号扩展：推到最高位
    return shifted >> 5; // 算术右移恢复
}
static inline am_float_t am_value_to_float(am_value_t v) {
    uintptr_t data = v >> 5; // 剥离类型标签
    am_float_bits_t bits = (am_float_bits_t)(data << 5); // 左移 5 位恢复高位，低 5 位自动补 0
    am_float_t f;
    memcpy(&f, &bits, sizeof(am_float_t)); // 安全还原为浮点数
    return f;
}

// 打包
static inline am_value_t am_make_value_of_ptr(am_object_t* obj_p) { return (am_value_t)obj_p; }
static inline am_value_t am_make_value_of_handle(am_handle_t x) { return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_HANDLE); }
static inline am_value_t am_make_value_of_iaddr(am_iaddr_t x) { return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_IADDR); }
static inline am_value_t am_make_value_of_varid(am_varid_t x) { return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_VARID); }
static inline am_value_t am_make_value_of_label(am_label_t x) { return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_LABEL); }
static inline am_value_t am_make_value_of_boolean(am_boolean_t x) { return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_BOOLEAN); }
static inline am_value_t am_make_value_of_null(am_null_t x) { (void)x; return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_NULL); } // 单例：常函数，输入不重要
static inline am_value_t am_make_value_of_undefined(am_undefined_t x) { (void)x; return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_UNDEFINED); } // 单例：常函数，输入不重要
static inline am_value_t am_make_value_of_symbol(am_symbol_t x) { return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_SYMBOL); }
static inline am_value_t am_make_value_of_wchar(am_wchar_t x) { return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_WCHAR); }
static inline am_value_t am_make_value_of_uint(am_uint_t x) { return AM_MAKE_VALUE_OF_UINT_LIKE(x, AM_VALUE_TAG_UINT); }
static inline am_value_t am_make_value_of_int(am_int_t x) {
    am_value_t bits = (am_value_t)x;
    am_value_t shifted = bits << 5;
    return (shifted | (AM_VALUE_TAG_INT));
}
static inline am_value_t am_make_value_of_float(am_float_t x) {
    am_float_bits_t bits;
    memcpy(&bits, &x, sizeof(am_float_t)); // 安全获取 IEEE754 位模式
    uintptr_t data = (uintptr_t)(bits >> 5); // 无论32位还是64位都截断低5位尾数，保留符号位和指数位
    return ((data << 5) | (AM_VALUE_TAG_FLOAT)); // 左移5位腾出类型标签，并填入
}











///////////////////////////////////////////
// 对象语言数据对象：Object
///////////////////////////////////////////


// Object类型枚举
#define AM_OBJECT_TYPE_BASE         (0x00)  // 默认类型（基类）
#define AM_OBJECT_TYPE_LIST         (0x01)  // 通用线性表List<am_value_t>
#define AM_OBJECT_TYPE_MAP          (0x02)  // 通用散列表Map<am_value_t, am_value_t>
#define AM_OBJECT_TYPE_WSTRING      (0x03)  // 字符串（wstring表示uint32_t构成的宽字符串，即由unicode码点直接构成，无任何压缩编码如utf-16等）
#define AM_OBJECT_TYPE_PORT         (0x04)  // 端口（对IO的抽象）
#define AM_OBJECT_TYPE_CLOSURE      (0x05)  // 闭包
#define AM_OBJECT_TYPE_CONTINUATION (0x06)  // 续体
#define AM_OBJECT_TYPE_FRAME        (0x07)  // 栈帧
#define AM_OBJECT_TYPE_ILCODE       (0x08)  // 中间语言指令 TODO
#define AM_OBJECT_TYPE_BOX          (0x09)  // 基本类型装箱 TODO
#define AM_OBJECT_TYPE_TOKEN        (0x0A)  // 词元
#define AM_OBJECT_TYPE_SCOPE        (0x0B)  // 词法作用域（环境帧）
#define AM_OBJECT_TYPE_VOCAB        (0x0C)  // 词典（字符串集合）
#define AM_OBJECT_TYPE_MODULE       (0x0D)  // 模块
#define AM_OBJECT_TYPE_PROCESS      (0x0E)  // 进程
#define AM_OBJECT_TYPE_STRINDEX     (0x0F)  // 字符串索引（多值哈希表，用于字符串驻留）


// Object基类（公共头）
typedef struct am_object_t {
    uint32_t header; // TODO 预留，包括魔法值、static标记等
    uint32_t hash;   // T散列值
    uint32_t gcmark; // TODO 用于垃圾回收，具体用法待定，取决于垃圾回收算法
    int32_t  type;   // 对象类型（AM_OBJECT_TYPE_*）
} am_object_t;




///////////////////////////////////////////
// 对象头元数据操作
///////////////////////////////////////////

// 获取/设置对象“静态”属性（header最低位，1为static，0为非static）
// 是静态则返回/输入0，不是静态则返回/输入-1。
int32_t am_object_check_static(am_object_t *obj);
int32_t am_object_set_static(am_object_t *obj, int32_t is_static);

// 获取/设置对象“保持存活”属性（header从LSB倒数第二位，1为keepalive，0为非keepalive）
// 是“保持存活”则返回/输入0，不是“保持存活”则返回/输入-1。
int32_t am_object_check_keepalive(am_object_t *obj);
int32_t am_object_set_keepalive(am_object_t *obj, int32_t is_keepalive);

// 获取/设置对象“存活”状态，用于GC（gcmark最高位，1为alive，0为非alive）
// 是“存活”则返回/输入0，不是“存活”则返回/输入-1。
int32_t am_object_check_alive(am_object_t *obj);
int32_t am_object_set_alive(am_object_t *obj, int32_t is_alive);




///////////////////////////////////////////
// WString对象
///////////////////////////////////////////

// WString堆对象（作为对象语言的数据对象，实质上是am_wstring_t）
typedef am_wstring_t am_obj_wstring_t;



///////////////////////////////////////////
// List对象
///////////////////////////////////////////

// List堆对象（作为对象语言的数据对象，实质上是am_list_t）
typedef am_list_t am_obj_list_t;



///////////////////////////////////////////
// Map对象
///////////////////////////////////////////

// Map堆对象（作为对象语言的数据对象，实质上是am_map_t）
typedef am_map_t am_obj_map_t;













///////////////////////////////////////////
// 平台无关固定宽度磁盘格式序列化原语
//
// 设计目标：
//   1. 与宿主字长（32/64位）、指针长度、size_t 长度、结构体填充完全无关；
//   2. 与宿主字节序无关：所有多字节整数一律以小端序（LE）显式按字节读写；
//   3. 尽可能紧凑：计数、索引、句柄等小值整数采用 ULEB128 变长编码，
//      有符号整数采用 zigzag+ULEB128 编码；TPV 采用 1字节类型标签+变长负载。
//
// 基本编码规则：
//   - u8/u16/u32/u64：定长小端；
//   - uvarint：ULEB128（每字节7位有效载荷，MSB为续位标志）；
//   - svarint：zigzag 映射后的 ULEB128；
//   - f64：IEEE-754 double 的 64 位位模式（小端）。
//
// TPV（am_value_t）磁盘编码 dvalue：
//   - 第1字节：类型标签 = AM_VALUE_TYPE_*（0x00~0x0C）；
//   - 负载：
//       PTR      (0x00)：uvarint(原始指针位模式)（仅用于堆转储中的对象偏移量，必须为偶数）
//       HANDLE/IADDR/VARID/LABEL/BOOLEAN/SYMBOL/WCHAR/UINT：uvarint(运行时值 >> 5)
//       NULL/UNDEFINED：无负载（载荷隐含为0）
//       INT：svarint(整数值)（磁盘上统一视为64位有符号整数）
//       FLOAT：f64（IEEE-754 double；32位平台上由float精确提升/舍入还原）
//
// 所有写函数允许 buffer 为 NULL（仅计算字节数，不实际写入）。
///////////////////////////////////////////


// ===============================================================================
// 定长小端整数
// ===============================================================================

static inline void am_disk_write_u16(uint8_t *buf, size_t off, uint16_t v) {
    if (!buf) return;
    buf[off + 0] = (uint8_t)(v & 0xFFu);
    buf[off + 1] = (uint8_t)((v >> 8) & 0xFFu);
}

static inline void am_disk_write_u32(uint8_t *buf, size_t off, uint32_t v) {
    if (!buf) return;
    buf[off + 0] = (uint8_t)(v & 0xFFu);
    buf[off + 1] = (uint8_t)((v >> 8) & 0xFFu);
    buf[off + 2] = (uint8_t)((v >> 16) & 0xFFu);
    buf[off + 3] = (uint8_t)((v >> 24) & 0xFFu);
}

static inline void am_disk_write_u64(uint8_t *buf, size_t off, uint64_t v) {
    if (!buf) return;
    for (int i = 0; i < 8; i++) {
        buf[off + i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
    }
}

static inline uint16_t am_disk_read_u16(const uint8_t *buf, size_t off) {
    return (uint16_t)((uint16_t)buf[off] | ((uint16_t)buf[off + 1] << 8));
}

static inline uint32_t am_disk_read_u32(const uint8_t *buf, size_t off) {
    return ((uint32_t)buf[off + 0]) |
           ((uint32_t)buf[off + 1] << 8) |
           ((uint32_t)buf[off + 2] << 16) |
           ((uint32_t)buf[off + 3] << 24);
}

static inline uint64_t am_disk_read_u64(const uint8_t *buf, size_t off) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= ((uint64_t)buf[off + i]) << (8 * i);
    }
    return v;
}


// ===============================================================================
// 变长整数（ULEB128 / zigzag+ULEB128）
// ===============================================================================

// 写入 ULEB128 编码的无符号整数。返回占用字节数。buf 为 NULL 时仅计算字节数。
static inline size_t am_disk_write_uvarint(uint8_t *buf, size_t off, uint64_t v) {
    size_t n = 0;
    do {
        uint8_t b = (uint8_t)(v & 0x7Fu);
        v >>= 7;
        if (v) b |= 0x80u;
        if (buf) buf[off + n] = b;
        n++;
    } while (v);
    return n;
}

// 读取 ULEB128 编码的无符号整数。成功返回消耗字节数（>=1），失败（溢出/超长）返回0。
static inline size_t am_disk_read_uvarint(const uint8_t *buf, size_t off, uint64_t *out) {
    uint64_t v = 0;
    for (size_t n = 0; n < 10; n++) {
        uint8_t b = buf[off + n];
        if (n == 9 && b > 1) return 0; // 超过64位
        v |= ((uint64_t)(b & 0x7Fu)) << (7 * n);
        if (!(b & 0x80u)) {
            *out = v;
            return n + 1;
        }
    }
    return 0;
}

// 写入 zigzag+ULEB128 编码的有符号整数。返回占用字节数。buf 为 NULL 时仅计算字节数。
static inline size_t am_disk_write_svarint(uint8_t *buf, size_t off, int64_t v) {
    uint64_t z = ((uint64_t)v << 1) ^ (uint64_t)(v >> 63);
    return am_disk_write_uvarint(buf, off, z);
}

// 读取 zigzag+ULEB128 编码的有符号整数。成功返回消耗字节数，失败返回0。
static inline size_t am_disk_read_svarint(const uint8_t *buf, size_t off, int64_t *out) {
    uint64_t z = 0;
    size_t n = am_disk_read_uvarint(buf, off, &z);
    if (!n) return 0;
    *out = (int64_t)(z >> 1) ^ (-(int64_t)(z & 1u));
    return n;
}


// ===============================================================================
// IEEE-754 double（小端位模式）
// ===============================================================================

static inline void am_disk_write_f64(uint8_t *buf, size_t off, double d) {
    uint64_t bits = 0;
    memcpy(&bits, &d, sizeof(bits));
    am_disk_write_u64(buf, off, bits);
}

static inline double am_disk_read_f64(const uint8_t *buf, size_t off) {
    uint64_t bits = am_disk_read_u64(buf, off);
    double d = 0.0;
    memcpy(&d, &bits, sizeof(d));
    return d;
}


// ===============================================================================
// 对象基类头 am_object_t（固定16字节：u32 header, u32 hash, u32 gcmark, i32 type）
// ===============================================================================

#define AM_DISK_BASE_SIZE (16)

static inline void am_disk_write_base(uint8_t *buf, size_t off, const am_object_t *base) {
    am_disk_write_u32(buf, off + 0,  base->header);
    am_disk_write_u32(buf, off + 4,  base->hash);
    am_disk_write_u32(buf, off + 8,  base->gcmark);
    am_disk_write_u32(buf, off + 12, (uint32_t)base->type);
}

static inline void am_disk_read_base(const uint8_t *buf, size_t off, am_object_t *base) {
    base->header = am_disk_read_u32(buf, off + 0);
    base->hash   = am_disk_read_u32(buf, off + 4);
    base->gcmark = am_disk_read_u32(buf, off + 8);
    base->type   = (int32_t)am_disk_read_u32(buf, off + 12);
}


// ===============================================================================
// TPV（am_value_t）磁盘编码
// ===============================================================================

// 本宿主 TPV 立即数能够容纳的无符号载荷上限（payload = value >> 5）
#define AM_DISK_UINT_PAYLOAD_MAX ((uint64_t)(UINTPTR_MAX >> 5))

// 本宿主 TPV 能够容纳的有符号整数范围（make/to_int 往返不失真）
#define AM_DISK_INT_BITS ((int)(sizeof(am_value_t) * 8) - 6)

// 计算 TPV 编码后的字节数。失败（不支持的类型）返回 SIZE_MAX。
static inline size_t am_disk_value_size(am_value_t v) {
    if (am_value_is_float(v)) {
        return 1 + 8;
    }
    if (am_value_is_null(v) || am_value_is_undefined(v)) {
        return 1;
    }
    if (am_value_is_int(v)) {
        return 1 + am_disk_write_svarint(NULL, 0, (int64_t)am_value_to_int(v));
    }
    if (am_value_is_ptr(v)) {
        return 1 + am_disk_write_uvarint(NULL, 0, (uint64_t)v);
    }
    // 其余 uint_like 立即数
    return 1 + am_disk_write_uvarint(NULL, 0, (uint64_t)(v >> 5));
}

// 将 TPV 编码写入 buffer[off]。返回写入字节数；buffer 为 NULL 时仅计算字节数。
static inline size_t am_disk_write_value(uint8_t *buf, size_t off, am_value_t v) {
    uint8_t tag = (uint8_t)am_value_type(v);
    if (buf) buf[off] = tag;

    if (am_value_is_float(v)) {
        if (buf) am_disk_write_f64(buf, off + 1, (double)am_value_to_float(v));
        return 1 + 8;
    }
    if (am_value_is_null(v) || am_value_is_undefined(v)) {
        return 1;
    }
    if (am_value_is_int(v)) {
        return 1 + am_disk_write_svarint(buf, off + 1, (int64_t)am_value_to_int(v));
    }
    if (am_value_is_ptr(v)) {
        // 仅用于堆转储中的对象相对偏移量（必须保持偶数，以维持PTR标签位）
        return 1 + am_disk_write_uvarint(buf, off + 1, (uint64_t)v);
    }
    // 其余 uint_like 立即数
    return 1 + am_disk_write_uvarint(buf, off + 1, (uint64_t)(v >> 5));
}

// 从 buffer[off] 解码 TPV，结果写入 *out。
// 成功返回消耗字节数；失败返回0（含：未知标签、变长整数溢出、32位宿主值域越界）。
static inline size_t am_disk_read_value(const uint8_t *buf, size_t off, am_value_t *out) {
    uint8_t tag = buf[off];
    if (tag > AM_VALUE_TYPE_FLOAT) return 0;

    if (tag == AM_VALUE_TYPE_NULL || tag == AM_VALUE_TYPE_UNDEFINED) {
        *out = AM_MAKE_VALUE_OF_UINT_LIKE(0, ((am_value_t)tag << 1) | 1);
        return 1;
    }
    if (tag == AM_VALUE_TYPE_FLOAT) {
        double d = am_disk_read_f64(buf, off + 1);
        *out = am_make_value_of_float((am_float_t)d);
        return 1 + 8;
    }
    if (tag == AM_VALUE_TYPE_INT) {
        int64_t sv = 0;
        size_t n = am_disk_read_svarint(buf, off + 1, &sv);
        if (!n) return 0;
        // 检查是否超出本宿主 TPV 可表示的整数范围
        if ((sv >> AM_DISK_INT_BITS) != 0 && (sv >> AM_DISK_INT_BITS) != -1) return 0;
        *out = am_make_value_of_int((am_int_t)sv);
        return 1 + n;
    }

    uint64_t payload = 0;
    size_t n = am_disk_read_uvarint(buf, off + 1, &payload);
    if (!n) return 0;

    if (tag == AM_VALUE_TYPE_PTR) {
        // 原始指针位模式（堆转储中的对象偏移量）：必须适配本宿主指针宽度且为偶数
        if (payload > (uint64_t)UINTPTR_MAX) return 0;
        if (payload & 1u) return 0;
        *out = (am_value_t)(uintptr_t)payload;
        return 1 + n;
    }

    // 其余 uint_like 立即数
    if (payload > AM_DISK_UINT_PAYLOAD_MAX) return 0;
    *out = AM_MAKE_VALUE_OF_UINT_LIKE(payload, ((am_value_t)tag << 1) | 1);
    return 1 + n;
}









#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_object.h ===== */

/* ===== begin: include/am_map.h ===== */
#ifndef __AM_MAP_H__
#define __AM_MAP_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////
// 基础数据结构：通用散列表 am_map_t<am_value_t,am_value_t>
///////////////////////////////////////////

// 定义两个特殊key
#define AM_MAP_KEY_EMPTY AM_VALUE_NULL
#define AM_MAP_KEY_TOMBSTONE AM_VALUE_UNDEFINED

// 表项
typedef struct am_map_entry_t {
    am_value_t key;
    am_value_t value;
} am_map_entry_t;

// 散列表（开放寻址法）
// NOTE 说明：虽然am_map_t是基础设施，但定义上让它携带am_object_t头（base），赋予其解释器基础设施和语言对象的双重身份。
//           am_map_t作为解释器的底层基础设施使用时，由解释器本身和宿主环境管理；
//           而作为对象语言的数据对象使用时，它作为am_object_t的一个派生类，接受抽象堆和内存分配器的管理。
typedef struct am_map_t {
    am_object_t base;

    size_t length;     // 当前有效键值对数量
    size_t capacity;   // 物理槽位数 (必须是2的幂)
    size_t mask;       // capacity - 1，用于快速取模
    size_t tombstones; // 墓碑数量，用于触发重哈希
    am_map_entry_t slots[];  // 连续槽位区
} am_map_t;

// 遍历回调类型
typedef void (*am_map_iter_callback_t)(am_value_t key, am_value_t value, void *user_data);

// ===============================================================================
// 通用辅助函数（按位操作）
// ===============================================================================

// 计算 am_value_t 的哈希值（基于其底层位模式）
static inline uint32_t am_value_hash(am_value_t v) {
    uint32_t h = (uint32_t)v;
#if UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFu
    h ^= (uint32_t)(v >> 32);
#endif
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// am_value_t 相等性：按位比较
static inline bool am_value_equal(am_value_t a, am_value_t b) {
    return a == b;
}

// ===============================================================================
// 构造函数
// ===============================================================================

// 以初始容量新建哈希表。capacity 会被向上取整为不小于它的最小 2 的幂。
// 所有 key 初始化为 AM_MAP_KEY_EMPTY，value 初始化为 AM_VALUE_NULL。
am_map_t *am_map_create(am_allocator_t *alloc, size_t capacity);

// ===============================================================================
// 析构与清理
// ===============================================================================

// 清空哈希表：对所有有效 entry，若 value 是指针则先释放，再将 key 置为 EMPTY、value 置为 NULL
int32_t am_map_clear(am_allocator_t *alloc, am_map_t *map);

// 彻底销毁哈希表对象
int32_t am_map_destroy(am_allocator_t *alloc, am_map_t *map);

// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝：创建并返回一个与原 map 内容完全一致的新 map 对象。
// 所有 key/value 按位拷贝（与闭包 Copy 语义一致，不递归拷贝指针指向的对象）。
am_map_t *am_map_copy(am_allocator_t *alloc, am_map_t *map);

// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_map_size(am_allocator_t *alloc, am_map_t *obj);

// ===============================================================================
// 对象二进制转储 TODO
// ===============================================================================

// 功能说明：将散列表对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，丢弃墓碑和空闲槽位。
size_t am_map_dump(am_allocator_t *alloc, am_map_t *map, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的散列表对象，构造散列表对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_map_t对象的指针，失败则返回NULL。
am_map_t *am_map_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);

// ===============================================================================
// 基本操作
// ===============================================================================

// 查找：返回对应的 value；若不存在返回 AM_VALUE_NULL
am_value_t am_map_get(am_allocator_t *alloc, am_map_t *map, am_value_t key);

// 存在性检查：存在返回 0，不存在返回 -1
int32_t am_map_contains(am_allocator_t *alloc, am_map_t *map, am_value_t key);

// 不扩容地插入或修改（stable 版本）。
// 仅做插入/替换，绝不分配或释放 map 对象本身，因此 map 指针保持稳定。
// 若 map 已满且 key 不存在，返回 -1；成功返回 0。
// 替换已存在的 key 时，会释放旧的指针 value。
int32_t am_map_set_stable(am_allocator_t *alloc, am_map_t *map, am_value_t key, am_value_t value);

// 插入或修改。
// 插入新键值对；若 key 已存在则替换 value，并释放旧的指针 value。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的 map 对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有 map 指针。
am_map_t *am_map_set(am_allocator_t *alloc, am_map_t *map, am_value_t key, am_value_t value);

// 删除指定 key。若存在且 value 为指针则释放。
// 删除成功返回 0；key 不存在返回 -1。
int32_t am_map_delete(am_allocator_t *alloc, am_map_t *map, am_value_t key);

// 当前有效键值对数量
size_t am_map_length(am_allocator_t *alloc, am_map_t *map);

// 物理槽位数
size_t am_map_capacity(am_allocator_t *alloc, am_map_t *map);

// ===============================================================================
// 遍历与键列表
// ===============================================================================

// 遍历所有有效键值对，调用回调 cb
void am_map_iter(am_allocator_t *alloc, am_map_t *map, am_map_iter_callback_t cb, void *user_data);

// 获取所有 key 的副本列表，使用 allocator 分配。
// 调用者负责使用 am_free(alloc, ...) 释放返回的指针；size 为 0 时返回 NULL。
am_value_t *am_map_keys(am_allocator_t *alloc, am_map_t *map);

#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_map.h ===== */

/* ===== begin: include/am_list.h ===== */
#ifndef __AM_LIST_H__
#define __AM_LIST_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


///////////////////////////////////////////
// 线性表及其子类型
///////////////////////////////////////////

// List子类型，决定了编译器和虚拟机如何解释List对象，这是Homoiconicity的基石
#define AM_LIST_TYPE_DEFAULT     (0) // 一般的运行时对象
#define AM_LIST_TYPE_LAMBDA      (1) // TODO lambda的对象布局不做特殊处理，以实现Homoiconicity
#define AM_LIST_TYPE_APPLICATION (2) // 实际等同于AM_SLIST_TYPE_DEFAULT
#define AM_LIST_TYPE_QUOTE       (3)
#define AM_LIST_TYPE_QUASIQUOTE  (4)
#define AM_LIST_TYPE_UNQUOTE     (5)

// 通用线性表（动态扩容）：同时作为基础数据结构和语言数据对象
// NOTE 说明：am_list_t虽然是基础数据结构，但实质上可作为对象语言的数据对象。详见am_map_t的说明。
typedef struct am_list_t {
    am_object_t base;

    size_t      capacity;   // children容量
    size_t      length;     // children元素个数（最后一个元素的下标+1）
    int32_t     type;       // List子类型（AM_LIST_TYPE_*）
    am_handle_t parent;     // 亲list的把柄
    am_value_t  children[]; // Array<am_value_t> 柔性数组
} am_list_t;

// 遍历回调类型
typedef void (*am_list_iter_callback_t)(size_t index, am_value_t item, void *user_data);

// NOTE 动态扩容策略参考
//      cpython：new_allocated = ((size_t)newsize + (newsize >> 3) + 6) & ~(size_t)3;
//      v8：old_capacity * 1.5 + 16


am_list_t *am_list_create(am_allocator_t *alloc, size_t capacity, int32_t type, am_handle_t parent); // V8采用的默认初始容量是4

// 销毁列表。lst 为 NULL 时视为成功。成功返回 0，失败返回 -1。
int32_t am_list_destroy(am_allocator_t *alloc, am_list_t *lst);

am_list_t *am_list_copy(am_allocator_t *alloc, am_list_t *lst);

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_list_size(am_allocator_t *alloc, am_list_t *obj);

void am_list_iter(am_allocator_t *alloc, am_list_t *lst, am_list_iter_callback_t cb, void *user_data);

// 功能说明：将列表对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分。
size_t am_list_dump(am_allocator_t *alloc, am_list_t *lst, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的列表对象，构造列表对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_list_t对象的指针，失败则返回NULL。
am_list_t *am_list_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);


am_value_t am_list_get(am_allocator_t *alloc, am_list_t *lst, size_t index);

// 设置指定下标元素。成功返回 0；lst 为 NULL 或 index 越界返回 -1。
int32_t am_list_set(am_allocator_t *alloc, am_list_t *lst, size_t index, am_value_t item);

am_list_t *am_list_push(am_allocator_t *alloc, am_list_t *lst, am_value_t item); // 带动态扩容

am_value_t am_list_pop(am_allocator_t *alloc, am_list_t *lst);

am_value_t am_list_shift(am_allocator_t *alloc, am_list_t *lst); // 弹出第一个元素并全部左移


// 从from_index开始遍历查找，找到第一个相同的则返回index，不管后面的；没有找到则返回SIZE_MAX
size_t am_list_find(am_allocator_t *alloc, am_list_t *lst, am_value_t item, size_t from_index);





///////////////////////////////////////////
// Lambda表相关函数
///////////////////////////////////////////


// Lambda表结构说明：Lambda表采用形参列表扁平化存储的设计，具体如下。
// children = ['lambda , n_param , param0 , ... , param(n-1) , body0 , ...]
// 
// - length = children项数，等于lambda关键字1项+形参数量字段1项+形参数量+函数体项数
// - children[0] = AM_VALUE_KW_lambda
// - children[1] = am_value_t(满足am_value_is_uint) 引数（形参）数量，记为n
// - children[2 ~ (2+n)] = n个形参的am_varid_t，且都必须为am_varid_n类型
// - children[(2+n) ~ length] = lambda函数体各项的am_value_t
// 
// 例如：(lambda (x y) 666) 对应列表对象的children为：['lambda , 2 , x_varid , y_varid , 666]，因而形参数为2，函数体项数=length-形参数-2=1



// 向Lambda表 增加一个形式参数
// 返回：列表对象指针。若执行成功，则返回原列表指针（无扩容）或新列表指针（有扩容）。若执行失败，则返回NULL作为标记。
// 参数：am_value_t param 必须满足am_value_is_varid(param)
am_list_t *am_list_lambda_add_parameter(am_allocator_t *alloc, am_list_t *lst, am_value_t param);


// 向Lambda表 增加一个函数体
// 返回：列表对象指针。若执行成功，则返回原列表指针（无扩容）或新列表指针（有扩容）。若执行失败，则返回NULL作为标记。
// 参数：am_value_t body
am_list_t *am_list_lambda_add_body(am_allocator_t *alloc, am_list_t *lst, am_value_t body);


// 从Lambda表中 获取函数体数量：根据length、children[1]值和列表结构，计算出函数体数量，并将其值转为size_t
size_t am_list_lambda_get_body_number(am_allocator_t *alloc, am_list_t *lst);


// 从Lambda表中 获取函数体列表和数量
// 返回：函数体am_value_t的数组指针（调用者负责free）
// 参数：am_value_t body
am_value_t *am_list_lambda_get_bodies(am_allocator_t *alloc, am_list_t *lst, size_t *n_body);


// 对Lambda表 重新设置（覆盖）所有的函数体
// 返回：列表对象指针。若执行成功，则返回原列表指针（无扩容）或新列表指针（有扩容）。若执行失败，则返回NULL作为标记。
// 说明：将函数体列表整体替换掉，如果原来的函数体较多，则将多余的旧函数体全部清空，同时保证length字段正确
am_list_t *am_list_lambda_set_bodies(am_allocator_t *alloc, am_list_t *lst, am_value_t *bodies, size_t *n_body);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_list.h ===== */

/* ===== begin: include/am_wstring.h ===== */
#ifndef __AM_WSTRING_H__
#define __AM_WSTRING_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>



// 宽字符串（不可扩容）：同时作为基础数据结构和语言数据对象
// NOTE 说明：虽然是基础数据结构，但实质上可作为对象语言的数据对象。详见am_map_t的说明。
typedef struct am_wstring_t {
    am_object_t base;

    size_t     length;     // content字符个数（最后一个字符的下标+1）=content容量
    am_value_t content[];  // Array<am_value_t(am_wchar_t)> 柔性数组
} am_wstring_t;


// 创建并初始化一个字符串对象。字符串对象是不可变的。
// 注意：am_wstring_t.content是am_value_t数组，每个元素是一个am_wchar_t。
am_wstring_t *am_wstring_create(am_allocator_t *alloc, wchar_t *str, size_t length);

// 销毁对象。obj 为 NULL 时视为成功。成功返回 0，失败返回 -1。
int32_t am_wstring_destroy(am_allocator_t *alloc, am_wstring_t *obj);

// 功能说明：拷贝wstring对象。成功则返回新副本对象的指针，失败则返回NULL。
am_wstring_t *am_wstring_copy(am_allocator_t *alloc, am_wstring_t *obj);

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_wstring_size(am_allocator_t *alloc, am_wstring_t *obj);

// 功能说明：将字符串对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
size_t am_wstring_dump(am_allocator_t *alloc, am_wstring_t *obj, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的字符串对象，构造字符串对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_wstring_t对象的指针，失败则返回NULL。
am_wstring_t *am_wstring_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);








///////////////////////////////////////////
// 多值字符串索引表 am_strindex_t<hash_t, am_value_t>
// 用于全局字符串驻留：一个字符串 hash 可对应多个 candidate handle。
///////////////////////////////////////////

// 多值哈希表特殊 key
#define AM_STRINDEX_KEY_EMPTY     ((uint32_t)UINT32_MAX)
#define AM_STRINDEX_KEY_TOMBSTONE ((uint32_t)(UINT32_MAX - 1))

// 表项
typedef struct am_strindex_entry_t {
    uint32_t   hash;  // 字符串内容的 FNV-1a hash tag
    am_value_t value; // 对应的 handle（或其他 am_value_t）
} am_strindex_entry_t;

// 多值字符串索引表（开放寻址 + 线性探测）
// 作为解释器底层基础设施和语言对象的双重身份，与 am_map_t 一致。
typedef struct am_strindex_t {
    am_object_t base;

    size_t length;     // 当前有效键值对数量
    size_t capacity;   // 物理槽位数（必须是2的幂）
    size_t mask;       // capacity - 1
    size_t tombstones; // 墓碑数量
    am_strindex_entry_t slots[]; // 连续槽位区
} am_strindex_t;

// ===============================================================================
// 哈希函数
// ===============================================================================

// 计算 wchar_t 字符串的 FNV-1a 32-bit 哈希值
uint32_t am_strindex_hash_string(const wchar_t *str);

// ===============================================================================
// 构造函数
// ===============================================================================

// 以初始容量新建多值哈希表。capacity 会被向上取整为不小于它的最小 2 的幂。
// 所有 key 初始化为 AM_STRINDEX_KEY_EMPTY，value 初始化为 AM_VALUE_NULL。
am_strindex_t *am_strindex_create(am_allocator_t *alloc, size_t capacity);

// ===============================================================================
// 析构与清理
// ===============================================================================

// 彻底销毁
int32_t am_strindex_destroy(am_allocator_t *alloc, am_strindex_t *obj);

// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝：创建并返回一个与原 strindex 内容完全一致的新对象。所有 key/value 按位拷贝。
am_strindex_t *am_strindex_copy(am_allocator_t *alloc, am_strindex_t *obj);

// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_strindex_size(am_allocator_t *alloc, am_strindex_t *obj);

// ===============================================================================
// 对象二进制转储
// ===============================================================================

// 功能说明：将表对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，丢弃墓碑和空闲槽位。
size_t am_strindex_dump(am_allocator_t *alloc, am_strindex_t *obj, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的对象，构造对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_strindex_t对象的指针，失败则返回NULL。
am_strindex_t *am_strindex_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);

// ===============================================================================
// 基本操作
// ===============================================================================

// 查找：输入一个wchar_t字符串，计算其uint32_t哈希值，得到所有对应的value的列表（values由调用者管理）。
// values 为 NULL 或 n_values 为 0 时，仅返回匹配条目的数量，不写入 values。
// 返回值为实际匹配条目数量；若不存在则返回 0；若出错则返回 SIZE_MAX。
size_t am_strindex_get_all(am_allocator_t *alloc, am_strindex_t *obj, wchar_t *str, am_value_t *values, size_t n_values);

// 插入新键值对。对输入的字符串计算hash，插入(key=hash,handle)时，直接根据hash找到对应的桶，如果被占用，则往后寻找第一个空桶插入。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有指针。
am_strindex_t *am_strindex_set(am_allocator_t *alloc, am_strindex_t *obj, wchar_t *str, am_value_t value);

// 按已知 hash 直接插入 (hash, value)，不重新计算字符串 hash。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有指针。
am_strindex_t *am_strindex_set_raw(am_allocator_t *alloc, am_strindex_t *obj, uint32_t hash, am_value_t value);

// 删除指定 value（handle）所在的条目。按 value 的位模式精确匹配；删除成功返回 0；未找到返回 -1。
int32_t am_strindex_delete(am_allocator_t *alloc, am_strindex_t *obj, am_value_t value);

// 当前有效键值对数量
size_t am_strindex_length(am_allocator_t *alloc, am_strindex_t *obj);

// 物理槽位数
size_t am_strindex_capacity(am_allocator_t *alloc, am_strindex_t *obj);









#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_wstring.h ===== */

/* ===== begin: include/am_vocab.h ===== */
#ifndef __AM_VOCAB_H__
#define __AM_VOCAB_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>



// 基础数据结构 vocab：词典，即字符串集合，实现为不重复的字符串数组，保证相同的字符串在数组中至多存在一个
// 该数据结构用于编译阶段，记录variable和symbol的集合，并通过递增index为其赋予am_varid_t或am_symbol_t
typedef struct am_vocab_t {
    am_object_t base;

    size_t  capacity; // words数组的容量
    size_t  length;   // words数组实际容纳的元素数
    wchar_t *words[]; // 弹性数组
} am_vocab_t;




// 创建词典对象，其中vocab->words初始化为长度为capacity的全0数组。
am_vocab_t *am_vocab_create(am_allocator_t *alloc,size_t capacity);

// 销毁词典对象，穿透销毁vocab->words的每一项指向的字符串。
// vocab 为 NULL 时视为成功。成功返回 0，失败返回 -1。
int32_t am_vocab_destroy(am_allocator_t *alloc,am_vocab_t *vocab);

// 深拷贝词典对象
am_vocab_t *am_vocab_copy(am_allocator_t *alloc, am_vocab_t *vocab);

// 功能说明：将词典对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       将words所指向的wchar_t*宽字符串依次展平拼接，各字符串之间以L'\0'为间隔符，最后一个字符串以L'\0'结束。
//       压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分。
size_t am_vocab_dump(am_allocator_t *alloc, am_vocab_t *vocab, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的词典对象，构造词典对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_vocab_t对象的指针，失败则返回NULL。
am_vocab_t *am_vocab_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);


// 检查词典中是否存在某个词，返回其在vocab->words中的index。
// 实现提示：将word与vocab->words中的词逐个进行比较，如果有相同的，返回其index；
//           如果遍历完所有words也不存在相同的，返回SIZE_MAX，表示不存在。
size_t am_vocab_find(am_allocator_t *alloc,am_vocab_t *vocab, wchar_t *word);

// 向词典中插入一个词，返回新的容器对象指针；失败返回 NULL。
// 插入的 index 通过 out_index 输出；若 word 已存在，则返回原 vocab 指针并将已有 index 写入 out_index。
// 实现提示：首先检查word是否存在，若存在，则返回原指针并将已有 index 写入 out_index；
//           若不存在，则尝试在vocab->words尾部插入word，成功则更新length字段并将新 index 写入 out_index，返回新的 vocab 指针。
// 注意：插入过程可能触发扩容并重新分配vocab对象本身，因此调用者必须使用返回的指针替换原有容器对象指针。
am_vocab_t *am_vocab_insert(am_allocator_t *alloc, am_vocab_t *vocab, wchar_t *word, size_t *out_index);

// 根据index获取词
// 实现提示：直接返回vocab->words[index]
wchar_t *am_vocab_get(am_allocator_t *alloc,am_vocab_t *vocab, size_t *index);





#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_vocab.h ===== */

/* ===== begin: include/am_heap.h ===== */
#ifndef __AM_HEAP_H__
#define __AM_HEAP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////
// 解释器基础设施：抽象堆
//   抽象堆的实质是 am_handle_t 到 am_value_t 的映射表，以及管理用元数据。
//   抽象堆的功能是 管理把柄（逻辑地址），将逻辑地址与物理地址解耦，使得无论物理地址怎么变化，把柄永远指向同一个对象。
//   抽象堆的归属是 每个进程拥有属于自己的堆实例，以此实现逻辑地址的进程隔离。
//   抽象堆的基础是 RT提供的内存分配器，堆上存储的value的指针指向的都是RT（宿主环境）提供的物理内存。
//
//   从内存管理角度看，am_heap_t 本身（结构体、table、metadata）属于“容器/元数据”，
//   由 container_alloc 管理；table 中存储的指针所指向的用户数据对象，由 obj_alloc 管理。
//   在编译阶段，container_alloc 和 obj_alloc 通常是同一个分配器；
//   在运行时阶段，container_alloc 对应 vm_alloc，obj_alloc 对应 heap_alloc。
///////////////////////////////////////////

// 堆数据结构
typedef struct am_heap_t {
    size_t   capacity; // 当前 table 的物理槽位数（随 table 扩容同步更新）
    am_map_t *table;
    am_map_t *metadata;
    am_handle_t handle_counter; // 简单的自增计数器
} am_heap_t;

// 遍历回调类型
typedef void (*am_heap_iter_callback_t)(am_handle_t handle, am_value_t value, void *user_data);


// 创建堆。container_alloc 管理堆结构本身及其 table/metadata；obj_alloc 管理堆中对象。
am_heap_t *am_heap_create(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, size_t capacity);

int32_t am_heap_destroy(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap);

am_heap_t *am_heap_copy(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap);

void am_heap_iter(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_heap_iter_callback_t cb, void *user_data);

// 功能说明：将am_heap_t对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩底层map对象，将table和metadata的capacity压缩到跟length一致，删除多余分配的空闲部分。
size_t am_heap_dump(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, uint8_t *buffer, size_t offset);

// 功能说明：深度转储整个heap及其指向的对象
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       仅处理value为ptr且指向AM_OBJECT_TYPE_LIST或AM_OBJECT_TYPE_WSTRING类型对象的情况。
size_t am_heap_deep_dump(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, uint8_t *buffer, size_t offset);

// 功能说明：am_heap_dump的逆操作。从二进制字节序列buffer[offset]开始，读取转储的heap对象，构造heap并返回其指针。
// 实现说明：成功则返回加载后am_heap_t对象的指针，失败则返回NULL。
am_heap_t *am_heap_load(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, uint8_t *buffer, size_t offset);

// 功能说明：am_heap_deep_dump的逆操作。从二进制字节序列buffer[offset]开始，读取转储的heap及其指向的对象，构造heap并返回其指针。
// 实现说明：成功则返回加载后am_heap_t对象的指针，失败则返回NULL。
// 注意：仅处理value为ptr且指向AM_OBJECT_TYPE_LIST或AM_OBJECT_TYPE_WSTRING类型对象的情况。
am_heap_t *am_heap_deep_load(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, uint8_t *buffer, size_t offset);



// 存在性检查：存在返回 0，不存在或 heap/table 为空返回 -1。
int32_t am_heap_has_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle);

am_handle_t am_heap_alloc_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap);

// 不仅删除entry，还要穿透free对应的堆对象（被GC调用）。
// 释放成功返回 0；handle 不存在或 heap/table 为空返回 -1。
int32_t am_heap_free_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle);

int32_t am_heap_set_metadata(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle, am_uint_t property); // TODO

am_uint_t am_heap_get_metadata(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle); // TODO


am_value_t am_heap_get(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle);

int32_t am_heap_set(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle, am_value_t value); // 不扩容，且set前检查把柄有效性





#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_heap.h ===== */

/* ===== begin: include/am_closure.h ===== */
#ifndef __AM_CLOSURE_H__
#define __AM_CLOSURE_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////
// 语言数据对象：闭包
///////////////////////////////////////////

// 变量绑定类型（约束绑定、自由绑定）
#define AM_BINDING_BOUND (1)
#define AM_BINDING_FREE  (2)

// 闭包中的变量绑定字段
typedef struct am_binding_t {
    am_varid_t  varid;
    int32_t     type;       // 变量绑定类型（约束绑定、自由绑定）
    int32_t     dirty_flag; // 脏标记
    am_value_t  value;
} am_binding_t;

// 闭包堆对象（变量绑定和元数据用线性表（柔性数组）实现）
typedef struct am_obj_closure_t {
    am_object_t base;

    am_iaddr_t   iaddr;      // 所在call指令的iaddr
    am_handle_t  parent;     // 亲闭包把柄
    size_t       length;     // 指的是bindings的元素个数
    size_t       capacity;   // bindings数组的容量（涉及动态扩容和重新分配）
    am_binding_t bindings[]; // 柔性数组，按顺序逐个插入
} am_obj_closure_t;


// ===============================================================================
// 构造函数
// ===============================================================================

// 创建闭包。capacity 为 0 时默认使用 16。
am_obj_closure_t *am_closure_create(am_allocator_t *alloc, am_iaddr_t iaddr, am_handle_t parent, size_t capacity);


// ===============================================================================
// 析构
// ===============================================================================

// 销毁闭包对象。binding 中的 value 按引用处理，不由闭包释放。
int32_t am_closure_destroy(am_allocator_t *alloc, am_obj_closure_t *closure);


// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝（头部与所有 binding）。value 按位拷贝（与 TS Copy 语义一致，不递归释放对象）。
am_obj_closure_t *am_closure_copy(am_allocator_t *alloc, am_obj_closure_t *closure);


// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_closure_size(am_allocator_t *alloc, am_obj_closure_t *obj);


// ===============================================================================
// 对象二进制转储 TODO
// ===============================================================================

// 功能说明：将闭包对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分。
size_t am_closure_dump(am_allocator_t *alloc, am_obj_closure_t *closure, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的闭包对象，构造闭包对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_obj_closure_t对象的指针，失败则返回NULL。
am_obj_closure_t *am_closure_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);


// ===============================================================================
// 约束变量操作
// ===============================================================================

// 初始化约束变量（不加脏标记）。若已存在则更新 value 并清除脏标记。
// 如涉及扩容，返回新闭包对象指针；否则返回原指针。失败返回 NULL。
am_obj_closure_t *am_closure_init_bound_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable, am_value_t value);

// 设置约束变量（加脏标记，仅用于 set 指令）。若不存在则插入。
// 返回新指针或原指针；失败返回 NULL。
am_obj_closure_t *am_closure_set_bound_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable, am_value_t value);

// 获取约束变量。未找到返回 AM_VALUE_UNDEFINED。
am_value_t am_closure_get_bound_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable);


// ===============================================================================
// 自由变量操作
// ===============================================================================

// 初始化自由变量（不加脏标记）。若已存在则更新 value 并清除脏标记。
am_obj_closure_t *am_closure_init_free_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable, am_value_t value);

// 设置自由变量（加脏标记，仅用于 set 指令）。若不存在则插入。
am_obj_closure_t *am_closure_set_free_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable, am_value_t value);

// 获取自由变量。未找到返回 AM_VALUE_UNDEFINED。
am_value_t am_closure_get_free_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable);


// ===============================================================================
// 查询
// ===============================================================================

// 判断变量是否为脏。为脏返回 0，未找到或不为脏返回 -1。
int32_t am_closure_is_dirty_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable);

// 是否存在约束变量绑定。存在返回 0，不存在返回 -1。
int32_t am_closure_has_bound_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable);

// 是否存在自由变量绑定。存在返回 0，不存在返回 -1。
int32_t am_closure_has_free_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_closure.h ===== */

/* ===== begin: include/am_continuation.h ===== */
#ifndef __AM_CONTINUATION_H__
#define __AM_CONTINUATION_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////
// 语言数据对象：计算续体
///////////////////////////////////////////

// 计算续体（continuation）数据结构，是am_object_t的子类。
// 设计说明：计算续体保存了进程在某一时刻的运行状态快照，包含续体返回iaddr、当前闭包handle、opstack和fstack四个字段。由于opstack和fstack都是朴素数组模拟的栈，且捕获续体后只读不写，故将其展平紧密排列存储到柔性数组stacks中。
// stacks的布局是：[0 ...opstack... (fstack_offset-1)  |  (fstack_offset) ...fstack... (length-1)]
// 即以fstack_offset为界，0<=index<fstack_offset属于opstack，fstack_offset<=index<length属于fstack。index较大的方向是栈顶。
typedef struct am_continuation_t {
    am_object_t base;

    size_t length; // 续体对象stacks字段的长度
    size_t fstack_offset; // stacks数组中，fstack区段起点（栈底）在stacks数组中的offset
    am_iaddr_t cont_return_target;
    am_handle_t current_closure_handle;
    am_handle_t dynamic_wind_stack_handle; // 捕获时刻 dynamic_wind_stack 的深拷贝快照的 handle
    am_handle_t dynamic_wind_after_stack_handle; // 捕获时刻 dynamic_wind_after_stack 的深拷贝快照的 handle
    am_handle_t current_dynamic_wind_entry_handle; // 捕获时刻 proc->current_dynamic_wind_entry
    am_handle_t current_dynamic_wind_thunk_handle; // 捕获时刻 proc->current_dynamic_wind_thunk
    am_value_t stacks[];
} am_continuation_t;


// 构造函数。成功返回指针，失败返回NULL。
am_continuation_t *am_continuation_create(
    am_allocator_t *alloc, am_iaddr_t cont_return_target, am_handle_t current_closure_handle,
    am_value_t *opstack, size_t opstack_length, am_value_t *fstack, size_t fstack_length,
    am_handle_t dynamic_wind_stack_handle);

// 析构函数。成功返回0，失败返回-1
int32_t am_continuation_destroy(am_allocator_t *alloc, am_continuation_t *obj);

// 拷贝
am_continuation_t *am_continuation_copy(am_allocator_t *alloc, am_continuation_t *obj);

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_continuation_size(am_allocator_t *alloc, am_continuation_t *obj);

// 获取opstack数组，用于GC遍历和续体恢复。成功返回新数组指针（通过alloc分配，由调用者负责释放），失败返回NULL。
am_value_t *am_continuation_get_opstack(am_allocator_t *alloc, am_continuation_t *obj, size_t *length);

// 获取fstack数组，用于GC遍历和续体恢复。成功返回新数组指针（通过alloc分配，由调用者负责释放），失败返回NULL。
am_value_t *am_continuation_get_fstack(am_allocator_t *alloc, am_continuation_t *obj, size_t *length);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_continuation.h ===== */

/* ===== begin: include/am_scope.h ===== */
#ifndef __AM_SCOPE_H__
#define __AM_SCOPE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>



///////////////////////////////////////////
// 词法作用域对象（环境帧）
///////////////////////////////////////////

// 词法作用域中的变量绑定字段
typedef struct am_scope_binding_t {
    am_varid_t  varid;
    am_value_t  value;
} am_scope_binding_t;

// 环境帧（支持扩容）
typedef struct am_scope_t {
    am_object_t base;

    am_handle_t parent_scope_handle;
    am_handle_t parent_lambda_handle;
    am_handle_t current_lambda_handle;
    size_t capacity;
    size_t length;
    am_scope_binding_t bindings[]; // 柔性数组，按顺序逐个插入
} am_scope_t;

// 创建环境帧。capacity 为 0 时默认使用 16。
am_scope_t *am_scope_create(am_allocator_t *alloc, am_handle_t parent_scope_handle, am_handle_t parent_lambda_handle, am_handle_t current_lambda_handle, size_t capacity);

// 销毁环境帧。scope 为 NULL 时视为成功。成功返回 0，失败返回 -1。
int32_t am_scope_destroy(am_allocator_t *alloc, am_scope_t *scope);

// 深拷贝（头部与所有 binding）。value 按位拷贝（与 TS Copy 语义一致，不递归释放对象）。
am_scope_t *am_scope_copy(am_allocator_t *alloc, am_scope_t *scope);

// 将对象的二进制内存布局从alloc管理的内存中倒出来，返回一个系统malloc的二进制序列，以及序列长度
//   注意：压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分
uint8_t *am_scope_dump(am_allocator_t *alloc, am_scope_t *scope, size_t *size);

// 查询是否存在变量绑定。存在返回 0，不存在或 scope 为 NULL 返回 -1。
int32_t am_scope_has_var(am_allocator_t *alloc, am_scope_t *scope, am_varid_t variable);

// 新增一个变量绑定。若已存在相同变量，则返回NULL表示失败。
// 如涉及扩容，返回新闭包对象指针；否则返回原指针。扩容失败返回NULL。
am_scope_t *am_scope_add_var(am_allocator_t *alloc, am_scope_t *scope, am_varid_t variable, am_value_t value);





#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_scope.h ===== */

/* ===== begin: include/am_lexer.h ===== */
#ifndef __AM_LEXER_H__
#define __AM_LEXER_H__

#include <stdint.h>
#include <wchar.h>
#include <wctype.h>
#include <string.h>


#ifdef __cplusplus
extern "C" {
#endif

/* ===== Token类型定义 ===== */
#define AM_TOKEN_TYPE_DELIMITER  (0)   // 分隔符: 空白字符
#define AM_TOKEN_TYPE_LB         (1)   // 左括号: ( [ {
#define AM_TOKEN_TYPE_RB         (2)   // 右括号: ) ] }
#define AM_TOKEN_TYPE_KEYWORD    (3)   // 关键字: define if cond while lambda begin 等
#define AM_TOKEN_TYPE_BOOLEAN    (4)   // 字面值：#t #f
#define AM_TOKEN_TYPE_UNDEFINED  (5)   // 字面值：#undefined
#define AM_TOKEN_TYPE_NULL       (6)   // 字面值：#null
#define AM_TOKEN_TYPE_NUMBER     (7)   // 字面值：数字，如 -3.14 +12.3 2e-5 等
#define AM_TOKEN_TYPE_SYMBOL     (8)   // 字面值：符号，即单撇号开头的符号，如 'symbol 等
#define AM_TOKEN_TYPE_IDENTIFIER (9)   // 标识符（变量、运算符等）
#define AM_TOKEN_TYPE_STRING     (10)  // 字符串: "hello"
#define AM_TOKEN_TYPE_QUOTE      (11)  // 出现在括号前面的单引号'
#define AM_TOKEN_TYPE_QUASIQUOTE (12)  // 反引号`
#define AM_TOKEN_TYPE_UNQUOTE    (13)  // 逗号,
#define AM_TOKEN_TYPE_UNEXPECTED (99)  // 意料之外的token

typedef struct am_token_t {
    am_object_t base;

    size_t  index;   // token首字符在code中的偏移
    size_t  length;  // token长度(字符数)
    int32_t type;    // token类型
    int32_t line;    // 行号(从1开始)
    int32_t column;  // 列号(从0开始)
    size_t  id;      // 如果是 AM_TOKEN_TYPE_SYMBOL 或 AM_TOKEN_TYPE_IDENTIFIER，记录其在编译时分配到的 am_symbol_t 或 am_varid_t
} am_token_t;

// 关键字
#define AM_KEYWORDS_NUM (28)
extern const wchar_t* AM_KEYWORDS[];


/* ===== 通用辅助函数 ===== */

// 定界符判断
static inline int is_delimiter(wchar_t c) {
    return c == L'(' || c == L')' || c == L'[' || c == L']' ||
           c == L'{' || c == L'}' || c == L'"' || c == L'`' || c == L',';
}

static inline int is_whitespace(wchar_t c) {
    return c == L' ' || c == L'\t' || c == L'\n' || c == L'\r';
}

static inline int is_escaped(wchar_t *code, int32_t pos) {
    if(pos <= 0) return 0;
    int32_t count = 0, i = pos - 1;
    while(i >= 0 && code[i--] == L'\\') count++;
    return (count & 1);
}

// 增强：数字字符判断（支持科学计数法）
static inline int is_num_char(wchar_t c) {
    return iswdigit(c) || c == L'.' || c == L'e' || c == L'E' ||
           c == L'+' || c == L'-';
}

/* ===== 主Lexer函数 ===== */

// 对 code 进行词法分析，结果写入 tokens 数组。
// 返回 token 数量；出错返回 -1。
int32_t am_lexer(wchar_t *code, am_token_t *tokens);

// 安全获取 token 文本（处理虚拟 token）。
// 注意：返回指向静态缓冲区的指针，非线程安全。
const wchar_t* token_text(am_token_t *tok, wchar_t *code);

#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_lexer.h ===== */

/* ===== begin: include/am_ast.h ===== */
#ifndef __AM_AST_H__
#define __AM_AST_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


// VM指令集（操作码）枚举

#define AM_VM_OP_nop         (0)
#define AM_VM_OP_store       (1)
#define AM_VM_OP_load        (2)
#define AM_VM_OP_loadclosure (3)
#define AM_VM_OP_push        (4)
#define AM_VM_OP_pop         (5)
#define AM_VM_OP_swap        (6)
#define AM_VM_OP_set         (7)
#define AM_VM_OP_call        (8)
#define AM_VM_OP_callnative  (9)
#define AM_VM_OP_tailcall    (10)
#define AM_VM_OP_return      (11)
#define AM_VM_OP_capturecc   (12)
#define AM_VM_OP_iftrue      (13)
#define AM_VM_OP_iffalse     (14)
#define AM_VM_OP_goto        (15)
#define AM_VM_OP_read        (16)
#define AM_VM_OP_write       (17)
#define AM_VM_OP_pause       (18)
#define AM_VM_OP_halt        (19)
#define AM_VM_OP_fork        (20)
#define AM_VM_OP_display     (21)
#define AM_VM_OP_newline     (22)
#define AM_VM_OP_add         (23)
#define AM_VM_OP_sub         (24)
#define AM_VM_OP_mul         (25)
#define AM_VM_OP_div         (26)
#define AM_VM_OP_mod         (27)
#define AM_VM_OP_pow         (28)
#define AM_VM_OP_eq          (29)
#define AM_VM_OP_eqv         (30)
#define AM_VM_OP_equal       (31)
#define AM_VM_OP_ge          (32)
#define AM_VM_OP_le          (33)
#define AM_VM_OP_gt          (34)
#define AM_VM_OP_lt          (35)
#define AM_VM_OP_not         (36)
#define AM_VM_OP_and         (37)
#define AM_VM_OP_or          (38)
#define AM_VM_OP_isnull      (39)
#define AM_VM_OP_isundef     (40)
#define AM_VM_OP_isatom      (41)
#define AM_VM_OP_islist      (42)
#define AM_VM_OP_isnumber    (43)
#define AM_VM_OP_isnan       (44)
#define AM_VM_OP_typeof      (45)
#define AM_VM_OP_car         (46)
#define AM_VM_OP_cdr         (47)
#define AM_VM_OP_cons        (48)
#define AM_VM_OP_get_item    (49)
#define AM_VM_OP_set_item    (50)
#define AM_VM_OP_list_push   (51)
#define AM_VM_OP_list_pop    (52)
#define AM_VM_OP_length      (53)
#define AM_VM_OP_concat      (54)
#define AM_VM_OP_duplicate   (55)
#define AM_VM_OP_evalcleanup (56)
#define AM_VM_OP_dynamicwind              (57)
#define AM_VM_OP_dynamicwind_after_before (58)
#define AM_VM_OP_dynamicwind_before_after (59)
#define AM_VM_OP_dynamicwind_done         (60)
#define AM_VM_OP_wind                     (61)


// 全局内置变量
#define AM_GLOBAL_BUILTIN_VAR_NUM (37)
extern const wchar_t* AM_GLOBAL_BUILTIN_VAR[];

// 全局内置变量到 VM opcode 的映射表。
// 下标与 AM_GLOBAL_BUILTIN_VAR 一一对应；-1 表示该 builtin 没有对应 opcode。
extern const int32_t AM_BUILTIN_OPCODE_MAP[AM_GLOBAL_BUILTIN_VAR_NUM];


// 顶级词法节点、顶级作用域和顶级闭包的parent字段，用于判断上溯结束
// 注意其类型为 am_handle_t，不要与 am_value_t AM_VALUE_HANDLE_NULL 混淆
#define AM_TOP_NODE_HANDLE AM_HANDLE_NULL


// ast.var_type的值
#define AM_VAR_TYPE_OLD          (0) // 默认：普通变量（ARN换名前）
#define AM_VAR_TYPE_NEW          (1) // 普通变量（ARN换名后）
#define AM_VAR_TYPE_BUILTIN      (2) // 全局内置符号（不ARN）
#define AM_VAR_TYPE_IMPORT_REF   (3) // 导入模块符号引用（不ARN）：点号分隔的引用了外部import模块的变量，例如Mod.foo
#define AM_VAR_TYPE_NATIVE_REF   (4) // 本地模块符号引用（不ARN）：点号分隔的对native函数的调用，例如"Math.exp"
#define AM_VAR_TYPE_EXT_REF     (34) // 点号分割形式：实际上就是 AM_VAR_TYPE_IMPORT_REF 或者 AM_VAR_TYPE_NATIVE_REF，用于暂时无法确定是哪种的情况
#define AM_VAR_TYPE_IMPORT_ALIAS (5) // 导入模块的别名（不ARN）：也就是(import Mod "mod.scm")中的Mod
#define AM_VAR_TYPE_NATIVE_ID    (6) // 本地模块名（不ARN）：也就是(native Math)中的Math
#define AM_VAR_TYPE_ILTEMP       (7) // 编译过程引入的临时中间变量，AST中不存在
#define AM_VAR_TYPE_GLOBAL_FREE  (8) // 用于eval：全局无所属作用域的自由变量，普通代码属于错误，但evalee中应特殊处理


// AST数据结构
typedef struct am_ast_t {
    wchar_t *absolute_path;      // 模块代码文件所在的文件系统绝对路径
    wchar_t *module_id;          // 模块ID，从absolute_path转换而来

    wchar_t *code;               // 一切字符串的总源头
    am_token_t *tokens;          // Lexer输出的token列表
    size_t token_count;          // token数量

    am_vocab_t *symbol_vocab;    // 保存所有的symbol字符串集合，以其index为am_symbol_t
    am_vocab_t *var_vocab;       // 保存所有的变量字符串集合，以其index为am_varid_t
    am_list_t  *var_type;        // 记录每个变量的类型（取值为AM_VAR_TYPE_*），其index即为var_vocab的index

    am_allocator_t *alloc;       // 编译阶段AST专用的内存分配器
    am_heap_t *nodes;            // AST临时堆，保存编译阶段所有数据对象（包括SList也就是AST节点、词法作用域、var/sym表等）的临时堆，它们之间都是通过handle互相引用，建立起树结构
    am_map_t *node_token_mapping; // 记录AST节点把柄与token索引的映射关系（对应TS的nodeIndexes）
    am_strindex_t *strindex;     // 用于全局字符串驻留的多值哈希表，检查某个字符串（的哈希值）是否已存在于nodes

    am_map_t *scopes;            // 词法作用域：Map<handle(lambda), handle(scope)>
    am_map_t *var_arn_mapping;   // 变量ARN（Alpha-renaming）前后的映射：Map<varid, varid>，key是ARN后的新varid，value是ARN前的旧varid

    am_handle_t top_lambda_handle; // 最顶层lambda节点的把柄（通过am_ast_get_top_lambda_node_handle计算）
    am_list_t *lambda_handles;   // 记录所有的lambda节点的把柄（对应TS的lambdaHandles）
    am_list_t *tailcall_handles; // 记录所有的尾调用节点的把柄（对应TS的tailcall）
    am_list_t *var_top;          // 顶级变量varid列表（即顶层作用域define的变量）（对应TS的topVariables）
    am_map_t *dependencies;      // 依赖模块记录：Map<varid, handle>（对应TS的dependencies）根据(import mod_alias "path/to/mod.scm")记录
    am_map_t *natives;           // 本地库记录：Map<varid, handle>（对应TS的natives）根据(native Math)记录，其中handle可暂时设置为AM_VALUE_HANDLE_NULL备用

    size_t opstack_depth;        // 静态分析得到的最大opstack栈深度（在link后最后分析）
} am_ast_t;


// 功能描述：创建AST对象。调用者保留code、absolute_path、tokens的所有权，AST只保存指针。
// 实现说明：成功返回AST指针，失败返回NULL。
am_ast_t *am_ast_create(am_allocator_t *alloc, wchar_t *code, wchar_t *absolute_path, am_token_t *tokens, size_t token_count);


// 功能描述：销毁AST对象，释放AST自身及其内部所有堆对象。
// 实现说明：成功返回0，失败返回-1。注意不释放调用者传入的code、absolute_path、tokens。
int32_t am_ast_destroy(am_ast_t *ast);


// 功能描述：深拷贝AST对象（对应TS的AST.Copy）
// 实现说明：创建新的AST，深拷贝所有内部集合和堆对象。code、absolute_path、tokens与源AST共享指针。
am_ast_t *am_ast_copy(am_ast_t *ast);


// 功能描述：设置AST节点把柄对应的token索引。
// 实现说明：成功返回0，失败返回-1。
int32_t am_ast_set_node_token_index(am_ast_t *ast, am_handle_t node_handle, size_t token_index);


// 功能描述：获取AST节点把柄对应的token索引（对应TS的nodeIndexes.get）。
// 实现说明：若不存在，返回SIZE_MAX。
size_t am_ast_get_node_token_index(am_ast_t *ast, am_handle_t node_handle);


// 功能描述：将importee融合进importer，也就是importer吃掉importee。
// 实现说明：成功返回0；失败返回-1。
int32_t am_ast_merge(am_ast_t *importer, am_ast_t *importee, int32_t order);


// 功能描述：遍历tokens，使用其中的KEYWORD和SYMBOL构建ast->symbol_vocab，同时等于是注册了am_symbol_t，并将am_symbol_t记录在token中
// 实现说明：返回symbol总数。注意将object.h中定义的24个Keyword置于词典的前24个条目。
size_t am_build_symbol_vocabulary(am_ast_t *ast);


// 功能描述：遍历tokens，使用其中的VARIABLE构建ast->var_vocab，同时等于是注册了am_varid_t，并将varid记录在token中。
// 实现说明：返回varid总数。
size_t am_build_variable_vocabulary(am_ast_t *ast);




// 功能描述：判断某个变量在形式上是否是“前缀.后缀”的格式（统称为EXT_REF，外部引用格式），是返回0，否返回-1
// 设计说明：parse和ARN阶段，这种形式的变量可能是AM_VAR_TYPE_IMPORT_REF或AM_VAR_TYPE_NATIVE_REF，保留原形，不参与ARN。
// 实现说明：(varid)--[ast->var_vocab]-->var_str-->判断其是否是被唯一点号分成两部分的形式（只有一个“.”，且不在开头和末尾）
int32_t am_ast_check_ext_ref(am_ast_t *ast, am_varid_t v);


// 功能描述：判断某个变量是否是AM_VAR_TYPE_NATIVE_REF，也就是对本地宿主库native的调用，是返回0，否返回-1（对应TS的AST.IsNativeCall）
// 实现说明：(varid)(t.id)--[ast->var_vocab]-->var_str-->提取点号分隔的第1部分--[ast->var_vocab]-->native_varid--[ast->natives]-->是否存在
int32_t am_ast_check_native_ref(am_ast_t *ast, am_varid_t v);


// 功能描述：判断某个变量是否是AM_VAR_TYPE_IMPORT_REF，即导入模块的外部引用（“别名.标识符”的格式），是返回0，否返回-1
// 设计说明：外部引用：指的是通过import和点号分隔标识符，引用外部模块变量。(import Alias "/path/to/module.scm")表达式，声明对外部模块的导入，并赋予其“别名”Alias，别名属于特殊变量，其类型为AM_VAR_TYPE_IMPORT_ALIAS。代码中通过“别名.标识符”的格式，引用外部模块的变量。“别名.标识符”整体也是一个变量，在parse阶段，其类型为AM_VAR_TYPE_IMPORT_REF。
// 实现说明：(varid)--[ast->var_vocab]-->var_str-->提取最后一个点号分隔的第1部分--[ast->var_vocab]-->alias_varid--[ast->dependencies]-->是否存在
int32_t am_ast_check_import_ref(am_ast_t *ast, am_varid_t v);


// 功能描述：根据把柄，从AST->nodes堆中获取相应的am_value_t（由调用者解包并使用）（对应TS的AST.GetNode）
// 设计说明：解释器的值-对象映射机制比较复杂。就本函数来说，根据handle从nodes堆中获得了am_value_t，这是一个打包后的值（因为只有打包后才能装进map、list等容器）。调用者通过该函数获得了am_value_t之后，应当自行判断其类型并解包。例如，如果调用者期望通过handle从nodes堆中获得一个am_object_t的指针，则通过本函数获得am_value_t后，将其作为am_object_t*也就是AM_VALUE_TYPE_PTR进行解包（使用am_value_to_ptr），这样就可以通过解包得到的ptr直接在C语言层面访问allocator管理的内存中（也就是被AST->nodes堆所封装起来的内存）的object对象。
am_value_t am_ast_get_node(am_ast_t *ast, am_handle_t handle);


// 功能描述：创建lambda对象，返回其在AST->nodes堆中的把柄（对应TS的AST.MakeLambdaNode）
// 实现说明：先从heap中申请一个把柄，再创建一个类型为AM_LIST_TYPE_LAMBDA的am_obj_list_t对象，以32为初始容量，再将对象指针打包成am_value_t与已分配把柄绑定在一起。同时，在ast->lambda_handles中登记这个把柄。最后返回把柄。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_make_lambda_node(am_ast_t *ast, am_handle_t parent);


// 功能描述：创建SList对象，返回其在AST->nodes堆中的把柄（对应TS的AST.MakeApplicationNode）
// 实现说明：先从heap中申请一个把柄，再创建一个类型为type=AM_LIST_TYPE_APPLICATION/AM_LIST_TYPE_QUOTE/AM_LIST_TYPE_QUASIQUOTE/AM_LIST_TYPE_UNQUOTE的am_obj_list_t对象，以32为初始容量，再将对象指针打包成am_value_t与已分配把柄绑定在一起。最后返回把柄。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_make_slist_node(am_ast_t *ast, am_handle_t parent, int32_t type);


// 功能描述：创建WString对象，返回其在AST->nodes堆中的把柄（对应TS的AST.MakeStringNode）
// 实现说明：先从AST->nodes中申请一个把柄，再根据AM_TOKEN_TYPE_STRING类型的am_token_t t 所表示的字符串（注意：可以根据其指示的index和length从ast->code中获取），创建一个am_obj_wstring_t对象，再将对象指针打包成am_value_t与已分配把柄绑定在一起。最后返回把柄。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_make_wstring_node(am_ast_t *ast, am_token_t *str_token);


// 功能描述：查找AST->nodes堆中最顶级am_obj_list_t对象的handle，也就是parent字段为AM_HANDLE_NULL的am_obj_list_t对象。（对应TS的AST.TopApplicationNodeHandle）
// 设计说明：根据编译器的约定，合法Scheme代码的顶层结构应当是一个thunk的调用，即((lambda () ...))，这个函数就是用来获取这个顶层APPLICATION的。
// 实现说明：如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_get_top_node_handle(am_ast_t *ast);


// 功能描述：查找AST->nodes堆中顶级am_obj_list_t（Lambda）对象的handle，也就是最顶级application list对象的第一个child。（对应TS的AST.TopLambdaNodeHandle）
// 设计说明：根据编译器的约定，合法Scheme代码的顶层结构应当是一个thunk的调用，即((lambda () ...))，这个函数就是用来获取这个顶层APPLICATION的第一个child也就是顶层lambda（thunk）的。
// 实现说明：如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_get_top_lambda_node_handle(am_ast_t *ast);


// 功能描述：获取位于全局作用域的node列表（也就是函数体列表）。（对应TS的AST.GetGlobalNodes）
// 设计说明：取am_ast_get_top_lambda_node_handle也就是顶层lambda（thunk）的bodies，返回一个am_value_t的数组，由调用者负责解包、解释、释放。
// 实现说明：如有异常情况，返回NULL，以示失败。
am_value_t *am_ast_get_global_nodes(am_ast_t *ast);



// 功能描述：设置全局作用域（顶层lambda）的node列表（也就是函数体列表）。（对应TS的AST.SetGlobalNodes）
// 设计说明：用bodies整体替换am_ast_get_top_lambda_node_handle也就是顶层lambda（thunk）的bodies。
// 实现说明：通过am_list_lambda_set_bodies实现，这个过程可能涉及lambda对象指针的变化，如有变化，则更新AST->nodes中对应handle的值（打包成am_value_t的）。n_body为bodies数组的长度。如有扩容失败等异常情况，返回-1。执行成功则返回0。
int32_t am_ast_set_global_nodes(am_ast_t *ast, am_value_t *bodies, size_t n_body);




// 功能描述：从某个节点开始，向上上溯查找某个varid归属的lambda节点把柄，也就是该varid作为哪个lambda节点的parameter（对应TS的Analyser.ts中的searchVarLambdaHandle）
// 设计说明：该函数用于“变量换名”过程，旨在寻找其最近上级lambda节点的把柄，进而确定其所在的词法作用域。该函数依赖于AST第一趟扫描“作用域分析”的结果。
// 实现说明：该函数的输入是变量换名前的varid，以及上溯查找起点节点的handle。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_find_var_lambda_handle(am_ast_t *ast, am_varid_t varid, am_handle_t from_node_handle);


// 功能描述：从某个节点开始，向上上溯查找最近的lambda节点的把柄（对应TS的Analyser.ts中的nearestLambdaHandle）
// 设计说明：该函数用于确定某个节点最近上级lambda节点的把柄，进而确定其所在的词法作用域。
// 实现说明：该函数的输入是上溯查找起点节点的handle。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_find_nearest_lambda_handle(am_ast_t *ast, am_handle_t from_node_handle);


// 功能表述：生成模块（AST）内唯一的变量名（对应TS的Analyser.ts中的MakeUniqueVariable）
// 设计说明：该函数用于“变量换名”阶段，用于生成携带作用域信息的、全局唯一的变量名，并将其新增注册到ast->var_vocab中。
// 实现说明：基于varid和所在lambda节点的handle，生成一个新的变量名字符串。规则是："V.module_id.lambda_handle.var_string"，并将其新增注册到ast->var_vocab，返回值是新变量名的ast->var_vocab的index。如有异常情况，返回SIZE_MAX，以示失败。
am_varid_t am_ast_make_unique_variable(am_ast_t *ast, am_varid_t varid, am_handle_t lambda_handle);


// 功能描述：为 import 别名生成模块级唯一变量名（module_id.alias），并将其注册到 ast->var_vocab，同时设置其 var_type 为 AM_VAR_TYPE_IMPORT_ALIAS。
// 实现说明：成功返回新的 varid，失败返回 SIZE_MAX。
am_varid_t am_ast_make_unique_module_alias(am_ast_t *ast, am_varid_t alias_varid);


// 功能描述：为 import 外部引用生成模块级唯一变量名（module_id.import_ref），并将其注册到 ast->var_vocab，同时设置其 var_type 为 AM_VAR_TYPE_IMPORT_REF。
// 实现说明：成功返回新的 varid，失败返回 SIZE_MAX。
am_varid_t am_ast_make_unique_import_ref(am_ast_t *ast, am_varid_t import_ref_varid);


// 功能描述：向 tailcall_handles 中添加一个尾调用节点把柄。
// 实现说明：成功返回0，失败返回-1。
int32_t am_ast_add_tailcall(am_ast_t *ast, am_handle_t handle);


// 功能描述：向 var_top 中添加一个顶级变量 varid。
// 实现说明：成功返回0，失败返回-1。
int32_t am_ast_add_var_top(am_ast_t *ast, am_varid_t varid);


// 功能描述：设置依赖模块记录。
// 实现说明：alias_varid 为 import 语句中模块别名对应的 varid；path_handle 为模块路径字符串节点在 ast->nodes 中的把柄。成功返回0，失败返回-1。
int32_t am_ast_set_dependency(am_ast_t *ast, am_varid_t alias_varid, am_handle_t path_handle);


// 功能描述：设置本地库记录。
// 实现说明：native_varid 为 native 语句中库名对应的 varid；handle 可暂时设置为 AM_VALUE_HANDLE_NULL。成功返回0，失败返回-1。
int32_t am_ast_set_native(am_ast_t *ast, am_varid_t native_varid, am_handle_t handle);


// 功能描述：为lambda节点设置对应的词法作用域把柄。
// 实现说明：成功返回0，失败返回-1。
int32_t am_ast_set_scope(am_ast_t *ast, am_handle_t lambda_handle, am_handle_t scope_handle);


// 功能描述：获取lambda节点对应的词法作用域把柄。
// 实现说明：若不存在，返回 AM_HANDLE_NULL。
am_handle_t am_ast_get_scope(am_ast_t *ast, am_handle_t lambda_handle);








// 功能描述：将模块绝对路径转换为模块ID。
// 实现说明：规则见 AGENTS.md。返回使用 ast 分配器分配的 wchar_t*，失败返回 NULL。
wchar_t *am_absolute_path_to_module_id(am_allocator_t *alloc, const wchar_t *absolute_path);


// 功能描述：将AST中的某个节点转成Scheme代码字符串（对应TS的AST.NodeToString）。
// 实现说明：返回使用 alloc 分配器分配的以 L'\0' 结尾的宽字符串，失败返回 NULL。
//         若 length 不为 NULL，则将字符串的逻辑长度（字符数）写入 *length。
wchar_t *am_ast_node_to_string(am_allocator_t *alloc, am_ast_t *ast, am_handle_t node_handle, size_t *length);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_ast.h ===== */

/* ===== begin: include/am_parser.h ===== */
#ifndef __AM_PARSER_H__
#define __AM_PARSER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <wchar.h>




/************

每个源码文件就是一个模块。
Parser的输入是模块，输出是模块的AST。
Linker将所有的AST链接成一个大AST。
Compiler将AST编译成ILCode，并封装成Module。



Parser需要维护的全局信息：
- am_ast_t AST（含tokens）
- List<am_value_t> node_stack; // 其元素是am_value_t，语义上可能是ptr、也可能是包括handle在内的imme，由使用者解释
- List<uint> state_stack;


# Parser：语法分析器（LL(1)递归下降分析器）设计原理说明

Parser的输入是Lexer输出的token序列，基于以下BNF文法对token序列进行解析，输出初步生成的AST。

输入代码必须是`((lambda () <code>))`格式。

BNF：
    <SourceCode> ::= (lambda () <TERM>*) CRLF
          <Term> ::= <SList> | <Lambda> | <Quote> | <Unquote> | <Quasiquote> | <Identifier>
         <SList> ::= ( <SListSeq> )
      <SListSeq> ::= <Term> <SListSeq> | ε
        <Lambda> ::= ( lambda <ArgList> <Body> )
       <ArgList> ::= ( <ArgListSeq> )
    <ArgListSeq> ::= <ArgIdentifier> <ArgListSeq> | ε
 <ArgIdentifier> ::= <Identifier>
          <Body> ::= <BodyTerm> <Body_>
         <Body_> ::= <BodyTerm> <Body_> | ε
      <BodyTerm> ::= <Term>
         <Quote> ::= ' <QuoteTerm> | ( quote <QuoteTerm> )
       <Unquote> ::= , <UnquoteTerm> | ( unquote <QuoteTerm> )
    <Quasiquote> ::= ` <QuasiquoteTerm> | ( quasiquote <QuoteTerm> )
     <QuoteTerm> ::= <Term>
   <UnquoteTerm> ::= <Term>
<QuasiquoteTerm> ::= <Term>
    <Identifier> ::= IDENTIFIER

# Analyser：AST作用域分析器设计原理说明

Analyser需要对AST做两趟扫描。分别是“词法作用域分析”和“变量换名”。经过这两趟扫描，原有AST被更新，同时得到作用域信息。

例如：以下Scheme代码中，外层x和内层x实际上是两个不同的变量

(define foo (lambda (x y) 
  (define bar (lambda (x) (+ x y)))
  (bar y)
))

语法分析Parse，得到AST之后，得到以下var_vocab：[0:foo, 1:x, 2:y, 3:bar]。随后对AST进行分析。

第一趟扫描“词法作用域分析”，发现两个作用域，分别是Scope-foo[x(1),y(2),bar(3)]和Scope-bar[x(1)]，此时尽管两个作用域都有同名的x（进而varid也相同，都是1），但如果某个x沿着作用域链上溯，第一个找到的x就是正确的约束变量，这说明分散在不同的scope中的同名x已经是词法作用域意义上不同的x了。因此，在现有的携带环境帧的带脏标记的闭包实现中，必须在编译阶段就将这两个x区分成两个不同的varid。此外，在这趟扫描中，也处理了作用域内的define的变量，这实质上是作用域范围内的、出现在lambda参数列表之外的一种全局绑定，几乎等同于JavaScript的var变量声明，或者约等于标准Scheme的letrec*。

第二趟扫描“变量换名”，将AST中所有的varid，根据其所在的scope，替换成全局唯一的、携带了词法作用域信息的、新的varid。例如，换名结果如下：

(define mod.0.foo (lambda (mod.0.x mod.0.y) 
  (define mod.0.bar (lambda (mod.1.x) (+ mod.1.x mod.0.y)))
  (mod.0.bar mod.0.y)
))

同时在var_vocab中追加[... 10:mod.0.foo , 11:mod.0.x , 12:mod.0.y , 13:mod.0.bar , 14:mod.1.x]。这样就实现了所有变量都可以通过其varid唯一确定其scope，而不致混淆。


# 实现说明

- 调用 am_parse(code, absolute_path) 即可完成词法分析、词汇表构建、语法分析和预处理指令解析。
- 返回的 am_ast_t 由调用者负责销毁。
- 若解析失败，返回 NULL。

************/


// 语法分析器入口。
// 输入：内存分配器 alloc、Scheme 源码 code、模块绝对路径 absolute_path、is_keep_free。
//       当 is_keep_free 为 0 时，保留现有逻辑；为 1 时，在 alpha-renaming 阶段将“未定义变量”的 var_type 设为 AM_VAR_TYPE_GLOBAL_FREE。
// 输出：解析得到的 AST；失败返回 NULL。
// 说明：code 与 absolute_path 由调用者所有；tokens 由返回的 AST 所有，随 AST 销毁而释放。
am_ast_t *am_parse(am_allocator_t *alloc, wchar_t *code, wchar_t *absolute_path, int32_t is_keep_free);


// 对 AST 执行整体的尾位置分析，将处于尾位置的 application 节点把柄记录到 ast->tailcall_handles。
// 通常在 am_link 完成所有模块合并后调用；也可在独立使用 am_parser 后手动调用。
// 调用前会清空已有的 tailcall_handles。
// 成功返回 0，失败返回 -1。
int32_t am_parser_tail_call_analysis(am_ast_t *ast);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_parser.h ===== */

/* ===== begin: include/am_macro.h ===== */
#ifndef __AM_MACRO_H__
#define __AM_MACRO_H__

#ifdef __cplusplus
extern "C" {
#endif


// 功能描述：对 AST 执行 syntax-rules 卫生宏展开。
// 设计说明：该函数在 Alpha-renaming 之后、清理 scope 对象之前调用，
//         将 define-syntax / let-syntax / letrec-syntax 定义的宏展开为
//         普通 AST 节点。展开完成后会自动重建 lambda_handles 和 var_top
//         等元数据；tailcall_handles 由 am_parse / am_link 的尾位置分析统一重建。
// 实现说明：成功返回 0；失败返回 -1，并在 stderr 输出错误信息。
int32_t am_macro_expand(am_ast_t *ast);

#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_macro.h ===== */

/* ===== begin: include/am_linker.h ===== */
#ifndef __AM_LINKER_H__
#define __AM_LINKER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <wchar.h>



// 模块源码读取回调类型。由调用方（宿主）注入，使链接器与具体的源码获取方式（文件系统、
// Flash、网络、内存表等）解耦，实现依赖倒置。
// 参数说明：alloc 为链接器使用的分配器，回调必须用它分配返回的缓冲区（链接器将用 am_free 释放）；
//          abs_path 为链接器已解析出的模块绝对路径（宽字符）；
//          user_data 为调用方透传的上下文指针。
// 返回值：  成功返回以 L'\0' 结尾的模块源码字符串；失败（读取不到、分配失败等）返回 NULL。
typedef wchar_t *(*am_linker_read_source_fn)(am_allocator_t *alloc, const wchar_t *abs_path, void *user_data);


// 功能描述：链接器入口。从 main_ast 出发，递归解析所有依赖模块，按拓扑顺序合并成一个大 AST。
// 参数说明：main_ast 为引用根模块的 AST；base_dir 为基准工作目录（用于解析相对路径 import）；
//          read_source 为模块源码读取回调（不可为 NULL）；user_data 透传给 read_source。
// 返回值：  成功返回链接后的 AST（即基于 main_ast 修改后的 AST）；失败返回 NULL。
am_ast_t *am_link(am_ast_t *main_ast, wchar_t *base_dir,
                  am_linker_read_source_fn read_source, void *user_data);


// 前向声明：链接器上下文（opaque pointer）
struct am_linker_ctx_t;
typedef struct am_linker_ctx_t am_linker_ctx_t;


// 功能描述：对合并后的 AST 执行外部引用解析。
// 参数说明：merged_ast 为已完成模块合并的 AST。base_dir为搜索基准目录。
// 返回值：  成功返回 0；失败返回 -1。
int32_t am_linker_import_ref_resolution(am_ast_t *merged_ast, wchar_t *base_dir);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_linker.h ===== */

/* ===== begin: include/am_compiler.h ===== */
#ifndef __AM_COMPILER_H__
#define __AM_COMPILER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>



// 单条IL指令
typedef struct am_instruction_t {
    uint32_t opcode;    // 指令代码：在 @include/ast.h 中定义的AM_VM_OP_*
    am_value_t operand;  // 操作数：统一为TPV，不同的指令有不同的具体类型要求。无参数则设为AM_VALUE_UNDEFINED。
} am_instruction_t;


// 编译器工作语境
typedef struct am_compiler_ctx_t {
    am_ast_t *ast; // 编译输入的AST，编译过程中会被修改，作为编译结果的一部分（概念上相当于“静态数据段”）
    am_iaddr_t icount; // 中间语言指令计数器
    am_iaddr_t ilcode_capacity; // ilcode 当前分配的容量（以指令数计）
    am_instruction_t *ilcode; // 编译得到的中间语言指令序列
    size_t label_counter; // 用于生成标签枚举值的计数器
    am_map_t *value_label_mapping; // Map<am_value_t(any), am_value_t(label)> 从任何类型的索引TPV到标签TPV的映射
    am_map_t *label_iaddr_mapping; // Map<am_value_t(label), am_value_t(iaddr)> 从label值到iaddr值的映射
    am_list_t *while_tag_stack; // while块的标签跟踪栈：用于处理break/continue
    size_t unique_id_counter; // 用于生成唯一枚举值的计数器
    am_iaddr_t offset; // 生成的IL代码在目标进程ilcode中的起始偏移量
    am_iaddr_t ret;    // 程序执行完毕后跳转返回的目标iaddr；为0时则使用halt结束
} am_compiler_ctx_t;


// 功能描述：创建编译器上下文。
// 实现说明：成功返回上下文指针，失败返回NULL。
am_compiler_ctx_t *am_compiler_ctx_create(am_ast_t *ast);


// 功能描述：销毁编译器上下文。
// 实现说明：释放上下文自身及其内部资源（包括ilcode）。
void am_compiler_ctx_destroy(am_compiler_ctx_t *ctx);


// 功能描述：AST编译的起点，将AST编译为中间语言指令序列。
// 实现说明：成功返回0，失败返回-1。编译结果写入ctx->ilcode和ctx->icount。
//         注意：本函数不执行标签解析，调用者应在am_compile_all结束后调用am_compiler_label_resolution。
int32_t am_compile_all(am_compiler_ctx_t *ctx);


// 功能描述：编译后处理——全局标签解析，该函数在am_compile_all结束后调用，用于将所有的label替换为绝对iaddr。
// 实现描述：遍历所有ilcode，检查am_instruction.operand的am_value_t的TPV类型是否是AM_VALUE_TYPE_LABEL。如果是，则调用am_compiler_parse_label_to_iaddr将其转换为iaddr，加上offset后替换掉原来的label。成功返回0，失败返回-1。
int32_t am_compiler_label_resolution(am_compiler_ctx_t *ctx, am_iaddr_t offset);

typedef struct am_module_t am_module_t;

// opstack最大深度的静态分析。成功返回最大深度，失败返回SIZE_MAX。
// 说明：本分析基于编译器生成的中间语言指令序列（ilcode），估算运行时操作数栈可能达到的最大深度。
//       分析覆盖所有lambda函数体、临时lambda（η变换生成）以及顶层thunk，取其中的最大值。
size_t am_compiler_opstack_depth_analysis(am_compiler_ctx_t *ctx);


// 功能描述：编译器入口。将AST编译为am_module_t。
// 实现说明：offset 为生成的IL代码在目标进程中的起始偏移量；ret 为程序执行完毕后跳转返回的目标iaddr，为0时使用halt结束。
//         成功返回指针，失败返回NULL。
am_module_t *am_compile(am_ast_t *ast, am_iaddr_t offset, am_iaddr_t ret);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_compiler.h ===== */

/* ===== begin: include/am_module.h ===== */
#ifndef __AM_MODULE_H__
#define __AM_MODULE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>



///////////////////////////////////////////
// 模块数据结构
// 说明：模块是编译器的输出产物，包含静态AST节点和中间语言指令序列。
//       进程由模块加载而来，模块本身不管理运行时状态。
///////////////////////////////////////////

typedef struct am_module_t {
    am_object_t base; // 基类头：am_module_t也视为对象语言的数据对象

    uint64_t header; // 保留：元数据头
    size_t opstack_depth; // 编译期分析出来的opstack最大深度
    am_ast_t *ast;
    am_instruction_t *ilcode;
    am_iaddr_t ilcode_length; // ilcode数组长度（指令条数）
} am_module_t;


// 将 am_module_t 序列化为二进制数据。
// container_alloc 用于分配模块/AST 结构本身；obj_alloc 用于分配 AST 子对象。
// buffer == NULL 或 offset == SIZE_MAX 时仅计算所需字节数。
// 成功返回新增字节数，失败返回 SIZE_MAX。
size_t am_module_dump(am_allocator_t *container_alloc,
                      am_allocator_t *obj_alloc,
                      am_module_t *mod,
                      uint8_t *buffer,
                      size_t offset);

// 从二进制数据恢复 am_module_t。参数含义与 am_module_dump 对应。
// 成功返回模块指针，失败返回 NULL。
am_module_t *am_module_load(am_allocator_t *container_alloc,
                            am_allocator_t *obj_alloc,
                            uint8_t *buffer,
                            size_t offset);

// 使用 PackBits 算法压缩字节流。
// dst == NULL 时仅计算压缩后字节数。
// 成功返回压缩后字节数，失败返回 SIZE_MAX。
size_t am_packbits_compress(uint8_t *src, size_t src_len, uint8_t *dst);

// 使用 PackBits 算法解压字节流。
// dst == NULL 时仅计算解压后字节数。
// 成功返回解压后字节数，失败返回 SIZE_MAX。
size_t am_packbits_decompress(uint8_t *src, size_t src_len, uint8_t *dst);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_module.h ===== */

/* ===== begin: include/am_js2scm.h ===== */
#ifndef __AM_JS2SCM_H__
#define __AM_JS2SCM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <wchar.h>

/* 将 JavaScript 代码机械翻译成非标准 Scheme 子集。
 * 输入：宽字符 JS 源码。
 * 输出：宽字符 Scheme 源码；失败或在翻译过程中发生词法/语法错误时返回 NULL。
 * 返回的指针由调用者使用 free() 释放。 */
wchar_t *am_js_to_scheme(const wchar_t *js_source);

// JS 翻译器最近一次词法/语法错误消息（UTF-32），无错误时为空字符串；
// am_js_to_scheme 每次进入翻译时清空。供上层（REPL）在翻译失败（返回 NULL）时取用。
const wchar_t *am_js_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_js2scm.h ===== */

/* ===== begin: include/am_process.h ===== */
#ifndef __AM_PROCESS_H__
#define __AM_PROCESS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>



///////////////////////////////////////////
// 进程状态常量
///////////////////////////////////////////

#define AM_PROCESS_STATE_READY     (1)
#define AM_PROCESS_STATE_RUNNING   (2)
#define AM_PROCESS_STATE_SLEEPING  (3)
#define AM_PROCESS_STATE_SUSPENDED (4)
#define AM_PROCESS_STATE_STOPPED   (5)
#define AM_PROCESS_STATE_BLOCKED   (6)
#define AM_PROCESS_STATE_KILLED    (7)


///////////////////////////////////////////
// 进程ID
///////////////////////////////////////////

typedef size_t am_pid_t;


///////////////////////////////////////////
// 进程数据结构
// 说明：进程是Scheme解释器的核心运行时数据结构，包含执行所需的全部状态。
//       每个进程拥有独立的堆、操作数栈和函数调用栈，是虚拟机调度的基本单位。
///////////////////////////////////////////

typedef struct am_process_t {
    am_object_t base; // 基类头：am_process_t也视为对象语言的数据对象

    am_allocator_t *vm_alloc; // VM工作内存分配器
    am_allocator_t *heap_alloc; // 堆内存分配器

    am_pid_t pid;        // 进程ID
    am_pid_t parent_pid; // 亲进程ID
    int32_t state;       // 进程状态

    am_iaddr_t PC;     // 程序计数器：代表下一条指令的iaddr
    am_instruction_t *ilcode; // 中间语言代码
    am_iaddr_t ilcode_length; // 中间语言代码长度

    am_heap_t *heap;   // 进程私有堆（由堆内存专用allocator管理）

    am_strindex_t *strindex;  // 用于全局字符串驻留的多值哈希表，检查某个字符串（的哈希值）是否已存在于heap

    am_vocab_t *var_vocab;    // 变量词表
    am_vocab_t *symbol_vocab; // 符号词表
    am_list_t *var_type;      // 变量类型表（内容同AST，用于运行时判断变量类型，尤其是native_ref）
    am_map_t *natives;        // 本地库记录（内容同AST，用于判断模块使用了哪些本地库）
    am_list_t *var_top;       // 顶级变量varid列表（内容同AST）
    am_map_t *var_arn_mapping; // 变量ARN（Alpha-renaming）前后的映射（内容同AST）

    am_handle_t current_closure_handle; // 指向当前闭包的把柄

    bool pending_kill; // 延迟kill标记：在进程自己的native调用中触发kill时，由调度器安全点完成实际销毁

    size_t gc_count; // GC 触发次数，用于控制标记-压缩频率

    // 操作数栈（其容量为opstack_depth）
    am_value_t *opstack;
    am_value_t *opstack_top; // opstack栈顶指针
    size_t opstack_capacity; // opstack容量（am_value_t元素个数）

    // 函数调用栈（默认容量1000，TODO 后面改成可配置）
    // 注意，成对入栈出栈，栈帧结构为{am_value_t(handle) closure_handle; am_value_t(iaddr) return_target_iaddr; }
    am_value_t *fstack;
    am_value_t *fstack_top; // fstack栈顶指针，注意每次操作加减2个元素
    size_t fstack_capacity; // fstack容量（am_value_t元素个数）

    // dynamic-wind 相关状态
    am_list_t *dynamic_wind_stack;      // 当前 dynamic-wind 栈，元素为 entry handle
    am_list_t *dynamic_wind_after_stack; // 正在执行 after 的条目 handle 栈（与 dynamic_wind_stack 配合）
    size_t     dynamic_wind_mark_counter; // 自增唯一 mark
    am_handle_t current_dynamic_wind_entry; // 当前 dynamic-wind 条目 handle（在 before/thunk 之间暂存）
    am_handle_t current_dynamic_wind_thunk; // 当前 dynamic-wind 的 thunk handle（在 thunk 执行前暂存）

    // continuation 恢复时的 wind 跳板状态
    am_iaddr_t wind_trampoline_iaddr;   // 进程中预留的 wind 指令地址
    int32_t    wind_state;              // 0=空闲, 1=执行 afters, 2=执行 befores, 3=恢复续体
    am_handle_t pending_cont_handle;    // 待恢复的目标续体 handle
    am_value_t  pending_cont_value;     // 调用续体时传入的值
    am_handle_t *pending_after_entries; // 待执行的 after 条目 handle 数组
    size_t      pending_after_count;
    am_handle_t *pending_before_entries;// 待执行的 before 条目 handle 数组
    size_t      pending_before_count;

    void *host_context;     // 宿主提供的不透明上下文
} am_process_t;


///////////////////////////////////////////
// 字符串驻留相关
///////////////////////////////////////////

// 运行时字符串驻留长度阈值：仅对长度不超过该值的字符串启用同值复用
#ifndef AM_PROCESS_STRINDEX_MAX_LEN
#define AM_PROCESS_STRINDEX_MAX_LEN (32)
#endif

// 功能说明：根据 wchar_t 缓冲区和长度创建/复用字符串堆对象，并返回其 handle。
// 实现说明：当 len <= AM_PROCESS_STRINDEX_MAX_LEN 时，会先查询 proc->strindex；
//         若已存在内容相同的字符串则复用其 handle，否则新建并登记。
//         超过阈值的字符串直接新建，不参与驻留。
//         失败返回 AM_HANDLE_NULL。
am_handle_t am_process_make_wstring_handle(am_process_t *proc, const wchar_t *str, size_t len);


///////////////////////////////////////////
// 生命周期
///////////////////////////////////////////

// 功能说明：从模块构造并初始化一个新的进程数据结构
// 实现说明：成功返回新进程对象指针；失败返回NULL
am_process_t *am_process_load_from_module(am_allocator_t *vm_alloc, am_allocator_t *heap_alloc, am_module_t *mod);

// 功能说明：销毁进程数据结构，释放其占用的全部资源
// 实现说明：成功返回0，失败返回-1
int32_t am_process_destroy(am_process_t *proc);


///////////////////////////////////////////
// 操作数栈操作
///////////////////////////////////////////

// 功能说明：向操作数栈中压入值。成功返回0，失败返回-1
int32_t am_process_push_operand(am_process_t *proc, am_value_t v);

// 功能说明：从操作数栈中弹出一个值。成功返回弹出值，失败返回UINTPTR_MAX
am_value_t am_process_pop_operand(am_process_t *proc);

// 功能说明：根据栈顶指针计算opstack中有多少个am_value_t。成功返回长度值，失败返回SIZE_MAX
size_t am_process_length_of_opstack(am_process_t *proc);


///////////////////////////////////////////
// 函数调用栈操作
///////////////////////////////////////////

// 功能说明：向fstack中压入栈帧（两个值）。成功返回0，失败返回-1
int32_t am_process_push_stack_frame(am_process_t *proc, am_value_t closure_handle_value, am_value_t return_target_iaddr_value);

// 功能说明：从fstack中弹出栈帧的两个值，通过两个指针传出。成功返回0，失败返回-1
int32_t am_process_pop_stack_frame(am_process_t *proc, am_value_t *closure_handle_value, am_value_t *return_target_iaddr_value);

// 功能说明：根据栈顶指针计算fstack中有多少个am_value_t（因为是成对push/pop，所以正常情况下必为偶数）。成功返回长度值，失败返回SIZE_MAX
size_t am_process_length_of_fstack(am_process_t *proc);


///////////////////////////////////////////
// 闭包操作
///////////////////////////////////////////

// 功能说明：新建闭包并返回其handle。成功返回handle，失败返回AM_HANDLE_NULL
am_handle_t am_process_make_closure(am_process_t *proc, am_iaddr_t iaddr, am_handle_t parent);

// 功能说明：根据闭包handle获取闭包对象。成功返回指针，失败返回NULL
am_obj_closure_t *am_process_get_closure(am_process_t *proc, am_handle_t hd);

// 功能说明：获取进程的当前闭包对象。成功返回指针，失败返回NULL
am_obj_closure_t *am_process_get_current_closure(am_process_t *proc);

// 功能说明：设置进程的当前闭包handle字段。成功返回0，失败返回-1
static inline int32_t am_process_set_current_closure(am_process_t *proc, am_handle_t hd) {
    if (!proc) return -1;
    proc->current_closure_handle = hd;
    return 0;
}

// 功能说明：变量解引用。成功返回TPV，失败返回UINTPTR_MAX
am_value_t am_process_dereference(am_process_t *proc, am_varid_t varid);


// 功能说明：将进程堆中的列表对象转换为可显示宽字符串。成功返回新分配的 wchar_t*（由调用者释放），失败返回 NULL。
// 实现说明：从 proc->heap 中取得对象，从 proc->var_vocab / proc->symbol_vocab 中解析变量名和符号名。
//          symbol 的处理规则：不在 quote 列表内时带前导单引号；在 quote 列表内时不带前导单引号。
wchar_t *am_process_list_to_string(am_process_t *proc, am_handle_t hd, size_t *length);


///////////////////////////////////////////
// 程序流程控制
///////////////////////////////////////////

// 功能说明：获取当前指令，并取出opcode和operand。成功返回0，失败返回-1
int32_t am_process_current_instruction(am_process_t *proc, uint32_t *opcode, am_value_t *operand);

// 功能说明：前进一步（PC加1）
void am_process_step(am_process_t *proc);

// 功能说明：无条件跳转（PC置数iaddr）
void am_process_goto(am_process_t *proc, am_iaddr_t iaddr);

// 功能说明：设置进程状态
void am_process_set_state(am_process_t *proc, int32_t s);


///////////////////////////////////////////
// 计算续体（continuation）的捕获和恢复
///////////////////////////////////////////

// 功能说明：捕获当前续体，保存为堆对象，并返回其handle。成功返回handle，失败返回AM_HANDLE_NULL
am_handle_t am_process_capture_continuation(am_process_t *proc, am_iaddr_t cont_return_target_iaddr);

// 功能说明：恢复指定的计算续体到当前进程。成功返回其返回目标位置的iaddr，失败返回SIZE_MAX
// 实现说明：传入的 value 为调用续体时传入的值；若需要 wind 调整，则 value 暂存于 proc，待跳板恢复时压栈。
am_iaddr_t am_process_load_continuation(am_process_t *proc, am_handle_t hd, am_value_t value);

// 功能说明：直接恢复续体快照（opstack/fstack/closure），不执行 wind 调整。成功返回 cont_return_target，失败返回 SIZE_MAX
am_iaddr_t am_process_restore_continuation_snapshot(am_process_t *proc, am_handle_t hd);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_process.h ===== */

/* ===== begin: include/am_gc.h ===== */
#ifndef __AM_GC_H__
#define __AM_GC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>



///////////////////////////////////////////
// 垃圾回收（GC）
// 说明：GC 是解释器的核心功能，统一由本模块实现：
//   - 分进程标记-清除（am_gc_process）：GC 根收集、递归标记、清除；
//   - 全局标记-压缩（am_gc_compact）：收集各进程逻辑堆中的存活对象，
//     调用 allocator 的纯物理压缩引擎搬移对象，再回写所有 heap 表指针；
//   - 编排（am_gc_collect）：对进程池执行一轮完整 GC。
// 层级：位于 process/heap 之上、runtime 之下，不依赖 runtime.h。
///////////////////////////////////////////

// GC 配置

#ifndef AM_ENABLE_GC
#define AM_ENABLE_GC (1)
#endif

// 每经历 AM_HEAP_COMPACT_INTERVAL 次 GC 后触发一次标记-压缩。
// 设为 0 表示不在 GC 时自动触发压缩（可手动调用 am_gc_compact）。
#ifndef AM_HEAP_COMPACT_INTERVAL
#define AM_HEAP_COMPACT_INTERVAL (1)
#endif

// GC 触发策略：堆水位 + 慢速周期兜底（见 am_runtime.c 的调用点）。
// 堆区已用比例达到 AM_GC_HEAP_HIGH_WATER_RATIO 时触发一轮标记-清除；
// 达到 AM_GC_HEAP_CRITICAL_RATIO 时当轮强制标记-压缩（无视 AM_HEAP_COMPACT_INTERVAL）。
#ifndef AM_GC_HEAP_HIGH_WATER_RATIO
#define AM_GC_HEAP_HIGH_WATER_RATIO (0.75)
#endif
#ifndef AM_GC_HEAP_CRITICAL_RATIO
#define AM_GC_HEAP_CRITICAL_RATIO (0.90)
#endif
// 碎片维度：堆用量达到 AM_GC_HEAP_FRAG_FLOOR_RATIO 且最大空闲块小于容量的
// AM_GC_HEAP_FRAG_MIN_BLOCK_RATIO 时，视为临界水位（first-fit 随时可能失败，需压缩整理）。
#ifndef AM_GC_HEAP_FRAG_FLOOR_RATIO
#define AM_GC_HEAP_FRAG_FLOOR_RATIO (0.30)
#endif
#ifndef AM_GC_HEAP_FRAG_MIN_BLOCK_RATIO
#define AM_GC_HEAP_FRAG_MIN_BLOCK_RATIO (0.03125)
#endif
// 事件循环每 AM_GC_PERIODIC_INTERVAL 轮执行一轮兜底 GC（保证分配缓慢但持续
// 产生垃圾的程序最终也能回收）。设为 0 表示禁用周期兜底（纯水位触发）。
#ifndef AM_GC_PERIODIC_INTERVAL
#define AM_GC_PERIODIC_INTERVAL (32)
#endif
// 进程执行的每个 tick 内，每 AM_GC_WATERMARK_CHECK_STRIDE 条指令检查一次堆水位，
// 将失控分配的逃逸窗口从整个时间片收窄到 STRIDE 条指令。
#ifndef AM_GC_WATERMARK_CHECK_STRIDE
#define AM_GC_WATERMARK_CHECK_STRIDE (256)
#endif


// 对单个进程执行全量的标记-清除 GC。成功返回 0，失败返回 -1。
int32_t am_gc_process(am_process_t *proc);

// 对多个进程堆一起执行全局标记-压缩：扫描所有 heap 表收集存活对象，
// 调用 am_allocator_heap_compact 引擎搬移对象，并回写所有 heap 表中的指针。
// 用于多进程共享同一个 heap_alloc 的场景。必须在 GC 安全点调用
//（所有相关进程已完成标记-清除）。成功返回 0，失败返回 -1。
int32_t am_gc_compact(am_allocator_t *heap_alloc, am_heap_t **heaps, size_t heap_count);

// 对进程池执行一轮完整 GC：逐进程标记-清除，随后按 gc_seq 与
// AM_HEAP_COMPACT_INTERVAL 决定是否执行全局标记-压缩与内存池边界自动调整。
// process_pool 为进程指针数组（允许含 NULL 槽位），process_count 为数组长度；
// gc_seq 为本轮 GC 的序号（通常由调用方维护的计数器提供）；
// force_compact 非 0 时无视 AM_HEAP_COMPACT_INTERVAL 当轮强制压缩。
// 仅 GC 成功的进程堆才会纳入压缩。成功返回 0，失败返回 -1。
int32_t am_gc_collect(am_allocator_t *heap_alloc, am_process_t **process_pool,
                      size_t process_count, size_t gc_seq, int32_t force_compact);

// 查询堆区水位级别（供运行期按水位触发 GC）：
//   0 = 正常（已用 < AM_GC_HEAP_HIGH_WATER_RATIO）；
//   1 = 高水位（应执行一轮标记-清除）；
//   2 = 临界水位（应执行一轮标记-清除并强制压缩）；
//  负值 = 查询失败（alloc 非池的堆区分配器等）。
int32_t am_gc_heap_watermark_level(am_allocator_t *heap_alloc);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_gc.h ===== */

/* ===== begin: include/am_runtime.h ===== */
#ifndef __AM_RUNTIME_H__
#define __AM_RUNTIME_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>


// 前向声明：回调函数指针需要引用运行时类型
typedef struct am_runtime_t am_runtime_t;


///////////////////////////////////////////
// 虚拟机状态
///////////////////////////////////////////

#define AM_VM_STATE_IDLE    (0)
#define AM_VM_STATE_RUNNING (1)


///////////////////////////////////////////
// 运行时事件处理器配置
///////////////////////////////////////////

#ifndef AM_COMPUTATION_PHASE_LENGTH
#define AM_COMPUTATION_PHASE_LENGTH (1)
#endif


///////////////////////////////////////////
// 异步定时器基础设施
///////////////////////////////////////////

// 时间戳类型：毫秒
typedef uint64_t am_timestamp_t;

// 定时器条目（不透明类型，具体定义见 runtime.c）
typedef struct am_timer_t am_timer_t;

// 注册一个定时器。delay_ms 为首次触发的延迟，repeat 表示是否周期触发，
// interval_ms 为周期触发的间隔。成功返回大于 0 的定时器编号，失败返回 0。
size_t am_runtime_set_timer(am_runtime_t *rt, am_pid_t pid, am_handle_t callback,
                            am_timestamp_t delay_ms, bool repeat, am_timestamp_t interval_ms);

// 根据编号取消一个定时器。成功返回 true，未找到返回 false。
bool am_runtime_clear_timer(am_runtime_t *rt, size_t timer_id);

// 获取当前时间戳（毫秒）。经由 vtable 分派到宿主实现。
am_timestamp_t am_runtime_now_ms(am_runtime_t *rt);

// 以异步方式调用一个闭包：压入栈帧并跳转到闭包入口，返回地址为 return_target。
// 用于定时器回调等场景。成功返回 0，失败返回 -1。
int32_t am_runtime_call_async(am_runtime_t *rt, am_process_t *proc, am_handle_t callback,
                              am_iaddr_t return_target);


///////////////////////////////////////////
// 宿主虚函数表（依赖倒置）
// 说明：runtime 不直接依赖宿主提供的输入输出回调、定时器与时间戳函数，
// 而是由宿主在调用 am_runtime_create 时，通过本虚函数表注入具体实现。
// on_tick/on_event/on_halt/on_error 允许为 NULL（不触发）；
// now_ms/sleep_in_ms 为必需能力，为 NULL 时 am_runtime_create 失败。
///////////////////////////////////////////

typedef struct am_runtime_vtable_t {
    void (*on_tick)(am_runtime_t *rt);    // 每个 Tick 结束后触发
    void (*on_event)(am_runtime_t *rt);   // 每个事件循环结束后触发
    void (*on_halt)(am_runtime_t *rt);    // 虚拟机进入 IDLE 后触发
    void (*on_error)(am_runtime_t *rt);   // 虚拟机捕获异常时触发
    void (*sleep_in_ms)(am_runtime_t *rt, am_timestamp_t ms); // 短时睡眠（毫秒）
    am_timestamp_t (*now_ms)(am_runtime_t *rt);               // 获取当前时间戳（毫秒）
} am_runtime_vtable_t;


///////////////////////////////////////////
// 运行时环境
// 说明：运行时是虚拟机调度的核心，管理进程池、进程队列、FIFO 和回调。
///////////////////////////////////////////

typedef struct am_runtime_t {
    am_allocator_t *vm_alloc;   // 运行时工作内存分配器
    am_allocator_t *heap_alloc; // 进程堆内存分配器

    wchar_t *working_dir;       // 基准工作目录

    size_t process_pool_capacity; // 进程池容量
    am_process_t **process_pool;  // 进程池：am_process_t* 动态数组
    size_t process_poll_counter;  // 进程计数器，也作为下一个 PID
    am_list_t *process_queue;     // 进程队列：List<am_value_t(uint:pid)>

    am_list_t *input_fifo;   // 输入 FIFO（存储 wchar 值）
    am_list_t *output_fifo;  // 输出 FIFO（存储 wchar 值）
    am_list_t *error_fifo;   // 错误 FIFO（存储 wchar 值）

    am_list_t *queue_list;   // 队列列表：List<am_queue_t*>
    size_t queue_next_id;    // 下一个队列编号，从 1 开始递增

    const am_runtime_vtable_t *vtable;  // 宿主虚函数表（由 am_runtime_create 注入，宿主拥有其生命周期）

    size_t tick_counter;     // Tick 计数器
    size_t gc_count;         // 全局 GC 周期计数器（作为 am_gc_collect 的 gc_seq）
    size_t gc_periodic_counter; // 事件循环轮计数器（周期兜底 GC 用）

    uint32_t timeslice;      // 默认时间片长度（单位：VM指令周期数）

    am_timer_t *timer_list;  // 定时器链表头
    size_t timer_next_id;    // 下一个定时器编号

    void *host_context;      // 宿主提供的全局不透明上下文
} am_runtime_t;


///////////////////////////////////////////
// 生命周期
///////////////////////////////////////////

// 创建运行时。成功返回运行时指针，失败返回 NULL。
// base_dir 为基准工作目录，允许为 NULL。
// vtable 为宿主虚函数表，不允许为 NULL，且其 now_ms/sleep_in_ms 成员不允许为 NULL；
// runtime 仅保存指针，不拷贝，宿主须保证 vtable 的生命周期不短于 runtime。
am_runtime_t *am_runtime_create(am_allocator_t *vm_alloc, am_allocator_t *heap_alloc, const wchar_t *base_dir,
                                const am_runtime_vtable_t *vtable);

// 销毁运行时，释放其占用的全部资源。成功返回 0，失败返回 -1。
int32_t am_runtime_destroy(am_runtime_t *rt);


///////////////////////////////////////////
// 入口函数（兼容参考用法）
///////////////////////////////////////////

// 将模块加载到运行时中，创建并启动一个进程。成功返回 PID，失败返回 -1。
am_pid_t am_runtime_load_module(am_runtime_t *rt, am_module_t *mod);

// 启动虚拟机主循环，直到所有进程执行结束进入 IDLE。
void am_runtime_start(am_runtime_t *rt);


///////////////////////////////////////////
// 队列 IPC 基础设施
///////////////////////////////////////////

typedef struct am_queue_waiter_t am_queue_waiter_t;
typedef struct am_queue_t am_queue_t;

// 队列阻塞等待者
struct am_queue_waiter_t {
    am_pid_t pid;                 // 阻塞的进程 ID
    am_value_t value;             // 发送等待者要写入的值（接收等待者忽略）
    am_timestamp_t deadline_ms;   // 超时绝对时间（毫秒）
    bool is_writer;               // true=发送等待者，false=接收等待者
    am_queue_waiter_t *next;      // 链表下一个节点
};

// 多生产者多消费者 FIFO 队列
struct am_queue_t {
    size_t id;                    // 队列编号
    size_t capacity;              // 最大容量
    am_list_t *items;             // 数据项列表（FIFO）
    am_queue_waiter_t *send_waiters; // 等待可写的发送者链表
    am_queue_waiter_t *recv_waiters; // 等待可读的接收者链表
};

// 根据 ID 查找队列。成功返回指针，失败返回 NULL。
am_queue_t *am_runtime_get_queue(am_runtime_t *rt, size_t queue_id);

// 创建一个容量为 capacity 的队列。成功返回队列指针，失败返回 NULL。
am_queue_t *am_runtime_queue_create(am_runtime_t *rt, size_t capacity);

// 销毁队列并释放其占用的全部资源。成功返回 0，失败返回 -1。
int32_t am_runtime_queue_destroy(am_runtime_t *rt, am_queue_t *q);

// 尝试/阻塞地向队列写入一个值。由 native_System.write 调用。
// 立即成功、立即失败或超时失败时都会直接修改 proc 的操作数栈并步进 PC；
// 进入阻塞时设置进程状态并返回 0，不步进 PC。
int32_t am_runtime_queue_write(am_runtime_t *rt, am_queue_t *q, am_value_t value,
                               am_timestamp_t timeout_ms, am_process_t *proc);

// 尝试/阻塞地从队列读取一个值。由 native_System.read 调用。
// 立即成功、立即失败或超时失败时都会直接修改 proc 的操作数栈并步进 PC；
// 进入阻塞时设置进程状态并返回 0，不步进 PC。
int32_t am_runtime_queue_read(am_runtime_t *rt, am_queue_t *q, am_timestamp_t timeout_ms,
                              am_process_t *proc);


///////////////////////////////////////////
// 模块与进程管理
///////////////////////////////////////////

// 将模块加载到运行时中。成功返回 PID，失败返回 (am_pid_t)-1。
am_pid_t am_runtime_load_module(am_runtime_t *rt, am_module_t *mod);

// 根据 PID 获取进程。成功返回进程指针，失败返回 NULL。
am_process_t *am_runtime_get_process(am_runtime_t *rt, am_pid_t pid);

// 彻底终止指定 PID 的进程：释放其堆、栈、AST 相关表及异步任务，但保留 am_process_t 壳。
// 允许在目标进程自己的 native 调用中同步调用，此时会标记为延迟销毁，由调度器安全点完成。
// 成功返回 0；pid 无效或进程已是 KILLED 返回 -1。
int32_t am_runtime_kill_process(am_runtime_t *rt, am_pid_t pid);

// 直接设置 rt->timeslice 字段（单位：VM指令周期数）
void am_runtime_set_default_timeslice(am_runtime_t *rt, uint32_t ticks);

// 根据 pid 返回对应的 process 对象。若失败，返回 NULL。
am_process_t *am_rumtime_get_process_by_pid(am_runtime_t *rt, am_pid_t pid);

// 设置/获取 VM 的全局宿主上下文（不透明数据）。设置成功返回 0，失败返回 -1。
int32_t am_set_runtime_host_context(am_runtime_t *rt, void *ctx);
void *am_get_runtime_host_context(am_runtime_t *rt);

// 设置/获取某进程的宿主上下文（不透明数据）。设置成功返回 0，失败返回 -1。
int32_t am_set_process_host_context(am_runtime_t *rt, am_process_t *proc, void *ctx);
void *am_get_process_host_context(am_runtime_t *rt, am_process_t *proc);


///////////////////////////////////////////
// 调度器
///////////////////////////////////////////

// 执行一次事件循环：执行若干 Tick，触发 GC 和事件回调。
// 返回 AM_VM_STATE_IDLE 或 AM_VM_STATE_RUNNING。
int32_t am_runtime_event_handler(am_runtime_t *rt);

// 执行一个时间片。返回 AM_VM_STATE_IDLE 或 AM_VM_STATE_RUNNING。
int32_t am_runtime_tick(am_runtime_t *rt, uint32_t timeslice);

// 根据opcode和operand分派具体的执行逻辑（指令译码）
int32_t am_runtime_op_dispatch(am_runtime_t *rt, am_process_t *proc, uint32_t opcode, am_value_t operand);

// 执行当前进程的一条指令。成功返回 0，失败返回 -1。
int32_t am_runtime_execute(am_runtime_t *rt, am_process_t *proc);

// 启动虚拟机主循环。
void am_runtime_start(am_runtime_t *rt);


///////////////////////////////////////////
// 内存统计
///////////////////////////////////////////

// 运行时内存统计快照（与 allocator 实现策略无关的抽象结构）。
typedef struct {
    size_t vm_capacity;   // VM 工作区容量（bytes）
    size_t vm_used;       // VM 工作区已用（bytes）
    size_t heap_capacity; // 用户堆区容量（bytes）
    size_t heap_used;     // 用户堆区已用（bytes）
} am_runtime_memory_stats_t;

// 获取运行时内存统计快照。成功返回 0，失败返回 -1。
int32_t am_runtime_get_memory_stats(am_runtime_t *rt, am_runtime_memory_stats_t *out);

// 打印运行时内存总体使用状况（VM 工作区 + 用户堆区）。
// 可在运行时任意时刻调用，接口与 allocator 实现策略无关。
void am_runtime_print_memory_stats(am_runtime_t *rt);


///////////////////////////////////////////
// 控制台输入输出
///////////////////////////////////////////

// 向 stdout 输出字符串，并记录到 output_fifo。
void am_runtime_output(am_runtime_t *rt, const wchar_t *str);

// 向 stderr 输出字符串，并记录到 error_fifo。
void am_runtime_error(am_runtime_t *rt, const wchar_t *str);


///////////////////////////////////////////
// 其他辅助函数
///////////////////////////////////////////

// 将数值 TPV 统一转换为浮点数
am_float_t am_runtime_number_to_float(am_value_t v);

// 将数值 TPV 统一（强制）转换为int
am_int_t am_runtime_number_to_int(am_value_t v);

// 将数值 TPV 统一（强制）转换为uint
am_int_t am_runtime_number_to_uint(am_value_t v);



///////////////////////////////////////////
// 本地宿主函数机制（Native）
///////////////////////////////////////////

// Native函数指针类型：与op_*指令函数签名一致
typedef int32_t (*am_native_func_t)(am_runtime_t *rt, am_process_t *proc);


// 函数表项：库内的单个函数
typedef struct am_native_func_entry_t {
    const wchar_t *name; // 函数名（suffix）
    am_native_func_t func;
} am_native_func_entry_t;


// 库表项：一个native库及其函数表
typedef struct am_native_lib_entry_t {
    const wchar_t *name;                  // 库名（prefix / native_id）
    const am_native_func_entry_t *funcs;  // 该库的函数表
    size_t func_count;                    // 函数表长度
} am_native_lib_entry_t;


#ifndef AM_NATIVE_MAX_LIBS
#define AM_NATIVE_MAX_LIBS (16)
#endif

// 向运行时注册一个native库。成功返回0，失败返回-1。
int32_t am_runtime_register_native_lib(am_runtime_t *rt, const am_native_lib_entry_t *lib);


// 运行时查表：根据库名和函数名查找对应的Native函数实现。
// 成功返回函数指针，失败返回NULL。
am_native_func_t am_native_find_func(const wchar_t *lib_name, const wchar_t *func_name);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_runtime.h ===== */

/* ===== begin: include/am_debug.h ===== */
#ifndef __AM_DEBUG_H__
#define __AM_DEBUG_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <wchar.h>



// 功能描述：将 AST 以 JSON-like 树状格式输出到指定文件流。
// 实现说明：输出内容包括 AST 元数据、nodes 映射以及顶层节点。
void am_debug_ast_print(FILE *out, am_ast_t *ast);


// 功能描述：将 AST 可视化结果输出到 stdout。
// 实现说明：由于 stdout 可能被 printf 置为字节取向，直接 fwprintf 可能无输出，
//          因此先写入临时文件再读取到 wchar_t 缓冲区，最后用 printf("%ls") 输出。
void am_debug_ast_print_to_stdout(am_ast_t *ast);


// 功能描述：输出单个 AST 节点（am_list_t / am_wstring_t）的摘要信息。
// 实现说明：主要用于 nodes 映射遍历时的单行输出。
void am_debug_ast_print_node_summary(FILE *out, am_ast_t *ast, am_handle_t handle);


// 功能描述：将 opcode 转换为其名称字符串。
// 实现说明：未知 opcode 返回 "?"。
const char *am_debug_opcode_name(uint32_t opcode);


// 功能描述：将 IL 指令的操作数以人类可读形式输出到 stdout。
// 实现说明：根据 operand 的 TPV 类型，输出 varid、handle、iaddr、label、symbol、number 等。
void am_debug_print_operand(am_ast_t *ast, am_value_t operand);


// 功能描述：将 IL 指令序列输出到 stdout。
// 实现说明：逐行打印每条指令的索引、opcode 名称和操作数。
void am_debug_print_ilcode(am_ast_t *ast, am_instruction_t *ilcode, am_iaddr_t icount);


// 功能描述：将 IL 指令序列以原始十六进制操作数形式输出到 stdout。
// 实现说明：用于进程测试等无需 AST 词汇表上下文、仅需查看操作数原始值的场景。
void am_debug_print_ilcode_raw(am_instruction_t *ilcode, am_iaddr_t icount);


#ifdef __cplusplus
}
#endif

#endif
/* ===== end:   include/am_debug.h ===== */

#ifdef __cplusplus
}
#endif

#endif /* __ANIMAC_CORE_H__ */
