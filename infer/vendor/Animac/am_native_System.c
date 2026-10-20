#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <time.h>

#include "am_native_System.h"
#include "am_host.h"


// 前向声明：System.eval 的辅助函数，供 System.exec 复用
static wchar_t *eval_wstring_to_vm_buffer(am_process_t *proc, am_wstring_t *ws);
static wchar_t *eval_make_path(am_allocator_t *alloc, const wchar_t *base_dir);


// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 将数值 TPV 统一转换为浮点数
static am_float_t system_number_to_float(am_value_t v) {
    if (am_value_is_float(v)) return am_value_to_float(v);
    if (am_value_is_int(v)) return (am_float_t)am_value_to_int(v);
    if (am_value_is_uint(v)) return (am_float_t)am_value_to_uint(v);
    return 0.0;
}


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


// 从操作数栈中弹出一个闭包把柄。
// 成功返回 true，失败返回 false。
static bool native_pop_closure(am_process_t *proc, am_handle_t *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_obj_closure_t *closure = am_process_get_closure(proc, hd);
    if (!closure) return false;

    *out = hd;
    return true;
}


// 将 wstring 对象的内容复制到堆分配器新分配的 wchar_t 缓冲区中。
// 调用者负责使用 proc->heap_alloc 释放返回的缓冲区。
static wchar_t *wstring_content_to_buffer(am_process_t *proc, am_wstring_t *ws) {
    if (!ws || ws->length == 0) return NULL;

    wchar_t *buf = (wchar_t *)am_malloc(proc->heap_alloc, (ws->length + 1) * sizeof(wchar_t));
    if (!buf) return NULL;

    for (size_t i = 0; i < ws->length; i++) {
        buf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    }
    buf[ws->length] = L'\0';
    return buf;
}


// 从操作数栈中弹出一个字符串对象，并将其内容转换为多字节 char* 缓冲区。
// 成功返回 true，*out 为堆分配器分配的缓冲区（调用者负责释放），失败返回 false。
static bool native_pop_wstring_as_mb(am_process_t *proc, char **out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return false;

    am_wstring_t *ws = (am_wstring_t *)obj;
    wchar_t *wbuf = wstring_content_to_buffer(proc, ws);
    if (ws->length > 0 && !wbuf) return false;

    // UTF-8 最坏情况每个码点 4 字节
    size_t buf_size = (wbuf ? wcslen(wbuf) : 0) * 4 + 1;
    char *mb_buf = (char *)am_malloc(proc->heap_alloc, buf_size);
    if (!mb_buf) {
        if (wbuf) am_free(proc->heap_alloc, wbuf);
        return false;
    }

    am_wcstombs(mb_buf, wbuf ? wbuf : L"", (uint32_t)buf_size);

    if (wbuf) am_free(proc->heap_alloc, wbuf);
    *out = mb_buf;
    return true;
}


// 从 wchar_t 缓冲区创建字符串对象并压回操作数栈。
// 成功返回 0，失败返回 -1。
static int32_t native_push_wstring_buf(am_process_t *proc, const wchar_t *buf, size_t len) {
    am_handle_t hd = am_process_make_wstring_handle(proc, buf, len);
    if (hd == AM_HANDLE_NULL) return -1;

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// 将 float 结果压回操作数栈；若结果为 NaN，则压入 null。
static int32_t native_push_float_or_null(am_process_t *proc, am_float_t result) {
    if (isnan(result)) {
        if (am_process_push_operand(proc, AM_VALUE_NULL) != 0) return -1;
    }
    else {
        if (am_process_push_operand(proc, am_make_value_of_float(result)) != 0) return -1;
    }
    am_process_step(proc);
    return 0;
}


// 设置闭包对象的 keepalive 标记，防止异步回调闭包被 GC 回收。
static void native_keepalive_closure(am_process_t *proc, am_handle_t hd) {
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type == AM_OBJECT_TYPE_CLOSURE) {
        am_object_set_keepalive(obj, 0);
    }
}


// ===============================================================================
// System.fork 内部辅助函数
// ===============================================================================

// 堆对象深拷贝的映射表条目
typedef struct {
    am_handle_t old_handle;
    am_handle_t new_handle;
    am_object_t *old_obj;
    am_object_t *new_obj;
} am_fork_heap_mapping_t;


typedef struct {
    am_allocator_t *vm_alloc;
    am_allocator_t *heap_alloc;
    am_heap_t *src_heap;
    am_heap_t *dst_heap;
    am_fork_heap_mapping_t *entries;
    size_t count;
    size_t capacity;
} am_fork_heap_ctx_t;


static int32_t am_fork_heap_ctx_init(am_fork_heap_ctx_t *ctx,
                                     am_allocator_t *vm_alloc,
                                     am_allocator_t *heap_alloc,
                                     am_heap_t *src_heap,
                                     am_heap_t *dst_heap) {
    if (!ctx || !vm_alloc || !heap_alloc || !src_heap || !dst_heap) return -1;
    ctx->vm_alloc = vm_alloc;
    ctx->heap_alloc = heap_alloc;
    ctx->src_heap = src_heap;
    ctx->dst_heap = dst_heap;
    ctx->entries = NULL;
    ctx->count = 0;
    ctx->capacity = 0;
    return 0;
}


static void am_fork_heap_ctx_destroy(am_fork_heap_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->entries && ctx->vm_alloc) {
        am_free(ctx->vm_alloc, ctx->entries);
    }
    ctx->entries = NULL;
    ctx->count = 0;
    ctx->capacity = 0;
}


static am_handle_t am_fork_heap_map_handle(am_fork_heap_ctx_t *ctx, am_handle_t old_handle) {
    if (old_handle == AM_HANDLE_NULL) return AM_HANDLE_NULL;
    for (size_t i = 0; i < ctx->count; i++) {
        if (ctx->entries[i].old_handle == old_handle) {
            return ctx->entries[i].new_handle;
        }
    }
    return AM_HANDLE_NULL;
}


static am_value_t am_fork_heap_map_value(am_fork_heap_ctx_t *ctx, am_value_t old_value) {
    if (am_value_is_handle(old_value)) {
        am_handle_t new_handle = am_fork_heap_map_handle(ctx, am_value_to_handle(old_value));
        return am_make_value_of_handle(new_handle);
    }
    return old_value;
}


static void am_fork_heap_free_object(am_allocator_t *heap_alloc, am_object_t *obj) {
    if (!heap_alloc || !obj) return;
    switch (obj->type) {
        case AM_OBJECT_TYPE_LIST:         am_list_destroy(heap_alloc, (am_list_t *)obj); break;
        case AM_OBJECT_TYPE_MAP:          am_map_destroy(heap_alloc, (am_map_t *)obj); break;
        case AM_OBJECT_TYPE_WSTRING:      am_wstring_destroy(heap_alloc, (am_wstring_t *)obj); break;
        case AM_OBJECT_TYPE_CLOSURE:      am_closure_destroy(heap_alloc, (am_obj_closure_t *)obj); break;
        case AM_OBJECT_TYPE_CONTINUATION: am_continuation_destroy(heap_alloc, (am_continuation_t *)obj); break;
        default:                          am_free(heap_alloc, obj); break;
    }
}


static am_object_t *am_fork_heap_copy_object(am_fork_heap_ctx_t *ctx, am_object_t *old_obj) {
    if (!old_obj) return NULL;
    switch (old_obj->type) {
        case AM_OBJECT_TYPE_LIST:
            return (am_object_t *)am_list_copy(ctx->heap_alloc, (am_list_t *)old_obj);
        case AM_OBJECT_TYPE_MAP:
            // map 在第二遍重新插入，第一遍先分配空 map
            return (am_object_t *)am_map_create(ctx->heap_alloc, ((am_map_t *)old_obj)->capacity);
        case AM_OBJECT_TYPE_WSTRING:
            return (am_object_t *)am_wstring_copy(ctx->heap_alloc, (am_wstring_t *)old_obj);
        case AM_OBJECT_TYPE_CLOSURE:
            return (am_object_t *)am_closure_copy(ctx->heap_alloc, (am_obj_closure_t *)old_obj);
        case AM_OBJECT_TYPE_CONTINUATION:
            return (am_object_t *)am_continuation_copy(ctx->heap_alloc, (am_continuation_t *)old_obj);
        default:
            // 不支持的对象类型，深拷贝失败
            return NULL;
    }
}


static int32_t am_fork_heap_remap_object(am_fork_heap_ctx_t *ctx, size_t entry_index) {
    am_object_t *new_obj = ctx->entries[entry_index].new_obj;
    am_object_t *old_obj = ctx->entries[entry_index].old_obj;
    if (!new_obj || !old_obj) return 0;

    switch (new_obj->type) {
        case AM_OBJECT_TYPE_LIST: {
            am_list_t *lst = (am_list_t *)new_obj;
            for (size_t i = 0; i < lst->length; i++) {
                lst->children[i] = am_fork_heap_map_value(ctx, lst->children[i]);
            }
            lst->parent = am_fork_heap_map_handle(ctx, lst->parent);
            break;
        }
        case AM_OBJECT_TYPE_MAP: {
            am_map_t *src_m = (am_map_t *)old_obj;
            am_map_t *dst_m = (am_map_t *)new_obj;
            for (size_t i = 0; i < src_m->capacity; i++) {
                am_value_t k = src_m->slots[i].key;
                if (k == AM_MAP_KEY_EMPTY || k == AM_MAP_KEY_TOMBSTONE) continue;
                am_value_t new_k = am_fork_heap_map_value(ctx, k);
                am_value_t new_v = am_fork_heap_map_value(ctx, src_m->slots[i].value);
                am_map_t *new_m = am_map_set(ctx->heap_alloc, dst_m, new_k, new_v);
                if (!new_m) return -1;
                if (new_m != dst_m) {
                    am_handle_t hd = ctx->entries[entry_index].new_handle;
                    if (am_heap_set(ctx->vm_alloc, ctx->heap_alloc, ctx->dst_heap, hd,
                                    am_make_value_of_ptr((am_object_t *)new_m)) != 0) {
                        return -1;
                    }
                    dst_m = new_m;
                    ctx->entries[entry_index].new_obj = (am_object_t *)new_m;
                }
            }
            break;
        }
        case AM_OBJECT_TYPE_CLOSURE: {
            am_obj_closure_t *closure = (am_obj_closure_t *)new_obj;
            closure->parent = am_fork_heap_map_handle(ctx, closure->parent);
            for (size_t i = 0; i < closure->length; i++) {
                closure->bindings[i].value = am_fork_heap_map_value(ctx, closure->bindings[i].value);
            }
            break;
        }
        case AM_OBJECT_TYPE_CONTINUATION: {
            am_continuation_t *cont = (am_continuation_t *)new_obj;
            cont->current_closure_handle = am_fork_heap_map_handle(ctx, cont->current_closure_handle);
            for (size_t i = 0; i < cont->length; i++) {
                cont->stacks[i] = am_fork_heap_map_value(ctx, cont->stacks[i]);
            }
            cont->dynamic_wind_stack_handle = am_fork_heap_map_handle(ctx, cont->dynamic_wind_stack_handle);
            cont->dynamic_wind_after_stack_handle = am_fork_heap_map_handle(ctx, cont->dynamic_wind_after_stack_handle);
            cont->current_dynamic_wind_entry_handle = am_fork_heap_map_handle(ctx, cont->current_dynamic_wind_entry_handle);
            cont->current_dynamic_wind_thunk_handle = am_fork_heap_map_handle(ctx, cont->current_dynamic_wind_thunk_handle);
            break;
        }
        case AM_OBJECT_TYPE_WSTRING:
            // 字符串不含把柄引用，无需重映射
            break;
        default:
            // 不应到达：拷贝阶段已过滤不支持的类型
            return -1;
    }
    return 0;
}


static int32_t am_fork_heap_deep_copy(am_fork_heap_ctx_t *ctx) {
    if (!ctx || !ctx->src_heap || !ctx->src_heap->table) return -1;

    size_t src_count = am_map_length(ctx->vm_alloc, ctx->src_heap->table);
    am_value_t *keys = am_map_keys(ctx->vm_alloc, ctx->src_heap->table);
    if (!keys && src_count > 0) return -1;

    // 第一遍：为源堆中每个 handle 分配新 handle，并深拷贝对象（map 先分配空壳）
    for (size_t i = 0; i < src_count; i++) {
        am_handle_t old_handle = am_value_to_handle(keys[i]);
        am_value_t old_value = am_heap_get(ctx->vm_alloc, ctx->heap_alloc, ctx->src_heap, old_handle);
        am_object_t *old_obj = am_value_is_ptr(old_value) ? am_value_to_ptr(old_value) : NULL;

        am_handle_t new_handle = am_heap_alloc_handle(ctx->vm_alloc, ctx->heap_alloc, ctx->dst_heap);
        if (new_handle == AM_HANDLE_NULL) {
            am_free(ctx->vm_alloc, keys);
            return -1;
        }

        am_object_t *new_obj = NULL;
        am_value_t new_value = old_value;
        if (old_obj) {
            new_obj = am_fork_heap_copy_object(ctx, old_obj);
            if (!new_obj) {
                am_free(ctx->vm_alloc, keys);
                return -1;
            }
            new_value = am_make_value_of_ptr(new_obj);
        }

        if (am_heap_set(ctx->vm_alloc, ctx->heap_alloc, ctx->dst_heap, new_handle, new_value) != 0) {
            if (new_obj) am_fork_heap_free_object(ctx->heap_alloc, new_obj);
            am_free(ctx->vm_alloc, keys);
            return -1;
        }

        if (ctx->count >= ctx->capacity) {
            size_t new_capacity = ctx->capacity ? ctx->capacity * 2 : 16;
            am_fork_heap_mapping_t *new_entries = (am_fork_heap_mapping_t *)am_realloc(
                ctx->vm_alloc, ctx->entries, new_capacity * sizeof(am_fork_heap_mapping_t));
            if (!new_entries) {
                am_free(ctx->vm_alloc, keys);
                return -1;
            }
            ctx->entries = new_entries;
            ctx->capacity = new_capacity;
        }

        ctx->entries[ctx->count].old_handle = old_handle;
        ctx->entries[ctx->count].new_handle = new_handle;
        ctx->entries[ctx->count].old_obj = old_obj;
        ctx->entries[ctx->count].new_obj = new_obj;
        ctx->count++;
    }
    am_free(ctx->vm_alloc, keys);

    // 第二遍：重映射各对象内部的把柄引用
    for (size_t i = 0; i < ctx->count; i++) {
        if (am_fork_heap_remap_object(ctx, i) != 0) {
            return -1;
        }
    }

    return 0;
}


static am_process_t *am_fork_process_copy(am_runtime_t *rt, am_process_t *parent) {
    (void)rt;
    if (!parent || !parent->vm_alloc || !parent->heap_alloc) return NULL;

    am_allocator_t *vm_alloc = parent->vm_alloc;
    am_allocator_t *heap_alloc = parent->heap_alloc;

    am_process_t *child = (am_process_t *)am_calloc(vm_alloc, sizeof(am_process_t));
    if (!child) return NULL;

    child->base = parent->base;
    child->vm_alloc = vm_alloc;
    child->heap_alloc = heap_alloc;
    child->pid = 0;
    child->parent_pid = parent->pid;
    child->state = AM_PROCESS_STATE_READY;
    child->PC = parent->PC;
    child->ilcode_length = parent->ilcode_length;
    child->current_closure_handle = parent->current_closure_handle;

    child->ilcode = (am_instruction_t *)am_malloc(vm_alloc, parent->ilcode_length * sizeof(am_instruction_t));
    if (!child->ilcode) goto fail;
    memcpy(child->ilcode, parent->ilcode, parent->ilcode_length * sizeof(am_instruction_t));

    child->var_vocab = am_vocab_copy(vm_alloc, parent->var_vocab);
    child->symbol_vocab = am_vocab_copy(vm_alloc, parent->symbol_vocab);
    child->var_type = am_list_copy(vm_alloc, parent->var_type);
    child->natives = am_map_copy(vm_alloc, parent->natives);
    child->var_top = am_list_copy(vm_alloc, parent->var_top);
    child->var_arn_mapping = am_map_copy(vm_alloc, parent->var_arn_mapping);
    if (!child->var_vocab || !child->symbol_vocab || !child->var_type || !child->natives || !child->var_top || !child->var_arn_mapping) goto fail;

    child->opstack_capacity = parent->opstack_capacity;
    child->opstack = (am_value_t *)am_calloc(vm_alloc, child->opstack_capacity * sizeof(am_value_t));
    if (!child->opstack) goto fail;
    size_t opstack_len = am_process_length_of_opstack(parent);
    if (opstack_len != SIZE_MAX) {
        memcpy(child->opstack, parent->opstack, opstack_len * sizeof(am_value_t));
        child->opstack_top = child->opstack + opstack_len;
    } else {
        child->opstack_top = child->opstack;
    }

    child->fstack_capacity = parent->fstack_capacity;
    child->fstack = (am_value_t *)am_calloc(vm_alloc, child->fstack_capacity * sizeof(am_value_t));
    if (!child->fstack) goto fail;
    size_t fstack_len = am_process_length_of_fstack(parent);
    if (fstack_len != SIZE_MAX) {
        memcpy(child->fstack, parent->fstack, fstack_len * sizeof(am_value_t));
        child->fstack_top = child->fstack + fstack_len;
    } else {
        child->fstack_top = child->fstack;
    }

    // 拷贝 dynamic-wind 栈（vm_alloc 中的列表，条目 handle 在深拷贝后重映射）
    child->dynamic_wind_stack = am_list_copy(vm_alloc, parent->dynamic_wind_stack);
    if (!child->dynamic_wind_stack) goto fail;
    child->dynamic_wind_after_stack = am_list_copy(vm_alloc, parent->dynamic_wind_after_stack);
    if (!child->dynamic_wind_after_stack) goto fail;
    child->dynamic_wind_mark_counter = parent->dynamic_wind_mark_counter;
    child->current_dynamic_wind_entry = AM_HANDLE_NULL;
    child->current_dynamic_wind_thunk = AM_HANDLE_NULL;
    child->wind_trampoline_iaddr = parent->wind_trampoline_iaddr;
    child->wind_state = 0;
    child->pending_cont_handle = AM_HANDLE_NULL;
    child->pending_cont_value = AM_VALUE_UNDEFINED;
    child->pending_after_entries = NULL;
    child->pending_after_count = 0;
    child->pending_before_entries = NULL;
    child->pending_before_count = 0;

    size_t heap_capacity = parent->heap ? parent->heap->capacity : 16;
    child->heap = am_heap_create(vm_alloc, heap_alloc, heap_capacity);
    if (!child->heap) goto fail;

    am_fork_heap_ctx_t ctx;
    if (am_fork_heap_ctx_init(&ctx, vm_alloc, heap_alloc, parent->heap, child->heap) != 0) goto fail;

    int32_t copy_ok = am_fork_heap_deep_copy(&ctx);
    if (copy_ok == 0) {
        child->current_closure_handle = am_fork_heap_map_handle(&ctx, child->current_closure_handle);

        // 重映射 IL 指令中的字面 handle（如字符串字面量）
        for (size_t i = 0; i < child->ilcode_length; i++) {
            am_value_t operand = child->ilcode[i].operand;
            if (am_value_is_handle(operand)) {
                child->ilcode[i].operand = am_make_value_of_handle(
                    am_fork_heap_map_handle(&ctx, am_value_to_handle(operand)));
            }
        }

        size_t op_len = (size_t)(child->opstack_top - child->opstack);
        for (size_t i = 0; i < op_len; i++) {
            child->opstack[i] = am_fork_heap_map_value(&ctx, child->opstack[i]);
        }

        size_t f_len = (size_t)(child->fstack_top - child->fstack);
        for (size_t i = 0; i < f_len; i++) {
            child->fstack[i] = am_fork_heap_map_value(&ctx, child->fstack[i]);
        }

        // 重映射 dynamic-wind 栈中的 entry handle
        if (child->dynamic_wind_stack) {
            for (size_t i = 0; i < child->dynamic_wind_stack->length; i++) {
                child->dynamic_wind_stack->children[i] = am_fork_heap_map_value(
                    &ctx, child->dynamic_wind_stack->children[i]);
            }
        }

        // 重映射正在执行 after 的 dynamic-wind 条目 handle
        if (child->dynamic_wind_after_stack) {
            for (size_t i = 0; i < child->dynamic_wind_after_stack->length; i++) {
                child->dynamic_wind_after_stack->children[i] = am_fork_heap_map_value(
                    &ctx, child->dynamic_wind_after_stack->children[i]);
            }
        }
    }
    am_fork_heap_ctx_destroy(&ctx);

    if (copy_ok != 0) goto fail;

    return child;

fail:
    if (child) {
        am_process_destroy(child);
    }
    return NULL;
}


static am_pid_t am_fork_runtime_add_process(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc) return (am_pid_t)-1;

    am_pid_t pid = rt->process_poll_counter;
    if (pid >= rt->process_pool_capacity) {
        size_t new_cap = rt->process_pool_capacity * 2;
        am_process_t **new_pool = (am_process_t **)am_realloc(
            rt->vm_alloc, rt->process_pool, new_cap * sizeof(am_process_t *));
        if (!new_pool) return (am_pid_t)-1;
        rt->process_pool = new_pool;
        rt->process_pool_capacity = new_cap;
    }

    proc->pid = pid;
    rt->process_pool[pid] = proc;
    rt->process_poll_counter++;
    return pid;
}


// ===============================================================================
// Native 函数实现
// ===============================================================================

// (System.exit) : void
// 立即停止当前进程，将其状态置为 STOPPED。不影响已存在的异步任务。
int32_t am_native_System_exit(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!proc) return -1;
    am_process_set_state(proc, AM_PROCESS_STATE_STOPPED);
    return 0;
}


// (System.kill pid:Number) : Boolean
// 彻底终止指定 PID 的进程：释放其堆、栈、AST 相关表及异步任务，但保留 am_process_t 壳。
int32_t am_native_System_kill(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc) return -1;

    am_float_t pid_f;
    if (!native_pop_number(proc, &pid_f)) return -1;

    if (isnan(pid_f) || pid_f < 0.0) {
        if (am_process_push_operand(proc, AM_VALUE_FALSE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    int32_t ret = am_runtime_kill_process(rt, (am_pid_t)pid_f);

    if (am_process_push_operand(proc, ret == 0 ? AM_VALUE_TRUE : AM_VALUE_FALSE) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// ===============================================================================
// System.exec 内部辅助函数
// ===============================================================================

// exec 失败时压栈 -1，并步进到下一条指令。
static int32_t exec_push_failure(am_process_t *proc) {
    if (am_process_push_operand(proc, am_make_value_of_int(-1)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// 释放编译产生的 module（ilcode + ast + module 结构体本身）。
static void exec_free_module(am_allocator_t *vm_alloc, am_module_t *mod) {
    if (!mod) return;
    if (mod->ilcode) am_free(vm_alloc, mod->ilcode);
    if (mod->ast) am_ast_destroy(mod->ast);
    am_free(vm_alloc, mod);
}


// 用 new_proc 的内容原地替换 proc，保持 proc 指针不变（调用链持有该指针）。
static int32_t exec_replace_process(am_process_t *proc, am_process_t *new_proc) {
    if (!proc || !new_proc) return -1;

    am_pid_t old_pid = proc->pid;
    am_pid_t old_parent_pid = proc->parent_pid;

    // 释放旧进程占用的全部资源
    if (proc->ilcode) {
        am_free(proc->vm_alloc, proc->ilcode);
        proc->ilcode = NULL;
    }
    if (proc->opstack) {
        am_free(proc->vm_alloc, proc->opstack);
        proc->opstack = NULL;
        proc->opstack_top = NULL;
    }
    if (proc->fstack) {
        am_free(proc->vm_alloc, proc->fstack);
        proc->fstack = NULL;
        proc->fstack_top = NULL;
    }
    if (proc->var_type) {
        am_list_destroy(proc->vm_alloc, proc->var_type);
        proc->var_type = NULL;
    }
    if (proc->natives) {
        am_map_destroy(proc->vm_alloc, proc->natives);
        proc->natives = NULL;
    }
    if (proc->var_top) {
        am_list_destroy(proc->vm_alloc, proc->var_top);
        proc->var_top = NULL;
    }
    if (proc->var_arn_mapping) {
        am_map_destroy(proc->vm_alloc, proc->var_arn_mapping);
        proc->var_arn_mapping = NULL;
    }
    if (proc->strindex) {
        am_strindex_destroy(proc->vm_alloc, proc->strindex);
        proc->strindex = NULL;
    }
    if (proc->heap) {
        am_heap_destroy(proc->vm_alloc, proc->heap_alloc, proc->heap);
        proc->heap = NULL;
    }

    // 整体替换为 new_proc 的内容
    memcpy(proc, new_proc, sizeof(am_process_t));
    proc->pid = old_pid;
    proc->parent_pid = old_parent_pid;
    proc->state = AM_PROCESS_STATE_RUNNING;
    proc->PC = 0;
    proc->current_closure_handle = AM_HANDLE_NULL;

    // new_proc 结构体本身不再使用，其字段已归属 proc
    am_free(proc->vm_alloc, new_proc);
    return 0;
}


// (System.exec code:String) : Number|-1
// 将 Scheme 源码编译成 module，原地替换当前 process 的全部内容并从 PC=0 开始执行。
// 失败时压栈 -1 并继续执行。
int32_t am_native_System_exec(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc || !proc->heap) return -1;

    // 弹出源码字符串对象
    am_value_t code_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(code_val)) return -1;

    am_handle_t code_h = am_value_to_handle(code_val);
    am_value_t code_obj = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, code_h);
    if (!am_value_is_ptr(code_obj)) return -1;

    am_object_t *obj = am_value_to_ptr(code_obj);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return -1;

    wchar_t *code_buf = eval_wstring_to_vm_buffer(proc, (am_wstring_t *)obj);
    if (!code_buf) return -1;

    wchar_t *path_buf = eval_make_path(proc->vm_alloc, rt->working_dir);
    if (!path_buf) {
        am_free(proc->vm_alloc, code_buf);
        return -1;
    }

    // 包装成顶层 thunk 形式，与 main.c 保持一致
    const wchar_t *prefix = L"((lambda () \n";
    const wchar_t *suffix = L"\n))";
    size_t code_len = wcslen(code_buf);
    size_t wrapped_len = wcslen(prefix) + code_len + wcslen(suffix);
    wchar_t *wrapped_code = (wchar_t *)am_malloc(proc->vm_alloc,
                                                  (wrapped_len + 1) * sizeof(wchar_t));
    if (!wrapped_code) {
        am_free(proc->vm_alloc, code_buf);
        am_free(proc->vm_alloc, path_buf);
        return exec_push_failure(proc);
    }
    wcscpy(wrapped_code, prefix);
    wcscat(wrapped_code, code_buf);
    wcscat(wrapped_code, suffix);

    // 解析、链接、编译
    am_ast_t *ast = am_parse(proc->vm_alloc, wrapped_code, path_buf, 0);
    if (!ast) goto fail;

    am_ast_t *linked = am_link(ast, rt->working_dir, am_host_read_source_from_file, NULL);
    if (!linked) {
        am_ast_destroy(ast);
        goto fail;
    }

    am_module_t *mod = am_compile(linked, 0, 0);
    if (!mod) {
        am_ast_destroy(linked);
        goto fail;
    }

    // 从 module 构造新进程
    am_process_t *new_proc = am_process_load_from_module(proc->vm_alloc, proc->heap_alloc, mod);
    if (!new_proc) {
        exec_free_module(proc->vm_alloc, mod);
        goto fail;
    }

    // 原地替换当前进程内容
    if (exec_replace_process(proc, new_proc) != 0) {
        am_process_destroy(new_proc);
        exec_free_module(proc->vm_alloc, mod);
        goto fail;
    }

    // 清理编译中间产物
    exec_free_module(proc->vm_alloc, mod);
    am_free(proc->vm_alloc, wrapped_code);
    am_free(proc->vm_alloc, code_buf);
    am_free(proc->vm_alloc, path_buf);

    // 成功：proc 已经被替换，PC=0，无需压栈
    return 0;

fail:
    am_free(proc->vm_alloc, wrapped_code);
    am_free(proc->vm_alloc, code_buf);
    am_free(proc->vm_alloc, path_buf);
    return exec_push_failure(proc);
}


// (System.set_timeout time_ms:Number callback:(void->undefined)) : Number(计时器编号)
int32_t am_native_System_set_timeout(am_runtime_t *rt, am_process_t *proc) {
    am_float_t time_ms;
    am_handle_t callback;

    // 参数退栈顺序与参数列表顺序相反：先 callback，后 time_ms
    if (!native_pop_closure(proc, &callback)) return -1;
    if (!native_pop_number(proc, &time_ms)) return -1;

    if (isnan(time_ms) || time_ms < 0) {
        // 非法延时：返回 0 计时器编号
        return native_push_float_or_null(proc, 0.0);
    }

    // 防止回调闭包被 GC 回收
    native_keepalive_closure(proc, callback);

    am_timestamp_t delay = (am_timestamp_t)time_ms;
    size_t timer_id = am_runtime_set_timer(rt, proc->pid, callback, delay, false, 0);
    if (timer_id == 0) return -1;

    return native_push_float_or_null(proc, (am_float_t)timer_id);
}


// (System.set_interval time_ms:Number callback:(void->undefined)) : Number(计时器编号)
int32_t am_native_System_set_interval(am_runtime_t *rt, am_process_t *proc) {
    am_float_t time_ms;
    am_handle_t callback;

    // 参数退栈顺序与参数列表顺序相反：先 callback，后 time_ms
    if (!native_pop_closure(proc, &callback)) return -1;
    if (!native_pop_number(proc, &time_ms)) return -1;

    if (isnan(time_ms) || time_ms < 0) {
        return native_push_float_or_null(proc, 0.0);
    }

    native_keepalive_closure(proc, callback);

    am_timestamp_t interval = (am_timestamp_t)time_ms;
    size_t timer_id = am_runtime_set_timer(rt, proc->pid, callback, interval, true, interval);
    if (timer_id == 0) return -1;

    return native_push_float_or_null(proc, (am_float_t)timer_id);
}


// (System.clear_timeout timer:Number) : undefined
// 语句型 native（单值纪律 N2）：不产出有意义结果，压 1 个 #undefined 再返回
int32_t am_native_System_clear_timeout(am_runtime_t *rt, am_process_t *proc) {
    am_float_t timer_id_f;
    if (!native_pop_number(proc, &timer_id_f)) return -1;

    if (!isnan(timer_id_f)) {
        am_runtime_clear_timer(rt, (size_t)timer_id_f);
    }

    if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (System.clear_interval timer:Number) : undefined
// 语句型 native（单值纪律 N2）：不产出有意义结果，压 1 个 #undefined 再返回
int32_t am_native_System_clear_interval(am_runtime_t *rt, am_process_t *proc) {
    am_float_t timer_id_f;
    if (!native_pop_number(proc, &timer_id_f)) return -1;

    if (!isnan(timer_id_f)) {
        am_runtime_clear_timer(rt, (size_t)timer_id_f);
    }

    if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (System.timestamp) : Number
int32_t am_native_System_timestamp(am_runtime_t *rt, am_process_t *proc) {
    am_timestamp_t now = am_runtime_now_ms(rt);
    if (am_process_push_operand(proc, am_make_value_of_uint(now)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (System.memstat) : List
// 返回一个包含四项的列表：'(vm_capacity vm_used heap_capacity heap_used)，单位 bytes。
int32_t am_native_System_memstat(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc || !proc->heap) return -1;

    am_runtime_memory_stats_t stats;
    if (am_runtime_get_memory_stats(rt, &stats) != 0) return -1;

    am_list_t *lst = am_list_create(proc->heap_alloc, 4, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!lst) return -1;

    lst = am_list_push(proc->heap_alloc, lst, am_make_value_of_uint((am_uint_t)stats.vm_capacity));
    if (!lst) return -1;
    lst = am_list_push(proc->heap_alloc, lst, am_make_value_of_uint((am_uint_t)stats.vm_used));
    if (!lst) return -1;
    lst = am_list_push(proc->heap_alloc, lst, am_make_value_of_uint((am_uint_t)stats.heap_capacity));
    if (!lst) return -1;
    lst = am_list_push(proc->heap_alloc, lst, am_make_value_of_uint((am_uint_t)stats.heap_used));
    if (!lst) return -1;

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) return -1;

    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return -1;
    }

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (System.gc) : Boolean
// 强制对当前进程执行一次 GC，主要用于测试 wind 调整等关键路径的 GC 安全性。
int32_t am_native_System_gc(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!proc || !proc->heap) return -1;

    int32_t ret = am_gc_process(proc);
    if (am_process_push_operand(proc, (ret == 0) ? AM_VALUE_TRUE : AM_VALUE_FALSE) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (System.fork) : Number
// 深度复制当前进程，保留除 pid 和 parent_pid 外的全部状态。
// 亲进程返回子进程 pid，子进程返回 0。
int32_t am_native_System_fork(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc) return -1;

    // 亲进程 PC 前进到 fork 调用后的下一条指令
    am_process_step(proc);

    // 深拷贝子进程
    am_process_t *child = am_fork_process_copy(rt, proc);
    if (!child) return -1;

    // 子进程操作数栈压入 0
    if (am_process_push_operand(child, am_make_value_of_int(0)) != 0) {
        am_process_destroy(child);
        return -1;
    }
    // 将子进程加入运行时
    am_pid_t child_pid = am_fork_runtime_add_process(rt, child);
    if (child_pid == (am_pid_t)-1) {
        am_process_destroy(child);
        return -1;
    }
    // 亲进程操作数栈压入子进程 pid
    if (am_process_push_operand(proc, am_make_value_of_int((am_int_t)child_pid)) != 0) {
        return -1;
    }

    // 释放亲进程 CPU，将亲进程和子进程都加入调度队列
    am_process_set_state(proc, AM_PROCESS_STATE_READY);
    am_list_t *new_queue = am_list_push(rt->vm_alloc, rt->process_queue,
                                        am_make_value_of_uint((am_uint_t)proc->pid));
    if (!new_queue) {
        return -1;
    }
    rt->process_queue = new_queue;

    new_queue = am_list_push(rt->vm_alloc, rt->process_queue,
                             am_make_value_of_uint((am_uint_t)child_pid));
    if (!new_queue) {
        return -1;
    }
    rt->process_queue = new_queue;

    return 0;
}


// (System.make_queue len:Number) : Number|null
// 创建一个容量为 len 的队列，返回队列编号；失败返回 #null。
int32_t am_native_System_make_queue(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc) return -1;

    am_float_t len_f;
    if (!native_pop_number(proc, &len_f)) return -1;

    if (isnan(len_f) || len_f <= 0.0) {
        if (am_process_push_operand(proc, AM_VALUE_NULL) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    am_queue_t *q = am_runtime_queue_create(rt, (size_t)len_f);
    if (!q) {
        if (am_process_push_operand(proc, AM_VALUE_NULL) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    if (am_process_push_operand(proc, am_make_value_of_uint((am_uint_t)q->id)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (System.write qid:Number v:any timeout_ms:Number) : Boolean
// 向队列 qid 写入 v，timeout_ms 为超时时间。成功返回 #t，失败返回 #f。
int32_t am_native_System_write(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc) return -1;

    am_float_t timeout_f;
    if (!native_pop_number(proc, &timeout_f)) return -1;

    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return -1;

    am_float_t qid_f;
    if (!native_pop_number(proc, &qid_f)) return -1;

    if (isnan(timeout_f) || timeout_f < 0.0 || isnan(qid_f) || qid_f < 0.0) {
        if (am_process_push_operand(proc, AM_VALUE_FALSE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    am_queue_t *q = am_runtime_get_queue(rt, (size_t)qid_f);
    if (!q) {
        if (am_process_push_operand(proc, AM_VALUE_FALSE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    return am_runtime_queue_write(rt, q, v, (am_timestamp_t)timeout_f, proc);
}


// (System.read qid:Number timeout_ms:Number) : any
// 从队列 qid 读取一个值，timeout_ms 为超时时间。成功返回值，失败返回 #undefined。
int32_t am_native_System_read(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc) return -1;

    am_float_t timeout_f;
    if (!native_pop_number(proc, &timeout_f)) return -1;

    am_float_t qid_f;
    if (!native_pop_number(proc, &qid_f)) return -1;

    if (isnan(timeout_f) || timeout_f < 0.0 || isnan(qid_f) || qid_f < 0.0) {
        if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    am_queue_t *q = am_runtime_get_queue(rt, (size_t)qid_f);
    if (!q) {
        if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    return am_runtime_queue_read(rt, q, (am_timestamp_t)timeout_f, proc);
}


// (System.test x:any) : String
// 保留的测试函数，用于兼容现有测试用例 mob.scm。
int32_t am_native_System_test(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_value_t v = am_process_pop_operand(proc);
    // 单值纪律 N5：每次裸 pop 都必须检查 UINTPTR_MAX 哨兵（栈下溢 = 调用方少传参数 = 硬错误）
    if (v == (am_value_t)UINTPTR_MAX) return -1;
    wchar_t buf[128];
    swprintf(buf, 128, L"Value=%zu", (size_t)v);
    return native_push_wstring_buf(proc, buf, wcslen(buf));
}


// ===============================================================================
// System.eval 内部辅助函数
// ===============================================================================

// 构造 eval 代码的临时绝对路径：base_dir/__eval__.scm
static wchar_t *eval_make_path(am_allocator_t *alloc, const wchar_t *base_dir) {
    if (!alloc) return NULL;
    const wchar_t *filename = L"__eval__.scm";
    size_t base_len = base_dir ? wcslen(base_dir) : 0;
    size_t file_len = wcslen(filename);
    int need_sep = (base_len > 0 && base_dir[base_len - 1] != L'/' && base_dir[base_len - 1] != L'\\');
    wchar_t *path = (wchar_t *)am_malloc(alloc,
                                          (base_len + (need_sep ? 1 : 0) + file_len + 1) * sizeof(wchar_t));
    if (!path) return NULL;
    if (base_len > 0) wcscpy(path, base_dir);
    if (need_sep) path[base_len] = L'/';
    wcscpy(path + base_len + (need_sep ? 1 : 0), filename);
    return path;
}


// 将 wstring 对象内容复制到 vm_alloc 管理的 wchar_t 缓冲区
static wchar_t *eval_wstring_to_vm_buffer(am_process_t *proc, am_wstring_t *ws) {
    if (!proc || !ws || ws->length == 0) return NULL;
    wchar_t *buf = (wchar_t *)am_malloc(proc->vm_alloc, (ws->length + 1) * sizeof(wchar_t));
    if (!buf) return NULL;
    for (size_t i = 0; i < ws->length; i++) {
        buf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    }
    buf[ws->length] = L'\0';
    return buf;
}


// 建立 symbol 映射：evalee symbol -> proc symbol（按字面全局统一）
static int32_t eval_build_symbol_mapping(am_process_t *proc, am_ast_t *evalee, am_map_t **mapping) {
    if (!proc || !evalee || !mapping) return -1;
    size_t count = evalee->symbol_vocab ? evalee->symbol_vocab->length : 0;
    am_map_t *m = am_map_create(proc->vm_alloc, count > 0 ? count : 8);
    if (!m) return -1;

    for (size_t i = 0; i < count; i++) {
        wchar_t *word = am_vocab_get(evalee->alloc, evalee->symbol_vocab, &i);
        if (!word) {
            am_map_destroy(proc->vm_alloc, m);
            return -1;
        }
        size_t proc_idx = am_vocab_find(proc->vm_alloc, proc->symbol_vocab, word);
        if (proc_idx == SIZE_MAX) {
            proc->symbol_vocab = am_vocab_insert(proc->vm_alloc, proc->symbol_vocab, word, &proc_idx);
            if (!proc->symbol_vocab || proc_idx == SIZE_MAX) {
                am_map_destroy(proc->vm_alloc, m);
                return -1;
            }
        }
        am_map_t *new_m = am_map_set(proc->vm_alloc, m,
                                      am_make_value_of_symbol((am_symbol_t)i),
                                      am_make_value_of_symbol((am_symbol_t)proc_idx));
        if (!new_m) {
            am_map_destroy(proc->vm_alloc, m);
            return -1;
        }
        m = new_m;
    }
    *mapping = m;
    return 0;
}


// 判断某个 varid 是否仍被 evalee AST 引用（从顶层节点可达的节点）。
// 宏展开会消除 macro-use 与 define-syntax 节点，但宏关键字等标识符在 parse 阶段
// 已注册进 var_vocab（GLOBAL_FREE）；这些未被引用的残留条目不参与映射，直接跳过即可。
// 注意：nodes 堆是只增的，被消除的旧节点仍留在堆中，因此不能遍历全堆，
// 只能从顶层节点做可达性遍历。
static void eval_var_ref_walk(am_ast_t *ast, am_value_t item, am_varid_t varid, int32_t *found) {
    if (*found) return;
    if (am_value_is_varid(item)) {
        if (am_value_to_varid(item) == varid) *found = 1;
        return;
    }
    if (!am_value_is_handle(item)) return;
    am_value_t node_val = am_ast_get_node(ast, am_value_to_handle(item));
    if (!am_value_is_ptr(node_val)) return;
    am_object_t *obj = am_value_to_ptr(node_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;
    am_list_t *lst = (am_list_t *)obj;
    for (size_t i = 0; i < lst->length && !*found; i++) {
        eval_var_ref_walk(ast, am_list_get(ast->alloc, lst, i), varid, found);
    }
}

static int32_t eval_var_is_referenced(am_ast_t *evalee, am_varid_t varid) {
    if (!evalee) return 0;
    am_handle_t top = am_ast_get_top_node_handle(evalee);
    if (top == AM_HANDLE_NULL) return 0;
    int32_t found = 0;
    eval_var_ref_walk(evalee, am_make_value_of_handle(top), varid, &found);
    return found;
}


// 建立 varid 映射：
// - GLOBAL_FREE：在 proc->var_top 中通过 proc->var_arn_mapping 查找原形，映射到对应 proc varid
// - 其他：插入 proc->var_vocab，并同步 proc->var_type
static int32_t eval_build_var_mapping(am_process_t *proc, am_ast_t *evalee, am_map_t **mapping) {
    if (!proc || !evalee || !mapping) return -1;
    size_t count = evalee->var_vocab ? evalee->var_vocab->length : 0;
    am_map_t *m = am_map_create(proc->vm_alloc, count > 0 ? count : 8);
    if (!m) return -1;

    size_t old_var_vocab_len = proc->var_vocab ? proc->var_vocab->length : 0;

    for (size_t i = 0; i < count; i++) {
        am_varid_t old_varid = (am_varid_t)i;
        am_value_t type_val = am_list_get(evalee->alloc, evalee->var_type, i);
        am_uint_t type = am_value_is_uint(type_val) ? am_value_to_uint(type_val) : AM_VAR_TYPE_OLD;
        am_varid_t new_varid = SIZE_MAX;

        if (type == AM_VAR_TYPE_GLOBAL_FREE) {
            wchar_t *name = am_vocab_get(evalee->alloc, evalee->var_vocab, &old_varid);
            if (!name) {
                am_map_destroy(proc->vm_alloc, m);
                return -1;
            }
            size_t top_len = proc->var_top ? proc->var_top->length : 0;
            for (size_t j = 0; j < top_len; j++) {
                am_value_t top_val = am_list_get(proc->vm_alloc, proc->var_top, j);
                if (!am_value_is_varid(top_val)) continue;
                am_varid_t top_varid = am_value_to_varid(top_val);
                am_value_t old_top_val = am_map_get(proc->vm_alloc, proc->var_arn_mapping,
                                                     am_make_value_of_varid(top_varid));
                if (!am_value_is_varid(old_top_val)) continue;
                am_varid_t old_top_varid = am_value_to_varid(old_top_val);
                wchar_t *old_name = am_vocab_get(proc->vm_alloc, proc->var_vocab, &old_top_varid);
                if (old_name && wcscmp(old_name, name) == 0) {
                    new_varid = top_varid;
                    break;
                }
            }
            if (new_varid == SIZE_MAX) {
                // 宏展开后残留的未引用条目（如宏关键字），不属于未定义变量错误，跳过
                if (!eval_var_is_referenced(evalee, old_varid)) continue;
                fprintf(stderr, "[System.eval] 未定义的变量：%ls\n", name);
                am_map_destroy(proc->vm_alloc, m);
                return -1;
            }
        }
        else {
            wchar_t *name = am_vocab_get(evalee->alloc, evalee->var_vocab, &old_varid);
            if (!name) {
                am_map_destroy(proc->vm_alloc, m);
                return -1;
            }
            size_t proc_idx = SIZE_MAX;
            proc->var_vocab = am_vocab_insert(proc->vm_alloc, proc->var_vocab, name, &proc_idx);
            if (!proc->var_vocab || proc_idx == SIZE_MAX) {
                am_map_destroy(proc->vm_alloc, m);
                return -1;
            }
            new_varid = (am_varid_t)proc_idx;

            // 仅当真正新增 varid 时才写入 var_type，避免覆盖已有 proc 变量类型
            if (proc_idx >= old_var_vocab_len) {
                while (proc->var_type->length <= proc_idx) {
                    am_list_t *vt = am_list_push(proc->vm_alloc, proc->var_type,
                                                  am_make_value_of_uint(AM_VAR_TYPE_OLD));
                    if (!vt) {
                        am_map_destroy(proc->vm_alloc, m);
                        return -1;
                    }
                    proc->var_type = vt;
                }
                if (am_list_set(proc->vm_alloc, proc->var_type, proc_idx,
                                am_make_value_of_uint(type)) != 0) {
                    am_map_destroy(proc->vm_alloc, m);
                    return -1;
                }
            }
        }

        am_map_t *new_m = am_map_set(proc->vm_alloc, m,
                                      am_make_value_of_varid(old_varid),
                                      am_make_value_of_varid(new_varid));
        if (!new_m) {
            am_map_destroy(proc->vm_alloc, m);
            return -1;
        }
        m = new_m;
    }
    *mapping = m;
    return 0;
}


// 编译后，为编译器引入的临时变量（ILTEMP）补充映射。
// 这些 varid 在 eval_build_var_mapping 之后才产生，因此需要额外处理。
static int32_t eval_extend_var_mapping_for_temps(am_process_t *proc, am_ast_t *evalee,
                                                  am_map_t **mapping, size_t old_evalee_var_vocab_len) {
    if (!proc || !evalee || !mapping || !*mapping) return -1;
    size_t count = evalee->var_vocab ? evalee->var_vocab->length : 0;
    if (old_evalee_var_vocab_len >= count) return 0;

    size_t old_proc_var_vocab_len = proc->var_vocab ? proc->var_vocab->length : 0;
    am_map_t *m = *mapping;

    // 用于保证每次 eval 的临时变量在 proc 中都有唯一的 varid，避免 call/cc 等续体
    // 与后续 eval 生成的同名临时变量发生 slot 冲突。
    static size_t eval_temp_seq = 0;

    for (size_t i = old_evalee_var_vocab_len; i < count; i++) {
        am_varid_t old_varid = (am_varid_t)i;
        am_value_t type_val = am_list_get(evalee->alloc, evalee->var_type, i);
        am_uint_t type = am_value_is_uint(type_val) ? am_value_to_uint(type_val) : AM_VAR_TYPE_OLD;

        // 仅处理编译期引入的 ILTEMP；其余类型应在编译前完成映射
        if (type != AM_VAR_TYPE_ILTEMP) {
            fprintf(stderr, "[System.eval] 未预期的编译后变量类型：varid=%zu type=%u\n",
                    (size_t)i, (unsigned int)type);
            return -1;
        }

        wchar_t *name = am_vocab_get(evalee->alloc, evalee->var_vocab, &old_varid);
        if (!name) return -1;

        // 生成唯一名称，避免不同 eval 调用的临时变量共享 proc varid
        wchar_t unique_name[256];
        int n = swprintf(unique_name, 256, L"__eval_%zu_%ls", eval_temp_seq++, name);
        if (n <= 0 || (size_t)n >= 256) return -1;

        size_t proc_idx = SIZE_MAX;
        proc->var_vocab = am_vocab_insert(proc->vm_alloc, proc->var_vocab, unique_name, &proc_idx);
        if (!proc->var_vocab || proc_idx == SIZE_MAX) return -1;

        if (proc_idx >= old_proc_var_vocab_len) {
            while (proc->var_type->length <= proc_idx) {
                am_list_t *vt = am_list_push(proc->vm_alloc, proc->var_type,
                                              am_make_value_of_uint(AM_VAR_TYPE_OLD));
                if (!vt) return -1;
                proc->var_type = vt;
            }
            if (am_list_set(proc->vm_alloc, proc->var_type, proc_idx,
                            am_make_value_of_uint(type)) != 0) return -1;
        }

        am_map_t *new_m = am_map_set(proc->vm_alloc, m,
                                      am_make_value_of_varid(old_varid),
                                      am_make_value_of_varid((am_varid_t)proc_idx));
        if (!new_m) return -1;
        m = new_m;
    }

    *mapping = m;
    return 0;
}


// 迁移 natives：把 evalee 中声明的 native 库记录合并到 proc->natives
static int32_t eval_migrate_natives(am_process_t *proc, am_ast_t *evalee, am_map_t *var_mapping) {
    if (!proc || !evalee || !evalee->natives || !var_mapping) return 0;

    size_t count = am_map_length(evalee->alloc, evalee->natives);
    am_value_t *keys = am_map_keys(evalee->alloc, evalee->natives);
    if (!keys && count > 0) return -1;

    for (size_t i = 0; i < count; i++) {
        am_value_t old_varid_val = keys[i];
        am_value_t new_varid_val = am_map_get(proc->vm_alloc, var_mapping, old_varid_val);
        if (!am_value_is_varid(new_varid_val)) continue;

        am_value_t old_handle_val = am_map_get(evalee->alloc, evalee->natives, old_varid_val);
        am_handle_t new_handle = AM_HANDLE_NULL;

        if (am_value_is_handle(old_handle_val)) {
            am_handle_t old_h = am_value_to_handle(old_handle_val);
            am_value_t obj_val = am_heap_get(evalee->alloc, evalee->alloc, evalee->nodes, old_h);
            if (am_value_is_ptr(obj_val)) {
                am_object_t *obj = am_value_to_ptr(obj_val);
                if (obj->type == AM_OBJECT_TYPE_WSTRING) {
                    am_wstring_t *new_ws = am_wstring_copy(proc->heap_alloc, (am_wstring_t *)obj);
                    if (new_ws) {
                        new_handle = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
                        if (new_handle != AM_HANDLE_NULL) {
                            if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, new_handle,
                                            am_make_value_of_ptr((am_object_t *)new_ws)) != 0) {
                                am_wstring_destroy(proc->heap_alloc, new_ws);
                                new_handle = AM_HANDLE_NULL;
                            } else {
                                am_object_set_static((am_object_t *)new_ws, 0);
                            }
                        } else {
                            am_wstring_destroy(proc->heap_alloc, new_ws);
                        }
                    }
                }
            }
        }

        am_map_t *new_m = am_map_set(proc->vm_alloc, proc->natives,
                                      new_varid_val,
                                      new_handle != AM_HANDLE_NULL
                                          ? am_make_value_of_handle(new_handle)
                                          : AM_VALUE_HANDLE_NULL);
        if (!new_m) {
            am_free(evalee->alloc, keys);
            return -1;
        }
        proc->natives = new_m;
    }
    am_free(evalee->alloc, keys);
    return 0;
}


// 将 symbol 映射应用到 evalee 的所有 list 节点 children
// varid 保持 evalee 原值，在 IL 生成后再统一 remap，以便编译器仍能用 evalee->var_vocab 解析名称
static int32_t eval_apply_symbol_mapping(am_ast_t *evalee, am_map_t *symbol_mapping) {
    if (!evalee || !evalee->nodes) return 0;

    size_t count = am_map_length(evalee->alloc, evalee->nodes->table);
    am_value_t *keys = am_map_keys(evalee->alloc, evalee->nodes->table);
    if (!keys && count > 0) return -1;

    for (size_t i = 0; i < count; i++) {
        am_handle_t hd = am_value_to_handle(keys[i]);
        am_value_t val = am_heap_get(evalee->alloc, evalee->alloc, evalee->nodes, hd);
        if (!am_value_is_ptr(val)) continue;
        am_object_t *obj = am_value_to_ptr(val);
        if (obj->type != AM_OBJECT_TYPE_LIST) continue;

        am_list_t *lst = (am_list_t *)obj;
        for (size_t j = 0; j < lst->length; j++) {
            am_value_t child = am_list_get(evalee->alloc, lst, j);
            if (am_value_is_symbol(child)) {
                am_value_t new_child = am_map_get(evalee->alloc, symbol_mapping, child);
                if (am_value_is_symbol(new_child) && new_child != child) {
                    if (am_list_set(evalee->alloc, lst, j, new_child) != 0) {
                        am_free(evalee->alloc, keys);
                        return -1;
                    }
                }
            }
        }
    }
    am_free(evalee->alloc, keys);
    return 0;
}


// 把 evalee 临时堆中的对象递归深拷贝到 proc->heap，并建立 handle 映射
typedef struct {
    am_allocator_t *vm_alloc;
    am_allocator_t *heap_alloc;
    am_ast_t *evalee;
    am_process_t *proc;
    am_map_t *handle_mapping;
} eval_copy_ctx_t;

static am_handle_t eval_copy_object_to_proc(eval_copy_ctx_t *ctx, am_handle_t old_handle) {
    if (old_handle == AM_HANDLE_NULL) return AM_HANDLE_NULL;

    am_value_t mapped = am_map_get(ctx->vm_alloc, ctx->handle_mapping,
                                    am_make_value_of_handle(old_handle));
    if (am_value_is_handle(mapped)) return am_value_to_handle(mapped);

    am_value_t old_val = am_heap_get(ctx->evalee->alloc, ctx->evalee->alloc,
                                      ctx->evalee->nodes, old_handle);
    if (!am_value_is_ptr(old_val)) return AM_HANDLE_NULL;

    am_object_t *obj = am_value_to_ptr(old_val);
    am_handle_t new_handle = AM_HANDLE_NULL;
    am_object_t *new_obj = NULL;

    if (obj->type == AM_OBJECT_TYPE_WSTRING) {
        am_wstring_t *new_ws = am_wstring_copy(ctx->heap_alloc, (am_wstring_t *)obj);
        if (!new_ws) return AM_HANDLE_NULL;
        new_obj = (am_object_t *)new_ws;
    }
    else if (obj->type == AM_OBJECT_TYPE_LIST) {
        am_list_t *old_lst = (am_list_t *)obj;
        am_list_t *new_lst = am_list_copy(ctx->heap_alloc, old_lst);
        if (!new_lst) return AM_HANDLE_NULL;

        for (size_t i = 0; i < new_lst->length; i++) {
            am_value_t child = new_lst->children[i];
            if (am_value_is_handle(child)) {
                am_handle_t old_child_h = am_value_to_handle(child);
                am_handle_t new_child_h = eval_copy_object_to_proc(ctx, old_child_h);
                if (new_child_h == AM_HANDLE_NULL && old_child_h != AM_HANDLE_NULL) {
                    am_list_destroy(ctx->heap_alloc, new_lst);
                    return AM_HANDLE_NULL;
                }
                new_lst->children[i] = am_make_value_of_handle(new_child_h);
            }
        }
        new_obj = (am_object_t *)new_lst;
    }
    else {
        return AM_HANDLE_NULL;
    }

    new_handle = am_heap_alloc_handle(ctx->vm_alloc, ctx->heap_alloc, ctx->proc->heap);
    if (new_handle == AM_HANDLE_NULL) {
        if (new_obj->type == AM_OBJECT_TYPE_WSTRING) {
            am_wstring_destroy(ctx->heap_alloc, (am_wstring_t *)new_obj);
        } else {
            am_list_destroy(ctx->heap_alloc, (am_list_t *)new_obj);
        }
        return AM_HANDLE_NULL;
    }

    if (am_heap_set(ctx->vm_alloc, ctx->heap_alloc, ctx->proc->heap, new_handle,
                    am_make_value_of_ptr(new_obj)) != 0) {
        if (new_obj->type == AM_OBJECT_TYPE_WSTRING) {
            am_wstring_destroy(ctx->heap_alloc, (am_wstring_t *)new_obj);
        } else {
            am_list_destroy(ctx->heap_alloc, (am_list_t *)new_obj);
        }
        am_heap_free_handle(ctx->vm_alloc, ctx->heap_alloc, ctx->proc->heap, new_handle);
        return AM_HANDLE_NULL;
    }

    am_object_set_static(new_obj, 0);

    am_map_t *new_m = am_map_set(ctx->vm_alloc, ctx->handle_mapping,
                                  am_make_value_of_handle(old_handle),
                                  am_make_value_of_handle(new_handle));
    if (!new_m) return AM_HANDLE_NULL;
    ctx->handle_mapping = new_m;

    return new_handle;
}


// 扫描 evalee IL，把 varid/handle 操作数分别映射到 proc 的 varid 和 proc->heap
static int32_t eval_remap_il_operands(eval_copy_ctx_t *ctx, am_map_t *var_mapping,
                                       am_instruction_t *ilcode, am_iaddr_t length) {
    if (!ctx || !var_mapping || !ilcode) return -1;
    for (am_iaddr_t i = 0; i < length; i++) {
        am_value_t operand = ilcode[i].operand;
        if (am_value_is_varid(operand)) {
            am_value_t new_val = am_map_get(ctx->vm_alloc, var_mapping, operand);
            if (!am_value_is_varid(new_val)) {
                fprintf(stderr, "[System.eval] IL varid 映射失败：iaddr=%zu\n", (size_t)i);
                return -1;
            }
            ilcode[i].operand = new_val;
        }
        else if (am_value_is_handle(operand)) {
            am_handle_t old_h = am_value_to_handle(operand);
            if (old_h == AM_HANDLE_NULL) continue;
            am_handle_t new_h = eval_copy_object_to_proc(ctx, old_h);
            if (new_h == AM_HANDLE_NULL) {
                fprintf(stderr, "[System.eval] IL handle 映射失败：iaddr=%zu\n", (size_t)i);
                return -1;
            }
            ilcode[i].operand = am_make_value_of_handle(new_h);
        }
    }
    return 0;
}


// 将 evalee IL 追加到 proc->ilcode 尾部
static int32_t eval_append_ilcode(am_process_t *proc, am_instruction_t *new_ilcode, am_iaddr_t new_length) {
    if (!proc || !new_ilcode || new_length == 0) return -1;
    am_iaddr_t old_length = proc->ilcode_length;
    am_instruction_t *combined = (am_instruction_t *)am_realloc(
        proc->vm_alloc, proc->ilcode,
        (old_length + new_length) * sizeof(am_instruction_t));
    if (!combined) return -1;
    proc->ilcode = combined;
    memcpy(proc->ilcode + old_length, new_ilcode, new_length * sizeof(am_instruction_t));
    proc->ilcode_length = old_length + new_length;
    return 0;
}


// 创建 System.eval 清理记录。
// 记录为 6 元素列表：
//   [ret_iaddr, saved_opstack_len, first_handle, last_handle, old_ilcode_length, old_var_vocab_len]
// 该对象不标记为 static，但会设置 keepalive，避免在 evalcleanup 执行前被 GC 回收。
static am_handle_t eval_create_cleanup_record(am_process_t *proc, am_iaddr_t ret_iaddr,
                                              size_t saved_opstack_len,
                                              am_handle_t first_handle,
                                              am_handle_t last_handle,
                                              am_iaddr_t old_ilcode_length,
                                              size_t old_var_vocab_len) {
    if (!proc || !proc->heap) return AM_HANDLE_NULL;

    am_list_t *rec = am_list_create(proc->heap_alloc, 5, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!rec) return AM_HANDLE_NULL;

    rec = am_list_push(proc->heap_alloc, rec, am_make_value_of_iaddr(ret_iaddr));
    if (!rec) return AM_HANDLE_NULL;
    rec = am_list_push(proc->heap_alloc, rec, am_make_value_of_uint((am_uint_t)saved_opstack_len));
    if (!rec) return AM_HANDLE_NULL;
    rec = am_list_push(proc->heap_alloc, rec, am_make_value_of_handle(first_handle));
    if (!rec) return AM_HANDLE_NULL;
    rec = am_list_push(proc->heap_alloc, rec, am_make_value_of_handle(last_handle));
    if (!rec) return AM_HANDLE_NULL;
    rec = am_list_push(proc->heap_alloc, rec, am_make_value_of_uint((am_uint_t)old_ilcode_length));
    if (!rec) return AM_HANDLE_NULL;
    rec = am_list_push(proc->heap_alloc, rec, am_make_value_of_uint((am_uint_t)old_var_vocab_len));
    if (!rec) return AM_HANDLE_NULL;

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, rec);
        return AM_HANDLE_NULL;
    }

    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                    am_make_value_of_ptr((am_object_t *)rec)) != 0) {
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        am_list_destroy(proc->heap_alloc, rec);
        return AM_HANDLE_NULL;
    }

    am_object_set_keepalive((am_object_t *)rec, 0);
    return hd;
}


// (System.eval codestr:String) : void
// 运行时动态编译并执行 Scheme 代码字符串。仅捕获当前进程的顶级变量绑定。
int32_t am_native_System_eval(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc || !proc->heap) return -1;

    // 弹出代码字符串
    am_value_t code_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(code_val)) return -1;
    am_handle_t code_h = am_value_to_handle(code_val);
    am_value_t code_obj = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, code_h);
    if (!am_value_is_ptr(code_obj)) return -1;
    am_object_t *obj = am_value_to_ptr(code_obj);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return -1;

    wchar_t *code_buf = eval_wstring_to_vm_buffer(proc, (am_wstring_t *)obj);
    if (!code_buf) return -1;

    wchar_t *path_buf = eval_make_path(proc->vm_alloc, rt->working_dir);
    if (!path_buf) {
        am_free(proc->vm_alloc, code_buf);
        return -1;
    }

    // 将 eval 代码包装成 Parser 要求的顶层 thunk 形式：((lambda () <code>))
    const wchar_t *prefix = L"((lambda () ";
    const wchar_t *suffix = L"))";
    size_t code_len = wcslen(code_buf);
    size_t wrapped_len = wcslen(prefix) + code_len + wcslen(suffix);
    wchar_t *wrapped_code = (wchar_t *)am_malloc(proc->vm_alloc,
                                                  (wrapped_len + 1) * sizeof(wchar_t));
    if (!wrapped_code) {
        am_free(proc->vm_alloc, code_buf);
        am_free(proc->vm_alloc, path_buf);
        return -1;
    }
    wcscpy(wrapped_code, prefix);
    wcscat(wrapped_code, code_buf);
    wcscat(wrapped_code, suffix);

    // 解析为孤立 AST，保留 GLOBAL_FREE
    am_ast_t *evalee = am_parse(proc->vm_alloc, wrapped_code, path_buf, 1);
    if (!evalee) {
        am_free(proc->vm_alloc, wrapped_code);
        am_free(proc->vm_alloc, code_buf);
        am_free(proc->vm_alloc, path_buf);
        return -1;
    }

    am_map_t *symbol_mapping = NULL;
    am_map_t *var_mapping = NULL;
    am_module_t *mod = NULL;
    eval_copy_ctx_t copy_ctx = { 0 };
    am_handle_t cleanup_rec_h = AM_HANDLE_NULL;

    // 记录 eval 调用前的操作数栈高度，用于执行完毕后恢复
    size_t saved_opstack_len = am_process_length_of_opstack(proc);

    // 记录 eval 前 proc 的 var_vocab 长度，用于事后清理编译期引入的 ILTEMP 临时变量
    size_t old_var_vocab_len = proc->var_vocab ? proc->var_vocab->length : 0;

    // 建立 symbol / var 映射
    if (eval_build_symbol_mapping(proc, evalee, &symbol_mapping) != 0) goto fail;
    if (eval_build_var_mapping(proc, evalee, &var_mapping) != 0) goto fail;

    // 迁移 natives 以支持 eval 中的 native 调用
    if (eval_migrate_natives(proc, evalee, var_mapping) != 0) goto fail;

    // 应用 symbol 映射到 evalee AST（varid 保持原值）
    if (eval_apply_symbol_mapping(evalee, symbol_mapping) != 0) goto fail;

    // 计算偏移和返回地址
    am_iaddr_t offset = proc->ilcode_length;
    am_iaddr_t ret = proc->PC + 1;

    // 记录 eval 即将在 proc->heap 中分配的首个把柄，用于事后清理静态标记
    am_handle_t eval_first_handle = proc->heap->handle_counter;

    // 记录编译前 evalee 的 var_vocab 长度，编译器可能引入 ILTEMP 临时变量
    size_t old_evalee_var_vocab_len = evalee->var_vocab ? evalee->var_vocab->length : 0;

    // 编译 evalee AST
    mod = am_compile(evalee, offset, ret);
    if (!mod) {
        fprintf(stderr, "[System.eval] 编译失败\n");
        goto fail;
    }

    // 为编译期引入的临时变量补充 varid 映射
    if (eval_extend_var_mapping_for_temps(proc, evalee, &var_mapping,
                                           old_evalee_var_vocab_len) != 0) goto fail;

    // 建立 handle 映射并把 IL 中 push 的 handle 映射到 proc->heap
    copy_ctx.vm_alloc = proc->vm_alloc;
    copy_ctx.heap_alloc = proc->heap_alloc;
    copy_ctx.evalee = evalee;
    copy_ctx.proc = proc;
    copy_ctx.handle_mapping = am_map_create(proc->vm_alloc, 16);
    if (!copy_ctx.handle_mapping) goto fail;

    if (eval_remap_il_operands(&copy_ctx, var_mapping, mod->ilcode, mod->ilcode_length) != 0) goto fail;

    // 记录 eval 在 proc->heap 中分配的最后一个把柄
    am_handle_t eval_last_handle = (proc->heap->handle_counter > 0)
                                       ? proc->heap->handle_counter - 1
                                       : 0;

    // 创建清理记录，并在 IL 中将返回处的 goto 替换为 evalcleanup
    cleanup_rec_h = eval_create_cleanup_record(proc, ret, saved_opstack_len,
                                               eval_first_handle, eval_last_handle,
                                               offset, old_var_vocab_len);
    if (cleanup_rec_h == AM_HANDLE_NULL) goto fail;

    if (mod->ilcode_length < 2 || mod->ilcode[1].opcode != AM_VM_OP_goto) {
        fprintf(stderr, "[System.eval] 无法插入 evalcleanup 指令\n");
        goto fail;
    }
    mod->ilcode[1].opcode = AM_VM_OP_evalcleanup;
    mod->ilcode[1].operand = am_make_value_of_handle(cleanup_rec_h);

    // 追加 IL 到 proc
    if (eval_append_ilcode(proc, mod->ilcode, mod->ilcode_length) != 0) {
        fprintf(stderr, "[System.eval] 追加 IL 代码失败\n");
        goto fail;
    }

    // 跳转到 evalee 入口
    proc->PC = offset;

    // 清理临时资源：mod 的 ilcode 已复制到 proc，释放原始块，再释放 mod 结构体
    if (mod->ilcode) {
        am_free(proc->vm_alloc, mod->ilcode);
        mod->ilcode = NULL;
    }
    am_free(proc->vm_alloc, mod);
    am_map_destroy(proc->vm_alloc, copy_ctx.handle_mapping);
    am_map_destroy(proc->vm_alloc, symbol_mapping);
    am_map_destroy(proc->vm_alloc, var_mapping);
    am_ast_destroy(evalee);
    am_free(proc->vm_alloc, wrapped_code);
    am_free(proc->vm_alloc, code_buf);
    am_free(proc->vm_alloc, path_buf);
    return 0;

fail:
    if (cleanup_rec_h != AM_HANDLE_NULL) {
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, cleanup_rec_h);
    }
    if (copy_ctx.handle_mapping) am_map_destroy(proc->vm_alloc, copy_ctx.handle_mapping);
    if (symbol_mapping) am_map_destroy(proc->vm_alloc, symbol_mapping);
    if (var_mapping) am_map_destroy(proc->vm_alloc, var_mapping);
    if (mod) {
        if (mod->ilcode) {
            am_free(proc->vm_alloc, mod->ilcode);
            mod->ilcode = NULL;
        }
        am_free(proc->vm_alloc, mod);
    }
    am_ast_destroy(evalee);
    am_free(proc->vm_alloc, wrapped_code);
    am_free(proc->vm_alloc, code_buf);
    am_free(proc->vm_alloc, path_buf);
    return -1;
}


static const am_native_func_entry_t am_native_System_funcs[] = {
    { L"exit",         am_native_System_exit },
    { L"kill",         am_native_System_kill },
    { L"exec",         am_native_System_exec },
    { L"set_timeout",  am_native_System_set_timeout },
    { L"set_interval", am_native_System_set_interval },
    { L"clear_timeout",am_native_System_clear_timeout },
    { L"clear_interval",am_native_System_clear_interval },
    { L"timestamp",    am_native_System_timestamp },
    { L"memstat",      am_native_System_memstat },
    { L"gc",           am_native_System_gc },
    { L"fork",         am_native_System_fork },
    { L"eval",         am_native_System_eval },
    { L"make_queue",   am_native_System_make_queue },
    { L"write",        am_native_System_write },
    { L"read",         am_native_System_read },
    { L"test",         am_native_System_test },
};

const am_native_lib_entry_t am_native_System_lib = {
    L"System",
    am_native_System_funcs,
    sizeof(am_native_System_funcs) / sizeof(am_native_System_funcs[0])
};
