#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "am_native_Table.h"


// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 从操作数栈中弹出一个 table 对象（am_map_t）。
// 成功返回 true，失败返回 false。
static bool native_pop_table(am_process_t *proc, am_map_t **out_map, am_handle_t *out_handle) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_MAP) return false;

    *out_map = (am_map_t *)obj;
    if (out_handle) *out_handle = hd;
    return true;
}


// 判断一个 am_value_t 是否是合法的 table key 类型（number、symbol、wstring handle）。
// 若是，返回 true 并通过 out_key 输出可用于 am_map_t 的 key。
// 对 number 进行归一化：非负 int 转为 uint；float 保持 float。
// 对 wstring 检查长度是否超过 AM_PROCESS_STRINDEX_MAX_LEN。
static bool native_pop_table_key(am_process_t *proc, am_value_t *out_key) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;

    // number：归一化非负 int 为 uint
    if (am_value_is_number(v)) {
        if (am_value_is_int(v)) {
            am_int_t iv = am_value_to_int(v);
            if (iv >= 0) {
                *out_key = am_make_value_of_uint((am_uint_t)iv);
            } else {
                *out_key = v;
            }
        } else {
            *out_key = v;
        }
        return true;
    }

    // symbol：直接作为 key
    if (am_value_is_symbol(v)) {
        *out_key = v;
        return true;
    }

    // wstring handle：检查长度限制
    if (am_value_is_handle(v)) {
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, am_value_to_handle(v));
        if (am_value_is_ptr(obj_val)) {
            am_object_t *obj = am_value_to_ptr(obj_val);
            if (obj->type == AM_OBJECT_TYPE_WSTRING) {
                am_wstring_t *ws = (am_wstring_t *)obj;
                if (ws->length > AM_PROCESS_STRINDEX_MAX_LEN) return false;
                *out_key = v;
                return true;
            }
        }
    }

    return false;
}


// 将 #undefined 压回操作数栈并前进 PC。
static int32_t native_push_undefined(am_process_t *proc) {
    if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// 将 boolean 值压回操作数栈并前进 PC。
static int32_t native_push_bool(am_process_t *proc, bool value) {
    if (am_process_push_operand(proc, am_make_value_of_boolean(value)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// ===============================================================================
// Native 函数实现
// ===============================================================================

// (Table.make) : Table
int32_t am_native_Table_make(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!proc || !proc->heap_alloc) return -1;

    am_map_t *map = am_map_create(proc->heap_alloc, 16);
    if (!map) return -1;

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) {
        am_map_destroy(proc->heap_alloc, map);
        return -1;
    }

    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                    am_make_value_of_ptr((am_object_t *)map)) != 0) {
        am_map_destroy(proc->heap_alloc, map);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return -1;
    }

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (Table.set tbl key value) : #undefined
int32_t am_native_Table_set(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_value_t value = am_process_pop_operand(proc);
    if (value == (am_value_t)UINTPTR_MAX) return -1;

    am_value_t key;
    if (!native_pop_table_key(proc, &key)) return -1;

    am_map_t *map = NULL;
    am_handle_t tbl_h = AM_HANDLE_NULL;
    if (!native_pop_table(proc, &map, &tbl_h)) return -1;

    if (value == AM_VALUE_UNDEFINED) {
        // value 为 #undefined 时语义上相当于删除
        am_map_delete(proc->heap_alloc, map, key);
    } else {
        am_map_t *new_map = am_map_set(proc->heap_alloc, map, key, value);
        if (!new_map) return -1;
        if (new_map != map) {
            // map 对象已扩容并重新分配，需要更新 heap 中 handle 的绑定
            if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, tbl_h,
                            am_make_value_of_ptr((am_object_t *)new_map)) != 0) {
                am_map_destroy(proc->heap_alloc, new_map);
                return -1;
            }
        }
    }

    return native_push_undefined(proc);
}


// (Table.get tbl key) : value | #undefined
int32_t am_native_Table_get(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_value_t key;
    if (!native_pop_table_key(proc, &key)) return -1;

    am_map_t *map = NULL;
    if (!native_pop_table(proc, &map, NULL)) return -1;

    am_value_t result;
    if (am_map_contains(proc->heap_alloc, map, key) == 0) {
        result = am_map_get(proc->heap_alloc, map, key);
    } else {
        result = AM_VALUE_UNDEFINED;
    }

    if (am_process_push_operand(proc, result) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (Table.keys tbl) : List
int32_t am_native_Table_keys(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_map_t *map = NULL;
    if (!native_pop_table(proc, &map, NULL)) return -1;

    size_t count = am_map_length(proc->heap_alloc, map);
    am_list_t *lst = am_list_create(proc->heap_alloc, count > 0 ? count : 1,
                                    AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!lst) return -1;

    if (count > 0) {
        am_value_t *keys = am_map_keys(proc->vm_alloc, map);
        if (!keys) {
            am_list_destroy(proc->heap_alloc, lst);
            return -1;
        }
        for (size_t i = 0; i < count; i++) {
            am_list_t *new_lst = am_list_push(proc->heap_alloc, lst, keys[i]);
            if (!new_lst) {
                am_free(proc->vm_alloc, keys);
                am_list_destroy(proc->heap_alloc, lst);
                return -1;
            }
            lst = new_lst;
        }
        am_free(proc->vm_alloc, keys);
    }

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, lst);
        return -1;
    }

    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return -1;
    }

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// (Table.contains tbl key) : Boolean
int32_t am_native_Table_contains(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_value_t key;
    if (!native_pop_table_key(proc, &key)) return -1;

    am_map_t *map = NULL;
    if (!native_pop_table(proc, &map, NULL)) return -1;

    bool contains = (am_map_contains(proc->heap_alloc, map, key) == 0);
    return native_push_bool(proc, contains);
}


// (Table.delete tbl key) : #undefined
int32_t am_native_Table_delete(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_value_t key;
    if (!native_pop_table_key(proc, &key)) return -1;

    am_map_t *map = NULL;
    if (!native_pop_table(proc, &map, NULL)) return -1;

    am_map_delete(proc->heap_alloc, map, key);
    return native_push_undefined(proc);
}


// (Table.length tbl) : Number
int32_t am_native_Table_length(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_map_t *map = NULL;
    if (!native_pop_table(proc, &map, NULL)) return -1;

    size_t len = am_map_length(proc->heap_alloc, map);
    if (am_process_push_operand(proc, am_make_value_of_float((am_float_t)len)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// ===============================================================================
// Native 库表
// ===============================================================================

static const am_native_func_entry_t am_native_Table_funcs[] = {
    { L"make",     am_native_Table_make },
    { L"set",      am_native_Table_set },
    { L"get",      am_native_Table_get },
    { L"keys",     am_native_Table_keys },
    { L"contains", am_native_Table_contains },
    { L"delete",   am_native_Table_delete },
    { L"length",   am_native_Table_length },
};

const am_native_lib_entry_t am_native_Table_lib = {
    L"Table",
    am_native_Table_funcs,
    sizeof(am_native_Table_funcs) / sizeof(am_native_Table_funcs[0])
};
