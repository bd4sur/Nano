#ifndef __AM_HOST_H__
#define __AM_HOST_H__

#include <stdint.h>
#include <wchar.h>

#include "animac.h"

#ifdef __cplusplus
extern "C" {
#endif

// 短时睡眠（毫秒）
void am_sleep_in_ms(uint64_t ms);

// 获取当前时间戳（毫秒）
uint64_t am_current_timestamp_in_ms();

// 读取文件内容（UTF-8），并转换为 wchar_t* 字符串
// 成功时返回动态分配的 wchar_t*（以 L'\0' 结尾），失败返回 NULL。调用者需用 free() 释放返回值。
wchar_t* am_read_file_to_wchar(char* filename);

// 链接器模块源码读取回调（am_linker_read_source_fn）的宿主侧默认实现：文件系统版。
// 将宽字符绝对路径转换为 UTF-8 后读取文件内容，并用 alloc 分配返回的源码字符串，
// 使读取到的代码纳入 allocator 管理（由链接器用 am_free 释放）。
// user_data 未使用，调用 am_link 时可传 NULL。
// 成功返回以 L'\0' 结尾的源码字符串；失败返回 NULL。
wchar_t *am_host_read_source_from_file(am_allocator_t *alloc, const wchar_t *abs_path, void *user_data);

// 从 Linux 格式的文件路径中提取文件所在目录的绝对路径
// 即最后一个 '/' 之前的内容，不包含末尾的 '/'
// 返回值：动态分配的字符串，调用者需 free()；失败或路径不含 '/' 时返回 NULL
char* am_path_dirname(const char *path);


// ===============================================================================
// 宿主内存分配
// ===============================================================================

void *am_host_calloc(size_t n, size_t sizeoftype);
void *am_host_malloc(size_t nbytes);
void *am_host_realloc(void *ptr, size_t n);
void  am_host_free(void *ptr);

// 宿主内存分配虚函数表（am_allocator_host_vtable_t）的默认实例，
// 成员即上述四个 am_host_* 参考实现，可直接传给 am_allocator_pool_create。
extern const am_allocator_host_vtable_t am_host_default_vtable;


// ===============================================================================
// 字符编码
// ===============================================================================

// 将 UTF-32 码点（wchar_t）数组转换为 UTF-8 字符串
// 注意：此函数假设 wchar_t 为 32 位（即 UTF-32）
uint32_t am_wcstombs(char *dest, const wchar_t *src, uint32_t dest_size);

// 将 UTF-8 字符串转换为 null-terminated 的 UTF-32 (wchar_t) 字符串
// 返回值：成功转换的 wchar_t 字符数量（不包括结尾的 L'\0'）
// 注意：dest 必须有至少 (length + 1) 个 wchar_t 的空间（最坏情况）
uint32_t am_mbstowcs(wchar_t *dest, const char *src, uint32_t length);


#ifdef __cplusplus
}
#endif

#endif
