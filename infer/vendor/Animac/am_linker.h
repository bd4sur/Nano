#ifndef __AM_LINKER_H__
#define __AM_LINKER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <wchar.h>

#include "am_ast.h"
#include "am_allocator.h"


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
