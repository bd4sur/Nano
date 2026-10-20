#ifndef __AM_MACRO_H__
#define __AM_MACRO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "am_ast.h"

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
