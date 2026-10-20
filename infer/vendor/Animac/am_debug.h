#ifndef __AM_DEBUG_H__
#define __AM_DEBUG_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <wchar.h>

#include "am_ast.h"
#include "am_object.h"
#include "am_compiler.h"


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
