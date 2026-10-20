#ifndef __AM_COMPILER_H__
#define __AM_COMPILER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#include "am_ast.h"
#include "am_object.h"


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
