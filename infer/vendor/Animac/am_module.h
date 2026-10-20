#ifndef __AM_MODULE_H__
#define __AM_MODULE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

#include "am_object.h"
#include "am_ast.h"
#include "am_compiler.h"


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
