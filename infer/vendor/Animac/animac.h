#ifndef __ANIMAC_H__
#define __ANIMAC_H__

/* =============================================================================
 * Animac（灵机）解释器核心公开 API 伞形头文件
 *
 * 嵌入解释器或编写上层工具（REPL、解释器入口、debugger 等）时，
 * 只需包含本头文件即可获得解释器核心的全部公开 API。
 *
 * 收录范围：解释器核心（基础设施、前端、运行时）。
 * 明确排除：宿主适配（host.h）、native 库实现（native_*.h）、
 *           终端呈现工具（highlight.h）、上层消费者自身（repl.h）。
 * 上层程序可按需另行包含上述被排除的头文件。
 *
 * 约定：新增解释器核心头文件时，必须登记进本文件对应分组。
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

// 基础设施
#include "am_base.h"
#include "am_allocator.h"
#include "am_object.h"
#include "am_map.h"
#include "am_list.h"
#include "am_wstring.h"
#include "am_vocab.h"
#include "am_heap.h"
#include "am_closure.h"
#include "am_continuation.h"
#include "am_scope.h"
#include "am_debug.h"

// 前端（词法/语法/宏/链接/编译/模块/JS 转换）
#include "am_lexer.h"
#include "am_ast.h"
#include "am_parser.h"
#include "am_macro.h"
#include "am_linker.h"
#include "am_compiler.h"
#include "am_module.h"
#include "am_js2scm.h"

// 运行时（进程/垃圾回收/虚拟机）
#include "am_process.h"
#include "am_gc.h"
#include "am_runtime.h"

#ifdef __cplusplus
}
#endif

#endif
