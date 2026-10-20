#ifndef __ANIMAC_H__
#define __ANIMAC_H__

/* =============================================================================
 * Animac（灵机）解释器核心公开 API 伞形头文件 —— 本项目（玲珑）的 amalgam 适配层
 *
 * 【这个文件不是上游文件，是本项目唯一的自有适配文件，同步上游时不要覆盖】
 *
 * 上游的 include/animac.h 通过 #include 逐个登记 23 个核心头文件
 * （am_base.h / am_allocator.h / am_object.h / … / am_runtime.h / am_debug.h）。
 * 本项目采用上游的「单文件（amalgamation）集成形态」：核心的 23 个头文件与全部
 * 核心实现已被 amalgamate.sh 合并进 animac_core.h / animac_core.c，产物不再随附
 * 原始头文件。因此本文件把伞形头文件的收录语义等价地转发到合并产物：
 *
 *     animac.h  ==  include/animac.h  （逐个头文件列举）  ==  animac_core.h
 *
 * 两者靠同名 include guard（__AM_XXX_H__）互斥，与 animac_core.h 同时包含也不会重复定义。
 *
 * 维护约定：
 *   - 上游新增/调整核心头文件时，无需改本文件（合并产物的收录范围由上游
 *     amalgamate.sh 依据 include/animac.h 自动同步）；
 *   - 宿主适配（am_host.h）、本地库（am_native_*.h）、REPL（am_repl.h）按上游
 *     约定不并入单文件，由使用者按需另行包含。
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

#include "animac_core.h"

#ifdef __cplusplus
}
#endif

#endif