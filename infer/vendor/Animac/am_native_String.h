#ifndef __AM_NATIVE_STRING_H__
#define __AM_NATIVE_STRING_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"


////////////////////////////////////////////////////////////////////////////
//  Native Library : String
////////////////////////////////////////////////////////////////////////////

// 由 src/native_String.c 定义
extern const am_native_lib_entry_t am_native_String_lib;


// (String.length str:String) : Number
int32_t am_native_String_length(am_runtime_t *rt, am_process_t *proc);

// (String.atom_to_string x:Boolean|Number|Symbol) : String
int32_t am_native_String_atom_to_string(am_runtime_t *rt, am_process_t *proc);

// (String.concat str1:String str2:String) : String
int32_t am_native_String_concat(am_runtime_t *rt, am_process_t *proc);

// (String.charCodeAt index:Number str:String) : Number
int32_t am_native_String_charCodeAt(am_runtime_t *rt, am_process_t *proc);

// (String.fromCharCode charcode:Number) : String
int32_t am_native_String_fromCharCode(am_runtime_t *rt, am_process_t *proc);

// (String.slice str:String start:Number end:Number) : String
int32_t am_native_String_slice(am_runtime_t *rt, am_process_t *proc);

// (String.equals str1:String str2:String) : Boolean
int32_t am_native_String_equals(am_runtime_t *rt, am_process_t *proc);

// (String.charAt str:String index:Number) : String
int32_t am_native_String_charAt(am_runtime_t *rt, am_process_t *proc);

// (String.parseNumber x:String) : Number|#undefined
int32_t am_native_String_parseNumber(am_runtime_t *rt, am_process_t *proc);



#ifdef __cplusplus
}
#endif

#endif
