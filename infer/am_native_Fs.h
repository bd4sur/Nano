#ifndef __AM_NATIVE_FS_H__
#define __AM_NATIVE_FS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Fs —— 灵机（Animac）文件系统本地库
//
//  为什么需要本库：
//    此前「灵机应用程序」是**只读世界**——数据要么硬编码在 .scm 里，要么经
//    (import 别名 "app/lib/xxx.scm") 以「数据模块」的形式链接进来（星历/星表
//    就是这么做的）。于是所有需要「上次留下的东西」的 App 全部不可写：高分榜、
//    存档、便签、待办、关卡编辑器、日志……本库把 hal_fs 的读写原语以最小面暴露
//    给 Scheme，让 App 能留下数据（浏览器目标还会把写入同步进 IndexedDB，
//    见下方「持久化」一节）。
//
//  能力刻意收窄（与 hal_net 同一取舍）：只做**文本文件**的整读整写 + 目录枚举；
//  不做随机访问、不做二进制、不做删除/改名（hal_fs 目前也没有）。
//
//  路径规则（易错，务必按此书写）：
//    · 以 '/' 开头的路径 = **宿主绝对路径**，原样使用；
//    · 其他路径 = **相对业务资源根**（PLATFORM_ROOT_DIR）：例如 "data/note.txt"
//      在桌面/WASM 上落到 /home/bd4sur/linglong/data/note.txt，在 ESP32 上落到
//      SD 卡根下的 /data/note.txt；
//    · 含 ".." 的路径一律拒绝（不允许穿越出资源根）。
//    App 请一律用相对路径，才能在「桌面 / 浏览器 / 掌机」三处通用。
//
//  Scheme 接口（全部值型，压 1 值 + step 1 次；失败按软失败约定返回 #null / #f）：
//
//    (Fs.exists path)      : Boolean          —— 文件或目录是否存在
//    (Fs.is_dir path)      : Boolean          —— 是否目录
//    (Fs.size path)        : Number | #null   —— 文件字节数（不存在/是目录 → #null）
//    (Fs.list dir)         : List | #null     —— 目录条目名（不含路径，按名字升序）；
//                                                目录打不开 → #null
//    (Fs.mkdir path)       : Boolean          —— 创建目录（已存在视为成功）
//    (Fs.read path)        : String | #null   —— 整文件读入并按 UTF-8 解码为字符串；
//                                                不存在 / 读失败 / 超过
//                                                AM_NATIVE_FS_READ_MAX → #null
//    (Fs.write path text)  : Boolean          —— 整文件写入（覆盖；自动创建父目录）
//    (Fs.append path text) : Boolean          —— 追加写入（文件不存在等价于写入）
//
//  读入上限：AM_NATIVE_FS_READ_MAX（128 KB）。超限返回 #null 而不是截断——
//  截断会让 App 在不知情的情况下损坏数据；App 可先用 (Fs.size path) 判断。
//
//  持久化（WASM 目标）：Emscripten 的 MEMFS 是内存文件系统，刷新页面即丢失。
//  因此写入成功后本库会调用一个**可选弱钩子** am_native_fs_on_write()；WASM 目标
//  由 hal_fs_persist_wasm.c 提供实现，把文件同步进浏览器 IndexedDB（nano-store），
//  下次启动由 loader.js 以「本机来源」挂载（与用户上传的资源同一通路）。
//  未提供钩子的平台（桌面/掌机）不产生任何额外动作。
//
//  与宿主 GUI 的关系：hal_fs 的随机访问 API 是**全局单句柄**（ebook/dict 用），
//  本库的 exists/size/read 会在内部短暂 open/close 该句柄。灵机 App 运行期间宿主
//  GUI 不读文件（单前台 App 模型），故安全；若将来出现并发读，需给 hal_fs 换
//  成多句柄。
//
//  Scheme 侧用法：源码中声明 (native Fs) 后以 Fs.函数名 调用。
////////////////////////////////////////////////////////////////////////////

// 单次读入的字节上限（超出返回 #null，见上）
#define AM_NATIVE_FS_READ_MAX (128u * 1024u)

// 由本文件定义
extern const am_native_lib_entry_t am_native_Fs_lib;

// (Fs.exists path:String) : Boolean
int32_t am_native_Fs_exists(am_runtime_t *rt, am_process_t *proc);

// (Fs.is_dir path:String) : Boolean
int32_t am_native_Fs_is_dir(am_runtime_t *rt, am_process_t *proc);

// (Fs.size path:String) : Number | #null
int32_t am_native_Fs_size(am_runtime_t *rt, am_process_t *proc);

// (Fs.list dir:String) : List | #null
int32_t am_native_Fs_list(am_runtime_t *rt, am_process_t *proc);

// (Fs.mkdir path:String) : Boolean
int32_t am_native_Fs_mkdir(am_runtime_t *rt, am_process_t *proc);

// (Fs.read path:String) : String | #null
int32_t am_native_Fs_read(am_runtime_t *rt, am_process_t *proc);

// (Fs.write path:String text:String) : Boolean
int32_t am_native_Fs_write(am_runtime_t *rt, am_process_t *proc);

// (Fs.append path:String text:String) : Boolean
int32_t am_native_Fs_append(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif