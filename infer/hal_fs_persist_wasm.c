// hal_fs_persist_wasm.c —— Nano-WASM-GUI 的文件写入「落盘」钩子（Emscripten）
//
// 背景：WASM 目标的 hal_fs 是 hal_fs_linux.c（纯 POSIX），在浏览器里跑在
// Emscripten 的 MEMFS 上——写入是**内存文件系统**，刷新页面即丢失。而浏览器侧
// 早有一条持久化通路：webgui/store.js 的 IndexedDB（nano-store），用户上传的
// 资源就存在那里，loader.js 启动挂载时 store 优先于 HTTP 默认源。
//
// 因此 (native Fs) 的写入成功后，am_native_Fs.c 会调用可选弱钩子
// am_native_fs_on_write()；本文件就是 WASM 目标的实现：把「业务资源根相对路径」
// 与文件内容作为 Blob 写进 nano-store。下次启动时 loader.js 把它当作「本机来源」
// 的资源挂载回 MEMFS，于是 App 的存档/便签在浏览器里真正持久。
// 删除/改名同理，经可选弱钩子 am_native_fs_on_remove() 把旧键从 nano-store 删掉
// （目录树改名 = 逐文件重新 put + 删旧键，由 am_native_Fs.c 完成）。
//
// 约束与降级：
//   · 钩子是弱符号——其他平台不链接本文件即可，零改动、零开销；
//   · 页面未加载 store.js（例如 node 端回归台）时 nanoStorePut 不存在，直接返回；
//   · IndexedDB 写入是异步的，这里 fire-and-forget（App 不需要写盘结果；
//     失败只影响"下次启动是否还在"，不影响本次会话的读回，因为 MEMFS 已写好）；
//   · 路径在资源根之外（宿主绝对路径）时跳过——只持久化资源根内的文件。

#include <stddef.h>
#include <stdint.h>

#include <emscripten.h>

#include "platform.h"

// 把字节写入 nano-store（IndexedDB）。key = 资源根相对路径（与 manifest 的
// path 同一口径，如 "data/note.txt"）。
EM_JS(void, nano_fs_store_put_js, (const char *path_ptr, const uint8_t *data_ptr, int32_t size, const char *root_ptr), {
    if (typeof nanoStorePut !== "function") return;
    try {
        var abs = UTF8ToString(path_ptr);
        var root = UTF8ToString(root_ptr);
        var key = abs;
        if (root.length > 0 && abs.indexOf(root + "/") === 0) {
            key = abs.slice(root.length + 1);
        }
        else if (abs.charAt(0) === "/") {
            return; // 资源根之外：不持久化
        }
        var bytes = HEAPU8.subarray(data_ptr, data_ptr + size).slice(); // 复制：C 侧缓冲随后即释放
        var p = nanoStorePut(key, new Blob([bytes]));
        if (p && typeof p.catch === "function") p.catch(function () {});
    }
    catch (e) { /* 存储不可用：忽略（本次会话仍可读回） */ }
});

void am_native_fs_on_write(const char *path, const uint8_t *data, size_t size) {
    if (!path || (!data && size > 0)) return;
    nano_fs_store_put_js(path, data, (int32_t)size, PLATFORM_ROOT_DIR);
}

// ---------------------------------------------------------------------------
// 删除/改名通道（2026-11：文件管理器）：MEMFS 里的删除不会影响 IndexedDB，
// 不同步的话刷新页面后「已删的文件」会从 store 里重新活过来（loader 把 store
// 当「本机来源」挂载），改名后新旧两份会同时存在。因此 am_native_Fs.c 在删除/改名
// 成功后会调本钩子，把对应键从 nano-store 删掉；目录树改名由 C 侧逐文件重新 put
// （新路径）+ 删键（旧路径）完成。
// ---------------------------------------------------------------------------
EM_JS(void, nano_fs_store_delete_js, (const char *path_ptr, const char *root_ptr), {
    if (typeof nanoStoreDelete !== "function") return;
    try {
        var abs = UTF8ToString(path_ptr);
        var root = UTF8ToString(root_ptr);
        var key = abs;
        if (root.length > 0 && abs.indexOf(root + "/") === 0) {
            key = abs.slice(root.length + 1);
        }
        else if (abs.charAt(0) === "/") {
            return; // 资源根之外：本就不在 store 里
        }
        var p = nanoStoreDelete(key);
        if (p && typeof p.catch === "function") p.catch(function () {});
    }
    catch (e) { /* 存储不可用：忽略 */ }
});

void am_native_fs_on_remove(const char *path) {
    if (!path) return;
    nano_fs_store_delete_js(path, PLATFORM_ROOT_DIR);
}