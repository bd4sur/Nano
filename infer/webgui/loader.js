// loader.js —— Nano-WASM-GUI 加载器：WASM 二进制下载进度 + 业务资源外置挂载进度
//
// 职责：
//   1. 自带进度地下载 nano_gui.wasm（Module.instantiateWasm 覆盖在 index.html 中调用
//      nanoFetchWithProgress；emscripten 默认的流式实例化不暴露进度，故接管）；
//   2. 运行时资源挂载：manifest.json → FS.createPath 建目录 → 按 priority 分两批
//      经自带进度 fetch 拉取并 FS.writeFile 进 MEMFS：
//        priority 1 = 启动必需（icon/、app/*.scm、data/ 存档…），**阻塞**启动；
//        priority 2 = 其它一切（电子书/词典/music/、llm/ 模型…），**不自动拉**：
//          nanoMountResources 只返回这份清单，页面用「下载资源」按钮调
//          nanoMountPending() 才真正下载（283MB，用户点了才付这份流量）。
//      p1 完成后调用 onP1Ready（index.html 据此 callMain 启动系统）。
//      没有第三/第四层，也没有按 App 的按需钩子（见 doc/AGENTS.md 2026-11-17 条目）。
//   3. 浏览器本地存储（store.js，IndexedDB）优先于 HTTP 默认源：
//      每个文件先查 store（用户上传的），命中则用之；否则回退 HTTP。
//      store 中 manifest 之外的文件（用户全新上传）一律并入 p1（用户自己的东西）。
//
// 浏览器与 node 双兼容（node 用于端到端回归：fetch/FS 等经参数注入）：
//   nanoFetchWithProgress(url, onProgress)                      —— 带字节进度下载为 Uint8Array
//   nanoMountResources(opts)                                    —— 挂载全部清单资源（async）
//     opts = { FS, fetchImpl, manifestUrl, filesBaseUrl, resRoot,
//              onStage(text),                                  —— 阶段文案
//              onP1Progress(done,total,fileDone,fileTotal),    —— 阶段1进度（阻塞启动路径）
//              onBgProgress(text) }                            —— 阶段2后台进度
"use strict";

// 带字节进度地下载 URL 为 Uint8Array（onProgress(receivedBytes, totalBytes)；total 可为 0=未知）
async function nanoFetchWithProgress(url, onProgress) {
  const resp = await fetchImpl(url);
  if (!resp.ok && resp.ok !== undefined) throw new Error("HTTP " + resp.status);
  const total = (resp.headers && resp.headers.get)
    ? (parseInt(resp.headers.get("Content-Length") || "0", 10) || 0) : 0;
  if (resp.body && resp.body.getReader) {
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (onProgress) onProgress(received, total);
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    return buf;
  }
  // 退化路径（无流式读取）：整取
  const buf = new Uint8Array(await resp.arrayBuffer());
  if (onProgress) onProgress(buf.length, buf.length);
  return buf;
}
// fetch 注入点（node 测试时替换；浏览器默认全局 fetch）
var fetchImpl = (typeof fetch !== "undefined") ? fetch : null;
function nanoSetFetchImpl(f) { fetchImpl = f; }

// 优先级规则（只有两级）：
//   p1 = 启动必需：manifest 的 priority 1，**以及本机存储里的全部文件**；
//   p2 = 服务器上那堆可选的大块资源（ebook/dict/music/llm…），等页面「下载资源」按钮。
// 为什么 store 文件一律 p1：它们是**用户自己的东西**（上传的文档/图片/音乐、便签、
// App 存档……），体积由用户决定，读的是 IndexedDB 而不是网络。若按目录给 p2，就会出现
// "上次上传到 doc/ 的文件重启后不见了，得先点一下下载服务器资源"（2026-11-17 由
// regress_webgui_files 的"移动后的文件还在"用例抓到）。
// 注意：manifest 里已有的文件即使在 store 中被覆盖，仍沿用 manifest 的 priority。

// 单文件读取：store 优先（rec.blob → Uint8Array），失败/未命中回退 HTTP 带进度下载
async function nanoReadResource(opts, f, onFileProgress) {
  if (f.source === "store" && opts.store) {
    try {
      const rec = await opts.store.get(f.path);
      if (rec && rec.blob) {
        const buf = new Uint8Array(await rec.blob.arrayBuffer());
        if (onFileProgress) onFileProgress(buf.length, buf.length);
        return buf;
      }
    } catch (e) { /* 存储读取失败：回退 HTTP */ }
  }
  const url = opts.filesBaseUrl + encodeURI(f.path);
  return nanoFetchWithProgress(url, onFileProgress);
}

// 收集资源清单：manifest ∪ 本机存储（同名以 store 为准，用户上传优先）。
// store 里的目录条目（网页文件管理器“新建目录”，可为空目录）同样并入；
// store 里的**墓碑**（deleted 标记）在最后统一剔除——把同路径的服务器默认资源一并隐藏。
async function nanoCollect(opts) {
  const { manifestUrl } = opts;
  const manifest = await (await fetchImpl(manifestUrl)).json();
  const byPath = new Map();
  for (const f of (manifest.files || [])) byPath.set(f.path, { ...f, source: "http" });
  const tombstones = new Set();
  if (opts.store) {
    let storeFiles = [];
    try { storeFiles = await opts.store.list(); } catch (e) { console.warn("[mount] store list failed:", e); }
    for (const sf of storeFiles) {
      if (sf.deleted) { tombstones.add(sf.path); continue; }
      const existing = byPath.get(sf.path);
      if (sf.dir) {
        // 目录条目：不挂载内容，只保证 MEMFS 里有这个（可能是空的）目录
        byPath.set(sf.path, { path: sf.path, size: 0, dir: true, priority: 1, source: "store" });
      }
      else if (existing) {
        existing.source = "store";
        existing.size = sf.size;
      }
      else {
        byPath.set(sf.path, { path: sf.path, size: sf.size, priority: 1, source: "store" });
      }
    }
  }
  for (const p of tombstones) byPath.delete(p);
  const all = [...byPath.values()];
  const dirs = new Set();
  for (const f of all) {
    const parts = f.path.split("/");
    for (let i = 1; i < parts.length; i++) dirs.add(parts.slice(0, i).join("/"));
  }
  return { all, files: all.filter(f => !f.dir), dirs, dirsExplicit: all.filter(f => f.dir) };
}

// 建目录树（createPath 递归逐层创建；显式目录条目也要建，空目录才留得住）
function nanoEnsureDirs(FS, resRoot, dirs, dirsExplicit) {
  try { FS.createPath("/", "home"); } catch (e) {}
  try { FS.createPath("/home", "bd4sur"); } catch (e) {}
  try { FS.createPath("/home/bd4sur", "linglong"); } catch (e) {}
  for (const d of [...dirs].sort()) { try { FS.createPath(resRoot, d); } catch (e) {} }
  for (const d of dirsExplicit) { try { FS.createPath(resRoot, d.path); } catch (e) {} }
}

// MEMFS 里这个相对路径是否已有内容（按文件大小判：0 字节文件也算"有"）
function nanoMemfsHas(FS, resRoot, relPath) {
  try { return !!FS.stat(resRoot + "/" + relPath); } catch (e) { return false; }
}

// 挂一批文件进 MEMFS（8 并发，字节级汇总进度；已在 MEMFS 的跳过 → 重复点击安全）。
// onProgress(完成数, 总数, 已传字节, 总字节)。
async function nanoMountList(opts, list, onProgress) {
  const { FS, resRoot } = opts;
  const report = onProgress || (() => {});
  // 只挂 MEMFS 里还没有的：store 来源也要读入 MEMFS 才能被 C 侧 fopen
  const todo = list.filter(f => !nanoMemfsHas(FS, resRoot, f.path));
  const totalBytes = todo.reduce((s, f) => s + (f.size || 0), 0);
  let doneBytes = 0, doneCount = 0, idx = 0;
  const inflight = new Map();
  const tick = () => {
    let cur = doneBytes;
    for (const v of inflight.values()) cur += v;
    report(doneCount, todo.length, cur, totalBytes);
  };

  // 挂载单文件（store 优先，HTTP 回退；失败重试一次）
  async function mountFile(f, onFileProgress) {
    const target = resRoot + "/" + f.path;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const buf = await nanoReadResource(opts, f, onFileProgress);
        FS.writeFile(target, buf);
        return true;
      } catch (e) { /* 重试 */ }
    }
    console.warn("[mount] failed:", f.path);
    return false;
  }

  const worker = async () => {
    while (idx < todo.length) {
      const f = todo[idx++];
      await mountFile(f, (recv) => { inflight.set(f.path, recv); tick(); });
      inflight.delete(f.path);
      doneBytes += f.size || 0;
      doneCount++;
      tick();
    }
  };
  const workers = [];
  for (let i = 0; i < Math.min(8, todo.length); i++) workers.push(worker());
  await Promise.all(workers);
  return { done: doneCount, total: todo.length, bytes: totalBytes };
}

// 阶段 1：挂载**启动必需**资源（阻塞启动路径），完成后回调 onP1Ready。
// 返回值 = 待用户点「下载资源」才拉的 p2 清单（本函数**不**拉它）。
async function nanoMountResources(opts) {
  const { FS, resRoot } = opts;
  const onStage = opts.onStage || (() => {});
  const onP1Progress = opts.onP1Progress || (() => {});

  const { all, files, dirs, dirsExplicit } = await nanoCollect(opts);
  if (opts.onManifest) opts.onManifest(all); // 合并清单（含 source/dir 标记），资源面板展示用
  nanoEnsureDirs(FS, resRoot, dirs, dirsExplicit);

  // 只有两批：1=启动必需（阻塞），2=其余一切（页面按需，用户点按钮才下）。没有按需层。
  const p1 = files.filter(f => f.priority !== 2);
  const p2 = files.filter(f => f.priority === 2);

  const p1Bytes = p1.reduce((s, f) => s + f.size, 0);
  onStage(`挂载启动资源（${(p1Bytes / 1048576).toFixed(1)}MB）…`);
  await nanoMountList(opts, p1, (dc, dn, cur, total) => onP1Progress(dc, dn, cur, total));
  onStage("");

  // 阶段 1 就绪 → 通知启动 wasm main
  if (opts.onP1Ready) await opts.onP1Ready();

  const p2Bytes = p2.reduce((s, f) => s + f.size, 0);
  console.log("[mount] 启动资源就绪:", p1.length, "个；待下载 p2:", p2.length, "个 /",
              (p2Bytes / 1048576).toFixed(1), "MB（等页面按钮）");
  return p2;
}

// 阶段 2：把 p2 清单下载进 MEMFS（页面「下载资源」按钮 / 回归脚本的 nanoEnsureP2 调用）。
// 幂等：已在 MEMFS 的文件跳过；重复点击不会重复下载。
async function nanoMountPending(opts, pending, onProgress) {
  const r = await nanoMountList(opts, pending || [], onProgress);
  console.log("[mount] p2 下载完成:", r.done, "/", r.total, "个 /", (r.bytes / 1048576).toFixed(1), "MB");
  return r;
}

// node 可测性导出（浏览器 <script> 顶层为全局函数）
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = { nanoMountResources, nanoMountPending, nanoCollect, nanoFetchWithProgress, nanoSetFetchImpl };
}
