// loader.js —— Nano-WASM-GUI 加载器：WASM 二进制下载进度 + 业务资源外置挂载进度
//
// 职责：
//   1. 自带进度地下载 nano_gui.wasm（Module.instantiateWasm 覆盖在 index.html 中调用
//      nanoFetchWithProgress；emscripten 默认的流式实例化不暴露进度，故接管）；
//   2. 运行时资源挂载：manifest.json → FS.createPath 建目录 → 按 priority 分两批
//      经自带进度 fetch 拉取并 FS.writeFile 进 MEMFS：
//        priority 1 = 启动必需（icon/、app/*.scm、data/ 存档…），**阻塞**启动；
//        priority 2 = 其它一切（电子书/词典/music/、llm/ 模型…），p1 就绪后**后台**拉。
//      p1 完成后调用 onP1Ready（index.html 据此 callMain 启动系统），p2 后台继续。
//      **没有第三/第四层**：音乐与 LLM 模型不搞按需挂载/预载按钮，和别的资源一样
//      随页面加载拉一次（宁可重下，不做特殊逻辑；见 doc/AGENTS.md 2026-11-17 条目）。
//   3. 浏览器本地存储（store.js，IndexedDB）优先于 HTTP 默认源：
//      每个文件先查 store（用户上传的），命中则用之；否则回退 HTTP。
//      store 中 manifest 之外的文件（用户全新上传）按目录推断优先级并入挂载表。
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

// 目录推断优先级（store 独有文件：manifest 清单之外的用户上传）
function nanoInferPriority(path) {
  const top = path.split("/")[0];
  if (top === "icon") return 1;
  if (!path.includes("/") && (path.startsWith("wp") || path === "mob.scm" || path === "jstest.js")) return 1;
  // data/ 是「灵机应用程序」自己的存档/模型（便签、2048 存档、Flappy 的 Q 表……），
  // 都是 KB 级小文件；提到 p1 让 App 启动时就能读到自己的存档，否则首帧读不到、
  // 训练一轮就把旧模型覆盖掉（2026-11 Flappy Bird 条目）。
  if (top === "data") return 1;
  return 2;
}

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

async function nanoMountResources(opts) {
  const { FS, manifestUrl, filesBaseUrl, resRoot } = opts;
  const onStage = opts.onStage || (() => {});
  const onP1Progress = opts.onP1Progress || (() => {});
  const onBgProgress = opts.onBgProgress || (() => {});

  // 拉取清单
  const manifest = await (await fetchImpl(manifestUrl)).json();

  // 合并浏览器本地存储清单：同名覆盖（用户上传优先），store 独有按目录推断优先级并入；
  // store 里的目录条目（网页文件管理器“新建目录”，可为空目录）同样并入；
  // store 里的**墓碑**（deleted 标记）在最后统一剔除——把同路径的服务器默认资源一并隐藏。
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
        byPath.set(sf.path, { path: sf.path, size: 0, dir: true, priority: nanoInferPriority(sf.path), source: "store" });
      }
      else if (existing) {
        existing.source = "store";
        existing.size = sf.size;
      }
      else {
        byPath.set(sf.path, { path: sf.path, size: sf.size, priority: nanoInferPriority(sf.path), source: "store" });
      }
    }
  }
  for (const p of tombstones) byPath.delete(p);
  const all = [...byPath.values()];
  const files = all.filter(f => !f.dir);
  const dirsExplicit = all.filter(f => f.dir);
  // 只有两批：1=启动必需（阻塞），2=其它一切（后台）。没有按需层。
  const p1 = files.filter(f => f.priority !== 2);
  const p2 = files.filter(f => f.priority === 2);
  if (opts.onManifest) opts.onManifest(all); // 合并清单（含 source/dir 标记），资源面板展示用

  // 建目录树（createPath 递归逐层创建；显式目录条目也要建，空目录才留得住）
  const dirs = new Set();
  for (const f of all) {
    const parts = f.path.split("/");
    for (let i = 1; i < parts.length; i++) dirs.add(parts.slice(0, i).join("/"));
  }
  try { FS.createPath("/", "home"); } catch (e) {}
  try { FS.createPath("/home", "bd4sur"); } catch (e) {}
  try { FS.createPath("/home/bd4sur", "linglong"); } catch (e) {}
  for (const d of [...dirs].sort()) { try { FS.createPath(resRoot, d); } catch (e) {} }
  for (const d of dirsExplicit) { try { FS.createPath(resRoot, d.path); } catch (e) {} }

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

  // 并发池（8 并发），带字节级汇总进度
  async function runPool(list, totalBytes, bytesBase, onProgress) {
    let idx = 0;
    let doneBytes = 0;   // 已完成文件的字节数
    let doneCount = 0;
    const inflight = new Map(); // 当前下载中文件的已收字节（汇总用）
    const workers = [];
    for (let i = 0; i < Math.min(8, list.length); i++) {
      workers.push((async () => {
        while (idx < list.length) {
          const f = list[idx++];
          await mountFile(f, (recv, total) => {
            inflight.set(f.path, recv);
            let cur = bytesBase + doneBytes;
            for (const v of inflight.values()) cur += v;
            onProgress(doneCount, list.length, cur, totalBytes);
          });
          inflight.delete(f.path);
          doneBytes += f.size;
          doneCount++;
          {   // 完成事件同样汇总 inflight（其他下载中文件的已收字节），保证进度单调
            let cur = bytesBase + doneBytes;
            for (const v of inflight.values()) cur += v;
            onProgress(doneCount, list.length, cur, totalBytes);
          }
        }
      })());
    }
    await Promise.all(workers);
  }

  // 阶段 1：启动必需资源（阻塞启动路径，精细进度）
  const p1Bytes = p1.reduce((s, f) => s + f.size, 0);
  onStage(`挂载启动资源（${(p1Bytes / 1048576).toFixed(1)}MB）…`);
  await runPool(p1, p1Bytes, 0, (dc, dn, cur, total) => onP1Progress(dc, dn, cur, total));

  // 阶段 1 就绪 → 通知启动 wasm main
  if (opts.onP1Ready) await opts.onP1Ready();

  // 阶段 2：后台（电子书/词典/相册/music/、llm/ 模型等一切非启动必需资源）
  const rest = p2;
  const restBytes = rest.reduce((s, f) => s + f.size, 0);
  await runPool(rest, restBytes, 0, (dc, dn, cur, total) => {
    onBgProgress(`后台挂载资源 ${dc}/${dn}（${(cur / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)}MB）`);
  });
  onStage("");
  console.log("[mount] resources mounted:", p1.length + p2.length);
}

// node 可测性导出（浏览器 <script> 顶层为全局函数）
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = { nanoMountResources, nanoFetchWithProgress, nanoSetFetchImpl, nanoInferPriority };
}
