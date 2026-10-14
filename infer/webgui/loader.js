// loader.js —— Nano-WASM-GUI 加载器：WASM 二进制下载进度 + 业务资源外置挂载进度
//
// 职责：
//   1. 自带进度地下载 nano_gui.wasm（Module.instantiateWasm 覆盖在 index.html 中调用
//      nanoFetchWithProgress；emscripten 默认的流式实例化不暴露进度，故接管）；
//   2. 运行时资源挂载：manifest.json → FS.createPath 建目录 → 分级（p1 启动必需 /
//      p2 业务资源）经自带进度 fetch 拉取并 FS.writeFile 进 MEMFS；
//      p1 完成后调用 onP1Ready（index.html 据此 callMain 启动系统），p2 后台继续。
//      p3（LLM 模型，数百 MB）【不自动挂载】——回调 onModelsPending 交由页面提供
//      「按需预载」入口（nanoMountModel），用户触发后才拉取。
//   3. 浏览器本地存储（store.js，IndexedDB）优先于 HTTP 默认源：
//      每个文件先查 store（用户上传/曾预载的），命中则用之；否则回退 HTTP。
//      store 中 manifest 之外的文件（用户全新上传）按目录推断优先级并入挂载表。
//
// 浏览器与 node 双兼容（node 用于端到端回归：fetch/FS 等经参数注入）：
//   nanoFetchWithProgress(url, onProgress)                      —— 带字节进度下载为 Uint8Array
//   nanoMountResources(opts)                                    —— 挂载全部清单资源（async）
//     opts = { FS, fetchImpl, manifestUrl, filesBaseUrl, resRoot,
//              onStage(text),                                  —— 阶段文案
//              onP1Progress(done,total,fileDone,fileTotal),    —— 阶段1进度（阻塞启动路径）
//              onBgProgress(text) }                            —— 阶段2/3后台进度
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
  if (top === "llm") return 3;
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

  // 合并浏览器本地存储清单：同名覆盖（用户上传优先），store 独有按目录推断优先级并入
  const byPath = new Map();
  for (const f of (manifest.files || [])) byPath.set(f.path, { ...f, source: "http" });
  if (opts.store) {
    let storeFiles = [];
    try { storeFiles = await opts.store.list(); } catch (e) { console.warn("[mount] store list failed:", e); }
    for (const sf of storeFiles) {
      const existing = byPath.get(sf.path);
      if (existing) {
        existing.source = "store";
        existing.size = sf.size;
      }
      else {
        byPath.set(sf.path, { path: sf.path, size: sf.size, priority: nanoInferPriority(sf.path), source: "store" });
      }
    }
  }
  const files = [...byPath.values()];
  const p1 = files.filter(f => f.priority === 1);
  const p2 = files.filter(f => f.priority === 2);
  const p3 = files.filter(f => f.priority === 3); // LLM 模型：不自动挂载，页面按需预载
  if (opts.onManifest) opts.onManifest(files); // 合并清单（含 source 标记），资源面板展示用

  // 建目录树（createPath 递归逐层创建）
  const dirs = new Set();
  for (const f of files) {
    const parts = f.path.split("/");
    for (let i = 1; i < parts.length; i++) dirs.add(parts.slice(0, i).join("/"));
  }
  try { FS.createPath("/", "home"); } catch (e) {}
  try { FS.createPath("/home", "bd4sur"); } catch (e) {}
  try { FS.createPath("/home/bd4sur", "linglong"); } catch (e) {}
  for (const d of [...dirs].sort()) { try { FS.createPath(resRoot, d); } catch (e) {} }

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

  // 阶段 2：后台（电子书/词典/相册/音乐等）
  const rest = p2;
  const restBytes = rest.reduce((s, f) => s + f.size, 0);
  await runPool(rest, restBytes, 0, (dc, dn, cur, total) => {
    onBgProgress(`后台挂载资源 ${dc}/${dn}（${(cur / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)}MB）`);
  });
  onStage("");

  // 阶段 3：LLM 模型不自动挂载——交由页面提供「按需预载」入口
  //（模型就绪前鹦鹉笼菜单显示「无可用模型」属预期；菜单按文件存在性动态过滤）
  if (opts.onModelsPending) {
    opts.onModelsPending(p3.map(f => ({ path: f.path, size: f.size, source: f.source })));
  }
  console.log("[mount] resources mounted:", p1.length + p2.length, "; models pending:", p3.length);
}

// 按需挂载单个模型（页面「预载模型」按钮调用）：store 优先；HTTP 拉取成功后写入 store
//（下次启动/离线可直接从浏览器存储读，不再耗流量）。onProgress(recv,total) 汇报进度。
async function nanoMountModel(opts, path, onProgress) {
  const { FS, resRoot } = opts;
  // store 命中则直接用
  if (opts.store) {
    try {
      const rec = await opts.store.get(path);
      if (rec && rec.blob) {
        const buf = new Uint8Array(await rec.blob.arrayBuffer());
        FS.writeFile(resRoot + "/" + path, buf);
        if (onProgress) onProgress(buf.length, buf.length);
        return true;
      }
    } catch (e) { /* 回退 HTTP */ }
  }
  // HTTP 拉取（带进度），成功后尝试写入 store 持久化（失败不影响本次使用）
  const buf = await nanoFetchWithProgress(opts.filesBaseUrl + encodeURI(path), onProgress);
  FS.writeFile(resRoot + "/" + path, buf);
  if (opts.store) {
    try { await opts.store.put(path, new Blob([buf])); } catch (e) { console.warn("[mount] persist model failed:", e); }
  }
  return true;
}

// node 可测性导出（浏览器 <script> 顶层为全局函数）
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = { nanoMountResources, nanoMountModel, nanoFetchWithProgress, nanoSetFetchImpl, nanoInferPriority };
}
