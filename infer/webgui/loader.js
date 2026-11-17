// loader.js —— Nano-WASM-GUI 加载器：WASM 二进制下载进度 + 业务资源外置挂载
//
// 职责：
//   1. 自带进度地下载 nano_gui.wasm（Module.instantiateWasm 覆盖在 index.html 中调用
//      nanoFetchWithProgress；emscripten 默认的流式实例化不暴露进度，故接管）；
//   2. 运行时资源挂载：manifest.json → 建目录 → 按优先级分两类处理（见下）；
//   3. 浏览器本地存储（store.js，IndexedDB）作为 p2 缓存与用户自建文件的落点。
//
// ============================================================================
// 资源两类（2026-11-18 定稿，见 doc/AGENTS.md 2026-11-18（五））
// ============================================================================
//   p1 = manifest 里 priority 1 的条目，就是 app/ icon/ wallpaper/ 三处：
//        **服务器是唯一来源**。每次页面加载无条件从 HTTP 全量重拉、无条件覆盖 MEMFS，
//        并把 IndexedDB 里属于 p1 的条目**全部删掉**（含用户覆盖与墓碑）。
//        这样浏览器里永远不会留着旧版 App 源码/图标——过去"改了 .scm 刷新不生效、
//        必须清站点数据"就是 IndexedDB 缓存遮挡服务器版本造成的（2026-10 踩过）。
//        因此 p1 **不写入** IndexedDB：写了也只是让下一次加载多一份要被清掉的垃圾。
//
//   p2 = manifest 里其余一切（ebook/ dict/ music/ llm/ …）：
//        只有用户点页面上的「下载资源」按钮才拉（nanoMountPending）。规则：
//          · MEMFS 已有同名文件 → 跳过（静默）；
//          · IndexedDB 已有同名文件 → 从本机灌进 MEMFS，**不联网、不重写**（静默）；
//          · 都没有 → 从服务器拉，写入 MEMFS **并**写入 IndexedDB（下次就命中上一条）。
//
//   另有第三类：IndexedDB 里存在、但 manifest 里没有的路径 —— 用户自己上传/App 落盘的
//   本机文件（如 doc/我的笔记.md）。它们不走网络，随启动直接灌进 MEMFS（否则用户上传完
//   刷新一下就"不见了"，必须点下载按钮才回来，明显不合理）。
//
// 浏览器与 node 双兼容（node 用于端到端回归：fetch/FS 等经参数注入）：
//   nanoFetchWithProgress(url, onProgress)                      —— 带字节进度下载为 Uint8Array
//   nanoMountResources(opts)                                    —— 挂载 p1 + 本机文件（async）
//     opts = { FS, fetchImpl, manifestUrl, filesBaseUrl, resRoot,
//              store: { get, put, list, delete },
//              onStage(text),                                  —— 阶段文案
//              onP1Progress(done,total,fileDone,fileTotal),     —— 阶段1进度（阻塞启动路径）
//              onBgProgress(text) }                            —— 阶段2后台进度
//     p1 + 本机文件就绪后调用 onP1Ready（index.html 据此 callMain 启动系统），
//     返回待用户点「下载资源」才拉的 p2 清单（本函数**不**拉它）。
//   nanoMountPending(opts, pending, onProgress)                 —— 拉 p2（按钮/回归脚本）
// ============================================================================
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

// ---------------- 路径分类 ----------------

// 路径的顶层目录（"app/lib/a.scm" → "app"；根文件 → ""）
function nanoTopDir(p) {
  const i = p.indexOf("/");
  return i < 0 ? "" : p.slice(0, i);
}

// 从清单解析 p1 归属：
//   p1Set  = priority 1 的路径集合（逐文件判定的唯一依据）
//   p1Tops = p1 的顶层目录（清单里带 p1_dirs 就直接用，否则由 p1Set 反推）——
//            用来识别"服务器上已删除、但浏览器 IndexedDB 里还留着"的 p1 残骸。
function nanoResolveP1(manifest) {
  const files = (manifest && manifest.files) || [];
  const p1Set = new Set();
  for (const f of files) if (f.priority === 1) p1Set.add(f.path);
  let p1Tops;
  if (manifest && Array.isArray(manifest.p1_dirs)) {
    p1Tops = new Set(manifest.p1_dirs);
  }
  else {
    p1Tops = new Set();
    for (const p of p1Set) { const t = nanoTopDir(p); if (t) p1Tops.add(t); }
  }
  return { p1Set, p1Tops };
}

// ---------------- 清单收集 / 存储清理 ----------------

// 收集资源清单：
//   p1Files    —— 从服务器拉的（每次全量，覆盖 MEMFS）
//   p2Files    —— 等按钮的
//   localFiles —— IndexedDB 里有、manifest 里没有（用户自己的文件）→ 随启动灌进 MEMFS
//   stalePaths —— 需要从 IndexedDB 删掉的 p1 条目（覆盖 + 墓碑 + 服务器上已删的残骸）
//   dirs/dirsExplicit —— MEMFS 目录树（含用户新建的空目录）
async function nanoCollect(opts) {
  const { manifestUrl } = opts;
  const manifest = await (await fetchImpl(manifestUrl)).json();
  const { p1Set, p1Tops } = nanoResolveP1(manifest);
  const manifestFiles = (manifest.files || []).map(f => ({ ...f, source: "http" }));
  const manifestPaths = new Set(manifestFiles.map(f => f.path));

  let storeFiles = [];
  if (opts.store) {
    try { storeFiles = await opts.store.list(); } catch (e) { console.warn("[mount] store list failed:", e); }
  }

  const stalePaths = [];   // 要从 IndexedDB 删掉的键
  const localFiles = [];   // 本机独有（manifest 里没有）
  const dirsExplicit = []; // 用户新建的目录条目（可为空目录）
  for (const sf of storeFiles) {
    const p = sf.path;
    if (p1Set.has(p) || (p1Tops.has(nanoTopDir(p)) && !manifestPaths.has(p))) {
      // p1 归服务器管：本机覆盖、墓碑、以及服务器上已删的残骸一律清掉
      stalePaths.push(p);
      continue;
    }
    if (sf.dir) { dirsExplicit.push({ path: p, dir: true, priority: 1, source: "store" }); continue; }
    if (sf.deleted) {
      // p2 的墓碑：用户在网页文件管理器里删掉了某个服务器默认资源 → 保持隐藏
      continue;
    }
    if (!manifestPaths.has(p)) {
      localFiles.push({ path: p, size: sf.size || 0, priority: 1, source: "store" });
    }
  }

  // p2 墓碑：从待下载清单里剔除（删了就别再问用户要不要下）
  const tombstones = new Set(storeFiles.filter(sf => sf.deleted && !p1Set.has(sf.path)).map(sf => sf.path));

  const all = [...manifestFiles, ...localFiles, ...dirsExplicit]
    .filter(f => !tombstones.has(f.path));
  const dirs = new Set();
  for (const f of all) {
    const parts = f.path.split("/");
    for (let i = 1; i < parts.length; i++) dirs.add(parts.slice(0, i).join("/"));
  }
  return {
    all,
    p1Files: manifestFiles.filter(f => f.priority === 1),
    p2Files: manifestFiles.filter(f => f.priority !== 1 && !tombstones.has(f.path)),
    localFiles,
    stalePaths,
    dirs,
    dirsExplicit,
  };
}

// 清掉 IndexedDB 里属于 p1 的条目（"重建 p1 分片"：让浏览器里 p1 的唯一来源只剩服务器）
async function nanoPurgeP1(opts, stalePaths) {
  if (!opts.store || !opts.store.delete || !stalePaths.length) return 0;
  let n = 0;
  for (const p of stalePaths) {
    try { await opts.store.delete(p); n++; } catch (e) { /* 单条失败不影响挂载 */ }
  }
  if (n) console.log("[mount] 已清除 IndexedDB 中属于 p1 的条目:", n, "个");
  return n;
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

// 从 IndexedDB 取一个文件的字节（没有/是目录/读取失败 → null）
async function nanoStoreBytes(opts, relPath) {
  if (!opts.store || !opts.store.get) return null;
  try {
    const rec = await opts.store.get(relPath);
    if (rec && rec.blob) return new Uint8Array(await rec.blob.arrayBuffer());
  } catch (e) { /* 读取失败：当作没有 */ }
  return null;
}

// ---------------- 挂载一批文件 ----------------
// mode：
//   "p1"    从服务器拉（覆盖 MEMFS，不写 IndexedDB）
//   "local" 从 IndexedDB 读（本机独有文件）
//   "p2"    MEMFS 有→跳过；IndexedDB 有→灌进 MEMFS（不联网/不重写）；都没有→拉+两边都写
// 并发 8，字节级汇总进度；返回 { done, total, bytes, fetched, cached, skipped }
async function nanoMountList(opts, list, onProgress, mode) {
  const { FS, resRoot } = opts;
  const report = onProgress || (() => {});

  // 待处理集合：p1/local 全做；p2 剔除 MEMFS 已有的（静默跳过）
  const todo = [];
  let skipped = 0;
  for (const f of list) {
    if (mode === "p2" && nanoMemfsHas(FS, resRoot, f.path)) { skipped++; continue; }
    todo.push(f);
  }
  const totalBytes = todo.reduce((s, f) => s + (f.size || 0), 0);
  let doneBytes = 0, doneCount = 0, fetched = 0, cached = 0;

  // 挂载单文件：按 mode 决定字节从哪来
  async function mountFile(f) {
    const target = resRoot + "/" + f.path;
    let buf = null, fromStore = false;

    if (mode !== "p1") {
      buf = await nanoStoreBytes(opts, f.path);
      if (buf) fromStore = true;
    }
    if (!buf) {
      buf = await nanoFetchWithProgress(opts.filesBaseUrl + encodeURI(f.path),
        (recv) => { inflight.set(f.path, recv); tick(); });
      fetched++;
    }
    else cached++;

    // 1) MEMFS（p1 无条件覆盖；p2/local 只补缺）
    FS.writeFile(target, buf);
    // 2) IndexedDB（只有 p2 且是刚从服务器拉的才落地：已有的不覆盖、p1 永不落库）
    if (mode === "p2" && !fromStore && opts.store && opts.store.put) {
      try {
        await opts.store.put(f.path, new Blob([buf]));
      } catch (e) {
        // 配额不足/隐私模式：静默降级为"本次会话可用"，不影响本次挂载
        if (!nanoMountList._storeWarned) {
          nanoMountList._storeWarned = true;
          console.warn("[mount] 写入 IndexedDB 失败（配额/权限），本次仅驻留 MEMFS:", e);
        }
      }
    }
  }

  const inflight = new Map();
  const tick = () => {
    let cur = doneBytes;
    for (const v of inflight.values()) cur += v;
    report(doneCount, todo.length, cur, totalBytes);
  };

  let idx = 0;
  const worker = async () => {
    while (idx < todo.length) {
      const f = todo[idx++];
      for (let attempt = 0; attempt < 2; attempt++) {
        try { await mountFile(f); break; }
        catch (e) {
          if (attempt === 1) console.warn("[mount] failed:", f.path, e);
          else { inflight.delete(f.path); }
        }
      }
      inflight.delete(f.path);
      doneBytes += f.size || 0;
      doneCount++;
      tick();
    }
  };
  const workers = [];
  for (let i = 0; i < Math.min(8, todo.length); i++) workers.push(worker());
  await Promise.all(workers);
  return { done: doneCount, total: todo.length, bytes: totalBytes, fetched, cached, skipped };
}

// ---------------- 对外：启动挂载（p1 + 本机文件） ----------------
// 阶段 1：清 p1 旧账 → 建目录 → 从服务器全量重拉 p1（覆盖 MEMFS）→ 灌本机独有文件
//         → 回调 onP1Ready（页面据此 callMain 启动）。
// 返回值 = 待用户点「下载资源」才拉的 p2 清单（本函数**不**拉它）。
async function nanoMountResources(opts) {
  const { FS, resRoot } = opts;
  const onStage = opts.onStage || (() => {});
  const onP1Progress = opts.onP1Progress || (() => {});

  const { all, p1Files, p2Files, localFiles, stalePaths, dirs, dirsExplicit } = await nanoCollect(opts);

  // ① 重建 IndexedDB 的 p1 分片：服务器是唯一来源，本机副本/墓碑一律作废
  onStage("清理本机缓存的启动资源…");
  await nanoPurgeP1(opts, stalePaths);

  // ② 目录树 + p1 全量拉取（无条件覆盖 MEMFS）
  if (opts.onManifest) opts.onManifest(all);
  nanoEnsureDirs(FS, resRoot, dirs, dirsExplicit);

  const p1Bytes = p1Files.reduce((s, f) => s + f.size, 0);
  onStage(`挂载启动资源（${(p1Bytes / 1048576).toFixed(1)}MB）…`);
  await nanoMountList(opts, p1Files, (dc, dn, cur, total) => onP1Progress(dc, dn, cur, total), "p1");

  // ③ 本机独有文件（用户上传 / App 落盘的，不走网络）
  if (localFiles.length) {
    onStage(`挂载本机文件（${localFiles.length} 个）…`);
    await nanoMountList(opts, localFiles, null, "local");
  }
  onStage("");

  // ④ 启动 wasm main
  if (opts.onP1Ready) await opts.onP1Ready();

  const p2Bytes = p2Files.reduce((s, f) => s + f.size, 0);
  console.log("[mount] 启动资源就绪:", p1Files.length, "个（服务器全量）; 本机文件:", localFiles.length,
              "个; 待下载 p2:", p2Files.length, "个 /", (p2Bytes / 1048576).toFixed(1), "MB（等页面按钮）");
  return p2Files;
}

// ---------------- 对外：按钮拉 p2 ----------------
// 幂等：MEMFS 已有的跳过；IndexedDB 已有的只灌 MEMFS、不联网也不重写。
async function nanoMountPending(opts, pending, onProgress) {
  const r = await nanoMountList(opts, pending || [], onProgress, "p2");
  console.log("[mount] p2 处理完成:", r.done, "/", r.total, "个 /", (r.bytes / 1048576).toFixed(1),
              "MB（服务器下载", r.fetched, "，本机缓存", r.cached, "，已在 MEMFS 跳过", r.skipped, "）");
  return r;
}

// node 可测性导出（浏览器 <script> 顶层为全局函数）
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = {
    nanoMountResources, nanoMountPending, nanoCollect, nanoFetchWithProgress, nanoSetFetchImpl,
    nanoResolveP1, nanoTopDir,
  };
}
