// store.js —— Nano-WASM-GUI 浏览器本地资源存储（IndexedDB）
//
// 业务资源（电子书/相册/音乐/LLM 模型，以及 icon/壁纸/词典的覆盖替换）可由用户经网页
// 上传到浏览器 IndexedDB——不依赖后端服务器，作为临时文件系统（清除站点数据即消失，
// 符合预期语义）；loader.js 挂载时优先读本存储，缺失再回退 HTTP 默认源。
//
// 结构：库 nano-store，两个 object store——meta（path → {size, mtime, dir?, deleted?}）
// 与 blobs（path → Blob）分离，list 不物化大 Blob。驱动注入式设计（nanoStoreSetDriver），
// 浏览器默认 IndexedDB 驱动；node 回归可注入内存驱动（见本文件末尾 makeMemDriver）。
//
// 记录形态（三条互斥的语义，统一用 meta 里的标记位表达）：
//   · 普通文件：blobs[key] = Blob，meta = {size, mtime}
//   · 目录    ：无 blob，meta = {size: 0, mtime, dir: 1}   —— 空目录也要能持久（网页文件管理器的“新建目录”）
//   · 墓碑    ：无 blob，meta = {size, mtime, deleted: 1}  —— “删除服务器默认资源”的持久标记，
//               loader 见到墓碑就不挂载同路径的默认文件（否则刷新后删掉的东西又活过来）
// put() 写文件会整体覆盖 meta，因此自动清掉 dir/deleted 标记（删除后同名重建即恢复）。
"use strict";

const NANO_STORE_DB = "nano-store";
const NANO_STORE_META = "meta";
const NANO_STORE_BLOBS = "blobs";

// ---------------- IndexedDB 驱动（浏览器） ----------------

function makeIdbDriver() {
  let dbPromise = null;
  function openDb() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(NANO_STORE_DB, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(NANO_STORE_META)) db.createObjectStore(NANO_STORE_META);
        if (!db.objectStoreNames.contains(NANO_STORE_BLOBS)) db.createObjectStore(NANO_STORE_BLOBS);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return dbPromise;
  }
  function txDone(tx) {
    return new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error || new Error("tx abort"));
    });
  }
  const driver = {
    // 文件写入（同名覆盖；同时清掉 dir/deleted 标记）
    async put(path, blob) {
      const db = await openDb();
      const tx = db.transaction([NANO_STORE_META, NANO_STORE_BLOBS], "readwrite");
      tx.objectStore(NANO_STORE_BLOBS).put(blob, path);
      tx.objectStore(NANO_STORE_META).put({ size: blob.size, mtime: Date.now() }, path);
      await txDone(tx);
    },
    // 目录（空目录也持久；同路径若有文件内容则丢弃）
    async putDir(path) {
      const db = await openDb();
      const tx = db.transaction([NANO_STORE_META, NANO_STORE_BLOBS], "readwrite");
      tx.objectStore(NANO_STORE_BLOBS).delete(path);
      tx.objectStore(NANO_STORE_META).put({ size: 0, mtime: Date.now(), dir: 1 }, path);
      await txDone(tx);
    },
    // 墓碑：把“服务器默认资源”从虚拟文件系统里持久隐藏
    async tombstone(path, size) {
      const db = await openDb();
      const tx = db.transaction([NANO_STORE_META, NANO_STORE_BLOBS], "readwrite");
      tx.objectStore(NANO_STORE_BLOBS).delete(path);
      tx.objectStore(NANO_STORE_META).put({ size: size || 0, mtime: Date.now(), deleted: 1 }, path);
      await txDone(tx);
    },
    // 返回 { path, blob, size, mtime, dir }；不存在（或已删）返回 null
    async get(path) {
      const db = await openDb();
      const tx = db.transaction([NANO_STORE_META, NANO_STORE_BLOBS], "readonly");
      const blobReq = tx.objectStore(NANO_STORE_BLOBS).get(path);
      const metaReq = tx.objectStore(NANO_STORE_META).get(path);
      await txDone(tx).catch(() => {});
      const meta = metaReq.result;
      if (!meta) return null;
      const blob = blobReq.result;
      if (blob === undefined || blob === null) {
        if (meta.dir) return { path, blob: null, size: 0, mtime: meta.mtime || 0, dir: true };
        return null;
      }
      return { path, blob, size: meta.size !== undefined ? meta.size : blob.size, mtime: meta.mtime || 0, dir: false };
    },
    async del(path) {
      const db = await openDb();
      const tx = db.transaction([NANO_STORE_META, NANO_STORE_BLOBS], "readwrite");
      tx.objectStore(NANO_STORE_BLOBS).delete(path);
      tx.objectStore(NANO_STORE_META).delete(path);
      await txDone(tx);
    },
    // 改名/移动单条记录（文件复用同一个 Blob 对象，不物化字节）
    async rename(from, to) {
      const rec = await driver.get(from);
      if (!rec) return false;
      if (rec.dir) await driver.putDir(to);
      else await driver.put(to, rec.blob);
      await driver.del(from);
      return true;
    },
    // 返回 [{path, size, mtime, dir, deleted}]（只读 meta，不物化 Blob）
    async list() {
      const db = await openDb();
      const tx = db.transaction(NANO_STORE_META, "readonly");
      const store = tx.objectStore(NANO_STORE_META);
      const keysReq = store.getAllKeys();
      const metasReq = store.getAll();
      await txDone(tx).catch(() => {});
      const keys = keysReq.result || [];
      const metas = metasReq.result || [];
      return keys.map((k, i) => {
        const meta = metas[i] || {};
        return {
          path: String(k),
          size: meta.size || 0,
          mtime: meta.mtime || 0,
          dir: !!meta.dir,
          deleted: !!meta.deleted,
        };
      });
    },
  };
  return driver;
}

// ---------------- 内存驱动（node 测试/退化兜底） ----------------

function makeMemDriver() {
  const map = new Map(); // path → { blob, size, mtime, dir, deleted }
  const driver = {
    async put(path, blob) { map.set(path, { blob, size: blob.size, mtime: Date.now(), dir: false, deleted: false }); },
    async putDir(path) { map.set(path, { blob: null, size: 0, mtime: Date.now(), dir: true, deleted: false }); },
    async tombstone(path, size) { map.set(path, { blob: null, size: size || 0, mtime: Date.now(), dir: false, deleted: true }); },
    async get(path) {
      const r = map.get(path);
      if (!r || r.deleted) return null;
      return { path, blob: r.blob, size: r.size, mtime: r.mtime, dir: r.dir };
    },
    async del(path) { map.delete(path); },
    async rename(from, to) {
      const r = map.get(from);
      if (!r) return false;
      map.set(to, r);
      map.delete(from);
      return true;
    },
    async list() {
      return [...map.entries()].map(([path, r]) => ({
        path, size: r.size, mtime: r.mtime, dir: r.dir, deleted: r.deleted,
      }));
    },
  };
  return driver;
}

// ---------------- 对外 API（驱动可注入） ----------------

let nanoStoreDriver = null; // null=惰性创建 IndexedDB 驱动
function nanoStoreSetDriver(d) { nanoStoreDriver = d; }
function nanoStoreGetDriver() {
  if (!nanoStoreDriver) nanoStoreDriver = makeIdbDriver();
  return nanoStoreDriver;
}

async function nanoStorePut(path, blob) { return nanoStoreGetDriver().put(path, blob); }
async function nanoStoreGet(path) { return nanoStoreGetDriver().get(path); }
async function nanoStoreDelete(path) { return nanoStoreGetDriver().del(path); }
async function nanoStoreList() { return nanoStoreGetDriver().list(); }
async function nanoStorePutDir(path) { return nanoStoreGetDriver().putDir(path); }
async function nanoStoreTombstone(path, size) { return nanoStoreGetDriver().tombstone(path, size); }
async function nanoStoreRename(from, to) { return nanoStoreGetDriver().rename(from, to); }

// node 可测性导出
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = {
    nanoStorePut, nanoStoreGet, nanoStoreDelete, nanoStoreList,
    nanoStorePutDir, nanoStoreTombstone, nanoStoreRename,
    nanoStoreSetDriver, nanoStoreGetDriver, makeMemDriver, makeIdbDriver,
  };
}
