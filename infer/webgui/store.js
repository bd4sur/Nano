// store.js —— Nano-WASM-GUI 浏览器本地资源存储（IndexedDB）
//
// 业务资源（电子书/相册/音乐/LLM 模型，以及 icon/壁纸/词典的覆盖替换）可由用户经网页
// 上传到浏览器 IndexedDB——不依赖后端服务器，作为临时文件系统（清除站点数据即消失，
// 符合预期语义）；loader.js 挂载时优先读本存储，缺失再回退 HTTP 默认源。
//
// 结构：库 nano-store，两个 object store——meta（path → {size, mtime}）与 blobs
//（path → Blob）分离，list 不物化大 Blob。驱动注入式设计（nanoStoreSetDriver），
// 浏览器默认 IndexedDB 驱动；node 回归可注入内存驱动（见本文件末尾 makeMemDriver）。
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
  return {
    async put(path, blob) {
      const db = await openDb();
      const tx = db.transaction([NANO_STORE_META, NANO_STORE_BLOBS], "readwrite");
      tx.objectStore(NANO_STORE_BLOBS).put(blob, path);
      tx.objectStore(NANO_STORE_META).put({ size: blob.size, mtime: Date.now() }, path);
      await txDone(tx);
    },
    // 返回 { path, blob, size, mtime }；不存在返回 null
    async get(path) {
      const db = await openDb();
      const tx = db.transaction([NANO_STORE_META, NANO_STORE_BLOBS], "readonly");
      const blobReq = tx.objectStore(NANO_STORE_BLOBS).get(path);
      const metaReq = tx.objectStore(NANO_STORE_META).get(path);
      await txDone(tx).catch(() => {});
      const blob = blobReq.result;
      if (blob === undefined || blob === null) return null;
      const meta = metaReq.result || {};
      return { path, blob, size: meta.size !== undefined ? meta.size : blob.size, mtime: meta.mtime || 0 };
    },
    async del(path) {
      const db = await openDb();
      const tx = db.transaction([NANO_STORE_META, NANO_STORE_BLOBS], "readwrite");
      tx.objectStore(NANO_STORE_BLOBS).delete(path);
      tx.objectStore(NANO_STORE_META).delete(path);
      await txDone(tx);
    },
    // 返回 [{path, size, mtime}]（只读 meta，不物化 Blob）
    async list() {
      const db = await openDb();
      const tx = db.transaction(NANO_STORE_META, "readonly");
      const store = tx.objectStore(NANO_STORE_META);
      const keysReq = store.getAllKeys();
      const metasReq = store.getAll();
      await txDone(tx).catch(() => {});
      const keys = keysReq.result || [];
      const metas = metasReq.result || [];
      return keys.map((k, i) => ({ path: String(k), size: metas[i] ? metas[i].size : 0, mtime: metas[i] ? metas[i].mtime : 0 }));
    },
  };
}

// ---------------- 内存驱动（node 测试/退化兜底） ----------------

function makeMemDriver() {
  const map = new Map(); // path → { blob, size, mtime }
  return {
    async put(path, blob) { map.set(path, { blob, size: blob.size, mtime: Date.now() }); },
    async get(path) { const r = map.get(path); return r ? { path, ...r } : null; },
    async del(path) { map.delete(path); },
    async list() { return [...map.entries()].map(([path, r]) => ({ path, size: r.size, mtime: r.mtime })); },
  };
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

// node 可测性导出
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = {
    nanoStorePut, nanoStoreGet, nanoStoreDelete, nanoStoreList,
    nanoStoreSetDriver, makeMemDriver,
  };
}
