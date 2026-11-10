// vfs.js —— 虚拟文件系统（VFS）逻辑层：路径 / 目录树 / 可见行 / 文件操作
//
// 与 DOM 无关，node 可直接测（依赖全注入）。三个关注点分开：
//   1) 视图模型：把「本地存储（store.js）+ 服务器默认源（manifest）」合成一棵树，
//      **同级目录在前、文件在后，同类按名称字符串升序**（nanoBuildTree）；
//   2) 可见行：传统目录树的前序展开（nanoFlattenTree）——传入展开集合，返回
//      [{ node, depth }]；默认全折叠由调用方用空 expanded 集合表达；
//   3) 操作：新建目录 / 写文件 / 改名 / 移动 / 复制 / 删除 / 读字节 / 恢复墓碑。
//      持久层（store）与运行时文件系统（MEMFS 镜像）同时更新，保证「当前会话立即生效」
//      且「刷新页面仍然存在」。
//
// 路径口径：一律是**业务资源根下面的相对路径**（如 "app/lib/menu_list.scm"）；"" 表示根。
"use strict";

// ---------------- 路径工具 ----------------

function nanoPathNorm(p) {
  if (p === undefined || p === null) return "";
  return String(p).replace(/\\/g, "/").split("/").filter(function (s) { return s.length > 0; }).join("/");
}
function nanoPathJoin(dir, name) {
  const d = nanoPathNorm(dir), n = String(name === undefined || name === null ? "" : name).replace(/^\/+|\/+$/g, "");
  if (!d) return nanoPathNorm(n);
  if (!n) return d;
  return d + "/" + n;
}
function nanoPathParent(p) {
  const s = nanoPathNorm(p);
  const i = s.lastIndexOf("/");
  return i < 0 ? "" : s.slice(0, i);
}
function nanoPathBase(p) {
  const s = nanoPathNorm(p);
  const i = s.lastIndexOf("/");
  return i < 0 ? s : s.slice(i + 1);
}
function nanoPathDepth(p) {
  const s = nanoPathNorm(p);
  return s ? s.split("/").length : 0;
}
// p 是否在 root 之下（含 root 自身）
function nanoPathWithin(p, root) {
  const a = nanoPathNorm(p), b = nanoPathNorm(root);
  if (!b) return true;
  return a === b || a.indexOf(b + "/") === 0;
}
// 文件名合法性（单段，不含路径分隔符）
function nanoNameValid(name) {
  const s = String(name === undefined || name === null ? "" : name).trim();
  if (!s) return false;
  if (s === "." || s === "..") return false;
  if (/[\/\\\0]/.test(s)) return false;
  if (/^\s|\s$/.test(s)) return false;
  return true;
}
// 字符串升序（按码元比较：结果与宿主 locale 无关，MEMFS/IndexedDB/浏览器一致）
function nanoCompareNames(a, b) {
  return a < b ? -1 : (a > b ? 1 : 0);
}

// ---------------- 目录树 ----------------

// entries: [{ path, dir?, size?, mtime?, source? }]（已剔除墓碑）
// 返回 { root, index }；root.path === ""，index: path → node
function nanoBuildTree(entries) {
  const root = {
    name: "资源根", path: "", dir: true, size: 0, source: "", mtime: 0,
    children: [], parent: null, implied: true,
  };
  const index = new Map();
  index.set("", root);

  function ensureDir(path) {
    if (!path) return root;
    let node = index.get(path);
    if (node) return node;
    const parent = ensureDir(nanoPathParent(path));
    node = {
      name: nanoPathBase(path), path, dir: true, size: 0, source: "", mtime: 0,
      children: [], parent, implied: true,
    };
    parent.children.push(node);
    index.set(path, node);
    return node;
  }

  const list = Array.isArray(entries) ? entries : [];
  // 先建显式目录（空目录也要出现），再落文件；顺序不影响结果，只影响 index 里的 implied 标记
  for (const e of list) {
    if (!e || !e.path || e.deleted) continue;
    const path = nanoPathNorm(e.path);
    if (!path) continue;
    if (e.dir) {
      const d = ensureDir(path);
      d.implied = false;
      d.mtime = e.mtime || 0;
      d.source = e.source || "";
    }
  }
  for (const e of list) {
    if (!e || !e.path || e.deleted || e.dir) continue;
    const path = nanoPathNorm(e.path);
    if (!path) continue;
    const parent = ensureDir(nanoPathParent(path));
    let node = index.get(path);
    if (!node) {
      node = { name: nanoPathBase(path), path, dir: false, size: 0, source: "", mtime: 0, children: [], parent };
      parent.children.push(node);
      index.set(path, node);
    }
    node.dir = false;
    node.size = e.size || 0;
    node.mtime = e.mtime || 0;
    node.source = e.source || "";
  }

  // 排序：目录在前、文件在后；各自按名称升序（递归）
  (function sortNode(node) {
    node.children.sort(function (a, b) {
      if (a.dir !== b.dir) return a.dir ? -1 : 1;
      return nanoCompareNames(a.name, b.name);
    });
    for (const c of node.children) sortNode(c);
  })(root);

  return { root, index };
}

// 目录的子孙文件/目录计数（不含自身）
function nanoCountDescendants(node) {
  let n = 0;
  const walk = function (x) {
    for (const c of x.children) { n++; walk(c); }
  };
  walk(node);
  return n;
}

// ---------------- 可见行（传统目录树） ----------------

// 可见行（传统目录树）：根的一级条目 + 所有已展开目录的子条目，前序
//   expanded: Set<path>；返回 [{ node, depth }]（顺序 = nanoBuildTree 的排序结果）
function nanoFlattenTree(root, isExpanded) {
  const expanded = isExpanded || new Set();
  const rows = [];
  if (!root) return rows;
  (function walk(node, depth) {
    for (const c of node.children) {
      rows.push({ node: c, depth });
      if (c.dir && expanded.has(c.path)) walk(c, depth + 1);
    }
  })(root, 0);
  return rows;
}

// ---------------- VFS：读 ----------------

function nanoDefaultsEntries(defaults) {
  if (!defaults) return [];
  if (defaults instanceof Map) {
    return [...defaults.entries()].map(function (kv) {
      const meta = kv[1];
      return { path: kv[0], size: meta && meta.size ? meta.size : 0 };
    });
  }
  return [...defaults].map(function (p) { return { path: p, size: 0 }; });
}
function nanoDefaultsHas(defaults, path) {
  if (!defaults) return false;
  if (defaults instanceof Map) return defaults.has(path);
  return defaults.has(path);
}

// 合并视图：服务器默认源（manifest）∪ 本地存储（store），剔除墓碑
// deps = { store, defaults() }
async function nanoVfsList(deps) {
  const defaults = deps.defaults ? deps.defaults() : null;
  const map = new Map();
  for (const e of nanoDefaultsEntries(defaults)) {
    const path = nanoPathNorm(e.path);
    if (path) map.set(path, { path, dir: false, size: e.size, source: "http", mtime: 0 });
  }
  let storeList = [];
  try { storeList = await deps.store.list(); } catch (e) { storeList = []; }
  const tombstones = new Set();
  for (const e of storeList) {
    if (e.deleted) { tombstones.add(nanoPathNorm(e.path)); continue; }
    const path = nanoPathNorm(e.path);
    if (!path) continue;
    if (e.dir) {
      map.set(path, { path, dir: true, size: 0, source: "store", mtime: e.mtime || 0 });
    } else {
      map.set(path, { path, dir: false, size: e.size || 0, source: "store", mtime: e.mtime || 0 });
    }
  }
  // 墓碑最后删：即使同路径有服务器默认源也一并隐藏
  for (const p of tombstones) map.delete(p);
  return [...map.values()];
}

// 墓碑清单（用于“已隐藏，可恢复”）
async function nanoVfsListDeleted(deps) {
  const defaults = deps.defaults ? deps.defaults() : null;
  let storeList = [];
  try { storeList = await deps.store.list(); } catch (e) { return []; }
  return storeList
    .filter(function (e) { return e.deleted; })
    .map(function (e) { return { path: e.path, size: e.size || 0, inDefault: nanoDefaultsHas(defaults, e.path) }; })
    .sort(function (a, b) { return nanoCompareNames(a.path, b.path); });
}

// 读字节：本地存储 → MEMFS → 服务器默认源
async function nanoVfsReadBytes(deps, path) {
  path = nanoPathNorm(path);
  if (!path) throw new Error("资源根不是文件");
  if (deps.store) {
    const rec = await deps.store.get(path);
    if (rec && rec.blob) return new Uint8Array(await rec.blob.arrayBuffer());
  }
  if (deps.memfs) {
    const b = await deps.memfs.read(path);
    if (b && b.length !== undefined) return b;
  }
  if (deps.fetchDefault) {
    const b = await deps.fetchDefault(path);
    if (b) return b;
  }
  throw new Error("读不到文件内容：" + path);
}

// ---------------- VFS：写 ----------------

// 写文件（上传/恢复）：持久层 + 当前会话 MEMFS 同时生效
async function nanoVfsWriteFile(deps, path, bytes) {
  path = nanoPathNorm(path);
  if (!path) throw new Error("资源根不是文件");
  const arr = (bytes instanceof Uint8Array) ? bytes : new Uint8Array(bytes);
  await deps.store.put(path, new Blob([arr]));
  if (deps.memfs) await deps.memfs.write(path, arr);
  return arr.length;
}

// 新建目录
async function nanoVfsMkdir(deps, path) {
  path = nanoPathNorm(path);
  if (!path) throw new Error("资源根已存在");
  await deps.store.putDir(path);
  if (deps.memfs) await deps.memfs.mkdir(path);
  return path;
}

// 移动/复制（同一实现；目录会整棵树一起走）
//   mode: "move" | "copy"；toPath 是**目标完整路径**（含新名字）
async function nanoVfsTransfer(deps, fromPath, toPath, mode) {
  fromPath = nanoPathNorm(fromPath);
  toPath = nanoPathNorm(toPath);
  if (!fromPath) throw new Error("资源根不可移动或复制");
  if (!toPath) throw new Error("目标路径为空");
  if (fromPath === toPath) return { count: 0 };
  if (nanoPathWithin(toPath, fromPath)) throw new Error("不能把目录放进它自己的子目录");
  if (!nanoNameValid(nanoPathBase(toPath))) throw new Error("目标名称不合法");

  const entries = await nanoVfsList(deps);
  const affected = entries
    .filter(function (e) { return nanoPathWithin(e.path, fromPath); })
    .sort(function (a, b) { return nanoPathDepth(a.path) - nanoPathDepth(b.path); });
  if (!affected.length) throw new Error("源不存在：" + fromPath);
  // 目录判定：自身是目录条目，或存在严格位于其下的条目（隐含目录，如 icon/）
  const isDir = affected.some(function (e) {
    return e.dir || e.path.indexOf(fromPath + "/") === 0;
  });
  const defaults = deps.defaults ? deps.defaults() : null;

  // ① 持久层：先读后写（此刻 MEMFS 里还是旧路径，读得到字节）
  for (const e of affected) {
    const dest = toPath + e.path.slice(fromPath.length);
    if (e.dir) {
      if (mode === "move") await deps.store.del(e.path);
      await deps.store.putDir(dest);
      continue;
    }
    if (mode === "move" && e.source === "store") {
      await deps.store.rename(e.path, dest);
    } else {
      let blob = null;
      if (e.source === "store") {
        const rec = await deps.store.get(e.path);
        blob = rec && rec.blob;
      }
      if (!blob) blob = new Blob([await nanoVfsReadBytes(deps, e.path)]);
      await deps.store.put(dest, blob);
      if (mode === "move") await deps.store.del(e.path);
    }
    // 移动走的若是“服务器默认资源”，旧路径要立墓碑，否则刷新后又挂回来
    if (mode === "move" && nanoDefaultsHas(defaults, e.path)) {
      await deps.store.tombstone(e.path, e.size);
    }
  }

  // ② 当前会话：整棵树一次搬（MEMFS rename 比逐文件读写真写快得多）
  if (deps.memfs) {
    if (mode === "move") await deps.memfs.move(fromPath, toPath, isDir);
    else await deps.memfs.copy(fromPath, toPath, isDir);
  }
  return { count: affected.length };
}

// 删除（目录递归）。服务器默认源用墓碑隐藏，本地存储记录直接删。
async function nanoVfsDelete(deps, path) {
  path = nanoPathNorm(path);
  if (!path) throw new Error("资源根不可删除");
  const entries = await nanoVfsList(deps);
  const affected = entries.filter(function (e) { return nanoPathWithin(e.path, path); });
  const defaults = deps.defaults ? deps.defaults() : null;
  for (const e of affected) {
    if (e.dir) { await deps.store.del(e.path); continue; }
    if (e.source === "store") await deps.store.del(e.path);
    if (nanoDefaultsHas(defaults, e.path)) await deps.store.tombstone(e.path, e.size);
  }
  if (deps.memfs) await deps.memfs.remove(path);
  return affected.length;
}

// 恢复墓碑（删除的服务器默认资源 / 被覆盖后删掉的默认版本）
async function nanoVfsRestore(deps, path) {
  path = nanoPathNorm(path);
  await deps.store.del(path); // 清墓碑（若同路径还有本地文件，这里会一并删掉 → 回到默认版本）
  let mounted = false;
  let error = null;
  if (deps.mountDefault) {
    try { mounted = !!(await deps.mountDefault(path)); } catch (e) { error = e; }
  }
  return { path, mounted, error };
}

// node 可测性导出
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = {
    nanoPathNorm, nanoPathJoin, nanoPathParent, nanoPathBase, nanoPathDepth, nanoPathWithin,
    nanoNameValid, nanoCompareNames,
    nanoBuildTree, nanoCountDescendants, nanoFlattenTree,
    nanoVfsList, nanoVfsListDeleted, nanoVfsReadBytes, nanoVfsWriteFile, nanoVfsMkdir,
    nanoVfsTransfer, nanoVfsDelete, nanoVfsRestore,
  };
}
