// https://github.com/dmarcos/llama2.c-web
// https://github.com/taybenlor/runno

var st = Object.defineProperty;
var ct = (e, t, i) => t in e ? st(e, t, { enumerable: !0, configurable: !0, writable: !0, value: i }) : e[t] = i;
var r = (e, t, i) => (ct(e, typeof t != "symbol" ? t + "" : t, i), i);
var c;
(function(e) {
  e[e.SUCCESS = 0] = "SUCCESS", e[e.E2BIG = 1] = "E2BIG", e[e.EACCESS = 2] = "EACCESS", e[e.EADDRINUSE = 3] = "EADDRINUSE", e[e.EADDRNOTAVAIL = 4] = "EADDRNOTAVAIL", e[e.EAFNOSUPPORT = 5] = "EAFNOSUPPORT", e[e.EAGAIN = 6] = "EAGAIN", e[e.EALREADY = 7] = "EALREADY", e[e.EBADF = 8] = "EBADF", e[e.EBADMSG = 9] = "EBADMSG", e[e.EBUSY = 10] = "EBUSY", e[e.ECANCELED = 11] = "ECANCELED", e[e.ECHILD = 12] = "ECHILD", e[e.ECONNABORTED = 13] = "ECONNABORTED", e[e.ECONNREFUSED = 14] = "ECONNREFUSED", e[e.ECONNRESET = 15] = "ECONNRESET", e[e.EDEADLK = 16] = "EDEADLK", e[e.EDESTADDRREQ = 17] = "EDESTADDRREQ", e[e.EDOM = 18] = "EDOM", e[e.EDQUOT = 19] = "EDQUOT", e[e.EEXIST = 20] = "EEXIST", e[e.EFAULT = 21] = "EFAULT", e[e.EFBIG = 22] = "EFBIG", e[e.EHOSTUNREACH = 23] = "EHOSTUNREACH", e[e.EIDRM = 24] = "EIDRM", e[e.EILSEQ = 25] = "EILSEQ", e[e.EINPROGRESS = 26] = "EINPROGRESS", e[e.EINTR = 27] = "EINTR", e[e.EINVAL = 28] = "EINVAL", e[e.EIO = 29] = "EIO", e[e.EISCONN = 30] = "EISCONN", e[e.EISDIR = 31] = "EISDIR", e[e.ELOOP = 32] = "ELOOP", e[e.EMFILE = 33] = "EMFILE", e[e.EMLINK = 34] = "EMLINK", e[e.EMSGSIZE = 35] = "EMSGSIZE", e[e.EMULTIHOP = 36] = "EMULTIHOP", e[e.ENAMETOOLONG = 37] = "ENAMETOOLONG", e[e.ENETDOWN = 38] = "ENETDOWN", e[e.ENETRESET = 39] = "ENETRESET", e[e.ENETUNREACH = 40] = "ENETUNREACH", e[e.ENFILE = 41] = "ENFILE", e[e.ENOBUFS = 42] = "ENOBUFS", e[e.ENODEV = 43] = "ENODEV", e[e.ENOENT = 44] = "ENOENT", e[e.ENOEXEC = 45] = "ENOEXEC", e[e.ENOLCK = 46] = "ENOLCK", e[e.ENOLINK = 47] = "ENOLINK", e[e.ENOMEM = 48] = "ENOMEM", e[e.ENOMSG = 49] = "ENOMSG", e[e.ENOPROTOOPT = 50] = "ENOPROTOOPT", e[e.ENOSPC = 51] = "ENOSPC", e[e.ENOSYS = 52] = "ENOSYS", e[e.ENOTCONN = 53] = "ENOTCONN", e[e.ENOTDIR = 54] = "ENOTDIR", e[e.ENOTEMPTY = 55] = "ENOTEMPTY", e[e.ENOTRECOVERABLE = 56] = "ENOTRECOVERABLE", e[e.ENOTSOCK = 57] = "ENOTSOCK", e[e.ENOTSUP = 58] = "ENOTSUP", e[e.ENOTTY = 59] = "ENOTTY", e[e.ENXIO = 60] = "ENXIO", e[e.EOVERFLOW = 61] = "EOVERFLOW", e[e.EOWNERDEAD = 62] = "EOWNERDEAD", e[e.EPERM = 63] = "EPERM", e[e.EPIPE = 64] = "EPIPE", e[e.EPROTO = 65] = "EPROTO", e[e.EPROTONOSUPPORT = 66] = "EPROTONOSUPPORT", e[e.EPROTOTYPE = 67] = "EPROTOTYPE", e[e.ERANGE = 68] = "ERANGE", e[e.EROFS = 69] = "EROFS", e[e.ESPIPE = 70] = "ESPIPE", e[e.ESRCH = 71] = "ESRCH", e[e.ESTALE = 72] = "ESTALE", e[e.ETIMEDOUT = 73] = "ETIMEDOUT", e[e.ETXTBSY = 74] = "ETXTBSY", e[e.EXDEV = 75] = "EXDEV", e[e.ENOTCAPABLE = 76] = "ENOTCAPABLE";
})(c || (c = {}));
var E;
(function(e) {
  e[e.REALTIME = 0] = "REALTIME", e[e.MONOTONIC = 1] = "MONOTONIC", e[e.PROCESS_CPUTIME_ID = 2] = "PROCESS_CPUTIME_ID", e[e.THREAD_CPUTIME_ID = 3] = "THREAD_CPUTIME_ID";
})(E || (E = {}));
var N;
(function(e) {
  e[e.SET = 0] = "SET", e[e.CUR = 1] = "CUR", e[e.END = 2] = "END";
})(N || (N = {}));
var S;
(function(e) {
  e[e.UNKNOWN = 0] = "UNKNOWN", e[e.BLOCK_DEVICE = 1] = "BLOCK_DEVICE", e[e.CHARACTER_DEVICE = 2] = "CHARACTER_DEVICE", e[e.DIRECTORY = 3] = "DIRECTORY", e[e.REGULAR_FILE = 4] = "REGULAR_FILE", e[e.SOCKET_DGRAM = 5] = "SOCKET_DGRAM", e[e.SOCKET_STREAM = 6] = "SOCKET_STREAM", e[e.SYMBOLIC_LINK = 7] = "SYMBOLIC_LINK";
})(S || (S = {}));
var Q;
(function(e) {
  e[e.DIR = 0] = "DIR";
})(Q || (Q = {}));
var G;
(function(e) {
  e[e.CLOCK = 0] = "CLOCK", e[e.FD_READ = 1] = "FD_READ", e[e.FD_WRITE = 2] = "FD_WRITE";
})(G || (G = {}));
const lt = {
  SYMLINK_FOLLOW: 1
  // As long as the resolved path corresponds to a symbolic
  // link, it is expanded.
}, T = {
  CREAT: 1,
  DIRECTORY: 2,
  EXCL: 4,
  TRUNC: 8
  // Truncate file to size 0.
}, p = {
  APPEND: 1,
  DSYNC: 2,
  NONBLOCK: 4,
  RSYNC: 8,
  SYNC: 16
  // Write according to synchronized I/O file integrity completion. In addition to synchronizing the data stored in the file, the implementation may also synchronously update the file's metadata.
}, o = {
  FD_DATASYNC: BigInt(1) << BigInt(0),
  FD_READ: BigInt(1) << BigInt(1),
  FD_SEEK: BigInt(1) << BigInt(2),
  FD_FDSTAT_SET_FLAGS: BigInt(1) << BigInt(3),
  FD_SYNC: BigInt(1) << BigInt(4),
  FD_TELL: BigInt(1) << BigInt(5),
  FD_WRITE: BigInt(1) << BigInt(6),
  FD_ADVISE: BigInt(1) << BigInt(7),
  FD_ALLOCATE: BigInt(1) << BigInt(8),
  PATH_CREATE_DIRECTORY: BigInt(1) << BigInt(9),
  PATH_CREATE_FILE: BigInt(1) << BigInt(10),
  PATH_LINK_SOURCE: BigInt(1) << BigInt(11),
  PATH_LINK_TARGET: BigInt(1) << BigInt(12),
  PATH_OPEN: BigInt(1) << BigInt(13),
  FD_READDIR: BigInt(1) << BigInt(14),
  PATH_READLINK: BigInt(1) << BigInt(15),
  PATH_RENAME_SOURCE: BigInt(1) << BigInt(16),
  PATH_RENAME_TARGET: BigInt(1) << BigInt(17),
  PATH_FILESTAT_GET: BigInt(1) << BigInt(18),
  PATH_FILESTAT_SET_SIZE: BigInt(1) << BigInt(19),
  PATH_FILESTAT_SET_TIMES: BigInt(1) << BigInt(20),
  FD_FILESTAT_GET: BigInt(1) << BigInt(21),
  FD_FILESTAT_SET_SIZE: BigInt(1) << BigInt(22),
  FD_FILESTAT_SET_TIMES: BigInt(1) << BigInt(23),
  PATH_SYMLINK: BigInt(1) << BigInt(24),
  PATH_REMOVE_DIRECTORY: BigInt(1) << BigInt(25),
  PATH_UNLINK_FILE: BigInt(1) << BigInt(26),
  POLL_FD_READWRITE: BigInt(1) << BigInt(27),
  SOCK_SHUTDOWN: BigInt(1) << BigInt(28),
  SOCK_ACCEPT: BigInt(1) << BigInt(29)
}, U = {
  ATIM: 1,
  ATIM_NOW: 2,
  MTIM: 4,
  MTIM_NOW: 8
  // Adjust the last data modification timestamp to the time of clock clockid::realtime.
}, $ = {
  SUBSCRIPTION_CLOCK_ABSTIME: 1
  // If set, treat the timestamp provided in subscription_clock::timeout as an absolute timestamp of clock subscription_clock::id. If clear, treat the timestamp provided in subscription_clock::timeout relative to the current time value of clock subscription_clock::id.
}, dt = {
  FD_READWRITE_HANGUP: 1
  // The peer of this socket has closed or disconnected.
}, O = 64, x = 48, B = 32, Xt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  get Clock() {
    return E;
  },
  EVENT_SIZE: B,
  EventReadWriteFlags: dt,
  get EventType() {
    return G;
  },
  FILESTAT_SIZE: O,
  FileDescriptorFlags: p,
  FileStatTimestampFlags: U,
  get FileType() {
    return S;
  },
  LookupFlags: lt,
  OpenFlags: T,
  get PreopenType() {
    return Q;
  },
  get Result() {
    return c;
  },
  RightsFlags: o,
  SUBSCRIPTION_SIZE: x,
  SubscriptionClockFlags: $,
  get Whence() {
    return N;
  }
}, Symbol.toStringTag, { value: "Module" }));
var I;
(function(e) {
  e[e.CUR = 0] = "CUR", e[e.END = 1] = "END", e[e.SET = 2] = "SET";
})(I || (I = {}));
class at {
  constructor(t) {
    r(this, "fs");
    r(this, "args");
    // Program args (like from a terminal program)
    r(this, "env");
    // Environment (like a .env file)
    r(this, "stdin");
    r(this, "stdout");
    r(this, "stderr");
    r(this, "debug");
    r(this, "isTTY");
    this.fs = (t == null ? void 0 : t.fs) ?? {}, this.args = (t == null ? void 0 : t.args) ?? [], this.env = (t == null ? void 0 : t.env) ?? {}, this.stdin = (t == null ? void 0 : t.stdin) ?? (() => null), this.stdout = (t == null ? void 0 : t.stdout) ?? (() => {
    }), this.stderr = (t == null ? void 0 : t.stderr) ?? (() => {
    }), this.debug = t == null ? void 0 : t.debug, this.isTTY = !!(t != null && t.isTTY);
  }
}
class Vt {
  constructor(t) {
    r(this, "fs");
    r(this, "nextFD", 10);
    r(this, "openMap", /* @__PURE__ */ new Map());
    this.fs = { ...t }, this.openMap.set(3, new b(this.fs, "/"));
  }
  //
  // Helpers
  //
  openFile(t, i, n) {
    const s = new X(t, n);
    i && (s.buffer = new Uint8Array(new ArrayBuffer(1024), 0, 0));
    const l = this.nextFD;
    return this.openMap.set(l, s), this.nextFD++, [c.SUCCESS, l];
  }
  openDir(t, i) {
    const n = new b(t, i), s = this.nextFD;
    return this.openMap.set(s, n), this.nextFD++, [c.SUCCESS, s];
  }
  hasDir(t, i) {
    return i === "." ? !0 : t.containsDirectory(i);
  }
  //
  // Public Interface
  //
  open(t, i, n, s) {
    const l = !!(n & T.CREAT), d = !!(n & T.DIRECTORY), V = !!(n & T.EXCL), a = !!(n & T.TRUNC), Z = this.openMap.get(t);
    if (!(Z instanceof b))
      return [c.EBADF];
    if (Z.containsFile(i))
      return d ? [c.ENOTDIR] : V ? [c.EEXIST] : this.openFile(Z.get(i), a, s);
    if (this.hasDir(Z, i)) {
      if (i === ".")
        return this.openDir(this.fs, "/");
      const h = `/${i}/`, m = Object.entries(this.fs).filter(([u]) => u.startsWith(h));
      return this.openDir(Object.fromEntries(m), h);
    } else {
      if (l) {
        const h = Z.fullPath(i);
        return this.fs[h] = {
          path: h,
          mode: "binary",
          content: new Uint8Array(),
          timestamps: {
            access: /* @__PURE__ */ new Date(),
            modification: /* @__PURE__ */ new Date(),
            change: /* @__PURE__ */ new Date()
          }
        }, this.openFile(this.fs[h], a, s);
      }
      return [c.ENOTCAPABLE];
    }
  }
  close(t) {
    if (!this.openMap.has(t))
      return c.EBADF;
    const i = this.openMap.get(t);
    return i instanceof X && i.sync(), this.openMap.delete(t), c.SUCCESS;
  }
  read(t, i) {
    const n = this.openMap.get(t);
    return !n || n instanceof b ? [c.EBADF] : [c.SUCCESS, n.read(i)];
  }
  pread(t, i, n) {
    const s = this.openMap.get(t);
    return !s || s instanceof b ? [c.EBADF] : [c.SUCCESS, s.pread(i, n)];
  }
  write(t, i) {
    const n = this.openMap.get(t);
    return !n || n instanceof b ? c.EBADF : (n.write(i), c.SUCCESS);
  }
  pwrite(t, i, n) {
    const s = this.openMap.get(t);
    return !s || s instanceof b ? c.EBADF : (s.pwrite(i, n), c.SUCCESS);
  }
  sync(t) {
    const i = this.openMap.get(t);
    return !i || i instanceof b ? c.EBADF : (i.sync(), c.SUCCESS);
  }
  seek(t, i, n) {
    const s = this.openMap.get(t);
    return !s || s instanceof b ? [c.EBADF] : [c.SUCCESS, s.seek(i, n)];
  }
  tell(t) {
    const i = this.openMap.get(t);
    return !i || i instanceof b ? [c.EBADF] : [c.SUCCESS, i.tell()];
  }
  renumber(t, i) {
    return !this.exists(t) || !this.exists(i) ? c.EBADF : (t === i || (this.close(i), this.openMap.set(i, this.openMap.get(t))), c.SUCCESS);
  }
  unlink(t, i) {
    const n = this.openMap.get(t);
    if (!(n instanceof b))
      return c.EBADF;
    if (!n.contains(i))
      return c.ENOENT;
    for (const s of Object.keys(this.fs))
      (s === n.fullPath(i) || s.startsWith(`${n.fullPath(i)}/`)) && delete this.fs[s];
    return c.SUCCESS;
  }
  rename(t, i, n, s) {
    const l = this.openMap.get(t), d = this.openMap.get(n);
    if (!(l instanceof b) || !(d instanceof b))
      return c.EBADF;
    if (!l.contains(i))
      return c.ENOENT;
    if (d.contains(s))
      return c.EEXIST;
    const V = l.fullPath(i), a = d.fullPath(s);
    for (const Z of Object.keys(this.fs))
      if (Z.startsWith(V)) {
        const h = Z.replace(V, a);
        this.fs[h] = this.fs[Z], this.fs[h].path = h, delete this.fs[Z];
      }
    return c.SUCCESS;
  }
  list(t) {
    const i = this.openMap.get(t);
    return i instanceof b ? [c.SUCCESS, i.list()] : [c.EBADF];
  }
  stat(t) {
    const i = this.openMap.get(t);
    return i instanceof X ? [c.SUCCESS, i.stat()] : [c.EBADF];
  }
  pathStat(t, i) {
    const n = this.openMap.get(t);
    if (!(n instanceof b))
      return [c.EBADF];
    if (n.containsFile(i)) {
      const s = n.fullPath(i), l = new X(this.fs[s], 0).stat();
      return [c.SUCCESS, l];
    } else if (this.hasDir(n, i)) {
      if (i === ".")
        return [c.SUCCESS, new b(this.fs, "/").stat()];
      const s = `/${i}/`, l = Object.entries(this.fs).filter(([V]) => V.startsWith(s)), d = new b(Object.fromEntries(l), s).stat();
      return [c.SUCCESS, d];
    } else
      return [c.ENOTCAPABLE];
  }
  setFlags(t, i) {
    const n = this.openMap.get(t);
    return n instanceof X ? (n.setFlags(i), c.SUCCESS) : c.EBADF;
  }
  setSize(t, i) {
    const n = this.openMap.get(t);
    return n instanceof X ? (n.setSize(Number(i)), c.SUCCESS) : c.EBADF;
  }
  setAccessTime(t, i) {
    const n = this.openMap.get(t);
    return n instanceof X ? (n.setAccessTime(i), c.SUCCESS) : c.EBADF;
  }
  setModificationTime(t, i) {
    const n = this.openMap.get(t);
    return n instanceof X ? (n.setModificationTime(i), c.SUCCESS) : c.EBADF;
  }
  pathSetAccessTime(t, i, n) {
    const s = this.openMap.get(t);
    if (!(s instanceof b))
      return c.EBADF;
    const l = s.get(i);
    if (!l)
      return c.ENOTCAPABLE;
    const d = new X(l, 0);
    return d.setAccessTime(n), d.sync(), c.SUCCESS;
  }
  pathSetModificationTime(t, i, n) {
    const s = this.openMap.get(t);
    if (!(s instanceof b))
      return c.EBADF;
    const l = s.get(i);
    if (!l)
      return c.ENOTCAPABLE;
    const d = new X(l, 0);
    return d.setModificationTime(n), d.sync(), c.SUCCESS;
  }
  pathCreateDir(t, i) {
    const n = this.openMap.get(t);
    if (!(n instanceof b))
      return c.EBADF;
    if (n.contains(i))
      return c.ENOTCAPABLE;
    const s = `${n.fullPath(i)}/.runno`;
    return this.fs[s] = {
      path: s,
      timestamps: {
        access: /* @__PURE__ */ new Date(),
        modification: /* @__PURE__ */ new Date(),
        change: /* @__PURE__ */ new Date()
      },
      mode: "string",
      content: ""
    }, c.SUCCESS;
  }
  //
  // Public Helpers
  //
  exists(t) {
    return this.openMap.has(t);
  }
  fileType(t) {
    const i = this.openMap.get(t);
    return i ? i instanceof X ? S.REGULAR_FILE : S.DIRECTORY : S.UNKNOWN;
  }
  fileFdflags(t) {
    const i = this.openMap.get(t);
    return i instanceof X ? i.fdflags : 0;
  }
}
class X {
  constructor(t, i) {
    r(this, "file");
    r(this, "buffer");
    r(this, "_offset", BigInt(0));
    r(this, "isDirty", !1);
    r(this, "fdflags");
    r(this, "flagAppend");
    r(this, "flagDSync");
    r(this, "flagNonBlock");
    r(this, "flagRSync");
    r(this, "flagSync");
    if (this.file = t, this.file.mode === "string") {
      const n = new TextEncoder();
      this.buffer = n.encode(this.file.content);
    } else
      this.buffer = this.file.content;
    this.fdflags = i, this.flagAppend = !!(i & p.APPEND), this.flagDSync = !!(i & p.DSYNC), this.flagNonBlock = !!(i & p.NONBLOCK), this.flagRSync = !!(i & p.RSYNC), this.flagSync = !!(i & p.SYNC);
  }
  get offset() {
    return Number(this._offset);
  }
  read(t) {
    const i = this.buffer.subarray(this.offset, this.offset + t);
    return this._offset += BigInt(i.length), i;
  }
  pread(t, i) {
    return this.buffer.subarray(i, i + t);
  }
  write(t) {
    if (this.isDirty = !0, this.flagAppend) {
      const i = this.buffer.length;
      this.resize(i + t.byteLength), this.buffer.set(t, i);
    } else {
      const i = Math.max(this.offset + t.byteLength, this.buffer.byteLength);
      this.resize(i), this.buffer.set(t, this.offset), this._offset += BigInt(t.byteLength);
    }
    (this.flagDSync || this.flagSync) && this.sync();
  }
  pwrite(t, i) {
    if (this.isDirty = !0, this.flagAppend) {
      const n = this.buffer.length;
      this.resize(n + t.byteLength), this.buffer.set(t, n);
    } else {
      const n = Math.max(i + t.byteLength, this.buffer.byteLength);
      this.resize(n), this.buffer.set(t, i);
    }
    (this.flagDSync || this.flagSync) && this.sync();
  }
  sync() {
    if (!this.isDirty)
      return;
    if (this.isDirty = !1, this.file.mode === "binary") {
      this.file.content = new Uint8Array(this.buffer);
      return;
    }
    const t = new TextDecoder();
    this.file.content = t.decode(this.buffer);
  }
  seek(t, i) {
    switch (i) {
      case N.SET:
        this._offset = t;
        break;
      case N.CUR:
        this._offset += t;
        break;
      case N.END:
        this._offset = BigInt(this.buffer.length) + t;
        break;
    }
    return this._offset;
  }
  tell() {
    return this._offset;
  }
  stat() {
    return {
      path: this.file.path,
      timestamps: this.file.timestamps,
      type: S.REGULAR_FILE,
      byteLength: this.buffer.length
    };
  }
  setFlags(t) {
    this.fdflags = t;
  }
  setSize(t) {
    this.resize(t);
  }
  setAccessTime(t) {
    this.file.timestamps.access = t;
  }
  setModificationTime(t) {
    this.file.timestamps.modification = t;
  }
  /**
   * Resizes the buffer to be exactly requiredBytes length, while resizing the
   * underlying buffer to be larger if necessary.
   *
   * Resizing will internally double the buffer size to reduce the need for
   * resizing often.
   *
   * @param requiredBytes how many bytes the buffer needs to have available
   */
  resize(t) {
    if (t <= this.buffer.buffer.byteLength) {
      this.buffer = new Uint8Array(this.buffer.buffer, 0, t);
      return;
    }
    let i;
    this.buffer.buffer.byteLength === 0 ? i = new ArrayBuffer(t < 1024 ? 1024 : t * 2) : t > this.buffer.buffer.byteLength * 2 ? i = new ArrayBuffer(t * 2) : i = new ArrayBuffer(this.buffer.buffer.byteLength * 2);
    const n = new Uint8Array(i, 0, t);
    n.set(this.buffer), this.buffer = n;
  }
}
function Y(e, t) {
  const i = t.replace(/[/\-\\^$*+?.()|[\]{}]/g, "\\$&"), n = new RegExp(`^${i}`);
  return e.replace(n, "");
}
class b {
  // full folder path including /
  constructor(t, i) {
    r(this, "dir");
    r(this, "prefix");
    this.dir = t, this.prefix = i;
  }
  containsFile(t) {
    for (const i of Object.keys(this.dir))
      if (Y(i, this.prefix) === t)
        return !0;
    return !1;
  }
  containsDirectory(t) {
    for (const i of Object.keys(this.dir))
      if (Y(i, this.prefix).startsWith(`${t}/`))
        return !0;
    return !1;
  }
  contains(t) {
    for (const i of Object.keys(this.dir)) {
      const n = Y(i, this.prefix);
      if (n === t || n.startsWith(`${t}/`))
        return !0;
    }
    return !1;
  }
  get(t) {
    return this.dir[this.fullPath(t)];
  }
  fullPath(t) {
    return `${this.prefix}${t}`;
  }
  list() {
    const t = [], i = /* @__PURE__ */ new Set();
    for (const n of Object.keys(this.dir)) {
      const s = Y(n, this.prefix);
      if (s.includes("/")) {
        const l = s.split("/")[0];
        if (i.has(l))
          continue;
        i.add(l), t.push({ name: l, type: S.DIRECTORY });
      } else
        t.push({
          name: s,
          type: S.REGULAR_FILE
        });
    }
    return t;
  }
  stat() {
    return {
      path: this.prefix,
      timestamps: {
        access: /* @__PURE__ */ new Date(),
        modification: /* @__PURE__ */ new Date(),
        change: /* @__PURE__ */ new Date()
      },
      type: S.DIRECTORY,
      byteLength: 0
    };
  }
}
let z = [];
function f(e) {
  z.push(e);
}
function Zt() {
  const e = z;
  return z = [], e;
}
class q {
  constructor(t) {
    r(this, "instance");
    r(this, "module");
    r(this, "memory");
    r(this, "context");
    r(this, "drive");
    this.context = new at(t), this.drive = new Vt(this.context.fs);
  }
  static async start(t, i = {}) {
    const n = new q(i), s = await WebAssembly.instantiateStreaming(t, n.getImportObject());
    return n.start(s);
  }
  getImportObject() {
    return {
      wasi_snapshot_preview1: this.getImports("preview1", this.context.debug),
      wasi_unstable: this.getImports("unstable", this.context.debug)
    };
  }
  start(t, i = {}) {
    this.instance = t.instance, this.module = t.module, this.memory = i.memory ?? this.instance.exports.memory;
    const n = this.instance.exports._start;
    try {
      n();
    } catch (s) {
      if (s instanceof H)
        return {
          exitCode: s.code,
          fs: this.drive.fs
        };
      if (s instanceof WebAssembly.RuntimeError)
        return {
          exitCode: 134,
          fs: this.drive.fs
        };
      throw s;
    }
    return {
      exitCode: 0,
      fs: this.drive.fs
    };
  }
  getImports(t, i) {
    const n = {
      args_get: this.args_get.bind(this),
      args_sizes_get: this.args_sizes_get.bind(this),
      clock_res_get: this.clock_res_get.bind(this),
      clock_time_get: this.clock_time_get.bind(this),
      environ_get: this.environ_get.bind(this),
      environ_sizes_get: this.environ_sizes_get.bind(this),
      proc_exit: this.proc_exit.bind(this),
      random_get: this.random_get.bind(this),
      sched_yield: this.sched_yield.bind(this),
      // File Descriptors
      fd_advise: this.fd_advise.bind(this),
      fd_allocate: this.fd_allocate.bind(this),
      fd_close: this.fd_close.bind(this),
      fd_datasync: this.fd_datasync.bind(this),
      fd_fdstat_get: this.fd_fdstat_get.bind(this),
      fd_fdstat_set_flags: this.fd_fdstat_set_flags.bind(this),
      fd_fdstat_set_rights: this.fd_fdstat_set_rights.bind(this),
      fd_filestat_get: this.fd_filestat_get.bind(this),
      fd_filestat_set_size: this.fd_filestat_set_size.bind(this),
      fd_filestat_set_times: this.fd_filestat_set_times.bind(this),
      fd_pread: this.fd_pread.bind(this),
      fd_prestat_dir_name: this.fd_prestat_dir_name.bind(this),
      fd_prestat_get: this.fd_prestat_get.bind(this),
      fd_pwrite: this.fd_pwrite.bind(this),
      fd_read: this.fd_read.bind(this),
      fd_readdir: this.fd_readdir.bind(this),
      fd_renumber: this.fd_renumber.bind(this),
      fd_seek: this.fd_seek.bind(this),
      fd_sync: this.fd_sync.bind(this),
      fd_tell: this.fd_tell.bind(this),
      fd_write: this.fd_write.bind(this),
      // Paths
      path_filestat_get: this.path_filestat_get.bind(this),
      path_filestat_set_times: this.path_filestat_set_times.bind(this),
      path_open: this.path_open.bind(this),
      path_rename: this.path_rename.bind(this),
      path_unlink_file: this.path_unlink_file.bind(this),
      path_create_directory: this.path_create_directory.bind(this),
      // Unimplemented
      path_link: this.path_link.bind(this),
      path_readlink: this.path_readlink.bind(this),
      path_remove_directory: this.path_remove_directory.bind(this),
      path_symlink: this.path_symlink.bind(this),
      poll_oneoff: this.poll_oneoff.bind(this),
      proc_raise: this.proc_raise.bind(this),
      sock_accept: this.sock_accept.bind(this),
      sock_recv: this.sock_recv.bind(this),
      sock_send: this.sock_send.bind(this),
      sock_shutdown: this.sock_shutdown.bind(this),
      // Unimplemented - WASMEdge compatibility
      sock_open: this.sock_open.bind(this),
      sock_listen: this.sock_listen.bind(this),
      sock_connect: this.sock_connect.bind(this),
      sock_setsockopt: this.sock_setsockopt.bind(this),
      sock_bind: this.sock_bind.bind(this),
      sock_getlocaladdr: this.sock_getlocaladdr.bind(this),
      sock_getpeeraddr: this.sock_getpeeraddr.bind(this),
      sock_getaddrinfo: this.sock_getaddrinfo.bind(this)
    };
    t === "unstable" && (n.path_filestat_get = this.unstable_path_filestat_get.bind(this), n.fd_filestat_get = this.unstable_fd_filestat_get.bind(this), n.fd_seek = this.unstable_fd_seek.bind(this));
    for (const [s, l] of Object.entries(n))
      n[s] = function() {
        let d = l.apply(this, arguments);
        if (i) {
          const V = Zt();
          d = i(s, [...arguments], d, V) ?? d;
        }
        return d;
      };
    return n;
  }
  //
  // Helpers
  //
  get envArray() {
    return Object.entries(this.context.env).map(([t, i]) => `${t}=${i}`);
  }
  //
  // WASI Implementation
  //
  /**
   * Read command-line argument data. The size of the array should match that
   * returned by args_sizes_get. Each argument is expected to be \0 terminated.
   */
  args_get(t, i) {
    const n = new DataView(this.memory.buffer);
    for (const s of this.context.args) {
      n.setUint32(t, i, !0), t += 4;
      const l = new TextEncoder().encode(`${s}\0`);
      new Uint8Array(this.memory.buffer, i, l.byteLength).set(l), i += l.byteLength;
    }
    return c.SUCCESS;
  }
  /**
   * Return command-line argument data sizes.
   */
  args_sizes_get(t, i) {
    const n = this.context.args, s = n.reduce((d, V) => d + new TextEncoder().encode(`${V}\0`).byteLength, 0), l = new DataView(this.memory.buffer);
    return l.setUint32(t, n.length, !0), l.setUint32(i, s, !0), c.SUCCESS;
  }
  /**
   * Return the resolution of a clock. Implementations are required to provide a
   * non-zero value for supported clocks. For unsupported clocks, return
   * errno::inval. Note: This is similar to clock_getres in POSIX.
   */
  clock_res_get(t, i) {
    switch (t) {
      case E.REALTIME:
      case E.MONOTONIC:
      case E.PROCESS_CPUTIME_ID:
      case E.THREAD_CPUTIME_ID:
        return new DataView(this.memory.buffer).setBigUint64(i, BigInt(1e6), !0), c.SUCCESS;
    }
    return c.EINVAL;
  }
  /**
   * Return the time value of a clock.
   * Note: This is similar to clock_gettime in POSIX.
   */
  clock_time_get(t, i, n) {
    switch (t) {
      case E.REALTIME:
      case E.MONOTONIC:
      case E.PROCESS_CPUTIME_ID:
      case E.THREAD_CPUTIME_ID:
        return new DataView(this.memory.buffer).setBigUint64(n, W(/* @__PURE__ */ new Date()), !0), c.SUCCESS;
    }
    return c.EINVAL;
  }
  /**
   * Read environment variable data. The sizes of the buffers should match that
   * returned by environ_sizes_get. Key/value pairs are expected to be joined
   * with =s, and terminated with \0s.
   */
  environ_get(t, i) {
    const n = new DataView(this.memory.buffer);
    for (const s of this.envArray) {
      n.setUint32(t, i, !0), t += 4;
      const l = new TextEncoder().encode(`${s}\0`);
      new Uint8Array(this.memory.buffer, i, l.byteLength).set(l), i += l.byteLength;
    }
    return c.SUCCESS;
  }
  /**
   * Return environment variable data sizes.
   */
  environ_sizes_get(t, i) {
    const n = this.envArray.reduce((l, d) => l + new TextEncoder().encode(`${d}\0`).byteLength, 0), s = new DataView(this.memory.buffer);
    return s.setUint32(t, this.envArray.length, !0), s.setUint32(i, n, !0), c.SUCCESS;
  }
  /**
   * Terminate the process normally. An exit code of 0 indicates successful
   * termination of the program. The meanings of other values is dependent on
   * the environment.
   */
  proc_exit(t) {
    throw new H(t);
  }
  /**
   * Write high-quality random data into a buffer. This function blocks when the
   * implementation is unable to immediately provide sufficient high-quality
   * random data. This function may execute slowly, so when large mounts of
   * random data are required, it's advisable to use this function to seed a
   * pseudo-random number generator, rather than to provide the random data
   * directly.
   */
  random_get(t, i) {
    const n = new Uint8Array(this.memory.buffer, t, i);
    return globalThis.crypto.getRandomValues(n), c.SUCCESS;
  }
  /**
   * Temporarily yield execution of the calling thread.
   * Note: This is similar to sched_yield in POSIX.
   */
  sched_yield() {
    return c.SUCCESS;
  }
  //
  // File Descriptors
  //
  /**
   * Read from a file descriptor. Note: This is similar to readv in POSIX.
   */
  fd_read(t, i, n, s) {
    if (t === 1 || t === 2)
      return c.ENOTSUP;
    const l = new DataView(this.memory.buffer), d = g(l, i, n), V = new TextEncoder();
    let a = 0, Z = c.SUCCESS;
    for (const h of d) {
      let m;
      if (t === 0) {
        const R = this.context.stdin(h.byteLength);
        if (!R)
          break;
        m = V.encode(R);
      } else {
        const [R, L] = this.drive.read(t, h.byteLength);
        if (R) {
          Z = R;
          break;
        } else
          m = L;
      }
      const u = Math.min(h.byteLength, m.byteLength);
      h.set(m.subarray(0, u)), a += u;
    }
    return f({ bytesRead: a }), l.setUint32(s, a, !0), Z;
  }
  /**
   * Write to a file descriptor. Note: This is similar to writev in POSIX.
   */
  fd_write(t, i, n, s) {
    if (t === 0)
      return c.ENOTSUP;
    const l = new DataView(this.memory.buffer), d = g(l, i, n), V = new TextDecoder();
    let a = 0, Z = c.SUCCESS;
    for (const h of d)
      if (h.byteLength !== 0) {
        if (t === 1 || t === 2) {
          const m = t === 1 ? this.context.stdout : this.context.stderr, u = V.decode(h);
          m(u), f({ output: u });
        } else if (Z = this.drive.write(t, h), Z != c.SUCCESS)
          break;
        a += h.byteLength;
      }
    return l.setUint32(s, a, !0), Z;
  }
  /**
   * Provide file advisory information on a file descriptor.
   * Note: This is similar to posix_fadvise in POSIX.
   */
  fd_advise() {
    return c.SUCCESS;
  }
  /**
   * Force the allocation of space in a file.
   * Note: This is similar to posix_fallocate in POSIX.
   */
  fd_allocate(t, i, n) {
    return this.drive.pwrite(t, new Uint8Array(Number(n)), Number(i));
  }
  /**
   * Close a file descriptor.
   * Note: This is similar to close in POSIX.
   *
   * @param fd
   */
  fd_close(t) {
    return this.drive.close(t);
  }
  /**
   * Synchronize the data of a file to disk.
   * Note: This is similar to fdatasync in POSIX.
   *
   * @param fd
   */
  fd_datasync(t) {
    return this.drive.sync(t);
  }
  /**
   * Get the attributes of a file descriptor.
   * Note: This returns similar flags to fsync(fd, F_GETFL) in POSIX,
   * as well as additional fields.
   *
   * Returns fdstat - the buffer where the file descriptor's attributes
   * are stored.
   *
   * @returns Result<fdstat, errno>
   */
  fd_fdstat_get(t, i) {
    if (t < 3) {
      let V;
      if (this.context.isTTY) {
        const Z = D ^ o.FD_SEEK ^ o.FD_TELL;
        V = M(S.CHARACTER_DEVICE, 0, Z);
      } else
        V = M(S.CHARACTER_DEVICE, 0);
      return new Uint8Array(this.memory.buffer, i, V.byteLength).set(V), c.SUCCESS;
    }
    if (!this.drive.exists(t))
      return c.EBADF;
    const n = this.drive.fileType(t), s = this.drive.fileFdflags(t), l = M(n, s);
    return new Uint8Array(this.memory.buffer, i, l.byteLength).set(l), c.SUCCESS;
  }
  /**
   * Adjust the flags associated with a file descriptor.
   * Note: This is similar to fcntl(fd, F_SETFL, flags) in POSIX.
   */
  fd_fdstat_set_flags(t, i) {
    return this.drive.setFlags(t, i);
  }
  /**
   * Adjust the rights associated with a file descriptor. This can only be used
   * to remove rights, and returns errno::notcapable if called in a way that
   * would attempt to add rights
   */
  fd_fdstat_set_rights() {
    return c.SUCCESS;
  }
  /**
   * Return the attributes of an open file.
   */
  fd_filestat_get(t, i) {
    return this.shared_fd_filestat_get(t, i, "preview1");
  }
  /**
   * Return the attributes of an open file.
   * This version is used
   */
  unstable_fd_filestat_get(t, i) {
    return this.shared_fd_filestat_get(t, i, "unstable");
  }
  /**
   * Return the attributes of an open file.
   */
  shared_fd_filestat_get(t, i, n) {
    const s = n === "unstable" ? A : _;
    if (t < 3) {
      let Z;
      switch (t) {
        case 0:
          Z = "/dev/stdin";
          break;
        case 1:
          Z = "/dev/stdout";
          break;
        case 2:
          Z = "/dev/stderr";
          break;
        default:
          Z = "/dev/undefined";
          break;
      }
      const h = s({
        path: Z,
        byteLength: 0,
        timestamps: {
          access: /* @__PURE__ */ new Date(),
          modification: /* @__PURE__ */ new Date(),
          change: /* @__PURE__ */ new Date()
        },
        type: S.CHARACTER_DEVICE
      });
      return new Uint8Array(this.memory.buffer, i, h.byteLength).set(h), c.SUCCESS;
    }
    const [l, d] = this.drive.stat(t);
    if (l != c.SUCCESS)
      return l;
    f({ resolvedPath: d.path, stat: d });
    const V = s(d);
    return new Uint8Array(this.memory.buffer, i, V.byteLength).set(V), c.SUCCESS;
  }
  /**
   * Adjust the size of an open file. If this increases the file's size, the
   * extra bytes are filled with zeros. Note: This is similar to ftruncate in
   * POSIX.
   */
  fd_filestat_set_size(t, i) {
    return this.drive.setSize(t, i);
  }
  /**
   * Adjust the timestamps of an open file or directory.
   * Note: This is similar to futimens in POSIX.
   */
  fd_filestat_set_times(t, i, n, s) {
    let l = null;
    s & U.ATIM && (l = k(i)), s & U.ATIM_NOW && (l = /* @__PURE__ */ new Date());
    let d = null;
    if (s & U.MTIM && (d = k(n)), s & U.MTIM_NOW && (d = /* @__PURE__ */ new Date()), l) {
      const V = this.drive.setAccessTime(t, l);
      if (V != c.SUCCESS)
        return V;
    }
    if (d) {
      const V = this.drive.setModificationTime(t, d);
      if (V != c.SUCCESS)
        return V;
    }
    return c.SUCCESS;
  }
  /**
   * Read from a file descriptor, without using and updating the file
   * descriptor's offset. Note: This is similar to preadv in POSIX.
   */
  fd_pread(t, i, n, s, l) {
    if (t === 1 || t === 2)
      return c.ENOTSUP;
    if (t === 0)
      return this.fd_read(t, i, n, l);
    const d = new DataView(this.memory.buffer), V = g(d, i, n);
    let a = 0, Z = c.SUCCESS;
    for (const h of V) {
      const [m, u] = this.drive.pread(t, h.byteLength, Number(s) + a);
      if (m !== c.SUCCESS) {
        Z = m;
        break;
      }
      const R = Math.min(h.byteLength, u.byteLength);
      h.set(u.subarray(0, R)), a += R;
    }
    return d.setUint32(l, a, !0), Z;
  }
  /**
   * Return a description of the given preopened file descriptor.
   */
  fd_prestat_dir_name(t, i, n) {
    if (t !== 3)
      return c.EBADF;
    const s = new TextEncoder().encode("/");
    return new Uint8Array(this.memory.buffer, i, n).set(s.subarray(0, n)), c.SUCCESS;
  }
  /**
   * Return a description of the given preopened file descriptor.
   */
  fd_prestat_get(t, i) {
    if (t !== 3)
      return c.EBADF;
    const n = new TextEncoder().encode("."), s = new DataView(this.memory.buffer, i);
    return s.setUint8(0, Q.DIR), s.setUint32(4, n.byteLength, !0), c.SUCCESS;
  }
  /**
   * Write to a file descriptor, without using and updating the file
   * descriptor's offset. Note: This is similar to pwritev in POSIX.
   */
  fd_pwrite(t, i, n, s, l) {
    if (t === 0)
      return c.ENOTSUP;
    if (t === 1 || t === 2)
      return this.fd_write(t, i, n, l);
    const d = new DataView(this.memory.buffer), V = g(d, i, n);
    let a = 0, Z = c.SUCCESS;
    for (const h of V)
      if (h.byteLength !== 0) {
        if (Z = this.drive.pwrite(t, h, Number(s)), Z != c.SUCCESS)
          break;
        a += h.byteLength;
      }
    return d.setUint32(l, a, !0), Z;
  }
  /**
   * Read directory entries from a directory. When successful, the contents of
   * the output buffer consist of a sequence of directory entries. Each
   * directory entry consists of a dirent object, followed by dirent::d_namlen
   * bytes holding the name of the directory entry. This function fills the
   * output buffer as much as possible, potentially truncating the last
   * directory entry. This allows the caller to grow its read buffer size in
   * case it's too small to fit a single large directory entry, or skip the
   * oversized directory entry.
   */
  fd_readdir(t, i, n, s, l) {
    const [d, V] = this.drive.list(t);
    if (d != c.SUCCESS)
      return d;
    let a = [], Z = 0;
    for (const { name: y, type: F } of V) {
      const K = mt(y, F, Z);
      a.push(K), Z++;
    }
    a = a.slice(Number(s));
    const h = a.reduce((y, F) => y + F.byteLength, 0), m = new Uint8Array(h);
    let u = 0;
    for (const y of a)
      m.set(y, u), u += y.byteLength;
    const R = new Uint8Array(this.memory.buffer, i, n), L = m.subarray(0, n);
    return R.set(L), new DataView(this.memory.buffer).setUint32(l, L.byteLength, !0), c.SUCCESS;
  }
  /**
   * Atomically replace a file descriptor by renumbering another file
   * descriptor. Due to the strong focus on thread safety, this environment does
   * not provide a mechanism to duplicate or renumber a file descriptor to an
   * arbitrary number, like dup2(). This would be prone to race conditions, as
   * an actual file descriptor with the same number could be allocated by a
   * different thread at the same time. This function provides a way to
   * atomically renumber file descriptors, which would disappear if dup2() were
   * to be removed entirely.
   */
  fd_renumber(t, i) {
    return this.drive.renumber(t, i);
  }
  /**
   * Move the offset of a file descriptor.
   *
   * The offset is specified as a bigint here
   * Note: This is similar to lseek in POSIX.
   *
   * The offset, and return type are FileSize (u64) which is represented by
   * bigint in JavaScript.
   */
  fd_seek(t, i, n, s) {
    const [l, d] = this.drive.seek(t, i, n);
    return l !== c.SUCCESS || (f({ newOffset: d.toString() }), new DataView(this.memory.buffer).setBigUint64(s, d, !0)), l;
  }
  unstable_fd_seek(t, i, n, s) {
    const l = ot[n];
    return this.fd_seek(t, i, l, s);
  }
  /**
   * Synchronize the data and metadata of a file to disk.
   * Note: This is similar to fsync in POSIX.
   */
  fd_sync(t) {
    return this.drive.sync(t);
  }
  /**
   * Return the current offset of a file descriptor.
   * Note: This is similar to lseek(fd, 0, SEEK_CUR) in POSIX.
   *
   * The return type is FileSize (u64) which is represented by bigint in JS.
   *
   */
  fd_tell(t, i) {
    const [n, s] = this.drive.tell(t);
    return n !== c.SUCCESS || new DataView(this.memory.buffer).setBigUint64(i, s, !0), n;
  }
  //
  // Paths
  //
  path_filestat_get(t, i, n, s, l) {
    return this.shared_path_filestat_get(t, i, n, s, l, "preview1");
  }
  unstable_path_filestat_get(t, i, n, s, l) {
    return this.shared_path_filestat_get(t, i, n, s, l, "unstable");
  }
  /**
   * Return the attributes of a file or directory.
   * Note: This is similar to stat in POSIX.
   */
  shared_path_filestat_get(t, i, n, s, l, d) {
    const V = d === "unstable" ? A : _, a = new TextDecoder().decode(new Uint8Array(this.memory.buffer, n, s));
    f({ path: a });
    const [Z, h] = this.drive.pathStat(t, a);
    if (Z != c.SUCCESS)
      return Z;
    const m = V(h);
    return new Uint8Array(this.memory.buffer, l, m.byteLength).set(m), Z;
  }
  /**
   * Adjust the timestamps of a file or directory.
   * Note: This is similar to utimensat in POSIX.
   */
  path_filestat_set_times(t, i, n, s, l, d, V) {
    let a = null;
    V & U.ATIM && (a = k(l)), V & U.ATIM_NOW && (a = /* @__PURE__ */ new Date());
    let Z = null;
    V & U.MTIM && (Z = k(d)), V & U.MTIM_NOW && (Z = /* @__PURE__ */ new Date());
    const h = new TextDecoder().decode(new Uint8Array(this.memory.buffer, n, s));
    if (a) {
      const m = this.drive.pathSetAccessTime(t, h, a);
      if (m != c.SUCCESS)
        return m;
    }
    if (Z) {
      const m = this.drive.pathSetModificationTime(t, h, Z);
      if (m != c.SUCCESS)
        return m;
    }
    return c.SUCCESS;
  }
  /**
   * Open a file or directory. The returned file descriptor is not guaranteed to
   * be the lowest-numbered file descriptor not currently open; it is randomized
   * to prevent applications from depending on making assumptions about indexes,
   * since this is error-prone in multi-threaded contexts. The returned file
   * descriptor is guaranteed to be less than 2**31.
   * Note: This is similar to openat in POSIX.
   * @param fd: fd
   * @param dirflags: lookupflags Flags determining the method of how the path
   *                  is resolved. Not supported by Runno (symlinks)
   * @param path: string The relative path of the file or directory to open,
   *              relative to the path_open::fd directory.
   * @param oflags: oflags The method by which to open the file.
   * @param fs_rights_base: rights The initial rights of the newly created file
   *                        descriptor. The implementation is allowed to return
   *                        a file descriptor with fewer rights than specified,
   *                        if and only if those rights do not apply to the type
   *                        of file being opened. The base rights are rights
   *                        that will apply to operations using the file
   *                        descriptor itself, while the inheriting rights are
   *                        rights that apply to file descriptors derived from
   *                        it.
   * @param fs_rights_inheriting: rights
   * @param fdflags: fdflags
   *
   */
  path_open(t, i, n, s, l, d, V, a, Z) {
    const h = new DataView(this.memory.buffer), m = C(this.memory, n, s), u = !!(l & T.CREAT), R = !!(l & T.DIRECTORY), L = !!(l & T.EXCL), w = !!(l & T.TRUNC), y = !!(a & p.APPEND), F = !!(a & p.DSYNC), K = !!(a & p.NONBLOCK), et = !!(a & p.RSYNC), it = !!(a & p.SYNC);
    f({
      path: m,
      openFlags: {
        createFileIfNone: u,
        failIfNotDir: R,
        failIfFileExists: L,
        truncateFile: w
      },
      fileDescriptorFlags: {
        flagAppend: y,
        flagDSync: F,
        flagNonBlock: K,
        flagRSync: et,
        flagSync: it
      }
    });
    const [J, nt] = this.drive.open(t, m, l, a);
    return J || (h.setUint32(Z, nt, !0), J);
  }
  /**
   * Rename a file or directory. Note: This is similar to renameat in POSIX.
   */
  path_rename(t, i, n, s, l, d) {
    const V = C(this.memory, i, n), a = C(this.memory, l, d);
    return f({ oldPath: V, newPath: a }), this.drive.rename(t, V, s, a);
  }
  /**
   * Unlink a file. Return errno::isdir if the path refers to a directory.
   * Note: This is similar to unlinkat(fd, path, 0) in POSIX.
   */
  path_unlink_file(t, i, n) {
    const s = C(this.memory, i, n);
    return f({ path: s }), this.drive.unlink(t, s);
  }
  /**
   * Concurrently poll for the occurrence of a set of events.
   */
  poll_oneoff(t, i, n, s) {
    for (let d = 0; d < n; d++) {
      const V = new Uint8Array(this.memory.buffer, t + d * x, x), a = ht(V), Z = new Uint8Array(this.memory.buffer, i + d * B, B);
      let h = 0, m = c.SUCCESS;
      switch (a.type) {
        case G.CLOCK:
          for (; /* @__PURE__ */ new Date() < a.timeout; )
            ;
          Z.set(rt(a.userdata, c.SUCCESS));
          break;
        case G.FD_READ:
          if (a.fd < 3)
            a.fd === 0 ? (m = c.SUCCESS, h = 32) : m = c.EBADF;
          else {
            const [u, R] = this.drive.stat(a.fd);
            m = u, h = R ? R.byteLength : 0;
          }
          Z.set(v(a.userdata, m, G.FD_READ, BigInt(h)));
          break;
        case G.FD_WRITE:
          if (h = 0, m = c.SUCCESS, a.fd < 3)
            a.fd === 0 ? m = c.EBADF : (m = c.SUCCESS, h = 1024);
          else {
            const [u, R] = this.drive.stat(a.fd);
            m = u, h = R ? R.byteLength : 0;
          }
          Z.set(v(a.userdata, m, G.FD_READ, BigInt(h)));
          break;
      }
    }
    return new DataView(this.memory.buffer, s, 4).setUint32(0, n, !0), c.SUCCESS;
  }
  /**
   * Create a directory. Note: This is similar to mkdirat in POSIX.
   */
  path_create_directory(t, i, n) {
    const s = C(this.memory, i, n);
    return this.drive.pathCreateDir(t, s);
  }
  //
  // Unimplemented - these operations are not supported by Runno
  //
  /**
   * Create a hard link. Note: This is similar to linkat in POSIX.
   */
  path_link() {
    return c.ENOSYS;
  }
  /**
   * Read the contents of a symbolic link.
   * Note: This is similar to readlinkat in POSIX.
   */
  path_readlink() {
    return c.ENOSYS;
  }
  /**
   * Remove a directory. Return errno::notempty if the directory is not empty.
   * Note: This is similar to unlinkat(fd, path, AT_REMOVEDIR) in POSIX.
   */
  path_remove_directory() {
    return c.ENOSYS;
  }
  /**
   * Create a symbolic link. Note: This is similar to symlinkat in POSIX.
   */
  path_symlink() {
    return c.ENOSYS;
  }
  /**
   * Send a signal to the process of the calling thread.
   * Note: This is similar to raise in POSIX.
   */
  proc_raise() {
    return c.ENOSYS;
  }
  /**
   * Accept a new incoming connection. Note: This is similar to accept in POSIX.
   */
  sock_accept() {
    return c.ENOSYS;
  }
  /**
   * Receive a message from a socket. Note: This is similar to recv in POSIX,
   * though it also supports reading the data into multiple buffers in the
   * manner of readv.
   */
  sock_recv() {
    return c.ENOSYS;
  }
  /**
   * Send a message on a socket. Note: This is similar to send in POSIX, though
   * it also supports writing the data from multiple buffers in the manner of
   * writev.
   */
  sock_send() {
    return c.ENOSYS;
  }
  /**
   * Shut down socket send and receive channels. Note: This is similar to
   * shutdown in POSIX.
   */
  sock_shutdown() {
    return c.ENOSYS;
  }
  //
  // Unimplemented - these are for compatibility with Wasmedge
  //
  sock_open() {
    return c.ENOSYS;
  }
  sock_listen() {
    return c.ENOSYS;
  }
  sock_connect() {
    return c.ENOSYS;
  }
  sock_setsockopt() {
    return c.ENOSYS;
  }
  sock_bind() {
    return c.ENOSYS;
  }
  sock_getlocaladdr() {
    return c.ENOSYS;
  }
  sock_getpeeraddr() {
    return c.ENOSYS;
  }
  sock_getaddrinfo() {
    return c.ENOSYS;
  }
}
const D = o.FD_DATASYNC | o.FD_READ | o.FD_SEEK | o.FD_FDSTAT_SET_FLAGS | o.FD_SYNC | o.FD_TELL | o.FD_WRITE | o.FD_ADVISE | o.FD_ALLOCATE | o.PATH_CREATE_DIRECTORY | o.PATH_CREATE_FILE | o.PATH_LINK_SOURCE | o.PATH_LINK_TARGET | o.PATH_OPEN | o.FD_READDIR | o.PATH_READLINK | o.PATH_RENAME_SOURCE | o.PATH_RENAME_TARGET | o.PATH_FILESTAT_GET | o.PATH_FILESTAT_SET_SIZE | o.PATH_FILESTAT_SET_TIMES | o.FD_FILESTAT_GET | o.FD_FILESTAT_SET_SIZE | o.FD_FILESTAT_SET_TIMES | o.PATH_SYMLINK | o.PATH_REMOVE_DIRECTORY | o.PATH_UNLINK_FILE | o.POLL_FD_READWRITE | o.SOCK_SHUTDOWN | o.SOCK_ACCEPT;
class H extends Error {
  constructor(i) {
    super();
    r(this, "code");
    this.code = i;
  }
}
function C(e, t, i) {
  return new TextDecoder().decode(new Uint8Array(e.buffer, t, i));
}
function g(e, t, i) {
  let n = Array(i);
  for (let s = 0; s < i; s++) {
    const l = e.getUint32(t, !0);
    t += 4;
    const d = e.getUint32(t, !0);
    t += 4, n[s] = new Uint8Array(e.buffer, l, d);
  }
  return n;
}
function ht(e) {
  const t = new Uint8Array(8);
  t.set(e.subarray(0, 8));
  const i = e[8], n = new DataView(e.buffer, e.byteOffset + 9);
  switch (i) {
    case G.FD_READ:
    case G.FD_WRITE:
      return {
        userdata: t,
        type: i,
        fd: n.getUint32(0, !0)
      };
    case G.CLOCK:
      const s = n.getUint16(24, !0), l = W(/* @__PURE__ */ new Date()), d = n.getBigUint64(8, !0), V = n.getBigUint64(16, !0), a = s & $.SUBSCRIPTION_CLOCK_ABSTIME ? d : l + d;
      return {
        userdata: t,
        type: i,
        id: n.getUint32(0, !0),
        timeout: k(a),
        precision: k(a + V)
      };
  }
}
function _(e) {
  const t = new Uint8Array(O), i = new DataView(t.buffer);
  return i.setBigUint64(0, BigInt(0), !0), i.setBigUint64(8, BigInt(P(e.path)), !0), i.setUint8(16, e.type), i.setBigUint64(24, BigInt(1), !0), i.setBigUint64(32, BigInt(e.byteLength), !0), i.setBigUint64(40, W(e.timestamps.access), !0), i.setBigUint64(48, W(e.timestamps.modification), !0), i.setBigUint64(56, W(e.timestamps.change), !0), t;
}
function A(e) {
  const t = new Uint8Array(O), i = new DataView(t.buffer);
  return i.setBigUint64(0, BigInt(0), !0), i.setBigUint64(8, BigInt(P(e.path)), !0), i.setUint8(16, e.type), i.setUint32(20, 1, !0), i.setBigUint64(24, BigInt(e.byteLength), !0), i.setBigUint64(32, W(e.timestamps.access), !0), i.setBigUint64(40, W(e.timestamps.modification), !0), i.setBigUint64(48, W(e.timestamps.change), !0), t;
}
function M(e, t, i) {
  const n = i ?? D, s = i ?? D, l = new Uint8Array(24), d = new DataView(l.buffer, 0, 24);
  return d.setUint8(0, e), d.setUint32(2, t, !0), d.setBigUint64(8, n, !0), d.setBigUint64(16, s, !0), l;
}
function mt(e, t, i) {
  const n = new TextEncoder().encode(e), s = 24 + n.byteLength, l = new Uint8Array(s), d = new DataView(l.buffer);
  return d.setBigUint64(0, BigInt(i + 1), !0), d.setBigUint64(8, BigInt(P(e)), !0), d.setUint32(16, n.length, !0), d.setUint8(20, t), l.set(n, 24), l;
}
function rt(e, t) {
  const i = new Uint8Array(32);
  i.set(e, 0);
  const n = new DataView(i.buffer);
  return n.setUint16(8, t, !0), n.setUint16(10, G.CLOCK, !0), i;
}
function v(e, t, i, n) {
  const s = new Uint8Array(32);
  s.set(e, 0);
  const l = new DataView(s.buffer);
  return l.setUint16(8, t, !0), l.setUint16(10, i, !0), l.setBigUint64(16, n, !0), s;
}
function P(e, t = 0) {
  let i = 3735928559 ^ t, n = 1103547991 ^ t;
  for (let s = 0, l; s < e.length; s++)
    l = e.charCodeAt(s), i = Math.imul(i ^ l, 2654435761), n = Math.imul(n ^ l, 1597334677);
  return i = Math.imul(i ^ i >>> 16, 2246822507) ^ Math.imul(n ^ n >>> 13, 3266489909), n = Math.imul(n ^ n >>> 16, 2246822507) ^ Math.imul(i ^ i >>> 13, 3266489909), 4294967296 * (2097151 & n) + (i >>> 0);
}
function W(e) {
  return BigInt(e.getTime()) * BigInt(1e6);
}
function k(e) {
  return new Date(Number(e / BigInt(1e6)));
}
const ot = {
  [I.CUR]: N.CUR,
  [I.END]: N.END,
  [I.SET]: N.SET
}, tt = "dmFyIHV0PU9iamVjdC5kZWZpbmVQcm9wZXJ0eTt2YXIgZ3Q9KHMsVCxEKT0+VCBpbiBzP3V0KHMsVCx7ZW51bWVyYWJsZTohMCxjb25maWd1cmFibGU6ITAsd3JpdGFibGU6ITAsdmFsdWU6RH0pOnNbVF09RDt2YXIgZD0ocyxULEQpPT4oZ3Qocyx0eXBlb2YgVCE9InN5bWJvbCI/VCsiIjpULEQpLEQpOyhmdW5jdGlvbigpeyJ1c2Ugc3RyaWN0Ijt2YXIgcz0oZT0+KGVbZS5TVUNDRVNTPTBdPSJTVUNDRVNTIixlW2UuRTJCSUc9MV09IkUyQklHIixlW2UuRUFDQ0VTUz0yXT0iRUFDQ0VTUyIsZVtlLkVBRERSSU5VU0U9M109IkVBRERSSU5VU0UiLGVbZS5FQUREUk5PVEFWQUlMPTRdPSJFQUREUk5PVEFWQUlMIixlW2UuRUFGTk9TVVBQT1JUPTVdPSJFQUZOT1NVUFBPUlQiLGVbZS5FQUdBSU49Nl09IkVBR0FJTiIsZVtlLkVBTFJFQURZPTddPSJFQUxSRUFEWSIsZVtlLkVCQURGPThdPSJFQkFERiIsZVtlLkVCQURNU0c9OV09IkVCQURNU0ciLGVbZS5FQlVTWT0xMF09IkVCVVNZIixlW2UuRUNBTkNFTEVEPTExXT0iRUNBTkNFTEVEIixlW2UuRUNISUxEPTEyXT0iRUNISUxEIixlW2UuRUNPTk5BQk9SVEVEPTEzXT0iRUNPTk5BQk9SVEVEIixlW2UuRUNPTk5SRUZVU0VEPTE0XT0iRUNPTk5SRUZVU0VEIixlW2UuRUNPTk5SRVNFVD0xNV09IkVDT05OUkVTRVQiLGVbZS5FREVBRExLPTE2XT0iRURFQURMSyIsZVtlLkVERVNUQUREUlJFUT0xN109IkVERVNUQUREUlJFUSIsZVtlLkVET009MThdPSJFRE9NIixlW2UuRURRVU9UPTE5XT0iRURRVU9UIixlW2UuRUVYSVNUPTIwXT0iRUVYSVNUIixlW2UuRUZBVUxUPTIxXT0iRUZBVUxUIixlW2UuRUZCSUc9MjJdPSJFRkJJRyIsZVtlLkVIT1NUVU5SRUFDSD0yM109IkVIT1NUVU5SRUFDSCIsZVtlLkVJRFJNPTI0XT0iRUlEUk0iLGVbZS5FSUxTRVE9MjVdPSJFSUxTRVEiLGVbZS5FSU5QUk9HUkVTUz0yNl09IkVJTlBST0dSRVNTIixlW2UuRUlOVFI9MjddPSJFSU5UUiIsZVtlLkVJTlZBTD0yOF09IkVJTlZBTCIsZVtlLkVJTz0yOV09IkVJTyIsZVtlLkVJU0NPTk49MzBdPSJFSVNDT05OIixlW2UuRUlTRElSPTMxXT0iRUlTRElSIixlW2UuRUxPT1A9MzJdPSJFTE9PUCIsZVtlLkVNRklMRT0zM109IkVNRklMRSIsZVtlLkVNTElOSz0zNF09IkVNTElOSyIsZVtlLkVNU0dTSVpFPTM1XT0iRU1TR1NJWkUiLGVbZS5FTVVMVElIT1A9MzZdPSJFTVVMVElIT1AiLGVbZS5FTkFNRVRPT0xPTkc9MzddPSJFTkFNRVRPT0xPTkciLGVbZS5FTkVURE9XTj0zOF09IkVORVRET1dOIixlW2UuRU5FVFJFU0VUPTM5XT0iRU5FVFJFU0VUIixlW2UuRU5FVFVOUkVBQ0g9NDBdPSJFTkVUVU5SRUFDSCIsZVtlLkVORklMRT00MV09IkVORklMRSIsZVtlLkVOT0JVRlM9NDJdPSJFTk9CVUZTIixlW2UuRU5PREVWPTQzXT0iRU5PREVWIixlW2UuRU5PRU5UPTQ0XT0iRU5PRU5UIixlW2UuRU5PRVhFQz00NV09IkVOT0VYRUMiLGVbZS5FTk9MQ0s9NDZdPSJFTk9MQ0siLGVbZS5FTk9MSU5LPTQ3XT0iRU5PTElOSyIsZVtlLkVOT01FTT00OF09IkVOT01FTSIsZVtlLkVOT01TRz00OV09IkVOT01TRyIsZVtlLkVOT1BST1RPT1BUPTUwXT0iRU5PUFJPVE9PUFQiLGVbZS5FTk9TUEM9NTFdPSJFTk9TUEMiLGVbZS5FTk9TWVM9NTJdPSJFTk9TWVMiLGVbZS5FTk9UQ09OTj01M109IkVOT1RDT05OIixlW2UuRU5PVERJUj01NF09IkVOT1RESVIiLGVbZS5FTk9URU1QVFk9NTVdPSJFTk9URU1QVFkiLGVbZS5FTk9UUkVDT1ZFUkFCTEU9NTZdPSJFTk9UUkVDT1ZFUkFCTEUiLGVbZS5FTk9UU09DSz01N109IkVOT1RTT0NLIixlW2UuRU5PVFNVUD01OF09IkVOT1RTVVAiLGVbZS5FTk9UVFk9NTldPSJFTk9UVFkiLGVbZS5FTlhJTz02MF09IkVOWElPIixlW2UuRU9WRVJGTE9XPTYxXT0iRU9WRVJGTE9XIixlW2UuRU9XTkVSREVBRD02Ml09IkVPV05FUkRFQUQiLGVbZS5FUEVSTT02M109IkVQRVJNIixlW2UuRVBJUEU9NjRdPSJFUElQRSIsZVtlLkVQUk9UTz02NV09IkVQUk9UTyIsZVtlLkVQUk9UT05PU1VQUE9SVD02Nl09IkVQUk9UT05PU1VQUE9SVCIsZVtlLkVQUk9UT1RZUEU9NjddPSJFUFJPVE9UWVBFIixlW2UuRVJBTkdFPTY4XT0iRVJBTkdFIixlW2UuRVJPRlM9NjldPSJFUk9GUyIsZVtlLkVTUElQRT03MF09IkVTUElQRSIsZVtlLkVTUkNIPTcxXT0iRVNSQ0giLGVbZS5FU1RBTEU9NzJdPSJFU1RBTEUiLGVbZS5FVElNRURPVVQ9NzNdPSJFVElNRURPVVQiLGVbZS5FVFhUQlNZPTc0XT0iRVRYVEJTWSIsZVtlLkVYREVWPTc1XT0iRVhERVYiLGVbZS5FTk9UQ0FQQUJMRT03Nl09IkVOT1RDQVBBQkxFIixlKSkoc3x8e30pLFQ9KGU9PihlW2UuUkVBTFRJTUU9MF09IlJFQUxUSU1FIixlW2UuTU9OT1RPTklDPTFdPSJNT05PVE9OSUMiLGVbZS5QUk9DRVNTX0NQVVRJTUVfSUQ9Ml09IlBST0NFU1NfQ1BVVElNRV9JRCIsZVtlLlRIUkVBRF9DUFVUSU1FX0lEPTNdPSJUSFJFQURfQ1BVVElNRV9JRCIsZSkpKFR8fHt9KSxEPShlPT4oZVtlLlNFVD0wXT0iU0VUIixlW2UuQ1VSPTFdPSJDVVIiLGVbZS5FTkQ9Ml09IkVORCIsZSkpKER8fHt9KSxiPShlPT4oZVtlLlVOS05PV049MF09IlVOS05PV04iLGVbZS5CTE9DS19ERVZJQ0U9MV09IkJMT0NLX0RFVklDRSIsZVtlLkNIQVJBQ1RFUl9ERVZJQ0U9Ml09IkNIQVJBQ1RFUl9ERVZJQ0UiLGVbZS5ESVJFQ1RPUlk9M109IkRJUkVDVE9SWSIsZVtlLlJFR1VMQVJfRklMRT00XT0iUkVHVUxBUl9GSUxFIixlW2UuU09DS0VUX0RHUkFNPTVdPSJTT0NLRVRfREdSQU0iLGVbZS5TT0NLRVRfU1RSRUFNPTZdPSJTT0NLRVRfU1RSRUFNIixlW2UuU1lNQk9MSUNfTElOSz03XT0iU1lNQk9MSUNfTElOSyIsZSkpKGJ8fHt9KSxHPShlPT4oZVtlLkRJUj0wXT0iRElSIixlKSkoR3x8e30pLEE9KGU9PihlW2UuQ0xPQ0s9MF09IkNMT0NLIixlW2UuRkRfUkVBRD0xXT0iRkRfUkVBRCIsZVtlLkZEX1dSSVRFPTJdPSJGRF9XUklURSIsZSkpKEF8fHt9KTtjb25zdCBOPXtDUkVBVDoxLERJUkVDVE9SWToyLEVYQ0w6NCxUUlVOQzo4fSxPPXtBUFBFTkQ6MSxEU1lOQzoyLE5PTkJMT0NLOjQsUlNZTkM6OCxTWU5DOjE2fSxfPXtGRF9EQVRBU1lOQzpCaWdJbnQoMSk8PEJpZ0ludCgwKSxGRF9SRUFEOkJpZ0ludCgxKTw8QmlnSW50KDEpLEZEX1NFRUs6QmlnSW50KDEpPDxCaWdJbnQoMiksRkRfRkRTVEFUX1NFVF9GTEFHUzpCaWdJbnQoMSk8PEJpZ0ludCgzKSxGRF9TWU5DOkJpZ0ludCgxKTw8QmlnSW50KDQpLEZEX1RFTEw6QmlnSW50KDEpPDxCaWdJbnQoNSksRkRfV1JJVEU6QmlnSW50KDEpPDxCaWdJbnQoNiksRkRfQURWSVNFOkJpZ0ludCgxKTw8QmlnSW50KDcpLEZEX0FMTE9DQVRFOkJpZ0ludCgxKTw8QmlnSW50KDgpLFBBVEhfQ1JFQVRFX0RJUkVDVE9SWTpCaWdJbnQoMSk8PEJpZ0ludCg5KSxQQVRIX0NSRUFURV9GSUxFOkJpZ0ludCgxKTw8QmlnSW50KDEwKSxQQVRIX0xJTktfU09VUkNFOkJpZ0ludCgxKTw8QmlnSW50KDExKSxQQVRIX0xJTktfVEFSR0VUOkJpZ0ludCgxKTw8QmlnSW50KDEyKSxQQVRIX09QRU46QmlnSW50KDEpPDxCaWdJbnQoMTMpLEZEX1JFQURESVI6QmlnSW50KDEpPDxCaWdJbnQoMTQpLFBBVEhfUkVBRExJTks6QmlnSW50KDEpPDxCaWdJbnQoMTUpLFBBVEhfUkVOQU1FX1NPVVJDRTpCaWdJbnQoMSk8PEJpZ0ludCgxNiksUEFUSF9SRU5BTUVfVEFSR0VUOkJpZ0ludCgxKTw8QmlnSW50KDE3KSxQQVRIX0ZJTEVTVEFUX0dFVDpCaWdJbnQoMSk8PEJpZ0ludCgxOCksUEFUSF9GSUxFU1RBVF9TRVRfU0laRTpCaWdJbnQoMSk8PEJpZ0ludCgxOSksUEFUSF9GSUxFU1RBVF9TRVRfVElNRVM6QmlnSW50KDEpPDxCaWdJbnQoMjApLEZEX0ZJTEVTVEFUX0dFVDpCaWdJbnQoMSk8PEJpZ0ludCgyMSksRkRfRklMRVNUQVRfU0VUX1NJWkU6QmlnSW50KDEpPDxCaWdJbnQoMjIpLEZEX0ZJTEVTVEFUX1NFVF9USU1FUzpCaWdJbnQoMSk8PEJpZ0ludCgyMyksUEFUSF9TWU1MSU5LOkJpZ0ludCgxKTw8QmlnSW50KDI0KSxQQVRIX1JFTU9WRV9ESVJFQ1RPUlk6QmlnSW50KDEpPDxCaWdJbnQoMjUpLFBBVEhfVU5MSU5LX0ZJTEU6QmlnSW50KDEpPDxCaWdJbnQoMjYpLFBPTExfRkRfUkVBRFdSSVRFOkJpZ0ludCgxKTw8QmlnSW50KDI3KSxTT0NLX1NIVVRET1dOOkJpZ0ludCgxKTw8QmlnSW50KDI4KSxTT0NLX0FDQ0VQVDpCaWdJbnQoMSk8PEJpZ0ludCgyOSl9LG09e0FUSU06MSxBVElNX05PVzoyLE1USU06NCxNVElNX05PVzo4fSx0dD17U1VCU0NSSVBUSU9OX0NMT0NLX0FCU1RJTUU6MX0sVz02NCx6PTQ4LCQ9MzI7dmFyIE09KGU9PihlW2UuQ1VSPTBdPSJDVVIiLGVbZS5FTkQ9MV09IkVORCIsZVtlLlNFVD0yXT0iU0VUIixlKSkoTXx8e30pO2NsYXNzIFh7Y29uc3RydWN0b3IodCl7ZCh0aGlzLCJmcyIpO2QodGhpcywiYXJncyIpO2QodGhpcywiZW52Iik7ZCh0aGlzLCJzdGRpbiIpO2QodGhpcywic3Rkb3V0Iik7ZCh0aGlzLCJzdGRlcnIiKTtkKHRoaXMsImRlYnVnIik7ZCh0aGlzLCJpc1RUWSIpO3RoaXMuZnM9KHQ9PW51bGw/dm9pZCAwOnQuZnMpPz97fSx0aGlzLmFyZ3M9KHQ9PW51bGw/dm9pZCAwOnQuYXJncyk/P1tdLHRoaXMuZW52PSh0PT1udWxsP3ZvaWQgMDp0LmVudik/P3t9LHRoaXMuc3RkaW49KHQ9PW51bGw/dm9pZCAwOnQuc3RkaW4pPz8oKCk9Pm51bGwpLHRoaXMuc3Rkb3V0PSh0PT1udWxsP3ZvaWQgMDp0LnN0ZG91dCk/PygoKT0+e30pLHRoaXMuc3RkZXJyPSh0PT1udWxsP3ZvaWQgMDp0LnN0ZGVycik/PygoKT0+e30pLHRoaXMuZGVidWc9dD09bnVsbD92b2lkIDA6dC5kZWJ1Zyx0aGlzLmlzVFRZPSEhKHQhPW51bGwmJnQuaXNUVFkpfX1jbGFzcyBldHtjb25zdHJ1Y3Rvcih0KXtkKHRoaXMsImZzIik7ZCh0aGlzLCJuZXh0RkQiLDEwKTtkKHRoaXMsIm9wZW5NYXAiLG5ldyBNYXApO3RoaXMuZnM9ey4uLnR9LHRoaXMub3Blbk1hcC5zZXQoMyxuZXcgdSh0aGlzLmZzLCIvIikpfW9wZW5GaWxlKHQsbixpKXtjb25zdCByPW5ldyBJKHQsaSk7biYmKHIuYnVmZmVyPW5ldyBVaW50OEFycmF5KG5ldyBBcnJheUJ1ZmZlcigxMDI0KSwwLDApKTtjb25zdCBhPXRoaXMubmV4dEZEO3JldHVybiB0aGlzLm9wZW5NYXAuc2V0KGEsciksdGhpcy5uZXh0RkQrKyxbcy5TVUNDRVNTLGFdfW9wZW5EaXIodCxuKXtjb25zdCBpPW5ldyB1KHQsbikscj10aGlzLm5leHRGRDtyZXR1cm4gdGhpcy5vcGVuTWFwLnNldChyLGkpLHRoaXMubmV4dEZEKyssW3MuU1VDQ0VTUyxyXX1oYXNEaXIodCxuKXtyZXR1cm4gbj09PSIuIj8hMDp0LmNvbnRhaW5zRGlyZWN0b3J5KG4pfW9wZW4odCxuLGkscil7Y29uc3QgYT0hIShpJk4uQ1JFQVQpLGY9ISEoaSZOLkRJUkVDVE9SWSksYz0hIShpJk4uRVhDTCksbz0hIShpJk4uVFJVTkMpLEU9dGhpcy5vcGVuTWFwLmdldCh0KTtpZighKEUgaW5zdGFuY2VvZiB1KSlyZXR1cm5bcy5FQkFERl07aWYoRS5jb250YWluc0ZpbGUobikpcmV0dXJuIGY/W3MuRU5PVERJUl06Yz9bcy5FRVhJU1RdOnRoaXMub3BlbkZpbGUoRS5nZXQobiksbyxyKTtpZih0aGlzLmhhc0RpcihFLG4pKXtpZihuPT09Ii4iKXJldHVybiB0aGlzLm9wZW5EaXIodGhpcy5mcywiLyIpO2NvbnN0IGg9YC8ke259L2AsUz1PYmplY3QuZW50cmllcyh0aGlzLmZzKS5maWx0ZXIoKFtnXSk9Pmcuc3RhcnRzV2l0aChoKSk7cmV0dXJuIHRoaXMub3BlbkRpcihPYmplY3QuZnJvbUVudHJpZXMoUyksaCl9ZWxzZXtpZihhKXtjb25zdCBoPUUuZnVsbFBhdGgobik7cmV0dXJuIHRoaXMuZnNbaF09e3BhdGg6aCxtb2RlOiJiaW5hcnkiLGNvbnRlbnQ6bmV3IFVpbnQ4QXJyYXksdGltZXN0YW1wczp7YWNjZXNzOm5ldyBEYXRlLG1vZGlmaWNhdGlvbjpuZXcgRGF0ZSxjaGFuZ2U6bmV3IERhdGV9fSx0aGlzLm9wZW5GaWxlKHRoaXMuZnNbaF0sbyxyKX1yZXR1cm5bcy5FTk9UQ0FQQUJMRV19fWNsb3NlKHQpe2lmKCF0aGlzLm9wZW5NYXAuaGFzKHQpKXJldHVybiBzLkVCQURGO2NvbnN0IG49dGhpcy5vcGVuTWFwLmdldCh0KTtyZXR1cm4gbiBpbnN0YW5jZW9mIEkmJm4uc3luYygpLHRoaXMub3Blbk1hcC5kZWxldGUodCkscy5TVUNDRVNTfXJlYWQodCxuKXtjb25zdCBpPXRoaXMub3Blbk1hcC5nZXQodCk7cmV0dXJuIWl8fGkgaW5zdGFuY2VvZiB1P1tzLkVCQURGXTpbcy5TVUNDRVNTLGkucmVhZChuKV19cHJlYWQodCxuLGkpe2NvbnN0IHI9dGhpcy5vcGVuTWFwLmdldCh0KTtyZXR1cm4hcnx8ciBpbnN0YW5jZW9mIHU/W3MuRUJBREZdOltzLlNVQ0NFU1Msci5wcmVhZChuLGkpXX13cml0ZSh0LG4pe2NvbnN0IGk9dGhpcy5vcGVuTWFwLmdldCh0KTtyZXR1cm4haXx8aSBpbnN0YW5jZW9mIHU/cy5FQkFERjooaS53cml0ZShuKSxzLlNVQ0NFU1MpfXB3cml0ZSh0LG4saSl7Y29uc3Qgcj10aGlzLm9wZW5NYXAuZ2V0KHQpO3JldHVybiFyfHxyIGluc3RhbmNlb2YgdT9zLkVCQURGOihyLnB3cml0ZShuLGkpLHMuU1VDQ0VTUyl9c3luYyh0KXtjb25zdCBuPXRoaXMub3Blbk1hcC5nZXQodCk7cmV0dXJuIW58fG4gaW5zdGFuY2VvZiB1P3MuRUJBREY6KG4uc3luYygpLHMuU1VDQ0VTUyl9c2Vlayh0LG4saSl7Y29uc3Qgcj10aGlzLm9wZW5NYXAuZ2V0KHQpO3JldHVybiFyfHxyIGluc3RhbmNlb2YgdT9bcy5FQkFERl06W3MuU1VDQ0VTUyxyLnNlZWsobixpKV19dGVsbCh0KXtjb25zdCBuPXRoaXMub3Blbk1hcC5nZXQodCk7cmV0dXJuIW58fG4gaW5zdGFuY2VvZiB1P1tzLkVCQURGXTpbcy5TVUNDRVNTLG4udGVsbCgpXX1yZW51bWJlcih0LG4pe3JldHVybiF0aGlzLmV4aXN0cyh0KXx8IXRoaXMuZXhpc3RzKG4pP3MuRUJBREY6KHQ9PT1ufHwodGhpcy5jbG9zZShuKSx0aGlzLm9wZW5NYXAuc2V0KG4sdGhpcy5vcGVuTWFwLmdldCh0KSkpLHMuU1VDQ0VTUyl9dW5saW5rKHQsbil7Y29uc3QgaT10aGlzLm9wZW5NYXAuZ2V0KHQpO2lmKCEoaSBpbnN0YW5jZW9mIHUpKXJldHVybiBzLkVCQURGO2lmKCFpLmNvbnRhaW5zKG4pKXJldHVybiBzLkVOT0VOVDtmb3IoY29uc3QgciBvZiBPYmplY3Qua2V5cyh0aGlzLmZzKSkocj09PWkuZnVsbFBhdGgobil8fHIuc3RhcnRzV2l0aChgJHtpLmZ1bGxQYXRoKG4pfS9gKSkmJmRlbGV0ZSB0aGlzLmZzW3JdO3JldHVybiBzLlNVQ0NFU1N9cmVuYW1lKHQsbixpLHIpe2NvbnN0IGE9dGhpcy5vcGVuTWFwLmdldCh0KSxmPXRoaXMub3Blbk1hcC5nZXQoaSk7aWYoIShhIGluc3RhbmNlb2YgdSl8fCEoZiBpbnN0YW5jZW9mIHUpKXJldHVybiBzLkVCQURGO2lmKCFhLmNvbnRhaW5zKG4pKXJldHVybiBzLkVOT0VOVDtpZihmLmNvbnRhaW5zKHIpKXJldHVybiBzLkVFWElTVDtjb25zdCBjPWEuZnVsbFBhdGgobiksbz1mLmZ1bGxQYXRoKHIpO2Zvcihjb25zdCBFIG9mIE9iamVjdC5rZXlzKHRoaXMuZnMpKWlmKEUuc3RhcnRzV2l0aChjKSl7Y29uc3QgaD1FLnJlcGxhY2UoYyxvKTt0aGlzLmZzW2hdPXRoaXMuZnNbRV0sdGhpcy5mc1toXS5wYXRoPWgsZGVsZXRlIHRoaXMuZnNbRV19cmV0dXJuIHMuU1VDQ0VTU31saXN0KHQpe2NvbnN0IG49dGhpcy5vcGVuTWFwLmdldCh0KTtyZXR1cm4gbiBpbnN0YW5jZW9mIHU/W3MuU1VDQ0VTUyxuLmxpc3QoKV06W3MuRUJBREZdfXN0YXQodCl7Y29uc3Qgbj10aGlzLm9wZW5NYXAuZ2V0KHQpO3JldHVybiBuIGluc3RhbmNlb2YgST9bcy5TVUNDRVNTLG4uc3RhdCgpXTpbcy5FQkFERl19cGF0aFN0YXQodCxuKXtjb25zdCBpPXRoaXMub3Blbk1hcC5nZXQodCk7aWYoIShpIGluc3RhbmNlb2YgdSkpcmV0dXJuW3MuRUJBREZdO2lmKGkuY29udGFpbnNGaWxlKG4pKXtjb25zdCByPWkuZnVsbFBhdGgobiksYT1uZXcgSSh0aGlzLmZzW3JdLDApLnN0YXQoKTtyZXR1cm5bcy5TVUNDRVNTLGFdfWVsc2UgaWYodGhpcy5oYXNEaXIoaSxuKSl7aWYobj09PSIuIilyZXR1cm5bcy5TVUNDRVNTLG5ldyB1KHRoaXMuZnMsIi8iKS5zdGF0KCldO2NvbnN0IHI9YC8ke259L2AsYT1PYmplY3QuZW50cmllcyh0aGlzLmZzKS5maWx0ZXIoKFtjXSk9PmMuc3RhcnRzV2l0aChyKSksZj1uZXcgdShPYmplY3QuZnJvbUVudHJpZXMoYSkscikuc3RhdCgpO3JldHVybltzLlNVQ0NFU1MsZl19ZWxzZSByZXR1cm5bcy5FTk9UQ0FQQUJMRV19c2V0RmxhZ3ModCxuKXtjb25zdCBpPXRoaXMub3Blbk1hcC5nZXQodCk7cmV0dXJuIGkgaW5zdGFuY2VvZiBJPyhpLnNldEZsYWdzKG4pLHMuU1VDQ0VTUyk6cy5FQkFERn1zZXRTaXplKHQsbil7Y29uc3QgaT10aGlzLm9wZW5NYXAuZ2V0KHQpO3JldHVybiBpIGluc3RhbmNlb2YgST8oaS5zZXRTaXplKE51bWJlcihuKSkscy5TVUNDRVNTKTpzLkVCQURGfXNldEFjY2Vzc1RpbWUodCxuKXtjb25zdCBpPXRoaXMub3Blbk1hcC5nZXQodCk7cmV0dXJuIGkgaW5zdGFuY2VvZiBJPyhpLnNldEFjY2Vzc1RpbWUobikscy5TVUNDRVNTKTpzLkVCQURGfXNldE1vZGlmaWNhdGlvblRpbWUodCxuKXtjb25zdCBpPXRoaXMub3Blbk1hcC5nZXQodCk7cmV0dXJuIGkgaW5zdGFuY2VvZiBJPyhpLnNldE1vZGlmaWNhdGlvblRpbWUobikscy5TVUNDRVNTKTpzLkVCQURGfXBhdGhTZXRBY2Nlc3NUaW1lKHQsbixpKXtjb25zdCByPXRoaXMub3Blbk1hcC5nZXQodCk7aWYoIShyIGluc3RhbmNlb2YgdSkpcmV0dXJuIHMuRUJBREY7Y29uc3QgYT1yLmdldChuKTtpZighYSlyZXR1cm4gcy5FTk9UQ0FQQUJMRTtjb25zdCBmPW5ldyBJKGEsMCk7cmV0dXJuIGYuc2V0QWNjZXNzVGltZShpKSxmLnN5bmMoKSxzLlNVQ0NFU1N9cGF0aFNldE1vZGlmaWNhdGlvblRpbWUodCxuLGkpe2NvbnN0IHI9dGhpcy5vcGVuTWFwLmdldCh0KTtpZighKHIgaW5zdGFuY2VvZiB1KSlyZXR1cm4gcy5FQkFERjtjb25zdCBhPXIuZ2V0KG4pO2lmKCFhKXJldHVybiBzLkVOT1RDQVBBQkxFO2NvbnN0IGY9bmV3IEkoYSwwKTtyZXR1cm4gZi5zZXRNb2RpZmljYXRpb25UaW1lKGkpLGYuc3luYygpLHMuU1VDQ0VTU31wYXRoQ3JlYXRlRGlyKHQsbil7Y29uc3QgaT10aGlzLm9wZW5NYXAuZ2V0KHQpO2lmKCEoaSBpbnN0YW5jZW9mIHUpKXJldHVybiBzLkVCQURGO2lmKGkuY29udGFpbnMobikpcmV0dXJuIHMuRU5PVENBUEFCTEU7Y29uc3Qgcj1gJHtpLmZ1bGxQYXRoKG4pfS8ucnVubm9gO3JldHVybiB0aGlzLmZzW3JdPXtwYXRoOnIsdGltZXN0YW1wczp7YWNjZXNzOm5ldyBEYXRlLG1vZGlmaWNhdGlvbjpuZXcgRGF0ZSxjaGFuZ2U6bmV3IERhdGV9LG1vZGU6InN0cmluZyIsY29udGVudDoiIn0scy5TVUNDRVNTfWV4aXN0cyh0KXtyZXR1cm4gdGhpcy5vcGVuTWFwLmhhcyh0KX1maWxlVHlwZSh0KXtjb25zdCBuPXRoaXMub3Blbk1hcC5nZXQodCk7cmV0dXJuIG4/biBpbnN0YW5jZW9mIEk/Yi5SRUdVTEFSX0ZJTEU6Yi5ESVJFQ1RPUlk6Yi5VTktOT1dOfWZpbGVGZGZsYWdzKHQpe2NvbnN0IG49dGhpcy5vcGVuTWFwLmdldCh0KTtyZXR1cm4gbiBpbnN0YW5jZW9mIEk/bi5mZGZsYWdzOjB9fWNsYXNzIEl7Y29uc3RydWN0b3IodCxuKXtkKHRoaXMsImZpbGUiKTtkKHRoaXMsImJ1ZmZlciIpO2QodGhpcywiX29mZnNldCIsQmlnSW50KDApKTtkKHRoaXMsImlzRGlydHkiLCExKTtkKHRoaXMsImZkZmxhZ3MiKTtkKHRoaXMsImZsYWdBcHBlbmQiKTtkKHRoaXMsImZsYWdEU3luYyIpO2QodGhpcywiZmxhZ05vbkJsb2NrIik7ZCh0aGlzLCJmbGFnUlN5bmMiKTtkKHRoaXMsImZsYWdTeW5jIik7aWYodGhpcy5maWxlPXQsdGhpcy5maWxlLm1vZGU9PT0ic3RyaW5nIil7Y29uc3QgaT1uZXcgVGV4dEVuY29kZXI7dGhpcy5idWZmZXI9aS5lbmNvZGUodGhpcy5maWxlLmNvbnRlbnQpfWVsc2UgdGhpcy5idWZmZXI9dGhpcy5maWxlLmNvbnRlbnQ7dGhpcy5mZGZsYWdzPW4sdGhpcy5mbGFnQXBwZW5kPSEhKG4mTy5BUFBFTkQpLHRoaXMuZmxhZ0RTeW5jPSEhKG4mTy5EU1lOQyksdGhpcy5mbGFnTm9uQmxvY2s9ISEobiZPLk5PTkJMT0NLKSx0aGlzLmZsYWdSU3luYz0hIShuJk8uUlNZTkMpLHRoaXMuZmxhZ1N5bmM9ISEobiZPLlNZTkMpfWdldCBvZmZzZXQoKXtyZXR1cm4gTnVtYmVyKHRoaXMuX29mZnNldCl9cmVhZCh0KXtjb25zdCBuPXRoaXMuYnVmZmVyLnN1YmFycmF5KHRoaXMub2Zmc2V0LHRoaXMub2Zmc2V0K3QpO3JldHVybiB0aGlzLl9vZmZzZXQrPUJpZ0ludChuLmxlbmd0aCksbn1wcmVhZCh0LG4pe3JldHVybiB0aGlzLmJ1ZmZlci5zdWJhcnJheShuLG4rdCl9d3JpdGUodCl7aWYodGhpcy5pc0RpcnR5PSEwLHRoaXMuZmxhZ0FwcGVuZCl7Y29uc3Qgbj10aGlzLmJ1ZmZlci5sZW5ndGg7dGhpcy5yZXNpemUobit0LmJ5dGVMZW5ndGgpLHRoaXMuYnVmZmVyLnNldCh0LG4pfWVsc2V7Y29uc3Qgbj1NYXRoLm1heCh0aGlzLm9mZnNldCt0LmJ5dGVMZW5ndGgsdGhpcy5idWZmZXIuYnl0ZUxlbmd0aCk7dGhpcy5yZXNpemUobiksdGhpcy5idWZmZXIuc2V0KHQsdGhpcy5vZmZzZXQpLHRoaXMuX29mZnNldCs9QmlnSW50KHQuYnl0ZUxlbmd0aCl9KHRoaXMuZmxhZ0RTeW5jfHx0aGlzLmZsYWdTeW5jKSYmdGhpcy5zeW5jKCl9cHdyaXRlKHQsbil7aWYodGhpcy5pc0RpcnR5PSEwLHRoaXMuZmxhZ0FwcGVuZCl7Y29uc3QgaT10aGlzLmJ1ZmZlci5sZW5ndGg7dGhpcy5yZXNpemUoaSt0LmJ5dGVMZW5ndGgpLHRoaXMuYnVmZmVyLnNldCh0LGkpfWVsc2V7Y29uc3QgaT1NYXRoLm1heChuK3QuYnl0ZUxlbmd0aCx0aGlzLmJ1ZmZlci5ieXRlTGVuZ3RoKTt0aGlzLnJlc2l6ZShpKSx0aGlzLmJ1ZmZlci5zZXQodCxuKX0odGhpcy5mbGFnRFN5bmN8fHRoaXMuZmxhZ1N5bmMpJiZ0aGlzLnN5bmMoKX1zeW5jKCl7aWYoIXRoaXMuaXNEaXJ0eSlyZXR1cm47aWYodGhpcy5pc0RpcnR5PSExLHRoaXMuZmlsZS5tb2RlPT09ImJpbmFyeSIpe3RoaXMuZmlsZS5jb250ZW50PW5ldyBVaW50OEFycmF5KHRoaXMuYnVmZmVyKTtyZXR1cm59Y29uc3QgdD1uZXcgVGV4dERlY29kZXI7dGhpcy5maWxlLmNvbnRlbnQ9dC5kZWNvZGUodGhpcy5idWZmZXIpfXNlZWsodCxuKXtzd2l0Y2gobil7Y2FzZSBELlNFVDp0aGlzLl9vZmZzZXQ9dDticmVhaztjYXNlIEQuQ1VSOnRoaXMuX29mZnNldCs9dDticmVhaztjYXNlIEQuRU5EOnRoaXMuX29mZnNldD1CaWdJbnQodGhpcy5idWZmZXIubGVuZ3RoKSt0O2JyZWFrfXJldHVybiB0aGlzLl9vZmZzZXR9dGVsbCgpe3JldHVybiB0aGlzLl9vZmZzZXR9c3RhdCgpe3JldHVybntwYXRoOnRoaXMuZmlsZS5wYXRoLHRpbWVzdGFtcHM6dGhpcy5maWxlLnRpbWVzdGFtcHMsdHlwZTpiLlJFR1VMQVJfRklMRSxieXRlTGVuZ3RoOnRoaXMuYnVmZmVyLmxlbmd0aH19c2V0RmxhZ3ModCl7dGhpcy5mZGZsYWdzPXR9c2V0U2l6ZSh0KXt0aGlzLnJlc2l6ZSh0KX1zZXRBY2Nlc3NUaW1lKHQpe3RoaXMuZmlsZS50aW1lc3RhbXBzLmFjY2Vzcz10fXNldE1vZGlmaWNhdGlvblRpbWUodCl7dGhpcy5maWxlLnRpbWVzdGFtcHMubW9kaWZpY2F0aW9uPXR9cmVzaXplKHQpe2lmKHQ8PXRoaXMuYnVmZmVyLmJ1ZmZlci5ieXRlTGVuZ3RoKXt0aGlzLmJ1ZmZlcj1uZXcgVWludDhBcnJheSh0aGlzLmJ1ZmZlci5idWZmZXIsMCx0KTtyZXR1cm59bGV0IG47dGhpcy5idWZmZXIuYnVmZmVyLmJ5dGVMZW5ndGg9PT0wP249bmV3IEFycmF5QnVmZmVyKHQ8MTAyND8xMDI0OnQqMik6dD50aGlzLmJ1ZmZlci5idWZmZXIuYnl0ZUxlbmd0aCoyP249bmV3IEFycmF5QnVmZmVyKHQqMik6bj1uZXcgQXJyYXlCdWZmZXIodGhpcy5idWZmZXIuYnVmZmVyLmJ5dGVMZW5ndGgqMik7Y29uc3QgaT1uZXcgVWludDhBcnJheShuLDAsdCk7aS5zZXQodGhpcy5idWZmZXIpLHRoaXMuYnVmZmVyPWl9fWZ1bmN0aW9uIFAoZSx0KXtjb25zdCBuPXQucmVwbGFjZSgvWy9cLVxcXiQqKz8uKCl8W1xde31dL2csIlxcJCYiKSxpPW5ldyBSZWdFeHAoYF4ke259YCk7cmV0dXJuIGUucmVwbGFjZShpLCIiKX1jbGFzcyB1e2NvbnN0cnVjdG9yKHQsbil7ZCh0aGlzLCJkaXIiKTtkKHRoaXMsInByZWZpeCIpO3RoaXMuZGlyPXQsdGhpcy5wcmVmaXg9bn1jb250YWluc0ZpbGUodCl7Zm9yKGNvbnN0IG4gb2YgT2JqZWN0LmtleXModGhpcy5kaXIpKWlmKFAobix0aGlzLnByZWZpeCk9PT10KXJldHVybiEwO3JldHVybiExfWNvbnRhaW5zRGlyZWN0b3J5KHQpe2Zvcihjb25zdCBuIG9mIE9iamVjdC5rZXlzKHRoaXMuZGlyKSlpZihQKG4sdGhpcy5wcmVmaXgpLnN0YXJ0c1dpdGgoYCR7dH0vYCkpcmV0dXJuITA7cmV0dXJuITF9Y29udGFpbnModCl7Zm9yKGNvbnN0IG4gb2YgT2JqZWN0LmtleXModGhpcy5kaXIpKXtjb25zdCBpPVAobix0aGlzLnByZWZpeCk7aWYoaT09PXR8fGkuc3RhcnRzV2l0aChgJHt0fS9gKSlyZXR1cm4hMH1yZXR1cm4hMX1nZXQodCl7cmV0dXJuIHRoaXMuZGlyW3RoaXMuZnVsbFBhdGgodCldfWZ1bGxQYXRoKHQpe3JldHVybmAke3RoaXMucHJlZml4fSR7dH1gfWxpc3QoKXtjb25zdCB0PVtdLG49bmV3IFNldDtmb3IoY29uc3QgaSBvZiBPYmplY3Qua2V5cyh0aGlzLmRpcikpe2NvbnN0IHI9UChpLHRoaXMucHJlZml4KTtpZihyLmluY2x1ZGVzKCIvIikpe2NvbnN0IGE9ci5zcGxpdCgiLyIpWzBdO2lmKG4uaGFzKGEpKWNvbnRpbnVlO24uYWRkKGEpLHQucHVzaCh7bmFtZTphLHR5cGU6Yi5ESVJFQ1RPUll9KX1lbHNlIHQucHVzaCh7bmFtZTpyLHR5cGU6Yi5SRUdVTEFSX0ZJTEV9KX1yZXR1cm4gdH1zdGF0KCl7cmV0dXJue3BhdGg6dGhpcy5wcmVmaXgsdGltZXN0YW1wczp7YWNjZXNzOm5ldyBEYXRlLG1vZGlmaWNhdGlvbjpuZXcgRGF0ZSxjaGFuZ2U6bmV3IERhdGV9LHR5cGU6Yi5ESVJFQ1RPUlksYnl0ZUxlbmd0aDowfX19bGV0IGs9W107ZnVuY3Rpb24gVShlKXtrLnB1c2goZSl9ZnVuY3Rpb24gbnQoKXtjb25zdCBlPWs7cmV0dXJuIGs9W10sZX1jbGFzcyB4e2NvbnN0cnVjdG9yKHQpe2QodGhpcywiaW5zdGFuY2UiKTtkKHRoaXMsIm1vZHVsZSIpO2QodGhpcywibWVtb3J5Iik7ZCh0aGlzLCJjb250ZXh0Iik7ZCh0aGlzLCJkcml2ZSIpO3RoaXMuY29udGV4dD1uZXcgWCh0KSx0aGlzLmRyaXZlPW5ldyBldCh0aGlzLmNvbnRleHQuZnMpfXN0YXRpYyBhc3luYyBzdGFydCh0LG49e30pe2NvbnN0IGk9bmV3IHgobikscj1hd2FpdCBXZWJBc3NlbWJseS5pbnN0YW50aWF0ZVN0cmVhbWluZyh0LGkuZ2V0SW1wb3J0T2JqZWN0KCkpO3JldHVybiBpLnN0YXJ0KHIpfWdldEltcG9ydE9iamVjdCgpe3JldHVybnt3YXNpX3NuYXBzaG90X3ByZXZpZXcxOnRoaXMuZ2V0SW1wb3J0cygicHJldmlldzEiLHRoaXMuY29udGV4dC5kZWJ1Zyksd2FzaV91bnN0YWJsZTp0aGlzLmdldEltcG9ydHMoInVuc3RhYmxlIix0aGlzLmNvbnRleHQuZGVidWcpfX1zdGFydCh0LG49e30pe3RoaXMuaW5zdGFuY2U9dC5pbnN0YW5jZSx0aGlzLm1vZHVsZT10Lm1vZHVsZSx0aGlzLm1lbW9yeT1uLm1lbW9yeT8/dGhpcy5pbnN0YW5jZS5leHBvcnRzLm1lbW9yeTtjb25zdCBpPXRoaXMuaW5zdGFuY2UuZXhwb3J0cy5fc3RhcnQ7dHJ5e2koKX1jYXRjaChyKXtpZihyIGluc3RhbmNlb2YgailyZXR1cm57ZXhpdENvZGU6ci5jb2RlLGZzOnRoaXMuZHJpdmUuZnN9O2lmKHIgaW5zdGFuY2VvZiBXZWJBc3NlbWJseS5SdW50aW1lRXJyb3IpcmV0dXJue2V4aXRDb2RlOjEzNCxmczp0aGlzLmRyaXZlLmZzfTt0aHJvdyByfXJldHVybntleGl0Q29kZTowLGZzOnRoaXMuZHJpdmUuZnN9fWdldEltcG9ydHModCxuKXtjb25zdCBpPXthcmdzX2dldDp0aGlzLmFyZ3NfZ2V0LmJpbmQodGhpcyksYXJnc19zaXplc19nZXQ6dGhpcy5hcmdzX3NpemVzX2dldC5iaW5kKHRoaXMpLGNsb2NrX3Jlc19nZXQ6dGhpcy5jbG9ja19yZXNfZ2V0LmJpbmQodGhpcyksY2xvY2tfdGltZV9nZXQ6dGhpcy5jbG9ja190aW1lX2dldC5iaW5kKHRoaXMpLGVudmlyb25fZ2V0OnRoaXMuZW52aXJvbl9nZXQuYmluZCh0aGlzKSxlbnZpcm9uX3NpemVzX2dldDp0aGlzLmVudmlyb25fc2l6ZXNfZ2V0LmJpbmQodGhpcykscHJvY19leGl0OnRoaXMucHJvY19leGl0LmJpbmQodGhpcykscmFuZG9tX2dldDp0aGlzLnJhbmRvbV9nZXQuYmluZCh0aGlzKSxzY2hlZF95aWVsZDp0aGlzLnNjaGVkX3lpZWxkLmJpbmQodGhpcyksZmRfYWR2aXNlOnRoaXMuZmRfYWR2aXNlLmJpbmQodGhpcyksZmRfYWxsb2NhdGU6dGhpcy5mZF9hbGxvY2F0ZS5iaW5kKHRoaXMpLGZkX2Nsb3NlOnRoaXMuZmRfY2xvc2UuYmluZCh0aGlzKSxmZF9kYXRhc3luYzp0aGlzLmZkX2RhdGFzeW5jLmJpbmQodGhpcyksZmRfZmRzdGF0X2dldDp0aGlzLmZkX2Zkc3RhdF9nZXQuYmluZCh0aGlzKSxmZF9mZHN0YXRfc2V0X2ZsYWdzOnRoaXMuZmRfZmRzdGF0X3NldF9mbGFncy5iaW5kKHRoaXMpLGZkX2Zkc3RhdF9zZXRfcmlnaHRzOnRoaXMuZmRfZmRzdGF0X3NldF9yaWdodHMuYmluZCh0aGlzKSxmZF9maWxlc3RhdF9nZXQ6dGhpcy5mZF9maWxlc3RhdF9nZXQuYmluZCh0aGlzKSxmZF9maWxlc3RhdF9zZXRfc2l6ZTp0aGlzLmZkX2ZpbGVzdGF0X3NldF9zaXplLmJpbmQodGhpcyksZmRfZmlsZXN0YXRfc2V0X3RpbWVzOnRoaXMuZmRfZmlsZXN0YXRfc2V0X3RpbWVzLmJpbmQodGhpcyksZmRfcHJlYWQ6dGhpcy5mZF9wcmVhZC5iaW5kKHRoaXMpLGZkX3ByZXN0YXRfZGlyX25hbWU6dGhpcy5mZF9wcmVzdGF0X2Rpcl9uYW1lLmJpbmQodGhpcyksZmRfcHJlc3RhdF9nZXQ6dGhpcy5mZF9wcmVzdGF0X2dldC5iaW5kKHRoaXMpLGZkX3B3cml0ZTp0aGlzLmZkX3B3cml0ZS5iaW5kKHRoaXMpLGZkX3JlYWQ6dGhpcy5mZF9yZWFkLmJpbmQodGhpcyksZmRfcmVhZGRpcjp0aGlzLmZkX3JlYWRkaXIuYmluZCh0aGlzKSxmZF9yZW51bWJlcjp0aGlzLmZkX3JlbnVtYmVyLmJpbmQodGhpcyksZmRfc2Vlazp0aGlzLmZkX3NlZWsuYmluZCh0aGlzKSxmZF9zeW5jOnRoaXMuZmRfc3luYy5iaW5kKHRoaXMpLGZkX3RlbGw6dGhpcy5mZF90ZWxsLmJpbmQodGhpcyksZmRfd3JpdGU6dGhpcy5mZF93cml0ZS5iaW5kKHRoaXMpLHBhdGhfZmlsZXN0YXRfZ2V0OnRoaXMucGF0aF9maWxlc3RhdF9nZXQuYmluZCh0aGlzKSxwYXRoX2ZpbGVzdGF0X3NldF90aW1lczp0aGlzLnBhdGhfZmlsZXN0YXRfc2V0X3RpbWVzLmJpbmQodGhpcykscGF0aF9vcGVuOnRoaXMucGF0aF9vcGVuLmJpbmQodGhpcykscGF0aF9yZW5hbWU6dGhpcy5wYXRoX3JlbmFtZS5iaW5kKHRoaXMpLHBhdGhfdW5saW5rX2ZpbGU6dGhpcy5wYXRoX3VubGlua19maWxlLmJpbmQodGhpcykscGF0aF9jcmVhdGVfZGlyZWN0b3J5OnRoaXMucGF0aF9jcmVhdGVfZGlyZWN0b3J5LmJpbmQodGhpcykscGF0aF9saW5rOnRoaXMucGF0aF9saW5rLmJpbmQodGhpcykscGF0aF9yZWFkbGluazp0aGlzLnBhdGhfcmVhZGxpbmsuYmluZCh0aGlzKSxwYXRoX3JlbW92ZV9kaXJlY3Rvcnk6dGhpcy5wYXRoX3JlbW92ZV9kaXJlY3RvcnkuYmluZCh0aGlzKSxwYXRoX3N5bWxpbms6dGhpcy5wYXRoX3N5bWxpbmsuYmluZCh0aGlzKSxwb2xsX29uZW9mZjp0aGlzLnBvbGxfb25lb2ZmLmJpbmQodGhpcykscHJvY19yYWlzZTp0aGlzLnByb2NfcmFpc2UuYmluZCh0aGlzKSxzb2NrX2FjY2VwdDp0aGlzLnNvY2tfYWNjZXB0LmJpbmQodGhpcyksc29ja19yZWN2OnRoaXMuc29ja19yZWN2LmJpbmQodGhpcyksc29ja19zZW5kOnRoaXMuc29ja19zZW5kLmJpbmQodGhpcyksc29ja19zaHV0ZG93bjp0aGlzLnNvY2tfc2h1dGRvd24uYmluZCh0aGlzKSxzb2NrX29wZW46dGhpcy5zb2NrX29wZW4uYmluZCh0aGlzKSxzb2NrX2xpc3Rlbjp0aGlzLnNvY2tfbGlzdGVuLmJpbmQodGhpcyksc29ja19jb25uZWN0OnRoaXMuc29ja19jb25uZWN0LmJpbmQodGhpcyksc29ja19zZXRzb2Nrb3B0OnRoaXMuc29ja19zZXRzb2Nrb3B0LmJpbmQodGhpcyksc29ja19iaW5kOnRoaXMuc29ja19iaW5kLmJpbmQodGhpcyksc29ja19nZXRsb2NhbGFkZHI6dGhpcy5zb2NrX2dldGxvY2FsYWRkci5iaW5kKHRoaXMpLHNvY2tfZ2V0cGVlcmFkZHI6dGhpcy5zb2NrX2dldHBlZXJhZGRyLmJpbmQodGhpcyksc29ja19nZXRhZGRyaW5mbzp0aGlzLnNvY2tfZ2V0YWRkcmluZm8uYmluZCh0aGlzKX07dD09PSJ1bnN0YWJsZSImJihpLnBhdGhfZmlsZXN0YXRfZ2V0PXRoaXMudW5zdGFibGVfcGF0aF9maWxlc3RhdF9nZXQuYmluZCh0aGlzKSxpLmZkX2ZpbGVzdGF0X2dldD10aGlzLnVuc3RhYmxlX2ZkX2ZpbGVzdGF0X2dldC5iaW5kKHRoaXMpLGkuZmRfc2Vlaz10aGlzLnVuc3RhYmxlX2ZkX3NlZWsuYmluZCh0aGlzKSk7Zm9yKGNvbnN0W3IsYV1vZiBPYmplY3QuZW50cmllcyhpKSlpW3JdPWZ1bmN0aW9uKCl7bGV0IGY9YS5hcHBseSh0aGlzLGFyZ3VtZW50cyk7aWYobil7Y29uc3QgYz1udCgpO2Y9bihyLFsuLi5hcmd1bWVudHNdLGYsYyk/P2Z9cmV0dXJuIGZ9O3JldHVybiBpfWdldCBlbnZBcnJheSgpe3JldHVybiBPYmplY3QuZW50cmllcyh0aGlzLmNvbnRleHQuZW52KS5tYXAoKFt0LG5dKT0+YCR7dH09JHtufWApfWFyZ3NfZ2V0KHQsbil7Y29uc3QgaT1uZXcgRGF0YVZpZXcodGhpcy5tZW1vcnkuYnVmZmVyKTtmb3IoY29uc3QgciBvZiB0aGlzLmNvbnRleHQuYXJncyl7aS5zZXRVaW50MzIodCxuLCEwKSx0Kz00O2NvbnN0IGE9bmV3IFRleHRFbmNvZGVyKCkuZW5jb2RlKGAke3J9XDBgKTtuZXcgVWludDhBcnJheSh0aGlzLm1lbW9yeS5idWZmZXIsbixhLmJ5dGVMZW5ndGgpLnNldChhKSxuKz1hLmJ5dGVMZW5ndGh9cmV0dXJuIHMuU1VDQ0VTU31hcmdzX3NpemVzX2dldCh0LG4pe2NvbnN0IGk9dGhpcy5jb250ZXh0LmFyZ3Mscj1pLnJlZHVjZSgoZixjKT0+ZituZXcgVGV4dEVuY29kZXIoKS5lbmNvZGUoYCR7Y31cMGApLmJ5dGVMZW5ndGgsMCksYT1uZXcgRGF0YVZpZXcodGhpcy5tZW1vcnkuYnVmZmVyKTtyZXR1cm4gYS5zZXRVaW50MzIodCxpLmxlbmd0aCwhMCksYS5zZXRVaW50MzIobixyLCEwKSxzLlNVQ0NFU1N9Y2xvY2tfcmVzX2dldCh0LG4pe3N3aXRjaCh0KXtjYXNlIFQuUkVBTFRJTUU6Y2FzZSBULk1PTk9UT05JQzpjYXNlIFQuUFJPQ0VTU19DUFVUSU1FX0lEOmNhc2UgVC5USFJFQURfQ1BVVElNRV9JRDpyZXR1cm4gbmV3IERhdGFWaWV3KHRoaXMubWVtb3J5LmJ1ZmZlcikuc2V0QmlnVWludDY0KG4sQmlnSW50KDFlNiksITApLHMuU1VDQ0VTU31yZXR1cm4gcy5FSU5WQUx9Y2xvY2tfdGltZV9nZXQodCxuLGkpe3N3aXRjaCh0KXtjYXNlIFQuUkVBTFRJTUU6Y2FzZSBULk1PTk9UT05JQzpjYXNlIFQuUFJPQ0VTU19DUFVUSU1FX0lEOmNhc2UgVC5USFJFQURfQ1BVVElNRV9JRDpyZXR1cm4gbmV3IERhdGFWaWV3KHRoaXMubWVtb3J5LmJ1ZmZlcikuc2V0QmlnVWludDY0KGkseShuZXcgRGF0ZSksITApLHMuU1VDQ0VTU31yZXR1cm4gcy5FSU5WQUx9ZW52aXJvbl9nZXQodCxuKXtjb25zdCBpPW5ldyBEYXRhVmlldyh0aGlzLm1lbW9yeS5idWZmZXIpO2Zvcihjb25zdCByIG9mIHRoaXMuZW52QXJyYXkpe2kuc2V0VWludDMyKHQsbiwhMCksdCs9NDtjb25zdCBhPW5ldyBUZXh0RW5jb2RlcigpLmVuY29kZShgJHtyfVwwYCk7bmV3IFVpbnQ4QXJyYXkodGhpcy5tZW1vcnkuYnVmZmVyLG4sYS5ieXRlTGVuZ3RoKS5zZXQoYSksbis9YS5ieXRlTGVuZ3RofXJldHVybiBzLlNVQ0NFU1N9ZW52aXJvbl9zaXplc19nZXQodCxuKXtjb25zdCBpPXRoaXMuZW52QXJyYXkucmVkdWNlKChhLGYpPT5hK25ldyBUZXh0RW5jb2RlcigpLmVuY29kZShgJHtmfVwwYCkuYnl0ZUxlbmd0aCwwKSxyPW5ldyBEYXRhVmlldyh0aGlzLm1lbW9yeS5idWZmZXIpO3JldHVybiByLnNldFVpbnQzMih0LHRoaXMuZW52QXJyYXkubGVuZ3RoLCEwKSxyLnNldFVpbnQzMihuLGksITApLHMuU1VDQ0VTU31wcm9jX2V4aXQodCl7dGhyb3cgbmV3IGoodCl9cmFuZG9tX2dldCh0LG4pe2NvbnN0IGk9bmV3IFVpbnQ4QXJyYXkodGhpcy5tZW1vcnkuYnVmZmVyLHQsbik7cmV0dXJuIGdsb2JhbFRoaXMuY3J5cHRvLmdldFJhbmRvbVZhbHVlcyhpKSxzLlNVQ0NFU1N9c2NoZWRfeWllbGQoKXtyZXR1cm4gcy5TVUNDRVNTfWZkX3JlYWQodCxuLGkscil7aWYodD09PTF8fHQ9PT0yKXJldHVybiBzLkVOT1RTVVA7Y29uc3QgYT1uZXcgRGF0YVZpZXcodGhpcy5tZW1vcnkuYnVmZmVyKSxmPXYoYSxuLGkpLGM9bmV3IFRleHRFbmNvZGVyO2xldCBvPTAsRT1zLlNVQ0NFU1M7Zm9yKGNvbnN0IGggb2YgZil7bGV0IFM7aWYodD09PTApe2NvbnN0IEM9dGhpcy5jb250ZXh0LnN0ZGluKGguYnl0ZUxlbmd0aCk7aWYoIUMpYnJlYWs7Uz1jLmVuY29kZShDKX1lbHNle2NvbnN0W0MsbF09dGhpcy5kcml2ZS5yZWFkKHQsaC5ieXRlTGVuZ3RoKTtpZihDKXtFPUM7YnJlYWt9ZWxzZSBTPWx9Y29uc3QgZz1NYXRoLm1pbihoLmJ5dGVMZW5ndGgsUy5ieXRlTGVuZ3RoKTtoLnNldChTLnN1YmFycmF5KDAsZykpLG8rPWd9cmV0dXJuIFUoe2J5dGVzUmVhZDpvfSksYS5zZXRVaW50MzIocixvLCEwKSxFfWZkX3dyaXRlKHQsbixpLHIpe2lmKHQ9PT0wKXJldHVybiBzLkVOT1RTVVA7Y29uc3QgYT1uZXcgRGF0YVZpZXcodGhpcy5tZW1vcnkuYnVmZmVyKSxmPXYoYSxuLGkpLGM9bmV3IFRleHREZWNvZGVyO2xldCBvPTAsRT1zLlNVQ0NFU1M7Zm9yKGNvbnN0IGggb2YgZilpZihoLmJ5dGVMZW5ndGghPT0wKXtpZih0PT09MXx8dD09PTIpe2NvbnN0IFM9dD09PTE/dGhpcy5jb250ZXh0LnN0ZG91dDp0aGlzLmNvbnRleHQuc3RkZXJyLGc9Yy5kZWNvZGUoaCk7UyhnKSxVKHtvdXRwdXQ6Z30pfWVsc2UgaWYoRT10aGlzLmRyaXZlLndyaXRlKHQsaCksRSE9cy5TVUNDRVNTKWJyZWFrO28rPWguYnl0ZUxlbmd0aH1yZXR1cm4gYS5zZXRVaW50MzIocixvLCEwKSxFfWZkX2FkdmlzZSgpe3JldHVybiBzLlNVQ0NFU1N9ZmRfYWxsb2NhdGUodCxuLGkpe3JldHVybiB0aGlzLmRyaXZlLnB3cml0ZSh0LG5ldyBVaW50OEFycmF5KE51bWJlcihpKSksTnVtYmVyKG4pKX1mZF9jbG9zZSh0KXtyZXR1cm4gdGhpcy5kcml2ZS5jbG9zZSh0KX1mZF9kYXRhc3luYyh0KXtyZXR1cm4gdGhpcy5kcml2ZS5zeW5jKHQpfWZkX2Zkc3RhdF9nZXQodCxuKXtpZih0PDMpe2xldCBjO2lmKHRoaXMuY29udGV4dC5pc1RUWSl7Y29uc3QgRT1ZXl8uRkRfU0VFS15fLkZEX1RFTEw7Yz1WKGIuQ0hBUkFDVEVSX0RFVklDRSwwLEUpfWVsc2UgYz1WKGIuQ0hBUkFDVEVSX0RFVklDRSwwKTtyZXR1cm4gbmV3IFVpbnQ4QXJyYXkodGhpcy5tZW1vcnkuYnVmZmVyLG4sYy5ieXRlTGVuZ3RoKS5zZXQoYykscy5TVUNDRVNTfWlmKCF0aGlzLmRyaXZlLmV4aXN0cyh0KSlyZXR1cm4gcy5FQkFERjtjb25zdCBpPXRoaXMuZHJpdmUuZmlsZVR5cGUodCkscj10aGlzLmRyaXZlLmZpbGVGZGZsYWdzKHQpLGE9VihpLHIpO3JldHVybiBuZXcgVWludDhBcnJheSh0aGlzLm1lbW9yeS5idWZmZXIsbixhLmJ5dGVMZW5ndGgpLnNldChhKSxzLlNVQ0NFU1N9ZmRfZmRzdGF0X3NldF9mbGFncyh0LG4pe3JldHVybiB0aGlzLmRyaXZlLnNldEZsYWdzKHQsbil9ZmRfZmRzdGF0X3NldF9yaWdodHMoKXtyZXR1cm4gcy5TVUNDRVNTfWZkX2ZpbGVzdGF0X2dldCh0LG4pe3JldHVybiB0aGlzLnNoYXJlZF9mZF9maWxlc3RhdF9nZXQodCxuLCJwcmV2aWV3MSIpfXVuc3RhYmxlX2ZkX2ZpbGVzdGF0X2dldCh0LG4pe3JldHVybiB0aGlzLnNoYXJlZF9mZF9maWxlc3RhdF9nZXQodCxuLCJ1bnN0YWJsZSIpfXNoYXJlZF9mZF9maWxlc3RhdF9nZXQodCxuLGkpe2NvbnN0IHI9aT09PSJ1bnN0YWJsZSI/UTpaO2lmKHQ8Myl7bGV0IEU7c3dpdGNoKHQpe2Nhc2UgMDpFPSIvZGV2L3N0ZGluIjticmVhaztjYXNlIDE6RT0iL2Rldi9zdGRvdXQiO2JyZWFrO2Nhc2UgMjpFPSIvZGV2L3N0ZGVyciI7YnJlYWs7ZGVmYXVsdDpFPSIvZGV2L3VuZGVmaW5lZCI7YnJlYWt9Y29uc3QgaD1yKHtwYXRoOkUsYnl0ZUxlbmd0aDowLHRpbWVzdGFtcHM6e2FjY2VzczpuZXcgRGF0ZSxtb2RpZmljYXRpb246bmV3IERhdGUsY2hhbmdlOm5ldyBEYXRlfSx0eXBlOmIuQ0hBUkFDVEVSX0RFVklDRX0pO3JldHVybiBuZXcgVWludDhBcnJheSh0aGlzLm1lbW9yeS5idWZmZXIsbixoLmJ5dGVMZW5ndGgpLnNldChoKSxzLlNVQ0NFU1N9Y29uc3RbYSxmXT10aGlzLmRyaXZlLnN0YXQodCk7aWYoYSE9cy5TVUNDRVNTKXJldHVybiBhO1Uoe3Jlc29sdmVkUGF0aDpmLnBhdGgsc3RhdDpmfSk7Y29uc3QgYz1yKGYpO3JldHVybiBuZXcgVWludDhBcnJheSh0aGlzLm1lbW9yeS5idWZmZXIsbixjLmJ5dGVMZW5ndGgpLnNldChjKSxzLlNVQ0NFU1N9ZmRfZmlsZXN0YXRfc2V0X3NpemUodCxuKXtyZXR1cm4gdGhpcy5kcml2ZS5zZXRTaXplKHQsbil9ZmRfZmlsZXN0YXRfc2V0X3RpbWVzKHQsbixpLHIpe2xldCBhPW51bGw7ciZtLkFUSU0mJihhPXAobikpLHImbS5BVElNX05PVyYmKGE9bmV3IERhdGUpO2xldCBmPW51bGw7aWYociZtLk1USU0mJihmPXAoaSkpLHImbS5NVElNX05PVyYmKGY9bmV3IERhdGUpLGEpe2NvbnN0IGM9dGhpcy5kcml2ZS5zZXRBY2Nlc3NUaW1lKHQsYSk7aWYoYyE9cy5TVUNDRVNTKXJldHVybiBjfWlmKGYpe2NvbnN0IGM9dGhpcy5kcml2ZS5zZXRNb2RpZmljYXRpb25UaW1lKHQsZik7aWYoYyE9cy5TVUNDRVNTKXJldHVybiBjfXJldHVybiBzLlNVQ0NFU1N9ZmRfcHJlYWQodCxuLGkscixhKXtpZih0PT09MXx8dD09PTIpcmV0dXJuIHMuRU5PVFNVUDtpZih0PT09MClyZXR1cm4gdGhpcy5mZF9yZWFkKHQsbixpLGEpO2NvbnN0IGY9bmV3IERhdGFWaWV3KHRoaXMubWVtb3J5LmJ1ZmZlciksYz12KGYsbixpKTtsZXQgbz0wLEU9cy5TVUNDRVNTO2Zvcihjb25zdCBoIG9mIGMpe2NvbnN0W1MsZ109dGhpcy5kcml2ZS5wcmVhZCh0LGguYnl0ZUxlbmd0aCxOdW1iZXIocikrbyk7aWYoUyE9PXMuU1VDQ0VTUyl7RT1TO2JyZWFrfWNvbnN0IEM9TWF0aC5taW4oaC5ieXRlTGVuZ3RoLGcuYnl0ZUxlbmd0aCk7aC5zZXQoZy5zdWJhcnJheSgwLEMpKSxvKz1DfXJldHVybiBmLnNldFVpbnQzMihhLG8sITApLEV9ZmRfcHJlc3RhdF9kaXJfbmFtZSh0LG4saSl7aWYodCE9PTMpcmV0dXJuIHMuRUJBREY7Y29uc3Qgcj1uZXcgVGV4dEVuY29kZXIoKS5lbmNvZGUoIi8iKTtyZXR1cm4gbmV3IFVpbnQ4QXJyYXkodGhpcy5tZW1vcnkuYnVmZmVyLG4saSkuc2V0KHIuc3ViYXJyYXkoMCxpKSkscy5TVUNDRVNTfWZkX3ByZXN0YXRfZ2V0KHQsbil7aWYodCE9PTMpcmV0dXJuIHMuRUJBREY7Y29uc3QgaT1uZXcgVGV4dEVuY29kZXIoKS5lbmNvZGUoIi4iKSxyPW5ldyBEYXRhVmlldyh0aGlzLm1lbW9yeS5idWZmZXIsbik7cmV0dXJuIHIuc2V0VWludDgoMCxHLkRJUiksci5zZXRVaW50MzIoNCxpLmJ5dGVMZW5ndGgsITApLHMuU1VDQ0VTU31mZF9wd3JpdGUodCxuLGkscixhKXtpZih0PT09MClyZXR1cm4gcy5FTk9UU1VQO2lmKHQ9PT0xfHx0PT09MilyZXR1cm4gdGhpcy5mZF93cml0ZSh0LG4saSxhKTtjb25zdCBmPW5ldyBEYXRhVmlldyh0aGlzLm1lbW9yeS5idWZmZXIpLGM9dihmLG4saSk7bGV0IG89MCxFPXMuU1VDQ0VTUztmb3IoY29uc3QgaCBvZiBjKWlmKGguYnl0ZUxlbmd0aCE9PTApe2lmKEU9dGhpcy5kcml2ZS5wd3JpdGUodCxoLE51bWJlcihyKSksRSE9cy5TVUNDRVNTKWJyZWFrO28rPWguYnl0ZUxlbmd0aH1yZXR1cm4gZi5zZXRVaW50MzIoYSxvLCEwKSxFfWZkX3JlYWRkaXIodCxuLGkscixhKXtjb25zdFtmLGNdPXRoaXMuZHJpdmUubGlzdCh0KTtpZihmIT1zLlNVQ0NFU1MpcmV0dXJuIGY7bGV0IG89W10sRT0wO2Zvcihjb25zdHtuYW1lOncsdHlwZTpGfW9mIGMpe2NvbnN0IEs9cnQodyxGLEUpO28ucHVzaChLKSxFKyt9bz1vLnNsaWNlKE51bWJlcihyKSk7Y29uc3QgaD1vLnJlZHVjZSgodyxGKT0+dytGLmJ5dGVMZW5ndGgsMCksUz1uZXcgVWludDhBcnJheShoKTtsZXQgZz0wO2Zvcihjb25zdCB3IG9mIG8pUy5zZXQodyxnKSxnKz13LmJ5dGVMZW5ndGg7Y29uc3QgQz1uZXcgVWludDhBcnJheSh0aGlzLm1lbW9yeS5idWZmZXIsbixpKSxsPVMuc3ViYXJyYXkoMCxpKTtyZXR1cm4gQy5zZXQobCksbmV3IERhdGFWaWV3KHRoaXMubWVtb3J5LmJ1ZmZlcikuc2V0VWludDMyKGEsbC5ieXRlTGVuZ3RoLCEwKSxzLlNVQ0NFU1N9ZmRfcmVudW1iZXIodCxuKXtyZXR1cm4gdGhpcy5kcml2ZS5yZW51bWJlcih0LG4pfWZkX3NlZWsodCxuLGkscil7Y29uc3RbYSxmXT10aGlzLmRyaXZlLnNlZWsodCxuLGkpO3JldHVybiBhIT09cy5TVUNDRVNTfHwoVSh7bmV3T2Zmc2V0OmYudG9TdHJpbmcoKX0pLG5ldyBEYXRhVmlldyh0aGlzLm1lbW9yeS5idWZmZXIpLnNldEJpZ1VpbnQ2NChyLGYsITApKSxhfXVuc3RhYmxlX2ZkX3NlZWsodCxuLGkscil7Y29uc3QgYT1hdFtpXTtyZXR1cm4gdGhpcy5mZF9zZWVrKHQsbixhLHIpfWZkX3N5bmModCl7cmV0dXJuIHRoaXMuZHJpdmUuc3luYyh0KX1mZF90ZWxsKHQsbil7Y29uc3RbaSxyXT10aGlzLmRyaXZlLnRlbGwodCk7cmV0dXJuIGkhPT1zLlNVQ0NFU1N8fG5ldyBEYXRhVmlldyh0aGlzLm1lbW9yeS5idWZmZXIpLnNldEJpZ1VpbnQ2NChuLHIsITApLGl9cGF0aF9maWxlc3RhdF9nZXQodCxuLGkscixhKXtyZXR1cm4gdGhpcy5zaGFyZWRfcGF0aF9maWxlc3RhdF9nZXQodCxuLGkscixhLCJwcmV2aWV3MSIpfXVuc3RhYmxlX3BhdGhfZmlsZXN0YXRfZ2V0KHQsbixpLHIsYSl7cmV0dXJuIHRoaXMuc2hhcmVkX3BhdGhfZmlsZXN0YXRfZ2V0KHQsbixpLHIsYSwidW5zdGFibGUiKX1zaGFyZWRfcGF0aF9maWxlc3RhdF9nZXQodCxuLGkscixhLGYpe2NvbnN0IGM9Zj09PSJ1bnN0YWJsZSI/UTpaLG89bmV3IFRleHREZWNvZGVyKCkuZGVjb2RlKG5ldyBVaW50OEFycmF5KHRoaXMubWVtb3J5LmJ1ZmZlcixpLHIpKTtVKHtwYXRoOm99KTtjb25zdFtFLGhdPXRoaXMuZHJpdmUucGF0aFN0YXQodCxvKTtpZihFIT1zLlNVQ0NFU1MpcmV0dXJuIEU7Y29uc3QgUz1jKGgpO3JldHVybiBuZXcgVWludDhBcnJheSh0aGlzLm1lbW9yeS5idWZmZXIsYSxTLmJ5dGVMZW5ndGgpLnNldChTKSxFfXBhdGhfZmlsZXN0YXRfc2V0X3RpbWVzKHQsbixpLHIsYSxmLGMpe2xldCBvPW51bGw7YyZtLkFUSU0mJihvPXAoYSkpLGMmbS5BVElNX05PVyYmKG89bmV3IERhdGUpO2xldCBFPW51bGw7YyZtLk1USU0mJihFPXAoZikpLGMmbS5NVElNX05PVyYmKEU9bmV3IERhdGUpO2NvbnN0IGg9bmV3IFRleHREZWNvZGVyKCkuZGVjb2RlKG5ldyBVaW50OEFycmF5KHRoaXMubWVtb3J5LmJ1ZmZlcixpLHIpKTtpZihvKXtjb25zdCBTPXRoaXMuZHJpdmUucGF0aFNldEFjY2Vzc1RpbWUodCxoLG8pO2lmKFMhPXMuU1VDQ0VTUylyZXR1cm4gU31pZihFKXtjb25zdCBTPXRoaXMuZHJpdmUucGF0aFNldE1vZGlmaWNhdGlvblRpbWUodCxoLEUpO2lmKFMhPXMuU1VDQ0VTUylyZXR1cm4gU31yZXR1cm4gcy5TVUNDRVNTfXBhdGhfb3Blbih0LG4saSxyLGEsZixjLG8sRSl7Y29uc3QgaD1uZXcgRGF0YVZpZXcodGhpcy5tZW1vcnkuYnVmZmVyKSxTPUIodGhpcy5tZW1vcnksaSxyKSxnPSEhKGEmTi5DUkVBVCksQz0hIShhJk4uRElSRUNUT1JZKSxsPSEhKGEmTi5FWENMKSxxPSEhKGEmTi5UUlVOQyksdz0hIShvJk8uQVBQRU5EKSxGPSEhKG8mTy5EU1lOQyksSz0hIShvJk8uTk9OQkxPQ0spLFN0PSEhKG8mTy5SU1lOQyksX3Q9ISEobyZPLlNZTkMpO1Uoe3BhdGg6UyxvcGVuRmxhZ3M6e2NyZWF0ZUZpbGVJZk5vbmU6ZyxmYWlsSWZOb3REaXI6QyxmYWlsSWZGaWxlRXhpc3RzOmwsdHJ1bmNhdGVGaWxlOnF9LGZpbGVEZXNjcmlwdG9yRmxhZ3M6e2ZsYWdBcHBlbmQ6dyxmbGFnRFN5bmM6RixmbGFnTm9uQmxvY2s6SyxmbGFnUlN5bmM6U3QsZmxhZ1N5bmM6X3R9fSk7Y29uc3RbUixkdF09dGhpcy5kcml2ZS5vcGVuKHQsUyxhLG8pO3JldHVybiBSfHwoaC5zZXRVaW50MzIoRSxkdCwhMCksUil9cGF0aF9yZW5hbWUodCxuLGkscixhLGYpe2NvbnN0IGM9Qih0aGlzLm1lbW9yeSxuLGkpLG89Qih0aGlzLm1lbW9yeSxhLGYpO3JldHVybiBVKHtvbGRQYXRoOmMsbmV3UGF0aDpvfSksdGhpcy5kcml2ZS5yZW5hbWUodCxjLHIsbyl9cGF0aF91bmxpbmtfZmlsZSh0LG4saSl7Y29uc3Qgcj1CKHRoaXMubWVtb3J5LG4saSk7cmV0dXJuIFUoe3BhdGg6cn0pLHRoaXMuZHJpdmUudW5saW5rKHQscil9cG9sbF9vbmVvZmYodCxuLGkscil7Zm9yKGxldCBmPTA7ZjxpO2YrKyl7Y29uc3QgYz1uZXcgVWludDhBcnJheSh0aGlzLm1lbW9yeS5idWZmZXIsdCtmKnoseiksbz1pdChjKSxFPW5ldyBVaW50OEFycmF5KHRoaXMubWVtb3J5LmJ1ZmZlcixuK2YqJCwkKTtsZXQgaD0wLFM9cy5TVUNDRVNTO3N3aXRjaChvLnR5cGUpe2Nhc2UgQS5DTE9DSzpmb3IoO25ldyBEYXRlPG8udGltZW91dDspO0Uuc2V0KHN0KG8udXNlcmRhdGEscy5TVUNDRVNTKSk7YnJlYWs7Y2FzZSBBLkZEX1JFQUQ6aWYoby5mZDwzKW8uZmQ9PT0wPyhTPXMuU1VDQ0VTUyxoPTMyKTpTPXMuRUJBREY7ZWxzZXtjb25zdFtnLENdPXRoaXMuZHJpdmUuc3RhdChvLmZkKTtTPWcsaD1DP0MuYnl0ZUxlbmd0aDowfUUuc2V0KEooby51c2VyZGF0YSxTLEEuRkRfUkVBRCxCaWdJbnQoaCkpKTticmVhaztjYXNlIEEuRkRfV1JJVEU6aWYoaD0wLFM9cy5TVUNDRVNTLG8uZmQ8MylvLmZkPT09MD9TPXMuRUJBREY6KFM9cy5TVUNDRVNTLGg9MTAyNCk7ZWxzZXtjb25zdFtnLENdPXRoaXMuZHJpdmUuc3RhdChvLmZkKTtTPWcsaD1DP0MuYnl0ZUxlbmd0aDowfUUuc2V0KEooby51c2VyZGF0YSxTLEEuRkRfUkVBRCxCaWdJbnQoaCkpKTticmVha319cmV0dXJuIG5ldyBEYXRhVmlldyh0aGlzLm1lbW9yeS5idWZmZXIsciw0KS5zZXRVaW50MzIoMCxpLCEwKSxzLlNVQ0NFU1N9cGF0aF9jcmVhdGVfZGlyZWN0b3J5KHQsbixpKXtjb25zdCByPUIodGhpcy5tZW1vcnksbixpKTtyZXR1cm4gdGhpcy5kcml2ZS5wYXRoQ3JlYXRlRGlyKHQscil9cGF0aF9saW5rKCl7cmV0dXJuIHMuRU5PU1lTfXBhdGhfcmVhZGxpbmsoKXtyZXR1cm4gcy5FTk9TWVN9cGF0aF9yZW1vdmVfZGlyZWN0b3J5KCl7cmV0dXJuIHMuRU5PU1lTfXBhdGhfc3ltbGluaygpe3JldHVybiBzLkVOT1NZU31wcm9jX3JhaXNlKCl7cmV0dXJuIHMuRU5PU1lTfXNvY2tfYWNjZXB0KCl7cmV0dXJuIHMuRU5PU1lTfXNvY2tfcmVjdigpe3JldHVybiBzLkVOT1NZU31zb2NrX3NlbmQoKXtyZXR1cm4gcy5FTk9TWVN9c29ja19zaHV0ZG93bigpe3JldHVybiBzLkVOT1NZU31zb2NrX29wZW4oKXtyZXR1cm4gcy5FTk9TWVN9c29ja19saXN0ZW4oKXtyZXR1cm4gcy5FTk9TWVN9c29ja19jb25uZWN0KCl7cmV0dXJuIHMuRU5PU1lTfXNvY2tfc2V0c29ja29wdCgpe3JldHVybiBzLkVOT1NZU31zb2NrX2JpbmQoKXtyZXR1cm4gcy5FTk9TWVN9c29ja19nZXRsb2NhbGFkZHIoKXtyZXR1cm4gcy5FTk9TWVN9c29ja19nZXRwZWVyYWRkcigpe3JldHVybiBzLkVOT1NZU31zb2NrX2dldGFkZHJpbmZvKCl7cmV0dXJuIHMuRU5PU1lTfX1jb25zdCBZPV8uRkRfREFUQVNZTkN8Xy5GRF9SRUFEfF8uRkRfU0VFS3xfLkZEX0ZEU1RBVF9TRVRfRkxBR1N8Xy5GRF9TWU5DfF8uRkRfVEVMTHxfLkZEX1dSSVRFfF8uRkRfQURWSVNFfF8uRkRfQUxMT0NBVEV8Xy5QQVRIX0NSRUFURV9ESVJFQ1RPUll8Xy5QQVRIX0NSRUFURV9GSUxFfF8uUEFUSF9MSU5LX1NPVVJDRXxfLlBBVEhfTElOS19UQVJHRVR8Xy5QQVRIX09QRU58Xy5GRF9SRUFERElSfF8uUEFUSF9SRUFETElOS3xfLlBBVEhfUkVOQU1FX1NPVVJDRXxfLlBBVEhfUkVOQU1FX1RBUkdFVHxfLlBBVEhfRklMRVNUQVRfR0VUfF8uUEFUSF9GSUxFU1RBVF9TRVRfU0laRXxfLlBBVEhfRklMRVNUQVRfU0VUX1RJTUVTfF8uRkRfRklMRVNUQVRfR0VUfF8uRkRfRklMRVNUQVRfU0VUX1NJWkV8Xy5GRF9GSUxFU1RBVF9TRVRfVElNRVN8Xy5QQVRIX1NZTUxJTkt8Xy5QQVRIX1JFTU9WRV9ESVJFQ1RPUll8Xy5QQVRIX1VOTElOS19GSUxFfF8uUE9MTF9GRF9SRUFEV1JJVEV8Xy5TT0NLX1NIVVRET1dOfF8uU09DS19BQ0NFUFQ7Y2xhc3MgaiBleHRlbmRzIEVycm9ye2NvbnN0cnVjdG9yKG4pe3N1cGVyKCk7ZCh0aGlzLCJjb2RlIik7dGhpcy5jb2RlPW59fWZ1bmN0aW9uIEIoZSx0LG4pe3JldHVybiBuZXcgVGV4dERlY29kZXIoKS5kZWNvZGUobmV3IFVpbnQ4QXJyYXkoZS5idWZmZXIsdCxuKSl9ZnVuY3Rpb24gdihlLHQsbil7bGV0IGk9QXJyYXkobik7Zm9yKGxldCByPTA7cjxuO3IrKyl7Y29uc3QgYT1lLmdldFVpbnQzMih0LCEwKTt0Kz00O2NvbnN0IGY9ZS5nZXRVaW50MzIodCwhMCk7dCs9NCxpW3JdPW5ldyBVaW50OEFycmF5KGUuYnVmZmVyLGEsZil9cmV0dXJuIGl9ZnVuY3Rpb24gaXQoZSl7Y29uc3QgdD1uZXcgVWludDhBcnJheSg4KTt0LnNldChlLnN1YmFycmF5KDAsOCkpO2NvbnN0IG49ZVs4XSxpPW5ldyBEYXRhVmlldyhlLmJ1ZmZlcixlLmJ5dGVPZmZzZXQrOSk7c3dpdGNoKG4pe2Nhc2UgQS5GRF9SRUFEOmNhc2UgQS5GRF9XUklURTpyZXR1cm57dXNlcmRhdGE6dCx0eXBlOm4sZmQ6aS5nZXRVaW50MzIoMCwhMCl9O2Nhc2UgQS5DTE9DSzpjb25zdCByPWkuZ2V0VWludDE2KDI0LCEwKSxhPXkobmV3IERhdGUpLGY9aS5nZXRCaWdVaW50NjQoOCwhMCksYz1pLmdldEJpZ1VpbnQ2NCgxNiwhMCksbz1yJnR0LlNVQlNDUklQVElPTl9DTE9DS19BQlNUSU1FP2Y6YStmO3JldHVybnt1c2VyZGF0YTp0LHR5cGU6bixpZDppLmdldFVpbnQzMigwLCEwKSx0aW1lb3V0OnAobykscHJlY2lzaW9uOnAobytjKX19fWZ1bmN0aW9uIFooZSl7Y29uc3QgdD1uZXcgVWludDhBcnJheShXKSxuPW5ldyBEYXRhVmlldyh0LmJ1ZmZlcik7cmV0dXJuIG4uc2V0QmlnVWludDY0KDAsQmlnSW50KDApLCEwKSxuLnNldEJpZ1VpbnQ2NCg4LEJpZ0ludChIKGUucGF0aCkpLCEwKSxuLnNldFVpbnQ4KDE2LGUudHlwZSksbi5zZXRCaWdVaW50NjQoMjQsQmlnSW50KDEpLCEwKSxuLnNldEJpZ1VpbnQ2NCgzMixCaWdJbnQoZS5ieXRlTGVuZ3RoKSwhMCksbi5zZXRCaWdVaW50NjQoNDAseShlLnRpbWVzdGFtcHMuYWNjZXNzKSwhMCksbi5zZXRCaWdVaW50NjQoNDgseShlLnRpbWVzdGFtcHMubW9kaWZpY2F0aW9uKSwhMCksbi5zZXRCaWdVaW50NjQoNTYseShlLnRpbWVzdGFtcHMuY2hhbmdlKSwhMCksdH1mdW5jdGlvbiBRKGUpe2NvbnN0IHQ9bmV3IFVpbnQ4QXJyYXkoVyksbj1uZXcgRGF0YVZpZXcodC5idWZmZXIpO3JldHVybiBuLnNldEJpZ1VpbnQ2NCgwLEJpZ0ludCgwKSwhMCksbi5zZXRCaWdVaW50NjQoOCxCaWdJbnQoSChlLnBhdGgpKSwhMCksbi5zZXRVaW50OCgxNixlLnR5cGUpLG4uc2V0VWludDMyKDIwLDEsITApLG4uc2V0QmlnVWludDY0KDI0LEJpZ0ludChlLmJ5dGVMZW5ndGgpLCEwKSxuLnNldEJpZ1VpbnQ2NCgzMix5KGUudGltZXN0YW1wcy5hY2Nlc3MpLCEwKSxuLnNldEJpZ1VpbnQ2NCg0MCx5KGUudGltZXN0YW1wcy5tb2RpZmljYXRpb24pLCEwKSxuLnNldEJpZ1VpbnQ2NCg0OCx5KGUudGltZXN0YW1wcy5jaGFuZ2UpLCEwKSx0fWZ1bmN0aW9uIFYoZSx0LG4pe2NvbnN0IGk9bj8/WSxyPW4/P1ksYT1uZXcgVWludDhBcnJheSgyNCksZj1uZXcgRGF0YVZpZXcoYS5idWZmZXIsMCwyNCk7cmV0dXJuIGYuc2V0VWludDgoMCxlKSxmLnNldFVpbnQzMigyLHQsITApLGYuc2V0QmlnVWludDY0KDgsaSwhMCksZi5zZXRCaWdVaW50NjQoMTYsciwhMCksYX1mdW5jdGlvbiBydChlLHQsbil7Y29uc3QgaT1uZXcgVGV4dEVuY29kZXIoKS5lbmNvZGUoZSkscj0yNCtpLmJ5dGVMZW5ndGgsYT1uZXcgVWludDhBcnJheShyKSxmPW5ldyBEYXRhVmlldyhhLmJ1ZmZlcik7cmV0dXJuIGYuc2V0QmlnVWludDY0KDAsQmlnSW50KG4rMSksITApLGYuc2V0QmlnVWludDY0KDgsQmlnSW50KEgoZSkpLCEwKSxmLnNldFVpbnQzMigxNixpLmxlbmd0aCwhMCksZi5zZXRVaW50OCgyMCx0KSxhLnNldChpLDI0KSxhfWZ1bmN0aW9uIHN0KGUsdCl7Y29uc3Qgbj1uZXcgVWludDhBcnJheSgzMik7bi5zZXQoZSwwKTtjb25zdCBpPW5ldyBEYXRhVmlldyhuLmJ1ZmZlcik7cmV0dXJuIGkuc2V0VWludDE2KDgsdCwhMCksaS5zZXRVaW50MTYoMTAsQS5DTE9DSywhMCksbn1mdW5jdGlvbiBKKGUsdCxuLGkpe2NvbnN0IHI9bmV3IFVpbnQ4QXJyYXkoMzIpO3Iuc2V0KGUsMCk7Y29uc3QgYT1uZXcgRGF0YVZpZXcoci5idWZmZXIpO3JldHVybiBhLnNldFVpbnQxNig4LHQsITApLGEuc2V0VWludDE2KDEwLG4sITApLGEuc2V0QmlnVWludDY0KDE2LGksITApLHJ9ZnVuY3Rpb24gSChlLHQ9MCl7bGV0IG49MzczNTkyODU1OV50LGk9MTEwMzU0Nzk5MV50O2ZvcihsZXQgcj0wLGE7cjxlLmxlbmd0aDtyKyspYT1lLmNoYXJDb2RlQXQociksbj1NYXRoLmltdWwobl5hLDI2NTQ0MzU3NjEpLGk9TWF0aC5pbXVsKGleYSwxNTk3MzM0Njc3KTtyZXR1cm4gbj1NYXRoLmltdWwobl5uPj4+MTYsMjI0NjgyMjUwNyleTWF0aC5pbXVsKGleaT4+PjEzLDMyNjY0ODk5MDkpLGk9TWF0aC5pbXVsKGleaT4+PjE2LDIyNDY4MjI1MDcpXk1hdGguaW11bChuXm4+Pj4xMywzMjY2NDg5OTA5KSw0Mjk0OTY3Mjk2KigyMDk3MTUxJmkpKyhuPj4+MCl9ZnVuY3Rpb24geShlKXtyZXR1cm4gQmlnSW50KGUuZ2V0VGltZSgpKSpCaWdJbnQoMWU2KX1mdW5jdGlvbiBwKGUpe3JldHVybiBuZXcgRGF0ZShOdW1iZXIoZS9CaWdJbnQoMWU2KSkpfWNvbnN0IGF0PXtbTS5DVVJdOkQuQ1VSLFtNLkVORF06RC5FTkQsW00uU0VUXTpELlNFVH07b25tZXNzYWdlPWFzeW5jIGU9Pntjb25zdCB0PWUuZGF0YTtzd2l0Y2godC50eXBlKXtjYXNlInN0YXJ0Ijp0cnl7Y29uc3Qgbj1hd2FpdCBmdCh0LmJpbmFyeVVSTCx0LnN0ZGluQnVmZmVyLHQpO0woe3RhcmdldDoiaG9zdCIsdHlwZToicmVzdWx0IixyZXN1bHQ6bn0pfWNhdGNoKG4pe2xldCBpO24gaW5zdGFuY2VvZiBFcnJvcj9pPXttZXNzYWdlOm4ubWVzc2FnZSx0eXBlOm4uY29uc3RydWN0b3IubmFtZX06aT17bWVzc2FnZTpgdW5rbm93biBlcnJvciAtICR7bn1gLHR5cGU6IlVua25vd24ifSxMKHt0YXJnZXQ6Imhvc3QiLHR5cGU6ImNyYXNoIixlcnJvcjppfSl9YnJlYWt9fTtmdW5jdGlvbiBMKGUpe3Bvc3RNZXNzYWdlKGUpfWFzeW5jIGZ1bmN0aW9uIGZ0KGUsdCxuKXtyZXR1cm4geC5zdGFydChmZXRjaChlKSxuZXcgWCh7Li4ubixzdGRvdXQ6b3Qsc3RkZXJyOmN0LHN0ZGluOmk9Pmh0KGksdCksZGVidWc6RXR9KSl9ZnVuY3Rpb24gb3QoZSl7TCh7dGFyZ2V0OiJob3N0Iix0eXBlOiJzdGRvdXQiLHRleHQ6ZX0pfWZ1bmN0aW9uIGN0KGUpe0woe3RhcmdldDoiaG9zdCIsdHlwZToic3RkZXJyIix0ZXh0OmV9KX1mdW5jdGlvbiBFdChlLHQsbixpKXtyZXR1cm4gaT1KU09OLnBhcnNlKEpTT04uc3RyaW5naWZ5KGkpKSxMKHt0YXJnZXQ6Imhvc3QiLHR5cGU6ImRlYnVnIixuYW1lOmUsYXJnczp0LHJldDpuLGRhdGE6aX0pLG59ZnVuY3Rpb24gaHQoZSx0KXtBdG9taWNzLndhaXQobmV3IEludDMyQXJyYXkodCksMCwwKTtjb25zdCBuPW5ldyBEYXRhVmlldyh0KSxpPW4uZ2V0SW50MzIoMCk7aWYoaTwwKXJldHVybiBuLnNldEludDMyKDAsMCksbnVsbDtjb25zdCByPW5ldyBVaW50OEFycmF5KHQsNCxpKSxhPW5ldyBUZXh0RGVjb2RlcigpLmRlY29kZShyLnNsaWNlKDAsZSkpLGY9ci5zbGljZShlLHIubGVuZ3RoKTtyZXR1cm4gbi5zZXRJbnQzMigwLGYuYnl0ZUxlbmd0aCksci5zZXQoZiksYX19KSgpOwo=", j = typeof window < "u" && window.Blob && new Blob([atob(tt)], { type: "text/javascript;charset=utf-8" });
function ut() {
  let e;
  try {
    if (e = j && (window.URL || window.webkitURL).createObjectURL(j), !e)
      throw "";
    return new Worker(e);
  } catch {
    return new Worker("data:application/javascript;base64," + tt);
  } finally {
    e && (window.URL || window.webkitURL).revokeObjectURL(e);
  }
}
function bt(e, t) {
  e.postMessage(t);
}
class Rt extends Error {
}
class Gt {
  constructor(t, i) {
    r(this, "binaryURL");
    // 8kb should be big enough
    r(this, "stdinBuffer", new SharedArrayBuffer(8 * 1024));
    r(this, "context");
    r(this, "result");
    r(this, "worker");
    r(this, "reject");
    this.binaryURL = t, this.context = i;
  }
  async start() {
    if (this.result)
      throw new Error("WASIWorker Host can only be started once");
    return this.result = new Promise((t, i) => {
      this.reject = i, this.worker = new ut(), this.worker.addEventListener("message", (n) => {
        var l, d, V, a, Z, h;
        const s = n.data;
        switch (s.type) {
          case "stdout":
            (d = (l = this.context).stdout) == null || d.call(l, s.text);
            break;
          case "stderr":
            (a = (V = this.context).stderr) == null || a.call(V, s.text);
            break;
          case "debug":
            (h = (Z = this.context).debug) == null || h.call(Z, s.name, s.args, s.ret, s.data);
            break;
          case "result":
            t(s.result);
            break;
          case "crash":
            i(s.error);
            break;
        }
      }), bt(this.worker, {
        target: "client",
        type: "start",
        binaryURL: this.binaryURL,
        stdinBuffer: this.stdinBuffer,
        // Unfortunately can't just splat these because it includes types
        // that can't be sent as a message.
        args: this.context.args,
        env: this.context.env,
        fs: this.context.fs,
        isTTY: this.context.isTTY
      });
    }), this.result;
  }
  kill() {
    var t;
    if (!this.worker)
      throw new Error("WASIWorker has not started");
    this.worker.terminate(), (t = this.reject) == null || t.call(this, new Rt("WASI Worker was killed"));
  }
  async pushStdin(t) {
    const i = new DataView(this.stdinBuffer);
    for (; i.getInt32(0) !== 0; )
      await new Promise((l) => setTimeout(l, 0));
    const n = new TextEncoder().encode(t);
    new Uint8Array(this.stdinBuffer, 4).set(n), i.setInt32(0, n.byteLength), Atomics.notify(new Int32Array(this.stdinBuffer), 0);
  }
  async pushEOF() {
    const t = new DataView(this.stdinBuffer);
    for (; t.getInt32(0) !== 0; )
      await new Promise((i) => setTimeout(i, 0));
    t.setInt32(0, -1), Atomics.notify(new Int32Array(this.stdinBuffer), 0);
  }
}
export {
  q as WASI,
  at as WASIContext,
  Xt as WASISnapshotPreview1,
  Gt as WASIWorkerHost,
  Rt as WASIWorkerHostKilledError
};
