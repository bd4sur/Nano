////////////////////////////////////////////////////////////////////////////
//  Native Library : Net
//  设计说明见 am_native_Net.h。
//
//  实现要点（阅读顺序建议与代码一致）：
//    1) 句柄表：静态槽位 + **单调递增 id**（id 永不复用，旧 id 不会命中新请求）；
//    2) 行缓冲：每请求一个字节环形队列，凑满一行才交出去（UTF-8 安全，见头文件）；
//    3) 每拍只做一件小事：(Net.line) 一次最多从 hal_net 拉 1KB，绝不循环到流结束
//       ——这是"不阻塞 UI"的硬约束，不是性能优化；
//    4) 诊断日志：请求发起/结束时各打一行，浏览器 console 里能直接看到
//       （与 ui_llm.c 的 [llm_rm] 日志同一体例，e2e 排查靠它）。
////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "am_native_Net.h"
#include "am_host.h"
#include "hal_net.h"

// ---------------------------------------------------------------------------
// 请求槽位
// ---------------------------------------------------------------------------

typedef struct {
    int32_t  id;                    // 0 = 空闲；否则为 (Net.* id) 用的句柄
    Net_Http_Request *req;          // hal_net 句柄（NULL = 无）
    uint8_t  buf[AM_NATIVE_NET_LINE_MAX]; // 行缓冲（字节队列，未消费部分在 [0, len)）
    size_t   len;                   // 缓冲内字节数
    int32_t  final_flushed;         // 流结束后残余尾巴是否已交出
    int32_t  logged_done;           // 结束日志只打一次
} Net_Slot;

static Net_Slot s_slots[AM_NATIVE_NET_MAX_REQ];
static int32_t  s_next_id = 1;

static Net_Slot *net_find(int32_t id) {
    if (id <= 0) return NULL;
    for (int32_t i = 0; i < AM_NATIVE_NET_MAX_REQ; i++) {
        if (s_slots[i].id == id) return &s_slots[i];
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// 操作数栈小工具
// ---------------------------------------------------------------------------

static int32_t net_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t net_push_int(am_process_t *proc, int32_t x) {
    return net_push(proc, am_make_value_of_int((am_int_t)x));
}

static int32_t net_push_null(am_process_t *proc) {
    return net_push(proc, AM_VALUE_NULL);
}

static int32_t net_push_undefined(am_process_t *proc) {
    return net_push(proc, AM_VALUE_UNDEFINED);
}

static int32_t net_push_bool(am_process_t *proc, int32_t b) {
    return net_push(proc, am_make_value_of_boolean(b ? 1 : 0));
}

// 由字节串构造 Scheme 字符串并压栈（失败压 #null）。
// 入参必须是合法的 UTF-8 且不以 NUL 结尾（len 显式给出）。
static int32_t net_push_utf8(am_process_t *proc, const char *utf8, size_t len) {
    // 最坏情况：每个字节一个码点（am_mbstowcs 遇残缺序列会停止，此处已保证完整）
    wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (len + 1) * sizeof(wchar_t));
    if (wbuf == NULL) return net_push_null(proc);

    uint32_t wlen = am_mbstowcs(wbuf, utf8, (uint32_t)len);
    am_handle_t hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
    am_free(proc->heap_alloc, wbuf);
    if (hd == AM_HANDLE_NULL) return net_push_null(proc);
    return net_push(proc, am_make_value_of_handle(hd));
}

// 弹出一个**列表**对象（不复制，仅返回对象指针）
static bool net_pop_list(am_process_t *proc, am_list_t **out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return false;

    *out = (am_list_t *)obj;
    return true;
}

// 弹出一个字符串，转成 UTF-8 写入 buf（最多 cap-1 字节 + NUL）。
// 返回实际字节数；参数非法或超长返回 -1（调用方按软失败处理）。
static int32_t net_pop_utf8(am_process_t *proc, char *buf, size_t cap) {
    am_value_t v = am_process_pop_operand(proc);
    buf[0] = '\0';
    if (v == (am_value_t)UINTPTR_MAX) return -1;
    if (!am_value_is_handle(v)) return -1;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return -1;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return -1;

    am_wstring_t *ws = (am_wstring_t *)obj;
    // 先判容量（UTF-8 最坏 4 字节/码点）：宁可按软失败处理，也不静默截断 URL
    if (ws->length * 4 + 1 > cap) return -1;

    // am_wcstombs 直接按 UTF-32 → UTF-8 转换（与 am_native_Chat 的 chat_pop_utf8 同一做法）
    wchar_t wbuf[1024];
    size_t n = ws->length;
    if (n > sizeof(wbuf) / sizeof(wbuf[0]) - 1) n = sizeof(wbuf) / sizeof(wbuf[0]) - 1;
    for (size_t i = 0; i < n; i++) wbuf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    wbuf[n] = L'\0';
    uint32_t wrote = am_wcstombs(buf, wbuf, (uint32_t)cap);
    return (int32_t)wrote;
}

// 弹出数值 id（无效/越界 → 0）
static int32_t net_pop_id(am_process_t *proc) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return 0;
    int64_t x = 0;
    if (am_value_is_int(v))        x = (int64_t)am_value_to_int(v);
    else if (am_value_is_uint(v))  x = (int64_t)am_value_to_uint(v);
    else if (am_value_is_float(v)) x = (int64_t)am_value_to_float(v);
    else return 0;
    if (x < 0 || x > 2147483647) return 0;
    return (int32_t)x;
}

// ---------------------------------------------------------------------------
// 行缓冲：把已到达的字节追加进 slot，并按需切出一行
// ---------------------------------------------------------------------------

// 剔除尾部不完整的多字节 UTF-8 序列（只可能出现在"流被截断"的残余尾巴上）
static void net_trim_incomplete_utf8(uint8_t *buf, size_t *len) {
    size_t n = *len;
    size_t back = 0;
    while (n > 0 && back < 4 && (buf[n - 1] & 0xC0) == 0x80) { n--; back++; }
    if (n == 0) { *len = 0; return; }
    uint8_t lead = buf[n - 1];
    size_t need = (lead < 0x80) ? 1 : (lead >= 0xF0) ? 4 : (lead >= 0xE0) ? 3 : (lead >= 0xC0) ? 2 : 1;
    // 已有 (back+1) 字节（前导字节 + 已读到的续字节）；不够 need 就整段丢掉
    if (back + 1 < need) *len = n - 1;
}

// 从 slot 缓冲里取出一条完整行（不含 CR/LF），写入 line（NUL 结尾）。
// 返回 1 = 取到；0 = 暂无完整行且流未结束；-1 = 流已结束且缓冲为空。
//
// 【顺序铁律】先取缓冲 → 再读 hal_net → **最后**才看“流是否结束”。
//   早期版本把“检查结束”放在“读 hal_net”之前，于是流一结束就再也不去读环形缓冲里
//   已经收下的数据——表现为“模型最后一段正文消失”（实测：59 行 SSE 只交出了 46 行，
//   恰好丢掉收尾那几行，而 content 往往就在那里）。
static int32_t net_pop_line(Net_Slot *s, char *line, size_t line_cap) {
    for (;;) {
        // ① 缓冲里已有换行：切一行出来
        uint8_t *nl = (uint8_t *)memchr(s->buf, '\n', s->len);
        if (nl != NULL) {
            size_t n = (size_t)(nl - s->buf);
            size_t take = n;
            if (take > 0 && s->buf[take - 1] == '\r') take--; // 兼容 CRLF
            if (take > line_cap - 1) take = line_cap - 1;
            memcpy(line, s->buf, take);
            line[take] = '\0';
            size_t consumed = n + 1;
            memmove(s->buf, s->buf + consumed, s->len - consumed);
            s->len -= consumed;
            return 1;
        }

        // ② 缓冲满且没有换行：强制交出一整段（不丢数据，只是把长行拆段）
        if (s->len >= AM_NATIVE_NET_LINE_MAX) {
            size_t take = s->len;
            if (take > line_cap - 1) take = line_cap - 1;
            memcpy(line, s->buf, take);
            line[take] = '\0';
            memmove(s->buf, s->buf + take, s->len - take);
            s->len -= take;
            return 1;
        }

        // ③ 再拉一块（一次最多 1KB；本过程每拍只做这一件小事）
        size_t room = AM_NATIVE_NET_LINE_MAX - s->len;
        if (room > 1024) room = 1024;
        if (room > 0) {
            int32_t n = net_http_stream_read(s->req, s->buf + s->len, room);
            if (n > 0) {
                s->len += (size_t)n;
                continue; // 回到 ①：可能已经凑出行了
            }
            if (n < 0) {
                // 网络层出错：先把缓冲里剩下的字节交出去一次（可能还有没取走的行/尾巴）
                if (s->len > 0 && s->final_flushed == 0) {
                    s->final_flushed = 1;
                    net_trim_incomplete_utf8(s->buf, &s->len);
                    size_t take = s->len;
                    if (take > line_cap - 1) take = line_cap - 1;
                    memcpy(line, s->buf, take);
                    line[take] = '\0';
                    memmove(s->buf, s->buf + take, s->len - take);
                    s->len -= take;
                    if (take > 0) return 1;
                }
                return -1;
            }
        }

        // ④ 读不到新数据了：这时才看流是否已结束
        Net_Stream_State st = net_http_stream_state(s->req);
        if (st == NET_STREAM_DONE || st == NET_STREAM_ERROR) {
            if (s->len > 0 && s->final_flushed == 0) {
                net_trim_incomplete_utf8(s->buf, &s->len);
                size_t take = s->len;
                if (take > line_cap - 1) take = line_cap - 1;
                memcpy(line, s->buf, take);
                line[take] = '\0';
                memmove(s->buf, s->buf + take, s->len - take);
                s->len -= take;
                s->final_flushed = 1;
                if (take == 0) return -1;
                return 1;
            }
            return -1;
        }
        return 0; // 暂无新数据（流仍在进行中）
    }
}

// ---------------------------------------------------------------------------
// 发起请求（post/get 共用）
// ---------------------------------------------------------------------------

// headers 列表 → hal_net 的键值数组。列表元素形如 ("K" "V")；
// 非两元素子列表 / 非字符串项一律跳过（宽进严出，坏数据不致整请求失败）。
static void net_parse_headers(am_process_t *proc, am_list_t *hdrs,
                              char keys[][AM_NATIVE_NET_HKEY_MAX],
                              char vals[][AM_NATIVE_NET_HVAL_MAX],
                              int32_t *num) {
    size_t n = (hdrs != NULL) ? hdrs->length : 0;
    for (size_t i = 0; i < n && *num < NANO_NET_MAX_HEADERS; i++) {
        am_value_t iv = am_list_get(proc->heap_alloc, hdrs, i);
        if (!am_value_is_handle(iv)) continue;
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap,
                                        am_value_to_handle(iv));
        if (!am_value_is_ptr(obj_val)) continue;
        am_object_t *obj = am_value_to_ptr(obj_val);
        if (obj->type != AM_OBJECT_TYPE_LIST) continue;

        am_list_t *pair = (am_list_t *)obj;
        if (pair->length < 2) continue;

        // key
        am_value_t kv = am_list_get(proc->heap_alloc, pair, 0);
        if (!am_value_is_handle(kv)) continue;
        am_value_t kobj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap,
                                         am_value_to_handle(kv));
        if (!am_value_is_ptr(kobj_val)) continue;
        am_object_t *kobj = am_value_to_ptr(kobj_val);
        if (kobj->type != AM_OBJECT_TYPE_WSTRING) continue;
        am_wstring_t *kws = (am_wstring_t *)kobj;

        // value
        am_value_t vv = am_list_get(proc->heap_alloc, pair, 1);
        if (!am_value_is_handle(vv)) continue;
        am_value_t vobj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap,
                                         am_value_to_handle(vv));
        if (!am_value_is_ptr(vobj_val)) continue;
        am_object_t *vobj = am_value_to_ptr(vobj_val);
        if (vobj->type != AM_OBJECT_TYPE_WSTRING) continue;
        am_wstring_t *vws = (am_wstring_t *)vobj;

        wchar_t wbuf[AM_NATIVE_NET_HVAL_MAX];
        size_t kn = kws->length;
        if (kn > AM_NATIVE_NET_HKEY_MAX - 1) kn = AM_NATIVE_NET_HKEY_MAX - 1;
        for (size_t k = 0; k < kn; k++) wbuf[k] = (wchar_t)am_value_to_wchar(kws->content[k]);
        wbuf[kn] = L'\0';
        am_wcstombs(keys[*num], wbuf, AM_NATIVE_NET_HKEY_MAX);

        size_t vn = vws->length;
        if (vn > AM_NATIVE_NET_HVAL_MAX - 1) vn = AM_NATIVE_NET_HVAL_MAX - 1;
        for (size_t k = 0; k < vn; k++) wbuf[k] = (wchar_t)am_value_to_wchar(vws->content[k]);
        wbuf[vn] = L'\0';
        am_wcstombs(vals[*num], wbuf, AM_NATIVE_NET_HVAL_MAX);

        if (keys[*num][0] != '\0') (*num)++;
    }
}

// 真正发起请求：url/body 已是 UTF-8；method 为 "POST"/"GET"
static int32_t net_start(am_process_t *proc, const char *method,
                         const char *url, const char *body, size_t body_len,
                         char *keys, char *vals, int32_t header_num) {
    Net_Http_Request_Config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.url = url;
    cfg.method = method;
    cfg.body = (const uint8_t *)body;
    cfg.body_len = body_len;

    for (int32_t i = 0; i < header_num; i++) {
        cfg.headers[i].key = keys + (size_t)i * AM_NATIVE_NET_HKEY_MAX;
        cfg.headers[i].value = vals + (size_t)i * AM_NATIVE_NET_HVAL_MAX;
    }
    cfg.header_num = header_num;

    printf("[net] %s %s (body %u bytes, %d headers)\n",
           method, url, (unsigned)body_len, (int)header_num);

    Net_Http_Request *req = NULL;
    if (net_http_post_stream(&cfg, &req) != 0 || req == NULL) {
        printf("[net] request start failed\n");
        return net_push_null(proc);
    }

    // 找空槽位
    Net_Slot *slot = NULL;
    for (int32_t i = 0; i < AM_NATIVE_NET_MAX_REQ; i++) {
        if (s_slots[i].id == 0) { slot = &s_slots[i]; break; }
    }
    if (slot == NULL) {
        // 槽位满：关掉刚建好的请求（调用方应先把旧请求 close 掉）
        printf("[net] no free slot (%d in flight), request dropped\n", (int)AM_NATIVE_NET_MAX_REQ);
        net_http_close(req);
        return net_push_null(proc);
    }

    slot->id = s_next_id++;
    if (s_next_id <= 0) s_next_id = 1; // 极端回绕保护
    slot->req = req;
    slot->len = 0;
    slot->final_flushed = 0;
    slot->logged_done = 0;
    return net_push_int(proc, slot->id);
}

int32_t am_native_Net_post(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;

    // 参数自右向左弹（先压先得：body → headers → url）
    am_value_t body_v = am_process_pop_operand(proc);
    am_list_t *hdrs = NULL;
    if (!net_pop_list(proc, &hdrs)) {
        // 仍要把 url 弹掉，保持栈平衡
        (void)am_process_pop_operand(proc);
        return net_push_null(proc);
    }

    char url[AM_NATIVE_NET_URL_MAX];
    if (net_pop_utf8(proc, url, sizeof url) < 0) return net_push_null(proc);

    // 请求体：从已弹出的字符串值里取（这里单独处理，避免再压栈）
    char *body = NULL;
    size_t body_len = 0;
    if (am_value_is_handle(body_v)) {
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap,
                                         am_value_to_handle(body_v));
        if (am_value_is_ptr(obj_val)) {
            am_object_t *obj = am_value_to_ptr(obj_val);
            if (obj->type == AM_OBJECT_TYPE_WSTRING) {
                am_wstring_t *ws = (am_wstring_t *)obj;
                // UTF-8 最坏 4 字节/码点
                size_t cap = ws->length * 4 + 1;
                if (cap > AM_NATIVE_NET_BODY_MAX) {
                    printf("[net] body too large (%u bytes > %d)\n",
                           (unsigned)cap, (int)AM_NATIVE_NET_BODY_MAX);
                    return net_push_null(proc);
                }
                body = (char *)malloc(cap);
                if (body == NULL) return net_push_null(proc);
                wchar_t *wbuf = (wchar_t *)malloc((ws->length + 1) * sizeof(wchar_t));
                if (wbuf == NULL) { free(body); return net_push_null(proc); }
                for (size_t i = 0; i < ws->length; i++) wbuf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
                wbuf[ws->length] = L'\0';
                body_len = (size_t)am_wcstombs(body, wbuf, (uint32_t)cap);
                free(wbuf);
            }
        }
    }

    // 头表（栈上分配：8×64 + 8×384 ≈ 3.6KB，native 调用期间有效即可——
    //   hal_net 的实现都会在 net_http_post_stream 内深拷贝）
    char keys[NANO_NET_MAX_HEADERS][AM_NATIVE_NET_HKEY_MAX];
    char vals[NANO_NET_MAX_HEADERS][AM_NATIVE_NET_HVAL_MAX];
    int32_t hnum = 0;
    net_parse_headers(proc, hdrs, keys, vals, &hnum);

    int32_t rc = net_start(proc, "POST", url, body ? body : "", body_len,
                           (char *)keys, (char *)vals, hnum);
    free(body);
    return rc;
}

int32_t am_native_Net_get(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;

    am_list_t *hdrs = NULL;
    if (!net_pop_list(proc, &hdrs)) {
        (void)am_process_pop_operand(proc); // 弹掉 url
        return net_push_null(proc);
    }

    char url[AM_NATIVE_NET_URL_MAX];
    if (net_pop_utf8(proc, url, sizeof url) < 0) return net_push_null(proc);

    char keys[NANO_NET_MAX_HEADERS][AM_NATIVE_NET_HKEY_MAX];
    char vals[NANO_NET_MAX_HEADERS][AM_NATIVE_NET_HVAL_MAX];
    int32_t hnum = 0;
    net_parse_headers(proc, hdrs, keys, vals, &hnum);

    return net_start(proc, "GET", url, NULL, 0, (char *)keys, (char *)vals, hnum);
}

// ---------------------------------------------------------------------------
// 取数据 / 查状态
// ---------------------------------------------------------------------------

int32_t am_native_Net_line(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t id = net_pop_id(proc);
    Net_Slot *s = net_find(id);
    if (s == NULL) return net_push_undefined(proc);

    // 结束日志（只打一次，便于 e2e 判定"流真的结束了"）
    Net_Stream_State st = net_http_stream_state(s->req);
    if (!s->logged_done && (st == NET_STREAM_DONE || st == NET_STREAM_ERROR)) {
        printf("[net] id=%d finished: state=%d http=%d\n",
               (int)s->id, (int)st, (int)net_http_status_code(s->req));
        s->logged_done = 1;
    }

    char line[AM_NATIVE_NET_LINE_MAX + 1];
    int32_t got = net_pop_line(s, line, sizeof line);
    if (got == 1) return net_push_utf8(proc, line, strlen(line));
    return net_push_undefined(proc);
}

int32_t am_native_Net_state(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t id = net_pop_id(proc);
    Net_Slot *s = net_find(id);
    if (s == NULL) return net_push_int(proc, 0);
    return net_push_int(proc, (int32_t)net_http_stream_state(s->req));
}

int32_t am_native_Net_status(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t id = net_pop_id(proc);
    Net_Slot *s = net_find(id);
    if (s == NULL) return net_push_int(proc, 0);
    return net_push_int(proc, net_http_status_code(s->req));
}

int32_t am_native_Net_done(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t id = net_pop_id(proc);
    Net_Slot *s = net_find(id);
    if (s == NULL) return net_push_bool(proc, 1); // 无效 id 视为"已结束"，避免 App 死等
    Net_Stream_State st = net_http_stream_state(s->req);
    return net_push_bool(proc, (st == NET_STREAM_DONE || st == NET_STREAM_ERROR) ? 1 : 0);
}

int32_t am_native_Net_error(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t id = net_pop_id(proc);
    Net_Slot *s = net_find(id);
    if (s == NULL) return net_push_bool(proc, 1);
    return net_push_bool(proc, net_http_stream_state(s->req) == NET_STREAM_ERROR ? 1 : 0);
}

int32_t am_native_Net_close(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    int32_t id = net_pop_id(proc);
    Net_Slot *s = net_find(id);
    if (s != NULL) {
        if (s->req != NULL) net_http_close(s->req);
        s->req = NULL;
        s->id = 0;
        s->len = 0;
        s->final_flushed = 0;
        s->logged_done = 0;
    }
    return net_push_undefined(proc);
}

int32_t am_native_Net_version(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    static const char *ver = "Net/1";
    return net_push_utf8(proc, ver, strlen(ver));
}

// ---------------------------------------------------------------------------
// 库表
// ---------------------------------------------------------------------------

static const am_native_func_entry_t am_native_Net_funcs[] = {
    { L"post",    am_native_Net_post },
    { L"get",     am_native_Net_get },
    { L"line",    am_native_Net_line },
    { L"state",   am_native_Net_state },
    { L"status",  am_native_Net_status },
    { L"done",    am_native_Net_done },
    { L"error",   am_native_Net_error },
    { L"close",   am_native_Net_close },
    { L"version", am_native_Net_version },
};

const am_native_lib_entry_t am_native_Net_lib = {
    L"Net",
    am_native_Net_funcs,
    sizeof(am_native_Net_funcs) / sizeof(am_native_Net_funcs[0])
};
