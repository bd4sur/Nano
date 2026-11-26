#ifndef __AM_NATIVE_NET_H__
#define __AM_NATIVE_NET_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Net —— 灵机（Animac）网络访问本地库（类 ajax 的 HTTP 客户端）
//
//  为什么需要本库：
//    hal_net.h 已经把"上网"这件事按整机协作式调度模型抽象好了（非阻塞/轮询），
//    但目前只有 C 业务在用（ui_llm.c 的远端模型）。灵机（Scheme）程序要访问网络
//    ——例如调用大模型 API——此前完全没有入口。本库把 hal_net 包成 Scheme 可见的
//    「发起 → 逐行取 → 查状态 → 关闭」四件套，各平台实现由 hal_net_<平台>.c 决定：
//      · WASM（NANO_WASM_GUI）：EM_JS + 浏览器 fetch（http/https 均可，TLS 由浏览器做；
//        受同源策略约束，跨源需服务器给 CORS 头，或走同源反向代理）；
//      · Linux（tty/web 等）：后台 pthread + 阻塞 socket（**仅明文 http**）；
//      · 其余平台：hal_net_stub.c 打桩，一律失败（Scheme 侧按"网络不可用"降级）。
//
//  交互模型（关键设计——「一拍一步」，绝不阻塞）：
//    浏览器主线程上 JS 的 fetch 回调与 WASM 的帧回调天然分时：WASM 把控制权交回
//    事件循环（rAF 帧回调返回）时浏览器才执行网络回调，下一帧 Scheme 再从
//    (Net.line) 里同步取走已到达的数据。因此本库的过程**全部立即返回**：
//
//        (Net.post url headers body)  → 建请求（JS fetch 立即发起），返回请求 id
//        (Net.line id)                → 取出「一条完整行」（没有则 #undefined）
//        (Net.state id)               → 0 无此 id / 1 进行中 / 2 完成 / 3 错误
//        (Net.status id)              → HTTP 状态码（尚未收到响应头时为 0）
//        (Net.done id) / (Net.error id) → 状态谓词
//        (Net.close id)               → 中止并释放（幂等）
//
//    App 的正确写法：在自己的每拍（System.set_interval 回调）里调一两次 (Net.line)
//    解析已到达的行，然后照常绘图返回。**不要在 while 里等**——时间片耗光会冻住
//    界面，而且 JS 的网络回调根本没有机会执行（单线程）。
//
//  为什么是「按行」而不是「按字节流」：
//    1) 大模型 API 的流式响应（SSE）天生按行组织（`data: {...}`），按行交出后
//       Scheme 侧不需要 split/index-of（本解释器的 String 库没有这两个过程）；
//    2) 字符串返回值要走 UTF-8→UTF-32 转换，若按任意字节块交出，被块边界切断的
//       多字节序列会解码成乱码（宿主 am_mbstowcs 遇残缺字节即停止）。按行交出时
//       换行符必然落在字符边界上，天然无此问题；流结束（DONE/ERROR）时的残余尾巴
//       由本库剔除不完整的多字节序列后再交出。
//
//  行缓冲与截断：单行上限 AM_NATIVE_NET_LINE_MAX 字节。超限时**强制按已有内容
//    交出一行**（不丢数据，只是把一长行拆成几段），Scheme 侧按需处理。
//
//  句柄：同一时刻最多 AM_NATIVE_NET_MAX_REQ 个请求；id 单调递增永不复用，
//    因此"已关闭的旧 id"不会被误认成新请求。App 退出时务必 (Net.close …)，
//    否则浏览器里那个 fetch 会一直跑到结束。
//
//  Scheme 侧用法：
//      (native Net)
//      (define id (Net.post "http://host:11050/v1/chat/completions"
//                           (quote (("Content-Type" "application/json")
//                                   ("Authorization" "Bearer <key>")))
//                           "{\"model\":\"m\",\"stream\":true,\"messages\":[...]}"))
//      (define ln (Net.line id))       ; 每拍一次；#undefined = 暂时没有新行
//      (if (Net.done id) (Net.close id))
//
//  配套的应用库：app/lib/net.scm（JSON 字符串转义、SSE delta 提取、JSON 字段取值），
//    配套的示例 App：app/remotechat.scm（远端大模型对话）。
//
//  栈纪律（上游本地库契约 N1/N2/N6）：每个过程弹完参数后恰好压 1 个值并 step 一次；
//    参数类型不符 / 请求 id 无效按**软失败**处理（返回 #null / #undefined / 0，
//    不抛运行时错误）——网络失败不应杀死 App。
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Net_lib;

// 并发请求上限（超出后 (Net.post) 软失败返回 #null）
#define AM_NATIVE_NET_MAX_REQ (4)

// 单行缓冲上限（字节；超限即强制交出一行）
#define AM_NATIVE_NET_LINE_MAX (8192)

// URL 上限（字节）
#define AM_NATIVE_NET_URL_MAX (512)

// 单个请求头的 key/value 上限（字节）
#define AM_NATIVE_NET_HKEY_MAX (64)
#define AM_NATIVE_NET_HVAL_MAX (384)

// 单次请求体上限（字节；超出 (Net.post) 软失败，防止误传巨大字符串把池撑爆）
#define AM_NATIVE_NET_BODY_MAX (64 * 1024)

// ---------------------------------------------------------------------------
// Scheme 可见过程（参数经虚拟机操作数栈传递，顺序为「先压先得」）
// ---------------------------------------------------------------------------

// (Net.post url:String headers:List body:String) : Number | #null —— 发起 POST
int32_t am_native_Net_post(am_runtime_t *rt, am_process_t *proc);

// (Net.get url:String headers:List) : Number | #null —— 发起 GET（无请求体）
int32_t am_native_Net_get(am_runtime_t *rt, am_process_t *proc);

// (Net.line id:Number) : String | #undefined —— 取一条完整行（无新数据则 #undefined）
int32_t am_native_Net_line(am_runtime_t *rt, am_process_t *proc);

// (Net.state id:Number) : Number —— 0 无此 id / 1 进行中 / 2 完成 / 3 错误
int32_t am_native_Net_state(am_runtime_t *rt, am_process_t *proc);

// (Net.status id:Number) : Number —— HTTP 状态码（未知为 0）
int32_t am_native_Net_status(am_runtime_t *rt, am_process_t *proc);

// (Net.done id:Number) : Boolean —— 已完成或已失败
int32_t am_native_Net_done(am_runtime_t *rt, am_process_t *proc);

// (Net.error id:Number) : Boolean —— 网络层失败（连接失败/中断/被中止）
int32_t am_native_Net_error(am_runtime_t *rt, am_process_t *proc);

// (Net.close id:Number) : #undefined —— 中止并释放（幂等；无效 id 静默忽略）
int32_t am_native_Net_close(am_runtime_t *rt, am_process_t *proc);

// (Net.version) : String —— 本地库版本串（"Net/1"），供 App 自检/排查
int32_t am_native_Net_version(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif
