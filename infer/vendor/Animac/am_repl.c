#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <unistd.h>
#include <limits.h>

#include "am_repl.h"
#include "am_host.h"

#include "am_native_System.h"
#include "am_native_Math.h"
#include "am_native_String.h"
#include "am_native_LLM.h"
#include "am_native_Table.h"

// 前向声明
static void repl_ctx_output_wchar(am_repl_ctx_t *ctx, wchar_t c);
static void repl_ctx_output_wcs(am_repl_ctx_t *ctx, const wchar_t *s);
static void repl_ctx_error_wchar(am_repl_ctx_t *ctx, wchar_t c);
static void repl_ctx_error_wcs(am_repl_ctx_t *ctx, const wchar_t *s);
static void repl_ctx_clear_output(am_repl_ctx_t *ctx);
static int repl_ctx_accum_append(am_repl_ctx_t *ctx, const wchar_t *line);
static void repl_ctx_reset_accum(am_repl_ctx_t *ctx);
static int repl_ctx_session_append(am_repl_ctx_t *ctx, const wchar_t *code, size_t len);
static am_repl_result_t repl_result_from_ctx(am_repl_ctx_t *ctx);
static int32_t repl_eval_session(am_repl_ctx_t *ctx, const wchar_t *session_code,
                                  am_module_t **out_mod);
static am_module_t *repl_create_initial_module(am_repl_ctx_t *ctx);
static int repl_ctx_js_indent(const wchar_t *code);
static am_repl_result_t repl_ctx_eval_js_accum(am_repl_ctx_t *ctx, int force);
static am_repl_result_t repl_ctx_submit(am_repl_ctx_t *ctx);
static int repl_ctx_reset(am_repl_ctx_t *ctx);
static wchar_t *repl_session_strip_display(const wchar_t *session);
static int repl_ctx_init_runtime(am_repl_ctx_t *ctx);

// 运行时回调：捕获输出到当前上下文的输出缓冲区。
static void on_tick(am_runtime_t *rt) {
    if (!rt || !rt->output_fifo || !rt->error_fifo) return;
    am_repl_ctx_t *ctx = (am_repl_ctx_t *)am_get_runtime_host_context(rt);
    if (!ctx) return;
    while (rt->output_fifo->length > 0) {
        am_value_t v = am_list_shift(rt->vm_alloc, rt->output_fifo);
        if (am_value_is_wchar(v)) {
            repl_ctx_output_wchar(ctx, (wchar_t)am_value_to_wchar(v));
        }
    }
    while (rt->error_fifo->length > 0) {
        am_value_t v = am_list_shift(rt->vm_alloc, rt->error_fifo);
        if (am_value_is_wchar(v)) {
            repl_ctx_error_wchar(ctx, (wchar_t)am_value_to_wchar(v));
        }
    }
}

static void on_halt(am_runtime_t *rt) { (void)rt; }

static void on_error(am_runtime_t *rt) {
    if (!rt) return;
    am_repl_ctx_t *ctx = (am_repl_ctx_t *)am_get_runtime_host_context(rt);
    if (ctx) ctx->runtime_error = 1;
}

// 宿主时间函数适配：实现 runtime vtable 要求的带 rt 引数签名
static am_timestamp_t host_now_ms(am_runtime_t *rt) {
    (void)rt;
    return (am_timestamp_t)am_current_timestamp_in_ms();
}

static void host_sleep_in_ms(am_runtime_t *rt, am_timestamp_t ms) {
    (void)rt;
    am_sleep_in_ms((uint64_t)ms);
}

// 宿主虚函数表：向 runtime 注入事件回调与时间/睡眠实现
static const am_runtime_vtable_t g_host_vtable = {
    .on_tick = on_tick,
    .on_event = NULL,
    .on_halt = on_halt,
    .on_error = on_error,
    .sleep_in_ms = host_sleep_in_ms,
    .now_ms = host_now_ms,
};

static wchar_t *mb_to_wchar(const char *src) {
    if (!src) return NULL;
    // 使用项目统一的 UTF-8 -> UTF-32 转换，不依赖当前 C locale。
    // 每个 UTF-8 字节最多对应一个 wchar_t，因此 strlen(src) + 1 足够容纳。
    size_t src_len = strlen(src);
    wchar_t *dst = (wchar_t *)malloc((src_len + 1) * sizeof(wchar_t));
    if (!dst) return NULL;
    am_mbstowcs(dst, src, (uint32_t)src_len);
    return dst;
}

static char *wchar_to_mb(const wchar_t *src) {
    if (!src) return NULL;
    // 使用项目统一的 UTF-32 -> UTF-8 转换，不依赖当前 C locale。
    // 每个 UTF-32 码点最多需要 4 字节 UTF-8 表示。
    size_t src_len = wcslen(src);
    size_t buf_size = src_len * 4 + 1;
    char *dst = (char *)malloc(buf_size);
    if (!dst) return NULL;
    am_wcstombs(dst, src, (uint32_t)buf_size);
    return dst;
}

// 获取当前工作目录并转换为 wchar_t*（调用者负责 free）。失败返回 NULL。
static wchar_t *repl_get_cwd(void) {
    char cwd_mb[PATH_MAX];
    if (!getcwd(cwd_mb, sizeof(cwd_mb))) return NULL;
    size_t len = strlen(cwd_mb);
    wchar_t *cwd_w = (wchar_t *)malloc((len + 1) * sizeof(wchar_t));
    if (!cwd_w) return NULL;
    am_mbstowcs(cwd_w, cwd_mb, (uint32_t)(len + 1));
    return cwd_w;
}

// 检查代码是否构成完整的 S-表达式。
// 返回值： 0 完整；1 未完成（需要更多输入）；-1 括号不匹配等明显错误。
// 若 out_indent 非空，则写入建议的续行缩进层级（未闭合的左括号数）。
static int check_expression_complete(const wchar_t *code, int *out_indent) {
    int depth = 0;
    int in_string = 0;
    int escape = 0;
    int line_comment = 0;

    for (size_t i = 0; code[i] != L'\0'; i++) {
        wchar_t c = code[i];

        if (line_comment) {
            if (c == L'\n') line_comment = 0;
            continue;
        }

        if (in_string) {
            if (escape) {
                escape = 0;
                continue;
            }
            if (c == L'\\') {
                escape = 1;
                continue;
            }
            if (c == L'"') {
                in_string = 0;
            }
            continue;
        }

        if (c == L';') {
            line_comment = 1;
            continue;
        }
        if (c == L'"') {
            in_string = 1;
            continue;
        }

        if (c == L'(' || c == L'[' || c == L'{') {
            depth++;
        } else if (c == L')' || c == L']' || c == L'}') {
            depth--;
            if (depth < 0) return -1;
        }
    }

    if (in_string) return 1;
    if (depth > 0) {
        if (out_indent) *out_indent = depth;
        return 1;
    }
    if (out_indent) *out_indent = 0;
    return 0;
}

// 构造 eval 代码的临时绝对路径：base_dir/__repl__.scm
static wchar_t *repl_make_path(am_allocator_t *alloc, const wchar_t *base_dir) {
    if (!alloc) return NULL;
    const wchar_t *filename = L"__repl__.scm";
    size_t base_len = base_dir ? wcslen(base_dir) : 0;
    size_t file_len = wcslen(filename);
    int need_sep = (base_len > 0 && base_dir[base_len - 1] != L'/' && base_dir[base_len - 1] != L'\\');
    wchar_t *path = (wchar_t *)am_malloc(alloc,
                                          (base_len + (need_sep ? 1 : 0) + file_len + 1) * sizeof(wchar_t));
    if (!path) return NULL;
    if (base_len > 0) wcscpy(path, base_dir);
    if (need_sep) path[base_len] = L'/';
    wcscpy(path + base_len + (need_sep ? 1 : 0), filename);
    return path;
}


// 判断 varid 名字是否对应有副作用的内置函数或本地宿主函数。
static int repl_is_side_effect_varid(const wchar_t *word) {
    if (!word) return 0;

    // 有副作用的全局 builtin（对应 runtime 中的 OP）
    if (wcscmp(word, L"display") == 0) return 1;
    if (wcscmp(word, L"newline") == 0) return 1;
    if (wcscmp(word, L"write") == 0) return 1;
    if (wcscmp(word, L"read") == 0) return 1;
    if (wcscmp(word, L"set_item!") == 0) return 1;
    if (wcscmp(word, L"push") == 0) return 1;
    if (wcscmp(word, L"pop") == 0) return 1;
    if (wcscmp(word, L"fork") == 0) return 1;

    // 有副作用的本地宿主函数（按库分类）
    if (wcscmp(word, L"System.exit") == 0) return 1;
    if (wcscmp(word, L"System.kill") == 0) return 1;
    if (wcscmp(word, L"System.exec") == 0) return 1;
    if (wcscmp(word, L"System.set_timeout") == 0) return 1;
    if (wcscmp(word, L"System.set_interval") == 0) return 1;
    if (wcscmp(word, L"System.clear_timeout") == 0) return 1;
    if (wcscmp(word, L"System.clear_interval") == 0) return 1;
    if (wcscmp(word, L"System.fork") == 0) return 1;
    if (wcscmp(word, L"System.eval") == 0) return 1;
    if (wcscmp(word, L"System.make_queue") == 0) return 1;
    if (wcscmp(word, L"System.write") == 0) return 1;
    if (wcscmp(word, L"System.read") == 0) return 1;
    if (wcscmp(word, L"Table.set") == 0) return 1;
    if (wcscmp(word, L"Table.delete") == 0) return 1;
    if (wcscmp(word, L"LLM.init") == 0) return 1;

    return 0;
}

// 判断节点是否是 define / set! / native / import 调用，或是有副作用的 builtin / native 调用。
// 这些表达式不再额外包装 display。
static int repl_last_expr_has_side_effect(am_ast_t *ast, am_handle_t node_handle) {
    if (!ast || node_handle == AM_HANDLE_NULL) return 0;
    am_value_t node_val = am_ast_get_node(ast, node_handle);
    if (!am_value_is_ptr(node_val)) return 0;
    am_object_t *obj = am_value_to_ptr(node_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return 0;
    am_list_t *lst = (am_list_t *)obj;
    if (lst->length == 0) return 0;
    am_value_t first = am_list_get(ast->alloc, lst, 0);

    if (am_value_is_symbol(first)) {
        if (first == AM_VALUE_KW_define ||
            first == AM_VALUE_KW_set ||
            first == AM_VALUE_KW_native ||
            first == AM_VALUE_KW_import) {
            return 1;
        }
        // (begin ...) 的副作用取决于最后一个子表达式
        if (first == AM_VALUE_KW_begin && lst->length > 1) {
            am_value_t last = am_list_get(ast->alloc, lst, lst->length - 1);
            if (am_value_is_handle(last)) {
                return repl_last_expr_has_side_effect(ast, am_value_to_handle(last));
            }
        }
    }
    if (am_value_is_varid(first)) {
        am_varid_t varid = am_value_to_varid(first);
        wchar_t *word = am_vocab_get(ast->alloc, ast->var_vocab, &varid);
        if (word) {
            return repl_is_side_effect_varid(word);
        }
    }
    return 0;
}

// 若用户输入的最后一个顶层表达式不是定义/赋值/输出，则将其包装为 (display <expr>) (newline)。
// 返回新分配的 wchar_t* 代码（调用者负责释放），失败返回 NULL。
// 关键：直接从原始 session_code 切分最后一个表达式，避免 ARN 后的内部变量名污染。
static wchar_t *repl_build_session_code(am_repl_ctx_t *ctx, const wchar_t *session_code) {
    if (!ctx || !session_code) return NULL;
    am_allocator_t *alloc = ctx->vm_alloc;

    const wchar_t *prefix = L"((lambda () \n";
    const wchar_t *suffix = L"\n))";
    size_t prefix_len = wcslen(prefix);
    size_t suffix_len = wcslen(suffix);
    size_t session_len = wcslen(session_code);
    size_t wrapped_len = prefix_len + session_len + suffix_len;

    wchar_t *wrapped_code = (wchar_t *)am_malloc(alloc, (wrapped_len + 1) * sizeof(wchar_t));
    if (!wrapped_code) return NULL;
    wcscpy(wrapped_code, prefix);
    wcscat(wrapped_code, session_code);
    wcscat(wrapped_code, suffix);

    wchar_t *path_buf = repl_make_path(alloc, L".");
    am_ast_t *ast = am_parse(alloc, wrapped_code, path_buf, 0);
    am_free(alloc, path_buf);
    if (!ast) {
        am_free(alloc, wrapped_code);
        return NULL;
    }

    am_value_t *bodies = am_ast_get_global_nodes(ast);
    size_t n_body = 0;
    if (bodies) {
        am_value_t lambda_val = am_ast_get_node(ast, ast->top_lambda_handle);
        if (am_value_is_ptr(lambda_val)) {
            am_list_t *lambda = (am_list_t *)am_value_to_ptr(lambda_val);
            n_body = am_list_lambda_get_body_number(ast->alloc, lambda);
        }
    }

    if (n_body == 0 || repl_last_expr_has_side_effect(ast, am_value_to_handle(bodies[n_body - 1]))) {
        if (bodies) free(bodies);
        am_ast_destroy(ast);
        return wrapped_code; // 无需包装，调用者负责释放
    }

    am_handle_t last_handle = am_value_to_handle(bodies[n_body - 1]);
    size_t last_token_idx = am_ast_get_node_token_index(ast, last_handle);
    size_t char_index = 0;
    if (last_token_idx != SIZE_MAX && last_token_idx < ast->token_count) {
        char_index = ast->tokens[last_token_idx].index;
    }

    if (bodies) free(bodies);
    am_ast_destroy(ast);

    // char_index 是最后一个表达式在 wrapped_code 中的起始字符偏移。
    // 转换到 session_code 中的起始偏移。
    size_t session_start = 0;
    if (last_token_idx != SIZE_MAX && char_index >= prefix_len) {
        session_start = char_index - prefix_len;
    } else {
        // 对于没有 token 映射的原子表达式（如单独的数字/字符串），回退到整个 session_code
        char_index = prefix_len;
    }
    // 最后一个表达式在 session_code 中的结束偏移，忽略尾部空白。
    size_t session_end = session_len;
    while (session_end > session_start &&
           (session_code[session_end - 1] == L'\n' ||
            session_code[session_end - 1] == L'\r' ||
            session_code[session_end - 1] == L' ' ||
            session_code[session_end - 1] == L'\t')) {
        session_end--;
    }
    size_t last_expr_len = (session_end > session_start) ? (session_end - session_start) : 0;

    size_t new_len = char_index + wcslen(L"\n(display ") + last_expr_len + wcslen(L")\n(newline)") + suffix_len + 8;
    wchar_t *new_code = (wchar_t *)am_malloc(alloc, (new_len + 1) * sizeof(wchar_t));
    if (!new_code) {
        am_free(alloc, wrapped_code);
        return NULL;
    }

    wcsncpy(new_code, wrapped_code, char_index);
    new_code[char_index] = L'\0';
    wcscat(new_code, L"\n(display ");
    wcsncat(new_code, session_code + session_start, last_expr_len);
    wcscat(new_code, L")\n(newline)");
    wcscat(new_code, suffix);

    am_free(alloc, wrapped_code);
    return new_code;
}

// 编译完整的 REPL 会话代码。成功返回 0 并通过 out_mod 输出模块，失败返回 -1。
// 注意：本函数不创建进程，进程由调用者在合适的时机通过 am_runtime_load_module 创建。
static int32_t repl_eval_session(am_repl_ctx_t *ctx, const wchar_t *session_code,
                                  am_module_t **out_mod) {
    if (!ctx || !session_code || !out_mod) return -1;

    wchar_t *display_wrapped = repl_build_session_code(ctx, session_code);
    if (!display_wrapped) {
        repl_ctx_error_wcs(ctx, L"[REPL] 语法解析失败\n");
        return -1;
    }

    const wchar_t *base_dir = ctx->base_dir ? ctx->base_dir : L".";
    wchar_t *path_buf = repl_make_path(ctx->vm_alloc, base_dir);
    if (!path_buf) {
        am_free(ctx->vm_alloc, display_wrapped);
        return -1;
    }

    am_ast_t *ast = am_parse(ctx->vm_alloc, display_wrapped, path_buf, 0);
    am_free(ctx->vm_alloc, display_wrapped);
    if (!ast) {
        am_free(ctx->vm_alloc, path_buf);
        repl_ctx_error_wcs(ctx, L"[REPL] 语法解析失败\n");
        return -1;
    }

    am_ast_t *linked = am_link(ast, (wchar_t *)base_dir, am_host_read_source_from_file, NULL);
    if (!linked) {
        am_ast_destroy(ast);
        am_free(ctx->vm_alloc, path_buf);
        repl_ctx_error_wcs(ctx, L"[REPL] 模块链接失败\n");
        return -1;
    }

    am_module_t *mod = am_compile(linked, 0, 0);
    am_free(ctx->vm_alloc, path_buf);
    if (!mod) {
        am_ast_destroy(linked);
        repl_ctx_error_wcs(ctx, L"[REPL] 编译失败\n");
        return -1;
    }

    *out_mod = mod;
    return 0;
}


static am_module_t *repl_create_initial_module(am_repl_ctx_t *ctx) {
    if (!ctx) return NULL;
    const wchar_t *init_code = L"((lambda () (begin)))";
    const wchar_t *base_dir = ctx->base_dir ? ctx->base_dir : L".";
    wchar_t *path_buf = repl_make_path(ctx->vm_alloc, base_dir);
    if (!path_buf) return NULL;

    am_ast_t *ast = am_parse(ctx->vm_alloc, (wchar_t *)init_code, path_buf, 0);
    if (!ast) {
        am_free(ctx->vm_alloc, path_buf);
        return NULL;
    }

    am_ast_t *linked = am_link(ast, (wchar_t *)base_dir, am_host_read_source_from_file, NULL);
    if (!linked) {
        am_ast_destroy(ast);
        am_free(ctx->vm_alloc, path_buf);
        return NULL;
    }

    am_module_t *mod = am_compile(linked, 0, 0);
    am_free(ctx->vm_alloc, path_buf);
    if (!mod) {
        am_ast_destroy(linked);
        return NULL;
    }

    return mod;
}

// 输出缓冲区操作
static void repl_ctx_clear_output(am_repl_ctx_t *ctx) {
    if (!ctx) return;
    ctx->stdout_len = 0;
    if (ctx->stdout_buf) ctx->stdout_buf[0] = '\0';
    ctx->output = NULL;
    ctx->stderr_len = 0;
    if (ctx->stderr_buf) ctx->stderr_buf[0] = '\0';
    ctx->error = NULL;
}

static void repl_ctx_output_mb(am_repl_ctx_t *ctx, const char *s) {
    if (!ctx || !s) return;
    size_t len = strlen(s);
    size_t need = ctx->stdout_len + len + 1;
    if (need > ctx->stdout_cap) {
        size_t new_cap = ctx->stdout_cap ? ctx->stdout_cap * 2 : 256;
        while (new_cap < need) new_cap *= 2;
        char *new_buf = (char *)realloc(ctx->stdout_buf, new_cap);
        if (!new_buf) return;
        ctx->stdout_buf = new_buf;
        ctx->stdout_cap = new_cap;
    }
    memcpy(ctx->stdout_buf + ctx->stdout_len, s, len);
    ctx->stdout_len += len;
    ctx->stdout_buf[ctx->stdout_len] = '\0';
    ctx->output = ctx->stdout_buf;
}

static void repl_ctx_error_mb(am_repl_ctx_t *ctx, const char *s) {
    if (!ctx || !s) return;
    size_t len = strlen(s);
    size_t need = ctx->stderr_len + len + 1;
    if (need > ctx->stderr_cap) {
        size_t new_cap = ctx->stderr_cap ? ctx->stderr_cap * 2 : 256;
        while (new_cap < need) new_cap *= 2;
        char *new_buf = (char *)realloc(ctx->stderr_buf, new_cap);
        if (!new_buf) return;
        ctx->stderr_buf = new_buf;
        ctx->stderr_cap = new_cap;
    }
    memcpy(ctx->stderr_buf + ctx->stderr_len, s, len);
    ctx->stderr_len += len;
    ctx->stderr_buf[ctx->stderr_len] = '\0';
    ctx->error = ctx->stderr_buf;
}

static void repl_ctx_output_wchar(am_repl_ctx_t *ctx, wchar_t c) {
    wchar_t src[2] = { c, L'\0' };
    char mb[8];
    uint32_t n = am_wcstombs(mb, src, sizeof(mb));
    if (n == 0) return;
    mb[n] = '\0';
    repl_ctx_output_mb(ctx, mb);
}

static void repl_ctx_error_wchar(am_repl_ctx_t *ctx, wchar_t c) {
    wchar_t src[2] = { c, L'\0' };
    char mb[8];
    uint32_t n = am_wcstombs(mb, src, sizeof(mb));
    if (n == 0) return;
    mb[n] = '\0';
    repl_ctx_error_mb(ctx, mb);
}

static void repl_ctx_output_wcs(am_repl_ctx_t *ctx, const wchar_t *s) {
    if (!ctx || !s) return;
    char *mb = wchar_to_mb(s);
    if (!mb) return;
    repl_ctx_output_mb(ctx, mb);
    free(mb);
}

static void repl_ctx_error_wcs(am_repl_ctx_t *ctx, const wchar_t *s) {
    if (!ctx || !s) return;
    char *mb = wchar_to_mb(s);
    if (!mb) return;
    repl_ctx_error_mb(ctx, mb);
    free(mb);
}

// 累积输入与会话操作
static int repl_ctx_accum_append(am_repl_ctx_t *ctx, const wchar_t *line) {
    if (!ctx || !line) return 0;
    size_t line_len = wcslen(line);
    size_t need = ctx->accum_len + (ctx->accum_len > 0 ? 1 : 0) + line_len;
    wchar_t *new_accum = (wchar_t *)realloc(ctx->accum, (need + 1) * sizeof(wchar_t));
    if (!new_accum) return 0;
    ctx->accum = new_accum;
    if (ctx->accum_len > 0) {
        ctx->accum[ctx->accum_len] = L'\n';
        ctx->accum_len++;
    }
    memcpy(ctx->accum + ctx->accum_len, line, line_len * sizeof(wchar_t));
    ctx->accum_len += line_len;
    ctx->accum[ctx->accum_len] = L'\0';
    return 1;
}

static void repl_ctx_reset_accum(am_repl_ctx_t *ctx) {
    if (!ctx) return;
    free(ctx->accum);
    ctx->accum = NULL;
    ctx->accum_len = 0;
    ctx->multiline = 0;
}

static int repl_ctx_session_append(am_repl_ctx_t *ctx, const wchar_t *code, size_t len) {
    if (!ctx || !code) return 0;
    size_t need = ctx->session_len + (ctx->session_len > 0 ? 1 : 0) + len;
    wchar_t *new_session = (wchar_t *)realloc(ctx->session, (need + 1) * sizeof(wchar_t));
    if (!new_session) return 0;
    ctx->session = new_session;
    if (ctx->session_len > 0) {
        ctx->session[ctx->session_len] = L'\n';
        ctx->session_len++;
    }
    memcpy(ctx->session + ctx->session_len, code, len * sizeof(wchar_t));
    ctx->session_len += len;
    ctx->session[ctx->session_len] = L'\0';
    return 1;
}

static void repl_ctx_rollback_session(am_repl_ctx_t *ctx, size_t submitted_len) {
    if (!ctx) return;
    if (ctx->session_len > submitted_len) {
        ctx->session_len -= submitted_len + 1;
    } else {
        ctx->session_len = 0;
    }
    if (ctx->session) ctx->session[ctx->session_len] = L'\0';
}

// REPL 特殊指令类型。
typedef enum {
    REPL_CMD_NONE = 0,
    REPL_CMD_EXIT,
    REPL_CMD_HELP,
    REPL_CMD_RESET,
    REPL_CMD_JS,
    REPL_CMD_SCHEME,
} repl_cmd_t;

// 解析一行输入是否为特殊指令。仅在未进入多行输入且累积缓冲区为空时生效。
static repl_cmd_t repl_ctx_parse_command(am_repl_ctx_t *ctx, const wchar_t *line_w) {
    if (!ctx || !line_w) return REPL_CMD_NONE;
    if (ctx->multiline || (ctx->accum && ctx->accum_len > 0)) return REPL_CMD_NONE;

    if (wcscmp(line_w, L"exit") == 0 ||
        wcscmp(line_w, L"quit") == 0 ||
        wcscmp(line_w, L"(exit)") == 0 ||
        wcscmp(line_w, L"(quit)") == 0 ||
        wcscmp(line_w, L".exit") == 0 ||
        wcscmp(line_w, L".quit") == 0 ||
        wcscmp(line_w, L"(System.exit)") == 0) {
        return REPL_CMD_EXIT;
    }
    if (wcscmp(line_w, L".help") == 0) {
        return REPL_CMD_HELP;
    }
    if (wcscmp(line_w, L".reset") == 0) {
        return REPL_CMD_RESET;
    }
    if (wcscmp(line_w, L".javascript") == 0 || wcscmp(line_w, L".js") == 0) {
        return REPL_CMD_JS;
    }
    if (wcscmp(line_w, L".scheme") == 0 || wcscmp(line_w, L".scm") == 0) {
        return REPL_CMD_SCHEME;
    }
    return REPL_CMD_NONE;
}

static am_repl_result_t repl_ctx_handle_command(am_repl_ctx_t *ctx, repl_cmd_t cmd) {
    if (!ctx) return repl_result_from_ctx(ctx);

    switch (cmd) {
        case REPL_CMD_EXIT:
            ctx->status = AM_REPL_STATUS_EXIT;
            break;
        case REPL_CMD_HELP:
            ctx->status = AM_REPL_STATUS_OUTPUT;
            repl_ctx_output_wcs(ctx,
                L"Animac Interpreter\n"
                L"Copyright (c) 2018-2026 BD4SUR\n"
                L"https://github.com/bd4sur/Animac\n"
                L"REPL Command Reference:\n"
                L"  .help    show this information.\n"
                L"  .reset   reset REPL context and clear all history.\n"
                L"  .js      switch to JavaScript mode (with REPL context reset)\n"
                L"  .scm     switch to Scheme mode (with REPL context reset)\n"
                L"  .exit    exit the REPL\n"
                L"   Ctrl+C  exit the REPL\n"
                L"\n");
            if (ctx->js_mode) {
                repl_ctx_output_wcs(ctx, L"Current mode: JavaScript\n");
            } else {
                repl_ctx_output_wcs(ctx, L"Current mode: Scheme\n");
            }
            repl_ctx_output_wcs(ctx, L"\n");
            break;
        case REPL_CMD_RESET:
            if (repl_ctx_reset(ctx) == 0) {
                ctx->status = AM_REPL_STATUS_OUTPUT;
                repl_ctx_output_wcs(ctx, L"[REPL] 上下文已重置，历史记录已清空。\n");
            } else {
                ctx->status = AM_REPL_STATUS_ERROR;
                repl_ctx_error_wcs(ctx, L"[REPL] 重置失败\n");
            }
            break;
        case REPL_CMD_JS:
            if (ctx->js_mode) {
                ctx->status = AM_REPL_STATUS_OUTPUT;
                repl_ctx_output_wcs(ctx, L"[REPL] 当前已经是 JavaScript 模式。\n");
            } else if (repl_ctx_reset(ctx) == 0) {
                am_repl_ctx_set_js_mode(ctx, 1);
                ctx->status = AM_REPL_STATUS_OUTPUT;
                repl_ctx_output_wcs(ctx, L"[REPL] 已重置上下文并切换到 JavaScript 模式。\n");
            } else {
                ctx->status = AM_REPL_STATUS_ERROR;
                repl_ctx_error_wcs(ctx, L"[REPL] 模式切换失败\n");
            }
            break;
        case REPL_CMD_SCHEME:
            if (!ctx->js_mode) {
                ctx->status = AM_REPL_STATUS_OUTPUT;
                repl_ctx_output_wcs(ctx, L"[REPL] 当前已经是 Scheme 模式。\n");
            } else if (repl_ctx_reset(ctx) == 0) {
                am_repl_ctx_set_js_mode(ctx, 0);
                ctx->status = AM_REPL_STATUS_OUTPUT;
                repl_ctx_output_wcs(ctx, L"[REPL] 已重置上下文并切换到 Scheme 模式。\n");
            } else {
                ctx->status = AM_REPL_STATUS_ERROR;
                repl_ctx_error_wcs(ctx, L"[REPL] 模式切换失败\n");
            }
            break;
        default:
            ctx->status = AM_REPL_STATUS_CONTINUE;
            break;
    }
    return repl_result_from_ctx(ctx);
}

static am_repl_result_t repl_result_from_ctx(am_repl_ctx_t *ctx) {
    am_repl_result_t res;
    res.status = ctx ? ctx->status : AM_REPL_STATUS_ERROR;
    res.output = ctx ? ctx->output : NULL;
    res.error = ctx ? ctx->error : NULL;
    res.indent = ctx ? ctx->indent : 0;
    return res;
}

// 去掉 JS 翻译器自动添加的最外层 ((lambda () ...)) 包装，
// 使后续的 repl_build_session_code 能重新包装并自动 display 最后一个表达式。
// 返回指向 scheme 缓冲区内部的指针；scheme 仍由调用者负责 free。
static wchar_t *repl_ctx_unwrap_js_scheme(wchar_t *scheme) {
    if (!scheme) return NULL;
    size_t len = wcslen(scheme);
    const wchar_t *prefix = L"((lambda ()";
    size_t prefix_len = wcslen(prefix);
    if (len < prefix_len || wcsncmp(scheme, prefix, prefix_len) != 0) {
        return scheme;
    }

    // 去掉尾部空白，找到最后的 "))"
    while (len > 0 &&
           (scheme[len - 1] == L'\n' || scheme[len - 1] == L'\r' ||
            scheme[len - 1] == L' ' || scheme[len - 1] == L'\t')) {
        scheme[len - 1] = L'\0';
        len--;
    }

    if (len >= 2 && scheme[len - 1] == L')' && scheme[len - 2] == L')') {
        scheme[len - 2] = L'\0';
        return scheme + prefix_len;
    }

    return scheme;
}

// 计算 JS 代码的建议缩进层级（未闭合的括号/花括号/方括号数）。
static int repl_ctx_js_indent(const wchar_t *code) {
    if (!code) return 0;
    int depth = 0;
    int in_string = 0;
    int escape = 0;
    int line_comment = 0;
    int block_comment = 0;

    for (size_t i = 0; code[i] != L'\0'; i++) {
        wchar_t c = code[i];
        wchar_t next = code[i + 1];

        if (line_comment) {
            if (c == L'\n') line_comment = 0;
            continue;
        }
        if (block_comment) {
            if (c == L'*' && next == L'/') {
                block_comment = 0;
                i++;
            }
            continue;
        }

        if (in_string) {
            if (escape) {
                escape = 0;
                continue;
            }
            if (c == L'\\') {
                escape = 1;
                continue;
            }
            if (c == L'"' || c == L'\'') {
                in_string = 0;
            }
            continue;
        }

        if (c == L'/' && next == L'/') {
            line_comment = 1;
            i++;
            continue;
        }
        if (c == L'/' && next == L'*') {
            block_comment = 1;
            i++;
            continue;
        }
        if (c == L'"' || c == L'\'') {
            in_string = 1;
            continue;
        }

        if (c == L'(' || c == L'[' || c == L'{') {
            depth++;
        } else if (c == L')' || c == L']' || c == L'}') {
            depth--;
        }
    }

    if (depth < 0) depth = 0;
    return depth;
}

// 将当前 accum 中的 JS 代码翻译成 Scheme 并执行。
// force=0 时翻译失败仅表示代码尚不完整，返回 CONTINUE；
// force=1 时（空行触发）翻译失败则报错误并清空 accum。
static am_repl_result_t repl_ctx_eval_js_accum(am_repl_ctx_t *ctx, int force) {
    if (!ctx) return repl_result_from_ctx(ctx);

    // 非强制提交时，若 JS 代码括号/花括号明显未闭合，暂不翻译，避免打印错误信息。
    int js_indent = repl_ctx_js_indent(ctx->accum);
    if (!force && js_indent > 0) {
        ctx->status = AM_REPL_STATUS_CONTINUE;
        ctx->indent = js_indent;
        return repl_result_from_ctx(ctx);
    }

    wchar_t *scheme = am_js_to_scheme(ctx->accum);
    if (!scheme) {
        // 翻译失败：强制提交，或括号已闭合（js_indent <= 0）但仍失败 = 真实的语法错误。
        // 报错并重置累积缓冲——原实现静默返回 CONTINUE 且不清理 accum，
        // 坏输入会留在累积缓冲里毒化后续所有输入（表现为无回显且永久卡死）。
        if (force || js_indent <= 0) {
            repl_ctx_reset_accum(ctx);
            ctx->status = AM_REPL_STATUS_ERROR;
            const wchar_t *err = am_js_last_error();
            if (err && err[0] != L'\0') {
                wchar_t err_buf[320];
                swprintf(err_buf, sizeof(err_buf) / sizeof(err_buf[0]), L"[JS] ");
                wcsncat(err_buf, err, sizeof(err_buf) / sizeof(err_buf[0]) - wcslen(err_buf) - 2);
                wcsncat(err_buf, L"\n", sizeof(err_buf) / sizeof(err_buf[0]) - wcslen(err_buf) - 1);
                repl_ctx_error_wcs(ctx, err_buf);
            } else {
                repl_ctx_error_wcs(ctx, L"[REPL] JS 翻译失败\n");
            }
        } else {
            ctx->status = AM_REPL_STATUS_CONTINUE;
            ctx->indent = js_indent;
        }
        return repl_result_from_ctx(ctx);
    }

    // 翻译成功：去掉 JS 翻译器自带的外层 lambda 包装，
    // 让 repl_build_session_code 重新包装并自动 display 最后一个表达式。
    wchar_t *unwrapped = repl_ctx_unwrap_js_scheme(scheme);
    size_t unwrapped_len = wcslen(unwrapped);
    wchar_t *scheme_copy = (wchar_t *)malloc((unwrapped_len + 1) * sizeof(wchar_t));
    if (!scheme_copy) {
        free(scheme);
        repl_ctx_reset_accum(ctx);
        ctx->status = AM_REPL_STATUS_ERROR;
        repl_ctx_error_wcs(ctx, L"[REPL] 内存不足\n");
        return repl_result_from_ctx(ctx);
    }
    wcscpy(scheme_copy, unwrapped);
    free(scheme);

    wchar_t *js_accum = ctx->accum;
    ctx->accum = scheme_copy;
    ctx->accum_len = unwrapped_len;
    ctx->multiline = 0;

    am_repl_result_t res = repl_ctx_submit(ctx);

    // submit 已释放 scheme_copy 并清空 accum；仍需释放原始 JS 累积缓冲区。
    free(js_accum);
    return res;
}

// 查找从 start 开始的下一个 S-表达式的结束位置（返回末尾后一个字符的索引）。
// 失败返回 SIZE_MAX。此函数为轻量级扫描器，仅用于识别顶层 display 表达式。
static size_t repl_find_expr_end(const wchar_t *s, size_t start, size_t len) {
    size_t i = start;
    while (i < len && iswspace(s[i])) i++;
    if (i >= len) return i;

    // 处理引用简写前缀：' ` , ,@
    int has_prefix = 1;
    while (has_prefix && i < len) {
        if (s[i] == L'\'' || s[i] == L'`') {
            i++;
            continue;
        }
        if (s[i] == L',' && i + 1 < len && s[i + 1] == L'@') {
            i += 2;
            continue;
        }
        if (s[i] == L',') {
            i++;
            continue;
        }
        has_prefix = 0;
    }
    if (i >= len) return SIZE_MAX;

    // 处理向量前缀 #( ... )
    if (s[i] == L'#' && i + 1 < len &&
        (s[i + 1] == L'(' || s[i + 1] == L'[' || s[i + 1] == L'{')) {
        i++;
    }

    if (s[i] == L'(' || s[i] == L'[' || s[i] == L'{') {
        wchar_t open = s[i];
        wchar_t close;
        if (open == L'(') close = L')';
        else if (open == L'[') close = L']';
        else close = L'}';

        int depth = 1;
        i++;
        int in_string = 0;
        int escape = 0;
        int line_comment = 0;

        while (i < len && depth > 0) {
            wchar_t c = s[i];
            if (line_comment) {
                if (c == L'\n') line_comment = 0;
                i++;
                continue;
            }
            if (in_string) {
                if (escape) {
                    escape = 0;
                    i++;
                    continue;
                }
                if (c == L'\\') {
                    escape = 1;
                    i++;
                    continue;
                }
                if (c == L'"') in_string = 0;
                i++;
                continue;
            }
            if (c == L';') {
                line_comment = 1;
                i++;
                continue;
            }
            if (c == L'"') {
                in_string = 1;
                i++;
                continue;
            }
            if (c == open) depth++;
            else if (c == close) depth--;
            i++;
        }
        if (depth != 0) return SIZE_MAX;
        return i;
    } else if (s[i] == L'"') {
        // 字符串字面量
        i++;
        int in_string = 1;
        int escape = 0;
        while (i < len && in_string) {
            wchar_t c = s[i];
            if (escape) {
                escape = 0;
                i++;
                continue;
            }
            if (c == L'\\') {
                escape = 1;
                i++;
                continue;
            }
            if (c == L'"') in_string = 0;
            i++;
        }
        return i;
    } else {
        // 原子：读取到空白、行注释或分隔符
        while (i < len) {
            wchar_t c = s[i];
            if (iswspace(c) || c == L';' ||
                c == L'(' || c == L')' ||
                c == L'[' || c == L']' ||
                c == L'{' || c == L'}') {
                break;
            }
            i++;
        }
        return i;
    }
}

// 判断 [start, end) 区间的表达式是否为顶层 (display ...) 调用。
static int repl_expr_is_display(const wchar_t *s, size_t start, size_t end) {
    size_t i = start;
    while (i < end && iswspace(s[i])) i++;
    if (i >= end || s[i] != L'(') return 0;
    i++;
    while (i < end && iswspace(s[i])) i++;
    const wchar_t *kw = L"display";
    size_t kw_len = wcslen(kw);
    if (i + kw_len > end || wcsncmp(s + i, kw, kw_len) != 0) return 0;
    i += kw_len;
    if (i < end && !iswspace(s[i]) && s[i] != L')') return 0;
    return 1;
}

// 移除 session 中所有顶层 (display ...) 表达式，返回新分配的宽字符串。
// 调用者负责 free。若分配失败返回 NULL。
static wchar_t *repl_session_strip_display(const wchar_t *session) {
    if (!session) return NULL;
    size_t len = wcslen(session);
    wchar_t *out = (wchar_t *)malloc((len + 1) * sizeof(wchar_t));
    if (!out) return NULL;
    size_t out_len = 0;
    size_t i = 0;

    while (i < len) {
        // 跳过并保留非首部的空白字符
        while (i < len && iswspace(session[i])) {
            if (out_len > 0) {
                out[out_len++] = session[i];
            }
            i++;
        }
        if (i >= len) break;

        // 保留行注释
        if (session[i] == L';') {
            size_t comment_start = i;
            while (i < len && session[i] != L'\n') i++;
            size_t n = i - comment_start;
            memcpy(out + out_len, session + comment_start, n * sizeof(wchar_t));
            out_len += n;
            continue;
        }

        size_t expr_start = i;
        size_t expr_end = repl_find_expr_end(session, expr_start, len);
        if (expr_end == SIZE_MAX || expr_end <= expr_start) {
            // 格式异常，复制剩余部分并结束
            size_t n = len - expr_start;
            memcpy(out + out_len, session + expr_start, n * sizeof(wchar_t));
            out_len += n;
            break;
        }

        if (!repl_expr_is_display(session, expr_start, expr_end)) {
            size_t n = expr_end - expr_start;
            memcpy(out + out_len, session + expr_start, n * sizeof(wchar_t));
            out_len += n;
        }

        i = expr_end;
    }

    // 去除尾部空白
    while (out_len > 0 && iswspace(out[out_len - 1])) {
        out_len--;
    }
    out[out_len] = L'\0';
    return out;
}

// 彻底重置 REPL 上下文：释放所有资源并冷启动整个系统。
// 包括 Runtime、Allocator Pool、输入/输出缓冲区、会话历史、进程状态等。
static int repl_ctx_reset(am_repl_ctx_t *ctx) {
    if (!ctx) return -1;

    // 保留解释器模式与内存池大小，其余全部重置。
    int saved_js_mode = ctx->js_mode;
    size_t saved_pool_size = ctx->pool_size;

    // 释放标准 C 分配的缓冲区。
    free(ctx->accum);
    free(ctx->session);
    free(ctx->stdout_buf);
    free(ctx->stderr_buf);
    free(ctx->base_dir);

    // 销毁旧的运行时与分配器池，彻底归还其管理的整片内存。
    if (ctx->rt) {
        am_runtime_destroy(ctx->rt);
    }
    if (ctx->pool) {
        am_allocator_pool_destroy(ctx->pool);
    }

    // 清空上下文所有字段，随后恢复冷启动所需的默认值。
    memset(ctx, 0, sizeof(am_repl_ctx_t));

    ctx->pid = (am_pid_t)-1;
    ctx->status = AM_REPL_STATUS_CONTINUE;
    ctx->js_mode = saved_js_mode;
    ctx->pool_size = saved_pool_size;
    ctx->should_stop = 0;
    ctx->prompt_main = "> ";
    ctx->prompt_cont = "... ";

    ctx->base_dir = repl_get_cwd();
    if (!ctx->base_dir) {
        ctx->base_dir = (wchar_t *)malloc(2 * sizeof(wchar_t));
        if (!ctx->base_dir) return -1;
        ctx->base_dir[0] = L'.';
        ctx->base_dir[1] = L'\0';
    }

    // 重新初始化运行时基础设施，相当于从头冷启动。
    if (repl_ctx_init_runtime(ctx) != 0) {
        return -1;
    }

    ctx->runtime_error = 0;

    return 0;
}

// 提交当前 accum 到 session 并执行。
static am_repl_result_t repl_ctx_submit(am_repl_ctx_t *ctx) {
    if (!ctx) return repl_result_from_ctx(ctx);

    // 追加新输入前，先清除历史记录中的 display 表达式（仅保留最新输入中的 display）。
    if (ctx->session && ctx->session_len > 0) {
        wchar_t *stripped = repl_session_strip_display(ctx->session);
        if (stripped) {
            free(ctx->session);
            ctx->session = stripped;
            ctx->session_len = wcslen(stripped);
        }
    }

    size_t submitted_len = ctx->accum_len;
    if (!repl_ctx_session_append(ctx, ctx->accum, ctx->accum_len)) {
        repl_ctx_reset_accum(ctx);
        ctx->status = AM_REPL_STATUS_ERROR;
        repl_ctx_error_wcs(ctx, L"[REPL] 内存不足\n");
        return repl_result_from_ctx(ctx);
    }
    repl_ctx_reset_accum(ctx);

    am_module_t *new_mod = NULL;
    if (repl_eval_session(ctx, ctx->session, &new_mod) != 0) {
        repl_ctx_rollback_session(ctx, submitted_len);
        ctx->status = AM_REPL_STATUS_ERROR;
        return repl_result_from_ctx(ctx);
    }

    // 新模块编译成功后，立即终止旧进程，避免新旧进程同时驻留导致内存持续累积。
    if (ctx->pid != (am_pid_t)-1) {
        am_runtime_kill_process(ctx->rt, ctx->pid);
        ctx->pid = (am_pid_t)-1;
    }

    ctx->runtime_error = 0;
    am_pid_t new_pid = am_runtime_load_module(ctx->rt, new_mod);

    if (new_pid == (am_pid_t)-1) {
        if (new_mod->ilcode) {
            am_free(ctx->vm_alloc, new_mod->ilcode);
            new_mod->ilcode = NULL;
        }
        if (new_mod->ast) {
            am_ast_destroy(new_mod->ast);
            new_mod->ast = NULL;
        }
        am_free(ctx->vm_alloc, new_mod);
        repl_ctx_rollback_session(ctx, submitted_len);
        ctx->status = AM_REPL_STATUS_ERROR;
        repl_ctx_error_wcs(ctx, L"[REPL] 加载新进程失败\n");
        return repl_result_from_ctx(ctx);
    }

    if (new_mod->ilcode) {
        am_free(ctx->vm_alloc, new_mod->ilcode);
        new_mod->ilcode = NULL;
    }
    if (new_mod->ast) {
        am_ast_destroy(new_mod->ast);
        new_mod->ast = NULL;
    }
    am_free(ctx->vm_alloc, new_mod);

    int32_t running = 1;
    while (running && !ctx->should_stop) {
        running = am_runtime_tick(ctx->rt, ctx->rt->timeslice);
    }

    // 确保所有输出都被捕获
    on_tick(ctx->rt);

    if (ctx->runtime_error) {
        am_runtime_kill_process(ctx->rt, new_pid);
        repl_ctx_rollback_session(ctx, submitted_len);
        ctx->pid = (am_pid_t)-1;
        ctx->status = AM_REPL_STATUS_ERROR;
        return repl_result_from_ctx(ctx);
    }

    ctx->pid = new_pid;
    ctx->status = AM_REPL_STATUS_OUTPUT;
    return repl_result_from_ctx(ctx);
}

// 初始化 REPL 上下文的运行时基础设施：allocator pool、runtime、native libs、初始模块。
// 调用前 ctx 的 pid 应为 -1，pool/rt 应为 NULL。
static int repl_ctx_init_runtime(am_repl_ctx_t *ctx) {
    if (!ctx) return -1;

    ctx->pool = am_allocator_pool_create(ctx->pool_size, &am_host_default_vtable);
    if (!ctx->pool) {
        ctx->vm_alloc = NULL;
        ctx->heap_alloc = NULL;
        return -1;
    }
    ctx->vm_alloc = am_allocator_pool_get_vm(ctx->pool);
    ctx->heap_alloc = am_allocator_pool_get_heap(ctx->pool);

    const wchar_t *base_dir = ctx->base_dir ? ctx->base_dir : L".";
    ctx->rt = am_runtime_create(ctx->vm_alloc, ctx->heap_alloc, base_dir, &g_host_vtable);
    if (!ctx->rt) {
        am_allocator_pool_destroy(ctx->pool);
        ctx->pool = NULL;
        ctx->vm_alloc = NULL;
        ctx->heap_alloc = NULL;
        return -1;
    }

    am_runtime_set_default_timeslice(ctx->rt, 8192);
    am_set_runtime_host_context(ctx->rt, ctx);

    am_runtime_register_native_lib(ctx->rt, &am_native_System_lib);
    am_runtime_register_native_lib(ctx->rt, &am_native_Math_lib);
    am_runtime_register_native_lib(ctx->rt, &am_native_String_lib);
    am_runtime_register_native_lib(ctx->rt, &am_native_LLM_lib);
    am_runtime_register_native_lib(ctx->rt, &am_native_Table_lib);

    am_module_t *mod = repl_create_initial_module(ctx);
    if (!mod) {
        am_runtime_destroy(ctx->rt);
        am_allocator_pool_destroy(ctx->pool);
        ctx->rt = NULL;
        ctx->pool = NULL;
        ctx->vm_alloc = NULL;
        ctx->heap_alloc = NULL;
        return -1;
    }

    ctx->pid = am_runtime_load_module(ctx->rt, mod);
    if (ctx->pid == (am_pid_t)-1) {
        if (mod->ilcode) am_free(ctx->vm_alloc, mod->ilcode);
        if (mod->ast) am_ast_destroy(mod->ast);
        am_free(ctx->vm_alloc, mod);
        am_runtime_destroy(ctx->rt);
        am_allocator_pool_destroy(ctx->pool);
        ctx->rt = NULL;
        ctx->pool = NULL;
        ctx->vm_alloc = NULL;
        ctx->heap_alloc = NULL;
        return -1;
    }

    if (mod->ilcode) {
        am_free(ctx->vm_alloc, mod->ilcode);
        mod->ilcode = NULL;
    }
    if (mod->ast) {
        am_ast_destroy(mod->ast);
        mod->ast = NULL;
    }
    am_free(ctx->vm_alloc, mod);

    return 0;
}

// 建立 REPL 上下文。pool_size 指定内存池总大小（单位：字节）。
am_repl_ctx_t *am_repl_ctx_create(size_t pool_size) {
    am_repl_ctx_t *ctx = (am_repl_ctx_t *)calloc(1, sizeof(am_repl_ctx_t));
    if (!ctx) return NULL;

    ctx->pool_size = pool_size;
    ctx->pid = (am_pid_t)-1;
    ctx->status = AM_REPL_STATUS_CONTINUE;
    ctx->js_mode = 0;
    ctx->should_stop = 0;
    ctx->prompt_main = "> ";
    ctx->prompt_cont = "... ";

    ctx->base_dir = repl_get_cwd();
    if (!ctx->base_dir) {
        ctx->base_dir = (wchar_t *)malloc(2 * sizeof(wchar_t));
        if (!ctx->base_dir) {
            free(ctx);
            return NULL;
        }
        ctx->base_dir[0] = L'.';
        ctx->base_dir[1] = L'\0';
    }

    if (repl_ctx_init_runtime(ctx) != 0) {
        free(ctx->base_dir);
        free(ctx);
        return NULL;
    }

    return ctx;
}

// 请求停止 REPL 运行。
void am_repl_ctx_interrupt(am_repl_ctx_t *ctx) {
    if (ctx) ctx->should_stop = 1;
}

// 设置 JS 解释器模式。
void am_repl_ctx_set_js_mode(am_repl_ctx_t *ctx, int js_mode) {
    if (!ctx) return;
    ctx->js_mode = js_mode ? 1 : 0;
    if (ctx->js_mode) {
        ctx->prompt_main = "> ";
        ctx->prompt_cont = "... ";
    } else {
        ctx->prompt_main = "> ";
        ctx->prompt_cont = "... ";
    }
}

// 销毁 REPL 上下文。
void am_repl_ctx_destroy(am_repl_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->rt) {
        if (ctx->pid != (am_pid_t)-1) {
            am_runtime_kill_process(ctx->rt, ctx->pid);
        }
        am_runtime_destroy(ctx->rt);
    }
    if (ctx->pool) am_allocator_pool_destroy(ctx->pool);
    free(ctx->accum);
    free(ctx->session);
    free(ctx->stdout_buf);
    free(ctx->stderr_buf);
    free(ctx->base_dir);
    free(ctx);
}

// Scheme 模式：逐行累积并检查 S-表达式完整性。
static am_repl_result_t am_repl_ctx_feed_scheme(am_repl_ctx_t *ctx, const char *input) {
    if (!ctx) return repl_result_from_ctx(ctx);

    // 空行：尝试提交已累积的输入
    if (!input || input[0] == '\0') {
        if (!ctx->accum || ctx->accum_len == 0) {
            ctx->status = AM_REPL_STATUS_CONTINUE;
            return repl_result_from_ctx(ctx);
        }
        int status = check_expression_complete(ctx->accum, &ctx->indent);
        if (status == 1) {
            ctx->status = AM_REPL_STATUS_CONTINUE;
            return repl_result_from_ctx(ctx);
        }
        if (status == -1) {
            repl_ctx_reset_accum(ctx);
            ctx->status = AM_REPL_STATUS_ERROR;
            repl_ctx_error_wcs(ctx, L"[REPL] 括号不匹配\n");
            return repl_result_from_ctx(ctx);
        }
        // status == 0，fall through 到提交
    } else {
        wchar_t *line_w = mb_to_wchar(input);
        if (!line_w) {
            ctx->status = AM_REPL_STATUS_ERROR;
            repl_ctx_error_wcs(ctx, L"[REPL] 输入编码失败\n");
            return repl_result_from_ctx(ctx);
        }

        // 特殊指令（仅在尚未进入多行输入时生效）
        repl_cmd_t cmd = repl_ctx_parse_command(ctx, line_w);
        if (cmd != REPL_CMD_NONE) {
            free(line_w);
            return repl_ctx_handle_command(ctx, cmd);
        }

        if (!repl_ctx_accum_append(ctx, line_w)) {
            free(line_w);
            ctx->status = AM_REPL_STATUS_ERROR;
            repl_ctx_error_wcs(ctx, L"[REPL] 内存不足\n");
            return repl_result_from_ctx(ctx);
        }
        free(line_w);

        int status = check_expression_complete(ctx->accum, &ctx->indent);
        if (status == 1) {
            ctx->multiline = 1;
            ctx->status = AM_REPL_STATUS_CONTINUE;
            return repl_result_from_ctx(ctx);
        }
        if (status == -1) {
            repl_ctx_reset_accum(ctx);
            ctx->status = AM_REPL_STATUS_ERROR;
            repl_ctx_error_wcs(ctx, L"[REPL] 括号不匹配\n");
            return repl_result_from_ctx(ctx);
        }
        // status == 0，fall through 到提交
    }

    return repl_ctx_submit(ctx);
}

// JS 模式：逐行累积 JS，每次尝试翻译成 Scheme；翻译成功则执行。
static am_repl_result_t am_repl_ctx_feed_js(am_repl_ctx_t *ctx, const char *input) {
    if (!ctx) return repl_result_from_ctx(ctx);

    // 空行：强制尝试翻译一次
    if (!input || input[0] == '\0') {
        if (!ctx->accum || ctx->accum_len == 0) {
            ctx->status = AM_REPL_STATUS_CONTINUE;
            return repl_result_from_ctx(ctx);
        }
        return repl_ctx_eval_js_accum(ctx, 1);
    }

    wchar_t *line_w = mb_to_wchar(input);
    if (!line_w) {
        ctx->status = AM_REPL_STATUS_ERROR;
        repl_ctx_error_wcs(ctx, L"[REPL] 输入编码失败\n");
        return repl_result_from_ctx(ctx);
    }

    // 特殊指令（仅在尚未进入多行输入时生效）
    repl_cmd_t cmd = repl_ctx_parse_command(ctx, line_w);
    if (cmd != REPL_CMD_NONE) {
        free(line_w);
        return repl_ctx_handle_command(ctx, cmd);
    }

    if (!repl_ctx_accum_append(ctx, line_w)) {
        free(line_w);
        ctx->status = AM_REPL_STATUS_ERROR;
        repl_ctx_error_wcs(ctx, L"[REPL] 内存不足\n");
        return repl_result_from_ctx(ctx);
    }
    free(line_w);

    // 每次追加后都尝试翻译；若失败则认为 JS 仍不完整，继续输入。
    return repl_ctx_eval_js_accum(ctx, 0);
}

// 向 REPL 上下文传入一个字符串（通常是一行输入），返回结果并更新上下文内部状态。
am_repl_result_t am_repl_ctx_feed(am_repl_ctx_t *ctx, const char *input) {
    if (!ctx) {
        am_repl_result_t res;
        res.status = AM_REPL_STATUS_ERROR;
        res.output = NULL;
        res.indent = 0;
        return res;
    }

    repl_ctx_clear_output(ctx);
    ctx->status = AM_REPL_STATUS_CONTINUE;
    ctx->indent = 0;
    ctx->output = NULL;
    ctx->error = NULL;
    ctx->should_stop = 0;

    if (ctx->js_mode) {
        return am_repl_ctx_feed_js(ctx, input);
    }
    return am_repl_ctx_feed_scheme(ctx, input);
}


