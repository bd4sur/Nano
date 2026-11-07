#!/usr/bin/env python3
# ============================================================================
# run.py —— 证明「纯 Scheme LLM 库」与「上游 (native LLM) 本地库」完全功能等价
#
# 做法：把 app/llm.scm 里的**前向传播与采样代码**原样抽出来（按顶层 define 名字挑选，
# 不复制粘贴），配上同一份驱动脚本，分别链接：
#   A) 上游本地库 am_native_LLM.c（本目录的 native_shim.scm 封装成同名接口）
#   B) 纯 Scheme 实现 app/lib/llm_pure.scm
# 两次运行的 stdout 必须**逐字节相同**（配置、14 个张量的长度/求和/首中末元素、分词、
# 前向 logits 序列、贪心解码的文本、排序任务的输出）。
#
# 用法：
#   python3 run.py                 # 构建 + 对跑 + 比对
#   python3 run.py --keep          # 保留 /tmp/llm_equiv 下的中间产物
#
# 注意：nref 是 64 位宿主，故浮点是双精度；本测试比对的是两条**同一宿主上**的实现，
# 与 wasm32（单精度）上的数值不同但同样成立。
# ============================================================================

import os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.abspath(os.path.join(HERE, '..', '..'))
VENDOR = os.path.join(INFER, 'vendor', 'Animac')
APP = os.path.join(INFER, 'app', 'llm.scm')
LLM_DIR = os.path.join(INFER, 'app', 'lib', 'llm')
WORK = '/tmp/llm_equiv'


def split_forms(s):
    """按顶层平衡括号切分（能正确跳过字符串、注释）。"""
    forms, depth, start, instr, incomment, i = [], 0, None, False, False, 0
    while i < len(s):
        ch = s[i]
        if incomment:
            if ch == '\n':
                incomment = False
            i += 1
            continue
        if instr:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                instr = False
            i += 1
            continue
        if ch == ';':
            incomment = True
            i += 1
            continue
        if ch == '"':
            instr = True
            i += 1
            continue
        if ch in '([':
            if depth == 0:
                start = i
            depth += 1
        elif ch in ')]':
            depth -= 1
            if depth == 0:
                forms.append(s[start:i + 1])
                start = None
        i += 1
    return forms


# 从 App 里抽取的顶层名字：模块状态 + 配置绑定 + 前向/采样 + 生成与排序的步进函数
KEEP = '''f4,f6,mkbuf,block_size,vocab_size,n_layer,n_embd,n_head,n_kv_head,n_hidden,
is_shared,head_dim,kv_dim,kv_mul,p_rms_attn,p_rms_ffn,p_rms_final,p_emb,p_wq,p_wk,p_wv,
p_wo,p_w1,p_w2,p_w3,p_fcr,p_fci,p_cls,bx,bxb,bxba,bxb2,bhb,bhb2,bq,k_cache,v_cache,batt,
blogits,bind_model,b_accum,b_rms_norm,b_softmax,b_rope,llm_forward,sample_argmax,
sample_multinomial,sample_token,N_GEN,TEMPERATURE,SORT_LEN,gen_prompt,gen_ids,gen_pos,
gen_next,gen_line,gen_t0,gen_n,gen_setup,gen_step,sort_input,sort_ids,sort_ref,sort_out,
sort_setup,sort_pos,sort_prefill_step,sort_gen_step,digits_ok,sorted_ok,concat_of,csv_of'''.replace('\n', '').split(',')

HEAD = '''(import LLM "@@LIB@@")
(native String)
(native Math)
(native System)
(native Fs)

;; 模型数据目录（绝对路径：(native Fs) 对以 '/' 开头的路径按宿主绝对路径处理）
(define MODEL_DIR "@@MODEL_DIR@@")
'''

STUBS = '''
;; App 侧的日志/界面函数在无头参考台里空实现（不参与比对内容）
(define say (lambda (s) #t))
(define set_status (lambda (s) #t))
(define narr_to_str
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s "(")
    (while (< i n) {
      (if (> i 0) (set! s (String.concat s " ")) #f)
      (set! s (String.concat s (String.atom_to_string (get_item lst i))))
      (set! i (+ i 1))
    })
    (String.concat s ")")))

'''

DRIVER = '''
;; ---------------- 加载（分块 base64 → 流式解码）----------------
(define i2s2
  (lambda (i)
    (if (< i 10) (String.concat "0" (String.atom_to_string i))
        (String.atom_to_string i))))
(define chunk_i 0)
(define load_model
  (lambda (name nchunk)
    (define r 0)
    (define ok #f)
    (define s #f)
    (LLM.init_begin)
    (set! chunk_i 0)
    (while (== 1 1) {
      (set! r 0)
      (if (>= chunk_i nchunk)
          (set! r 1)
          {
            (set! s (Fs.read (String.concat MODEL_DIR (String.concat "/" (String.concat name
              (String.concat "/m" (String.concat (i2s2 chunk_i) ".b64")))))))
            (set! chunk_i (+ chunk_i 1))
            (if (null? s)
                (set! r -1)
                {
                  (LLM.init_feed s)
                  (if (>= chunk_i nchunk) (set! r 1) (set! r 0))
                })
          })
      (if (== r 1) break #f)
      (if (== r -1) break #f)
    })
    (set! ok (LLM.init_end))
    (if ok (bind_model) #f)
    ok))

;; 张量摘要：长度 + 求和 + 首/中/末元素
(define tensor_stats
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s 0.0)
    (while (< i n) { (set! s (+ s (get_item lst i))) (set! i (+ i 1)) })
    (String.concat (String.atom_to_string n) (String.concat " sum=" (String.concat (f6 s)
      (String.concat " head=" (String.concat (f6 (get_item lst 0))
      (String.concat " mid=" (String.concat (f6 (get_item lst (Math.floor (/ n 2))))
      (String.concat " last=" (f6 (get_item lst (- n 1)))))))))))))

(define dump_model
  (lambda (name nchunk)
    (define ok #f)
    (define cfg #f)
    (define prm #f)
    (define i 0)
    (display "== model ") (display name) (newline)
    (set! ok (load_model name nchunk))
    (display "init ") (display ok) (newline)
    (set! cfg (LLM.get_config))
    (display "config ") (display cfg) (newline)
    (set! prm (LLM.get_param))
    (display "nparam ") (display (length prm)) (newline)
    (set! i 0)
    (while (< i (length prm)) {
      (display "T") (display i) (display " ") (display (tensor_stats (get_item prm i))) (newline)
      (set! i (+ i 1))
    })
    (display "cls_eq_emb ") (display (equal? (get_item prm 13) (get_item prm 3))) (newline)
    (display "cls_same_obj ") (display (eq? (get_item prm 13) (get_item prm 3))) (newline)
    #t))

;; ---------------- 分词 ----------------
(define dump_tokens
  (lambda (texts)
    (define i 0)
    (define ids #f)
    (define back "")
    (define j 0)
    (while (< i (length texts)) {
      (set! ids (LLM.encode (get_item texts i)))
      (display "enc ") (display (get_item texts i)) (display " -> ") (display ids) (newline)
      (set! back "")
      (set! j 0)
      (while (< j (length ids)) {
        (set! back (String.concat back (LLM.decode (get_item ids j))))
        (set! j (+ j 1))
      })
      (display "dec ") (display back) (newline)
      (set! i (+ i 1))
    })
    (set! i 0)
    (set! back "")
    (while (< i 40) {
      (set! back (String.concat back (String.concat "/" (LLM.decode i))))
      (set! i (+ i 1))
    })
    (display "vocab0_40 ") (display back) (newline)))

;; ---------------- 前向传播（逐位打印 logits 摘要）----------------
(define dump_forward
  (lambda (ids maxlen is_causal)
    (define pos 0)
    (define lg #f)
    (while (< pos maxlen) {
      (set! lg (llm_forward (get_item ids pos) pos maxlen is_causal))
      (display "pos ") (display pos)
      (display " argmax ") (display (sample_argmax lg vocab_size))
      (display " lg ") (display (f6 (get_item lg 0)))
      (display " ") (display (f6 (get_item lg 1)))
      (display " ") (display (f6 (get_item lg (Math.floor (/ vocab_size 2)))))
      (newline)
      (set! pos (+ pos 1))
    })))

;; ================ 主流程 ================
(dump_model "sort6" 1)
(dump_tokens '("0123456789" "a b c"))
;; 排序任务（全局自注意力，输入 351420）
(set! sort_input '(3 5 1 4 2 0))
(sort_setup)
(display "sort ids ") (display sort_ids) (newline)
(define pre_done #f)
(while (== pre_done #f) { (set! pre_done (sort_prefill_step)) })
(set! sort_pos 0)
(define gen_done #f)
(while (== gen_done #f) { (set! gen_done (sort_gen_step)) })
(display "sort out ") (display sort_out) (newline)

(dump_model "psycho" 14)
(dump_tokens '("你好" "人类的本质是" "精神分析" "3,5,1,4,2,6"))
;; 因果注意力前向（前 6 个位置）
(dump_forward (LLM.encode "人类的本质是") 6 #t)
;; 自回归生成（贪心，24 步）
(gen_setup)
(define going #t)
(define steps 0)
(while going {
  (if (gen_step) (set! going #f) #f)
  (set! steps (+ steps 1))
})
(display "gen steps ") (display steps) (newline)
(display "gen text ") (display gen_line) (newline)
(display "EQUIV DONE") (newline)
'''


def build_script(lib_import):
    forms = split_forms(open(APP, encoding='utf-8').read())
    kept = []
    for f in forms:
        m = re.match(r'\(\s*define\s+([^\s()]+)', f)
        if m and m.group(1) in KEEP:
            kept.append(f)
    missing = [k for k in KEEP if k not in [re.match(r'\(\s*define\s+([^\s()]+)', f).group(1)
                                            for f in kept]]
    if missing:
        sys.exit('从 app/llm.scm 抽取失败，缺少：' + ','.join(missing))
    head = HEAD.replace('@@LIB@@', lib_import).replace('@@MODEL_DIR@@', LLM_DIR)
    return head + '\n' + '\n\n'.join(kept) + '\n' + STUBS + DRIVER


def main():
    os.makedirs(WORK, exist_ok=True)
    exe = os.path.join(WORK, 'nref')
    srcs = [os.path.join(HERE, 'nref.c')]
    srcs += [os.path.join(VENDOR, f) for f in sorted(os.listdir(VENDOR)) if f.endswith('.c')]
    srcs += [os.path.join(INFER, f) for f in ('am_native_Fs.c', 'hal_fs_linux.c',
                                              'hal_ram_linux.c', 'utils.c')]
    cmd = ['gcc', '-std=gnu11', '-O2', '-w', '-I', INFER, '-I', VENDOR, '-o', exe] + srcs + ['-lm']
    print('[build]', ' '.join(cmd[:8]), '...')
    subprocess.run(cmd, check=True)

    variants = {
        'native': os.path.join(HERE, 'native_shim.scm'),
        'pure': os.path.relpath(os.path.join(INFER, 'app', 'lib', 'llm_pure.scm'), INFER),
    }
    outs = {}
    for tag, lib in variants.items():
        script = os.path.join(WORK, 'equiv_%s.scm' % tag)
        open(script, 'w', encoding='utf-8').write(build_script(lib))
        args = [exe, script, INFER] + (['--llm'] if tag == 'native' else [])
        print('[run]', tag)
        r = subprocess.run(args, capture_output=True, text=True)
        open(os.path.join(WORK, '%s.out' % tag), 'w', encoding='utf-8').write(r.stdout + r.stderr)
        outs[tag] = r.stdout

    a = [ln for ln in outs['native'].split('\n') if ln.strip() and ln.strip() != 'RESULT done']
    b = [ln for ln in outs['pure'].split('\n') if ln.strip() and ln.strip() != 'RESULT done']
    if a == b:
        print('\nPASS：本地库与纯 Scheme 库的 %d 行输出逐字节相同' % len(a))
        if not any('EQUIV DONE' in ln for ln in a):
            print('警告：没有跑到 EQUIV DONE')
            return 1
        return 0

    print('\nFAIL：输出不一致（详见 %s/{native,pure}.out）' % WORK)
    import difflib
    for line in list(difflib.unified_diff(a, b, 'native', 'pure', lineterm='', n=1))[:60]:
        print(line)
    return 1


if __name__ == '__main__':
    sys.exit(main())
