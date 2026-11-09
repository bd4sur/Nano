;; ============================================================================
;; app/lib/numtower.scm —— 数值类型塔（整数 ⊂ 有理数 ⊂ 实数）
;;
;; 灵机本身**没有**数值塔（manual.md §6.2.2：无精确/非精确之分、无有理数、整数还
;; 会静默溢出）。本文件在 Scheme 层补上「精确数」这一层，思路取自上游 Animac 用例
;; test/cas.scm（eint / efrac / ereal 三级塔），但把它做成**可运算完备**的一套：
;;
;;   精确整数  ('eint sign digits)       —— 见 app/lib/bigint.scm（任意精度）
;;   精确分数  ('efrac p q)              —— p、q 为精确整数，q ≥ 2、gcd(|p|,q)=1
;;   实数      解释器原生浮点（number?）  —— 精确数经 to_float 落入这一层
;;
;; 不变量（由 fr_make 统一维持）：分数恒为既约、分母恒正、分母为 1 时退化为整数、
;; 零恒为整数 0 —— 于是「相等」只有一种写法，gcd 不会越滚越大。
;;
;; 值域规则（与 R^nRS 的塔一致、但按浮点实现能承受的方式简化）：
;;   · + - * / % ^ 只要**两个操作数都精确**且运算本身是代数运算，结果就精确；
;;   · 任一侧是浮点，或运算本身是超越函数（sin/cos/tan/exp/log/非完全平方的 sqrt），
;;     结果就是浮点（"浮点传染"）；
;;   · 例外：sqrt 在完全平方数（含分数）上给出精确值，^ 在整数指数上给出精确值。
;;
;; 对外接口（NT.*）：
;;   NT.num? v / NT.exact? v / NT.int? v / NT.zero? v
;;   NT.parse_literal s     十进制字面量串（可带 +/- 与小数点）→ 精确数
;;   NT.num? / NT.exact? / NT.int? / NT.zero?      谓词
;;   NT.from_small n        原生整数 → 精确整数（走十进制串，不经浮点，27 位内都精确）
;;   NT.parse_literal s     十进制字面量（可带 +/- 与小数点）→ 精确数
;;   NT.add / NT.sub / NT.mul / NT.div            + - * /（精确；一侧浮点则浮点）
;;   NT.remainder a b        %（截断除法语义，结果符号随被除数，与内建 mod 一致）
;;   NT.power a b            ^（整数指数精确，含负指数；其余落浮点）
;;   NT.sqrt                 开方（完全平方精确，其余浮点）
;;   NT.neg / NT.abs / NT.floor / NT.ceil / NT.trunc
;;   NT.cmp a b              -1 / 0 / 1
;;   NT.to_float v           近似浮点（混合运算/浮点兜底）
;;   NT.to_string v          显示串（"" 表示不可显示：NaN/无穷）
;;   失败（除零、负数开方、非实数）一律返回符号 'err，由调用方判断——
;;   灵机没有异常机制，运行期报错会直接杀掉进程，所以这里绝不抛错。
;; ============================================================================

(native Math)
(native String)

(import BI "app/lib/bigint.scm")

;; 显示上限：精确值的十进制串超过这个长度就改用科学计数法（见 fmt_int / fmt_frac）。
;; 屏幕一行约 40 个 ASCII 字符（320px / 7px），当前行的值还要给表达式让位。
(define SHOW_MAX 20)
;; 精确幂的结果位数上限（估算）：超过就退回浮点，避免一次按键算到天荒地老。
;; 实测（wasm32）：2^1000（302 位）9ms、99^400（799 位）56ms、1000 位以内 ≤ 约 90ms——
;; 都在"一次按键"的可接受范围内；再往上就落浮点（显示科学计数法）。
(define POW_MAX_DIGITS 1000)
;; 精确幂的指数上限
(define POW_MAX_EXP 4096)

;; ---------------------------------------------------------------------------
;; 谓词与取值
;; ---------------------------------------------------------------------------

(define efrac? (lambda (v) (and (list? v) (== (length v) 3) (eq? (car v) 'efrac))))

(define exact? (lambda (v) (or (BI.int? v) (efrac? v))))

(define num? (lambda (v) (or (number? v) (BI.int? v) (efrac? v))))

(define int? (lambda (v) (BI.int? v)))

(define zero?
  (lambda (v)
    (cond ((number? v) (== v 0))
          ((BI.int? v) (BI.is_zero v))
          ((efrac? v) #f)
          (else #f))))

;; 分数的分子/分母；整数视作分母为 1 的分数
(define fr_p (lambda (x) (if (efrac? x) (get_item x 1) x)))
(define fr_q (lambda (x) (if (efrac? x) (get_item x 2) (BI.one))))

;; ---------------------------------------------------------------------------
;; 精确整数的工具
;; ---------------------------------------------------------------------------

(define zero (lambda () (BI.zero)))
(define one (lambda () (BI.one)))

;; 精确整数 → 原生小整数；超过 7 位返回 #f。
;; 用整数累加（全程整数运算，不经浮点），在 wasm32 的 27 位整数范围内安全。
(define bi_to_small
  (lambda (x) {
    (define ds (BI.digits x))
    (define n (length ds))
    (define i 0)
    (define acc 0)
    (if (> n 7)
        #f
        {
          (while (< i n) {
            (set! acc (+ (* acc 10) (get_item ds i)))
            (set! i (+ i 1))
          })
          (if (< (BI.sign x) 0) (- 0 acc) acc)
        })
  }))

;; 原生小整数 → 精确整数。
;; **只能用整数运算**：wasm32 宿主的 mod / Math.floor 都会先把参数转成 19 位尾数的
;; float，超过 2^19 = 524288 就不再精确（曾把 123456789 变成 123456442）。
;; 整数走 display 的整数打印路径（精确，不经浮点），故这里取十进制串再解析。
;; 只接受整数：传入带小数部分的浮点，结果按截断后的整数部分理解。
(define from_small
  (lambda (v) (BI.from_string (String.atom_to_string v))))

;; ---------------------------------------------------------------------------
;; 分数的规范化
;; ---------------------------------------------------------------------------

;; p/q → 既约分数；分母取正；q 化为 1 时退化为整数；分子为 0 时退化为整数 0
(define fr_make
  (lambda (p q) {
    (define pp p)
    (define qq q)
    (define g (BI.one))
    (if (< (BI.cmp qq (BI.zero)) 0) {
      (set! pp (BI.neg pp))
      (set! qq (BI.neg qq))
    } #f)
    (cond ((BI.is_zero pp) (BI.zero))
          ((== (BI.cmp qq (BI.one)) 0) pp)
          (else {
            (set! g (BI.gcd pp qq))
            (if (not (== (BI.cmp g (BI.one)) 0)) {
              (set! pp (car_of (BI.divmod pp g)))
              (set! qq (car_of (BI.divmod qq g)))
            } #f)
            (if (== (BI.cmp qq (BI.one)) 0) pp `('efrac ,pp ,qq))
          }))
  }))

(define car_of (lambda (x) (get_item x 0)))

;; ---------------------------------------------------------------------------
;; 精确 ↔ 浮点
;; ---------------------------------------------------------------------------

(define to_float
  (lambda (v)
    (cond ((number? v) v)
          ((BI.int? v) (BI.to_float v))
          ((efrac? v) (/ (BI.to_float (fr_p v)) (BI.to_float (fr_q v))))
          (else 0))))

;; ---------------------------------------------------------------------------
;; 四则运算
;; ---------------------------------------------------------------------------

;; 精确加减的公共实现：sg = 1 加、-1 减
(define exact_addsub
  (lambda (a b sg) {
    (define pa (fr_p a))
    (define qa (fr_q a))
    (define pb (if (== sg 1) (fr_p b) (BI.neg (fr_p b))))
    (define qb (fr_q b))
    (if (== (BI.cmp qa qb) 0)
        (fr_make (BI.add pa pb) qa)
        (fr_make (BI.add (BI.mul pa qb) (BI.mul pb qa)) (BI.mul qa qb)))
  }))

(define add
  (lambda (a b)
    (if (and (exact? a) (exact? b))
        (exact_addsub a b 1)
        (+ (to_float a) (to_float b)))))

(define sub
  (lambda (a b)
    (if (and (exact? a) (exact? b))
        (exact_addsub a b -1)
        (- (to_float a) (to_float b)))))

(define mul
  (lambda (a b)
    (if (and (exact? a) (exact? b))
        (fr_make (BI.mul (fr_p a) (fr_p b)) (BI.mul (fr_q a) (fr_q b)))
        (* (to_float a) (to_float b)))))

(define neg
  (lambda (v)
    (cond ((number? v) (- 0 v))
          ((BI.int? v) (BI.neg v))
          ((efrac? v) `('efrac ,(BI.neg (fr_p v)) ,(fr_q v)))
          (else v))))

(define abs
  (lambda (v)
    (cond ((number? v) (Math.abs v))
          ((BI.int? v) (BI.abs v))
          ((efrac? v) (if (< (BI.cmp (fr_p v) (BI.zero)) 0)
                          `('efrac ,(BI.neg (fr_p v)) ,(fr_q v))
                          v))
          (else v))))

(define div
  (lambda (a b)
    (cond ((zero? b) 'err)
          ((and (exact? a) (exact? b))
           (fr_make (BI.mul (fr_p a) (fr_q b)) (BI.mul (fr_q a) (fr_p b))))
          (else (/ (to_float a) (to_float b))))))

;; 向下取整（精确分数用截断除法 + 余数符号修正）
(define floor
  (lambda (v) {
    (define qr (BI.new_list))
    (define qt (BI.zero))
    (define rt (BI.zero))
    (cond ((number? v) (Math.floor v))
          ((BI.int? v) v)
          ((efrac? v) {
            (set! qr (BI.divmod (fr_p v) (fr_q v)))
            (set! qt (car_of qr))
            (set! rt (get_item qr 1))
            (if (and (not (BI.is_zero rt)) (< (BI.cmp (fr_p v) (BI.zero)) 0))
                (BI.sub qt (BI.one))
                qt)
          })
          (else v))
  }))

(define ceil
  (lambda (v)
    (if (number? v) (Math.ceil v) (BI.neg (floor (neg v))))))

;; 向零取整（截断）
(define trunc
  (lambda (v) {
    (define qr (BI.new_list))
    (cond ((number? v) (if (< v 0) (Math.ceil v) (Math.floor v)))
          ((BI.int? v) v)
          ((efrac? v) (get_item (BI.divmod (fr_p v) (fr_q v)) 0))
          (else v))
  }))

;; 取余：a - b·trunc(a/b)，与内建 mod 的 fmod 语义一致（结果的符号随被除数）
(define remainder
  (lambda (a b) {
    (define t 'err)
    (define f (BI.zero))
    (cond ((zero? b) 'err)
          ((or (not (exact? a)) (not (exact? b))) (mod (to_float a) (to_float b)))
          (else {
            (set! t (div a b))
            (set! f (trunc t))
            (sub a (mul f b))
          }))
  }))

;; ---------------------------------------------------------------------------
;; 比较
;; ---------------------------------------------------------------------------

(define cmp
  (lambda (a b) {
    (define d 'err)
    (cond ((or (not (num? a)) (not (num? b))) 0)
          ((or (not (exact? a)) (not (exact? b)))
           (cond ((< (to_float a) (to_float b)) -1)
                 ((> (to_float a) (to_float b)) 1)
                 (else 0)))
          (else {
            (set! d (sub a b))
            (cond ((BI.int? d) (BI.cmp d (BI.zero)))
                  ((efrac? d) (if (< (BI.cmp (fr_p d) (BI.zero)) 0) -1 1))
                  (else 0))
          }))
  }))

;; ---------------------------------------------------------------------------
;; 幂与开方
;; ---------------------------------------------------------------------------

;; 精确底数的整数次幂；指数为负时取倒数
(define exact_pow
  (lambda (a k) {
    (define m k)
    (define r 'err)
    (if (< m 0) (set! m (- 0 m)) #f)
    (if (> m POW_MAX_EXP)
        'overflow
        {
          (set! r (BI.pow_small (fr_p a) m))
          (if (< k 0)
              (fr_make (BI.pow_small (fr_q a) m) r)
              (fr_make r (BI.pow_small (fr_q a) m)))
        })
  }))

(define power
  (lambda (a b) {
    (define k 0)
    (define r 'err)
    (define digits 0)
    (cond ((not (num? a)) 'err)
          ((not (num? b)) 'err)
          ((and (exact? a) (BI.int? b) (not (BI.is_zero a))) {
            (set! k (bi_to_small b))
            (cond ((not k) (power_float a b))
                  ((and (> k 0)
                        (> (* (BI.ndigits (fr_p a)) k) POW_MAX_DIGITS)) (power_float a b))
                  ((and (< k 0)
                        (> (* (BI.ndigits (fr_q a)) (- 0 k)) POW_MAX_DIGITS)) (power_float a b))
                  (else {
                    (set! r (exact_pow a k))
                    (if (eq? r 'overflow) (power_float a b) r)
                  }))
          })
          (else (power_float a b)))
  }))

(define power_float
  (lambda (a b) {
    (define v (Math.pow (to_float a) (to_float b)))
    (if (number? v) v 'err)
  }))

;; 开方：完全平方（含分数）给精确值，否则浮点
(define sqrt
  (lambda (a) {
    (define rp (BI.one))
    (define rq (BI.one))
    (define fp 'err)
    (define fq 'err)
    (cond ((not (num? a)) 'err)
          ((< (cmp a (zero)) 0) 'err)
          ((not (exact? a)) (sqrt_float a))
          ((zero? a) (BI.zero))
          (else {
            (set! rp (BI.isqrt (fr_p a)))
            (set! rq (BI.isqrt (fr_q a)))
            (set! fp (BI.sub (BI.mul rp rp) (fr_p a)))
            (set! fq (BI.sub (BI.mul rq rq) (fr_q a)))
            (if (and (BI.is_zero fp) (BI.is_zero fq))
                (fr_make rp rq)
                (sqrt_float a))
          }))
  }))

;; 精确数的浮点开方兜底：**先缩放再整数开方**，避免 to_float 在 wasm32 上溢出。
;; 取 m 使 p·10^(2m)/q 的量级约 10^16，其整数平方根约 10^8（float 装得下），
;; 再除以 10^m 即 sqrt(p/q) 的约 8 位有效数字。
;; 反例（修之前）：sqrt(10^41) 走 to_float(10^41) → inf → sqrt → inf → 显示"非实数"。
(define sqrt_exact_float
  (lambda (p q) {
    (define e (- (BI.ndigits p) (BI.ndigits q)))
    (define m (- 8 (Math.floor (/ (+ e 1) 2))))
    (define ten (from_small 10))
    (define num p)
    (define den q)
    (define s (BI.zero))
    (if (>= m 0)
        (set! num (BI.mul p (BI.pow_small ten (* 2 m))))
        (set! den (BI.mul q (BI.pow_small ten (* 2 (- 0 m))))))
    (set! s (BI.isqrt (car_of (BI.divmod num den))))
    (/ (BI.to_float s) (Math.pow 10 m))
  }))

;; 有限、非零、非 NaN 的浮点？（wasm32 的 float 上限只有 3.4e38，
;; 精确数很容易越过；判据同样避开"与常数比较"的写法）
(define float_ok?
  (lambda (v)
    (and (number? v) (not (nan? v)) (not (== v 0))
         (not (== (* v 2) v)) (not (== (* v 0.5) v)))))

;; 开方的浮点兜底：先走最直接的 Math.sqrt(to_float x)（截断次数最少、精度最好）；
;; 只有当 x 超出 float 表示范围（上溢成 inf，或下溢成 0）时才改用"缩放 + 整数开方"。
;; 反例（修之前）：sqrt(10^41) 直接 to_float 上溢成 inf，于是显示"非实数"。
(define sqrt_float
  (lambda (a) {
    (define v 'err)
    (if (exact? a)
        {
          (set! v (Math.sqrt (to_float a)))
          (if (not (float_ok? v)) (set! v (sqrt_exact_float (fr_p a) (fr_q a))) #f)
        }
        (set! v (Math.sqrt a)))
    (if (number? v) v 'err)
  }))

;; ---------------------------------------------------------------------------
;; 字面量解析
;; ---------------------------------------------------------------------------

;; 十进制字面量串 → 精确数。接受 [+-]? digits [ . digits? ]
;; （上游中缀 lexer 的 NUMBER 词元就是这个形状：只认数字与小数点，不认指数形式。）
(define parse_literal
  (lambda (s) {
    (define n (String.length s))
    (define i 0)
    (define sg 1)
    (define c 0)
    (define after_dot 0)
    (define fracs 0)
    (define ds (BI.new_list))
    (define ip 'err)
    (if (and (> n 0) (String.equals (String.charAt s 0) "-")) {
      (set! sg -1)
      (set! i 1)
    } (if (and (> n 0) (String.equals (String.charAt s 0) "+")) (set! i 1) #f))
    (while (< i n) {
      (set! c (String.charCodeAt i s))
      (cond ((== c 46) (set! after_dot 1))
            ((and (>= c 48) (<= c 57)) {
               (push ds (- c 48))
               (if (== after_dot 1) (set! fracs (+ fracs 1)) #f)
             })
            (else break))
      (set! i (+ i 1))
    })
    (cond ((== (length ds) 0) 'err)
          ((== fracs 0) (BI.norm sg ds))
          (else {
            (set! ip (BI.norm sg ds))
            (fr_make ip (BI.pow_small (from_small 10) fracs))
          }))
  }))

;; ---------------------------------------------------------------------------
;; 显示
;; ---------------------------------------------------------------------------

;; 浮点 → 显示串（NaN/无穷返回 ""，由界面层显示"非实数"）
;; 浮点 → 显示串（NaN / ±inf 返回 ""，由界面层显示"非实数"）。
;; 注意**不能**用 `(> (Math.abs v) 1e300)` 判无穷：wasm32 宿主上 1e300 这个字面量
;; 本身就是 inf（float 上限 3.4e38），`inf > inf` 为假，于是 inf 会被原样打印成 "inf"。
;; 无穷的判据改成「v≠0 且 2v==v」——对所有有限值都不成立，与宿主浮点范围无关。
(define fmt_float
  (lambda (v)
    (cond ((nan? v) "")
          ((and (not (== v 0)) (== (* v 2) v)) "")
          (else (String.atom_to_string v)))))

;; 精确整数 → 显示串：短则全写，长则科学计数法（保留全部精度靠 Ans/续算）
(define fmt_int
  (lambda (x) {
    (define s (BI.to_string x))
    (if (<= (String.length s) SHOW_MAX)
        s
        (if (< (BI.cmp x (BI.zero)) 0)
            (String.concat "-" (sci_string (BI.abs x) (BI.one)))
            (sci_string x (BI.one))))
  }))

;; 有理数 p/q（p ≥ 0、q ≥ 1）→ 6 位有效数字的科学计数法串，**全程整数运算**。
;; 为什么不用浮点：wasm32 宿主的 float 只有 19 位尾数，10^20 会被显示成 9.99999e+19
;; ——看起来像"算错了"。这里用整数除法直接取前 6 位（多算 1 位用于四舍五入），
;; 于是显示与宿主精度无关，且不会给出比真实值更多或更少的数字。
(define sci_string
  (lambda (p q) {
    (define e0 (- (BI.ndigits p) (BI.ndigits q)))    ;; 科学计数法指数的初估
    (define sh (- 6 e0))                             ;; 需要乘/除 10 的幂次
    (define ten (from_small 10))
    (define m (BI.one))
    (define m6 (BI.one))
    (define ds "")
    (define fr "")
    (define out "")
    (cond ((>= sh 0)
           (set! m (get_item (BI.divmod (BI.mul p (BI.pow_small ten sh)) q) 0)))
          (else
           (set! m (get_item (BI.divmod p (BI.mul q (BI.pow_small ten (- 0 sh)))) 0))))
    ;; 规范到 7 位（多留 1 位保护位）
    (while (> (BI.ndigits m) 7) {
      (set! m (get_item (BI.divmod_small m 10) 0))
      (set! e0 (+ e0 1))
    })
    (while (< (BI.ndigits m) 7) {
      (set! m (BI.mul m ten))
      (set! e0 (- e0 1))
    })
    ;; 四舍五入到 6 位
    (set! m6 (get_item (BI.divmod_small (BI.add m (from_small 5)) 10) 0))
    (if (> (BI.ndigits m6) 6) {
      (set! m6 (get_item (BI.divmod_small m6 10) 0))
      (set! e0 (+ e0 1))
    } #f)
    (set! ds (BI.to_string m6))
    (set! fr (String.slice ds 1 (String.length ds)))
    ;; 去掉尾随零（"1.20000" → "1.2"）
    (while (and (> (String.length fr) 0)
                (String.equals "0" (String.slice fr (- (String.length fr) 1) (String.length fr)))) {
      (set! fr (String.slice fr 0 (- (String.length fr) 1)))
    })
    (set! out (String.slice ds 0 1))
    (cond ((== e0 0) {
            ;; 指数为 0：就是普通小数（pi*pi → 9.8696，而不是 9.8696e+0）
            (set! out (String.concat (String.slice ds 0 1)
                                     (String.concat "." (String.slice ds 1 (String.length ds)))))
            (set! out (strip_zeros out))
          })
          (else {
            (if (> (String.length fr) 0) (set! out (String.concat out (String.concat "." fr))) #f)
            (set! out (String.concat out (sci_exp e0)))
          }))
    out
  }))

;; 去掉 "9.8696" 这类串尾部的零与孤立小数点
(define strip_zeros
  (lambda (s) {
    (define n (String.length s))
    (while (and (> n 1)
                (String.equals "0" (String.slice s (- n 1) n))) {
      (set! n (- n 1))
    })
    (if (and (> n 1) (String.equals "." (String.slice s (- n 1) n)))
        (set! n (- n 1))
        #f)
    (String.slice s 0 n)
  }))

;; 科学计数法的指数后缀：20 → "+20"，-5 → "-5"（与 C 的 %g 一致）
(define sci_exp
  (lambda (e)
    (if (< e 0)
        (String.concat "e-" (BI.to_string (from_small (- 0 e))))
        (String.concat "e+" (BI.to_string (from_small e))))))

;; 分数 → 显示串：
;;   1) 有限小数且不太长 → 写成小数（0.5、"3.14"）；
;;   2) "p/q" 不太长 → 写成既约分数（1/3、22/7）；
;;   3) 否则科学计数法兜底。
(define fmt_frac
  (lambda (x) {
    (define p (fr_p x))
    (define q (fr_q x))
    (define r2 'err)
    (define r5 'err)
    (define a 0)
    (define b 0)
    (define k 0)
    (define m 'err)
    (define s "")
    (define fs "")
    (define neg 0)
    (set! r2 (count_factor q 2))
    (set! a (get_item r2 1))
    (set! r5 (count_factor (get_item r2 0) 5))
    (set! b (get_item r5 1))
    (cond
      ;; 分母只剩 2、5 因子 → 有限小数
      ((== (BI.cmp (get_item r5 0) (BI.one)) 0) {
         (set! k (if (> a b) a b))
         (set! m (BI.abs p))
         (if (> (- k a) 0) (set! m (BI.mul m (BI.pow_small (from_small 2) (- k a)))) #f)
         (if (> (- k b) 0) (set! m (BI.mul m (BI.pow_small (from_small 5) (- k b)))) #f)
         (set! s (BI.to_string m))
         (set! fs (place_point s k))
         (set! neg (if (< (BI.cmp p (BI.zero)) 0) 1 0))
         (if (== neg 1) (set! fs (String.concat "-" fs)) #f)
         (if (<= (String.length fs) SHOW_MAX)
             fs
             (fmt_frac_big p q))
       })
      (else (fmt_frac_big p q)))
  }))

;; 分数退化为 "p/q" 或科学计数法
(define fmt_frac_big
  (lambda (p q) {
    (define s (String.concat (BI.to_string p) (String.concat "/" (BI.to_string q))))
    (if (<= (String.length s) SHOW_MAX)
        s
        (if (< (BI.cmp p (BI.zero)) 0)
            (String.concat "-" (sci_string (BI.abs p) q))
            (sci_string p q)))
  }))

;; 把纯数字串 s 的小数点放在倒数第 k 位（k ≥ 1）
(define place_point
  (lambda (s k) {
    (define n (String.length s))
    (define head "")
    (define tail "")
    (if (> n k)
        (String.concat (String.concat (String.slice s 0 (- n k)) ".") (String.slice s (- n k) n))
        (String.concat (String.concat "0." (zeros (- k n))) s))
  }))

;; n 个 "0"
(define zeros
  (lambda (n) {
    (define i 0)
    (define out "")
    (while (< i n) {
      (set! out (String.concat out "0"))
      (set! i (+ i 1))
    })
    out
  }))

;; 反复除以小因子 m，返回 (剩余 次数)
(define count_factor
  (lambda (x m) {
    (define cur x)
    (define c 0)
    (define qr (BI.new_list))
    (if (BI.is_zero x)
        (cons x (cons 0 '()))
        {
          (set! qr (BI.divmod_small cur m))
          (while (== (get_item qr 1) 0) {
            (set! cur (get_item qr 0))
            (set! c (+ c 1))
            (set! qr (BI.divmod_small cur m))
          })
          (cons cur (cons c '()))
        })
  }))

;; 数值 → 显示串
(define to_string
  (lambda (v)
    (cond ((not (num? v)) "")
          ((number? v) (fmt_float v))
          ((BI.int? v) (fmt_int v))
          ((efrac? v) (fmt_frac v))
          (else ""))))
