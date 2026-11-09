;; ============================================================================
;; app/lib/bigint.scm —— 任意精度十进制大整数（含 FFT 卷积乘法路径）
;;
;; 为什么需要它：灵机的整数是「机器字长减 5 位类型标签」的立即值（64 位宿主 59 位，
;; **wasm32 宿主只有 27 位**），且算术**不做溢出检查**（`(* 123456789012345 1000000)`
;; 静默给出错误结果）；浮点是 IEEE-754（wasm32 宿主为 32 位 float，且 am 层再截掉
;; 低 5 位尾数 → 约 19 位 ≈ 6 位有效数字）。所以「精确大整数 / 精确分数」只能在
;; Scheme 层用十进制数字列表实现。本文件只做**整数**，分数与数值塔见
;; app/lib/numtower.scm。
;;
;; 表示：
;;   ('eint sign digits)   sign ∈ {1, -1}；digits = (d1 d2 ... dn) 十进制数字，
;;                         最高位在前、无前导零（只有 0 时 digits = (0)）。
;;   **数字列表一经构造即视为不可变**：push 只用于本地新建的列表，绝不 push 到入参。
;;
;; 对外接口（全部为值语义，返回新的 eint）：
;;   BI.int? v              是否为大整数
;;   BI.make sign digits    构造（**不做**规范化，调用方保证 digits 已规范）
;;   BI.norm sign digits    规范化构造（去前导零、零一律记作 +0）——解析时用这个
;;   BI.zero / BI.one       零 / 一（函数，每次返回新的 eint）
;;   BI.is_zero / BI.is_even / BI.ndigits / BI.sign / BI.digits
;;   BI.from_string s       十进制字符串（可带 +/-）→ 大整数
;;   BI.to_string x         大整数 → 十进制字符串（精确、无指数形式）
;;   BI.to_float x          近似浮点（取前 7 位有效数字；仅供混合运算兜底）
;;   BI.abs / BI.neg / BI.cmp
;;   BI.add / BI.sub / BI.mul
;;   BI.divmod a b          (商 余数) 二元素列表；商向零取整，余数符号随被除数
;;   BI.rem a b             同 divmod 的余数
;;   BI.divmod_small a m    除以原生小整数 m（1..99）
;;   BI.pow_small x k       x^k（k 为原生小非负整数，平方-乘）
;;   BI.gcd a b             最大公约数（二进制 GCD，非负）
;;   BI.isqrt x             floor(sqrt(x))（整数牛顿迭代，x ≥ 0）
;;   BI.new_list            运行期新建空列表（`'()` 是静态共享对象，不能拿来 push）
;;   BI.mul_school / BI.mul_kara / BI.mul_fft  ×2  → 三条乘法路径（供测试与自检）
;;
;; 三条乘法路径（BI.mul 自动选择，见 mul_dispatch）：
;;   1) 学校式竖式 O(n²)：位数少时最快（常数小、无额外分配）；
;;   2) Karatsuba O(n^1.585)：**纯整数运算，任何宿主都精确**，大数运算主力；
;;   3) FFT 卷积 O(n log n)：卷积定理 a·b = IFFT(FFT(a)·FFT(b))，数字序列的线性
;;      卷积正是「未进位的按位系数」，再做一趟进位规范化即得乘积。思路与上游
;;      Animac 用例 test/big_int.scm + test/fft.scm 一致，但这里的实现在
;;      set_item!/get_item 上做**原地迭代式 radix-2 FFT**（上游是递归 + 列表拼接，
;;      每层 O(n²)），实部/虚部用两条平行列表（不为每个点分配二元素列表）。
;;      **但它是浮点算法**：舍入误差 ~ eps·log2(N)·81·sqrt(la·lb)，wasm32 宿主
;;      float 只有 19 位尾数（eps ≈ 1.9e-6），所以只能在小规模上保证卷积系数正确
;;      取整。故 FFT 路径由 fft_ok? 做「误差预算」门控（按宿主实测精度自适应，
;;      64 位双精度宿主上可覆盖任意大的操作数），并在算完后做一次**模小素数校验**
;;      （fft_verify），校验不过就退回 Karatsuba —— 因此**大数乘法永远精确**。
;; ============================================================================

(native Math)
(native String)

;; 学校式与 Karatsuba 的分界（十进制位数）：低于此值竖式更快。
;; 实测（wasm32 宿主，n×n 位乘法，单位 ms）：n=256 时两者相当（29/30），
;; n=320 起 Karatsuba 开始领先（23/21），n=512 是 63/49，n=1024 是 543/约 300。
;; Karatsuba 在这台 VM 上的常数不小（每层都要切分/拼接数字列表），
;; 故阈值取实测交叉点，避免「为了算法优雅把常用规模算慢了」。
(define KARA_MIN 256)

;; ---------------------------------------------------------------------------
;; 数字列表（digits，最高位在前）
;; ---------------------------------------------------------------------------

(define ds_zero '(0))
(define ds_one '(1))

;; **运行期**新建空列表。注意：源码里的 '() 字面量是**静态共享对象**——同一个字面量
;; 在每次调用时都是同一个列表对象，对它 push 会就地增长、跨调用累积（实测
;; (define f (lambda () { (define o (new_list)) (push o 1) o })) 连调三次得 (1) (1 1) (1 1 1)）。
;; 所以凡是要 push 的列表，一律从下面的 new_list 开始。
(define new_list (lambda () (cdr (cons 0 '()))))

;; 规范化：去掉前导零（返回新列表；入参已规范时原样返回，共享是安全的，因为
;; 数字列表不可变）
(define ds_norm
  (lambda (ds) {
    (define n (length ds))
    (define i 0)
    (define out (new_list))
    (if (== n 0)
        ds_zero
        {
          (while (and (< i (- n 1)) (== (get_item ds i) 0)) (set! i (+ i 1)))
          (if (== i 0)
              ds
              {
                (while (< i n) {
                  (push out (get_item ds i))
                  (set! i (+ i 1))
                })
                out
              })
        })
  }))

;; 反转数字列表
(define ds_rev
  (lambda (ds) {
    (define n (length ds))
    (define i (- n 1))
    (define out (new_list))
    (while (>= i 0) {
      (push out (get_item ds i))
      (set! i (- i 1))
    })
    out
  }))

(define ds_zero? (lambda (ds) (and (== (length ds) 1) (== (get_item ds 0) 0))))

;; 偶数？（末位 % 2 == 0）
(define ds_even? (lambda (ds) (== (mod (get_item ds (- (length ds) 1)) 2) 0)))

;; 比较（两个都已规范、非负）：-1 / 0 / 1
(define ds_cmp
  (lambda (a b) {
    (define la (length a))
    (define lb (length b))
    (define i 0)
    (define r 0)
    (cond ((< la lb) (set! r -1))
          ((> la lb) (set! r 1))
          (else
           (while (and (== r 0) (< i la)) {
             (if (< (get_item a i) (get_item b i))
                 (set! r -1)
                 (if (> (get_item a i) (get_item b i)) (set! r 1) #f))
             (set! i (+ i 1))
           })))
    r
  }))

;; 无符号加
(define ds_add
  (lambda (a b) {
    (define ia (- (length a) 1))
    (define ib (- (length b) 1))
    (define carry 0)
    (define s 0)
    (define out (new_list))
    (while (or (>= ia 0) (>= ib 0)) {
      (set! s carry)
      (if (>= ia 0) {
        (set! s (+ s (get_item a ia)))
        (set! ia (- ia 1))
      } #f)
      (if (>= ib 0) {
        (set! s (+ s (get_item b ib)))
        (set! ib (- ib 1))
      } #f)
      (if (>= s 10) {
        (set! s (- s 10))
        (set! carry 1)
      } (set! carry 0))
      (push out s)
    })
    (if (== carry 1) (push out 1) #f)
    (ds_rev out)
  }))

;; 无符号减（要求 a >= b）
(define ds_sub
  (lambda (a b) {
    (define ia (- (length a) 1))
    (define ib (- (length b) 1))
    (define borrow 0)
    (define d 0)
    (define out (new_list))
    (while (>= ia 0) {
      (set! d (- (get_item a ia) borrow))
      (if (>= ib 0) {
        (set! d (- d (get_item b ib)))
        (set! ib (- ib 1))
      } #f)
      (if (< d 0) {
        (set! d (+ d 10))
        (set! borrow 1)
      } (set! borrow 0))
      (push out d)
      (set! ia (- ia 1))
    })
    (ds_norm (ds_rev out))
  }))

;; 乘一个原生小整数（m < 100）
(define ds_mul_small
  (lambda (ds m) {
    (define n (length ds))
    (define i (- n 1))
    (define carry 0)
    (define v 0)
    (define out (new_list))
    (while (>= i 0) {
      (set! v (+ (* (get_item ds i) m) carry))
      (set! carry (Math.floor (/ v 10)))
      (push out (- v (* carry 10)))
      (set! i (- i 1))
    })
    (while (> carry 0) {
      (push out (mod carry 10))
      (set! carry (Math.floor (/ carry 10)))
    })
    (ds_norm (ds_rev out))
  }))

;; 除以 2（无符号）
(define ds_halve
  (lambda (ds) {
    (define n (length ds))
    (define i 0)
    (define rem 0)
    (define cur 0)
    (define out (new_list))
    (while (< i n) {
      (set! cur (+ (* rem 10) (get_item ds i)))
      (push out (Math.floor (/ cur 2)))
      (set! rem (mod cur 2))
      (set! i (+ i 1))
    })
    (ds_norm out)
  }))

;; 除以原生小整数（1 <= m < 100）：返回 (商 余数)，余数为原生小整数
(define ds_divmod_small
  (lambda (ds m) {
    (define n (length ds))
    (define i 0)
    (define rem 0)
    (define cur 0)
    (define out (new_list))
    (while (< i n) {
      (set! cur (+ (* rem 10) (get_item ds i)))
      (push out (Math.floor (/ cur m)))
      (set! rem (mod cur m))
      (set! i (+ i 1))
    })
    (cons (ds_norm out) (cons rem '()))
  }))

;; 追加一位（x → 10x + d）
(define ds_append_digit
  (lambda (ds d) {
    (define n (length ds))
    (define i 0)
    (define out (new_list))
    (while (< i n) {
      (push out (get_item ds i))
      (set! i (+ i 1))
    })
    (push out d)
    (ds_norm out)
  }))

;; 左移 k 位十进制（x → x·10^k）
(define ds_shift
  (lambda (ds k) {
    (define n (length ds))
    (define i 0)
    (define out (new_list))
    (while (< i n) {
      (push out (get_item ds i))
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i k) {
      (push out 0)
      (set! i (+ i 1))
    })
    (ds_norm out)
  }))

;; 高位部分（去掉低 k 位）
(define ds_high
  (lambda (ds k) {
    (define e (- (length ds) k))
    (define i 0)
    (define out (new_list))
    (if (<= e 0)
        ds_zero
        {
          (while (< i e) {
            (push out (get_item ds i))
            (set! i (+ i 1))
          })
          (ds_norm out)
        })
  }))

;; 低位部分（保留低 k 位，去掉前导零）
(define ds_low
  (lambda (ds k) {
    (define n (length ds))
    (define st (- n k))
    (define i 0)
    (define out (new_list))
    (if (<= st 0)
        (ds_norm ds)
        {
          (set! i st)
          (while (< i n) {
            (push out (get_item ds i))
            (set! i (+ i 1))
          })
          (ds_norm out)
        })
  }))

;; ---------------------------------------------------------------------------
;; 乘法路径 1：学校式竖式 O(n²)
;; ---------------------------------------------------------------------------

(define ds_mul_school
  (lambda (a b) {
    (define la (length a))
    (define lb (length b))
    (define acc (new_list))
    (define i 0)
    (define j 0)
    (define ai 0)
    (define v 0)
    (define carry 0)
    (define out (new_list))
    ;; acc 为「低位在前」的未进位系数表，长度 la+lb
    (while (< i (+ la lb)) {
      (push acc 0)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i la) {
      (set! ai (get_item a (- (- la 1) i)))
      (set! j 0)
      (while (< j lb) {
        (set! v (+ (get_item acc (+ i j)) (* ai (get_item b (- (- lb 1) j)))))
        (set_item! acc (+ i j) v)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    ;; 进位规范化
    (set! i 0)
    (while (< i (length acc)) {
      (set! v (+ (get_item acc i) carry))
      (set! carry (Math.floor (/ v 10)))
      (push out (- v (* carry 10)))
      (set! i (+ i 1))
    })
    (while (> carry 0) {
      (push out (mod carry 10))
      (set! carry (Math.floor (/ carry 10)))
    })
    (ds_norm (ds_rev out))
  }))

;; ---------------------------------------------------------------------------
;; 乘法路径 2：Karatsuba O(n^1.585)（纯整数，永远精确）
;;   a = a1·10^m + a0, b = b1·10^m + b0
;;   a·b = z2·10^2m + ((a0+a1)(b0+b1) - z0 - z2)·10^m + z0
;; ---------------------------------------------------------------------------

(define ds_mul_kara
  (lambda (a b) {
    (define la (length a))
    (define lb (length b))
    (define n 0)
    (define m 0)
    (define a0 ds_zero)
    (define a1 ds_zero)
    (define b0 ds_zero)
    (define b1 ds_zero)
    (define z0 ds_zero)
    (define z1 ds_zero)
    (define z2 ds_zero)
    (if (or (< la KARA_MIN) (< lb KARA_MIN))
        (ds_mul_school a b)
        {
          (set! n (if (> la lb) la lb))
          (set! m (Math.floor (/ n 2)))
          (set! a0 (ds_low a m))
          (set! a1 (ds_high a m))
          (set! b0 (ds_low b m))
          (set! b1 (ds_high b m))
          (set! z0 (ds_mul_kara a0 b0))
          (set! z2 (ds_mul_kara a1 b1))
          (set! z1 (ds_sub (ds_sub (ds_mul_kara (ds_add a0 a1) (ds_add b0 b1)) z0) z2))
          (ds_add (ds_shift z2 (* 2 m)) (ds_add (ds_shift z1 m) z0))
        })
  }))

;; ---------------------------------------------------------------------------
;; 乘法路径 3：FFT 卷积（卷积定理）
;; ---------------------------------------------------------------------------

;; 本宿主浮点的相对精度（1 附近的最小可分辨增量），首次使用时实测：
;; 64 位宿主 ≈ 1.1e-16（双精度）；wasm32 宿主 ≈ 1.9e-6（19 位有效尾数）。
;; 探测法：不断把 r 减半，直到 1+r == 1 —— 此时 r 已小于半个 ulp。
(define probe_float_eps
  (lambda () {
    (define r 0.5)
    (define k 0)
    (while (and (< k 200) (not (== (+ 1 r) 1))) {
      (set! r (/ r 2))
      (set! k (+ k 1))
    })
    r
  }))

;; 模块加载时实测一次（64 位双精度宿主 ≈ 3.6e-15；wasm32 宿主 ≈ 1.9e-6）。
;; 注意必须在这里**立即求值**：fft_ok 用的是这个值，若保持 0 会让门控永远放行。
(define float_eps (probe_float_eps))

;; 位反转置换表（长度 n = 2^k）
(define fft_bitrev
  (lambda (n) {
    (define out (new_list))
    (define i 0)
    (define j 0)
    (define bit 0)
    (while (< i n) {
      (push out j)
      (set! bit (/ n 2))
      (while (and (> bit 0) (>= j bit)) {
        (set! j (- j bit))
        (set! bit (/ bit 2))
      })
      (set! j (+ j bit))
      (set! i (+ i 1))
    })
    out
  }))

;; 原地 radix-2 FFT：re/im 等长（长度 n = 2^k）；inv = 1 时做逆变换（含 1/n 缩放）
(define fft_core
  (lambda (re im n inv) {
    (define tab (fft_bitrev n))
    (define i 0)
    (define j 0)
    (define t 0)
    (define len 2)
    (define half 1)
    (define ang 0)
    (define wr 0)
    (define wi 0)
    (define cr 0)
    (define ci 0)
    (define ur 0)
    (define ui 0)
    (define vr 0)
    (define vi 0)
    (define p 0)
    (define q 0)
    (define k 0)
    ;; 1) 位反转置换
    (while (< i n) {
      (set! j (get_item tab i))
      (if (< i j) {
        (set! t (get_item re i))
        (set_item! re i (get_item re j))
        (set_item! re j t)
        (set! t (get_item im i))
        (set_item! im i (get_item im j))
        (set_item! im j t)
      } #f)
      (set! i (+ i 1))
    })
    ;; 2) 蝶形
    (while (<= len n) {
      (set! half (/ len 2))
      (set! ang (/ (* 2 (Math.PI)) len))
      (if (== inv 1) #f (set! ang (- 0 ang)))
      (set! wr (Math.cos ang))
      (set! wi (Math.sin ang))
      (set! i 0)
      (while (< i n) {
        (set! cr 1)
        (set! ci 0)
        (set! k 0)
        (while (< k half) {
          (set! p (+ i k))
          (set! q (+ p half))
          (set! ur (get_item re p))
          (set! ui (get_item im p))
          (set! vr (- (* (get_item re q) cr) (* (get_item im q) ci)))
          (set! vi (+ (* (get_item re q) ci) (* (get_item im q) cr)))
          (set_item! re p (+ ur vr))
          (set_item! im p (+ ui vi))
          (set_item! re q (- ur vr))
          (set_item! im q (- ui vi))
          ;; 旋转因子递推：w ← w·wlen
          (set! t (- (* cr wr) (* ci wi)))
          (set! ci (+ (* cr wi) (* ci wr)))
          (set! cr t)
          (set! k (+ k 1))
        })
        (set! i (+ i len))
      })
      (set! len (* len 2))
    })
    ;; 3) 逆变换缩放
    (if (== inv 1)
        {
          (set! i 0)
          (while (< i n) {
            (set_item! re i (/ (get_item re i) n))
            (set_item! im i (/ (get_item im i) n))
            (set! i (+ i 1))
          })
        }
        #f)
  }))

;; 数字列表（最高位在前）→ 长度 n 的实部向量（低位在前，尾部补零）；虚部全 0
(define fft_real_vec
  (lambda (ds n) {
    (define m (length ds))
    (define i 0)
    (define out (new_list))
    (while (< i n) {
      (if (< i m) (push out (get_item ds (- (- m 1) i))) (push out 0))
      (set! i (+ i 1))
    })
    out
  }))

(define fft_zero_vec
  (lambda (n) {
    (define i 0)
    (define out (new_list))
    (while (< i n) {
      (push out 0)
      (set! i (+ i 1))
    })
    out
  }))

;; FFT 卷积乘法：a、b 为无符号（规范）数字列表
(define ds_mul_fft
  (lambda (a b) {
    (define n 1)
    (define ra (new_list))
    (define ia (new_list))
    (define rb (new_list))
    (define ib (new_list))
    (define i 0)
    (define tr 0)
    (define ti 0)
    (define carry 0)
    (define v 0)
    (define out (new_list))
    (while (< n (+ (length a) (length b))) (set! n (* n 2)))
    (set! ra (fft_real_vec a n))
    (set! ia (fft_zero_vec n))
    (set! rb (fft_real_vec b n))
    (set! ib (fft_zero_vec n))
    (fft_core ra ia n 0)
    (fft_core rb ib n 0)
    ;; 逐点相乘（复数乘法），结果写回 ra/ia
    (set! i 0)
    (while (< i n) {
      (set! tr (get_item ra i))
      (set! ti (get_item ia i))
      (set_item! ra i (- (* tr (get_item rb i)) (* ti (get_item ib i))))
      (set_item! ia i (+ (* tr (get_item ib i)) (* ti (get_item rb i))))
      (set! i (+ i 1))
    })
    (fft_core ra ia n 1)
    ;; 取整 + 进位规范化（ra 为低位在前的卷积系数）
    (set! i 0)
    (while (< i n) {
      (set! v (+ (Math.round (get_item ra i)) carry))
      (set! carry (Math.floor (/ v 10)))
      (push out (- v (* carry 10)))
      (set! i (+ i 1))
    })
    (while (> carry 0) {
      (push out (mod carry 10))
      (set! carry (Math.floor (/ carry 10)))
    })
    (ds_norm (ds_rev out))
  }))

;; 误差预算门控：卷积系数 |c_k| ≤ 81·min(la,lb)，FFT 的相对误差 ≈ eps·log2(n)·K，
;; 于是系数绝对误差 ≈ 81·K·eps·log2(n)·sqrt(la·lb)（Cauchy-Schwarz 界）。
;; 要求它 < FFT_ERR_MAX（远小于 0.5 的取整半径），否则走 Karatsuba。
(define FFT_ERR_MAX 0.02)
(define FFT_ERR_K 4)

(define fft_ok
  (lambda (la lb) {
    (define n 1)
    (define lg 0)
    (define est 0)
    (while (< n (+ la lb)) {
      (set! n (* n 2))
      (set! lg (+ lg 1))
    })
    (set! est (* 81 FFT_ERR_K))
    (set! est (* est (Math.sqrt (* la lb))))
    (set! est (* est float_eps))
    (set! est (* est (+ lg 1)))
    (< est FFT_ERR_MAX)
  }))

;; ---------------------------------------------------------------------------
;; 模校验：把 a、b、积分别取模若干个小素数，验证乘积（O(n)，与乘法本身同阶）
;; 素数都 < 256，故 a·b < 2^18 —— 在 wasm32 的 27 位整数与 19 位浮点下都精确。
;; ---------------------------------------------------------------------------

(define verify_primes '(251 241 233 229 227 223))

(define ds_mod_small
  (lambda (ds m) {
    (define n (length ds))
    (define i 0)
    (define r 0)
    (while (< i n) {
      (set! r (mod (+ (* r 10) (get_item ds i)) m))
      (set! i (+ i 1))
    })
    r
  }))

(define fft_verify
  (lambda (r a b) {
    (define ps verify_primes)
    (define ok #t)
    (define p 0)
    (define ma 0)
    (define mb 0)
    (define mr 0)
    (while (and ok (not (null? ps))) {
      (set! p (car ps))
      (set! ma (ds_mod_small a p))
      (set! mb (ds_mod_small b p))
      (set! mr (ds_mod_small r p))
      (if (not (== (mod (* ma mb) p) mr)) (set! ok #f) #f)
      (set! ps (cdr ps))
    })
    ok
  }))

;; ---------------------------------------------------------------------------
;; 乘法调度
;; ---------------------------------------------------------------------------

(define ds_mul
  (lambda (a b) {
    (define la (length a))
    (define lb (length b))
    (define r ds_zero)
    (cond ((or (< la KARA_MIN) (< lb KARA_MIN)) (ds_mul_school a b))
          (else
           (if (fft_ok la lb)
               {
                 (set! r (ds_mul_fft a b))
                 (if (fft_verify r a b) r (ds_mul_kara a b))
               }
               (ds_mul_kara a b))))
  }))

;; ---------------------------------------------------------------------------
;; 除法 / 取余 / 幂 / 平方根 / GCD
;; ---------------------------------------------------------------------------

;; 无符号除法：返回 (商 余数)
(define ds_divmod
  (lambda (a b) {
    (define na (length a))
    (define i 0)
    (define r ds_zero)
    (define q (new_list))
    (define d 0)
    (while (< i na) {
      (set! r (ds_append_digit r (get_item a i)))
      (set! d 0)
      (while (>= (ds_cmp r b) 0) {
        (set! r (ds_sub r b))
        (set! d (+ d 1))
      })
      (push q d)
      (set! i (+ i 1))
    })
    (cons (ds_norm q) (cons (ds_norm r) '()))
  }))

;; 二进制 GCD（Stein）：只用减半/减法/比较，不需除法
(define ds_gcd
  (lambda (a b) {
    (define x a)
    (define y b)
    (define k 0)
    (while (and (not (ds_zero? x)) (not (ds_zero? y))) {
      (cond ((and (ds_even? x) (ds_even? y)) {
               (set! x (ds_halve x))
               (set! y (ds_halve y))
               (set! k (+ k 1))
             })
            ((ds_even? x) (set! x (ds_halve x)))
            ((ds_even? y) (set! y (ds_halve y)))
            ((>= (ds_cmp x y) 0) (set! x (ds_halve (ds_sub x y))))
            (else (set! y (ds_halve (ds_sub y x))))
      )
    })
    ;; 结果 = 非零的那一个 · 2^k（注意：是二进制左移 k 位，不是十进制左移）
    (if (ds_zero? x) (set! x y) #f)
    (while (> k 0) {
      (set! x (ds_mul_small x 2))
      (set! k (- k 1))
    })
    x
  }))

;; 整数平方根 floor(sqrt(x))，x ≥ 0：整数牛顿迭代（初值 10^ceil(n/2) ≥ sqrt(x)）
(define ds_isqrt
  (lambda (x) {
    (define n (length x))
    (define g ds_one)
    (define y ds_zero)
    (define i 0)
    (if (ds_zero? x)
        ds_zero
        {
          ;; 初值：1 后面跟 ceil(n/2) 个 0
          (set! g (ds_shift ds_one (Math.floor (/ (+ n 1) 2))))
          (while #t {
            (set! y (ds_halve (ds_add g (car (ds_divmod x g)))))
            (if (>= (ds_cmp y g) 0) break)
            (set! g y)
          })
          g
        })
  }))

;; ---------------------------------------------------------------------------
;; 有符号大整数
;; ---------------------------------------------------------------------------

(define int?
  (lambda (v) (and (list? v) (== (length v) 3) (eq? (car v) 'eint))))

(define make (lambda (sg ds) `('eint ,sg ,ds)))

(define sign (lambda (x) (get_item x 1)))

(define digits (lambda (x) (get_item x 2)))

;; 规范化构造：零一律记作 (+1, (0))
(define norm
  (lambda (sg ds) {
    (define d (ds_norm ds))
    (if (ds_zero? d) (make 1 ds_zero) (make sg d))
  }))

(define zero (lambda () (make 1 ds_zero)))
(define one (lambda () (make 1 ds_one)))

(define is_zero (lambda (x) (ds_zero? (digits x))))
(define is_even (lambda (x) (ds_even? (digits x))))
(define ndigits (lambda (x) (length (digits x))))

(define abs (lambda (x) (make 1 (digits x))))

(define neg (lambda (x) (if (is_zero x) x (make (- 0 (sign x)) (digits x)))))

;; 原生小整数 → 大整数（要求 |n| < 100000：再大就要靠字符串，见 from_string）
(define from_string
  (lambda (s) {
    (define n (String.length s))
    (define i 0)
    (define sg 1)
    (define c 0)
    (define ds (new_list))
    (if (and (> n 0) (String.equals (String.charAt s 0) "-")) {
      (set! sg -1)
      (set! i 1)
    } (if (and (> n 0) (String.equals (String.charAt s 0) "+")) (set! i 1) #f))
    (while (< i n) {
      (set! c (String.charCodeAt i s))
      (if (and (>= c 48) (<= c 57))
          (push ds (- c 48))
          #f)
      (set! i (+ i 1))
    })
    (norm sg ds)
  }))

;; 大整数 → 十进制字符串（精确，无指数形式）
(define to_string
  (lambda (x) {
    (define ds (digits x))
    (define n (length ds))
    (define i 0)
    (define out "")
    (if (== (sign x) -1) (set! out "-") #f)
    (while (< i n) {
      (set! out (String.concat out (String.fromCharCode (+ 48 (get_item ds i)))))
      (set! i (+ i 1))
    })
    out
  }))

;; 近似浮点（只供显示兜底与混合运算）：取前 7 位有效数字再乘 10 的幂。
;; 只取 7 位是**有意**的：wasm32 宿主的 float 只有 19 位尾数（≈6 位有效数字），
;; 取更多位反而会引入截断误差，让 10^20 显示成 9.99991e+19。显示只要 6 位足够
;; （%g 也只输出 6 位有效数字）。
(define to_float
  (lambda (x) {
    (define ds (digits x))
    (define n (length ds))
    (define take (if (> n 7) 7 n))
    (define i 0)
    (define m 0)
    (while (< i take) {
      (set! m (+ (* m 10) (get_item ds i)))
      (set! i (+ i 1))
    })
    (if (> n take) (set! m (* m (Math.pow 10 (- n take)))) #f)
    (if (== (sign x) -1) (- 0 m) m)
  }))

(define cmp
  (lambda (a b) {
    (define sa (sign a))
    (define sb (sign b))
    (define za (is_zero a))
    (define zb (is_zero b))
    (define c 0)
    (cond ((and za zb) 0)
          (za (if (== sb -1) 1 -1))
          (zb (if (== sa -1) -1 1))
          ((and (== sa 1) (== sb -1)) 1)
          ((and (== sa -1) (== sb 1)) -1)
          (else {
            (set! c (ds_cmp (digits a) (digits b)))
            (if (== sa 1) c (- 0 c))
          }))
  }))

(define add
  (lambda (a b) {
    (define sa (sign a))
    (define sb (sign b))
    (define c 0)
    (cond ((== sa sb) (norm sa (ds_add (digits a) (digits b))))
          (else {
            (set! c (ds_cmp (digits a) (digits b)))
            (cond ((== c 0) (make 1 ds_zero))
                  ((> c 0) (norm sa (ds_sub (digits a) (digits b))))
                  (else (norm sb (ds_sub (digits b) (digits a)))))
          }))
  }))

(define sub (lambda (a b) (add a (neg b))))

(define mul
  (lambda (a b) {
    (define d (ds_mul (digits a) (digits b)))
    (if (ds_zero? d) (make 1 d) (make (* (sign a) (sign b)) d))
  }))

;; 三种乘法路径的直接入口（同号、无符号语义；供测试与自检交叉验证）
(define mul_school
  (lambda (a b) (make (* (sign a) (sign b)) (ds_mul_school (digits a) (digits b)))))
(define mul_kara
  (lambda (a b) (make (* (sign a) (sign b)) (ds_mul_kara (digits a) (digits b)))))
(define mul_fft
  (lambda (a b) (make (* (sign a) (sign b)) (ds_mul_fft (digits a) (digits b)))))

;; 截断除法：返回 (商 余数) 二元素列表；商向零取整，余数符号随 a
(define divmod
  (lambda (a b) {
    (define qr (ds_divmod (digits a) (digits b)))
    (define q (car qr))
    (define r (get_item qr 1))
    (cons (norm (* (sign a) (sign b)) q) (cons (norm (sign a) r) '()))
  }))

(define rem
  (lambda (a b) (get_item (divmod a b) 1)))

;; 原生小整数除法：返回 (商 余数)，余数为原生小整数
(define divmod_small
  (lambda (a m) {
    (define qr (ds_divmod_small (digits a) m))
    (cons (norm (sign a) (car qr)) (cons (get_item qr 1) '()))
  }))

;; x^k（k 为原生小非负整数），平方-乘
(define pow_small
  (lambda (x k) {
    (define r (one))
    (define base x)
    (define e k)
    (while (> e 0) {
      (if (== (mod e 2) 1) (set! r (mul r base)) #f)
      (set! e (Math.floor (/ e 2)))
      (if (> e 0) (set! base (mul base base)) #f)
    })
    r
  }))

(define gcd
  (lambda (a b) (make 1 (ds_gcd (digits a) (digits b)))))

(define isqrt (lambda (x) (make 1 (ds_isqrt (digits x)))))
