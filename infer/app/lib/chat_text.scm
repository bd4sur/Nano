;; ============================================================================
;; chat_text.scm —— 对话文本排版工具（硬换行 + 软折行 + 控制字符）
;;
;; 为什么单独成模块：对话类 App 的"文本 → 屏幕行"这一步有两类容易出错的地方——
;;   ① **换行符**：模型的输出里经常带 '\n'。12px 字模没有 '\n' 的字形，直接把整段
;;      文本丢给 Gfx.draw_text 会"什么也不画"，看起来像换行被吃掉了；必须按 '\n'
;;      切成硬行再逐行绘制。
;;   ② **软折行**：按半角单位宽度折行（半角 1、全角 2，与 12px 字模的步进一致）。
;; 把它抽成模块后可以用最小宿主直接单测（见 tools/e2e/native/chat_text_probe.scm），
;; 不必起浏览器。
;;
;; 接口（导出名由导入方以 别名.名字 使用）：
;;   (TXT.units s)            半角单位数
;;   (TXT.clean s)            去控制字符：'\t'→4 个空格、'\r' 丢弃；无需清理时原样返回
;;   (TXT.hard_lines s)       → 行列表：仅按 '\n' 切分（保留空行；结尾换行不额外产空行）
;;   (TXT.wrap_lines s maxu)  → 行列表：先硬换行，再逐行按 maxu 个半角单位软折行
;;
;; 解释器约束（manual.md）：算术/比较恒二参数、无一元取负、无 let/map、块内不 define、
;; 空表用 (cdr '(0)) 现造。
;; 注意（易错）：`native` 声明是**按模块**生效的——被 import 的模块要自己声明它用到的
;; 本地库，否则调用处会报"变量 X 未定义"。
;; ============================================================================

(native String)

;; 半角/全角判定（>= 0x3000 视作全角；8592..8703 的箭头区按全角算，与 12px 字模一致）
(define is_wide
  (lambda (code)
    (if (>= code 12288) #t
        (if (>= code 8192) #t
            (if (>= code 8592) (<= code 8703) #f)))))

(define units
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define u 0)
    (while (< i n) {
      (set! u (+ u (if (is_wide (String.charCodeAt i s)) 2 1)))
      (set! i (+ i 1))
    })
    u))

;; 行内控制字符清理：'\t' → 4 个空格、'\r' 丢弃（其余原样）。
;; 绝大多数文本没有控制字符，先扫一遍，没有就原样返回（不产生任何新字符串）。
(define clean
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define c 0)
    (define need #f)
    (define out "")
    (while (< i n) {
      (set! c (String.charCodeAt i s))
      (if (== c 9) (set! need #t) #f)
      (if (== c 13) (set! need #t) #f)
      (if (< c 32)
          (if (not (== c 9))
              (if (not (== c 10)) (set! need #t) #f)
              #f)
          #f)
      (set! i (+ i 1))
    })
    (if need
        {
          (set! i 0)
          (while (< i n) {
            (set! c (String.charCodeAt i s))
            (cond ((== c 9) (set! out (String.concat out "    ")))
                  ((== c 13) #f)
                  ((< c 32) #f)
                  (else (set! out (String.concat out (String.slice s i (+ i 1))))))
            (set! i (+ i 1))
          })
          out
        }
        s)))

;; 按 '\n' 切硬行：空行保留（连续换行 = 空行），结尾换行不额外产生一个空行。
(define hard_lines
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define start 0)
    (define out (cdr '(0)))
    (while (< i n) {
      (if (== (String.charCodeAt i s) 10)
          {
            (push out (String.slice s start i))
            (set! start (+ i 1))
          }
          #f)
      (set! i (+ i 1))
    })
    (if (< start n) (push out (String.slice s start n)) #f)
    out))

;; 软折行：把一行按 maxu 个半角单位切开（全角字符不劈成两半）
(define wrap_one
  (lambda (line kind_unused maxu out)
    (define n (String.length line))
    (define i 0)
    (define u 0)
    (define start 0)
    (define code 0)
    (define w 1)
    (while (< i n) {
      (set! code (String.charCodeAt i line))
      (set! w (if (is_wide code) 2 1))
      (if (> (+ u w) maxu)
          {
            (push out (String.slice line start i))
            (set! start i)
            (set! u 0)
          }
          #f)
      (set! u (+ u w))
      (set! i (+ i 1))
    })
    (if (< start n) (push out (String.slice line start n)) #f)
    (if (== n 0) (push out "") #f)
    out))

(define wrap_lines
  (lambda (s maxu)
    (define hs (hard_lines s))
    (define n (length hs))
    (define i 0)
    (define out (cdr '(0)))
    (if (< maxu 1) (set! maxu 1) #f)
    (set! i 0)
    (while (< i n) {
      (set! out (wrap_one (clean (get_item hs i)) 0 maxu out))
      (set! i (+ i 1))
    })
    out))
