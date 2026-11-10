;; ============================================================================
;; chat_prompts.scm —— 预置问题（输入为空时的随机提示词）
;;
;; 与原生 C 实现保持一致（infer/utils.c 的 default_prompts[] + set_random_prompt()）：
;;   · 内容：同一组 20 个问题，逐条照抄（改那边就要同步这里）；
;;   · 逻辑：输入为空时用"当前时间戳"做一个 xorshift 伪随机，按结果取模选一条。
;;
;; 与 C 的差异（实现层，语义等价）：C 用 64 位 xorshift64*（state>>12、<<25、>>27，
;; 再乘 0x2545F4914F6CDD1D 取高位）。灵机的整数在 wasm32 宿主上只有 27 位有效宽度、
;; 浮点只有 24 位尾数，**大数乘法取模类 PRNG 在 wasm32 上会失真**（乘积超出 2^24 就不再
;; 逐位精确，低位先坏——实测 xorshift + mod 会退化成"5 10 15 0"式短周期）。因此这里
;; 直接用宿主 C 实现的 `(Math.random)`（[0,1) 均匀分布，首次调用以宿主时间播种）：
;; 分布好、不碰整数位宽，语义仍是"输入为空 → 随机取一条预置问题"。
;;
;; 接口：
;;   (PICK.count)      预置问题条数
;;   (PICK.at i)       第 i 条（越界返回空串）
;;   (PICK.random)     按当前时间戳随机取一条
;;
;; 解释器约束：算术/比较恒二参数；位运算的参数必须是无符号整数（本模块只在
;; < 2^27 的值上做位运算，wasm32 安全）；块内不 define；空表用 (cdr '(0)) 现造。
;; ============================================================================

(native Math)
(native System)

(define prompts (cdr '(0)))

(push prompts "人类的本质是什么？")
(push prompts "人类的本质是复读机吗？")
(push prompts "天空为什么是蓝色的？")
(push prompts "你是谁训练的大模型？")
(push prompts "西红柿炒鸡蛋怎么做？")
(push prompts "我想洗车，洗车店离我家50米，应该开车去还是走路去？")
(push prompts "射频功率-40dBm的一半是多少dBm？")
(push prompts "质权、典权、留置权和抵押权的区别是什么？")
(push prompts "太阳系中最小的大行星是哪个？")
(push prompts "某业余电台操作者听到业余专用频率上出现某种显然出自非业余电台的人为干扰发射，于是按下话筒向该发射者宣传无线电管理法规知识。这种做法是正确的还是错误的？")
(push prompts "证明 $\\sqrt{2}$ 是无理数。")
(push prompts "为什么阿塔卡马沙漠位于大洋沿岸却极度干燥？")
(push prompts "澳大利亚的首都在哪里？")
(push prompts "如何构造Quine，也就是自己输出自己的程序？")
(push prompts "将以下句子翻译为英文：“人工智能即将统治人类！”")
(push prompts "写一篇申论，题目是《绿水青山就是金山银山》。")
(push prompts "1小时能晒干1条毛巾，晒干10条毛巾需要几小时？")
(push prompts "原样复述引号中内容：“我叫Nano，是BD4SUR训练的电子鹦鹉。”")
(push prompts "列举一些常见的逻辑谬误。")
(push prompts "9.9和9.11哪个大？")

(define count
  (lambda () (length prompts)))

(define at
  (lambda (i)
    (if (>= i 0)
        (if (< i (length prompts)) (get_item prompts i) "")
        "")))

(define random
  (lambda ()
    (define n (count))
    (define i 0)
    (if (< n 1)
        ""
        {
          (set! i (Math.floor (* (Math.random) n)))
          ;; Math.random 名义上是 [0,1)，保险起见钳一下
          (if (< i 0) (set! i 0) #f)
          (if (>= i n) (set! i (- n 1)) #f)
          (at i)
        })))
