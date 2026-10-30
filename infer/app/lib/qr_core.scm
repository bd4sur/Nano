;; ============================================================================
;; 二维码编码核心（QR Code Model 2 / 字节模式）—— 纯逻辑模块
;;
;; 运行环境：灵机（Animac）解释器 + Math / String 本地库（不碰 Gfx/Key/Touch/Fs，
;;           因此可以脱离图形在宿主命令行解释器里单独回归，见 tools/e2e/native）。
;;
;; 支持范围：
;;   · 版本 1..20（21x21 .. 97x97 模块），字节模式（UTF-8 编码任意文本，
;;     字节容量 L 17..858 / M 14..666 / Q 11..482 / H 7..382）；
;;   · 四种纠错等级 L/M/Q/H（分级索引 0..3）；
;;   · 自动选最小版本；8 种掩码按罚分择优（见 penalty 一节：实现了标准四条规则里的
;;     第 1/2/4 条，第 3 条为速度略去，理由与实测影响见那里）；
;;   · 版本信息（v>=7）与格式信息（纠错等级+掩码）按标准 BCH 码写入（表已预计算）。
;;
;; 正确性验证（tools/e2e/native/test_qrcode.py + 本机额外扫测）：
;;   · 版本 1..20 × 4 个纠错等级的**容量边界**（如 17/18、14/15、718/719、858/859）逐一对齐；
;;   · 矩阵与 Python qrcode 库**逐模块相同**（同版本同掩码同纠错等级，96 个用例 0 差异）；
;;   · 渲染成位图后由 zxing-cpp 解码：>300 个用例（含中文/ASCII/随机/满载）全部还原原文，
;;     从 320x240 屏幕帧缓冲直接解码亦成功。
;;
;; 对外接口（全部值型，便于另一个模块读取结果）：
;;   (QR.encode text ec)  : 0=成功；1=文本过长（超出该纠错等级下版本 20 的容量）
;;       成功后可用下列过程取结果：
;;   (QR.get_version)     : 版本号 1..20
;;   (QR.get_size)        : 边长模块数 n（= 4*version+17）
;;   (QR.get_mask)        : 采用的掩码 0..7
;;   (QR.get_nbytes)      : 文本的 UTF-8 字节数
;;   (QR.get_ec)          : 纠错等级索引 0..3（0=L 1=M 2=Q 3=H）
;;   (QR.get_matrix)      : 长度 n*n 的扁平列表，元素 1=黑 0=白，下标 = 行*n + 列
;;   (QR.ec_capacity ec)  : 该纠错等级下版本 20 的字节容量（界面显示"上限"用）
;;   (QR.ec_name ec)      : "L"/"M"/"Q"/"H"
;;
;; ⚠ 解释器约束（manual.md，逐条遵守）：
;;   · 算术/比较恒二参数；**没有一元取负**（写 (- 0 x)）；/ 与 mod 恒返回浮点；
;;   · 没有 list/append/map；空表用 (cdr '(0)) 现场造（'() 是模块级静态槽）；
;;   · define 无过程简写；块内不 define（循环体不是独立词法区域）；
;;   · cond 每个子句只接受单个表达式。
;;
;; ⚠ 位运算（Math.bit_*）的**参数必须是 uint 表示**（非负整数字面量、uint 算术结果、
;;   或位运算结果），Math.floor 等返回的**浮点数不能**直接喂给位运算（会报参数类型
;;   不符）。故本模块只在两处用位运算：GF(256) 的 exp/log 表初始化与 RS 的加（异或），
;;   涉及的数值一律经 U8 表（`(u8 v)`：由 256 个 uint 字面量组成的查表）转成 uint。
;;
;; ⚠ 模块级临时量按"谁写谁读"分配（前缀即所属函数），跨函数调用链绝不共用——
;;   否则内层函数会悄悄覆盖外层循环变量（见 doc/AGENTS.md 记的俄罗斯方块踩坑）。
;; ============================================================================

(native Math)
(native String)

;; ---- 常量 ----------------------------------------------------------------

(define MAXV 20)          ; 支持的最大版本
(define QUIET 4)          ; 标准要求的静区模块数（本模块不负责绘制，供界面参考）

;; ---- 编码结果（由 encode 写入，getter 读出）-------------------------------

(define Q_VERSION 0)
(define Q_SIZE 0)
(define Q_MASK 0)
(define Q_NBYTES 0)
(define Q_EC 1)
(define Q_MATRIX (cdr '(0)))

;; ---- 0..255 的 uint 查表：把浮点字节值转成位运算可用的 uint ---------------
;; （每一位运算的输入都必须是 uint 表示，见文件头说明）

(define U8 '(
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
    64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
    80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
    96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
    112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
    128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
    144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
    160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175
    176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
    192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207
    208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
    224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
    240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255))

;; 浮点（或任何数值）-> uint，仅在 0..255 范围有意义
(define u8
  (lambda (v)
    (get_item U8 (Math.floor v))))

;; ---- 标准表 --------------------------------------------------------------

;; RS 分块表：下标 = (版本-1)*4 + 纠错等级（0=L 1=M 2=Q 3=H）
;; 每项为 (纠错码字数 块数1 数据码字数1 [块数2 数据码字数2])：
;; 第 1 组若干块的每块数据码字数 = 第 1 个数据码字数，第 2 组比它多 1
(define RS_TABLE '(
    (7 1 19) (10 1 16) (13 1 13) (17 1 9)
    (10 1 34) (16 1 28) (22 1 22) (28 1 16)
    (15 1 55) (26 1 44) (18 2 17) (22 2 13)
    (20 1 80) (18 2 32) (26 2 24) (16 4 9)
    (26 1 108) (24 2 43) (18 2 15 2 16) (22 2 11 2 12)
    (18 2 68) (16 4 27) (24 4 19) (28 4 15)
    (20 2 78) (18 4 31) (18 2 14 4 15) (26 4 13 1 14)
    (24 2 97) (22 2 38 2 39) (22 4 18 2 19) (26 4 14 2 15)
    (30 2 116) (22 3 36 2 37) (20 4 16 4 17) (24 4 12 4 13)
    (18 2 68 2 69) (26 4 43 1 44) (24 6 19 2 20) (28 6 15 2 16)
    (20 4 81) (30 1 50 4 51) (28 4 22 4 23) (24 3 12 8 13)
    (24 2 92 2 93) (22 6 36 2 37) (26 4 20 6 21) (28 7 14 4 15)
    (26 4 107) (22 8 37 1 38) (24 8 20 4 21) (22 12 11 4 12)
    (30 3 115 1 116) (24 4 40 5 41) (20 11 16 5 17) (24 11 12 5 13)
    (22 5 87 1 88) (24 5 41 5 42) (30 5 24 7 25) (24 11 12 7 13)
    (24 5 98 1 99) (28 7 45 3 46) (24 15 19 2 20) (30 3 15 13 16)
    (28 1 107 5 108) (28 10 46 1 47) (28 1 22 15 23) (28 2 14 17 15)
    (30 5 120 1 121) (26 9 43 4 44) (28 17 22 1 23) (28 2 14 19 15)
    (28 3 113 4 114) (26 3 44 11 45) (26 17 21 4 22) (26 9 13 16 14)
    (28 3 107 5 108) (26 3 41 13 42) (30 15 24 5 25) (28 15 15 10 16)))

;; 校正图形中心坐标表：下标 = 版本-1（版本 1 没有校正图形）
(define ALIGN_TABLE '(
    ()
    (6 18) (6 22) (6 26) (6 30) (6 34)
    (6 22 38) (6 24 42) (6 26 46) (6 28 50) (6 30 54)
    (6 32 58) (6 34 62) (6 26 46 66) (6 26 48 70) (6 26 50 74)
    (6 30 54 78) (6 30 56 82) (6 30 58 86) (6 34 62 90)))

;; 格式信息（纠错等级 + 掩码 的 15 位 BCH 码，已异或 0x5412）：
;; 下标 = 纠错等级*8 + 掩码（纠错等级 0=L 1=M 2=Q 3=H，二进制指示符 L=01 M=00 Q=11 H=10）
(define FORMAT_INFO '(
    30660 29427 32170 30877 26159 25368 27713 26998
    21522 20773 24188 23371 17913 16590 20375 19104
    13663 12392 16177 14854 9396 8579 11994 11245
    5769 5054 7399 6608 1890 597 3340 2107))

;; 版本信息（版本 7..20 的 18 位 BCH 码）：下标 = 版本-7
(define VERSION_INFO '(
    31892 34236 39577 42195 48118 51042 55367
    58893 63784 68472 70749 76311 79154 84390))

;; 模块级临时量：本解释器的 set! 要求变量已有绑定，故统一提前 define。
;; （前缀表示归属：z=zeros gf=GF 表 gn=生成多项式 rs=RS u=UTF-8 pb=写位 b=组装
;;   il=交错 ed/cp=容量 fa/am/pe/cm=功能图形/掩码/罚分 rf/rv/tm/bp=预留与放置 wf/wv=格式/版本 en=入口）
(define q_al_c 0)
(define q_al_dk 0)
(define q_al_r 0)
(define q_am_c 0)
(define q_am_idx 0)
(define q_am_r12 0)
(define q_am_t 0)
(define q_am_tend 0)
(define q_pe_idx 0)
(define q_pe_up 0)
(define q_pe_upprev 0)
(define q_am_out (cdr '(0)))
(define q_am_r 0)
(define q_am_v 0)
(define q_am_base 0)
(define q_am_tile (cdr '(0)))
(define q_mt_m 0)
(define q_mt_r 0)
(define q_mt_c 0)
(define q_mt_tile (cdr '(0)))
(define q_b_cw 0)
(define q_b_i 0)
(define q_b_j 0)
(define q_b_pad 0)
(define q_b_t 0)
(define q_b_v 0)
(define q_bp_i 0)
(define q_cm_best 0)
(define q_cm_bp 0)
(define q_cm_m 0)
(define q_cm_p 0)
(define q_cp_d 0)
(define q_ed_v 0)
(define q_en_bytes 0)
(define q_en_cw 0)
(define q_en_dcw 0)
(define q_en_entry 0)
(define q_en_final 0)
(define q_en_mask 0)
(define q_en_v 0)
(define q_en_ver 0)
(define q_fi_c 0)
(define q_fi_cc 0)
(define q_fi_dk 0)
(define q_fi_r 0)
(define q_fi_rr 0)
(define q_gf_i 0)
(define q_gf_x 0)
(define q_gn_a 0)
(define q_gn_g 0)
(define q_gn_i 0)
(define q_gn_j 0)
(define q_gn_len 0)
(define q_gn_tmp 0)
(define q_il_b 0)
(define q_il_bd 0)
(define q_il_be 0)
(define q_il_blk 0)
(define q_il_d1 0)
(define q_il_d2 0)
(define q_il_dl 0)
(define q_il_ecpb 0)
(define q_il_gen 0)
(define q_il_i 0)
(define q_il_k 0)
(define q_il_maxd 0)
(define q_il_n1 0)
(define q_il_n2 0)
(define q_il_nb 0)
(define q_il_out 0)
(define q_il_pos 0)
(define q_pa_c 0)
(define q_pa_i 0)
(define q_pa_j 0)
(define q_pa_pos 0)
(define q_pa_r 0)
(define q_pb_j 0)
(define q_pb_sh 0)
(define q_pd_bidx 0)
(define q_pd_bit 0)
(define q_pd_c 0)
(define q_pd_cc 0)
(define q_pd_clen 0)
(define q_pd_col 0)
(define q_pd_go 0)
(define q_pd_inc 0)
(define q_pd_row 0)
(define q_pd_sh 0)
(define q_pd_widx 0)
(define q_pe_a 0)
(define q_pe_acc 0)
(define q_pe_b 0)
(define q_pe_c 0)
(define q_pe_dark 0)
(define q_pe_len 0)
(define q_pe_pct 0)
(define q_pe_prev 0)
(define q_pe_r 0)
(define q_pe_v 0)
(define q_rf_k 0)
(define q_rs_fac 0)
(define q_rs_i 0)
(define q_rs_j 0)
(define q_rs_len 0)
(define q_rs_rem 0)
(define q_rv_i 0)
(define q_rv_j 0)
(define q_tm_k 0)
(define q_u_c 0)
(define q_u_i 0)
(define q_u_n 0)
(define q_u_out 0)
(define q_wf_b 0)
(define q_wf_bits 0)
(define q_wf_i 0)
(define q_wf_sh 0)
(define q_wv_b 0)
(define q_wv_bits 0)
(define q_wv_c 0)
(define q_wv_i 0)
(define q_wv_r 0)
(define q_wv_sh 0)
(define q_z_i 0)
(define q_z_l 0)

;; ---- 通用小工具 ----------------------------------------------------------

(define new_list (lambda () (cdr '(0))))

;; 造 k 个 0 的列表（'() 是模块级静态槽，push 前必须现造）
(define zeros
  (lambda (k)
    (begin
      (set! q_z_l (new_list))
      (set! q_z_i 0)
      (while (< q_z_i k) (begin (push q_z_l 0) (set! q_z_i (+ q_z_i 1))))
      q_z_l)))

(define is_even
  (lambda (k)
    (if (== (mod k 2) 0) 1 0)))

;; ---- GF(256)：exp / log 表（本原多项式 x^8+x^4+x^3+x^2+1 = 0x11D）----------
;; 全过程只做 uint 算术（1 是 uint 字面量，*2 仍是 uint，异或用 uint 字面量 285），
;; 因此不需要经 U8 转换。

(define GF_EXP (zeros 512))   ; 元素 -> uint；尾部复制一份，乘法取下标时免去 mod 255
(define GF_LOG (zeros 256))
(define gf_ready 0)

(define gf_init
  (lambda ()
    (if (== gf_ready 0)
        (begin
          (set! q_gf_x 1)
          (set! q_gf_i 0)
          (while (< q_gf_i 255)
            (begin
              (set_item! GF_EXP q_gf_i q_gf_x)
              (set_item! GF_LOG q_gf_x q_gf_i)
              (set! q_gf_x (* q_gf_x 2))
              (if (>= q_gf_x 256) (set! q_gf_x (Math.bit_xor q_gf_x 285)))
              (set! q_gf_i (+ q_gf_i 1))))
          (set! q_gf_i 0)
          (while (< q_gf_i 255)
            (begin
              (set_item! GF_EXP (+ q_gf_i 255) (get_item GF_EXP q_gf_i))
              (set! q_gf_i (+ q_gf_i 1))))
          (set! gf_ready 1)))))

;; a、b 为 uint（0..255），返回 uint
(define gf_mul
  (lambda (a b)
    (if (== a 0)
        0
        (if (== b 0)
            0
            (get_item GF_EXP (+ (get_item GF_LOG a) (get_item GF_LOG b)))))))

;; 生成多项式 g(x) = Π(x + α^i)，i=0..n-1；返回 n+1 个 uint 系数（g[0]=1，高次在前）
(define gen_make
  (lambda (n)
    (begin
      (set! q_gn_g (new_list))
      (push q_gn_g 1)
      (set! q_gn_i 0)
      (while (< q_gn_i n)
        (begin
          (set! q_gn_a (get_item GF_EXP q_gn_i))     ; α^i（uint）
          (set! q_gn_len (length q_gn_g))
          (set! q_gn_tmp (zeros (+ q_gn_len 1)))
          (set! q_gn_j 0)
          (while (< q_gn_j q_gn_len)
            (begin
              ;; new[j] ^= g[j]（x 乘）
              (set_item! q_gn_tmp q_gn_j
                         (Math.bit_xor (get_item q_gn_tmp q_gn_j) (get_item q_gn_g q_gn_j)))
              ;; new[j+1] ^= α^i * g[j]
              (set_item! q_gn_tmp (+ q_gn_j 1)
                         (Math.bit_xor (get_item q_gn_tmp (+ q_gn_j 1))
                                       (gf_mul (get_item q_gn_g q_gn_j) q_gn_a)))
              (set! q_gn_j (+ q_gn_j 1))))
          (set! q_gn_g q_gn_tmp)
          (set! q_gn_i (+ q_gn_i 1))))
      q_gn_g)))

;; RS 纠错码字：data 为 uint 码字列表，gen 为生成多项式，n 个纠错码字
(define rs_ec
  (lambda (data gen n)
    (begin
      (set! q_rs_rem (zeros n))
      (set! q_rs_len (length data))
      (set! q_rs_i 0)
      (while (< q_rs_i q_rs_len)
        (begin
          (set! q_rs_fac (Math.bit_xor (get_item data q_rs_i) (get_item q_rs_rem 0)))
          ;; 余式左移一位
          (set! q_rs_j 0)
          (while (< q_rs_j (- n 1))
            (begin
              (set_item! q_rs_rem q_rs_j (get_item q_rs_rem (+ q_rs_j 1)))
              (set! q_rs_j (+ q_rs_j 1))))
          (set_item! q_rs_rem (- n 1) 0)
          (set! q_rs_j 0)
          (while (< q_rs_j n)
            (begin
              (set_item! q_rs_rem q_rs_j
                         (Math.bit_xor (get_item q_rs_rem q_rs_j)
                                       (gf_mul (get_item gen (+ q_rs_j 1)) q_rs_fac)))
              (set! q_rs_j (+ q_rs_j 1))))
          (set! q_rs_i (+ q_rs_i 1))))
      q_rs_rem)))

;; ---- UTF-8 编码 ----------------------------------------------------------

(define utf8_bytes
  (lambda (s)
    (begin
      (set! q_u_out (new_list))
      (set! q_u_i 0)
      (set! q_u_n (String.length s))
      (while (< q_u_i q_u_n)
        (begin
          (set! q_u_c (String.charCodeAt q_u_i s))
          (if (< q_u_c 128)
              (push q_u_out q_u_c)
              (if (< q_u_c 2048)
                  (begin
                    (push q_u_out (+ 192 (Math.floor (/ q_u_c 64))))
                    (push q_u_out (+ 128 (Math.floor (mod q_u_c 64)))))
                  (if (< q_u_c 65536)
                      (begin
                        (push q_u_out (+ 224 (Math.floor (/ q_u_c 4096))))
                        (push q_u_out (+ 128 (Math.floor (mod (/ q_u_c 64) 64))))
                        (push q_u_out (+ 128 (Math.floor (mod q_u_c 64)))))
                      (begin
                        (push q_u_out (+ 240 (Math.floor (/ q_u_c 262144))))
                        (push q_u_out (+ 128 (Math.floor (mod (/ q_u_c 4096) 64))))
                        (push q_u_out (+ 128 (Math.floor (mod (/ q_u_c 64) 64))))
                        (push q_u_out (+ 128 (Math.floor (mod q_u_c 64))))))))
          (set! q_u_i (+ q_u_i 1))))
      q_u_out)))

;; ---- 数据位流 ------------------------------------------------------------

;; 位流累加器（由 build_codewords 与 put_bits 共用：put_bits 只追加到它上面）
(define q_b_bits (cdr '(0)))

;; 把 v 的低 k 位按高位在前追加为 0/1（元素为浮点，仅供拼字节用）
(define put_bits
  (lambda (v k)
    (begin
      (set! q_pb_j (- k 1))
      (while (>= q_pb_j 0)
        (begin
          (set! q_pb_sh (pow 2 q_pb_j))
          (push q_b_bits (Math.floor (mod (Math.floor (/ v q_pb_sh)) 2)))
          (set! q_pb_j (- q_pb_j 1)))))))

;; 数据码字：模式指示(0100) + 字符计数 + 数据字节 + 终止符 + 补齐 + 填充字节
;; 返回 dcw 个 uint 码字（U8 查表把浮点字节值变成 uint）
(define build_codewords
  (lambda (bytes dcw ver)
    (begin
      (set! q_b_bits (new_list))
      (put_bits 4 4)
      (if (< ver 10)
          (put_bits (length bytes) 8)
          (put_bits (length bytes) 16))
      (set! q_b_i 0)
      (while (< q_b_i (length bytes))
        (begin
          (put_bits (get_item bytes q_b_i) 8)
          (set! q_b_i (+ q_b_i 1))))
      ;; 终止符：最多 4 个 0（不超过容量）
      (set! q_b_t 0)
      (while (and (< q_b_t 4) (< (length q_b_bits) (* dcw 8)))
        (begin (push q_b_bits 0) (set! q_b_t (+ q_b_t 1))))
      ;; 补齐到字节边界
      (while (> (mod (length q_b_bits) 8) 0) (push q_b_bits 0))
      ;; 填充字节 0xEC / 0x11 交替
      (set! q_b_pad 1)
      (while (< (length q_b_bits) (* dcw 8))
        (begin
          (if (== q_b_pad 1) (put_bits 236 8) (put_bits 17 8))
          (set! q_b_pad (- 1 q_b_pad))))
      ;; 位 -> uint 码字
      (set! q_b_cw (new_list))
      (set! q_b_i 0)
      (while (< q_b_i dcw)
        (begin
          (set! q_b_v 0)
          (set! q_b_j 0)
          (while (< q_b_j 8)
            (begin
              (set! q_b_v (+ (* q_b_v 2) (get_item q_b_bits (+ (* q_b_i 8) q_b_j))))
              (set! q_b_j (+ q_b_j 1))))
          (push q_b_cw (u8 q_b_v))
          (set! q_b_i (+ q_b_i 1))))
      q_b_cw)))

;; ---- 分块 + 纠错 + 交错 ---------------------------------------------------

(define interleave
  (lambda (cw entry)
    (begin
      (set! q_il_ecpb (get_item entry 0))
      (set! q_il_n1 (get_item entry 1))
      (set! q_il_d1 (get_item entry 2))
      (set! q_il_n2 0)
      (set! q_il_d2 0)
      (if (> (length entry) 3)
          (begin
            (set! q_il_n2 (get_item entry 3))
            (set! q_il_d2 (get_item entry 4))))
      (set! q_il_gen (gen_make q_il_ecpb))
      (set! q_il_nb (+ q_il_n1 q_il_n2))
      (set! q_il_bd (new_list))     ; 每块的数据码字
      (set! q_il_be (new_list))     ; 每块的纠错码字
      (set! q_il_pos 0)
      (set! q_il_b 0)
      (while (< q_il_b q_il_nb)
        (begin
          (set! q_il_dl q_il_d1)
          (if (>= q_il_b q_il_n1) (set! q_il_dl q_il_d2))
          (set! q_il_blk (new_list))
          (set! q_il_k 0)
          (while (< q_il_k q_il_dl)
            (begin
              (push q_il_blk (get_item cw q_il_pos))
              (set! q_il_pos (+ q_il_pos 1))
              (set! q_il_k (+ q_il_k 1))))
          (push q_il_bd q_il_blk)
          (push q_il_be (rs_ec q_il_blk q_il_gen q_il_ecpb))
          (set! q_il_b (+ q_il_b 1))))
      ;; 交错：先按列取各块数据码字（组 2 的块多 1 个），再按列取各块纠错码字
      (set! q_il_out (new_list))
      (set! q_il_maxd q_il_d1)
      (if (> q_il_d2 q_il_maxd) (set! q_il_maxd q_il_d2))
      (set! q_il_i 0)
      (while (< q_il_i q_il_maxd)
        (begin
          (set! q_il_b 0)
          (while (< q_il_b q_il_nb)
            (begin
              (set! q_il_blk (get_item q_il_bd q_il_b))
              (if (< q_il_i (length q_il_blk))
                  (push q_il_out (get_item q_il_blk q_il_i)))
              (set! q_il_b (+ q_il_b 1))))
          (set! q_il_i (+ q_il_i 1))))
      (set! q_il_i 0)
      (while (< q_il_i q_il_ecpb)
        (begin
          (set! q_il_b 0)
          (while (< q_il_b q_il_nb)
            (begin
              (push q_il_out (get_item (get_item q_il_be q_il_b) q_il_i))
              (set! q_il_b (+ q_il_b 1))))
          (set! q_il_i (+ q_il_i 1))))
      q_il_out)))

;; 某项 RS 表条目里的数据码字总数
(define entry_datacw
  (lambda (entry)
    (begin
      (set! q_ed_v (* (get_item entry 1) (get_item entry 2)))
      (if (> (length entry) 3)
          (set! q_ed_v (+ q_ed_v (* (get_item entry 3) (get_item entry 4)))))
      q_ed_v)))

;; 版本 v、纠错等级 ec 的字节模式容量（数据码字 - 模式/计数头）
(define capacity_of
  (lambda (v ec)
    (begin
      (set! q_cp_d (entry_datacw (get_item RS_TABLE (+ (* (- v 1) 4) ec))))
      (if (< v 10) (- q_cp_d 2) (- q_cp_d 3)))))

;; ---- 矩阵：功能图形 ------------------------------------------------------

(define q_n 0)              ; 边长模块数
(define q_mods (cdr '(0)))  ; 扁平矩阵：功能图形为最终值，数据区为**未掩码**的原始位
(define q_func (cdr '(0)))  ; 扁平标记：1=功能图形（不参与数据放置与掩码）
(define q_mat (cdr '(0)))   ; 最终矩阵（含掩码与格式信息）

(define mi (lambda (r c) (+ (* r q_n) c)))

(define set_func
  (lambda (r c v)
    (begin
      (set_item! q_mods (mi r c) v)
      (set_item! q_func (mi r c) 1))))

;; 7x7 定位图形 + 分隔符（r0/c0 为左上角，向外扩 1 圈）
(define put_finder
  (lambda (r0 c0)
    (begin
      (set! q_fi_r (- 0 1))
      (while (< q_fi_r 8)
        (begin
          (set! q_fi_c (- 0 1))
          (while (< q_fi_c 8)
            (begin
              (set! q_fi_rr (+ r0 q_fi_r))
              (set! q_fi_cc (+ c0 q_fi_c))
              (if (and (>= q_fi_rr 0) (< q_fi_rr q_n) (>= q_fi_cc 0) (< q_fi_cc q_n))
                  (begin
                    (set! q_fi_dk 0)
                    (if (and (>= q_fi_r 0) (<= q_fi_r 6) (or (== q_fi_c 0) (== q_fi_c 6)))
                        (set! q_fi_dk 1))
                    (if (and (>= q_fi_c 0) (<= q_fi_c 6) (or (== q_fi_r 0) (== q_fi_r 6)))
                        (set! q_fi_dk 1))
                    (if (and (>= q_fi_r 2) (<= q_fi_r 4) (>= q_fi_c 2) (<= q_fi_c 4))
                        (set! q_fi_dk 1))
                    (set_func q_fi_rr q_fi_cc q_fi_dk)))
              (set! q_fi_c (+ q_fi_c 1))))
          (set! q_fi_r (+ q_fi_r 1)))))))

;; 5x5 校正图形（中心 r0/c0）
(define put_align
  (lambda (r0 c0)
    (begin
      (set! q_al_r (- 0 2))
      (while (<= q_al_r 2)
        (begin
          (set! q_al_c (- 0 2))
          (while (<= q_al_c 2)
            (begin
              (set! q_al_dk 0)
              (if (or (== (Math.abs q_al_r) 2) (== (Math.abs q_al_c) 2)
                      (and (== q_al_r 0) (== q_al_c 0)))
                  (set! q_al_dk 1))
              (set_func (+ r0 q_al_r) (+ c0 q_al_c) q_al_dk)
              (set! q_al_c (+ q_al_c 1))))
          (set! q_al_r (+ q_al_r 1)))))))

(define put_aligns
  (lambda (ver)
    (begin
      (set! q_pa_pos (get_item ALIGN_TABLE (- ver 1)))
      (set! q_pa_i 0)
      (while (< q_pa_i (length q_pa_pos))
        (begin
          (set! q_pa_j 0)
          (while (< q_pa_j (length q_pa_pos))
            (begin
              (set! q_pa_r (get_item q_pa_pos q_pa_i))
              (set! q_pa_c (get_item q_pa_pos q_pa_j))
              ;; 只画中心尚未被占用者（避开三个定位图形）
              (if (and (== (get_item q_mods (mi q_pa_r q_pa_c)) (- 0 1))
                       (== (get_item q_func (mi q_pa_r q_pa_c)) 0))
                  (put_align q_pa_r q_pa_c))
              (set! q_pa_j (+ q_pa_j 1))))
          (set! q_pa_i (+ q_pa_i 1)))))))

;; 时序图形：第 6 行与第 6 列（未被占用处）
(define put_timing
  (lambda ()
    (begin
      (set! q_tm_k 8)
      (while (< q_tm_k (- q_n 8))
        (begin
          (if (== (get_item q_mods (mi q_tm_k 6)) (- 0 1))
              (set_func q_tm_k 6 (is_even q_tm_k)))
          (if (== (get_item q_mods (mi 6 q_tm_k)) (- 0 1))
              (set_func 6 q_tm_k (is_even q_tm_k)))
          (set! q_tm_k (+ q_tm_k 1)))))))

;; 预留格式信息区（值先填 0，掩码选定后由 write_format 覆写）
(define reserve_format
  (lambda ()
    (begin
      (set! q_rf_k 0)
      (while (< q_rf_k 9)
        (begin
          (if (not (== q_rf_k 6)) (set_func q_rf_k 8 0))
          (set! q_rf_k (+ q_rf_k 1))))
      (set! q_rf_k 0)
      (while (< q_rf_k 7)
        (begin
          (set_func (- (- q_n 1) q_rf_k) 8 0)
          (set! q_rf_k (+ q_rf_k 1))))
      (set! q_rf_k 0)
      (while (< q_rf_k 9)
        (begin
          (if (not (== q_rf_k 6)) (set_func 8 q_rf_k 0))
          (set! q_rf_k (+ q_rf_k 1))))
      (set! q_rf_k 0)
      (while (< q_rf_k 8)
        (begin
          (set_func 8 (- (- q_n 1) q_rf_k) 0)
          (set! q_rf_k (+ q_rf_k 1)))))))

;; 预留版本信息区（6x3 两块，仅版本 >= 7）
(define reserve_version
  (lambda ()
    (begin
      (set! q_rv_i 0)
      (while (< q_rv_i 6)
        (begin
          (set! q_rv_j 0)
          (while (< q_rv_j 3)
            (begin
              (set_func q_rv_i (+ (- q_n 11) q_rv_j) 0)
              (set_func (+ (- q_n 11) q_rv_j) q_rv_i 0)
              (set! q_rv_j (+ q_rv_j 1))))
          (set! q_rv_i (+ q_rv_i 1)))))))

(define build_patterns
  (lambda (ver)
    (begin
      (set! q_n (+ (* ver 4) 17))
      (set! q_mods (zeros (* q_n q_n)))
      (set! q_func (zeros (* q_n q_n)))
      (set! q_bp_i 0)
      (while (< q_bp_i (* q_n q_n))
        (begin (set_item! q_mods q_bp_i (- 0 1)) (set! q_bp_i (+ q_bp_i 1))))
      (put_finder 0 0)
      (put_finder 0 (- q_n 7))
      (put_finder (- q_n 7) 0)
      (if (>= ver 2) (put_aligns ver))
      (put_timing)
      (set_func (- q_n 8) 8 1)     ; 固定暗模块
      (reserve_format)
      (if (>= ver 7) (reserve_version)))))

;; ---- 矩阵：数据放置 ------------------------------------------------------

;; 从右下角起、两列一组呈蛇形填充未占用的模块（掩码稍后统一施加）
(define place_data
  (lambda (cw)
    (begin
      (set! q_pd_clen (length cw))
      (set! q_pd_row (- q_n 1))
      (set! q_pd_inc (- 0 1))
      (set! q_pd_bidx 7)
      (set! q_pd_widx 0)
      (set! q_pd_col (- q_n 1))
      (while (> q_pd_col 0)
        (begin
          ;; 跳过第 6 列（竖直时序图形）
          ;; 第 6 列是竖直时序图形，整列跳过（只在这一列上左移 1 格，其余列保持奇偶）
          (if (== q_pd_col 6) (set! q_pd_col (- q_pd_col 1)))
          (set! q_pd_go 1)
          (while (== q_pd_go 1)
            (begin
              (set! q_pd_cc 0)
              (while (< q_pd_cc 2)
                (begin
                  (set! q_pd_c (- q_pd_col q_pd_cc))
                  (if (== (get_item q_func (mi q_pd_row q_pd_c)) 0)
                      (begin
                        (set! q_pd_bit 0)
                        (if (< q_pd_widx q_pd_clen)
                            (begin
                              (set! q_pd_sh (pow 2 q_pd_bidx))
                              (set! q_pd_bit (Math.floor (mod (Math.floor (/ (get_item cw q_pd_widx) q_pd_sh)) 2)))))
                        (set_item! q_mods (mi q_pd_row q_pd_c) (+ 2 q_pd_bit))
                        (set! q_pd_bidx (- q_pd_bidx 1))
                        (if (== q_pd_bidx (- 0 1))
                            (begin
                              (set! q_pd_widx (+ q_pd_widx 1))
                              (set! q_pd_bidx 7)))))
                  (set! q_pd_cc (+ q_pd_cc 1))))
              (set! q_pd_row (+ q_pd_row q_pd_inc))
              (if (or (< q_pd_row 0) (>= q_pd_row q_n))
                  (begin
                    (set! q_pd_row (- q_pd_row q_pd_inc))
                    (set! q_pd_inc (- 0 q_pd_inc))
                    (set! q_pd_go 0)))))
          (set! q_pd_col (- q_pd_col 2)))))))

;; ---- 掩码 ----------------------------------------------------------------

;; 掩码位（ISO/IEC 18004 表 10）：1 = 该处数据位取反
(define mask_bit
  (lambda (m r c)
    (cond ((== m 0) (is_even (+ r c)))
          ((== m 1) (is_even r))
          ((== m 2) (if (== (mod c 3) 0) 1 0))
          ((== m 3) (if (== (mod (+ r c) 3) 0) 1 0))
          ((== m 4) (is_even (+ (Math.floor (/ r 2)) (Math.floor (/ c 3)))))
          ((== m 5) (if (== (+ (mod (* r c) 2) (mod (* r c) 3)) 0) 1 0))
          ((== m 6) (is_even (+ (mod (* r c) 2) (mod (* r c) 3))))
          (else (is_even (+ (mod (* r c) 3) (mod (+ r c) 2)))))))

;; 掩码平铺表：八种掩码的周期都不超过 12（掩码 4 为 行:4 / 列:6，其余 ≤ 6），
;; 故预先算一张 12x12 平铺图，逐格查表代替每格的 mod/floor 运算。
;; 掩码只在 n<=97 的矩阵上使用，一次性的 8*144 次计算可忽略。
(define MASK_TILES (cdr '(0)))
(define mask_ready 0)

(define mask_init
  (lambda ()
    (if (== mask_ready 0)
        (begin
          (set! q_mt_m 0)
          (while (< q_mt_m 8)
            (begin
              (set! q_mt_tile (new_list))
              (set! q_mt_r 0)
              (while (< q_mt_r 12)
                (begin
                  (set! q_mt_c 0)
                  (while (< q_mt_c 12)
                    (begin
                      (push q_mt_tile (mask_bit q_mt_m q_mt_r q_mt_c))
                      (set! q_mt_c (+ q_mt_c 1))))
                  (set! q_mt_r (+ q_mt_r 1))))
              (push MASK_TILES q_mt_tile)
              (set! q_mt_m (+ q_mt_m 1))))
          (set! mask_ready 1)))))

;; 施加掩码 m，返回新的 n*n 列表（值 0/1）。
;; q_mods 中功能图形存 0/1、数据区存 2+原始位（见 place_data 的写入处），因此一次
;; get_item 就能区分两者：功能图形原样输出，数据区按掩码平铺位取反。
;; 内层循环用"递增下标 + 预先算好的行偏移"代替 (mi r c)，省掉每格的乘加与闭包调用
;; （解释器里一次过程调用比一次算术贵得多）。
(define apply_mask
  (lambda (m)
    (begin
      (mask_init)
      (set! q_am_tile (get_item MASK_TILES m))
      (set! q_am_out (new_list))
      (set! q_am_r 0)
      (set! q_am_r12 0)
      (set! q_am_idx 0)
      (while (< q_am_r q_n)
        (begin
          ;; 平铺图的行偏移；列方向的 12 循环用递增计数器（省掉每格一次 mod+floor）
          (set! q_am_base (* q_am_r12 12))
          (set! q_am_tend (+ q_am_base 12))
          (set! q_am_t q_am_base)
          (set! q_am_c 0)
          (while (< q_am_c q_n)
            (begin
              (set! q_am_v (get_item q_mods q_am_idx))
              (if (< q_am_v 2)
                  (push q_am_out q_am_v)
                  (if (== (get_item q_am_tile q_am_t) 1)
                      (push q_am_out (- 3 q_am_v))
                      (push q_am_out (- q_am_v 2))))
              (set! q_am_t (+ q_am_t 1))
              (if (== q_am_t q_am_tend) (set! q_am_t q_am_base))
              (set! q_am_idx (+ q_am_idx 1))
              (set! q_am_c (+ q_am_c 1))))
          (set! q_am_r12 (+ q_am_r12 1))
          (if (== q_am_r12 12) (set! q_am_r12 0))
          (set! q_am_r (+ q_am_r 1))))
      q_am_out)))

;; 罚分：用于在 8 种掩码中择优（ISO/IEC 18004 §8.8.2 的评分思想）。
;;
;; 已实现三条：
;;   规则 1 —— 行/列上连续同色 >= 5 格，每段记 3 + (len-5)；
;;   规则 2 —— 2x2 同色块，每块记 3；
;;   规则 4 —— 深色占比每偏离 50% 达 5% 记 10。
;; 未实现规则 3（行/列上出现 1:1:3:1:1 且一侧有 4 格浅色的"伪定位图形"）：
;;   · 它几乎只被三个定位图形与校正图形触发，而这些在 8 种掩码下逐格相同，
;;     对"选哪个掩码"的贡献小；
;;   · 代价却最高（每个位置要看 11 格，相当于再扫一遍全矩阵）。
;;   略去后 8 掩码搜索快一倍多；二维码本身仍然完全合法——掩码只影响可扫性余量，
;;   不影响可解码性（编码结果已用 zxing-cpp 做全量解码验证）。
;;
;; 性能约束（本解释器里每次 get_item 都是一次 native 调用，约百条 VM 指令）：
;;   行主序一遍同时算规则 1（行）、规则 2、深色计数 —— 每格 2 次取指；
;;   规则 1（列）单独一遍 —— 每格 1 次取指；
;;   下标一律用递增计数器/行偏移，不用 (mi r c)。
(define penalty
  (lambda (mat)
    (begin
      (set! q_pe_acc 0)
      (set! q_pe_dark 0)
      ;; ---- 行主序：规则 1（行内连续）+ 规则 2（2x2）+ 深色计数 ----
      (set! q_pe_r 0)
      (set! q_pe_idx 0)
      (while (< q_pe_r q_n)
        (begin
          (set! q_pe_prev (get_item mat q_pe_idx))       ; 行首
          (if (== q_pe_prev 1) (set! q_pe_dark (+ q_pe_dark 1)))
          (set! q_pe_len 1)
          (set! q_pe_upprev 0)
          (if (> q_pe_r 0) (set! q_pe_upprev (get_item mat (- q_pe_idx q_n))))
          (set! q_pe_idx (+ q_pe_idx 1))
          (set! q_pe_c 1)
          (while (< q_pe_c q_n)
            (begin
              (set! q_pe_v (get_item mat q_pe_idx))
              (if (== q_pe_v 1) (set! q_pe_dark (+ q_pe_dark 1)))
              (if (== q_pe_v q_pe_prev)
                  (set! q_pe_len (+ q_pe_len 1))
                  (begin
                    (if (>= q_pe_len 5) (set! q_pe_acc (+ q_pe_acc (- q_pe_len 2))))
                    (set! q_pe_len 1)))
              ;; 2x2 同色：本格 = 左邻 = 上邻 = 左上邻
              (if (> q_pe_r 0)
                  (begin
                    (set! q_pe_up (get_item mat (- q_pe_idx q_n)))
                    (if (and (== q_pe_v q_pe_prev)
                             (== q_pe_v q_pe_up)
                             (== q_pe_prev q_pe_upprev))
                        (set! q_pe_acc (+ q_pe_acc 3)))
                    (set! q_pe_upprev q_pe_up)))
              (set! q_pe_prev q_pe_v)
              (set! q_pe_idx (+ q_pe_idx 1))
              (set! q_pe_c (+ q_pe_c 1))))
          (if (>= q_pe_len 5) (set! q_pe_acc (+ q_pe_acc (- q_pe_len 2))))
          (set! q_pe_r (+ q_pe_r 1))))
      ;; ---- 规则 1（列内连续）----
      (set! q_pe_c 0)
      (while (< q_pe_c q_n)
        (begin
          (set! q_pe_idx q_pe_c)
          (set! q_pe_prev (get_item mat q_pe_idx))
          (set! q_pe_len 1)
          (set! q_pe_idx (+ q_pe_idx q_n))
          (set! q_pe_r 1)
          (while (< q_pe_r q_n)
            (begin
              (set! q_pe_v (get_item mat q_pe_idx))
              (if (== q_pe_v q_pe_prev)
                  (set! q_pe_len (+ q_pe_len 1))
                  (begin
                    (if (>= q_pe_len 5) (set! q_pe_acc (+ q_pe_acc (- q_pe_len 2)) ))
                    (set! q_pe_len 1)))
              (set! q_pe_prev q_pe_v)
              (set! q_pe_idx (+ q_pe_idx q_n))
              (set! q_pe_r (+ q_pe_r 1))))
          (if (>= q_pe_len 5) (set! q_pe_acc (+ q_pe_acc (- q_pe_len 2))))
          (set! q_pe_c (+ q_pe_c 1))))
      ;; ---- 规则 4：深色占比每偏离 50% 满 5% 记 10 ----
      (set! q_pe_pct (/ q_pe_dark (* q_n q_n)))
      (set! q_pe_acc
        (+ q_pe_acc (* 10 (Math.floor (/ (Math.abs (- (* q_pe_pct 100) 50)) 5)))))
      q_pe_acc)))

;; 依次试 8 种掩码，罚分最低者胜（同分取小掩码）
(define choose_mask
  (lambda ()
    (begin
      (set! q_cm_best 0)
      (set! q_cm_bp 0)
      (set! q_cm_m 0)
      (while (< q_cm_m 8)
        (begin
          (set! q_cm_p (penalty (apply_mask q_cm_m)))
          (if (or (== q_cm_m 0) (< q_cm_p q_cm_bp))
              (begin
                (set! q_cm_bp q_cm_p)
                (set! q_cm_best q_cm_m)))
          (set! q_cm_m (+ q_cm_m 1))))
      q_cm_best)))

;; ---- 格式信息 / 版本信息写入 ---------------------------------------------

(define write_format
  (lambda (mask ec)
    (begin
      (set! q_wf_bits (get_item FORMAT_INFO (+ (* ec 8) mask)))
      (set! q_wf_i 0)
      (while (< q_wf_i 15)
        (begin
          (set! q_wf_sh (pow 2 q_wf_i))
          (set! q_wf_b (Math.floor (mod (Math.floor (/ q_wf_bits q_wf_sh)) 2)))
          ;; 竖排（第 8 列，右上 8 位 / 左上 7 位 / 左下 7 位）
          (if (< q_wf_i 6)
              (set_item! q_mat (mi q_wf_i 8) q_wf_b)
              (if (< q_wf_i 8)
                  (set_item! q_mat (mi (+ q_wf_i 1) 8) q_wf_b)
                  (set_item! q_mat (mi (+ (- q_n 15) q_wf_i) 8) q_wf_b)))
          ;; 横排（第 8 行，右上 8 位 / 左上 7 位）
          (if (< q_wf_i 8)
              (set_item! q_mat (mi 8 (- (- q_n 1) q_wf_i)) q_wf_b)
              (if (< q_wf_i 9)
                  (set_item! q_mat (mi 8 7) q_wf_b)
                  (set_item! q_mat (mi 8 (- 14 q_wf_i)) q_wf_b)))
          (set! q_wf_i (+ q_wf_i 1))))
      (set_item! q_mat (mi (- q_n 8) 8) 1))))   ; 固定暗模块

(define write_version
  (lambda (ver)
    (begin
      (set! q_wv_bits (get_item VERSION_INFO (- ver 7)))
      (set! q_wv_i 0)
      (while (< q_wv_i 18)
        (begin
          (set! q_wv_sh (pow 2 q_wv_i))
          (set! q_wv_b (Math.floor (mod (Math.floor (/ q_wv_bits q_wv_sh)) 2)))
          (set! q_wv_r (Math.floor (/ q_wv_i 3)))
          (set! q_wv_c (+ (Math.floor (mod q_wv_i 3)) (- q_n 11)))
          (set_item! q_mat (mi q_wv_r q_wv_c) q_wv_b)
          (set_item! q_mat (mi q_wv_c q_wv_r) q_wv_b)
          (set! q_wv_i (+ q_wv_i 1)))))))

;; ---- 主入口 --------------------------------------------------------------

(define encode
  (lambda (text ec)
    (begin
      (set! Q_EC ec)
      (set! q_en_bytes (utf8_bytes text))
      (set! Q_NBYTES (length q_en_bytes))
      (set! q_en_ver 0)
      (set! q_en_v 1)
      (while (and (== q_en_ver 0) (<= q_en_v MAXV))
        (begin
          (if (>= (capacity_of q_en_v ec) Q_NBYTES) (set! q_en_ver q_en_v))
          (set! q_en_v (+ q_en_v 1))))
      (if (== q_en_ver 0)
          1
          (begin
            (gf_init)
            (set! q_en_entry (get_item RS_TABLE (+ (* (- q_en_ver 1) 4) ec)))
            (set! q_en_dcw (entry_datacw q_en_entry))
            (set! q_en_cw (build_codewords q_en_bytes q_en_dcw q_en_ver))
            (set! q_en_final (interleave q_en_cw q_en_entry))
            (build_patterns q_en_ver)
            (place_data q_en_final)
            (set! q_en_mask (choose_mask))
            (set! q_mat (apply_mask q_en_mask))
            (write_format q_en_mask ec)
            (if (>= q_en_ver 7) (write_version q_en_ver))
            (set! Q_VERSION q_en_ver)
            (set! Q_SIZE q_n)
            (set! Q_MASK q_en_mask)
            (set! Q_MATRIX q_mat)
            0)))))

;; ---- getter --------------------------------------------------------------

(define get_version (lambda () Q_VERSION))
(define get_size (lambda () Q_SIZE))
(define get_mask (lambda () Q_MASK))
(define get_nbytes (lambda () Q_NBYTES))
(define get_ec (lambda () Q_EC))
(define get_matrix (lambda () Q_MATRIX))

(define ec_name
  (lambda (ec)
    (cond ((== ec 0) "L")
          ((== ec 1) "M")
          ((== ec 2) "Q")
          (else "H"))))

(define ec_capacity
  (lambda (ec)
    (capacity_of MAXV ec)))
