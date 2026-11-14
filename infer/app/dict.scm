;; ============================================================================
;; 词典（Dict）—— 灵机应用程序 · 英汉词典（C 版 ui_dict.c 的灵机重实现）
;;
;; 功能与 C 版对齐：
;;   · 前缀检索（大小写不敏感）：输入 "ab" 列出词典里所有以 ab 开头的词条；
;;   · 候选表最多 64 条（与 C 版 UI_DICT_MATCH_MAX 一致），点选/回车进入详情；
;;   · 详情页（与 C 版同构）：正文第一行是词条（青色）、接着音标与释义，
;;     正文字体 16px 抗锯齿（其余界面 12px）；页脚「上一个 / 下一个」翻词条，
;;     支持按住连发；右上角「返回」回查询界面（硬 A/ESC 在本页同样是"上一层"）。
;;
;; 与 C 版的差异，以及为什么（三条都是被灵机的运行环境"逼"出来的取舍）：
;;
;;  1) **索引**：C 版把 CSV 全量扫描成 2.4MB 的二进制索引（ecdict.idx）再做二分。
;;     灵机侧做不到：本地库 (native Fs) 只给文本文件 API，二进制 .idx 无法可靠
;;     逐字节解析；13 万条索引项在 Scheme 堆里也要几十 MB。本实现改用**稀疏块
;;     索引**（约 500 项 / 60KB 堆，按 16KB 分块定向探测建立，不做全量扫描），
;;     查询仍是"索引定位块 → 块内精细定位 → 前缀顺序枚举"，只是块内定位改用
;;     十几次小窗口探测替代十几 KB 文本扫描（桌面实测单次查询 3~9ms）。细节与
;;     权衡见 app/lib/dict_index.scm 的头注释（数据层单独成模块，便于与 C 版/
;;     Python 真值逐条比对）。
;;
;;  2) **软键盘**：不复用 C 层 ui_softkbd.c，改为本文件内自绘的 QWERTY 键盘——
;;     26 字母 + 退格(DEL) + 输入(ENTER) + 空格(SPACE) + Shift，键帽是**分立的
;;     圆角矩形**（圆角半径 2px，外圈 1px 描边），字模统一 12px 抗锯齿（字模 1）。
;;     触屏语义：按下武装（键帽高亮）→ 在同一键上松手才生效，按住拖走可取消。
;;     Shift 是"粘滞一格"：打出一个大写字母后自动松开（手机键盘惯例）。
;;
;;  3) **候选表**：复用灵机侧已有的菜单控件组件 app/lib/menu_list.scm（高亮行、
;;     拖动滚动、滚动条、轻点选中），不再自己写一套列表。注意绘制顺序：菜单控件
;;     的视口边界是"半行"，滚动时它会画出视口上/下各一个半行，因此必须先画候选表、
;;     再用页眉/查询行/键盘底板把它们盖掉（见 draw_query 的注释）。
;;
;; 数据文件：dict/ecdict.csv（相对业务资源根；桌面/WASM 为
;;   $PLATFORM_ROOT_DIR/dict/ecdict.csv，掌机为 SD 卡 /dict/ecdict.csv）。
;;   词典目录里自带的 ecdict.idx 本实现不使用（原因见上），因此 CSV 更新后
;;   无需重建任何缓存。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数、无一元取负、无 let/map/
;;   append/apply、define 无过程简写、块内不 define、'() 是模块级静态槽（空表一律
;;   用 (cdr '(0)) 现造）、/ 与 mod 恒返回浮点（下标一律 Math.floor）、cond 子句
;;   只吃一个表达式。热路径上不做整串拼接（manual §3.4：整串拼接会制造大量短命
;;   对象），故"状态行 / 查询行"等文本只在取值变化时重建。
;; ============================================================================

(native Gfx)
(native Touch)
(native Key)
(native String)
(native Math)
(native System)
(native App)

(import EX "app/lib/exit.scm")         ; 硬 A/ESC → 全局退出确认框 + 右上角「退出」按钮
(import ML "app/lib/menu_list.scm")    ; 候选表控件（高亮/滚动/点选）
(import TXT "app/lib/chat_text.scm")   ; 硬换行 / 控制字符清理 / 全角判定
(import DI "app/lib/dict_index.scm")   ; 词典数据层（稀疏索引 + 前缀查询 + 字段解析）

;; ============================================================================
;; 常量
;; ============================================================================

(define PFX_MAX 31)              ; 前缀长度上限（词典最长词条 31 字节）
(define FONT 1)                  ; 12px 抗锯齿字模（GFX_FONT_ALPHA_12）：界面其余部分
(define FONT_BODY 2)             ; 16px 抗锯齿字模（GFX_FONT_ALPHA_16）：详情页正文（含词条）
(define ROW_H 15)                ; 12px 抗锯齿字模的行高（gfx_font_line_height）
(define HALF_W 7)                ; 12px 字模半角步进（像素）
(define FULL_W 12)               ; 12px 字模全角步进（像素）
(define BODY_HALF 9)             ; 16px 字模半角步进（(Gfx.font_info 2) 实测）
(define BODY_ROW_H 18)           ; 16px 字模的行高
(define DET_MAXU 34)             ; 详情正文每行半角单位数 = (320-12)/9

;; 版式（320x240）：页眉 17 / 查询行 15 / 候选表 105（7 行）/ 键盘 4 行
(define HDR_H 17)
(define QRY_Y 17)
(define LIST_Y 32)
(define LIST_H 105)
(define KB_Y0 141)
(define KB_H 21)
(define KB_STEP 25)
(define KB_M 3)                  ; 键盘左右边距
(define KB_G 3)                  ; 键间距
(define DET_Y 19)                ; 详情正文起点（页眉 17 + 2）
(define DET_LINES 11)            ; 详情一屏行数（16px：19 + 11*18 = 217）
(define DET_FOOT_Y 222)          ; 详情页脚（翻词条按钮）
(define DET_BTN_W 62)            ; 页脚按钮宽
(define DET_BTN_H 15)            ; 页脚按钮高（= 12px 字模行高）
(define BACK_W 36)               ; 详情页右上角「返回」按钮（与宿主退出按钮同尺寸）
(define BACK_H 16)
;; 页眉文字的垂直补偿：本地库现在把 (Gfx.draw_text) 的 y 当「墨迹顶」绘制
;; （见 am_native_Gfx.c 的 native_text_top_gap，12px 抗锯齿字模的行顶空隙是 2px），
;; 而本 App 的页眉是按旧口径（y = 行顶，墨迹再落 2px）调的；不补回去，标题（以及
;; 同一行的状态文字）会帖到 17px 页眉的上沿。
(define HDR_TXT_DY 2)
(define REP_DELAY 380)           ; 按住翻词条：首次后的等待（ms）
(define REP_PERIOD 130)          ; 按住翻词条：随后的连发周期（ms）

;; 键的动作码：>= 32 是"上屏这个字符码"，其余是功能键
(define K_SHIFT 1001)
(define K_DEL 1002)
(define K_SPACE 1003)
(define K_ENTER 1004)

;; 配色（深色主题，与便签/文件管理器同一色系）
(define C_BG_R 16)   (define C_BG_G 19)   (define C_BG_B 25)
(define C_HDR_R 30)  (define C_HDR_G 38)  (define C_HDR_B 52)
(define C_TXT_R 225) (define C_TXT_G 231) (define C_TXT_B 241)
(define C_DIM_R 140) (define C_DIM_G 150) (define C_DIM_B 166)
(define C_KEY_R 46)  (define C_KEY_G 55)  (define C_KEY_B 74)    ; 键帽底色
(define C_EDG_R 86)  (define C_EDG_G 97)  (define C_EDG_B 124)   ; 键帽描边
(define C_KON_R 96)  (define C_KON_G 132) (define C_KON_B 196)   ; 按下/Shift 亮态
(define C_QRY_R 255) (define C_QRY_G 226) (define C_QRY_B 74)    ; 查询前缀（黄）
(define C_WRD_R 120) (define C_WRD_G 200) (define C_WRD_B 255)   ; 词条（青）

;; ============================================================================
;; 状态
;; ============================================================================

(define SW 320)
(define SH 240)
(define matches (cdr '(0)))      ; 当前候选：(词条 记录偏移)
(define pfx "")                  ; 查询前缀（原样大小写）
(define qline "_")               ; 查询行显示文本（前缀 + 光标）
(define status "")               ; 页眉状态（候选数/提示）
(define shift 0)                 ; Shift 粘滞状态
(define page 0)                  ; 0=查询 1=详情 2=错误
(define err_line1 "")
(define err_line2 "")
(define dirty 1)                 ; 需要重绘
(define ts 0)                    ; 定时器编号
(define keys (cdr '(0)))         ; 键表：(标签 动作 x y w h)
(define armed -1)                ; 触屏按下时武装的键下标（-1 无）

;; 详情页状态
(define d_word "")
(define d_phon "")
(define d_def "")
(define d_lines (cdr '(0)))      ; 折行后的正文行（第 0 行是音标）
(define d_scroll 0)
(define d_idx 0)                 ; 当前词条在候选表中的下标
(define d_prev 0)                ; 详情页触屏上一拍是否按住
(define d_last 0)                ; 上一拍触点 y
(define d_y0 0)                  ; 本次按下的起始 y（轻点判定）
(define d_moved 0)               ; 本次按住期间累计位移
(define d_btn 0)                 ; 按住中的按钮：0 无 / 1 上一个 / 2 下一个 / 3 返回
(define d_rep_at 0)              ; 下一次连发时刻（ms）

;; ============================================================================
;; 小工具（只放界面侧用得上的；数据层在 app/lib/dict_index.scm）
;; ============================================================================

(define new_list (lambda () (cdr '(0))))
(define max2 (lambda (a b) (if (> a b) a b)))

;; 把 src 的全部元素追加到 dst 尾部（dst 被原地 push）
(define push_all
  (lambda (dst src)
    (define n (length src))
    (define i 0)
    (while (< i n) {
      (push dst (get_item src i))
      (set! i (+ i 1))
    })
    dst))

;; ---- 释义折行 ------------------------------------------------------------
;; 复用 chat_text 的硬换行(TXT.hard_lines)、控制字符清理(TXT.clean) 与全角判定
;; (TXT.is_wide)，只把"软折行"自己写一遍：TXT.wrap_lines 严格按半角单位切断，
;; 会出现"上一行以（喂 结尾、下一行只剩一个 )"的孤字行（本词典释义里很常见）。
;; 这里的规则是**不在收尾标点前断行**（右括号/右引号/句读等），允许该行略微超宽。

(define is_tail
  (lambda (c)
    (cond ((== c 41) #t)     ; )
          ((== c 93) #t)     ; ]
          ((== c 125) #t)    ; }
          ((== c 44) #t)     ; ,
          ((== c 46) #t)     ; .
          ((== c 59) #t)     ; ;
          ((== c 58) #t)     ; :
          ((== c 33) #t)     ; !
          ((== c 63) #t)     ; ?
          ((== c 12289) #t)  ; 、
          ((== c 12290) #t)  ; 。
          ((== c 65292) #t)  ; ，
          ((== c 65294) #t)  ; ．
          ((== c 65307) #t)  ; ；
          ((== c 65306) #t)  ; ：
          ((== c 65281) #t)  ; ！
          ((== c 65311) #t)  ; ？
          ((== c 65289) #t)  ; ）
          ((== c 12301) #t)  ; 」
          ((== c 12299) #t)  ; 〉
          ((== c 12309) #t)  ; 】
          ((== c 8250) #t)   ; ›
          ((== c 8217) #t)   ; ’
          ((== c 8221) #t)   ; ”
          (else #f))))

(define wrap_para
  (lambda (line maxu out)
    (define n (String.length line))
    (define i 0)
    (define u 0)
    (define start 0)
    (define c 0)
    (define w 1)
    (while (< i n) {
      (set! c (String.charCodeAt i line))
      (set! w (if (TXT.is_wide c) 2 1))
      (if (and (> (+ u w) maxu) (> i start) (not (is_tail c)))
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

(define wrap_text
  (lambda (s maxu)
    (define hs (TXT.hard_lines s))
    (define n (length hs))
    (define i 0)
    (define out (new_list))
    (if (< maxu 1) (set! maxu 1) #f)
    (while (< i n) {
      (set! out (wrap_para (TXT.clean (get_item hs i)) maxu out))
      (set! i (+ i 1))
    })
    out))

;; ---- 音标转写 ------------------------------------------------------------
;; 12px 字模只有 ASCII + CJK，本词典音标用到的 18 个 IPA 字符全都没有字形，
;; 直接画会满屏 '?'。这里按"传统词典的 ASCII 音标"习惯做一次显示层转写
;; （只影响显示，不动数据；θ 字模里有，走 else 分支原样保留）。

(define ipa_map
  (lambda (c)
    (cond ((== c 618) "i")      ; ɪ
          ((== c 712) "'")      ; ˈ 主重音
          ((== c 716) ",")      ; ˌ 次重音
          ((== c 720) ":")      ; ː 长音
          ((== c 601) "e")      ; ə
          ((== c 604) "e:")     ; ɜ
          ((== c 603) "e")      ; ɛ
          ((== c 650) "u")      ; ʊ
          ((== c 652) "u")      ; ʌ
          ((== c 596) "o")      ; ɔ
          ((== c 594) "o")      ; ɒ
          ((== c 593) "a")      ; ɑ
          ((== c 230) "a")      ; æ
          ((== c 643) "sh")     ; ʃ
          ((== c 658) "zh")     ; ʒ
          ((== c 331) "ng")     ; ŋ
          ((== c 240) "dh")     ; ð
          (else (String.fromCharCode c)))))

(define ipa_ascii
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define c 0)
    (define out "")
    (while (< i n) {
      (set! c (String.charCodeAt i s))
      (if (< c 128)
          (set! out (String.concat out (String.charAt s i)))
          (set! out (String.concat out (ipa_map c))))
      (set! i (+ i 1))
    })
    out))

;; ============================================================================
;; 候选表：查询 + 重建菜单控件
;; ============================================================================

(define rebuild
  (lambda ()
    (define rows (new_list))
    (define n 0)
    (define i 0)
    (set! matches (DI.query pfx))
    (set! n (length matches))
    (if (== n 0)
        (if (== (String.length pfx) 0)
            (set! status "输入字母检索")
            (set! status "无匹配词条"))
        (if (>= n (DI.match_max))
            (set! status (String.concat (String.atom_to_string n) "+ 条"))
            (set! status (String.concat (String.atom_to_string n) " 条"))))
    ;; 候选表控件：有匹配给候选行，否则给一行提示（复用同一套绘制与高亮）
    (if (== n 0)
        {
          (push rows (ML.item (if (== (String.length pfx) 0)
                                  "（输入前缀以检索）"
                                  "（无匹配词条）")
                              150 158 172 0 0 150 158 172))
          (ML.reset rows ROW_H)
        }
        {
          (set! i 0)
          (while (< i n) {
            (push rows (ML.item (get_item (get_item matches i) 0)
                                205 214 230 0 0 255 255 255))
            (set! i (+ i 1))
          })
          (ML.reset rows ROW_H)
        })
    (set! dirty 1)
    #t))

;; ============================================================================
;; 详情：取字段 → 折行 → 展示
;; ============================================================================

(define build_detail
  (lambda ()
    (define ph "")
    (set! d_lines (new_list))
    ;; 与 C 版一致：正文第一行是**词条**（青色），接着音标（暗色）、一个空行、释义正文
    (push d_lines d_word)
    (if (> (String.length d_phon) 0)
        (set! ph (String.concat "/" (String.concat (ipa_ascii d_phon) "/")))
        (set! ph ""))
    (push d_lines ph)
    (push d_lines "")
    (push_all d_lines (wrap_text d_def DET_MAXU))
    #t))

;; 打开展示候选表第 idx 条
(define open_detail
  (lambda (idx)
    (define e #null)
    (define flds #null)
    (set! d_idx idx)
    (set! e (get_item matches idx))
    (set! flds (DI.entry (get_item e 1)))
    (if (null? flds)
        {
          (set! d_word (get_item e 0))
          (set! d_phon "")
          (set! d_def "（读取该词条失败）")
        }
        {
          (set! d_word (get_item flds 0))
          (set! d_phon (get_item flds 1))
          (set! d_def (get_item flds 2))
          (if (== (String.length d_word) 0) (set! d_word (get_item e 0)) #f)
        })
    (set! d_scroll 0)
    ;; 注意：**不要**在这里重置 d_prev / d_btn —— 它们是"本次触屏手势"的状态，
    ;; 而按住翻词条时每一步都会走到这里（重置会让连发退化成每个节拍一次，
    ;; 且按住高亮会闪掉）。
    (build_detail)
    (set! page 1)
    (set! dirty 1)
    ;; 详情页右上角是「返回」：撤掉宿主登记的退出按钮热区，否则点它会变成退出请求
    (App.set_exit_button 0 0 0 0)
    #t))

;; 详情页翻词条（dir=-1 上一个 / 1 下一个，端点钳制）
(define detail_step
  (lambda (dir)
    (define n (length matches))
    (define k (+ d_idx dir))
    (if (>= k 0)
        (if (< k n) (open_detail k) #f)
        #f)))

;; 详情页滚动（按行，端点钳制；本词典释义最长 143 字节 ≈ 4 行，滚动是给
;; 用户换成大词典（或更窄的屏）时留的余量）
(define detail_scroll_by
  (lambda (dy)
    (define n (length d_lines))
    (define mx (max2 0 (- n DET_LINES)))
    (set! d_scroll (+ d_scroll dy))
    (if (< d_scroll 0) (set! d_scroll 0) #f)
    (if (> d_scroll mx) (set! d_scroll mx) #f)
    (set! dirty 1)
    #t))

;; ============================================================================
;; 键盘：QWERTY，分立圆角矩形键帽（半径 2px），12px 抗锯齿字模
;;
;; 键表项 = (标签 动作 x y w h)。动作 >= 32 表示"上屏该字符码"，否则是功能键码。
;; 每行按"权重"分配宽度：先求单位宽，再取整，最后整行居中。
;; ============================================================================

(define add_row
  (lambda (labels actions weights y)
    (define n (length labels))
    (define i 0)
    (define sw 0)
    (define avail 0)
    (define unit 0.0)
    (define w 0)
    (define tw 0)
    (define x 0.0)
    (while (< i n) {
      (set! sw (+ sw (get_item weights i)))
      (set! i (+ i 1))
    })
    (set! avail (- (- SW (* 2 KB_M)) (* (- n 1) KB_G)))
    (set! unit (/ avail sw))
    (set! i 0)
    (while (< i n) {
      (set! tw (+ tw (Math.floor (* unit (get_item weights i)))))
      (set! i (+ i 1))
    })
    (set! x (+ KB_M (Math.floor (/ (- avail tw) 2))))
    (set! i 0)
    (while (< i n) {
      (set! w (Math.floor (* unit (get_item weights i))))
      (push keys (cons (get_item labels i)
                       (cons (get_item actions i)
                             (cons x (cons y (cons w (cons KB_H (new_list))))))))
      (set! x (+ x w))
      (set! x (+ x KB_G))
      (set! i (+ i 1))
    })
    #t))

(define build_keys
  (lambda ()
    (set! keys (new_list))
    ;; 第 1 行：q w e r t y u i o p
    (add_row '("Q" "W" "E" "R" "T" "Y" "U" "I" "O" "P")
             '(113 119 101 114 116 121 117 105 111 112)
             '(1 1 1 1 1 1 1 1 1 1) KB_Y0)
    ;; 第 2 行：a s d f g h j k l
    (add_row '("A" "S" "D" "F" "G" "H" "J" "K" "L")
             '(97 115 100 102 103 104 106 107 108)
             '(1 1 1 1 1 1 1 1 1) (+ KB_Y0 KB_STEP))
    ;; 第 3 行：SHIFT z x c v b n m DEL
    (add_row '("SHIFT" "Z" "X" "C" "V" "B" "N" "M" "DEL")
             '(1001 122 120 99 118 98 110 109 1002)
             '(1.6 1 1 1 1 1 1 1 1.6) (+ KB_Y0 (* 2 KB_STEP)))
    ;; 第 4 行：SPACE ENTER
    (add_row '("SPACE" "ENTER")
             '(1003 1004)
             '(4 2.2) (+ KB_Y0 (* 3 KB_STEP)))
    #t))

;; 屏幕坐标 → 键下标（不在任何键上返回 -1）
(define kb_hit
  (lambda (px py)
    (define n (length keys))
    (define i 0)
    (define k #null)
    (define r -1)
    (while (< i n) {
      (set! k (get_item keys i))
      (if (and (>= px (get_item k 2)) (< px (+ (get_item k 2) (get_item k 4))))
          (if (and (>= py (get_item k 3)) (< py (+ (get_item k 3) (get_item k 5))))
              (if (< r 0) (set! r i) #f)
              #f)
          #f)
      (set! i (+ i 1))
    })
    r))

;; 编辑：上屏一个字符码（Shift 生效时转大写，并在打完一格后自动松开）
(define insert_ch
  (lambda (code)
    (define c code)
    (if (== shift 1)
        (if (and (>= code 97) (<= code 122)) (set! c (- code 32)) #f)
        #f)
    (if (< (String.length pfx) PFX_MAX) {
      (set! pfx (String.concat pfx (String.fromCharCode c)))
      (set! qline (String.concat pfx "_"))
      (set! shift 0)
      (rebuild)
    } #f)
    #t))

(define do_del
  (lambda ()
    (if (> (String.length pfx) 0) {
      (set! pfx (String.slice pfx 0 (- (String.length pfx) 1)))
      (set! qline (String.concat pfx "_"))
      (rebuild)
    } #f)
    #t))

;; 回车：打开当前选中的候选词条
(define open_selected
  (lambda ()
    (define n (length matches))
    (define k 0)
    (if (> n 0) {
      (set! k (ML.sel_of))
      (if (>= k n) (set! k (- n 1)) #f)
      (if (>= k 0) (open_detail k) #f)
    } #f)
    #t))

;; 键动作分发
(define activate
  (lambda (i)
    (define k (get_item keys i))
    (define a (get_item k 1))
    (if (== a K_SHIFT)
        { (set! shift (if (== shift 0) 1 0))
          (set! dirty 1) }
        (if (== a K_DEL) (do_del)
            (if (== a K_SPACE) (insert_ch 32)
                (if (== a K_ENTER) (open_selected)
                    (insert_ch a)))))
    #t))

;; ============================================================================
;; 绘制
;; ============================================================================

(define draw_header
  (lambda (title)
    (Gfx.fill_rect 0 0 SW HDR_H C_HDR_R C_HDR_G C_HDR_B)
    (Gfx.draw_text title 4 (+ 1 HDR_TXT_DY) C_TXT_R C_TXT_G C_TXT_B)
    #t))

;; 一个键帽：外圈 1px 描边 + 内凹圆角矩形（圆角半径 2px；"分立圆角矩形"的要求）
(define draw_key
  (lambda (k on)
    (define x (get_item k 2))
    (define y (get_item k 3))
    (define w (get_item k 4))
    (define h (get_item k 5))
    (define fr C_KEY_R)
    (define fg C_KEY_G)
    (define fb C_KEY_B)
    (if (== on 1) {
      (set! fr C_KON_R)
      (set! fg C_KON_G)
      (set! fb C_KON_B)
    } #f)
    (Gfx.rounded_rect x y w h 2 2 2 2 C_EDG_R C_EDG_G C_EDG_B 1)
    (Gfx.rounded_rect (+ x 1) (+ y 1) (- w 2) (- h 2) 2 2 2 2 fr fg fb 1)
    (Gfx.draw_text_centered FONT (get_item k 0)
                            (+ x (Math.floor (/ w 2)))
                            (+ y (Math.floor (/ h 2)))
                            C_TXT_R C_TXT_G C_TXT_B 1)
    #t))

(define draw_kb
  (lambda ()
    (define n (length keys))
    (define i 0)
    (define k #null)
    (define a 0)
    (define on 0)
    ;; 键盘底板：与候选区分色，并盖掉候选表最后一个半行（若有）
    (Gfx.fill_rect 0 (+ LIST_Y LIST_H) SW (- SH (+ LIST_Y LIST_H)) 20 23 30)
    (while (< i n) {
      (set! k (get_item keys i))
      (set! a (get_item k 1))
      (set! on 0)
      (if (== i armed) (set! on 1) #f)
      (if (and (== a K_SHIFT) (== shift 1)) (set! on 1) #f)
      (draw_key k on)
      (set! i (+ i 1))
    })
    #t))

(define draw_query
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH C_BG_R C_BG_G C_BG_B)
    (Gfx.set_font FONT)
    ;; 绘制顺序即叠加顺序（图形层没有裁剪矩形）：候选表控件的视口边界是"半行"，
    ;; 滚动时它会画出视口上/下各一个半行，因此必须先画候选表，再用页眉、查询行、
    ;; 键盘底板把它们盖掉（menu_list 的约定：「边界上的半行由调用方随后覆盖绘制」）。
    (ML.draw)
    (draw_header "电子词典")
    (Gfx.draw_text status 60 (+ 1 HDR_TXT_DY) C_DIM_R C_DIM_G C_DIM_B)
    ;; 查询行（前缀 + 光标）
    (Gfx.fill_rect 0 QRY_Y SW ROW_H 22 26 34)
    (Gfx.draw_text qline 6 (+ QRY_Y 1) C_QRY_R C_QRY_G C_QRY_B)
    ;; 软键盘（内部先铺底板，盖掉候选表下边界的半行）
    (draw_kb)
    (EX.draw_button)
    ;; 撤掉宿主登记的「退出」热区：本 App 每拍 33ms，(EX.check) 自己的电平判定足够
    ;; 可靠（宿主热区是给星历那种 1000ms/拍的慢 App 准备的）。留着它会有一个坑：
    ;; 从详情页点右上角「返回」时，宿主是在**触摸 UP 沿（带去抖，晚几帧）**才做热区
    ;; 命中判定的，而那时我们已回到查询页并重新登记了热区，于是同一次点按被误判成
    ;; "退出"→ 弹全局确认框。清掉热区后，退出按钮仍由 (EX.check) 正常处理。
    (App.set_exit_button 0 0 0 0)
    (Gfx.refresh)
    #t))

;; 详情页右上角「返回」按钮（回词条查询界面）：与宿主退出按钮同尺寸同位置，
;; 但语义是"上一层"，故不用 (EX.draw_button)——那枚按钮会把点按变成**退出请求**。
(define draw_back_btn
  (lambda ()
    (define bx (- SW (+ BACK_W 4)))
    (if (== d_btn 3)
        (Gfx.rounded_rect bx 2 BACK_W BACK_H 2 2 2 2 C_KON_R C_KON_G C_KON_B 1)
        (Gfx.rounded_rect bx 2 BACK_W BACK_H 2 2 2 2 C_KEY_R C_KEY_G C_KEY_B 1))
    (Gfx.draw_text_centered FONT "返回" (+ bx (Math.floor (/ BACK_W 2)))
                            (+ 2 (Math.floor (/ BACK_H 2))) C_TXT_R C_TXT_G C_TXT_B 1)
    #t))

;; 页脚按钮：**无边框**的圆角矩形（半径 2px），按住时换底色（连发期间持续高亮）
(define draw_foot_btn
  (lambda (id label cx)
    (define x (- cx 31))
    (if (== d_btn id)
        (Gfx.rounded_rect x DET_FOOT_Y DET_BTN_W DET_BTN_H 2 2 2 2 C_KON_R C_KON_G C_KON_B 1)
        (Gfx.rounded_rect x DET_FOOT_Y DET_BTN_W DET_BTN_H 2 2 2 2 C_KEY_R C_KEY_G C_KEY_B 1))
    (Gfx.draw_text_centered FONT label cx
                            (+ DET_FOOT_Y (Math.floor (/ DET_BTN_H 2)))
                            C_TXT_R C_TXT_G C_TXT_B 1)
    #t))

(define draw_detail
  (lambda ()
    (define i d_scroll)
    (define n (length d_lines))
    (define y DET_Y)
    (define lim (+ DET_Y (* DET_LINES BODY_ROW_H)))
    (define ln "")
    (define idx_txt "")
    (Gfx.fill_rect 0 0 SW SH C_BG_R C_BG_G C_BG_B)
    ;; 页眉、页脚、按钮统一 12px；正文（含词条）用 16px 抗锯齿
    (Gfx.set_font FONT)
    (draw_header "电子词典")
    (draw_back_btn)
    (Gfx.set_font FONT_BODY)
    (while (and (< i n) (< y lim)) {
      (set! ln (get_item d_lines i))
      ;; 正文第 0 行是词条（青色）、第 1 行是音标（暗色），其余是释义
      (if (== i 0)
          (Gfx.draw_text ln 6 y C_WRD_R C_WRD_G C_WRD_B)
          (if (== i 1)
              (Gfx.draw_text ln 6 y C_DIM_R C_DIM_G C_DIM_B)
              (Gfx.draw_text ln 6 y C_TXT_R C_TXT_G C_TXT_B)))
      (set! y (+ y BODY_ROW_H))
      (set! i (+ i 1))
    })
    (Gfx.set_font FONT)
    ;; 页脚：上一个 / 序号 / 下一个
    (Gfx.fill_rect 0 DET_FOOT_Y SW (- SH DET_FOOT_Y) 24 28 38)
    (draw_foot_btn 1 "上一个" 35)
    (draw_foot_btn 2 "下一个" 285)
    (set! idx_txt (String.concat (String.atom_to_string (+ d_idx 1))
                                 (String.concat "/" (String.atom_to_string (length matches)))))
    (Gfx.draw_text_centered FONT idx_txt 160
                            (+ DET_FOOT_Y (Math.floor (/ DET_BTN_H 2)))
                            C_DIM_R C_DIM_G C_DIM_B 1)
    (Gfx.refresh)
    #t))

(define draw_error
  (lambda ()
    (Gfx.set_font FONT)
    (Gfx.fill_rect 0 0 SW SH C_BG_R C_BG_G C_BG_B)
    (draw_header "电子词典")
    (Gfx.draw_text_centered FONT err_line1 160 100 C_TXT_R C_TXT_G C_TXT_B 1)
    (Gfx.draw_text_centered FONT err_line2 160 124 C_DIM_R C_DIM_G C_DIM_B 1)
    (EX.draw_button)
    (Gfx.refresh)
    #t))

(define draw_splash
  (lambda (msg)
    (Gfx.set_font FONT)
    (Gfx.fill_rect 0 0 SW SH C_BG_R C_BG_G C_BG_B)
    (Gfx.draw_text_centered FONT msg 160 112 C_TXT_R C_TXT_G C_TXT_B 1)
    (Gfx.refresh)
    #t))

(define draw_current
  (lambda ()
    (if (== page 1) (draw_detail)
        (if (== page 2) (draw_error)
            (draw_query)))
    #t))

;; ============================================================================
;; 输入
;; ============================================================================

;; 查询页触屏：按下武装键 → 在同一键上松手才生效（拖走即取消）
(define touch_query
  (lambda ()
    (if (Touch.take_press) (set! armed (kb_hit (Touch.x) (Touch.y))) #f)
    (if (Touch.take_release) {
      (if (>= armed 0)
          (if (== (kb_hit (Touch.x) (Touch.y)) armed)
              (activate armed)
              #f)
          #f)
      (set! armed -1)
    } #f)
    #t))

;; 详情页按钮命中：0 无 / 1 上一个 / 2 下一个 / 3 返回
(define det_hit
  (lambda (px py)
    (define bx (- SW (+ BACK_W 4)))
    (define r 0)
    (if (and (>= py DET_FOOT_Y) (< py (+ DET_FOOT_Y DET_BTN_H)))
        (if (and (>= px 4) (< px (+ 4 DET_BTN_W)))
            (set! r 1)
            (if (and (>= px 254) (< px (+ 254 DET_BTN_W)))
                (set! r 2)))
        #f)
    (if (and (== r 0) (>= px bx) (< px (+ bx BACK_W)))
        (if (and (>= py 2) (< py (+ 2 BACK_H))) (set! r 3) #f)
        #f)
    r))

;; 详情页的"回查询页"（「返回」按钮、页眉轻点、硬 A/ESC 三条路都走它）
(define detail_back
  (lambda ()
    (set! page 0)
    (set! d_btn 0)
    (set! dirty 1)
    #t))

;; 触发一次翻词条（按住连发也走它）
(define foot_fire
  (lambda (id)
    (if (== id 1) (detail_step -1) (detail_step 1))
    #t))

;; 详情页触屏：拖动滚动；轻点页眉 / 右上角「返回」回查询页；页脚按钮翻词条，
;; 且**按住连发**（380ms 后每 130ms 一次，与掌机/手机长按翻页的手感一致）。
(define touch_detail
  (lambda ()
    (define now (System.timestamp))
    (define dy 0)
    (define hit 0)
    (if (Touch.is_pressed) {
      (set! hit (det_hit (Touch.x) (Touch.y)))
      (if (== d_prev 0) {
        (set! d_prev 1)
        (set! d_last (Touch.y))
        (set! d_y0 (Touch.y))
        (set! d_moved 0)
        (set! d_btn hit)
        (if (and (== d_btn 1) (> (length matches) 0)) (foot_fire 1) #f)
        (if (and (== d_btn 2) (> (length matches) 0)) (foot_fire 2) #f)
        (set! d_rep_at (+ now REP_DELAY))
        (set! dirty 1)
      } {
        (if (> d_btn 0)
            ;; 按住按钮：手指移出按钮即取消（与虚拟键盘同一语义）
            (if (not (== hit d_btn))
                {
                  (set! d_btn 0)
                  (set! dirty 1)
                }
                (if (and (or (== d_btn 1) (== d_btn 2)) (>= now d_rep_at))
                    {
                      (foot_fire d_btn)
                      (set! d_rep_at (+ now REP_PERIOD))
                    }
                    #f))
            ;; 普通拖动：滚动正文（1:1 跟手）
            {
              (set! dy (- (Touch.y) d_last))
              (set! d_last (Touch.y))
              (set! d_moved (+ d_moved (Math.abs dy)))
              (if (not (== dy 0)) (detail_scroll_by (- 0 dy)) #f)
            })
      })
    } {
      (if (== d_prev 1) {
        (set! d_prev 0)
        (if (and (== d_btn 3) (== (det_hit (Touch.x) (Touch.y)) 3))
            (detail_back)
            (if (and (== d_btn 0) (< d_moved 8) (< d_y0 HDR_H))
                (detail_back)
                #f))
        (set! d_btn 0)
        (set! dirty 1)
      } #f)
    })
    #t))

;; 查询页的按键：字母/空格上屏、退格删字、回车选词、↑↓ 移动候选高亮
(define key_query
  (lambda (k)
    (define done 0)
    (if (and (>= k 97) (<= k 122)) { (insert_ch k) (set! done 1) } #f)
    (if (and (== done 0) (>= k 65) (<= k 90)) { (insert_ch (+ k 32)) (set! done 1) } #f)
    (if (and (== done 0) (== k 32)) { (insert_ch 32) (set! done 1) } #f)
    (if (and (== done 0) (or (== k 8) (== k 127))) { (do_del) (set! done 1) } #f)
    (if (and (== done 0) (== k 13)) { (open_selected) (set! done 1) } #f)
    (if (and (== done 0) (or (== k 1) (== k 3)))
        { (ML.move -1) (set! dirty 1) (set! done 1) } #f)
    (if (and (== done 0) (or (== k 2) (== k 4)))
        { (ML.move 1) (set! dirty 1) (set! done 1) } #f)
    done))

;; 详情页的按键：←→ / ↑↓ 翻词条，退格/回车/ESC 回查询页，换行滚动一行
(define key_detail
  (lambda (k)
    (if (or (== k 1) (== k 3)) (detail_step -1)
        (if (or (== k 2) (== k 4)) (detail_step 1)
            (if (or (== k 8) (== k 127) (== k 13) (== k 27))
                { (set! page 0) (set! dirty 1) }
                (if (== k 10) (detail_scroll_by 1) #f))))))

;; 实体键盘（浏览器物理键盘 / 掌机 16 键矩阵 / 终端）：一拍内可能积压多个按下沿
(define handle_keys
  (lambda ()
    (define k 0)
    (define stop 0)
    (while (== stop 0) {
      (set! k (Key.take_press))
      (if (== k 0)
          (set! stop 1)
          (if (== page 0) (key_query k)
              (if (== page 1) (key_detail k) #f)))
    })
    #t))

;; ============================================================================
;; 主循环
;; ============================================================================

(define on_tick
  (lambda ()
    ;; 硬 A/ESC：在**详情页**是"回查询页"（本模块自己消费返回请求，因此这一拍
    ;; 不能同时调 EX.check —— 否则它会拿同一份请求去弹退出确认框）；
    ;; 查询页/错误页仍交给 app/lib/exit.scm 走全局退出确认。
    (if (== page 1)
        (if (App.back_requested) (detail_back) #f)
        (EX.check))
    (handle_keys)
    (if (== page 0) {
      (touch_query)
      (if (>= (ML.step_touch) 0) (open_selected)
          (set! dirty 1))
    } (if (== page 1) (touch_detail) #f))
    (if (== dirty 1) {
      (set! dirty 0)
      (draw_current)
    } #f)
    #t))

;; ============================================================================
;; 启动
;; ============================================================================

(define boot
  (lambda ()
    (define fi #null)
    (set! SW (Gfx.width))
    (set! SH (Gfx.height))
    (Gfx.set_font FONT)                      ; 12px 抗锯齿字模（整个 App 统一）
    ;; 菜单控件的字模度量（半角步进 / 全角步进 / 墨高 12 / 字模框顶内缩 1 / 行顶空隙 2）
    (set! fi (Gfx.font_info FONT))
    (if (null? fi)
        (ML.metrics HALF_W FULL_W 12 1 2)
        (ML.metrics (get_item fi 0) (get_item fi 1) 12 1 2))
    (ML.geom 0 LIST_Y SW LIST_H)
    (build_keys)
    (if (< (DI.open) 0)
        {
          (set! page 2)
          (set! err_line1 "未找到词典文件")
          (set! err_line2 "请把 ecdict.csv 放到 dict/ 目录")
          (draw_current)
        }
        {
          (draw_splash "正在建立索引……")
          (set! page 0)
          (rebuild)
          (draw_current)
        })
    (set! ts (System.set_interval 33 on_tick))
    #t))

(boot)

;; 硬 A/ESC → 宿主全局退出确认框；确认后本实例在帧末销毁（本 App 无落盘状态）
(EX.install_with draw_current)
