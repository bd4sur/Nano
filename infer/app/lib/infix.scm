;; ============================================================================
;; 中缀表达式解析器（Infix Expression Parser）—— 灵机（Animac）应用库
;;
;; 来源：上游 Animac 回归用例 /home/bd4sur/Animac/test/calculator.scm（2017, 2025-07-11）
;;       **lexer / parser / type_of_token / eval / show_* 逐字照抄**，仅做两处机械适配：
;;         1) import 路径 "list.scm" → "app/lib/list.scm"（本项目的模块搜索基准是资源根，
;;            见 ui_scheme_app.c 的 s_app_root；app/lib/list.scm 是该文件的逐字副本）；
;;         2) 删去上游的 `run` 演示入口（它是自带表达式与断言的冒烟测试，界面由
;;            app/calculator.scm 承担）。
;;       因此语法、优先级、结合性、词法规则与上游完全一致，行为可直接对照上游用例。
;;
;; 文法（BNF，同上游文件头注释）：
;;      <expr_list> ::= <expr> <expr_list_tail>
;; <expr_list_tail> ::= , <expr> <expr_list_tail> | ε
;;           <expr> ::= <term> <expr_tail>
;;      <expr_tail> ::= [+-] <term> <expr_tail> | ε
;;           <term> ::= <factor> <term_tail>
;;      <term_tail> ::= [*/] <factor> <term_tail> | ε
;;         <factor> ::= <unaried> [^%] <factor> | <unaried>
;;        <unaried> ::= [!-] <unaried> | <primary>
;;       <arg_list> ::= <expr_list> | ε
;;        <primary> ::= ( <expr> ) | <IDENTIFIER> ( <expr> ) | <literal>
;;        <literal> ::= <NUMBER> | <IDENTIFIER>
;;         <NUMBER> ::= [\+\-]*\d+\.*\d*
;;     <IDENTIFIER> ::= [a-zA-Z_][a-zA-Z0-9_]*
;;
;; 对外接口（供 import 方以 infix. 前缀调用，别名由 import 方决定）：
;;   (lexer 表达式字符串)              → 词元列表，每个词元为 (位置 类型 文本)
;;   (parser 词元列表)                 → 抽象语法树；语法错误时返回 #f（并 display 提示）
;;   (eval AST)                        → 数值；无法识别的算子返回 #f（并 display 提示）
;;
;; 支持的内建运算符与函数（同上游 eval）：+ - * / % ^ 与
;;   sqrt exp log sin cos tan atan floor ceil（写作 sqrt(9) 这种函数调用形式）
;;
;; 注意（调用方须自行防御，本库不抛异常也不报错）：
;;   · 语法错误：parser 返回 #f，需要判断 `list?` / `#f` 再决定是否 eval；
;;   · 除零、非法算子、数字解析失败（String.parseNumber → #undefined）会触发虚拟机
;;     运行时错误并**终止出错进程**——界面侧须把求值放在 fork 出的子进程里做，
;;     见 app/calculator.scm 的「= 键求值」段落。
;; ============================================================================

(native String)
(native Math)
(import List "app/lib/list.scm")

(define lexer
  (lambda (expr)
    (define len (String.length expr))

    (define is_sep
      (lambda (ch)
        (or (String.equals ch " ") (String.equals ch "\t") (String.equals ch "\n") (String.equals ch "\r"))))

    (define is_comma
      (lambda (ch)
        (String.equals ch ",")))

    (define is_operator
      (lambda (ch)
        (or (String.equals ch "+") (String.equals ch "-") (String.equals ch "*") (String.equals ch "/")
            (String.equals ch "^") (String.equals ch "%") (String.equals ch "&") (String.equals ch "|")
            (String.equals ch "!") (String.equals ch "<") (String.equals ch ">") (String.equals ch "="))))
    
    (define is_digit
      (lambda (ch)
        (and (>= (String.charCodeAt 0 ch) 48) (<= (String.charCodeAt 0 ch) 57))))

    (define is_letter
      (lambda (ch)
        (or (and (>= (String.charCodeAt 0 ch) 97) (<= (String.charCodeAt 0 ch) 122))
            (and (>= (String.charCodeAt 0 ch) 65) (<= (String.charCodeAt 0 ch) 90))
            (== (String.charCodeAt 0 ch) 95))))

    (define read_number
      (lambda (start)
        (define end start)
        (define has_decimal #f)
        (define ch 0)
        (while (< end len) {
          (set! ch (String.charAt expr end))
          (cond ((is_digit ch) {
                  (set! end (+ end 1))
                })
                ((String.equals ch ".") {
                  (if has_decimal break)
                  (set! has_decimal #t)
                  (set! end (+ end 1))
                })
                (else break))
        })
        (String.slice expr start end)))

    (define read_identifier
      (lambda (start)
        (define end start)
        (define ch 0)
        (if (is_letter (String.charAt expr end)) {
          (set! end (+ start 1))
          (while (and (< end len) (or (is_letter (String.charAt expr end)) (is_digit (String.charAt expr end)))) {
            (set! end (+ end 1))
          })
          (String.slice expr start end)
        } {
          #f
        })))

    ;; 主体部分
    (define tokens '())
    (define is_prev_number #f)
    (define num_start 0)
    (define sign 0)
    (define number 0)
    (define identifier 0)
    (define ch 0)
    (define index 0)
    (while (< index len) {
      (set! ch (String.charAt expr index))

      (if (is_sep ch) {
        (set! index (+ index 1))
        continue
      })

      (cond ((or (is_digit ch)
                 (and (or (String.equals ch "-") (String.equals ch "+"))
                      (< (+ index 1) len)
                      (not is_prev_number)
                      (or (is_digit (String.charAt expr (+ index 1)))
                          (String.equals (String.charAt expr (+ index 1)) ".")))) {
              (set! is_prev_number #t)
              (set! num_start index)
              (if (or (String.equals ch "-") (String.equals ch "+")) {
                (set! sign ch)
                (set! index (+ index 1))
              } {
                (set! sign "")
                (set! num_start index)
              })
              (set! number (read_number index))
              (set! tokens (List.append `(,index 'number ,(String.concat sign number)) tokens))
              (set! index (+ index (String.length number)))
            })
            ;; 运算符
            ((is_operator ch) {
              (set! is_prev_number #f)
              (set! tokens (List.append `(,index 'operator ,ch) tokens))
              (set! index (+ index 1))
            })
            ;; 左右括号
            ((or (String.equals ch "(") (String.equals ch ")")) {
              (set! is_prev_number (String.equals ch ")"))
              (set! tokens (List.append `(,index 'bracket ,ch) tokens))
              (set! index (+ index 1))
            })
            ;; 标识符
            ((is_letter ch) {
              (set! is_prev_number #f)
              (set! identifier (read_identifier index))
              (set! tokens (List.append `(,index 'identifier ,identifier) tokens))
              (set! index (+ index (String.length identifier)))
            })
            ;; 逗号
            ((is_comma ch) {
              (set! is_prev_number #f)
              (set! tokens (List.append `(,index 'comma ,ch) tokens))
              (set! index (+ index 1))
            })
            (else {
              (display "Unexpected character: ") (display ch) (newline)
              break
            })
      )
    })
    tokens
  ))


(define parser
  (lambda (tokens)
    (define index 0)

    (define eat
      (lambda ()
        (define ret (get_item tokens index))
        (set! index (+ index 1))
        ret))

    (define match
      (lambda (expected)
        (if (< index (length tokens)) 
          (String.equals expected (get_item (get_item tokens index) 2))
          #f)))

    (define is_number
      (lambda ()
        (eq? 'number (get_item (get_item tokens index) 1))))

    (define is_identifier
      (lambda ()
        (eq? 'identifier (get_item (get_item tokens index) 1))))

    (define parse_literal
      (lambda ()
        (if (or (is_number) (is_identifier)) (eat) {
          (display "Error @ <literal>") (newline)
          #f
        })))

    (define parse_primary
      (lambda ()
        (define expr 0)
        (define func 0)
        (cond ((match "(") {
                (eat)
                (set! expr (parse_expr))
                (if (match ")") (eat) { (display "Error @ <primary>: missing ')'") (newline) #f })
                expr
              })
              ((is_identifier) {
                (set! func (eat))
                (if (match "(") (eat) { (display "Error @ <primary>: missing '('") (newline) #f })
                (set! expr (parse_expr))
                (if (match ")") (eat) { (display "Error @ <primary>: missing ')'") (newline) #f })
                `(,func ,expr)
              })
              (else {
                (parse_literal)
              }))))

    (define parse_unaried
      (lambda ()
        (if (or (match "!") (match "-")) {
          (define op (eat))
          (define unaried (parse_unaried))
          `(,op ,unaried)
        } {
          (parse_primary)
        })))

    (define parse_factor
      (lambda ()
        (define unaried (parse_unaried))
        (if (or (match "^") (match "%")) {
          (define op (eat))
          `(,op ,unaried ,(parse_factor))
        } {
          unaried
        })))

    (define parse_term_tail
      (lambda (left)
        (if (or (match "*") (match "/")) {
          (define op (eat))
          (define factor (parse_factor))
          (define new_left `(,op ,left ,factor))
          (parse_term_tail new_left)
        } {
          left
        })))

    (define parse_term
      (lambda ()
        (parse_term_tail (parse_factor))))

    (define parse_expr_tail
      (lambda (left)
        (if (or (match "+") (match "-")) {
          (define op (eat))
          (define term (parse_term))
          (define new_left `(,op ,left ,term))
          (parse_expr_tail new_left)
        } {
          left
        })))

    (define parse_expr
      (lambda ()
        (parse_expr_tail (parse_term))))

    (define parse
      (lambda ()
        (define ast (parse_expr))
        (if (< index (length tokens)) {
          (display "Unexpected token '") (display (get_item tokens index)) (display "'") (newline)
          #f
        } {
          ast
        })))

    (parse)
  ))


(define type_of_token
  (lambda (token)
    (if (atom? (get_item token 1)) {
      (get_item token 1)
    } {
      'application
    })))

;; ---------------------------------------------------------------------------
;; eval —— 求值（上游逻辑 + 交互式安全加固）
;;
;; 与上游的差异只有「失效路径」：上游对非法输入会落到 (eval #f) / (/ x 0) /
;; (#null 参与算术) 等情形，这些都会**触发虚拟机运行时错误并终止进程**。
;; 交互式计算器上这些都是常见输入（残缺表达式、5/0、log(-1)），故统一改为返回
;; 符号 'err（任意层级的操作数非数字 → 整体 'err，天然向上传播），运算表本身与
;; 上游逐条一致（含一元/二元 '-' 的区分）。
;; ---------------------------------------------------------------------------

(define num_or_err
  (lambda (v) (if (number? v) v 'err)))

(define eval
  (lambda (ast)
    (define tot (if (list? ast) (type_of_token ast) 'bad))
    (cond ((eq? tot 'number) {
            (num_or_err (String.parseNumber (get_item ast 2)))
          })
          ((eq? tot 'application) {
            (define op (get_item (get_item ast 0) 2))
            (define left  (get_item ast 1))
            (define right (get_item ast 2))
            (define lv (eval left))
            (define u (eq? #undefined right))            ;; 一元运算（'-' 单目）
            (define rv (if u #undefined (eval right)))
            (if (or (not (number? lv)) (and (not u) (not (number? rv))))
                'err
                (cond ((String.equals op "+") (+ lv rv))
                      ((String.equals op "-") (if u (- 0 lv) (- lv rv))) ;; 区分一元和二元运算
                      ((String.equals op "*") (* lv rv))
                      ((String.equals op "/") (if (== 0 rv) 'err (/ lv rv)))
                      ((String.equals op "%") (mod lv rv))
                      ((String.equals op "^") (pow lv rv))
                      ((String.equals op "sqrt") (pow lv 0.5))
                      ((String.equals op "exp") (num_or_err (Math.exp lv)))
                      ((String.equals op "log") (num_or_err (Math.log lv)))
                      ((String.equals op "sin") (num_or_err (Math.sin lv)))
                      ((String.equals op "cos") (num_or_err (Math.cos lv)))
                      ((String.equals op "tan") (num_or_err (Math.tan lv)))
                      ((String.equals op "atan") (num_or_err (Math.atan lv)))
                      ((String.equals op "floor") (num_or_err (Math.floor lv)))
                      ((String.equals op "ceil") (num_or_err (Math.ceil lv)))
                      (else 'err))
            )})
          (else 'err))))



;; ===========================================================================
;; 以下为本项目新增（上游无）：交互式输入的前置校验与一站式求值
;;
;; 为什么需要 validate：上游 parser 在「期待操作数却已到输入末尾」时会越界取词元
;; （get_item 返回 #undefined，再对其取下标即运行时错误、进程被杀）。批处理用例
;; 不会出现这种输入，交互式键盘则必然出现（"1+"、"sqrt"、"("）。validate 按文法
;; 对词元序列做一遍等价的结构检查，只在通过后才调用 parser，从而把「残缺输入」
;; 变成一句提示，而不是一次进程崩溃。
;; ===========================================================================

;; 校验失败原因（validate 侧）与 calc 的中文原因，供界面直接显示
(define v_reason "表达式不完整")
(define c_reason "表达式不完整")

;; 支持的函数名白名单（与上面 eval 的运算表一致）
(define FUNC_NAMES '("sqrt" "exp" "log" "sin" "cos" "tan" "atan" "floor" "ceil"))
(define OP_NAMES '("+" "-" "*" "/" "%" "^"))

(define in_list
  (lambda (x lst)
    (if (null? lst) #f
        (if (String.equals x (car lst)) #t (in_list x (cdr lst))))))

;; 词元序列是否构成一个完整的中缀表达式；#t/#f，原因见 v_reason
(define validate
  (lambda (tokens)
    (define n (length tokens))
    (define i 0)
    (define depth 0)
    (define want_operand #t)   ;; #t = 期待操作数（或左括号/一元负号）
    (define ty 'none)
    (define val "")
    (define ok #t)
    (set! v_reason "表达式不完整")
    (while (and ok (< i n)) {
      (set! ty (get_item (get_item tokens i) 1))
      (set! val (get_item (get_item tokens i) 2))
      (cond ((eq? ty 'number)
             (if want_operand
                 (set! want_operand #f)
                 { (set! ok #f) (set! v_reason "两个数字之间缺少运算符") }))
            ((eq? ty 'bracket)
             (if (String.equals val "(")
                 { (set! depth (+ depth 1)) (set! want_operand #t) }
                 { (if (or want_operand (<= depth 0))
                       { (set! ok #f) (set! v_reason "括号不匹配") }
                       { (set! depth (- depth 1)) (set! want_operand #f) }) }))
            ((eq? ty 'identifier)
             ;; 文法只允许 <IDENTIFIER> ( <expr> )，且实参不能为空
             (if (not want_operand)
                 { (set! ok #f) (set! v_reason "两个数字之间缺少运算符") }
                 (if (not (in_list val FUNC_NAMES))
                     { (set! ok #f) (set! v_reason (String.concat "不支持的函数：" val)) }
                     (if (or (>= (+ i 1) n)
                             (not (eq? 'bracket (get_item (get_item tokens (+ i 1)) 1)))
                             (not (String.equals "(" (get_item (get_item tokens (+ i 1)) 2)))
                             (>= (+ i 2) n)
                             (and (eq? 'bracket (get_item (get_item tokens (+ i 2)) 1))
                                  (String.equals ")" (get_item (get_item tokens (+ i 2)) 2))))
                         { (set! ok #f) (set! v_reason "函数调用缺少参数") }
                         ;; 通过：不消费 '('，交给下一轮的 bracket 分支开括号（depth+1）
                         (set! want_operand #t)))))
            ((eq? ty 'operator)
             (if (not (in_list val OP_NAMES))
                 { (set! ok #f) (set! v_reason (String.concat "不支持的运算符：" val)) }
                 (if (String.equals val "-")
                     (set! want_operand #t)            ;; 一元负号或二元减号均可
                     (if want_operand
                         { (set! ok #f) (set! v_reason "缺少左操作数") }
                         (set! want_operand #t)))))
            (else { (set! ok #f) (set! v_reason "输入中有无法识别的符号") }))
      (set! i (+ i 1))
    })
    (if (not ok) #f
        (if want_operand #f
            (if (not (== depth 0)) { (set! v_reason "括号不匹配") #f } #t)))))

;; 一站式求值：表达式字符串 → 数值；失败返回符号 'syntax 或 'value
;; （原因经 calc_reason 取；界面只做文案，不做语法/求值判断）
(define calc
  (lambda (s)
    (define toks (lexer s))
    (define ast 0)
    (define v 0)
    (set! c_reason "表达式不完整")
    (if (not (validate toks))
        'syntax
        {
          (set! ast (parser toks))
          (if (not (list? ast))
              { (set! c_reason "表达式语法错误") 'syntax }
              {
                (set! v (eval ast))
                (if (number? v)
                    v
                    { (set! c_reason "无法计算（除数为零或结果非实数）") 'value })
              })
        })))

;; 失败原因文案（对上 calc 的返回值）
(define calc_reason
  (lambda (v)
    (if (eq? v 'syntax) v_reason c_reason)))


(define show_tokens
  (lambda (tokens)
    (if (null? tokens) {
      (newline)
    } {
      (display (get_item (car tokens) 2))
      (display " | ")
      (show_tokens (cdr tokens))
    })))

(define show_ast
  (lambda (ast level)
    (cond ((not (list? ast)) #f)
          ((number? (get_item ast 0)) (display (get_item ast 2)))
          (else {
            (define i 0)
            (display "( ")
            (show_ast (get_item ast 0) (+ level 1))
            (display "\n  ")
            (set! i 0) (while (< i level) { (display "  ") (set! i (+ i 1)) })
            (show_ast (get_item ast 1) (+ level 1))
            (display "\n  ")
            (set! i 0) (while (< i level) { (display "  ") (set! i (+ i 1)) })
            (show_ast (get_item ast 2) (+ level 1))
            (display "\n")
            (set! i 0) (while (< i level) { (display "  ") (set! i (+ i 1)) })
            (display ")")
          }))))


