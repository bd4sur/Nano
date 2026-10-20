#ifndef __AM_PARSER_H__
#define __AM_PARSER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <wchar.h>

#include "am_ast.h"



/************

每个源码文件就是一个模块。
Parser的输入是模块，输出是模块的AST。
Linker将所有的AST链接成一个大AST。
Compiler将AST编译成ILCode，并封装成Module。



Parser需要维护的全局信息：
- am_ast_t AST（含tokens）
- List<am_value_t> node_stack; // 其元素是am_value_t，语义上可能是ptr、也可能是包括handle在内的imme，由使用者解释
- List<uint> state_stack;


# Parser：语法分析器（LL(1)递归下降分析器）设计原理说明

Parser的输入是Lexer输出的token序列，基于以下BNF文法对token序列进行解析，输出初步生成的AST。

输入代码必须是`((lambda () <code>))`格式。

BNF：
    <SourceCode> ::= (lambda () <TERM>*) CRLF
          <Term> ::= <SList> | <Lambda> | <Quote> | <Unquote> | <Quasiquote> | <Identifier>
         <SList> ::= ( <SListSeq> )
      <SListSeq> ::= <Term> <SListSeq> | ε
        <Lambda> ::= ( lambda <ArgList> <Body> )
       <ArgList> ::= ( <ArgListSeq> )
    <ArgListSeq> ::= <ArgIdentifier> <ArgListSeq> | ε
 <ArgIdentifier> ::= <Identifier>
          <Body> ::= <BodyTerm> <Body_>
         <Body_> ::= <BodyTerm> <Body_> | ε
      <BodyTerm> ::= <Term>
         <Quote> ::= ' <QuoteTerm> | ( quote <QuoteTerm> )
       <Unquote> ::= , <UnquoteTerm> | ( unquote <QuoteTerm> )
    <Quasiquote> ::= ` <QuasiquoteTerm> | ( quasiquote <QuoteTerm> )
     <QuoteTerm> ::= <Term>
   <UnquoteTerm> ::= <Term>
<QuasiquoteTerm> ::= <Term>
    <Identifier> ::= IDENTIFIER

# Analyser：AST作用域分析器设计原理说明

Analyser需要对AST做两趟扫描。分别是“词法作用域分析”和“变量换名”。经过这两趟扫描，原有AST被更新，同时得到作用域信息。

例如：以下Scheme代码中，外层x和内层x实际上是两个不同的变量

(define foo (lambda (x y) 
  (define bar (lambda (x) (+ x y)))
  (bar y)
))

语法分析Parse，得到AST之后，得到以下var_vocab：[0:foo, 1:x, 2:y, 3:bar]。随后对AST进行分析。

第一趟扫描“词法作用域分析”，发现两个作用域，分别是Scope-foo[x(1),y(2),bar(3)]和Scope-bar[x(1)]，此时尽管两个作用域都有同名的x（进而varid也相同，都是1），但如果某个x沿着作用域链上溯，第一个找到的x就是正确的约束变量，这说明分散在不同的scope中的同名x已经是词法作用域意义上不同的x了。因此，在现有的携带环境帧的带脏标记的闭包实现中，必须在编译阶段就将这两个x区分成两个不同的varid。此外，在这趟扫描中，也处理了作用域内的define的变量，这实质上是作用域范围内的、出现在lambda参数列表之外的一种全局绑定，几乎等同于JavaScript的var变量声明，或者约等于标准Scheme的letrec*。

第二趟扫描“变量换名”，将AST中所有的varid，根据其所在的scope，替换成全局唯一的、携带了词法作用域信息的、新的varid。例如，换名结果如下：

(define mod.0.foo (lambda (mod.0.x mod.0.y) 
  (define mod.0.bar (lambda (mod.1.x) (+ mod.1.x mod.0.y)))
  (mod.0.bar mod.0.y)
))

同时在var_vocab中追加[... 10:mod.0.foo , 11:mod.0.x , 12:mod.0.y , 13:mod.0.bar , 14:mod.1.x]。这样就实现了所有变量都可以通过其varid唯一确定其scope，而不致混淆。


# 实现说明

- 调用 am_parse(code, absolute_path) 即可完成词法分析、词汇表构建、语法分析和预处理指令解析。
- 返回的 am_ast_t 由调用者负责销毁。
- 若解析失败，返回 NULL。

************/


// 语法分析器入口。
// 输入：内存分配器 alloc、Scheme 源码 code、模块绝对路径 absolute_path、is_keep_free。
//       当 is_keep_free 为 0 时，保留现有逻辑；为 1 时，在 alpha-renaming 阶段将“未定义变量”的 var_type 设为 AM_VAR_TYPE_GLOBAL_FREE。
// 输出：解析得到的 AST；失败返回 NULL。
// 说明：code 与 absolute_path 由调用者所有；tokens 由返回的 AST 所有，随 AST 销毁而释放。
am_ast_t *am_parse(am_allocator_t *alloc, wchar_t *code, wchar_t *absolute_path, int32_t is_keep_free);


// 对 AST 执行整体的尾位置分析，将处于尾位置的 application 节点把柄记录到 ast->tailcall_handles。
// 通常在 am_link 完成所有模块合并后调用；也可在独立使用 am_parser 后手动调用。
// 调用前会清空已有的 tailcall_handles。
// 成功返回 0，失败返回 -1。
int32_t am_parser_tail_call_analysis(am_ast_t *ast);


#ifdef __cplusplus
}
#endif

#endif
