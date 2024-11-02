#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

#include "trie.h"
#include "infer.h"



int main(int argc, char **argv) {
    if(!setlocale(LC_CTYPE, "")) {
        fprintf(stderr, "Can't set the specified locale! Check LANG, LC_CTYPE, LC_ALL.\n");
        return -1;
    }

    LLM llm;
    Tokenizer tokenizer;
    build_transformer(&llm, &tokenizer, "/home/bd4sur/ai/Nano/test.bin");

    wchar_t *text = L"<|bos|>你好！Nano是<|BD4SUR|>开发的大模型，是一只电子鹦鹉parrot，它说人类的本质就是复读机repeater。在一本哲学著作的序言里，如果也象在普通的书序里惯常所做的那样先作一个声明，以说明作者所怀抱的著述目的和动机以及作者所认为他的著作与这同一问题上早期和同时的其他论著的关系，那么这样的一种声明似乎不仅是多余的，而且就一部哲学著作的性质来说是不适宜的、不合目的的。因为，在一篇序言里，不论对哲学作出怎么样周详的陈述，比如说，给哲学的趋势和观点、一般内容和结果作一种历史性的叙述，或就真理问题上各家各派的主张和断言作一种兼容并蓄的罗列，如此等等，毕竟不能算是适合于陈述哲学真理的方式和办法。而且，由于在本质上哲学所探讨的那种普遍性的因素本身就包含着特殊，所以在哲学里比在其他科学里更容易使人觉得，仿佛就在目的或最终结果里事情自身甚至其全部本质都已得到了表达，至于实现过程，与此结果相比，则根本不是什么本质的事情。相反，譬如在解剖学是什么这样的一般观念里，我们则深信我们尚未占有事实本身，尚未占有这门科学的内容，而必须进一步去探讨特殊。——再者，在这样一种不配被称之为科学的知识堆积里，谈论目的之类普遍性的东西时所采用的方式，通常也就是叙述内容本身如神经、肌肉等等时所使用的那种历史性的无概念的方式，两者没有什么不同。但在哲学里，如果也采取这样的一种方式先作说明，而哲学本身随后又证明这种方式不能把握真理，那就很不一致了。";
    uint32_t token_count = 0;

    uint32_t *token_ids = encode(&tokenizer, text, &token_count);

    printf("Token count = %d\n", token_count);
    for(int i = 0; i < token_count; i++) {
        printf("%ls | ", tokenizer.token_list[token_ids[i]]);
    }
    printf("\n");

    wchar_t *out = decode(&tokenizer, token_ids, token_count);
    printf("%ls", out);
    printf("\n");

    return 0;
}
