#include "prompt.h"

#include <wchar.h>

static wchar_t *default_prompts[] = {
    L"人类的本质是什么？",
    L"人类的本质是复读机吗？",
    L"天空为什么是蓝色的？",
    L"你是谁训练的大模型？",
    L"西红柿炒鸡蛋怎么做？",
    L"为什么天上的星星会一闪一闪？",
    L"射频功率-40dBm的一半是多少dBm？",
    L"质权、典权、留置权和抵押权的区别是什么？",
    L"太阳系中最小的大行星是哪个？",
    L"中国十大名校有哪些？",
    L"证明$\\sqrt{2}$是无理数。",
    L"为什么阿塔卡马沙漠位于大洋沿岸却极度干燥？",
    L"澳大利亚的首都在哪里？",
    L"如何构造Quine，也就是自己输出自己的程序？",
    L"将以下句子翻译为英文：“人工智能即将统治人类！”",
    L"写一篇申论，题目是《绿水青山就是金山银山》。",
    L"光速是多少？",
    L"原样复述引号中内容：“我叫Nano，是BD4SUR训练的电子鹦鹉。”",
    L"列举一些常见的逻辑谬误。",
    L"9.9和9.11哪个大？"
};

void set_random_prompt(wchar_t *dest, uint64_t seed) {
    uint64_t state = seed;
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    uint32_t index = (state * 0x2545F4914F6CDD1Dull) >> 32;
    wcscpy(dest, default_prompts[index % 20]);
}
