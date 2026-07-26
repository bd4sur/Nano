#ifndef PINYIN_IME_H
#define PINYIN_IME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 拼音转候选汉字（完全匹配，忽略音调）
 * @param pinyin            输入拼音（无音调数字），如 "di"
 * @param hanzi_candidates  输出缓冲区，写入 UTF-32 码点
 * @return 候选汉字数量
 *
 * @note 调用者需保证 hanzi_candidates 有足够的元素。
 * @note 返回的候选字已按字频降序预排列，无需运行时排序。
 */
size_t pinyin_to_hanzi(char *pinyin, uint32_t *hanzi_candidates);

#ifdef __cplusplus
}
#endif

#endif /* PINYIN_IME_H */
