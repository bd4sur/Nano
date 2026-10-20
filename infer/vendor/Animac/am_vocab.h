#ifndef __AM_VOCAB_H__
#define __AM_VOCAB_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "am_object.h"
#include "am_allocator.h"


// 基础数据结构 vocab：词典，即字符串集合，实现为不重复的字符串数组，保证相同的字符串在数组中至多存在一个
// 该数据结构用于编译阶段，记录variable和symbol的集合，并通过递增index为其赋予am_varid_t或am_symbol_t
typedef struct am_vocab_t {
    am_object_t base;

    size_t  capacity; // words数组的容量
    size_t  length;   // words数组实际容纳的元素数
    wchar_t *words[]; // 弹性数组
} am_vocab_t;




// 创建词典对象，其中vocab->words初始化为长度为capacity的全0数组。
am_vocab_t *am_vocab_create(am_allocator_t *alloc,size_t capacity);

// 销毁词典对象，穿透销毁vocab->words的每一项指向的字符串。
// vocab 为 NULL 时视为成功。成功返回 0，失败返回 -1。
int32_t am_vocab_destroy(am_allocator_t *alloc,am_vocab_t *vocab);

// 深拷贝词典对象
am_vocab_t *am_vocab_copy(am_allocator_t *alloc, am_vocab_t *vocab);

// 功能说明：将词典对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       将words所指向的wchar_t*宽字符串依次展平拼接，各字符串之间以L'\0'为间隔符，最后一个字符串以L'\0'结束。
//       压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分。
size_t am_vocab_dump(am_allocator_t *alloc, am_vocab_t *vocab, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的词典对象，构造词典对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_vocab_t对象的指针，失败则返回NULL。
am_vocab_t *am_vocab_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);


// 检查词典中是否存在某个词，返回其在vocab->words中的index。
// 实现提示：将word与vocab->words中的词逐个进行比较，如果有相同的，返回其index；
//           如果遍历完所有words也不存在相同的，返回SIZE_MAX，表示不存在。
size_t am_vocab_find(am_allocator_t *alloc,am_vocab_t *vocab, wchar_t *word);

// 向词典中插入一个词，返回新的容器对象指针；失败返回 NULL。
// 插入的 index 通过 out_index 输出；若 word 已存在，则返回原 vocab 指针并将已有 index 写入 out_index。
// 实现提示：首先检查word是否存在，若存在，则返回原指针并将已有 index 写入 out_index；
//           若不存在，则尝试在vocab->words尾部插入word，成功则更新length字段并将新 index 写入 out_index，返回新的 vocab 指针。
// 注意：插入过程可能触发扩容并重新分配vocab对象本身，因此调用者必须使用返回的指针替换原有容器对象指针。
am_vocab_t *am_vocab_insert(am_allocator_t *alloc, am_vocab_t *vocab, wchar_t *word, size_t *out_index);

// 根据index获取词
// 实现提示：直接返回vocab->words[index]
wchar_t *am_vocab_get(am_allocator_t *alloc,am_vocab_t *vocab, size_t *index);





#ifdef __cplusplus
}
#endif

#endif
