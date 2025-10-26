#ifndef __NANO_TRIE_H__
#define __NANO_TRIE_H__

#include <stdio.h>
#include <string.h>
#include "hashmap.h"

#define VOCAB_SIZE        (16384) // Trie树字符数。为效率考虑（避免动态内存分配），固定为16384。
#define INITIAL_POOL_SIZE (16384) // 初始内存池大小。


// Trie树的节点结构
struct TrieNode {
    uint32_t is_end_of_token;  // 标记是否为词的结尾
    uint32_t token_id;         // 保存词对应的ID
    struct TrieNode *children[VOCAB_SIZE]; // 子节点指针
};

// Trie树结构
struct Trie {
    struct TrieNode *root;       // 根节点
    struct TrieNode *node_pool;  // 内存池
    uint32_t pool_size;          // 内存池大小
    uint32_t next_free_node;     // 下一个可用节点的索引
};


// 扩展内存池
void expand_memory_pool(struct Trie *trie);

// 初始化一个Trie树
//   注：当前使用动态内存池的实现中，没有用到两个参数。仅为兼容性而保留。
struct Trie *new_trie(uint32_t vocab_size, uint32_t is_end_of_token);

// 从内存池中分配一个新的Trie节点
struct TrieNode *allocate_node(struct Trie *trie);

// 释放Trie树
void free_trie(struct Trie *trie);

// 向Trie树中增加一个词
int add_token(struct Trie *trie, uint32_t *token, uint32_t token_len, uint32_t token_id);

// 在Trie树中匹配一个词
int match_token(struct Trie *trie, uint32_t *pattern, uint32_t pattern_len, uint32_t *token_id);

uint32_t tokenize(struct Trie *vocab_trie, uint32_t *output_token_ids, const uint32_t *input_char_ids, uint32_t input_length, uint32_t max_token_length);


#endif