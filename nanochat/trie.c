// ===============================================================================
// 词元编解码、分词器（基于Trie树）
// ===============================================================================

#include "trie.h"

// Trie树


// 扩展内存池
void expand_memory_pool(struct Trie *trie) {
    uint32_t new_pool_size = trie->pool_size * 2; // 新内存池大小
    struct TrieNode *new_pool = (struct TrieNode *)calloc(new_pool_size, sizeof(struct TrieNode));
    if (!new_pool) {
        printf("Failed to allocate expanded memory pool\n");
        exit(EXIT_FAILURE);
    }

    // 拷贝旧内存池的内容到新池中
    memcpy(new_pool, trie->node_pool, trie->pool_size * sizeof(struct TrieNode));
    free(trie->node_pool);

    trie->node_pool = new_pool;
    trie->pool_size = new_pool_size;
    printf("Memory pool expanded to %u nodes\n", trie->pool_size);
}

// 初始化一个Trie树
//   注：当前使用动态内存池的实现中，没有用到两个参数。仅为兼容性而保留。
struct Trie *new_trie(uint32_t vocab_size, uint32_t is_end_of_token) {
    struct Trie *trie = (struct Trie *)malloc(sizeof(struct Trie));
    if (!trie) {
        printf("Failed to allocate memory for Trie\n");
        exit(EXIT_FAILURE);
    }

    trie->pool_size = INITIAL_POOL_SIZE;
    trie->node_pool = (struct TrieNode *)calloc(trie->pool_size, sizeof(struct TrieNode));
    if (!trie->node_pool) {
        printf("Failed to allocate initial memory pool for Trie nodes\n");
        free(trie);
        exit(EXIT_FAILURE);
    }

    trie->next_free_node = 0;
    trie->root = &trie->node_pool[trie->next_free_node++];
    return trie;
}

// 从内存池中分配一个新的Trie节点
struct TrieNode *allocate_node(struct Trie *trie) {
    if (trie->next_free_node >= trie->pool_size) {
        expand_memory_pool(trie); // 扩展内存池
    }
    return &trie->node_pool[trie->next_free_node++];
}

// 释放Trie树
void free_trie(struct Trie *trie) {
    if (trie) {
        free(trie->node_pool);
        free(trie);
    }
}

// 向Trie树中增加一个词
int add_token(struct Trie *trie, uint32_t *token, uint32_t token_len, uint32_t token_id) {
    if (!trie || !token || token_len == 0) return -1;

    struct TrieNode *node = trie->root;
    for (uint32_t i = 0; i < token_len; ++i) {
        uint32_t index = token[i];
        if (index >= VOCAB_SIZE) return -1; // 超出VOCAB_SIZE范围

        if (!node->children[index]) {
            node->children[index] = allocate_node(trie);
        }
        node = node->children[index];
    }
    if (node->is_end_of_token) return -1; // 防止重复添加

    node->is_end_of_token = 1;
    node->token_id = token_id;
    return 0;
}

// 在Trie树中匹配一个词
int match_token(struct Trie *trie, uint32_t *pattern, uint32_t pattern_len, uint32_t *token_id) {
    if (!trie || !pattern || pattern_len == 0) return -1;

    struct TrieNode *node = trie->root;
    for (uint32_t i = 0; i < pattern_len; ++i) {
        uint32_t index = pattern[i];
        if (index >= VOCAB_SIZE || !node->children[index]) return -1;

        node = node->children[index];
    }
    if (node->is_end_of_token) {
        if (token_id) *token_id = node->token_id;
        return 0;
    }
    return -1;
}

uint32_t tokenize(struct Trie *vocab_trie, uint32_t *output_token_ids, const uint32_t *input_char_ids, uint32_t input_length, uint32_t max_token_length) {
    uint32_t token_count = 0;
    uint32_t pos = 0;
    // uint32_t prefix[MAX_TOKEN_LENGTH];
    while(pos < input_length) {
        uint32_t available_max_token_length = (input_length - pos < max_token_length) ? (input_length - pos) : max_token_length;
        for(uint32_t n = available_max_token_length; n > 0; n--) {
            uint32_t *prefix = (uint32_t*)calloc(n, sizeof(uint32_t));
            uint32_t tid = 0;
            for(uint32_t i = 0; i < n; i++) {
                prefix[i] = input_char_ids[pos + i];
            }
            if(n == 1 || match_token(vocab_trie, prefix, n, &tid) == 0) {
                output_token_ids[token_count] = (n == 1) ? prefix[0] : tid;
                token_count++;
                pos += n;
                break;
            }
            free(prefix);
        }
    }
    return token_count;
}
