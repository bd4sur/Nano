#include "utils.h"

// ===============================================================================
// HashMap
// ===============================================================================

struct Map *new_map(uint32_t bucket_num) {
    struct Map *pm = (struct Map *)calloc(1, sizeof(struct Map));
    pm->bucket_num = bucket_num;
    pm->buckets = (struct Entry **)calloc(bucket_num, sizeof(struct Entry *));
    for(uint32_t i = 0; i < bucket_num; i++) {
        pm->buckets[i] = NULL;
    }
    return pm;
}

void free_map(struct Map *pm) {
    for(uint32_t i = 0; i < pm->bucket_num; i++) {
        struct Entry *current_entry = pm->buckets[i];
        while(NULL != current_entry) {
            struct Entry *next_entry = current_entry->next;
            free(current_entry);
            current_entry = next_entry;
        }
    }
    free(pm->buckets);
    free(pm);
}

struct Entry *new_entry(uint32_t key, uint32_t value) {
    struct Entry *e = (struct Entry *)calloc(1, sizeof(struct Entry));
    e->key = key;
    e->value = value;
    e->next = NULL;
    return e;
}

uint32_t map_hash(uint32_t key, uint32_t bucket_num) {
    // TODO 有巨大优化空间：大部分汉字是0x4e00~0xa000
    return key % bucket_num;
}

uint32_t map_set(struct Map *m, uint32_t key, uint32_t value) {
    uint32_t hashcode = map_hash(key, m->bucket_num);
    struct Entry *e = m->buckets[hashcode];
    if(NULL == e) {
        m->buckets[hashcode] = new_entry(key, value);
    }
    else {
        while(NULL != e->next) {
            e = e->next;
        }
        e->next = new_entry(key, value);
    }
    return hashcode;
}

uint32_t map_get(struct Map *m, uint32_t key) {
    uint32_t hashcode = map_hash(key, m->bucket_num);
    struct Entry *e = m->buckets[hashcode];
    if(NULL == e) {
        return 0; // 等同于<|padding|>
    }
    else {
        do {
            if(e->key == key) {
                return e->value;
            }
            e = e->next;
        } while(NULL != e);
    }
    return 0;
}


// ===============================================================================
// Trie树
// ===============================================================================

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
struct Trie *new_trie(uint32_t vocab_size, uint8_t is_end_of_token) {
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

// 从内存池中分配一个新的Trie节点，返回它在节点内存池中的索引
uint32_t allocate_node(struct Trie *trie) {
    if (trie->next_free_node >= trie->pool_size) {
        expand_memory_pool(trie); // 扩展内存池
    }
    return trie->next_free_node++;
}

// 释放Trie树
void free_trie(struct Trie *trie) {
    if (trie) {
        for (int32_t i = 0; i < trie->next_free_node; i++) {
            struct TrieNode *node = &trie->node_pool[i];
            if (node->children) {
                free_map(node->children);
            }
        }
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

        uint32_t child_node_index = 0;
        if (!node->children) {
            node->children = new_map(30);
            child_node_index = allocate_node(trie);
            map_set(node->children, index, child_node_index);
        }
        else {
            child_node_index = map_get(node->children, index);
            if (!child_node_index) { // 返回0就是不存在（没找到）
                child_node_index = allocate_node(trie);
                map_set(node->children, index, child_node_index);
            }
        }
        node = &trie->node_pool[child_node_index];
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
        uint32_t child_node_index = 0;

        if (index >= VOCAB_SIZE) return -1;

        if (!node->children) {
            return -1;
        }
        else {
            child_node_index = map_get(node->children, index);
            if (!child_node_index) return -1;
            node = &trie->node_pool[child_node_index];
        }
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


// ===============================================================================
// AVL树
// ===============================================================================

uint32_t max(uint32_t a, uint32_t b) {
    return (a > b) ? a : b;
}

int height(AVLNode* node) {
    if (node == NULL)
        return 0;
    return node->height;
}

int balanceFactor(AVLNode* node) {
    if (node == NULL)
        return 0;
    return height(node->left) - height(node->right);
}

AVLNode* newNode(uint32_t key, uint32_t index) {
    AVLNode* node = (AVLNode*)malloc(sizeof(AVLNode));
    node->key = key;
    node->index = index;
    node->left = NULL;
    node->right = NULL;
    node->height = 1;
    return node;
}

AVLNode* rightRotate(AVLNode* y) {
    AVLNode* x = y->left;
    AVLNode* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = max(height(y->left), height(y->right)) + 1;
    x->height = max(height(x->left), height(x->right)) + 1;

    return x;
}

AVLNode* leftRotate(AVLNode* x) {
    AVLNode* y = x->right;
    AVLNode* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = max(height(x->left), height(x->right)) + 1;
    y->height = max(height(y->left), height(y->right)) + 1;

    return y;
}

AVLNode* insert(AVLNode* node, uint32_t key, uint32_t index) {
    if (node == NULL)
        return newNode(key, index);

    if (key < node->key)
        node->left = insert(node->left, key, index);
    else if (key > node->key)
        node->right = insert(node->right, key, index);
    else
        return node;

    node->height = 1 + max(height(node->left), height(node->right));

    int balance = balanceFactor(node);

    if (balance > 1) {
        if (key < node->left->key) {
            return rightRotate(node);
        } else {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }
    }

    if (balance < -1) {
        if (key > node->right->key) {
            return leftRotate(node);
        } else {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }
    }

    return node;
}

AVLNode* buildAVLTree(uint32_t arr[], uint32_t n) {
    AVLNode* root = NULL;
    for (int i = 0; i < n; i++) {
        root = insert(root, arr[i], i);
    }
    return root;
}

uint32_t findIndex(AVLNode* root, uint32_t key) {
    AVLNode* current = root;
    while (current != NULL) {
        if (key == current->key) {
            return current->index;
        } else if (key < current->key) {
            current = current->left;
        } else {
            current = current->right;
        }
    }
    return -1;
}

void freeTree(AVLNode* root) {
    if (root != NULL) {
        freeTree(root->left);
        freeTree(root->right);
        free(root);
    }
}


// ===============================================================================
// 其他平台无关的工具函数
// ===============================================================================

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random float32 in [0,1)
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}
