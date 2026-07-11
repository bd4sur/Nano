#include "utils.h"

#include <stdint.h>
#include <math.h>

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
// Radix Tree（基数树 / 压缩前缀树）
// ===============================================================================

// 创建一个新的 Radix 节点
static struct RadixNode *new_radix_node(void) {
    struct RadixNode *node = (struct RadixNode *)calloc(1, sizeof(struct RadixNode));
    if (!node) {
        printf("Failed to allocate RadixNode\n");
        exit(EXIT_FAILURE);
    }
    return node;
}

// 递归释放 Radix 节点及其所有子树
static void free_radix_node(struct RadixNode *node) {
    if (!node) return;
    if (node->prefix) {
        free(node->prefix);
    }
    for (uint32_t i = 0; i < node->num_children; i++) {
        free_radix_node(node->children[i]);
    }
    if (node->children) {
        free(node->children);
    }
    free(node);
}

// 在父节点的子节点中查找首元素与给定 key 匹配的子节点
static struct RadixNode *find_child(struct RadixNode *parent, uint32_t key) {
    for (uint32_t i = 0; i < parent->num_children; i++) {
        if (parent->children[i]->prefix_len > 0 && parent->children[i]->prefix[0] == key) {
            return parent->children[i];
        }
    }
    return NULL;
}

// 向父节点添加一个子节点
static void add_child(struct RadixNode *parent, struct RadixNode *child) {
    if (parent->num_children >= parent->child_capacity) {
        uint32_t new_cap = parent->child_capacity == 0 ? 4 : parent->child_capacity * 2;
        struct RadixNode **new_children = (struct RadixNode **)realloc(parent->children, new_cap * sizeof(struct RadixNode *));
        if (!new_children) {
            printf("Failed to realloc children array\n");
            exit(EXIT_FAILURE);
        }
        parent->children = new_children;
        parent->child_capacity = new_cap;
    }
    parent->children[parent->num_children++] = child;
}

// 初始化一个 Trie 树（内部使用 Radix Tree 实现）
//   注：参数 vocab_size 和 is_end_of_token 仅为兼容性而保留。
struct Trie *new_trie(uint32_t vocab_size, uint8_t is_end_of_token) {
    (void)vocab_size;
    (void)is_end_of_token;
    struct Trie *trie = (struct Trie *)malloc(sizeof(struct Trie));
    if (!trie) {
        printf("Failed to allocate memory for Trie\n");
        exit(EXIT_FAILURE);
    }
    trie->root = new_radix_node();
    return trie;
}

// 释放 Trie 树
void free_trie(struct Trie *trie) {
    if (trie) {
        free_radix_node(trie->root);
        free(trie);
    }
}

// 向 Trie 树中增加一个词
int add_token(struct Trie *trie, uint32_t *token, uint32_t token_len, uint32_t token_id) {
    if (!trie || !token || token_len == 0) return -1;

    // 合法性校验
    for (uint32_t i = 0; i < token_len; i++) {
        if (token[i] >= VOCAB_SIZE) return -1;
    }

    struct RadixNode *node = trie->root;
    uint32_t pos = 0;

    while (pos < token_len) {
        struct RadixNode *child = find_child(node, token[pos]);
        if (!child) {
            // 没有匹配的子节点，直接创建新节点并挂载
            struct RadixNode *new_node = new_radix_node();
            new_node->prefix_len = token_len - pos;
            new_node->prefix = (uint32_t *)malloc(new_node->prefix_len * sizeof(uint32_t));
            if (!new_node->prefix) {
                printf("Failed to allocate prefix\n");
                exit(EXIT_FAILURE);
            }
            memcpy(new_node->prefix, token + pos, new_node->prefix_len * sizeof(uint32_t));
            new_node->is_end_of_token = 1;
            new_node->token_id = token_id;
            add_child(node, new_node);
            return 0;
        }

        // 计算当前待插入序列与 child 前缀的公共长度
        uint32_t common_len = 0;
        uint32_t min_len = (token_len - pos < child->prefix_len) ? (token_len - pos) : child->prefix_len;
        while (common_len < min_len && token[pos + common_len] == child->prefix[common_len]) {
            common_len++;
        }

        if (common_len == child->prefix_len && common_len == token_len - pos) {
            // 完全匹配已有节点
            if (child->is_end_of_token) return -1; // 防止重复添加
            child->is_end_of_token = 1;
            child->token_id = token_id;
            return 0;
        }
        else if (common_len < child->prefix_len) {
            // 需要分裂 child 节点
            struct RadixNode *split = new_radix_node();
            split->prefix_len = common_len;
            split->prefix = (uint32_t *)malloc(common_len * sizeof(uint32_t));
            if (!split->prefix) {
                printf("Failed to allocate prefix for split\n");
                exit(EXIT_FAILURE);
            }
            memcpy(split->prefix, child->prefix, common_len * sizeof(uint32_t));

            // 调整原 child 节点的前缀为剩余部分
            uint32_t remaining = child->prefix_len - common_len;
            uint32_t *new_prefix = (uint32_t *)malloc(remaining * sizeof(uint32_t));
            if (!new_prefix) {
                printf("Failed to allocate new prefix\n");
                exit(EXIT_FAILURE);
            }
            memcpy(new_prefix, child->prefix + common_len, remaining * sizeof(uint32_t));
            free(child->prefix);
            child->prefix = new_prefix;
            child->prefix_len = remaining;

            // 将原 child 挂载到 split 下
            add_child(split, child);

            // 在 node 的子节点列表中用 split 替换 child
            for (uint32_t i = 0; i < node->num_children; i++) {
                if (node->children[i] == child) {
                    node->children[i] = split;
                    break;
                }
            }

            if (common_len == token_len - pos) {
                // 插入的词在 split 处结束
                split->is_end_of_token = 1;
                split->token_id = token_id;
            } else {
                // 还需要创建一个剩余部分的新节点挂载到 split 下
                struct RadixNode *new_node = new_radix_node();
                new_node->prefix_len = token_len - pos - common_len;
                new_node->prefix = (uint32_t *)malloc(new_node->prefix_len * sizeof(uint32_t));
                if (!new_node->prefix) {
                    printf("Failed to allocate suffix prefix\n");
                    exit(EXIT_FAILURE);
                }
                memcpy(new_node->prefix, token + pos + common_len, new_node->prefix_len * sizeof(uint32_t));
                new_node->is_end_of_token = 1;
                new_node->token_id = token_id;
                add_child(split, new_node);
            }
            return 0;
        }
        else {
            // common_len == child->prefix_len < token_len - pos
            // 继续向下匹配
            pos += common_len;
            node = child;
        }
    }

    return -1; // 理论上不会到达此处
}

// 在 Trie 树中匹配一个词
int match_token(struct Trie *trie, uint32_t *pattern, uint32_t pattern_len, uint32_t *token_id) {
    if (!trie || !pattern || pattern_len == 0) return -1;

    // 合法性校验
    for (uint32_t i = 0; i < pattern_len; i++) {
        if (pattern[i] >= VOCAB_SIZE) return -1;
    }

    struct RadixNode *node = trie->root;
    uint32_t pos = 0;

    while (pos < pattern_len) {
        struct RadixNode *child = find_child(node, pattern[pos]);
        if (!child) return -1;

        uint32_t remaining = pattern_len - pos;
        if (remaining < child->prefix_len) return -1;

        for (uint32_t i = 0; i < child->prefix_len; i++) {
            if (pattern[pos + i] != child->prefix[i]) return -1;
        }

        pos += child->prefix_len;
        node = child;
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
                free(prefix);
                break;
            }
            free(prefix);
        }
    }
    return token_count;
}


// ===============================================================================
// 对有序数组的二分查找
// ===============================================================================

int32_t binary_search(const uint32_t *lut_sorted, const uint32_t *lut_indexs, uint32_t n, uint32_t target) {
    uint32_t left = 0;
    uint32_t right = n - 1;
    while (left <= right) {
        uint32_t mid = left + (right - left) / 2; // 防止溢出
        if (lut_sorted[mid] == target) {
            return lut_indexs[mid];
        }
        else if (lut_sorted[mid] < target) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
    }
    return -1;
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
// 对字符串列表进行原地排序
// ===============================================================================

static int _sort_strings_compare_asc(const void *a, const void *b) {
    const char *sa = *(const char **)a;
    const char *sb = *(const char **)b;
    if (sa == NULL && sb == NULL) return 0;
    if (sa == NULL) return -1;
    if (sb == NULL) return 1;
    return strcmp(sa, sb);
}

// order 0-升序 1-降序
int32_t sort_strings(char **strings, int32_t num, int32_t order) {
    if (strings == NULL || num < 0) return -1;
    if (num <= 1) return 0;
    if (order != 0 && order != 1) return -1;

    qsort(strings, (size_t)num, sizeof(char *), _sort_strings_compare_asc);

    if (order == 1) {
        int32_t i = 0, j = num - 1;
        while (i < j) {
            char *tmp = strings[i];
            strings[i] = strings[j];
            strings[j] = tmp;
            i++;
            j--;
        }
    }

    return 0;
}


// ===============================================================================
// 数字格式字符串
// ===============================================================================

// 获取十进制整数 num 的第 n 位（从右往左，个位是第 0 位）
uint8_t get_digit(int32_t num, int32_t n) {
    if (n < 0) return (uint8_t)'?';
    num = abs(num);
    for (int i = 0; i < n; i++) {
        num /= 10;
        if (num == 0) return '0';
    }
    return (uint8_t)(num % 10 + '0');
}

// 将浮点数格式的时区转为 shhmm 格式，并返回指定位置的 ASCII 字符
// shhmm 格式说明：
//   位置 0: 符号位 ('+' 或 '-')
//   位置 1: 小时十位 (0-9)
//   位置 2: 小时个位 (0-9)
//   位置 3: 分钟十位 (0-5)
//   位置 4: 分钟个位 (0-9)
// 注意：时区 0.0 视为 "+0000"
uint8_t get_timezone_digit(float tz, int32_t digit) {
    // 1. 处理符号位（0.0 视为正）
    char sign = (tz < 0.0f && tz != -0.0f) ? '-' : '+';

    // 2. 取绝对值（处理 -0.0 的特殊情况）
    float abs_tz = fabsf(tz);

    // 3. 分解小时和分钟（四舍五入到最近分钟）
    int hours = (int)floorf(abs_tz);          // 小时整数部分
    float fractional = abs_tz - hours;        // 小数部分
    int minutes = (int)roundf(fractional * 60.0f); // 转换为分钟并四舍五入

    // 4. 处理分钟进位（例如 59.9 分钟 -> 60 分钟 -> 进位）
    if (minutes >= 60) {
        hours += minutes / 60;
        minutes %= 60;
    }

    // 5. 限制小时范围（时区标准范围 -12~+14，但保留健壮性）
    if (hours > 99) hours = 99;  // 极端情况保护

    // 6. 根据 digit 位置返回对应 ASCII 字符
    switch (digit) {
        case 0:  // 符号位
            return (uint8_t)sign;
        case 1:  // 小时十位
            return (uint8_t)('0' + (hours / 10));
        case 2:  // 小时个位
            return (uint8_t)('0' + (hours % 10));
        case 3:  // 分钟十位
            return (uint8_t)('0' + (minutes / 10));
        case 4:  // 分钟个位
            return (uint8_t)('0' + (minutes % 10));
        default: // 无效位置返回空字符（或根据需求返回错误码）
            return (uint8_t)'\0';
    }
}



// 将十进制格式的浮点数经度或者纬度转换为shhhmmss格式，并返回指定位置的 ASCII 字符
// shhhmmss格式说明：
//   位置 0: 符号位 ('+' 或 '-')，经纬度0的符号一律为'+'
//   位置 1: 小时百位 (0-9)
//   位置 2: 小时十位 (0-9)
//   位置 3: 小时个位 (0-9)
//   位置 4: 分钟十位 (0-9)
//   位置 5: 分钟个位 (0-9)
//   位置 6: 秒数十位 (0-9)
//   位置 7: 秒数个位 (0-9)
uint8_t get_lon_lat_digit(float decimal, int32_t digit) {
    // 1. 确定符号位：0值（含-0.0）统一使用'+'
    char sign = '+';
    if (decimal < -1e-6f) {
        sign = '-';
    }

    // 2. 取绝对值进行度分秒计算
    float abs_val = fabsf(decimal);

    // 3. 计算度、分、秒（带四舍五入）
    int degrees = (int)abs_val;
    float minutes_frac = (abs_val - degrees) * 60.0f;
    int minutes = (int)minutes_frac;
    int seconds = (int)((minutes_frac - minutes) * 60.0f + 0.5f);  // 四舍五入

    // 4. 处理进位（秒→分→度）
    if (seconds >= 60) {
        seconds -= 60;
        minutes++;
    }
    if (minutes >= 60) {
        minutes -= 60;
        degrees++;
    }

    // 5. 根据digit位置返回对应ASCII字符
    switch (digit) {
        case 0:  // 符号位
            return (uint8_t)sign;
        case 1:  // 度百位 (0-1，经度最大180，纬度最大90)
            return (uint8_t)('0' + (degrees / 100) % 10);
        case 2:  // 度十位
            return (uint8_t)('0' + (degrees / 10) % 10);
        case 3:  // 度个位
            return (uint8_t)('0' + degrees % 10);
        case 4:  // 分十位
            return (uint8_t)('0' + (minutes / 10) % 10);
        case 5:  // 分个位
            return (uint8_t)('0' + minutes % 10);
        case 6:  // 秒十位
            return (uint8_t)('0' + (seconds / 10) % 10);
        case 7:  // 秒个位
            return (uint8_t)('0' + seconds % 10);
        default: // 无效位置返回问号
            return (uint8_t)'?';
    }
}


static int32_t _days_in_month(int32_t y, int32_t m) {
    static const int32_t dim[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (m == 2 && ((y % 4 == 0 && y % 100 != 0) || (y % 400 == 0))) {
        return 29;
    }
    return dim[m - 1];
}

void local_time_to_utc(
    int32_t year, int32_t month, int32_t day,
    int32_t hour, int32_t minute, int32_t second,
    float timezone,
    int32_t *utc_year, int32_t *utc_month, int32_t *utc_day,
    int32_t *utc_hour, int32_t *utc_minute, int32_t *utc_second
) {
    // timezone 为正表示东半球（如北京时间 +8.0），为负表示西半球
    // UTC = 地方时 - timezone
    int32_t offset_minutes_total = (int32_t)(timezone * 60.0f);
    int32_t offset_hours = offset_minutes_total / 60;
    int32_t offset_mins  = offset_minutes_total % 60;

    int32_t s = second;
    int32_t mi = minute - offset_mins;
    int32_t h = hour - offset_hours;
    int32_t d = day;
    int32_t mo = month;
    int32_t y = year;

    while (s < 0) { s += 60; mi -= 1; }
    while (s >= 60) { s -= 60; mi += 1; }

    while (mi < 0) { mi += 60; h -= 1; }
    while (mi >= 60) { mi -= 60; h += 1; }

    while (h < 0) { h += 24; d -= 1; }
    while (h >= 24) { h -= 24; d += 1; }

    while (d < 1) {
        mo -= 1;
        if (mo < 1) { mo = 12; y -= 1; }
        d += _days_in_month(y, mo);
    }
    while (d > _days_in_month(y, mo)) {
        d -= _days_in_month(y, mo);
        mo += 1;
        if (mo > 12) { mo = 1; y += 1; }
    }

    while (mo < 1) { mo += 12; y -= 1; }
    while (mo > 12) { mo -= 12; y += 1; }

    *utc_year = y;
    *utc_month = mo;
    *utc_day = d;
    *utc_hour = h;
    *utc_minute = mi;
    *utc_second = s;
}


// ===============================================================================
// 字符编码相关
// ===============================================================================

// 将 UTF-32 码点（wchar_t）数组转换为 UTF-8 字符串
// 注意：此函数假设 wchar_t 为 32 位（即 UTF-32），符合 ESP32 的配置（通常 -fshort-wchar 未启用）
uint32_t _wcstombs(char *dest, const wchar_t *src, uint32_t dest_size) {
    const wchar_t *p = src;
    char *q = dest;
    uint32_t dest_len = 0;
    char *end = dest + dest_size - 1; // 留 1 字节给 \0

    while (*p != L'\0' && q < end) {
        uint32_t cp = (uint32_t)*p++;
        if (cp <= 0x7F && q < end) {
            *q++ = (char)cp;
            dest_len++;
        } else if (cp <= 0x7FF && q + 1 < end) {
            *q++ = 0xC0 | (cp >> 6);
            *q++ = 0x80 | (cp & 0x3F);
            dest_len += 2;
        } else if (cp <= 0xFFFF && q + 2 < end) {
            *q++ = 0xE0 | (cp >> 12);
            *q++ = 0x80 | ((cp >> 6) & 0x3F);
            *q++ = 0x80 | (cp & 0x3F);
            dest_len += 3;
        } else if (cp <= 0x10FFFF && q + 3 < end) {
            *q++ = 0xF0 | (cp >> 18);
            *q++ = 0x80 | ((cp >> 12) & 0x3F);
            *q++ = 0x80 | ((cp >> 6) & 0x3F);
            *q++ = 0x80 | (cp & 0x3F);
            dest_len += 4;
        } else if (q < end) {
            *q++ = '?';
            dest_len++;
        }
    }
    *q = '\0';
    return dest_len;
}

// 将 UTF-8 字符串转换为 null-terminated 的 UTF-32 (wchar_t) 字符串
// 返回值：成功转换的 wchar_t 字符数量（不包括结尾的 L'\0'）
// 注意：dest 必须有至少 (length + 1) 个 wchar_t 的空间（最坏情况）
uint32_t _mbstowcs(wchar_t *dest, const char *src, uint32_t length) {
    const uint8_t *p = (const uint8_t *)src;
    const uint8_t *end = p + length;
    wchar_t *out = dest;

    while (p < end) {
        uint8_t byte = *p;
        if (byte == '\0') break;
        p++;

        if ((byte & 0x80) == 0) {
            // 1-byte: ASCII
            if (dest) *out++ = (wchar_t)byte;
            else out++;
        } else if ((byte & 0xE0) == 0xC0) {
            // 2-byte
            if (p >= end) goto invalid;
            uint8_t b2 = *p++;
            if ((b2 & 0xC0) != 0x80) goto invalid;
            uint32_t cp = ((byte & 0x1F) << 6) | (b2 & 0x3F);
            if (cp < 0x80) goto invalid; // overlong
            if (dest) *out++ = (wchar_t)cp;
            else out++;
        } else if ((byte & 0xF0) == 0xE0) {
            // 3-byte
            if (p + 1 >= end) goto invalid;
            uint8_t b2 = *p++;
            uint8_t b3 = *p++;
            if ((b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) goto invalid;
            uint32_t cp = ((byte & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
            if (cp < 0x800) goto invalid;
            if (cp >= 0xD800 && cp <= 0xDFFF) goto invalid; // surrogate
            if (dest) *out++ = (wchar_t)cp;
            else out++;
        } else if ((byte & 0xF8) == 0xF0) {
            // 4-byte
            if (p + 2 >= end) goto invalid;
            uint8_t b2 = *p++;
            uint8_t b3 = *p++;
            uint8_t b4 = *p++;
            if ((b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 || (b4 & 0xC0) != 0x80) goto invalid;
            uint32_t cp = ((byte & 0x07) << 18) | ((b2 & 0x3F) << 12) |
                          ((b3 & 0x3F) << 6) | (b4 & 0x3F);
            if (cp < 0x10000) goto invalid;
            if (cp > 0x10FFFF) goto invalid;
            if (dest) *out++ = (wchar_t)cp;
            else out++;
        } else {
            goto invalid;
        }
    }

    // 正常结束：添加 null 终止符
    *out = (wchar_t)0;
    return (uint32_t)(out - dest);

invalid:
    // 遇到无效 UTF-8：用 '?' 替代并终止
    if (dest) *out++ = (wchar_t)'?';
    else out++;
    if (dest) *out = (wchar_t)0;  // 仍然保证 null-terminated
    return (uint32_t)(out - dest); // 返回包含 '?' 的字符数
}



// ===============================================================================
// 预置提示词
// ===============================================================================

static const wchar_t *default_prompts[] = {
    L"人类的本质是什么？",
    L"人类的本质是复读机吗？",
    L"天空为什么是蓝色的？",
    L"你是谁训练的大模型？",
    L"西红柿炒鸡蛋怎么做？",
    L"我想洗车，洗车店离我家50米，应该开车去还是走路去？",
    L"射频功率-40dBm的一半是多少dBm？",
    L"质权、典权、留置权和抵押权的区别是什么？",
    L"太阳系中最小的大行星是哪个？",
    L"某业余电台操作者听到业余专用频率上出现某种显然出自非业余电台的人为干扰发射，于是按下话筒向该发射者宣传无线电管理法规知识。这种做法是正确的还是错误的？",
    L"证明$\\sqrt{2}$是无理数。",
    L"为什么阿塔卡马沙漠位于大洋沿岸却极度干燥？",
    L"澳大利亚的首都在哪里？",
    L"如何构造Quine，也就是自己输出自己的程序？",
    L"将以下句子翻译为英文：“人工智能即将统治人类！”",
    L"写一篇申论，题目是《绿水青山就是金山银山》。",
    L"1小时能晒干1条毛巾，晒干10条毛巾需要几小时？",
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



// ===============================================================================
// IMU坐标变换
// ===============================================================================

// 欧拉角转换：避免pitch=90°时的环架锁问题
void transform_euler_angles(float pitch_in, float roll_in, float yaw_in, float *pitch_out, float *roll_out, float *yaw_out) {
    const float degtorad = M_PI / 180.0f;
    const float radtodeg = 180.0f / M_PI;

    float alpha = yaw_in;
    float beta = pitch_in;
    float gamma = roll_in;
    
    // 计算输入角度的三角函数值
    float cX = cosf(beta * degtorad);   // cos(beta)
    float cY = cosf(gamma * degtorad);    // cos(gamma)
    float cZ = cosf(alpha * degtorad);  // cos(alpha)
    float sX = sinf(beta * degtorad);   // sin(beta)
    float sY = sinf(gamma * degtorad);    // sin(gamma)
    float sZ = sinf(alpha * degtorad);  // sin(alpha)
    
    // 计算旋转矩阵元素
    float m11 = cZ * cY - sZ * sX * sY;
    float m12 = - cX * sZ;
    float m13 = cY * sZ * sX + cZ * sY;

    float m21 = cY * sZ + cZ * sX * sY;
    float m22 = cZ * cX;
    float m23 = sZ * sY - cZ * cY * sX;

    float m31 = - cX * sY;
    float m32 = sX;
    float m33 = cX * cY;

    (void)m11; (void)m12;

    // 计算 sy 判断是否奇异
    float sy = sqrtf(m13 * m13 + m23 * m23);
    
    float x, y, z;
    
    if (sy >= 1e-6f) {
        x = atan2f(m31, m32);
        y = atan2f(-m33, sy);
        z = atan2f(m23, m13);
    } else {
        x = atan2f(-m22, m21);
        y = atan2f(-m33, sy);
        z = 0.0f;
    }
    
    *pitch_out = y * radtodeg;
    *roll_out = x * radtodeg;
    *yaw_out = z * radtodeg;
}


/**
 * 将四元数转换为 ZXY 欧拉角（Tait-Bryan 角）
 * 
 * 坐标系约定:
 *   - yaw   (alpha): 绕 Z 轴方位角，范围 [0°, 360°)
 *   - pitch (beta):  绕 X 轴俯仰角，范围 [-90°, 90°]
 *   - roll  (gamma): 绕 Y 轴横滚角，范围 (-180°, 180°]
 *
 * 参数:
 *   q0,q1,q2,q3 - 输入四元数分量 (x, y, z, w)
 *   pitch, roll, yaw - 输出指针，结果以度为单位
 */
void quaternion_to_euler(float q0, float q1, float q2, float q3, float *pitch, float *roll, float *yaw) {
    // 映射到标准 x,y,z,w 以便阅读
    const float x = q1;
    const float y = q2;
    const float z = q3;
    const float w = q0;
    
    const float RAD2DEG = 180.0f / M_PI;
    const float epsilon = 1e-6f;
    
    // ── 第一步：四元数 → 3×3 旋转矩阵 (R = Rz(yaw)*Rx(pitch)*Ry(roll)) ──
    const float R00 = 1.0f - 2.0f * (y*y + z*z);
    const float R01 = 2.0f * (x*y - w*z);
    const float R10 = 2.0f * (x*y + w*z);
    const float R11 = 1.0f - 2.0f * (x*x + z*z);
    const float R20 = 2.0f * (x*z - w*y);
    const float R21 = 2.0f * (y*z + w*x);
    const float R22 = 1.0f - 2.0f * (x*x + y*y);
    
    // ── 第二步：从旋转矩阵提取 ZXY 欧拉角 ──────────────────────────────
    // R[2][1] = sin(pitch)
    float sin_pitch = R21;
    
    // 钳制到 [-1, 1]，防止浮点误差导致 asin 返回 NaN
    if (sin_pitch > 1.0f)  sin_pitch = 1.0f;
    if (sin_pitch < -1.0f) sin_pitch = -1.0f;
    
    float pitch_rad = asinf(sin_pitch);
    float cos_pitch = cosf(pitch_rad);
    
    float roll_rad, yaw_rad;
    
    if (fabsf(cos_pitch) > epsilon) {
        // 正常情况：无万向锁
        roll_rad = atan2f(-R20, R22);
        yaw_rad  = atan2f(-R01, R11);
    } else {
        // 万向锁：pitch ≈ ±90°，cos(pitch)≈0，yaw 与 roll 耦合
        // 约定 roll = 0，通过矩阵残余项恢复 yaw
        roll_rad = 0.0f;
        yaw_rad  = atan2f(R10, R00);
    }
    
    // ── 第三步：转换为角度并归一化 ────────────────────────────────────
    *pitch = pitch_rad * RAD2DEG;               // [-90, 90]
    *roll  = roll_rad  * RAD2DEG;               // (-180, 180]
    *yaw   = yaw_rad   * RAD2DEG;
}


// ===============================================================================
// 其他平台无关的工具函数
// ===============================================================================

uint32_t random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random float32 in [0,1)
float random_f32(uint64_t *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}
