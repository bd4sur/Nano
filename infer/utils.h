#ifndef __NANO_UTILITIES_H__
#define __NANO_UTILITIES_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <ctype.h>
#include <wchar.h>
#include <locale.h>
#include <math.h>


// #if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32P4
//     #define uint64_t unsigned long long
//     #define int64_t signed long long
//     #define uint32_t unsigned int
//     #define int32_t signed int
//     #define uint16_t unsigned short
//     #define int16_t signed short
//     #define uint8_t unsigned char
//     #define int8_t signed char
// #else
//     #include <stdint.h>
// #endif

#include <stdint.h>

#ifndef M_PI
    #define M_PI (3.14159265358979323846)
#endif

#ifndef M_PI_2
    # define M_PI_2 (1.57079632679489661923)
#endif

#ifndef MIN
    #define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif
#ifndef MAX
    #define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif

// ===============================================================================
// HashMap
// ===============================================================================

struct Entry {
    uint32_t key;
    uint32_t value;
    struct Entry *next;
};

struct Map {
    uint32_t bucket_num;
    struct Entry **buckets;
};

struct Map *new_map(uint32_t bucket_num);
void free_map(struct Map *pm);
struct Entry *new_entry(uint32_t key, uint32_t value);
uint32_t map_hash(uint32_t key, uint32_t bucket_num);
uint32_t map_set(struct Map *m, uint32_t key, uint32_t value);
uint32_t map_get(struct Map *m, uint32_t key);


// ===============================================================================
// Radix Tree（基数树 / 压缩前缀树）
// ===============================================================================

#define VOCAB_SIZE        (16384) // 树字符数上限，用于输入合法性校验。

// Radix Tree 节点结构
struct RadixNode {
    uint32_t token_id;         // 保存词对应的ID
    uint8_t  is_end_of_token;  // 标记是否为词的结尾
    uint32_t prefix_len;       // 从父节点到本节点的路径标签长度
    uint32_t *prefix;          // 从父节点到本节点的路径标签（uint32_t 数组）
    struct RadixNode **children; // 子节点指针数组
    uint32_t num_children;     // 子节点数量
    uint32_t child_capacity;   // 子节点数组容量
};

// Trie 树结构（对外接口名称保持兼容，内部已改为 Radix Tree）
struct Trie {
    struct RadixNode *root;    // 根节点
};

// 初始化一个 Trie 树（内部使用 Radix Tree 实现）
//   注：参数 vocab_size 和 is_end_of_token 仅为兼容性而保留。
struct Trie *new_trie(uint32_t vocab_size, uint8_t is_end_of_token);

// 释放 Trie 树
void free_trie(struct Trie *trie);

// 向 Trie 树中增加一个词
int add_token(struct Trie *trie, uint32_t *token, uint32_t token_len, uint32_t token_id);

// 在 Trie 树中匹配一个词
int match_token(struct Trie *trie, uint32_t *pattern, uint32_t pattern_len, uint32_t *token_id);

uint32_t tokenize(struct Trie *vocab_trie, uint32_t *output_token_ids, const uint32_t *input_char_ids, uint32_t input_length, uint32_t max_token_length);


// ===============================================================================
// 对有序数组的二分查找
// ===============================================================================

int32_t binary_search(const uint32_t *lut_sorted, const uint32_t *lut_indexs, uint32_t n, uint32_t target);

// ===============================================================================
// AVL树
// ===============================================================================

struct AVLNode {
    uint32_t key;
    uint32_t index;
    struct AVLNode *left;
    struct AVLNode *right;
    int height;
};

typedef struct AVLNode AVLNode;

AVLNode* buildAVLTree(uint32_t arr[], uint32_t n);
uint32_t findIndex(AVLNode* root, uint32_t key);
void freeTree(AVLNode* root);


// ===============================================================================
// 数字格式字符串
// ===============================================================================

uint8_t get_digit(int32_t num, int32_t n);
uint8_t get_timezone_digit(float tz, int32_t digit);
uint8_t get_lon_lat_digit(float decimal, int32_t digit);

// 地方时+时区 转换为 UTC
void local_time_to_utc(
    int32_t year, int32_t month, int32_t day,
    int32_t hour, int32_t minute, int32_t second,
    float timezone,
    int32_t *utc_year, int32_t *utc_month, int32_t *utc_day,
    int32_t *utc_hour, int32_t *utc_minute, int32_t *utc_second
);


// ===============================================================================
// 字符编码相关
// ===============================================================================

uint32_t _wcstombs(char *dest, const wchar_t *src, uint32_t length);
uint32_t _mbstowcs(wchar_t *dest, const char *src, uint32_t length);


// ===============================================================================
// 预置提示词
// ===============================================================================

void set_random_prompt(wchar_t *dest, uint64_t seed);


// ===============================================================================
// IMU坐标变换
// ===============================================================================

void transform_euler_angles(float pitch_in, float roll_in, float yaw_in, float *pitch_out, float *roll_out, float *yaw_out);
void quaternion_to_euler(float q0, float q1, float q2, float q3, float *pitch, float *roll, float *yaw);


// ===============================================================================
// 其他平台无关的工具函数
// ===============================================================================

uint32_t random_u32(uint64_t *state);
float random_f32(uint64_t *state);


#ifdef __cplusplus
}
#endif

#endif
