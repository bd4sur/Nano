#ifndef __NANO_AVL_TREE__
#define __NANO_AVL_TREE__

#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>

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

#endif