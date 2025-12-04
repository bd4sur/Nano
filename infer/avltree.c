#include "avltree.h"

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
