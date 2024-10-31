#include <stdio.h>
#include <stdlib.h>

#define VOCAB_SIZE (16384)

struct trie {
    struct trie *children[VOCAB_SIZE];
    unsigned char is_end_of_token;
};

struct trie *make_node(unsigned char is_end_of_token) {
    struct trie *pnode = calloc(1, sizeof(struct trie));
    if(NULL == pnode) {
        return NULL;
    }
    else {
        pnode->is_end_of_token = is_end_of_token;
        return pnode;
    }
}

int add_token(struct trie *root, unsigned int *token, unsigned int token_len) {
    struct trie *current_node = root;
    for(unsigned int i = 0; i < token_len; i++) {
        unsigned int cid = token[i];
        unsigned char is_eot = (i == token_len - 1) ? 1 : 0;
        struct trie *next_node = current_node->children[cid];
        if(NULL == next_node) {
            next_node = make_node(is_eot);
            if(NULL == next_node) {
                return -1;
            }
            else {
                current_node->children[cid] = next_node;
                current_node = next_node;
            }
        }
        else {
            current_node = next_node;
        }
    }
    current_node->is_end_of_token = 1;
    return 0;
}

int match_token(struct trie *root, unsigned int *pattern, unsigned int pattern_len) {
    struct trie *current_node = root;
    for(unsigned int i = 0; i < pattern_len; i++) {
        unsigned int cid = pattern[i];
        struct trie *next_node = current_node->children[cid];
        if(NULL == next_node) {
            return -1;
        }
        current_node = next_node;
        if(i == pattern_len - 1) {
            if(current_node->is_end_of_token == 1) {
                return 0;
            }
            else {
                return -1;
            }
        }
    }
}

int main(int argc, char **argv) {
    struct trie *root = make_node(0);

    unsigned int token0[3] = {1, 3, 5};
    unsigned int token1[3] = {1, 3, 6};
    unsigned int token2[2] = {1, 3};
    unsigned int token3[4] = {2, 4, 6, 8};
    add_token(root, token0, 3);
    add_token(root, token1, 3);
    add_token(root, token2, 2);
    add_token(root, token3, 4);

    unsigned int pattern[5] = {2, 4, 6, 8, 10};
    int is_match = match_token(root, pattern, 5);
    printf("Match? = %d\n", is_match);

    return 0;
}
