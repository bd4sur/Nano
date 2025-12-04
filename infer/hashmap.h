#ifndef __NANO_HASHMAP_H__
#define __NANO_HASHMAP_H__

#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>

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


#endif