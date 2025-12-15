#ifndef __NANO_PROMPT_H__
#define __NANO_PROMPT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

void set_random_prompt(wchar_t *dest, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif
