#ifndef __NANO_ASR_H__
#define __NANO_ASR_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

int32_t set_ptt_status(uint8_t status);
int32_t close_ptt_fifo();

int32_t open_asr_fifo();
int32_t close_asr_fifo();
int32_t read_asr_fifo(wchar_t *asr_text);
int32_t check_asr_server_status();

#ifdef __cplusplus
}
#endif

#endif
