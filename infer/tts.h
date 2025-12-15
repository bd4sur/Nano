#ifndef __NANO_TTS_H__
#define __NANO_TTS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

int32_t send_tts_request(wchar_t *text, int32_t is_finished);
int32_t stop_tts();
void reset_tts_split_status();

#ifdef __cplusplus
}
#endif

#endif
