#include "platform.h"
#include "audio_out.h"

int32_t audio_out_init(uint32_t sample_rate, uint8_t volume) {
    return 0;
}

int32_t audio_out_queue_free(void) {
    return 0;
}

int32_t audio_out_enqueue(const int16_t *pcm, uint32_t samples) {
    return 0;
}

void audio_out_stop(void) {
    return;
}

void audio_out_set_volume(uint8_t volume) {
    return;
}

void audio_out_close(void) {
    return;
}
