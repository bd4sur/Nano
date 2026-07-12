#include "platform.h"
#include "input_device.h"

#include <Arduino.h>
#include "M5Unified.h"

m5::touch_detail_t touchDetail;

#define X0 (0)
#define X1 (SCREEN_WIDTH / 4 * 1)
#define X2 (SCREEN_WIDTH / 4 * 2)
#define X3 (SCREEN_WIDTH / 4 * 3)
#define X4 (SCREEN_WIDTH)
#define Y0 (0)
#define Y1 (SCREEN_HEIGHT / 4 * 1)
#define Y2 (SCREEN_HEIGHT / 4 * 2)
#define Y3 (SCREEN_HEIGHT / 4 * 3)
#define Y4 (SCREEN_HEIGHT)

int32_t input_device_init() {
    return 0;
}


uint8_t input_device_read_key() {
    touchDetail = M5.Touch.getDetail();
    if (touchDetail.isPressed()) {
        int x = touchDetail.x;
        int y = touchDetail.y;
        if (y >= Y0 && y < Y1) {
            if (x >= X0 && x <  X1) return NANO_KEY_1;
            if (x >= X1 && x <  X2) return NANO_KEY_2;
            if (x >= X2 && x <  X3) return NANO_KEY_3;
            if (x >= X3 && x <= X4) return NANO_KEY_esc;
            else return NANO_KEY_IDLE;
        }
        else if (y >= Y1 && y < Y2) {
            if (x >= X0 && x <  X1) return NANO_KEY_4;
            if (x >= X1 && x <  X2) return NANO_KEY_5;
            if (x >= X2 && x <  X3) return NANO_KEY_6;
            if (x >= X3 && x <= X4) return NANO_KEY_shift;
            else return NANO_KEY_IDLE;
        }
        else if (y >= Y2 && y < Y3) {
            if (x >= X0 && x <  X1) return NANO_KEY_7;
            if (x >= X1 && x <  X2) return NANO_KEY_8;
            if (x >= X2 && x <  X3) return NANO_KEY_9;
            if (x >= X3 && x <= X4) return NANO_KEY_ctrl;
            else return NANO_KEY_IDLE;
        }
        else if (y >= Y3 && y <= Y4) {
            if (x >= X0 && x <  X1) return NANO_KEY_left;
            if (x >= X1 && x <  X2) return NANO_KEY_0;
            if (x >= X2 && x <  X3) return NANO_KEY_right;
            if (x >= X3 && x <= X4) return NANO_KEY_enter;
            else return NANO_KEY_IDLE;
        }
        else {
            return NANO_KEY_IDLE;
        }
    }
    else return NANO_KEY_IDLE;
}
