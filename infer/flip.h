#ifndef FLIP_H
#define FLIP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "graphics.h"

/*
 * Initialize the FLIP fluid simulator with a given scene size.
 * Call this before the first render_flip(), or let render_flip()
 * auto-initialize on first call.
 *
 * Parameters:
 *   hourglass_enabled - Non-zero to set up an hourglass obstacle scene;
 *                       zero for the simple open-tank demo.
 */
void flip_init(float pool_width, float pool_height, int32_t resolution, int32_t hourglass_enabled);

/*
 * Clean up all internally allocated memory.
 */
void flip_cleanup(void);

/*
 * Simulate one step and render particles into the provided RGB frame buffer.
 *
 * Parameters:
 *   frame_buffer      - Pointer to a writable RGB buffer (size = fb_width * fb_height * 3 bytes)
 *   fb_width          - Width of the frame buffer in pixels
 *   fb_height         - Height of the frame buffer in pixels
 *   pool_width       - Width of the simulation domain in world units
 *   pool_height      - Height of the simulation domain in world units
 *   gravity_x         - Gravity acceleration along the X axis (world units/s^2)
 *   gravity_y         - Gravity acceleration along the Y axis (world units/s^2)
 *   dt                - Time step for the simulation (seconds)
 *   flip_ratio        - Blend between PIC (0.0) and FLIP (1.0)
 *   num_pressure_iters- Number of pressure solver iterations
 *   num_particle_iters- Number of particle separation iterations
 *   over_relaxation   - SOR over-relaxation factor (e.g. 1.9)
 *   compensate_drift  - Non-zero to enable drift compensation
 *   separate_particles- Non-zero to enable particle separation
 *   show_particles    - Non-zero to draw particles
 *   show_grid         - Non-zero to draw MAC grid cells
 *   funnel_neck_damping      - Velocity damping in the hourglass neck (0.0 ~ 0.6)
 *   funnel_pressure_resistance - Artificial pressure resistance in the neck (0.0 ~ 0.3)
 */
void render_flip(Nano_GFX *gfx,
                 int32_t center_x, int32_t center_y, int32_t width, int32_t height,
                 float pool_width, float pool_height, int32_t resolution,
                 float gravity_x, float gravity_y,
                 float dt, float flip_ratio,
                 int32_t num_pressure_iters, int32_t num_particle_iters,
                 float over_relaxation, int32_t compensate_drift,
                 int32_t separate_particles,
                 int32_t show_particles, int32_t show_grid,
                 float funnel_neck_damping, float funnel_pressure_resistance);

#ifdef __cplusplus
}
#endif

#endif /* FLIP_H */
