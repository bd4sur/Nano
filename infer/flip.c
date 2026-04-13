#include "flip.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils.h"
#include "graphics.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define U_FIELD 0
#define V_FIELD 1
#define FLUID_CELL 0
#define AIR_CELL   1
#define SOLID_CELL 2

typedef struct {
    float density;
    int f_num_x;
    int f_num_y;
    float h;
    float f_inv_spacing;
    int f_num_cells;

    float *u;
    float *v;
    float *du;
    float *dv;
    float *prev_u;
    float *prev_v;
    float *p;
    float *s;
    int *cell_type;
    float *cell_color;

    int max_particles;
    float *particle_pos;
    float *particle_color;
    float *particle_vel;
    float *particle_density;
    float particle_rest_density;

    float particle_radius;
    float p_inv_spacing;
    int p_num_x;
    int p_num_y;
    int p_num_cells;

    int *num_cell_particles;
    int *first_cell_particle;
    int *cell_particle_ids;
    int num_particles;
} FlipFluid;

static FlipFluid g_fluid;
static int g_initialized = 0;

static inline float clampf(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

static inline int mini(int a, int b) { return a < b ? a : b; }
static inline int maxi(int a, int b) { return a > b ? a : b; }
static inline int clampi(int x, int min, int max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

static inline int cell_type_safe(const int *cell_type, int idx, int num_cells) {
    if (idx < 0 || idx >= num_cells) return SOLID_CELL;
    return cell_type[idx];
}

static void integrate_particles(FlipFluid *f, float dt, float gravity_x, float gravity_y) {
    for (int i = 0; i < f->num_particles; i++) {
        f->particle_vel[2 * i]     += dt * gravity_x;
        f->particle_vel[2 * i + 1] += dt * gravity_y;
        f->particle_pos[2 * i]     += f->particle_vel[2 * i] * dt;
        f->particle_pos[2 * i + 1] += f->particle_vel[2 * i + 1] * dt;
    }
}

static void push_particles_apart(FlipFluid *f, int num_iters) {
    float color_diffusion_coeff = 0.001f;
    memset(f->num_cell_particles, 0, f->p_num_cells * sizeof(int));

    for (int i = 0; i < f->num_particles; i++) {
        float x = f->particle_pos[2 * i];
        float y = f->particle_pos[2 * i + 1];
        int xi = clampi((int)floorf(x * f->p_inv_spacing), 0, f->p_num_x - 1);
        int yi = clampi((int)floorf(y * f->p_inv_spacing), 0, f->p_num_y - 1);
        int cell_nr = xi * f->p_num_y + yi;
        f->num_cell_particles[cell_nr]++;
    }

    int first = 0;
    for (int i = 0; i < f->p_num_cells; i++) {
        first += f->num_cell_particles[i];
        f->first_cell_particle[i] = first;
    }
    f->first_cell_particle[f->p_num_cells] = first;

    for (int i = 0; i < f->num_particles; i++) {
        float x = f->particle_pos[2 * i];
        float y = f->particle_pos[2 * i + 1];
        int xi = clampi((int)floorf(x * f->p_inv_spacing), 0, f->p_num_x - 1);
        int yi = clampi((int)floorf(y * f->p_inv_spacing), 0, f->p_num_y - 1);
        int cell_nr = xi * f->p_num_y + yi;
        f->first_cell_particle[cell_nr]--;
        f->cell_particle_ids[f->first_cell_particle[cell_nr]] = i;
    }

    float min_dist = 2.0f * f->particle_radius;
    float min_dist2 = min_dist * min_dist;

    for (int iter = 0; iter < num_iters; iter++) {
        for (int i = 0; i < f->num_particles; i++) {
            float px = f->particle_pos[2 * i];
            float py = f->particle_pos[2 * i + 1];
            int pxi = (int)floorf(px * f->p_inv_spacing);
            int pyi = (int)floorf(py * f->p_inv_spacing);
            int x0 = maxi(pxi - 1, 0);
            int y0 = maxi(pyi - 1, 0);
            int x1 = mini(pxi + 1, f->p_num_x - 1);
            int y1 = mini(pyi + 1, f->p_num_y - 1);

            for (int xi = x0; xi <= x1; xi++) {
                for (int yi = y0; yi <= y1; yi++) {
                    int cell_nr = xi * f->p_num_y + yi;
                    int fs = f->first_cell_particle[cell_nr];
                    int ls = f->first_cell_particle[cell_nr + 1];
                    for (int j = fs; j < ls; j++) {
                        int id = f->cell_particle_ids[j];
                        if (id == i) continue;
                        float qx = f->particle_pos[2 * id];
                        float qy = f->particle_pos[2 * id + 1];
                        float dx = qx - px;
                        float dy = qy - py;
                        float d2 = dx * dx + dy * dy;
                        if (d2 > min_dist2 || d2 == 0.0f) continue;
                        float d = sqrtf(d2);
                        float s = 0.5f * (min_dist - d) / d;
                        dx *= s; dy *= s;
                        f->particle_pos[2 * i]     -= dx;
                        f->particle_pos[2 * i + 1] -= dy;
                        f->particle_pos[2 * id]     += dx;
                        f->particle_pos[2 * id + 1] += dy;

                        for (int k = 0; k < 3; k++) {
                            float color0 = f->particle_color[3 * i + k];
                            float color1 = f->particle_color[3 * id + k];
                            float color = (color0 + color1) * 0.5f;
                            f->particle_color[3 * i + k] = color0 + (color - color0) * color_diffusion_coeff;
                            f->particle_color[3 * id + k] = color1 + (color - color1) * color_diffusion_coeff;
                        }
                    }
                }
            }
        }
    }
}

static void handle_particle_collisions(FlipFluid *f) {
    float h = 1.0f / f->f_inv_spacing;
    float r = f->particle_radius;
    float min_x = h + r, max_x = (f->f_num_x - 1) * h - r;
    float min_y = h + r, max_y = (f->f_num_y - 1) * h - r;

    for (int i = 0; i < f->num_particles; i++) {
        float x = f->particle_pos[2 * i];
        float y = f->particle_pos[2 * i + 1];
        if (x < min_x) { x = min_x; f->particle_vel[2 * i] = 0.0f; }
        if (x > max_x) { x = max_x; f->particle_vel[2 * i] = 0.0f; }
        if (y < min_y) { y = min_y; f->particle_vel[2 * i + 1] = 0.0f; }
        if (y > max_y) { y = max_y; f->particle_vel[2 * i + 1] = 0.0f; }
        f->particle_pos[2 * i] = x;
        f->particle_pos[2 * i + 1] = y;
    }
}

static void update_particle_density(FlipFluid *f) {
    int n = f->f_num_y;
    float h = f->h, h1 = f->f_inv_spacing, h2 = 0.5f * h;
    float *d = f->particle_density;
    memset(d, 0, f->f_num_cells * sizeof(float));

    for (int i = 0; i < f->num_particles; i++) {
        float x = f->particle_pos[2 * i];
        float y = f->particle_pos[2 * i + 1];
        x = clampf(x, h, (f->f_num_x - 1) * h);
        y = clampf(y, h, (f->f_num_y - 1) * h);

        int x0 = (int)floorf((x - h2) * h1);
        float tx = ((x - h2) - x0 * h) * h1;
        int x1 = mini(x0 + 1, f->f_num_x - 2);
        int y0 = (int)floorf((y - h2) * h1);
        float ty = ((y - h2) - y0 * h) * h1;
        int y1 = mini(y0 + 1, f->f_num_y - 2);
        float sx = 1.0f - tx, sy = 1.0f - ty;

        if (x0 < f->f_num_x && y0 < f->f_num_y) d[x0 * n + y0] += sx * sy;
        if (x1 < f->f_num_x && y0 < f->f_num_y) d[x1 * n + y0] += tx * sy;
        if (x1 < f->f_num_x && y1 < f->f_num_y) d[x1 * n + y1] += tx * ty;
        if (x0 < f->f_num_x && y1 < f->f_num_y) d[x0 * n + y1] += sx * ty;
    }

    if (f->particle_rest_density == 0.0f) {
        float sum = 0.0f;
        int num_fluid_cells = 0;
        for (int i = 0; i < f->f_num_cells; i++) {
            if (f->cell_type[i] == FLUID_CELL) {
                sum += d[i];
                num_fluid_cells++;
            }
        }
        if (num_fluid_cells > 0) f->particle_rest_density = sum / num_fluid_cells;
    }
}

static void transfer_velocities(FlipFluid *f, int to_grid, float flip_ratio) {
    int n = f->f_num_y;
    float h = f->h, h1 = f->f_inv_spacing, h2 = 0.5f * h;

    if (to_grid) {
        memcpy(f->prev_u, f->u, f->f_num_cells * sizeof(float));
        memcpy(f->prev_v, f->v, f->f_num_cells * sizeof(float));
        memset(f->du, 0, f->f_num_cells * sizeof(float));
        memset(f->dv, 0, f->f_num_cells * sizeof(float));
        memset(f->u, 0, f->f_num_cells * sizeof(float));
        memset(f->v, 0, f->f_num_cells * sizeof(float));
        for (int i = 0; i < f->f_num_cells; i++)
            f->cell_type[i] = (f->s[i] == 0.0f) ? SOLID_CELL : AIR_CELL;
        for (int i = 0; i < f->num_particles; i++) {
            float x = f->particle_pos[2 * i];
            float y = f->particle_pos[2 * i + 1];
            int xi = clampi((int)floorf(x * h1), 0, f->f_num_x - 1);
            int yi = clampi((int)floorf(y * h1), 0, f->f_num_y - 1);
            int cell_nr = xi * n + yi;
            if (f->cell_type[cell_nr] == AIR_CELL) f->cell_type[cell_nr] = FLUID_CELL;
        }
    }

    for (int component = 0; component < 2; component++) {
        float dx = (component == 0) ? 0.0f : h2;
        float dy = (component == 0) ? h2 : 0.0f;
        float *F = (component == 0) ? f->u : f->v;
        float *prevF = (component == 0) ? f->prev_u : f->prev_v;
        float *d_arr = (component == 0) ? f->du : f->dv;

        for (int i = 0; i < f->num_particles; i++) {
            float x = f->particle_pos[2 * i];
            float y = f->particle_pos[2 * i + 1];
            x = clampf(x, h, (f->f_num_x - 1) * h);
            y = clampf(y, h, (f->f_num_y - 1) * h);

            int x0 = mini((int)floorf((x - dx) * h1), f->f_num_x - 2);
            float tx = ((x - dx) - x0 * h) * h1;
            int x1 = mini(x0 + 1, f->f_num_x - 2);
            int y0 = mini((int)floorf((y - dy) * h1), f->f_num_y - 2);
            float ty = ((y - dy) - y0 * h) * h1;
            int y1 = mini(y0 + 1, f->f_num_y - 2);
            float sx = 1.0f - tx, sy = 1.0f - ty;
            float d0 = sx * sy, d1 = tx * sy, d2 = tx * ty, d3 = sx * ty;
            int nr0 = x0 * n + y0, nr1 = x1 * n + y0, nr2 = x1 * n + y1, nr3 = x0 * n + y1;

            if (to_grid) {
                float pv = f->particle_vel[2 * i + component];
                F[nr0] += pv * d0; d_arr[nr0] += d0;
                F[nr1] += pv * d1; d_arr[nr1] += d1;
                F[nr2] += pv * d2; d_arr[nr2] += d2;
                F[nr3] += pv * d3; d_arr[nr3] += d3;
            } else {
                int offset = (component == 0) ? n : 1;
                int ncells = f->f_num_cells;
                float valid0 = (f->cell_type[nr0] != AIR_CELL || cell_type_safe(f->cell_type, nr0 - offset, ncells) != AIR_CELL) ? 1.0f : 0.0f;
                float valid1 = (f->cell_type[nr1] != AIR_CELL || cell_type_safe(f->cell_type, nr1 - offset, ncells) != AIR_CELL) ? 1.0f : 0.0f;
                float valid2 = (f->cell_type[nr2] != AIR_CELL || cell_type_safe(f->cell_type, nr2 - offset, ncells) != AIR_CELL) ? 1.0f : 0.0f;
                float valid3 = (f->cell_type[nr3] != AIR_CELL || cell_type_safe(f->cell_type, nr3 - offset, ncells) != AIR_CELL) ? 1.0f : 0.0f;
                float d = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;

                if (d > 0.0f) {
                    float pic_v = (valid0 * d0 * F[nr0] + valid1 * d1 * F[nr1] + valid2 * d2 * F[nr2] + valid3 * d3 * F[nr3]) / d;
                    float corr = (valid0 * d0 * (F[nr0] - prevF[nr0]) + valid1 * d1 * (F[nr1] - prevF[nr1])
                                + valid2 * d2 * (F[nr2] - prevF[nr2]) + valid3 * d3 * (F[nr3] - prevF[nr3])) / d;
                    float flip_v = f->particle_vel[2 * i + component] + corr;
                    f->particle_vel[2 * i + component] = (1.0f - flip_ratio) * pic_v + flip_ratio * flip_v;
                }
            }
        }

        if (to_grid) {
            for (int i = 0; i < f->f_num_cells; i++) if (d_arr[i] > 0.0f) F[i] /= d_arr[i];
            for (int i = 0; i < f->f_num_x; i++) {
                for (int j = 0; j < f->f_num_y; j++) {
                    int solid = (f->cell_type[i * n + j] == SOLID_CELL);
                    if (solid || (i > 0 && f->cell_type[(i - 1) * n + j] == SOLID_CELL))
                        f->u[i * n + j] = f->prev_u[i * n + j];
                    if (solid || (j > 0 && f->cell_type[i * n + j - 1] == SOLID_CELL))
                        f->v[i * n + j] = f->prev_v[i * n + j];
                }
            }
        }
    }
}

static void solve_incompressibility(FlipFluid *f, int num_iters, float dt, float over_relaxation, int compensate_drift) {
    memset(f->p, 0, f->f_num_cells * sizeof(float));
    memcpy(f->prev_u, f->u, f->f_num_cells * sizeof(float));
    memcpy(f->prev_v, f->v, f->f_num_cells * sizeof(float));
    int n = f->f_num_y;
    float cp = f->density * f->h / dt;

    for (int iter = 0; iter < num_iters; iter++) {
        for (int i = 1; i < f->f_num_x - 1; i++) {
            for (int j = 1; j < f->f_num_y - 1; j++) {
                int center = i * n + j;
                if (f->cell_type[center] != FLUID_CELL) continue;
                int left   = (i - 1) * n + j;
                int right  = (i + 1) * n + j;
                int bottom = i * n + j - 1;
                int top    = i * n + j + 1;
                float sx0 = f->s[left], sx1 = f->s[right];
                float sy0 = f->s[bottom], sy1 = f->s[top];
                float s = sx0 + sx1 + sy0 + sy1;
                if (s == 0.0f) continue;
                float div = f->u[right] - f->u[center] + f->v[top] - f->v[center];
                if (f->particle_rest_density > 0.0f && compensate_drift) {
                    float compression = f->particle_density[center] - f->particle_rest_density;
                    if (compression > 0.0f) div -= 1.0f * compression;
                }
                float p_val = -div / s * over_relaxation;
                f->p[center] += cp * p_val;
                f->u[center] -= sx0 * p_val; f->u[right] += sx1 * p_val;
                f->v[center] -= sy0 * p_val; f->v[top]    += sy1 * p_val;
            }
        }
    }
}

static void update_particle_colors(FlipFluid *f) {
    float h1 = f->f_inv_spacing;
    for (int i = 0; i < f->num_particles; i++) {
        float s = 0.01f;
        f->particle_color[3 * i]     = clampf(f->particle_color[3 * i]     - s, 0.0f, 1.0f);
        f->particle_color[3 * i + 1] = clampf(f->particle_color[3 * i + 1] - s, 0.0f, 1.0f);
        f->particle_color[3 * i + 2] = clampf(f->particle_color[3 * i + 2] + s, 0.0f, 1.0f);
        float x = f->particle_pos[2 * i];
        float y = f->particle_pos[2 * i + 1];
        int xi = clampi((int)floorf(x * h1), 1, f->f_num_x - 1);
        int yi = clampi((int)floorf(y * h1), 1, f->f_num_y - 1);
        int cell_nr = xi * f->f_num_y + yi;
        if (f->particle_rest_density > 0.0f) {
            float rel_density = f->particle_density[cell_nr] / f->particle_rest_density;
            if (rel_density < 0.7f) {
                f->particle_color[3 * i]     = 0.8f;
                f->particle_color[3 * i + 1] = 0.8f;
                f->particle_color[3 * i + 2] = 1.0f;
            }
        }
    }
}

static void set_sci_color(FlipFluid *f, int cell_nr, float val, float min_val, float max_val) {
    val = clampf(val, min_val, max_val - 0.0001f);
    float d = max_val - min_val;
    val = (d == 0.0f) ? 0.5f : (val - min_val) / d;
    float m = 0.25f;
    int num = (int)floorf(val / m);
    float s = (val - num * m) / m;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    switch (num) {
        case 0: r = 0.0f; g = s; b = 1.0f; break;
        case 1: r = 0.0f; g = 1.0f; b = 1.0f - s; break;
        case 2: r = s; g = 1.0f; b = 0.0f; break;
        case 3: r = 1.0f; g = 1.0f - s; b = 0.0f; break;
    }
    f->cell_color[3 * cell_nr]     = r;
    f->cell_color[3 * cell_nr + 1] = g;
    f->cell_color[3 * cell_nr + 2] = b;
}

static void update_cell_colors(FlipFluid *f) {
    memset(f->cell_color, 0, 3 * f->f_num_cells * sizeof(float));
    for (int i = 0; i < f->f_num_cells; i++) {
        if (f->cell_type[i] == SOLID_CELL) {
            f->cell_color[3 * i]     = 0.5f;
            f->cell_color[3 * i + 1] = 0.5f;
            f->cell_color[3 * i + 2] = 0.5f;
        } else if (f->cell_type[i] == FLUID_CELL) {
            float d = f->particle_density[i];
            if (f->particle_rest_density > 0.0f) d /= f->particle_rest_density;
            set_sci_color(f, i, d, 0.0f, 2.0f);
        }
    }
}

static void simulate_step(FlipFluid *f, float dt, float gravity_x, float gravity_y,
                          float flip_ratio, int num_pressure_iters, int num_particle_iters,
                          float over_relaxation, int compensate_drift, int separate_particles) {
    int num_sub_steps = 1;
    float sdt = dt / num_sub_steps;
    for (int step = 0; step < num_sub_steps; step++) {
        integrate_particles(f, sdt, gravity_x, gravity_y);
        if (separate_particles) push_particles_apart(f, num_particle_iters);
        handle_particle_collisions(f);
        transfer_velocities(f, 1, 0.0f);
        update_particle_density(f);
        solve_incompressibility(f, num_pressure_iters, sdt, over_relaxation, compensate_drift);
        transfer_velocities(f, 0, flip_ratio);
    }
    update_particle_colors(f);
    update_cell_colors(f);
}

static void draw_particle_to_framebuffer(Nano_GFX *gfx,
    int32_t center_x, int32_t center_y, int width, int height,
    int cx, int cy, int radius,
    uint8_t r, uint8_t g, uint8_t b
) {
    int x0 = maxi(cx - radius, 0);
    int y0 = maxi(cy - radius, 0);
    int x1 = mini(cx + radius, width - 1);
    int y1 = mini(cy + radius, height - 1);
    int r2 = radius * radius;
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            int dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy <= r2) {
                gfx_draw_point(gfx, x+center_x, y+center_y, r, g, b, 1);
            }
        }
    }
}

static void render_to_framebuffer(
    Nano_GFX *gfx, int32_t center_x, int32_t center_y, int32_t width, int32_t height,
    FlipFluid *f,
    float scene_w, float scene_h,
    int show_particles, int show_grid
) {

    int32_t fb_w = width;
    int32_t fb_h = height;

    float c_scale = fb_h / scene_h;

    if (show_grid) {
        float point_size = 0.9f * f->h / scene_w * fb_w;
        int radius = (int)(point_size * 0.5f);
        if (radius < 1) radius = 1;
        int n = f->f_num_y;
        for (int i = 0; i < f->f_num_cells; i++) {
            int xi = i / n;
            int yi = i % n;
            float px = (xi + 0.5f) * f->h * c_scale;
            float py = fb_h - (yi + 0.5f) * f->h * c_scale;
            int cx = (int)px;
            int cy = (int)py;
            uint8_t r = (uint8_t)clampf(f->cell_color[3 * i]     * 255.0f, 0.0f, 255.0f);
            uint8_t g = (uint8_t)clampf(f->cell_color[3 * i + 1] * 255.0f, 0.0f, 255.0f);
            uint8_t b = (uint8_t)clampf(f->cell_color[3 * i + 2] * 255.0f, 0.0f, 255.0f);
            draw_particle_to_framebuffer(gfx, center_x, center_y, width, height, cx, cy, radius, r, g, b);
        }
    }

    if (show_particles) {
        float point_size = 2.2f * f->particle_radius / scene_w * fb_w;
        int radius = (int)(point_size * 0.5f);
        if (radius < 1) radius = 1;

        for (int i = 0; i < f->num_particles; i++) {
            float px = f->particle_pos[2 * i] * c_scale;
            float py = fb_h - f->particle_pos[2 * i + 1] * c_scale;
            int cx = (int)px;
            int cy = (int)py;
            uint8_t r = (uint8_t)clampf(f->particle_color[3 * i]     * 255.0f, 0.0f, 255.0f);
            uint8_t g = (uint8_t)clampf(f->particle_color[3 * i + 1] * 255.0f, 0.0f, 255.0f);
            uint8_t b = (uint8_t)clampf(f->particle_color[3 * i + 2] * 255.0f, 0.0f, 255.0f);
            draw_particle_to_framebuffer(gfx, center_x, center_y, width, height, cx, cy, radius, r, g, b);
        }
    }
}

/* -------------------------------------------------------------------- */
/* Public API                                                           */
/* -------------------------------------------------------------------- */

void flip_init(float pool_width, float pool_height) {
    if (g_initialized) flip_cleanup();

    int res = 32;
    float tank_height = pool_height;
    float tank_width = pool_width;
    float h = tank_height / res;
    float density = 1000.0f;
    float rel_water_height = 0.8f, rel_water_width = 0.6f;
    float r = 0.3f * h;
    float dx = 2.0f * r;
    float dy = sqrtf(3.0f) / 2.0f * dx;
    int num_x = (int)floorf((rel_water_width * tank_width - 2.0f * h - 2.0f * r) / dx);
    int num_y = (int)floorf((rel_water_height * tank_height - 2.0f * h - 2.0f * r) / dy);
    int max_particles = num_x * num_y;

    FlipFluid *f = &g_fluid;
    f->density = density;
    f->f_num_x = (int)floorf(tank_width / h) + 1;
    f->f_num_y = (int)floorf(tank_height / h) + 1;
    f->h = (tank_width / f->f_num_x > tank_height / f->f_num_y) ? (tank_width / f->f_num_x) : (tank_height / f->f_num_y);
    f->f_inv_spacing = 1.0f / f->h;
    f->f_num_cells = f->f_num_x * f->f_num_y;

    f->u = (float *)calloc(f->f_num_cells, sizeof(float));
    f->v = (float *)calloc(f->f_num_cells, sizeof(float));
    f->du = (float *)calloc(f->f_num_cells, sizeof(float));
    f->dv = (float *)calloc(f->f_num_cells, sizeof(float));
    f->prev_u = (float *)calloc(f->f_num_cells, sizeof(float));
    f->prev_v = (float *)calloc(f->f_num_cells, sizeof(float));
    f->p = (float *)calloc(f->f_num_cells, sizeof(float));
    f->s = (float *)calloc(f->f_num_cells, sizeof(float));
    f->cell_type = (int *)calloc(f->f_num_cells, sizeof(int));
    f->cell_color = (float *)calloc(3 * f->f_num_cells, sizeof(float));
    f->particle_density = (float *)calloc(f->f_num_cells, sizeof(float));

    f->max_particles = max_particles;
    f->particle_pos = (float *)calloc(2 * max_particles, sizeof(float));
    f->particle_color = (float *)calloc(3 * max_particles, sizeof(float));
    for (int i = 0; i < max_particles; i++) f->particle_color[3 * i + 2] = 1.0f;
    f->particle_vel = (float *)calloc(2 * max_particles, sizeof(float));
    f->particle_rest_density = 0.0f;

    f->particle_radius = r;
    f->p_inv_spacing = 1.0f / (2.2f * r);
    f->p_num_x = (int)floorf(tank_width * f->p_inv_spacing) + 1;
    f->p_num_y = (int)floorf(tank_height * f->p_inv_spacing) + 1;
    f->p_num_cells = f->p_num_x * f->p_num_y;

    f->num_cell_particles = (int *)calloc(f->p_num_cells, sizeof(int));
    f->first_cell_particle = (int *)calloc(f->p_num_cells + 1, sizeof(int));
    f->cell_particle_ids = (int *)calloc(max_particles, sizeof(int));
    f->num_particles = num_x * num_y;

    int p = 0;
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < num_y; j++) {
            f->particle_pos[p++] = h + r + dx * i + (j % 2 == 0 ? 0.0f : r);
            f->particle_pos[p++] = h + r + dy * j;
        }
    }

    int n = f->f_num_y;
    for (int i = 0; i < f->f_num_x; i++) {
        for (int j = 0; j < f->f_num_y; j++) {
            float s = 1.0f;
            if (i == 0 || i == f->f_num_x - 1 || j == 0) s = 0.0f;
            f->s[i * n + j] = s;
        }
    }

    g_initialized = 1;
}

void flip_cleanup(void) {
    if (!g_initialized) return;
    FlipFluid *f = &g_fluid;
    free(f->u); free(f->v); free(f->du); free(f->dv);
    free(f->prev_u); free(f->prev_v); free(f->p); free(f->s);
    free(f->cell_type); free(f->cell_color); free(f->particle_density);
    free(f->particle_pos); free(f->particle_color); free(f->particle_vel);
    free(f->num_cell_particles); free(f->first_cell_particle); free(f->cell_particle_ids);
    memset(f, 0, sizeof(FlipFluid));
    g_initialized = 0;
}

void render_flip(Nano_GFX *gfx,
                 int32_t center_x, int32_t center_y, int32_t width, int32_t height,
                 int32_t pool_width, int32_t pool_height,
                 float gravity_x, float gravity_y,
                 float dt, float flip_ratio,
                 int32_t num_pressure_iters, int32_t num_particle_iters,
                 float over_relaxation, int32_t compensate_drift,
                 int32_t separate_particles,
                 int32_t show_particles, int32_t show_grid) {
    if (!g_initialized) {
        flip_init((float)pool_width, (float)pool_height);
    }

    FlipFluid *f = &g_fluid;
    simulate_step(f, dt, gravity_x, gravity_y, flip_ratio,
                  num_pressure_iters, num_particle_iters,
                  over_relaxation, compensate_drift, separate_particles);

    render_to_framebuffer(gfx, center_x, center_y, width, height, f,
                          (float)pool_width, (float)pool_height,
                          show_particles, show_grid);
}
