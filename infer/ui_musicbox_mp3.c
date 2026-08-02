// ============================================================================
// minimp3 实现单元 / BD4SUR 2026-08
// 单独一个编译单元承载 MINIMP3_IMPLEMENTATION（单头库惯例，同 vendor/stb_image.h），
// 避免实现代码污染其他编译单元。声明见 vendor/minimp3.h（CC0 公共领域）。
// 默认输出 int16 交错 PCM（未定义 MINIMP3_FLOAT_OUTPUT）。
//
// MINIMP3_EXTERNAL_SCRATCH（本项目对 vendor/minimp3.h 的最小 patch）：
// mp3dec_decode_frame 的 scratch 工作区约 16KB，放栈上会撑爆渲染任务 12KB 栈；
// 改由 minimp3_set_scratch() 注入外部分配缓冲（音乐盒进入时 PSRAM 分配、退出时释放）。
// ============================================================================

#define MINIMP3_IMPLEMENTATION
#define MINIMP3_ONLY_MP3            // 只保留 MP3 解码（裁掉 MP1/MP2，减小体积）
#define MINIMP3_NO_SIMD             // Xtensa 无 SSE/NEON，走通用 C 路径
#define MINIMP3_EXTERNAL_SCRATCH
#include "vendor/minimp3.h"

mp3dec_scratch_t *g_minimp3_scratch = NULL;

// 注入外部分配的 scratch 工作区（大小须 ≥ minimp3_scratch_size()）
void minimp3_set_scratch(void *scratch) {
    g_minimp3_scratch = (mp3dec_scratch_t *)scratch;
}

// scratch 工作区大小（字节），供调用方分配
uint32_t minimp3_scratch_size(void) {
    return (uint32_t)sizeof(mp3dec_scratch_t);
}
