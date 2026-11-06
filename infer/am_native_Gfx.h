#ifndef __AM_NATIVE_GFX_H__
#define __AM_NATIVE_GFX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

#include "graphics.h"


////////////////////////////////////////////////////////////////////////////
//  Native Library : Gfx
//
//  将 Nano 图形层（Nano_GFX）封装为灵机（Animac）本地库，供 Scheme 程序
//  直接绘图。宿主须在创建解释器运行时后、加载进程前调用
//  am_native_gfx_bind() 绑定目标帧缓冲；未绑定时所有绘图函数报运行时错误。
//
//  Scheme 侧用法：源码中声明 (native Gfx) 后以 Gfx.函数名 调用。
//
//  栈纪律（上游本地库契约 N1/N2）：值型函数（width/height）压 1 个结果值；
//  语句型函数（clear/set_pixel/draw_*/fill_*/refresh）弹完参数后压 1 个 #undefined。
//  两者都恰好向操作数栈留下 1 个值，因此语句型调用也可出现在需要值的位置
//  （其值为 #undefined）；编译器在语句位置会自动将其弹出。
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Gfx_lib;

// 绑定绘图目标帧缓冲（传 NULL 解除绑定）。宿主在解释器进程运行期间
// 必须保证 gfx 指向有效的、已完成 gfx_init 的 Nano_GFX。
void am_native_gfx_bind(Nano_GFX *gfx);

// (Gfx.width) : Number —— 画布宽度（像素）
int32_t am_native_Gfx_width(am_runtime_t *rt, am_process_t *proc);

// (Gfx.height) : Number —— 画布高度（像素）
int32_t am_native_Gfx_height(am_runtime_t *rt, am_process_t *proc);

// (Gfx.clear) : #undefined —— 清屏为黑
int32_t am_native_Gfx_clear(am_runtime_t *rt, am_process_t *proc);

// (Gfx.fill_white) : #undefined —— 清屏为白
int32_t am_native_Gfx_fill_white(am_runtime_t *rt, am_process_t *proc);

// (Gfx.set_pixel x:Number y:Number r:Number g:Number b:Number) : #undefined
// —— 直写像素颜色（自动越界裁剪）
int32_t am_native_Gfx_set_pixel(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_point x:Number y:Number r:Number g:Number b:Number) : #undefined
// —— 画点（置色语义，同 mode 1）
int32_t am_native_Gfx_draw_point(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_line x1:Number y1:Number x2:Number y2:Number r:Number g:Number b:Number) : #undefined
// —— 画线段
int32_t am_native_Gfx_draw_line(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_line_aa x1:Number y1:Number x2:Number y2:Number width:Number r:Number g:Number b:Number) : #undefined
// —— 抗锯齿直线（吴小林算法，= C 版 gfx_draw_line_anti_aliasing）。端点/线宽按浮点取，
//    不做取整（抗锯齿的意义就在亚像素端点）。底层为**加法混合**（底色 + 覆盖率×颜色），
//    与上面置色语义的 draw_line 不同。
int32_t am_native_Gfx_draw_line_aa(am_runtime_t *rt, am_process_t *proc);

// (Gfx.fill_rect x:Number y:Number w:Number h:Number r:Number g:Number b:Number) : #undefined
// —— 实心矩形
int32_t am_native_Gfx_fill_rect(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_rect x:Number y:Number w:Number h:Number r:Number g:Number b:Number) : #undefined
// —— 空心矩形（1px 边框）
int32_t am_native_Gfx_draw_rect(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_circle cx:Number cy:Number r:Number rr:Number gg:Number bb:Number) : #undefined
// —— 空心圆
int32_t am_native_Gfx_draw_circle(am_runtime_t *rt, am_process_t *proc);

// (Gfx.fill_circle cx:Number cy:Number r:Number rr:Number gg:Number bb:Number) : #undefined
// —— 实心圆
int32_t am_native_Gfx_fill_circle(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_text str:String x:Number y:Number r:Number g:Number b:Number) : #undefined
// —— 绘制一行文本（12px 平滑字模；(x,y) 为行框左上角）
int32_t am_native_Gfx_draw_text(am_runtime_t *rt, am_process_t *proc);

// (Gfx.set_font id:Number) : Boolean —— 选择 Gfx.draw_text 使用的字模：
//   0 = 12px 二值点阵（GFX_FONT_BITMAP_12）、1 = 12px 抗锯齿（默认）、2 = 16px 抗锯齿。
//   字模决定**行高与字符步进**，切换后调用方应按 (Gfx.font_info id) 重新排版。
//   非法 id 返回 #f（软失败）。
int32_t am_native_Gfx_set_font(am_runtime_t *rt, am_process_t *proc);

// (Gfx.font_info id:Number) : List | #null —— 该字模的 (半角步进 全角步进 行高 基线)；
//   id 非法返回 #null。排版端用它算折行宽度与行高。
int32_t am_native_Gfx_font_info(am_runtime_t *rt, am_process_t *proc);

// (Gfx.image_size path:String) : List | #null
// —— 图片原始尺寸 (宽 高)。只读文件头（stbi_info），不解码像素，
//    供**排版阶段**算等比缩放与行高；文件缺失/格式不支持返回 #null。
int32_t am_native_Gfx_image_size(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_image path:String x:Number y:Number w:Number h:Number) : Boolean
// —— 从文件读图、缩放到 (w,h) 并绘制（宿主侧带 LRU 缓存；像素永不进解释器堆）。
//    路径规则与 (native Fs) 完全一致（相对业务资源根、拒绝 ".."）。
//    失败（文件缺失/解码失败/尺寸非法）返回 #f —— 供调用方画占位框兜底。
//    注：这条路径按 **RGB 直写**（丢弃 alpha），用于照片/插图；透明背景的图标/精灵
//    请用下面的 sprite 系列。
int32_t am_native_Gfx_draw_image(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_image_warp path:String x:Number y:Number w:Number h:Number
//                      grid_w:Number grid_h:Number disp:List neutral:Number) : Boolean
// —— **位移贴图**：图像缩放到 (w,h) 后，按 grid_w×grid_h 的位移栅格做"以矩形中心
//    为原点的径向缩放"采样（水波/水面折射、放大镜效果）。位移公式与 ui_ripple 的定点
//    实现一致：scale = disp[j*grid_w+i] / neutral（neutral 为无位移基准值，如 1024），
//    src = (dst - 矩形中心) * scale + 矩形中心，越界钳位。
//    两处平滑：位移栅格按**格中心双线性插值**放大到像素（栅格不必与目标尺寸成整数倍
//    关系，故 App 可把波场降采样到 80x60 这类低分辨率后再由宿主无块状地放大）；
//    纹理采样也用双线性过滤（最近邻会把连续位移量化为整数像素 → 折射出现马赛克）。
//    纹理经图像缓存（键 = 路径 + 目标尺寸），像素不进解释器堆；
//    失败（列表长度不足 / 参数非法 / 读文件或解码失败）返回 #f。
//    Scheme 侧示例（绘制 320x240 底图，波场 80x60、neutral=1024）：
//        (Gfx.draw_image_warp "wp.png" 0 0 320 240 80 60 disp 1024)
int32_t am_native_Gfx_draw_image_warp(am_runtime_t *rt, am_process_t *proc);

// (Gfx.sprite_info path:String) : List | #null
// —— 透明背景贴图的原始尺寸 (宽 高)。解码一次即常驻宿主侧 RGBA 缓存（预热作用），
//    失败返回 #null。
int32_t am_native_Gfx_sprite_info(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_sprite path:String cx:Number cy:Number) : Boolean
// —— 按**原始尺寸**、以 (cx,cy) 为**中心**做 alpha 混合绘制（透明背景 PNG 专用；
//    LRU 缓存按路径键，像素不进解释器堆）。失败返回 #f。
int32_t am_native_Gfx_draw_sprite(am_runtime_t *rt, am_process_t *proc);

// (Gfx.draw_sprite_rotated path:String px:Number py:Number angle:Number) : Boolean
// —— 同上，但以 (px,py) 为枢轴（贴图**上缘横向中点**）旋转 angle 弧度后绘制。
//    几何与 C 版 ui_icon_draw_rotated 一致（黄金矿工的钩爪随绳角转动）。
int32_t am_native_Gfx_draw_sprite_rotated(am_runtime_t *rt, am_process_t *proc);

// (Gfx.refresh) : #undefined —— 推帧上屏（脏区局部刷新）
int32_t am_native_Gfx_refresh(am_runtime_t *rt, am_process_t *proc);


#ifdef __cplusplus
}
#endif

#endif
