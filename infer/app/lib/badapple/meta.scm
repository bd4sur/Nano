;; 由 infer/tools/gen_badapple_data.py 从 infer/badapple.h 自动生成，请勿手工编辑。
;; 帧数据：app/lib/badapple/frames.txt（定长记录，偏移 = 帧号 × rec_bytes）。
;; 记录布局常量一并导出，App 侧不手抄数字（改格式只改本脚本）。
(define n_frames 2193)
(define rec_bytes 656)
(define frame_w 64)
(define frame_h 64)
(define fps 10)
(define slot_ms 100)
;; 记录字符 c → 数值：(- c base_char)
(define base_char 33)
;; 段数头：c0*base_radix + c1（两个字符）
(define base_radix 94)
(define head_chars 2)
;; 每段 3 字符：(行号, 段起始列, 段长-1)
(define chars_per_run 3)
