#!/bin/bash
# build_wasm_gui.sh —— Nano-WASM-GUI 构建脚本（Emscripten）
#
# 把 infer/ 的完整 GUI 业务代码编译为浏览器内运行的 WebAssembly：
#   产物：infer/webgui/{nano_gui.js, nano_gui.wasm, index.html, loader.js, manifest.json}
#   运行：cd infer/webgui && python3 ../tools/https_static_server.py 9002 .（或任意静态 HTTP 服务）
#
# 与 build_wasm.sh（LLM-only 的 main_wasm.c/wasi-sdk 路线，产物在 infer/web/）互不相关。
#
# 业务资源【不打进产物】（隐私要求）：不使用 --preload-file，改为运行时从外部 HTTP
# 「挂载」——本脚本把 linglong 资源软链接为 webgui/files 并生成 manifest.json 清单；
# 页面启动时经 fetch 拉取写入 MEMFS（见 webgui/loader.js 与下方第 41 行起的说明）。
# 故 nano_gui.wasm / nano_gui.js 中不含任何业务文件内容与用户文件名
# （审计记录见 doc/AGENTS.md 2026-10-13 条目）。
#
# 工具链：Ubuntu noble deb 解包版 emscripten 3.1.6（安装于 /home/bd4sur/app/emscripten-deb，
# 含预编译 sysroot 缓存；emsdk 在线下载过慢时的替代，经清华镜像 apt 源下载解包组装）。
# 也可用标准 emsdk：EMSDK_PATH=/path/to/emsdk 且存在 upstream/emscripten/emcc 时优先使用之。

set -e

EMSDK_DEB=${EMSDK_DEB:-/home/bd4sur/app/emscripten-deb}
if [ -n "$EMSDK_PATH" ] && [ -x "$EMSDK_PATH/upstream/emscripten/emcc" ]; then
    source "$EMSDK_PATH/emsdk_env.sh" >/dev/null
    EMCC="$EMSDK_PATH/upstream/emscripten/emcc"
else
    export EM_CONFIG="$EMSDK_DEB/emconfig.py"
    export NODE_PATH="$EMSDK_DEB/usr/share/nodejs"
    export LD_LIBRARY_PATH="$EMSDK_DEB/usr/lib/llvm-15/lib:$EMSDK_DEB/usr/lib/x86_64-linux-gnu"
    export PATH="$EMSDK_DEB/bin-wrappers:$EMSDK_DEB/usr/bin:$EMSDK_DEB/usr/lib/llvm-15/bin:$PATH"
    EMCC="$EMSDK_DEB/bin-wrappers/emcc"
fi
if [ ! -x "$EMCC" ]; then
    echo "emcc not found (EMSDK_PATH=$EMSDK_PATH EMSDK_DEB=$EMSDK_DEB)" >&2
    exit 1
fi

cd "$(dirname "$0")"
LINGLONG=/home/bd4sur/linglong
OUT_DIR=webgui

# 业务资源：不打进产物（不用 --preload-file），改为运行时从外部 HTTP「挂载」——
#   本脚本把 linglong 资源软链接进 webgui/files/ 并生成 manifest.json（文件清单+大小）；
#   页面 JS 启动时经 createPreloadedFile 逐文件拉入 MEMFS（路径与 PLATFORM_ROOT_DIR 一致）。
#   分级加载：priority 1（图标/壁纸/mob.scm 等启动必需）先拉，立即可进系统；
#             priority 2（ebook/dict/image 等）后台拉，状态栏显示进度；
#             priority 3（LLM 模型）页面「预载」按钮按需拉；
#             priority 4（music 音频）进入音乐盒时按需拉（见 webgui/index.html 的 nanoEnsureMusic）。
#   浏览器平台约束：主线程同步 XHR 已被禁用，「打开文件时才下载」的纯按需挂载做不到
#   （createLazyFile 依赖 sync XHR），清单式挂载+分级加载是务实等价物。
#   不打包 llm 中的 qwen3-0b6（599MB，单线程 WASM 不可用）；music 走 p4 按需层（93MB）。
LINGLONG=/home/bd4sur/linglong
mkdir -p "$OUT_DIR"
ln -sfn "$LINGLONG" "$OUT_DIR/files"

# 同步灵机 App 源码（infer/app/*.scm）到业务资源目录
mkdir -p "$LINGLONG/app"
mkdir -p "$LINGLONG/app/lib"
cp -f app/*.scm "$LINGLONG/app/" 2>/dev/null || true
cp -f app/lib/*.scm "$LINGLONG/app/lib/" 2>/dev/null || true
# 大块数据（不入 .scm 源码：灵机按源码字符数预分配 token 数组，见 app/hwr.scm 注释）
mkdir -p "$LINGLONG/app/lib/hwr"
cp -f app/lib/hwr/* "$LINGLONG/app/lib/hwr/" 2>/dev/null || true
# LLM 模型数据（app/lib/llm/{sort6,psycho}/mNN.b64 + meta.scm）：base64 分块，
# 供纯 Scheme 的 LLM 测试台（app/llm.scm + app/lib/llm_pure.scm）流式读入解码。
# 真源由 tools/gen_llm_model_b64.py 生成并随仓库提供。
mkdir -p "$LINGLONG/app/lib/llm/sort6" "$LINGLONG/app/lib/llm/psycho"
cp -f app/lib/llm/sort6/* "$LINGLONG/app/lib/llm/sort6/" 2>/dev/null || true
cp -f app/lib/llm/psycho/* "$LINGLONG/app/lib/llm/psycho/" 2>/dev/null || true
# Bad Apple 帧数据（app/lib/badapple/frames.txt + meta.scm）：定长记录文本，一帧一条，
# 由 app/badapple.scm 逐帧 (Fs.read_range) 读取；(native Fs) 只读文本，故数据必须落在这里。
# 真源是 tools/gen_badapple_data.py（从 badapple.h 的 C 数组生成）——真源改过就自动重生，
# 避免「改了 badapple.h 却忘了 make badapple-data」的隐患。
if [ "badapple.h" -nt "app/lib/badapple/frames.txt" ]; then
    echo "badapple.h 比 frames.txt 新，重新生成帧数据…"
    python3 tools/gen_badapple_data.py
fi
mkdir -p "$LINGLONG/app/lib/badapple"
cp -f app/lib/badapple/* "$LINGLONG/app/lib/badapple/" 2>/dev/null || true
# 把内置示例文档导出成真正的 .md（真源是 app/lib/md_demo.scm；阅读器首次运行也会落盘，
# 但构建期先摊出来，用户才能直接在资源目录里看到/编辑它）
python3 tools/gen_md_demo.py "$LINGLONG" || true
python3 - "$LINGLONG" "$OUT_DIR/manifest.json" <<'EOF'
import json, os, sys
root, out = sys.argv[1], sys.argv[2]
P1_DIRS = {"icon", "app"}
# app/ 下的大块数据不算「启动必需」：降为 p2 后台挂载（App 侧缺文件会自动等重试）
#   app/lib/llm/      —— 纯 Scheme LLM 测试台的 base64 模型分块
#   app/lib/badapple/ —— Bad Apple 帧数据（1.4MB 定长记录文本）
P2_PREFIXES = ("app/lib/llm/", "app/lib/badapple/")
P1_FILES = {"mob.scm", "jstest.js"}
SKIP_DIRS = {"data"}   # data 是 App 运行时数据（不入产物）
# music/ 不跳过，而是归为 p4「音频按需挂载」层：不进启动路径、不自动拉取，
# 进入音乐盒时由页面调 nanoMountTier(4) 拉入 MEMFS 并落本机存储（见 webgui/index.html）。
# 这样 wasm+js 里仍不含任何业务文件内容（隐私要求不变），首次进音乐盒一次性下载。
P4_DIRS = {"music"}
# LLM 模型：标准引擎（infer.c）的 Nano-168M/56M-Q80 作为 p3（最后台优先级，鹦鹉笼可用）；
# qwen3-0b6（599MB，单线程 WASM 不可用）不挂载。菜单按白名单+存在性过滤（ui_llm.c）。
P3_FILES = {"llm/nano-168m-q80.bin", "llm/nano-56m-q80.bin"}
files = []
for dirpath, dirnames, filenames in os.walk(root):
    rel_dir = os.path.relpath(dirpath, root)
    top = rel_dir.split(os.sep)[0]
    if top in SKIP_DIRS:
        dirnames[:] = []
        continue
    for fn in filenames:
        rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
        if top in P1_DIRS or (rel_dir == "." and (fn in P1_FILES or fn.startswith("wp"))):
            pri = 2 if rel.startswith(P2_PREFIXES) else 1
        elif rel in P3_FILES:
            pri = 3
        elif top in P4_DIRS:
            pri = 4
        elif top == "llm":
            continue
        else:
            pri = 2
        files.append({"path": rel, "size": os.path.getsize(os.path.join(dirpath, fn)), "priority": pri})
files.sort(key=lambda f: (f["priority"], f["path"]))
with open(out, "w") as fp:
    json.dump({"files": files}, fp, ensure_ascii=False)
n1 = sum(1 for f in files if f["priority"] == 1)
n3 = sum(1 for f in files if f["priority"] == 3)
n4 = sum(1 for f in files if f["priority"] == 4)
print(f"manifest: {len(files)} files (p1={n1}, p2={len(files)-n1-n3-n4}, p3={n3}, p4={n4})")
EOF

ANIMAC_DIR=vendor/Animac
ANIMAC_SRCS=$(ls $ANIMAC_DIR/*.c | grep -v '/exclude.c$')

"$EMCC" -O3 -ffast-math -DNANO_WASM_GUI \
    -I. -I$ANIMAC_DIR \
    $ANIMAC_SRCS \
    main_wasm_gui.c \
    hal_fs_linux.c \
    hal_fs_persist_wasm.c \
    nano_key_events.c \
    hal_net_wasm.c \
    hal_ram_wasm.c \
    hal_os_wasm.c \
    hal_audio_in_wasm.c \
    hal_audio_out_wasm.c \
    hal_display_wasm.c \
    hal_imu_wasm.c \
    hal_key_wasm.c \
    hal_touch_wasm.c \
    hal_power_linux.c \
    hal_misc_linux.c \
    celestial.c \
    celestial_ocean.c \
    ephemeris.c \
    flip.c \
    gfx_font_12.c \
    gfx_font_16.c \
    graphics.c \
    am_native_Gfx.c \
    am_native_Time.c \
    am_native_Diag.c \
    am_native_Touch.c \
    am_native_Imu.c \
    am_native_Key.c \
    am_native_Fs.c \
    am_native_Ui.c \
    am_native_App.c \
    am_native_Chat.c \
    am_native_Raster.c \
    infer.c \
    nano_fft.c \
    nano_min.c \
    nongli.c \
    ofdm_modem.c \
    pinyin_ime.c \
    tensor.c \
    tokenizer.c \
    utils.c \
    vsop87c_milli.c \
    ui.c \
    ui_app.c \
    ui_almanac.c \
    ui_calendar.c \
    ui_cloud.c \
    ui_dict.c \
    ui_ebook.c \
    ui_goldminer.c \
    ui_grid16kbd.c \
    ui_linglong_ctrl.c \
    ui_icon.c \
    ui_llm.c \
    ui_musicbox_mp3.c \
    ui_musicbox.c \
    ui_ofdm.c \
    ui_particlelife.c \
    ui_pedometer.c \
    ui_pinyin_ime.c \
    ui_ripple.c \
    ui_softkbd.c \
    ui_spectrogram.c \
    ui_tetris.c \
    ui_water.c \
    ui_scheme_app.c \
    -o "$OUT_DIR/nano_gui.js" \
    -s WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=402653184 \
    -s FILESYSTEM=1 \
    -s FORCE_FILESYSTEM=1 \
    -s EXIT_RUNTIME=0 \
    -s EXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,FS,callMain \
    -s EXPORTED_FUNCTIONS=_main,_wasm_gui_get_fb_rgba,_wasm_gui_get_fb_epoch,_wasm_gui_get_fb_width,_wasm_gui_get_fb_height,_wasm_gui_on_touch,_wasm_gui_on_key,_wasm_gui_on_key_event,_wasm_gui_on_orientation,_wasm_gui_input_tick,_wasm_gui_get_state

echo "built: $OUT_DIR/nano_gui.js / nano_gui.wasm（资源外置：$OUT_DIR/files → $LINGLONG）"
ls -lh "$OUT_DIR"/nano_gui.js "$OUT_DIR"/nano_gui.wasm "$OUT_DIR"/manifest.json | awk '{print " ", $5, $9}'
