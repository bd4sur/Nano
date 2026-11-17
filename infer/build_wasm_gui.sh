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
#   本脚本把 linglong 资源软链接进 webgui/files/ 并生成 manifest.json（文件清单+大小+优先级）；
#   页面 JS 启动时逐文件拉入 MEMFS（路径与 PLATFORM_ROOT_DIR 一致）。
#   资源分两类（见 doc/AGENTS.md 2026-11-18（五））：
#     p1 = app/ icon/ wallpaper/（含其下**任意深度**的全部内容，不做例外）：**服务器是
#          唯一来源**——每次页面加载全量重拉、无条件覆盖 MEMFS，并把 IndexedDB 里属于
#          这三处的条目（含墓碑）清空。
#     p2 = 其余一切（ebook/dict/music/llm 模型……**一个不落**，不按文件大小/模型种类
#          挑肥拣瘦）：**不自动拉**——页面状态栏有个「下载资源 <体积>」按钮，用户点了才拉，
#          写入 MEMFS + IndexedDB；两边已有同名文件则静默跳过（不重下、不覆盖）。
#
# 【写入边界】$LINGLONG 里**只有 app/ 是构建产出的**（下面整树镜像）；其余子目录
#   （icon/ wallpaper/ ebook/ dict/ music/ llm/ doc/ data/ …）一律由人工维护，
#   本脚本不去建、不去改、不去删。清单是从磁盘读来的（只读），不改动文件。
mkdir -p "$OUT_DIR"
ln -sfn "$LINGLONG" "$OUT_DIR/files"

# 同步灵机 App 资源目录（infer/app/ → $LINGLONG/app/）
#
# app/ 是**一个整体**：里面既有 .scm 源码（含 app/lib/ 的库），也有运行时要读的数据
# （hwr/hwr2 的训练权重 txt、llm 的 base64 分块、badapple 帧数据…）。所以这里做一次
# 整树镜像，而不是按子目录分别 rsync/cp。
#
# 【为什么】以前是“scm 走两次 rsync --delete（只认 *.scm），数据走三段 cp（只加不删）”，
# 实测已经出过一次事：app/lib/hwr2/ 的 14 个权重文件漏登记（没有任何一行拷它），
# 从干净资源目录构建出来的 hwr2 App 在浏览器里读不到权重；它现在能跑，只是因为
# 那些文件早先被手工拷过一次，而 rsync 的 --delete 对 exclude 掉的文件是“保护”不是删除
#（默认的 --delete 不删被 exclude 的目的端文件），于是残骸永远留着。
# 整树镜像同时把另一个坑堵上：数据/目录增删不用再到脚本里登记一行（漏登记就静默缺文件）。
#
# --delete：源里删了什么，资源目录就跟着删（宿主 App 列表是扫描资源目录得来的，
# 残留副本会让“已删除”的 App 继续出现在主菜单；数据残骸同理）。副作用是**手工放进
# $LINGLONG/app/ 的东西也会被清掉**——那本来就是这个目录的约定（它是构建产物）。
mkdir -p "$LINGLONG/app"
# 同步是**纯粹的目录镜像**：不做任何生成/转换。app/ 里的每一样东西（含 app/lib/badapple
# 的帧数据、app/lib/llm 的 base64 分块、hwr* 的权重）都是仓库里的普通文件，构建不参与
# 生产它们（2026-11-18 起取消了 badapple.h → frames.txt 那一步）。
rsync -a --delete \
    --exclude='.DS_Store' --exclude='__pycache__/' --exclude='*.pyc' \
    app/ "$LINGLONG/app/" \
    || { echo "未装 rsync，退化为整树拷贝（不会清理已删除文件）"; cp -rf app/. "$LINGLONG/app/"; }

# 内置示例文档不再由构建期写进资源目录：`$LINGLONG` 里**只有 app/ 是构建产出的**，
# 其余子目录（doc/ icon/ wallpaper/ ebook/ dict/ music/ llm/ …）一律由人工维护，
# 构建脚本不得去建、去改、去删（见 doc/AGENTS.md 2026-11-18（五））。
# 阅读器首次运行时仍会自己把示例 md 落盘到 doc/（浏览器侧只进 MEMFS + IndexedDB）。
python3 - "$LINGLONG" "$OUT_DIR/manifest.json" <<'EOF'
import json, os, sys
root, out = sys.argv[1], sys.argv[2]
# p1 = 这三个目录**递归下去的一切**，不做任何例外（app/lib/badapple、app/lib/llm 同样
#      算 p1：它们在 infer/app/ 里，就跟着 app/ 一起整体同步、一起每次全量刷新）。
#      它们也是**服务器唯一来源**（页面每次加载全量重拉、无条件覆盖 MEMFS，并清空
#      IndexedDB 里属于这三处的条目）。页面靠 p1_dirs 判定“哪些路径归服务器管”。
P1_DIRS = {"app", "icon", "wallpaper"}
SKIP_DIRS = {"data"}   # data 是 App 运行时数据（不入产物）
# 清单是 llm/ 下的**全量**：构建不对任何具体文件做取舍（大模型也一样——它们只是 p2，
# 用户不点「下载资源」就不会花那份流量）。是否"跑得动"属于运行时/App 的判断
# （如 app/nanochat.scm 的模型表按文件存在性过滤），不是构建该替用户决定的事。
files = []
for dirpath, dirnames, filenames in os.walk(root):
    rel_dir = os.path.relpath(dirpath, root)
    top = rel_dir.split(os.sep)[0]
    if top in SKIP_DIRS:
        dirnames[:] = []
        continue
    for fn in filenames:
        rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
        pri = 1 if top in P1_DIRS else 2
        files.append({"path": rel, "size": os.path.getsize(os.path.join(dirpath, fn)), "priority": pri})
files.sort(key=lambda f: (f["priority"], f["path"]))
with open(out, "w") as fp:
    json.dump({"p1_dirs": sorted(P1_DIRS), "files": files}, fp, ensure_ascii=False)
n1 = sum(1 for f in files if f["priority"] == 1)
b1 = sum(f["size"] for f in files if f["priority"] == 1)
bt = sum(f["size"] for f in files)
print(f"manifest: {len(files)} files (p1={n1}/{b1/1048576:.1f}MB，其余 {len(files)-n1}/{(bt-b1)/1048576:.1f}MB 后台挂载)")
EOF

ANIMAC_DIR=vendor/Animac
ANIMAC_SRCS=$(ls $ANIMAC_DIR/*.c | grep -v '/exclude.c$')

# ---------------------------------------------------------------------------
# 代码生成选项（性能专项 A1）
#   -O3 -ffast-math        既有（-ffast-math 含 -fno-math-errno / -freciprocal-math 等）
#   -mnontrapping-fptoint  float→int 用 i32.trunc_sat（单条指令）代替 trapping 转换 +
#                          越界分支：本工程逐像素内核有大量 (int)floorf（噪声三线性采样、
#                          上采样、调色映射），默认模式下 LLVM 要为每次转换生成饱和检查。
#   -mbulk-memory          memcpy/memset → memory.copy / memory.fill（帧缓冲整行拷贝、
#                          清屏、上采样都靠它）。需 2020+ 浏览器。
#   -msimd128              开 SIMD128 自动向量化。**实测默认关**：玲珑仪的三条热点路径
#                          （大气积分、云 ray-march、海面 march/着色）都是分支密集 + 逐像素
#                          标量，LLVM 无法自动向量化；A/B 实测开/关差异在噪声内
#                          （m0+海洋 38.44 vs 38.67 fps、体积云 10.19 vs 10.13 fps），
#                          而产物大 100KB 且要求 2021+ 浏览器。手写 SIMD 内核时再打开。
#   -flto                  全程序优化（跨翻译单元内联，如 ui_cloud ↔ celestial ↔ graphics）
# 可用环境变量覆盖做基准：WASM_SIMD=1 WASM_LTO=0 ./build_wasm_gui.sh
# ---------------------------------------------------------------------------
WASM_SIMD=${WASM_SIMD:-0}
WASM_LTO=${WASM_LTO:-1}
WASM_OPT="-O3 -ffast-math -mnontrapping-fptoint -mbulk-memory"
if [ "$WASM_SIMD" = "1" ]; then WASM_OPT="$WASM_OPT -msimd128"; fi
if [ "$WASM_LTO" = "1" ]; then WASM_OPT="$WASM_OPT -flto"; fi

# -DAM_NATIVE_MAX_LIBS=32：宿主本地库数量上限（默认 16 已用满，第 17 个是 am_native_Audio.c）
"$EMCC" $WASM_OPT -DNANO_WASM_GUI -DAM_NATIVE_MAX_LIBS=32 \
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
    am_native_Music.c \
    am_native_Audio.c \
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
    ui_cloud.c \
    ui_dict.c \
    ui_ebook.c \
    ui_goldminer.c \
    ui_grid16kbd.c \
    ui_linglong_ctrl.c \
    ui_settings.c \
    ui_icon.c \
    ui_llm.c \
    ui_musicbox_mp3.c \
    ui_musicbox.c \
    ui_ofdm.c \
    ui_pinyin_ime.c \
    ui_ripple.c \
    ui_softkbd.c \
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
