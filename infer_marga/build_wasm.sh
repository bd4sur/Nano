#!/bin/bash

# 工具链部署备忘：
# 1）下载：https://github.com/WebAssembly/wasi-sdk/releases/tag/wasi-sdk-24
# 2）解压到BASE_PATH
# 3）下载clang_rt.builtins-wasm32.a（LLVM18）
# 3）放置到${WASI_SDK_PATH}/lib/wasi

# 注意！
# 编译前，取消设置infer.h中的 `NANO_USE_MMAP` 和 `MATMUL_PTHREAD` 两个宏

export BASE_PATH=/home/bd4sur/app/wasi
export WASI_VERSION=24.0
export WASI_SDK_PATH=${BASE_PATH}/wasi-sdk-${WASI_VERSION}-arm64-linux

export CC="${WASI_SDK_PATH}/bin/clang --sysroot=${WASI_SDK_PATH}/share/wasi-sysroot"
export LD="${WASI_SDK_PATH}/bin/wasm-ld"

$CC --target=wasm32-wasi -O3 -I. -o obj_main_wasm.o -c main_wasm.c
$CC --target=wasm32-wasi -O3 -I. -o obj_bpe.o -c bpe.c
$CC --target=wasm32-wasi -O3 -I. -o obj_hashmap.o -c hashmap.c
$CC --target=wasm32-wasi -O3 -I. -o obj_trie.o -c trie.c
$CC --target=wasm32-wasi -O3 -I. -o obj_infer.o -c infer.c

$LD --export-dynamic --allow-undefined --lto-O3 \
  -L${WASI_SDK_PATH}/share/wasi-sysroot/lib/wasm32-wasi \
  -lc -lc++ -lc++abi ${WASI_SDK_PATH}/share/wasi-sysroot/lib/wasm32-wasi/crt1.o \
  --export=malloc \
  --export=free \
  --export=init_nano \
  --export=set_sampler \
  --export=generate_next_token_external \
  --export=encode_external \
  --export=decode_external \
  --export=load_lora_external \
  --export=unload_lora_external \
  --export=close_nano \
  --no-entry \
  --import-memory -L${WASI_SDK_PATH}/lib/wasi -lclang_rt.builtins-wasm32 obj_main_wasm.o obj_bpe.o obj_hashmap.o obj_trie.o obj_infer.o -o ../infer/nano_infer.wasm \

rm -f obj_main_wasm.o obj_bpe.o obj_hashmap.o obj_trie.o obj_infer.o

# 作为参考，用emscripten编译的选项如下
if false; then
/home/bd4sur/emsdk/upstream/emscripten/emcc -O3 -ffast-math \
    ../infer_c/infer.c \
    -o infer.wasm \
    -s WASM=1 \
    -s STANDALONE_WASM \
    -s EXPORTED_FUNCTIONS="[_malloc, _free, _init_nano, _generate_next_token_external, _encode_external, _decode_external, _close_nano]" \
    --no-entry
fi
