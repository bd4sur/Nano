#!/bin/bash

export BASE_PATH=/home/bd4sur/ai/wasi

export WASI_VERSION=24
export WASI_VERSION_FULL=${WASI_VERSION}.0
export WASI_FILENAME=wasi-sdk-${WASI_VERSION_FULL}-arm64-linux.tar.gz
export BUILTINS_LIBRARY_FILENAME=libclang_rt.builtins-wasm32
export BUILTINS_LIBRARY_TAR_FILENAME=${BUILTINS_LIBRARY_FILENAME}-${WASI_VERSION_FULL}.tar.gz
export WASI_SDK_PATH=${BASE_PATH}/wasi-sdk-${WASI_VERSION_FULL}-arm64-linux
export CC="${WASI_SDK_PATH}/bin/clang --sysroot=${WASI_SDK_PATH}/share/wasi-sysroot"
export LD="${WASI_SDK_PATH}/bin/wasm-ld"

$CC --target=wasm32-wasi -O3 -I. -o infer.o -c ../infer_c/infer.c
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
  --export=close_nano \
  --no-entry \
  --import-memory -L${BASE_PATH}/lib/wasi -lclang_rt.builtins-wasm32 infer.o -o infer.wasm \

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
