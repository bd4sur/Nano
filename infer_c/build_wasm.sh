/home/bd4sur/emsdk/upstream/emscripten/emcc -O2 -ffast-math \
    infer.c \
    -o infer.wasm \
    -s WASM=1 \
    -s STANDALONE_WASM \
    -s EXPORTED_FUNCTIONS="[_malloc, _free, _test_wasm, _init_nano, _generate_next_token_external, _encode_external, _decode_external, _close_nano]" \
    -s INITIAL_MEMORY=4294967296 \
    --no-entry \

