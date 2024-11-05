emcc -O3 -ffast-math \
    infer.c \
    -o infer.html \
    -s WASM=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s ALLOW_TABLE_GROWTH=1 \
    --preload-file "base.bin" \
    --preload-file "lora.bin"
