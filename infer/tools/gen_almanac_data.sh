#!/usr/bin/env bash
# gen_almanac_data.sh —— 重新生成 app/lib/almanac_data.scm（黄历数据表）
#
# 真源是 infer/almanac.h（cnlunar C 实现）。生成器直接 include 该头文件逐表导出，
# 因此数据表与 C 实现永远一致。改动 almanac.h 后跑一次本脚本即可。
#
# 用法： bash tools/gen_almanac_data.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=/tmp/gen_almanac_tables
gcc -O2 -I"$ROOT" -o "$OUT" "$HERE/gen_almanac_tables.c"
"$OUT" emit > "$ROOT/app/lib/almanac_data.scm"
echo "generated: $ROOT/app/lib/almanac_data.scm ($(wc -c < "$ROOT/app/lib/almanac_data.scm") bytes)"
