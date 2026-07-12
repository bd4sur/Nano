#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_pinyin_table.py

读取 字频+拼音 CSV，生成 C 语言硬编码查找表。
规则：
  1. 完全忽略拼音尾部的音调数字（如 de2/dei3 均视为 de/dei）
  2. 仅支持完全匹配，无前缀查找
  3. 所有候选字按字频降序预排列，运行时零排序开销

用法：
  python generate_pinyin_table.py hanzi.csv pinyin_lut.c
  gcc -o pinyin pinyin_lut.c -DPINYIN_TEST_MAIN
  ./pinyin

"""

import csv
import sys
import os
from collections import defaultdict


def strip_tone(pinyin):
    """去掉拼音尾部的音调数字（1-9）"""
    if not pinyin:
        return pinyin
    i = len(pinyin) - 1
    while i >= 0 and pinyin[i].isdigit():
        i -= 1
    return pinyin[:i + 1]


def sanitize_for_identifier(s):
    """将拼音转为合法的 C 标识符后缀"""
    return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in s)


def parse_csv(csv_path):
    """
    解析 CSV，返回：
      hanzi_list      : [(unicode, freq), ...]  按 CSV 出现顺序（字频降序）
      pinyin_entries  : [(pinyin, [hanzi_idx, ...]), ...]  按拼音字母序
    """
    hanzi_list = []          # idx -> (unicode, freq)
    hanzi_to_idx = {}        # unicode -> idx

    # pinyin -> [(hanzi_idx, freq), ...]
    pinyin_to_hanzi = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 5:
                continue
            try:
                _ = int(row[0].strip())
            except ValueError:
                continue

            hanzi_char = row[1].strip()
            if not hanzi_char:
                continue

            try:
                freq = int(row[2].strip())
            except ValueError:
                continue

            pinyin_field = row[4].strip()
            if not pinyin_field:
                continue

            # 解析拼音，去掉音调数字，同一行内去重
            raw_pinyins = [p.strip() for p in pinyin_field.split('/') if p.strip()]
            seen_py = set()
            pinyins = []
            for py in raw_pinyins:
                py_no_tone = strip_tone(py).lower()
                if py_no_tone and py_no_tone not in seen_py:
                    seen_py.add(py_no_tone)
                    pinyins.append(py_no_tone)

            if not pinyins:
                continue

            unicode_cp = ord(hanzi_char)

            if unicode_cp not in hanzi_to_idx:
                hanzi_to_idx[unicode_cp] = len(hanzi_list)
                hanzi_list.append((unicode_cp, freq))

            hanzi_idx = hanzi_to_idx[unicode_cp]

            for py in pinyins:
                # 去重：同一拼音下同一汉字只保留一次
                # CSV 本身按字频降序，首次出现即为最高字频
                exists = any(existing_idx == hanzi_idx
                             for existing_idx, _ in pinyin_to_hanzi[py])
                if not exists:
                    pinyin_to_hanzi[py].append((hanzi_idx, freq))

    # 整理：按拼音字母序，每个拼音下的汉字按字频降序
    pinyin_entries = []
    for py in sorted(pinyin_to_hanzi.keys()):
        items = pinyin_to_hanzi[py]
        # 按字频降序（CSV 大概率已有序，此处保险排序）
        items.sort(key=lambda x: x[1], reverse=True)
        indices = [idx for idx, _ in items]
        pinyin_entries.append((py, indices))

    return hanzi_list, pinyin_entries


def generate_c_code(hanzi_list, pinyin_entries, output_path):
    lines = []
    lines.append('/* Auto-generated pinyin to hanzi lookup table */')
    lines.append('/* DO NOT EDIT MANUALLY */')
    lines.append('')
    lines.append('#include <stdint.h>')
    lines.append('#include <string.h>')
    lines.append('')
    lines.append('/* ===================== 数据表 ===================== */')
    lines.append('')

    # 1. 汉字表（仅保留 Unicode，字频仅用于离线排序）
    lines.append('/* Hanzi table: index -> unicode code point */')
    lines.append(f'/* Total unique hanzi: {len(hanzi_list)} */')
    lines.append('static const uint32_t g_hanzi_table[] = {')
    for unicode, freq in hanzi_list:
        ch = chr(unicode)
        lines.append(f'    0x{unicode:04X},  /* {ch} (freq={freq}) */')
    lines.append('};')
    lines.append('')

    # 2. 拼音对应的汉字索引数组
    lines.append('/* Pinyin to hanzi index arrays (pre-sorted by frequency desc) */')
    for py, indices in pinyin_entries:
        var_name = 'idx_' + sanitize_for_identifier(py)
        idx_str = ', '.join(str(i) for i in indices)
        lines.append(f'static const uint16_t {var_name}[] = {{{idx_str}}};')
    lines.append('')

    # 3. 拼音条目表（按拼音字符串排序，用于二分查找）
    lines.append('/* Pinyin lookup table (sorted by pinyin string) */')
    lines.append(f'/* Total pinyin entries: {len(pinyin_entries)} */')
    lines.append('static const struct {')
    lines.append('    const char *pinyin;')
    lines.append('    const uint16_t *hanzi_indices;')
    lines.append('    uint16_t hanzi_count;')
    lines.append('} g_pinyin_table[] = {')
    for py, indices in pinyin_entries:
        var_name = 'idx_' + sanitize_for_identifier(py)
        lines.append(f'    {{"{py}", {var_name}, {len(indices)}}},')
    lines.append('};')
    lines.append('')

    lines.append('#define PINYIN_COUNT (sizeof(g_pinyin_table) / sizeof(g_pinyin_table[0]))')
    lines.append('#define HANZI_COUNT (sizeof(g_hanzi_table) / sizeof(g_hanzi_table[0]))')
    lines.append('')

    # 4. 核心算法（完全匹配，零运行时排序/去重）
    lines.append('/* ===================== 查找算法 ===================== */')
    lines.append('')
    lines.append('/**')
    lines.append(' * @brief 拼音转候选汉字（完全匹配，忽略音调）')
    lines.append(' * @param pinyin            输入拼音（无音调数字），如 "di"')
    lines.append(' * @param hanzi_candidates  输出缓冲区，写入 UTF-32 码点')
    lines.append(' * @return 候选汉字数量')
    lines.append(' *')
    lines.append(' * @note 调用者需保证 hanzi_candidates 至少有 HANZI_COUNT 个元素。')
    lines.append(' * @note 返回的候选字已按字频降序预排列，无需运行时排序。')
    lines.append(' */')
    lines.append('size_t pinyin_to_hanzi(char *pinyin, uint32_t *hanzi_candidates) {')
    lines.append('    size_t left = 0, right = PINYIN_COUNT;')
    lines.append('    size_t mid;')
    lines.append('    int cmp;')
    lines.append('')
    lines.append('    /* 二分查找完全匹配的拼音 */')
    lines.append('    while (left < right) {')
    lines.append('        mid = left + (right - left) / 2;')
    lines.append('        cmp = strcmp(g_pinyin_table[mid].pinyin, pinyin);')
    lines.append('        if (cmp < 0) {')
    lines.append('            left = mid + 1;')
    lines.append('        } else if (cmp > 0) {')
    lines.append('            right = mid;')
    lines.append('        } else {')
    lines.append('            /* 找到完全匹配，直接复制预排序的候选字 */')
    lines.append('            int i;')
    lines.append('            for (i = 0; i < g_pinyin_table[mid].hanzi_count; i++) {')
    lines.append('                hanzi_candidates[i] = g_hanzi_table[g_pinyin_table[mid].hanzi_indices[i]];')
    lines.append('            }')
    lines.append('            return g_pinyin_table[mid].hanzi_count;')
    lines.append('        }')
    lines.append('    }')
    lines.append('    return 0;  /* 未找到 */')
    lines.append('}')
    lines.append('')

    # 5. 测试代码
    lines.append('/* ===================== 测试代码 ===================== */')
    lines.append('#ifdef PINYIN_TEST_MAIN')
    lines.append('#include <stdio.h>')
    lines.append('')
    lines.append('/* 将 UTF-32 码点转为 UTF-8 字节序列并输出，不依赖 locale */')
    lines.append('static void print_utf32_char(uint32_t cp) {')
    lines.append('    char buf[5] = {0};')
    lines.append('    if (cp <= 0x7F) {')
    lines.append('        buf[0] = (char)cp;')
    lines.append('    } else if (cp <= 0x7FF) {')
    lines.append('        buf[0] = (char)(0xC0 | (cp >> 6));')
    lines.append('        buf[1] = (char)(0x80 | (cp & 0x3F));')
    lines.append('    } else if (cp <= 0xFFFF) {')
    lines.append('        buf[0] = (char)(0xE0 | (cp >> 12));')
    lines.append('        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));')
    lines.append('        buf[2] = (char)(0x80 | (cp & 0x3F));')
    lines.append('    } else {')
    lines.append('        buf[0] = (char)(0xF0 | (cp >> 18));')
    lines.append('        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));')
    lines.append('        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));')
    lines.append('        buf[3] = (char)(0x80 | (cp & 0x3F));')
    lines.append('    }')
    lines.append('    printf("%s", buf);')
    lines.append('}')
    lines.append('')
    lines.append('int main(int argc, char *argv[]) {')
    lines.append('    uint32_t candidates[HANZI_COUNT];')
    lines.append('    const char *tests[] = {"de", "zhi", "li", "ji", "yi", "xyz"};')
    lines.append('    size_t n_tests = sizeof(tests) / sizeof(tests[0]);')
    lines.append('    size_t t, i, n;')
    lines.append('    (void)argc; (void)argv;')
    lines.append('')
    lines.append('    for (t = 0; t < n_tests; t++) {')
    lines.append('        n = pinyin_to_hanzi((char*)tests[t], candidates);')
    lines.append('        printf("pinyin: %-6s -> %lu candidates: ",')
    lines.append('               tests[t], (unsigned long)n);')
    lines.append('        for (i = 0; i < n; i++) {')
    lines.append('            printf("U+%04X(", candidates[i]);')
    lines.append('            print_utf32_char(candidates[i]);')
    lines.append('            printf(") ");')
    lines.append('        }')
    lines.append('        printf("\\n");')
    lines.append('    }')
    lines.append('    return 0;')
    lines.append('}')
    lines.append('#endif')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[OK] Generated: {output_path}')
    print(f'     Unique hanzi: {len(hanzi_list)}')
    print(f'     Pinyin entries: {len(pinyin_entries)}')
    for py, indices in pinyin_entries:
        chars = ''.join(chr(hanzi_list[i][0]) for i in indices)
        print(f'     {py}: {chars}')


def main():
    if len(sys.argv) < 3:
        print('Usage: python generate_pinyin_table.py <input.csv> <output.c>')
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2]
    if not os.path.exists(csv_path):
        print(f'Error: file not found: {csv_path}')
        sys.exit(1)

    hanzi_list, pinyin_entries = parse_csv(csv_path)
    generate_c_code(hanzi_list, pinyin_entries, out_path)


if __name__ == '__main__':
    main()
