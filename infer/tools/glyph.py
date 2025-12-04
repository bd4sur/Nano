from tqdm import tqdm
import math
import json
from bdfparser import Font

font = Font("D:/Desktop/ssd1309/wenquanyi_9pt.bdf") #github.com/larryli/u8g2_wqy

def get_glyph_bitmap(char, font_width, font_height):
    g = font.glyph(char)
    if g is None:
        return []
    img = g.draw().crop(font_width, font_height, 0, 1).todata(2)
    glyph =[]
    for p in range(math.ceil(font_height / 8)):
        for x in range(font_width):
            block = 0
            for y in range(p * 8, min(font_height, (p+1) * 8)):
                if img[y][x] == 1:
                    block = block | (1 << (y - p * 8))
            glyph.append(block)
    return glyph

def render(glyph, font_width, font_height):
    matrix =[]
    for x in range(font_width):
        matrix.append([])
        for y in range(font_height):
            matrix[x].append(0)
    block_count = 0
    for p in range(math.ceil(font_height / 8)):
        for x in range(font_width):
            block = glyph[block_count]
            for y in range(p * 8, min(font_height, (p+1) * 8)):
                pixel = 1 & (block >> (y - p * 8))
                matrix[x][y] = pixel
            block_count = block_count + 1
    for y in range(font_height):
        line = ""
        for x in range(font_width):
            line += "." if matrix[x][y] == 0 else"#"
        print(line)


gbindex_to_glyph = [] # 从（连续的）码点到字模的映射，硬编码到头文件

index = 0
for i in range(32, 127):
    gbindex_to_glyph.append(chr(i))
    index = index + 1
for zone in range(1, 95):    # 区码范围1-94
    for pos in range(1, 95): # 位码范围1-94
        high = zone + 0xA0   # 高位字节 = 区码 + A0
        low = pos + 0xA0     # 低位字节 = 位码 + A0
        try:
            char = bytes([high, low]).decode('gb2312')
            gbindex_to_glyph.append(char)
            index = index + 1
        except UnicodeDecodeError:
            pass

u2g_map = [[], []] # 从UTF32码点到连续码点的映射，硬编码到头文件

gbindex_to_char = {}

gbindex_to_glyph_string = ""

for i,ch in tqdm(enumerate(gbindex_to_glyph)):
    if i < 95:
        continue
    utf32 = ord(ch)
    u2g_map[0].append(utf32)
    u2g_map[1].append(i)
    gbindex_to_char[i] = ch
    glyph = get_glyph_bitmap(ch, 12, 12)
    gbindex_to_glyph[i] = glyph

    glyph_str = "{"
    for n in range(len(glyph)):
        glyph_str += str(hex(glyph[n]))
        glyph_str += ","
    glyph_str += f"}}, /* [{ch}] , {i} */"

    gbindex_to_glyph_string += glyph_str
    gbindex_to_glyph_string += "\n"

"""
content ="中文和全角，符号。￥啊啊啊"
for c in content:
    utf32 = ord(c)
    for i in range(len(u2g_map[0])):
        if u2g_map[0][i] == utf32:
            gbindex = u2g_map[1][i]
            break
    glyph = gbindex_to_glyph[gbindex]
    render(glyph, 12, 12)
"""


with open("utf32_to_gbindex.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(u2g_map))

"""
with open("gbindex_to_char.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(gbindex_to_char))
"""


with open("gbindex_to_glyph.txt", "w", encoding="utf-8") as f:
    f.write(gbindex_to_glyph_string)