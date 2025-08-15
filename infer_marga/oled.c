#include "oled.h"
#include "oledfont.h"

// uint8_t OLED_FRAME_BUFFER[OLED_PAGES][OLED_WIDTH];
static uint8_t **OLED_FRAME_BUFFER;

static AVLNode *GLYPH;

static int i2cdev_fd;

// 发送命令
void OLED_WriteData(uint8_t data) {
    uint8_t buf[2];
    buf[0] = OLED_Data_Mode;
    buf[1] = data;
    if( write(i2cdev_fd, &buf, 2) != 2 ) printf("I2C write error\n");
    /*
    Wire.beginTransmission(OLED_I2C_ADDR);
    Wire.write(OLED_Data_Mode);
    Wire.write(data);
    Wire.endTransmission();
    */
}

// 发送数据
void OLED_WriteCommand(uint8_t cmd) {
    uint8_t buf[2];
    buf[0] = OLED_Command_Mode;
    buf[1] = cmd;
    if( write(i2cdev_fd, &buf, 2) != 2 ) printf("I2C write error\n");
    /*
    Wire.beginTransmission(OLED_I2C_ADDR);
    Wire.write(OLED_Command_Mode);
    Wire.write(cmd);
    Wire.endTransmission();
    */
}

// 更新显存到OLED
void OLED_Refresh(void) {
    for (uint8_t row = 0; row < OLED_PAGES; row++) {
        uint8_t col = 0;
        OLED_WriteCommand(0xb0 + row); // 设置行起始地址
        OLED_WriteCommand(0x00 + (col & 0x0F));        // 设置低列起始地址
        OLED_WriteCommand(0x10 + (col & 0x0F)); // 设置高列起始地址

        // for (u8 col = 0; col < OLED_WIDTH; col++) {
        //     OLED_WriteData(OLED_FRAME_BUFFER[row][col]);
        // }

        uint8_t buf[129];
        buf[0] = OLED_Data_Mode;
        for (uint8_t col = 0; col < OLED_WIDTH; col++) {
            buf[col + 1] = OLED_FRAME_BUFFER[row][col];
        }
        if( write(i2cdev_fd, buf, (OLED_WIDTH+1)) != (OLED_WIDTH+1) ) printf("I2C write error\n");

        // Wire.beginTransmission(OLED_I2C_ADDR);
        // Wire.write(OLED_Data_Mode);
        // Wire.write(OLED_FRAME_BUFFER[row], 128);
        // Wire.endTransmission();
    
        // 有32字节的缓冲区??
/*
        while (col < OLED_WIDTH) {
            uint8_t buf[64];
            buf[0] = OLED_Data_Mode;
            uint8_t actual_buf_length = 1;
            for (uint8_t i = 1; i < 64; i++) {
                buf[i] = OLED_FRAME_BUFFER[row][col];
                col++;
                actual_buf_length++;
                if(col >= OLED_WIDTH) break;
            }
            if( write(i2cdev_fd, buf, actual_buf_length) != actual_buf_length ) printf("I2C write error\n");
        }
*/
    }
}

// OLED的初始化
void OLED_Init(void) {
    // 初始化GDRAM
    OLED_FRAME_BUFFER = (uint8_t **)calloc(OLED_PAGES, sizeof(uint8_t *));
    for(uint8_t i = 0; i < OLED_PAGES; i++) {
        OLED_FRAME_BUFFER[i] = (uint8_t *)calloc(OLED_WIDTH, sizeof(uint8_t));
    }

    // 初始化字库
    GLYPH = buildAVLTree(UTF32_LUT, 7445);

    // 初始化屏幕设备
    i2cdev_fd = open(OLED_I2C_DEVFILE, O_RDWR);
    if (i2cdev_fd < 0) {
        printf("OLED open error : %s\r\n", strerror(errno));
    }
    if (ioctl(i2cdev_fd, I2C_SLAVE, OLED_I2C_ADDR) < 0) {
        printf("OLED ioctl error : %s\r\n", strerror(errno));
    }

    delay(100);

    OLED_WriteCommand(0xFD);
    OLED_WriteCommand(0x12);
    OLED_WriteCommand(0xAE); //--turn off oled panel
    OLED_WriteCommand(0xd5); //--set display clock divide ratio/oscillator frequency
    OLED_WriteCommand(0xf0);
    OLED_WriteCommand(0xA8); //--set multiplex ratio(1 to 64)
    OLED_WriteCommand(0x3f); //--1/64 duty
    OLED_WriteCommand(0xD3); //-set display offset  Shift Mapping RAM Counter (0x00~0x3F)
    OLED_WriteCommand(0x00); //-not offset
    OLED_WriteCommand(0x40); //--set start line address  Set Mapping RAM Display Start Line (0x00~0x3F)
    OLED_WriteCommand(0xA1); //--Set SEG/Column Mapping     0xa0左右反置 0xa1正常
    OLED_WriteCommand(0xC8); // Set COM/Row Scan Direction   0xc0上下反置 0xc8正常
    OLED_WriteCommand(0xDA); //--set com pins hardware configuration
    OLED_WriteCommand(0x12);
    OLED_WriteCommand(0x81); //--set contrast control register
    OLED_WriteCommand(0xFF); // Set SEG Output Current Brightness
    OLED_WriteCommand(0xD9); //--set pre-charge period
    OLED_WriteCommand(0x82); // Set Pre-Charge as 15 Clocks & Discharge as 1 Clock
    OLED_WriteCommand(0xDB); //--set vcomh
    OLED_WriteCommand(0x34); // Set VCOM Deselect Level
    OLED_WriteCommand(0xA4); // Disable Entire Display On (0xa4/0xa5)
    OLED_WriteCommand(0xA6); // Disable Inverse Display On (0xa6/a7)
    OLED_Clear();
    OLED_WriteCommand(0x2E); // Stop scroll
    OLED_WriteCommand(0x20); // Set Memory Addressing Mode
    OLED_WriteCommand(0x00); // Set Memory Addressing Mode ab Horizontal addressing mode
    OLED_WriteCommand(0xAF);
}

void OLED_Close(void) {
    close(i2cdev_fd);
    freeTree(GLYPH);
}

// 开启OLED显示
void OLED_Display_On(void) {
    OLED_WriteCommand(0x8D); // 电荷泵使能
    OLED_WriteCommand(0x14); // 开启电荷泵
    OLED_WriteCommand(0xAF); // 点亮屏幕
}

// 关闭OLED显示
void OLED_Display_Off(void) {
    OLED_WriteCommand(0x8D); // 电荷泵使能
    OLED_WriteCommand(0x10); // 关闭电荷泵
    OLED_WriteCommand(0xAE); // 关闭屏幕
}

// 清屏函数
void OLED_Clear(void) {
    for (uint8_t i = 0; i < OLED_PAGES; i++) {
        for (uint8_t n = 0; n < OLED_WIDTH; n++) {
            OLED_FRAME_BUFFER[i][n] = 0;
        }
    }
    OLED_Refresh();
}

// 清屏函数
void OLED_SoftClear(void) {
    for (uint8_t i = 0; i < OLED_PAGES; i++) {
        for (uint8_t n = 0; n < OLED_WIDTH; n++) {
            OLED_FRAME_BUFFER[i][n] = 0;
        }
    }
}

// 画点
// x:0~127
// y:0~63
// t:1 填充 0,清空
void OLED_DrawPoint(uint8_t x, uint8_t y, uint8_t t) {
    if (x >= OLED_WIDTH || y >= OLED_HEIGHT) {
        // printf("WARNING: Plot Coord Out of bound! Cancelled.\n");
        return;
    }
    uint8_t i, m, n;
    i = y / OLED_PAGES;
    m = y % OLED_PAGES;
    n = 1 << m;
    if (t) {
        OLED_FRAME_BUFFER[i][x] |= n;
    }
    else {
        OLED_FRAME_BUFFER[i][x] = ~OLED_FRAME_BUFFER[i][x];
        OLED_FRAME_BUFFER[i][x] |= n;
        OLED_FRAME_BUFFER[i][x] = ~OLED_FRAME_BUFFER[i][x];
    }
}

// 画线
// x1,y1:起点坐标
// x2,y2:结束坐标
void OLED_DrawLine(uint8_t x1, uint8_t y1, uint8_t x2, uint8_t y2, uint8_t mode) {
    int t;
    int xerr = 0, yerr = 0, delta_x, delta_y, distance;
    int incx, incy, uRow, uCol;
    delta_x = x2 - x1; // 计算坐标增量
    delta_y = y2 - y1;
    uRow = x1; // 画线起点坐标
    uCol = y1;
    if (delta_x > 0)
        incx = 1; // 设置单步方向
    else if (delta_x == 0)
        incx = 0; // 垂直线
    else
    {
        incx = -1;
        delta_x = -delta_x;
    }
    if (delta_y > 0)
        incy = 1;
    else if (delta_y == 0)
        incy = 0; // 水平线
    else
    {
        incy = -1;
        delta_y = -delta_x;
    }
    if (delta_x > delta_y)
        distance = delta_x; // 选取基本增量坐标轴
    else
        distance = delta_y;
    for (t = 0; t < distance + 1; t++)
    {
        OLED_DrawPoint(uRow, uCol, mode); // 画点
        xerr += delta_x;
        yerr += delta_y;
        if (xerr > distance)
        {
            xerr -= distance;
            uRow += incx;
        }
        if (yerr > distance)
        {
            yerr -= distance;
            uCol += incy;
        }
    }
}

// x,y:圆心坐标
// r:圆的半径
void OLED_DrawCircle(uint8_t x, uint8_t y, uint8_t r)
{
    int a, b, num;
    a = 0;
    b = r;
    while (2 * b * b >= r * r) {
        OLED_DrawPoint(x + a, y - b, 1);
        OLED_DrawPoint(x - a, y - b, 1);
        OLED_DrawPoint(x - a, y + b, 1);
        OLED_DrawPoint(x + a, y + b, 1);

        OLED_DrawPoint(x + b, y + a, 1);
        OLED_DrawPoint(x + b, y - a, 1);
        OLED_DrawPoint(x - b, y - a, 1);
        OLED_DrawPoint(x - b, y + a, 1);

        a++;
        num = (a * a + b * b) - r * r; // 计算画的点离圆心的距离
        if (num > 0)
        {
            b--;
            a--;
        }
    }
}

// 显示汉字
// x,y:起点坐标
// num:汉字对应的序号
// mode:0,反色显示;1,正常显示
void OLED_ShowChar(uint8_t x, uint8_t y, uint8_t *glyph, uint8_t font_width, uint8_t font_height, uint8_t mode) {
    uint8_t x0 = x, y0 = y;
    uint32_t bytenum = (font_height / 8 + ((font_height % 8) ? 1 : 0)) * font_width; // 得到字体一个字符对应点阵集所占的字节数
    for (uint32_t i = 0; i < bytenum; i++) {
        uint8_t temp = glyph[i];
        for (uint8_t m = 0; m < 8; m++) {
            if (temp & 0x01)
                OLED_DrawPoint(x, y, mode);
            else
                OLED_DrawPoint(x, y, !mode);
            temp >>= 1;
            y++;
        }
        x++;
        if ((x - x0) == font_width) {
            x = x0;
            y0 = y0 + 8;
        }
        y = y0;
    }
}

// x,y：起点坐标
// sizex,sizey,图片长宽
// BMP[]：要写入的图片数组
// mode:0,反色显示;1,正常显示
void OLED_ShowPicture(uint8_t x, uint8_t y, uint8_t sizex, uint8_t sizey, uint8_t BMP[], uint8_t mode) {
    u16 j = 0;
    uint8_t i, n, temp, m;
    uint8_t x0 = x, y0 = y;
    sizey = sizey / 8 + ((sizey % 8) ? 1 : 0);
    for (n = 0; n < sizey; n++)
    {
        for (i = 0; i < sizex; i++)
        {
            temp = BMP[j];
            j++;
            for (m = 0; m < 8; m++)
            {
                if (temp & 0x01)
                    OLED_DrawPoint(x, y, mode);
                else
                    OLED_DrawPoint(x, y, !mode);
                temp >>= 1;
                y++;
            }
            x++;
            if ((x - x0) == sizex)
            {
                x = x0;
                y0 = y0 + 8;
            }
            y = y0;
        }
    }
}


// 反显函数
void OLED_ColorTurn(uint8_t i) {
    if (i == 0) {
        OLED_WriteCommand(0xA6); // 正常显示
    }
    if (i == 1) {
        OLED_WriteCommand(0xA7); // 反色显示
    }
}

// 屏幕旋转180度
void OLED_DisplayTurn(uint8_t i) {
    if (i == 0) {
        OLED_WriteCommand(0xC8); // 正常显示
        OLED_WriteCommand(0xA1);
    }
    if (i == 1) {
        OLED_WriteCommand(0xC0); // 反转显示
        OLED_WriteCommand(0xA0);
    }
}

uint8_t *get_glyph(uint32_t utf32, uint8_t *font_width, uint8_t *font_height) {
    if(utf32 < 127) {
        *font_width = 6;
        *font_height = 12;
        return ASCII_6_12[utf32 - 32];
    }
    else {
        uint32_t index = findIndex(GLYPH, utf32);
        if (index >= 0 && index < 7445) {
            *font_width = 12;
            *font_height = 12;
            return GB2312_12_12[index];
        }
        else {
            return NULL;
        }
    }
}

// 渲染一行文本，mode为1则为正显，为0则为反白
void render_line(wchar_t *line, uint32_t x, uint32_t y, uint8_t mode) {
    uint32_t x_pos = x;
    uint32_t y_pos = y;
    for (uint32_t i = 0; i < wcslen(line); i++) {
        uint32_t current_char = line[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        uint8_t *glyph = get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            printf("出现了字库之外的字符！\n");
            break;
        }
        if (x_pos + font_width >= 128) {
            break;
        }
        OLED_ShowChar(x_pos, y_pos, glyph, font_width, font_height, (mode % 2));
        x_pos += font_width;
    }
}

// 返回值：文本折行后的行数（含换行符）
int32_t render_text(wchar_t *text, int32_t line_shift) {
    wchar_t wrapped[STRING_BUFFER_LENGTH];
    int32_t char_count = 0;
    int32_t break_pos[STRING_BUFFER_LENGTH];
    int32_t break_count = 0;
    int32_t line_x_pos = 0;
    for (char_count = 0; char_count < wcslen(text); char_count++) {
        wchar_t ch = text[char_count];
        int32_t char_width = (ch < 127) ? ((ch == '\n') ? 0 : 6) : 12;
        if (line_x_pos + char_width >= 128 || ch == '\n') {
            break_pos[break_count] = char_count;
            break_count++;
            line_x_pos = 0;
        }
        line_x_pos += char_width;
        wrapped[char_count] = ch;
    }
    wrapped[char_count] = 0;
    break_pos[break_count] = char_count; // 最后一个字符视为换行，但不计入break_count

    // 向上卷动 |line_shift| 行，注意line_shift非负。默认为0。

    wchar_t wrapped_clipped[STRING_BUFFER_LENGTH];

    if (line_shift < 0) {
        line_shift = 0;
    }

    int32_t break_from = 0;
    int32_t break_to = 0;

    if (break_count >= 5) {
        if (line_shift < (break_count + 1) - 5) {
            break_from = break_pos[break_count - 5 - line_shift];
            break_to = break_pos[break_count - line_shift];
        }
        else {
            break_from = 0;
            break_to = break_pos[5 - 1];
        }
    }
    else {
        break_from = 0;
        break_to = wcslen(wrapped);
    }

    wcscpy(wrapped_clipped, wrapped + break_from);
    wrapped_clipped[break_to - break_from] = 0;

    int x_pos = 0;
    int y_pos = 0;
    for (int i = 0; i < wcslen(wrapped_clipped); i++) {
        uint32_t current_char = wrapped_clipped[i];
        uint8_t font_width = 12;
        uint8_t font_height = 12;
        if (current_char == '\n') {
            x_pos = 0;
            if(i > 0) y_pos += (font_height + 1);
            continue;
        }
        uint8_t *glyph = get_glyph(current_char, &font_width, &font_height);
        if (!glyph) {
            printf("出现了字库之外的字符[%d]\n", current_char);
            glyph = get_glyph(12307, &font_width, &font_height); // 用字脚符号“〓”代替，参考https://ja.wikipedia.org/wiki/下駄記号
        }
        if (x_pos + font_width >= 128) {
            y_pos += (font_height + 1);
            x_pos = 0;
        }
        OLED_ShowChar(x_pos, y_pos, glyph, font_width, font_height, 1);
        x_pos += font_width;
    }

    // free(wrapped);
    // free(wrapped_clipped);

    return break_count + 1;
}
