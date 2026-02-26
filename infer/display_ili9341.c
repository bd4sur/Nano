/*****************************************************************************
 * Author: Waveshare team
 ******************************************************************************/

#include "platform.h"
#include "display_hal.h"
#include <string.h>
#include <stdlib.h> //itoa()

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/gpio.h>

#include <stdlib.h>
#include <stdio.h>

#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>

#ifdef USE_WIRINGPI_LIB
#include <wiringPi.h>
#include <wiringPiSPI.h>
#elif defined(USE_DEV_LIB)
// #include "sysfs_gpio.h"
// #include "dev_hardware_SPI.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#endif

#include <unistd.h>

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>


#define ILI9341_WIDTH 240  // LCD width
#define ILI9341_HEIGHT 320 // LCD height

#define ILI9341_CS_0 LCD_CS_0
#define ILI9341_CS_1 LCD_CS_1

#define ILI9341_RST_0 LCD_RST_0
#define ILI9341_RST_1 LCD_RST_1

#define ILI9341_DC_0 LCD_DC_0
#define ILI9341_DC_1 LCD_DC_1

#define ILI9341_BL_0 LCD_BL_0
#define ILI9341_BL_1 LCD_BL_1



#define RGB565_WHITE          0xFFFF
#define RGB565_BLACK          0x0000
#define RGB565_BLUE           0x001F
#define RGB565_BRED           0XF81F
#define RGB565_GRED            0XFFE0
#define RGB565_GBLUE           0X07FF
#define RGB565_RED            0xF800
#define RGB565_MAGENTA        0xF81F
#define RGB565_GREEN          0x07E0
#define RGB565_CYAN           0x7FFF
#define RGB565_YELLOW         0xFFE0
#define RGB565_BROWN            0XBC40
#define RGB565_BRRED            0XFC07
#define RGB565_GRAY             0X8430



// ===============================================================================
// GPIO
// ===============================================================================

#define SYSFS_GPIO_IN 0
#define SYSFS_GPIO_OUT 1

#define SYSFS_GPIO_LOW 0
#define SYSFS_GPIO_HIGH 1

#define NUM_MAXBUF 4
#define DIR_MAXSIZ 60

#define GPIO_CHIP_PATH "/dev/gpiochip0"
#define MAX_GPIO_PINS 512

// 全局状态管理
static int gpio_chip_fd = -1;
static int gpio_line_fds[MAX_GPIO_PINS];
static int gpio_initialized = 0;

// 内部辅助函数：初始化映射表
static void init_gpio_map(void)
{
    if (!gpio_initialized)
    {
        for (int i = 0; i < MAX_GPIO_PINS; i++)
        {
            gpio_line_fds[i] = -1;
        }
        gpio_initialized = 1;
    }
}

// 内部辅助函数：获取或打开芯片设备
static int get_chip_fd(void)
{
    if (gpio_chip_fd < 0)
    {
        gpio_chip_fd = open(GPIO_CHIP_PATH, O_RDONLY);
        if (gpio_chip_fd < 0)
        {
            printf("无法打开 GPIO 芯片设备 %s\n", GPIO_CHIP_PATH);
            return -1;
        }
    }
    return gpio_chip_fd;
}

// 内部辅助函数：申请 GPIO 线路
static int request_gpio_line(int pin, int direction)
{
    struct gpiohandle_request req;
    int ret;
    int chip_fd;

    init_gpio_map();
    chip_fd = get_chip_fd();
    if (chip_fd < 0)
        return -1;

    if (pin < 0 || pin >= MAX_GPIO_PINS)
    {
        printf("Pin 编号超出范围：%d\n", pin);
        return -1;
    }

    // 如果已经申请过，先释放（用于 Direction 切换）
    if (gpio_line_fds[pin] >= 0)
    {
        close(gpio_line_fds[pin]);
        gpio_line_fds[pin] = -1;
    }

    memset(&req, 0, sizeof(req));
    req.lineoffsets[0] = pin;
    req.lines = 1;

    // 设置方向标志
    if (direction == SYSFS_GPIO_IN)
    {
        req.flags = GPIOHANDLE_REQUEST_INPUT;
    }
    else
    {
        req.flags = GPIOHANDLE_REQUEST_OUTPUT;
        req.default_values[0] = 0; // 默认输出低电平
    }

    snprintf(req.consumer_label, sizeof(req.consumer_label), "sysfs_compat");

    ret = ioctl(chip_fd, GPIO_GET_LINEHANDLE_IOCTL, &req);
    if (ret < 0)
    {
        printf("申请 GPIO 失败 (Pin%d): %m\n", pin);
        return -1;
    }

    gpio_line_fds[pin] = req.fd;
    return 0;
}

int SYSFS_GPIO_Export(int Pin)
{
    init_gpio_map();

    // 检查是否已经 Export
    if (Pin >= 0 && Pin < MAX_GPIO_PINS && gpio_line_fds[Pin] >= 0)
    {
        printf("Export Failed: Pin%d 已被占用\n", Pin);
        return -1;
    }

    // Character Device 需要在 Export 时申请线路，默认先设为输入（安全）
    // 用户随后调用 Direction 会重新申请
    if (request_gpio_line(Pin, SYSFS_GPIO_IN) < 0)
    {
        return -1;
    }

    printf("Export: Pin%d\r\n", Pin);
    return 0;
}

int SYSFS_GPIO_Unexport(int Pin)
{
    init_gpio_map();

    if (Pin < 0 || Pin >= MAX_GPIO_PINS || gpio_line_fds[Pin] < 0)
    {
        printf("unexport Failed: Pin%d 未导出或无效\n", Pin);
        return -1;
    }

    close(gpio_line_fds[Pin]);
    gpio_line_fds[Pin] = -1;

    printf("Unexport: Pin%d\r\n", Pin);
    return 0;
}

int SYSFS_GPIO_Direction(int Pin, int Dir)
{
    // 检查是否已 Export
    if (Pin < 0 || Pin >= MAX_GPIO_PINS || gpio_line_fds[Pin] < 0)
    {
        printf("Set Direction failed: Pin%d 未 Export\n", Pin);
        return -1;
    }

    // 在 Character Device 中，改变方向需要重新 Request 线路
    // 我们复用 request_gpio_line，它会自动关闭旧的 fd
    if (request_gpio_line(Pin, Dir) < 0)
    {
        return -1;
    }

    if (Dir == SYSFS_GPIO_IN)
    {
        printf("Pin%d:intput\r\n", Pin);
    }
    else
    {
        printf("Pin%d:Output\r\n", Pin);
    }

    return 0;
}

int SYSFS_GPIO_Read(int Pin)
{
    struct gpiohandle_data data;
    int ret;
    int line_fd;

    init_gpio_map();

    if (Pin < 0 || Pin >= MAX_GPIO_PINS || gpio_line_fds[Pin] < 0)
    {
        printf("Read failed Pin%d: 未 Export\n", Pin);
        return -1;
    }

    line_fd = gpio_line_fds[Pin];

    ret = ioctl(line_fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &data);
    if (ret < 0)
    {
        printf("failed to read value! Pin%d\n", Pin);
        return -1;
    }

    return data.values[0];
}

int SYSFS_GPIO_Write(int Pin, int value)
{
    struct gpiohandle_data data;
    int ret;
    int line_fd;

    init_gpio_map();

    if (Pin < 0 || Pin >= MAX_GPIO_PINS || gpio_line_fds[Pin] < 0)
    {
        printf("Write failed : Pin%d,value = %d (未 Export)\n", Pin, value);
        return -1;
    }

    line_fd = gpio_line_fds[Pin];
    data.values[0] = value ? 1 : 0;

    ret = ioctl(line_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, &data);
    if (ret < 0)
    {
        // 如果引脚是输入模式，ioctl 会失败，这符合预期
        printf("failed to write value! Pin%d (可能是输入模式)\n", Pin);
        return -1;
    }

    return 0;
}

// ===============================================================================
// SPI
// ===============================================================================

// #define SPI_CPHA 0x01
// #define SPI_CPOL 0x02
// #define SPI_MODE_0 (0 | 0)
// #define SPI_MODE_1 (0 | SPI_CPHA)
// #define SPI_MODE_2 (SPI_CPOL | 0)
// #define SPI_MODE_3 (SPI_CPOL | SPI_CPHA)

typedef enum
{
    SPI_MODE0 = SPI_MODE_0, /*!< CPOL = 0, CPHA = 0 */
    SPI_MODE1 = SPI_MODE_1, /*!< CPOL = 0, CPHA = 1 */
    SPI_MODE2 = SPI_MODE_2, /*!< CPOL = 1, CPHA = 0 */
    SPI_MODE3 = SPI_MODE_3  /*!< CPOL = 1, CPHA = 1 */
} SPIMode;

typedef enum
{
    DISABLE = 0,
    ENABLE = 1
} SPICSEN;

typedef enum
{
    SPI_CS_Mode_LOW = 0,  /*!< Chip Select 0 */
    SPI_CS_Mode_HIGH = 1, /*!< Chip Select 1 */
    SPI_CS_Mode_NONE = 3  /*!< No CS, control it yourself */
} SPIChipSelect;

typedef enum
{
    SPI_BIT_ORDER_LSBFIRST = 0, /*!< LSB First */
    SPI_BIT_ORDER_MSBFIRST = 1  /*!< MSB First */
} SPIBitOrder;

typedef enum
{
    SPI_3WIRE_Mode = 0,
    SPI_4WIRE_Mode = 1
} BusMode;

/**
 * Define SPI attribute
 **/
typedef struct SPIStruct
{
    // GPIO
    uint16_t SCLK_PIN;
    uint16_t MOSI_PIN;
    uint16_t MISO_PIN;

    uint16_t CS0_PIN;
    uint16_t CS1_PIN;

    uint32_t speed;
    uint16_t mode;
    uint16_t delay;
    int fd; //
} HARDWARE_SPI;



HARDWARE_SPI hardware_SPI;

static uint8_t bits = 8;

// #define SPI_CS_HIGH 0x04   // Chip select high
// #define SPI_LSB_FIRST 0x08 // LSB
// #define SPI_3WIRE 0x10     // 3-wire mode SI and SO same line
// #define SPI_LOOP 0x20      // Loopback mode
// #define SPI_NO_CS 0x40     // A single device occupies one SPI bus, so there is no chip select
// #define SPI_READY 0x80     // Slave pull low to stop data transmission


struct spi_ioc_transfer tr;

/******************************************************************************
function:   Set SPI speed
parameter:
Info:   Return 1 success
        Return -1 failed
******************************************************************************/
int DEV_HARDWARE_SPI_setSpeed(uint32_t speed)
{
    uint32_t speed1 = hardware_SPI.speed;

    hardware_SPI.speed = speed;

    // Write speed
    if (ioctl(hardware_SPI.fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) == -1)
    {
        printf("can't set max speed hz\r\n");
        hardware_SPI.speed = speed1; // Setting failure rate unchanged
        return -1;
    }

    // Read the speed of just writing
    if (ioctl(hardware_SPI.fd, SPI_IOC_RD_MAX_SPEED_HZ, &speed) == -1)
    {
        printf("can't get max speed hz\r\n");
        hardware_SPI.speed = speed1; // Setting failure rate unchanged
        return -1;
    }
    hardware_SPI.speed = speed;
    tr.speed_hz = hardware_SPI.speed;
    return 1;
}

/******************************************************************************
function:   Set SPI Mode
parameter:
Info:
    SPIMode:
        SPI_MODE0
        SPI_MODE1
        SPI_MODE2
        SPI_MODE3
    Return :
        Return 1 success
        Return -1 failed
******************************************************************************/
int DEV_HARDWARE_SPI_Mode(SPIMode mode)
{
    hardware_SPI.mode &= 0xfC; // Clear low 2 digits
    hardware_SPI.mode |= mode; // Setting mode

    // Write device
    if (ioctl(hardware_SPI.fd, SPI_IOC_WR_MODE, &hardware_SPI.mode) == -1)
    {
        printf("DEV_HARDWARE_SPI_Mode can't set spi mode\r\n");
        return -1;
    }
    return 1;
}

/******************************************************************************
function:   Set SPI CS Enable
parameter:
Info:
    EN:
        DISABLE
        ENABLE
    Return :
        Return 1 success
        Return -1 failed
******************************************************************************/
int DEV_HARDWARE_SPI_CSEN(SPICSEN EN)
{
    if (EN == ENABLE)
    {
        hardware_SPI.mode |= SPI_NO_CS;
    }
    else
    {
        hardware_SPI.mode &= ~SPI_NO_CS;
    }
    // Write device
    if (ioctl(hardware_SPI.fd, SPI_IOC_WR_MODE, &hardware_SPI.mode) == -1)
    {
        printf("can't set spi CS EN\r\n");
        return -1;
    }
    return 1;
}

/******************************************************************************
function:   Chip Select
parameter:
Info:
    CS_Mode:
        SPI_CS_Mode_LOW
        SPI_CS_Mode_HIGH
        SPI_CS_Mode_NONE
    Return :
        Return 1 success
        Return -1 failed
******************************************************************************/
int DEV_HARDWARE_SPI_ChipSelect(SPIChipSelect CS_Mode)
{
    if (CS_Mode == SPI_CS_Mode_HIGH)
    {
        hardware_SPI.mode |= SPI_CS_HIGH;
        hardware_SPI.mode &= ~SPI_NO_CS;
        printf("CS HIGH \r\n");
    }
    else if (CS_Mode == SPI_CS_Mode_LOW)
    {
        hardware_SPI.mode &= ~SPI_CS_HIGH;
        hardware_SPI.mode &= ~SPI_NO_CS;
    }
    else if (CS_Mode == SPI_CS_Mode_NONE)
    {
        hardware_SPI.mode |= SPI_NO_CS;
    }

    if (ioctl(hardware_SPI.fd, SPI_IOC_WR_MODE, &hardware_SPI.mode) == -1)
    {
        printf("DEV_HARDWARE_SPI_ChipSelect can't set spi mode\r\n");
        return -1;
    }
    return 1;
}

/******************************************************************************
function:   Sets the SPI bit order
parameter:
Info:
    Order:
        SPI_BIT_ORDER_LSBFIRST
        SPI_BIT_ORDER_MSBFIRST
    Return :
        Return 1 success
        Return -1 failed
******************************************************************************/
int DEV_HARDWARE_SPI_SetBitOrder(SPIBitOrder Order)
{
    if (Order == SPI_BIT_ORDER_LSBFIRST)
    {
        hardware_SPI.mode |= SPI_LSB_FIRST;
        printf("SPI_LSB_FIRST\r\n");
    }
    else if (Order == SPI_BIT_ORDER_MSBFIRST)
    {
        hardware_SPI.mode &= ~SPI_LSB_FIRST;
        printf("SPI_MSB_FIRST\r\n");
    }

    // printf("hardware_SPI.mode = 0x%02x\r\n", hardware_SPI.mode);
    int ret = ioctl(hardware_SPI.fd, SPI_IOC_WR_MODE, &hardware_SPI.mode);
    if (ret == -1)
    {
        printf("can't set spi SPI_LSB_FIRST\r\n");
        return -1;
    }
    return 1;
}

/******************************************************************************
function:   Sets the SPI Bus Mode
parameter:
Info:
    Order:
        SPI_3WIRE_Mode
        SPI_4WIRE_Mode
    Return :
        Return 1 success
        Return -1 failed
******************************************************************************/
int DEV_HARDWARE_SPI_SetBusMode(BusMode mode)
{
    if (mode == SPI_3WIRE_Mode)
    {
        hardware_SPI.mode |= SPI_3WIRE;
    }
    else if (mode == SPI_4WIRE_Mode)
    {
        hardware_SPI.mode &= ~SPI_3WIRE;
    }
    if (ioctl(hardware_SPI.fd, SPI_IOC_WR_MODE, &hardware_SPI.mode) == -1)
    {
        printf("can't set spi mode\r\n");
        return -1;
    }
    return 1;
}

/******************************************************************************
function:
    Time interval after transmission of one byte during continuous transmission
parameter:
    us :   Interval time (us)
Info:
******************************************************************************/
void DEV_HARDWARE_SPI_SetDataInterval(uint16_t us)
{
    hardware_SPI.delay = us;
    tr.delay_usecs = hardware_SPI.delay;
}

/******************************************************************************
function: SPI port sends one byte of data
parameter:
    buf :   Sent data
Info:
******************************************************************************/
uint8_t DEV_HARDWARE_SPI_TransferByte(uint8_t buf)
{
    uint8_t rbuf[1];
    tr.len = 1;
    tr.tx_buf = (unsigned long)&buf;
    tr.rx_buf = (unsigned long)rbuf;

    // ioctl Operation, transmission of data
    if (ioctl(hardware_SPI.fd, SPI_IOC_MESSAGE(1), &tr) < 1)
        printf("can't send spi message\r\n");
    return rbuf[0];
}

/******************************************************************************
function: The SPI port reads a byte
parameter:
Info: Return read data
******************************************************************************/
int DEV_HARDWARE_SPI_Transfer(uint8_t *buf, uint32_t len)
{
    tr.len = len;
    tr.tx_buf = (unsigned long)buf;
    tr.rx_buf = (unsigned long)buf;

    // ioctl Operation, transmission of data
    if (ioctl(hardware_SPI.fd, SPI_IOC_MESSAGE(1), &tr) < 1)
    {
        printf("can't send spi message\r\n");
        return -1;
    }

    return 1;
}

/******************************************************************************
function:   SPI port initialization
parameter:
    SPI_device : Device name
Info:
    /dev/spidev0.0
    /dev/spidev0.1
******************************************************************************/
void DEV_HARDWARE_SPI_begin(char *SPI_device)
{
    // device
    int ret = 0;
    printf("try to open : %s\r\n", SPI_device);
    if ((hardware_SPI.fd = open(SPI_device, O_RDWR)) < 0)
    {
        printf("Failed to open SPI device\r\n");
        exit(1);
    }
    else
    {
        printf("open : %s\r\n", SPI_device);
    }
    printf("SPI fd : %d\r\n", hardware_SPI.fd);
    hardware_SPI.mode = 0;

    DEV_HARDWARE_SPI_Mode(SPI_MODE_0);
    // ret = ioctl(hardware_SPI.fd, SPI_IOC_WR_MODE, &hardware_SPI.mode);
    // if (ret == -1) {
    //     printf("can't set spi mode\r\n");
    // }

    ret = ioctl(hardware_SPI.fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
    if (ret == -1)
    {
        printf("can't set bits per word\r\n");
    }

    ret = ioctl(hardware_SPI.fd, SPI_IOC_RD_BITS_PER_WORD, &bits);
    if (ret == -1)
    {
        printf("can't get bits per word\r\n");
    }

    int speed = 80000000;
    ret = ioctl(hardware_SPI.fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed);
    if (ret == -1)
    {
        printf("can't SPI_IOC_WR_MAX_SPEED_HZ\r\n");
    }

    tr.bits_per_word = bits;

    DEV_HARDWARE_SPI_ChipSelect(SPI_CS_Mode_LOW);
    // DEV_HARDWARE_SPI_ChipSelect(SPI_CS_Mode_NONE);
    // DEV_HARDWARE_SPI_SetBitOrder(SPI_BIT_ORDER_LSBFIRST);
    // DEV_HARDWARE_SPI_setSpeed(80000000);
    DEV_HARDWARE_SPI_SetDataInterval(0);
}


/******************************************************************************
function:   SPI device End
parameter:
Info:
******************************************************************************/
void DEV_HARDWARE_SPI_end(void)
{
    hardware_SPI.mode = 0;
    if (close(hardware_SPI.fd) != 0)
    {
        printf("Failed to close SPI device\r\n");
        perror("Failed to close SPI device.\n");
    }
}














// ===============================================================================
// Dev API
// ===============================================================================


#define LCD_CS 110
#define LCD_RST 36
#define LCD_DC 39
#define LCD_BL 40



/*****************************************
                GPIO
*****************************************/
void DEV_Digital_Write(uint16_t Pin, uint8_t Value)
{
#ifdef USE_WIRINGPI_LIB
    digitalWrite(Pin, Value);

#elif defined(USE_DEV_LIB)
    SYSFS_GPIO_Write(Pin, Value);

#endif
}

uint8_t DEV_Digital_Read(uint16_t Pin)
{
    uint8_t Read_value = 0;

#ifdef USE_WIRINGPI_LIB
    Read_value = digitalRead(Pin);

#elif defined(USE_DEV_LIB)
    Read_value = SYSFS_GPIO_Read(Pin);
#endif
    return Read_value;
}

void DEV_GPIO_Mode(uint16_t Pin, uint16_t Mode)
{
#ifdef USE_WIRINGPI_LIB
    if (Mode == 0 || Mode == INPUT)
    {
        pinMode(Pin, INPUT);
        pullUpDnControl(Pin, PUD_UP);
    }
    else
    {
        pinMode(Pin, OUTPUT);
        // printf (" %d OUT \r\n",Pin);
    }
#elif defined(USE_DEV_LIB)
    SYSFS_GPIO_Export(Pin);
    if (Mode == 0 || Mode == SYSFS_GPIO_IN)
    {
        SYSFS_GPIO_Direction(Pin, SYSFS_GPIO_IN);
        printf("IN Pin = %d\r\n",Pin);
    }
    else
    {
        SYSFS_GPIO_Direction(Pin, SYSFS_GPIO_OUT);
        printf("OUT Pin = %d\r\n",Pin);
    }
#endif
}

/**
 * delay x ms
 **/
void DEV_Delay_ms(uint32_t xms)
{
#ifdef USE_WIRINGPI_LIB
    delay(xms);
#elif defined(USE_DEV_LIB)
    uint32_t i;
    for (i = 0; i < xms; i++)
    {
        usleep(1000);
    }
#endif
}

static void DEV_GPIO_Init(void)
{
    DEV_GPIO_Mode(LCD_CS, 1);
    DEV_GPIO_Mode(LCD_RST, 1);
    DEV_GPIO_Mode(LCD_DC, 1);
    DEV_GPIO_Mode(LCD_BL, 1);

    DEV_Digital_Write(LCD_CS, 1);
    DEV_Digital_Write(LCD_BL, 1);
}

/******************************************************************************
function:    Module Initialize, the library and initialize the pins, SPI protocol
parameter:
Info:
******************************************************************************/
uint8_t DEV_ModuleInit(void)
{

#ifdef USE_WIRINGPI_LIB
    // if(wiringPiSetup() < 0)//use wiringpi Pin number table
    if (wiringPiSetupGpio() < 0)
    { // use BCM2835 Pin number table
        printf("set wiringPi lib failed    !!! \r\n");
        return 1;
    }
    else
    {
        printf("set wiringPi lib success  !!! \r\n");
    }
    DEV_GPIO_Init();
    wiringPiSPISetup(0, 80000000);
    pinMode(LCD_BL, PWM_OUTPUT);
    pwmWrite(LCD_BL, 1023);
#elif defined(USE_DEV_LIB)
    DEV_GPIO_Init();
    DEV_HARDWARE_SPI_begin("/dev/spidev1.0");
#endif
    return 0;
}

void DEV_SPI_WriteByte(uint8_t Value)
{

#ifdef USE_WIRINGPI_LIB
    wiringPiSPIDataRW(0, &Value, 1);

#elif defined(USE_DEV_LIB)
    DEV_HARDWARE_SPI_TransferByte(Value);

#endif
}

void DEV_SPI_Write_nByte(uint8_t *pData, uint32_t Len)
{
#ifdef USE_WIRINGPI_LIB
    wiringPiSPIDataRW(0, (unsigned char *)pData, Len);

#elif defined(USE_DEV_LIB)
    DEV_HARDWARE_SPI_Transfer(pData, Len);

#endif
}

/******************************************************************************
function:    Module exits, closes SPI and BCM2835 library
parameter:
Info:
******************************************************************************/
void DEV_ModuleExit(void)
{
#ifdef USE_WIRINGPI_LIB

#elif defined(USE_DEV_LIB)

    DEV_HARDWARE_SPI_end();
#ifdef USE_DEV_LIB_PWM
    pthread_cancel(t1);
#endif
    DEV_Digital_Write(LCD_BL, 1);
#endif
}



















// ===============================================================================
// ILI9341
// ===============================================================================


// LCD
#define LCD_CS_0 DEV_Digital_Write(LCD_CS, 0)
#define LCD_CS_1 DEV_Digital_Write(LCD_CS, 1)

#define LCD_RST_0 DEV_Digital_Write(LCD_RST, 0)
#define LCD_RST_1 DEV_Digital_Write(LCD_RST, 1)

#define LCD_DC_0 DEV_Digital_Write(LCD_DC, 0)
#define LCD_DC_1 DEV_Digital_Write(LCD_DC, 1)

#define LCD_BL_0 DEV_Digital_Write(LCD_BL, 0)
#define LCD_BL_1 DEV_Digital_Write(LCD_BL, 1)






/*******************************************************************************
function:
    Hardware reset
*******************************************************************************/
static void ILI9341_Reset(void)
{
    DEV_Digital_Write(LCD_CS, 1);
    DEV_Delay_ms(100);
    DEV_Digital_Write(LCD_RST, 0);
    DEV_Delay_ms(100);
    DEV_Digital_Write(LCD_RST, 1);
    DEV_Delay_ms(100);
}

/*******************************************************************************
function:
        Write data and commands
*******************************************************************************/
static void ILI9341_Write_Command(uint8_t data)
{
    DEV_Digital_Write(LCD_CS, 0);
    DEV_Digital_Write(LCD_DC, 0);
    DEV_SPI_WriteByte(data);
}

static void ILI9341_WriteData_Byte(uint8_t data)
{
    DEV_Digital_Write(LCD_CS, 0);
    DEV_Digital_Write(LCD_DC, 1);
    DEV_SPI_WriteByte(data);
    DEV_Digital_Write(LCD_CS, 1);
}

void ILI9341_WriteData_Word(uint16_t data)
{
    DEV_Digital_Write(LCD_CS, 0);
    DEV_Digital_Write(LCD_DC, 1);
    DEV_SPI_WriteByte((data >> 8) & 0xff);
    DEV_SPI_WriteByte(data);
    DEV_Digital_Write(LCD_CS, 1);
}

/******************************************************************************
function:
        Common register initialization
******************************************************************************/
void ILI9341_Init(void)
{
    ILI9341_Reset();

    ILI9341_Write_Command(0x11); // Sleep out

    ILI9341_Write_Command(0xCF);
    ILI9341_WriteData_Byte(0x00);
    ILI9341_WriteData_Byte(0xC1);
    ILI9341_WriteData_Byte(0X30);
    ILI9341_Write_Command(0xED);
    ILI9341_WriteData_Byte(0x64);
    ILI9341_WriteData_Byte(0x03);
    ILI9341_WriteData_Byte(0X12);
    ILI9341_WriteData_Byte(0X81);
    ILI9341_Write_Command(0xE8);
    ILI9341_WriteData_Byte(0x85);
    ILI9341_WriteData_Byte(0x00);
    ILI9341_WriteData_Byte(0x79);
    ILI9341_Write_Command(0xCB);
    ILI9341_WriteData_Byte(0x39);
    ILI9341_WriteData_Byte(0x2C);
    ILI9341_WriteData_Byte(0x00);
    ILI9341_WriteData_Byte(0x34);
    ILI9341_WriteData_Byte(0x02);
    ILI9341_Write_Command(0xF7);
    ILI9341_WriteData_Byte(0x20);
    ILI9341_Write_Command(0xEA);
    ILI9341_WriteData_Byte(0x00);
    ILI9341_WriteData_Byte(0x00);
    ILI9341_Write_Command(0xC0);  // Power control
    ILI9341_WriteData_Byte(0x1D); // VRH[5:0]
    ILI9341_Write_Command(0xC1);  // Power control
    ILI9341_WriteData_Byte(0x12); // SAP[2:0];BT[3:0]
    ILI9341_Write_Command(0xC5);  // VCM control
    ILI9341_WriteData_Byte(0x33);
    ILI9341_WriteData_Byte(0x3F);
    ILI9341_Write_Command(0xC7); // VCM control
    ILI9341_WriteData_Byte(0x92);
    ILI9341_Write_Command(0x3A); // Memory Access Control
    ILI9341_WriteData_Byte(0x55);
    ILI9341_Write_Command(0x36); // Memory Access Control
    ILI9341_WriteData_Byte(0x08);
    ILI9341_Write_Command(0xB1);
    ILI9341_WriteData_Byte(0x00);
    ILI9341_WriteData_Byte(0x12);
    ILI9341_Write_Command(0xB6); // Display Function Control
    ILI9341_WriteData_Byte(0x0A);
    ILI9341_WriteData_Byte(0xA2);

    ILI9341_Write_Command(0x44);
    ILI9341_WriteData_Byte(0x02);

    ILI9341_Write_Command(0xF2); // 3Gamma Function Disable
    ILI9341_WriteData_Byte(0x00);
    ILI9341_Write_Command(0x26); // Gamma curve selected
    ILI9341_WriteData_Byte(0x01);
    ILI9341_Write_Command(0xE0); // Set Gamma
    ILI9341_WriteData_Byte(0x0F);
    ILI9341_WriteData_Byte(0x22);
    ILI9341_WriteData_Byte(0x1C);
    ILI9341_WriteData_Byte(0x1B);
    ILI9341_WriteData_Byte(0x08);
    ILI9341_WriteData_Byte(0x0F);
    ILI9341_WriteData_Byte(0x48);
    ILI9341_WriteData_Byte(0xB8);
    ILI9341_WriteData_Byte(0x34);
    ILI9341_WriteData_Byte(0x05);
    ILI9341_WriteData_Byte(0x0C);
    ILI9341_WriteData_Byte(0x09);
    ILI9341_WriteData_Byte(0x0F);
    ILI9341_WriteData_Byte(0x07);
    ILI9341_WriteData_Byte(0x00);
    ILI9341_Write_Command(0XE1); // Set Gamma
    ILI9341_WriteData_Byte(0x00);
    ILI9341_WriteData_Byte(0x23);
    ILI9341_WriteData_Byte(0x24);
    ILI9341_WriteData_Byte(0x07);
    ILI9341_WriteData_Byte(0x10);
    ILI9341_WriteData_Byte(0x07);
    ILI9341_WriteData_Byte(0x38);
    ILI9341_WriteData_Byte(0x47);
    ILI9341_WriteData_Byte(0x4B);
    ILI9341_WriteData_Byte(0x0A);
    ILI9341_WriteData_Byte(0x13);
    ILI9341_WriteData_Byte(0x06);
    ILI9341_WriteData_Byte(0x30);
    ILI9341_WriteData_Byte(0x38);
    ILI9341_WriteData_Byte(0x0F);
    ILI9341_Write_Command(0x29); // Display on
}

/******************************************************************************
function:    Set the cursor position
parameter    :
      Xstart:     Start uint16_t x coordinate
      Ystart:    Start uint16_t y coordinate
      Xend  :    End uint16_t coordinates
      Yend  :    End uint16_t coordinatesen
******************************************************************************/
void ILI9341_SetWindow(uint16_t Xstart, uint16_t Ystart, uint16_t Xend, uint16_t Yend)
{
    ILI9341_Write_Command(0x2a);
    ILI9341_WriteData_Byte(Xstart >> 8);
    ILI9341_WriteData_Byte(Xstart & 0xff);
    ILI9341_WriteData_Byte((Xend - 1) >> 8);
    ILI9341_WriteData_Byte((Xend - 1) & 0xff);

    ILI9341_Write_Command(0x2b);
    ILI9341_WriteData_Byte(Ystart >> 8);
    ILI9341_WriteData_Byte(Ystart & 0xff);
    ILI9341_WriteData_Byte((Yend - 1) >> 8);
    ILI9341_WriteData_Byte((Yend - 1) & 0xff);

    ILI9341_Write_Command(0x2C);
}

/******************************************************************************
function:    Settings window
parameter    :
      Xstart:     Start uint16_t x coordinate
      Ystart:    Start uint16_t y coordinate

******************************************************************************/
void ILI9341_SetCursor(uint16_t X, uint16_t Y)
{
    ILI9341_Write_Command(0x2a);
    ILI9341_WriteData_Byte(X >> 8);
    ILI9341_WriteData_Byte(X);
    ILI9341_WriteData_Byte(X >> 8);
    ILI9341_WriteData_Byte(X);

    ILI9341_Write_Command(0x2b);
    ILI9341_WriteData_Byte(Y >> 8);
    ILI9341_WriteData_Byte(Y);
    ILI9341_WriteData_Byte(Y >> 8);
    ILI9341_WriteData_Byte(Y);

    ILI9341_Write_Command(0x2C);
}

/******************************************************************************
function:    Clear screen function, refresh the screen to a certain color
parameter    :
      Color :        The color you want to clear all the screen
******************************************************************************/
void ILI9341_Clear(uint16_t Color)
{
    uint16_t i;
    uint16_t image[ILI9341_WIDTH];
    for (i = 0; i < ILI9341_WIDTH; i++) {
        image[i] = Color >> 8 | (Color & 0xff) << 8;
    }
    ILI9341_SetWindow(0, 0, ILI9341_WIDTH, ILI9341_HEIGHT);
    DEV_Digital_Write(LCD_DC, 1);
    for (i = 0; i < ILI9341_HEIGHT; i++) {
        DEV_SPI_Write_nByte((uint8_t *)(image), ILI9341_WIDTH * 2);
    }
}

/******************************************************************************
function:    Refresh a certain area to the same color
parameter    :
      Xstart: Start uint16_t x coordinate
      Ystart:    Start uint16_t y coordinate
      Xend  :    End uint16_t coordinates
      Yend  :    End uint16_t coordinates
      color :    Set the color
******************************************************************************/
void ILI9341_ClearWindow(uint16_t Xstart, uint16_t Ystart, uint16_t Xend, uint16_t Yend, uint16_t color)
{
    uint16_t i, j;
    ILI9341_SetWindow(Xstart, Ystart, Xend - 1, Yend - 1);
    for (i = Ystart; i <= Yend - 1; i++)
    {
        for (j = Xstart; j <= Xend - 1; j++)
        {
            ILI9341_WriteData_Word(color);
        }
    }
}

/******************************************************************************
function: Show a picture
parameter    :
        image: Picture buffer
******************************************************************************/
void ILI9341_Display(uint8_t *image)
{
    uint16_t i;
    ILI9341_SetWindow(0, 0, ILI9341_WIDTH, ILI9341_HEIGHT);
    DEV_Digital_Write(LCD_DC, 1);
    for (i = 0; i < ILI9341_HEIGHT; i++)
    {
        DEV_SPI_Write_nByte((uint8_t *)image + ILI9341_WIDTH * 2 * i, ILI9341_WIDTH * 2);
    }
}





void ILI9341_SetBacklight(uint16_t Value)
{
#ifdef USE_WIRINGPI_LIB
    pwmWrite(LCD_BL, Value);
#elif defined(USE_DEV_LIB)
    DEV_Digital_Write(LCD_BL, 1);
#endif
}


void Handler_2IN4_LCD(int signo)
{
    // System Exit
    printf("\r\nHandler:Program stop\r\n");
    DEV_ModuleExit();
    exit(0);
}

// MADCTL 寄存器位定义
#define MADCTL_MY  0x80  // 垂直翻转
#define MADCTL_MX  0x40  // 水平翻转
#define MADCTL_MV  0x20  // 行列交换（旋转90度的关键）
#define MADCTL_ML  0x10  // 垂直刷新顺序
#define MADCTL_RGB 0x00  // RGB 顺序
#define MADCTL_BGR 0x08  // BGR 顺序

// 设置旋转角度
void ILI9341_SetRotation(uint8_t rotation) {
    uint8_t madctl = 0;
    
    switch(rotation) {
        case 0: // 0度
            madctl = MADCTL_MX | MADCTL_BGR;
            break;
        case 1: // 90度
            madctl = MADCTL_MV | MADCTL_BGR;
            break;
        case 2: // 180度
            madctl = MADCTL_MY | MADCTL_BGR;
            break;
        case 3: // 270度
            madctl = MADCTL_MX | MADCTL_MY | MADCTL_MV | MADCTL_BGR;
            break;
    }
    
    ILI9341_Write_Command(0x36);  // MADCTL
    ILI9341_WriteData_Byte(madctl);
}








// === RGB888 → RGB565 转换函数（高效版）===
static inline uint16_t rgb8888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    uint16_t color = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
    color = ((color<<8)&0xff00)|(color>>8);
    return color;
}

void convert_rgb888_to_rgb565(const uint8_t *rgb888, uint16_t *rgb565, int width, int height) {
    const uint8_t *p = rgb888;
    uint16_t *out = rgb565;
    for (int i = 0; i < width * height; i++)
    {
        uint8_t r = p[0];
        uint8_t g = p[1];
        uint8_t b = p[2];
        *out++ = rgb8888_to_rgb565(r, g, b);
        p += 3;
    }
}

static uint16_t *g_frame_buffer_rgb565 = NULL; // RGB565, size = W*H

void display_hal_refresh(uint8_t *frame_buffer_rgb888, uint32_t fb_width, uint32_t fb_height,
    uint32_t x0, uint32_t y0, uint32_t view_width, uint32_t view_height
) {
    memset(g_frame_buffer_rgb565, 0, ILI9341_WIDTH * ILI9341_HEIGHT * sizeof(uint16_t));
    convert_rgb888_to_rgb565(frame_buffer_rgb888, g_frame_buffer_rgb565, fb_width, fb_height);
    ILI9341_Display((uint8_t *)g_frame_buffer_rgb565);
}


void display_hal_init(void) {
    g_frame_buffer_rgb565 = (uint16_t *)malloc(ILI9341_WIDTH * ILI9341_HEIGHT * sizeof(uint16_t));
    memset(g_frame_buffer_rgb565, 0, ILI9341_WIDTH * ILI9341_HEIGHT * sizeof(uint16_t));

    if (DEV_ModuleInit() != 0) {
        DEV_ModuleExit();
        exit(0);
    }

    ILI9341_Init();
    ILI9341_SetBacklight(1023);
    // ILI9341_SetRotation(1);
    ILI9341_Clear(RGB565_GREEN);
    DEV_Delay_ms(1000);
}


void display_hal_close(void) {
    free(g_frame_buffer_rgb565);
    g_frame_buffer_rgb565 = NULL;
    DEV_ModuleExit();
}