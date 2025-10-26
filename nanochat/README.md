# Qwen3树莓派部署和使用指南

![ ](../doc/overview.jpg)

本文简要介绍了将Qwen3部署在树莓派上进行推理的方法。

## 硬件准备

准备以下材料：

- 树莓派5代：内存4GB或以上，建议16GB。建议加装官方主动散热器。
- microSD卡：存储树莓派操作系统和语言模型，建议不小于16GB。
- 电源：建议使用树莓派官方5V5A电源，以免输入功率不足导致性能下降。
- OLED屏幕：基于SSD1309芯片的128x64点阵OLED显示屏，I2C接口。
- 矩阵键盘：I2C接口。
- 杜邦线、跳线等线缆若干。
- 显示器、键盘、鼠标，用于直接在树莓派上操作；或者准备一台电脑，通过SSH和SFTP以“无头”方式远程操作树莓派。

按照以下图示，使用杜邦线等线缆连接各个模块：

![ ](../doc/blocks.png)


上面图示中的I2C总线信号线，可以按照下图的方式制作，将同名信号线、VCC=3V3、GND连接在一起，形成一个三端的总线结构，分别连接树莓派、OLED屏幕和矩阵键盘：

![ ](../doc/bus_wire.jpg)


注意：切勿带电插拔模块。切勿接反或短路电源线和地线。避免导电物体接触裸露的电子模块，以防意外短路。建议操作前先释放身上的静电，或者戴静电手环操作。

## 软件准备

**第一步：环境配置**

首先，按照[树莓派官方文档](https://www.raspberrypi.com/documentation/computers/getting-started.html)的说明，在电脑上下载树莓派系统烧录工具，将 Raspberry Pi OS (64-bit) 烧录进microSD卡。

随后，将显示器、键盘、网线连接到树莓派，将烧录了操作系统的microSD卡插入插槽，确保所有模块连接正确后，插入电源，树莓派应能自动启动。按照[树莓派官方文档](https://www.raspberrypi.com/documentation/computers/getting-started.html)的说明，完成网络、账户密码等配置（**下文使用的用户名为`pi`**），进入 Raspberry Pi OS。

打开终端，执行`gcc --version`，如果没有报错，则意味着编译工具链已成功安装，进入第二步。否则，执行以下命令，更新并安装必要软件：

```
sudo apt update
sudo apt install git build-essential
```

**第二步：启用并设置I2C端口**

打开终端，执行

```
sudo nano /boot/firmware/config.txt
```

编辑器打开后，在config文件中，将`dtparam=i2c_arm=off`这一行改成以下内容，以启用I2C端口，并将其速率设置为400kHz：

```
dtparam=i2c_arm=on,i2c_arm_baudrate=400000
```

保存并退出，随后执行`sudo reboot`重启树莓派。

重启之后，执行以下命令，检查能否正确识别OLED屏幕和矩阵键盘两个设备：

```
sudo i2cdetect 1 -y
```

如果显示的内容中有27和3c如下，说明树莓派已经识别到了两个I2C设备，其中0x27是矩阵键盘，0x3c是OLED屏幕。

```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- 27 -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- 3c -- -- --
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- --
```

**第三步：拉取代码并编译**

首先，拉取代码仓库到本地：

```
cd /home/pi
git clone https://github.com/bd4sur/Nano.git
cd Nano/nanochat
```

然后，从[HuggingFace](https://huggingface.co/bd4sur/Qwen3)或者[ModelScope]()下载转换好的模型文件，并将其放置于`model`目录下。

```
cd model

# 从HuggingFace下载
wget -c https://huggingface.co/bd4sur/Qwen3/resolve/main/qwen3-0b6-q80.bin
wget -c https://huggingface.co/bd4sur/Qwen3/resolve/main/qwen3-1b7-q80.bin
wget -c https://huggingface.co/bd4sur/Qwen3/resolve/main/qwen3-4b-instruct-2507-q80.bin
wget -c https://huggingface.co/bd4sur/Nano-168M/resolve/main/nano_168m_625000_sft_947000_q80.bin

# 或者从ModelScope下载

wget -c https://modelscope.cn/models/bd4sur/qwen3_nano/resolve/master/qwen3-0b6-q80.bin
wget -c https://modelscope.cn/models/bd4sur/qwen3_nano/resolve/master/qwen3-1b7-q80.bin
wget -c https://modelscope.cn/models/bd4sur/qwen3_nano/resolve/master/qwen3-4b-instruct-2507-q80.bin
wget -c https://modelscope.cn/models/bd4sur/Nano-168M/resolve/master/nano_168m_625000_sft_947000_q80.bin
```

模型下载完成后，返回上一级目录，将代码编译为可执行文件：

```
cd ..
make -j4
```

编译完成后，在当前目录中会出现一个新的可执行文件`nanochat`，执行它：

```
./nanochat
```

如果一切正常，OLED屏幕亮起，可以开始与电子鹦鹉对话啦。

## 使用方法

首先介绍16键矩阵键盘的功能。

|1|2|3|4|
|--|--|--|--|
|1<br>英文符号|2<br>ABC|3<br>DEF|A<br>返回/退格|
|4<br>GHI|5<br>JKL|6<br>MNO|B<br>汉英数切换|
|7<br>PQRS|8<br>TUV|9<br>WXYZ|C<br>无功能|
|*<br>向上|0<br>符号|#<br>向下|D<br>确认/输入|

程序启动，首先显示主屏幕（图1）。在主屏幕中，按任意键，进入主菜单（图2）。在菜单中，按【*】和【#】键移动光标，按【D】键确认。

选择“电子鹦鹉”选项，进入模型选择菜单（图3），选择所需的模型，待模型加载完毕后（模型加载需要几秒到几十秒的时间，具体因模型的尺寸而异），进入文字输入状态（图4）。

![ ](../doc/screenshots.jpg)


在文字输入状态（图4）下，按【*】和【#】键移动光标，按【A】键删除光标左侧的1个字符，按【B】键切换汉字/英文字母/数字输入状态，按【D】键确认输入。如果输入框内没有内容，则按【A】键会返回主菜单。

汉字输入状态，类似于手机的九键拼音输入法。例如，要输入“你”字，依次按【6】键（mno）和【4】键【ghi】，随着按键输入，屏幕最下方会出现已输入的按键组合所对应的全部候选字（图5）。若拼音输入完毕，按下【D】键，开始选字，此时在候选字列表上方会出现一行数字（图6），直接按下对应的数字键，即可选中并输入相应的数字。数字上方的（1/5）是候选字列表的页码，按【*】和【#】键可以向前向后翻页，查看更多候选字。在拼音输入的任何阶段，按【A】键都会退出拼音输入状态，回到文字输入状态（图4）。

英文字母输入状态，类似于传统的T9英文输入法。例如，要输入字母“d”，则按【2】键（def），屏幕下方会出现这个按键对应的候选字母（图7），同时出现一个倒计时进度条。反复按同一个键，光标向右滚动，直至停留在想要的字母上，停止按键，待倒计时进度掉读完，则选中的字母被输入。【1】键对应的是常用的英文符号，输入方法与普通的字母按键一致。

数字输入状态，按下某个数字键，直接输入对应的数字。

无论在哪种输入状态，长按【0】键，都会呼出符号候选列表。按【*】和【#】键可以向前向后翻页，按数字键，可选中并输入对应的符号。

文字输入完成后，按【D】键确认输入，此时屏幕上显示“Pre-filling...”和进度条，意味着模型推理引擎正在逐词读取输入内容。读取完毕后，进入解码阶段，此时屏幕上开始显示大模型的回答内容，同时自动翻页到最底部。

待大模型回答完毕后，屏幕底部显示本次对话的生成速度。此时，按【*】和【#】键可以向上向下翻页，查看全部回答内容，每按1次滚动1行，滚动到顶部或底部时可自动返回最底部或者最顶部。按【A】键，返回到文字输入状态。按【D】键，可以再次询问刚刚问过的问题。
