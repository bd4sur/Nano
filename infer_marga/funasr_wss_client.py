# -*- encoding: utf-8 -*-
"""
# FunASR服务启动方式

## 原始镜像

sudo docker run -p 10096:10095 -it --rm --privileged=true --name funasr \
--volume /home/bd4sur/ai/_model/FunASR:/workspace/models \
--workdir /workspace/FunASR/runtime \
registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12

/bin/bash run_server_2pass.sh \
--download-model-dir /workspace/models \
--vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
--model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
--online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
--punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
--itn-dir thuduj12/fst_itn_zh \
--hotword /workspace/models/hotwords.txt \
--certfile 0

## 安装了ffmpeg的镜像，一行命令启动

sudo docker run -p 10096:10095 -d --privileged=true --name funasr \
--restart=always \
--volume /home/bd4sur/ai/_model/FunASR:/workspace/models \
--workdir /workspace/FunASR/runtime \
funasr-online-cpu-0.1.12-20250820:latest \
/bin/bash -c "/workspace/FunASR/runtime/start_2pass.sh \
--download-model-dir /workspace/models \
--hotword /workspace/models/hotwords.txt \
--certfile 0"

## 客户端启动命令

python funasr_wss_client.py --host "0.0.0.0" --port 10096 --mode 2pass --chunk_size "5,10,5" --ssl 0

## 解决作为系统服务启动时找不到默认音频设备的问题

sudo nano /etc/asound.conf
添加：defaults.pcm.card 2

"""
import os
import errno
import websockets, ssl
import asyncio
import argparse
import json
from multiprocessing import Process
import pyaudio

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="localhost",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10096,
                    required=False,
                    help="grpc server port")
parser.add_argument("--chunk_size",
                    type=str,
                    default="5, 10, 5",
                    help="chunk")
parser.add_argument("--chunk_interval",
                    type=int,
                    default=10,
                    help="chunk")
parser.add_argument("--hotword",
                    type=str,
                    default="",
                    help="hotword file path, one hotword perline (e.g.:阿里巴巴 20)")
parser.add_argument("--words_max_print",
                    type=int,
                    default=10000,
                    help="chunk")
parser.add_argument("--ssl",
                    type=int,
                    default=1,
                    help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--use_itn",
                    type=int,
                    default=1,
                    help="1 for using itn, 0 for not itn")
parser.add_argument("--mode",
                    type=str,
                    default="2pass",
                    help="offline, online, 2pass")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]

g_offline_msg_done=False

g_output_text = ""
g_text_2pass_online = ""
g_text_2pass_offline = ""

PTT_STATUS = False

# 进程间通信的两个fifo
ASR_FIFO_PATH = "/tmp/asr_fifo"
ASR_FIFO_FD = None
PTT_FIFO_PATH = "/tmp/ptt_fifo"
PTT_FIFO_FD = None

def create_all_fifo():
    try:
        if os.path.exists(ASR_FIFO_PATH):
            os.unlink(ASR_FIFO_PATH)
        os.mkfifo(ASR_FIFO_PATH)
        print(f"命名管道 {ASR_FIFO_PATH} 创建成功")

        if os.path.exists(PTT_FIFO_PATH):
            os.unlink(PTT_FIFO_PATH)
        os.mkfifo(PTT_FIFO_PATH)
        print(f"命名管道 {PTT_FIFO_PATH} 创建成功")

        return True

    except OSError as e:
        print(f"创建命名管道失败: {e}")
        return False

def open_asr_fifo():
    global ASR_FIFO_FD
    try:
        if ASR_FIFO_FD:
            try:
                os.close(ASR_FIFO_FD)
            except:
                pass
        ASR_FIFO_FD = os.open(ASR_FIFO_PATH, os.O_WRONLY | os.O_NONBLOCK)
        print("ASR管道已打开（非阻塞模式）")
        return True
    except OSError as e:
        print("ASR管道打开失败")
        ASR_FIFO_FD = None
        return False

def write_text_to_asr_fifo(text):
    global ASR_FIFO_FD
    try:
        if not ASR_FIFO_FD:
            if not open_asr_fifo():
                return False
        os.write(ASR_FIFO_FD, text.encode())
        return True
    except OSError as e:
        if ASR_FIFO_FD:
            try:
                os.close(ASR_FIFO_FD)
            except:
                pass
            ASR_FIFO_FD = None
        return False

async def check_ptt_status():
    global PTT_STATUS

    while True:
        try:
            # 以非阻塞方式打开FIFO进行读取
            fd = os.open(PTT_FIFO_PATH, os.O_RDONLY | os.O_NONBLOCK)
            while True:
                try:
                    # 尝试读取一个字节
                    data = os.read(fd, 1)
                    if data:
                        byte_value = data[0]
                        if byte_value > 0:
                            PTT_STATUS = True
                        else:
                            PTT_STATUS = False
                    else:
                        pass
                    
                except BlockingIOError:
                    pass
                except OSError as e:
                    if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                        pass
                    else:
                        print(f"Error reading from FIFO: {e}")
                        break
                # 让出控制权，避免忙等待
                await asyncio.sleep(0.05)
        except FileNotFoundError:
            print("FIFO not found, waiting...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error opening FIFO: {e}")
            await asyncio.sleep(1)
        finally:
            try:
                os.close(fd)
            except:
                pass


async def record_microphone():
    global g_output_text, g_text_2pass_online, g_text_2pass_offline, PTT_STATUS
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

    device_count = p.get_device_count()
    print(f"设备数：{device_count}")
    for i in range(device_count):
        info = p.get_device_info_by_index(i)
        print(info)

    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        f_scp = open(args.hotword)
        hot_lines = f_scp.readlines()
        for line in hot_lines:
            words = line.strip().split(" ")
            if len(words) < 2:
                print("Please checkout format of hotwords")
                continue
            try:
                fst_dict[" ".join(words[:-1])] = int(words[-1])
            except ValueError:
                print("Please checkout format of hotwords")
        hotword_msg=json.dumps(fst_dict)

    use_itn=True
    if args.use_itn == 0:
        use_itn=False

    while True:
        if PTT_STATUS == True:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
            message = json.dumps({"mode": args.mode, "chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval,
                                "wav_name": "microphone", "is_speaking": True, "hotwords":hotword_msg, "itn": use_itn})
            await websocket.send(message)
            while PTT_STATUS == True:
                data = stream.read(CHUNK)
                message = data
                await websocket.send(message)
                await asyncio.sleep(0.005)
            print("PTT松开下降沿")
            end_message = json.dumps({"is_speaking": False})
            await websocket.send(end_message)
            g_output_text = ""
            g_text_2pass_online = ""
            g_text_2pass_offline = ""
        await asyncio.sleep(0.1)

async def message(id):
    global websocket, g_offline_msg_done, g_output_text, g_text_2pass_online, g_text_2pass_offline
    g_output_text = ""
    g_text_2pass_online = ""
    g_text_2pass_offline = ""
    try:
        while True:
            meg = await websocket.recv()
            print(meg)
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]
            timestamp=""
            g_offline_msg_done = meg.get("is_final", False)
            if "timestamp" in meg:
                timestamp = meg["timestamp"]

            if 'mode' not in meg:
                continue
            if meg["mode"] == "online":
                g_output_text += "{}".format(text)
                g_output_text = g_output_text[-args.words_max_print:]
                os.system('clear')
                print("\rpid" + str(id) + ": " + g_output_text)
            elif meg["mode"] == "offline":
                if timestamp !="":
                    g_output_text += "{} timestamp: {}".format(text, timestamp)
                else:
                    g_output_text += "{}".format(text)

                # g_output_text = g_output_text[-args.words_max_print:]
                print("\rpid" + str(id) + ": " + wav_name + ": " + g_output_text)
                g_offline_msg_done = True
            else:
                if meg["mode"] == "2pass-online":
                    g_text_2pass_online += "{}".format(text)
                    g_output_text = g_text_2pass_offline + g_text_2pass_online
                else:
                    g_text_2pass_online = ""
                    g_output_text = g_text_2pass_offline + "{}".format(text)
                    g_text_2pass_offline += "{}".format(text)
                g_output_text = g_output_text[-args.words_max_print:]
                print("\rpid" + str(id) + ": " + g_output_text)
                write_text_to_asr_fifo(g_output_text)
                # g_offline_msg_done=True

    except Exception as e:
            print("Exception:", e)
            await websocket.close()




async def ws_client(id):
    global websocket, g_offline_msg_done

    g_offline_msg_done=False

    if args.ssl == 1:
        ssl_context = ssl.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        uri = "wss://{}:{}".format(args.host, args.port)
    else:
        uri = "ws://{}:{}".format(args.host, args.port)
        ssl_context = None
    print("connect to", uri)
    async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket:
        task = asyncio.create_task(record_microphone())
        task2 = asyncio.create_task(check_ptt_status())
        task3 = asyncio.create_task(message(str(id)))
        await asyncio.gather(task, task2, task3)

    exit(0)
    

def one_thread(id):
    asyncio.get_event_loop().run_until_complete(ws_client(id))
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':

    create_all_fifo()

    p = Process(target=one_thread, args=(0,))
    p.start()
    p.join()
    print('end')
