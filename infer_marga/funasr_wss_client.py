# -*- encoding: utf-8 -*-
"""
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

python funasr_wss_client.py --host "0.0.0.0" --port 10096 --mode 2pass --chunk_size "5,10,5" --ssl 0 --thread_num 4

"""
import os
import time
import websockets, ssl
import asyncio
# import threading
import argparse
import json
import traceback
from multiprocessing import Process
# from funasr.fileio.datadir_writer import DatadirWriter

import logging

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="localhost",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10095,
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
parser.add_argument("--audio_in",
                    type=str,
                    default=None,
                    help="audio_in")
parser.add_argument("--audio_fs",
                    type=int,
                    default=16000,
                    help="audio_fs")
parser.add_argument("--send_without_sleep",
                    action="store_true",
                    default=True,
                    help="if audio_in is set, send_without_sleep")
parser.add_argument("--thread_num",
                    type=int,
                    default=1,
                    help="thread_num")
parser.add_argument("--words_max_print",
                    type=int,
                    default=10000,
                    help="chunk")
parser.add_argument("--output_dir",
                    type=str,
                    default=None,
                    help="output_dir")
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
print(args)
# voices = asyncio.Queue()
from queue import Queue

voices = Queue()
offline_msg_done=False

text_print = ""
text_print_2pass_online = ""
text_print_2pass_offline = ""

if args.output_dir is not None:
    # if os.path.exists(args.output_dir):
    #     os.remove(args.output_dir)
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)




# 命名管道路径
PIPE_NAME = "/tmp/asr_pipe"
pipe_fd = None

def create_named_pipe():
    """创建命名管道"""
    try:
        # 如果管道已存在，先删除
        if os.path.exists(PIPE_NAME):
            os.unlink(PIPE_NAME)
        
        # 创建命名管道
        os.mkfifo(PIPE_NAME)
        print(f"命名管道 {PIPE_NAME} 创建成功")
        return True
    except OSError as e:
        print(f"创建命名管道失败: {e}")
        return False

def open_pipe_nonblocking():
    """以非阻塞方式打开管道"""
    global pipe_fd
    try:
        if pipe_fd:
            try:
                os.close(pipe_fd)
            except:
                pass
        
        # 以非阻塞写模式打开管道
        pipe_fd = os.open(PIPE_NAME, os.O_WRONLY | os.O_NONBLOCK)
        print("管道已打开（非阻塞模式）")
        return True
    except OSError as e:
        # 如果没有读取者，open会失败
        print("管道打开失败")
        pipe_fd = None
        return False

def clear_pipe_content():
    """清空管道中已有的内容"""
    global pipe_fd
    if not pipe_fd:
        return False
    
    try:
        # 尝试读取并丢弃所有现有数据
        while True:
            data = os.read(pipe_fd, 1024)  # 读取1KB数据
            print("读取1KB数据")
            if len(data) == 0:
                print("管道清空")
                break  # 没有更多数据可读
    except OSError:
        print("没有数据可读或其它错误")
        pass
    return True

def write_text_to_pipe(text):
    """尝试向命名管道写入字符串"""
    global pipe_fd
    try:
        if not pipe_fd:
            # 尝试打开管道
            if not open_pipe_nonblocking():
                return False
        # 清空管道中已有的内容
        clear_pipe_content()
        # 尝试写入数据
        os.write(pipe_fd, (text + '\n').encode())
        return True
    except OSError as e:
        # 如果写入失败（通常是管道没有读取者），关闭文件描述符
        if pipe_fd:
            try:
                os.close(pipe_fd)
            except:
                pass
            pipe_fd = None
        return False


def check_ptt_status():
    try:
        with open('/tmp/ptt_status', 'r') as f:
            content = f.read().strip()
            if content == '1':
                return True
            elif content == '0':
                return False
    except Exception as e:
        print(f"错误: {e}")


async def record_microphone():
    is_finished = False
    import pyaudio
    # print("2")
    global voices, text_print, text_print_2pass_online, text_print_2pass_offline
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

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
        ptt_status = check_ptt_status()
        if ptt_status == True:
            print("PTT已按下，打开音频流")
            stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
            message = json.dumps({"mode": args.mode, "chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval,
                                "wav_name": "microphone", "is_speaking": True, "hotwords":hotword_msg, "itn": use_itn})
            await websocket.send(message)
            while ptt_status == True:
                ptt_status = check_ptt_status()
                data = stream.read(CHUNK)
                message = data
                await websocket.send(message)
                await asyncio.sleep(0.005)

            print("PTT松开")
            text_print = ""
            text_print_2pass_online = ""
            text_print_2pass_offline = ""

async def message(id):
    global websocket, voices, offline_msg_done, text_print, text_print_2pass_online, text_print_2pass_offline
    text_print = ""
    text_print_2pass_online = ""
    text_print_2pass_offline = ""
    if args.output_dir is not None:
        ibest_writer = open(os.path.join(args.output_dir, "text.{}".format(id)), "a", encoding="utf-8")
    else:
        ibest_writer = None
    try:
        while True:
            meg = await websocket.recv()
            print(meg)
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]
            timestamp=""
            offline_msg_done = meg.get("is_final", False)
            if "timestamp" in meg:
                timestamp = meg["timestamp"]

            if ibest_writer is not None:
                if timestamp !="":
                    text_write_line = "{}\t{}\t{}\n".format(wav_name, text, timestamp)
                else:
                    text_write_line = "{}\t{}\n".format(wav_name, text)
                ibest_writer.write(text_write_line)

            if 'mode' not in meg:
                continue
            if meg["mode"] == "online":
                text_print += "{}".format(text)
                text_print = text_print[-args.words_max_print:]
                os.system('clear')
                print("\rpid" + str(id) + ": " + text_print)
            elif meg["mode"] == "offline":
                if timestamp !="":
                    text_print += "{} timestamp: {}".format(text, timestamp)
                else:
                    text_print += "{}".format(text)

                # text_print = text_print[-args.words_max_print:]
                # os.system('clear')
                print("\rpid" + str(id) + ": " + wav_name + ": " + text_print)
                offline_msg_done = True
            else:
                if meg["mode"] == "2pass-online":
                    text_print_2pass_online += "{}".format(text)
                    text_print = text_print_2pass_offline + text_print_2pass_online
                else:
                    text_print_2pass_online = ""
                    text_print = text_print_2pass_offline + "{}".format(text)
                    text_print_2pass_offline += "{}".format(text)
                text_print = text_print[-args.words_max_print:]
                # os.system('clear')
                print("\rpid" + str(id) + ": " + text_print)
                write_text_to_pipe(text_print)
                # offline_msg_done=True

    except Exception as e:
            print("Exception:", e)
            #traceback.print_exc()
            #await websocket.close()




async def ws_client(id, chunk_begin, chunk_size):
    if args.audio_in is None:
        chunk_begin=0
        chunk_size=1
    global websocket,voices,offline_msg_done
    
    for i in range(chunk_begin,chunk_begin+chunk_size):
        offline_msg_done=False
        voices = Queue()
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
            task3 = asyncio.create_task(message(str(id)+"_"+str(i))) #processid+fileid
            await asyncio.gather(task, task3)
    exit(0)
    

def one_thread(id, chunk_begin, chunk_size):
    asyncio.get_event_loop().run_until_complete(ws_client(id, chunk_begin, chunk_size))
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':

    create_named_pipe()



    # for microphone
    if args.audio_in is None:
        p = Process(target=one_thread, args=(0, 0, 0))
        p.start()
        p.join()
        print('end')
    else:
        # calculate the number of wavs for each preocess
        wavs = [args.audio_in]

        for wav in wavs:
            wav_splits = wav.strip().split()
            wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
            wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
            audio_type = os.path.splitext(wav_path)[-1].lower()


        total_len = len(wavs)
        if total_len >= args.thread_num:
            chunk_size = int(total_len / args.thread_num)
            remain_wavs = total_len - chunk_size * args.thread_num
        else:
            chunk_size = 1
            remain_wavs = 0

        process_list = []
        chunk_begin = 0
        for i in range(args.thread_num):
            now_chunk_size = chunk_size
            if remain_wavs > 0:
                now_chunk_size = chunk_size + 1
                remain_wavs = remain_wavs - 1
            # process i handle wavs at chunk_begin and size of now_chunk_size
            p = Process(target=one_thread, args=(i, chunk_begin, now_chunk_size))
            chunk_begin = chunk_begin + now_chunk_size
            p.start()
            process_list.append(p)

        for i in process_list:
            p.join()

        print('end')
