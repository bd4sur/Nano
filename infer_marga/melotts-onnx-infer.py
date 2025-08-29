import librosa
from melo_onnx import MeloTTS_ONNX
import numpy as np
import sounddevice as sd
import os
import errno
import time
import threading
import queue

model_path = "/home/bd4sur/ai/_model/melotts_zh_mix_en_onnx"
tts = MeloTTS_ONNX(model_path)

# 创建线程安全的队列
text_queue = queue.Queue()  # 存储待处理的文本
audio_queue = queue.Queue()  # 存储待播放的音频

# TTS合成工作线程，按顺序处理文本
def tts_worker():
    while True:
        try:
            # 从文本队列获取文本
            text = text_queue.get(timeout=1.0)
            
            # 执行TTS合成
            try:
                audio = tts.speak(text, tts.speakers[0])
                # 将合成好的音频放入音频队列
                audio_queue.put(audio)
            except Exception as e:
                print(f"TTS合成失败: {e}")
                
            # 标记任务完成
            text_queue.task_done()
            
        except queue.Empty:
            # 超时，继续循环等待新文本
            continue

# 音频播放工作线程，按顺序播放音频
def play_audio_worker():
    while True:
        try:
            # 从音频队列获取音频数据
            audio = audio_queue.get(timeout=1.0)
            
            # 归一化并减慢音频
            pcm_data = np.array(audio, dtype=np.float32)
            pcm_data = librosa.util.normalize(pcm_data, norm=np.inf)
            slowed_pcm_data = librosa.effects.time_stretch(pcm_data, rate=0.8)

            # 播放音频（阻塞方式）
            sd.play(slowed_pcm_data, samplerate=48000, blocking=True)

            # 标记任务完成
            audio_queue.task_done()
            
        except queue.Empty:
            # 超时，继续循环等待新音频
            continue
        except Exception as e:
            print(f"播放音频时出错: {e}")

# 读取FIFO并在独立线程中处理TTS
def read_fifo(fifo_path):
    # 创建FIFO（如果不存在）
    try:
        os.mkfifo(fifo_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # 启动TTS合成线程和音频播放线程
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    audio_thread = threading.Thread(target=play_audio_worker, daemon=True)
    audio_thread.start()

    # 初始提示
    initial_audio = tts.speak("T T S 准备就绪", tts.speakers[0])
    audio_queue.put(initial_audio)

    # 以非阻塞模式打开FIFO
    fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)

    try:
        while True:
            try:
                # 读取数据
                data = os.read(fd, 256)
                if data:
                    text = data.decode('utf-8', errors='ignore').strip().replace('\n', '，').replace('！', '，').replace('。', '，').replace('*', '')
                    print(f"收到文本：{text}")

                    # 将文本放入队列，由TTS工作线程处理
                    text_queue.put(text)

            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                # 没有数据时继续循环
                    
            # time.sleep(0.01)  # 短暂休眠以减少CPU使用

    except KeyboardInterrupt:
        print("\n正在停止...")
        # 等待所有任务完成
        text_queue.join()
        audio_queue.join()
    finally:
        os.close(fd)

# 使用示例
if __name__ == "__main__":
    fifo_path = "/tmp/tts_fifo"
    read_fifo(fifo_path)
