import librosa
import sherpa_onnx
import numpy as np
import sounddevice as sd
import os
import errno
import time
import threading
import queue

model_path = "/home/bd4sur/ai/_model/vits-melo-tts-zh_en"
# model_path = "/home/bd4sur/ai/_model/sherpa-onnx-vits-zh-ll"

SPEAKER_ID = 0
SPEED = 1.0

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{model_path}/model.onnx",
            lexicon=f"{model_path}/lexicon.txt",
            dict_dir=f"{model_path}/dict",
            tokens=f"{model_path}/tokens.txt",
        ),
        num_threads=4,
        provider="cpu",
    ),
    rule_fsts=f"{model_path}/date.fst,{model_path}/number.fst",
    max_num_sentences=1,
)

if not tts_config.validate():
    raise ValueError("Please check your config")

tts = sherpa_onnx.OfflineTts(tts_config)

global SAMPLE_RATE
SAMPLE_RATE = tts.sample_rate
print(f"sample_rate = {SAMPLE_RATE}\n")


# 创建线程安全的队列
text_queue = queue.Queue()  # 存储待处理的文本
audio_queue = queue.Queue()  # 存储待播放的音频

# Create a stop event for immediate audio stopping
stop_audio_event = threading.Event()

# def generated_audio_callback(samples: np.ndarray, progress: float):
#     audio_queue.put(samples)


# TTS合成工作线程，按顺序处理文本
def tts_worker():
    while True:
        try:
            text = text_queue.get(timeout=1.0)
            try:
                audio = tts.generate(
                    text,
                    sid=SPEAKER_ID,
                    speed=SPEED,
                    # callback=generated_audio_callback,
                )
                audio_queue.put(audio.samples)
            except Exception as e:
                print(f"TTS合成失败: {e}")
            text_queue.task_done()
        except queue.Empty:
            continue

# 音频播放工作线程，按顺序播放音频
def play_audio_worker():
    global SAMPLE_RATE
    while True:
        try:
            audio = audio_queue.get(timeout=1.0)

            # Check if we should stop playback
            if stop_audio_event.is_set():
                stop_audio_event.clear()  # Reset for next time
                audio_queue.task_done()
                continue

            # 归一化并减慢音频
            pcm_data = np.array(audio, dtype=np.float32)
            pcm_data = librosa.util.normalize(pcm_data, norm=np.inf)
            # pcm_data = librosa.effects.time_stretch(pcm_data, rate=1.0)
            # 播放音频（阻塞方式）
            sd.play(pcm_data, samplerate=SAMPLE_RATE, blocking=True)
            # 标记任务完成
            audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"播放音频时出错: {e}")

# 读取FIFO并在独立线程中处理TTS
def read_fifo(fifo_path):
    try:
        os.mkfifo(fifo_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    audio_thread = threading.Thread(target=play_audio_worker, daemon=True)
    audio_thread.start()

    # 初始提示
    audio = tts.generate(
        "T T S 准备就绪",
        sid=SPEAKER_ID,
        speed=SPEED,
        # callback=generated_audio_callback,
    )
    audio_queue.put(audio.samples)
    # print(audio)

    # 以非阻塞模式打开FIFO
    fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)

    try:
        while True:
            try:
                data = os.read(fd, 1024)
                if data:
                    text = data.decode('utf-8', errors='ignore').strip()
                    if "_TTS_STOP_" in text:
                        print("收到停止指令，清空队列并停止播放")
                        # 设置停止事件以中断当前播放
                        if not audio_queue.empty():
                            stop_audio_event.set()

                        # 清空文本队列
                        while not text_queue.empty():
                            try:
                                text_queue.get_nowait()
                                text_queue.task_done()
                            except queue.Empty:
                                break
                                
                        # 清空音频队列
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                                audio_queue.task_done()
                            except queue.Empty:
                                break
                    else:
                        text = text.replace('\n', '，').replace('！', '，').replace('：', '，').replace('。', '，').replace('*', '').replace('. ', '，').replace('.', '点')
                        print(f"收到文本：{text}")
                        text_queue.put(text)
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
            time.sleep(0.01)  # 短暂休眠以减少CPU使用
    except KeyboardInterrupt:
        print("\n正在停止...")
        text_queue.join()
        audio_queue.join()
    finally:
        os.close(fd)

# 使用示例
if __name__ == "__main__":
    fifo_path = "/tmp/tts_fifo"
    read_fifo(fifo_path)
