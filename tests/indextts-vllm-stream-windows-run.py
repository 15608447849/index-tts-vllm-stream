import requests
import numpy as np
import sounddevice as sd
from queue import Queue


url = "http://127.0.0.1:9999/tts_live_stream"

text_content = '''水瓶座像一颗悬在天际的孤星，总以疏离又热忱的姿态打量世界。他们的独立是刻在骨子里的，从不愿被世俗的框架束缚 —— 别人追逐潮流时，他们偏要在旧物里翻找新意；众人在人群中寻求认同，他们却能在独处时听见更清晰的声音。这种特立独行并非刻意标榜，而是源于对 “独特” 的本能向往。
理性是水瓶座的另一枚标签。面对纷争，他们总能跳出情绪的漩涡，用逻辑梳理脉络，仿佛自带 “理性滤镜”。但这份冷静之下，藏着滚烫的人道主义精神：他们会为素不相识者的困境揪心，为社会不公发声，只是表达善意的方式总带着点 “水瓶座式” 的特别 —— 可能是匿名捐赠，也可能是用代码搭建公益平台。
他们的思维像万花筒，能从落叶联想到宇宙膨胀，在日常琐事里提炼哲学命题。这种跳跃性常常让身边人跟不上节奏，却也让他们成为创意领域的 “破局者”。只是，过度沉浸在自己的精神世界时，他们偶尔会显得有些 “不近人情”，就像星群虽璀璨，却始终隔着光年的距离。
这就是水瓶座，一半是清醒的观察者，一半是热忱的理想主义者，在世俗与理想的间隙里，活成了独一无二的星轨。'''

data = {
    "text": text_content,
    "audio_paths": [  # 支持多参考音频
        "file:///绝对路径/sample_prompt.wav"
    ]
}


# 音频参数
SAMPLE_RATE = 24000
CHUNK_SIZE = 1024  # 每次处理的帧数
CHANNELS = 1

# 创建线程安全的音频缓冲区
audio_queue = Queue(maxsize=10)  # 限制队列大小防止内存爆炸

def audio_callback(outdata, frames, time, status):
    """声音设备回调函数"""
    try:
        if not audio_queue.empty():
            # 从队列获取数据并确保形状正确 (frames, channels)
            data = audio_queue.get_nowait()
            if len(data) < frames:
                # 如果数据不足，补零
                outdata[:len(data)] = data.reshape(-1, 1)
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:frames].reshape(-1, 1)
        else:
            # 队列为空时输出静音
            outdata.fill(0)
    except Exception as e:
        print(f"Audio callback error: {e}")
        outdata.fill(0)


# 初始化音频输出流
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=CHUNK_SIZE,
    channels=CHANNELS,
    dtype='float32',
    callback=audio_callback,
    device='default'
)


# 启动音频流
with stream:
    # 请求
    response = requests.post(url, json=data, stream=True)
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE * 4):  # 16-bit样本*通道数
        if chunk:
            try:
                # 将字节数据转换为numpy数组并归一化
                pcm_data = np.frombuffer(chunk, dtype=np.float32)

                # 确保数据是二维的 (frames, channels)
                if pcm_data.ndim == 1:
                    pcm_data = pcm_data.reshape(-1, CHANNELS)

                # 将数据分块放入队列
                for i in range(0, len(pcm_data), CHUNK_SIZE):
                    block = pcm_data[i:i + CHUNK_SIZE]
                    audio_queue.put(block)

            except Exception as e:
                print(f"Error processing audio chunk: {e}")

    # 等待队列中的音频播放完毕
    while not audio_queue.empty():
        sd.sleep(100)

    print("--complete--")