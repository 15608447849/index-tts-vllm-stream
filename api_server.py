
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import argparse
import json
import asyncio
import time
import numpy as np
import soundfile as sf

from indextts.gpt.perceiver import print_once
# from indextts.infer_vllm import IndexTTS
from indextts.infer_vllm_stream import IndexTTS




tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    tts = IndexTTS(model_dir=args.model_dir, cfg_path=cfg_path, gpu_memory_utilization=args.gpu_memory_utilization)

    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = os.path.join(cur_dir, "assets/speaker.json")
    if os.path.exists(speaker_path):
        speaker_dict = json.load(open(speaker_path, 'r'))

        for speaker, audio_paths in speaker_dict.items():
            tts.registry_speaker(speaker, audio_paths)
    yield
    # Clean up the ML models and release the resources
    # ml_models.clear()

app = FastAPI(lifespan=lifespan)

# 20250708 lsp 下载音频文件
async def download_audio(url: str) -> str:
    """下载音频到临时文件，返回本地路径"""
    import requests
    from pathlib import Path
    import tempfile
    from urllib.parse import unquote, urlparse
    import os
    from datetime import datetime

    # 从 URL 中提取文件名
    parsed_url = urlparse(url)
    original_filename = unquote(os.path.basename(parsed_url.path))
    # 如果 URL 没有文件名，则生成时间戳文件名
    if not original_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"tts_audio_{timestamp}.wav"

        # 创建临时文件路径
    temp_dir = Path(tempfile.gettempdir())
    local_path = temp_dir / original_filename
    print(f"下载临时文件 url={url} path={str(local_path)}")

    # 下载文件
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查 HTTP 错误
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return str(local_path)



@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        text = data["text"]

        # 20250708 lsp 本地文件改为兼容网络文件
        # audio_paths = data["audio_paths"]
        audio_urls =  data["audio_paths"]
        audio_paths = [await download_audio(url) for url in audio_urls]

        print(f"tts_api_url audio_paths={audio_paths}\ntext={text} ")

        global tts
        sr, wav = await tts.infer(audio_paths, text)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts_live_stream")
async def tts_live_stream(request: Request):
    try:
        from io import BytesIO

        data = await request.json()
        text = data["text"]
        audio_urls = data["audio_paths"]
        audio_paths = [await download_audio(url) for url in audio_urls]

        print(f"tts_api_url audio_paths={audio_paths}\ntext={text} ")

        async def generate_audio_frames():
            async for sr, pcm_data in tts.stream_infer(audio_paths, text):
                # 将PCM数据转换为float32格式（适合Web Audio API）
                pcm_float = pcm_data.astype(np.float32) / 32767.0

                # 使用RAW格式输出，不带WAV头
                with BytesIO() as bio:
                    sf.write(bio, pcm_float, sr, format='RAW', subtype='FLOAT')
                    yield bio.getvalue()

        return StreamingResponse(
            content=generate_audio_frames(),
            media_type="audio/x-raw",
            headers={
                "Content-Type": "audio/x-raw; rate=24000; channels=1",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as ex:
        return {"error": str(ex)}

@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11996)
    parser.add_argument("--model_dir", type=str, default="/path/to/IndexTeam/Index-TTS")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    args = parser.parse_args()

    uvicorn.run(app=app, host=args.host, port=args.port)