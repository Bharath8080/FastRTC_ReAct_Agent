"""
FastRTC Voice Agent (Clean Logs + Colored + Cleaned TTS)
SpeechRecognition STT + Cartesia Sonic 3 TTS
"""

import os
import io
import time
import argparse
import numpy as np
import gradio as gr
import speech_recognition as sr
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
import re
from cartesia import Cartesia
from fastrtc import AlgoOptions, ReplyOnPause, Stream
from scripts.agent import agent, agent_config

load_dotenv()

# ----------------------------- CLEAN + COLORED LOGGER ---------------------------------

logger.remove()

WANTED = ["🎙", "👂", "💬", "🔊", "⚡"]

def filter_logs(record):
    return any(e in record["message"] for e in WANTED)

logger.add(
    lambda msg: print(msg),
    level="INFO",
    filter=filter_logs,
    format="<green>{time:HH:mm:ss}</green> | {message}",
)

# Color helper
CYAN   = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# ----------------------------- TEXT CLEANER FOR TTS ------------------------------------

def clean_text_for_tts(text: str) -> str:
    """
    Clean markdown formatting for better TTS output.
    """
    clean_text = text
    clean_text = re.sub(r'\*+', '', clean_text)
    clean_text = re.sub(r'[#_`]', '', clean_text)
    clean_text = re.sub(r'-{2,}', ' ', clean_text)
    clean_text = re.sub(r'\|', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# ----------------------------- INIT CLIENTS --------------------------------------------

logger.info(f"{CYAN}🎙 Initializing SpeechRecognition + Cartesia Sonic-3 TTS..{RESET}")

recognizer = sr.Recognizer()

cartesia_client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))

CARTESIA_TTS_CONFIG = {
    "model_id": "sonic-3",
    "voice": {
        "mode": "id",
        "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02",
    },
    "output_format": {
        "container": "raw",
        "sample_rate": 24000,
        "encoding": "pcm_f32le",
    },
}

# ----------------------------- FUNCTIONS ------------------------------------

def stt_transcribe(audio):
    """
    Convert audio → text (SpeechRecognition STT)
    """
    sample_rate, audio_array = audio

    if audio_array.dtype == np.float32:
        audio_int16 = (audio_array * 32767).astype(np.int16)
    else:
        audio_int16 = audio_array.astype(np.int16)

    if audio_int16.ndim > 1:
        audio_int16 = audio_int16.flatten()

    audio_bytes = audio_int16.tobytes()
    audio_data = sr.AudioData(audio_bytes, sample_rate, 2)

    try:
        return recognizer.recognize_google(audio_data)
    except:
        return ""

def generate_speech(text):
    iter_chunks = cartesia_client.tts.bytes(
        model_id=CARTESIA_TTS_CONFIG["model_id"],
        transcript=text,
        voice=CARTESIA_TTS_CONFIG["voice"],
        output_format=CARTESIA_TTS_CONFIG["output_format"],
    )

    buffer = b""
    element_size = 4
    sample_rate = CARTESIA_TTS_CONFIG["output_format"]["sample_rate"]

    for chunk in iter_chunks:
        buffer += chunk
        n = len(buffer) // element_size

        if n:
            size = n * element_size
            block = buffer[:size]
            buffer = buffer[size:]

            arr = np.frombuffer(block, dtype=np.float32)
            yield (sample_rate, arr)

    if buffer:
        rem = len(buffer) % element_size
        if rem:
            buffer += b"\x00" * (element_size - rem)
        arr = np.frombuffer(buffer, dtype=np.float32)
        yield (sample_rate, arr)

# ----------------------------- MAIN PIPELINE ------------------------------------

def response(audio):
    start_time = time.time()
    logger.info(f"{CYAN}🎙 Received audio input{RESET}")

    # --- STT ---
    stt_start = time.time()
    transcript = stt_transcribe(audio)
    stt_time = time.time() - stt_start

    logger.info(f'{YELLOW}👂 Transcribed: "{transcript}"{RESET}')

    if not transcript.strip():
        return

    # --- LLM ---
    llm_start = time.time()
    agent_reply = agent.invoke(
        {"messages": [{"role": "user", "content": transcript}]},
        config=agent_config,
    )

    reply_raw = agent_reply["messages"][-1].content
    llm_time = time.time() - llm_start

    logger.info(f'{MAGENTA}💬 Response: "{reply_raw}"{RESET}')

    # Clean for TTS only
    reply_clean = clean_text_for_tts(reply_raw)

    # --- TTS ---
    logger.info(f"{GREEN}🔊 Speaking...{RESET}")

    tts_start = time.time()
    chunk_count = 0

    for chunk in generate_speech(reply_clean):
        chunk_count += 1
        yield chunk

    tts_time = time.time() - tts_start
    total_time = time.time() - start_time

    # --- PERFORMANCE LOG ---
    logger.info(
        f"{CYAN}⚡ Performance:{RESET} "
        f"{YELLOW}STT={stt_time:.2f}s{RESET} | "
        f"{MAGENTA}LLM={llm_time:.2f}s{RESET} | "
        f"{GREEN}TTS={tts_time:.2f}s{RESET} | "
        f"{CYAN}Total={total_time:.2f}s{RESET} | "
        f"{RED}Chunks={chunk_count}{RESET}"
    )

# ----------------------------- STREAM SETUP ------------------------------------

def create_stream():
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(speech_threshold=0.4),
        ),
    )

stream = create_stream()
app = FastAPI()
app = gr.mount_gradio_app(app, stream.ui, path="/")

# ----------------------------- MAIN ------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone", action="store_true")
    args = parser.parse_args()

    os.environ["GRADIO_SSR_MODE"] = "false"

    if args.phone:
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        stream.ui.launch(server_port=7860)
