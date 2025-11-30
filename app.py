"""
FastRTC Voice Agent with Cartesia Integration
Ultra-low latency voice agent using Cartesia STT and Sonic 3 TTS
"""

import argparse
import os
import re
import time
from typing import Generator, Tuple

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from cartesia import Cartesia
from fastrtc import AlgoOptions, ReplyOnPause, Stream
from scripts.agent import agent, agent_config

# ============================================================================
# Configuration & Initialization
# ============================================================================

load_dotenv()

# Configure logging with clean, colored output
logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# Initialize Cartesia client
logger.info("🎤 Initializing Cartesia Sonic 3 TTS...")
cartesia_client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))

# Cartesia Sonic 3 TTS Configuration
CARTESIA_TTS_CONFIG = {
    "model_id": "sonic-3",
    "voice": {
        "mode": "id",
        "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02",  # Katie voice
    },
    "output_format": {
        "container": "raw",
        "sample_rate": 24000,
        "encoding": "pcm_f32le",
    },
}

# STT Configuration
STT_CONFIG = {
    "model": "ink-whisper",
    "language": "en",
    "encoding": "pcm_s16le",
    "sample_rate": 16000,
    "min_volume": 0.1,
    "max_silence_duration_secs": 0.3,
}

logger.info("✅ Cartesia Sonic 3 TTS configured successfully")


# ============================================================================
# Audio Processing Functions
# ============================================================================

def preprocess_audio(
    audio: Tuple[int, np.ndarray],
    target_sample_rate: int = 16000
) -> bytes:
    """
    Preprocess audio data for Cartesia STT.
    
    Args:
        audio: Tuple of (sample_rate, audio_data)
        target_sample_rate: Target sample rate for STT (default: 16000)
    
    Returns:
        bytes: PCM audio bytes ready for STT
    """
    sample_rate, audio_data = audio
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        import librosa
        
        # Convert to float32
        if audio_data.dtype != np.float32:
            audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        else:
            audio_float = audio_data
        
        # Resample
        audio_resampled = librosa.resample(
            audio_float.T.flatten() if audio_float.ndim > 1 else audio_float,
            orig_sr=sample_rate,
            target_sr=target_sample_rate
        )
        audio_data = audio_resampled
    
    # Convert to 16-bit PCM
    if audio_data.dtype == np.float32:
        audio_int16 = (audio_data * 32767).astype(np.int16)
    else:
        audio_int16 = audio_data.astype(np.int16)
    
    return audio_int16.tobytes()


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio using Cartesia STT.
    
    Args:
        audio_bytes: PCM audio bytes
    
    Returns:
        str: Transcribed text
    """
    # Create WebSocket connection
    ws = cartesia_client.stt.websocket(**STT_CONFIG)
    
    # Send audio in 20ms chunks for streaming
    chunk_size = int(STT_CONFIG["sample_rate"] * 0.02 * 2)
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        if chunk:
            ws.send(chunk)
    
    # Finalize transcription
    ws.send("finalize")
    ws.send("done")
    
    # Receive transcription results
    transcript = ""
    for result in ws.receive():
        if result['type'] == 'transcript' and result['is_final']:
            transcript = result['text']
            break
        elif result['type'] == 'done':
            break
    
    ws.close()
    return transcript


def clean_text_for_tts(text: str) -> str:
    """
    Clean markdown formatting for better TTS output.
    
    Args:
        text: Raw text with potential markdown
    
    Returns:
        str: Cleaned text suitable for TTS
    """
    clean_text = text
    
    # Remove markdown formatting
    clean_text = re.sub(r'\*+', '', clean_text)  # Bold/italic
    clean_text = re.sub(r'[#_`]', '', clean_text)  # Headers, underscores, code
    clean_text = re.sub(r'-{2,}', ' ', clean_text)  # Multiple dashes
    clean_text = re.sub(r'\|', ' ', clean_text)  # Table pipes
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Extra whitespace
    
    return clean_text


def generate_speech(text: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generate speech audio using Cartesia Sonic 3 TTS.
    
    Args:
        text: Text to synthesize
    
    Yields:
        Tuple[int, np.ndarray]: (sample_rate, audio_array) chunks
    """
    clean_text = clean_text_for_tts(text)
    
    if clean_text != text:
        logger.debug(f"Cleaned text for TTS: {clean_text}")
    
    # Generate speech chunks
    chunk_iter = cartesia_client.tts.bytes(
        model_id=CARTESIA_TTS_CONFIG["model_id"],
        transcript=clean_text,
        voice=CARTESIA_TTS_CONFIG["voice"],
        output_format=CARTESIA_TTS_CONFIG["output_format"],
    )
    
    buffer = b""
    element_size = 4  # float32 is 4 bytes
    chunk_count = 0
    
    # Stream audio chunks
    for chunk in chunk_iter:
        buffer += chunk
        
        # Process complete float32 samples
        num_complete_samples = len(buffer) // element_size
        if num_complete_samples > 0:
            complete_bytes = num_complete_samples * element_size
            complete_buffer = buffer[:complete_bytes]
            buffer = buffer[complete_bytes:]
            
            audio_array = np.frombuffer(complete_buffer, dtype=np.float32)
            chunk_count += 1
            
            yield (CARTESIA_TTS_CONFIG["output_format"]["sample_rate"], audio_array)
    
    # Process remaining bytes
    if len(buffer) > 0:
        remainder = len(buffer) % element_size
        if remainder != 0:
            buffer += b'\x00' * (element_size - remainder)
        
        if len(buffer) >= element_size:
            audio_array = np.frombuffer(buffer, dtype=np.float32)
            chunk_count += 1
            yield (CARTESIA_TTS_CONFIG["output_format"]["sample_rate"], audio_array)
    
    logger.debug(f"Generated {chunk_count} audio chunks")


# ============================================================================
# Main Response Handler
# ============================================================================

def response(
    audio: Tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input and generate voice response.
    
    Pipeline: STT → LLM → TTS
    
    Args:
        audio: Tuple of (sample_rate, audio_data)
    
    Yields:
        Tuple[int, np.ndarray]: Audio chunks for playback
    """
    start_time = time.time()
    logger.info("🎙 Received audio input")
    
    try:
        # ========== Speech-to-Text ==========
        stt_start = time.time()
        logger.debug("🔄 Transcribing audio with Cartesia...")
        
        audio_bytes = preprocess_audio(audio, target_sample_rate=STT_CONFIG["sample_rate"])
        transcript = transcribe_audio(audio_bytes)
        
        stt_time = time.time() - stt_start
        logger.info(f'👂 Transcribed in {stt_time:.2f}s: "{transcript}"')
        
        # ========== Language Model ==========
        llm_start = time.time()
        logger.debug("🧠 Running agent...")
        
        agent_response = agent.invoke(
            {"messages": [{"role": "user", "content": transcript}]},
            config=agent_config
        )
        response_text = agent_response["messages"][-1].content
        
        llm_time = time.time() - llm_start
        logger.info(f'💬 Response in {llm_time:.2f}s: "{response_text}"')
        
        # ========== Text-to-Speech ==========
        tts_start = time.time()
        logger.debug("🔊 Generating speech with Cartesia Sonic 3...")
        
        chunk_count = 0
        for audio_chunk in generate_speech(response_text):
            chunk_count += 1
            yield audio_chunk
        
        tts_time = time.time() - tts_start
        total_time = time.time() - start_time
        
        logger.info(
            f'⚡ Performance: STT={stt_time:.2f}s | '
            f'LLM={llm_time:.2f}s | TTS={tts_time:.2f}s | '
            f'Total={total_time:.2f}s | Chunks={chunk_count}'
        )
        
    except Exception as e:
        logger.error(f"Error in response pipeline: {str(e)}")
        raise


# ============================================================================
# Stream Configuration
# ============================================================================

def create_stream() -> Stream:
    """
    Create and configure FastRTC Stream instance.
    
    Returns:
        Stream: Configured stream with audio capabilities
    """
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                speech_threshold=0.4,
            ),
        ),
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="FastRTC Cartesia Voice Agent (Ultra-Low Latency)"
    )
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    args = parser.parse_args()
    
    stream = create_stream()
    logger.info("🎧 Stream handler configured")
    
    if args.phone:
        logger.info("📞 Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("🌈 Launching with Gradio UI...")
        stream.ui.launch()


if __name__ == "__main__":
    main()