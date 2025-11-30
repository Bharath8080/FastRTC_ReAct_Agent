"""
FastRTC Voice Agent with Groq Integration
Voice agent using Groq Whisper for STT and FastRTC Kokoro for TTS
"""

import argparse
import io
import os
from typing import Generator, Tuple

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from loguru import logger

from fastrtc import AlgoOptions, ReplyOnPause, Stream, get_tts_model
from groq import Groq
from websearch_agent import agent, agent_config


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

# Initialize clients
logger.info("🎤 Initializing voice processing components...")
groq_client = Groq()
tts_model = get_tts_model()  # FastRTC Kokoro TTS
logger.info("✅ Voice processing initialized successfully")

# Configuration constants
STT_CONFIG = {
    "model": "whisper-large-v3-turbo",
    "response_format": "text",
}

STREAM_CONFIG = {
    "speech_threshold": 0.5,
}


# ============================================================================
# Audio Processing Functions
# ============================================================================

def audio_to_wav_bytes(audio: Tuple[int, np.ndarray]) -> bytes:
    """
    Convert audio numpy array to WAV format bytes.
    
    Args:
        audio: Tuple of (sample_rate, audio_data)
    
    Returns:
        bytes: WAV format audio data
    """
    sample_rate, audio_data = audio
    buffer = io.BytesIO()
    sf.write(buffer, audio_data.T, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio using Groq Whisper.
    
    Args:
        audio_bytes: WAV format audio bytes
    
    Returns:
        str: Transcribed text
    """
    transcript = groq_client.audio.transcriptions.create(
        file=("audio-file.wav", audio_bytes),
        model=STT_CONFIG["model"],
        response_format=STT_CONFIG["response_format"],
    )
    return transcript


def generate_response(transcript: str) -> str:
    """
    Generate response using LangGraph agent.
    
    Args:
        transcript: Input text from STT
    
    Returns:
        str: Generated response text
    """
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": transcript}]},
        config=agent_config
    )
    return agent_response["messages"][-1].content


def synthesize_speech(text: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Synthesize speech using FastRTC Kokoro TTS.
    
    Args:
        text: Text to synthesize
    
    Yields:
        Tuple[int, np.ndarray]: Audio chunks for playback
    """
    for audio_chunk in tts_model.stream_tts_sync(text):
        yield audio_chunk


# ============================================================================
# Main Response Handler
# ============================================================================

def response(
    audio: Tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input and generate voice response.
    
    Pipeline: STT (Groq Whisper) → LangGraph Agent → TTS (Kokoro)
    
    Args:
        audio: Tuple of (sample_rate, audio_data)
    
    Yields:
        Tuple[int, np.ndarray]: Audio chunks for playback
    """
    logger.info("🎙 Received audio input")
    
    try:
        # ========== Speech-to-Text ==========
        logger.debug("🔄 Transcribing audio with Groq Whisper...")
        audio_bytes = audio_to_wav_bytes(audio)
        transcript = transcribe_audio(audio_bytes)
        logger.info(f'👂 Transcribed: "{transcript}"')
        
        # ========== Agent Processing ==========
        logger.debug("🧠 Running LangGraph agent...")
        response_text = generate_response(transcript)
        logger.info(f'💬 Response: "{response_text}"')
        
        # ========== Text-to-Speech ==========
        logger.debug("🔊 Generating speech with FastRTC Kokoro TTS...")
        for audio_chunk in synthesize_speech(response_text):
            yield audio_chunk
            
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
                speech_threshold=STREAM_CONFIG["speech_threshold"],
            ),
        ),
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="FastRTC Voice Agent with Groq Whisper and Kokoro TTS"
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