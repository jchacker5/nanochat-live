"""
Live NanoChat Components

Real-time multimodal interaction modules.
"""

from .tts import (
    SpeechSynthesizer,
    speak_response,
    speak_text,
    get_speech_synthesizer,
    CHATGPT_VOICES,
    demo_voices
)

from .voice_output import (
    VoiceOutput,
    get_voice_output,
    speak_text as speak_text_wrapper
)

from .capture import (
    WebcamCapture,
    AudioCapture,
    MultimodalCapture,
    get_multimodal_capture,
    capture_demo
)

__all__ = [
    # TTS
    "SpeechSynthesizer",
    "speak_response",
    "speak_text",
    "get_speech_synthesizer",
    "CHATGPT_VOICES",
    "demo_voices",

    # Voice Output
    "VoiceOutput",
    "get_voice_output",
    "speak_text_wrapper",

    # Multimodal Capture
    "WebcamCapture",
    "AudioCapture",
    "MultimodalCapture",
    "get_multimodal_capture",
    "capture_demo",
]
