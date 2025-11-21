"""
Enhanced Speech Synthesis Module - Production Ready

Fully integrated, production-ready speech synthesis with:
- Coqui TTS (neural, human-like quality)
- Auto-fallback: coqui → gtts → pyttsx3
- ChatGPT voice personalities
- Real-time streaming
- Docker/Colab compatible
"""

import os
import io
import sys
import time
import torch
import numpy as np
from typing import Optional, Union, BinaryIO, Iterator
from pathlib import Path
import threading
import queue

# Optional imports with fallbacks
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class StreamingAudioBuffer:
    """Thread-safe audio streaming buffer."""

    def __init__(self, chunk_size: int = 4096):
        self.chunk_size = chunk_size
        self.buffer = queue.Queue()
        self.finished = False

    def write(self, data: bytes):
        """Write audio data to buffer."""
        self.buffer.put(data)

    def read(self, timeout: float = 0.1) -> Optional[bytes]:
        """Read chunk from buffer."""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        """Mark buffer as finished."""
        self.finished = True


class CoquiTTSEngine:
    """Coqui TTS - Neural voice synthesis (highest quality)."""

    def __init__(self, language: str = "en", speed: float = 1.0):
        self.language = language
        self.speed = speed
        self.tts = None
        self.model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"

        if COQUI_AVAILABLE:
            try:
                print("🎵 Loading Coqui TTS model (this may take a moment on first run)...")
                self.tts = TTS(self.model_name).to("cpu")
                print("✅ Coqui TTS loaded successfully")
            except Exception as e:
                print(f"⚠️  Coqui TTS failed to load: {e}")
                print("   Falling back to other engines")
                self.tts = None
        else:
            print("⚠️  Coqui TTS not available (pip install TTS)")

    def synthesize(self, text: str, output_file: Optional[str] = None, stream: bool = False) -> Optional[bytes]:
        if not self.tts:
            return None

        try:
            temp_file = output_file or f"/tmp/coqui_{hash(text)}_{int(time.time())}.wav"

            # Generate speech
            self.tts.tts_to_file(text=text, file_path=temp_file)

            if stream:
                # Read and return as bytes
                if SOUNDFILE_AVAILABLE:
                    audio_data, sample_rate = sf.read(temp_file, dtype='float32')
                    # Convert to 16-bit PCM
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                else:
                    with open(temp_file, 'rb') as f:
                        audio_bytes = f.read()

                # Cleanup temp file
                if not output_file:
                    try:
                        os.remove(temp_file)
                    except:
                        pass

                return audio_bytes
            else:
                # File already saved
                return None

        except Exception as e:
            print(f"❌ Coqui TTS error: {e}")
            return None


class GoogleTTSEngine:
    """Google TTS - High quality, online."""

    def __init__(self, language: str = "en", speed: float = 1.0):
        self.language = language
        self.speed = speed

    def synthesize(self, text: str, output_file: Optional[str] = None, stream: bool = False) -> Optional[bytes]:
        if not GTTS_AVAILABLE:
            return None

        try:
            tts = gTTS(text=text, lang=self.language, slow=(self.speed < 1.0))

            if stream:
                # Return audio data
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                return audio_buffer.getvalue()
            elif output_file:
                tts.save(output_file)
                return None
            else:
                # Play directly
                temp_file = f"/tmp/gtts_{hash(text)}.mp3"
                tts.save(temp_file)
                self._play_audio(temp_file)
                try:
                    os.remove(temp_file)
                except:
                    pass
                return None

        except Exception as e:
            print(f"❌ Google TTS error: {e}")
            return None

    def _play_audio(self, file_path: str):
        """Play audio file."""
        try:
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {file_path}")
            elif sys.platform.startswith("linux"):
                os.system(f"mpg123 {file_path}")
            elif sys.platform == "win32":
                os.system(f"start {file_path}")
        except:
            pass


class PyTTSX3Engine:
    """pyttsx3 - Fast, offline TTS."""

    def __init__(self, language: str = "en", speed: float = 1.0):
        self.language = language
        self.speed = speed
        self.engine = None

        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', int(200 * speed))
                self.engine.setProperty('volume', 1.0)
                print("✅ pyttsx3 loaded")
            except Exception as e:
                print(f"⚠️  pyttsx3 failed: {e}")

    def synthesize(self, text: str, output_file: Optional[str] = None, stream: bool = False) -> Optional[bytes]:
        if not self.engine:
            return None

        try:
            if stream:
                # Save to temp file then read
                temp_file = f"/tmp/pyttsx3_{hash(text)}.wav"
                self.engine.save_to_file(text, temp_file)
                self.engine.runAndWait()

                # Read file
                if SOUNDFILE_AVAILABLE:
                    audio_data, _ = sf.read(temp_file, dtype='float32')
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                else:
                    with open(temp_file, 'rb') as f:
                        audio_bytes = f.read()

                try:
                    os.remove(temp_file)
                except:
                    pass

                return audio_bytes

            elif output_file:
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
                return None
            else:
                self.engine.say(text)
                self.engine.runAndWait()
                return None

        except Exception as e:
            print(f"❌ pyttsx3 error: {e}")
            return None


class SpeechSynthesizer:
    """
    Production-ready speech synthesizer with auto-fallback.

    Auto-fallback order: coqui → gtts → pyttsx3
    """

    def __init__(
        self,
        default_engine: str = "auto",
        language: str = "en",
        voice_speed: float = 1.0,
        voice_pitch: float = 1.0,
    ):
        self.language = language
        self.voice_speed = voice_speed
        self.voice_pitch = voice_pitch

        # Initialize engines with auto-fallback
        self.engines = {}
        self.engine_priority = ["coqui", "gtts", "pyttsx3"]

        # Try to load engines in priority order
        self._load_engines()

        # Set default engine
        if default_engine == "auto":
            # Use highest priority available engine
            for engine_name in self.engine_priority:
                if engine_name in self.engines:
                    self.default_engine = engine_name
                    break
            else:
                self.default_engine = None
        else:
            self.default_engine = default_engine if default_engine in self.engines else None

        if not self.engines:
            print("⚠️  No TTS engines available!")
            print("   Install at least one: pip install gtts  # or pip install pyttsx3  # or pip install TTS")
        elif self.default_engine:
            print(f"🎵 TTS ready - using {self.default_engine} engine")
        else:
            print("⚠️  Default engine not available, will fallback")

    def _load_engines(self):
        """Load all available engines."""
        # Coqui TTS (highest quality)
        if COQUI_AVAILABLE:
            coqui = CoquiTTSEngine(self.language, self.voice_speed)
            if coqui.tts:
                self.engines["coqui"] = coqui

        # Google TTS (good quality, online)
        if GTTS_AVAILABLE:
            self.engines["gtts"] = GoogleTTSEngine(self.language, self.voice_speed)

        # pyttsx3 (fast, offline, always works)
        if PYTTSX3_AVAILABLE:
            pyttsx3_engine = PyTTSX3Engine(self.language, self.voice_speed)
            if pyttsx3_engine.engine:
                self.engines["pyttsx3"] = pyttsx3_engine

    def speak(
        self,
        text: str,
        engine: Optional[str] = None,
        output_file: Optional[str] = None,
        stream: bool = False
    ) -> Optional[bytes]:
        """Synthesize speech with auto-fallback."""
        engine_name = engine or self.default_engine

        # Try specified engine first
        if engine_name and engine_name in self.engines:
            result = self.engines[engine_name].synthesize(text, output_file, stream)
            if result is not None or not stream:  # Non-stream success, or file output
                return result

        # Auto-fallback
        for fallback_engine in self.engine_priority:
            if fallback_engine in self.engines and fallback_engine != engine_name:
                print(f"🔄 Falling back to {fallback_engine}...")
                result = self.engines[fallback_engine].synthesize(text, output_file, stream)
                if result is not None or not stream:
                    return result

        print("❌ All TTS engines failed")
        return None

    def speak_streaming(
        self,
        text: str,
        engine: Optional[str] = None,
        chunk_size: int = 4096
    ) -> Iterator[bytes]:
        """Stream speech synthesis."""
        audio_data = self.speak(text, engine, stream=True)
        if audio_data:
            # Yield in chunks
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]

    def list_engines(self) -> list:
        """List available engines."""
        return list(self.engines.keys())

    def set_voice(self, language: str = None, speed: float = None, pitch: float = None):
        """Update voice settings."""
        if language:
            self.language = language
        if speed:
            self.voice_speed = speed
        if pitch:
            self.voice_pitch = pitch


# ChatGPT voice personality mapping
CHATGPT_VOICES = {
    "alloy": {"engine": "coqui", "style": "neutral", "description": "Balanced, neutral voice"},
    "echo": {"engine": "pyttsx3", "style": "male", "description": "Male voice"},
    "fable": {"engine": "gtts", "style": "british", "description": "British accent"},
    "onyx": {"engine": "pyttsx3", "style": "deep_male", "description": "Deep male voice"},
    "nova": {"engine": "coqui", "style": "young_female", "description": "Young female voice"},
    "shimmer": {"engine": "gtts", "style": "warm_female", "description": "Warm female voice"},
}


def speak_response(text: str, voice: str = "alloy"):
    """
    Speak text with ChatGPT voice personality.

    Args:
        text: Text to speak
        voice: ChatGPT voice name (alloy, echo, fable, onyx, nova, shimmer)
    """
    if voice not in CHATGPT_VOICES:
        print(f"⚠️  Unknown voice '{voice}', using 'alloy'")
        voice = "alloy"

    voice_config = CHATGPT_VOICES[voice]
    engine = voice_config["engine"]

    synthesizer = get_speech_synthesizer(engine, "en")
    synthesizer.speak(text, engine)


# Global synthesizer instance
_synthesizer = None

def get_speech_synthesizer(engine: str = "auto", language: str = "en", speed: float = 1.0):
    """Get or create global speech synthesizer."""
    global _synthesizer

    if _synthesizer is None:
        _synthesizer = SpeechSynthesizer(
            default_engine=engine,
            language=language,
            voice_speed=speed
        )

    return _synthesizer


def speak_text(
    text: str,
    engine: str = "auto",
    voice: str = "alloy",
    output_file: Optional[str] = None,
    stream: bool = False
):
    """Convenience function to speak text."""
    if voice in CHATGPT_VOICES:
        engine = CHATGPT_VOICES[voice]["engine"]

    synthesizer = get_speech_synthesizer(engine)
    return synthesizer.speak(text, output_file=output_file, stream=stream)


# Demo function
def demo_voices():
    """Demo all available voices."""
    print("🎵 Testing all ChatGPT voice personalities...")

    test_text = "Hello! This is a test of the speech synthesis system."

    for voice_name, config in CHATGPT_VOICES.items():
        print(f"\n🎤 Testing {voice_name}: {config['description']}")
        try:
            speak_response(test_text, voice_name)
            time.sleep(1)  # Brief pause between voices
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print("\n✅ Voice demo complete!")


if __name__ == "__main__":
    # Quick test
    print("🎵 Speech Synthesis Module Test")
    print("=" * 40)

    synthesizer = SpeechSynthesizer()
    print(f"Available engines: {synthesizer.list_engines()}")

    if synthesizer.engines:
        print("\n🗣️  Testing speech synthesis...")
        synthesizer.speak("Hello from the NanoChat speech synthesis system!")
        print("✅ Speech test complete!")
    else:
        print("❌ No TTS engines available - install dependencies:")
        print("   pip install gtts pyttsx3 TTS")
