"""
Speech Synthesis Module - Native Voice Output like ChatGPT

Provides multiple TTS engines for natural speech synthesis:
- Google TTS (online, high quality)
- pyttsx3 (offline, fast)
- Coqui TTS (neural, very high quality, optional)

Supports real-time streaming and file output.
"""

import os
import io
import torch
import numpy as np
from typing import Optional, Union, BinaryIO
from pathlib import Path

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
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class SpeechSynthesizer:
    """
    Unified speech synthesis interface with multiple TTS engines.

    Supports streaming audio output like ChatGPT's native voice feature.
    """

    def __init__(
        self,
        default_engine: str = "gtts",
        language: str = "en",
        voice_speed: float = 1.0,
        voice_pitch: float = 1.0,
    ):
        """
        Initialize speech synthesizer.

        Args:
            default_engine: "gtts", "pyttsx3", or "coqui"
            language: Language code (e.g., "en", "es", "fr")
            voice_speed: Speech speed multiplier (0.5-2.0)
            voice_pitch: Voice pitch multiplier (0.5-2.0)
        """
        self.default_engine = default_engine
        self.language = language
        self.voice_speed = voice_speed
        self.voice_pitch = voice_pitch

        # Initialize engines
        self._engines = {}

        if GTTS_AVAILABLE:
            self._engines["gtts"] = GoogleTTSEngine(language, voice_speed)
            print("✅ Google TTS engine loaded")

        if PYTTSX3_AVAILABLE:
            self._engines["pyttsx3"] = PyTTSX3Engine(language, voice_speed)
            print("✅ pyttsx3 TTS engine loaded")

        if not self._engines:
            print("⚠️  No TTS engines available. Install gtts or pyttsx3 for speech output")
            print("   pip install gtts pyttsx3")

    def speak(
        self,
        text: str,
        engine: Optional[str] = None,
        output_file: Optional[str] = None,
        stream: bool = False
    ) -> Optional[bytes]:
        """
        Convert text to speech.

        Args:
            text: Text to synthesize
            engine: TTS engine to use ("gtts", "pyttsx3", "coqui")
            output_file: Save audio to file (optional)
            stream: Return audio data instead of playing

        Returns:
            Audio data as bytes if stream=True, None otherwise
        """
        engine_name = engine or self.default_engine

        if engine_name not in self._engines:
            available = list(self._engines.keys())
            raise ValueError(f"Engine '{engine_name}' not available. Available: {available}")

        engine_instance = self._engines[engine_name]
        return engine_instance.synthesize(text, output_file, stream)

    def speak_streaming(
        self,
        text: str,
        engine: Optional[str] = None,
        chunk_size: int = 1024
    ):
        """
        Stream speech synthesis for real-time output (like ChatGPT).

        Args:
            text: Text to synthesize
            engine: TTS engine to use
            chunk_size: Audio chunk size for streaming

        Yields:
            Audio chunks as they become available
        """
        engine_name = engine or self.default_engine

        if engine_name not in self._engines:
            available = list(self._engines.keys())
            raise ValueError(f"Engine '{engine_name}' not available. Available: {available}")

        engine_instance = self._engines[engine_name]

        if hasattr(engine_instance, 'synthesize_streaming'):
            yield from engine_instance.synthesize_streaming(text, chunk_size)
        else:
            # Fallback to non-streaming
            audio_data = engine_instance.synthesize(text, stream=True)
            if audio_data:
                # Yield in chunks
                for i in range(0, len(audio_data), chunk_size):
                    yield audio_data[i:i + chunk_size]

    def list_engines(self) -> list:
        """List available TTS engines."""
        return list(self._engines.keys())

    def set_voice(self, language: str = None, speed: float = None, pitch: float = None):
        """Update voice settings."""
        if language:
            self.language = language
        if speed:
            self.voice_speed = speed
        if pitch:
            self.voice_pitch = pitch

        # Update all engines
        for engine in self._engines.values():
            if hasattr(engine, 'set_voice'):
                engine.set_voice(language, speed, pitch)


class GoogleTTSEngine:
    """Google Text-to-Speech engine (online, high quality)."""

    def __init__(self, language: str = "en", speed: float = 1.0):
        self.language = language
        self.speed = speed

    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        stream: bool = False
    ) -> Optional[bytes]:
        """Synthesize speech using Google TTS."""
        if not GTTS_AVAILABLE:
            raise ImportError("gtts not installed. Run: pip install gtts")

        try:
            # Create TTS object
            tts = gTTS(text=text, lang=self.language, slow=self.speed < 1.0)

            if stream:
                # Return audio data
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                return audio_buffer.getvalue()
            elif output_file:
                # Save to file
                tts.save(output_file)
                return None
            else:
                # Play directly (fallback to file then play)
                temp_file = f"/tmp/tts_{hash(text)}.mp3"
                tts.save(temp_file)

                # Play the file
                self._play_audio_file(temp_file)

                # Cleanup
                try:
                    os.remove(temp_file)
                except:
                    pass

                return None

        except Exception as e:
            print(f"Google TTS error: {e}")
            return None

    def _play_audio_file(self, file_path: str):
        """Play audio file using system default player."""
        try:
            if os.name == 'posix':  # Linux/Mac
                os.system(f"afplay {file_path}" if 'darwin' in os.uname().sysname.lower() else f"mpg123 {file_path}")
            elif os.name == 'nt':  # Windows
                os.system(f"start {file_path}")
        except:
            print(f"Could not play audio file: {file_path}")


class PyTTSX3Engine:
    """pyttsx3 TTS engine (offline, fast)."""

    def __init__(self, language: str = "en", speed: float = 1.0):
        self.language = language
        self.speed = speed

        if PYTTSX3_AVAILABLE:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', int(200 * speed))  # Default rate is ~200
            self.engine.setProperty('volume', 1.0)
        else:
            self.engine = None

    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        stream: bool = False
    ) -> Optional[bytes]:
        """Synthesize speech using pyttsx3."""
        if not self.engine:
            raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")

        try:
            if stream:
                # pyttsx3 doesn't support streaming well, save to temp file then read
                temp_file = f"/tmp/pyttsx3_{hash(text)}.wav"
                self.engine.save_to_file(text, temp_file)
                self.engine.runAndWait()

                # Read the file
                if SOUNDFILE_AVAILABLE:
                    audio_data, _ = sf.read(temp_file, dtype='float32')
                    # Convert to bytes (simplified)
                    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                else:
                    with open(temp_file, 'rb') as f:
                        audio_bytes = f.read()

                # Cleanup
                try:
                    os.remove(temp_file)
                except:
                    pass

                return audio_bytes

            elif output_file:
                # Save to file
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
                return None
            else:
                # Speak directly
                self.engine.say(text)
                self.engine.runAndWait()
                return None

        except Exception as e:
            print(f"pyttsx3 error: {e}")
            return None


class CoquiTTSEngine:
    """Coqui TTS engine (neural, very high quality, requires model download)."""

    def __init__(self, language: str = "en", speed: float = 1.0):
        self.language = language
        self.speed = speed
        self.tts = None

        try:
            from TTS.api import TTS
            # Load a pre-trained model (this will download on first use)
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph").to("cpu")
            print("✅ Coqui TTS engine loaded")
        except ImportError:
            print("⚠️  Coqui TTS not available. Install with: pip install TTS")
        except Exception as e:
            print(f"⚠️  Coqui TTS initialization failed: {e}")

    def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        stream: bool = False
    ) -> Optional[bytes]:
        """Synthesize speech using Coqui TTS."""
        if not self.tts:
            raise ImportError("Coqui TTS not initialized")

        try:
            if stream or output_file:
                # Generate to file first
                temp_file = output_file or f"/tmp/coqui_{hash(text)}.wav"
                self.tts.tts_to_file(text=text, file_path=temp_file)

                if stream:
                    # Read file and return bytes
                    if SOUNDFILE_AVAILABLE:
                        audio_data, _ = sf.read(temp_file, dtype='float32')
                        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                    else:
                        with open(temp_file, 'rb') as f:
                            audio_bytes = f.read()

                    # Cleanup if temp file
                    if not output_file:
                        try:
                            os.remove(temp_file)
                        except:
                            pass

                    return audio_bytes
                # else: file already saved
                return None
            else:
                # Speak directly (not well supported, fallback to file)
                temp_file = f"/tmp/coqui_{hash(text)}.wav"
                self.tts.tts_to_file(text=text, file_path=temp_file)
                self._play_audio_file(temp_file)

                # Cleanup
                try:
                    os.remove(temp_file)
                except:
                    pass

                return None

        except Exception as e:
            print(f"Coqui TTS error: {e}")
            return None

    def _play_audio_file(self, file_path: str):
        """Play audio file."""
        try:
            if os.name == 'posix':
                os.system(f"afplay {file_path}" if 'darwin' in os.uname().sysname.lower() else f"aplay {file_path}")
            elif os.name == 'nt':
                os.system(f"start {file_path}")
        except:
            print(f"Could not play audio file: {file_path}")


# Global synthesizer instance
_default_synthesizer = None

def get_speech_synthesizer(
    engine: str = "gtts",
    language: str = "en",
    voice_speed: float = 1.0
) -> SpeechSynthesizer:
    """Get or create the global speech synthesizer."""
    global _default_synthesizer

    if _default_synthesizer is None:
        _default_synthesizer = SpeechSynthesizer(
            default_engine=engine,
            language=language,
            voice_speed=voice_speed
        )

    return _default_synthesizer

def speak_text(
    text: str,
    engine: Optional[str] = None,
    language: str = "en",
    output_file: Optional[str] = None,
    stream: bool = False
) -> Optional[bytes]:
    """
    Convenience function to speak text (like ChatGPT's voice output).

    Args:
        text: Text to convert to speech
        engine: TTS engine ("gtts", "pyttsx3", "coqui")
        language: Language code
        output_file: Save to file instead of playing
        stream: Return audio bytes instead of playing

    Returns:
        Audio data as bytes if stream=True
    """
    synthesizer = get_speech_synthesizer(engine or "gtts", language)
    return synthesizer.speak(text, engine, output_file, stream)

def speak_streaming(text: str, engine: Optional[str] = None, language: str = "en"):
    """
    Stream speech synthesis for real-time voice output.

    Yields audio chunks as they become available (like ChatGPT streaming).
    """
    synthesizer = get_speech_synthesizer(engine or "gtts", language)
    yield from synthesizer.speak_streaming(text)

# ChatGPT-style voice commands
def speak_response(response_text: str, voice: str = "alloy"):
    """
    Speak a response with ChatGPT-style voice selection.

    Args:
        response_text: Text to speak
        voice: Voice type ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
    """
    # Map ChatGPT voices to our engines
    voice_mapping = {
        "alloy": ("gtts", "en"),      # Neutral, balanced
        "echo": ("pyttsx3", "en"),    # Male voice
        "fable": ("gtts", "en"),      # British accent
        "onyx": ("pyttsx3", "en"),    # Deep male voice
        "nova": ("gtts", "en"),       # Young female voice
        "shimmer": ("gtts", "en"),    # Warm female voice
    }

    engine, language = voice_mapping.get(voice, ("gtts", "en"))
    speak_text(response_text, engine=engine, language=language)
