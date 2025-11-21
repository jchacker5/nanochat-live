"""
Voice Output Wrapper for NanoChat

Integrates speech synthesis into chat interfaces.
Used by chat_web.py and chat_cli.py for voice output.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.tts import speak_response, get_speech_synthesizer, CHATGPT_VOICES

class VoiceOutput:
    """Voice output wrapper for chat interfaces."""

    def __init__(self, voice: str = "alloy", engine: str = "auto", enabled: bool = True):
        self.voice = voice
        self.engine = engine
        self.enabled = enabled
        self.synthesizer = None

        if enabled:
            self.synthesizer = get_speech_synthesizer(engine)

    def speak(self, text: str, voice: str = None):
        """Speak text with specified voice."""
        if not self.enabled or not self.synthesizer:
            return

        voice_to_use = voice or self.voice
        try:
            speak_response(text, voice_to_use)
        except Exception as e:
            print(f"⚠️  Voice output failed: {e}")

    def set_voice(self, voice: str):
        """Change voice."""
        if voice in CHATGPT_VOICES:
            self.voice = voice
            print(f"🎭 Voice changed to {voice}: {CHATGPT_VOICES[voice]['description']}")
        else:
            print(f"⚠️  Unknown voice: {voice}")

    def list_voices(self):
        """List available voices."""
        return list(CHATGPT_VOICES.keys())

    def enable(self):
        """Enable voice output."""
        self.enabled = True
        if not self.synthesizer:
            self.synthesizer = get_speech_synthesizer(self.engine)
        print("🔊 Voice output enabled")

    def disable(self):
        """Disable voice output."""
        self.enabled = False
        print("🔇 Voice output disabled")

    def is_enabled(self) -> bool:
        """Check if voice output is enabled."""
        return self.enabled and self.synthesizer is not None


# Global voice output instance
_voice_output = None

def get_voice_output(voice: str = "alloy", engine: str = "auto", enabled: bool = True):
    """Get or create global voice output instance."""
    global _voice_output

    if _voice_output is None:
        _voice_output = VoiceOutput(voice, engine, enabled)

    return _voice_output

def speak_text(text: str, voice: str = "alloy"):
    """Convenience function to speak text."""
    voice_output = get_voice_output()
    voice_output.speak(text, voice)
