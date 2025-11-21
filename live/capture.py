"""
Live Input Capture for SRGI Mini-AGI

Captures real-time multimodal input streams:
- Webcam frames (rolling canvas)
- Microphone audio (16kHz mono)
- Screen capture
- Video files

Unified preprocessing pipeline for vision, audio, and text inputs.
"""

import torch
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Iterator
import cv2

# Optional imports
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


class WebcamCapture:
    """Real-time webcam capture with rolling canvas."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.thread = None

    def start(self):
        """Start webcam capture."""
        if self.cap is not None:
            return

        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam device {self.device_id}")

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        print(f"🎥 Webcam capture started: {self.width}x{self.height}@{self.fps}fps")

    def stop(self):
        """Stop webcam capture."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("🎥 Webcam capture stopped")

    def _capture_loop(self):
        """Main capture loop."""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        # Remove oldest frame
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, timeout=0.1)
                        except:
                            pass
            time.sleep(1.0 / self.fps)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_frame_tensor(self) -> Optional[torch.Tensor]:
        """Get latest frame as tensor."""
        frame = self.get_frame()
        if frame is not None:
            # Convert BGR to RGB and to tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb).float() / 255.0
            # HWC to CHW
            tensor = tensor.permute(2, 0, 1)
            return tensor
        return None


class AudioCapture:
    """Real-time microphone audio capture."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        self.audio = None
        self.stream = None
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=50)
        self.thread = None

        if not PYAUDIO_AVAILABLE:
            print("⚠️  pyaudio not available. Install with: pip install pyaudio")
            return

    def start(self):
        """Start audio capture."""
        if self.audio is not None or not PYAUDIO_AVAILABLE:
            return

        self.audio = pyaudio.PyAudio()

        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

            print(f"🎤 Audio capture started: {self.sample_rate}Hz, {self.channels}ch")

        except Exception as e:
            print(f"❌ Audio capture failed: {e}")
            self.audio = None

    def stop(self):
        """Stop audio capture."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        print("🎤 Audio capture stopped")

    def _capture_loop(self):
        """Main audio capture loop."""
        while self.is_running and self.stream:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                try:
                    self.audio_queue.put(audio_chunk, timeout=0.1)
                except queue.Full:
                    # Remove oldest chunk
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put(audio_chunk, timeout=0.1)
                    except:
                        pass

            except Exception as e:
                print(f"Audio capture error: {e}")
                break

    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get latest audio chunk."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def get_audio_tensor(self) -> Optional[torch.Tensor]:
        """Get latest audio chunk as tensor."""
        chunk = self.get_audio_chunk()
        if chunk is not None:
            return torch.from_numpy(chunk).float()
        return None


class MultimodalCapture:
    """
    Unified multimodal input capture system.

    Manages webcam, microphone, and provides unified interface
    for SRGI mini-AGI input processing.
    """

    def __init__(
        self,
        webcam_device: int = 0,
        mic_sample_rate: int = 16000,
        enable_webcam: bool = True,
        enable_microphone: bool = True
    ):
        self.enable_webcam = enable_webcam
        self.enable_microphone = enable_microphone

        self.webcam = None
        self.microphone = None

        if enable_webcam:
            self.webcam = WebcamCapture(device_id=webcam_device)

        if enable_microphone:
            self.microphone = AudioCapture(sample_rate=mic_sample_rate)

    def start(self):
        """Start all enabled capture devices."""
        print("🎬 Starting multimodal capture...")

        if self.webcam:
            self.webcam.start()

        if self.microphone:
            self.microphone.start()

        print("✅ Multimodal capture active")

    def stop(self):
        """Stop all capture devices."""
        print("🛑 Stopping multimodal capture...")

        if self.webcam:
            self.webcam.stop()

        if self.microphone:
            self.microphone.stop()

        print("✅ Multimodal capture stopped")

    def get_current_frame(self) -> Optional[torch.Tensor]:
        """Get current webcam frame as tensor."""
        if self.webcam:
            return self.webcam.get_frame_tensor()
        return None

    def get_current_audio(self) -> Optional[torch.Tensor]:
        """Get current audio chunk as tensor."""
        if self.microphone:
            return self.microphone.get_audio_tensor()
        return None

    def get_multimodal_input(self) -> dict:
        """
        Get current multimodal input state.

        Returns dict with 'vision' and 'audio' tensors ready for SRGI processing.
        """
        vision = self.get_current_frame()
        audio = self.get_current_audio()

        return {
            'vision': vision,      # (C, H, W) tensor or None
            'audio': audio,        # (samples,) tensor or None
            'timestamp': time.time()
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Global capture instance
_capture_instance = None

def get_multimodal_capture(
    webcam_device: int = 0,
    mic_sample_rate: int = 16000,
    enable_webcam: bool = True,
    enable_microphone: bool = True
) -> MultimodalCapture:
    """Get or create global multimodal capture instance."""
    global _capture_instance

    if _capture_instance is None:
        _capture_instance = MultimodalCapture(
            webcam_device=webcam_device,
            mic_sample_rate=mic_sample_rate,
            enable_webcam=enable_webcam,
            enable_microphone=enable_microphone
        )

    return _capture_instance


def capture_demo():
    """Demo function showing multimodal capture."""
    print("🎬 Multimodal Capture Demo")
    print("This will capture webcam and microphone for 10 seconds...")

    with MultimodalCapture() as capture:
        start_time = time.time()

        while time.time() - start_time < 10:
            inputs = capture.get_multimodal_input()

            if inputs['vision'] is not None:
                print("📸 Captured video frame"
            if inputs['audio'] is not None:
                print("🎤 Captured audio chunk"
            time.sleep(1.0)

    print("✅ Demo complete!")


if __name__ == "__main__":
    capture_demo()
