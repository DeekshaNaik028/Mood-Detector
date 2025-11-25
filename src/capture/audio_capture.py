"""
Audio capture module
"""

import sounddevice as sd
import queue
from ..config import Config


class AudioCapture:
    """Handles audio capture from microphone"""
    
    def __init__(self, sample_rate=None, duration=None):
        self.sample_rate = sample_rate or Config.SAMPLE_RATE
        self.duration = duration or Config.AUDIO_DURATION
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
    
    def callback(self, indata, frames, time, status):
        """Audio callback function"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def start(self):
        """Start audio capture"""
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=Config.AUDIO_CHANNELS,
            callback=self.callback,
            blocksize=int(self.sample_rate * self.duration)
        )
        self.stream.start()
    
    def stop(self):
        """Stop audio capture"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
    
    def get_audio(self):
        """Get latest audio chunk"""
        if not self.audio_queue.empty():
            audio = self.audio_queue.get()
            return audio.flatten()
        return None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()