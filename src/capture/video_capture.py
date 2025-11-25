"""
Video capture module
"""

import cv2
from ..config import Config


class VideoCapture:
    """Handles video capture from webcam"""
    
    def __init__(self, camera_index=None):
        self.camera_index = camera_index or Config.CAMERA_INDEX
        self.capture = None
        self.is_opened = False
    
    def start(self):
        """Start video capture"""
        self.capture = cv2.VideoCapture(self.camera_index)
        self.is_opened = self.capture.isOpened()
        
        if not self.is_opened:
            raise Exception(f"Could not open camera {self.camera_index}")
        
        return self.is_opened
    
    def read_frame(self):
        """Read a frame from the camera"""
        if not self.is_opened:
            return None
        
        ret, frame = self.capture.read()
        return frame if ret else None
    
    def stop(self):
        """Stop video capture"""
        if self.capture:
            self.capture.release()
            self.is_opened = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()