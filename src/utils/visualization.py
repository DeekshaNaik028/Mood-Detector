"""
Visualization utilities
"""

import cv2
import numpy as np


class Visualizer:
    """Handles visualization of emotion detection results"""
    
    @staticmethod
    def draw_face_box(frame, bbox, emotion, confidence):
        """Draw bounding box and emotion text on frame"""
        if bbox is None:
            return frame
        
        x, y, w, h = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text
        text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    @staticmethod
    def create_info_panel(width, face_emotion, face_conf, voice_emotion, voice_conf, 
                         final_emotion, final_conf):
        """Create information panel"""
        panel_height = 150
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        y_offset = 30
        
        if face_emotion:
            text = f"Face: {face_emotion} ({face_conf:.2f})"
            cv2.putText(panel, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        
        if voice_emotion:
            text = f"Voice: {voice_emotion} ({voice_conf:.2f})"
            cv2.putText(panel, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y_offset += 30
        
        if final_emotion:
            text = f"FINAL: {final_emotion} ({final_conf:.2f})"
            cv2.putText(panel, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return panel
    
    @staticmethod
    def combine_frame_panel(frame, panel):
        """Combine frame and panel vertically"""
        return np.vstack([frame, panel])