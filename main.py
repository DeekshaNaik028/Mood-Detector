"""
Main entry point for the mood detection system
"""

import cv2
from collections import deque
from datetime import datetime

from src.config import Config
from src.detectors.face_detector import FaceEmotionDetector
from src.detectors.voice_detector import VoiceEmotionDetector
from src.fusion.emotion_fusion import EmotionFusion
from src.capture.video_capture import VideoCapture
from src.capture.audio_capture import AudioCapture
from src.utils.visualization import Visualizer
from src.utils.data_utils import save_history, get_mood_summary


class MoodDetectionSystem:
    """Main system coordinating all components"""
    
    def __init__(self):
        print("Initializing Mood Detection System...")
        
        # Initialize components
        self.face_detector = FaceEmotionDetector(Config.FACE_MODEL_PATH)
        self.voice_detector = VoiceEmotionDetector(
            Config.VOICE_MODEL_PATH, 
            Config.SCALER_PATH
        )
        self.fusion = EmotionFusion()
        self.visualizer = Visualizer()
        
        # Capture devices
        self.video_capture = None
        self.audio_capture = None
        
        # History tracking
        self.emotion_history = deque(maxlen=Config.HISTORY_LENGTH)
        self.timestamp_history = deque(maxlen=Config.HISTORY_LENGTH)
        
        self.is_running = False
        
        print("System initialized!")
    
    def start(self):
        """Start the detection system"""
        print("\n" + "="*60)
        print("STARTING MOOD DETECTION SYSTEM")
        print("="*60)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save history")
        print("  - Press 'r' to show summary report")
        print("="*60 + "\n")
        
        try:
            # Start video
            self.video_capture = VideoCapture(Config.CAMERA_INDEX)
            self.video_capture.start()
            
            # Start audio
            self.audio_capture = AudioCapture()
            self.audio_capture.start()
            
            self.is_running = True
            
            # Main loop
            self.run_detection_loop()
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def run_detection_loop(self):
        """Main detection loop"""
        while self.is_running:
            # Capture frame
            frame = self.video_capture.read_frame()
            if frame is None:
                break
            
            # Detect face emotion
            face, bbox = self.face_detector.detect_face(frame)
            face_emotion, face_conf = self.face_detector.predict_emotion(face)
            
            # Detect voice emotion
            audio = self.audio_capture.get_audio()
            voice_emotion, voice_conf = None, 0.0
            if audio is not None and len(audio) > 0:
                voice_emotion, voice_conf = self.voice_detector.predict_emotion(audio)
            
            # Fuse predictions
            final_emotion, final_conf = self.fusion.fuse_predictions(
                face_emotion, face_conf, voice_emotion, voice_conf
            )
            
            # Record history
            if final_emotion:
                self.emotion_history.append(final_emotion)
                self.timestamp_history.append(datetime.now())
            
            # Visualize
            self.visualize(
                frame, bbox, face_emotion, face_conf,
                voice_emotion, voice_conf, final_emotion, final_conf
            )
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_history(self.emotion_history, self.timestamp_history)
            elif key == ord('r'):
                self.show_summary()
    
    def visualize(self, frame, bbox, face_emotion, face_conf,
                  voice_emotion, voice_conf, final_emotion, final_conf):
        """Visualize results"""
        # Draw face box
        frame = self.visualizer.draw_face_box(frame, bbox, face_emotion or "No face", face_conf or 0)
        
        # Create info panel
        h, w = frame.shape[:2]
        panel = self.visualizer.create_info_panel(
            w, face_emotion, face_conf, voice_emotion, voice_conf,
            final_emotion, final_conf
        )
        
        # Combine and display
        combined = self.visualizer.combine_frame_panel(frame, panel)
        cv2.imshow('Mood Detection System', combined)
    
    def show_summary(self):
        """Display mood summary"""
        summary = get_mood_summary(self.emotion_history)
        if summary:
            print("\n" + "="*60)
            print("MOOD SUMMARY")
            print("="*60)
            print(f"Total samples: {summary['total_samples']}")
            print(f"Dominant emotion: {summary['dominant_emotion']}")
            print("\nDistribution:")
            for emotion, percentage in sorted(
                summary['distribution'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                bar = '█' * int(percentage * 50)
                print(f"  {emotion:>10}: {bar} {percentage:.1%}")
            print("="*60 + "\n")
    
    def stop(self):
        """Stop the system"""
        print("\nStopping system...")
        self.is_running = False
        
        if self.video_capture:
            self.video_capture.stop()
        
        if self.audio_capture:
            self.audio_capture.stop()
        
        cv2.destroyAllWindows()
        print("System stopped")


def main():
    """Main function"""
    try:
        system = MoodDetectionSystem()
        system.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()