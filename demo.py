"""
Quick demo script for mood detection system
Tests individual components without full system startup
"""

import cv2
import numpy as np
import torch
from pathlib import Path

from src.config import Config
from src.detectors.face_detector import FaceEmotionDetector
from src.detectors.voice_detector import VoiceEmotionDetector
from src.features.audio_features import AudioFeatureExtractor


def demo_face_detection():
    """Demo face emotion detection on sample image"""
    print("\n" + "="*60)
    print("FACE EMOTION DETECTION DEMO")
    print("="*60)
    
    # Check if model exists
    if not Config.FACE_MODEL_PATH.exists():
        print("❌ Face model not found. Please train the model first.")
        print(f"Expected path: {Config.FACE_MODEL_PATH}")
        return
    
    # Initialize detector
    detector = FaceEmotionDetector(Config.FACE_MODEL_PATH)
    
    # Try to capture from webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("✓ Webcam opened")
    print("\nPress 'q' to quit, 'c' to capture and analyze\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect face
        face, bbox = detector.detect_face(frame)
        
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection Demo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and face is not None:
            # Analyze emotion
            emotion, confidence = detector.predict_emotion(face)
            print(f"Detected: {emotion} (confidence: {confidence:.2%})")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Face detection demo complete")


def demo_voice_detection():
    """Demo voice emotion detection"""
    print("\n" + "="*60)
    print("VOICE EMOTION DETECTION DEMO")
    print("="*60)
    
    # Check if model exists
    if not Config.VOICE_MODEL_PATH.exists():
        print("❌ Voice model not found. Please train the model first.")
        print(f"Expected path: {Config.VOICE_MODEL_PATH}")
        return
    
    # Initialize detector
    detector = VoiceEmotionDetector(Config.VOICE_MODEL_PATH, Config.SCALER_PATH)
    
    if not detector.is_trained:
        print("❌ Model not trained properly")
        return
    
    print("✓ Voice model loaded")
    
    # Try to record audio
    try:
        import sounddevice as sd
        
        print("\nRecording 3 seconds of audio...")
        print("Speak now!")
        
        audio = sd.rec(
            int(Config.AUDIO_DURATION * Config.SAMPLE_RATE),
            samplerate=Config.SAMPLE_RATE,
            channels=1
        )
        sd.wait()
        
        print("Recording complete. Analyzing...")
        
        audio = audio.flatten()
        emotion, confidence = detector.predict_emotion(audio)
        
        print(f"\nDetected emotion: {emotion}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"❌ Error recording audio: {e}")
        print("Make sure you have a microphone connected")
    
    print("\n✓ Voice detection demo complete")


def demo_feature_extraction():
    """Demo audio feature extraction"""
    print("\n" + "="*60)
    print("AUDIO FEATURE EXTRACTION DEMO")
    print("="*60)
    
    extractor = AudioFeatureExtractor(sample_rate=Config.SAMPLE_RATE)
    
    # Generate sample audio (sine wave)
    duration = 3
    t = np.linspace(0, duration, int(Config.SAMPLE_RATE * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    print("Extracting features from sample audio...")
    
    features = extractor.extract_all_features(audio)
    
    print(f"\n✓ Extracted {len(features)} features")
    print(f"Feature vector shape: {features.shape}")
    
    # Show some features
    print("\nSample features:")
    print(f"  - MFCC features: 26 values")
    print(f"  - Pitch features: 4 values")
    print(f"  - Energy features: 2 values")
    print(f"  - ZCR features: 2 values")
    print(f"  - Spectral features: 3 values")
    print(f"  - Tempo: 1 value")
    
    print("\n✓ Feature extraction demo complete")


def demo_system_check():
    """Check if system is ready to run"""
    print("\n" + "="*60)
    print("SYSTEM READINESS CHECK")
    print("="*60)
    
    checks = {
        "Face Model": Config.FACE_MODEL_PATH.exists(),
        "Voice Model": Config.VOICE_MODEL_PATH.exists(),
        "Scaler": Config.SCALER_PATH.exists(),
        "Models Directory": Config.MODEL_DIR.exists(),
        "Data Directory": Config.DATA_DIR.exists(),
    }
    
    print("\nComponent Status:")
    all_ready = True
    for component, status in checks.items():
        icon = "✓" if status else "❌"
        print(f"  {icon} {component}: {'Ready' if status else 'Missing'}")
        if not status:
            all_ready = False
    
    print("\nLibrary Checks:")
    libraries = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("librosa", "Librosa"),
        ("sounddevice", "SoundDevice"),
        ("sklearn", "scikit-learn")
    ]
    
    for lib_name, display_name in libraries:
        try:
            __import__(lib_name)
            print(f"  ✓ {display_name}: Installed")
        except ImportError:
            print(f"  ❌ {display_name}: Not installed")
            all_ready = False
    
    print("\nHardware Checks:")
    
    # Check webcam
    cap = cv2.VideoCapture(0)
    webcam_ok = cap.isOpened()
    cap.release()
    print(f"  {'✓' if webcam_ok else '❌'} Webcam: {'Available' if webcam_ok else 'Not found'}")
    
    # Check audio devices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        mic_ok = any('input' in str(d).lower() for d in str(devices).lower().split('\n'))
        print(f"  {'✓' if mic_ok else '❌'} Microphone: {'Available' if mic_ok else 'Not found'}")
    except:
        print("  ❌ Microphone: Cannot check")
        mic_ok = False
    
    print("\n" + "="*60)
    if all_ready and webcam_ok and mic_ok:
        print("✓ SYSTEM READY! Run 'python main.py' to start")
    else:
        print("⚠ SYSTEM NOT READY")
        if not checks["Face Model"] or not checks["Voice Model"]:
            print("\nNext steps:")
            print("  1. Prepare training data in data/ directory")
            print("  2. Run training scripts:")
            print("     from training.train_face import FaceTrainer")
            print("     from training.train_voice import VoiceTrainer")
    print("="*60)


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("MOOD DETECTION SYSTEM - DEMO")
    print("="*60)
    print("\nAvailable demos:")
    print("  1. System Readiness Check")
    print("  2. Face Detection Demo")
    print("  3. Voice Detection Demo")
    print("  4. Feature Extraction Demo")
    print("  5. Run All Demos")
    print("  0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-5): ").strip()
            
            if choice == "0":
                print("Exiting...")
                break
            elif choice == "1":
                demo_system_check()
            elif choice == "2":
                demo_face_detection()
            elif choice == "3":
                demo_voice_detection()
            elif choice == "4":
                demo_feature_extraction()
            elif choice == "5":
                demo_system_check()
                input("\nPress Enter to continue to face detection...")
                demo_face_detection()
                input("\nPress Enter to continue to voice detection...")
                demo_voice_detection()
                input("\nPress Enter to continue to feature extraction...")
                demo_feature_extraction()
            else:
                print("Invalid choice. Please select 0-5.")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()