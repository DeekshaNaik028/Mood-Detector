from pathlib import Path


class Config:
    """System configuration"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    MODEL_DIR = PROJECT_ROOT / 'models'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    # Emotion labels
    EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted']
    NUM_EMOTIONS = len(EMOTIONS)
    
    # Audio settings
    SAMPLE_RATE = 22050
    AUDIO_DURATION = 3  # seconds for each analysis window
    AUDIO_CHANNELS = 1
    
    # Video settings
    FACE_SIZE = (48, 48)
    VIDEO_FPS = 30
    CAMERA_INDEX = 0
    
    # Model paths
    FACE_MODEL_PATH = MODEL_DIR / 'face_emotion_model.pth'
    VOICE_MODEL_PATH = MODEL_DIR / 'voice_emotion_model.pkl'
    SCALER_PATH = MODEL_DIR / 'audio_scaler.pkl'
    
    # Fusion weights
    FACE_WEIGHT = 0.6
    VOICE_WEIGHT = 0.4
    
    # Tracking
    HISTORY_LENGTH = 100  # frames to keep in history
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)


# Initialize directories
Config.create_directories()