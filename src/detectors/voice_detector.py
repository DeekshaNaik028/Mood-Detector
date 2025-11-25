"""
Voice emotion detection module
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

from ..features.audio_features import AudioFeatureExtractor
from ..models.voice_model import VoiceEmotionModel
from ..config import Config


class VoiceEmotionDetector:
    """Voice emotion detection"""
    
    def __init__(self, model_path=None, scaler_path=None):
        self.extractor = AudioFeatureExtractor(sample_rate=Config.SAMPLE_RATE)
        
        # Load or create model
        if model_path and Path(model_path).exists():
            self.model = VoiceEmotionModel.load(model_path)
            print(f"Loaded voice model from {model_path}")
            self.is_trained = True
        else:
            self.model = VoiceEmotionModel(model_type='random_forest')
            print("Using untrained voice model. Train before use!")
            self.is_trained = False
        
        # Load or create scaler
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = StandardScaler()
    
    def predict_emotion(self, audio):
        """Predict emotion from audio"""
        if not self.is_trained:
            return None, 0.0
        
        # Extract features
        features = self.extractor.extract_all_features(audio)
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return Config.EMOTIONS[prediction], confidence