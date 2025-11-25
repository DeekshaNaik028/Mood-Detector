"""
Multimodal emotion fusion
"""

import numpy as np
from ..config import Config


class EmotionFusion:
    """Fuse predictions from multiple modalities"""
    
    def __init__(self, face_weight=None, voice_weight=None):
        self.face_weight = face_weight or Config.FACE_WEIGHT
        self.voice_weight = voice_weight or Config.VOICE_WEIGHT
        self.emotions = Config.EMOTIONS
    
    def fuse_predictions(self, face_emotion, face_conf, voice_emotion, voice_conf):
        """
        Fuse face and voice predictions
        
        Args:
            face_emotion: Predicted emotion from face
            face_conf: Confidence score for face prediction
            voice_emotion: Predicted emotion from voice
            voice_conf: Confidence score for voice prediction
            
        Returns:
            final_emotion: Fused emotion prediction
            final_confidence: Confidence score for fused prediction
        """
        if face_emotion is None and voice_emotion is None:
            return None, 0.0
        
        if face_emotion is None:
            return voice_emotion, voice_conf
        
        if voice_emotion is None:
            return face_emotion, face_conf
        
        # Create probability distributions
        face_probs = np.zeros(len(self.emotions))
        voice_probs = np.zeros(len(self.emotions))
        
        face_idx = self.emotions.index(face_emotion)
        voice_idx = self.emotions.index(voice_emotion)
        
        face_probs[face_idx] = face_conf
        voice_probs[voice_idx] = voice_conf
        
        # Weighted fusion
        fused_probs = (self.face_weight * face_probs + 
                       self.voice_weight * voice_probs)
        
        # Get final prediction
        final_idx = np.argmax(fused_probs)
        final_emotion = self.emotions[final_idx]
        final_confidence = fused_probs[final_idx]
        
        return final_emotion, final_confidence