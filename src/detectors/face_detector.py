"""
Face emotion detection module
"""

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

from ..models.face_model import FaceEmotionCNN
from ..config import Config


class FaceEmotionDetector:
    """Facial emotion detection"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FaceEmotionCNN(num_classes=Config.NUM_EMOTIONS)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded face model from {model_path}")
        else:
            print("Using randomly initialized face model. Train before use!")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(Config.FACE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def detect_face(self, frame):
        """Detect face in frame and return cropped face"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face = gray[y:y+h, x:x+w]
            return face, (x, y, w, h)
        
        return None, None
    
    def predict_emotion(self, face):
        """Predict emotion from face image"""
        if face is None:
            return None, None
        
        # Preprocess
        face_tensor = self.transform(face).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion_idx = predicted.item()
        confidence_score = confidence.item()
        
        return Config.EMOTIONS[emotion_idx], confidence_score