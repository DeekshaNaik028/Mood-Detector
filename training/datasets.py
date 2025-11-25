"""
Dataset classes for training
"""

import torch
from torch.utils.data import Dataset
import cv2
import librosa
from pathlib import Path
import numpy as np

from src.config import Config
from src.features.audio_features import AudioFeatureExtractor


class FaceEmotionDataset(Dataset):
    """
    Dataset for facial emotion images
    
    Expected directory structure:
    data/face_emotions/
        neutral/
            image1.jpg
            image2.jpg
        happy/
            image1.jpg
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images and labels
        for emotion_idx, emotion in enumerate(Config.EMOTIONS):
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            image_files = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(emotion_idx)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class VoiceEmotionDataset:
    """
    Dataset for voice emotion audio files
    
    Expected directory structure:
    data/voice_emotions/
        neutral/
            audio1.wav
        happy/
            audio1.wav
    """
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.extractor = AudioFeatureExtractor(sample_rate=Config.SAMPLE_RATE)
        self.features = []
        self.labels = []
        
        print(f"Loading audio dataset from {root_dir}")
        print("Extracting features...")
        
        for emotion_idx, emotion in enumerate(Config.EMOTIONS):
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                print(f"Warning: {emotion_dir} does not exist")
                continue
            
            audio_files = list(emotion_dir.glob('*.wav')) + list(emotion_dir.glob('*.mp3'))
            
            for i, audio_path in enumerate(audio_files):
                try:
                    # Load audio
                    audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
                    
                    # Extract features
                    features = self.extractor.extract_all_features(audio)
                    
                    self.features.append(features)
                    self.labels.append(emotion_idx)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  {emotion}: {i+1}/{len(audio_files)} processed")
                        
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        print(f"Extracted features from {len(self.features)} audio files")
        print(f"Feature shape: {self.features.shape}")