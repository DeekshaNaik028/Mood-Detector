"""
Model evaluation utilities
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pickle

from src.config import Config
from src.models.face_model import FaceEmotionCNN
from .datasets import FaceEmotionDataset, VoiceEmotionDataset


class ModelEvaluator:
    """Evaluate trained models"""
    
    @staticmethod
    def evaluate_face_model(model_path, test_data_dir):
        """Evaluate face emotion model"""
        print("="*60)
        print("EVALUATING FACE EMOTION MODEL")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FaceEmotionCNN(num_classes=Config.NUM_EMOTIONS)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(Config.FACE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        test_dataset = FaceEmotionDataset(test_data_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"\nTest Accuracy: {accuracy*100:.2f}%\n")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(ModelEvaluator._format_confusion_matrix(cm))
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            all_labels, all_preds, 
            target_names=Config.EMOTIONS, 
            digits=3
        ))
        print("="*60)
    
    @staticmethod
    def evaluate_voice_model(model_path, scaler_path, test_data_dir):
        """Evaluate voice emotion model"""
        print("="*60)
        print("EVALUATING VOICE EMOTION MODEL")
        print("="*60)
        
        # Load model and scaler
        from src.models.voice_model import VoiceEmotionModel
        model = VoiceEmotionModel.load(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load test data
        test_dataset = VoiceEmotionDataset(test_data_dir)
        
        if len(test_dataset.features) == 0:
            print("No test data found!")
            return
        
        # Scale and predict
        X_test_scaled = scaler.transform(test_dataset.features)
        y_pred = model.predict(X_test_scaled)
        y_true = test_dataset.labels
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        print(f"\nTest Accuracy: {accuracy*100:.2f}%\n")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(ModelEvaluator._format_confusion_matrix(cm))
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred, 
            target_names=Config.EMOTIONS, 
            digits=3
        ))
        print("="*60)
    
    @staticmethod
    def _format_confusion_matrix(cm):
        """Format confusion matrix for display"""
        lines = []
        
        # Header
        header = "     " + "".join([f"{e[:3]:>6}" for e in Config.EMOTIONS])
        lines.append(header)
        
        # Rows
        for i, emotion in enumerate(Config.EMOTIONS):
            row = f"{emotion[:3]:>5}"
            for j in range(len(Config.EMOTIONS)):
                row += f"{cm[i][j]:>6}"
            lines.append(row)
        
        return "\n".join(lines)