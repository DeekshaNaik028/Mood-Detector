"""
Voice emotion model training
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from src.config import Config
from src.models.voice_model import VoiceEmotionModel
from .datasets import VoiceEmotionDataset


class VoiceTrainer:
    """Trainer for voice emotion model"""
    
    def __init__(self, data_dir, model_type='random_forest'):
        self.data_dir = Path(data_dir)
        self.model_type = model_type
    
    def train(self, save_model_path=None, save_scaler_path=None):
        """Train the model"""
        print("="*60)
        print("TRAINING VOICE EMOTION MODEL")
        print("="*60)
        
        # Load dataset
        dataset = VoiceEmotionDataset(self.data_dir)
        
        if len(dataset.features) == 0:
            print("Error: No audio files found!")
            return None, None
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            dataset.features, 
            dataset.labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=dataset.labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        print("Training model...")
        model = VoiceEmotionModel(model_type=self.model_type)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = model.score(X_train_scaled, y_train)
        val_acc = model.score(X_val_scaled, y_val)
        
        print(f'Training Accuracy: {train_acc*100:.2f}%')
        print(f'Validation Accuracy: {val_acc*100:.2f}%')
        
        # Save model and scaler
        save_model_path = save_model_path or Config.VOICE_MODEL_PATH
        save_scaler_path = save_scaler_path or Config.SCALER_PATH
        
        model.save(save_model_path)
        with open(save_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print()
        print(f"Model saved to: {save_model_path}")
        print(f"Scaler saved to: {save_scaler_path}")
        print("="*60)
        
        return model, scaler