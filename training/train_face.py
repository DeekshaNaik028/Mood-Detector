"""
Face emotion model training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path

from src.config import Config
from src.models.face_model import FaceEmotionCNN
from .datasets import FaceEmotionDataset


class FaceTrainer:
    """Trainer for face emotion model"""
    
    def __init__(self, data_dir, batch_size=None, lr=None, epochs=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size or Config.BATCH_SIZE
        self.lr = lr or Config.LEARNING_RATE
        self.epochs = epochs or Config.EPOCHS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
    
    def prepare_data(self):
        """Prepare training and validation data"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(Config.FACE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load dataset
        full_dataset = FaceEmotionDataset(self.data_dir, transform=transform)
        
        # Split into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self, save_path=None):
        """Train the model"""
        print("="*60)
        print("TRAINING FACE EMOTION MODEL")
        print("="*60)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        # Initialize model
        model = FaceEmotionCNN(num_classes=Config.NUM_EMOTIONS).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_val_acc = 0
        save_path = save_path or Config.FACE_MODEL_PATH
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{self.epochs}:')
            print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f'  ✓ Saved best model (val_acc: {val_acc:.2f}%)')
        
        print("="*60)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Model saved to: {save_path}")
        print("="*60)
        
        return model