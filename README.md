# 🎭 Multimodal Mood Detection System

A real-time emotion recognition system that combines facial expression analysis and voice emotion detection for accurate mood assessment.

## 🌟 Features

- **Dual Modality Detection**: Combines face and voice analysis
- **Real-time Processing**: Live webcam and microphone input
- **7 Emotion Classes**: Neutral, Happy, Sad, Angry, Fearful, Surprised, Disgusted
- **Confidence Scoring**: Provides prediction confidence for each modality
- **Emotion History Tracking**: Records and analyzes mood over time
- **Visual Interface**: Interactive display with bounding boxes and info panels

## 📋 Requirements

- Python 3.8+
- Webcam
- Microphone
- CUDA-capable GPU (optional, for faster face detection)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd mood-detection-system
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets (for training)

**Face Dataset**: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) or [AffectNet](http://mohammadmahoor.com/affectnet/)

**Voice Dataset**: [RAVDESS](https://zenodo.org/record/1188976)

Organize datasets:
```
data/
├── face_emotions/
│   ├── neutral/
│   ├── happy/
│   ├── sad/
│   └── ...
└── voice_emotions/
    ├── neutral/
    ├── happy/
    ├── sad/
    └── ...
```

## 🎓 Training Models

### Train Face Emotion Model
```python
from training.train_face import FaceTrainer

trainer = FaceTrainer('data/face_emotions')
trainer.train()
```

### Train Voice Emotion Model
```python
from training.train_voice import VoiceTrainer

trainer = VoiceTrainer('data/voice_emotions')
trainer.train()
```

### Organize RAVDESS Dataset
```bash
python organize_ravdess_simple.py
```

## 🎮 Usage

### Run the Main System
```bash
python main.py
```

### Keyboard Controls
- **Q**: Quit the application
- **S**: Save emotion history to JSON
- **R**: Show mood summary report

### Run Demo
```bash
python demo.py
```

## 📊 Model Architecture

### Face Emotion CNN
- 4 Convolutional blocks with batch normalization
- MaxPooling and dropout for regularization
- 3 Fully connected layers
- Input: 48x48 grayscale images
- Output: 7 emotion classes

### Voice Emotion Model
- Feature extraction: MFCC, pitch, energy, ZCR, spectral features
- Classifier: Random Forest (default) or SVM
- Feature normalization with StandardScaler

### Fusion Strategy
- Weighted combination of predictions
- Default weights: Face (0.6), Voice (0.4)
- Confidence-based final decision

## 📁 Project Structure

```
mood-detection-system/
├── src/
│   ├── capture/          # Video and audio capture
│   ├── detectors/        # Face and voice emotion detectors
│   ├── features/         # Audio feature extraction
│   ├── fusion/           # Multimodal fusion
│   ├── models/           # Neural network architectures
│   └── utils/            # Visualization and data utilities
├── training/             # Training scripts and datasets
├── models/               # Saved model weights
├── data/                 # Training data
├── main.py               # Main application
├── demo.py               # Demo script
└── requirements.txt      # Dependencies
```

## 🔧 Configuration

Edit `src/config.py` to customize:
- Emotion labels
- Model paths
- Audio/video settings
- Fusion weights
- Training hyperparameters

## 📈 Evaluation

```python
from training.evaluate import ModelEvaluator

# Evaluate face model
ModelEvaluator.evaluate_face_model(
    'models/face_emotion_model.pth',
    'data/face_emotions_test'
)

# Evaluate voice model
ModelEvaluator.evaluate_voice_model(
    'models/voice_emotion_model.pkl',
    'models/audio_scaler.pkl',
    'data/voice_emotions_test'
)
```

## 🎯 Performance Tips

1. **Lighting**: Ensure good lighting for face detection
2. **Audio Quality**: Use a decent microphone for voice detection
3. **Distance**: Stay 1-2 feet from the webcam
4. **Background Noise**: Minimize for better voice emotion accuracy

## 🐛 Troubleshooting

### Camera not working
```python
# Try different camera indices in src/config.py
CAMERA_INDEX = 1  # or 2, 3, etc.
```

### Audio issues
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Low accuracy
- Train on more diverse data
- Adjust fusion weights in `src/config.py`
- Fine-tune model hyperparameters

## 📝 Citation

If you use this project, please cite:
```bibtex
@software{mood_detection_system,
  title={Multimodal Mood Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mood-detection-system}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 🙏 Acknowledgments

- FER-2013 dataset creators
- RAVDESS dataset creators
- PyTorch and scikit-learn communities

## 📧 Contact

For questions or feedback, please open an issue or contact [your.email@example.com]

---
⭐ Star this repo if you find it helpful!