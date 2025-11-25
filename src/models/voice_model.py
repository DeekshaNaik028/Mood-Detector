"""
Voice emotion recognition model
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle


class VoiceEmotionModel:
    """Wrapper for voice emotion classification models"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize voice emotion model
        
        Args:
            model_type: 'random_forest' or 'svm'
        """
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """Train the model"""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Predict emotion labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict emotion probabilities"""
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        return self.model.score(X, y)
    
    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)