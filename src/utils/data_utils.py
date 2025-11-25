"""
Data handling utilities
"""

import json
from datetime import datetime
from collections import Counter


def save_history(emotion_history, timestamp_history, filename=None):
    """Save emotion history to JSON file"""
    if len(emotion_history) == 0:
        print("No history to save")
        return None
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mood_history_{timestamp}.json"
    
    history_data = {
        'timestamps': [t.isoformat() for t in timestamp_history],
        'emotions': list(emotion_history)
    }
    
    with open(filename, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"History saved to {filename}")
    return filename


def load_history(filename):
    """Load emotion history from JSON file"""
    with open(filename, 'r') as f:
        history_data = json.load(f)
    
    timestamps = [datetime.fromisoformat(t) for t in history_data['timestamps']]
    emotions = history_data['emotions']
    
    return emotions, timestamps


def get_mood_summary(emotion_history):
    """Get summary statistics of emotion history"""
    if len(emotion_history) == 0:
        return None
    
    emotion_counts = Counter(emotion_history)
    total = len(emotion_history)
    
    summary = {
        'total_samples': total,
        'distribution': {
            emotion: count / total 
            for emotion, count in emotion_counts.items()
        },
        'dominant_emotion': emotion_counts.most_common(1)[0][0],
        'unique_emotions': len(emotion_counts)
    }
    
    return summary