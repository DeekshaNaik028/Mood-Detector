"""
Audio feature extraction
"""

import numpy as np
import librosa


class AudioFeatureExtractor:
    """Extract acoustic features from audio"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def extract_mfcc(self, audio, n_mfcc=13):
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        return {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1)
        }
    
    def extract_pitch(self, audio):
        """Extract pitch features"""
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            return {
                'pitch_mean': np.mean(pitch_values),
                'pitch_std': np.std(pitch_values),
                'pitch_min': np.min(pitch_values),
                'pitch_max': np.max(pitch_values)
            }
        else:
            return {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_min': 0,
                'pitch_max': 0
            }
    
    def extract_energy(self, audio):
        """Extract energy features"""
        energy = librosa.feature.rms(y=audio)
        return {
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy)
        }
    
    def extract_zcr(self, audio):
        """Extract zero crossing rate"""
        zcr = librosa.feature.zero_crossing_rate(audio)
        return {
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr)
        }
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff)
        }
    
    def extract_tempo(self, audio):
        """Extract tempo"""
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        return {'tempo': tempo}
    
    def extract_all_features(self, audio):
        """Extract all features and return as a single vector"""
        features = {}
        
        # Extract all feature types
        features.update(self.extract_mfcc(audio))
        features.update(self.extract_pitch(audio))
        features.update(self.extract_energy(audio))
        features.update(self.extract_zcr(audio))
        features.update(self.extract_spectral_features(audio))
        features.update(self.extract_tempo(audio))
        
        # Convert to flat vector
        feature_vector = []
        for key in sorted(features.keys()):
            val = features[key]
            if isinstance(val, np.ndarray):
                feature_vector.extend(val)
            else:
                feature_vector.append(val)
        
        return np.array(feature_vector)