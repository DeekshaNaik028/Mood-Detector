import shutil
from pathlib import Path

# Configure paths
source_dir = r"C:\Users\dell\Downloads\Audio_Speech_Actors_01-24.zip"  # CHANGE THIS
output_dir = "data/voice_emotions"

# Emotion mapping
emotions = {
    '01': 'neutral', '02': 'neutral',
    '03': 'happy', '04': 'sad', '05': 'angry',
    '06': 'fearful', '07': 'disgusted', '08': 'surprised'
}

# Create output directories
for emotion in set(emotions.values()):
    Path(output_dir, emotion).mkdir(parents=True, exist_ok=True)

# Process all wav files
for wav_file in Path(source_dir).rglob('*.wav'):
    emotion_code = wav_file.name.split('-')[2]
    emotion = emotions.get(emotion_code)
    
    if emotion:
        dest = Path(output_dir, emotion, wav_file.name)
        shutil.copy2(wav_file, dest)
        print(f"Copied {wav_file.name} to {emotion}/")

print("\nDone! Check data/voice_emotions/")