import os
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from praatio import pitch_and_intensity
from praatio import praat_scripts
from praatio import textgrid
from praatio.utilities import utils
from pydub import AudioSegment

# Directory to search for mp3 and wav files
directory = "corsal_boro_subset"

# Iterate through all files in the directory
for file in os.listdir(directory):
    if file.endswith('.mp3'):
        wav_file = os.path.join
        wav_path = os.path.join(directory,wav_file)
        mp3_path = os.path.join(directory, file)
        # Check if the corresponding wav file exists
        if not os.path.exists(wav_path):
            # Load mp3 and save as wav
            audio, sr = librosa.load(mp3_path, sr=16000)
            sf.write(wav_path, audio, sr)

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")


prediction = torch.argmax(logits,dim = -1)
transcription = tokenizer.batch_decode(prediction)
print(transcription)
