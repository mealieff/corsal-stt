import os
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch

directory = "corsal_boro_subset"
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

for file in os.listdir(directory):
    if file.endswith('.mp3'):
        mp3_path = os.path.join(directory, file)
        wav_file = file.rsplit('.', 1)[0] + '.wav'
        wav_path = os.path.join(directory, wav_file)
        if not os.path.exists(wav_path):
            audio, sr = librosa.load(mp3_path, sr=16000)
            sf.write(wav_path, audio, sr)

for file in os.listdir(directory):
    if file.endswith('.wav'):
        wav_path = os.path.join(directory, file)
        audio_input, _ = librosa.load(wav_path, sr=16000)
        input_values = tokenizer(audio_input, return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        prediction = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(prediction)
        print(f"{file}: {transcription[0]}")

