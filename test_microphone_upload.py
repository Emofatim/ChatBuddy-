import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import time

# Record audio
duration = 3  # seconds
fs = 22050  # sample rate

print("Recording...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
filename = "recorded_audio.wav"
wav.write(filename, fs, audio)
print("Recording complete!")

# Send to API
url = 'http://127.0.0.1:5000/predict'
files = {'file': open(filename, 'rb')}
response = requests.post(url, files=files)

print("Predicted Emotion:", response.text)
