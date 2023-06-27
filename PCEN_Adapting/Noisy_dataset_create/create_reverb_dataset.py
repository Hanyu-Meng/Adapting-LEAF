
import soundfile as sf
import librosa
import os
import pedalboard
from scipy.io import wavfile

reverb = pedalboard.Reverb()
reverb.wet_level = 0.5
reverb.damping = 0.8
directory = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d"
path_new = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d_reverb"

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        # load the audio signal
        audio_file = os.path.join(directory, filename)
        y, sr = librosa.load(audio_file, sr=16000)
        # add reverb to the audio file
        y_reverb = reverb(y, sr)
        # save the reverb-added signal to a new audio file
        new_file = os.path.join(path_new, filename)
        wavfile.write(new_file, sr, y_reverb)

