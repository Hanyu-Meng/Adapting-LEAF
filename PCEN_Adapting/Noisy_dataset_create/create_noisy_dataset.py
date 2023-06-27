import soundfile as sf
import os
from scipy.io import wavfile
import numpy as np
import librosa
directory = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d"
path_new = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d_gaussian_noise_15dB"

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        # load the audio signal
        audio_file = os.path.join(directory, filename)
        # Set the desired signal-to-noise ratio (SNR) in decibels (dB)
        snr_db = 15
        y, sr = librosa.load(audio_file, sr=16000)
        # Calculate the signal power
        signal_power = np.sum(y ** 2) / len(y)
        # Calculate the noise power using the SNR
        noise_power = signal_power / (10 ** (snr_db / 10))
        # Generate Gaussian noise with the same length as the speech signal
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        # Add the Gaussian noise to the speech signal
        y_noise = y + noise
        new_file = os.path.join(path_new, filename)
        wavfile.write(new_file, sr, y_noise)

