import numpy as np
from scipy.io import wavfile
import os


directory = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d"
path_new = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d_pink_noise"

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        audio_file = os.path.join(directory, filename)
        # Load the audio file
        sample_rate, audio_data = wavfile.read(audio_file)

        # Define the number of samples and the frequency range
        n_samples = len(audio_data)
        freqs = np.fft.fftfreq(n_samples, 1/sample_rate)

        # Calculate the pink noise power spectrum
        pink_noise_power = 1 / freqs

        # Set the desired noise level
        noise_level = 0.1

        # Generate the pink noise with the same length as the audio file
        pink_noise = np.fft.fft(np.random.normal(0, 1, n_samples))
        pink_noise = pink_noise * np.sqrt(pink_noise_power)
        pink_noise = np.real(np.fft.ifft(pink_noise))

        # Add the pink noise to the audio data
        audio_data_with_noise = audio_data + noise_level * pink_noise
        new_file = os.path.join(path_new, filename)
        # Write the audio data with noise to a new file
        wavfile.write(new_file, sample_rate, audio_data_with_noise)
