
import sys
import numpy as np
import soundfile as sf
import os
import random
directory = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d"
path_new = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d_babble_noise_{}dB".format(sys.argv[1])
if os.path.exists(path_new):
    pass
else:
    os.makedirs(path_new)
# Load the original speech signal
SNR = sys.argv[1]


speaker_count = 3
dir = '/media/unsw/172E-A21B/IS2023/leaf_dataset/musan'
# Pick an SNR and use it to compute the mixture amplitude factors




for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        # load the audio signal
        audio_file = os.path.join(directory, filename)
        signal, sr = sf.read(audio_file)
        noise = np.zeros(len(signal))
        for i in range(1, speaker_count):
            filename_babble = random.choice(os.listdir(dir))
            babble_path = os.path.join(dir, filename_babble)
            babble, _ = sf.read(babble_path)
            noise += babble[sr*10:sr*10+len(signal)]

        noise_power = np.mean(np.square(noise))
        audio_power = np.mean(np.square(signal))
        desired_noise_power = audio_power / np.power(10, (int(SNR) / 10))
        scale_factor = np.sqrt(desired_noise_power / noise_power)
        noise *= scale_factor
        # Add the noise to the signal
        noisy_signal = signal + noise
        new_file = os.path.join(path_new, filename)
        sf.write(new_file, noisy_signal, sr)



