import numpy as np
from math import pi
import matplotlib.pyplot as plt
import torchaudio
import argparse
import os
from pathlib import Path
from collections import OrderedDict
import librosa
## processing
from tqdm import tqdm

## torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

## efficentnet pytorch implementation
from efficientnet_pytorch import EfficientNet

## internal functions
from model import AudioClassifier
from model.leaf import Leaf, PCEN


audio_file = "/media/unsw/172E-A21B/IS2023/leaf_dataset/crema_d/1045_WSI_ANG_XX.wav"
data_waveform, rate_of_sample = torchaudio.load(audio_file)
path_1 = '/media/unsw/172E-A21B/IS2023/outputs/models/adaptive_pcen_reverb_test/net_last_model.pth'
model_1  = torch.load(path_1, map_location=torch.device('cpu'))
# Default LEAF parameters
n_filters = 40
sample_rate = 16000
window_len = 25.0
window_stride = 10.0
min_freq = 60.0
max_freq = 7800.0

compression_fn = PCEN(num_bands=n_filters,
                      s=0.04,
                      alpha=0.96,
                      delta=2.0,
                      r=0.5,
                      eps=1e-12,
                      learn_logs=False,
                      clamp=1e-5)

frontend = Leaf(n_filters=n_filters,
                min_freq=min_freq,
                max_freq=max_freq,
                sample_rate=sample_rate,
                window_len=window_len,
                window_stride=window_stride,
                compression=compression_fn)

## init encoder
frontend_channels = 1
encoder = EfficientNet.from_name("efficientnet-b0", num_classes=6, include_top=False,
                                 in_channels=frontend_channels)
encoder._avg_pooling = torch.nn.Identity()

## init classifier
network = AudioClassifier(
    num_outputs=6,
    frontend=frontend,
    encoder=encoder)


network_1 = model_1['network']
network.load_state_dict(network_1)
representation = network._frontend(data_waveform)
representation = representation.squeeze(0)
representation = representation.detach().numpy()
# representation = librosa.amplitude_to_db(representation, ref=np.max)

fig,ax = plt.subplots()
img = ax.pcolormesh(representation, cmap='plasma')
# fig.colorbar(img, format = "%2.0f dB")
fig.colorbar(img)
ax.set_xlabel('Time (frames)')
ax.set_ylabel('Frequence (Hz)')
plt.title('Adaptive (Clean)')
plt.show()