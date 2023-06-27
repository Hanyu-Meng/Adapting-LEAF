import torch
import numpy as np
from math import pi
from compare_baseline_fixed import PCEN as baseline_PCEN
import matplotlib.pyplot as plt

# select the speaker ID for different speakers
speaker_id = 65
freq_channel_1 = 1
freq_channel_2 = 20
freq_channel_3 = 40
# load trained model and system paramters
path = '/media/unsw/172E-A21B/IS2023/outputs/models/adaptive-speaker-{}/net_last_model.pth'.format(speaker_id)
model = torch.load(path, map_location=torch.device('cpu'))
network = model['network']
filtering_mu = network['_frontend.filterbank.center_freqs']
filtering_sigma = network['_frontend.filterbank.bandwidths']
pooling_sigma = network['_frontend.filterbank.pooling_widths']
pcen_s = network['_frontend.compression.s']
pcen_alpha = network['_frontend.compression.alpha']
pcen_delta = network['_frontend.compression.delta']
pcen_r = network['_frontend.compression.r']

sample_rate = 16000
filter_num = 40
filter_width = 401 # same as the frame size
# convert the tensor to np array, transfer the network weight to interpretable paramters
filtering_center_freq = np.array((filtering_mu/pi)*sample_rate/2)
filtering_bandwidth = np.array((sample_rate/2)/filtering_sigma)
pcen_r = np.array(pcen_r)
pcen_s = np.array(pcen_s)
pcen_delta = np.array(pcen_delta)
pcen_alpha = np.array(pcen_alpha)

# Rebuild PCEN
epsilon = 1e-12
test_sample = 2000
Max_Energy_dB = 15
Min_Energy_dB = 30
E_dB = np.linspace(Max_Energy_dB, Min_Energy_dB, test_sample)
E_lin = pow(10, E_dB/20)
M = np.zeros([test_sample, filter_num])
M[:,0] = E_lin
M0 = np.zeros(test_sample)
PCEN = np.zeros([test_sample, filter_num])

for i in range (1, filter_num):
    M[:,i] = pcen_s[i]*E_lin + (1-pcen_s[i])*M[:,i-1]


#plotting
# Compression -- I/O plots
for i in range (0,filter_num):
    PCEN[:,i] = 20*np.log10((E_lin/(epsilon+(M[:,i])**pcen_alpha[i]) + pcen_delta[i])**pcen_r[i] - (pcen_delta[i])**pcen_r[i])


plt.figure()
plt.plot(E_dB, PCEN[:,freq_channel_1-1],'--',color = 'r', label='Channel={}, Non-adaptive Speaker_id={}'.format(freq_channel_1, speaker_id))
plt.plot(E_dB, baseline_PCEN[:, freq_channel_1-1],color = 'r', label='Channel={}, Speaker Independent'.format(freq_channel_1))
plt.plot(E_dB, PCEN[:,freq_channel_2-1], '--', color = 'b', label='Channel={}, Non-adaptive Speaker_id={}'.format(freq_channel_2, speaker_id))
plt.plot(E_dB, baseline_PCEN[:, freq_channel_2-1], color = 'b', label='Channel={}, Speaker Independent'.format(freq_channel_2))
plt.plot(E_dB, PCEN[:,freq_channel_3-1], '--', color = 'g',label='Channel={}, Non-adaptive Speaker_id={}'.format(freq_channel_3, speaker_id))
plt.plot(E_dB, baseline_PCEN[:, freq_channel_3-1], color = 'g',label='Channel={}, Speaker Independent'.format(freq_channel_3))
plt.xlabel('Input Energy (dB)')
plt.ylabel('PCEN Output (dB)')
plt.title('I/O plot for PCEN Compression (Speaker ID = {})'. format(speaker_id))
plt.legend(loc='upper left', prop = {'size':10})
plt.show()

# Dynamic Range for different speakers
