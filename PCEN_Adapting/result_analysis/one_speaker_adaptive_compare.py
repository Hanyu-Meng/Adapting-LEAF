import torch
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import os

# select the speaker ID for different speakers
freq_channel_1 = 1
freq_channel_2 = 20
freq_channel_3 = 40
speaker_id = 22

# load trained model and system paramters
path_2 = '/media/unsw/172E-A21B/IS2023/outputs/models/non-adaptive-speaker-{}/net_last_model.pth'.format(speaker_id)
path_1 = '/media/unsw/172E-A21B/IS2023/outputs/models/adaptive-speaker-{}/net_last_model.pth'.format(speaker_id)
model_1 = torch.load(path_1, map_location=torch.device('cpu'))
model_2 = torch.load(path_2, map_location=torch.device('cpu'))


network_1 = model_1['network']
network_2 = model_2['network']

pcen_s1 = network_1['_frontend.compression.s']
pcen_alpha1 = network_1['_frontend.compression.alpha']
pcen_delta1 = network_1['_frontend.compression.delta']
pcen_r1 = network_1['_frontend.compression.r']

pcen_s2 = network_2['_frontend.compression.s']
pcen_alpha2 = network_2['_frontend.compression.alpha']
pcen_delta2 = network_2['_frontend.compression.delta']
pcen_r2 = network_2['_frontend.compression.r']



sample_rate = 16000
filter_num = 40
filter_width = 401 # same as the frame size
pcen_r1 = np.array(pcen_r1)
pcen_s1 = np.array(pcen_s1)
pcen_delta1 = np.array(pcen_delta1)
pcen_alpha1 = np.array(pcen_alpha1)

pcen_r2 = np.array(pcen_r2)
pcen_s2 = np.array(pcen_s2)
pcen_delta2 = np.array(pcen_delta2)
pcen_alpha2 = np.array(pcen_alpha2)
# Rebuild PCEN
epsilon = 1e-12
test_sample = 5000
Max_Energy_dB = 15
Min_Energy_dB = 30
E_dB = np.linspace(Max_Energy_dB, Min_Energy_dB, test_sample)
E_lin = pow(10, E_dB/20)



M1 = np.zeros([test_sample, filter_num])
M1[:,0] = E_lin

M2 = np.zeros([test_sample, filter_num])
M2[:,0] = E_lin

PCEN1 = np.zeros([test_sample, filter_num])
PCEN2 = np.zeros([test_sample, filter_num])


for i in range (1, filter_num):
    M1[:,i] = pcen_s1[i]*E_lin + (1-pcen_s1[i])*M1[:,i-1]
    M2[:, i] = pcen_s2[i] * E_lin + (1 - pcen_s2[i]) * M2[:, i-1]


#plotting
# Compression -- I/O plots
for i in range (0,filter_num):
    PCEN1[:,i] = 20*np.log10((E_lin/(epsilon+(M1[:,i])**pcen_alpha1[i]) + pcen_delta1[i])**pcen_r1[i] - (pcen_delta1[i])**pcen_r1[i])
    PCEN2[:, i] = 20 * np.log10((E_lin / ((epsilon + M2[:, i]) ** pcen_alpha2[i]) + pcen_delta2[i]) ** pcen_r2[i] - (pcen_delta2[i]) ** pcen_r2[i])


plt.figure()
plt.plot(E_dB, PCEN1[:,freq_channel_1-1], label='Adaptive Model Channel={}'.format(freq_channel_1))
plt.plot(E_dB, PCEN2[:, freq_channel_1-1], label= 'Non-Adaptive Model Channel={}'.format(freq_channel_1))

plt.plot(E_dB, PCEN1[:,freq_channel_2-1], label='Adaptive Model Channel={}'.format(freq_channel_2))
plt.plot(E_dB, PCEN2[:, freq_channel_2-1], label='Non-Adaptive Model Channel={}'.format(freq_channel_2))
#
plt.plot(E_dB, PCEN1[:,freq_channel_3-1], label='Adaptive Model Channel={}'.format(freq_channel_3))
plt.plot(E_dB, PCEN2[:, freq_channel_3-1], label='Non-Adaptive Model Channel={}'.format(freq_channel_3))

plt.legend()
plt.title("PCEN I/O (Speaker ID = {})".format(speaker_id))


