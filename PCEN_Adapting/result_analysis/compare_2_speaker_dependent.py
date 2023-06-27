import torch
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import os

# select the speaker ID for different speakers
speaker_id_1 = 1
speaker_id_2 = 80
freq_channel_1 = 1
freq_channel_2 = 20
freq_channel_3 = 40
# load trained model and system paramters

path_2 = '/media/unsw/172E-A21B/IS2023/outputs/models/adaptive-speaker-{}/net_last_model.pth'.format(speaker_id_2)
path_1 = '/media/unsw/172E-A21B/IS2023/outputs/models/adaptive-speaker-{}/net_last_model.pth'.format(speaker_id_1)
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
test_sample = 2000
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
plt.plot(E_dB, PCEN1[:,freq_channel_1-1], color = 'r', label='Channel={}, Speaker_id={}'.format(freq_channel_1, speaker_id_1))
plt.plot(E_dB, PCEN2[:, freq_channel_1-1],'--', color = 'r', label='Channel={}, Speaker_id={}'.format(freq_channel_1, speaker_id_2))

plt.plot(E_dB, PCEN1[:,freq_channel_2-1], color = 'b', label='Channel={}, Speaker_id={}'.format(freq_channel_2, speaker_id_1))
plt.plot(E_dB, PCEN2[:, freq_channel_2-1],'--', color = 'b', label='Channel={}, Speaker_id={}'.format(freq_channel_2, speaker_id_2))

plt.plot(E_dB, PCEN1[:,freq_channel_3-1], color = 'g', label='Channel={}, Speaker_id={}'.format(freq_channel_3, speaker_id_1))
plt.plot(E_dB, PCEN2[:, freq_channel_3-1],'--', color = 'g', label='Channel={}, Speaker_id={}'.format(freq_channel_3, speaker_id_2))

plt.xlabel('Input Energy (dB)')
plt.ylabel('PCEN Output (dB)')
plt.title('I/O plot for PCEN Compression (Speaker ID = {} and Speaker ID = {})'. format(speaker_id_1, speaker_id_2))
plt.legend(loc='upper left', prop = {'size':10})
plt.show()

# Dynamic Range Computation
# DR = np.zeros(filter_num)
# input_range = E_lin[test_sample-1] - E_lin[0]
# total_speakers = 91
# PCEN = np.zeros([test_sample, filter_num])
# M = np.zeros([test_sample, filter_num])
# M[:, 0] = E_lin
# for i in range (1,9):
#     speaker_id = i
#     path1 = '/media/unsw/172E-A21B/IS2023/outputs/models/adaptive-speaker-{}/net_last_model.pth'.format(
#         speaker_id)
#     path2 = '/media/unsw/172E-A21B/IS2023/outputs/models/adaptive-speaker-{}/net_last_model.pth'.format(
#             speaker_id)
#     if not os.path.exists(path1):
#         path = path2
#         print(speaker_id)
#         print("---------------------")
#     elif not os.path.exists(path2):
#         print(speaker_id)
#         speaker_id = speaker_id + 1
#         continue
#     else:
#         path = path1
#     model = torch.load(path, map_location=torch.device('cpu'))
#     network = model['network']
#
#     pcen_s = network['_frontend.compression.s']
#     pcen_alpha = network['_frontend.compression.alpha']
#     pcen_delta = network['_frontend.compression.delta']
#     pcen_r = network['_frontend.compression.r']
#
#     pcen_r = np.array(pcen_r)
#     pcen_s = np.array(pcen_s)
#     pcen_delta = np.array(pcen_delta)
#     pcen_alpha = np.array(pcen_alpha)
#
#     for i in range(1, filter_num):
#         M[:, i] = pcen_s[i] * E_lin + (1 - pcen_s[i]) * M[:, i - 1]
#
#     # Compression -- I/O plots
#     for i in range(0, filter_num):
#         PCEN[:, i] = (E_lin / ((epsilon + M[:, i]) ** pcen_alpha[i]) + pcen_delta[i]) ** pcen_r[i] - (pcen_delta[i]) ** pcen_r[i]
#
#     output_range = PCEN[:, i].max() - PCEN[:, i].min()
#     DR[i-1] = output_range/input_range
#
# print(DR)



