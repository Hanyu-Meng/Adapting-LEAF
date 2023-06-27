import torch
import numpy as np
from math import pi
import matplotlib.pyplot as plt

# select the speaker ID for different speakers
freq_channel = 40
sample_rate = 16000
filter_num = 40
filter_width = 401 # same as the frame size
# Parameters for Rebuilding PCEN
epsilon = 1e-12
test_sample = 2000
Max_Energy_dB = 15
Min_Energy_dB = 30

E_dB = np.linspace(Max_Energy_dB, Min_Energy_dB, test_sample)
E_lin = pow(10, E_dB/20)

plt.figure()
for spk in range (70, 81):
    # load trained model and system paramters
    path = '/media/unsw/172E-A21B/IS2023/outputs/models/non-adaptive-speaker-{}/net_last_model.pth'.format(spk)
    model = torch.load(path, map_location=torch.device('cpu'))
    network = model['network']
    filtering_mu = network['_frontend.filterbank.center_freqs']
    filtering_sigma = network['_frontend.filterbank.bandwidths']
    pooling_sigma = network['_frontend.filterbank.pooling_widths']
    pcen_s = network['_frontend.compression.s']
    pcen_alpha = network['_frontend.compression.alpha']
    pcen_delta = network['_frontend.compression.delta']
    pcen_r = network['_frontend.compression.r']
    pcen_r = np.array(pcen_r)
    pcen_s = np.array(pcen_s)
    pcen_delta = np.array(pcen_delta)
    pcen_alpha = np.array(pcen_alpha)

    M = np.zeros([test_sample, filter_num])
    M[:,0] = E_lin

    PCEN = np.zeros([test_sample, filter_num])

    for i in range (1, filter_num):
        M[:,i] = pcen_s[i]*E_lin + (1-pcen_s[i])*M[:,i-1]


    # Compression -- I/O plots
    for j in range (0,filter_num):
        PCEN[:,j] = 20*np.log10((E_lin/(epsilon+(M[:,j])**pcen_alpha[j]) + pcen_delta[j])**pcen_r[j] - (pcen_delta[j])**pcen_r[j])


    plt.plot(E_dB, PCEN[:,freq_channel-1], label='Speaker_id={}'.format(spk))



plt.xlabel('Input Energy (dB)')
plt.ylabel('PCEN Output (dB)')
plt.title('I/O plot for PCEN Compression (Frequency Channel= {})'. format(freq_channel))
plt.legend(loc='upper left', prop = {'size':10})
plt.show()
