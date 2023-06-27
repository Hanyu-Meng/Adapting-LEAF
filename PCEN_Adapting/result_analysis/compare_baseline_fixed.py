import torch
import numpy as np
import torch.nn as nn
from math import pi
from fixed_model import filtering_center_freq as initial_centered_freq
from fixed_model import filtering_bandwidth as initial_bandwidth
from fixed_model import pooling_sigma as initial_pooling_sigma
from fixed_model import PCEN as initial_PCEN
import matplotlib.pyplot as plt
from model.leaf import Leaf, gauss_windows

# load trained model and system paramters
path = '/media/unsw/172E-A21B/IS2023/outputs/models/adaptive_pcen_noise_test_new/net_last_model.pth'
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
# pooling_sigma = np.array(pooling_sigma)

pcen_r = np.array(pcen_r)
pcen_s = np.array(pcen_s)
pcen_delta = np.array(pcen_delta)
pcen_alpha = np.array(pcen_alpha)

# Rebuild PCEN
epsilon = 1e-12
test_sample = 2000
Max_Energy_dB = 0
Min_Energy_dB = 50
E_dB = np.linspace(Max_Energy_dB, Min_Energy_dB, test_sample)
E_lin = pow(10, E_dB/20)
M = np.zeros([test_sample, filter_num])
M[:,0] = E_lin
M0 = np.zeros(test_sample)
PCEN = np.zeros([test_sample, filter_num])
for i in range (1, filter_num):
    M[:,i] = pcen_s[i]*E_lin + (1-pcen_s[i])*M[:,i-1]


#plotting
# 1. Filterbank -- Centered Frequency
plt.figure()
plt.subplot(1,2,1)
x_axis = np.linspace(1,filter_num,filter_num)
plt.plot(x_axis, initial_centered_freq, '--', color='b', label = 'Initial')
plt.plot(x_axis, filtering_center_freq, '--', color='r', label = 'Baseline')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Number of Filters')
plt.title('Center Frequency of Filterbank')
plt.legend()

plt.subplot(1,2,2)
plt.plot(x_axis, filtering_center_freq-initial_centered_freq, marker= ".", color='k')
plt.xlabel('Number of Filters')
plt.title('Variations')


# 2. Filterbank -- Bandwidth
plt.figure()
plt.subplot(1,2,1)
plt.plot(x_axis, initial_bandwidth, '--', color='DarkGreen', label = 'Initial')
plt.plot(x_axis, filtering_bandwidth, '--', color='DarkOrange', label= 'Baseline')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Number of Filters')
plt.title('Bandwidth of Filterbank')
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_axis, filtering_bandwidth-initial_bandwidth, marker= ".", color='Purple')
plt.xlabel('Number of Filters')
plt.title('Variations')

# 3. Gaussian Lowpass
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x_axis,initial_pooling_sigma,'--', label = 'Fixed')
plt.plot(x_axis,pooling_sigma,'--', label = 'Baseline')
plt.xlabel('Number of Channels')
plt.title('Gausssian Sigma')
plt.legend()
# plot 2: magnitude response (dB)
# compute pooling windows
steps = int(filter_width/2)
freq = np.array(torch.linspace(0,sample_rate/2,steps))
fixed_filters = Leaf().filterbank
pooling_widths = fixed_filters.pooling_widths.clamp(min=2. / fixed_filters.pool_size,
                                           max=0.5)
fixed_gaussian = gauss_windows(fixed_filters.pool_size, pooling_widths)
freq_gaussian = abs(torch.fft.fft(fixed_gaussian)).detach().numpy()
trained_gaussian = gauss_windows(fixed_filters.pool_size, pooling_sigma)
trained_freq_gaussian = abs(torch.fft.fft(trained_gaussian)).detach().numpy()
plt.subplot(1, 2, 2)
plt.plot(freq,20*np.log10(freq_gaussian[0,:steps]),'--',color = 'b',label = 'Fixed')
plt.plot(freq,20*np.log10(trained_freq_gaussian[0,:steps]),color = 'r',label = '1st Channel')
plt.plot(freq,20*np.log10(trained_freq_gaussian[19,:steps]),color = 'DarkGreen', label = '20th Channel')
plt.plot(freq,20*np.log10(trained_freq_gaussian[39,:steps]),color = 'DarkOrange', label = '40th Channel')
plt.xlim((0,200))
plt.ylim((0,50))
plt.legend()
plt.xlabel('Frequency(Hz)')
plt.title('Amplitude Response (dB)')


# 4. Compression -- I/O plots
plt.figure()
for i in range (0,filter_num):
    PCEN[:,i] = 20*np.log10((E_lin/(epsilon+(M[:,i])**pcen_alpha[i]) + pcen_delta[i])**pcen_r[i] - (pcen_delta[i])**pcen_r[i])
    #plt.plot(E_dB, PCEN[:,i], label='{}'.format(i+1), linewidth=1)

plt.plot(E_dB, initial_PCEN[:,0], label='Initial PCEN', color = 'k', linewidth=2)
plt.plot(E_dB, PCEN[:,0], label='1st Channel', color='r', linewidth=2)
plt.plot(E_dB, PCEN[:,19], label='20th Channel', color='b', linewidth=2)
plt.plot(E_dB, PCEN[:,39], label='40th Channel', color='g', linewidth=2)
plt.xlabel('Input Energy (dB)')
plt.ylabel('PCEN Output (dB)')
plt.title('I/O plot for PCEN Compression (Speaker Independent Model)')
# plt.legend(loc='upper left', prop = {'size':8}, ncol=3)
plt.legend()
plt.show()