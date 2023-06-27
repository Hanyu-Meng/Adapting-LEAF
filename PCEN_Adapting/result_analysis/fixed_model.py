import torch
import numpy as np
import torch.nn as nn
from math import pi


# load the fixed model as the referenece
path = '/media/unsw/172E-A21B/EfficientLEAF/outputs/models/crema-d/models/leaf-fixed_r1/net_last_model.pth'
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

sample_rate = 16000
filter_num = 40
filter_width = 401 # same as the frame size
# convert the tensor to np array, transfer the network weight to interpretable paramters
filtering_center_freq = np.array((filtering_mu/pi)*sample_rate/2)
filtering_bandwidth = np.array((sample_rate/2)/filtering_sigma)
pooling_sigma = np.array(pooling_sigma)

# Rebuild PCEN
epsilon = 1e-12
test_sample = 2000
Max_Energy_dB = 0
Min_Energy_dB = 50
E_dB = np.linspace(Max_Energy_dB, Min_Energy_dB, test_sample)
E_lin = pow(10, E_dB/20)
M = np.zeros([test_sample, filter_num])
M[:,0] = E_lin
PCEN = np.zeros([test_sample, filter_num])
for i in range (1, filter_num):
    M[:,i] = pcen_s[i]*E_lin + (1-pcen_s[i])*M[:,i-1]

for i in range (0,filter_num):
    PCEN[:,i] = 20*np.log10((E_lin/(epsilon+(M[:,i])**pcen_alpha[i]) + pcen_delta[i])**pcen_r[i] - (pcen_delta[i])**pcen_r[i])
