import os

# SNR: 13- 20 dB

SNR = [0, 5, 10, 15, 20]
for i in SNR:
    print("Running Babble Noise Level {}dB".format(i))
    exec = "/home/unsw/anaconda3/envs/efficientleaf/bin/python /media/unsw/172E-A21B/IS2023/create_babble_noise_dataset.py"
    os.system(exec+" {}".format(str(i)))