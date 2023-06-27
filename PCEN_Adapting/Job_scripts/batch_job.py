import os

# For training the noisy model
# --data-set "CREMAD_SEN_90"
noise_level = [0, 5, 10, 15, 20]
for i in noise_level:
    print("Running Noise Level {}dB".format(i))
    exec = "/home/unsw/anaconda3/envs/efficientleaf/bin/python /media/unsw/172E-A21B/IS2023/main.py"
    os.system(exec+" --level {} --model-name babble_trained_{}db".format(i,i))