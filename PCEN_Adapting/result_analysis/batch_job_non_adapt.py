import os

# Change the --adaptation to True
noise_level = [0, 5, 10, 15, 20]
for i in noise_level:
    print("Running Noise Level {}dB".format(i))
    exec = "/home/unsw/anaconda3/envs/efficientleaf/bin/python /media/unsw/172E-A21B/IS2023/main.py"
    model_path = "/media/unsw/172E-A21B/IS2023/noise_outputs/models/bable_nonadapt_{}db/net_last_model.pth".format(i)
    os.system(exec+" --level {} --model-name babble_adaptive_{}db --resume {} --tune True".format(i,i,model_path))