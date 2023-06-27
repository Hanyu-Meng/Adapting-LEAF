import os

for i in range(1, 87):
    print("Running Speaker {}".format(i))
    exec = "/home/unsw/anaconda3/envs/efficientleaf/bin/python /media/unsw/172E-A21B/IS2023/main.py"
    resum_path = "/media/unsw/172E-A21B/IS2023/outputs/models/pcen_alpha_r_adaptive-speaker-{}/net_last_model.pth".format(i)
    os.system(exec+" --target_speaker {} --model-name pcen_alpha_r_adaptive-speaker-{} --fixed_PCEN True --batch-size 100 --resume {} --epoch 20".format(i,i,resum_path))