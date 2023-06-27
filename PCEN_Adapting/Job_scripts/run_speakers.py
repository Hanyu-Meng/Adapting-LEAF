import os
LOC_OF_LEAF = "/media/unsw/172E-A21B/EfficientLEAF"
DATA_PATH = "/media/unsw/172E-A21B/leaf_datasets"
MODEL_NAME = "compression_r1"
EPOCHS = 50

#
for i in range (1,92):
    OUT_DIR = "/media/unsw/172E-A21B/EfficientLEAF/outputs/models/crema-d-speaker-new/{}".format(i)
    SPEAKER_ID = "{}".format(i)
    CMD = "python {}/speaker_dependent_train.py --data-path {} --model-name {} --speaker_id {} --epochs {} --output-dir {}".format(LOC_OF_LEAF, DATA_PATH,MODEL_NAME,i,EPOCHS,OUT_DIR)
    os.system(CMD)
