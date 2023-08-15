## imports
#basic
import os
import collections
import requests

#processing
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

#torch
import torch
from torch.utils.data import Dataset

#torchaudio
import torchaudio
import platform
if platform.system() == 'Windows':
    torchaudio.set_audio_backend("soundfile") # using torchaudio on a windows machine

#shared
from . import (_compute_split_boundaries, _get_tune_splits,
               build_dataloaders,_get_speaker_adaptation_splits, _get_inter_splits_by_sentence, _get_tune_splits)


## Crema-D for training the data before tuning and then do the tuning
class Crema_D(Dataset):
    def __init__(self, root='/content', split='train', seed=10, babble=False, noise=False, tune=False, level = 0):
        # same seed for train and test(!)
        assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train" or "test"'
        # inits
        # urls
        self.wav_data_url = 'https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/'
        self.csv_summary_url = 'https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv'
        self.tune = tune
        # paths
        # for tuning, set noise == Ture
        if noise == True:
            self.root_data = os.path.join(root, 'crema_d_gaussian_noise_{}dB'.format(level))
        elif babble == True:
            self.root_data = os.path.join(root, 'crema_d_babble_noise_{}dB'.format(level))
        else:
            self.root_data = os.path.join(root, 'crema_d')

        if not os.path.isdir(self.root_data): os.mkdir(self.root_data)  # init folder
        # parameters
        self.split = split
        self.seed = seed
        # x,y
        self.waveforms = []
        self.labels = []

        # fill x,y
        self.split_load()

    def split_load(self):
        csv_summary = pd.read_csv(self.csv_summary_url, index_col=0)
        all_wav_files = []
        sentence = []
        spk_ids = []
        wav_names = []
        labels = []

        # get info from summary csv
        for _, row in csv_summary.iterrows():
            wav_name = row['FileName']

            wav_path = os.path.join(self.wav_data_url, '%s.wav' % wav_name)
            all_wav_files.append(wav_path)
            sentence.append(wav_name.split('_')[1])
            wav_names.append(wav_name)
            labels.append(wav_name.split('_')[2])
            spk_ids.append(wav_name.split('_')[0])

        # splitting train/test
        items_and_groups = list(zip(wav_names, sentence))
        all_wav_info = list(zip(all_wav_files, wav_names, spk_ids, labels))
        items_and_groups = list(zip(all_wav_info, sentence))
        # solit 90 speakers

        # further split according to ratio

        split_probs = [('train', 0.75), ('validation', 0.09), ('test', 0.16)]
        split_to_ids = _get_tune_splits(items_and_groups, split_probs, self.tune)

        # download file if not already in crema_d folder
        tqdm_dl = tqdm(range(1, len(split_to_ids[self.split])), desc=f'Downloading {self.split}')
        for wave_url, wave_name, speaker_id, label in split_to_ids[self.split]:

            # download
            wave_file_location = os.path.join(self.root_data, wave_name + '.wav')
            if not os.path.isfile(wave_file_location):
                with open(wave_file_location, 'wb') as file_:
                    file_.write(requests.get(wave_url).content)

            waveform, sample_rate = torchaudio.load(wave_file_location)
            if sample_rate != 16000:
                print('Samplerate of', wave_name, 'is', sample_rate)
            # adding noise and reverb to data
            self.waveforms.append(waveform)
            self.labels.append(label)
            tqdm_dl.update()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.waveforms[idx]
        label = self.labels[idx]

        return audio, label

# New Crema-D that has split in advance
class Crema_D_speakers(Dataset):
    def __init__(self, root = '/content', split='train', seed=10, target_spk=1):
        #same seed for train and test(!)
        assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train" or "test"'
        # inits
        #urls
        self.wav_data_url = 'https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/'
        self.csv_summary_url = 'https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv'
        #paths
        self.root_data = os.path.join(root, 'crema_d')
        if not os.path.isdir(self.root_data): os.mkdir(self.root_data) #init folder
        #parameters
        self.split = split
        self.seed = seed
        self.speaker_id = target_spk
        #x,y
        self.waveforms = []
        self.labels = []

        #fill x,y
        self.split_load()

    def split_load(self):
        csv_summary = pd.read_csv(self.csv_summary_url, index_col=0)
        all_wav_files = []
        spk_ids = []
        wav_names = []
        labels = []
        sentence = []

        # get info from summary csv
        for _, row in csv_summary.iterrows():
            wav_name = row['FileName']

            wav_path = os.path.join(self.wav_data_url, '%s.wav' % wav_name)
            all_wav_files.append(wav_path)
            sentence.append(wav_name.split('_')[1])
            wav_names.append(wav_name)
            labels.append(wav_name.split('_')[2])
            spk_ids.append(wav_name.split('_')[0])

        #splitting train/test
        items_and_groups =  list(zip(wav_names, sentence))
        all_wav_info = list(zip(all_wav_files, wav_names, spk_ids, labels))
        items_and_groups = list(zip(all_wav_info, sentence))
        # solit 90 speakers

        # further split according to ratio
        split_probs = [('train', 0.75), ('validation', 0.09), ('test', 0.16)]
        split_to_ids = _get_speaker_adaptation_splits(items_and_groups, split_probs, self.speaker_id)

        #download file if not already in crema_d folder
        tqdm_dl = tqdm(range(1, len(split_to_ids[self.split])), desc= f'Downloading {self.split}')
        for wave_url, wave_name, speaker_id, label in split_to_ids[self.split]:

            #download
            wave_file_location = os.path.join(self.root_data, wave_name+'.wav')
            if not os.path.isfile(wave_file_location):
                with open(wave_file_location, 'wb') as file_:
                    file_.write(requests.get(wave_url).content)

            waveform, sample_rate = torchaudio.load(wave_file_location)
            if sample_rate != 16000:
                print('Samplerate of', wave_name, 'is', sample_rate)
            self.waveforms.append(waveform)
            self.labels.append(label)
            tqdm_dl.update()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.waveforms[idx]
        label = self.labels[idx]

        return audio, label

class Crema_D_90_clean(Dataset):
    def __init__(self, root='/content', split='train', seed=10, target_spk=None, test_noise=False, test_babble=False, level = 0):
        # same seed for train and test(!)
        assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train" or "test"'
        # inits
        # urls
        self.wav_data_url = 'https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/'
        self.csv_summary_url = 'https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv'
        # paths
        if test_noise == True:
            self.root_data = os.path.join(root, 'crema_d_gaussian_noise_{}dB'.format(level))
        elif  test_babble == True:
            self.root_data = os.path.join(root, 'crema_d_babble_noise_{}dB'.format(level))
        else:
            self.root_data = os.path.join(root, 'crema_d')

        if not os.path.isdir(self.root_data): os.mkdir(self.root_data)  # init folder
        # parameters
        self.split = split
        self.seed = seed
        self.speaker_id = target_spk
        # x,y
        self.waveforms = []
        self.labels = []

        # fill x,y
        self.split_load()

    def split_load(self):
        csv_summary = pd.read_csv(self.csv_summary_url, index_col=0)
        all_wav_files = []
        sentence = []
        spk_ids = []
        wav_names = []
        labels = []

        # get info from summary csv
        for _, row in csv_summary.iterrows():
            wav_name = row['FileName']

            wav_path = os.path.join(self.wav_data_url, '%s.wav' % wav_name)
            all_wav_files.append(wav_path)
            sentence.append(wav_name.split('_')[1])
            wav_names.append(wav_name)
            labels.append(wav_name.split('_')[2])
            spk_ids.append(wav_name.split('_')[0])

        # splitting train/test
        items_and_groups = list(zip(wav_names, sentence))
        all_wav_info = list(zip(all_wav_files, wav_names, spk_ids, labels))
        items_and_groups = list(zip(all_wav_info, sentence))
        # solit 90 speakers

        # further split according to ratio
        split_probs = [('train', 0.75), ('validation', 0.09), ('test', 0.16)]
        split_to_ids = _get_inter_splits_by_sentence(items_and_groups, split_probs, self.speaker_id)
        # split_to_ids = _get_inter_splits_by_group(items_and_groups, split_probs, split_number=0)

        # download file if not already in crema_d folder
        tqdm_dl = tqdm(range(1, len(split_to_ids[self.split])), desc=f'Downloading {self.split}')
        for wave_url, wave_name, speaker_id, label in split_to_ids[self.split]:

            # download
            wave_file_location = os.path.join(self.root_data, wave_name + '.wav')
            if not os.path.isfile(wave_file_location):
                with open(wave_file_location, 'wb') as file_:
                    file_.write(requests.get(wave_url).content)

            waveform, sample_rate = torchaudio.load(wave_file_location)
            if sample_rate != 16000:
                print('Samplerate of', wave_name, 'is', sample_rate)
            # adding noise and reverb to data
            self.waveforms.append(waveform)
            self.labels.append(label)
            tqdm_dl.update()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.waveforms[idx]
        label = self.labels[idx]

        return audio, label

class Crema_D_sentences_90(Dataset):
    def __init__(self, root = '/content', split='train', seed=10, target_spk=1, babble=False, noise=False, level=0):
        #same seed for train and test(!)
        assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train" or "test"'
        # inits
        #urls
        self.wav_data_url = 'https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/'
        self.csv_summary_url = 'https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv'
        #paths
        if babble == True:
            self.root_data = os.path.join(root, 'crema_d_babble_noise_{}dB'.format(level))
        elif noise == True:
            self.root_data = os.path.join(root, 'crema_d_gaussian_noise_{}dB'.format(level))
        else:
            self.root_data = os.path.join(root, 'crema_d')

        if not os.path.isdir(self.root_data): os.mkdir(self.root_data) #init folder
        #parameters
        self.split = split
        self.seed = seed
        self.speaker_id = target_spk
        #x,y
        self.waveforms = []
        self.labels = []

        #fill x,y
        self.split_load()

    def split_load(self):
        csv_summary = pd.read_csv(self.csv_summary_url, index_col=0)
        all_wav_files = []
        sentence = []
        spk_ids = []
        wav_names = []
        labels = []

        # get info from summary csv
        for _, row in csv_summary.iterrows():
            wav_name = row['FileName']

            wav_path = os.path.join(self.wav_data_url, '%s.wav' % wav_name)
            all_wav_files.append(wav_path)
            sentence.append(wav_name.split('_')[1])
            wav_names.append(wav_name)
            labels.append(wav_name.split('_')[2])
            spk_ids.append(wav_name.split('_')[0])

        #splitting train/test
        items_and_groups =  list(zip(wav_names, sentence))
        all_wav_info = list(zip(all_wav_files, wav_names, spk_ids, labels))
        items_and_groups = list(zip(all_wav_info, sentence))
        # solit 90 speakers

        # further split according to ratio
        split_probs = [('train', 0.75), ('validation', 0.09), ('test', 0.16)]
        split_to_ids = _get_inter_splits_by_sentence(items_and_groups, split_probs, self.speaker_id)
        # split_to_ids = _get_inter_splits_by_group(items_and_groups, split_probs, split_number=0)

        #download file if not already in crema_d folder
        tqdm_dl = tqdm(range(1, len(split_to_ids[self.split])), desc= f'Downloading {self.split}')
        for wave_url, wave_name, speaker_id, label in split_to_ids[self.split]:

            #download
            wave_file_location = os.path.join(self.root_data, wave_name+'.wav')
            if not os.path.isfile(wave_file_location):
                with open(wave_file_location, 'wb') as file_:
                    file_.write(requests.get(wave_url).content)

            waveform, sample_rate = torchaudio.load(wave_file_location)
            if sample_rate != 16000:
                print('Samplerate of', wave_name, 'is', sample_rate)
            # adding noise and reverb to data
            self.waveforms.append(waveform)
            self.labels.append(label)
            tqdm_dl.update()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.waveforms[idx]
        label = self.labels[idx]

        return audio, label


def label_to_index(word):
    labels = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    labels = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


## Build dataset function for adapatation PCEN trained
# 1 sentence for validation, 2 for test, 8 clean for train, 1 noise for fine tuning
def build_dataset(args):
    test_set = Crema_D(root = args.data_path, split='test', seed=args.seed, babble = args.babble, noise=args.noise, tune = args.tune, level = args.level)
    val_set = Crema_D(root = args.data_path, split='validation', seed=args.seed, babble=args.babble, noise=False, tune=args.tune, level = args.level)
    train_set = Crema_D(root = args.data_path, split='train', seed=args.seed, babble=args.babble, noise=False, tune=args.tune, level=args.level)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=label_to_index, args=args)
    nb_classes = 6
    return train_loader, val_loader, test_loader, nb_classes


# the data split for train the adaptive speaker models
def build_dataset_speakers(args):
    test_set = Crema_D_speakers(root = args.data_path, split='test', seed=args.seed, target_spk = args.target_speaker)
    val_set = Crema_D_speakers(root = args.data_path, split='validation', seed=args.seed, target_spk = args.target_speaker)
    train_set = Crema_D_speakers(root = args.data_path, split='train', seed=args.seed, target_spk = args.target_speaker)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=label_to_index, args=args)
    nb_classes = 6
    return train_loader, val_loader, test_loader, nb_classes


# the data split for train the baseline
def build_dataset_90_sentences(args):
    if args.adaptation == True:
        test_set = Crema_D_speakers(root=args.data_path, split='test', seed=args.seed, target_spk=args.target_speaker, babble=args.babble, noise=args.noise)
    else:
        test_set = Crema_D_90_clean(root=args.data_path, split='test', seed=args.seed, target_spk=args.target_speaker, test_noise = args.noise_test, test_babble = args.babble_test, level=args.level)

    if args.noise == True or args.babble == True:
        val_set = Crema_D_sentences_90(root = args.data_path, split='validation', seed=args.seed, target_spk = args.target_speaker, babble=args.babble, noise=args.noise, level=args.level)
        train_set = Crema_D_sentences_90(root = args.data_path, split='train', seed=args.seed, target_spk = args.target_speaker, babble=args.babble, noise=args.noise, level=args.level)
    else:
        val_set = Crema_D_90_clean(root=args.data_path, split='validation', seed=args.seed, target_spk=args.target_speaker, test_noise = False, test_babble = False, level=args.level)
        train_set = Crema_D_90_clean(root=args.data_path, split='train', seed=args.seed, target_spk=args.target_speaker, test_noise = False, test_babble = False, level=args.level)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=label_to_index, args=args)
    nb_classes = 6
    return train_loader, val_loader, test_loader, nb_classes

