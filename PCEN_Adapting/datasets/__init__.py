import collections
from functools import partial

import numpy as np
import torch
import random
from torch.utils.data import IterableDataset, get_worker_info
import torchaudio
#SOURCE: https://github.com/tensorflow/datasets/blob/17b40dfdf6ce13adde74f82dd1214fe26545b0d3/tensorflow_datasets/audio/crema_d.py#L56
def _compute_split_boundaries(split_probs, n_items):
    """Computes boundary indices for each of the splits in split_probs.
    Args:
      split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
        ('test', 0.2)]
      n_items: Number of items we want to split.
    Returns:
      The item indices of boundaries between different splits. For the above
      example and n_items=100, these will be
      [('train', 0, 60), ('dev', 60, 80), ('test', 80, 100)].
    """
    if len(split_probs) > n_items:
        raise ValueError('Not enough items for the splits. There are {splits} '
                         'splits while there are only {items} items'.format(
            splits=len(split_probs), items=n_items))
    total_probs = sum(p for name, p in split_probs)
    if abs(1 - total_probs) > 1E-8:
        raise ValueError('Probs should sum up to 1. probs={}'.format(split_probs))
    split_boundaries = []
    sum_p = 0.0
    for name, p in split_probs:
        prev = sum_p
        sum_p += p
        split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))

    # Guard against rounding errors.
    split_boundaries[-1] = (split_boundaries[-1][0], split_boundaries[-1][1],
                            n_items)
    return split_boundaries


#SOURCE: https://github.com/tensorflow/datasets/blob/17b40dfdf6ce13adde74f82dd1214fe26545b0d3/tensorflow_datasets/audio/crema_d.py#L90
def _get_tune_splits(items_and_groups, split_probs, tune):
    groups = sorted(set(group_id for item_id, group_id in items_and_groups))

    split_boundaries = _compute_split_boundaries(split_probs, len(groups))
    group_id_to_split = {}
    for split_name, i_start, i_end in split_boundaries:
        for i in range(i_start, i_end):
            group_id_to_split[groups[i]] = split_name


    split_to_ids = collections.defaultdict(set)
    tune_sentence = ['IOM']
    other_train = ['IEO', 'IWW', 'TAI', 'MTI', 'IWL', 'ITH', 'DFA', 'ITS']
    for item_id, group_id in items_and_groups:
        split = group_id_to_split[group_id]
        split_to_ids[split].add(item_id)

    if tune == True:
        for item_id, group_id in items_and_groups:
            if group_id in other_train:
                split_to_ids['train'].remove(item_id)
    else:
        for item_id, group_id in items_and_groups:
            if group_id in tune_sentence:
                split_to_ids['train'].remove(item_id)

    return split_to_ids


def _get_inter_splits_by_sentence(items_and_groups, split_probs, speaker_id):
    """Split items to train/dev/test, so all items in group go into same split.
    Each group contains all the samples from the same sentences. The samples are
    splitted between train, validation and testing so that samples from each
    speaker belongs to exactly one split.
    Args:
      items_and_groups: Sequence of (item_id, group_id) pairs.
      split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
        ('test', 0.2)]
      split_number: Generated splits should change with split_number.
    Returns:
      Dictionary that looks like {split name -> set(ids)}.
    """
    spk_groups = sorted(set(item_id[2] for item_id, group_id in items_and_groups))
    groups = sorted(set(group_id for item_id, group_id in items_and_groups))
    new_items_and_groups = []
    i = 0
    # outlier_speakers = ['1002', '1008', '1009', '1019', '1076']
    # if speaker_id != None:
    #     for spk_id in outlier_speakers:
    #         spk_groups.remove(spk_id)

    #incorperate outlier speakers into training set

    # if the speaker id is none, then, get all speaker data except the outlier speakers
    # if the speaker id is a number (from 1-91), then, get all 85 speakers' data except the outlier speakers and the target speaker
    for item_id, group_id in items_and_groups:
        if speaker_id != None:
            if item_id[2] != spk_groups[speaker_id - 1]:
                new_items_and_groups.append(items_and_groups[i])

            i += 1
        elif speaker_id == None:
            new_items_and_groups.append(items_and_groups[i])
            i += 1

    random.shuffle(new_items_and_groups)
    # rng = np.random.RandomState(split_number)
    # rng.shuffle(groups)

    split_boundaries = _compute_split_boundaries(split_probs, len(groups))
    group_id_to_split = {}
    for split_name, i_start, i_end in split_boundaries:
        for i in range(i_start, i_end):
            # if i == 9:
            #     group_id_to_split[groups[i]] = 'validation'
            # elif i == 11 and split_name == 'validation':
            #     group_id_to_split[groups[i]] = 'train'
            # else:
            group_id_to_split[groups[i]] = split_name

    split_to_ids = collections.defaultdict(set)
    for item_id, group_id in new_items_and_groups:
        split = group_id_to_split[group_id]
        split_to_ids[split].add(item_id)

    # count = 0
    # for item_id, group_id in new_items_and_groups:
    #     if group_id == 'TIE':
    #         count += 1
    # test_sentence = ['TSI', 'WSI']
    # val_sentence = ['TIE']
    # for item_id, group_id in items_and_groups:
    #     if item_id[2] in outlier_speakers and group_id in test_sentence:
    #         split_to_ids['test'].remove(item_id)
    #     elif item_id[2] in outlier_speakers and group_id in val_sentence:
    #         split_to_ids['validation'].remove(item_id)


    return split_to_ids



def _get_speaker_adaptation_splits(items_and_groups, split_probs, speaker_id):
    spk_groups = sorted(set(item_id[2] for item_id, group_id in items_and_groups))
    groups = sorted(set(group_id for item_id, group_id in items_and_groups))
    new_items_and_groups = []
    i = 0
    outlier_speakers = ['1002', '1008', '1009', '1019', '1076']
    for spk_id in outlier_speakers:
        spk_groups.remove(spk_id)
    #incorperate outlier speakers into training set
    for item_id, group_id in items_and_groups:
        if item_id[2] == spk_groups[speaker_id - 1] and item_id[2] not in outlier_speakers:
            new_items_and_groups.append(items_and_groups[i])
        i += 1

    split_boundaries = _compute_split_boundaries(split_probs, len(groups))
    group_id_to_split = {}
    # split the dataset based on sentences
    for split_name, i_start, i_end in split_boundaries:
        for i in range(i_start, i_end):
            group_id_to_split[groups[i]] = split_name

    split_to_ids = collections.defaultdict(set)
    for item_id, group_id in new_items_and_groups:
        split = group_id_to_split[group_id]
        split_to_ids[split].add(item_id)

    return split_to_ids

class WindowedDataset(IterableDataset):
    """
    Iterates over recordings of an audio dataset in chunks of a given length,
    with a given amount or fraction of overlap. If `pad_incomplete` is "zero",
    the last chunk will be zero-padded; if "overlap", it will be overlapped
    with the previous chunk; if "drop", it will be omitted.
    """
    def __init__(self, dataset, window_size, overlap=0, pad_incomplete='zero'):
        super().__init__()
        self.dataset = dataset
        self.window_size = window_size
        self.pad_incomplete = pad_incomplete
        if 0 < abs(overlap) < 1:
            self.overlap = int(overlap * window_size)  # interpret as fraction
        else:
            self.overlap = int(overlap)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            offset = 0
            stride = 1
        else:
            offset = worker_info.id
            stride = worker_info.num_workers
        for idx in range(offset, len(self.dataset), stride):
            audio, label = self.dataset[idx]
            audio_size = audio.shape[1]
            hop_size = self.window_size - self.overlap
            start_pos = 0
            while audio_size - start_pos >= self.window_size:
                # yield all complete chunks, with the given amount of overlap
                yield audio[:, start_pos:start_pos + self.window_size], label, idx
                start_pos += hop_size
            if self.pad_incomplete == 'drop' and start_pos > 0:
                # drop any remainder of the recording, move to the next file
                continue
            elif self.pad_incomplete == 'overlap' and start_pos < audio_size:
                # overlap last chunk with the previous to last chunk
                start_pos = max(0, audio_size - self.window_size)
            if start_pos < audio_size:
                # return last chunk, zero-padded at the end if needed
                chunk = audio[:, start_pos:]
                if chunk.shape[1] < self.window_size:
                    chunk = torch.nn.functional.pad(
                        chunk, (0, self.window_size - chunk.shape[1]))
                yield chunk, label, idx


def align_sample(sample: torch.Tensor, seq_len: int=16000): #sample shape: (channels, seq_len)
    pad_length = seq_len - sample.shape[1]
    if pad_length == 0:
        return sample
    elif pad_length > 0: #padding
        return torch.nn.functional.pad(sample, pad=(0, pad_length), mode='constant', value=0.)
    else: #cropping
        max_start_pos = (pad_length * -1) + 1 #draw from 0 to max_start_pos
        pos = np.random.randint(max_start_pos)
        return sample[:, pos:pos + seq_len]


def db_to_linear(samples):
    return 10.0 ** (samples / 20.0)


def add_noise(samples: torch.Tensor):
    # add gaussian noise
    noise = torch.randn(samples.size())
    noise_level = 0.1
    noisy_waveform = samples + noise * noise_level

    return noisy_waveform

def add_reverb_2(waveform: torch.Tensor):
    # define the room parameters
    # Set the parameters for the reverb effect
    # Define the reverb effect
    # Define effects
    # Add reverb effect
    # Normalize waveform to between -1 and 1
    normalized = waveform / torch.abs(waveform).max()
    sample_rate = 16000
    # Define reverb effect parameters
    reverb = torchaudio.sox_effects.effect_names()

    # Apply reverb effect to waveform
    reverberated_waveform = torchaudio.sox_effects.apply_effects_tensor(normalized.unsqueeze(0), sample_rate, [reverb])[0]

    # Apply the reverb effect to the waveform
    # reverb_waveform = F.apply_reverb(samples, 16000, room_size=room_size, decay_time=decay_time, density=density)

    return reverberated_waveform

def loudness_normalization(samples: torch.Tensor,
                           target_db: float=15.0,
                           max_gain_db: float=30.0):
    """Normalizes the loudness of the input signal."""
    std = torch.std(samples) + 1e-9
    gain = np.minimum(db_to_linear(max_gain_db), db_to_linear(target_db) / std)
    return gain * samples


def collate_fn(batch, seq_len, label_transform_fn=None, args=None):
    # A data tuple has the form:
    # waveform, label, *anything else
    tensors, targets, anythings = [], [], []

    # Gather in lists, normalize waveforms, encode labels as indices
    for waveform, label, *anything in batch:
        norm_wave = waveform.float()
        if label_transform_fn:
            label = label_transform_fn(label)
        # align the waveform
        aligned_wave = align_sample(norm_wave, seq_len=seq_len)
        # loudness_normalization
        normalized = loudness_normalization(aligned_wave)
        # if args.noise == True:
        #     wave = add_noise(normalized)
        # if args.reverb == True:
        #     wave = add_reverb_2(normalized)

        tensors.append(normalized)
        targets.append(label)
        anythings.append(anything)

    # Group the list of tensors into a batched tensor
    #and loudness normalization
    data = torch.stack(tensors)
    targets = torch.stack(targets)
    return (data, targets) + tuple(zip(*anythings))


def build_dataloaders(train_set, val_set, test_set, label_transform_fn, args):
    collate = partial(collate_fn, seq_len=args.input_size,
                      label_transform_fn=label_transform_fn, args=args)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    val_loader = torch.utils.data.DataLoader(
        WindowedDataset(val_set, args.input_size, overlap=args.eval_overlap,
                        pad_incomplete=args.eval_pad),
        batch_size=args.batch_size_eval or args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=min(1, args.num_workers),  # must keep order
        pin_memory=args.pin_mem,
    )

    test_loader = torch.utils.data.DataLoader(
        WindowedDataset(test_set, args.input_size, overlap=args.eval_overlap,
                        pad_incomplete=args.eval_pad),
        batch_size=args.batch_size_eval or args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=min(1, args.num_workers),  # must keep order
        pin_memory=args.pin_mem,
    )
    return train_loader, val_loader, test_loader


