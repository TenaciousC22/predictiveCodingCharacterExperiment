import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence


def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b

def audio_batch_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    data_len = torch.cat([item[1].unsqueeze(0) for item in batch], dim=0)
    targets = [item[2] for item in batch]
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=43)
    target_len = torch.cat([item[3].unsqueeze(0) for item in batch], dim=0)
    return [data, data_len, targets, target_len]

def audiovisual_batch_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    audio_data = [item[0].permute(1, 0) for item in batch]
    visual_data = [item[1].permute(1, 0) for item in batch]


    audio_data = torch.nn.utils.rnn.pad_sequence(audio_data, batch_first=True).permute(0, 2, 1)
    visual_data = torch.nn.utils.rnn.pad_sequence(visual_data, batch_first=True).permute(0, 2, 1)

    targets = [item[2] for item in batch]
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=43)
    return [audio_data, visual_data, targets]

def audiovisual_embedding_batch_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    data_len = torch.cat([item[1].unsqueeze(0) for item in batch], dim=0)
    targets = [item[2] for item in batch]
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=43)
    target_len = torch.cat([item[3].unsqueeze(0) for item in batch], dim=0)
    return [data, data_len, targets, target_len]