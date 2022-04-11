import csv
import os
import random
import torch

from torch.utils import data
import numpy as np
import librosa
from util.pad import pad_along_axis

class LibrispeechUnsupervisedLoader():
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.ids = []

    def load(self):
        with open(self.csv_path, 'r') as rf:
            reader = csv.reader(rf)
            next(reader)

            for row in reader:
                self.ids.append(row[0].replace(".wav", ""))

        return self.ids


class LibrispeechUnsupervisedDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, batch_size, sample_len=20480):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir
        self.sample_len = sample_len

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        data, _ = librosa.load(os.path.join(self.dataset_dir, ID + '.wav'), sr=16000)
        data = pad_along_axis(data, self.sample_len, axis=0)

        randstart = random.randint(0, len(data) - self.sample_len)

        # Load data and get label
        X = torch.from_numpy(data[randstart:randstart +self.sample_len]).float()
        X = X.unsqueeze(0)
        return X


class LibrispeechPhonemeCachedDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, batch_size, embedding_extension='_audio_only_embeddings'):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir
        self.embedding_extension = embedding_extension

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x_data = np.load(os.path.join(self.dataset_dir+self.embedding_extension, ID+'.npy'))
        y_data = np.load(os.path.join(self.dataset_dir, ID+'-phn.npy'))

        # Load data and get label
        X = torch.from_numpy(x_data).float()
        X = X
        X_len = torch.LongTensor([x_data.shape[0]])

        Y = torch.from_numpy(y_data).int()
        Y_len = torch.LongTensor([y_data.shape[0]])

        return X, X_len, Y, Y_len

class LibrispeechCachedPhonemeDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, batch_size, embedding_extension='_audio_only_embeddings'):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir
        self.embedding_extension = embedding_extension

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x_data = np.load(os.path.join(self.dataset_dir+self.embedding_extension, ID+'.npy'))
        y_data = np.load(os.path.join(self.dataset_dir, ID+'-phn.npy'))

        # Load data and get label
        X = torch.from_numpy(x_data).float()
        X = X
        X_len = torch.LongTensor([x_data.shape[0]])

        Y = torch.from_numpy(y_data).int()
        Y_len = torch.LongTensor([y_data.shape[0]])

        return X, X_len, Y, Y_len

class LibrispeechPhonemeDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, batch_size):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x_data, _ = librosa.load(os.path.join(self.dataset_dir, ID + '.wav'), sr=16000)
        y_data = np.load(os.path.join(self.dataset_dir, ID+'-phn.npy'))

        # Load data and get label
        X = torch.from_numpy(x_data).float()
        X = X.unsqueeze(0)
        X_len = torch.LongTensor([x_data.shape[0]])

        Y = torch.from_numpy(y_data).int()
        Y_len = torch.LongTensor([y_data.shape[0]])

        return X, X_len, Y, Y_len

class LRS2AudioVisualCachedPhonemeDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, batch_size, embedding_extension='_audiovisual_embeddings'):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir
        self.embedding_extension = embedding_extension

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x_data = np.load(os.path.join(self.dataset_dir + self.embedding_extension, ID + '.npy'))
        y_data = np.load(os.path.join(self.dataset_dir, ID + '-phn.npy'))

        # Load data and get label
        X = torch.from_numpy(x_data).float()
        X = X
        X_len = torch.LongTensor([x_data.shape[0]])

        Y = torch.from_numpy(y_data).int()
        Y_len = torch.LongTensor([y_data.shape[0]])

        return X, X_len, Y, Y_len

class LRS2AudioVisualCachedCharacterDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, batch_size, embedding_extension='_audiovisual_embeddings'):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir
        self.embedding_extension = embedding_extension

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x_data = np.load(os.path.join(self.dataset_dir + self.embedding_extension, ID + '.npy'))
        y_data = np.load(os.path.join(self.dataset_dir, ID + '-chars.npy'))

        # Load data and get label
        X = torch.from_numpy(x_data).float()
        X = X
        X_len = torch.LongTensor([x_data.shape[0]])

        Y = torch.from_numpy(y_data).int()
        Y_len = torch.LongTensor([y_data.shape[0]])

        return X, X_len, Y, Y_len

class LRS2AudioVisualPhonemeDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, batch_size, return_lens=False):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir
        self.return_lens = return_lens

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x_audio_data, _ = librosa.load(os.path.join(self.dataset_dir, ID + '.wav'), sr=16000)
        y_data = np.load(os.path.join(self.dataset_dir, ID+'-phn.npy'))

        # Load data and get label
        X_audio = torch.from_numpy(x_audio_data).float()
        X_audio = X_audio.unsqueeze(0)

        X_visual_data = np.load(os.path.join(self.dataset_dir, ID + '.npy')).T
        X_visual = torch.from_numpy(X_visual_data).float()


        Y = torch.from_numpy(y_data).int()

        if not self.return_lens:
            return X_audio, X_visual, Y
        else:
            X_len = torch.LongTensor([x_audio_data.shape[0]])
            Y_len = torch.LongTensor([y_data.shape[0]])

            return X_audio, X_visual, X_len, Y, Y_len

class LRS2UnsupervisedDataset(data.Dataset):
    #Characterizes a dataset for PyTorch
    def __init__(self, list_IDs, dataset_dir, batch_size, sample_len=20480, video_sample_len=32):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir
        self.sample_len = sample_len
        self.video_sample_len = video_sample_len

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        audio_data, _ = librosa.load(os.path.join(self.dataset_dir, ID + '.wav'), sr=16000)
        audio_data = pad_along_axis(audio_data, self.sample_len, axis=0)

        randstart = random.randint(0, len(audio_data) - self.sample_len)


        # Load data and get label
        audio_X = torch.from_numpy(audio_data[randstart:randstart+self.sample_len]).float()
        audio_X = audio_X.unsqueeze(0)

        randstart_visual = randstart // 640

        visual_data = np.load(os.path.join(self.dataset_dir, ID + '.npy'))
        visual_X = visual_data[randstart_visual:randstart_visual+self.video_sample_len,:].T

        #pad for rounding issue
        visual_X = pad_along_axis(visual_X, self.video_sample_len,  axis=1)

        return audio_X, visual_X

class LRS2UnsupervisedLoader():
    def __init__(self, file_path):
        self.file_path = file_path
        self.ids = []

    def load(self):
        with open(self.file_path, 'r') as rf:

            for row in rf.readlines():
                self.ids.append(row.split(" ")[0].strip())

        return self.ids

class hybridNetworkDataset():
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, batch_size, return_lens=False):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)
        self.list_IDs = list_IDs[:maxlen]

        self.dataset_dir = dataset_dir
        self.return_lens = return_lens

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x_audio_data, _ = librosa.load(os.path.join(self.dataset_dir, ID + '.wav'), sr=16000)
        y_data = np.load(os.path.join(self.dataset_dir, ID+'-chars.npy'))

        # Load data and get label
        X_audio = torch.from_numpy(x_audio_data).float()
        X_audio = X_audio.unsqueeze(0)

        X_visual_data = np.load(os.path.join(self.dataset_dir, ID + '.npy')).T
        X_visual = torch.from_numpy(X_visual_data).float()


        Y = torch.from_numpy(y_data).int()

        if not self.return_lens:
            return X_audio, X_visual, Y
        else:
            X_len = torch.LongTensor([x_audio_data.shape[0]])
            Y_len = torch.LongTensor([y_data.shape[0]])

            return X_audio, X_visual, X_len, Y, Y_len