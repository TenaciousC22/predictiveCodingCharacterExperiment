import csv
import os
import random
import torch
import torchaudio
from torch.utils import data
import numpy as np

from util.pad import pad_along_axis


class SpeechCommandsLoader:
    def __init__(self, datasetDir, extension='.npy'):
        self.partition = {}
        self.labels = {}
        self.datasetDir = datasetDir

        self.partition['test'] = []
        self.partition['val'] = []
        self.partition['train'] = []

        with open(datasetDir+'testing_list.txt') as testingFile:
            for row in testingFile:
                self.partition['test'].append(row.replace('.wav', extension).strip())

        with open(datasetDir+'validation_list.txt') as validationFile:
            for row in validationFile:
                self.partition['val'].append(row.replace('.wav', extension).strip())

        for subfolder in os.listdir(self.datasetDir):

            subdir= os.path.join(self.datasetDir, subfolder)

            if not os.path.isdir(subdir):
                continue

            for file in os.listdir(subdir):
                fid = os.path.join(subfolder, file)

                if fid not in self.partition['test'] and fid not in self.partition['val']:
                    self.partition['train'].append(fid)

    def trim_dataset(self, n_per_class=64):
        keys = ['train', 'test', 'val']
        for key in keys:
            seen = {}
            new_part = []
            for item in self.partition[key]:
                label = item.split('/')[0]
                if label not in seen:
                    seen[label] = 0

                seen[label] += 1

                if seen[label] >= n_per_class:
                    continue

                new_part.append(item)

            self.partition[key] = new_part


    def load(self):
        pos=0

        label_strs = []

        for categoryFolder in os.listdir(self.datasetDir):
            categoryPath = os.path.join(self.datasetDir, categoryFolder)

            if not os.path.isdir(categoryPath):
                print("skipping {}".format(categoryPath))
                continue

            label_strs.append(categoryFolder)

        label_strs.sort()

        #loop through each folder, folders are named after each target class
        for categoryFolder in os.listdir(self.datasetDir):
            #skip if not folder
            categoryPath = self.datasetDir+categoryFolder
            if not os.path.isdir(categoryPath):
                continue

            if '_' in categoryFolder:
                continue

            for file in os.listdir(categoryPath):
                fileId = categoryFolder+'/'+file

                self.labels[fileId] = label_strs.index(categoryFolder)

        return self.partition, self.labels

class SpeechCommandsCPCDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dataset_dir, labels, batch_size):
        'Initialization'
        maxlen = len(list_IDs)-(len(list_IDs)%batch_size)

        self.list_IDs = list_IDs[:maxlen]
        self.labels = labels
        self.dataset_dir = dataset_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x = np.load(os.path.join(self.dataset_dir, ID))
        x = pad_along_axis(x, 100, axis=0)

        y = self.labels[ID]

        return x, y