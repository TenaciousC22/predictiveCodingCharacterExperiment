import csv
import math
import os
import wave
from torch.utils import data
import librosa
import torch
import numpy as np
from progressbar import progressbar
from tqdm import tqdm
from torch.multiprocessing import Lock, Manager
from generators.librispeech import LRS2UnsupervisedLoader, LRS2AudioVisualPhonemeDataset
from models.audiovisual_model import  FBAudioVisualCPCCharacterClassifierLightning
from util.pad import audiovisual_batch_collate
from util.seq_alignment import beam_search

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def createDatasetPaths():
	paths=[]

	for x in range(6):
		for y in range(28):
			for key in offsetMap:
				paths.append("speaker"+str(x+1)+"clip"+str(y+1)+offsetMap[key])

	return paths

#Get a full list of all videos with speakers, sentences, and offsets
per_ckpt="/home/analysis/Documents/studentHDD/chris/predictiveCodingCharacterExperiment/lrs2_audiovisual_lstm_character_classifier_lightning_logs/lightning_logs/version_3/checkpoints/epoch=6-step=40102.ckpt"
dest_csv="/home/analysis/Documents/studentHDD/chris/predicitiveCodingCharacterExperiment/predicitiveCodingCharacterResults.csv"
datasetPath="/home/analysis/Documents/studentHDD/chris/monoSubclips/"

model = FBAudioVisualCPCCharacterClassifierLightning(src_checkpoint_path=per_ckpt, batch_size=1, cached=False, LSTM=True).cuda()
per_checkpoint = torch.load(per_ckpt)

model.load_state_dict(per_checkpoint['state_dict'])

avgPER = 0
nItems = 0

downsamplingFactor = 160

test_params = {'batch_size': 1,
    'shuffle': False,
    'num_workers': 3}

model.eval()

testIDs=createDatasetPaths()